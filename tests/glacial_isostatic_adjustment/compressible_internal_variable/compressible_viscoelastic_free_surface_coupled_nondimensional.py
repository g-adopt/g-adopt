from gadopt import *
from gadopt.utility import vertical_component as vc
import numpy as np
import argparse
from mpi4py import MPI
import pandas as pd
OUTPUT = False
output_directory = "./2d_analytic_compressible_internalvariable_viscoelastic_freesurface_nondimensional/"

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="viscoelastic-compressible", type=str, help="Test case to run: elastic limit (dt << maxwell time, 1 step), viscoelastic (dt ~ maxwell time), viscous limit (dt >> maxwell time) ", required=False)
args = parser.parse_args()


def viscoelastic_model(nx=80, dt_factor=0.1, sim_time="long", viscosity=1e21, shear_modulus=1e11, bulk_modulus=2e11,lam_factor=64, compressible_adv_hyd_pre=True):
    # Set up geometry:
    nz = nx  # Number of vertical cells
    D = 3e6  # length of domain in m
    L = D/(lam_factor/4)  # Depth of the domain in m
    D_tilde = 1
    L_tilde = L / D
    mesh = RectangleMesh(nx, nz, L_tilde, D_tilde)  # Rectangle mesh generated via firedrake
    mesh.cartesian = True

    # Squash mesh to refine near top boundary modified from the ocean model
    # Roms e.g. https://www.myroms.org/wiki/Vertical_S-coordinate
    mesh.coordinates.dat.data[:, 1] -= D_tilde
    x, z = SpatialCoordinate(mesh)
    a = Constant(4)
    b = Constant(0)
    depth_c = 500.0/D
    z_scaled = z  # already 0 to 1.
    Cs = (1.-b) * sinh(a*z_scaled) / sinh(a) + b*(tanh(a*(z_scaled + 0.5))/(2*tanh(0.5*a)) - 0.5)
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([x, depth_c*z_scaled + (D/D - depth_c)*Cs]))
    mesh.coordinates.assign(f)
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

    # Set up function spaces - currently using P2P1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    TP1 = TensorFunctionSpace(mesh, "DG", 1)
    Z = MixedFunctionSpace([V, TP1])  # Mixed function space.
    R = FunctionSpace(mesh, "R", 0)

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, m = split(z)  # Returns symbolic UFL expression for u and p
    u_, m_ = z.subfunctions  # Returns individual Function for output
    u_.rename("u")
    m_.rename("internal variable")
    u_dim = Function(u_, name="u dimensional")

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # timestepping
    rho0 = Function(R).assign(Constant(4500))  # density in kg/m^3
    g = 10  # gravitational acceleration in m/s^2
    viscosity = Constant(viscosity)  # Viscosity Pa s
    shear_modulus = Constant(shear_modulus)  # Shear modulus in Pa
    maxwell_time = viscosity / shear_modulus  # Maxwell time in s. This is nondimensional timescale.
    log("maxwell time (used for nondimensional time scale)", float(maxwell_time))
    bulk_modulus = Constant(bulk_modulus)

    # Set up surface load
    lam = 1 / lam_factor  # nondimensional wavenumber
    kk = 2 * pi / lam  # nondimensional wavenumber
    kk_dim = 2 * pi / (D/lam_factor)  # wavenumber in m^-1
    F0 = Constant(1000/D)  # nondimensional initial free surface amplitude
    X = SpatialCoordinate(mesh)
    eta = -F0 * cos(kk * X[0])  # nondimensional surface load

    # Timestepping parameters
    tau0 = Constant(2 * kk_dim * viscosity / (rho0 * g) / maxwell_time)  # Nondimensional viscous relaxation time

    log("nondimensional tau0", float(tau0))
    time = Constant(0.0)
    dt = Constant(dt_factor)  # Initial time-step
    log("nondimensional dt", float(dt))
    if sim_time == "long":
        max_timesteps = round(160/dt) # 2tau0 = 149 maxwell times ish 
    else:
        max_timesteps = 1

    log("max timesteps", max_timesteps)
    dump_period = 1
    log("dump_period", dump_period)

    Vi = rho0 * D * g / shear_modulus
    log("Vi = rho0*g*D/mu", float(Vi))
    bulk_shear_ratio = bulk_modulus/shear_modulus
    log("k/mu", float(bulk_shear_ratio))
    approximation = CompressibleInternalVariableApproximation(bulk_modulus=1, density=Function(R).assign(Constant(1)), 
                                                              shear_modulus=1, viscosity=1, g=1, Vi=Vi, 
                                                              bulk_shear_ratio=bulk_shear_ratio,
                                                              compressible_buoyancy=False, 
                                                              compressible_adv_hyd_pre=compressible_adv_hyd_pre)

    # Create output file
    if OUTPUT:
        output_file = VTKFile(f"{output_directory}viscoelastic_freesurface_maxwelltime1_nx{nx}_dt{float(dt):.2f}_tau{float(tau0):.1f}_symetricload_bulk2x_trace_nobuoy.pvd")

    # Setup boundary conditions
    stokes_bcs = {
        bottom_id: {'uy': 0},
        top_id: {'normal_stress': Vi*eta, 'free_surface': {}},
        left_id: {'ux': 0},
        right_id: {'ux': 0},
    }

    # Setup analytical solution for the free surface from Cathles et al. 2024
    eta_analytical = Function(Q, name="eta analytical")
    eta_analytical_nondim = Function(Q, name="eta analytical nondim")

    lambda_lame = bulk_modulus - 2/3 * shear_modulus
    # Pseudo incompressible
    if float(bulk_shear_ratio) > 5:
        f_e = 1
        log(f"Comparing with incompressible analytical, f_e={f_e}")
    else:
        f_e = (lambda_lame + 2*shear_modulus) / (lambda_lame + shear_modulus)
        log(f"Comparing with compressible analytical, f_e={float(f_e)}")


    h_elastic2 = Constant(D*F0/(1 + f_e*maxwell_time/(maxwell_time*tau0)))
    h_elastic = Constant(D*F0 - h_elastic2)  # Constant(F0/(1 + maxwell_time/tau0))

    h_elastic2_nondim = Constant(F0/(1 + f_e/(tau0)))
    h_elastic_nondim = Constant(F0 - h_elastic2_nondim)  # Constant(F0/(1 + maxwell_time/tau0))

    eta_analytical.interpolate(((D*F0 - h_elastic) * (1-exp(-(time*maxwell_time)/(maxwell_time*tau0+f_e*maxwell_time)))+h_elastic) * cos(kk_dim * D*X[0]))
    eta_analytical_nondim.interpolate(((F0 - h_elastic_nondim) * (1-exp(-(time)/(tau0+f_e)))+h_elastic_nondim) * cos(kk * X[0]))
    error_nondim = 0  # Initialise error

    # FIXME really should provide list of coupled bcs? but stokes integrators only expects 1 set of bcs...
    coupled_solver = InternalVariableSolver(z, approximation, coupled_dt=dt, bcs=stokes_bcs, solver_parameters="direct")

    if OUTPUT:
        output_file.write(u_, m_, eta_analytical_nondim, u_dim, eta_analytical)
    
    vertical_displacement = Function(V.sub(1), name="vertical displacement")  # Function to store vertical displacement for output
    f = Function(V).interpolate(as_vector([X[0], X[1]]))
    bc_displacement = DirichletBC(V.sub(1), 0, 4)

    
    surface_x = f.sub(0).dat.data_ro_with_halos[bc_displacement.nodes]
    surface_x_all = f.sub(0).comm.gather(surface_x)
    displacement_df = pd.DataFrame()
    if MPI.COMM_WORLD.rank == 0:
        surface_nodes = np.concatenate(surface_x_all)
        displacement_df['surface_points'] = surface_nodes

    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve stokes system
        coupled_solver.solve()

        time.assign(time+dt)

        # Update analytical solution
        if sim_time == 'long':
            eta_analytical.interpolate(((D*F0 - h_elastic) * (1-exp(-(time*maxwell_time)/(maxwell_time*tau0+f_e*maxwell_time)))+h_elastic) * cos(kk_dim * D*X[0]))
            eta_analytical_nondim.interpolate(((F0 - h_elastic_nondim) * (1-exp(-(time)/(tau0+f_e)))+h_elastic_nondim) * cos(kk * X[0]))
        else:
            eta_analytical.interpolate(h_elastic * cos(kk_dim * D*X[0]))
            eta_analytical_nondim.interpolate(h_elastic_nondim * cos(kk * X[0]))
    
        # get displacement at surface for plotting
        vertical_displacement.interpolate(vc(z.subfunctions[0])*D)
        surface_disp = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes]
        surface_disp_all = vertical_displacement.comm.gather(surface_disp)


        if MPI.COMM_WORLD.rank == 0:
            surface_disp_concat = np.concatenate(surface_disp_all)
            displacement_df[f'surface_disp_step{timestep}'] = surface_disp_concat


        # Calculate error
        local_error_nondim = assemble(pow(u[1]-eta_analytical_nondim, 2)*ds(top_id))

        if sim_time == 'long':
            error_nondim += local_error_nondim * float(dt)
        else:
            # For elastic solve only one timestep so
            # don't scale error by timestep length
            # (get 0.5x convergence rate as artificially
            # making error smaller when in reality
            # displacement formulation shouldnt depend on
            # time)
            error_nondim += local_error_nondim

        # Write output:
        if timestep % dump_period == 0:
            log("timestep", timestep)
            log("time", float(time))
            if MPI.COMM_WORLD.rank == 0:
                displacement_df.to_csv(f"surface_displacement_dt{float(dt)}_nx{nx}arrays_bulktoshear{float(bulk_shear_ratio)}.csv")
            if OUTPUT:
                u_dim.interpolate(D*u)
                output_file.write(u_, m_, eta_analytical_nondim, u_dim, eta_analytical)

    final_error_nondim = pow(error_nondim, 0.5)/L_tilde
    return final_error_nondim


params = {
    "viscoelastic-compressible-visc1e21-shear1e11-bulk2e11-lam8-dtfstart16alpha-compahpFalse_160alpha": {
        "dtf_start": 16,
        "nx": 320,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 2e11,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscoelastic-compressible-visc1e21-shear5e9-bulk1e11-lam128": {
        "dtf_start": 0.1, # relative to tau0
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 5e9,
        "bulk_modulus": 1e11,
        "lam_factor": 128},
    "elastic-compressible-visc1e21-shear1e11-bulk2e11-lam8-compahpFalse": {
        "dtf_start": 0.1,
        "nx": 320,
        "sim_time": "short",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 2e11,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "elastic-compressible": {
        "dtf_start": 0.001, # old relative to tau0
        "nx": 320,
        "sim_time": "short",
        "shear_modulus": 1e11,
        "bulk_modulus": 10e11},
    "viscous-compressible-visc1e21-shear1e13-bulk2e11-lam8-compahpFalse": {
        "dtf_start": 0.1, # old relative to tau0
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e13,
        "bulk_modulus": 2e11,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscoelastic-incompressible-visc1e21-shear1e11-bulk1e12-lam8-dtfstart16alpha-compahpFalse_160alpha": {
        "dtf_start": 16,
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 1e12,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscoelastic-incompressible-visc1e21-shear1e11-bulk1e13-lam8-dtfstart16alpha-compahpFalse_160alpha": {
        "dtf_start": 16,
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 1e13,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscoelastic-incompressible-visc1e21-shear1e11-bulk1e14-lam8-dtfstart16alpha-compahpFalse_160alpha": {
        "dtf_start": 16,
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 1e14,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscoelastic-incompressible-visc1e21-shear1e11-bulk1e15-lam8-dtfstart16alpha-compahpFalse_160alpha": {
        "dtf_start": 16,
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 1e15,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscoelastic-incompressible-visc1e21-shear1e10-bulk1e15": {
        "dtf_start": 0.1, # old relative to tau0
        "nx": 1280,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e10,
        "bulk_modulus": 1e15},
    "elastic-incompressible-visc1e21-shear1e11-bulk1e15-lam8-compahpFalse": {
        "dtf_start": 0.1,
        "nx": 320,
        "sim_time": "short",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 1e15,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "elastic-incompressible-visc1e21-shear1e11-bulk1e14-lam8-compahpFalse": {
        "dtf_start": 0.1,
        "nx": 320,
        "sim_time": "short",
        "viscosity": 1e21,
        "shear_modulus": 1e11,
        "bulk_modulus": 1e14,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "elastic-incompressible-1e15": {
        "dtf_start": 0.001,
        "nx": 320,
        "sim_time": "short",
        "shear_modulus": 1e11,
        "bulk_modulus": 1e15},
    "viscous-incompressible-visc1e21-shear1e13-bulk1e15-lam8-compahpFalse": {
        "dtf_start": 0.1,
        "nx": 640,
        "sim_time": "long",
        "viscosity": 1e21,
        "shear_modulus": 1e13,
        "bulk_modulus": 1e15,
        "lam_factor": 8,
        "compressible_adv_hyd_pre": False},
    "viscous-incompressible-visc1e21-shear1e13-bulk1e15": {
        "dtf_start": 0.1,
        "nx": 1280,
        "sim_time": "long",
        "shear_modulus": 1e13,
        "bulk_modulus": 1e15}
}


def run_benchmark(case_name):

    # Run default case run for four dt factors
    dtf_start = params[case_name]["dtf_start"]
    params[case_name].pop("dtf_start")  # Don't pass this to viscoelastic_model
    dt_factors = dtf_start / (2 ** np.arange(6))
    nx = params[case_name]["nx"]
    prefix = f"errors-{case_name}-internalvariable-coupled-{nx}cells_nondimensional_direct_T2tau"
    errors = np.array([viscoelastic_model(dt_factor=dtf, **params[case_name]) for dtf in dt_factors])

    np.savetxt(f"{prefix}-free-surface.dat", errors)
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    log("dimensional conv", convergence)


if __name__ == "__main__":
    run_benchmark(args.case)
#    viscoelastic_model(nx=320, dt_factor=0.025, sim_time="long", viscosity=1e21, shear_modulus=1e11, bulk_modulus=1e15,lam_factor=8, compressible_adv_hyd_pre=False)

