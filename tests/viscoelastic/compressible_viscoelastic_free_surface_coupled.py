from gadopt import *
import numpy as np
import argparse
OUTPUT = False
output_directory = "./2d_analytic_compressible_internalvariable_viscoelastic_freesurface/"

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="viscoelastic-compressible", type=str, help="Test case to run: elastic limit (dt << maxwell time, 1 step), viscoelastic (dt ~ maxwell time), viscous limit (dt >> maxwell time) ", required=False)
args = parser.parse_args()


def viscoelastic_model(nx=80, dt_factor=0.1, sim_time="long", shear_modulus=1e11, bulk_modulus=2e11):
    # Set up geometry:
    nz = nx  # Number of vertical cells
    D = 3e6  # length of domain in m
    L = D/2  # Depth of the domain in m
    mesh = RectangleMesh(nx, nz, L, D)  # Rectangle mesh generated via firedrake
    mesh.cartesian = True

    # Squash mesh to refine near top boundary modified from the ocean model
    # Roms e.g. https://www.myroms.org/wiki/Vertical_S-coordinate
    mesh.coordinates.dat.data[:, 1] -= D
    x, z = SpatialCoordinate(mesh)
    a = Constant(4)
    b = Constant(0)
    depth_c = 500.0
    z_scaled = z / D
    Cs = (1.-b) * sinh(a*z_scaled) / sinh(a) + b*(tanh(a*(z_scaled + 0.5))/(2*tanh(0.5*a)) - 0.5)
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([x, depth_c*z_scaled + (D - depth_c)*Cs]))
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

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # timestepping
    rho0 = Function(R).assign(Constant(4500))  # density in kg/m^3
    g = 10  # gravitational acceleration in m/s^2
    viscosity = Constant(1e21)  # Viscosity Pa s
    shear_modulus = Constant(shear_modulus)  # Shear modulus in Pa
    maxwell_time = viscosity / shear_modulus
    bulk_modulus = Constant(bulk_modulus)

    # Set up surface load
    lam = D / 8  # wavelength of load in m
    kk = 2 * pi / lam  # wavenumber in m^-1
    F0 = Constant(1000)  # initial free surface amplitude in m
    X = SpatialCoordinate(mesh)
    eta = -F0 * cos(kk * X[0])

    # Timestepping parameters
    year_in_seconds = 3600*24*365
    tau0 = Constant(2 * kk * viscosity / (rho0 * g))
    log("tau0 in years", float(tau0/year_in_seconds))
    time = Constant(0.0)
    dt = Constant(dt_factor * tau0)  # Initial time-step
    if sim_time == "long":
        max_timesteps = round(2*tau0/dt)
    else:
        max_timesteps = 1

    log("max timesteps", max_timesteps)
    dump_period = 1
    log("dump_period", dump_period)
    log("dt in years", float(dt/year_in_seconds))
    log("maxwell time in years", float(maxwell_time/year_in_seconds))

    approximation = CompressibleInternalVariableApproximation(bulk_modulus, rho0, shear_modulus, viscosity, g=g)

    # Create output file
    if OUTPUT:
        output_file = VTKFile(f"{output_directory}viscoelastic_freesurface_maxwelltime{float(maxwell_time/year_in_seconds):.0f}a_nx{nx}_dt{float(dt/year_in_seconds):.0f}a_tau{float(tau0/year_in_seconds):.0f}_symetricload_bulk2x_trace_nobuoy.pvd")

    # Setup boundary conditions
    stokes_bcs = {
        bottom_id: {'uy': 0},
        top_id: {'normal_stress': rho0*g*eta, 'free_surface': {"delta_rho_fs": rho0}},
        left_id: {'ux': 0},
        right_id: {'ux': 0},
    }

    # Setup analytical solution for the free surface from Cathles et al. 2024
    eta_analytical = Function(Q, name="eta analytical")
    eta_analytical2 = Function(Q, name="eta analytical2")

    lambda_lame = bulk_modulus - 2/3 * shear_modulus

    f_e = (lambda_lame + 2*shear_modulus) / (lambda_lame + shear_modulus)
    log("f_e: {f_e}")
    h_elastic = Constant((F0*rho0*g/(2*kk*shear_modulus))/(1 + f_e*maxwell_time/tau0))
    log("Maximum initial elastic displacement:", float(h_elastic))
    h_elastic2 = Constant(F0/(1 + f_e*maxwell_time/tau0))
    h_elastic = Constant(F0 - h_elastic2)  # Constant(F0/(1 + maxwell_time/tau0))
    log(" new Maximum initial elastic displacement:", float(h_elastic))
    log("Maximum initial elastic displacement:", float(h_elastic2))
    log("diff F0-helastic2:", float(F0-h_elastic2))
#    exit()
    eta_analytical.interpolate(((F0 - h_elastic) * (1-exp(-(time)/(tau0+f_e*maxwell_time)))+h_elastic) * cos(kk * X[0]))
    h_elasticold = Constant((F0*rho0*g/(2*kk*shear_modulus)))
    eta_analytical2.interpolate(((F0 - h_elasticold) * (1-exp(-(time)/(tau0)))+h_elasticold) * cos(kk * X[0]))
    error = 0  # Initialise error

    direct_stokes_solver_parameters = {
        "snes_monitor": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    # FIXME really should provide list of coupled bcs? but stokes integrators only expects 1 set of bcs...
    coupled_solver = InternalVariableSolver(z, approximation, coupled_dt=dt, bcs=stokes_bcs, solver_parameters=direct_stokes_solver_parameters)

    if OUTPUT:
        output_file.write(u_, m_, eta_analytical)

    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve stokes system
        coupled_solver.solve()

        # Timestep the internal variable field
#        vertical_displacement.interpolate(vc(u))
#        bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, top_id)
#        displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
#        displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
#        log("Greatest (-ve) displacement", displacement_min)

        time.assign(time+dt)  # tried updating tie after but doesnt seem to help...
        # Update analytical solution
        eta_analytical.interpolate(((F0 - h_elastic) * (1-exp(-(time)/(tau0+f_e*maxwell_time)))+h_elastic) * cos(kk * X[0]))
        eta_analytical2.interpolate(((F0 - h_elasticold) * (1-exp(-(time)/(tau0)))+h_elasticold) * cos(kk * X[0]))

        # Calculate error
        local_error = assemble(pow(u[1]-eta_analytical, 2)*ds(top_id))
        print(local_error)

        if sim_time == 'long':
            error += local_error * float(dt)
        else:
            # For elastic solve only one timestep so
            # don't scale error by timestep length
            # (get 0.5x convergence rate as artificially
            # making error smaller when in reality
            # displacement formulation shouldnt depend on
            # time)
            error += local_error

        # Write output:
        if timestep % dump_period == 0:
            log("timestep", timestep)
            log("time", float(time))
            if OUTPUT:
                output_file.write(u_, m_, eta_analytical)

    final_error = pow(error, 0.5)/L
    return final_error


params = {
    "viscoelastic-compressible": {
        "dtf_start": 0.1,
        "nx": 160,
        "sim_time": "long",
        "shear_modulus": 1e11,
        "bulk_modulus": 2e11},
    "elastic-compressible": {
        "dtf_start": 0.001,
        "nx": 160,
        "sim_time": "short",
        "shear_modulus": 1e11,
        "bulk_modulus": 2e11},
    "viscoelastic-incompressible-1e15": {
        "dtf_start": 0.1,
        "nx": 320,
        "sim_time": "long",
        "shear_modulus": 1e11,
        "bulk_modulus": 1e15},
    "elastic-incompressible": {
        "dtf_start": 0.001,
        "nx": 160,
        "sim_time": "short",
        "shear_modulus": 1e11,
        "bulk_modulus": 1e16},
    "viscous-incompressible": {
        "dtf_start": 0.1,
        "nx": 320,
        "sim_time": "long",
        "shear_modulus": 1e14,
        "bulk_modulus": 1e14}
}


def run_benchmark(case_name):

    # Run default case run for four dt factors
    dtf_start = params[case_name]["dtf_start"]
    params[case_name].pop("dtf_start")  # Don't pass this to viscoelastic_model
    dt_factors = dtf_start / (2 ** np.arange(4))
    nx = params[case_name]["nx"]
    prefix = f"errors-{case_name}-internalvariable-coupled-{nx}cells"
    errors = np.array([viscoelastic_model(dt_factor=dtf, **params[case_name]) for dtf in dt_factors])

    np.savetxt(f"{prefix}-free-surface-1e16.dat", errors)
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    print(convergence)


if __name__ == "__main__":
    run_benchmark(args.case)
