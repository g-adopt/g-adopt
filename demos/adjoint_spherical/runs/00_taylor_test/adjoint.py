from gadopt import *
from gadopt.inverse import *
from gadopt.gplates import GplatesFunction, pyGplatesConnector
import numpy as np
from firedrake.adjoint_utils import blocks

# Quadrature degree:
dx = dx(degree=6)
ds_b = ds_b(degree=6)
ds_t = ds_t(degree=6)
ds_tb = ds_t + ds_b

# Projection solver parameters for nullspaces:
iterative_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
    "ksp_atol": 1e-10,
}

LinearSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
NonlinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
LinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
NonlinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters

blocks.solving.Block.evaluate_adj = collect_garbage(blocks.solving.Block.evaluate_adj)
blocks.solving.Block.recompute = collect_garbage(blocks.solving.Block.recompute)

# timer decorator for fwd and derivative calls.
ReducedFunctional.__call__ = collect_garbage(
    timer_decorator(ReducedFunctional.__call__)
)
ReducedFunctional.derivative = collect_garbage(
    timer_decorator(ReducedFunctional.derivative)
)

# Set up geometry:
rmax = 2.22
rmin = 1.222


def __main__():
    my_taylor_test()


def my_taylor_test():
    Tic, reduced_functional = forward_problem()
    log("Reduced Functional Repeat: ", reduced_functional([Tic]))
    reduced_functional.derivative()
    # Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    # Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    # _ = taylor_test(reduced_functional, Tic, Delta_temp)


@collect_garbage
def forward_problem():
    # Section:
    # Enable writing intermediary adjoint fields to disk
    enable_disk_checkpointing()

    with CheckpointFile("../../spherical_mesh.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    # Load mesh
    with CheckpointFile("../../linear_LLNLG3G_SLB_Q5_smooth_2.0_101.h5", "r") as fi:
        Tobs = fi.load_function(mesh, name="Tobs")  # reference tomography temperature
        Tave = fi.load_function(mesh, name="AverageTemperature")  # 1-D geotherm

    # Boundary markers to top and bottom
    bottom_id, top_id = "bottom", "top"

    # For accessing the coordinates
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Initial Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Test functions and functions to hold solutions:
    v, w = TestFunctions(Z)
    z = Function(Z)
    u, p = split(z)

    # Set up temperature field and initialise:
    Tic = Function(Q1, name="Tic")
    T = Function(Q, name="Temperature")
    mu = Function(Q, name="mu2_radial")
    assign_1d_profile(mu, "../../../gplates_global/mu2_radial.rad")

    T0 = Constant(0.091)  # Non-dimensional surface temperature
    Di = Constant(0.5)  # Dissipation number.
    H_int = Constant(10.0)  # Internal heating

    # Initial time step
    delta_t = Constant(1.0e-6)
    Tic.interpolate(((1.0 - (T0*exp(Di) - T0)) * (rmax-r)))

    pl_rec_model = pyGplatesConnector(
        rotation_filenames=[
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/optimisation/1000_0_rotfile_MantleOptimised.rot"
        ],
        topology_filenames=[
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/250-0_plate_boundaries.gpml",
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/410-250_plate_boundaries.gpml",
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Convergence.gpml",
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Divergence.gpml",
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Topologies.gpml",
            "../../../gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Transforms.gpml",
        ],
        nneighbours=4,
        nseeds=1e5,
        scaling_factor=1.0,
        oldest_age=1000,
        delta_t=1.0
    )

    # Top velocity boundary condition
    gplates_velocities = GplatesFunction(
        V,
        gplates_connector=pl_rec_model,
        top_boundary_marker=top_id,
        name="GPlates_Velocity"
    )

    # Setup Equations Stokes related constants
    Ra = Constant(2.0e6)  # Rayleigh number
    Di = Constant(0.5)  # Dissipation number.

    # Compressible reference state:
    rho_0, alpha = 1.0, 1.0
    weight = r-rmin
    rhobar = Function(Q, name="CompRefDensity").interpolate(rho_0 * exp(((1.0 - weight) * Di) / alpha))
    Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - weight) * Di) - T0)
    alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
    chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)

    # We use TALA for approximation
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra, Di, rho=rhobar, Tbar=Tbar,
        alpha=alphabar, chi=chibar, cp=cpbar)

    # Section: Setting up nullspaces
    # Nullspaces for stokes contains only a constant nullspace for pressure, as the top boundary is
    # imposed. The nullspace is generate with closed=True(for pressure) and rotational=False
    # as there are no rotational nullspace for velocity.
    # .. note: For compressible formulations we only provide `transpose_nullspace`
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False)
    # The near nullspaces gor gamg always include rotational and translational modes
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Section: Setting boundary conditions
    # Temperature boundary conditions (constant)
    # for the top and bottom boundaries
    temp_bcs = {
        bottom_id: {'T': 1.0 - (T0*exp(Di) - T0)},
        top_id: {'T': 0.0},
    }
    # Velocity boundary conditions
    stokes_bcs = {
        top_id: {'u': gplates_velocities},
        bottom_id: {'un': 0},
    }

    # Constructing Energy and Stokes solver
    energy_solver = EnergySolver(
        T, u, approximation, delta_t,
        ImplicitMidpoint, bcs=temp_bcs, su_advection=True)
    energy_solver.fields['source'] = rhobar * H_int
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                                 cartesian=False, constant_jacobian=True,
                                 transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    # tweaking solver parameters
    energy_solver.solver_parameters['ksp_converged_reason'] = None
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4
    stokes_solver.solver_parameters['snes_rtol'] = 1e-2
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # non-dimensionalised time for present geologic day (0)
    ndtime_now = pl_rec_model.age2ndtime(0.)

    # non-dimensionalised time for 10 Myrs ago
    time = pl_rec_model.age2ndtime(10.)

    # Write output files in VTK format:
    u_, p_ = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")

    # Defining control
    control = Control(Tic)

    # project the initial condition from Q1 to Q2, and imposing
    # boundary conditions
    project(
        Tic,
        T,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=energy_solver.strong_bcs,
    )

    # timestep counter
    timestep_index = 0

    # Now perform the time loop:
    while time < ndtime_now:
        # Update surface velocities
        gplates_velocities.update_plate_reconstruction(time)

        # Solve Stokes sytem
        stokes_solver.solve()

        # Make sure we are not going past present day
        if ndtime_now - time < float(delta_t):
            delta_t.assign(ndtime_now - time)

        # Temperature system:
        energy_solver.solve()

        # Updating time
        time += float(delta_t)
        timestep_index += 1

    # Temperature misfit between solution and observation
    objective = assemble((T - Tobs) ** 2 * dx)

    log(f"Value of objective: {objective}")

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()
    return Tic, ReducedFunctional(objective, control)


def mu_constructor(T, u):
    def step_func(r, center, mag, increasing=True, sharpness=30):
        """
        A step function designed to control viscosity jumps:
        input:
          r: is the radius array
          center: radius of the jump
          increasing: if True, the jump happens towards lower r, otherwise jump happens at higher r
          sharpness: how sharp should the jump should be (larger numbers = sharper).
        """
        if increasing:
            sign = 1
        else:
            sign = -1
        return mag * (0.5 * (1 + tanh(sign*(r-center)*sharpness)))

    # a constant mu
    mu_lin = 2.0

    # coordinates
    X = SpatialCoordinate(T.ufl_domain())
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)

    # Depth dependence: for the lower mantle increase we
    # multiply the profile with a linear function
    for line, step in zip([5.*(rmax-r), 1., 1.],
                          [step_func(r, 1.992, 30, False),
                           step_func(r, 2.078, 10, False),
                           step_func(r, 2.2, 10, True)]):
        mu_lin += line*step

    # Adding temperature dependence:
    delta_mu_T = Constant(100.)
    mu_lin *= exp(-ln(delta_mu_T) * T)
    mu_star, sigma_y = Constant(1.0), 5.0e5 + 2.5e6*(rmax-r)
    epsilon = sym(grad(u))  # strain-rate
    epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
    mu_plast = mu_star + (sigma_y / epsii)
    mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)
    return mu


def T_initialise(T, average):
    import scipy
    import math
    # Initial condition for T:
    # Evaluate P_lm node-wise using scipy lpmv
    X = SpatialCoordinate(T.ufl_domain())
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    theta = atan2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
    phi = atan2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
    l, m, eps_c, eps_s = 6, 4, 0.02, 0.02
    Plm = Function(T.function_space(), name="P_lm")
    cos_phi = assemble(interpolate(cos(phi), T.function_space()))
    Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
    Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
    if m == 0:
        Plm.assign(Plm/math.sqrt(2))
    T.interpolate(average +
                  (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))


def assign_1d_profile(q, one_d_filename):
    """
    Assign a one-dimensional profile to a Function `q` from a file.

    The function reads a one-dimensional radial viscosity profile from a file, broadcasts
    the read array to all processes, and then interpolates this
    array onto the function space of `q`.

    Args:
        q (firedrake.Function): The function onto which the 1D profile will be assigned.
        one_d_filename (str): The path to the file containing the 1D radial viscosity profile.

    Returns:
        None: This function does not return a value. It directly modifies the input function `q`.

    Note:
        - This function is designed to be run in parallel with MPI.
        - The input file should contain an array of viscosity values.
        - It assumes that the function space of `q` is defined on a radial mesh.
        - `rmax` and `rmin` should be defined before this function is called, representing
          the maximum and minimum radial bounds for the profile.
    """
    from firedrake.ufl_expr import extract_unique_domain
    from scipy.interpolate import interp1d
    # find the mesh
    mesh = extract_unique_domain(q)

    visc = None
    rshl = None
    # read the input file
    if mesh.comm.rank == 0:
        # The root process reads the file
        rshl, visc = np.loadtxt(one_d_filename, unpack=True, delimiter=",")

    # Broadcast the entire 'visc' array to all processes
    visc = mesh.comm.bcast(visc, root=0)
    # Similarly, broadcast 'rshl' if needed (assuming all processes need it)
    rshl = mesh.comm.bcast(rshl, root=0)

    element_family = q.function_space().ufl_element()
    X = Function(VectorFunctionSpace(mesh=mesh, family=element_family)).interpolate(SpatialCoordinate(mesh))
    rad = Function(q.function_space()).interpolate(sqrt(X**2))
    averager = LayerAveraging(mesh, cartesian=False)
    averager.extrapolate_layer_average(q, interp1d(rshl, visc, fill_value="extrapolate")(averager.get_layer_average(rad)))


if __name__ == "__main__":
    __main__()
