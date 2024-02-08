from gadopt import *
from gadopt.inverse import *
from gadopt.gplates import pyGplatesConnector
from pyadjoint import stop_annotating
import numpy as np
from wrappers import collect_garbage
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


def __main__():
    my_taylor_test()


def my_taylor_test():
    Tic, reduced_functional = forward_problem()
    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, Delta_temp)


@collect_garbage
def forward_problem():
    # Set up geometry:
    rmin, rmax = 1.22, 2.22

    enable_disk_checkpointing()

    # Load mesh
    with CheckpointFile("../../Adjoint_CheckpointFile.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        T_initial_guess = f.load_function(mesh, name="Temperature")  # initial guess
        Tobs = f.load_function(mesh, name="ReferenceTemperature")  # reference tomography temperature
        Tave = f.load_function(mesh, name="AverageTemperature")  # 1-D geotherm
        mu_function = f.load_function(mesh, name="mu1viscosity")  # viscosity function

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
    Tic.interpolate(T_initial_guess)
    T = Function(Q, name="Temperature")
    T0 = Constant(0.091)  # Non-dimensional surface temperature
    Di = Constant(0.5)  # Dissipation number.
    H_int = Constant(10.0)  # Internal heating

    # Initial time step
    delta_t = Constant(1.0e-6)

    # Top velocity boundary condition
    gplates_velocities = Function(V, name="GPlates_Velocity")

    # Setup Equations Stokes related constants
    Ra = Constant(2.0e7)  # Rayleigh number
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

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False)
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Temperature boundary conditions (constant)
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
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu_function,
                                 cartesian=False, constant_jacobian=True,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    # tweaking solver parameters
    energy_solver.solver_parameters['ksp_converged_reason'] = None
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4
    stokes_solver.solver_parameters['snes_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-4
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-3

    # Initiating a plate reconstruction model
    pl_rec_model = pyGplatesConnector(
        rotation_filenames=[
            '../../gplates_files/Zahirovic2022_CombinedRotations_fixed_crossovers.rot'],
        topology_filenames=[
            '../../gplates_files/Zahirovic2022_PlateBoundaries.gpmlz',
            '../../gplates_files/Zahirovic2022_ActiveDeformation.gpmlz',
            '../../gplates_files/Zahirovic2022_InactiveDeformation.gpmlz'],
        dbc=stokes_solver.strong_bcs[0],
        geologic_zero=409,
        delta_time=1.0
    )

    # non-dimensionalised time for present geologic day (0)
    ndtime_now = pl_rec_model.geotime2ndtime(0.)

    # non-dimensionalised time for 10 Myrs ago
    time = pl_rec_model.geotime2ndtime(25.)

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
        if timestep_index % 2 == 0:
            # Update surface velocities
            pl_rec_model.assign_plate_velocities(time)
            # Surface velocities should be considered as a new block if the
            #   content has changed. This happens when the updated
            #   reconstruction time is the one as requested time
            gplates_velocities.create_block_variable()

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

    # Define the component terms of the overall objective functional
    smoothing = assemble(dot(grad(Tic - Tave), grad(Tic - Tave)) * dx)
    norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
    norm_obs = assemble(Tobs**2 * dx)

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    # Assembling the objective
    objective = (
        t_misfit +
        0.01 * (norm_obs * smoothing / norm_smoothing)
    )

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


if __name__ == "__main__":
    __main__()
