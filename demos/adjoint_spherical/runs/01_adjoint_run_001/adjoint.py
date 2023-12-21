from gadopt import *
from gadopt.inverse import *
import numpy as np
import warnings
import libgplates
from spherical_utils import step_func
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


def inversion():
    Tic, reduced_functional = forward_problem()

    T_lb = Function(Tic.function_space(), name="Lower bound temperature")
    T_ub = Function(Tic.function_space(), name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    minimisation_parameters["Step"]["Trust Region"]["Lin-More"]["Cauchy Point"][
        "Normalize Initial Step Size"] = False
    minimisation_parameters["Step"]["Trust Region"]["Lin-More"]["Cauchy Point"][
        "Initial Step Size"] = 0.06

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )
    optimiser.run()


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
    with CheckpointFile("./Adjoint_CheckpointFile.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        Tic = f.load_function(mesh, name="Temperature")
        Tobs = f.load_function(mesh, name="ReferenceTemperature")
        Tave = f.load_function(mesh, name="AverageTemperature")
        mu_function = f.load_function(mesh, name="mu1viscosity")

    bottom_id, top_id = "bottom", "top"

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Test functions and functions to hold solutions:
    v, w = TestFunctions(Z)
    z = Function(Z)
    u, p = split(z)

    # Set up temperature field and initialise:
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
    T = Function(Q, name="Temperature")
    T0 = Constant(0.091)  # Non-dimensional surface temperature
    Di = Constant(0.5)  # Dissipation number.
    H_int = Constant(10.0)  # Internal heating

    time = (409 - 25.0) * (
        libgplates.myrs2sec *
        libgplates.plate_scaling_factor /
        libgplates.time_dim_factor
    )
    time_presentday = 409 * (
        libgplates.myrs2sec *
        libgplates.plate_scaling_factor /
        libgplates.time_dim_factor
    )

    delta_t = Constant(5.0e-7)

    # GPLATES
    X_val = interpolate(X, V)

    # set up a Function for gplate velocities
    gplates_velocities = Function(V, name="GPlates_Velocity")

    # Setup Equations Stokes related constants
    Ra = Constant(5.0e6)  # Rayleigh number
    Di = Constant(0.5)  # Dissipation number.

    # Compressible reference state:
    rho_0, alpha = 1.0, 1.0
    weight = r-rmin
    rhobar = Function(Q, name="CompRefDensity").interpolate(rho_0 * exp(((1.0 - weight) * Di) / alpha))
    Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - weight) * Di) - T0)
    alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
    chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)

    # approximation = BoussinesqApproximation(Ra)
    approximation = TruncatedAnelasticLiquidApproximation(Ra, Di, rho=rhobar, Tbar=Tbar, alpha=alphabar, chi=chibar, cp=cpbar)

    delta_t = Constant(5e-7)  # Initial time-step

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Write output files in VTK format:
    u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u.rename("Velocity")
    p.rename("Pressure")

    # Open file for logging diagnostic output:
    temp_bcs = {
        bottom_id: {'T': 1.0 - (T0*exp(Di) - T0)},
        top_id: {'T': 0.0},
    }
    stokes_bcs = {
        top_id: {'u': gplates_velocities},
        bottom_id: {'un': 0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs, su_advection=True)
    energy_solver.fields['source'] = rhobar * H_int
    energy_solver.solver_parameters['ksp_converged_reason'] = None
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu_function,
                                 cartesian=False,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)
    stokes_solver.solver_parameters['snes_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # No-Slip (prescribed) boundary condition for the top surface
    boundary_X = X_val.dat.data_ro_with_halos[stokes_solver.strong_bcs[0].nodes]

    control = Control(Tic)

    project(
        Tic,
        T,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=energy_solver.strong_bcs,
    )

    num_timesteps = 0
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    output_pvd = File("visual/output.pvd")
    # Now perform the time loop:
    while time < time_presentday:
        # Update gplates velocities
        libgplates.rec_model.set_time(model_time=time)
        gplates_velocities.dat.data_with_halos[
            stokes_solver.strong_bcs[0].nodes
        ] = libgplates.rec_model.get_velocities(boundary_X)
        # Solve Stokes sytem:
        stokes_solver.solve()

        if time_presentday - time < float(delta_t):
            delta_t.assign(time_presentday - time)

        # Temperature system:
        energy_solver.solve()
        if num_timesteps % 4 == 0:
            output_pvd.write(T, u, p)
        time += float(delta_t)
        num_timesteps += 1

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    objective = t_misfit

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()
    return Tic, ReducedFunctional(objective, control)


if __name__ == "__main__":
    inversion()
