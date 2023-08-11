from gadopt import *
from gadopt.inverse import *
import numpy as np

ds_t = ds_t(degree=6)
dx = dx(degree=6)


def main():
    for case in ["damping", "smoothing", "Tobs", "uobs"]:
        try:
            rectangle_taylor_test(case)
        except Exception:
            raise Exception(f"Taylor test for case {case} failed!")


def rectangle_taylor_test(case):
    """
    Perform a second-order taylor remainder convergence test
    for one term in the objective functional for the rectangular case
    and asserts if convergence is above 1.9

    Args:
        case (string): name of the objective functional term
            either of "damping", "smooothing", "Tobs", "uobs"
    """

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile("mesh.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    # enable_disk_checkpointing()

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Control function space
    Z = MixedFunctionSpace([V, W])  # Mixed function space

    # Test functions and functions to hold solutions:
    z = Function(Z)  # A field over the mixed function space Z
    u, p = z.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")

    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    # Define time stepping parameters:
    max_timesteps = 80
    delta_t = Constant(4e-6)  # Constant time step

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")
    Taverage = Function(Q1, name="Average Temperature")

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")
    # Initialise the control
    Tic.project(
        checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    )
    Taverage.project(checkpoint_file.load_function(mesh, "Average Temperature", idx=0))

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    stokes_bcs = {
        "top": {"uy": 0},
        "bottom": {"uy": 0},
        1: {"ux": 0},
        2: {"ux": 0},
    }
    temp_bcs = {
        "top": {"T": 0.0},
        "bottom": {"T": 1.0},
    }

    energy_solver = EnergySolver(
        T,
        u,
        approximation,
        delta_t,
        ImplicitMidpoint,
        bcs=temp_bcs,
    )

    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
    )

    # Control variable for optimisation
    control = Control(Tic)

    u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # If it is only for smoothing or damping, there is no need to do the time-steping
    initial_timestep = 0 if case in ["Tobs", "uobs"] else max_timesteps - 1

    # Populate the tape by running the forward simulation
    for timestep in range(initial_timestep, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()

        # Update the accumulated surface velocity misfit using the observed value
        uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=timestep)
        u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)

    # Load the observed final state
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("Observed Temperature")

    # Load the reference initial state
    # Needed to measure performance of weightings
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_ref.rename("Reference Initial Temperature")

    # Load the average temperature profile
    Taverage = checkpoint_file.load_function(mesh, "Average Temperature", idx=0)

    checkpoint_file.close()

    # Define the component terms of the overall objective functional
    damping = assemble((Tic - Taverage) ** 2 * dx)
    norm_damping = assemble(Taverage**2 * dx)
    smoothing = assemble(dot(grad(Tic - Taverage), grad(Tic - Taverage)) * dx)
    norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
    norm_obs = assemble(Tobs**2 * dx)
    norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    if case == "Tobs":
        objective = t_misfit
    elif case == "uobs":
        objective = norm_obs * u_misfit / max_timesteps / norm_u_surface
    elif case == "damping":
        objective = norm_obs * damping / norm_damping
    else:
        objective = norm_obs * smoothing / norm_smoothing

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    # Defining the object for pyadjoint
    reduced_functional = ReducedFunctional(objective, control)

    # Applying timer decorator to fwd and derivative calls.
    for func in [reduced_functional, reduced_functional.derivative]:
        func = timer_decorator(func)

    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, Delta_temp)

    log(
        (
            "\n\nEnd of Taylor Test ****: "
            f"case: {case} "
            f"conversion: {minconv:.8e}\n\n\n"
        )
    )
    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()

    # Making sure test results are satisfied
    assert minconv > 1.9


if __name__ == "__main__":
    main()
