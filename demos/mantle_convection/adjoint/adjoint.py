from gadopt import *
from gadopt.inverse import *
import gc


def forward_problem():
    # Open the checkpoint file and subsequently load the mesh:
    checkpoint_filename = "adjoint-demo-checkpoint-state.h5"
    checkpoint_file = CheckpointFile(checkpoint_filename, mode="r")
    mesh = checkpoint_file.load_mesh("firedrake_default_extruded")
    mesh.cartesian = True

    enable_disk_checkpointing()

    # Specify boundary markers, noting that for extruded meshes the upper and lower boundaries are tagged as
    # "top" and "bottom" respectively.
    bottom_id, top_id, left_id, right_id = "bottom", "top", 1, 2

    # Retrieve the timestepping information for the Velocity and Temperature functions from checkpoint file:
    temperature_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Temperature")

    # Load the final state, analagous to the present-day "observed" state:
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
    Tobs.rename("Observed Temperature")
    # Load the reference initial state - i.e. the state that we wish to recover:
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
    Tic_ref.rename("Reference Initial Temperature")
    checkpoint_file.close()

    # These fields can be visualised using standard VTK software, such as Paraview or pyvista.
    tape = get_working_tape()
    tape.clear_tape()

    # Set up function spaces:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space

    # Specify test functions and functions to hold solutions:
    z = Function(Z)  # A field over the mixed function space Z
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    T = Function(Q, name="Temperature")

    # Specify important constants for the problem, alongside the approximation:
    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    # Define time-stepping parameters:
    delta_t = Constant(4e-6)  # Constant time step
    timesteps = int(temperature_timestepping_info["index"][-1]) + 1  # number of timesteps from forward

    # Nullspaces for the problem are next defined:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Followed by boundary conditions, noting that all boundaries are free slip, whilst the domain is
    # heated from below (T = 1) and cooled from above (T = 0).
    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {"uy": 0},
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }
    temp_bcs = {
        bottom_id: {"T": 1.0},
        top_id: {"T": 0.0},
    }

    # Setup Energy and Stokes solver
    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, constant_jacobian=True)
    # -

    # Specify Problem Length
    # ------------------------
    #
    # For the purpose of this demo, we only invert for a total of 10 time-steps. This makes it
    # tractable to run this within a tutorial session.
    #
    # To run for the simulation's full duration, change the initial_timestep to `0` below, rather than
    # `timesteps - 10`.

    initial_timestep = timesteps - 10

    # Define the Control Space
    # ------------------------
    #
    # In this section, we define the control space, which can be restricted to reduce the risk of encountering an
    # undetermined problem. Here, we select the Q1 function space for the initial condition $T_{ic}$. We also provide an
    # initial guess for the control value, which in this synthetic test is the temperature field of the reference
    # simulation at the final time-step (`timesteps - 1`). In other words, our guess for the initial temperature
    # is the final model state.

    # +
    # Define control function space:
    Q1 = FunctionSpace(mesh, "CG", 1)

    # Create a function for the unknown initial temperature condition, which we will be inverting for. Our initial
    # guess is set to the 1-D average of the forward model. We first load that, at the relevant timestep.
    # Note that this layer average will later be used for the smoothing term in our objective functional.
    with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
        Taverage = checkpoint_file.load_function(mesh, "Average_Temperature", idx=initial_timestep)
    Tic = Function(Q1, name="Initial_Condition_Temperature").assign(Taverage)

    # Given that Tic will be updated during the optimisation, we also create a function to store our initial guess,
    # which we will later use for smoothing. Note that since smoothing is executed in the control space, we must
    # specify boundary conditions on this term in that same Q1 space.
    T0_bcs = [DirichletBC(Q1, 0., top_id), DirichletBC(Q1, 1., bottom_id)]
    T0 = Function(Q1, name="Initial_Guess_Temperature").project(Tic, bcs=T0_bcs)

    # We next make pyadjoint aware of our control problem:
    control = Control(Tic)

    # Take our initial guess and project to T, simultaneously applying boundary conditions in the Q2 space:
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # We continue by integrating the solutions at each time-step.
    # Notice that we cumulatively compute the misfit term with respect to the
    # surface velocity observable.

    # +
    u_misfit = 0.0

    # Next populate the tape by running the forward simulation.
    for time_idx in range(initial_timestep, timesteps):
        stokes_solver.solve()
        energy_solver.solve()
        # Update the accumulated surface velocity misfit using the observed value.
        with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
            uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=time_idx)
        u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)

    # Define component terms of overall objective functional and their normalisation terms:
    damping = assemble((T0 - Taverage) ** 2 * dx)
    norm_damping = assemble(Taverage**2 * dx)
    smoothing = assemble(dot(grad(T0 - Taverage), grad(T0 - Taverage)) * dx)
    norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
    norm_obs = assemble(Tobs**2 * dx)
    norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

    # Define temperature misfit between final state solution and observation:
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    # Weighting terms
    alpha_u = 1e-1
    alpha_d = 1e-3
    alpha_s = 1e-3

    # Define overall objective functional:
    objective = (
        t_misfit +
        alpha_u * (norm_obs * u_misfit / timesteps / norm_u_surface) +
        alpha_d * (norm_obs * damping / norm_damping) +
        alpha_s * (norm_obs * smoothing / norm_smoothing)
    )
    # -

    reduced_functional = ReducedFunctional(objective, control)

    pause_annotation()

    return Tic, reduced_functional


def run_forward_inverse():
    Tic, reduced_functional = forward_problem()
    for i in range(5):
        gc.collect()
        reduced_functional(Tic)
        gc.collect()
        reduced_functional.derivative()


def minimisation_problem():
    # We can print the contents of the tape at this stage to verify that it is not empty.
    # Define lower and upper bounds for the temperature
    T_lb = Function(Tic.function_space(), name="Lower Bound Temperature")
    T_ub = Function(Tic.function_space(), name="Upper Bound Temperature")

    # Assign the bounds
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    minimisation_parameters["Status Test"]["Iteration Limit"] = 5
    minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1e-2

    # Define the LinMore Optimiser class with checkpointing capability:
    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )

    solutions_vtk = VTKFile("solutions.pvd")
    solution_container = Function(Tic.function_space(), name="Solutions")
    functional_values = []

    def callback():
        solution_container.assign(Tic.block_variable.checkpoint)
        solutions_vtk.write(solution_container)
        final_temperature_misfit = assemble(
            (T.block_variable.checkpoint - Tobs) ** 2 * dx
        )
        log(f"Terminal Temperature Misfit: {final_temperature_misfit}")

    def record_value(value, *args):
        functional_values.append(value)

    optimiser.add_callback(callback)
    reduced_functional.eval_cb_post = record_value

    # Run the optimisation
    optimiser.run()


if __name__ == "__main__":
    run_forward_inverse()
