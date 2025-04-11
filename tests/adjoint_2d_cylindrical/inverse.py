"""
This runs the optimisation portion of the adjoint test case. A forward run first sets up
the tape with the adjoint information, then a misfit functional is constructed to be
used as the goal condition for nonlinear optimisation using ROL.

annulus_taylor_test is also added to this script for testing the correctness of the gradient for the inverse problem.
    taylor_test(alpha_T, alpha_u, alpha_d, alpha_s):
            alpha_T (float): The coefficient of the temperature misfit term.
            alpha_u (float): The coefficient of the velocity misfit term.
            alpha_d (float): The coefficient of the initial condition damping term.
            alpha_s (float): The coefficient of the smoothing term.
            float: The minimum convergence rate from the Taylor test. (Should be close to 2)
"""
from gadopt import *
from gadopt.inverse import *
import numpy as np
import sys
import itertools
from cases import cases, schedulers


def inverse(alpha_T=1e0, alpha_u=1e-1, alpha_d=1e-2, alpha_s=1e-1, checkpointing_schedule=None):

    # For solving the inverse problem we the reduced functional, any callback functions,
    # and the initial guess for the control variable
    inverse_problem = generate_inverse_problem(
        alpha_u=alpha_u,
        alpha_d=alpha_d,
        alpha_s=alpha_s,
        checkpointing_schedule=checkpointing_schedule
    )

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(inverse_problem["control"].function_space(), name="Lower_bound_temperature")
    T_ub = Function(inverse_problem["control"].function_space(), name="Upper_bound_temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(inverse_problem["reduced_functional"], bounds=(T_lb, T_ub))

    # Here we limit the number of optimisation iterations to ensure demo tractability and set other
    # parameters to help the optimisation procedure.
    minimisation_parameters["Status Test"]["Iteration Limit"] = 5
    minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1e-3

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )
    optimiser.add_callback(inverse_problem["callback"])
    optimiser.run()

    # If we're performing multiple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them.
    continue_annotation()


def run_forward_and_back(inverse_problem):
    # Make sure we are not annotating
    if annotate_tape():
        pause_annotation()

    # inverse_problem = generate_inverse_problem(alpha_T, alpha_u, alpha_d, alpha_s, checkpointing_schedule)
    value = inverse_problem["reduced_functional"](inverse_problem["control"])
    inverse_problem["reduced_functional"].derivative()

    continue_annotation()

    # Using a scheduler returns None in the callback for the derivative (A bug that shall be fixed)
    # For now we manually enter this value here
    inverse_problem["callback_function"].values[-1] = value

    return inverse_problem["objective"], inverse_problem["callback_function"]


def annulus_taylor_test(alpha_T, alpha_u, alpha_d, alpha_s, checkpointing_schedule):
    """
    Perform a Taylor test to verify the correctness of the gradient for the inverse problem.

    This function calls a main function to populate the tape for the inverse problem
    with specified regularization parameters, generates a random perturbation for the control variable,
    and performs a Taylor test to ensure the gradient is correct. Finally, it ensures that annotations
    are switched back on for any subsequent tests.

    Returns:
        minconv (float): The minimum convergence rate from the Taylor test.
    """

    # For solving the inverse problem we the reduced functional, any callback functions,
    # and the initial guess for the control variable
    inverse_problem = generate_inverse_problem(alpha_T, alpha_u, alpha_d, alpha_s, checkpointing_schedule)

    # generate perturbation for the control variable
    delta_temp = Function(inverse_problem["control"].function_space(), name="Delta_Temperature")
    delta_temp.dat.data[:] = np.random.random(delta_temp.dat.data.shape)

    # Perform a taylor test to ensure the gradient is correct
    minconv = taylor_test(
        inverse_problem["reduced_functional"],
        inverse_problem["control"],
        delta_temp
    )

    # If we're performing mulitple successive tests we want
    # to ensure the annotations are switched back on for the next code to use them
    continue_annotation()

    inverse_problem["minconv"] = minconv

    return inverse_problem


def generate_inverse_problem(alpha_T=1.0, alpha_u=-1, alpha_d=-1, alpha_s=-1, checkpointing_schedule=None):
    """
    Use adjoint-based optimisation to solve for the initial condition of the cylindrical
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        alpha_d: The coefficient of the initial condition damping term
        alpha_s: The coefficient of the smoothing term
    """

    # Get working tape
    tape = get_working_tape()
    tape.clear_tape()

    # If we are not annotating, let's switch on taping
    if not annotate_tape():
        continue_annotation()

    # Using SingleMemoryStorageSchedule
    if any([alpha_T > 0, alpha_u > 0]):
        tape.enable_checkpointing(checkpointing_schedule)

    # Set up geometry:
    rmax = 2.22
    rmax_earth = 6370  # Radius of Earth [km]
    rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
    r_410_earth = rmax_earth - 410  # 410 radius [km]
    r_660_earth = rmax_earth - 660  # 660 raidus [km]
    r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
    r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = False

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])
    R = FunctionSpace(mesh, "R", 0)  # Real number function space

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    z.assign(0)
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    Ra = Constant(1e7)  # Rayleigh number

    # Define time stepping parameters:
    max_timesteps = 250
    delta_t = Function(R, name="delta_t").assign(3e-6)  # Constant time step

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial_Temperature")
    T_0 = Function(Q, name="T_0")  # Temperature for zeroth time-step
    Taverage = Function(Q1, name="Average_Temperature")

    # Initialise the control to final state temperature from forward model.
    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")
    Tic.project(
        checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    )
    Taverage.project(checkpoint_file.load_function(mesh, "Average_Temperature", idx=0))

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

    # A step function designed to design viscosity jumps
    # Build a step centred at "centre" with given magnitude
    # Increase with radius if "increasing" is True
    def step_func(centre, mag, increasing=True, sharpness=50):
        return mag * (
            0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
        )

    # From this point, we define a depth-dependent viscosity mu_lin
    mu_lin = 2.0

    # Assemble the depth dependence
    for line, step in zip(
        [5.0 * (rmax - r), 1.0, 1.0],
        [
            step_func(r_660, 30, False),
            step_func(r_410, 10, False),
            step_func(2.2, 10, True),
        ],
    ):
        mu_lin += line * step

    # Add temperature dependence of viscosity
    mu_lin *= exp(-ln(Constant(80)) * T)

    # Assemble the viscosity expression in terms of velocity u
    eps = sym(grad(u))
    epsii = sqrt(inner(eps, eps) + 1e-10)
    # yield stress and its depth dependence
    # consistent with values used in Coltice et al. 2017
    sigma_y = 2e4 + 4e5 * (rmax - r)
    mu_plast = 0.1 + (sigma_y / epsii)
    mu_eff = 2 * (mu_lin * mu_plast) / (mu_lin + mu_plast)
    mu = conditional(mu_eff > 0.4, mu_eff, 0.4)

    # Configure approximation
    approximation = BoussinesqApproximation(Ra, mu=mu)

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1]
    )

    # Free-slip velocity boundary condition on all sides
    stokes_bcs = {
        "bottom": {"un": 0},
        "top": {"un": 0},
    }
    temp_bcs = {
        "bottom": {"T": 1.0},
        "top": {"T": 0.0},
    }

    energy_solver = EnergySolver(
        T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
    )

    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
        solver_parameters="direct"
    )

    # Control variable for optimisation
    control = Control(Tic)

    # If we are using surface veolocit misfit in the functional
    if alpha_u > 0:
        u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T_0.project(Tic, bcs=energy_solver.strong_bcs)
    T.assign(T_0)

    # if the weighting for misfit terms non-positive, then no need to integrate in time
    min_timesteps = max_timesteps - 5 if any([w > 0 for w in [alpha_T, alpha_u]]) else max_timesteps

    # Generate a surface velocity reference
    uobs = Function(V, name="uobs")

    # Populate the tape by running the forward simulation
    for timestep in tape.timestepper(iter(range(min_timesteps, max_timesteps))):
        stokes_solver.solve()
        energy_solver.solve()

        if alpha_u > 0:
            # Update the accumulated surface velocity misfit using the observed value
            uobs.assign(checkpoint_file.load_function(mesh, name="Velocity", idx=timestep))
            u_misfit += assemble(Function(R, name="alpha_u").assign(float(alpha_u)/(max_timesteps - min_timesteps)) * dot(u - uobs, u - uobs) * ds_t)

    # Load observed final state.
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("Observed Temperature")

    # Load reference initial state (needed to measure performance of weightings).
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_ref.rename("Reference Initial Temperature")

    # Load average temperature profile.
    Taverage = checkpoint_file.load_function(mesh, "Average_Temperature", idx=0)

    checkpoint_file.close()

    # Initiate objective functional.
    objective = 0.0

    if any([w > 0 for w in [alpha_u, alpha_d, alpha_s]]):
        # Calculate norm of the observed temperature, it will be used in multiple spots later to
        # weight other terms.
        norm_obs = assemble(Tobs**2 * dx)

    # Define the component terms of the overall objective functional
    # Temperature term
    if alpha_T > 0:
        # Temperature misfit between solution and observation
        objective += assemble(Function(R, name="alpha_T").assign(float(alpha_T)) * (T - Tobs) ** 2 * dx)

    # Velocity misfit term
    if alpha_u > 0:
        norm_u_surface = assemble(dot(uobs, uobs) * ds_t)  # measure of u_obs from the last timestep
        objective += (norm_obs * u_misfit / norm_u_surface)

    # Damping term
    if alpha_d > 0:
        damping = assemble(Function(R, name="alpha_d").assign(float(alpha_d)) * (T_0 - Taverage) ** 2 * dx)
        norm_damping = assemble((Tobs - Taverage)**2 * dx)
        objective += (norm_obs * damping / norm_damping)

    # Smoothing term
    if alpha_s > 0:
        smoothing = assemble(Function(R, name="alpha_s").assign(float(alpha_s)) * dot(grad(T_0 - Taverage), grad(T_0 - Taverage)) * dx)
        norm_smoothing = assemble(dot(grad(Tobs - Taverage), grad(Tobs - Taverage)) * dx)
        objective += (norm_obs * smoothing / norm_smoothing)

    # All done with the forward run, stop annotating anything else to the tape.
    pause_annotation()

    inverse_problem = {}

    # objective for zeroth iteration
    inverse_problem["objective"] = float(objective)

    # Specify control.
    inverse_problem["control"] = Tic

    # Call back function to store functional/gradient/control values
    class CallBack(object):
        def __init__(self):
            self.iteration = 0
            self.values = []
            self.derivatives = []
            self.controls = []

        def __call__(self, *args, **kwargs):
            # args[0, 1, 2] are functional, gradient, control
            self.iteration += 1
            self.values.append(args[0])
            self.derivatives.append(Function(args[1][0].function_space(), name="dJdm").assign(args[1][0]))
            self.controls.append(Function(args[2][0].function_space(), name="m").assign(args[2][0]))
            return args[1]

    # Recording the callback function for later access
    inverse_problem["callback_function"] = CallBack()

    # ReducedFunctional that is to be minimised.
    inverse_problem["reduced_functional"] = ReducedFunctional(
        objective,
        control,
        derivative_cb_post=inverse_problem["callback_function"],
    )

    return inverse_problem


if __name__ == "__main__":
    if len(sys.argv) == 1:
        product_of_cases_schedulers = itertools.product(cases.keys(), schedulers.keys())
        for case_name, scheduler_name in product_of_cases_schedulers:
            minconv = annulus_taylor_test(*cases.get(case_name), schedulers.get(scheduler_name))
            print(f"case {case_name} & scheduler {scheduler_name}: result: {minconv}.")
    else:
        case_name, scheduler_name = sys.argv[1].split("_")
        weightings = cases[case_name]
        scheduler = schedulers[scheduler_name]

        # Taylor tests:
        # For taylor tests we use full memory checkpointing for fastest results
        if scheduler_name == "fullmemory":
            inverse_problem = annulus_taylor_test(*weightings, scheduler)

            log_file = ParameterLog(
                f"{case_name}_{scheduler_name}.conv",
                UnitSquareMesh(1, 1),   # Just passing a dummy mesh here
            )
            log_file.log_str(f"{inverse_problem['minconv']}")
            log_file.close()

        # Forward and Reverse tap
        # cb is a class containing values/derivatives/controls
        # val is the first objective calculation while populating the tape
        val, cb = run_forward_and_back(inverse_problem)

        # store one control and one derivative for test
        with CheckpointFile(f"{case_name}_{scheduler_name}_cb_res.h5", mode="w") as fi:
            fi.save_mesh(cb.controls[0].ufl_domain())
            fi.save_function(cb.controls[0], name="Control")
            fi.save_function(cb.derivatives[0], name="Derivative")

        # Write out the functional at tape population and functional recorded by the callback
        log_file = ParameterLog(
            f"{case_name}_{scheduler_name}_functional.dat",
            UnitSquareMesh(1, 1),
        )
        log_file.log_str(f"{val}, {cb.values[-1]}")
        log_file.close()
