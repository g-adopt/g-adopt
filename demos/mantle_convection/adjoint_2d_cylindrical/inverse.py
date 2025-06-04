"""
This runs the optimisation portion of the adjoint test case. A forward run first sets up
the tape with the adjoint information, then a misfit functional is constructed to be
used as the goal condition for nonlinear optimisation using ROL.
"""

from gadopt import *
from gadopt.inverse import *


def main():
    inverse(alpha_u=1e-1, alpha_d=1e-2, alpha_s=1e-1)


def inverse(alpha_u, alpha_d, alpha_s):
    """
    Use adjoint-based optimisation to solve for the initial condition of the cylindrical
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        alpha_d: The coefficient of the initial condition damping term
        alpha_s: The coefficient of the smoothing term
    """

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

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

    enable_disk_checkpointing()

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    Ra = Constant(1e7)  # Rayleigh number

    # Define time stepping parameters:
    max_timesteps = 200
    delta_t = Constant(5e-6)  # Constant time step

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
        solver_parameters="direct",
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
    )

    # Control variable for optimisation
    control = Control(Tic)

    u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # Populate the tape by running the forward simulation
    for timestep in range(0, max_timesteps):
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

    objective = (
        t_misfit +
        alpha_u * (norm_obs * u_misfit / max_timesteps / norm_u_surface) +
        alpha_d * (norm_obs * damping / norm_damping) +
        alpha_s * (norm_obs * smoothing / norm_smoothing)
    )

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    # Defining the object for pyadjoint
    reduced_functional = ReducedFunctional(objective, control)

    def callback():
        initial_misfit = assemble(
            (Tic.block_variable.checkpoint.restore() - Tic_ref) ** 2 * dx
        )
        final_misfit = assemble(
            (T.block_variable.checkpoint.restore() - Tobs) ** 2 * dx
        )

        log(f"Initial misfit; {initial_misfit}; final misfit: {final_misfit}")

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(Tic.function_space(), name="Lower bound temperature")
    T_ub = Function(Tic.function_space(), name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    # Here we limit the number of optimisation iterations to 10, for CI and demo tractability.
    minimisation_parameters["Status Test"]["Iteration Limit"] = 10

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )
    optimiser.add_callback(callback)
    optimiser.run()

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()


if __name__ == "__main__":
    main()
