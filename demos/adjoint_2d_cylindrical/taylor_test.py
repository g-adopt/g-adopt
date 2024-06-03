"""
This standalone script tests the robustness of the derivatives
using the Taylor remainder convergence test.
"""

from gadopt import *
from gadopt.inverse import *
from mpi4py import MPI
import numpy as np
import sys

from cases import cases

ds_t = ds_t(degree=6)
dx = dx(degree=6)

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_stol": 0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
    "fieldsplit_0": {
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_converged_reason": None,
    },
}


def annulus_taylor_test(case):
    """
    Perform a second-order taylor remainder convergence test
    for one term in the objective functional for the cylindrical case
    and asserts if convergence is above 1.9

    Args:
        case (string): name of the objective functional term
            either of "damping", "smooothing", "Tobs", "uobs"
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

    with CheckpointFile("Checkpoint230.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    enable_disk_checkpointing()

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Control function space
    Z = MixedFunctionSpace([V, W])  # Mixed function space

    # Test functions and functions to hold solutions:
    z = Function(Z)  # A field over the mixed function space Z
    u, p = split(z)  # Symbolic UFL expressions for u and p

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    Ra = Constant(1e7)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra, cartesian=False)

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

    # From this point, we define a depth-dependent viscosity mu
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
    epsii = sqrt(0.5 * inner(eps, eps))
    sigma_y = 1e4 + 2.0e5 * (rmax - r)
    mu_plast = 0.1 + (sigma_y / epsii)
    mu_eff = 2 * (mu_lin * mu_plast) / (mu_lin + mu_plast)
    mu = conditional(mu_eff > 0.4, mu_eff, 0.4)

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1]
    )

    stokes_bcs = {
        "top": {"un": 0},
        "bottom": {"un": 0},
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
        mu=mu,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
        solver_parameters=newton_stokes_solver_parameters,
    )

    # Control variable for optimisation
    control = Control(Tic)

    u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # If it is only for smoothing or damping, there is no need to do the time-stepping
    initial_timestep = 0 if case in ["Tobs", "uobs"] else max_timesteps

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

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    if case == "Tobs":
        objective = t_misfit
    elif case == "uobs":
        norm_u_surface = assemble(dot(uobs, uobs) * ds_t)
        objective = norm_obs * u_misfit / max_timesteps / norm_u_surface
    elif case == "damping":
        objective = norm_obs * damping / norm_damping
    else:
        objective = norm_obs * smoothing / norm_smoothing

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    # Defining the object for pyadjoint
    reduced_functional = ReducedFunctional(objective, control)

    delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    delta_temp.dat.data[:] = np.random.random(delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, delta_temp)

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()

    return minconv


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for case_name in cases:
            minconv = annulus_taylor_test(case_name)
            print(f"case: {case_name}, result: {minconv}")
    else:
        case_name = sys.argv[1]
        minconv = annulus_taylor_test(case_name)

        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(f"{case_name}.conv", "w") as f:
                f.write(f"{minconv}")
