"""
This standalone script tests the robustness of the derivatives
using the Taylor remainder convergence test.
"""

import sys
from pathlib import Path

import numpy as np
from cases import cases
from mpi4py import MPI

from gadopt import *
from gadopt.inverse import *

ds_t = ds_t(degree=6)
dx = dx(degree=6)


def rectangle_taylor_test(case):
    """
    Perform a second-order Taylor remainder convergence test
    for one term in the objective functional for the rectangular case
    and asserts if convergence is above 1.9

    Args:
        case (str): name of the objective functional term
            either of "damping", "smooothing", "Tobs", "uobs"
    """
    checkpoint_path = Path(__file__).parent / "adjoint-demo-checkpoint-state.h5"
    checkpoint_file = CheckpointFile(str(checkpoint_path), "r")

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    mesh = checkpoint_file.load_mesh("firedrake_default_extruded")
    mesh.cartesian = True

    bottom_id, top_id, left_id, right_id = "bottom", "top", 1, 2

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

    approximation = EquationSystem(
        approximation="BA",
        dimensional=False,
        parameters={"Ra": 1e6},
        buoyancy_terms=["thermal"],
    )

    # Define time stepping parameters:
    max_timesteps = 80
    delta_t = Constant(4e-6)  # Constant time step

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")
    Taverage = Function(Q1, name="Average Temperature")

    # Initialise the control
    Tic.project(
        checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    )
    Taverage.project(checkpoint_file.load_function(mesh, "Average Temperature", idx=0))

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {"uy": 0},
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }
    temp_bcs = {bottom_id: {"T": 1.0}, top_id: {"T": 0.0}}

    energy_solver = EnergySolver(
        approximation,
        T,
        u,
        delta_t,
        ImplicitMidpoint,
        bcs=temp_bcs,
    )

    stokes_solver = StokesSolver(
        approximation,
        z,
        T,
        bcs=stokes_bcs,
        constant_jacobian=True,
        nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
    )

    # Control variable for optimisation
    control = Control(Tic)

    u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # If it is only for smoothing or damping, there is no need to do the time-stepping
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

    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, Delta_temp)

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()

    return minconv


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for case_name in cases:
            minconv = rectangle_taylor_test(case_name)
            print(f"case: {case_name}, result: {minconv}")
    else:
        case_name = sys.argv[1]
        minconv = rectangle_taylor_test(case_name)

        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(f"{case_name}.conv", "w") as f:
                f.write(f"{minconv}")
