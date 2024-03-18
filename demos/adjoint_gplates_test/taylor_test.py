"""
This standalone script tests the robustness of the derivatives
using the Taylor remainder convergence test.
"""

from gadopt import *
from gadopt.inverse import *
from gadopt.gplates import pyGplatesConnector, GplatesFunction
from mpi4py import MPI
import numpy as np
import sys

from cases import cases

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

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile("mesh.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

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

    # Initiating a plate reconstruction model
    rec_model = pyGplatesConnector(
        rotation_filenames=[
            '../gplates_global/gplates_files/Zahirovic2022_CombinedRotations_fixed_crossovers.rot'],
        topology_filenames=[
            '../gplates_global/gplates_files/Zahirovic2022_PlateBoundaries.gpmlz',
            '../gplates_global/gplates_files/Zahirovic2022_ActiveDeformation.gpmlz',
            '../gplates_global/gplates_files/Zahirovic2022_InactiveDeformation.gpmlz'],
        nseeds=1e5,
        nneighbours=4,
        geologic_zero=409,
        delta_time=0.9
    )

    gplate_velocities = GplatesFunction(V, name="TopVelocities", gplates_connector=rec_model, top_boundary_marker=top_id)
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
        constant_jacobian=True,
    )

    # Control variable for optimisation
    control = Control(Tic)

    u_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # If it is only for smoothing or damping, there is no need to do the time-stepping
    initial_timestep = max_timesteps - 2  if case in ["Tobs", "uobs"] else max_timesteps - 1

    # Populate the tape by running the forward simulation
    for timestep in range(initial_timestep, max_timesteps):
        gplate_velocities.update_plate_reconstruction(timestep*1000)
        stokes_solver.solve()
        energy_solver.solve()

        # Update the accumulated surface velocity misfit using the observed value
        uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=timestep)
        u_misfit += assemble(dot(u - gplate_velocities, u - gplate_velocities) * ds_t)
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
    norm_obs = assemble(Tobs**2 * dx)
    norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

    # Defining objective functional
    objective = norm_obs * u_misfit / max_timesteps / norm_u_surface
    log("First Call", objective)
    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    # Defining the object for pyadjoint
    reduced_functional = ReducedFunctional(objective, control)

    log("Repeat Call", reduced_functional([Tic]))

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
