import sys
from gadopt.transport_solver import direct_energy_solver_parameters
from gadopt.stokes_integrators import direct_stokes_solver_parameters
from gadopt import *
from gadopt.inverse import *

from cases import schedules
import gc


def rectangle_taylor_test(scheduler_name, **kwargs):

    a, b, c = 1.0079, 0.6283, 1.0
    nx, ny, nz = 20, int(b/c * 20), 20
    mesh2d = RectangleMesh(nx, ny, a, b, quadrilateral=True)  # Rectangular 2D mesh
    mesh = ExtrudedMesh(mesh2d, nz)

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    # At this point annotation should be switched off, so turn it on
    continue_annotation()

    # Only for fullstorage we need to enable disk checkpointing
    if scheduler_name == "fullstorage":
        enable_disk_checkpointing()
        mesh = checkpointable_mesh(mesh)

    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)

    X = SpatialCoordinate(mesh)


    # Enable the checkpointing scheduler
    if scheduler_name != "noscheduler":
        tape.enable_checkpointing(schedules[scheduler_name])

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "DQ", 3)
    u = Function(V, name="Velocity")
    u.assign(0.0)

    T = Function(Q, name="Temperature")

    approximation = BoussinesqApproximation(10**2)

    temp_bcs = {
        boundary.bottom: {"T": 1.0},
        boundary.top: {"T": 0.0},
    }

    # Setup Energy and Stokes solver
    energy_solver = EnergySolver(T, u, approximation, Constant(1e-9), ImplicitMidpoint, bcs=temp_bcs)

    Tic = Function(Q, name="Initial_Condition_Temperature").interpolate(
        0.5 * (erf((1 - X[1]) * 3.0) + erf(-X[1] * 3.0) + 1) +
        0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2, 0.5))) / Constant(0.1)) ** 2)
    )

    # We next make pyadjoint aware of our control problem:
    control = Control(Tic)

    # Take initial guess and project to T, simultaneously applying boundary conditions in the Q2 space:
    T.assign(Tic)

    # Next populate the tape by running the forward simulation.
    for time_idx in tape.timestepper(iter(range(2))):
        energy_solver.solve()

    # Define temperature misfit between final state solution and observation:
    t_misfit = assemble(T ** 2 * dx)

    pause_annotation()

    reduced_functional = ReducedFunctional(t_misfit, control)

    return Tic, reduced_functional


def test_a_scheduler(scheduler_name):
    Tic, rf = rectangle_taylor_test(scheduler_name)
    gc.collect()
    rf([Tic])
    gc.collect()
    rf.derivative()


if __name__ == "__main__":
    test_a_scheduler(sys.argv[1])
