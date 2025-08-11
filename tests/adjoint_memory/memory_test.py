import sys
from gadopt.transport_solver import direct_energy_solver_parameters
from gadopt.stokes_integrators import direct_stokes_solver_parameters
from gadopt import *
from gadopt.inverse import *

from cases import schedules
import gc


def rectangle_taylor_test(scheduler_name, **kwargs):
    # Making the mesh
    mesh1d = IntervalMesh(150, length_or_left=0.0, right=1.0)
    mesh = ExtrudedMesh(
        mesh1d, layers=150, layer_height=1. / 150, extrusion_type="uniform"
    )

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
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 2)
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    T = Function(Q, name="Temperature")

    # Specify important constants for the problem, alongside the approximation:
    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    # Nullspaces for the problem are next defined:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Followed by boundary conditions, noting that all boundaries are free slip, whilst the domain is
    # heated from below (T = 1) and cooled from above (T = 0).
    stokes_bcs = {
        boundary.bottom: {"uy": 0},
        boundary.top: {"uy": 0},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
    }
    temp_bcs = {
        boundary.bottom: {"T": 1.0},
        boundary.top: {"T": 0.0},
    }

    # Setup Energy and Stokes solver
    energy_solver = EnergySolver(T, u, approximation, Constant(4e-6), ImplicitMidpoint, bcs=temp_bcs, solver_parameters=direct_energy_solver_parameters)
    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        constant_jacobian=True,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        solver_parameters=direct_stokes_solver_parameters,
    )

    Tic = Function(Q, name="Initial_Condition_Temperature").interpolate(
        0.5 * (erf((1 - X[1]) * 3.0) + erf(-X[1] * 3.0) + 1) +
        0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2))) / Constant(0.1)) ** 2)
    )

    # We next make pyadjoint aware of our control problem:
    control = Control(Tic)

    # Take initial guess and project to T, simultaneously applying boundary conditions in the Q2 space:
    T.assign(Tic)

    # Next populate the tape by running the forward simulation.
    for time_idx in tape.timestepper(iter(range(50))):
        stokes_solver.solve()
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
