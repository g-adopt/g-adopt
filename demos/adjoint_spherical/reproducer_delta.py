from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule
import numpy as np
import sys


def run_taylor_test(scheduler):
    print(f"using scheduler: {scheduler.__class__.__name__}")

    # get tape
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    if scheduler is not None:
        tape.enable_checkpointing(scheduler)

    mesh = SquareMesh(1, 1, 1, quadrilateral=True)

    V = FunctionSpace(mesh, "CG", 1)

    u_0 = Function(V).assign(1.0)
    u = Function(V).assign(u_0)
    r = Function(V)

    for i in tape.timestepper(iter(range(10))):
        if i % 3 == 0:
            r.project(1.01 * u)
        u.project(r * u)

    J = assemble((u) ** 2 * dx)

    pause_annotation()

    reduced_functional = ReducedFunctional(J, Control(u_0))

    # Printing diagnostics here
    J = reduced_functional(u_0)
    print(f"\tJ1: {J}")
    dJdm = reduced_functional.derivative()
    print(f"\tdJdm 1: {sqrt(assemble(dJdm ** 2 * dx))}")

match sys.argv[1]:
    case "memory":
        run_taylor_test(scheduler=SingleMemoryStorageSchedule())
    case "disk":
        run_taylor_test(scheduler=SingleDiskStorageSchedule())
    case "none":
        run_taylor_test(scheduler=None)
