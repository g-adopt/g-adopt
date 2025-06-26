from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule
import numpy as np
from firedrake.petsc import PETSc
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
    R = FunctionSpace(mesh, "R", 0)

    u_0 = Function(V).assign(1.0)
    u = Function(V).assign(u_0)
    r = Function(R)


    r.assign(2.0)
    u.project(r * u)
    r.assign(1.0)
    u.project(r * u)

    J = assemble((u) ** 2 * dx)

    pause_annotation()

    reduced_functional = ReducedFunctional(J, Control(u_0))

    # Printing diagnostics here
    J = reduced_functional(u_0)
    dJdm = reduced_functional.derivative()

    # Choosing fixed perturbation
    dtemp = Function(u_0.function_space())
    rng = np.random.default_rng(42)
    dtemp.dat.data_wo[:] = rng.random(dtemp.dat.data_ro.shape)

    dJdm = dtemp._ad_dot(dJdm)
    print(f"\tdJdm 1: {dJdm}")

    u_0.assign(u_0 + 0.1)
    Jm = reduced_functional(u_0)
    print(f"\treduced functional: {Jm}")

    dJdm = reduced_functional.derivative()
    dJdm = dtemp._ad_dot(dJdm)
    print(f"\tdJdm 2: {dJdm}")


    Jm = reduced_functional(u_0)
    print(f"\treduced functional: {Jm}")
    dJdm = reduced_functional.derivative()
    dJdm = dtemp._ad_dot(dJdm)
    print(f"\tdJdm 3: {dJdm}")


    def perturbe(eps):
        return u_0._ad_add(dtemp._ad_mul(eps))


    epsilons = [0.01 / 2 ** i for i in range(4)]
    residuals = []
    for eps in epsilons:
        Jp = reduced_functional(perturbe(eps))
        residuals.append(abs(Jp - Jm - eps * dJdm))

    for eps, res in zip(epsilons, residuals):
        print(f"\teps: {eps}, res: {res}")


match sys.argv[1]:
    case "memory":
        run_taylor_test(scheduler=SingleMemoryStorageSchedule())
    case "disk":
        run_taylor_test(scheduler=SingleDiskStorageSchedule())
    case "none":
        run_taylor_test(scheduler=None)
