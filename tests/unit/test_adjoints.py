import pytest
import pickle
from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule, StorageType
import numpy as np
from pathlib import Path


def tape_generation_staggered_solves(scheduler):
    """
    Generating a tape with staggered solves.

    Tape structure:
    - u_0 (control) -> u -> r (every 3rd step) -> u -> r -> u -> ...
    - Creates long-range dependencies: u at step i affects r at step i+3
    """

    continue_annotation()

    tape = get_working_tape()

    mesh = compatible_mesh_for_scheduler(scheduler)

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

    # Storing the diagnostics
    ret = {}
    ret["J1"] = reduced_functional(u_0)
    ret["dJdm1"] = reduced_functional.derivative().dat.data_ro.copy()

    return ret


def tape_generation_control_invariant_assign(scheduler):
    """
    Generates a tape with control-invariant assignments.

    Tape structure:
    - u_0 (control) -> u -> r.assign(2.0) -> u.project(r*u) -> r.assign(1.0) -> u.project(r*u)

    Tests pyadjoint's handling of control-independent assignments that affect later computations.
    """
    continue_annotation()

    mesh = compatible_mesh_for_scheduler(scheduler)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    u_0 = Function(V).assign(1.0)
    u = Function(V).assign(u_0)
    r = Function(R)

    r.assign(2.0)
    u.project(r * u)
    r.assign(1.0)
    u.project(r * u)

    J = assemble((u - Function(V).assign(1.0)) ** 2 * dx)

    pause_annotation()
    reduced_functional = ReducedFunctional(J, Control(u_0))

    J = reduced_functional(u_0)
    dJdm = reduced_functional.derivative()

    ret = {}
    ret["djdm_0"] = dJdm.dat.data_ro
    ret["j_0"] = J

    # Choosing fixed perturbation
    dtemp = Function(u_0.function_space())
    rng = np.random.default_rng(42)
    dtemp.dat.data_wo[:] = rng.random(dtemp.dat.data_ro.shape)

    dJdm = dtemp._ad_dot(dJdm)
    ret["djdm_1"] = dJdm

    u_0.assign(u_0 + 0.1)
    Jm = reduced_functional(u_0)
    ret["j_1"] = Jm

    dJdm = reduced_functional.derivative()
    dJdm = dtemp._ad_dot(dJdm)
    ret["djdm_2"] = dJdm

    Jm = reduced_functional(u_0)
    ret["j_2"] = Jm
    dJdm = reduced_functional.derivative()
    dJdm = dtemp._ad_dot(dJdm)
    ret["djdm_3"] = dJdm

    def perturbe(eps):
        return u_0._ad_add(dtemp._ad_mul(eps))

    epsilons = [0.01 / 2 ** i for i in range(4)]
    residuals = []
    for eps in epsilons:
        Jp = reduced_functional(perturbe(eps))
        residuals.append(abs(Jp - Jm - eps * dJdm))

    ret["epsilons"] = epsilons
    ret["residuals"] = residuals

    return ret


def tape_generation_DirichletBCc(scheduler):
    """
    Generartes a tape with updating DirichletBCs.

    Tape structure:
    - F (control) -> T.assign(T+1.0) -> solve(a==L, uu, bcs) -> T.assign(T+1.0) -> solve(a==L, uu, bcs) -> ...
    - Tests BC evaluation correctness with disk checkpointing during timestepping

    Tests pyadjoint's handling of DirichletBC with disk checkpointing in timestepper context.
    """

    continue_annotation()

    tape = get_working_tape()
    mesh = compatible_mesh_for_scheduler(scheduler)

    V = FunctionSpace(mesh, "CG", 2)
    T = Function(V).interpolate(0.0)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    x = SpatialCoordinate(mesh)
    F = Function(V)
    control = Control(F)
    F.interpolate(sin(x[0] * pi) * sin(2 * x[1] * pi))
    L = F * v * dx

    bcs = [DirichletBC(V, T, (1,))]
    uu = Function(V)

    for i in tape.timestepper(iter(range(3))):
        T.assign(T + 1.0)
        solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    obj = assemble(uu * uu * dx)
    ret = {}
    ret["first_call"] = obj
    pause_annotation()
    rf = ReducedFunctional(obj, control)
    ret["first_call"] = rf(F)

    return ret


def compatible_mesh_for_scheduler(scheduler):
    """
    Returns a mesh that is compatible with the choice of the scheduler.
    Enables disk checkpointing if the scheduler if Disk storage is enabled.
    """
    mesh = UnitSquareMesh(5, 5)

    tape = get_working_tape()
    tape.clear_tape()

    if scheduler is None:
        return mesh

    if scheduler.uses_storage_type(StorageType.DISK):
        enable_disk_checkpointing()
        mesh = checkpointable_mesh(mesh)

    tape.enable_checkpointing(scheduler)
    return mesh


def load_reference_results(func):
    """
    For each problem, load the associated benchmarks from pickle files.
    """
    if func is tape_generation_staggered_solves:
        return pickle.load(open(Path(__file__).parent.resolve() / "data" / "taylor_test_staggered_results.pkl", "rb"))
    elif func is tape_generation_control_invariant_assign:
        return pickle.load(open(Path(__file__).parent.resolve() / "data" / "taylor_test_assign_results.pkl", "rb"))
    elif func is tape_generation_DirichletBCc:
        return pickle.load(open(Path(__file__).parent.resolve() / "data" / "taylor_test_DirichletBCc_results.pkl", "rb"))
    else:
        raise ValueError(f"Unknown function: {func}")


@pytest.mark.parametrize("tape_generator", [
    (tape_generation_staggered_solves),
    (tape_generation_control_invariant_assign),
    (tape_generation_DirichletBCc),
])
@pytest.mark.parametrize("scheduler_class", [
    SingleDiskStorageSchedule,
    SingleMemoryStorageSchedule,
    None,
])
def test_control_invariant_assign(tape_generator, scheduler_class):
    """
    Unit tests for the following issues:
        control invariant assign : https://github.com/dolfin-adjoint/pyadjoint/issues/209
        staggered solves: https://github.com/dolfin-adjoint/pyadjoint/issues/211
        modifying DirichletBCs: https://github.com/firedrakeproject/firedrake/issues/4206
    """

    scheduler = None
    if scheduler_class is not None:
        scheduler = scheduler_class()

    if isinstance(scheduler, SingleMemoryStorageSchedule) and tape_generator == tape_generation_control_invariant_assign:
        pytest.xfail("pyadjoint issue #209 not yet fixed")

    if isinstance(scheduler, SingleMemoryStorageSchedule) and tape_generator == tape_generation_staggered_solves:
        pytest.xfail("pyadjoint issue #211 not yet fixed")

    reference_results = load_reference_results(tape_generator)

    taylor_test_res = tape_generator(scheduler=scheduler)

    for key in reference_results:
        if key == "using scheduler":
            continue
        assert np.allclose(taylor_test_res[key], reference_results[key], rtol=1e-10), \
            f"Values differ for key '{key}'"
