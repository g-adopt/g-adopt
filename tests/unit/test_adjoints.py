import pytest
from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule
import numpy as np
from firedrake.petsc import PETSc
import pickle
from pathlib import Path
import sys


def tape_generation_control_invariant_assign(scheduler):
    """ See bellow test_control_invariant_assign for references """
    continue_annotation()
    tape = get_working_tape()
    tape.clear_tape()

    if isinstance(scheduler, SingleDiskStorageSchedule):
        enable_disk_checkpointing()

    if scheduler is not None:
        tape.enable_checkpointing(scheduler)

    mesh = UnitSquareMesh(1, 1)

    if isinstance(scheduler, SingleDiskStorageSchedule):
        mesh = checkpointable_mesh(mesh)

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

    ret["using scheduler"] = scheduler.__class__.__name__ if scheduler is not None else "None"
    return ret


def tape_generation_DirichletBCc(scheduler):
    """
    See bellow test_DirichletBCc for references
    """

    continue_annotation()

    if isinstance(scheduler, SingleDiskStorageSchedule):
        enable_disk_checkpointing()

    tape = get_working_tape()
    tape.clear_tape()

    if scheduler is not None:
        tape.enable_checkpointing(scheduler)

    mesh = UnitSquareMesh(5, 5)

    if isinstance(scheduler, SingleDiskStorageSchedule):
        mesh = checkpointable_mesh(mesh)

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


@pytest.fixture(scope="module")
def reference_results():
    """Load reference results from pickle file."""
    with open(Path(__file__).parent.resolve() / "data/taylor_test_results.pkl", "rb") as f:
        reference_taylor_test_res = pickle.load(f)
        reference_taylor_test_res.pop("using scheduler")
    return reference_taylor_test_res


@pytest.mark.parametrize("scheduler", [
    SingleMemoryStorageSchedule(),
    None
])
def test_control_invariant_assign(scheduler, reference_results):
    """
    See above tape_generation_control_invariant_assign for details.
    Test case for pyadjoint issue #209: Incorrect derivatives with SingleMemoryStorageSchedule.

    This test reproduces a bug in pyadjoint where adjoint derivatives are incorrect when using
    SingleMemoryStorageSchedule with control-independent assignments, but correct with no checkpointing.

    The bug occurs due to overly aggressive checkpoint pruning in pyadjoint's checkpointing.py.
    When a FunctionAssignBlock is control-independent and its output is used later in the graph,
    the pruning logic incorrectly discards the checkpoint because the output is not marked.
    This causes the value to be reconstructed incorrectly on the next reverse pass.

    Reference: https://github.com/dolfin-adjoint/pyadjoint/issues/209
    """

    if isinstance(scheduler, SingleMemoryStorageSchedule):
        pytest.xfail("pyadjoint issue #209 not yet fixed")

    taylor_test_res = tape_generation_control_invariant_assign(scheduler=scheduler)
    taylor_test_res.pop("using scheduler")

    for key in taylor_test_res:
        assert np.allclose(taylor_test_res[key], reference_results[key], rtol=1e-10), \
            f"Values differ for key '{key}'"


@pytest.mark.parametrize("scheduler", [
    SingleDiskStorageSchedule(),
    SingleMemoryStorageSchedule(),
    None
])
def test_DirichletBCc(scheduler):
    """Testing the fix for  https://github.com/firedrakeproject/firedrake/issues/4206"""

    taylor_test_res = tape_generation_DirichletBCc(scheduler=scheduler)
    for key, value in taylor_test_res.items():
        assert np.allclose(value, 9.000901957850, rtol=1e-10), \
            f"Values differ for key '{key}'"
