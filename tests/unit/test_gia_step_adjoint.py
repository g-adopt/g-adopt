"""The recorded time step of `SelfGravitatingGIASolver` on the tape.

Every step kind with a block of G-ADOPT's own (the Quadrature history with
either DtN representation) records the same structure: a sweep marker, the step block, a sweep marker
and the kernel projection (`gadopt.gia_step_adjoint`). The tests here check
that structure and that a finished run leaves no solver alive once the tape
is gone, with either preset.

The tangent sweep of the Quadrature step block (against the adjoint and
against a difference of replays) and the refusal of its second-order adjoint
are rows of the `adjoint` case in `tests/gia` (`sea-direct-viscosity`,
`load-direct-Lambda`, `sea-direct-dt` and `guards`). The gradients
themselves are checked there and by the `Real`-control Taylor tests of
`test_gia_solver.py`.
"""
import gc
import weakref

import firedrake as fd
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from petsc4py import PETSc
from pyadjoint.tape import annotate_tape

import gia_cases as C
from test_gia_solver import STEPS, march

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest

Case = C.Case

#: The step kinds with a step block of G-ADOPT's own.
STEP_KINDS = [
    pytest.param(Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                      model="maxwell2", core="fluid", centre=True),
                 id="quadrature-lowrank"),
    pytest.param(Case(family="Quadrature", representation="multiplier",
                      core="fluid", centre=True), id="quadrature-multiplier"),
]


@pytest.fixture(scope="module")
def meshes():
    return C.annulus_meshes()


@pytest.fixture(autouse=True)
def clean_tape():
    get_working_tape().clear_tape()
    yield
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


def _field_control(sub):
    """The viscosity control of `test_gia_solver.march` and a direction."""
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, "CG", 1)).assign(.1)
        direction = fd.Function(control.function_space()).interpolate(
            .1 + .03 * fd.SpatialCoordinate(sub)[0])
    return control, direction


@pytest.mark.parametrize("case", STEP_KINDS)
def test_every_step_kind_records_the_same_structure(meshes, case):
    """Marker, step block, marker, projection: once per solve, in this order."""
    from gadopt.gia_step_adjoint import GIAStepBlockMixin, TapeSweepMarkerBlock
    from gadopt.nullspaces import KernelProjectionBlock

    control, _ = _field_control(meshes[1])
    continue_annotation()
    march(meshes, case, control, steps=STEPS[:2])
    pause_annotation()
    blocks = get_working_tape().get_blocks()
    steps = [i for i, b in enumerate(blocks) if isinstance(b, GIAStepBlockMixin)]
    assert len(steps) == 2, [type(b).__name__ for b in blocks]
    for index, i in enumerate(steps, start=1):
        before, step, after, projection = blocks[i - 1:i + 3]
        assert isinstance(before, TapeSweepMarkerBlock) and before.before
        assert isinstance(after, TapeSweepMarkerBlock) and not after.before
        assert isinstance(projection, KernelProjectionBlock)
        # The step block carries the index of its forward solve; the
        # markers around it the same index.
        assert step.gia_solve_index == before.solve_index == after.solve_index
        # The projection is the one of the step solution: its input is the
        # output of the step block.
        assert any(dep is step.get_outputs()[0]
                   for dep in projection.get_dependencies())
    # One projection per step: no step block projects inside its own map.
    assert sum(isinstance(b, KernelProjectionBlock) for b in blocks) == 2


def _central_difference(rf, control, direction, eps=1e-4):
    """`(J(m + eps h) - J(m - eps h)) / (2 eps)` by two replays of the tape.

    Independent of the adjoint and of the tangent sweep: it uses only the
    replay of the step blocks.
    """
    with stop_annotating():
        plus = fd.Function(control.function_space()).assign(control + eps * direction)
        minus = fd.Function(control.function_space()).assign(control - eps * direction)
    return (float(rf(plus)) - float(rf(minus))) / (2 * eps)


#: The configurations of `test_a_finished_run_is_collected`: the step kind,
#: the preset and whether the run is taped. The `"direct"` preset puts
#: `firedrake.AssembledPC` (which holds the solver's `_SNESContext` and so its
#: callbacks) in the PETSc tree; the iterative preset puts the Python
#: preconditioners that hold the application context there.
_COLLECTED_CASES = [
    pytest.param(STEP_KINDS[0].values[0], "iterative", True, id="quadrature-lowrank-iterative"),
    pytest.param(STEP_KINDS[0].values[0], "direct", True, id="quadrature-lowrank-direct"),
    pytest.param(STEP_KINDS[0].values[0], "direct", False,
                 id="quadrature-lowrank-direct-untaped"),
    pytest.param(STEP_KINDS[1].values[0], "direct", True, id="quadrature-multiplier-direct"),
]


@pytest.mark.parametrize("case,preset,taped", _COLLECTED_CASES)
def test_a_finished_run_is_collected(meshes, case, preset, taped):
    """No solver of a finished run stays alive once the tape is gone.

    PETSc holds the Python preconditioner contexts from C. Two routes lead
    from such a context back to the solver: the application context, whose
    low-rank DtN operator reads `theta_psi` from the solver (iterative
    preset), and the `_SNESContext` that `firedrake.AssembledPC` holds,
    whose callbacks are methods of the solver and of the Quadrature backend
    (direct preset). If either route held the solver strongly, the solver,
    its taping solver, the cached replay and derivative solvers and all
    their preconditioner trees and factorisations would form a cycle
    through C that Python's collector cannot see, and stay alive to the end
    of the process. Both routes hold the solver weakly
    (`gadopt.utility.weak_callback`).
    """
    control, _ = _field_control(meshes[1])
    with stop_annotating():
        dt = C.dt_function(meshes[1])
        solver, u, psi = C.build_new(meshes, case, control=control, dt=dt,
                                     nullspace="rigid_rotation",
                                     solver_parameters=preset)
    if taped:
        continue_annotation()
    for h in STEPS[:2]:
        dt.assign(h)
        solver.solve()
    J = fd.assemble(fd.inner(u, u) * fd.dx(domain=meshes[1]))
    if taped:
        rf = ReducedFunctional(J, Control(control))
        pause_annotation()
        rf.derivative()
        del rf
    alive = weakref.ref(solver)
    del solver, u, psi, J
    get_working_tape().clear_tape()
    gc.collect()
    PETSc.garbage_cleanup(meshes[1].comm)
    gc.collect()
    assert alive() is None


# -- The start of the step and the solved state under a checkpoint schedule ------

def _march_without_datum(meshes, case, control, steps, schedule, directory):
    """A taped march in which no block of a step reads the solution of the step before.

    The march of `test_gia_solver.march` with one change: the refresh of the
    2-D monopole datum is replaced by a no-op. In 2-D the recorded datum
    solve of step k+1 reads the mixed solution of step k (the fluid-core
    sheet). A 3-D march has no datum, so there only the step block itself
    can read the mixed solution of the step before. This march has the
    dependencies of the 3-D march on the annulus. It is self-consistent:
    the forward run, the replay and the adjoint use the same frozen datum.

    Args:
      meshes: the parent annulus and its mantle submesh.
      case: the configuration.
      control: the viscosity control.
      steps: the step lengths.
      schedule: "memory" or "disk", the single-storage checkpoint schedule.
      directory: the directory of the disk checkpoints (`schedule="disk"`).

    Returns:
      `(J, solver)`: the functional of `test_gia_solver.march` and the solver.
    """
    from gadopt import gia_couplings as cp
    from test_gia_solver import surface_ice
    parent, sub = meshes
    with stop_annotating():
        sea = C.SeaFields(sub)
        dt = C.dt_function(sub)
        solver, u, psi = C.build_new(meshes, case, control=control, dt=dt, sea=sea,
                                     nullspace="rigid_rotation")
    solver.update_total_mass = lambda: None
    tape = get_working_tape()
    if schedule == "disk":
        from checkpoint_schedules import SingleDiskStorageSchedule
        from firedrake.adjoint import enable_disk_checkpointing
        if sub.comm.rank == 0:
            directory.mkdir()
        sub.comm.barrier()
        enable_disk_checkpointing(checkpoint_comm=sub.comm, checkpoint_dir=str(directory))
        tape.enable_checkpointing(SingleDiskStorageSchedule(), gc_timestep_frequency=1)
    else:
        from checkpoint_schedules import SingleMemoryStorageSchedule
        tape.enable_checkpointing(SingleMemoryStorageSchedule())
    for k, h in tape.timestepper(iter(enumerate(steps))):
        dt.assign(h)
        surface_ice(sea, k)
        solver.solve()
    dx_m = fd.Measure("dx", domain=sub, metadata={"quadrature_degree": 4})
    history = solver.internal_variables
    J = fd.assemble((fd.inner(u, u) + .1 * fd.inner(history, history)) * dx_m)
    J += fd.assemble(psi * psi * fd.dx(domain=parent))
    sea_level = [c for c in solver.couplings if isinstance(c, cp.SeaLevel)]
    if sea_level:
        J += fd.assemble(sea_level[0].sea_level() ** 2 * sea_level[0].measure())
    return J, solver


@pytest.mark.parametrize("schedule", ["memory", "disk"])
def test_the_gradient_after_a_replay_is_right_when_only_the_step_reads_the_solution(
        meshes, schedule, tmp_path):
    """The step block reads the mixed solution, and the gradient after a replay is right.

    The adjoint of a step linearises about the solved state of that step,
    the saved value of the first output of its block. Under a checkpoint
    schedule pyadjoint keeps that value across a replay only if a later
    block on the path to the functional reads it or a value computed from
    it. With the Quadrature history the next step reads the history field,
    so the step block must read the mixed solution itself (the start of its
    Newton solve), as Firedrake's solve block does on the DG family.

    Two checks, on a march of three steps without the 2-D datum
    (`_march_without_datum`):

    1. Every step block after the first has the output of the kernel
       projection of the step before among its dependencies.
    2. After a gradient and a replay at a second control value, the adjoint
       derivative along a direction equals the central difference of two
       replays about that value, to 1e-5 relative. The central difference
       uses no adjoint. Without the dependency the adjoint linearises about
       the solved state of an earlier run of the tape, and the two differ
       by 0.20 relative under the memory schedule and by 2.0e-4 under the
       disk schedule on this march.
    """
    from gadopt.gia_step_adjoint import GIAStepBlockMixin
    from gadopt.nullspaces import KernelProjectionBlock

    case = STEP_KINDS[0].values[0]
    control, direction = _field_control(meshes[1])
    continue_annotation()
    J, solver = _march_without_datum(meshes, case, control, STEPS[:3], schedule,
                                     tmp_path / "checkpoints")
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()

    blocks = get_working_tape().get_blocks()
    steps = [b for b in blocks if isinstance(b, GIAStepBlockMixin)]
    projections = [b for b in blocks if isinstance(b, KernelProjectionBlock)]
    assert len(steps) == len(projections) == 3
    for step, projection in zip(steps[1:], projections[:-1]):
        projected = projection.get_outputs()[0]
        assert projected.output is solver.solution
        assert any(dep is projected for dep in step.get_dependencies()), (
            "the step block does not read the mixed solution of the step before")

    with stop_annotating():
        other = fd.Function(control.function_space()).assign(control + .01 * direction)
    # The sequence of an optimisation: gradient, replay at a new control,
    # gradient. The first adjoint sweep is needed: in it pyadjoint works out
    # which saved values a later replay keeps, and before it a replay keeps
    # them all.
    rf.derivative()
    rf(other)
    adjoint = float(direction._ad_dot(rf.derivative()))
    difference = _central_difference(rf, other, direction)
    assert adjoint == pytest.approx(difference, rel=1e-5), (schedule, adjoint, difference)
