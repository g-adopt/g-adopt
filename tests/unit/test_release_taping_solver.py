"""Release of the taping solver after the taped forward run.

With cached solver blocks (Firedrake PR 4638) the replay and the adjoint of a
coupled low-rank GIA solve run through a cached forward solver and a cached
adjoint solver. The solver that recorded the tape ("taping solver") is not
used again, but it keeps its own preconditioner tree: a third preconditioner
set beside the two of the cached solvers.
`gadopt.dtn_coupled_adjoint.release_taping_solver` frees it. These tests
check three things on the GIAMIP Exp01 structure (low-rank DtN, moving
coastline, rotation, fluid core, three Maxwell elements on the uncondensed
layout with `CondensedBlockPC` on block 0 and the cached complement on
block 1), in a 2-D annulus with a march of `STEPS` steps:

1. The release changes no number. The taped J, the replay J, the J of
   every Taylor evaluation and every gradient vector are bit-identical to
   a run without the release, over several replays and adjoints, in the
   order of the Taylor mode of `invert_exp01.py` (replay first) and of its
   invert mode (adjoint first). A second run without the release checks
   that the baseline itself is deterministic, so that the equality means
   something.
2. The release frees the taping solver and its preconditioner tree: the
   solver object, the outer preconditioner instance of the taping solver
   and its PETSc SNES are dead after the release, the PETSc log shows the
   destroyed objects in the release stage, and at most two preconditioner
   sets are alive at any time after it (three without it).
3. The release also frees the tree when another Python wrapper of the
   taping SNES is alive (as in the monitor list of the Exp01 diagnostics),
   and any later use of the taping solver raises a clear error.

`STEPS = 6` is above the default lag of the rebuild rule (5 solves), so
every sweep crosses a lagged rebuild, as in production.
"""

import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import gc
import os
import re
import tempfile
import weakref

import firedrake as fd
import numpy as np
import pytest
from checkpoint_schedules import SingleDiskStorageSchedule
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    enable_disk_checkpointing,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from petsc4py import PETSc
from pyadjoint import Tape, set_working_tape
from pyadjoint.tape import annotate_tape

import gadopt.dtn_coupled_adjoint as dca
import gadopt.preconditioners as pcs
import test_gia_gravity
import test_gia_transient_power_law_gate as gate

#: The module fixture of `test_gia_gravity` (parent annulus, mantle
#: submesh), bound to a name of this module so that pytest finds it.
meshes = test_gia_gravity.meshes

#: The steps of the march. Above the default lag of the rebuild rule (5), so
#: the forward run, every replay and every adjoint sweep rebuild once inside
#: the sweep, as the 452-step Exp01 march does.
STEPS = 6

#: The ice of step k is `(1 - k MELT) I_init`.
MELT = 0.1

#: The step sizes of the Taylor evaluations along `gate.control_direction`.
EPSILONS = (0.1, 0.05, 0.025)

needs_cached = pytest.mark.skipif(
    not dca.CACHED_SOLVER_BLOCKS,
    reason="the release frees the taping solver only with cached solver "
           "blocks (Firedrake PR 4638)")


@pytest.fixture(autouse=True)
def fresh_tape():
    """A new tape for every test, and annotation off afterwards."""
    set_working_tape(Tape())
    yield
    if annotate_tape():
        pause_annotation()
    set_working_tape(Tape())


@pytest.fixture
def set_tracker(monkeypatch):
    """Record every `DtNTwoBlockSchurPC` that is initialised, by weak reference.

    One initialised `DtNTwoBlockSchurPC` is one preconditioner set: the outer
    preconditioner of one solver, which owns the cached complement and,
    through its inner fieldsplit, the block-0 `CondensedBlockPC` with its
    GAMG and `LowRankPotentialPC`. The tracker holds weak references only,
    so it keeps nothing alive. After every new set it records how many sets
    are alive, which gives the peak.
    """
    tracker = SetTracker()
    original = pcs.DtNTwoBlockSchurPC.initialize

    def initialize(obj, pc):
        result = original(obj, pc)
        tracker.instances.append(weakref.ref(obj))
        tracker.peak = max(tracker.peak, tracker.live())
        return result
    monkeypatch.setattr(pcs.DtNTwoBlockSchurPC, "initialize", initialize)
    return tracker


class SetTracker:
    """The weak references of the outer preconditioners and the peak count."""

    def __init__(self):
        self.instances = []
        self.peak = 0

    def live(self):
        """The number of initialised outer preconditioners alive now."""
        return sum(1 for ref in self.instances
                   if ref() is not None and getattr(ref(), "initialized",
                                                    False))

    def reset(self):
        """Forget the instances of an earlier run."""
        self.instances = []
        self.peak = 0


def collect(comm):
    """Three rounds of Python and PETSc garbage collection. Collective."""
    for _ in range(3):
        gc.collect()
        PETSc.garbage_cleanup(comm)


def checkpoint_directory(comm):
    """One directory for the disk checkpoints, named on rank 0. Collective."""
    path = tempfile.mkdtemp(prefix="gadopt-release-") if comm.rank == 0 else None
    return comm.bcast(path, root=0)


def build_case(meshes):
    """The solver, the control and the forcing of the case, untaped.

    The gate case of `test_gia_transient_power_law_gate` without the power
    law: three Maxwell elements, the uncondensed layout, the iterative preset
    of production (`CondensedBlockPC` on block 0, the cached complement on
    block 1), the moving coastline and rotation.

    Returns:
      A dict with `solver`, `SL_init` (the control), `I`, `ice`, `dt`,
      `target` and `h` (the Taylor direction).
    """
    _, sub = meshes
    with stop_annotating():
        SL_init, I_init, _ = gate.shoreline_state(sub)
        V = I_init.function_space()
        ice = [fd.Function(V, name=f"ice_{k}").interpolate(
            (1.0 - k * MELT) * I_init) for k in range(STEPS + 1)]
        I = fd.Function(V, name="ice").assign(ice[0])
        dt = fd.Function(fd.FunctionSpace(sub, "R", 0), name="dt")
        # The preset prints a converged-reason line for every block-0
        # iteration; four runs of a 6-step march make that the largest cost
        # of the test on a busy node. The keys change no number, so the
        # solver of this case is built without them (as `--diagnostics`
        # does in the Exp01 driver).
        preset = gate.selfgrav_dtn_iterative_solver_parameters

        def quiet(*args, **kwargs):
            return {key: value for key, value in preset(*args, **kwargs).items()
                    if not key.endswith("converged_reason")}
        gate.selfgrav_dtn_iterative_solver_parameters = quiet
        try:
            solver, _, _ = gate.build(meshes, SL_init, I_init, I, dt,
                                      power_law=False)
        finally:
            gate.selfgrav_dtn_iterative_solver_parameters = preset
        target = gate.misfit_target(sub)
        h = gate.control_direction(sub)
    return dict(solver=solver, SL_init=SL_init, I=I, ice=ice, dt=dt,
                target=target, h=h)


def tape_case(case, comm):
    """Tape the march of `case` as `invert_exp01.py` does; return `(rf, J)`.

    Disk checkpointing with `SingleDiskStorageSchedule(move_data=False)` and
    `gc_timestep_frequency=1`, one tape step per model step, `dt` assigned on
    the tape at the first step: the settings of the Exp01 driver.
    """
    tape = get_working_tape()
    enable_disk_checkpointing(checkpoint_comm=comm,
                              checkpoint_dir=checkpoint_directory(comm))
    tape.enable_checkpointing(SingleDiskStorageSchedule(move_data=False),
                              gc_timestep_frequency=1)
    solver = case["solver"]
    continue_annotation()
    try:
        control = Control(case["SL_init"])
        for k in tape.timestepper(iter(range(1, STEPS + 1))):
            if k == 1:
                case["dt"].assign(gate.DT)
            case["I"].assign(case["ice"][k])
            solver.solve()
        SL = solver.sea_level()
        J = fd.assemble((SL - case["target"]) ** 2
                        * solver.sea_level_measure()(gate.CURVE_RE))
    finally:
        pause_annotation()
    return ReducedFunctional(J, control), J


def perturbed(case, eps):
    """The control `SL_init + eps h`, a new Function off the tape."""
    with stop_annotating():
        m = case["SL_init"]
        return fd.Function(m.function_space()).assign(m + eps * case["h"])


def gradient_values(g):
    """The local values of a gradient (a Function or a Cofunction), copied."""
    return np.array(g.dat.data_ro, copy=True)


def run(meshes, tracker, *, release, order, stray_wrapper=False):
    """One taped run, then a sequence of replays and adjoints.

    Args:
      meshes: the module fixture.
      tracker: the `SetTracker` of the test.
      release: call `release_taping_solver` after the taped forward run.
      order: `"taylor"`: replay at the taped control, adjoint, three Taylor
        replays, one more adjoint at the last of them (the Taylor mode of
        `invert_exp01.py`, plus one adjoint at a new control); `"invert"`:
        adjoint first, then two pairs of replay and adjoint at new controls
        (the order of ROL in the invert mode, whose first evaluation of J
        is the taped value).
      stray_wrapper: keep Python wrappers of the taping SNES, its KSP and
        its outer PC alive through the release.

    Returns:
      A dict: `J` (a list of floats in the order computed), `gradients` (a
      list of arrays), `sets` (stage label to the number of live sets),
      `peak` (the largest number of live sets after each build and at each
      stage after the release; before the release, the build count), and
      with `release` the release record, `solver_alive`, `pc_alive`,
      `snes_handle_alive` and `log` (the PETSc object table of the release
      stage).
    """
    comm = meshes[1].comm
    # A new tape for every run: checkpointing must be enabled on an empty
    # tape, and the blocks of an earlier run must not be on it.
    set_working_tape(Tape())
    collect(comm)
    tracker.reset()
    case = build_case(meshes)
    rf, J = tape_case(case, comm)
    out = {"J": [float(J)], "gradients": [], "sets": {}}
    collect(comm)
    out["sets"]["after the taped forward"] = tracker.live()

    stray = None
    if release:
        gia_solver = case["solver"]
        taping = gia_solver.solver
        solver_ref = weakref.ref(taping)
        # The outer preconditioner of the taping solver is the only one
        # built during the taped run.
        assert len(tracker.instances) == 1
        pc_ref = tracker.instances[0]
        handle = taping.snes.handle
        if stray_wrapper:
            # Python wrappers that outlive the release: the SNES wrapper
            # object itself (what the monitor list of the Exp01 diagnostics
            # keeps), and new wrappers of the outer KSP and PC (what
            # `snes.ksp` or `ksp.pc` return; each holds its own PETSc
            # reference, so the KSP and the PC outlive the SNES).
            ksp = taping.snes.getKSP()
            stray = (taping.snes, ksp, ksp.getPC())
            del ksp
        del taping
        stage = PETSc.Log.Stage("release_taping_solver")
        stage.push()
        out["release"] = dca.release_taping_solver(gia_solver)
        stage.pop()
        out["solver_alive"] = solver_ref() is not None
        out["pc_alive"] = pc_ref() is not None
        # Any Python wrapper of the taping SNES that is still a live object.
        out["snes_handle_alive"] = any(
            isinstance(o, PETSc.SNES) and o.handle == handle
            for o in gc.get_objects())
        out["log"] = object_table("release_taping_solver", comm)
        out["sets"]["after the release"] = tracker.live()
        tracker.peak = tracker.live()
    else:
        tracker.peak = 0

    def stage_done(label):
        collect(comm)
        out["sets"][label] = tracker.live()
        tracker.peak = max(tracker.peak, tracker.live())

    if order == "taylor":
        out["J"].append(float(rf(case["SL_init"])))
        stage_done("after the replay")
        out["gradients"].append(gradient_values(rf.derivative()))
        stage_done("after the adjoint")
        for eps in EPSILONS:
            out["J"].append(float(rf(perturbed(case, eps))))
            stage_done(f"after Taylor eps {eps}")
        out["gradients"].append(gradient_values(rf.derivative()))
        stage_done("after the second adjoint")
    elif order == "invert":
        out["gradients"].append(gradient_values(rf.derivative()))
        stage_done("after the first adjoint")
        for eps in EPSILONS[:2]:
            out["J"].append(float(rf(perturbed(case, eps))))
            stage_done(f"after the replay eps {eps}")
            out["gradients"].append(gradient_values(rf.derivative()))
            stage_done(f"after the adjoint eps {eps}")
    else:
        raise ValueError(order)
    out["peak"] = tracker.peak
    if stray is not None:
        for wrapper in stray:
            wrapper.destroy()
    return out


def object_table(stage_name, comm):
    """The PETSc object creations and destructions of one log stage, rank 0.

    Parsed from the text of `PETSc.Log.view`. Collective.

    Returns:
      A dict from the PETSc class name ("Krylov Solver", "Preconditioner",
      "SNES", "Matrix", "Vector", ...) to `(creations, destructions)`, or an
      empty dict when logging is off or on ranks other than 0.
    """
    path = None
    if comm.rank == 0:
        handle, path = tempfile.mkstemp(suffix=".txt")
        os.close(handle)
    path = comm.bcast(path, root=0)
    viewer = PETSc.Viewer().createASCII(path, comm=comm)
    PETSc.Log.view(viewer)
    viewer.destroy()
    table = {}
    if comm.rank == 0:
        with open(path) as handle:
            text = handle.read()
        os.remove(path)
        objects = text.split("Object Type")[-1]
        sections = re.split(r"--- Event Stage \d+: ", objects)
        for section in sections[1:]:
            name, _, body = section.partition("\n")
            if name.strip() != stage_name:
                continue
            for line in body.splitlines():
                found = re.match(r"\s*(.+?)\s+(\d+)\s+(\d+)\s*$", line)
                if found:
                    table[found.group(1)] = (int(found.group(2)),
                                             int(found.group(3)))
    return table


def assert_bit_identical(a, b, label):
    """Every J and every gradient value of run `a` equals run `b` exactly."""
    assert len(a["J"]) == len(b["J"]), label
    for k, (x, y) in enumerate(zip(a["J"], b["J"])):
        assert x == y, f"{label}: J[{k}] {x!r} != {y!r}"
    assert len(a["gradients"]) == len(b["gradients"]), label
    for k, (x, y) in enumerate(zip(a["gradients"], b["gradients"])):
        assert np.array_equal(x, y), (
            f"{label}: gradient {k} differs by up to "
            f"{np.max(np.abs(x - y)):.3e}")


def print_run(label, result):
    """One line per run with the J values at 17 digits, for the report."""
    if PETSc.COMM_WORLD.rank == 0:
        values = ", ".join(f"{x:.17g}" for x in result["J"])
        norms = ", ".join(f"{np.linalg.norm(g):.17g}"
                          for g in result["gradients"])
        print(f"\n[{label}] J: {values}\n[{label}] |grad| (local): {norms}"
              f"\n[{label}] live sets: {result['sets']}, peak {result['peak']}")
        if "release" in result:
            r = result["release"]
            print(f"[{label}] release: {r['released']} {r['reason']!r}, "
                  f"rss {r['rss_gb_before']} -> {r['rss_gb_after']} GB, "
                  f"objects {r['objects_before']} -> {r['objects_after']}, "
                  f"tree {r['tree']}, {r['rounds']} cleanup rounds, "
                  f"{r['seconds']:.3f} s; "
                  f"PETSc log of the release stage "
                  f"{result['log']}")


@pytest.fixture(scope="module", autouse=True)
def petsc_logging():
    """PETSc logging on, so the log view has the object table per stage."""
    if not PETSc.Log.isActive():
        PETSc.Log.begin()
    yield


@needs_cached
@pytest.mark.parallel(nprocs=[1, 2, 4])
@pytest.mark.parametrize("order", ["taylor", "invert"])
def test_release_changes_no_number_and_frees_one_set(meshes, set_tracker,
                                                    order):
    """J and every gradient are bit-identical with the release; one set is freed.

    Four runs in one process: without the release, with it, with it and a
    stray wrapper of the taping SNES, and without it again. The last one
    checks that two runs without the release agree bit for bit, so that the
    equality of the others is a test.
    """
    base = run(meshes, set_tracker, release=False, order=order)
    print_run(f"{order} off", base)
    freed = run(meshes, set_tracker, release=True, order=order)
    print_run(f"{order} release", freed)
    # `GADOPT_RELEASE_TEST_NO_STRAY=1` skips the run with the stray wrappers,
    # so that the test fits in a short batch job with a cold kernel cache.
    stray = None
    if not os.environ.get("GADOPT_RELEASE_TEST_NO_STRAY"):
        stray = run(meshes, set_tracker, release=True, order=order,
                    stray_wrapper=True)
        print_run(f"{order} release + stray wrapper", stray)
    again = run(meshes, set_tracker, release=False, order=order)
    print_run(f"{order} off again", again)

    assert_bit_identical(base, again, "baseline twice")
    assert_bit_identical(base, freed, "release")
    if stray is not None:
        assert_bit_identical(base, stray, "release with a stray SNES wrapper")

    # Without the release all three sets stay alive to the end.
    assert base["sets"][list(base["sets"])[-1]] == 3
    for result in [r for r in (freed, stray) if r is not None]:
        assert result["release"]["released"]
        # The taping solver object and its outer preconditioner are gone.
        assert not result["solver_alive"]
        assert not result["pc_alive"]
        # Never more than the two sets of the cached solvers.
        assert result["sets"]["after the release"] == 0
        assert result["peak"] <= 2
        assert result["sets"][list(result["sets"])[-1]] == 2
        # The process-wide counts also hold the objects of the earlier runs
        # of this test, so the release is checked by the change: exactly one
        # set, with one instance of each class of its tree, is gone, and no
        # other solver object appears.
        before = result["release"]["objects_before"]
        after = result["release"]["objects_after"]
        for key in ("pc_sets", "_SchurAinvBCache", "CondensedBlockPC",
                    "LowRankPotentialPC", "DtNTwoBlockSchurPC"):
            assert before[key] - after[key] == 1, (key, before, after)
        for key in ("snes", "ksp", "pc", "mat", "vec", "dm"):
            assert after[key] <= before[key], (key, before, after)
    # Without a stray wrapper no Python object holds the taping SNES.
    assert not freed["snes_handle_alive"]
    # The PETSc log of the release stage shows the destroyed C objects:
    # at least the outer KSP and PC, the two sub-KSPs of the inner
    # fieldsplit, the condensed KSP and the KSPs of the two GAMG hierarchies
    # and of the low-rank PC; nothing new is created.
    if PETSc.COMM_WORLD.rank == 0:
        log = freed["log"]
        assert log, "no object table for the release stage in the log view"
        assert log.get("SNES", (0, 0))[1] >= 1, log
        assert log.get("Krylov Solver", (0, 0))[1] >= 5, log
        assert log.get("Preconditioner", (0, 0))[1] >= 5, log
        assert log.get("Matrix", (0, 0))[1] >= 5, log
        assert all(created == 0 for created, _ in log.values()), log


@needs_cached
def test_solve_after_release_raises(meshes):
    """A forward solve after the release stops with `TapingSolverReleasedError`."""
    comm = meshes[1].comm
    case = build_case(meshes)
    tape_case(case, comm)
    assert dca.release_taping_solver(case["solver"])["released"]
    # A second call does nothing.
    assert dca.release_taping_solver(case["solver"])["reason"] == "already released"
    with pytest.raises(dca.TapingSolverReleasedError):
        with stop_annotating():
            case["solver"].solve()


@needs_cached
def test_release_refuses_while_annotating(meshes):
    """The release refuses to run while the tape records."""
    comm = meshes[1].comm
    case = build_case(meshes)
    tape_case(case, comm)
    continue_annotation()
    try:
        with pytest.raises(RuntimeError, match="annotation is on"):
            dca.release_taping_solver(case["solver"])
    finally:
        pause_annotation()


def test_release_does_nothing_without_cached_blocks(monkeypatch):
    """On a Firedrake without cached solver blocks the release is a no-op.

    It returns before it reads anything of the solver, so any object will do.
    """
    monkeypatch.setattr(dca, "CACHED_SOLVER_BLOCKS", False)
    result = dca.release_taping_solver(object())
    assert not result["released"]
    assert "no cached solver blocks" in result["reason"]


def test_release_does_nothing_before_a_taped_solve(meshes):
    """A solver with no coupled block on the tape keeps its taping solver."""
    if not dca.CACHED_SOLVER_BLOCKS:
        pytest.skip("covered by the no-op test without cached blocks")
    case = build_case(meshes)
    with stop_annotating():
        case["dt"].assign(gate.DT)
        case["I"].assign(case["ice"][1])
        case["solver"].solve()
    result = dca.release_taping_solver(case["solver"])
    assert not result["released"]
    assert result["reason"] == "no coupled solve block on the tape"
    # The solver still works.
    with stop_annotating():
        case["solver"].solve()
