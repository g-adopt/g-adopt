"""Driver of the `adjoint` case: taped marches of the coupled GIA solver.

Usage: `python gia_adjoint.py --entry <name>`, with the entry names of
`gia_entries.ADJOINT` and `gia_entries.GUARDS`.

An entry of `ADJOINT` tapes one march of the adjoint scenario
(`gia_entries.ADJOINT_STEPS`, the ice `.002 (1 + .2 k) (1 - cos theta) / 2`
before step `k`) for one configuration and one control, and measures the
tape against independent evaluations:

- the replay of the tape at the recorded control and a fresh, untaped march
  there, against the recorded functional;
- the replay and a fresh march at `control + ADJOINT_OTHER_STEP * direction`;
- the directional derivative from the adjoint against the central
  difference of two fresh marches (`ADJOINT_FRESH_EPS`);
- the Taylor remainders of the replay with and without the gradient;
- the tangent sweep against the adjoint pairing and against the central
  difference of two replays (`ADJOINT_REPLAY_EPS`);
- the number of tape blocks of each step and the build counters of the
  cached preconditioners during the taped march, the first replay and the
  adjoint sweep (`CounterProbeBlock`).

The functional is the one of `tests/unit/test_gia_solver.py::march`:

    J = int_mantle (u . u + 0.1 h : h) dx + int_parent psi^2 dx + int SL^2 dS,

with `h` the history at the quadrature points and the last term only for a
case with a sea level. The values go as `name value` lines to
`out/adjoint-<entry>.dat` (`gia_entries.ADJOINT_VALUES`), the counts to
`out/adjoint-<entry>-counts.dat`. The test `test_gia_adjoint.py` reads both
and asserts; this script asserts nothing.

An entry of `REPLAY_GRADIENT` tapes one march of the same scenario with
the 2-D datum frozen (`FrozenDatumSolver`) under a single-storage checkpoint
schedule, and runs the sequence of an optimisation: gradient, replay,
gradient (`run_replay_gradient`). It writes the gradient after the replay,
the central difference of two replays, and the Newton iterations of each
step in the taped march and in the replay.

The entry `guards` runs the unit tests that need a taped solve and writes 1
or 0 per guard, with the numbers behind each flag (`gia_entries.GUARD_FLAGS`).
"""

import argparse
import gc
import math
import shutil
import tempfile
import time
import weakref
from dataclasses import replace

import gadopt  # imported before firedrake
import firedrake as fd
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation, stop_annotating)
from petsc4py import PETSc
from pyadjoint import Block
from pyadjoint.tape import annotate_tape

import gia_cases as C
import gia_march as M
from gia_entries import (ADJOINT, ADJOINT_FRESH_EPS, ADJOINT_OTHER_STEP,
                         ADJOINT_REPLAY_EPS, ADJOINT_STEPS, CASE_DIR, COUNT_PHASES,
                         COARSE_OUTER_RTOL, GUARD_FLAGS, GUARDS, LOOSE_OUTER_RTOL, OUT,
                         REPLAY_GRADIENT, REPLAY_RTOL,
                         SCALED_GRADIENT_RATE_MAX, TAYLOR_RATE_MIN, adjoint_blocks_name,
                         adjoint_count_name, adjoint_counts, adjoint_values, newton_name)
from gadopt import gia_couplings as cp
from gadopt.gia_solver import SelfGravitatingGIASolver


# -- Controls ----------------------------------------------------------------------

def ice_shape_expression(sub):
    """`.002 (1 - cos theta) / 2`: the ice before the first step, a cap with a grounding line."""
    X = fd.SpatialCoordinate(sub)
    return .002 * (1 - fd.cos(fd.atan2(X[1], X[0]))) / 2


def control_and_direction(kind, meshes):
    """The control at its recorded value and the direction of the derivative checks.

    The directions are smooth fields of the size of a tenth to a fifth of
    the control, so that the Taylor steps `0.01 / 2^i` move the functional
    well above round-off and the step `ADJOINT_OTHER_STEP` stays a small
    change of the march (Newton converges from the previous state at every
    step).

    Args:
      kind: a name of `gia_entries.ADJOINT_CONTROLS`.
      meshes: the parent annulus and its mantle submesh.

    Returns:
      `(control, direction)`, two Functions on the same space.
    """
    parent, sub = meshes
    X = fd.SpatialCoordinate(sub)
    theta = fd.atan2(X[1], X[0])
    with stop_annotating():
        if kind == "viscosity":
            # The log of a factor on every viscosity (`gia_cases.approximation`).
            space = fd.FunctionSpace(sub, "CG", 1)
            values = (.1, .1 + .03 * X[0])
        elif kind == "ice":
            # The shape of the ice; the direction thickens the cap unevenly.
            space = fd.FunctionSpace(sub, "CG", 2)
            values = (ice_shape_expression(sub),
                      .0004 * (1 - fd.cos(theta)) / 2 * (1 + .5 * fd.sin(2 * theta)))
        elif kind == "dt":
            # A factor on every step length.
            space = fd.FunctionSpace(sub, "R", 0)
            values = (1.0, .3)
        elif kind == "Lambda":
            space = fd.FunctionSpace(parent, "R", 0)
            values = (C.LAMBDA, .1 * C.LAMBDA)
        elif kind == "B_mu":
            space = fd.FunctionSpace(parent, "R", 0)
            values = (C.B_MU, .1 * C.B_MU)
        elif kind == "sl_init":
            # The reference sea level of `gia_cases.SeaFields`; the direction
            # moves the coastline and has no symmetry of the annulus.
            space = fd.FunctionSpace(sub, "CG", 2)
            values = (.005 * fd.cos(theta),
                      .001 * (fd.sin(theta) + .5 * fd.cos(2 * theta)))
        else:
            raise ValueError(kind)
        control, direction = (fd.Function(space) for _ in range(2))
        for f, value in zip((control, direction), values):
            if isinstance(value, float):
                f.assign(value)
            else:
                f.interpolate(value)
    return control, direction


def shifted(control, direction, eps):
    """`control + eps * direction` as a new Function, untaped."""
    with stop_annotating():
        out = control.copy(deepcopy=True)
        out.assign(control + eps * direction)
    return out


# -- Build counters through the tape ------------------------------------------------

class CounterProbeBlock(Block):
    """A tape block that reads the build counters of a solver whenever a sweep passes.

    It has no dependencies and no outputs, so it changes no value on the
    tape. `march` puts one after each step. A replay reaches it after the
    solve of its step, an adjoint sweep before the adjoint solve of its
    step (the sweep runs the tape backwards). Each visit appends
    `(phase, forward counters, adjoint counters)` to `log`: the counters of
    the PC tree of the nonlinear solver of the step, which the replay uses,
    and of the linear adjoint solve of the Quadrature step
    (`DirectConstitutiveBackend.derivative_solvers`), which exists after
    the first adjoint solve.

    Args:
      log: the list the readings are appended to.
      solver: the `SelfGravitatingGIASolver` of the march.
    """

    def __init__(self, log, solver):
        super().__init__()
        self.log = log
        self.solver = solver

    def __str__(self):
        return "CounterProbeBlock"

    def read(self):
        """The forward and the adjoint counters, two `{name: count}`."""
        adjoint = {}
        solvers = self.solver._step.backend.derivative_solvers
        if "adjoint" in solvers:
            adjoint = M.pc_counters(solvers["adjoint"][1].ksp.pc)
        return M.preconditioner_counters(self.solver), adjoint

    # pyadjoint calls these four for every block of a sweep; the stock
    # versions return early for a block without outputs, so they are
    # replaced (as in `gadopt.gia_step_adjoint.TapeSweepMarkerBlock`).
    def recompute(self, markings=False):
        self.log.append(("replay", *self.read()))

    def evaluate_adj(self, markings=False):
        self.log.append(("adjoint", *self.read()))

    def evaluate_tlm(self, markings=False):
        pass

    def evaluate_hessian(self, markings=False):
        pass


def increases(readings, before):
    """The increase of each counter over each step, from cumulative readings.

    Args:
      readings: one `{name: count}` per step, read after the step.
      before: `{name: count}` before the first step; a missing name is 0.

    Returns:
      One `{name: increase}` per step, over the names of the last reading.
    """
    names = sorted(readings[-1]) if readings else []
    out, last = [], before
    for reading in readings:
        out.append({n: reading.get(n, 0) - last.get(n, 0) for n in names})
        last = reading
    return out


def phase_counts(forward, log, steps):
    """The increases per step of the forward march, the replay and the adjoint sweep.

    Args:
      forward: the forward counters after each step of the taped march.
      log: the readings of the `CounterProbeBlock`s during one replay and
        the adjoint sweep after it, followed by one reading after the sweep
        (phase `"end"`).
      steps: the number of steps.

    Returns:
      `{phase: [{counter: increase} per step]}` for `COUNT_PHASES`.
    """
    replay = [r[1] for r in log if r[0] == "replay"]
    assert len(replay) == steps, (len(replay), steps)
    # The adjoint sweep visits the probe of step k before the adjoint solve
    # of step k, from the last step down. The reading after the adjoint
    # solve of step k is the probe of step k - 1, or the final reading for
    # the first step.
    adjoint = [r[2] for r in log if r[0] == "adjoint"]
    assert len(adjoint) == steps, (len(adjoint), steps)
    final = log[-1][2]
    # Index k - 1 holds the reading before, and after, the adjoint solve of step k.
    before_step = adjoint[::-1]
    after_step = [final] + before_step[:-1]
    adjoint_increase = []
    names = sorted(final)
    for before, after in zip(before_step, after_step):
        adjoint_increase.append({n: after.get(n, 0) - before.get(n, 0) for n in names})
    return {"forward": increases(forward, {}),
            "replay": increases(replay, forward[-1]),
            "adjoint": adjoint_increase}


class NewtonProbeBlock(Block):
    """A tape block that reads the Newton iterations of the last step solve in a replay.

    It has no dependencies and no outputs, so it changes no value on the
    tape. `Scenario.march` puts one after each step. A replay reaches it
    after the solve of its step and appends the Newton iteration count of
    that solve to `log` (`newton_iterations`). The adjoint and tangent
    sweeps leave it alone.

    Args:
      log: the list the counts are appended to.
      solver: the `SelfGravitatingGIASolver` of the march.
    """

    def __init__(self, log, solver):
        super().__init__()
        self.log = log
        self.solver = solver

    def __str__(self):
        return "NewtonProbeBlock"

    # The stock versions return early for a block without outputs, so they
    # are replaced (as in `CounterProbeBlock`).
    def recompute(self, markings=False):
        self.log.append(newton_iterations(self.solver))

    def evaluate_adj(self, markings=False):
        pass

    def evaluate_tlm(self, markings=False):
        pass

    def evaluate_hessian(self, markings=False):
        pass


def newton_iterations(solver):
    """The Newton iterations of the last step solve, forward or replayed.

    The forward step and its replay solve with the same variational solver
    (`SelfGravitatingGIASolver.solver`), so its SNES holds the count of the
    last of the two.
    """
    return int(solver.solver.snes.getIterationNumber())


# -- The solver with a frozen datum ----------------------------------------------------

class FrozenDatumSolver(SelfGravitatingGIASolver):
    """`SelfGravitatingGIASolver` with the 2-D monopole datum frozen at its initial value.

    Before each step, `SelfGravitatingGIASolver.solve` refreshes the 2-D
    monopole datum (`update_total_mass`) by recorded solves that read the
    mixed solution of the step before, because the fluid-core sheet enters
    the datum one solve late. A 3-D march has no such datum, so there no
    block but the step block reads the mixed solution of the step before.
    This subclass refreshes nothing, so the enclosed-mass scalars of the
    datum keep their initial value zero in the forward run, in every replay
    and in the adjoint, and the annulus march has the tape dependencies of
    the 3-D march. The march solves another problem than the one of
    `SelfGravitatingGIASolver`, and it is consistent with itself. Only the
    `REPLAY_GRADIENT` entries use it.
    """

    def update_total_mass(self) -> None:
        """Keep the datum at its initial value: solve and record nothing."""


# -- The march -----------------------------------------------------------------------

def zero_start(solver):
    """Make zero the starting point of the next step solve, as a user does: `u` and `psi`.

    The step solve starts from `u` and `psi` (`gadopt.gia_solver`, "The
    contract of `u`, `psi` and the history"), and the two assignments are
    on the tape. The step block reads the mixed solution, so a replay
    starts each step from the same point as the forward run: zero in `u`
    and `psi`, and the previous values in the `Real` fields of the
    couplings.

    Args:
      solver: the `SelfGravitatingGIASolver`.
    """
    solver.u.assign(0)
    solver.psi.assign(0)


class Scenario:
    """The configuration of one entry and the march of the adjoint scenario.

    Args:
      meshes: the parent annulus and its mantle submesh.
      spec: an entry of `gia_entries.ADJOINT` (or a dictionary with the
        same keys).
      preset: the solver preset, replacing the one of `spec`.
    """

    def __init__(self, meshes, spec, preset=None):
        self.meshes = meshes
        self.case = replace(getattr(M, spec["base"]), **spec.get("changes", {}))
        self.preset = preset or spec["preset"]
        self.default_options = spec.get("default_options", True)
        self.parameters = dict(spec.get("parameters", {}))
        self.kind = spec["control"]
        #: Start every step from zero (`gia_entries.ADJOINT`, `zero_start`).
        self.zero_start = spec.get("zero_start", False)
        #: Build a `FrozenDatumSolver` (`gia_entries.REPLAY_GRADIENT`).
        self.frozen_datum = spec.get("frozen_datum", False)
        with stop_annotating():
            # The fixed ice shape of the entries whose control is not the ice.
            self.ice_shape = fd.Function(fd.FunctionSpace(meshes[1], "CG", 2)).interpolate(
                ice_shape_expression(meshes[1]))

    def build(self, control, sea, **kwargs):
        """Build the solver with `control` in its place, untaped (`gia_march.build`)."""
        parameters = dict(self.parameters)
        if self.kind == "Lambda":
            parameters["approximation.self_gravity_number"] = control
        elif self.kind == "B_mu":
            parameters["approximation.B_mu"] = control
        if self.kind == "viscosity":
            kwargs["control"] = control
        if self.frozen_datum:
            kwargs["solver_class"] = FrozenDatumSolver
        return M.build(self.meshes, self.case, preset=self.preset,
                       default_options=self.default_options, sea=sea,
                       parameters=parameters or None, **kwargs)

    def march(self, control, *, steps=ADJOINT_STEPS, blocks=None, forward=None,
              probe=None, newton=None, schedule=False, restore_after=None, **kwargs):
        """March `steps` from rest with `control`; taped if annotation is on.

        Args:
          control: the control Function: the one of the `Control` for the
            taped march, a copy at another value for a fresh march.
          steps: the step lengths.
          blocks: a list that gets the number of tape blocks of each step.
          forward: a list that gets the counters of the forward solve after
            each step.
          probe: a list for the readings of a `CounterProbeBlock` put on
            the tape after each step, or `None` for no probe.
          newton: `None`, or two lists `(forward, replay)`: `forward` gets
            the Newton iterations of each step of this march, and a
            `NewtonProbeBlock` put on the tape after each step appends the
            Newton iterations of the step to `replay` in every replay.
          schedule: mark the steps with `tape.timestepper`, for a
            checkpoint schedule.
          restore_after: after this many steps, copy the state, build a
            second solver and `restore` the copy into it, on the tape
            (`test_gia_solver.py::march`).
          kwargs: further arguments of `gia_cases.build_new`.

        Returns:
          `(J, run)`: the functional and the `gia_march.Run` of the last
          solver.
        """
        sub = self.meshes[1]
        sea = None
        if self.case.load in ("sea_fixed", "sea_moving"):
            with stop_annotating():
                sea = C.SeaFields(sub)
            if self.kind == "sl_init":
                # The reference sea level enters the residual of every step.
                sea.SL_init = control
        ice_shape = control if self.kind == "ice" else self.ice_shape
        run = self.build(control, sea, **kwargs)
        tape = get_working_tape()
        iterator = enumerate(steps)
        if schedule and annotate_tape():
            iterator = tape.timestepper(iter(iterator))
        for k, h in iterator:
            if restore_after is not None and k == restore_after:
                copies = {}
                for name, f in run.solver.state().items():
                    copies[name] = (f if name == "previous_dt" else
                                    fd.Function(f.function_space().collapse()).assign(f))
                second = self.build(control, sea, **kwargs)
                second.solver.restore(copies)
                run = second
            n0 = len(tape.get_blocks())
            run.dt.assign(h * control if self.kind == "dt" else h)
            if sea is not None:
                sea.I.assign((1 + .2 * k) * ice_shape)
            if self.zero_start:
                zero_start(run.solver)
            run.solver.solve()
            if blocks is not None:
                blocks.append(len(tape.get_blocks()) - n0)
            if forward is not None:
                forward.append(M.preconditioner_counters(run.solver))
            if probe is not None and annotate_tape():
                tape.add_block(CounterProbeBlock(probe, run.solver))
            if newton is not None:
                newton[0].append(newton_iterations(run.solver))
                if annotate_tape():
                    tape.add_block(NewtonProbeBlock(newton[1], run.solver))
        return self.functional(run), run

    def functional(self, run):
        """The functional of `test_gia_solver.py::march` on the fields of `run`."""
        parent, sub = self.meshes
        dx_m = fd.Measure("dx", domain=sub, metadata={"quadrature_degree": 4})
        history = run.solver.internal_variables
        J = fd.assemble((fd.inner(run.u, run.u) + .1 * fd.inner(history, history)) * dx_m)
        J += fd.assemble(run.psi * run.psi * fd.dx(domain=parent))
        sea_level = [c for c in run.solver.couplings if isinstance(c, cp.SeaLevel)]
        if sea_level:
            # A sea-level expression of the coupling, as in the inversions.
            J += fd.assemble(sea_level[0].sea_level() ** 2 * sea_level[0].measure())
        return J

    def fresh(self, control):
        """The functional of an untaped march at `control`, a float."""
        with stop_annotating():
            return float(self.march(control.copy(deepcopy=True))[0])


def taylor_rates(rf, control, direction, J0, slopes):
    """The smallest rates of the first-order Taylor remainders, one per slope.

    The remainders `|J(m + eps h) - J(m) - eps s|` over `eps = 0.01 / 2^i`,
    `i = 0..3`, with the definition of `pyadjoint.taylor_test`. The four
    replays are shared by all slopes `s`: the adjoint derivative gives rate
    2, a slope of 0 (no gradient) rate 1, a wrong slope rate 1 once the
    error of the slope dominates the second-order term.

    Args:
      rf: the `ReducedFunctional`.
      control, direction: the expansion point and the direction.
      J0: the replay at `control`.
      slopes: the slopes `s`.

    Returns:
      One smallest rate per slope.
    """
    eps = [0.01 / 2 ** i for i in range(4)]
    values = [float(rf(shifted(control, direction, e))) for e in eps]
    rates = []
    for slope in slopes:
        residuals = [abs(v - J0 - e * slope) for v, e in zip(values, eps)]
        rates.append(min(math.log(residuals[i] / residuals[i + 1]) / math.log(2)
                         for i in range(3)))
    return rates


def tangent_of(rf, control, direction):
    """The tangent sweep `dJ/dm h` of the tape at the recorded control."""
    tape = get_working_tape()
    rf(control)
    tape.reset_tlm_values()
    rf.controls[0].block_variable.tlm_value = direction.copy(deepcopy=True)
    tape.evaluate_tlm()
    return float(rf.functional.block_variable.tlm_value)


def replay_difference(rf, control, direction, eps):
    """`(J(m + eps h) - J(m - eps h)) / (2 eps)` from two replays of the tape."""
    return (float(rf(shifted(control, direction, eps)))
            - float(rf(shifted(control, direction, -eps)))) / (2 * eps)


def enable_disk(comm):
    """Turn on disk checkpointing under `SingleDiskStorageSchedule` in a new directory.

    The directory is in `out/`, created on rank 0 and shared with the other
    ranks; the caller removes it.

    Returns:
      The path of the directory.
    """
    from checkpoint_schedules import SingleDiskStorageSchedule
    from firedrake.adjoint import enable_disk_checkpointing
    out = CASE_DIR / OUT
    out.mkdir(exist_ok=True)
    directory = tempfile.mkdtemp(prefix="adjoint-disk-", dir=out) if comm.rank == 0 else None
    directory = comm.bcast(directory, root=0)
    enable_disk_checkpointing(checkpoint_comm=comm, checkpoint_dir=directory)
    get_working_tape().enable_checkpointing(SingleDiskStorageSchedule(),
                                            gc_timestep_frequency=1)
    return directory


def enable_memory():
    """Turn on checkpointing under `SingleMemoryStorageSchedule`."""
    from checkpoint_schedules import SingleMemoryStorageSchedule
    get_working_tape().enable_checkpointing(SingleMemoryStorageSchedule())


def counted_run(scenario, control):
    """Tape a march with probes, replay it once and sweep the adjoint; return the counts.

    Returns:
      `(J, rf, blocks, counts)`: the taped functional and its
      `ReducedFunctional` (after the replay and the adjoint sweep), the
      tape blocks per step, and `phase_counts`.
    """
    blocks, forward, log = [], [], []
    continue_annotation()
    J, run = scenario.march(control, blocks=blocks, forward=forward, probe=log)
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    rf(control)
    rf.derivative()
    log.append(("end", *CounterProbeBlock(log, run.solver).read()))
    return J, rf, blocks, phase_counts(forward, log, len(ADJOINT_STEPS))


def adjoint_derivative(rf, direction):
    """The directional derivative from the adjoint, or NaN if the library refuses it.

    `DirectConstitutiveBackend._linear_solve` raises a `RuntimeError` when
    the true relative residual of a linear adjoint solve, with the transpose
    nullspace of its operator removed, is above its bound (ten times the
    tolerance of the solve, at least 1e-8). The refusal is caught
    here so that an entry still writes its other values if it happens; the
    residual is written, and the rows that need the derivative then fail.
    Any other error propagates.

    Returns:
      `(dJ, residual)`: `(<dJ/dm, h>, 0)`, or `(NaN, the refused residual)`.
    """
    try:
        return float(direction._ad_dot(rf.derivative())), 0.0
    except RuntimeError as error:
        message = str(error)
        if not message.startswith("Direct adjoint solve"):
            raise
        print(f"adjoint refused: {message}", flush=True)
        return math.nan, float(message.split()[-1])


def write_counts(name, blocks, counts):
    """Write the block counts and the counter increases of an entry."""
    out = {adjoint_blocks_name(k): n for k, n in enumerate(blocks, start=1)}
    for phase in COUNT_PHASES:
        for k, increase in enumerate(counts.get(phase, []), start=1):
            for counter, value in increase.items():
                out[adjoint_count_name(phase, counter, k)] = value
    M.write_values(CASE_DIR / adjoint_counts(name), out)


def run_entry(name):
    """Measure one entry of `ADJOINT` and write its values and counts."""
    spec = ADJOINT[name]
    start = time.time()
    meshes = C.annulus_meshes()
    tape = get_working_tape()
    tape.clear_tape()
    counts = {}
    if spec.get("counter_preset"):
        # The counters of this entry come from a second march with a preset
        # that caches preconditioner pieces; its tape is dropped afterwards.
        changes = {**spec.get("changes", {}), **spec.get("counter_changes", {})}
        counter_scenario = Scenario(meshes, {**spec, "changes": changes},
                                    preset=spec["counter_preset"])
        control, _ = control_and_direction(spec["control"], meshes)
        _, _, _, counts = counted_run(counter_scenario, control)
        print(f"counter run: {counts} ({time.time() - start:.0f} s)", flush=True)
        tape.clear_tape()
    scenario = Scenario(meshes, spec)
    control, direction = control_and_direction(spec["control"], meshes)
    directory = enable_disk(meshes[1].comm) if spec.get("checkpoint") == "disk" else None
    values = {}
    if scenario.preset == "direct" or directory is not None:
        # No cached preconditioner piece to count: tape without probes.
        blocks = []
        continue_annotation()
        J, _ = scenario.march(control, blocks=blocks, schedule=directory is not None)
        rf = ReducedFunctional(J, Control(control))
        pause_annotation()
        values["J_replay"] = float(rf(control))
    else:
        J, rf, blocks, counts = counted_run(scenario, control)
        values["J_replay"] = float(rf(control))
    values["J"] = float(J)
    values["dJ"], values["adjoint_refused_residual"] = adjoint_derivative(rf, direction)
    print(f"taped march, replay, adjoint: J {values['J']!r} replay {values['J_replay']!r} "
          f"dJ {values['dJ']!r} ({time.time() - start:.0f} s)", flush=True)
    values["J_fresh"] = scenario.fresh(control)
    other = shifted(control, direction, ADJOINT_OTHER_STEP)
    values["J_other_replay"] = float(rf(other))
    values["J_other_fresh"] = scenario.fresh(other)
    values["dJ_fresh"] = (scenario.fresh(shifted(control, direction, ADJOINT_FRESH_EPS))
                          - scenario.fresh(shifted(control, direction, -ADJOINT_FRESH_EPS))
                          ) / (2 * ADJOINT_FRESH_EPS)
    print(f"fresh marches: {values} ({time.time() - start:.0f} s)", flush=True)
    J0 = float(rf(control))
    values["taylor_rate"], values["taylor_rate_no_gradient"] = taylor_rates(
        rf, control, direction, J0, (values["dJ"], 0.0))
    if directory is None:
        # No tangent under a disk schedule: pyadjoint's checkpoint manager
        # has no tangent sweep, and `Tape.evaluate_tlm` walks the blocks
        # without it, so it reads the values the last replay left on them
        # and its result does not measure the solver.
        values["tangent"] = tangent_of(rf, control, direction)
        values["tangent_replay"] = replay_difference(rf, control, direction,
                                                     ADJOINT_REPLAY_EPS)
    values["wall"] = time.time() - start
    # The rank count of the run; the test compares it with `cores` of the
    # multi-rank entries.
    values["ranks"] = meshes[1].comm.size
    print(f"values: {values}", flush=True)
    M.write_values(CASE_DIR / adjoint_values(name), values)
    write_counts(name, blocks, counts)
    if directory is not None and meshes[1].comm.rank == 0:
        shutil.rmtree(directory, ignore_errors=True)


# -- The gradient after a replay ---------------------------------------------------------

def step_reads_previous_solution(solver):
    """Whether every step block after the first reads the projected solution of the step before.

    Each step records one step block and then one kernel projection of the
    mixed solution (`SelfGravitatingGIASolver.solve`). The output of the
    projection of step k is the mixed solution that step k + 1 starts
    from, and the step block of step k + 1 must have it among its
    dependencies, as Firedrake's solve block has its unknown.

    Args:
      solver: the `SelfGravitatingGIASolver` of the taped march.

    Returns:
      True if the tape holds one projection per step block and every step
      block after the first reads the output of the projection before it.
    """
    from gadopt.gia_step_adjoint import GIAStepBlockMixin
    from gadopt.nullspaces import KernelProjectionBlock
    blocks = get_working_tape().get_blocks()
    steps = [b for b in blocks if isinstance(b, GIAStepBlockMixin)]
    projections = [b for b in blocks if isinstance(b, KernelProjectionBlock)]
    if not steps or len(steps) != len(projections):
        return False
    for step, projection in zip(steps[1:], projections[:-1]):
        projected = projection.get_outputs()[0]
        if projected.output is not solver.solution:
            return False
        if not any(dependency is projected for dependency in step.get_dependencies()):
            return False
    return True


def run_replay_gradient(name):
    """Measure one entry of `REPLAY_GRADIENT` and write its values.

    The march is `SEA` with the direct preset, the viscosity control, the
    DtN representation of the entry and a `FrozenDatumSolver`, taped under
    the checkpoint schedule of the entry. On the tape, in this order:

    1. the first adjoint sweep. In it pyadjoint works out which saved
       values a later replay keeps; before it a replay keeps them all;
    2. a replay at the recorded control, with the Newton iterations of each
       step. A replay that starts each step from the saved start of the
       forward step repeats its Newton iterates and so its counts;
    3. a replay at `control + ADJOINT_OTHER_STEP * direction` and the
       adjoint sweep there;
    4. the central difference of two replays about that control, which
       uses no adjoint.

    The adjoint of each step linearises about the solved state of the step.
    If no block after a step reads that state, pyadjoint drops its saved
    value after step 1, and the sweep in step 3 linearises about the state
    of an earlier run of the tape.

    Args:
      name: a key of `REPLAY_GRADIENT`.
    """
    spec = REPLAY_GRADIENT[name]
    start = time.time()
    meshes = C.annulus_meshes()
    tape = get_working_tape()
    tape.clear_tape()
    scenario = Scenario(meshes, dict(base="SEA", preset="direct", control="viscosity",
                                     changes={"representation": spec["representation"]},
                                     frozen_datum=True))
    control, direction = control_and_direction("viscosity", meshes)
    directory = None
    if spec["schedule"] == "disk":
        directory = enable_disk(meshes[1].comm)
    else:
        enable_memory()
    forward, replay = [], []
    continue_annotation()
    J, run = scenario.march(control, newton=(forward, replay), schedule=True)
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    values = {"J": float(J)}
    values["step_reads_previous_solution"] = int(step_reads_previous_solution(run.solver))
    values["dJ_first"], refused_first = adjoint_derivative(rf, direction)
    # Only the readings of this replay count, so a schedule that replays
    # steps inside an adjoint sweep does not add readings to it.
    first = len(replay)
    values["J_replay"] = float(rf(control))
    replayed = replay[first:]
    assert len(replayed) == len(forward) == len(ADJOINT_STEPS), (replayed, forward)
    print(f"taped march, adjoint, replay: J {values['J']!r} replay {values['J_replay']!r} "
          f"dJ {values['dJ_first']!r} Newton {forward} / {replayed} "
          f"({time.time() - start:.0f} s)", flush=True)
    other = shifted(control, direction, ADJOINT_OTHER_STEP)
    values["J_other_replay"] = float(rf(other))
    values["dJ_after_replay"], refused_after = adjoint_derivative(rf, direction)
    values["dJ_replay_difference"] = replay_difference(rf, other, direction,
                                                       ADJOINT_REPLAY_EPS)
    values["adjoint_refused_residual"] = max(refused_first, refused_after)
    for k, (n_forward, n_replay) in enumerate(zip(forward, replayed), start=1):
        values[newton_name("forward", k)] = n_forward
        values[newton_name("replay", k)] = n_replay
    values["steps"] = len(ADJOINT_STEPS)
    values["ranks"] = meshes[1].comm.size
    values["wall"] = time.time() - start
    print(f"values: {values}", flush=True)
    M.write_values(CASE_DIR / adjoint_values(name), values)
    if directory is not None and meshes[1].comm.rank == 0:
        shutil.rmtree(directory, ignore_errors=True)


# -- guards --------------------------------------------------------------------------

#: The configuration of the unit tests of the step adjoint
#: (`test_gia_step_adjoint.STEP_KINDS`): the Quadrature history with BDF2,
#: the low-rank DtN, two Maxwell elements, a fluid core, the centre of mass
#: and the prescribed load.
STEP_CASE = C.Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                   model="maxwell2", core="fluid", centre=True)


def unit_march(meshes, case, control, steps, **kwargs):
    """A march of a case without sea level, with the functional of `test_gia_solver.march`.

    Returns:
      `(J, solver)`.
    """
    parent, sub = meshes
    with stop_annotating():
        dt = C.dt_function(sub)
        solver, u, psi = C.build_new(meshes, case, control=control, dt=dt,
                                     nullspace="rigid_rotation", **kwargs)
    for h in steps:
        dt.assign(h)
        solver.solve()
    dx_m = fd.Measure("dx", domain=sub, metadata={"quadrature_degree": 4})
    history = solver.internal_variables
    J = fd.assemble((fd.inner(u, u) + .1 * fd.inner(history, history)) * dx_m)
    J += fd.assemble(psi * psi * fd.dx(domain=parent))
    return J, solver


def guard_capped(meshes, out):
    """`derivative_max_it=1` makes the adjoint solve raise; 400 does not."""
    case = C.Case(family="Quadrature", representation="lowrank", core="fluid", centre=True)
    for cap, flag in ((1, "capped_derivative_raises"), (400, "uncapped_derivative_passes")):
        get_working_tape().clear_tape()
        control, _ = control_and_direction("viscosity", meshes)
        continue_annotation()
        J, _ = unit_march(meshes, case, control, ADJOINT_STEPS[:1],
                          solver_parameters="iterative", derivative_max_it=cap,
                          default_options=False)
        rf = ReducedFunctional(J, Control(control))
        pause_annotation()
        try:
            rf.derivative()
            raised = False
        except fd.ConvergenceError:
            raised = True
        out[flag] = int(raised if cap == 1 else not raised)


def guard_residual_bound(meshes, out):
    """The bound of the derivative residual follows the tolerance of the solve.

    Three taped marches of `sea-iterative-viscosity` (Quadrature history,
    low-rank DtN, moving coastline, iterative preset) with a loose outer
    Krylov tolerance, each followed by one adjoint sweep:

    - `loose`: the four steps at the tolerance of the GIAMIP drivers,
      `ksp_rtol = LOOSE_OUTER_RTOL` (1e-6). The outer FGMRES takes two
      iterations per adjoint solve and ends at a true relative residual of
      about 5e-9.
    - `coarse`: one step at `ksp_rtol = COARSE_OUTER_RTOL` (1e-4). The
      outer FGMRES stops after one iteration at a true relative residual of
      about 3e-5, which a fixed bound of 1e-8 refused.
    - `capped`: one step at `LOOSE_OUTER_RTOL`, with the outer adjoint KSP
      stopped after one iteration and that iteration counted as convergence
      (the command-line options `ksp_max_it 1` and `ksp_converged_maxits`
      under the prefix of the derivative solve). PETSc reports a converged
      solve, and `DirectConstitutiveBackend._linear_solve` must refuse it
      on its true residual, about three times its bound. The options are
      removed again whatever happens.

    `loose_derivative_accepted` is 1 when the first two sweeps give a finite
    derivative with no refusal and their bound is
    `DERIVATIVE_RESIDUAL_SAFETY` times the tolerance. `forced_cap_refused`
    is 1 when the third sweep is refused with a residual above its bound.
    The residual and the bound of the last adjoint solve of each march are
    written beside the flags.
    """
    from gadopt.constitutive_integrators import DERIVATIVE_RESIDUAL_SAFETY
    scenario = Scenario(meshes, ADJOINT["sea-iterative-viscosity"])
    accepted = True
    for name, rtol, steps, capped in (("loose", LOOSE_OUTER_RTOL, ADJOINT_STEPS, False),
                                      ("coarse", COARSE_OUTER_RTOL, ADJOINT_STEPS[:1], False),
                                      ("capped", LOOSE_OUTER_RTOL, ADJOINT_STEPS[:1], True)):
        get_working_tape().clear_tape()
        control, direction = control_and_direction("viscosity", meshes)
        continue_annotation()
        J, run = scenario.march(control, steps=steps, extra={"ksp_rtol": rtol})
        rf = ReducedFunctional(J, Control(control))
        pause_annotation()
        cap = {}
        if capped:
            # Command-line options under the prefix of the adjoint solve,
            # read when its solver is built at the first adjoint solve.
            prefix = f"{run.solver.name}_quadrature_adj_"
            cap = {prefix + "ksp_max_it": 1, prefix + "ksp_converged_maxits": True}
        options = PETSc.Options()
        try:
            for key, value in cap.items():
                options.setValue(key, value)
            dJ, refused = adjoint_derivative(rf, direction)
        finally:
            for key in cap:
                options.delValue(key)
        relative, bound = run.solver._step.backend.derivative_residuals.get(
            "adjoint", (math.nan, math.nan))
        out[f"{name}_dJ"] = dJ
        out[f"{name}_refused_residual"] = refused
        out[f"{name}_last_residual"] = relative
        out[f"{name}_bound"] = bound
        if capped:
            out["forced_cap_refused"] = int(math.isnan(dJ) and refused > bound)
        else:
            accepted = accepted and (
                math.isfinite(dJ) and refused == 0
                and math.isclose(bound, DERIVATIVE_RESIDUAL_SAFETY * rtol, rel_tol=1e-12))
    out["loose_derivative_accepted"] = int(accepted)


def guard_hessian_and_controls(meshes, out):
    """The Hessian of the Quadrature step raises; a control absent from the residual raises."""
    get_working_tape().clear_tape()
    case = C.Case(family="Quadrature", representation="multiplier", core="fluid", centre=True)
    control, direction = control_and_direction("viscosity", meshes)
    with stop_annotating():
        unrelated = fd.Function(control.function_space()).assign(1.0)
    continue_annotation()
    J, solver = unit_march(meshes, case, control, ADJOINT_STEPS[:1])
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    rf.derivative()
    try:
        rf.hessian(direction)
        out["hessian_refused"] = 0
    except NotImplementedError as error:
        out["hessian_refused"] = int("Quadrature history step" in str(error))
    try:
        solver.require_controls_reach_residual(unrelated)
        out["absent_control_refused"] = 0
    except ValueError:
        out["absent_control_refused"] = 1
    try:
        solver.require_controls_reach_residual(control)
        out["present_control_accepted"] = 1
    except ValueError:
        out["present_control_accepted"] = 0


def tangential_functional(solver, sub):
    """`int u . t (1 + sin 2 theta) dS` over the surface of the mantle.

    The constant part of the weight integrates a rigid rotation to
    `2 pi r^2` times its amplitude, so the functional sees the rotation
    content of `u`, which only the recorded kernel projection removes in a
    replay (`test_gia_nullspace_projection_adjoint._functionals`).
    """
    u = solver.field("displacement")
    X = fd.SpatialCoordinate(sub)
    n = fd.FacetNormal(sub)
    tangent = fd.as_vector([-n[1], n[0]])
    weight = 1 + fd.sin(2 * fd.atan2(X[1], X[0]))
    return fd.assemble(fd.dot(u, tangent) * weight * fd.ds(C.CURVE_RE, domain=sub))


def guard_tape(meshes, out):
    """Tape structure, the recorded projection and the scaled-gradient Taylor rate.

    One taped march of two steps of `SEA` with the viscosity control and two
    functionals on the same tape: the functional of the case and the
    tangential surface functional.
    """
    from gadopt.gia_step_adjoint import GIAStepBlockMixin, TapeSweepMarkerBlock
    from gadopt.nullspaces import KernelProjectionBlock
    get_working_tape().clear_tape()
    sub = meshes[1]
    scenario = Scenario(meshes, ADJOINT["sea-direct-viscosity"])
    steps = ADJOINT_STEPS[:2]
    control, direction = control_and_direction("viscosity", meshes)
    continue_annotation()
    J, run = scenario.march(control, steps=steps)
    J_tan = tangential_functional(run.solver, sub)
    rf = ReducedFunctional(J, Control(control))
    rf_tan = ReducedFunctional(J_tan, Control(control))
    pause_annotation()
    # Marker, step block, marker, projection: once per solve, in this order.
    blocks = get_working_tape().get_blocks()
    step_at = [i for i, b in enumerate(blocks) if isinstance(b, GIAStepBlockMixin)]
    ok = len(step_at) == len(steps)
    for i in step_at:
        before, step, after, projection = (blocks[i - 1:i + 3] + [None] * 4)[:4]
        ok = ok and isinstance(before, TapeSweepMarkerBlock) and before.before
        ok = ok and isinstance(after, TapeSweepMarkerBlock) and not after.before
        ok = ok and isinstance(projection, KernelProjectionBlock)
        ok = ok and step.gia_solve_index == before.solve_index == after.solve_index
        ok = ok and any(dep is step.get_outputs()[0]
                        for dep in projection.get_dependencies())
    projections = sum(isinstance(b, KernelProjectionBlock) for b in blocks)
    out["projection_blocks"] = projections
    out["tape_structure"] = int(ok and projections == len(steps))
    # The tangential functional: replay and fresh march against the recording.
    replay_tan = float(rf_tan(control))
    with stop_annotating():
        _, fresh_run = scenario.march(control.copy(deepcopy=True), steps=steps)
        fresh_tan = tangential_functional(fresh_run.solver, sub)
    scale = abs(float(J_tan))
    out["projection_J"] = float(J_tan)
    out["projection_replay_error"] = abs(replay_tan - float(J_tan)) / scale
    out["projection_fresh_error"] = abs(fresh_tan - float(J_tan)) / scale
    dJ_tan = float(direction._ad_dot(rf_tan.derivative()))
    (out["projection_rate"],) = taylor_rates(rf_tan, control, direction,
                                             float(rf_tan(control)), (dJ_tan,))
    out["projection_taylor"] = int(out["projection_rate"] >= TAYLOR_RATE_MIN)
    # A gradient scaled by 1.01 on the functional of the case.
    J0 = float(rf(control))
    dJ = float(direction._ad_dot(rf.derivative()))
    rate, scaled = taylor_rates(rf, control, direction, J0, (dJ, 1.01 * dJ))
    out["unscaled_gradient_rate"] = rate
    out["scaled_gradient_rate"] = scaled
    out["scaled_gradient_detected"] = int(scaled <= SCALED_GRADIENT_RATE_MAX
                                          and rate >= TAYLOR_RATE_MIN)


def guard_collected(meshes, out):
    """No solver of a finished taped run stays alive once the tape is cleared.

    `test_gia_step_adjoint.py::test_a_finished_run_is_collected` with the
    direct and the iterative preset: two taped steps, a gradient, then the
    tape is cleared and the garbage collected; a weak reference to the
    solver must be dead.
    """
    for preset in ("direct", "iterative"):
        get_working_tape().clear_tape()
        control, _ = control_and_direction("viscosity", meshes)
        with stop_annotating():
            dt = C.dt_function(meshes[1])
            solver, u, psi = C.build_new(meshes, STEP_CASE, control=control, dt=dt,
                                         nullspace="rigid_rotation", solver_parameters=preset)
        continue_annotation()
        for h in ADJOINT_STEPS[:2]:
            dt.assign(h)
            solver.solve()
        J = fd.assemble(fd.inner(u, u) * fd.dx(domain=meshes[1]))
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
        out[f"collected_{preset}"] = int(alive() is None)


def guard_restore(meshes, out):
    """A state restored into a second solver after two steps, on the tape: replay and Taylor."""
    get_working_tape().clear_tape()
    scenario = Scenario(meshes, ADJOINT["sea-direct-viscosity"])
    control, direction = control_and_direction("viscosity", meshes)
    continue_annotation()
    J, _ = scenario.march(control, restore_after=2)
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    J0 = float(rf(control))
    out["restore_replay_error"] = abs(J0 - float(J)) / abs(float(J))
    out["restore_replay"] = int(out["restore_replay_error"] <= REPLAY_RTOL)
    dJ = float(direction._ad_dot(rf.derivative()))
    (out["restore_rate"],) = taylor_rates(rf, control, direction, J0, (dJ,))
    out["restore_taylor"] = int(out["restore_rate"] >= TAYLOR_RATE_MIN)


def run_guards():
    """Run every guard and write the flags and their numbers.

    A guard that raises does not stop the others: its exception is printed,
    `<guard>_raised` is written as 1, and every flag it did not set is
    written as 0, so that the test reports the flags that a broken library
    no longer holds.
    """
    import traceback
    start = time.time()
    meshes = C.annulus_meshes()
    out = {}
    for guard in (guard_capped, guard_residual_bound, guard_hessian_and_controls, guard_tape,
                  guard_collected, guard_restore):
        try:
            guard(meshes, out)
            out[f"{guard.__name__}_raised"] = 0
        except Exception:
            traceback.print_exc()
            out[f"{guard.__name__}_raised"] = 1
        if annotate_tape():
            pause_annotation()
        get_working_tape().clear_tape()
        print(f"{guard.__name__}: {out} ({time.time() - start:.0f} s)", flush=True)
    for flag in GUARD_FLAGS:
        out.setdefault(flag, 0)
    out["wall"] = time.time() - start
    M.write_values(CASE_DIR / adjoint_values(GUARDS), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True,
                        choices=sorted([*ADJOINT, *REPLAY_GRADIENT, GUARDS]))
    name = parser.parse_args().entry
    print(f"gadopt from {gadopt.__file__}", flush=True)
    if name == GUARDS:
        run_guards()
    elif name in REPLAY_GRADIENT:
        run_replay_gradient(name)
    else:
        run_entry(name)
