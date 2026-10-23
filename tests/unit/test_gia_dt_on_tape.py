"""A time step that changes on the tape: the rebuild rule and the derivative.

Decision S28 of `NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-2026-10-20.md`:
the GIAMIP scripts hold the time step `dt` as a `Function` on the Real space
and assign it on the tape, so a march whose step changes (500 yr to 100 yr
at 26 ka in the protocol schedule) can be taped and replayed.

Two things are checked on the 2-D case of `test_gia_transient_power_law_gate`
(N = 3 elements, DG0 power law, moving coastline, low-rank DtN), with `dt =
1` at step 1 and `dt = 0.5` at steps 2 and 3:

1. The preconditioner rebuild rule (`gadopt.preconditioners._RebuildRule`,
   lagged branch) rebuilds at the first solve after the change of `dt` in
   the forward run, in the replay and in the adjoint sweep. In the replay
   and the adjoint, the solver's `dt` object holds the value of the last
   forward assignment (0.5), so the rule must read the time step of the
   step being solved from the solve block (`appctx["gia_dt"]`, bound by
   `gadopt.dtn_coupled_adjoint`). The default lag is 5 solves, so in this
   3-step march only the change of `dt` can trigger a rebuild.
2. A guarded Taylor test with respect to the initial sea level has rate 2
   across the change, and a replay at a new control equals a fresh march.
"""

import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import firedrake as fd
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from pyadjoint.tape import annotate_tape

import gadopt.preconditioners as preconditioners
import test_gia_transient_power_law_gate as gate
import test_gia_gravity

from test_gravity_adjoint import assert_taylor_with_guards

#: The module fixture of `test_gia_gravity` (parent annulus, mantle
#: submesh), bound to a name of this module so that pytest finds it.
meshes = test_gia_gravity.meshes

#: The time step of each step of the march (units of t_bar). The change is
#: at step 2; steps 2 and 3 share the shorter step.
DTS = (1.0, 0.5, 0.5)


@pytest.fixture(autouse=True)
def clean_tape():
    """A fresh tape for every test, and annotation off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


def march(meshes, SL_init):
    """Build the gate solver and march `DTS`; return `(J, solver, dt)`.

    `dt` is a `Function` on the Real space of the mantle, assigned when its
    value changes. When the tape is annotating, each assignment is a block,
    so the replay reads the value of each step from the tape.
    """
    _, sub = meshes
    with stop_annotating():
        _, I_init, ice = gate.shoreline_state(sub)
        I = fd.Function(I_init.function_space(), name="ice").assign(ice[0])
        dt = fd.Function(fd.FunctionSpace(sub, "R", 0), name="dt")
        solver, _, _ = gate.build(meshes, SL_init, I_init, I, dt)
    previous = None
    for k, value in enumerate(DTS, start=1):
        if value != previous:
            dt.assign(value)
            previous = value
        I.assign(ice[k])
        solver.solve()
    with stop_annotating():
        reference, _, _ = gate.shoreline_state(sub)
        target = fd.Function(reference.function_space()).interpolate(
            reference + 0.1 * fd.sin(gate.polar_angle(sub) - gate.PHASE))
    SL = solver.sea_level()
    J = fd.assemble((SL - target) ** 2
                    * solver.sea_level_measure()(gate.CURVE_RE))
    return J, solver, dt


def tape(meshes):
    """Tape the march with `SL_init` as control; return `(Jhat, m, J, dt, solver)`."""
    _, sub = meshes
    with stop_annotating():
        SL_init, _, _ = gate.shoreline_state(sub)
    get_working_tape().clear_tape()
    continue_annotation()
    try:
        control = Control(SL_init)
        J, solver, dt = march(meshes, SL_init)
        Jhat = ReducedFunctional(J, control)
    finally:
        pause_annotation()
    return Jhat, SL_init, J, dt, solver


def test_rebuild_rule_sees_the_change_of_dt_in_replay_and_adjoint(
        meshes, monkeypatch):
    """The lagged rule rebuilds at the dt change in forward, replay and adjoint.

    Every answer of `_RebuildRule.due` is recorded with the phase, the
    preconditioner, the solve index and the published time step. A rebuild
    must happen at solve index 2 (the first solve at `dt = 0.5`) in the
    forward run and in the replay, where the indices rise, and at solve
    index 1 (the first solve at `dt = 1` when walking back) in the adjoint,
    where the indices fall. The published time step must be the one of the
    step being solved in every phase.
    """
    events = []
    phase = ["forward"]
    due = preconditioners._RebuildRule.due

    def recording_due(self, pc):
        answer = due(self, pc)
        ctx = preconditioners._solver_appctx(pc)
        events.append((phase[0], ctx.get("gia_solve_index"),
                       preconditioners._published_dt(pc), answer))
        return answer

    monkeypatch.setattr(preconditioners._RebuildRule, "due", recording_due)
    Jhat, SL_init, _, dt, solver = tape(meshes)
    # The live value after the forward run is the last assignment.
    assert float(dt) == DTS[-1]
    phase[0] = "replay"
    Jhat(SL_init)
    phase[0] = "adjoint"
    Jhat.derivative()

    for name, rebuild_index in (("forward", 2), ("replay", 2),
                                ("adjoint", 1)):
        rows = [e for e in events if e[0] == name]
        print(f"    [dt-rule/{name}] {rows}")
        assert rows, f"no rebuild decision recorded in the {name} phase"
        # The published time step is that of the step solved (solve index
        # k solves step k of DTS).
        for _, index, published, _ in rows:
            assert published == DTS[index - 1], (name, index, published)
        # A rebuild at the first solve on the other side of the change, and
        # at no other solve: the lag of 5 solves does not expire in a
        # three-step march, so only the change of dt can rebuild. A rule
        # that rebuilt at every solve would fail this equality.
        rebuilt = {index for _, index, _, answer in rows if answer}
        assert rebuilt == {rebuild_index}, (
            f"{name}: rebuilds at solve indices {sorted(rebuilt)}, expected "
            f"only {rebuild_index}")
    # The binding is undone after each block solve: the live context reads
    # the solver's own dt again for the next forward solve.
    assert "gia_dt" not in solver.appctx


def test_taylor_and_replay_across_a_change_of_dt(meshes):
    """Rate 2 across the change of dt, and a replay equal to a fresh march."""
    _, sub = meshes
    Jhat, SL_init, J, _, _ = tape(meshes)
    h = gate.control_direction(sub)
    assert_taylor_with_guards(Jhat, SL_init, h, J, min_rate=1.90)
    with stop_annotating():
        shifted = SL_init.copy(deepcopy=True)
        shifted += h
        J_fresh, _, _ = march(meshes, shifted.copy(deepcopy=True))
    J_replay = float(Jhat(shifted))
    relative = abs(J_replay - float(J_fresh)) / abs(float(J_fresh))
    print(f"    [dt-taylor] replay {J_replay:.12e} fresh "
          f"{float(J_fresh):.12e} rel {relative:.3e}")
    assert relative <= 1e-8
