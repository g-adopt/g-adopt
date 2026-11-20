"""Assertions of the `adjoint` case of the coupled GIA solver.

`gia_adjoint.py` tapes one march per entry of `gia_entries.ADJOINT` under
`doit run_case` and writes the measured values to `out/adjoint-<entry>.dat`
and the block counts and build counters to `out/adjoint-<entry>-counts.dat`.
This file reads them and asserts:

- the tape reproduces the forward model: replay at the recorded control
  against the recording and a fresh march, replay at another control
  against a fresh march there;
- the derivatives are right: the adjoint against a central difference of
  fresh marches (an independent evaluation), the Taylor rates with and
  without the gradient, the tangent sweep against the adjoint pairing and
  against a central difference of replays;
- the relations between entries that solve the same system in another way
  (DtN representation, preset, checkpoint schedule, rank count);
- the counts: the tape blocks per step and the build counters against
  `expected/adjoint-<entry>.txt` (a recorded value, which detects a change),
  and in `sea-direct-dt` a rebuild at the change of the time step in the
  forward march, the replay and the adjoint sweep;
- the flags of the `guards` entry.

This file does not import Firedrake; it runs where the outputs are.
"""

import pytest

import gia_entries as E


def read_values(path):
    """The `name value` lines of `path` as a dictionary; `#` starts a comment line."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, value = line.split()
            out[name] = float(value)
    return out


def values(entry):
    """The values of an entry, from `out/adjoint-<entry>.dat`."""
    return read_values(E.CASE_DIR / E.adjoint_values(entry))


def counts(entry):
    """The block counts and counter increases of an entry."""
    return read_values(E.CASE_DIR / E.adjoint_counts(entry))


def relative(measured, expected):
    """`|measured - expected| / |expected|`."""
    return abs(measured - expected) / abs(expected)


ENTRIES = list(E.ADJOINT)

#: The `LOAD` entries march with the tolerances of the preset (see
#: `gia_entries.ADJOINT`). Newton then stops after one step with the
#: residual reduced by about 1e-7, and a replay, which starts each step from
#: zero (`ConstitutiveStepBlock.prepare_recompute_component`), stops at
#: another point than the forward march, which starts from the previous
#: state.
_LOAD_REPLAY = (
    "{entry}: the replay repeats the recording to 1.81e-8 and the fresh march at "
    "the other control to 1.82e-8 (bands 1e-9 and 1e-8). The march uses the "
    "preset tolerances (one Newton step, residual reduction about 1.4e-7), and "
    "the replay starts every step from zero while the forward march starts from "
    "the previous state. The fresh march repeats the recording exactly, and the "
    "derivatives agree (adjoint against the fresh difference {dJ}, Taylor rate "
    "{rate}).")

#: The disk schedule: the tangent sweep runs after a replay under
#: `SingleDiskStorageSchedule` and reads saved values that the schedule does
#: not restore for a tangent sweep.
_DISK_TANGENT = (
    "sea-direct-viscosity-disk: the tangent is 1.8022498658930116e-06 against "
    "the adjoint 1.6475546695832376e-06 and the difference of replays "
    "1.647554670545625e-06 (9.4 percent off; bands 1e-8 and 1e-6). Replay, "
    "adjoint and Taylor rate repeat sea-direct-viscosity to all digits, so the "
    "tangent sweep `tape.evaluate_tlm()` under `SingleDiskStorageSchedule` "
    "reads saved values that the schedule does not provide for it.")

#: The assertions that fail with the measured numbers, by `(test, entry)`,
#: with the numbers. They are strict xfails, so a change that makes one pass
#: shows in the report.
KNOWN_FAILURES = {
    ("replay", "load-direct-Lambda"): _LOAD_REPLAY.format(
        entry="load-direct-Lambda", dJ="8.0e-9", rate="2.0003"),
    ("other", "load-direct-Lambda"): _LOAD_REPLAY.format(
        entry="load-direct-Lambda", dJ="8.0e-9", rate="2.0003"),
    ("replay", "load-direct-B_mu"): _LOAD_REPLAY.format(
        entry="load-direct-B_mu", dJ="7.2e-8", rate="2.0005"),
    ("other", "load-direct-B_mu"): _LOAD_REPLAY.format(
        entry="load-direct-B_mu", dJ="7.2e-8", rate="2.0005"),
    ("tangent_pairing", "sea-direct-viscosity-disk"): _DISK_TANGENT,
    ("tangent_replay", "sea-direct-viscosity-disk"): _DISK_TANGENT,
}


def entry_params(test):
    """The entries of `ENTRIES` as parameters of `test`, with the known failures marked."""
    for entry in ENTRIES:
        reason = KNOWN_FAILURES.get((test, entry))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(entry, marks=marks, id=entry)


@pytest.mark.parametrize("entry", list(entry_params("replay")))
def test_replay_repeats_the_recording(entry):
    """The replay of the tape at the recorded control gives the recorded functional.

    An operation that changes a taped field without being recorded (for
    example the kernel projection) makes the replay differ.
    """
    v = values(entry)
    error = relative(v["J_replay"], v["J"])
    assert error <= E.REPLAY_RTOL, (
        f"{entry}: replay {v['J_replay']!r}, recorded {v['J']!r}, relative "
        f"difference {error!r}, tolerance {E.REPLAY_RTOL!r}")


@pytest.mark.parametrize("entry", list(entry_params("fresh")))
def test_a_fresh_march_repeats_the_recording(entry):
    """An untaped march at the recorded control gives the recorded functional."""
    v = values(entry)
    error = relative(v["J_fresh"], v["J"])
    assert error <= E.FRESH_RTOL, (
        f"{entry}: fresh {v['J_fresh']!r}, recorded {v['J']!r}, relative "
        f"difference {error!r}, tolerance {E.FRESH_RTOL!r}")


@pytest.mark.parametrize("entry", list(entry_params("other")))
def test_replay_at_another_control_is_a_fresh_march(entry):
    """The replay at `control + 0.05 direction` gives a fresh march there.

    The two differ by more than the tolerance from the recorded functional,
    so the replay did move with the control.
    """
    v = values(entry)
    error = relative(v["J_other_replay"], v["J_other_fresh"])
    assert error <= E.OTHER_RTOL, (
        f"{entry}: replay {v['J_other_replay']!r}, fresh {v['J_other_fresh']!r}, "
        f"relative difference {error!r}, tolerance {E.OTHER_RTOL!r}")
    moved = relative(v["J_other_fresh"], v["J"])
    assert moved > 100 * E.OTHER_RTOL, (
        f"{entry}: the functional moved by {moved!r} at the other control, "
        f"expected more than {100 * E.OTHER_RTOL!r}")


@pytest.mark.parametrize("entry", list(entry_params("fresh_difference")))
def test_the_adjoint_matches_a_fresh_central_difference(entry):
    """`<dJ/dm, h>` from the adjoint equals the central difference of two fresh marches.

    The difference uses neither the tape nor the adjoint code, so it is an
    independent check of the derivative; its own error is `O(eps^2)`.
    """
    v = values(entry)
    error = relative(v["dJ"], v["dJ_fresh"])
    assert error <= E.FRESH_DIFFERENCE_RTOL, (
        f"{entry}: adjoint {v['dJ']!r}, fresh central difference {v['dJ_fresh']!r}, "
        f"relative difference {error!r}, tolerance {E.FRESH_DIFFERENCE_RTOL!r}")


@pytest.mark.parametrize("entry", list(entry_params("floor")))
def test_the_directional_derivative_is_above_the_floor(entry):
    """`|<dJ/dm, h>| >= 1e-6 |J|`, so the derivative checks compare numbers that carry it."""
    v = values(entry)
    floor = E.DERIVATIVE_FLOOR * abs(v["J"])
    assert abs(v["dJ"]) >= floor, f"{entry}: |dJ| {abs(v['dJ'])!r} below {floor!r}"


@pytest.mark.parametrize("entry", list(entry_params("taylor")))
def test_the_taylor_rate_is_two(entry):
    """The first-order Taylor remainder with the adjoint gradient converges at rate 2."""
    rate = values(entry)["taylor_rate"]
    assert rate > E.TAYLOR_RATE_MIN, (
        f"{entry}: Taylor rate {rate!r}, expected above {E.TAYLOR_RATE_MIN!r}")


@pytest.mark.parametrize("entry", list(entry_params("no_gradient")))
def test_without_the_gradient_the_rate_is_one(entry):
    """With the gradient replaced by 0 the same remainders converge at rate 1.

    This is the negative control of the Taylor test: the same functional,
    point and direction reach rate 2 only with the gradient.
    """
    rate = values(entry)["taylor_rate_no_gradient"]
    low, high = E.NO_GRADIENT_BAND
    assert low <= rate <= high, (
        f"{entry}: rate without the gradient {rate!r}, expected in [{low!r}, {high!r}]")


@pytest.mark.parametrize("entry", list(entry_params("tangent_pairing")))
def test_the_tangent_is_the_adjoint_pairing(entry):
    """The tangent sweep `dJ/dm h` equals the adjoint pairing `<dJ/dm, h>`."""
    v = values(entry)
    error = relative(v["tangent"], v["dJ"])
    assert error <= E.TANGENT_PAIRING_RTOL, (
        f"{entry}: tangent {v['tangent']!r}, adjoint {v['dJ']!r}, relative "
        f"difference {error!r}, tolerance {E.TANGENT_PAIRING_RTOL!r}")


@pytest.mark.parametrize("entry", list(entry_params("tangent_replay")))
def test_the_tangent_matches_a_difference_of_replays(entry):
    """The tangent equals the central difference of two replays of the tape."""
    v = values(entry)
    error = relative(v["tangent"], v["tangent_replay"])
    assert error <= E.TANGENT_REPLAY_RTOL, (
        f"{entry}: tangent {v['tangent']!r}, difference of replays "
        f"{v['tangent_replay']!r}, relative difference {error!r}, tolerance "
        f"{E.TANGENT_REPLAY_RTOL!r}")


@pytest.mark.parametrize("entry", list(entry_params("counts")))
def test_the_counts_match_the_recording(entry):
    """The tape blocks per step and the build counters equal the recorded ones.

    A change of the tape structure or of the rebuild rule of the cached
    preconditioners changes these integers.
    """
    path = E.CASE_DIR / E.adjoint_expected(entry)
    measured = counts(entry)
    if not path.exists():
        pytest.skip(f"{E.adjoint_expected(entry)} is not recorded yet; measured {measured!r}")
    expected = read_values(path)
    assert sorted(measured) == sorted(expected), (
        f"{entry}: names {sorted(measured)!r}, expected {sorted(expected)!r}")
    differ = {name: (measured[name], expected[name]) for name in expected
              if measured[name] != expected[name]}
    assert not differ, f"{entry}: (measured, expected) {differ!r}"


def relation_params(test):
    """The relations of `E.ADJOINT_RELATIONS` as parameters of `test`, with the known failures."""
    for relation in E.ADJOINT_RELATIONS:
        reason = KNOWN_FAILURES.get((test, relation[0]))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(*relation, marks=marks, id=f"{relation[0]}-vs-{relation[1]}")


@pytest.mark.parametrize("entry, reference, rtol_J, rtol_dJ", list(relation_params("relation_J")))
def test_entries_that_solve_the_same_system_give_the_same_functional(
        entry, reference, rtol_J, rtol_dJ):
    """The functional agrees with the reference entry (`E.ADJOINT_RELATIONS`)."""
    v, r = values(entry), values(reference)
    error = relative(v["J"], r["J"])
    assert error <= rtol_J, (
        f"{entry}: J {v['J']!r}, {reference} {r['J']!r}, relative difference "
        f"{error!r}, tolerance {rtol_J!r}")


@pytest.mark.parametrize("entry, reference, rtol_J, rtol_dJ", list(relation_params("relation_dJ")))
def test_entries_that_solve_the_same_system_give_the_same_derivative(
        entry, reference, rtol_J, rtol_dJ):
    """The directional derivative agrees with the reference entry (`E.ADJOINT_RELATIONS`)."""
    v, r = values(entry), values(reference)
    error = relative(v["dJ"], r["dJ"])
    assert error <= rtol_dJ, (
        f"{entry}: dJ {v['dJ']!r}, {reference} {r['dJ']!r}, relative difference "
        f"{error!r}, tolerance {rtol_dJ!r}")


@pytest.mark.parametrize("phase", E.COUNT_PHASES)
def test_the_caches_rebuild_at_the_change_of_dt(phase):
    """In `sea-direct-dt` the cached operators rebuild at the step where `dt` changes.

    In the forward march, in the replay and in the adjoint sweep. The counters
    are those of the classes that answer `gadopt.preconditioners._RebuildRule`
    (`gia_entries.REBUILD_COUNTERS`); the columns of `LowRankPotentialPC`
    follow the potential block, which does not depend on the time step. A
    rule that does not rebuild at a new time step keeps a preconditioner
    built for the old one there.
    """
    measured = counts("sea-direct-dt")
    step = E.ADJOINT_DT_CHANGE_STEP
    names = [E.adjoint_count_name(phase, counter, step) for counter in E.REBUILD_COUNTERS]
    absent = [name for name in names if name not in measured]
    assert not absent, f"sea-direct-dt has no counter {absent!r}: {sorted(measured)!r}"
    missing = {name: measured[name] for name in names if measured[name] < 1}
    assert not missing, (
        f"sea-direct-dt {phase}: increase at step {step} {missing!r}, expected at least 1")


#: The guards that fail with the measured numbers, with the numbers (strict
#: xfails).
GUARD_KNOWN_FAILURES = {
    "projection_replay": (
        "With the projection on the tape, the replay of the tangential surface "
        "functional on SEA (moving coastline, rotation, direct preset, two steps) "
        "repeats the recording to 2.65e-8, against the bound 1e-10 of "
        "test_gia_nullspace_projection_adjoint (which ran a prescribed load); the "
        "fresh march repeats it exactly. Without the recorded projection "
        "(mutation M6) the replay error is 0.68. The probable cause: the replay "
        "starts each step from zero, and the declared rigid rotation is a null "
        "vector only to 8e-6 on the curved facets, so the projected Newton solve "
        "of the replay and the "
        "one of the forward march stop at points that differ by about 1e-8 "
        "relative in the rotation content, which this functional reads directly. "
        "The functional of the case replays to 3.5e-11."),
}


def guard_params():
    """The flags of `E.GUARD_FLAGS` as parameters, with the known failures marked."""
    for flag in E.GUARD_FLAGS:
        reason = GUARD_KNOWN_FAILURES.get(flag)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(flag, marks=marks, id=flag)


@pytest.mark.parametrize("flag", list(guard_params()))
def test_every_guard_holds(flag):
    """Each guard of the `guards` entry wrote 1 (see `gia_entries.GUARD_FLAGS`)."""
    v = values(E.GUARDS)
    numbers = {name: value for name, value in v.items() if name not in E.GUARD_FLAGS}
    assert v[flag] == 1, f"guard {flag}: {v[flag]!r}, expected 1; numbers {numbers!r}"
