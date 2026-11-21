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

#: The tests that do not apply to an entry, by test. pyadjoint evaluates a
#: tangent sweep with `Tape.evaluate_tlm`, which walks the blocks of the tape
#: and reads the values saved on them; the checkpoint manager of a
#: checkpoint schedule (`pyadjoint.checkpointing.CheckpointManager`) drives
#: the forward replay and the adjoint sweep only and has no tangent sweep. Under
#: `SingleDiskStorageSchedule` the values that the tangent sweep reads are
#: the ones the last replay left, not the ones of each step, so the tangent
#: of the disk entry (9.4 percent off) does not measure the solver. Replay,
#: adjoint and Taylor rate of that entry repeat `sea-direct-viscosity`.
NOT_APPLICABLE = {
    "tangent_pairing": {name for name, entry in E.ADJOINT.items()
                        if entry.get("checkpoint") == "disk"},
    "tangent_replay": {name for name, entry in E.ADJOINT.items()
                       if entry.get("checkpoint") == "disk"},
}

#: The assertions that fail with the measured numbers, by `(test, entry)`,
#: with the numbers. They are strict xfails, so a change that makes one pass
#: shows in the report. None at present. A `LOAD` march that starts each
#: step from the previous solution replays only to 1.8e-8, so the `LOAD`
#: entries start each step from zero (`gia_entries.ADJOINT`, `zero_start`).
KNOWN_FAILURES = {}


def entry_params(test):
    """The entries of `ENTRIES` as parameters of `test`, with the known failures marked.

    The entries of `NOT_APPLICABLE[test]` are left out.
    """
    for entry in ENTRIES:
        if entry in NOT_APPLICABLE.get(test, ()):
            continue
        reason = KNOWN_FAILURES.get((test, entry))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(entry, marks=marks, id=entry)


MULTI_RANK = {name: entry["cores"] for name, entry in E.ADJOINT.items()
              if entry.get("cores", 1) > 1}


@pytest.mark.parametrize("entry,cores", MULTI_RANK.items())
def test_each_multi_rank_entry_ran_on_its_ranks(entry, cores):
    """A multi-rank entry ran on the number of ranks that its entry gives.

    An entry for two ranks that ran as single-rank processes would pass its
    comparison with the serial entry without testing what it claims.
    """
    ranks = values(entry)["ranks"]
    assert ranks == cores, f"{entry}: ran on {ranks!r} rank(s), expected {cores!r}"


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
#: xfails). None at present.
GUARD_KNOWN_FAILURES = {}


def guard_params():
    """The flags of `E.GUARD_FLAGS` as parameters, with the known failures marked."""
    for flag in E.GUARD_FLAGS:
        reason = GUARD_KNOWN_FAILURES.get(flag)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(flag, marks=marks, id=flag)


def test_the_projection_replays_the_tangential_functional():
    """The replay and a fresh march repeat the tangential functional to `PROJECTION_RTOL`.

    With the rigid rotation declared and its projection on the tape, a
    functional of the tangential surface displacement, which a rigid
    rotation changes, replays to the recorded value. The bound 1e-6 is set
    by the measurement and by the defect it must catch: the replay differs by
    2.6e-8 (the replay starts each step from zero and the rotation is a
    kernel only to 8e-6, so the two projected solves stop at points about
    1e-8 apart in the rotation content), and without the recorded projection
    (mutation M6) the replay error is 0.68. The fresh march repeats the
    recording exactly.
    """
    v = values(E.GUARDS)
    for name in ("projection_replay_error", "projection_fresh_error"):
        assert v[name] <= E.PROJECTION_RTOL, (
            f"{name}: {v[name]!r}, at most {E.PROJECTION_RTOL!r}")


@pytest.mark.parametrize("flag", list(guard_params()))
def test_every_guard_holds(flag):
    """Each guard of the `guards` entry wrote 1 (see `gia_entries.GUARD_FLAGS`)."""
    v = values(E.GUARDS)
    numbers = {name: value for name, value in v.items() if name not in E.GUARD_FLAGS}
    assert v[flag] == 1, f"guard {flag}: {v[flag]!r}, expected 1; numbers {numbers!r}"
