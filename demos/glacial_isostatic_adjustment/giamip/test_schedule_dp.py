"""Serial tests of schedule D plus the protocol years (decision R14).

The "b" forward run of Exp09, Exp10 and Exp12 marches schedule D (the
march of their r2 inversions) with every GIAMIP protocol year added as a
step end (`--schedule windows-coarse-protocol` of `forward_expXX.py`), so
that the output exists at every protocol year. Each experiment directory
has its own copy of the schedule code (`expXX_common`), so every test runs
on all three copies.

The tests check:

- the step counts, the step of year 0, and that the steps sum to the span;
- that the years are exactly the union of the years of schedule D and the
  protocol years, so every schedule-D step is kept or split at protocol
  years, and nothing else changes;
- the step lengths and the years after which dt changes, written out here
  from the rule, not taken from the code under test;
- that no piece is shorter than `SCHEDULE_DP_MIN_STEP_YR` and no year was
  removed, and the removal rule of `merge_protocol_years` on small made-up
  marches (a short piece loses its schedule-D end; a short piece that
  cannot lose an end is refused);
- that a schedule-D state file is the input of this schedule
  (`input_schedule_of`), that the map from a step of this run to the step
  of the file is the one the forward script uses, and that every protocol
  position 0 .. 462 is the output of exactly one step;
- that every year that is not a year of the file is a protocol year, so
  its ice is a protocol slice (weight 0 in `ice_bracket`);
- that the forward script accepts the schedule name.

No solver runs here. Run with the PYTHONPATH of CLAUDE.md (worktree root,
gia-mip and meshing-gia):

    python -m pytest demos/glacial_isostatic_adjustment/giamip/test_schedule_dp.py
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The three experiment directories: their modules have distinct names, so
# all three can be imported in one process. The forward module imports the
# prepare module of its own directory, so each directory must be on the
# path.
for _number in ("09", "10", "12"):
    sys.path.insert(0, os.path.join(HERE, f"exp{_number}"))

import gadopt  # noqa: E402,F401  (gadopt before firedrake)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from giamip.mip.inputs import GiamipExperiment  # noqa: E402

#: Per experiment: the steps to year 0 and to +1000 yr, the step lengths,
#: and the years after which dt changes. Outside the windows the merge
#: gives the protocol steps (500 yr to year -26000, then 100 yr); the
#: windows keep the 50 yr and 20 yr steps of schedule D. Written out from
#: the rule: 192 steps of 500 yr to -26000, 100 yr steps from -26000 to
#: -16000 (100 steps), the window steps, 100 yr steps from the last window
#: to year 0, and 10 steps of 100 yr to +1000.
EXPECTED = {
    "09": {"ice": "iceHistory-GLAC3c_profile1.nc",
           # 192 + 100 + (20 + 50 + 40 + 50 + 20) + 100 = 572 to year 0.
           "year0": 572, "total": 582, "d_year0": 296,
           "changes": [-26000, -16000, -15000, -14000, -12000, -11000,
                       -10000]},
    "10": {"ice": "iceHistory-GLAC3c_profile2.nc",
           # 192 + 100 + (20 + 50 + 60 + 50 + 20) + 90 = 582 to year 0.
           "year0": 582, "total": 592, "d_year0": 315,
           "changes": [-26000, -16000, -15000, -14000, -11000, -10000,
                       -9000]},
    "12": {"ice": "iceHistory-ICE-6G_D.nc",
           # 192 + 100 + (20 + 100 + 40) + 110 = 562 to year 0.
           "year0": 562, "total": 572, "d_year0": 277,
           "changes": [-26000, -16000, -15000, -13000, -11000]},
}
NUMBERS = sorted(EXPECTED)


def common_of(number):
    """The `expXX_common` module of experiment `number`."""
    return importlib.import_module(f"exp{number}_common")


def protocol_of(number):
    """The start year, the protocol years and the ice file of an
    experiment, as giamip defines them (no data file is read)."""
    experiment = GiamipExperiment(f"Exp{number}")
    years = np.asarray(experiment.model_years(), dtype=float)
    ice_file = os.path.basename(
        experiment.definition["forcing"]["ice_history_file"])
    return int(years[0]), years, ice_file


def schedules(number):
    """The windows and years of schedule D and of schedule D plus the
    protocol years, and the protocol years."""
    common = common_of(number)
    start, protocol, ice_file = protocol_of(number)
    windows, d_years = common.window_schedule(common.SCHEDULE_D, ice_file,
                                              start, protocol)
    dp_windows, dp_years = common.window_schedule(common.SCHEDULE_DP,
                                                  ice_file, start, protocol)
    assert tuple(dp_windows) == tuple(windows)
    return d_years, dp_years, protocol


@pytest.mark.parametrize("number", NUMBERS)
def test_step_counts_and_time_span(number):
    """The step counts to year 0 and in total, and the steps sum to the
    span of the experiment."""
    common = common_of(number)
    _, years, _ = schedules(number)
    expected = EXPECTED[number]
    assert int(np.flatnonzero(years == 0)[0]) == expected["year0"]
    assert len(years) - 1 == expected["total"]
    assert (common.SCHEDULE_DP_STEPS_122KA[expected["ice"]]
            == expected["total"])
    assert years[0] == -122000 and years[-1] == 1000
    assert int(np.diff(years).sum()) == 123000


@pytest.mark.parametrize("number", NUMBERS)
def test_union_of_schedule_d_and_protocol(number):
    """The years are the union of the schedule-D years and the protocol
    years; every schedule-D step is kept or split at protocol years."""
    d_years, years, protocol = schedules(number)
    pset = set(int(p) for p in protocol)
    dset = set(d_years.tolist())
    assert years.tolist() == sorted(dset | pset)
    # Each schedule-D step [a, b] holds exactly the protocol years strictly
    # inside it as extra step ends.
    for a, b in zip(d_years[:-1].tolist(), d_years[1:].tolist()):
        inside = years[(years > a) & (years < b)].tolist()
        assert inside == sorted(p for p in pset if a < p < b)


@pytest.mark.parametrize("number", NUMBERS)
def test_step_lengths_and_dt_changes(number):
    """500 yr before -26000, 100 yr after it outside the windows, the
    schedule-D window steps inside; dt changes after the listed years."""
    common = common_of(number)
    _, years, _ = schedules(number)
    windows = common.schedule_d_windows(EXPECTED[number]["ice"])
    dts = np.diff(years)
    for year, dt in zip(years[:-1].tolist(), dts.tolist()):
        inside = [s for first, end, s in windows if first <= year < end]
        if inside:
            want = inside[0]
        else:
            want = 500 if year < -26000 else 100
        assert dt == want, (year, dt)
    assert sorted(set(dts.tolist())) == [20, 50, 100, 500]
    found = [int(years[i]) for i in range(1, len(dts))
             if dts[i] != dts[i - 1]]
    assert found == EXPECTED[number]["changes"]


@pytest.mark.parametrize("number", NUMBERS)
def test_no_short_piece_and_nothing_removed(number):
    """No piece is shorter than the tolerance, and the merge removed no
    schedule-D year."""
    common = common_of(number)
    d_years, years, protocol = schedules(number)
    merged, dropped = common.merge_protocol_years(d_years, protocol)
    assert dropped == []
    assert merged.tolist() == years.tolist()
    assert int(np.diff(years).min()) >= common.SCHEDULE_DP_MIN_STEP_YR == 10


@pytest.mark.parametrize("number", NUMBERS)
def test_merge_removes_the_schedule_d_end_of_a_short_piece(number):
    """A short piece made by a split loses its schedule-D end (not a
    protocol year); a short piece with no single such end is refused."""
    common = common_of(number)
    merge = common.merge_protocol_years
    # The protocol year 995 cuts the step 0 -> 2000 of the base 5 yr before
    # the base year 1000: the base year goes, the protocol year stays.
    years, dropped = merge([0, 1000, 2000], [0, 995, 2000])
    assert years.tolist() == [0, 995, 2000] and dropped == [1000]
    # The same 5 yr apart, the protocol year after the base year.
    years, dropped = merge([0, 1000, 2000], [0, 1005, 2000])
    assert years.tolist() == [0, 1005, 2000] and dropped == [1000]
    # Pieces of exactly the tolerance stay.
    years, dropped = merge([0, 1000, 2000], [0, 990, 2000])
    assert years.tolist() == [0, 990, 1000, 2000] and dropped == []
    # Both ends of the short piece are protocol years: refused.
    with pytest.raises(ValueError):
        merge([0, 2000], [0, 995, 1000, 2000])
    # Both ends are base years that are not protocol years: refused.
    with pytest.raises(ValueError):
        merge([0, 1000, 1005, 2000], [0, 2000])
    # The first year never goes: the piece 0 -> 3 has the first year and a
    # protocol year as its ends.
    with pytest.raises(ValueError):
        merge([0, 2000], [0, 3, 2000])


@pytest.mark.parametrize("number", NUMBERS)
def test_file_map_output_positions_and_ice(number):
    """The forward run of this schedule reads a schedule-D state file: its
    steps to year 0 map onto steps of this run with the same year, every
    other step ends at a protocol year (ice weight 0), and each protocol
    position is the output of one step."""
    common = common_of(number)
    assert common.input_schedule_of(common.SCHEDULE_DP) == common.SCHEDULE_D
    assert common.input_schedule_of(common.SCHEDULE_D) == common.SCHEDULE_D
    assert common.input_schedule_of(common.SCHEDULE_C) == common.SCHEDULE_C
    assert common.input_schedule_of("protocol") is None
    d_years, years, protocol = schedules(number)
    d_year0 = EXPECTED[number]["d_year0"]
    # The years of the state file: schedule D to year 0.
    file_years = d_years[:d_year0 + 1]
    # The map of the forward script: step of this run -> step of the file.
    fine_index = {int(y): k for k, y in enumerate(years)}
    file_steps = {fine_index[int(file_years[j])]: j
                  for j in range(1, len(file_years))}
    assert len(file_steps) == d_year0
    # The map keeps the order and ends at year 0 of both.
    keys = sorted(file_steps)
    assert [file_steps[k] for k in keys] == list(range(1, d_year0 + 1))
    assert years[keys[-1]] == 0 and keys[-1] == EXPECTED[number]["year0"]
    # Every other step ends at a protocol year: its ice is that slice.
    for index in range(1, len(years)):
        if index in file_steps:
            continue
        k, w = common.ice_bracket(protocol, float(years[index]))
        assert w == 0.0 and protocol[k] == years[index]
    # The output index of each step: its protocol position, or None.
    position = {int(y): k for k, y in enumerate(protocol)}
    outputs = [position.get(int(y)) for y in years[1:]]
    written = [o for o in outputs if o is not None]
    assert written == list(range(1, len(protocol)))


@pytest.mark.parametrize("number", NUMBERS)
def test_forward_accepts_the_schedule(number, monkeypatch):
    """The forward script takes `--schedule windows-coarse-protocol`."""
    forward = importlib.import_module(f"forward_exp{number}")
    monkeypatch.setattr(sys, "argv", [
        "forward", "--state", "state.h5", "--schedule",
        "windows-coarse-protocol", "--gia_rebuild_max_outer", "0"])
    args = forward.parse_args()
    assert args.schedule == "windows-coarse-protocol"
    assert args.gia_rebuild_max_outer == 0.0
