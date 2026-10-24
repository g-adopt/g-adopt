"""Serial tests of schedule D of Exp09, Exp10 and Exp12 (decision R12).

Schedule D is the march of the r2 inversions and forward runs of the three
experiments with transient or power-law rheology: 1 kyr steps everywhere,
and in the fast melt windows of schedule C twice the schedule-C step (50 yr
for 25 yr, 20 yr for 10 yr). Each experiment directory has its own copy of
the schedule code (`expXX_common`), so every test runs on all three copies.

The tests check:

- the step counts of each experiment, to year 0 and to +1000 yr, and that
  the steps sum to the full time span;
- the windows (twice the schedule-C step, same edges) and the step of every
  year: the window step inside a window, 1 kyr outside;
- that the 1 kyr steps line up with the window edges, year 0 and the last
  year, and that a window edge off the 1 kyr grid is refused;
- the protocol years that schedule D reaches and those it does not reach
  (the forward run writes its output at the reached years only);
- that schedule C is unchanged by the shared window check;
- that the `march` record of an input file of schedule D, written with
  `write_json_dataset` into a `CheckpointFile` (as `write_inputs` does),
  reads back the same, and gives the inversion the dt of schedule D;
- that the ice of the input file (`march_ice`) and of the forward run
  (`LinearIce`) are equal at every year of schedule D.

No solver runs here. Run with the PYTHONPATH of CLAUDE.md (worktree root,
gia-mip and meshing-gia):

    python -m pytest demos/glacial_isostatic_adjustment/giamip/test_schedule_d.py
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The three experiment directories: their modules have distinct names
# (`exp09_common`, `forward_exp10`, ...), so all three can be imported in
# one process. The forward module imports the prepare module of its own
# directory, so each directory must be on the path.
for _number in ("09", "10", "12"):
    sys.path.insert(0, os.path.join(HERE, f"exp{_number}"))

import gadopt  # noqa: E402,F401  (gadopt before firedrake)
import firedrake as fd  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from giamip.mip.inputs import GiamipExperiment  # noqa: E402

#: Per experiment: the steps to year 0 and to +1000 yr of schedule D, the
#: windows of schedule D, and the years after which dt changes. The windows
#: are written out here from the decision (twice the schedule-C step of
#: design handout 8.3), not taken from the code under test.
EXPECTED = {
    "09": {"ice": "iceHistory-GLAC3c_profile1.nc",
           "year0": 296, "total": 297,
           "windows": ((-16000, -15000, 50), (-15000, -14000, 20),
                       (-14000, -12000, 50), (-12000, -11000, 20),
                       (-11000, -10000, 50)),
           "changes": [-16000, -15000, -14000, -12000, -11000, -10000],
           "c_total": 762},
    "10": {"ice": "iceHistory-GLAC3c_profile2.nc",
           "year0": 315, "total": 316,
           "windows": ((-16000, -15000, 50), (-15000, -14000, 20),
                       (-14000, -11000, 50), (-11000, -10000, 20),
                       (-10000, -9000, 50)),
           "changes": [-16000, -15000, -14000, -11000, -10000, -9000],
           "c_total": 792},
    "12": {"ice": "iceHistory-ICE-6G_D.nc",
           "year0": 277, "total": 278,
           "windows": ((-16000, -15000, 50), (-15000, -13000, 20),
                       (-13000, -11000, 50)),
           "changes": [-16000, -15000, -13000, -11000],
           "c_total": 732},
}
NUMBERS = sorted(EXPECTED)


def common_of(number):
    """The `expXX_common` module of experiment `number`."""
    return importlib.import_module(f"exp{number}_common")


def protocol_of(number):
    """The start year, the protocol years and the ice file of an experiment,
    as giamip defines them (no data file is read)."""
    experiment = GiamipExperiment(f"Exp{number}")
    years = np.asarray(experiment.model_years(), dtype=float)
    ice_file = os.path.basename(
        experiment.definition["forcing"]["ice_history_file"])
    return int(years[0]), years, ice_file


def schedule_d(number):
    """The windows and years of schedule D of an experiment."""
    common = common_of(number)
    start, protocol, ice_file = protocol_of(number)
    return common.window_schedule(common.SCHEDULE_D, ice_file, start,
                                  protocol)


@pytest.mark.parametrize("number", NUMBERS)
def test_experiment_inputs(number):
    """giamip gives the ice history and the 122 ka start of the table."""
    start, protocol, ice_file = protocol_of(number)
    assert ice_file == EXPECTED[number]["ice"]
    assert start == -122000 and protocol[-1] == 1000.0
    # 192 protocol years before -26000 and 271 from -26000 to +1000.
    assert len(protocol) == 463


@pytest.mark.parametrize("number", NUMBERS)
def test_step_counts_and_time_span(number):
    """The step counts to year 0 and in total, and the steps sum to the
    span of the experiment."""
    common = common_of(number)
    windows, years = schedule_d(number)
    expected = EXPECTED[number]
    year0 = int(np.flatnonzero(years == 0)[0])
    assert year0 == expected["year0"]
    assert len(years) - 1 == expected["total"]
    assert common.SCHEDULE_D_STEPS_122KA[expected["ice"]] == expected["total"]
    assert years[0] == -122000 and years[-1] == 1000
    dts = np.diff(years)
    # The steps sum to the full span, and to year 0 the taped steps sum to
    # the span of the inversion.
    assert int(dts.sum()) == 1000 - (-122000)
    assert int(dts[:year0].sum()) == 122000
    # One step after year 0, of 1 kyr.
    assert dts[year0:].tolist() == [1000]


@pytest.mark.parametrize("number", NUMBERS)
def test_window_and_kyr_steps(number):
    """The windows are those of schedule C at twice the step, and every
    step has the step of its window or 1 kyr."""
    common = common_of(number)
    windows, years = schedule_d(number)
    expected = EXPECTED[number]
    assert tuple(windows) == expected["windows"]
    # Twice the schedule-C step, same edges.
    c_windows = common.schedule_c_windows(expected["ice"])
    assert [(f, e) for f, e, _ in windows] == [(f, e) for f, e, _ in c_windows]
    assert [s for _, _, s in windows] == [2 * s for _, _, s in c_windows]
    dts = np.diff(years)
    for year, dt in zip(years[:-1].tolist(), dts.tolist()):
        inside = [s for first, end, s in windows if first <= year < end]
        assert dt == (inside[0] if inside else 1000), (year, dt)
    assert sorted(set(dts.tolist())) == [20, 50, 1000]
    # Every window edge is a year of the schedule: no step crosses an edge.
    edges = {e for first, end, _ in windows for e in (first, end)}
    assert edges <= set(years.tolist())
    # The steps in each window: its length over its step.
    for first, end, step in windows:
        inside = years[(years >= first) & (years < end)]
        assert len(inside) == (end - first) // step
    # dt changes after these years only; the rebuild rule of the
    # preconditioner rebuilds at the first solve after each of them.
    found = [int(years[i]) for i in range(1, len(dts)) if dts[i] != dts[i - 1]]
    assert found == expected["changes"]


@pytest.mark.parametrize("number", NUMBERS)
def test_protocol_years_reached_and_lost(number):
    """Outside the windows schedule D reaches only the whole kyr; inside it
    reaches every protocol year. Every year of the schedule outside a window
    is a protocol year (its ice is a protocol slice)."""
    windows, years = schedule_d(number)
    _, protocol, _ = protocol_of(number)
    pset = set(int(p) for p in protocol)
    yset = set(years.tolist())
    first_w, end_w = windows[0][0], windows[-1][1]
    in_windows = {p for p in pset if first_w <= p <= end_w}
    assert in_windows <= yset
    outside = [y for y in yset
               if not any(f <= y < e for f, e, _ in windows)]
    assert set(outside) <= pset
    # The protocol years outside the windows that are reached: the whole
    # kyr, nothing else.
    reached_out = {p for p in pset - in_windows if p in yset}
    assert reached_out == {p for p in pset - in_windows if p % 1000 == 0}
    # The counts of the report: 463 protocol years in all.
    lost = sorted(pset - yset)
    span_kyr = (end_w - first_w) // 1000
    # Before -26000: 96 half-kyr years lost. From -26000 to year 0 the
    # 100 yr years off the kyr outside the windows: 9 per kyr over
    # (26 - span) kyr. After year 0: 100 .. 900, 9 years.
    assert len(lost) == 96 + 9 * (26 - span_kyr) + 9
    assert len(pset & yset) == 463 - len(lost)


@pytest.mark.parametrize("number", NUMBERS)
def test_schedule_c_unchanged(number):
    """Schedule C keeps its counts (the window check is shared)."""
    common = common_of(number)
    start, protocol, ice_file = protocol_of(number)
    _, years = common.window_schedule(common.SCHEDULE_C, ice_file, start,
                                      protocol)
    assert len(years) - 1 == EXPECTED[number]["c_total"]
    assert set(int(p) for p in protocol) <= set(years.tolist())


@pytest.mark.parametrize("number", NUMBERS)
@pytest.mark.parametrize("windows, start", [
    # A window edge half a kyr off the 1 kyr grid.
    (((-15500, -15000, 50),), -122000),
    # A start off the grid of the window edges.
    (((-16000, -15000, 50),), -121500),
    # Two windows that overlap.
    (((-16000, -15000, 50), (-15500, -14000, 20)), -122000),
    # A step that does not divide the window.
    (((-16000, -15000, 30),), -122000)])
def test_a_bad_window_is_refused(number, windows, start):
    """A window that does not line up with the 1 kyr steps, or that breaks
    a rule of schedule C, is refused; no step is shortened."""
    common = common_of(number)
    _, protocol, _ = protocol_of(number)
    with pytest.raises(ValueError):
        common.schedule_d_years(start, protocol, windows)


@pytest.mark.parametrize("number", NUMBERS)
def test_unknown_schedule_is_refused(number):
    """`window_schedule` knows schedules C and D only."""
    common = common_of(number)
    start, protocol, ice_file = protocol_of(number)
    with pytest.raises(ValueError):
        common.window_schedule("windows-fine", ice_file, start, protocol)


@pytest.mark.parametrize("number", NUMBERS)
def test_march_record_reads_back(number, tmp_path):
    """The `march` record of an input file of schedule D, written into a
    CheckpointFile with the ice entries (as `write_inputs`), reads back the
    same, and the inversion's dt and the forward script's check follow."""
    common = common_of(number)
    start, protocol, ice_file = protocol_of(number)
    windows, all_years = schedule_d(number)
    years = all_years.astype(float)
    year0 = int(np.flatnonzero(years == 0.0)[0])
    n_ice = year0 + 1
    brackets = [common.ice_bracket(protocol, y) for y in years[:n_ice]]
    # The protocol step of each year (the slice index here), -1 between two
    # protocol years, as the prepare script makes it.
    march_steps = [int(k) if w == 0.0 else -1 for k, w in brackets]
    info = {"ice_history": ice_file, "windows": windows,
            "steps_to_last_year": int(len(all_years) - 1)}
    march = common.window_march_record(common.SCHEDULE_D, years[:n_ice],
                                       march_steps, n_ice, year0, brackets,
                                       **info)
    record = {"experiment": f"giamip-exp{number}", "march": march}
    # Write as `write_inputs`: one ice entry per step with its year, and the
    # record as a JSON dataset.
    mesh = fd.UnitSquareMesh(2, 2, name=common.PARENT_NAME)
    ice = fd.Function(fd.FunctionSpace(mesh, "CG", 1), name=common.ICE_NAME)
    path = str(tmp_path / "inputs.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(mesh)
        for index in range(n_ice):
            ice.assign(float(index))
            chk.save_function(ice, idx=index,
                              timestepping_info={"year": float(years[index])})
        common.write_json_dataset(chk, f"{common.INPUT_GROUP}/record", record)
    with fd.CheckpointFile(path, "r") as chk:
        back = common.read_json_dataset(chk, f"{common.INPUT_GROUP}/record")
        loaded = chk.load_mesh(common.PARENT_NAME)
        history = chk.get_timestepping_history(loaded, common.ICE_NAME)
    # The record is the JSON form of what was written.
    assert back == common.plain_json(record)
    got = back["march"]
    assert got["schedule"] == common.SCHEDULE_D == "windows-coarse"
    assert got["year0_step"] == year0 == EXPECTED[number]["year0"]
    assert got["ice_steps"] == n_ice
    assert got["steps_to_last_year"] == EXPECTED[number]["total"]
    assert [tuple(w) for w in got["windows"]] == list(EXPECTED[number]["windows"])
    # The years of the record and of the ice entries are the schedule.
    assert got["years"] == all_years[:n_ice].astype(float).tolist()
    assert [float(y) for y in history["year"]] == got["years"]
    assert [int(i) for i in history["index"]] == list(range(n_ice))
    # The inversion takes dt from the differences of the record's years
    # (`invert_expXX.BedInversion`): the taped steps of schedule D.
    file_years = np.asarray(got["years"], dtype=float)
    taped_dt = [file_years[i] - file_years[i - 1]
                for i in range(1, got["year0_step"] + 1)]
    assert taped_dt == np.diff(all_years[:n_ice]).astype(float).tolist()
    # The check of the forward script: the march of the file is the start
    # of the schedule it builds, to year 0.
    _, fine_y = common.window_schedule(got["schedule"], got["ice_history"],
                                       int(file_years[0]), protocol)
    assert np.array_equal(file_years, fine_y[:n_ice].astype(float))
    assert float(fine_y[got["year0_step"]]) == 0.0
    # The years outside the windows carry a protocol slice (weight 0).
    for year, (k, w) in zip(got["years"], got["ice_brackets"]):
        if not any(f <= year < e for f, e, _ in windows):
            assert w == 0.0 and protocol[k] == year


class FakeForcing:
    """Protocol ice slices with random nodal values, counted per read."""

    def __init__(self, space, n, seed=3):
        rng = np.random.default_rng(seed)
        self.values = rng.random((n, space.dof_dset.size)) * 3.0e-4
        self.space = space
        self.calls = {}

    def ice(self, index):
        """A new Function with slice `index`, as `SurfaceForcing.ice`."""
        self.calls[index] = self.calls.get(index, 0) + 1
        f = fd.Function(self.space, name="ice")
        f.dat.data[:] = self.values[index]
        return f


@pytest.mark.parametrize("number", NUMBERS)
def test_prepare_and_forward_make_the_same_ice(number):
    """Around the windows (2 kyr before the first to 2 kyr after the last):
    the ice of the input file (`march_ice`) equals the ice of the forward
    run (`LinearIce`) at every year of schedule D; at a 1 kyr year it is
    the protocol slice; the check of the prepare script passes."""
    common = common_of(number)
    forward = importlib.import_module(f"forward_exp{number}")
    windows, all_years = schedule_d(number)
    _, protocol_all, _ = protocol_of(number)
    lo, hi = windows[0][0] - 2000, windows[-1][1] + 2000
    years = all_years[(all_years >= lo) & (all_years <= hi)].astype(float)
    protocol = protocol_all[(protocol_all >= lo) & (protocol_all <= hi)]
    V = fd.FunctionSpace(fd.UnitSquareMesh(3, 3), "CG", 1)
    forcing = FakeForcing(V, len(protocol))
    brackets = [common.ice_bracket(protocol, y) for y in years]
    ice = common.march_ice(forcing.ice, brackets, V)
    # The slices between two kyr years outside the windows are not read.
    unread = [k for k, p in enumerate(protocol)
              if p not in set(years.tolist())]
    assert unread and not set(unread) & set(forcing.calls)
    assert all(count == 1 for count in forcing.calls.values())
    check = common.check_linear_ice(years, ice, protocol, fd.Function(V))
    assert check["max_abs_diff"] == 0.0
    assert check["years_checked"] == sum(1 for _, w in brackets if w != 0.0)
    linear = forward.LinearIce(FakeForcing(V, len(protocol)), protocol)
    out = fd.Function(V)
    for j, year in enumerate(years):
        k, w = linear.set(float(year), out)
        assert (k, w) == brackets[j]
        assert np.array_equal(out.dat.data_ro, ice[j].dat.data_ro)
        if w == 0.0:
            assert np.array_equal(out.dat.data_ro, forcing.values[k])
