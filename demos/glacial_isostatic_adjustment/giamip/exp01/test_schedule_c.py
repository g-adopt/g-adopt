"""Serial tests of schedule C (design handout 8.3, decisions S33 and S34).

Schedule C is the march of the inversion and of the production forward run
of Exp09, Exp10 and Exp12: the protocol steps plus 25 yr and 10 yr steps in
the fast melt windows of the ice history. These tests check the step lists
against the counts of the handout, and that the ice of the input file (made
by the prepare script with `exp01_common.march_ice`) and the ice of the
forward run (`forward_exp01.LinearIce`) are the same numbers at every year,
and that after a write and a read of the file the ice is still linear to
round-off. No solver runs here. Run with the
PYTHONPATH of CLAUDE.md:

    python -m pytest demos/glacial_isostatic_adjustment/giamip/exp01/test_schedule_c.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_exp01 as fwd  # noqa: E402  (imports gadopt, then firedrake)
import exp01_common as common  # noqa: E402
import firedrake as fd  # noqa: E402

#: The protocol years of a 122 ka experiment: 500 yr steps to -26000, then
#: 100 yr steps to +1000 (`GiamipExperiment.model_years`, protocol v4.0).
PROTOCOL_122 = np.concatenate([np.arange(-122000, -26000, 500),
                               np.arange(-26000, 1001, 100)]).astype(float)

ICE6G = "iceHistory-ICE-6G_D.nc"
GLAC1 = "iceHistory-GLAC3c_profile1.nc"
GLAC2 = "iceHistory-GLAC3c_profile2.nc"


@pytest.mark.parametrize("ice_file, steps, changes", [
    # Handout 8.3: 732 steps for Exp12 (ICE-6G_D), 762 for Exp09 (GLAC3c
    # profile 1); Exp10 (profile 2) by the reading in SCHEDULE_C_WINDOWS.
    (ICE6G, 732, [-26000, -16000, -15000, -13000, -11000]),
    (GLAC1, 762, [-26000, -16000, -15000, -14000, -12000, -11000, -10000]),
    (GLAC2, 792, [-26000, -16000, -15000, -14000, -11000, -10000, -9000])])
def test_schedule_c_counts_and_steps(ice_file, steps, changes):
    """The step count of each history, the years at which dt changes, and
    the protocol steps outside the windows."""
    windows = common.schedule_c_windows(ice_file)
    years = common.schedule_c_years(-122000, PROTOCOL_122, windows)
    dts = np.diff(years)
    assert len(years) - 1 == steps == common.SCHEDULE_C_STEPS_122KA[ice_file]
    assert years[0] == -122000 and years[-1] == 1000
    # Every protocol year is a year of the schedule (the output is there).
    assert set(PROTOCOL_122.astype(int)) <= set(years.tolist())
    # Year 0 is 10 steps before the end (10 protocol steps of 100 yr).
    assert int(np.flatnonzero(years == 0)[0]) == steps - 10
    # dt changes after these years only (the rebuild rule rebuilds there).
    found = [int(years[i]) for i in range(1, len(dts)) if dts[i] != dts[i - 1]]
    assert found == changes
    # Each step has the length of its window, or reaches the next protocol
    # year: 500 yr before -26000, 100 yr after.
    for year, dt in zip(years[:-1].tolist(), dts.tolist()):
        window = [s for first, end, s in windows if first <= year < end]
        if window:
            assert dt == window[0]
        else:
            assert dt == (500 if year < -26000 else 100)


def test_the_windows_of_the_handout():
    """The windows of Exp12 and Exp09 as listed in handout 8.3."""
    years = common.schedule_c_years(-122000, PROTOCOL_122,
                                    common.schedule_c_windows(ICE6G))
    dts = dict(zip(years[:-1].tolist(), np.diff(years).tolist()))
    assert dts[-16100] == 100 and dts[-16000] == 25 and dts[-15025] == 25
    assert dts[-15000] == 10 and dts[-13010] == 10
    assert dts[-13000] == 25 and dts[-11025] == 25 and dts[-11000] == 100
    years = common.schedule_c_years(-122000, PROTOCOL_122,
                                    common.schedule_c_windows(GLAC1))
    dts = dict(zip(years[:-1].tolist(), np.diff(years).tolist()))
    assert dts[-15000] == 10 and dts[-14010] == 10 and dts[-14000] == 25
    assert dts[-12025] == 25 and dts[-12000] == 10 and dts[-11000] == 25
    assert dts[-10025] == 25 and dts[-10000] == 100


@pytest.mark.parametrize("windows", [
    # Starts between two protocol years.
    ((-16050, -15000, 25),),
    # 30 yr does not divide the 1000 yr of the window.
    ((-16000, -15000, 30),),
    # 1000 yr divides the window but jumps over the protocol year -29500.
    ((-30000, -29000, 1000),),
    # Two windows that overlap.
    ((-16000, -15000, 25), (-15500, -14000, 10))])
def test_a_bad_window_is_refused(windows):
    """A window that does not start and end at protocol years, or whose
    step does not divide it or misses a protocol year, is refused."""
    with pytest.raises(ValueError):
        common.schedule_c_years(-122000, PROTOCOL_122, windows)


def test_ice_bracket():
    """The bracket and weight; w is exactly 0 at a protocol year."""
    k, w = common.ice_bracket(PROTOCOL_122, -15010.0)
    assert PROTOCOL_122[k] == -15100.0 and w == pytest.approx(0.9)
    assert common.ice_bracket(PROTOCOL_122, -15000.0) == (
        int(np.flatnonzero(PROTOCOL_122 == -15000.0)[0]), 0.0)
    assert common.ice_bracket(PROTOCOL_122, 1000.0)[1] == 0.0
    with pytest.raises(ValueError):
        common.ice_bracket(PROTOCOL_122, 1010.0)


class FakeForcing:
    """The ice slices of `SurfaceForcing.ice`, from random nodal values.

    Every call returns a new Function, as the real transfer does; the calls
    are counted per slice.
    """

    def __init__(self, space, n, seed=2):
        rng = np.random.default_rng(seed)
        self.values = rng.random((n, space.dof_dset.size)) * 3.0e-4
        self.space = space
        self.calls = {}

    def ice(self, index):
        self.calls[index] = self.calls.get(index, 0) + 1
        f = fd.Function(self.space, name=common.ICE_NAME)
        f.dat.data[:] = self.values[index]
        return f


def window_march(ice_file, first=-16500, last=-10000):
    """The schedule-C years from `first` to `last` and the protocol years
    there (a part of the march around the windows, to keep the test small)."""
    years = common.schedule_c_years(-122000, PROTOCOL_122,
                                    common.schedule_c_windows(ice_file))
    years = years[(years >= first) & (years <= last)].astype(float)
    protocol = PROTOCOL_122[(PROTOCOL_122 >= first) & (PROTOCOL_122 <= last)]
    return years, protocol


@pytest.mark.parametrize("ice_file", [ICE6G, GLAC1, GLAC2])
def test_prepare_and_forward_make_the_same_ice(ice_file):
    """The ice of the input file (`march_ice`, the prepare script) and of the
    forward run (`LinearIce`) are equal bit for bit at every year; at a
    protocol year both are the protocol slice; between, the ice is linear."""
    mesh = fd.UnitSquareMesh(4, 4)
    V = fd.FunctionSpace(mesh, "CG", 1)
    years, protocol = window_march(ice_file)
    forcing = FakeForcing(V, len(protocol))
    brackets = [common.ice_bracket(protocol, y) for y in years]
    ice = common.march_ice(forcing.ice, brackets, V)
    # Each protocol slice read once by the prepare path.
    assert forcing.calls == {k: 1 for k in range(len(protocol))}
    # The check of the prepare script and of the forward script.
    check = common.check_linear_ice(years, ice, protocol, fd.Function(V))
    assert check["max_abs_diff"] == 0.0
    assert check["years_checked"] == len(years) - len(protocol)
    # The forward path from fresh slices.
    linear = fwd.LinearIce(FakeForcing(V, len(protocol)), protocol)
    out = fd.Function(V)
    for j, year in enumerate(years):
        k, w = linear.set(float(year), out)
        assert (k, w) == brackets[j]
        assert np.array_equal(out.dat.data_ro, ice[j].dat.data_ro)
        if year in protocol:
            assert np.array_equal(out.dat.data_ro, forcing.values[k])
        else:
            expected = (1 - w) * forcing.values[k] + w * forcing.values[k + 1]
            assert np.allclose(out.dat.data_ro, expected, rtol=0, atol=1e-18)


def test_ice_of_a_written_file_is_still_linear(tmp_path):
    """The path of the input file: the ice on a mantle `Submesh`, written
    through `ParentCopy` and `CheckpointFile` (as `write_inputs`), read back
    and interpolated onto the mantle again (as `read_inputs`). In memory the
    ice is linear bit for bit; after the round trip it is linear to
    round-off, inside the tolerance of the forward script (1e-12 of the
    largest ice), and the forward ice made from the protocol slices of the
    file equals the file's ice to the same tolerance."""
    base = fd.UnitSquareMesh(6, 6)
    x = fd.SpatialCoordinate(base)
    marker = fd.Function(fd.FunctionSpace(base, "DG", 0)).interpolate(
        fd.conditional(x[0] < 0.5, 1, 0))
    parent = fd.RelabeledMesh(base, [marker], [7], name="parent")
    mantle = fd.Submesh(parent, 2, 7)
    V = fd.FunctionSpace(mantle, "CG", 1)
    years, protocol = window_march(ICE6G, -15300, -14900)
    forcing = FakeForcing(V, len(protocol))
    ice = common.march_ice(forcing.ice,
                           [common.ice_bracket(protocol, y) for y in years], V)
    assert common.check_linear_ice(years, ice, protocol,
                                   fd.Function(V))["max_abs_diff"] == 0.0
    path = str(tmp_path / "ice.h5")
    copies = common.ParentCopy(parent)
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(parent)
        for j, f in enumerate(ice):
            chk.save_function(copies(common.ICE_NAME, f), idx=j)
    with fd.CheckpointFile(path, "r") as chk:
        loaded_parent = chk.load_mesh("parent")
        stored = [chk.load_function(loaded_parent, common.ICE_NAME, idx=j)
                  for j in range(len(years))]
    W = fd.FunctionSpace(fd.Submesh(loaded_parent, 2, 7), "CG", 1)
    loaded = [fd.Function(W).interpolate(f) for f in stored]
    check = common.check_linear_ice(years, loaded, protocol, fd.Function(W))
    tol = 1e-12 * check["max_abs_ice"]
    assert check["max_abs_diff"] <= tol
    # The forward ice from the protocol slices of the file.
    slices = {k: loaded[int(np.flatnonzero(years == p)[0])]
              for k, p in enumerate(protocol)}

    class FileForcing:
        def ice(self, k):
            return slices[k].copy(deepcopy=True)

    linear = fwd.LinearIce(FileForcing(), protocol)
    out = fd.Function(W)
    for j, year in enumerate(years):
        linear.set(float(year), out)
        assert np.max(np.abs(out.dat.data_ro - loaded[j].dat.data_ro)) <= tol


def test_check_linear_ice_finds_a_change():
    """The check reports a change of one node at one year."""
    mesh = fd.UnitSquareMesh(2, 2)
    V = fd.FunctionSpace(mesh, "CG", 1)
    years, protocol = window_march(ICE6G, -15300, -14900)
    forcing = FakeForcing(V, len(protocol))
    ice = common.march_ice(forcing.ice,
                           [common.ice_bracket(protocol, y) for y in years], V)
    j = int(np.flatnonzero(years == -14990.0)[0])
    ice[j].dat.data[0] += 1e-15
    check = common.check_linear_ice(years, ice, protocol, fd.Function(V))
    assert check["max_abs_diff"] > 0.0
