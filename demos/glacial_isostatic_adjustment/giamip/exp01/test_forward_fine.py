"""Serial tests of the fine schedules of `forward_exp01.py` (design handout 12.7).

The schedules, the ice between protocol years and the parameter log of a
restarted run. No solver runs here; the restart itself is tested on
Setonix with the 3-D mesh. Run with the PYTHONPATH of CLAUDE.md:

    python -m pytest demos/glacial_isostatic_adjustment/giamip/exp01/test_forward_fine.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_exp01 as fwd  # noqa: E402  (imports gadopt, then firedrake)
import firedrake as fd  # noqa: E402

#: The protocol years of a 122 ka experiment: 500 yr steps to -26000, then
#: 100 yr steps to +1000 (`GiamipExperiment.model_years`, protocol v4.0).
PROTOCOL_122 = np.concatenate([np.arange(-122000, -26000, 500),
                               np.arange(-26000, 1001, 100)]).astype(float)
PROTOCOL_80 = PROTOCOL_122[PROTOCOL_122 >= -80000]


@pytest.mark.parametrize("protocol, before, after", [
    (PROTOCOL_122, 1920, 2700), (PROTOCOL_80, 1080, 2700)])
def test_schedule_a_counts(protocol, before, after):
    """Schedule A: 50 yr to -26000, 10 yr to +1000; 4620 steps from 122 ka."""
    years = fwd.fine_years("fine", int(protocol[0]), protocol)
    dts = np.diff(years)
    assert len(years) - 1 == before + after
    assert np.all(dts[years[:-1] < -26000] == 50)
    assert np.all(dts[years[:-1] >= -26000] == 10)
    assert set(protocol.astype(int)) <= set(years.tolist())
    assert years[-1] == 1000 and 0 in years


def test_schedule_b_windows():
    """Schedule B: 100 yr, then 25 yr, and 10 yr in the windows of the ice history."""
    windows = fwd.FINE_B_WINDOWS["iceHistory-ICE-6G_D.nc"]
    years = fwd.fine_years("fine_b", -122000, PROTOCOL_122, windows)
    dts = dict(zip(years[:-1].tolist(), np.diff(years).tolist()))
    assert dts[-122000] == 100 and dts[-26100] == 100
    assert dts[-26000] == 25 and dts[-15025] == 25
    assert dts[-15000] == 10 and dts[-13010] == 10 and dts[-13000] == 25
    assert set(PROTOCOL_122.astype(int)) <= set(years.tolist())
    # 960 steps before 26 ka, 2000 yr of 10 yr steps and 25000 yr of 25 yr.
    assert len(years) - 1 == 960 + 200 + 1000


def test_a_schedule_that_misses_a_protocol_year_is_refused():
    """A schedule that misses a protocol year is refused."""
    with pytest.raises(RuntimeError):
        fwd.fine_years("fine", -122030, PROTOCOL_122)


class FakeForcing:
    """The ice slices of `SurfaceForcing.ice`, made from random nodal values.

    Every call returns a new Function, as the real transfer does, and the
    calls are counted.
    """

    def __init__(self, space, n):
        rng = np.random.default_rng(1)
        self.values = rng.random((n, space.dof_dset.size))
        self.space = space
        self.calls = 0

    def ice(self, index):
        self.calls += 1
        f = fd.Function(self.space)
        f.dat.data[:] = self.values[index]
        return f


def test_linear_ice_is_exact_at_protocol_years_and_linear_between():
    """Bit for bit at a protocol year; (1 - w) a + w b between; each slice read once."""
    mesh = fd.UnitSquareMesh(3, 3)
    V = fd.FunctionSpace(mesh, "CG", 1)
    years = PROTOCOL_122[:5]
    forcing = FakeForcing(V, len(years))
    linear = fwd.LinearIce(forcing, years)
    out = fd.Function(V)
    march = fwd.fine_years("fine", int(years[0]), PROTOCOL_122)
    march = march[march <= years[-1]]
    for year in march:
        k, w = linear.set(float(year), out)
        if year in years:
            assert w == 0.0
            assert np.array_equal(out.dat.data_ro, forcing.values[k])
        else:
            expected = ((1 - w) * forcing.values[k]
                        + w * forcing.values[k + 1])
            assert np.allclose(out.dat.data_ro, expected, rtol=0,
                               atol=1e-15)
            assert w == pytest.approx((year - years[k])
                                      / (years[k + 1] - years[k]))
    # Every slice read once, and never more than two kept.
    assert forcing.calls == len(years)
    assert len(linear.slices) <= 2


def test_params_log_restart_cuts_and_appends(tmp_path):
    """A restarted log keeps the header and the rows up to the checkpoint output."""
    path = tmp_path / "params.log"
    log = fwd.ParamsLog(str(path), ("step", "year"))
    for k in range(5):
        log.log_str(f"{k} {k * 100.0}")
    log.close()
    log = fwd.ParamsLog(str(path), ("step", "year"), append=True,
                        keep_upto=2)
    log.log_str("3 300.0")
    log.close()
    lines = path.read_text().splitlines()
    assert lines == ["step year", "0 0.0", "1 100.0", "2 200.0", "3 300.0"]
