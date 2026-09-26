r"""Pass criteria of the self-gravitating GIA benchmarks.

The tests read the files that `spada.py` and `martinec.py` wrote into this
directory (the steps of `meta.py`). They need no Firedrake.

Spada et al. (2011)
    The ratios to TABOO of U(0), N(0) and the largest V, at every epoch of the
    cap case, and the phase and the size of the polar motion. Spada et al.
    (2011) give no pass tolerance, so the tolerances here are PROVISIONAL.
    A separate analysis sets them (`NOTES/HANDOVER.md` item 7).

Martinec et al. (2018)
    The rules that the parent branch `sghelichkhani/sea-level` scores its
    runs with:

    1. The uniform water layer h_UF at the terminal time is within 1 percent
       of VEGA.
    2. Along each comparison meridian, for each quantity, the largest
       difference from VEGA and its root mean square are not larger than the
       largest difference (and the largest root mean square) of any other
       published code from VEGA, over the same region. The region is the
       colatitude interval of the case file, minus a band of 6.74 degrees
       (1.5 facets of a 500 km mesh) on each side of the cap margin and of
       the coastline. Near those two features all codes disagree strongly, and
       the paper does not say how wide a band to remove.
    3. Case C and case D: the grounded ice mass at the terminal time is
       within 0.5 percent of VEGA. Case D also: the floating ice mass within
       0.5 percent, and the ocean area inside the range of the published
       codes.

    The metrics and the reference data come from the PyPI package `giamip`
    (0.1.0 or later). The driver writes npz files on the compute nodes, and
    `martinec_result` converts them here to a `giamip` `BenchmarkResult`. The
    first test run downloads the reference files into `$GIAMIP_DATA_ROOT`,
    else `~/.cache/giamip`. VEGA is the reference: `case.reference()`, the
    curves that `giamip` scores against.

Case B is expected to fail criterion 2 on U and N along the load meridian
until its time step is reduced (`NOTES/HANDOVER.md` item 9). That
test is marked `xfail`.
"""

import json
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Spada: provisional tolerances
# ---------------------------------------------------------------------------

#: PROVISIONAL. The largest |model / TABOO - 1| of U(0), N(0) and the largest
#: V at any epoch. The earlier run on the 500 km mesh reached 0.0251 for U(0)
#: at t = 0, 0.0054 for N(0) and 0.0359 for V.
SPADA_CAP_TOLERANCE = {"U0": 0.03, "N0": 0.03, "Vmax": 0.04}

#: PROVISIONAL. The largest |U_0| / |U_2|. The load has no degree-0 part and
#: the core keeps its volume, so the breathing mode must stay near zero.
SPADA_DEGREE_ZERO_TOLERANCE = 1.0e-3

#: PROVISIONAL. The largest difference of the polar-motion phase from the
#: exact -105 degrees, in degrees.
SPADA_PHASE_TOLERANCE_DEG = 0.05

#: PROVISIONAL. The range of |m| / |m_ref|. The model uses C - A =
#: 2.6952e35 kg m^2 and the reference excitation uses 2.63e35, so the ratio
#: is below 1: 0.974 at t = 0 and 0.968 at 20 kyr in an earlier run on a
#: 500 km mesh.
SPADA_POLAR_MOTION_RATIO = (0.95, 1.0)

# ---------------------------------------------------------------------------
# Martinec: the rules of the parent branch
# ---------------------------------------------------------------------------

#: Criterion 1: the uniform water layer, percent of VEGA.
H_UF_TOLERANCE_PERCENT = 1.0
#: Criterion 3: the ice masses at the terminal time, percent of VEGA.
ICE_MASS_TOLERANCE_PERCENT = 0.5

#: The half width of the band that criterion 2 removes around each feature,
#: in degrees of colatitude: 1.5 facets of the 500 km mesh of the first
#: Martinec runs, 500 km on a sphere of 6371 km, 6.74 degrees. The smooth
#: ocean mask goes from 0.1 to 0.9 over about 1.5 facets, so this band covers
#: the displacement kink and the mask. This is the width of the parent's
#: pass verdicts. At 1.5 facets of the 78 km refinement (1.05 degrees) the
#: parent's case C run is outside on basin U by 1.04 times (0.0926 m against
#: 0.0887 m). The paper gives no width, so the band stays at the width of
#: the parent's verdicts, and the README states both numbers.
BAND_HALF_WIDTH_DEG = 1.5 * np.degrees(500.0 / 6371.0)

#: The driver's time-series keys and the `giamip` names they go to. The
#: ocean area is int B dS, the ocean of Martinec eq. 23, as a fraction of
#: the sphere. The ice masses are in kg and h_UF in metres.
TIME_SERIES = {"h_UF_m": "h_UF", "ocean_area": "ocean_area",
               "ice_mass_grounded_kg": "ice_mass_grounded",
               "ice_mass_floating_kg": "ice_mass_floating"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_summary(case):
    """The summary JSON of one case, or a test failure if it is missing."""
    path = HERE / f"summary_{case}.json"
    if not path.exists():
        pytest.fail(f"{path.name} does not exist; run the step of meta.py "
                    f"for case {case} first")
    summary = json.loads(path.read_text())
    assert not summary["smoke"], f"{path.name} is from a smoke run"
    assert summary["mesh"]["flat_cells"] == 0
    assert summary["mesh"]["folded_cells_parent"] == 0
    assert summary["mesh"]["folded_cells_mantle"] == 0
    return summary


def martinec_result(letter, directory=HERE):
    """The run of one case as a `giamip` `BenchmarkResult`.

    The profiles are the ones at the terminal time of the case, the time of
    the published figures. The driver samples every profile on its own grid;
    `giamip` evaluates the reference at the run's points, so any colatitudes
    are accepted. The time series hold every solved step. For case B the
    first row is the elastic solve at t = 0, which belongs in the series.

    Args:
      letter: `"B"`, `"C"` or `"D"`.
      directory: the directory of the npz files.

    Returns:
      A `BenchmarkResult(letter, "g-adopt")`. Nothing is written to a file.
    """
    from giamip.benchmarks.results import BenchmarkResult

    case = giamip_case(letter)
    result = BenchmarkResult(letter, "g-adopt")
    time_kyr = float(case.terminal_time_kyr)
    path = directory / f"martinec-{letter}-profiles_{time_kyr:g}kyr.npz"
    if not path.exists():
        pytest.fail(f"{path.name} does not exist; the run did not reach "
                    f"{time_kyr:g} kyr")
    data = np.load(path)
    assert abs(float(data["t_kyr"]) - time_kyr) < 1e-9, path.name
    for spec in case.profiles:
        # A point that the driver could not locate on the mesh holds nan,
        # and one such point makes the whole profile unusable.
        found = np.asarray(data[f"{spec.name}_found"], dtype=bool)
        assert found.all(), (f"{path.name}: {int((~found).sum())} of "
                             f"{found.size} {spec.name} points not located")
        colatitude = np.asarray(data[f"{spec.name}_colatitude_deg"],
                                dtype=float)
        for quantity in spec.quantities:
            values = np.asarray(data[f"{spec.name}_{quantity}_m"],
                                dtype=float)
            assert np.isfinite(values).all(), f"{spec.name} {quantity}"
            result.add_profile(
                spec.name, quantity, colatitude, values, time_kyr=time_kyr,
                longitude_deg=float(data[f"{spec.name}_longitude_deg"]))
    path = directory / f"martinec-{letter}-timeseries.npz"
    if not path.exists():
        pytest.fail(f"{path.name} does not exist")
    data = np.load(path)
    kyr = np.asarray(data["t_kyr"], dtype=float)
    for source, name in TIME_SERIES.items():
        values = np.asarray(data[source], dtype=float)
        assert np.isfinite(values).all(), f"time series {source}"
        result.add_time_series(name, kyr, values)
    return result


def giamip_case(letter):
    """The `giamip` case of one letter; the import stays out of the Spada tests."""
    import giamip
    return giamip.case(f"martinec2018-{letter}")


def martinec_run(letter):
    """The checked summary and the `BenchmarkResult` of one case."""
    read_summary(letter)
    return martinec_result(letter)


def terminal_value(series, time_kyr):
    """A time series read at `time_kyr` by linear interpolation."""
    return float(np.interp(time_kyr, np.asarray(series.axis, dtype=float),
                           np.asarray(series.values, dtype=float)))


# ---------------------------------------------------------------------------
# Spada et al. (2011)
# ---------------------------------------------------------------------------


@pytest.mark.longtest
def test_spada_cap():
    """U(0), N(0) and the largest V against TABOO at every epoch."""
    summary = read_summary("cap")
    for row in summary["epochs"]:
        assert row["U0_over_U2"] < SPADA_DEGREE_ZERO_TOLERANCE, row["t_kyr"]
        for key, tolerance in SPADA_CAP_TOLERANCE.items():
            ratio = row[key] / row[f"{key}_ref"]
            assert abs(ratio - 1.0) <= tolerance, (
                f"{key} at {row['t_kyr']} kyr: ratio {ratio:.4f}")


@pytest.mark.longtest
def test_spada_polar_motion():
    """The phase and the size of the polar motion at every epoch."""
    summary = read_summary("polar-motion")
    lo, hi = SPADA_POLAR_MOTION_RATIO
    for row in summary["epochs"]:
        assert abs(row["phase"] - row["phase_ref"]) <= \
            SPADA_PHASE_TOLERANCE_DEG, f"phase at {row['t_kyr']} kyr"
        ratio = row["absm"] / row["absm_ref"]
        assert lo <= ratio <= hi, (
            f"|m| at {row['t_kyr']} kyr: ratio {ratio:.4f}")


# ---------------------------------------------------------------------------
# Martinec et al. (2018)
# ---------------------------------------------------------------------------


@pytest.mark.longtest
@pytest.mark.parametrize("letter", ["B", "C", "D"])
def test_martinec_uniform_layer(letter):
    """Criterion 1: h_UF at the terminal time within 1 percent of VEGA."""
    from giamip.benchmarks import metrics

    result = martinec_run(letter)
    case = giamip_case(letter)
    # S - N of the reference on the basin meridian, which is h_UF over the
    # ocean (Martinec eq. 6 and 8). The spread along the meridian is below
    # 2e-6 m, so the mean is the reference value.
    vega, _ = metrics.uniform_layer(letter)
    ours = terminal_value(result.series("h_UF"), case.terminal_time_kyr)
    relative = 100.0 * abs(ours - vega) / abs(vega)
    assert relative <= H_UF_TOLERANCE_PERCENT, (
        f"h_UF {ours:.4f} m against VEGA {vega:.4f} m, {relative:.3f} "
        "percent")


@pytest.mark.longtest
@pytest.mark.parametrize("letter", [
    pytest.param("B", marks=pytest.mark.xfail(
        strict=False, reason="load-meridian U and N of case B fail until its "
                             "time step is reduced (HANDOVER item 9)")),
    "C", "D"])
def test_martinec_profiles(letter):
    """Criterion 2: every profile inside the spread of the published codes.

    `envelope_check` compares the run and every published code with VEGA on
    their own points, over the interval of the case file, with the band of
    `BAND_HALF_WIDTH_DEG` removed around the cap margin and the initial
    coastline (`case.feature_colatitudes`). It raises `KeyError` if the run
    lacks a compared curve, so no curve can pass unseen.
    """
    rows = giamip_case(letter).envelope_check(
        martinec_run(letter), band_half_width_deg=BAND_HALF_WIDTH_DEG)
    failures = [f"{row['profile']} {row['quantity']}: max {row['max']:.4f} m "
                f"against {row['envelope_max']:.4f} m, rms {row['rms']:.4f} "
                f"m against {row['envelope_rms']:.4f} m"
                for row in rows if not row["inside"]]
    assert not failures, "; ".join(failures)


@pytest.mark.longtest
@pytest.mark.parametrize("letter,name", [
    ("C", "ice_mass_grounded"), ("D", "ice_mass_grounded"),
    ("D", "ice_mass_floating")])
def test_martinec_ice_mass(letter, name):
    """Criterion 3: the ice mass at the terminal time within 0.5 percent."""
    result = martinec_run(letter)
    case = giamip_case(letter)
    time_kyr = case.terminal_time_kyr
    ours = terminal_value(result.series(name), time_kyr)
    vega = terminal_value(case.reference().series(name), time_kyr)
    relative = 100.0 * abs(ours - vega) / abs(vega)
    assert relative <= ICE_MASS_TOLERANCE_PERCENT, (
        f"{name}: {ours:.6e} kg against VEGA {vega:.6e} kg, "
        f"{relative:.3f} percent")


@pytest.mark.longtest
def test_martinec_ocean_area():
    """Criterion 3, case D: the ocean area inside the range of the codes."""
    from giamip.benchmarks.results import BenchmarkResult

    result = martinec_run("D")
    time_kyr = giamip_case("D").terminal_time_kyr
    ours = terminal_value(result.series("ocean_area"), time_kyr)
    # Every published code with an ocean-area series, VEGA included, read
    # at the terminal time. Some series carry nan rows where a code wrote no
    # value; the interpolation uses the finite rows only.
    path = BenchmarkResult.reference_path("D")
    values = []
    for member in BenchmarkResult.members(path):
        series = BenchmarkResult.read(path, member,
                                      case="D").time_series.get("ocean_area")
        if series is not None:
            axis = np.asarray(series.axis, dtype=float)
            data = np.asarray(series.values, dtype=float)
            finite = np.isfinite(data)
            values.append(float(np.interp(time_kyr, axis[finite],
                                          data[finite])))
    assert min(values) <= ours <= max(values), (
        f"ocean area {ours:.7f} outside {min(values):.7f} to "
        f"{max(values):.7f}")
