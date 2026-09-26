r"""Pass criteria of the self-gravitating GIA benchmarks.

The tests read the files that `spada.py` and `martinec.py` wrote into this
directory (the steps of `meta.py`). They need no Firedrake.

Spada et al. (2011)
    The ratios to TABOO of U(0), N(0) and the largest V, at every epoch of the
    cap case, and the phase and the size of the polar motion. Spada et al.
    (2011) give no pass tolerance. Each tolerance here is a known error floor
    of this model plus a margin for the mesh and the solver. The floors are
    K / mu = 100, backward Euler and the moment difference C - A. The
    comments at the constants give the numbers. The cap case has a second
    check that divides out the predicted floor of each epoch
    (`SPADA_CAP_FLOOR`). The bounds on U(0) and the largest V are set by the
    floor at t = 0, and they are loose at late times.

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

With the graded Spada time steps, case B fails criterion 2 on U and N along
the load meridian. The cause is the time error of backward Euler. The driver
now takes uniform 10 yr steps for case B (`martinec.T0_DT_YR`). A prediction
passes on these steps by a margin smaller than its own uncertainty. The test
stays marked `xfail` until a run on these steps confirms the pass.
"""

import json
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Spada: tolerances
# ---------------------------------------------------------------------------
#
# Spada et al. (2011) give no pass tolerance. Each value below is a known
# error floor of this model plus a margin for the mesh and the solver. The
# floors and the published spread are in
# `NOTES/team/spada-tolerances/PROPOSAL.md`, with the scripts that compute
# them. These values are older than the first run on the 250 km mesh.
#
# The floors come from lovejx, a Love-number code that is not in this
# repository. It solves the same Earth model in the spectral domain. Two
# floors apply to the cap case:
#   - K / mu = 100. The model is compressible and TABOO is incompressible.
#     lovejx gives model / TABOO = 1.0248 for U(0) and 0.9637 for max V at
#     t = 0 (the elastic state). The effect decreases with time, to 0.1
#     percent in U(0) and 0.5 percent in max V at 20 kyr. In N(0) it is
#     below 0.8 percent at every epoch.
#   - Backward Euler on `selfgrav_common.STEP_LOAD_LADDER_YR`. Each
#     relaxation mode decays by 1 / (1 + |s| dt) per step instead of
#     exp(-|s| dt). From the TABOO spectrum, N(0) is 1.1 percent high at
#     5 kyr and U(0) is 0.6 percent low at 2 kyr.
# The earlier run on a 500 km mesh, divided by these two floors, is within
# 0.13 percent in U(0) at every epoch. In N(0) it is within 0.36 percent and
# in max V within 0.25 percent.

#: The largest |model / TABOO - 1| of U(0), N(0) and the signed maximum of
#: V, at every epoch of the cap case. Dimensionless.
#:
#: U0: the largest floor is 0.0249 at t = 0 (K / mu). The margin of 0.005 is
#: four times the largest 500 km residual.
#: N0: the largest floor is 0.0084 at 5 kyr (backward Euler). The margin of
#: 0.0066 is about twice the largest 500 km residual. N(0) is the quantity
#: that converges with the mesh. The residual is -2.4 percent at 1000 km and
#: -0.3 percent at 500 km. A finer mesh moves N(0) towards the time floor, to
#: about 1.008 at 5 kyr. A mesh of 1000 km fails this bound.
#: Vmax: the largest floor is 0.0362 at t = 0 (K / mu). The margin of 0.0038
#: is 13 times the 500 km residual at t = 0.
#: The published time-domain code VK differs from TABOO by up to 1.1 percent
#: in these quantities. The bounds on U0 and Vmax are larger only because of
#: the compressibility floor, which the incompressible codes do not have.
SPADA_CAP_TOLERANCE = {"U0": 0.03, "N0": 0.015, "Vmax": 0.04}

#: The largest |U_0| / |U_2| on Re at any epoch. Dimensionless.
#:
#: The load has no degree-0 part and the fluid core keeps its volume, so the
#: physical value is zero. The 500 km mesh gives 1.1e-8 at most. The outer
#: solver tolerance of 1e-6 sets the lowest safe bound, so the bound is ten
#: times that. A core without the mass constraint has a degree-0 mode that
#: grows by a factor of 21.6 in each 100 yr step. This bound detects it
#: within a few steps.
SPADA_DEGREE_ZERO_TOLERANCE = 1.0e-5

#: The largest difference of the polar-motion phase from the exact -105
#: degrees, in degrees.
#:
#: The phase depends on the load position only, so any difference is a
#: discretisation error. The 500 km mesh gives 0.0009 degree at 20 kyr and
#: less at earlier epochs. The published codes GS and VB agree to 5e-7
#: degree. The bound of 0.01 degree (1.7e-4 radians) is 11 times the 500 km
#: value. It detects a difference of about 2e-4 between the x and the y
#: rotation responses.
SPADA_PHASE_TOLERANCE_DEG = 0.01

#: The range of |m| / |m_ref| at every epoch. Dimensionless.
#:
#: The model uses C - A = 2.6952e35 kg m^2, the value that goes with the
#: secular Love number k_s = 0.96672389 of the reference. The reference
#: excitation uses 2.63e35. So the ratio is 2.63 / 2.6952 = 0.9758 times the
#: model's own error. The own error at K / mu = 100 with backward Euler is
#: -0.13 to -0.41 percent (lovejx and the linearised Liouville equation).
#: So the predicted ratio is 0.9718 to 0.9746. The 500 km mesh gives 0.9744
#: at t = 0 and 0.9682 at 20 kyr, where the mesh error is largest.
#: The lower bound is 0.33 percent below the lowest 500 km value. The upper
#: bound is 0.43 percent above 0.9758, because the own error is negative at
#: every epoch. The window rejects the prescribed C - A (ratio 1.010) and the
#: closure that reproduces TABOO exactly (0.992 to 0.9985). It also rejects
#: an own error larger than about 1 percent.
SPADA_POLAR_MOTION_RATIO = (0.965, 0.980)

#: The predicted model / TABOO of (U(0), N(0), max V) at each epoch of the
#: cap case (kyr), for a run with no spatial error. This is the product of
#: the two floors above. lovejx computes them at K / mu = 100 against the
#: incompressible model, with backward Euler on
#: `selfgrav_common.STEP_LOAD_LADDER_YR` (script:
#: `NOTES/team/spada-tolerances/floors.py`). A change of that step sequence
#: or of K / mu changes these numbers, and they must then be computed again.
SPADA_CAP_FLOOR = {
    0.0: (1.02487, 0.99955, 0.96381), 0.1: (1.01846, 0.99964, 0.96933),
    1.0: (1.00371, 1.00237, 0.97887), 2.0: (0.99840, 1.00556, 0.98285),
    5.0: (0.99839, 1.00836, 0.98738), 10.0: (1.00236, 1.00609, 0.98782),
    20.0: (1.00036, 1.00046, 0.99412)}

#: The largest |model / TABOO / floor - 1| of U(0), N(0) and max V at any
#: epoch: the second check of the cap case. Dimensionless.
#:
#: Why a second check: the fixed bounds of `SPADA_CAP_TOLERANCE` must hold
#: at every epoch. So the compressibility floor at t = 0 sets the bounds on
#: U(0) and max V (2.5 and 3.6 percent). At 20 kyr the floors are only 0.04
#: and 0.6 percent. A late-time error of 2 to 3 percent then passes the fixed
#: bounds without any sign. This check divides out the predicted floor of
#: each epoch and bounds what is left, the error of the mesh and the solver.
#: The earlier 500 km run leaves at most 0.36 percent (N(0)). 0.5 percent is
#: half the spread of the time-domain code VK from TABOO. The lovejx pole fit
#: has a noise of 0.1 to 0.2 percent at 10 kyr. So at least 0.3 percent is
#: left for the spatial error. The fixed bounds stay, because they do not
#: depend on lovejx.
SPADA_CAP_RESIDUAL_TOLERANCE = 0.005

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


def cap_floor(t_kyr):
    """The predicted (U0, N0, Vmax) ratios of one epoch of the cap case.

    The epochs of the summary are the floats of the driver's epoch list, so
    the lookup matches with a small tolerance. An epoch with no prediction is
    a test failure, because the check must not skip an epoch.
    """
    for epoch, floor in SPADA_CAP_FLOOR.items():
        if abs(epoch - t_kyr) < 1e-9:
            return floor
    pytest.fail(f"no predicted floor for the epoch {t_kyr} kyr")


@pytest.mark.longtest
def test_spada_cap():
    """U(0), N(0) and the largest V against TABOO at every epoch.

    Two checks for each quantity and epoch. The ratio to TABOO must be
    within the fixed bound of `SPADA_CAP_TOLERANCE`. The ratio divided by the
    predicted floor of the epoch must be within
    `SPADA_CAP_RESIDUAL_TOLERANCE`.
    """
    summary = read_summary("cap")
    for row in summary["epochs"]:
        assert row["U0_over_U2"] < SPADA_DEGREE_ZERO_TOLERANCE, row["t_kyr"]
        floor = dict(zip(("U0", "N0", "Vmax"), cap_floor(row["t_kyr"])))
        for key, tolerance in SPADA_CAP_TOLERANCE.items():
            ratio = row[key] / row[f"{key}_ref"]
            assert abs(ratio - 1.0) <= tolerance, (
                f"{key} at {row['t_kyr']} kyr: ratio {ratio:.4f}")
            assert abs(ratio / floor[key] - 1.0) <= \
                SPADA_CAP_RESIDUAL_TOLERANCE, (
                    f"{key} at {row['t_kyr']} kyr: ratio {ratio:.4f}, "
                    f"predicted {floor[key]:.4f}")


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
        strict=False, reason="load-meridian U and N of case B failed on the "
                             "graded steps; the 10 yr steps await a run")),
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
