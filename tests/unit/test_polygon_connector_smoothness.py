"""Regression test for smooth-edge behaviour of PolygonConnector.

The current `PolygonConnector` produces a sharp 1->0 cliff at the polygon
edge: `IndicatorConnector._load_data` calls `PolygonFilter.filter_inside`
to discard exterior seeds, so the source cloud carries a constant
thickness over its whole support, and the lateral transition collapses
to the binary `too_far` cutoff (plus the `_apply_indicator` override
that snaps `too_far` mesh nodes to 0).

The proposed fix (see SMOOTH_POLYGON_INDICATOR.md) is "mask and relabel":
keep all seeds, set thickness inside the polygon to the requested value
and 0 outside, then let the Gaussian kNN interpolator smear the boundary
over a length set by `gaussian_sigma`. The spec mandates three production
changes:
    1. `PolygonConnector` uses a mask-and-relabel loader.
    2. `PolygonConnector._apply_indicator` override is deleted.
    3. `PolygonConfig.__post_init__` sets `interpolation.default_value=0.0`.

This module pins all three:
    * `test_polygon_config_default_value_is_zero` independently asserts
      the default-value contract from spec step 3, without depending on
      the indicator computation.
    * `test_polygon_indicator_is_smooth_across_continental_edge` exercises
      steps 1 and 2 by walking an arc across the Pacific edge of South
      America and out into the abyssal Pacific past the production-default
      `distance_threshold`, asserting smooth monotonic decay rather than
      a cliff drop at the `too_far` boundary.

The test is EXPECTED TO FAIL on the current `main` of this worktree --
that is the point. After the fix in SMOOTH_POLYGON_INDICATOR.md lands,
it should pass.

Runtime: ~30-60 s on a laptop.
"""

from pathlib import Path

import numpy as np
import pytest

from gadopt.gplates import (
    InterpolationConfig,
    PolygonConnector,
    PolygonConfig,
    pyGplatesConnector,
    ensure_reconstruction,
)


# ---------------------------------------------------------------------------
# Test geometry: an equatorial arc (which is a great circle exactly) across
# the western edge of South America. Longitudes are chosen so the arc starts
# well inside the continent (Amazon basin) and ends in the abyssal Pacific,
# deliberately past the production-default `distance_threshold` so that the
# tail samples exercise the `too_far` branch in `_apply_indicator`.
# ---------------------------------------------------------------------------
ARC_LAT = 0.0             # equator
ARC_LON_INSIDE = -65.0    # well inside South America (Amazon)
# abyssal Pacific, past distance_threshold from any continental seed
# (also well past the Galapagos so the bound is robust to polygon-set updates)
ARC_LON_OUTSIDE = -130.0
# samples along the arc; ~36 km step over the ~7200 km arc, fine enough to land
# >=5 samples in the (0.05, 0.95) mid-band for sigma=0.03 (~190 km transition
# width)
N_ARC = 200

# Sampling radius. We do NOT sample at `r_outer` exactly: with the
# mask-and-relabel fix in place, far-from-polygon mesh nodes interpolate to
# thickness ~ 0, so the base-class tanh evaluates `0.5 * (1 + tanh(0)) = 0.5`
# at the outer surface itself. That floors the outside-polygon indicator at
# 0.5, which makes the `indicator[-1] < 0.2` headline assertion structurally
# unreachable.
#
# Sample instead at `r = r_outer - 2.5 * transition_width_nondim`, i.e. a few
# transition widths *below* the outer surface and inside the lithosphere
# range. With the connector defaults (r_outer=2.208, transition_width=10 km,
# depth_scale=2890 km), transition_width_nondim = 10/2890 ~= 3.46e-3 and the
# sampling radius is 2.208 - 8.65e-3 ~= 2.1993. At this radius:
#   - Continental column (thickness=50 km): base_r = r_outer - 50/2890; the
#     argument (r - base_r)/w = (-2.5*w + 50/2890)/w = +2.5, so
#     indicator = 0.5*(1+tanh(2.5)) ~= 0.993.
#   - Ocean column (thickness~=0 after mask-and-relabel): base_r ~= r_outer;
#     (r - base_r)/w = -2.5, so indicator = 0.5*(1+tanh(-2.5)) ~= 0.007.
# Both endpoints land squarely on the [<0.2, >0.8] cliff regression bounds.
#
# This is the sampling-shell-in-the-lithosphere reading of the spec ("at
# depth in the lithosphere") rather than the above-surface reading; the
# above-surface reading is incompatible with the connector's tanh sign
# convention (indicator -> 1 as r -> infinity, not 0).
R_OUTER_NONDIM = PolygonConfig().r_outer
TRANSITION_WIDTH_NONDIM = PolygonConfig().transition_width / PolygonConfig().depth_scale
R_SAMPLE_NONDIM = R_OUTER_NONDIM - 2.5 * TRANSITION_WIDTH_NONDIM

# Parameters chosen to expose smoothing length cleanly:
# - thickness_data=50.0 km (matches saved Continental_Crust field)
# - gaussian_sigma=0.03 rad on unit sphere => ~190 km on Earth
# - distance_threshold defaults to 0.1 rad (~640 km): the production
#   default. The arc tail at -130 deg W lies past this threshold from any
#   continental seed, so the test exercises the `too_far` path and would
#   catch a retained `PolygonConnector._apply_indicator` override (which
#   would hard-snap those samples to 0).
SIGMA = 0.03
R_EARTH_KM = 6371.0
THICKNESS_KM = 50.0


def _arc_xyz(r=R_SAMPLE_NONDIM, n=N_ARC):
    """Return (n, 3) xyz coords on a sphere of radius `r`, walking along an
    equatorial arc from ARC_LON_INSIDE to ARC_LON_OUTSIDE.

    The equator is a great circle exactly, so there is no constant-latitude
    approximation to worry about.
    """
    lons = np.linspace(ARC_LON_INSIDE, ARC_LON_OUTSIDE, n)
    lat = np.deg2rad(ARC_LAT)
    lon_rad = np.deg2rad(lons)
    x = r * np.cos(lat) * np.cos(lon_rad)
    y = r * np.cos(lat) * np.sin(lon_rad)
    z = r * np.sin(lat) * np.ones_like(lon_rad)
    return np.column_stack([x, y, z]), lons


# ---------------------------------------------------------------------------
# Spec step 3: independent assertion that `PolygonConfig` ships a
# `default_value` of 0.0 in its bundled `InterpolationConfig`. This is the
# production contract; without it, base `_apply_indicator` would tanh the
# 200-km fill at `too_far` mesh nodes and give indicator ~ 1 in regions far
# from any seed -- exactly opposite to the desired behaviour.
# ---------------------------------------------------------------------------
def test_polygon_config_default_value_is_zero():
    cfg = PolygonConfig()
    assert cfg.interpolation.default_value == 0.0, (
        "PolygonConfig must default `interpolation.default_value` to 0.0 "
        "so that `too_far` mesh nodes are filled with zero thickness, not "
        "the 200 km fill that would tanh-up to indicator ~ 1 in the deep "
        "ocean. See SMOOTH_POLYGON_INDICATOR.md step 3."
    )


# ---------------------------------------------------------------------------
# Fixtures: plate-reconstruction data and the configured connector.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gplates_data_path():
    p = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    if not (p / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists():
        pytest.skip(
            "Plate reconstruction data not available. "
            "Run `make data` in demos/mantle_convection/gplates_global."
        )
    return p


@pytest.fixture(scope="module")
def gplates_connector(gplates_data_path):
    muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)
    return pyGplatesConnector(
        rotation_filenames=muller_files["rotation_filenames"],
        topology_filenames=muller_files["topology_filenames"],
        oldest_age=200,
        continental_polygons=muller_files.get("continental_polygons"),
        static_polygons=muller_files.get("static_polygons"),
    )


@pytest.fixture(scope="module")
def continental_polygons_path(gplates_data_path):
    muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)
    cp = muller_files.get("continental_polygons")
    if cp is None:
        pytest.skip("Continental polygons not available in plate-model bundle")
    return cp


@pytest.fixture(scope="module")
def smooth_polygon_connector(gplates_connector, continental_polygons_path):
    # Do NOT override `default_value` here. The test must run against the
    # production `PolygonConfig` default (asserted separately by
    # `test_polygon_config_default_value_is_zero`); pinning it locally would
    # mask a missed step-3 fix and let the smoothness test pass green even
    # if production still ships `default_value=200.0`.
    #
    # `k_neighbors=200` is needed here: at `gaussian_sigma=0.03` (~190 km)
    # and `n_points=20000` (~160 km seed spacing), the default k=50 cuts
    # off the Gaussian tail before it can bridge the boundary, collapsing
    # the kernel to near-IDW behaviour. The `InterpolationConfig` docstring
    # already flags that k>=200 is needed for small sigma. Production users
    # who pick sigma at this scale will need the same bump until/unless the
    # kernel auto-picks k from sigma vs. mean seed spacing (out of scope
    # here; see SMOOTH_POLYGON_INDICATOR.md tuning-knobs section, which
    # recommends `n_points=80000` as the alternative for narrower sigma).
    config = PolygonConfig(
        n_points=20000,
        interpolation=InterpolationConfig(
            kernel="gaussian",
            k_neighbors=200,
            gaussian_sigma=SIGMA,
        ),
        transition_width=10.0,
    )
    return PolygonConnector(
        gplates_connector=gplates_connector,
        polygons=continental_polygons_path,
        thickness_data=THICKNESS_KM,
        config=config,
    )


# ---------------------------------------------------------------------------
# Headline regression test.
# ---------------------------------------------------------------------------
def test_polygon_indicator_is_smooth_across_continental_edge(
    smooth_polygon_connector,
):
    """Walk an arc across the South-American Pacific coast and check the
    indicator transitions smoothly from ~1 inside the continent to ~0 over
    the abyssal Pacific.

    This test catches four failure modes:

      (i)   monotonicity violations -- a non-monotonic ramp would mean the
            kNN average is picking up isolated seeds that punch through the
            boundary, e.g. from seed-spacing aliasing.
      (ii)  cliff edges -- the OLD code transitions in a single sample
            (>=0.98 to <=0.02), so fewer than 5 samples lie strictly in
            (0.05, 0.95). This is the headline regression we want to guard.
      (iii) over-sharp gradients -- a peak |dI/ds| larger than the loose
            Gaussian roll-off bound indicates a residual cliff.
      (iv)  a retained `PolygonConnector._apply_indicator` override -- the
            arc tail sits past `distance_threshold` from any continental
            seed, so a kept override would hard-snap the trailing samples
            to exactly 0 and produce a cliff at the `too_far` boundary
            rather than the smooth tanh tail the spec requires.
    """
    target_coords, lons = _arc_xyz()
    ndtime = smooth_polygon_connector.age2ndtime(0.0)
    indicator = smooth_polygon_connector.get_indicator(target_coords, ndtime)

    assert indicator.shape == (N_ARC,)
    assert np.all(np.isfinite(indicator))
    assert indicator.min() >= 0.0 - 1e-12
    assert indicator.max() <= 1.0 + 1e-12

    # Sanity: the two endpoints should be on opposite sides of the edge.
    # Inside (Amazon) ~ 1, outside (abyssal Pacific) ~ 0.
    assert indicator[0] > 0.8, (
        f"Inside-continent sample at lon={lons[0]} should be >0.8, "
        f"got {indicator[0]:.3f}. Either the polygon set doesn't cover "
        f"the Amazon, or the connector failed to load."
    )
    assert indicator[-1] < 0.2, (
        f"Abyssal-Pacific sample at lon={lons[-1]} should be <0.2, "
        f"got {indicator[-1]:.3f}. The halo may be far too wide."
    )

    # (i) Monotonic decrease as we walk from inside to outside.
    # Allow a tiny epsilon for floating-point noise.
    diffs = np.diff(indicator)
    assert np.all(diffs <= 1e-6), (
        "Indicator must be monotonically non-increasing across the edge. "
        f"Found positive increments at indices "
        f"{np.where(diffs > 1e-6)[0].tolist()}."
    )

    # (ii) At least 5 samples strictly in (0.05, 0.95). This is the
    # headline regression: the old cliff implementation jumps in a single
    # sample.
    mid_band = (indicator > 0.05) & (indicator < 0.95)
    n_in_band = int(mid_band.sum())
    assert n_in_band >= 5, (
        f"Expected at least 5 samples strictly in (0.05, 0.95) along the "
        f"arc, got {n_in_band}. This usually means the indicator still "
        f"has a 1->0 cliff at the polygon edge. Full profile: "
        f"{np.array2string(indicator, precision=3)}."
    )

    # Geometry: arc step in great-circle length. On the equator, longitude
    # spacing equals arc length exactly, but compute it the unambiguous way
    # so this code remains correct if anyone ever moves the arc off the
    # equator.
    unit_a = target_coords[0] / np.linalg.norm(target_coords[0])
    unit_b = target_coords[1] / np.linalg.norm(target_coords[1])
    arc_step_rad = np.arccos(np.clip(unit_a @ unit_b, -1.0, 1.0))
    arc_step_km = arc_step_rad * R_EARTH_KM

    # (iii) Bound on peak gradient. The Gaussian RBF's maximum derivative
    # (per unit angular distance on the unit sphere) is bounded by
    # ~1 / (sqrt(2 pi) * sigma) for a step input. We compute the along-arc
    # finite difference per unit length (km) and assert the peak is below
    # 1 / (0.5 * sigma * R_earth) -- the spec's looser bound. This bound
    # is intentionally loose to tolerate the discrete kNN approximation;
    # the tighter width check below carries most of the regression-guard
    # weight.
    grad = np.abs(diffs) / arc_step_km
    peak_grad = grad.max()
    bound = 1.0 / (0.5 * SIGMA * R_EARTH_KM)
    assert peak_grad <= bound, (
        f"Peak gradient {peak_grad:.3e} per km exceeds Gaussian roll-off "
        f"bound {bound:.3e} per km (sigma={SIGMA} rad). Suggests a "
        f"residual cliff. Profile: "
        f"{np.array2string(indicator, precision=3)}."
    )

    # (iv) Width of the transition band. The headline-physics claim is
    # "the halo is ~150 km wide" for sigma=0.03. Assert that the arc
    # length between the last sample at >=0.95 and the first sample at
    # <=0.05 is at least 0.5 * sigma * R_earth (~95 km). This catches the
    # regression class "transition exists but is too narrow" with margin
    # to spare, without depending on finite-difference peak resolution
    # being well-sampled (which the (iii) bound does depend on).
    inside_idx = np.where(indicator >= 0.95)[0]
    outside_idx = np.where(indicator <= 0.05)[0]
    assert inside_idx.size > 0 and outside_idx.size > 0, (
        "Arc must straddle both the inside (>=0.95) and outside (<=0.05) "
        "regimes for the width check to be meaningful. Profile: "
        f"{np.array2string(indicator, precision=3)}."
    )
    last_inside = inside_idx.max()
    first_outside = outside_idx.min()
    transition_n_steps = first_outside - last_inside
    transition_width_km = transition_n_steps * arc_step_km
    min_width_km = 0.5 * SIGMA * R_EARTH_KM
    assert transition_width_km >= min_width_km, (
        f"Transition from indicator>=0.95 to indicator<=0.05 spans only "
        f"{transition_width_km:.0f} km along the arc, below the "
        f"{min_width_km:.0f} km lower bound implied by sigma={SIGMA} rad. "
        f"This is the cliff-regression signature. Profile: "
        f"{np.array2string(indicator, precision=3)}."
    )

    # Anti-cliff guard on the `too_far` tail. The arc end is far enough
    # outside the polygon that some trailing samples are past the
    # production-default `distance_threshold=0.1 rad` from any continental
    # seed (we deliberately did not override that threshold above). A
    # retained `PolygonConnector._apply_indicator` override would hard-snap
    # those samples to exactly 0; without the override, the base class at
    # our sub-surface sampling radius gives a small positive
    # `0.5*(1+tanh(-2.5)) ~= 0.0067` for thickness-0 columns (the
    # default_value=0 path through `_apply_indicator`). Distinct from 0, so
    # `not all == 0` discriminates: the override path produces exact 0 at
    # too_far nodes; the corrected base-class path produces ~0.0067.
    tail = indicator[-5:]
    assert not np.all(tail == 0.0), (
        "Last 5 indicator samples are all exactly 0. This is the "
        "signature of a retained `PolygonConnector._apply_indicator` "
        "override hard-snapping `too_far` mesh nodes to zero, instead "
        "of letting the base-class tanh produce a smooth tail. "
        "SMOOTH_POLYGON_INDICATOR.md step 2 mandates deleting that "
        f"override. Tail values: {tail.tolist()}."
    )


# ---------------------------------------------------------------------------
# Margin-aligned seed loss regression (SMOOTH-CONTINENTS-IMPLMEMENTATION-05-19.md)
#
# Before the 1-NN plate-ID inheritance fix, seeds in the polygon-set mismatch
# sliver (thickness>0 from the continental mask but plate_id=0 from the static
# plate polygons) were dropped by `assign_plate_ids(..., remove_undefined=True)`.
# The fix switches to `remove_undefined=False` and inherits each undefined
# continental seed's plate ID from its nearest defined continental neighbour
# on the unit sphere -- restricting donors to the continental subset so the
# halo cannot accidentally ride an oceanic plate at older ages.
# ---------------------------------------------------------------------------
def test_no_continental_seeds_dropped_at_polygon_set_mismatch(
    smooth_polygon_connector,
):
    """All continental seeds survive the plate-ID step with a defined plate.

    The mask step marks ~N continental seeds (thickness=50). Pre-fix, any of
    those seeds sitting in the static-vs-continental polygon sliver were
    dropped. Post-fix, they remain in the cloud and carry a non-zero plate ID
    inherited from a nearby continental seed.
    """
    cloud = smooth_polygon_connector._region_present
    assert cloud is not None, "Connector root state missing region cloud."

    thickness = cloud.get_property("thickness")
    continental = thickness > 0.0
    n_continental = int(continental.sum())
    assert n_continental > 0, "Fixture has no continental seeds; mask broken."

    undefined_continental = int(((cloud.plate_ids == 0) & continental).sum())
    assert undefined_continental == 0, (
        f"{undefined_continental} continental seeds (out of {n_continental}) "
        "still carry plate_id=0 after the 1-NN patch. The margin sliver fix "
        "must leave every continental seed with a defined plate ID."
    )


def test_undefined_continental_seeds_inherit_continental_plate_id(
    smooth_polygon_connector,
):
    """Re-derive the raw plate IDs and check the 1-NN inheritance contract.

    For each seed that was originally undefined AND continental (thickness>0),
    its post-fix plate_id must equal the plate_id of its nearest defined
    continental neighbour on the unit sphere. Restricting donors to the
    continental subset is the load-bearing safety property: it prevents the
    halo from riding an oceanic plate over geological time.
    """
    from scipy.spatial import cKDTree
    from gtrack.point_rotation import _get_plate_ids

    connector = smooth_polygon_connector
    cloud = connector._region_present
    rotator = connector._rotator

    raw_ids = _get_plate_ids(
        cloud.xyz,
        rotator.static_polygons,
        rotator.rotation_model,
        0.0,
    )
    thickness = cloud.get_property("thickness")
    continental = thickness > 0.0
    originally_undefined = raw_ids == 0
    target = originally_undefined & continental
    donor = (~originally_undefined) & continental

    if int(target.sum()) == 0:
        pytest.skip(
            "No undefined continental seeds in this fixture; the polygon "
            "sets agree everywhere a continental seed landed. Nothing to "
            "verify."
        )

    donor_xyz = cloud.xyz[donor]
    donor_unit = donor_xyz / np.linalg.norm(donor_xyz, axis=1, keepdims=True)
    target_xyz = cloud.xyz[target]
    target_unit = target_xyz / np.linalg.norm(target_xyz, axis=1, keepdims=True)
    tree = cKDTree(donor_unit)
    _, idx = tree.query(target_unit, k=1)

    expected = raw_ids[donor][idx]
    actual = cloud.plate_ids[target]
    np.testing.assert_array_equal(actual, expected)
