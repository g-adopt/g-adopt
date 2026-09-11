"""Tests for OutputStrategy subclasses and MeshConfig.

No reconstruction data needed; everything in this file constructs its inputs
directly. The tests check the numerical behaviour of the indicator and
geotherm transformations, and the validation contracts of the small
dataclasses: defaults are what we expect, input parameters are honoured,
and invalid values are rejected.
"""

import numpy as np
import pytest

from gadopt.gplates import (
    InterpolatedBaseDepth,
    MembershipCorrectedBaseDepth,
    FixedBaseDepth,
    HalfSpaceCoolingGeotherm,
    LinearGeotherm,
    LayerIndicator,
    MembershipField,
    BoundedLinearGeotherm,
    BoundedLayerIndicator,
    MembershipLateralWeight,
    MeshConfig,
    UniformLateralWeight,
    GlobalLayerIndicator,
    RadialQuinticTransition,
    SourceLateralWeight,
    MappedMembershipWeight,
    continental_linear,
    ocean_erf_normalized,
    radial_quintic_step,
)
# Internal to the deblend rather than public API, so imported from the module
# it lives in rather than widening gadopt.gplates.
from gadopt.gplates.outputs import MEMBERSHIP_FLOOR


# ---------------------------------------------------------------------------
# MeshConfig validation
# ---------------------------------------------------------------------------

class TestMeshConfig:
    def test_defaults(self):
        mesh = MeshConfig()
        assert mesh.r_outer == 2.208
        assert mesh.depth_scale == 2890.0

    def test_custom_values(self):
        mesh = MeshConfig(r_outer=1.5, depth_scale=1000.0)
        assert mesh.r_outer == 1.5
        assert mesh.depth_scale == 1000.0

    def test_rejects_nonpositive_r_outer(self):
        with pytest.raises(ValueError, match="r_outer must be positive"):
            MeshConfig(r_outer=-1.0)
        with pytest.raises(ValueError, match="r_outer must be positive"):
            MeshConfig(r_outer=0.0)

    def test_rejects_nonpositive_depth_scale(self):
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            MeshConfig(depth_scale=0.0)


# ---------------------------------------------------------------------------
# Geotherm functions
# ---------------------------------------------------------------------------

class TestOceanErfNormalized:
    def test_surface_is_zero(self):
        z_lab = np.array([100e3, 50e3, 150e3])
        depth = np.zeros_like(z_lab)
        age = np.array([50.0, 100.0, 10.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, thermal_diffusivity_m2_per_s=1e-6)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_lab_is_one(self):
        z_lab = np.array([100e3, 50e3, 150e3])
        depth = z_lab.copy()
        age = np.array([50.0, 100.0, 10.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, thermal_diffusivity_m2_per_s=1e-6)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_monotone_in_depth(self):
        z_lab = 100e3
        depths = np.linspace(0, z_lab, 50)
        z_labs = np.full_like(depths, z_lab)
        ages = np.full_like(depths, 80.0)
        result = ocean_erf_normalized(depths, z_labs, age_myr=ages, thermal_diffusivity_m2_per_s=1e-6)
        assert np.all(np.diff(result) >= 0)

    def test_clipped_to_unit_interval(self):
        z_lab = np.array([100e3])
        depth = np.array([200e3])  # deeper than LAB
        age = np.array([80.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, thermal_diffusivity_m2_per_s=1e-6)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_young_vs_old(self):
        # A young ocean has a steeper profile than an old ocean.
        depth = np.array([50e3])
        z_lab = np.array([100e3])
        young = ocean_erf_normalized(depth, z_lab, age_myr=np.array([5.0]), thermal_diffusivity_m2_per_s=1e-6)
        old = ocean_erf_normalized(depth, z_lab, age_myr=np.array([200.0]), thermal_diffusivity_m2_per_s=1e-6)
        assert young[0] > old[0]

    def test_zero_age_returns_finite(self):
        # The age floor keeps the calculation finite at zero material age.
        depth = np.array([10e3])
        z_lab = np.array([100e3])
        age = np.array([0.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, thermal_diffusivity_m2_per_s=1e-6)
        assert np.all(np.isfinite(result))
        assert np.all((result >= 0.0) & (result <= 1.0))


class TestContinentalLinear:
    def test_surface_is_zero(self):
        z_lab = np.array([200e3, 150e3])
        depth = np.zeros_like(z_lab)
        result = continental_linear(depth, z_lab)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_lab_is_one(self):
        z_lab = np.array([200e3, 150e3])
        depth = z_lab.copy()
        result = continental_linear(depth, z_lab)
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_midpoint(self):
        result = continental_linear(np.array([100e3]), np.array([200e3]))
        np.testing.assert_allclose(result, 0.5, atol=1e-12)

    def test_clipped_to_unit_interval(self):
        result = continental_linear(np.array([200e3]), np.array([100e3]))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_zero_lab_returns_zero(self):
        # A zero base depth represents no lithosphere.
        result = continental_linear(np.array([50e3]), np.array([0.0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# radial_quintic_step — shared radial primitive
# ---------------------------------------------------------------------------

class TestRadialQuinticTransition:
    """Test the limits and smoothness of the radial transition.

    The result is one above the base and zero below the transition band.
    The midpoint value is 0.5. The base radius can be scalar or node-dependent.
    """

    def test_exact_plateaus_and_midpoint(self):
        base_r = 2.0
        width = 0.01
        # The result is one at and above the base.
        assert radial_quintic_step(base_r, base_r, width) == 1.0
        assert radial_quintic_step(base_r + 0.05, base_r, width) == 1.0
        # The result is zero at and below the lower limit.
        np.testing.assert_allclose(
            radial_quintic_step(base_r - width, base_r, width), 0.0, atol=1e-30
        )
        assert radial_quintic_step(base_r - 0.05, base_r, width) == 0.0
        # The midpoint value is 0.5.
        np.testing.assert_allclose(
            radial_quintic_step(base_r - width / 2, base_r, width),
            0.5, rtol=1e-12,
        )
        # The transition increases monotonically.
        rs = np.linspace(base_r - width, base_r, 50)
        f = radial_quintic_step(rs, base_r, width)
        assert np.all(np.diff(f) > 0)

    def test_point_symmetry_about_band_midpoint(self):
        # The transition is point-symmetric about its midpoint.
        base_r = 2.0
        width = 0.02
        mid = base_r - width / 2
        d = np.array([0.001, 0.005, 0.009])
        np.testing.assert_allclose(
            radial_quintic_step(mid + d, base_r, width)
            + radial_quintic_step(mid - d, base_r, width),
            1.0, rtol=1e-12,
        )

    def test_flat_junctions(self):
        # The first derivative approaches zero at both limits.
        base_r = 2.0
        width = 0.02
        eps = 1e-6 * width

        def slope_at(r):
            return (
                radial_quintic_step(r + eps, base_r, width)
                - radial_quintic_step(r - eps, base_r, width)
            ) / (2 * eps)
        mid_slope = slope_at(base_r - width / 2)
        assert abs(slope_at(base_r - eps)) < 1e-6 * mid_slope
        assert abs(slope_at(base_r - width + eps)) < 1e-6 * mid_slope

    def test_scalar_and_array_base_broadcast(self):
        # Scalar base_r and a per-node array base_r both broadcast against an
        # array r_target. A constant array base must reproduce the scalar case.
        r = np.array([2.0, 2.05, 2.1])
        width = 0.02
        scalar = radial_quintic_step(r, 2.05, width)
        array = radial_quintic_step(r, np.full_like(r, 2.05), width)
        np.testing.assert_allclose(scalar, array, rtol=1e-12)
        # A genuinely per-node base shifts each band independently: each node
        # sitting exactly on its own base reads exactly 1.
        per_node_base = r.copy()
        np.testing.assert_array_equal(
            radial_quintic_step(r, per_node_base, width), 1.0
        )

    def test_narrower_width_is_steeper(self):
        # A narrower transition has a larger gradient magnitude.
        base_r = 2.0
        rs = np.linspace(base_r - 0.05, base_r + 0.05, 200)
        narrow = radial_quintic_step(rs, base_r, 0.005)
        wide = radial_quintic_step(rs, base_r, 0.05)
        assert np.abs(np.gradient(narrow)).max() > np.abs(np.gradient(wide)).max()


# ---------------------------------------------------------------------------
# GlobalLayerIndicator with variable base depth and uniform lateral weight
# ---------------------------------------------------------------------------

class TestGlobalLayerIndicatorVariableBase:
    """Test ``GlobalLayerIndicator.compute`` with synthetic inputs.

    Default configuration: per-node base depth from the thickness channel,
    no lateral weight. The indicator is exactly 1 from the surface down to the
    base, exactly 0 below base + width.
    """

    @staticmethod
    def _interp(thickness_km, n_targets=1):
        # Helper: a single interpolated thickness value broadcast to n_targets.
        return {"thickness": np.full(n_targets, thickness_km, dtype=float)}

    def test_validation_rejects_bad_args(self):
        with pytest.raises(ValueError, match="base_transition_width_km must be positive"):
            GlobalLayerIndicator(base_transition_width_km=0.0)
        with pytest.raises(ValueError, match="base_transition_width_km must be positive"):
            GlobalLayerIndicator(base_transition_width_km=-1.0)
        with pytest.raises(ValueError, match="fixed_base_depth_km must be positive"):
            GlobalLayerIndicator(fixed_base_depth_km=0.0)
        with pytest.raises(ValueError, match="fallback_thickness_km must be non-negative"):
            GlobalLayerIndicator(fallback_thickness_km=-1.0)
        # Zero is allowed (polygon seeds-plus-zero-background uses default 0).
        GlobalLayerIndicator(fallback_thickness_km=0.0)

    def test_value_at_base_is_one(self):
        # The transition lies below the base, so its value at the base is one.
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = GlobalLayerIndicator(base_transition_width_km=10.0)
        thickness = 100.0
        interp = self._interp(thickness)
        outside_source_range = np.array([False])
        r_target = np.array([mesh.r_outer - thickness / mesh.depth_scale])
        result = out.compute(interp, r_target, outside_source_range, mesh)
        np.testing.assert_array_equal(result, 1.0)

    def test_half_crossing_half_width_below_base(self):
        mesh = MeshConfig()
        width = 10.0
        thickness = 100.0
        out = GlobalLayerIndicator(base_transition_width_km=width)
        r_target = np.array(
            [mesh.r_outer - (thickness + width / 2) / mesh.depth_scale]
        )
        result = out.compute(
            self._interp(thickness), r_target, np.array([False]), mesh
        )
        np.testing.assert_allclose(result, 0.5, rtol=1e-10)

    def test_value_inside_lithosphere_is_one_exact(self):
        mesh = MeshConfig()
        out = GlobalLayerIndicator(base_transition_width_km=10.0)
        # 50 km depth, with lithosphere 200 km thick: well inside.
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = self._interp(200.0)
        result = out.compute(interp, r_target, np.array([False]), mesh)
        np.testing.assert_array_equal(result, 1.0)

    def test_value_below_transition_is_zero_exact(self):
        mesh = MeshConfig()
        out = GlobalLayerIndicator(base_transition_width_km=10.0)
        # 400 km depth, with lithosphere 200 km thick: well below base+width.
        r_target = np.array([mesh.r_outer - 400.0 / mesh.depth_scale])
        interp = self._interp(200.0)
        result = out.compute(interp, r_target, np.array([False]), mesh)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_thickness_surface_skin(self):
        # When thickness is zero, the base coincides with the surface.
        # The surface node reads one because the transition lies below it.
        # This is why a zero-outside source is refused a GlobalLayerIndicator
        # outright: "absent" and "present but infinitely thin" are the same
        # number in a thickness channel, and the step reads both as present.
        # One width below the surface the column is exactly 0 again.
        mesh = MeshConfig()
        width = 10.0
        out = GlobalLayerIndicator(base_transition_width_km=width)
        surface = out.compute(
            self._interp(0.0), np.array([mesh.r_outer]), np.array([False]), mesh
        )
        np.testing.assert_array_equal(surface, 1.0)
        below = out.compute(
            self._interp(0.0),
            np.array([mesh.r_outer - width / mesh.depth_scale]),
            np.array([False]), mesh,
        )
        np.testing.assert_allclose(below, 0.0, atol=1e-30)

    def test_outside_source_range_uses_default_thickness(self):
        # A zero fallback puts the base at the surface. Deeper targets read zero.
        mesh = MeshConfig()
        out = GlobalLayerIndicator(base_transition_width_km=10.0, fallback_thickness_km=0.0)
        r_target = np.array([mesh.r_outer - 100.0 / mesh.depth_scale])
        interp = self._interp(200.0)  # value ignored when outside_source_range=True
        outside_source_range = np.array([True])
        result = out.compute(interp, r_target, outside_source_range, mesh)
        np.testing.assert_array_equal(result, 0.0)

        # A positive fallback supplies a base depth outside the source range.
        out_lith = GlobalLayerIndicator(base_transition_width_km=10.0, fallback_thickness_km=100.0)
        # A target at 50 km depth lies above the 100 km fallback base.
        r_target_inside = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        result_inside = out_lith.compute(interp, r_target_inside, np.array([True]), mesh)
        np.testing.assert_array_equal(result_inside, 1.0)

    def test_variable_base_moves_per_node(self):
        # Distinct columns get distinct bases from their own thickness: each
        # reads exactly 1 at its own base and exactly 0.5 half a width below
        # it. A fixed-base configuration could not reproduce both at once.
        mesh = MeshConfig()
        w_r = 10.0
        out = GlobalLayerIndicator(base_transition_width_km=w_r)
        thickness = np.array([75.0, 300.0])
        outside_source_range = np.zeros(2, dtype=bool)

        r_at_base = mesh.r_outer - thickness / mesh.depth_scale
        at_base = out.compute(
            {"thickness": thickness.copy()}, r_at_base, outside_source_range, mesh
        )
        np.testing.assert_allclose(at_base, 1.0, rtol=1e-12)

        r_mid_band = mesh.r_outer - (thickness + w_r / 2) / mesh.depth_scale
        mid_band = out.compute(
            {"thickness": thickness.copy()}, r_mid_band, outside_source_range, mesh
        )
        np.testing.assert_allclose(mid_band, 0.5, rtol=1e-10)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = GlobalLayerIndicator(base_transition_width_km=10.0)
        thickness = np.array([999.0, 10.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)


# ---------------------------------------------------------------------------
# The oceanic-only lateral_weight channel
# ---------------------------------------------------------------------------

class TestSourceLateralWeight:
    """Read, clip, and apply the source-provided lateral weight.

    Nodes outside the source range receive full weight.
    """

    def test_reads_the_channel_clipped(self):
        lat = SourceLateralWeight()
        interp = {"lateral_weight": np.array([0.0, 0.3, 1.0, 1.5, -0.2])}
        outside_source_range = np.zeros(5, dtype=bool)
        np.testing.assert_array_equal(
            lat.weight(interp, outside_source_range), [0.0, 0.3, 1.0, 1.0, 0.0]
        )

    def test_outside_source_range_reads_full_weight(self):
        lat = SourceLateralWeight()
        interp = {"lateral_weight": np.array([0.1, 0.2])}
        np.testing.assert_array_equal(
            lat.weight(interp, np.array([False, True])), [0.1, 1.0]
        )

    def test_does_not_mutate_input(self):
        lat = SourceLateralWeight()
        a = np.array([0.2, 0.4])
        interp = {"lateral_weight": a.copy()}
        lat.weight(interp, np.array([True, False]))
        np.testing.assert_array_equal(interp["lateral_weight"], a)


class TestGlobalLayerIndicatorSourceWeight:
    """The source strategy multiplies the radial value by `lateral_weight`."""

    def test_requires_only_thickness_uniform(self):
        assert GlobalLayerIndicator(base_transition_width_km=10.0).requires == frozenset({"thickness"})

    def test_requires_adds_lateral_weight_with_source_weight(self):
        out = GlobalLayerIndicator(base_transition_width_km=10.0, lateral_weight=SourceLateralWeight())
        assert out.requires == frozenset({"thickness", "lateral_weight"})

    def test_weight_scales_the_surface_step(self):
        # At the surface skin the step reads 1, so the field equals the lateral weight.
        mesh = MeshConfig()
        out = GlobalLayerIndicator(base_transition_width_km=10.0, lateral_weight=SourceLateralWeight())
        r_target = np.full(3, mesh.r_outer - 5.0 / mesh.depth_scale)
        interp = {
            "thickness": np.full(3, 100.0),
            "lateral_weight": np.array([0.05, 0.5, 1.0]),
        }
        result = out.compute(interp, r_target, np.zeros(3, dtype=bool), mesh)
        np.testing.assert_allclose(result, [0.05, 0.5, 1.0])


# ---------------------------------------------------------------------------
# BoundedLayerIndicator
# ---------------------------------------------------------------------------

class TestBoundedLayerIndicator:
    """Test bounded indicators with synthetic source channels.

    The source supplies `masked_thickness = membership * thickness`.
    The indicator recovers thickness before it locates the radial transition.
    It then applies membership as the lateral weight.
    """

    @staticmethod
    def _interp(membership, thickness_km):
        """Source channels for a node with in-region fraction m and depth h."""
        m = np.asarray(membership, dtype=float)
        h = np.asarray(thickness_km, dtype=float)
        return {"membership": m, "masked_thickness": m * h}

    def test_validation_rejects_bad_args(self):
        with pytest.raises(ValueError, match="base_transition_width_km must be positive"):
            BoundedLayerIndicator(base_transition_width_km=0.0)
        with pytest.raises(ValueError, match="base_transition_width_km must be positive"):
            BoundedLayerIndicator(base_transition_width_km=-1.0)
        with pytest.raises(ValueError, match="fixed_base_depth_km must be positive"):
            BoundedLayerIndicator(fixed_base_depth_km=0.0)

    @pytest.mark.parametrize("cls", [BoundedLayerIndicator, GlobalLayerIndicator])
    def test_no_fade_ref_km_parameter(self, cls):
        """Reject the removed reference-thickness approximation.

        A single reference depth cannot recover spatially variable thickness.
        It also makes the interior weight depend on physical depth.
        """
        with pytest.raises(TypeError):
            cls(fade_ref_km=150.0)

    def test_surface_reads_membership_exactly(self):
        """Return membership as the surface value."""
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        m = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        interp = self._interp(m, 200.0)
        r_target = np.full(m.size, mesh.r_outer)
        result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
        np.testing.assert_allclose(result, m)

    def test_base_depth_does_not_shallow_with_membership(self):
        """Keep physical base depth independent of membership.

        A partially covered node must keep the data's own depth. Reading the
        weighted channel directly moves a 200 km base to 50 km at membership 0.25.
        """
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        m = np.array([1.0, 0.75, 0.5, 0.25])
        interp = self._interp(m, 200.0)
        # 190 km is above the 200 km base for every node, whatever m is.
        r_target = np.full(m.size, mesh.r_outer - 190.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
        np.testing.assert_allclose(result, m)

    def test_lateral_extent_is_independent_of_depth_data(self):
        """Keep lateral membership independent of physical depth.

        Equal membership produces equal values above both base depths.
        """
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        m = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        r_target = np.full(m.size, mesh.r_outer - 40.0 / mesh.depth_scale)
        outside_source_range = np.zeros(m.size, bool)
        shallow = out.compute(self._interp(m, 100.0), r_target, outside_source_range, mesh)
        deep = out.compute(self._interp(m, 400.0), r_target, outside_source_range, mesh)
        np.testing.assert_allclose(shallow, deep)
        np.testing.assert_allclose(shallow, m)

    def test_below_base_plus_width_is_zero(self):
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        m = np.array([1.0, 0.5])
        interp = self._interp(m, 200.0)
        r_target = np.full(m.size, mesh.r_outer - 260.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_membership_is_zero_at_the_surface(self):
        """Return zero at the surface outside the bounded region."""
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        interp = self._interp(np.array([0.0]), 200.0)
        result = out.compute(
            interp, np.array([mesh.r_outer]), np.array([False]), mesh
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_outside_source_range_is_outside_the_region(self):
        """No nearby source point is a statement about membership, not depth,
        so there is no default thickness to fill in."""
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        interp = self._interp(np.array([1.0, 1.0]), 200.0)
        r_target = np.full(2, mesh.r_outer)
        result = out.compute(interp, r_target, np.array([False, True]), mesh)
        np.testing.assert_allclose(result, [1.0, 0.0])

    def test_fixed_base_depth_overrides_the_data(self):
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=10.0, fixed_base_depth_km=50.0)
        m = np.array([1.0, 1.0])
        interp = self._interp(m, 300.0)  # data says 300 km; fixed base says 50
        # 70 km is clear of the 50 + 10 junction, so the fixed base has fully
        # decayed. The junction can contain floating-point residue from the
        # transition. See TestRadialQuinticTransition.
        r_target = np.full(2, mesh.r_outer - 70.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(2, bool), mesh)
        np.testing.assert_array_equal(result, 0.0)

    def test_recovers_a_laterally_varying_depth_exactly(self):
        """Recover a distinct physical base depth at each target node.

        The selected depths span a representative cratonic range.
        Each column uses its own depth and membership.
        """
        mesh = MeshConfig()
        w_r = 20.0
        out = BoundedLayerIndicator(base_transition_width_km=w_r)
        h = np.array([150.0, 200.0, 275.0, 350.0])
        m = np.array([1.0, 0.6, 0.35, 0.9])
        interp = {"membership": m.copy(), "masked_thickness": m * h}
        outside_source_range = np.zeros(m.size, bool)

        # Each column returns its membership at its own base.
        at_base = out.compute(
            interp, mesh.r_outer - h / mesh.depth_scale, outside_source_range, mesh
        )
        np.testing.assert_allclose(at_base, m, rtol=1e-12)

        # Half a transition width below the base gives half the membership.
        mid = out.compute(
            interp, mesh.r_outer - (h + w_r / 2) / mesh.depth_scale, outside_source_range, mesh
        )
        np.testing.assert_allclose(mid, 0.5 * m, rtol=1e-10)

    def test_transition_band_is_bounded_and_monotone(self):
        """Keep the field finite and monotonic near the membership floor.
        """
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=20.0)
        m = np.concatenate([
            np.geomspace(1.0, MEMBERSHIP_FLOOR, 60),
            np.geomspace(MEMBERSHIP_FLOOR, 1e-12, 40),
            np.array([0.0]),
        ])
        interp = {"membership": m.copy(), "masked_thickness": m * 200.0}
        outside_source_range = np.zeros(m.size, bool)

        for depth_km in (0.0, 100.0, 199.0, 210.0):
            r_target = np.full(m.size, mesh.r_outer - depth_km / mesh.depth_scale)
            result = out.compute(interp, r_target, outside_source_range, mesh)
            assert np.all(np.isfinite(result)), depth_km
            assert np.all(result >= 0.0) and np.all(result <= 1.0), depth_km
            # The field cannot increase as membership decreases.
            assert np.all(np.diff(result) <= 1e-12), depth_km

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = BoundedLayerIndicator(base_transition_width_km=50.0)
        interp = self._interp(np.array([0.5, 1.0]), 200.0)
        before = {k: v.copy() for k, v in interp.items()}
        out.compute(
            interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh
        )
        for k, v in before.items():
            np.testing.assert_array_equal(interp[k], v)


class TestLayerIndicator:
    """Test the composable strategies used by the indicator presets."""

    @staticmethod
    def _masked_interp(membership, thickness_km):
        m = np.asarray(membership, dtype=float)
        h = np.asarray(thickness_km, dtype=float)
        return {"membership": m, "masked_thickness": m * h}

    def test_requires_is_the_union_of_the_parts(self):
        assert LayerIndicator(
            RadialQuinticTransition(10.0), FixedBaseDepth(50.0), UniformLateralWeight()
        ).requires == frozenset()
        assert LayerIndicator(
            RadialQuinticTransition(10.0), InterpolatedBaseDepth(), UniformLateralWeight()
        ).requires == frozenset({"thickness"})
        assert LayerIndicator(
            RadialQuinticTransition(10.0), MembershipCorrectedBaseDepth(), MembershipLateralWeight()
        ).requires == frozenset({"masked_thickness", "membership"})
        # A fixed base with a membership weight reads membership only.
        assert LayerIndicator(
            RadialQuinticTransition(10.0), FixedBaseDepth(50.0), MembershipLateralWeight()
        ).requires == frozenset({"membership"})

    def test_reproduces_global_indicator(self):
        mesh = MeshConfig()
        interp = {"thickness": np.array([80.0, 150.0, 220.0])}
        r_target = np.full(3, mesh.r_outer - 100.0 / mesh.depth_scale)
        outside_source_range = np.array([False, True, False])
        composed = LayerIndicator(
            RadialQuinticTransition(10.0), InterpolatedBaseDepth(fallback_thickness_km=100.0), UniformLateralWeight()
        ).compute(interp, r_target, outside_source_range, mesh)
        preset = GlobalLayerIndicator(
            base_transition_width_km=10.0, fallback_thickness_km=100.0
        ).compute(interp, r_target, outside_source_range, mesh)
        np.testing.assert_array_equal(composed, preset)

    def test_reproduces_bounded_indicator(self):
        mesh = MeshConfig()
        m = np.array([1.0, 0.6, 0.25, 0.0])
        interp = self._masked_interp(m, np.array([150.0, 200.0, 300.0, 100.0]))
        r_target = np.full(4, mesh.r_outer - 120.0 / mesh.depth_scale)
        outside_source_range = np.array([False, False, True, False])
        composed = LayerIndicator(
            RadialQuinticTransition(20.0), MembershipCorrectedBaseDepth(), MembershipLateralWeight()
        ).compute(interp, r_target, outside_source_range, mesh)
        preset = BoundedLayerIndicator(base_transition_width_km=20.0).compute(
            interp, r_target, outside_source_range, mesh
        )
        np.testing.assert_array_equal(composed, preset)

    def test_tapered_weight_with_identity_is_membership(self):
        mesh = MeshConfig()
        m = np.array([1.0, 0.5, 0.0])
        interp = self._masked_interp(m, 200.0)
        r_target = np.full(3, mesh.r_outer)
        outside_source_range = np.zeros(3, bool)
        tapered = LayerIndicator(
            RadialQuinticTransition(20.0), MembershipCorrectedBaseDepth(), MappedMembershipWeight(lambda x: x)
        ).compute(interp, r_target, outside_source_range, mesh)
        plain = LayerIndicator(
            RadialQuinticTransition(20.0), MembershipCorrectedBaseDepth(), MembershipLateralWeight()
        ).compute(interp, r_target, outside_source_range, mesh)
        np.testing.assert_array_equal(tapered, plain)

    def test_tapered_weight_sharpens_the_margin(self):
        # A nonlinear map can reduce partial membership values.
        mesh = MeshConfig()
        m = np.array([0.25, 0.5, 0.75])
        interp = self._masked_interp(m, 200.0)
        r_target = np.full(3, mesh.r_outer)
        outside_source_range = np.zeros(3, bool)
        squared = LayerIndicator(
            RadialQuinticTransition(20.0), MembershipCorrectedBaseDepth(), MappedMembershipWeight(lambda x: x ** 2)
        ).compute(interp, r_target, outside_source_range, mesh)
        # At the surface the step is 1, so the field is exactly g(m) = m^2.
        np.testing.assert_allclose(squared, m ** 2, rtol=1e-12)


# ---------------------------------------------------------------------------
# HalfSpaceCoolingGeotherm and LinearGeotherm
# ---------------------------------------------------------------------------

class TestHalfSpaceCoolingGeotherm:
    def test_validation_rejects_nonpositive_kappa(self):
        with pytest.raises(ValueError, match="thermal_diffusivity_m2_per_s must be positive"):
            HalfSpaceCoolingGeotherm(thermal_diffusivity_m2_per_s=0.0)

    def test_validation_rejects_nonpositive_default_thickness(self):
        with pytest.raises(ValueError, match="fallback_thickness_km must be positive"):
            HalfSpaceCoolingGeotherm(fallback_thickness_km=0.0)

    def test_validation_rejects_nonpositive_outside_source_range_age(self):
        with pytest.raises(ValueError, match="fallback_age_myr must be positive"):
            HalfSpaceCoolingGeotherm(fallback_age_myr=-1.0)

    def test_surface_is_zero_lab_is_one(self):
        # Two target points: one at the surface, one at the LAB depth.
        # The output must match the underlying error-function geotherm.
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = HalfSpaceCoolingGeotherm(thermal_diffusivity_m2_per_s=1e-6)
        thickness_km = 100.0
        r_surface = mesh.r_outer
        r_lab = mesh.r_outer - thickness_km / mesh.depth_scale
        r_target = np.array([r_surface, r_lab])
        interp = {
            "thickness": np.full(2, thickness_km),
            "age": np.full(2, 80.0),
        }
        result = out.compute(interp, r_target, np.array([False, False]), mesh)
        assert result[0] < 1e-6
        assert result[1] > 1.0 - 1e-3

    def test_outside_source_range_uses_fallback_age_and_thickness(self):
        # Out-of-range nodes use the configured thickness and material age.
        mesh = MeshConfig()
        fallback_thick = 100.0
        fallback_age = 500.0
        out = HalfSpaceCoolingGeotherm(
            thermal_diffusivity_m2_per_s=1e-6,
            fallback_thickness_km=fallback_thick,
            fallback_age_myr=fallback_age,
        )
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = {
            "thickness": np.array([999.0]),  # value ignored when outside_source_range
            "age": np.array([12345.0]),       # value ignored when outside_source_range
        }
        result = out.compute(interp, r_target, np.array([True]), mesh)
        depth_m = (mesh.r_outer - r_target[0]) * mesh.depth_scale * 1e3
        expected = ocean_erf_normalized(
            np.array([depth_m]),
            np.array([fallback_thick * 1e3]),
            age_myr=np.array([fallback_age]),
            thermal_diffusivity_m2_per_s=1e-6,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestLinearGeotherm:
    def test_outside_source_range_is_mantle(self):
        # Out-of-range nodes receive mantle temperature.
        mesh = MeshConfig()
        out = LinearGeotherm()
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = {"thickness": np.array([100.0])}
        result = out.compute(interp, r_target, np.array([True]), mesh)
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_inside_region_linear(self):
        # Nodes within the source range follow the linear geotherm.
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = LinearGeotherm()
        thickness_km = 100.0
        # Mid-depth in a linear profile has a value of 0.5.
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = {"thickness": np.array([thickness_km])}
        result = out.compute(interp, r_target, np.array([False]), mesh)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)


class TestBoundedLinearGeotherm:
    """Recover physical base depth before evaluating the bounded geotherm.

    A bounded source supplies `masked_thickness = membership * thickness`.
    Reading this product as physical thickness makes the lithosphere shallower
    across the boundary. Division by membership removes this numerical effect.
    """

    @staticmethod
    def _interp(membership, thickness_km):
        m = np.asarray(membership, dtype=float)
        h = np.asarray(thickness_km, dtype=float)
        return {"membership": m, "masked_thickness": m * h}

    def test_deblended_profile_is_independent_of_membership(self):
        # Equal physical depths give equal temperatures for different nonzero
        # membership values.
        mesh = MeshConfig()
        out = BoundedLinearGeotherm()
        m = np.array([1.0, 0.5])
        interp = self._interp(m, 100.0)  # h = 100 km both nodes
        r_target = np.full(2, mesh.r_outer - 50.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(2, bool), mesh)
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-10)

    def test_outside_the_region_is_mantle_not_surface(self):
        # An uncovered node represents mantle, not the zero-depth surface value.
        mesh = MeshConfig()
        out = BoundedLinearGeotherm()
        # Node 0: covered. Node 1: membership below the floor. Node 2: outside_source_range.
        m = np.array([1.0, MEMBERSHIP_FLOOR / 2.0, 1.0])
        interp = self._interp(m, 100.0)
        r_target = np.full(3, mesh.r_outer - 50.0 / mesh.depth_scale)
        outside_source_range = np.array([False, False, True])
        result = out.compute(interp, r_target, outside_source_range, mesh)
        # Node 0 is a genuine mid-lithosphere reading; nodes 1 and 2 are mantle.
        np.testing.assert_allclose(result, [0.5, 1.0, 1.0], atol=1e-10)

    def test_stays_bounded_and_finite_through_the_floor(self):
        # The membership floor keeps the correction finite near the boundary.
        mesh = MeshConfig()
        out = BoundedLinearGeotherm()
        m = np.concatenate([
            np.geomspace(1.0, MEMBERSHIP_FLOOR, 50),
            np.geomspace(MEMBERSHIP_FLOOR, 1e-12, 50),
            np.array([0.0]),
        ])
        interp = self._interp(m, 200.0)
        for depth_km in (0.0, 50.0, 150.0, 250.0):
            r_target = np.full(m.size, mesh.r_outer - depth_km / mesh.depth_scale)
            result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
            assert np.all(np.isfinite(result)), depth_km
            assert np.all(result >= 0.0) and np.all(result <= 1.0), depth_km

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = BoundedLinearGeotherm()
        interp = self._interp(np.array([0.5, 1.0]), 200.0)
        before = {k: v.copy() for k, v in interp.items()}
        out.compute(
            interp, np.full(2, mesh.r_outer - 50.0 / mesh.depth_scale),
            np.array([True, False]), mesh,
        )
        for k, v in before.items():
            np.testing.assert_array_equal(interp[k], v)


# ---------------------------------------------------------------------------
# MembershipField — pure lateral membership, no radial dependence
# ---------------------------------------------------------------------------

class TestMembershipField:
    """Return clipped membership without radial dependence.

    Nodes outside the source range receive zero.
    """

    def test_no_radial_dependence(self):
        # Radius does not affect the membership field.
        mesh = MeshConfig()
        out = MembershipField()
        interp = {"membership": np.array([0.2, 0.8, 1.0])}
        outside_source_range = np.zeros(3, dtype=bool)
        shallow = out.compute(interp, np.full(3, mesh.r_outer), outside_source_range, mesh)
        deep = out.compute(interp, np.full(3, mesh.r_outer - 0.5), outside_source_range, mesh)
        np.testing.assert_array_equal(shallow, deep)

    def test_clipping_passthrough_and_outside_source_range(self):
        # Clip membership and set out-of-range nodes to zero.
        mesh = MeshConfig()
        out = MembershipField()
        interp = {"membership": np.array([-0.3, 0.0, 0.37, 1.0, 5.0, 0.9])}
        outside_source_range = np.array([False, False, False, False, False, True])
        result = out.compute(interp, np.full(6, mesh.r_outer), outside_source_range, mesh)
        np.testing.assert_allclose(
            result, [0.0, 0.0, 0.37, 1.0, 1.0, 0.0], rtol=1e-12
        )

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = MembershipField()
        membership = np.array([-0.3, 0.37, 5.0])
        interp = {"membership": membership.copy()}
        out.compute(interp, np.full(3, mesh.r_outer), np.array([False, False, True]), mesh)
        np.testing.assert_array_equal(interp["membership"], membership)


# ---------------------------------------------------------------------------
# GlobalLayerIndicator, fixed base — the radial step pinned at one depth
# ---------------------------------------------------------------------------

class TestGlobalLayerIndicatorFixedBaseDepth:
    """Keep the radial transition at one depth for every target node.

    The public channel contract still requires `thickness`, but a fixed base
    does not use its values.
    """

    def test_fixed_base_depth_independent_of_thickness(self):
        # Thickness does not move a fixed radial transition.
        mesh = MeshConfig()
        crust = 50.0
        out = GlobalLayerIndicator(base_transition_width_km=10.0, fixed_base_depth_km=crust)
        base_r = mesh.r_outer - crust / mesh.depth_scale
        for thickness in (50.0, 100.0, 300.0):
            r_target = np.array([base_r])
            result = out.compute(
                {"thickness": np.array([thickness])}, r_target, np.array([False]), mesh
            )
            np.testing.assert_array_equal(result, 1.0)


# ---------------------------------------------------------------------------
# The public ``requires`` contract of every concrete output, side by side.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output, expected",
    [
        (BoundedLayerIndicator(), frozenset({"masked_thickness", "membership"})),
        (HalfSpaceCoolingGeotherm(), frozenset({"thickness", "age"})),
        (LinearGeotherm(), frozenset({"thickness"})),
        (BoundedLinearGeotherm(), frozenset({"masked_thickness", "membership"})),
        # MembershipField reads the source's own membership channel rather
        # than a duplicated source carrying a constant in the thickness slot, so
        # it stays consistent with an indicator built from that source.
        (MembershipField(), frozenset({"membership"})),
    ],
)
def test_output_requires_contract(output, expected):
    """Each concrete output declares the source channels it reads via its
    class-level ``requires`` set; the connector validates this against a
    source's ``provides``. LayerIndicator's *computed* union is covered
    separately in TestLayerIndicator.test_requires_is_the_union_of_the_parts."""
    assert output.requires == expected
