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
    GeothermERFOutput,
    GeothermLinearOutput,
    LateralFractionOutput,
    MeshConfig,
    QuinticOutput,
    continental_linear,
    ocean_erf_normalized,
    radial_quintic_step,
)


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
# Geotherm functions (pure math; carried over from the old test file with the
# **kwargs/age=None branches removed)
# ---------------------------------------------------------------------------

class TestOceanErfNormalized:
    def test_surface_is_zero(self):
        z_lab = np.array([100e3, 50e3, 150e3])
        depth = np.zeros_like(z_lab)
        age = np.array([50.0, 100.0, 10.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, kappa=1e-6)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_lab_is_one(self):
        z_lab = np.array([100e3, 50e3, 150e3])
        depth = z_lab.copy()
        age = np.array([50.0, 100.0, 10.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, kappa=1e-6)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_monotone_in_depth(self):
        z_lab = 100e3
        depths = np.linspace(0, z_lab, 50)
        z_labs = np.full_like(depths, z_lab)
        ages = np.full_like(depths, 80.0)
        result = ocean_erf_normalized(depths, z_labs, age_myr=ages, kappa=1e-6)
        assert np.all(np.diff(result) >= 0)

    def test_clipped_to_unit_interval(self):
        z_lab = np.array([100e3])
        depth = np.array([200e3])  # deeper than LAB
        age = np.array([80.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, kappa=1e-6)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_young_vs_old(self):
        # Young ocean: steep erf, T at mid-depth > 0.5; old ocean: profile
        # approaches linear, T at mid-depth ~ 0.5.
        depth = np.array([50e3])
        z_lab = np.array([100e3])
        young = ocean_erf_normalized(depth, z_lab, age_myr=np.array([5.0]), kappa=1e-6)
        old = ocean_erf_normalized(depth, z_lab, age_myr=np.array([200.0]), kappa=1e-6)
        assert young[0] > old[0]

    def test_zero_age_returns_finite(self):
        # The internal safety floor (max(age_sec, 1.0)) means zero-age inputs
        # still produce a valid number rather than a divide-by-zero.
        depth = np.array([10e3])
        z_lab = np.array([100e3])
        age = np.array([0.0])
        result = ocean_erf_normalized(depth, z_lab, age_myr=age, kappa=1e-6)
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
        # z_lab=0 means no lithosphere; profile collapses to surface value.
        result = continental_linear(np.array([50e3]), np.array([0.0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# radial_quintic_step — shared radial primitive
# ---------------------------------------------------------------------------

class TestRadialQuinticStep:
    """One-sided quintic step: exactly 1 for r >= base_r, exactly 0 for
    r <= base_r - width, the smoothstep polynomial in between.

    The transition sits entirely BELOW the base: the base itself reads 1
    (the old tanh kernel read 0.5 there) and the 0.5-crossing is at
    base_r - width/2. base_r may be a scalar (fixed base depth) or a
    per-node array (variable base depth) and both broadcast correctly.
    """

    def test_exact_plateaus_and_midpoint(self):
        base_r = 2.0
        width = 0.01
        # Exactly 1 at the base and everywhere above it.
        assert radial_quintic_step(base_r, base_r, width) == 1.0
        assert radial_quintic_step(base_r + 0.05, base_r, width) == 1.0
        # Zero at base - width (up to round-off in forming base - width) and
        # exactly 0 everywhere below the band (no tanh tail).
        np.testing.assert_allclose(
            radial_quintic_step(base_r - width, base_r, width), 0.0, atol=1e-30
        )
        assert radial_quintic_step(base_r - 0.05, base_r, width) == 0.0
        # The 0.5-crossing sits at the middle of the band, base - width/2.
        np.testing.assert_allclose(
            radial_quintic_step(base_r - width / 2, base_r, width),
            0.5, rtol=1e-12,
        )
        # Monotone increasing across the band.
        rs = np.linspace(base_r - width, base_r, 50)
        f = radial_quintic_step(rs, base_r, width)
        assert np.all(np.diff(f) > 0)

    def test_point_symmetry_about_band_midpoint(self):
        # The smoothstep polynomial satisfies S(t) + S(1-t) == 1, so the step
        # is point-symmetric about (base - width/2, 0.5).
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
        # Zero slope at both ends of the band (C^1; the quintic is in fact
        # C^2). The numerical slope just inside each junction is tiny
        # compared to the mid-band slope.
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
        # Narrower transition width => larger gradient magnitude in the band.
        base_r = 2.0
        rs = np.linspace(base_r - 0.05, base_r + 0.05, 200)
        narrow = radial_quintic_step(rs, base_r, 0.005)
        wide = radial_quintic_step(rs, base_r, 0.05)
        assert np.abs(np.gradient(narrow)).max() > np.abs(np.gradient(wide)).max()


# ---------------------------------------------------------------------------
# QuinticOutput — variable base, no fade (the old TanhOutput role)
# ---------------------------------------------------------------------------

class TestQuinticOutputVariableBase:
    """Tests exercise QuinticOutput.compute directly with synthetic inputs.

    Default configuration: per-node base depth from the thickness channel,
    no lateral fade. The indicator is exactly 1 from the surface down to the
    base, exactly 0 below base + width.
    """

    @staticmethod
    def _interp(thickness_km, n_targets=1):
        # Helper: a single interpolated thickness value broadcast to n_targets.
        return {"thickness": np.full(n_targets, thickness_km, dtype=float)}

    def test_validation_rejects_bad_args(self):
        with pytest.raises(ValueError, match="width_km must be positive"):
            QuinticOutput(width_km=0.0)
        with pytest.raises(ValueError, match="width_km must be positive"):
            QuinticOutput(width_km=-1.0)
        with pytest.raises(ValueError, match="base_depth_km must be positive"):
            QuinticOutput(base_depth_km=0.0)
        with pytest.raises(ValueError, match="fade_ref_km must be positive"):
            QuinticOutput(fade_ref_km=-5.0)
        with pytest.raises(ValueError, match="default_thickness_km must be non-negative"):
            QuinticOutput(default_thickness_km=-1.0)
        # Zero is allowed (polygon mask-and-relabel uses default 0).
        QuinticOutput(default_thickness_km=0.0)

    def test_value_at_base_is_one(self):
        # The target sitting exactly at r_outer - thickness/depth_scale reads
        # exactly 1: the transition hangs entirely below the base (the old
        # tanh kernel read 0.5 here).
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = QuinticOutput(width_km=10.0)
        thickness = 100.0
        interp = self._interp(thickness)
        too_far = np.array([False])
        r_target = np.array([mesh.r_outer - thickness / mesh.depth_scale])
        result = out.compute(interp, r_target, too_far, mesh)
        np.testing.assert_array_equal(result, 1.0)

    def test_half_crossing_half_width_below_base(self):
        mesh = MeshConfig()
        width = 10.0
        thickness = 100.0
        out = QuinticOutput(width_km=width)
        r_target = np.array(
            [mesh.r_outer - (thickness + width / 2) / mesh.depth_scale]
        )
        result = out.compute(
            self._interp(thickness), r_target, np.array([False]), mesh
        )
        np.testing.assert_allclose(result, 0.5, rtol=1e-10)

    def test_value_inside_lithosphere_is_one_exact(self):
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0)
        # 50 km depth, with lithosphere 200 km thick: well inside.
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = self._interp(200.0)
        result = out.compute(interp, r_target, np.array([False]), mesh)
        np.testing.assert_array_equal(result, 1.0)

    def test_value_below_transition_is_zero_exact(self):
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0)
        # 400 km depth, with lithosphere 200 km thick: well below base+width.
        r_target = np.array([mesh.r_outer - 400.0 / mesh.depth_scale])
        interp = self._interp(200.0)
        result = out.compute(interp, r_target, np.array([False]), mesh)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_thickness_surface_skin(self):
        # Where thickness is exactly zero the base coincides with the surface
        # and the SURFACE NODE reads 1 (the transition hangs just below it).
        # This is the documented reason zero-outside polygon sources must pair
        # the step with a lateral fade; one width below the surface the column
        # is exactly 0 again.
        mesh = MeshConfig()
        width = 10.0
        out = QuinticOutput(width_km=width)
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

    def test_narrower_transition_steeper_gradient(self):
        mesh = MeshConfig()
        thickness = 100.0
        # Sweep a range of radii crossing the transition band.
        r_target = np.linspace(
            mesh.r_outer - (thickness + 50.0) / mesh.depth_scale,
            mesh.r_outer - (thickness - 50.0) / mesh.depth_scale,
            100,
        )
        interp = {"thickness": np.full_like(r_target, thickness)}
        too_far = np.zeros_like(r_target, dtype=bool)
        narrow = QuinticOutput(width_km=1.0).compute(interp, r_target, too_far, mesh)
        wide = QuinticOutput(width_km=20.0).compute(interp, r_target, too_far, mesh)
        assert np.abs(np.gradient(narrow)).max() > np.abs(np.gradient(wide)).max()

    def test_too_far_uses_default_thickness(self):
        # Polygon-style use: default_thickness_km=0 means too-far targets
        # read as "outside the region" (indicator -> 0 below the surface skin).
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0, default_thickness_km=0.0)
        r_target = np.array([mesh.r_outer - 100.0 / mesh.depth_scale])
        interp = self._interp(200.0)  # value ignored when too_far=True
        too_far = np.array([True])
        result = out.compute(interp, r_target, too_far, mesh)
        np.testing.assert_array_equal(result, 0.0)

        # Lithosphere-style use: default_thickness_km=100 means too-far
        # targets get a sensible fill rather than zero.
        out_lith = QuinticOutput(width_km=10.0, default_thickness_km=100.0)
        # Target at 50 km depth, default 100 km -> well inside.
        r_target_inside = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        result_inside = out_lith.compute(interp, r_target_inside, np.array([True]), mesh)
        np.testing.assert_array_equal(result_inside, 1.0)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0)
        thickness = np.array([999.0, 10.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)


# ---------------------------------------------------------------------------
# GeothermERFOutput and GeothermLinearOutput
# ---------------------------------------------------------------------------

class TestGeothermERFOutput:
    def test_requires_thickness_and_age(self):
        assert GeothermERFOutput().requires == frozenset({"thickness", "age"})

    def test_validation_rejects_nonpositive_kappa(self):
        with pytest.raises(ValueError, match="kappa must be positive"):
            GeothermERFOutput(kappa=0.0)

    def test_validation_rejects_nonpositive_default_thickness(self):
        with pytest.raises(ValueError, match="default_thickness_km must be positive"):
            GeothermERFOutput(default_thickness_km=0.0)

    def test_validation_rejects_nonpositive_too_far_age(self):
        with pytest.raises(ValueError, match="too_far_age_myr must be positive"):
            GeothermERFOutput(too_far_age_myr=-1.0)

    def test_surface_is_zero_lab_is_one(self):
        # Two target points: one at the surface, one at the LAB depth.
        # The output should match the underlying erf geotherm.
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = GeothermERFOutput(kappa=1e-6)
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

    def test_too_far_uses_fallback_age_and_thickness(self):
        # too_far=True targets should pick up the fallback thickness and age
        # rather than the interpolated values. Confirm by comparing against
        # a direct call to ocean_erf_normalized at the same point.
        mesh = MeshConfig()
        fallback_thick = 100.0
        fallback_age = 500.0
        out = GeothermERFOutput(
            kappa=1e-6,
            default_thickness_km=fallback_thick,
            too_far_age_myr=fallback_age,
        )
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = {
            "thickness": np.array([999.0]),  # value ignored when too_far
            "age": np.array([12345.0]),       # value ignored when too_far
        }
        result = out.compute(interp, r_target, np.array([True]), mesh)
        depth_m = (mesh.r_outer - r_target[0]) * mesh.depth_scale * 1e3
        expected = ocean_erf_normalized(
            np.array([depth_m]),
            np.array([fallback_thick * 1e3]),
            age_myr=np.array([fallback_age]),
            kappa=1e-6,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestGeothermLinearOutput:
    def test_requires_only_thickness(self):
        assert GeothermLinearOutput().requires == frozenset({"thickness"})

    def test_too_far_is_mantle(self):
        # Outside the polygon region (too_far=True), the output is 1.0
        # (mantle temperature).
        mesh = MeshConfig()
        out = GeothermLinearOutput()
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = {"thickness": np.array([100.0])}
        result = out.compute(interp, r_target, np.array([True]), mesh)
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_inside_region_linear(self):
        # Inside the polygon, output follows the linear geotherm.
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = GeothermLinearOutput()
        thickness_km = 100.0
        # Mid-depth in the lithosphere ⇒ T_norm ~ 0.5.
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = {"thickness": np.array([thickness_km])}
        result = out.compute(interp, r_target, np.array([False]), mesh)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# LateralFractionOutput — pure lateral membership, no radial dependence
# ---------------------------------------------------------------------------

class TestLateralFractionOutput:
    """Returns the interpolated membership clipped to [0, 1] with too_far->0.

    Its defining property is the absence of any radial dependence: the result
    is identical for any r_target.
    """

    def test_requires_only_thickness(self):
        assert LateralFractionOutput().requires == frozenset({"thickness"})

    def test_no_radial_dependence(self):
        # Same interpolated membership, wildly different r_target -> identical.
        mesh = MeshConfig()
        out = LateralFractionOutput()
        interp = {"thickness": np.array([0.2, 0.8, 1.0])}
        too_far = np.zeros(3, dtype=bool)
        shallow = out.compute(interp, np.full(3, mesh.r_outer), too_far, mesh)
        deep = out.compute(interp, np.full(3, mesh.r_outer - 0.5), too_far, mesh)
        np.testing.assert_array_equal(shallow, deep)

    def test_clipping_passthrough_and_too_far(self):
        # Below 0 clips to 0, above 1 clips to 1, mid-values pass through
        # unchanged, and too_far nodes are forced to 0 regardless of value.
        mesh = MeshConfig()
        out = LateralFractionOutput()
        interp = {"thickness": np.array([-0.3, 0.0, 0.37, 1.0, 5.0, 0.9])}
        too_far = np.array([False, False, False, False, False, True])
        result = out.compute(interp, np.full(6, mesh.r_outer), too_far, mesh)
        np.testing.assert_allclose(
            result, [0.0, 0.0, 0.37, 1.0, 1.0, 0.0], rtol=1e-12
        )

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = LateralFractionOutput()
        thickness = np.array([-0.3, 0.37, 5.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(3, mesh.r_outer), np.array([False, False, True]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)


# ---------------------------------------------------------------------------
# QuinticOutput, fixed base — separable lateral-fade × fixed-depth radial step
# ---------------------------------------------------------------------------

class TestQuinticOutputFixedBase:
    """I(r, x) = f_lat(x) * S(r) with f_lat = clip(thickness/fade_ref, 0, 1)
    and a radial step whose base depth is FIXED (independent of thickness).
    """

    def test_separable_product(self):
        # The whole field equals f_lat(x) * S(r) computed independently.
        mesh = MeshConfig()
        crust, w_r = 50.0, 10.0
        out = QuinticOutput(width_km=w_r, base_depth_km=crust, fade_ref_km=crust)
        thickness = np.array([10.0, 25.0, 50.0, 80.0])
        r_target = mesh.r_outer - np.array([20.0, 40.0, 60.0, 100.0]) / mesh.depth_scale
        too_far = np.zeros(4, dtype=bool)
        result = out.compute({"thickness": thickness.copy()}, r_target, too_far, mesh)

        f_lat = np.clip(thickness / crust, 0.0, 1.0)
        base_r = mesh.r_outer - crust / mesh.depth_scale
        S = radial_quintic_step(r_target, base_r, w_r / mesh.depth_scale)
        np.testing.assert_allclose(result, f_lat * S, rtol=1e-12)

    def test_fixed_base_depth_independent_of_thickness(self):
        # The radial step does NOT move with thickness: at the fixed base_r,
        # S = 1 exactly (one-sided step) for every thickness, so a saturated
        # column reads exactly 1 there regardless of how thick it is.
        mesh = MeshConfig()
        crust = 50.0
        out = QuinticOutput(width_km=10.0, base_depth_km=crust, fade_ref_km=crust)
        base_r = mesh.r_outer - crust / mesh.depth_scale
        for thickness in (50.0, 100.0, 300.0):
            r_target = np.array([base_r])
            result = out.compute(
                {"thickness": np.array([thickness])}, r_target, np.array([False]), mesh
            )
            np.testing.assert_array_equal(result, 1.0)

    def test_ocean_column_zeroed_vs_unfaded_surface_skin(self):
        # too_far (or zero thickness) zeroes the entire column, surface
        # included. Contrast: the UNFADED variable-base step on the same
        # zero-thickness surface point reads exactly 1 — the surface skin
        # that makes the fade mandatory for zero-outside sources.
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0, base_depth_km=50.0, fade_ref_km=50.0)
        r_surface = np.array([mesh.r_outer])
        # Zero thickness -> f_lat = 0 -> whole column 0 even at the surface.
        zero = out.compute({"thickness": np.array([0.0])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(zero, 0.0, atol=1e-15)
        # too_far sets thickness to 0 with the same effect.
        too_far = out.compute(
            {"thickness": np.array([999.0])}, r_surface, np.array([True]), mesh
        )
        np.testing.assert_allclose(too_far, 0.0, atol=1e-15)
        # Unfaded variable-base step at the same ocean surface point reads 1.
        unfaded_surface = QuinticOutput(width_km=10.0, default_thickness_km=0.0).compute(
            {"thickness": np.array([0.0])}, r_surface, np.array([False]), mesh
        )
        np.testing.assert_array_equal(unfaded_surface, 1.0)

    def test_amplitude_scaling_and_saturation(self):
        # The surface sits on the plateau of the fixed-base step (S = 1
        # exactly), so the surface value IS f_lat: linear in thickness up to
        # the fade reference, saturated at 1 beyond it.
        mesh = MeshConfig()
        crust = 50.0
        out = QuinticOutput(width_km=10.0, base_depth_km=crust, fade_ref_km=crust)
        r_surface = np.array([mesh.r_outer])
        # thickness = crust -> f_lat = 1 -> full surface amplitude, exactly 1.
        full = out.compute({"thickness": np.array([crust])}, r_surface, np.array([False]), mesh)
        np.testing.assert_array_equal(full, 1.0)
        # thickness = crust/2 -> f_lat = 0.5 -> exactly half amplitude.
        half = out.compute({"thickness": np.array([crust / 2])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(half, 0.5, rtol=1e-12)
        # thickness = 4*crust -> f_lat saturates at 1, same as full.
        sat = out.compute({"thickness": np.array([4 * crust])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(sat, full, rtol=1e-12)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0, base_depth_km=50.0, fade_ref_km=50.0)
        thickness = np.array([999.0, 10.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)


# ---------------------------------------------------------------------------
# QuinticOutput, variable base + fade — both from one thickness channel
# ---------------------------------------------------------------------------

class TestQuinticOutputVariableFaded:
    """I(r, x) = f_lat(x) * S(r; h(x)), with both the fade and the per-node
    base depth derived from the SAME thickness channel.

    Because the one-sided step reads exactly 1 at the surface for ANY base
    depth, the surface value of this configuration is exactly f_lat — the
    fade alone controls the surface, while the variable base depth controls
    where the column decays at depth.
    """

    def test_surface_is_exactly_the_fade(self):
        mesh = MeshConfig()
        thickness_ref, w_r = 150.0, 10.0
        out = QuinticOutput(width_km=w_r, fade_ref_km=thickness_ref)

        thickness = np.array([0.0, thickness_ref / 2.0, 300.0])
        r_surface = np.full(3, mesh.r_outer)
        too_far = np.zeros(3, dtype=bool)
        surf = out.compute({"thickness": thickness.copy()}, r_surface, too_far, mesh)

        # S(surface) = 1 exactly for every thickness, so surface == f_lat:
        # zero thickness gives exactly 0 (the faded-away surface skin),
        # half-reference gives exactly 0.5, saturated gives exactly 1.
        f_lat = np.clip(thickness / thickness_ref, 0.0, 1.0)
        np.testing.assert_allclose(surf, f_lat, rtol=1e-12)
        np.testing.assert_array_equal(surf[0], 0.0)
        np.testing.assert_array_equal(surf[2], 1.0)

    def test_variable_base_moves_with_thickness(self):
        # Each column reads f_lat * 1 at its OWN thickness-defined base and
        # f_lat * 0.5 half a width below it — the crossing radius differs per
        # node, which a fixed-base configuration could not reproduce.
        mesh = MeshConfig()
        thickness_ref, w_r = 150.0, 10.0
        out = QuinticOutput(width_km=w_r, fade_ref_km=thickness_ref)
        thickness = np.array([75.0, 300.0])
        too_far = np.zeros(2, dtype=bool)
        f_lat = np.clip(thickness / thickness_ref, 0.0, 1.0)

        r_at_base = mesh.r_outer - thickness / mesh.depth_scale
        at_base = out.compute(
            {"thickness": thickness.copy()}, r_at_base, too_far, mesh
        )
        np.testing.assert_allclose(at_base, f_lat, rtol=1e-12)

        r_mid_band = mesh.r_outer - (thickness + w_r / 2) / mesh.depth_scale
        mid_band = out.compute(
            {"thickness": thickness.copy()}, r_mid_band, too_far, mesh
        )
        np.testing.assert_allclose(mid_band, 0.5 * f_lat, rtol=1e-10)
        assert r_at_base[0] != r_at_base[1]  # crossing moves with thickness
