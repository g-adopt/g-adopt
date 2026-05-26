"""Pure-math tests for OutputStrategy subclasses and MeshConfig.

No reconstruction data needed; everything in this file constructs its inputs
directly. These tests cover the mathematical correctness of the indicator and
geotherm transformations and the validation contracts of the small dataclasses.
"""

import numpy as np
import pytest

from gadopt.gplates import (
    FadedRadialStepOutput,
    GeothermERFOutput,
    GeothermLinearOutput,
    LateralFractionOutput,
    MeshConfig,
    TanhOutput,
    continental_linear,
    ocean_erf_normalized,
    radial_tanh_step,
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
# TanhOutput
# ---------------------------------------------------------------------------

class TestTanhOutput:
    """Tests exercise TanhOutput.compute directly with synthetic inputs.

    The output produces a smooth indicator: ~1 inside lithosphere (above
    r = r_outer - thickness), ~0 in the mantle, 0.5 at the boundary.
    """

    @staticmethod
    def _interp(thickness_km, n_targets=1):
        # Helper: a single interpolated thickness value broadcast to n_targets.
        return {"thickness": np.full(n_targets, thickness_km, dtype=float)}

    def test_validation_rejects_nonpositive_transition(self):
        with pytest.raises(ValueError, match="transition_width_km must be positive"):
            TanhOutput(transition_width_km=0.0)
        with pytest.raises(ValueError, match="transition_width_km must be positive"):
            TanhOutput(transition_width_km=-1.0)

    def test_validation_rejects_negative_default_thickness(self):
        with pytest.raises(ValueError, match="default_thickness_km must be non-negative"):
            TanhOutput(default_thickness_km=-1.0)
        # Zero is allowed (polygon mask-and-relabel uses default 0).
        TanhOutput(default_thickness_km=0.0)

    def test_value_at_boundary_is_half(self):
        # When the target sits at r_outer - thickness/depth_scale, the
        # argument to tanh is zero and the indicator equals 0.5 exactly.
        mesh = MeshConfig(r_outer=2.208, depth_scale=2890.0)
        out = TanhOutput(transition_width_km=10.0)
        thickness = 100.0
        interp = self._interp(thickness)
        too_far = np.array([False])
        r_target = np.array([mesh.r_outer - thickness / mesh.depth_scale])
        result = out.compute(interp, r_target, too_far, mesh)
        np.testing.assert_allclose(result, 0.5, rtol=1e-10)

    def test_value_inside_lithosphere_is_one(self):
        mesh = MeshConfig()
        out = TanhOutput(transition_width_km=10.0)
        # 50 km depth, with lithosphere 200 km thick: well inside.
        r_target = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        interp = self._interp(200.0)
        result = out.compute(interp, r_target, np.array([False]), mesh)
        assert result[0] > 0.99

    def test_value_below_lithosphere_is_zero(self):
        mesh = MeshConfig()
        out = TanhOutput(transition_width_km=10.0)
        # 400 km depth, with lithosphere 200 km thick: well below.
        r_target = np.array([mesh.r_outer - 400.0 / mesh.depth_scale])
        interp = self._interp(200.0)
        result = out.compute(interp, r_target, np.array([False]), mesh)
        assert result[0] < 0.01

    def test_narrower_transition_steeper_gradient(self):
        mesh = MeshConfig()
        thickness = 100.0
        # Sweep a range of radii crossing the boundary.
        r_target = np.linspace(
            mesh.r_outer - (thickness + 50.0) / mesh.depth_scale,
            mesh.r_outer - (thickness - 50.0) / mesh.depth_scale,
            100,
        )
        interp = {"thickness": np.full_like(r_target, thickness)}
        too_far = np.zeros_like(r_target, dtype=bool)
        narrow = TanhOutput(transition_width_km=1.0).compute(interp, r_target, too_far, mesh)
        wide = TanhOutput(transition_width_km=20.0).compute(interp, r_target, too_far, mesh)
        assert np.abs(np.gradient(narrow)).max() > np.abs(np.gradient(wide)).max()

    def test_too_far_uses_default_thickness(self):
        # Polygon-style use: default_thickness_km=0 means too-far targets
        # read as "outside the region" (indicator -> 0 below r_outer).
        mesh = MeshConfig()
        out = TanhOutput(transition_width_km=10.0, default_thickness_km=0.0)
        r_target = np.array([mesh.r_outer - 100.0 / mesh.depth_scale])
        interp = self._interp(200.0)  # value ignored when too_far=True
        too_far = np.array([True])
        result = out.compute(interp, r_target, too_far, mesh)
        assert result[0] < 0.01

        # Lithosphere-style use: default_thickness_km=100 means too-far
        # targets get a sensible fill rather than zero.
        out_lith = TanhOutput(transition_width_km=10.0, default_thickness_km=100.0)
        # Target at 50 km depth, default 100 km -> well inside.
        r_target_inside = np.array([mesh.r_outer - 50.0 / mesh.depth_scale])
        result_inside = out_lith.compute(interp, r_target_inside, np.array([True]), mesh)
        assert result_inside[0] > 0.99


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
# radial_tanh_step — shared radial primitive
# ---------------------------------------------------------------------------

class TestRadialTanhStep:
    """The 0.5*(1+tanh((r-base_r)/width)) kernel shared by every tanh output.

    The core design point is that base_r may be a scalar (fixed base depth) or
    a per-node array (variable base depth) and both broadcast correctly.
    """

    def test_boundary_direction_and_saturation(self):
        # =0.5 exactly at r==base_r; >0.5 above toward 1; <0.5 below toward 0.
        base_r = 2.0
        width = 0.01
        # At the boundary the argument is zero -> exactly one half.
        assert radial_tanh_step(base_r, base_r, width) == 0.5
        # Five sigma above and below saturates to the golden tanh values.
        np.testing.assert_allclose(
            radial_tanh_step(base_r + 0.05, base_r, width),
            0.9999546021312975, rtol=1e-12,
        )
        np.testing.assert_allclose(
            radial_tanh_step(base_r - 0.05, base_r, width),
            1.0 - 0.9999546021312975, rtol=1e-9,
        )
        # Monotone increasing across the step.
        rs = np.linspace(base_r - 0.05, base_r + 0.05, 50)
        f = radial_tanh_step(rs, base_r, width)
        assert np.all(np.diff(f) > 0)
        assert f[0] < 0.5 < f[-1]

    def test_odd_symmetry_about_base(self):
        # f(base+d) + f(base-d) == 1 for any offset (tanh is odd).
        base_r = 2.0
        width = 0.02
        d = np.array([0.005, 0.01, 0.03, 0.1])
        np.testing.assert_allclose(
            radial_tanh_step(base_r + d, base_r, width)
            + radial_tanh_step(base_r - d, base_r, width),
            1.0, rtol=1e-12,
        )

    def test_scalar_and_array_base_broadcast(self):
        # Scalar base_r and a per-node array base_r both broadcast against an
        # array r_target. A constant array base must reproduce the scalar case.
        r = np.array([2.0, 2.05, 2.1])
        width = 0.02
        scalar = radial_tanh_step(r, 2.05, width)
        array = radial_tanh_step(r, np.full_like(r, 2.05), width)
        np.testing.assert_allclose(scalar, array, rtol=1e-12)
        # A genuinely per-node base shifts each crossing independently: each
        # node sitting exactly on its own base reads 0.5.
        per_node_base = r.copy()
        np.testing.assert_allclose(
            radial_tanh_step(r, per_node_base, width), 0.5, rtol=1e-12
        )

    def test_narrower_width_is_steeper(self):
        # Narrower transition width => larger gradient magnitude at the step.
        base_r = 2.0
        rs = np.linspace(base_r - 0.05, base_r + 0.05, 200)
        narrow = radial_tanh_step(rs, base_r, 0.005)
        wide = radial_tanh_step(rs, base_r, 0.05)
        assert np.abs(np.gradient(narrow)).max() > np.abs(np.gradient(wide)).max()


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
# FadedRadialStepOutput — separable lateral-fade × fixed-depth radial step
# ---------------------------------------------------------------------------

class TestFadedRadialStepOutput:
    """I(r, x) = f_lat(x) * S(r) with f_lat = clip(thickness/crust, 0, 1) and
    a radial step whose base depth is FIXED (independent of thickness).
    """

    def test_validation_rejects_nonpositive_args(self):
        with pytest.raises(ValueError, match="crust_thickness_km must be positive"):
            FadedRadialStepOutput(crust_thickness_km=0.0)
        with pytest.raises(ValueError, match="radial_width_km must be positive"):
            FadedRadialStepOutput(radial_width_km=-1.0)

    def test_requires_only_thickness(self):
        assert FadedRadialStepOutput().requires == frozenset({"thickness"})

    def test_separable_product(self):
        # The whole field equals f_lat(x) * S(r) computed independently.
        mesh = MeshConfig()
        crust, w_r = 50.0, 10.0
        out = FadedRadialStepOutput(crust_thickness_km=crust, radial_width_km=w_r)
        thickness = np.array([10.0, 25.0, 50.0, 80.0])
        r_target = mesh.r_outer - np.array([20.0, 40.0, 60.0, 100.0]) / mesh.depth_scale
        too_far = np.zeros(4, dtype=bool)
        result = out.compute({"thickness": thickness.copy()}, r_target, too_far, mesh)

        f_lat = np.clip(thickness / crust, 0.0, 1.0)
        base_r = mesh.r_outer - crust / mesh.depth_scale
        S = radial_tanh_step(r_target, base_r, w_r / mesh.depth_scale)
        np.testing.assert_allclose(result, f_lat * S, rtol=1e-12)

    def test_fixed_base_depth_independent_of_thickness(self):
        # Unlike TanhOutput, the radial 0.5-crossing of S does NOT move with
        # thickness: at the fixed base_r, S=0.5 for every thickness, so the
        # column value there is simply f_lat * 0.5.
        mesh = MeshConfig()
        crust = 50.0
        out = FadedRadialStepOutput(crust_thickness_km=crust, radial_width_km=10.0)
        base_r = mesh.r_outer - crust / mesh.depth_scale
        # Saturated f_lat (thickness >= crust) at base_r -> exactly 0.5.
        for thickness in (50.0, 100.0, 300.0):
            r_target = np.array([base_r])
            result = out.compute(
                {"thickness": np.array([thickness])}, r_target, np.array([False]), mesh
            )
            np.testing.assert_allclose(result, 0.5, rtol=1e-12)

    def test_ocean_column_zeroed_vs_tanh_floor(self):
        # too_far (or zero thickness) zeroes the entire column, surface
        # included -- the artefact this output fixes. Contrast: TanhOutput on
        # the same surface point floors at 0.5.
        mesh = MeshConfig()
        out = FadedRadialStepOutput(crust_thickness_km=50.0, radial_width_km=10.0)
        r_surface = np.array([mesh.r_outer])
        # Zero thickness -> f_lat = 0 -> whole column 0 even at the surface.
        zero = out.compute({"thickness": np.array([0.0])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(zero, 0.0, atol=1e-15)
        # too_far sets thickness to 0 with the same effect.
        too_far = out.compute(
            {"thickness": np.array([999.0])}, r_surface, np.array([True]), mesh
        )
        np.testing.assert_allclose(too_far, 0.0, atol=1e-15)
        # TanhOutput at the same ocean surface point still reads ~0.5.
        tanh_surface = TanhOutput(transition_width_km=10.0, default_thickness_km=0.0).compute(
            {"thickness": np.array([0.0])}, r_surface, np.array([False]), mesh
        )
        np.testing.assert_allclose(tanh_surface, 0.5, rtol=1e-10)

    def test_amplitude_scaling_and_saturation(self):
        # f_lat scales the amplitude of the radial profile linearly: a half
        # membership halves the surface value of a saturated column; thickness
        # above crust saturates f_lat at 1.
        mesh = MeshConfig()
        crust = 50.0
        out = FadedRadialStepOutput(crust_thickness_km=crust, radial_width_km=10.0)
        r_surface = np.array([mesh.r_outer])
        S_surf = 0.9999546021312975  # golden S(r_outer) for fixed base/width
        # thickness = crust -> f_lat = 1 -> full surface amplitude.
        full = out.compute({"thickness": np.array([crust])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(full, S_surf, rtol=1e-9)
        # thickness = crust/2 -> f_lat = 0.5 -> half amplitude.
        half = out.compute({"thickness": np.array([crust / 2])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(half, 0.5 * S_surf, rtol=1e-9)
        # thickness = 4*crust -> f_lat saturates at 1, same as full.
        sat = out.compute({"thickness": np.array([4 * crust])}, r_surface, np.array([False]), mesh)
        np.testing.assert_allclose(sat, full, rtol=1e-12)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = FadedRadialStepOutput()
        thickness = np.array([999.0, 10.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)
