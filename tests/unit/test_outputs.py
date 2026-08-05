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
    DataBase,
    DeblendedBase,
    FixedBase,
    GeothermERFOutput,
    GeothermLinearOutput,
    IndicatorOutput,
    LateralFractionOutput,
    MaskedGeothermLinearOutput,
    MaskedQuinticOutput,
    MembershipAmplitude,
    MeshConfig,
    NoAmplitude,
    QuinticOutput,
    QuinticStep,
    TaperedAmplitude,
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
        with pytest.raises(ValueError, match="default_thickness_km must be non-negative"):
            QuinticOutput(default_thickness_km=-1.0)
        # Zero is allowed (polygon seeds-plus-zero-background uses default 0).
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
        # This is why a zero-outside source is refused a QuinticOutput
        # outright: "absent" and "present but infinitely thin" are the same
        # number in a thickness channel, and the step reads both as present.
        # One width below the surface the column is exactly 0 again.
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

    def test_variable_base_moves_per_node(self):
        # Distinct columns get distinct bases from their own thickness: each
        # reads exactly 1 at its own base and exactly 0.5 half a width below
        # it. A fixed-base configuration could not reproduce both at once.
        mesh = MeshConfig()
        w_r = 10.0
        out = QuinticOutput(width_km=w_r)
        thickness = np.array([75.0, 300.0])
        too_far = np.zeros(2, dtype=bool)

        r_at_base = mesh.r_outer - thickness / mesh.depth_scale
        at_base = out.compute(
            {"thickness": thickness.copy()}, r_at_base, too_far, mesh
        )
        np.testing.assert_allclose(at_base, 1.0, rtol=1e-12)

        r_mid_band = mesh.r_outer - (thickness + w_r / 2) / mesh.depth_scale
        mid_band = out.compute(
            {"thickness": thickness.copy()}, r_mid_band, too_far, mesh
        )
        np.testing.assert_allclose(mid_band, 0.5, rtol=1e-10)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0)
        thickness = np.array([999.0, 10.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)


# ---------------------------------------------------------------------------
# MaskedQuinticOutput
# ---------------------------------------------------------------------------

class TestMaskedQuinticOutput:
    """Tests exercise MaskedQuinticOutput.compute directly with synthetic inputs.

    The inputs mimic what a polygon source publishes after the kNN blend:
    ``membership`` is the local in-region weight fraction m, and
    ``masked_thickness`` is the product m * h, because the source stores the
    region's depth inside and zero outside. The class exists to recover m and h
    separately, so the tests are written against that: lateral extent must be m
    alone, and base depth must be h alone.
    """

    @staticmethod
    def _interp(membership, thickness_km):
        """Source channels for a node with in-region fraction m and depth h."""
        m = np.asarray(membership, dtype=float)
        h = np.asarray(thickness_km, dtype=float)
        return {"membership": m, "masked_thickness": m * h}

    def test_validation_rejects_bad_args(self):
        with pytest.raises(ValueError, match="width_km must be positive"):
            MaskedQuinticOutput(width_km=0.0)
        with pytest.raises(ValueError, match="width_km must be positive"):
            MaskedQuinticOutput(width_km=-1.0)
        with pytest.raises(ValueError, match="base_depth_km must be positive"):
            MaskedQuinticOutput(base_depth_km=0.0)

    @pytest.mark.parametrize("cls", [MaskedQuinticOutput, QuinticOutput])
    def test_no_fade_ref_km_parameter(self, cls):
        """No indicator output takes a reference thickness to normalise
        against, so there is none to mis-set. `fade_ref_km` divided a channel
        that already held depth times membership by a single constant, which
        is only correct where the depth is spatially constant; it also capped
        interior amplitude at ``h / fade_ref_km`` far from any boundary, an
        error no choice of interpolation kernel could reduce. Guards against
        it being reintroduced on either class."""
        with pytest.raises(TypeError):
            cls(fade_ref_km=150.0)

    def test_surface_reads_membership_exactly(self):
        """'1 inside, 0 outside' — the surface value is the in-region fraction
        and nothing else."""
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        m = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        interp = self._interp(m, 200.0)
        r_target = np.full(m.size, mesh.r_outer)
        result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
        np.testing.assert_allclose(result, m)

    def test_base_depth_does_not_shallow_with_membership(self):
        """The regression this class was written for.

        A partially covered node must keep the data's own depth. Reading the
        blended channel as a depth instead would put a node at m=0.25 with
        h=200 km at a base of 50 km, so a target at 190 km depth would read
        zero rather than its membership.
        """
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        m = np.array([1.0, 0.75, 0.5, 0.25])
        interp = self._interp(m, 200.0)
        # 190 km is above the 200 km base for every node, whatever m is.
        r_target = np.full(m.size, mesh.r_outer - 190.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
        np.testing.assert_allclose(result, m)

    def test_lateral_extent_is_independent_of_depth_data(self):
        """Deepening the region must not move it sideways.

        Same membership, depth scaled by 4x: the indicator is unchanged
        everywhere above the shallowest base. This is the property a single
        thickness channel cannot have, because there the lateral amplitude is
        the depth divided by a constant.
        """
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        m = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        r_target = np.full(m.size, mesh.r_outer - 40.0 / mesh.depth_scale)
        too_far = np.zeros(m.size, bool)
        shallow = out.compute(self._interp(m, 100.0), r_target, too_far, mesh)
        deep = out.compute(self._interp(m, 400.0), r_target, too_far, mesh)
        np.testing.assert_allclose(shallow, deep)
        np.testing.assert_allclose(shallow, m)

    def test_below_base_plus_width_is_zero(self):
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        m = np.array([1.0, 0.5])
        interp = self._interp(m, 200.0)
        r_target = np.full(m.size, mesh.r_outer - 260.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(m.size, bool), mesh)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_membership_is_zero_at_the_surface(self):
        """The exterior surface reads 0, not 1. QuinticOutput without a fade
        reads 1 here, because zero thickness puts the base at the surface."""
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        interp = self._interp(np.array([0.0]), 200.0)
        result = out.compute(
            interp, np.array([mesh.r_outer]), np.array([False]), mesh
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_too_far_is_outside_the_region(self):
        """No nearby source point is a statement about membership, not depth,
        so there is no default thickness to fill in."""
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        interp = self._interp(np.array([1.0, 1.0]), 200.0)
        r_target = np.full(2, mesh.r_outer)
        result = out.compute(interp, r_target, np.array([False, True]), mesh)
        np.testing.assert_allclose(result, [1.0, 0.0])

    def test_fixed_base_depth_overrides_the_data(self):
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=10.0, base_depth_km=50.0)
        m = np.array([1.0, 1.0])
        interp = self._interp(m, 300.0)  # data says 300 km; fixed base says 50
        # 70 km is clear of the 50 + 10 junction, so the fixed base has fully
        # decayed. (Probing the junction itself would land on the float residue
        # the step leaves there; see TestRadialQuinticStep.)
        r_target = np.full(2, mesh.r_outer - 70.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(2, bool), mesh)
        np.testing.assert_array_equal(result, 0.0)

    def test_recovers_a_laterally_varying_depth_exactly(self):
        """Every other test here broadcasts one depth. The defect being fixed
        is specifically about depth that VARIES across the region — a single
        reference thickness can be right for one value at a time — so drive
        per-node depths spanning the cratonic range against per-node coverage
        and check each column against its own base.
        """
        mesh = MeshConfig()
        w_r = 20.0
        out = MaskedQuinticOutput(width_km=w_r)
        h = np.array([150.0, 200.0, 275.0, 350.0])
        m = np.array([1.0, 0.6, 0.35, 0.9])
        interp = {"membership": m.copy(), "masked_thickness": m * h}
        too_far = np.zeros(m.size, bool)

        # Each column at its OWN base reads its own membership, undiluted by
        # how deep it is or how deep its neighbours are.
        at_base = out.compute(
            interp, mesh.r_outer - h / mesh.depth_scale, too_far, mesh
        )
        np.testing.assert_allclose(at_base, m, rtol=1e-12)

        # Half a transition width below its own base, exactly half of it.
        mid = out.compute(
            interp, mesh.r_outer - (h + w_r / 2) / mesh.depth_scale, too_far, mesh
        )
        np.testing.assert_allclose(mid, 0.5 * m, rtol=1e-10)

    def test_transition_band_is_bounded_and_monotone(self):
        """The deblend divides by a membership that goes to zero at the edge,
        floored at MEMBERSHIP_FLOOR. The trailing multiply by the same
        membership bounds the result, but that is an argument, not a test —
        so sweep membership down through and past the floor and check the
        output stays finite, inside [0, 1], and non-increasing.
        """
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=20.0)
        m = np.concatenate([
            np.geomspace(1.0, MEMBERSHIP_FLOOR, 60),
            np.geomspace(MEMBERSHIP_FLOOR, 1e-12, 40),
            np.array([0.0]),
        ])
        interp = {"membership": m.copy(), "masked_thickness": m * 200.0}
        too_far = np.zeros(m.size, bool)

        for depth_km in (0.0, 100.0, 199.0, 210.0):
            r_target = np.full(m.size, mesh.r_outer - depth_km / mesh.depth_scale)
            result = out.compute(interp, r_target, too_far, mesh)
            assert np.all(np.isfinite(result)), depth_km
            assert np.all(result >= 0.0) and np.all(result <= 1.0), depth_km
            # m descends, so the field must never climb as coverage falls.
            assert np.all(np.diff(result) <= 1e-12), depth_km

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = MaskedQuinticOutput(width_km=50.0)
        interp = self._interp(np.array([0.5, 1.0]), 200.0)
        before = {k: v.copy() for k, v in interp.items()}
        out.compute(
            interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh
        )
        for k, v in before.items():
            np.testing.assert_array_equal(interp[k], v)


class TestIndicatorOutput:
    """The composable form. QuinticOutput and MaskedQuinticOutput are presets
    over it, so composing the same parts must reproduce them exactly; and
    ``requires`` must be the union of the chosen parts'."""

    @staticmethod
    def _masked_interp(membership, thickness_km):
        m = np.asarray(membership, dtype=float)
        h = np.asarray(thickness_km, dtype=float)
        return {"membership": m, "masked_thickness": m * h}

    def test_requires_is_the_union_of_the_parts(self):
        assert IndicatorOutput(
            QuinticStep(10.0), FixedBase(50.0), NoAmplitude()
        ).requires == frozenset()
        assert IndicatorOutput(
            QuinticStep(10.0), DataBase(), NoAmplitude()
        ).requires == frozenset({"thickness"})
        assert IndicatorOutput(
            QuinticStep(10.0), DeblendedBase(), MembershipAmplitude()
        ).requires == frozenset({"masked_thickness", "membership"})
        # A fixed base with a membership amplitude reads membership only.
        assert IndicatorOutput(
            QuinticStep(10.0), FixedBase(50.0), MembershipAmplitude()
        ).requires == frozenset({"membership"})

    def test_reproduces_quintic_output(self):
        mesh = MeshConfig()
        interp = {"thickness": np.array([80.0, 150.0, 220.0])}
        r_target = np.full(3, mesh.r_outer - 100.0 / mesh.depth_scale)
        too_far = np.array([False, True, False])
        composed = IndicatorOutput(
            QuinticStep(10.0), DataBase(default_thickness_km=100.0), NoAmplitude()
        ).compute(interp, r_target, too_far, mesh)
        preset = QuinticOutput(
            width_km=10.0, default_thickness_km=100.0
        ).compute(interp, r_target, too_far, mesh)
        np.testing.assert_array_equal(composed, preset)

    def test_reproduces_masked_quintic_output(self):
        mesh = MeshConfig()
        m = np.array([1.0, 0.6, 0.25, 0.0])
        interp = self._masked_interp(m, np.array([150.0, 200.0, 300.0, 100.0]))
        r_target = np.full(4, mesh.r_outer - 120.0 / mesh.depth_scale)
        too_far = np.array([False, False, True, False])
        composed = IndicatorOutput(
            QuinticStep(20.0), DeblendedBase(), MembershipAmplitude()
        ).compute(interp, r_target, too_far, mesh)
        preset = MaskedQuinticOutput(width_km=20.0).compute(
            interp, r_target, too_far, mesh
        )
        np.testing.assert_array_equal(composed, preset)

    def test_tapered_amplitude_with_identity_is_membership(self):
        mesh = MeshConfig()
        m = np.array([1.0, 0.5, 0.0])
        interp = self._masked_interp(m, 200.0)
        r_target = np.full(3, mesh.r_outer)
        too_far = np.zeros(3, bool)
        tapered = IndicatorOutput(
            QuinticStep(20.0), DeblendedBase(), TaperedAmplitude(lambda x: x)
        ).compute(interp, r_target, too_far, mesh)
        plain = IndicatorOutput(
            QuinticStep(20.0), DeblendedBase(), MembershipAmplitude()
        ).compute(interp, r_target, too_far, mesh)
        np.testing.assert_array_equal(tapered, plain)

    def test_tapered_amplitude_sharpens_the_margin(self):
        # A steeper g pulls partial-coverage nodes down harder than membership
        # alone, and is clipped to [0, 1].
        mesh = MeshConfig()
        m = np.array([0.25, 0.5, 0.75])
        interp = self._masked_interp(m, 200.0)
        r_target = np.full(3, mesh.r_outer)
        too_far = np.zeros(3, bool)
        squared = IndicatorOutput(
            QuinticStep(20.0), DeblendedBase(), TaperedAmplitude(lambda x: x ** 2)
        ).compute(interp, r_target, too_far, mesh)
        # At the surface the step is 1, so the field is exactly g(m) = m^2.
        np.testing.assert_allclose(squared, m ** 2, rtol=1e-12)


# ---------------------------------------------------------------------------
# GeothermERFOutput and GeothermLinearOutput
# ---------------------------------------------------------------------------

class TestGeothermERFOutput:
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


class TestMaskedGeothermLinearOutput:
    """The geotherm counterpart of MaskedQuinticOutput. A polygon source's depth
    channel is masked_thickness = m * h, so a plain GeothermLinearOutput would
    read the blended product as z_LAB and shallow the LAB across the edge. This
    class deblends z_LAB = masked_thickness / m and sets mantle (T_norm=1)
    outside the region.
    """

    @staticmethod
    def _interp(membership, thickness_km):
        m = np.asarray(membership, dtype=float)
        h = np.asarray(thickness_km, dtype=float)
        return {"membership": m, "masked_thickness": m * h}

    def test_deblended_profile_is_independent_of_membership(self):
        # The defect this fixes: at a fixed depth inside the region, T_norm must
        # depend on the data's own LAB depth h, not on the blended coverage m.
        # Two nodes at the same depth with the same h but different m must read
        # the same T_norm. A non-deblended read (z_LAB = m*h) would give the
        # m=0.5 node T_norm(50/50)=1.0 instead of 0.5.
        mesh = MeshConfig()
        out = MaskedGeothermLinearOutput()
        m = np.array([1.0, 0.5])
        interp = self._interp(m, 100.0)  # h = 100 km both nodes
        r_target = np.full(2, mesh.r_outer - 50.0 / mesh.depth_scale)
        result = out.compute(interp, r_target, np.zeros(2, bool), mesh)
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-10)

    def test_outside_the_region_is_mantle_not_surface(self):
        # The load-bearing check (OD1). Outside the region — too_far, or
        # membership below MEMBERSHIP_FLOOR — the geotherm is mantle (1.0), NOT
        # the surface (0.0) that continental_linear's z_lab<=0 branch returns.
        mesh = MeshConfig()
        out = MaskedGeothermLinearOutput()
        # Node 0: covered. Node 1: membership below the floor. Node 2: too_far.
        m = np.array([1.0, MEMBERSHIP_FLOOR / 2.0, 1.0])
        interp = self._interp(m, 100.0)
        r_target = np.full(3, mesh.r_outer - 50.0 / mesh.depth_scale)
        too_far = np.array([False, False, True])
        result = out.compute(interp, r_target, too_far, mesh)
        # Node 0 is a genuine mid-lithosphere reading; nodes 1 and 2 are mantle.
        np.testing.assert_allclose(result, [0.5, 1.0, 1.0], atol=1e-10)

    def test_stays_bounded_and_finite_through_the_floor(self):
        # Sweeping membership down through and past MEMBERSHIP_FLOOR, the deblend
        # divides by a vanishing m but must stay finite and inside [0, 1].
        mesh = MeshConfig()
        out = MaskedGeothermLinearOutput()
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
        out = MaskedGeothermLinearOutput()
        interp = self._interp(np.array([0.5, 1.0]), 200.0)
        before = {k: v.copy() for k, v in interp.items()}
        out.compute(
            interp, np.full(2, mesh.r_outer - 50.0 / mesh.depth_scale),
            np.array([True, False]), mesh,
        )
        for k, v in before.items():
            np.testing.assert_array_equal(interp[k], v)


# ---------------------------------------------------------------------------
# LateralFractionOutput — pure lateral membership, no radial dependence
# ---------------------------------------------------------------------------

class TestLateralFractionOutput:
    """Returns the interpolated membership clipped to [0, 1] with too_far->0.

    Its defining property is the absence of any radial dependence: the result
    is identical for any r_target.
    """

    def test_no_radial_dependence(self):
        # Same interpolated membership, wildly different r_target -> identical.
        mesh = MeshConfig()
        out = LateralFractionOutput()
        interp = {"membership": np.array([0.2, 0.8, 1.0])}
        too_far = np.zeros(3, dtype=bool)
        shallow = out.compute(interp, np.full(3, mesh.r_outer), too_far, mesh)
        deep = out.compute(interp, np.full(3, mesh.r_outer - 0.5), too_far, mesh)
        np.testing.assert_array_equal(shallow, deep)

    def test_clipping_passthrough_and_too_far(self):
        # Below 0 clips to 0, above 1 clips to 1, mid-values pass through
        # unchanged, and too_far nodes are forced to 0 regardless of value.
        mesh = MeshConfig()
        out = LateralFractionOutput()
        interp = {"membership": np.array([-0.3, 0.0, 0.37, 1.0, 5.0, 0.9])}
        too_far = np.array([False, False, False, False, False, True])
        result = out.compute(interp, np.full(6, mesh.r_outer), too_far, mesh)
        np.testing.assert_allclose(
            result, [0.0, 0.0, 0.37, 1.0, 1.0, 0.0], rtol=1e-12
        )

    def test_matches_the_masked_indicator_at_the_surface(self):
        """The lateral fraction is exactly what MaskedQuinticOutput paints at
        the surface, so the two cannot drift apart."""
        mesh = MeshConfig()
        membership = np.array([0.0, 0.25, 0.6, 1.0])
        interp = {
            "membership": membership.copy(),
            "masked_thickness": membership * 200.0,
        }
        r_surface = np.full(membership.size, mesh.r_outer)
        too_far = np.zeros(membership.size, dtype=bool)

        lateral = LateralFractionOutput().compute(interp, r_surface, too_far, mesh)
        indicator = MaskedQuinticOutput(width_km=50.0).compute(
            interp, r_surface, too_far, mesh
        )
        np.testing.assert_allclose(lateral, indicator, rtol=1e-12)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = LateralFractionOutput()
        membership = np.array([-0.3, 0.37, 5.0])
        interp = {"membership": membership.copy()}
        out.compute(interp, np.full(3, mesh.r_outer), np.array([False, False, True]), mesh)
        np.testing.assert_array_equal(interp["membership"], membership)


# ---------------------------------------------------------------------------
# QuinticOutput, fixed base — the radial step pinned at one depth
# ---------------------------------------------------------------------------

class TestQuinticOutputFixedBase:
    """``base_depth_km`` pins the step at one depth for every column, so the
    thickness channel stops moving the transition and only feeds the
    ``too_far`` bookkeeping.
    """

    def test_fixed_base_depth_independent_of_thickness(self):
        # The radial step does NOT move with thickness: at the fixed base_r,
        # S = 1 exactly (one-sided step) for every thickness, so a column
        # reads exactly 1 there regardless of how thick the data says it is.
        mesh = MeshConfig()
        crust = 50.0
        out = QuinticOutput(width_km=10.0, base_depth_km=crust)
        base_r = mesh.r_outer - crust / mesh.depth_scale
        for thickness in (50.0, 100.0, 300.0):
            r_target = np.array([base_r])
            result = out.compute(
                {"thickness": np.array([thickness])}, r_target, np.array([False]), mesh
            )
            np.testing.assert_array_equal(result, 1.0)

    def test_does_not_mutate_input(self):
        mesh = MeshConfig()
        out = QuinticOutput(width_km=10.0, base_depth_km=50.0)
        thickness = np.array([999.0, 10.0])
        interp = {"thickness": thickness.copy()}
        out.compute(interp, np.full(2, mesh.r_outer), np.array([True, False]), mesh)
        np.testing.assert_array_equal(interp["thickness"], thickness)


# ---------------------------------------------------------------------------
# The public ``requires`` contract of every concrete output, side by side.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output, expected",
    [
        (MaskedQuinticOutput(), frozenset({"masked_thickness", "membership"})),
        (GeothermERFOutput(), frozenset({"thickness", "age"})),
        (GeothermLinearOutput(), frozenset({"thickness"})),
        (MaskedGeothermLinearOutput(), frozenset({"masked_thickness", "membership"})),
        # LateralFractionOutput reads the source's own membership channel rather
        # than a duplicated source carrying a constant in the thickness slot, so
        # it stays consistent with an indicator built from that source.
        (LateralFractionOutput(), frozenset({"membership"})),
    ],
)
def test_output_requires_contract(output, expected):
    """Each concrete output declares the source channels it reads via its
    class-level ``requires`` set; the connector validates this against a
    source's ``provides``. IndicatorOutput's *computed* union is covered
    separately in TestIndicatorOutput.test_requires_is_the_union_of_the_parts."""
    assert output.requires == expected
