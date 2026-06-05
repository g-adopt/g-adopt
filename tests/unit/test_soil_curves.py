"""
Unit tests for soil curve models in the Richards equation module.

This module tests the implementation of various soil curve models including
Haverkamp, van Genuchten, and exponential models for unsaturated flow.
"""

import firedrake as fd
import numpy as np
import pytest
import ufl
from gadopt.soil_curves import (
    SoilCurve,
    HaverkampCurve,
    VanGenuchtenCurve,
    ExponentialCurve,
)

PROPERTY_NAMES = (
    "moisture_content",
    "hydraulic_conductivity",
    "water_retention",
)


def _make_curve(cls, reg_eps, n=1.5):
    """Build a curve of the given class with a NON-INTEGER exponent.

    A non-integer n/beta/gamma is what exposes the singular |.|^(p) derivative
    at h = 0 in the unsaturated branch. The exponential model has no such
    exponent, so it is excluded from the regularisation tests.
    """
    if cls is VanGenuchtenCurve:
        return VanGenuchtenCurve(
            theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
            alpha=0.5, n=n, reg_eps=reg_eps,
        )
    if cls is HaverkampCurve:
        return HaverkampCurve(
            theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
            alpha=1.0, beta=n, A=1.0, gamma=n, reg_eps=reg_eps,
        )
    raise ValueError(cls)


def _jacobian_diag(model, property_name, hval):
    """Assemble d(property)/dh at a uniform head hval and return the entries.

    This mirrors exactly the branch-wise UFL differentiation that goes into the
    Newton Jacobian and the pyadjoint adjoint, so finiteness here is finiteness
    of both.
    """
    mesh = create_test_mesh()
    V = fd.FunctionSpace(mesh, "CG", 1)
    h = fd.Function(V)
    h.interpolate(fd.Constant(hval))

    expr = getattr(model, property_name)(h)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    J = fd.derivative(expr * v * fd.dx, h, u)
    A = fd.assemble(J)
    return np.array(A.M.values).ravel()


def create_test_mesh():
    """Create a simple test mesh for soil curve testing."""
    return fd.UnitIntervalMesh(1)


def create_pressure_head_function(mesh, value):
    """Create a pressure head function with specified value."""
    V = fd.FunctionSpace(mesh, "CG", 1)
    h = fd.Function(V)
    h.interpolate(fd.Constant(value))
    return h


def evaluate_soil_curve(model, h, property_name):
    """Evaluate a soil curve property and return interpolated values."""
    if property_name == "moisture_content":
        expr = model.moisture_content(h)
    elif property_name == "hydraulic_conductivity":
        expr = model.hydraulic_conductivity(h)
    elif property_name == "water_retention":
        expr = model.water_retention(h)
    else:
        raise ValueError(f"Unknown property: {property_name}")

    result = fd.Function(h.function_space())
    result.interpolate(expr)
    return result


class TestSoilCurveBase:
    """Test base functionality of soil curve models."""

    def test_abstract_base_class(self):
        """Test that SoilCurve cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SoilCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0)

    def test_parameter_validation(self):
        """Test parameter validation for all models."""
        # Test Haverkamp with missing parameters (missing A, gamma)
        with pytest.raises(ValueError, match="Missing required parameter"):
            HaverkampCurve(
                theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                alpha=1.0, beta=2.0
            )

        # Test van Genuchten with invalid n parameter
        with pytest.raises(ValueError, match="Parameter n must be > 1.0"):
            VanGenuchtenCurve(
                theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0,
                alpha=0.5, n=0.5
            )

        # Test exponential with missing parameters (missing alpha)
        with pytest.raises(ValueError, match="Missing required parameter"):
            ExponentialCurve(theta_r=0.15, theta_s=0.45, Ks=1e-5, Ss=0.0)


class TestHaverkampCurve:
    """Test Haverkamp soil curve model."""

    @pytest.fixture
    def haverkamp_model(self):
        """Create Haverkamp model instance."""
        return HaverkampCurve(
            theta_r=0.15,
            theta_s=0.45,
            Ks=1e-5,
            Ss=0.0,
            alpha=1.0,
            beta=2.0,
            A=1.0,
            gamma=2.0,
        )

    def test_saturated_conditions(self, haverkamp_model):
        """Test soil curve behavior under saturated conditions (h >= 0)."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)  # Saturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(haverkamp_model, h_sat, "moisture_content")
        K = evaluate_soil_curve(haverkamp_model, h_sat, "hydraulic_conductivity")
        C = evaluate_soil_curve(haverkamp_model, h_sat, "water_retention")

        # Under saturated conditions, should return saturated values
        np.testing.assert_allclose(theta.dat.data, 0.45, rtol=1e-10)  # theta_s
        np.testing.assert_allclose(K.dat.data, 1e-5, rtol=1e-10)      # Ks
        np.testing.assert_allclose(C.dat.data, 0.0, rtol=1e-10)       # C = 0

    def test_unsaturated_conditions(self, haverkamp_model):
        """Test soil curve behavior under unsaturated conditions (h < 0)."""
        mesh = create_test_mesh()
        h_unsat = create_pressure_head_function(mesh, -1.0)  # Unsaturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(haverkamp_model, h_unsat, "moisture_content")
        K = evaluate_soil_curve(haverkamp_model, h_unsat, "hydraulic_conductivity")
        C = evaluate_soil_curve(haverkamp_model, h_unsat, "water_retention")

        # Under unsaturated conditions, should be less than saturated values
        assert theta.dat.data[0] < 0.45  # theta < theta_s
        assert K.dat.data[0] < 1e-5      # K < Ks
        assert C.dat.data[0] > 0.0       # C > 0 (positive capacity)

    def test_parameter_consistency(self, haverkamp_model):
        """Test that parameters are correctly used in calculations."""
        mesh = create_test_mesh()
        h = create_pressure_head_function(mesh, -1.0)

        theta = evaluate_soil_curve(haverkamp_model, h, "moisture_content")
        K = evaluate_soil_curve(haverkamp_model, h, "hydraulic_conductivity")

        # Test that residual water content is respected
        assert theta.dat.data[0] >= 0.15  # theta >= theta_r

        # Test that saturated conductivity is respected
        assert K.dat.data[0] <= 1e-5      # K <= Ks

    def test_mathematical_consistency(self, haverkamp_model):
        """Test mathematical consistency of the model."""
        mesh = create_test_mesh()
        h = create_pressure_head_function(mesh, -1.0)

        theta = evaluate_soil_curve(haverkamp_model, h, "moisture_content")
        C = evaluate_soil_curve(haverkamp_model, h, "water_retention")

        # Test that moisture content is within physical bounds
        assert 0.0 <= theta.dat.data[0] <= 1.0

        # Test that water retention capacity is positive for unsaturated conditions
        assert C.dat.data[0] >= 0.0


class TestVanGenuchtenCurve:
    """Test van Genuchten soil curve model."""

    @pytest.fixture
    def vg_model(self):
        """Create van Genuchten model instance."""
        return VanGenuchtenCurve(
            theta_r=0.15,
            theta_s=0.45,
            Ks=1e-5,
            Ss=0.0,
            alpha=0.5,
            n=2.0,
        )

    def test_saturated_conditions(self, vg_model):
        """Test soil curve behavior under saturated conditions (h >= 0)."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)  # Saturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(vg_model, h_sat, "moisture_content")
        K = evaluate_soil_curve(vg_model, h_sat, "hydraulic_conductivity")
        C = evaluate_soil_curve(vg_model, h_sat, "water_retention")

        # Under saturated conditions, should return saturated values. h = 0 is
        # exactly the point where the reg_eps smoothing acts, so K carries an
        # O(reg_eps) offset there (negligible, and the only place it shows up);
        # theta and C are unaffected to O(reg_eps^2).
        np.testing.assert_allclose(theta.dat.data, 0.45, rtol=1e-8)   # theta_s
        np.testing.assert_allclose(K.dat.data, 1e-5, rtol=1e-4)       # Ks
        np.testing.assert_allclose(C.dat.data, 0.0, atol=1e-12)       # C = 0

    def test_unsaturated_conditions(self, vg_model):
        """Test soil curve behavior under unsaturated conditions (h < 0)."""
        mesh = create_test_mesh()
        h_unsat = create_pressure_head_function(mesh, -1.0)  # Unsaturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(vg_model, h_unsat, "moisture_content")
        K = evaluate_soil_curve(vg_model, h_unsat, "hydraulic_conductivity")
        C = evaluate_soil_curve(vg_model, h_unsat, "water_retention")

        # Under unsaturated conditions, should be less than saturated values
        assert theta.dat.data[0] < 0.45  # theta < theta_s
        assert K.dat.data[0] < 1e-5      # K < Ks
        assert C.dat.data[0] > 0.0       # C > 0 (positive capacity)

    def test_parameter_consistency(self, vg_model):
        """Test that parameters are correctly used in calculations."""
        mesh = create_test_mesh()
        h = create_pressure_head_function(mesh, -1.0)

        theta = evaluate_soil_curve(vg_model, h, "moisture_content")
        K = evaluate_soil_curve(vg_model, h, "hydraulic_conductivity")

        # Test that residual water content is respected
        assert theta.dat.data[0] >= 0.15  # theta >= theta_r

        # Test that saturated conductivity is respected
        assert K.dat.data[0] <= 1e-5      # K <= Ks

    def test_mathematical_consistency(self, vg_model):
        """Test mathematical consistency of the model."""
        mesh = create_test_mesh()
        h = create_pressure_head_function(mesh, -1.0)

        theta = evaluate_soil_curve(vg_model, h, "moisture_content")
        C = evaluate_soil_curve(vg_model, h, "water_retention")

        # Test that moisture content is within physical bounds
        assert 0.0 <= theta.dat.data[0] <= 1.0

        # Test that water retention capacity is positive for unsaturated conditions
        assert C.dat.data[0] >= 0.0


class TestExponentialCurve:
    """Test exponential soil curve model."""

    @pytest.fixture
    def exp_model(self):
        """Create exponential model instance."""
        return ExponentialCurve(
            theta_r=0.15,
            theta_s=0.45,
            Ks=1e-5,
            Ss=0.0,
            alpha=0.5,
        )

    def test_saturated_conditions(self, exp_model):
        """Test soil curve behavior under saturated conditions (h >= 0)."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)  # Saturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(exp_model, h_sat, "moisture_content")
        K = evaluate_soil_curve(exp_model, h_sat, "hydraulic_conductivity")
        C = evaluate_soil_curve(exp_model, h_sat, "water_retention")

        # Under saturated conditions, should return saturated values
        np.testing.assert_allclose(theta.dat.data, 0.45, rtol=1e-10)  # theta_s
        np.testing.assert_allclose(K.dat.data, 1e-5, rtol=1e-10)      # Ks
        # For exponential model: C(0) = (theta_s - theta_r) * alpha
        expected_C = (0.45 - 0.15) * 0.5  # (theta_s - theta_r) * alpha
        np.testing.assert_allclose(C.dat.data, expected_C, rtol=1e-10)

    def test_unsaturated_conditions(self, exp_model):
        """Test soil curve behavior under unsaturated conditions (h < 0)."""
        mesh = create_test_mesh()
        h_unsat = create_pressure_head_function(mesh, -1.0)  # Unsaturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(exp_model, h_unsat, "moisture_content")
        K = evaluate_soil_curve(exp_model, h_unsat, "hydraulic_conductivity")
        C = evaluate_soil_curve(exp_model, h_unsat, "water_retention")

        # Under unsaturated conditions, should be less than saturated values
        assert theta.dat.data[0] < 0.45  # theta < theta_s
        assert K.dat.data[0] < 1e-5      # K < Ks
        assert C.dat.data[0] > 0.0       # C > 0 (positive capacity)

    def test_parameter_consistency(self, exp_model):
        """Test that parameters are correctly used in calculations."""
        mesh = create_test_mesh()
        h = create_pressure_head_function(mesh, -1.0)

        theta = evaluate_soil_curve(exp_model, h, "moisture_content")
        K = evaluate_soil_curve(exp_model, h, "hydraulic_conductivity")

        # Test that residual water content is respected
        assert theta.dat.data[0] >= 0.15  # theta >= theta_r

        # Test that saturated conductivity is respected
        assert K.dat.data[0] <= 1e-5      # K <= Ks

    def test_mathematical_consistency(self, exp_model):
        """Test mathematical consistency of the model."""
        mesh = create_test_mesh()
        h = create_pressure_head_function(mesh, -1.0)

        theta = evaluate_soil_curve(exp_model, h, "moisture_content")
        C = evaluate_soil_curve(exp_model, h, "water_retention")

        # Test that moisture content is within physical bounds
        assert 0.0 <= theta.dat.data[0] <= 1.0

        # Test that water retention capacity is positive for unsaturated conditions
        assert C.dat.data[0] >= 0.0


class TestSoilCurveComparison:
    """Test comparison between different soil curve models."""

    def test_model_consistency(self):
        """Test that all models behave consistently at saturation."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)

        # Common parameters
        theta_r, theta_s, Ks, Ss = 0.15, 0.45, 1e-5, 0.0

        # Create models with equivalent saturated properties
        haverkamp = HaverkampCurve(
            theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss,
            alpha=1.0, beta=2.0, A=1.0, gamma=2.0,
        )

        vg = VanGenuchtenCurve(
            theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss,
            alpha=0.5, n=2.0,
        )

        exp = ExponentialCurve(
            theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss,
            alpha=0.5,
        )

        # Test Haverkamp and van Genuchten models (should have C=0 at saturation).
        # h = 0 is exactly where the reg_eps smoothing acts, so K carries an
        # O(reg_eps) offset there; theta and C are unaffected to O(reg_eps^2).
        for model in [haverkamp, vg]:
            theta = evaluate_soil_curve(model, h_sat, "moisture_content")
            K = evaluate_soil_curve(model, h_sat, "hydraulic_conductivity")
            C = evaluate_soil_curve(model, h_sat, "water_retention")

            np.testing.assert_allclose(theta.dat.data, theta_s, rtol=1e-8)
            np.testing.assert_allclose(K.dat.data, Ks, rtol=1e-4)
            np.testing.assert_allclose(C.dat.data, 0.0, atol=1e-12)

        # Test exponential model (has different water retention at saturation)
        theta = evaluate_soil_curve(exp, h_sat, "moisture_content")
        K = evaluate_soil_curve(exp, h_sat, "hydraulic_conductivity")
        C = evaluate_soil_curve(exp, h_sat, "water_retention")

        np.testing.assert_allclose(theta.dat.data, theta_s, rtol=1e-10)
        np.testing.assert_allclose(K.dat.data, Ks, rtol=1e-10)
        # For exponential model: C(0) = (theta_s - theta_r) * alpha
        expected_C_exp = (theta_s - theta_r) * 0.5  # alpha = 0.5
        np.testing.assert_allclose(C.dat.data, expected_C_exp, rtol=1e-10)

    def test_unsaturated_behavior_differences(self):
        """Test that models show different behavior under unsaturated conditions."""
        mesh = create_test_mesh()
        h_unsat = create_pressure_head_function(mesh, -2.0)  # Strongly unsaturated

        # Common parameters
        theta_r, theta_s, Ks, Ss = 0.15, 0.45, 1e-5, 0.0

        haverkamp = HaverkampCurve(
            theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss,
            alpha=1.0, beta=2.0, A=1.0, gamma=2.0,
        )

        vg = VanGenuchtenCurve(
            theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss,
            alpha=0.5, n=2.0,
        )

        exp = ExponentialCurve(
            theta_r=theta_r, theta_s=theta_s, Ks=Ks, Ss=Ss,
            alpha=0.5,
        )

        # Get unsaturated values
        theta_h = evaluate_soil_curve(haverkamp, h_unsat, "moisture_content").dat.data[0]
        theta_vg = evaluate_soil_curve(vg, h_unsat, "moisture_content").dat.data[0]
        theta_exp = evaluate_soil_curve(exp, h_unsat, "moisture_content").dat.data[0]

        K_h = evaluate_soil_curve(haverkamp, h_unsat, "hydraulic_conductivity").dat.data[0]
        K_vg = evaluate_soil_curve(vg, h_unsat, "hydraulic_conductivity").dat.data[0]
        K_exp = evaluate_soil_curve(exp, h_unsat, "hydraulic_conductivity").dat.data[0]

        # Models should give different results (not testing specific values,
        # just that they're different from each other)
        assert not np.allclose([theta_h, theta_vg, theta_exp], theta_h, rtol=1e-6)
        assert not np.allclose([K_h, K_vg, K_exp], K_h, rtol=1e-6)


class TestRegularisationAtWaterTable:
    """Regularisation of the singular |.|^p derivative at exactly h = 0.

    For van Genuchten and Haverkamp with a non-integer exponent, the
    unsaturated branch contains a magnitude raised to a non-integer/negative
    power. Differentiated branch-wise by UFL this produces a term that is Inf
    (and NaN after sign(0)*Inf or 0*Inf) at exactly h = 0, poisoning the Newton
    Jacobian and the pyadjoint gradient. The reg_eps smoothing must keep all of
    moisture_content, hydraulic_conductivity and water_retention AND their
    derivatives finite at h = 0.
    """

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_values_and_derivatives_finite_at_zero(self, cls, property_name):
        """With the default reg_eps, value and derivative are finite at h = 0."""
        model = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS)

        # Value at h = 0.
        h0 = create_pressure_head_function(create_test_mesh(), 0.0)
        value = evaluate_soil_curve(model, h0, property_name)
        assert np.all(np.isfinite(value.dat.data)), (
            f"{cls.__name__}.{property_name} value not finite at h=0"
        )

        # Derivative (Jacobian/adjoint entry) at h = 0.
        deriv = _jacobian_diag(model, property_name, 0.0)
        assert np.all(np.isfinite(deriv)), (
            f"d({cls.__name__}.{property_name})/dh not finite at h=0"
        )

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    def test_unregularised_form_is_singular_at_zero(self, cls):
        """Sanity check: with reg_eps=0 the old form really does go NaN/Inf.

        At least one of the three property derivatives must be non-finite at
        h = 0 when the regularisation is disabled. This is the failure mode the
        fix removes; if this assertion ever stops holding the test above is no
        longer guarding anything.
        """
        model = _make_curve(cls, reg_eps=0.0)
        any_singular = False
        for property_name in PROPERTY_NAMES:
            deriv = _jacobian_diag(model, property_name, 0.0)
            if not np.all(np.isfinite(deriv)):
                any_singular = True
        assert any_singular, (
            f"{cls.__name__} unexpectedly finite at h=0 with reg_eps=0; "
            "the singular-derivative regression is no longer reproduced"
        )

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_forward_unchanged_away_from_zero(self, cls, property_name):
        """Forward value matches the exact (reg_eps=0) form away from h = 0.

        The regularisation is O(reg_eps^2) and must not perturb the constitutive
        relations in the bulk unsaturated region to several digits.
        """
        model_reg = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS)
        model_exact = _make_curve(cls, reg_eps=0.0)

        h = create_pressure_head_function(create_test_mesh(), -1.0)
        v_reg = evaluate_soil_curve(model_reg, h, property_name).dat.data
        v_exact = evaluate_soil_curve(model_exact, h, property_name).dat.data
        np.testing.assert_allclose(v_reg, v_exact, rtol=1e-7, atol=1e-30)


class TestRegularisationAdjointAccuracy:
    """Finite-difference vs UFL-derivative agreement on an unsaturated state.

    A lightweight stand-in for a full pyadjoint Taylor test: it confirms the
    regularised constitutive output is differentiable and that the UFL
    derivative (the object pyadjoint propagates) matches a central finite
    difference to second order at a strictly unsaturated point (h < 0, away
    from the water table).
    """

    @pytest.mark.parametrize("cls", [VanGenuchtenCurve, HaverkampCurve])
    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_taylor_second_order_unsaturated(self, cls, property_name):
        model = _make_curve(cls, reg_eps=SoilCurve.DEFAULT_REG_EPS)

        mesh = create_test_mesh()
        V = fd.FunctionSpace(mesh, "CG", 1)
        h0 = -1.3  # strictly unsaturated, well away from h = 0

        def value(hval):
            h = fd.Function(V)
            h.interpolate(fd.Constant(hval))
            out = fd.Function(V)
            out.interpolate(getattr(model, property_name)(h))
            return out.dat.data[0]

        # Exact (analytic) directional derivative via UFL at h0.
        h = fd.Function(V)
        h.interpolate(fd.Constant(h0))
        hv = fd.variable(h)
        dexpr = ufl.diff(getattr(model, property_name)(hv), hv)
        dval = fd.Function(V)
        dval.interpolate(ufl.replace(dexpr, {hv: h}))
        analytic = dval.dat.data[0]

        f0 = value(h0)

        # Residual of the first-order Taylor expansion should converge at
        # rate ~2 as the step is halved.
        steps = [1e-2, 5e-3, 2.5e-3]
        residuals = []
        for ds in steps:
            taylor = f0 + analytic * ds
            residuals.append(abs(value(h0 + ds) - taylor))

        rates = [
            np.log(residuals[i] / residuals[i + 1]) / np.log(steps[i] / steps[i + 1])
            for i in range(len(steps) - 1)
        ]
        assert min(rates) > 1.8, (
            f"{cls.__name__}.{property_name} Taylor rate {rates} not ~2"
        )
