"""
Unit tests for soil curve models in the Richards equation module.

This module tests the implementation of various soil curve models including
Haverkamp, van Genuchten, and exponential models for unsaturated flow.
"""

import firedrake as fd
import numpy as np
import pytest
from gadopt.richards import (
    SoilCurve,
    HaverkampCurve,
    VanGenuchtenCurve,
    ExponentialCurve,
)


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
    elif property_name == "relative_permeability":
        expr = model.relative_permeability(h)
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
            SoilCurve({})

    def test_parameter_validation(self):
        """Test parameter validation for all models."""
        # Test Haverkamp with missing parameters
        with pytest.raises(ValueError, match="Missing required parameter"):
            HaverkampCurve({'theta_r': 0.15})

        # Test van Genuchten with invalid n parameter
        with pytest.raises(ValueError, match="Parameter n must be > 1.0"):
            VanGenuchtenCurve({
                'theta_r': 0.15, 'theta_s': 0.45, 'alpha': 0.5,
                'n': 0.5, 'Ks': 1e-5
            })

        # Test exponential with missing parameters
        with pytest.raises(ValueError, match="Missing required parameter"):
            ExponentialCurve({'theta_r': 0.15})


class TestHaverkampCurve:
    """Test Haverkamp soil curve model."""

    @pytest.fixture
    def haverkamp_params(self):
        """Standard Haverkamp parameters for testing."""
        return {
            'theta_r': 0.15,
            'theta_s': 0.45,
            'alpha': 1.0,
            'beta': 2.0,
            'Ks': 1e-5,
            'A': 1.0,
            'gamma': 2.0
        }

    @pytest.fixture
    def haverkamp_model(self, haverkamp_params):
        """Create Haverkamp model instance."""
        return HaverkampCurve(haverkamp_params)

    def test_saturated_conditions(self, haverkamp_model):
        """Test soil curve behavior under saturated conditions (h >= 0)."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)  # Saturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(haverkamp_model, h_sat, "moisture_content")
        K = evaluate_soil_curve(haverkamp_model, h_sat, "relative_permeability")
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
        K = evaluate_soil_curve(haverkamp_model, h_unsat, "relative_permeability")
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
        K = evaluate_soil_curve(haverkamp_model, h, "relative_permeability")

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
    def vg_params(self):
        """Standard van Genuchten parameters for testing."""
        return {
            'theta_r': 0.15,
            'theta_s': 0.45,
            'alpha': 0.5,
            'n': 2.0,
            'Ks': 1e-5
        }

    @pytest.fixture
    def vg_model(self, vg_params):
        """Create van Genuchten model instance."""
        return VanGenuchtenCurve(vg_params)

    def test_saturated_conditions(self, vg_model):
        """Test soil curve behavior under saturated conditions (h >= 0)."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)  # Saturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(vg_model, h_sat, "moisture_content")
        K = evaluate_soil_curve(vg_model, h_sat, "relative_permeability")
        C = evaluate_soil_curve(vg_model, h_sat, "water_retention")

        # Under saturated conditions, should return saturated values
        np.testing.assert_allclose(theta.dat.data, 0.45, rtol=1e-10)  # theta_s
        np.testing.assert_allclose(K.dat.data, 1e-5, rtol=1e-10)      # Ks
        np.testing.assert_allclose(C.dat.data, 0.0, rtol=1e-10)       # C = 0

    def test_unsaturated_conditions(self, vg_model):
        """Test soil curve behavior under unsaturated conditions (h < 0)."""
        mesh = create_test_mesh()
        h_unsat = create_pressure_head_function(mesh, -1.0)  # Unsaturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(vg_model, h_unsat, "moisture_content")
        K = evaluate_soil_curve(vg_model, h_unsat, "relative_permeability")
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
        K = evaluate_soil_curve(vg_model, h, "relative_permeability")

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
    def exp_params(self):
        """Standard exponential parameters for testing."""
        return {
            'theta_r': 0.15,
            'theta_s': 0.45,
            'alpha': 0.5,
            'Ks': 1e-5
        }

    @pytest.fixture
    def exp_model(self, exp_params):
        """Create exponential model instance."""
        return ExponentialCurve(exp_params)

    def test_saturated_conditions(self, exp_model):
        """Test soil curve behavior under saturated conditions (h >= 0)."""
        mesh = create_test_mesh()
        h_sat = create_pressure_head_function(mesh, 0.0)  # Saturated

        # Evaluate soil curve properties
        theta = evaluate_soil_curve(exp_model, h_sat, "moisture_content")
        K = evaluate_soil_curve(exp_model, h_sat, "relative_permeability")
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
        K = evaluate_soil_curve(exp_model, h_unsat, "relative_permeability")
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
        K = evaluate_soil_curve(exp_model, h, "relative_permeability")

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
        theta_r, theta_s, Ks = 0.15, 0.45, 1e-5

        # Create models with equivalent saturated properties
        haverkamp = HaverkampCurve({
            'theta_r': theta_r, 'theta_s': theta_s, 'alpha': 1.0, 'beta': 2.0,
            'Ks': Ks, 'A': 1.0, 'gamma': 2.0
        })

        vg = VanGenuchtenCurve({
            'theta_r': theta_r, 'theta_s': theta_s, 'alpha': 0.5, 'n': 2.0, 'Ks': Ks
        })

        exp = ExponentialCurve({
            'theta_r': theta_r, 'theta_s': theta_s, 'alpha': 0.5, 'Ks': Ks
        })

        # Test Haverkamp and van Genuchten models (should have C=0 at saturation)
        for model in [haverkamp, vg]:
            theta = evaluate_soil_curve(model, h_sat, "moisture_content")
            K = evaluate_soil_curve(model, h_sat, "relative_permeability")
            C = evaluate_soil_curve(model, h_sat, "water_retention")

            np.testing.assert_allclose(theta.dat.data, theta_s, rtol=1e-10)
            np.testing.assert_allclose(K.dat.data, Ks, rtol=1e-10)
            np.testing.assert_allclose(C.dat.data, 0.0, rtol=1e-10)

        # Test exponential model (has different water retention at saturation)
        theta = evaluate_soil_curve(exp, h_sat, "moisture_content")
        K = evaluate_soil_curve(exp, h_sat, "relative_permeability")
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
        theta_r, theta_s, Ks = 0.15, 0.45, 1e-5

        haverkamp = HaverkampCurve({
            'theta_r': theta_r, 'theta_s': theta_s, 'alpha': 1.0, 'beta': 2.0,
            'Ks': Ks, 'A': 1.0, 'gamma': 2.0
        })

        vg = VanGenuchtenCurve({
            'theta_r': theta_r, 'theta_s': theta_s, 'alpha': 0.5, 'n': 2.0, 'Ks': Ks
        })

        exp = ExponentialCurve({
            'theta_r': theta_r, 'theta_s': theta_s, 'alpha': 0.5, 'Ks': Ks
        })

        # Get unsaturated values
        theta_h = evaluate_soil_curve(haverkamp, h_unsat, "moisture_content").dat.data[0]
        theta_vg = evaluate_soil_curve(vg, h_unsat, "moisture_content").dat.data[0]
        theta_exp = evaluate_soil_curve(exp, h_unsat, "moisture_content").dat.data[0]

        K_h = evaluate_soil_curve(haverkamp, h_unsat, "relative_permeability").dat.data[0]
        K_vg = evaluate_soil_curve(vg, h_unsat, "relative_permeability").dat.data[0]
        K_exp = evaluate_soil_curve(exp, h_unsat, "relative_permeability").dat.data[0]

        # Models should give different results (not testing specific values,
        # just that they're different from each other)
        assert not np.allclose([theta_h, theta_vg, theta_exp], theta_h, rtol=1e-6)
        assert not np.allclose([K_h, K_vg, K_exp], K_h, rtol=1e-6)
