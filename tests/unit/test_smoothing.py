"""
Test case for smoothing using DiffusiveSmoothingSolver
"""

import pytest

from gadopt import *
from gadopt.utility import upward_normal


@pytest.fixture
def cartesian_field():
    """Create a simple 2D Cartesian test field with known structure."""
    # Create a simple 2D mesh
    mesh = UnitSquareMesh(8, 8)
    mesh.cartesian = True

    V = FunctionSpace(mesh, "CG", 2)

    # Create a test temperature field with some structure
    x = SpatialCoordinate(mesh)
    T = Function(V, name="Temperature")
    # Create a field with some oscillations that can be smoothed
    T.interpolate(sin(4*pi*x[0]) * cos(4*pi*x[1]) + 0.5*(x[0] + x[1]))

    # Boundary conditions
    temp_bcs = {4: {"g": 0.0}, 3: {"g": 1.0}}  # Bottom and top boundaries

    return T, temp_bcs


@pytest.fixture
def cylindrical_field():
    """Create a cylindrical test field for anisotropic smoothing tests."""
    # Create a simple cylindrical mesh
    rmin, rmax = 1.0, 2.0
    mesh = CircleManifoldMesh(8, radius=rmax, degree=2)
    mesh = ExtrudedMesh(mesh, layers=4, layer_height=(rmax-rmin)/4, extrusion_type="radial")
    mesh.cartesian = False

    V = FunctionSpace(mesh, "CG", 1)

    # Create a test temperature field
    x = SpatialCoordinate(mesh)
    r = sqrt(x[0]**2 + x[1]**2)
    theta = atan2(x[1], x[0])
    T = Function(V, name="Temperature")
    # Create a field with radial and angular structure
    T.interpolate(0.5 + 0.3*sin(3*theta) + 0.2*(r - 1.5))

    # Boundary conditions for cylindrical geometry
    temp_bcs = {"bottom": {"g": 1.0}, "top": {"g": 0.0}}

    # Create a layer-averaged version for comparison
    T_avg = Function(V, name="Temperature (layer average)")
    averager = LayerAveraging(mesh, quad_degree=6)
    averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

    return T, T_avg, temp_bcs


def test_isotropic_smoothing(cartesian_field):
    """Test isotropic smoothing on a Cartesian field."""
    T, temp_bcs = cartesian_field

    # Create isotropic smoother with scalar diffusivity
    kappa = Constant(1.0)
    wavelength = 0.1

    # Create solution function for smoothed result
    result_field = Function(T.function_space(), name="smoothed_result")
    smoother = DiffusiveSmoothingSolver(
        result_field,
        wavelength=wavelength,
        K=kappa,
        bcs=temp_bcs,
    )

    # Apply smoothing to the temperature field
    smoother.action(T)

    # Check that smoothing reduces the field variation
    original_variation = assemble((T - Constant(0.5)) ** 2 * dx)
    smoothed_variation = assemble((result_field - Constant(0.5)) ** 2 * dx)

    # Smoothing should reduce variation
    assert smoothed_variation < original_variation, "Smoothing should reduce field variation"

    # But should not make the field completely uniform
    assert smoothed_variation > 1e-6, "Smoothing should not make field completely uniform"


def test_anisotropic_smoothing(cylindrical_field):
    """Test anisotropic smoothing on a cylindrical field."""
    T, T_avg, temp_bcs = cylindrical_field

    # Unit radial and tangential vectors
    mesh = T.function_space().mesh()
    e_r = upward_normal(mesh)
    e_t = as_vector((-e_r[1], e_r[0]))
    # Define the radial and tangential diffusivity values
    kappa_r = Constant(0.0)  # Radial diffusivity
    kappa_t = Constant(1.0)  # Tangential diffusivity
    # Construct the anisotropic diffusivity tensor
    kappa = kappa_r * outer(e_r, e_r) + kappa_t * outer(e_t, e_t)

    wavelength = 0.5  # Use smaller wavelength for faster convergence in tests

    # Create anisotropic smoother with tensor diffusivity
    result_field = Function(T.function_space(), name="anisotropic_result")
    smoother = DiffusiveSmoothingSolver(
        result_field,
        wavelength=wavelength,
        K=kappa,
        bcs=temp_bcs,
    )

    # Apply smoothing to the temperature field
    smoother.action(T)

    # Check that anisotropic smoothing works without errors and produces different results
    # than isotropic smoothing

    # Compare with isotropic smoothing
    kappa_iso = Constant(1.0)
    result_iso = Function(T.function_space(), name="isotropic_result")
    smoother_iso = DiffusiveSmoothingSolver(
        result_iso,
        wavelength=wavelength,
        K=kappa_iso,
        bcs=temp_bcs,
    )
    smoother_iso.action(T)

    # The anisotropic and isotropic results should be different
    diff_aniso_iso = assemble((result_field - result_iso) ** 2 * dx)
    assert diff_aniso_iso > 1e-8, "Anisotropic and isotropic smoothing should produce different results"

    # Both should be different from the original field
    diff_aniso_original = assemble((result_field - T) ** 2 * dx)
    diff_iso_original = assemble((result_iso - T) ** 2 * dx)

    assert diff_aniso_original > 1e-8, "Anisotropic smoothing should change the field"
    assert diff_iso_original > 1e-8, "Isotropic smoothing should change the field"


def test_custom_integration_quad_degree(cartesian_field):
    """Test that custom integration quadrature degree parameter works correctly."""
    T, temp_bcs = cartesian_field

    # Create tensor diffusivity to trigger quadrature degree usage
    # For Cartesian mesh, create a simple anisotropic tensor
    kappa = as_matrix([[2.0, 0.5], [0.5, 1.0]])

    wavelength = 0.2

    # Test with default integration quadrature degree
    result_default = Function(T.function_space(), name="default_result")
    smoother_default = DiffusiveSmoothingSolver(
        result_default,
        wavelength=wavelength,
        K=kappa,
        bcs=temp_bcs,
    )

    # Test with custom integration quadrature degree
    result_custom = Function(T.function_space(), name="custom_result")
    smoother_custom = DiffusiveSmoothingSolver(
        result_custom,
        wavelength=wavelength,
        K=kappa,
        bcs=temp_bcs,
        integration_quad_degree=8,
    )

    # Both should work without errors
    smoother_default.action(T)
    smoother_custom.action(T)

    # Results should be similar (both are solving the same equation)
    # but may differ slightly due to different quadrature accuracy
    diff = assemble((result_default - result_custom) ** 2 * dx)

    # The difference should be small (quadrature accuracy difference)
    assert diff < 1e-8, f"Results differ too much: {diff}"

    # Both should be different from the original field
    diff_default = assemble((result_default - T) ** 2 * dx)
    diff_custom = assemble((result_custom - T) ** 2 * dx)

    assert diff_default > 1e-6, "Default smoother should change the field"
    assert diff_custom > 1e-6, "Custom smoother should change the field"
