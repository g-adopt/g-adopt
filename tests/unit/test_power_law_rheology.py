"""Unit tests for composite power-law rheology in InternalVariableApproximation.

Three tests cover the key mathematical properties introduced by the power-law
changes:

1. second_stress_invariant returns sqrt(J2) for a known tensor.
2. power_law_factor is identically 1 for n=1 (Newtonian) at any stress level,
   including the edge case background_stress=0 where UFL would otherwise
   evaluate 0^0=0 incorrectly.
3. power_law_factor returns the analytically correct value for n=3 with
   background_stress=0.
"""

import pytest
import firedrake as fd
from firedrake import assemble, dx
import numpy as np

from gadopt import MaxwellApproximation


@pytest.fixture
def mesh_and_spaces():
    mesh = fd.UnitCubeMesh(1, 1, 1)
    mesh.cartesian = True
    S = fd.TensorFunctionSpace(mesh, "DG", 0)
    return mesh, S


def _eval_scalar(expr, mesh):
    """Integrate a scalar UFL expression over the unit cube and return the mean value."""
    vol = assemble(fd.Constant(1) * dx(domain=mesh))
    return assemble(expr * dx(domain=mesh)) / vol


def test_second_stress_invariant(mesh_and_spaces):
    """second_stress_invariant should compute sqrt(0.5 * s_ij * s_ij) = sqrt(J2).

    For a constant symmetric deviatoric tensor with known components the result
    must match the analytic sqrt(J2) value.
    """
    mesh, S = mesh_and_spaces
    approx = MaxwellApproximation(
        bulk_modulus=1, density=1, shear_modulus=1, viscosity=1
    )

    # Build a constant symmetric deviatoric tensor: diagonal (2, -1, -1) (trace=0)
    s = fd.Function(S)
    s.interpolate(fd.as_tensor([[2., 0., 0.],
                                [0., -1., 0.],
                                [0., 0., -1.]]))

    # Analytic: 0.5 * (4 + 1 + 1) = 3  ->  sqrt(J2) = sqrt(3)
    expected = np.sqrt(3.0)
    result = _eval_scalar(approx.second_stress_invariant(s), mesh)
    # small offset 1e-16 inside sqrt makes this approximate
    assert abs(result - expected) < 1e-6


def test_power_law_factor_newtonian(mesh_and_spaces):
    """power_law_factor must equal 1 for n=1 (Newtonian) at any stress level.

    This covers the 0^0 fix: with background_stress=0 and n=1 the numerator
    formerly evaluated to 0 (UFL's 0^0=0) giving factor=0.5 instead of 1.
    """
    mesh, S = mesh_and_spaces
    approx = MaxwellApproximation(
        bulk_modulus=1, density=1, shear_modulus=1, viscosity=1,
        exponent=1, transition_stress=1, background_stress=0,
    )

    # Non-zero stress to exercise the formula with a meaningful invariant.
    s = fd.Function(S)
    s.interpolate(fd.as_tensor([[2., 0., 0.],
                                [0., -1., 0.],
                                [0., 0., -1.]]))

    result = _eval_scalar(approx.power_law_factor(s), mesh)
    assert abs(result - 1.0) < 1e-10


def test_power_law_factor_nonlinear(mesh_and_spaces):
    """power_law_factor returns the analytic value for n=3, background_stress=0.

    With n=3 and background_stress=0:
        f = 1 / (1 + (sigma / sigma*)^(n-1))
          = 1 / (1 + (sigma / sigma*)^2)

    For the tensor with sqrt(J2) = sqrt(3) and transition_stress = sqrt(3):
        f = 1 / (1 + 1) = 0.5
    """
    mesh, S = mesh_and_spaces
    sigma_star = np.sqrt(3.0)
    approx = MaxwellApproximation(
        bulk_modulus=1, density=1, shear_modulus=1, viscosity=1,
        exponent=3, transition_stress=sigma_star, background_stress=0,
    )

    s = fd.Function(S)
    s.interpolate(fd.as_tensor([[2., 0., 0.],
                                [0., -1., 0.],
                                [0., 0., -1.]]))

    # sigma = sqrt(J2) = sqrt(3) = sigma_star  ->  f = 1 / (1 + 1^2) = 0.5
    expected = 0.5
    result = _eval_scalar(approx.power_law_factor(s), mesh)
    assert abs(result - expected) < 1e-6
