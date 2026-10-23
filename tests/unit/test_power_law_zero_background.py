"""The power-law factor with no background stress has a derivative in the exponent.

`InternalVariableApproximation.power_law_factor` with a DG0 `Function`
exponent and a background stress of plain 0 (the GIAMIP Exp10 and Exp12
configuration). The cached solve blocks of Firedrake PR 4638 differentiate
the residual with respect to every dependency of the solve, the exponent
included. Written as `a ** (n - 1)` with `a = 0 / sigma*`, that derivative
contains `ln(a)`, which UFL folds to `ln(0)` and evaluates when the form is
built ("math domain error"). These tests build that derivative and check
that the factor keeps its values.
"""

import gadopt  # noqa: F401  (import gadopt before firedrake)
import firedrake as fd
import numpy as np
import ufl
from ufl.algorithms import expand_derivatives

from gadopt import CompressibleInternalVariableApproximation


def make_case(background_stress):
    """A unit square, a DG0 exponent of 1 and 3.5, and the factor of a stress field."""
    mesh = fd.UnitSquareMesh(4, 4)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    exponent = fd.Function(Q)
    exponent.dat.data[:] = np.where(np.arange(exponent.dat.data.size) % 2,
                                    3.5, 1.0)
    approximation = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        exponent=exponent, transition_stress=fd.Function(Q).assign(0.5),
        background_stress=background_stress)
    x = fd.SpatialCoordinate(mesh)
    stress = ufl.as_matrix([[x[0], x[1]], [x[1], -x[0]]])
    return mesh, Q, exponent, approximation, stress


def test_derivative_in_the_exponent_builds_with_zero_background():
    """d f / d n assembles (no ln(0)) and is finite."""
    mesh, Q, exponent, approximation, stress = make_case(0.0)
    f = approximation.power_law_factor(stress)
    form = f * fd.TestFunction(Q) * fd.dx
    dform = expand_derivatives(fd.derivative(form, exponent))
    values = fd.assemble(dform).M.values
    assert np.all(np.isfinite(values))


def test_factor_values_are_those_of_the_full_formula():
    """With zero background the factor is 1 / (1 + b^(n-1)) for n > 1 and 1 for n = 1."""
    mesh, Q, exponent, approximation, stress = make_case(0.0)
    f = fd.Function(Q).interpolate(approximation.power_law_factor(stress))
    sigma = fd.Function(Q).interpolate(
        approximation.second_stress_invariant(stress))
    n = exponent.dat.data_ro
    b = sigma.dat.data_ro / 0.5
    expected = np.where(n == 1.0, 1.0, 1.0 / (1.0 + b ** (n - 1.0)))
    assert np.allclose(f.dat.data_ro, expected, rtol=1e-12, atol=0)
