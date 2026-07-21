"""Tests for gadopt.spherical_harmonics: convention, orthonormality, UFL path."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gadopt.spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonic_numpy,
)

L_MAX = 6


def all_modes(l_max):
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def test_scipy_convention():
    """Matches sqrt(2) (-1)^m Re/Im of scipy's complex Y_l^m."""
    sph_harm_y = pytest.importorskip("scipy.special").sph_harm_y

    rng = np.random.default_rng(7)
    theta = rng.uniform(0.05, np.pi - 0.05, 40)
    phi = rng.uniform(0.0, 2.0 * np.pi, 40)

    for l, m in all_modes(L_MAX):
        y_complex = sph_harm_y(l, abs(m), theta, phi)
        if m > 0:
            reference = np.sqrt(2.0) * (-1) ** m * y_complex.real
        elif m < 0:
            reference = np.sqrt(2.0) * (-1) ** m * y_complex.imag
        else:
            reference = y_complex.real
        assert_allclose(
            real_spherical_harmonic_numpy(l, m, theta, phi), reference,
            atol=1e-13, err_msg=f"(l, m) = ({l}, {m})")


def test_orthonormality():
    """The full Gram matrix on the unit sphere is the identity."""
    n_quad = 40
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta = np.arccos(nodes)
    phi = 2.0 * np.pi * np.arange(2 * n_quad) / (2 * n_quad)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    quad_weights = np.outer(weights, np.full(2 * n_quad, np.pi / n_quad))

    modes = all_modes(L_MAX)
    values = {
        lm: real_spherical_harmonic_numpy(*lm, theta_grid, phi_grid)
        for lm in modes
    }
    for i, lm1 in enumerate(modes):
        for lm2 in modes[i:]:
            gram = np.sum(quad_weights * values[lm1] * values[lm2])
            assert_allclose(
                gram, 1.0 if lm1 == lm2 else 0.0, atol=1e-12,
                err_msg=f"{lm1} x {lm2}")


def test_ufl_matches_numpy():
    """The UFL expressions agree with the numpy twin on a cubed sphere.

    The mesh radius is not 1: Y_lm must be a function of direction only.
    """
    import firedrake as fd

    mesh = fd.CubedSphereMesh(radius=2.22, refinement_level=2, degree=2)
    V = fd.FunctionSpace(mesh, "CG", 2)
    X = fd.SpatialCoordinate(mesh)
    coords = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(X)
    xyz = coords.dat.data_ro
    r = np.sqrt(np.sum(xyz**2, axis=1))
    theta = np.arccos(np.clip(xyz[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    for l, m in all_modes(4):
        f = fd.Function(V).interpolate(real_spherical_harmonic(l, m, X))
        reference = real_spherical_harmonic_numpy(l, m, theta, phi)
        assert_allclose(
            f.dat.data_ro, reference, atol=1e-12,
            err_msg=f"(l, m) = ({l}, {m})")


def test_invalid_modes():
    with pytest.raises(ValueError, match="l >= 0"):
        real_spherical_harmonic_numpy(-1, 0, 0.5, 0.5)
    with pytest.raises(ValueError, match="m"):
        real_spherical_harmonic_numpy(1, 2, 0.5, 0.5)
