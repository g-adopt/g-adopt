"""Convention tests for `gadopt.dtn_tabulate`.

The tabulation exists to replace `real_spherical_harmonic`'s per-mode UFL
Horner expressions inside the DtN machinery, so the only thing that can go
wrong quietly is the convention: a sign, a `sqrt(2)`, or a Condon-Shortley
phase that does not cancel changes what a trace coefficient *means* without
changing any residual, any iteration count, or any accuracy test. These tests
therefore pin the tabulation against two independent references - the module it
replaces, and scipy - for every `(l, m)` up to `L = 20`.
"""

import numpy as np
import pytest
import scipy.special as sp

from gadopt.dtn_tabulate import (
    cylindrical_index, spherical_index, tabulate_azimuthal_modes,
    tabulate_real_spherical_harmonics)
from gadopt.spherical_harmonics import real_spherical_harmonic_numpy

L_MAX = 20
M_MAX = 32


@pytest.fixture(scope="module")
def directions():
    """Random directions plus the three degenerate ones (both poles, equator).

    Radii are deliberately not 1: the harmonics are homogeneous of degree zero,
    and a hidden dependence on `|X|` would otherwise pass unnoticed - the DtN
    boundary is a *measured* sphere whose quadrature points are not at exactly
    the nominal radius.
    """
    rng = np.random.default_rng(20260727)
    theta = np.concatenate(
        [rng.uniform(0.0, np.pi, 200), [0.0, np.pi, 0.5 * np.pi]])
    phi = np.concatenate([rng.uniform(-np.pi, np.pi, 200), [0.3, 1.1, 2.0]])
    radius = np.concatenate([rng.uniform(0.5, 3.0, 200), [1.0, 2.22, 1.22]])
    points = np.stack([radius * np.sin(theta) * np.cos(phi),
                       radius * np.sin(theta) * np.sin(phi),
                       radius * np.cos(theta)], axis=1)
    return theta, phi, points


def scipy_real_harmonic(l, m, theta, phi):
    """`scipy.special.sph_harm_y` combined into the real orthonormal basis.

    The convention `spherical_harmonics.py` documents:
    `Y_lm = sqrt(2) (-1)^m Re[Y_l^m]` for `m > 0`, `Y_l0 = Y_l^0`,
    `Y_lm = sqrt(2) (-1)^|m| Im[Y_l^|m|]` for `m < 0`.
    """
    Y = sp.sph_harm_y(l, abs(m), theta, phi)
    if m == 0:
        return Y.real
    if m > 0:
        return np.sqrt(2.0) * (-1) ** m * Y.real
    return np.sqrt(2.0) * (-1) ** abs(m) * Y.imag


def test_matches_scipy(directions):
    """Every mode up to L=20 agrees with scipy to 1e-13.

    This is the convention check that matters: scipy is an independent
    implementation with a documented phase convention, so agreement pins the
    Condon-Shortley cancellation, the `sqrt(2)`, the `cos`/`sin` assignment to
    `m > 0` / `m < 0`, and the `4 pi` normalisation all at once.
    """
    theta, phi, points = directions
    table = tabulate_real_spherical_harmonics(L_MAX, points)
    worst, worst_mode = 0.0, None
    for k, (l, m) in enumerate(spherical_index(L_MAX)):
        deviation = np.max(np.abs(table[k] - scipy_real_harmonic(l, m, theta, phi)))
        if deviation > worst:
            worst, worst_mode = deviation, (l, m)
    assert worst <= 1e-13, f"worst deviation {worst:.3e} at mode {worst_mode}"


def test_incumbent_horner_error_is_the_parity_ceiling(directions):
    """Records how far the *incumbent* UFL construction is from scipy.

    This is a measurement, not a gate on the tabulation. Both routines are
    exact in exact arithmetic; in floating point the sympy/Horner evaluation of
    `d^m P_l / du^m` that `real_spherical_harmonic` and its numpy twin share
    cancels catastrophically as the degree grows, while the normalised
    recursion does not. So agreeing with the incumbent to `1e-13` at high
    degree is impossible, and demanding it would pin the *less* accurate of the
    two implementations.

    It is asserted rather than merely printed because the number it records is
    a hard ceiling on every parity comparison in the fast-DtN work: a tabulated
    path cannot agree with the symbolic one to better than the symbolic one
    agrees with itself. Below `l = 10` that ceiling is under `1e-13`, which is
    why the `1e-12` coefficient gate is meaningful there and not above.

    The wider consequence: at the truncations the low-rank representation
    exists to reach (`l ~ 32`), the current UFL path cannot represent its own
    modes to better than about `1e-6`, whatever it costs to evaluate them.
    """
    theta, phi, points = directions
    table = tabulate_real_spherical_harmonics(L_MAX, points)
    recursion, horner = {}, {}
    for k, (l, m) in enumerate(spherical_index(L_MAX)):
        exact = scipy_real_harmonic(l, m, theta, phi)
        incumbent = real_spherical_harmonic_numpy(l, m, theta, phi)
        recursion[l] = max(recursion.get(l, 0.0),
                           np.max(np.abs(table[k] - exact)))
        horner[l] = max(horner.get(l, 0.0),
                        np.max(np.abs(incumbent - exact)))
    for l in sorted(horner):
        print(f"    l={l:2d}  recursion {recursion[l]:.3e}   "
              f"incumbent {horner[l]:.3e}")

    # The recursion is uniformly the more accurate of the two above l = 6.
    assert recursion[L_MAX] < horner[L_MAX]
    # The ceiling the parity gates inherit, pinned so a change in either
    # implementation shows up here rather than as a mystery in a parity table.
    assert max(v for l, v in horner.items() if l <= 10) <= 1e-13
    assert horner[L_MAX] <= 1e-9


def test_azimuthal_modes(directions):
    """The 2-D `cos m phi` / `sin m phi` table, order and values."""
    _, phi, points_3d = directions
    points = np.stack([np.cos(phi), np.sin(phi)], axis=1) * 1.7
    table = tabulate_azimuthal_modes(M_MAX, points)
    worst = 0.0
    for k, (m, parity) in enumerate(cylindrical_index(M_MAX)):
        reference = np.cos(m * phi) if parity == "cos" else np.sin(m * phi)
        worst = max(worst, np.max(np.abs(table[k] - reference)))
    assert worst <= 1e-13, f"worst deviation {worst:.3e}"


def test_orthonormality(directions):
    """Discrete orthonormality on a Lebedev-free check: Gauss-Legendre x uniform.

    A product rule with enough points integrates `Y_lm Y_l'm'` exactly, so the
    Gram matrix must be the identity. This is independent of both references
    above - it tests the normalisation against the defining property rather
    than against another implementation - and it is the property
    `check_boundary_quadrature` asserts one diagonal entry of at a time.
    """
    L = 8
    nodes, weights = np.polynomial.legendre.leggauss(2 * L + 2)
    n_phi = 4 * L + 4
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    u = np.repeat(nodes, n_phi)
    w = np.repeat(weights, n_phi) * (2 * np.pi / n_phi)
    phi = np.tile(phi, nodes.size)
    sin_theta = np.sqrt(1.0 - u * u)
    points = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), u],
                      axis=1)
    table = tabulate_real_spherical_harmonics(L, points)
    gram = (table * w) @ table.T
    assert np.max(np.abs(gram - np.eye(gram.shape[0]))) <= 1e-12


def test_index_order_matches_the_mode_tables():
    """The tabulation rows are in `DtN.modes` order, which is what makes row
    `k` usable as mode `k` without a lookup."""
    import firedrake as fd

    from gadopt import CylindricalDtN, SphericalDtN

    mesh = fd.UnitCubeMesh(1, 1, 1)
    X = fd.SpatialCoordinate(mesh)
    spherical = SphericalDtN(L=3)
    keys = [mode.key for mode in spherical.modes("exterior", 1.0, X)]
    assert keys == [f"Y{l},{m}" for l, m in spherical_index(3)]

    mesh2 = fd.UnitSquareMesh(1, 1)
    X2 = fd.SpatialCoordinate(mesh2)
    cylindrical = CylindricalDtN(M=4)
    keys = [mode.key for mode in cylindrical.modes("exterior", 1.0, X2)]
    assert keys == [f"{parity}{m}" for m, parity in cylindrical_index(4)]

    interior = [mode.key for mode in cylindrical.modes("interior", 1.0, X2)]
    assert interior[0] == "mean"
    assert interior[1:] == keys


def test_input_validation():
    with pytest.raises(ValueError, match="Require L >= 0"):
        tabulate_real_spherical_harmonics(-1, np.zeros((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        tabulate_real_spherical_harmonics(2, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="Require M >= 0"):
        tabulate_azimuthal_modes(-1, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="shape"):
        tabulate_azimuthal_modes(2, np.zeros((1, 3)))
