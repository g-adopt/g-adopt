"""Tests for gadopt.spherical_harmonics: convention, orthonormality, UFL path.

`L_MAX` was 6. That is below `l = 7`, which is where the evaluation error of
this module starts to grow (NOTES/FINDING-HORNER-ACCURACY.md), so the suite was
structurally unable to see the defect it was nominally measuring: every
tolerance here passed with room to spare while the modes at the truncations the
solver actually runs at were losing digits. 15 is above the onset by enough for
the tolerances below to be gates rather than formalities.

Every `assert_allclose` below now passes `rtol=0`. Raising `L_MAX` alone
changed nothing, and the reason is worth keeping: `assert_allclose` tests
`|a - d| <= atol + rtol * |d|`, and it defaults to `rtol = 1e-7`. Against an
`O(1)` spherical harmonic that second term is about `5e-8`, so it dominated the
`atol` by five orders of magnitude and the stated `atol` was decorative -
`test_scipy_convention` passed an evaluation wrong by 3.1e-12 while advertising
a tolerance of 1e-13. The one assertion that ever bound was the *off-diagonal*
of the Gram matrix, where `desired` is exactly 0.0 and the relative term
vanishes. With `rtol = 0` the number written as `atol` is the whole tolerance
and means what it says.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gadopt.spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonic_numpy,
)

L_MAX = 15


def all_modes(l_max):
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def test_scipy_convention():
    """Matches sqrt(2) (-1)^m Re/Im of scipy's complex Y_l^m.

    This is a CONVENTION test: it pins signs, the `sqrt(2)`, and the
    Condon-Shortley cancellation. It is not an accuracy test, and reading it as
    one is a mistake worth spelling out, because the numbers invite it.

    Where it discriminates, and where it cannot. Measured over all 256 modes up
    to `L_MAX = 15`, at the 40 directions below, this module and scipy return
    **bit-identical doubles** on 16 of them - exactly the zonal `m = 0` modes,
    one per degree, and no mode with `m != 0`. On those 16 the difference this
    test computes is not small, it is *identically zero*, and no tolerance
    could make it otherwise: both sides run the same normalised upward
    recursion and commit the same roundings in the same order.

    That matters because the zonal modes are also where the deviation from
    truth is largest, so any accuracy-sounding number taken from this test is
    read off a mode where it has no accuracy-discriminating power at all. At
    `l = 15`, worst-mode deviation from 50-digit mpmath is 8.3323e-15 for this
    module and 8.3323e-15 for scipy - equal under `==`, not merely close,
    because at `m = 0` they are the same double.

    Two wrong readings of that coincidence, both recorded because both are
    plausible and one of them was written down here.

    It is not input conditioning - the idea that both evaluate at a `theta`
    that is itself a double. The mpmath reference is handed the *same* double
    `theta`, so any perturbation of the input is common to both sides of the
    comparison and cancels out of it.

    And it is not evidence of a shared double-precision floor, though that is
    the tempting reading and the conclusion happens to be true. Independent
    implementations sitting on a common floor produce *similar* numbers; they
    do not produce bit-identical ones. Exact equality is evidence of identical
    arithmetic, a stronger and different claim, and using the stronger
    observation to confirm the weaker one is how a measurement gets over-read.

    What a shared floor actually looks like is visible in a *different* pair.
    `dtn_tabulate`'s tabulation is the genuinely independent comparison against
    scipy - it reaches the same functions with the `sin^m(theta)` factor inside
    the recursion rather than divided out - and it is bit-identical to scipy at
    `l = 0` and nowhere else, at any order, through `l = 40`. Its deviations
    and scipy's are close without ever coinciding. This module, by contrast, is
    bit-identical to scipy at `m = 0` at *every* degree from 0 to 40, so the
    identity is not a low-degree artefact that dissolves as the recursion
    lengthens; it persists as far as it has been measured.

    One caution on how such pairs are quoted, including in the table at the top
    of `NOTES/FINDING-HORNER-ACCURACY.md`: figures like "1.095e-14 against
    scipy's 1.651e-14 at `l = 30`" are each a maximum over all orders, and the
    maxima are generally attained at *different* orders - measured on one
    sample, the tabulation's worst is at `m = 4` and scipy's at `m = 3`. So
    "close but not equal" there compares two argmaxes, not two evaluations of
    the same thing, and it is not a per-value statement.

    The coincidence at `m = 0` is bit-identity of the outputs, nothing subtler.

    None of which weakens the test for its actual purpose. A sign error, a
    missing `sqrt(2)` or an uncancelled `(-1)^m` is not a rounding difference,
    so it would break bit-identity at `m = 0` as loudly as anywhere else. The
    identity is a property of two implementations that currently agree, not a
    blind spot in what the test checks.

    Real discriminating power lives in the other 240 modes. There the worst
    disagreement is 2.8866e-15, at `(l, m) = (15, 1)`, so `atol = 1e-13` has a
    factor of 34.6 of headroom over the comparison it actually performs.

    Accuracy - referenced to arithmetic without a floor here - is
    `tests/unit/test_spherical_harmonics_accuracy.py`'s job, against mpmath.
    """
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
            rtol=0, atol=1e-13, err_msg=f"(l, m) = ({l}, {m})")


def test_orthonormality():
    """The full Gram matrix on the unit sphere is the identity.

    `atol` is 1e-13 rather than 1e-12 because 1e-12 was not a gate. Measured
    (NOTES/bench/lmax15_instrument_sensitivity.py), at `L_MAX = 15` the worst
    Gram entry is 5.562e-13 for an evaluation that is provably wrong by three
    orders of magnitude more than it should be: a factor of 1.8 inside a
    tolerance is not a test passing, it is a test failing to look.

    1e-13 is honest because it is set from the floor of the quantity rather
    than from what happens to pass. A stable evaluation puts the worst entry at
    1.446e-14, and that number is *flat* from `L = 12` to `L = 20` while the
    unstable one climbs from 5.15e-14 to 3.59e-11 over the same range - so
    1.446e-14 is this 40-point quadrature, not the harmonics, and the gate has
    a factor of 7 of real headroom under it.
    """
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
                gram, 1.0 if lm1 == lm2 else 0.0, rtol=0, atol=1e-13,
                err_msg=f"{lm1} x {lm2}")


def test_ufl_matches_tabulation():
    """The UFL expressions agree with an independent oracle on a cubed sphere.

    This compared against `real_spherical_harmonic_numpy` and ran to `l = 4`.
    Neither could discriminate. The numpy twin is not an independent
    implementation - it is the same construction with the same coefficients
    from the same generator, so it shares every error of the UFL path by
    design, and this test cannot see an evaluation defect at any degree, only a
    transcription slip between the two. And `l = 4` is below the degree at
    which anything interesting begins.

    The oracle is now `tabulate_real_spherical_harmonics`. One oracle, not two:
    adding a scipy comparison here would only re-test what
    `tests/unit/test_dtn_tabulate.py` already pins.

    How independent that oracle actually is, stated precisely, because it is
    partial and an overclaim here would be the same error this test was
    rewritten to fix. **Shared:** the four scalar constants of the recursion,
    from `gadopt/associated_legendre.py` - so a wrong `sqrt(2j+1)/(2j)` would
    move both sides together and this test would not see it. **Not shared:**
    the loop structure (one fused `O(L^2)` pass over all modes against a
    per-mode `O(l)` expression); the seeds and the factorisation (the
    tabulation carries `sin^m(theta)` inside the recursion, this path divides
    it out and supplies it as `A_m/r^m` built from `x/r`, `y/r`); the backend
    (interpreted numpy against C emitted by TSFC, with GEM free to
    re-associate); and the angles (the tabulation works from Cartesian points
    via `arctan2`, the expression from `X` directly).

    So this test discriminates evaluation, factorisation and compilation, and
    does *not* discriminate the shared constants. Those are covered against
    50-digit mpmath in `tests/unit/test_associated_legendre.py`, and the
    convention as a whole is pinned against scipy - a genuinely outside
    implementation - in `test_dtn_tabulate.py` and in `test_scipy_convention`
    above.

    Degree 16 is where the discrimination lives, and it is sampled at eleven
    orders rather than all 33, because each order is a separate TSFC
    compilation and the full block cost 27 s. The retained orders are
    `m = 0, +/-1, +/-2, +/-8, +/-15, +/-16`: zonal, both signs, mid-range,
    near-sectoral, and the sectoral `|m| = l` corner.

    The subset was checked to discriminate as well as the whole block rather
    than assumed to, since a cost trim that quietly removes the failing case is
    the obvious way for this to go wrong. Measured on this mesh, worst
    deviation from the tabulation over the retained eleven:

        stable recursion   5.773e-15  at m = 0   (17.3x inside the gate)
        Horner, pre-fix    3.420e-12  at m = 0   (34.2x outside it)

    Both figures are identical to the worst over all 33 orders - the extreme
    case is at `m = 0` in each direction, which is retained - so the trim costs
    nothing in discriminating power. `atol = 1e-13`, tightened from 1e-12.

    The `all_modes(4)` block is kept whole for convention coverage: every sign
    of m, m = 0, and the sectoral corner, where all implementations agree to
    1e-15 and the test is checking phases rather than accuracy.

    The mesh radius is not 1: Y_lm must be a function of direction only.
    """
    import firedrake as fd

    from gadopt.dtn_tabulate import (
        spherical_index,
        tabulate_real_spherical_harmonics,
    )

    mesh = fd.CubedSphereMesh(radius=2.22, refinement_level=2, degree=2)
    V = fd.FunctionSpace(mesh, "CG", 2)
    X = fd.SpatialCoordinate(mesh)
    coords = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(X)
    xyz = coords.dat.data_ro

    degree = 16
    table = tabulate_real_spherical_harmonics(degree, xyz)
    index = spherical_index(degree)

    orders = [0, 1, -1, 2, -2, 8, -8, 15, -15, 16, -16]
    modes = all_modes(4) + [(degree, m) for m in orders]
    for l, m in modes:
        f = fd.Function(V).interpolate(real_spherical_harmonic(l, m, X))
        assert_allclose(
            f.dat.data_ro, table[index.index((l, m))], rtol=0, atol=1e-13,
            err_msg=f"(l, m) = ({l}, {m})")


def test_invalid_modes():
    with pytest.raises(ValueError, match="l >= 0"):
        real_spherical_harmonic_numpy(-1, 0, 0.5, 0.5)
    with pytest.raises(ValueError, match="m"):
        real_spherical_harmonic_numpy(1, 2, 0.5, 0.5)
