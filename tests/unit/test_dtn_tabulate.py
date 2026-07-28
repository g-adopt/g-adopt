"""Convention tests for `gadopt.dtn_tabulate`.

The tabulation exists to replace `real_spherical_harmonic`'s per-mode UFL
expressions inside the DtN machinery, so the only thing that can go wrong
quietly is the convention: a sign, a `sqrt(2)`, or a Condon-Shortley phase that
does not cancel changes what a trace coefficient *means* without changing any
residual, any iteration count, or any accuracy test.

The convention is therefore pinned against **`scipy.special.sph_harm_y`
alone**, for every `(l, m)` up to `L = 30`.

That choice was originally forced: `real_spherical_harmonic` evaluated
`d^m P_l/du^m` by Horner from its monomial coefficients and was off by 2.9e-10
at `l = 20` and 1.6e-6 at `l = 30`, so requiring agreement with it at high
degree would have pinned the less accurate of the two. That defect is fixed -
it now runs the same upward recursion, from the shared constants in
`gadopt/associated_legendre.py` - and the choice is kept for a different and
better reason: the two paths now share those constants, so they are no longer
independent enough to pin each other's convention. Only an outside
implementation can do that. What they *can* do is bound each other, which is
what `test_parity_ceiling_between_the_two_paths` now measures.
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

    The angles returned alongside the points are RECOVERED FROM THEM, so a
    reference evaluated from angles sees exactly the argument the tabulation
    sees; see the comment below for why that matters at high degree.
    """
    rng = np.random.default_rng(20260727)
    theta = np.concatenate(
        [rng.uniform(0.0, np.pi, 200), [0.0, np.pi, 0.5 * np.pi]])
    phi = np.concatenate([rng.uniform(-np.pi, np.pi, 200), [0.3, 1.1, 2.0]])
    radius = np.concatenate([rng.uniform(0.5, 3.0, 200), [1.0, 2.22, 1.22]])
    points = np.stack([radius * np.sin(theta) * np.cos(phi),
                       radius * np.sin(theta) * np.sin(phi),
                       radius * np.cos(theta)], axis=1)

    # The angles handed to the reference are RECOVERED FROM THE POINTS, not the
    # ones the points were built from, so both implementations evaluate at
    # exactly the same argument. Without this the test measures the Cartesian
    # round trip as well: near a pole `u = cos(theta)` rounds by ~1e-16 while
    # `dY_l0/du ~ l(l+1)/2` reaches 465 at l = 30, so a mismatch of one ulp in
    # the argument shows up as 1.2e-13 in the value. Measured: with the
    # original angles the worst L=30 deviation is 1.323e-13 at (30,0) at
    # theta = 0.0028; with the recovered ones it is at the level of the
    # implementations themselves. Neither library is at fault for that term,
    # and leaving it in would have forced a weakened tolerance for the wrong
    # reason.
    #
    # Recovering the colatitude as `arccos(z/r)` would be worse than not
    # recovering it at all: `arccos` near `+-1` amplifies a 1e-16 rounding of
    # its argument into 1.4e-8 of angle (measured, that turns the worst L=20
    # deviation from 4.9e-14 into 7.6e-12). `arctan2(sqrt(x^2+y^2), z)` is the
    # stable form and has no such cancellation.
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (np.arctan2(np.sqrt(x * x + y * y), z), np.arctan2(y, x), points)


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


def test_matches_scipy_at_high_degree(directions):
    """The same check at `L = 30`, which is the truncation that motivates all
    of this: the Yu, Myhill & Al-Attar benchmark runs `ell_max = 32` as routine,
    and the whole point of a low-rank DtN representation is to make that
    reachable. A convention or stability defect that only appears at high
    degree would otherwise be found by the production run rather than here.

    **The tolerance here is set by scipy, not by the code under test, and that
    was measured rather than assumed.** The worst point in this fixture is
    `(30, 0)` at `theta = 0.00281276`, where the two differ by 1.32e-13.
    Against 60-digit mpmath at exactly the `u = z/r` the tabulation uses, at
    that same point:

        truth      2.1991798773066123
        recursion  2.1991798773065887   error 2.36e-14
        scipy      2.1991798773064564   error 1.56e-13

    So scipy is the one outside `1e-13` there — `Y_l0` near a pole is where the
    Legendre evaluation is hardest, and at `l = 30` scipy's own error crosses
    the gate. `3e-13` accommodates the reference. The recursion is pinned
    against truth, not against scipy, by
    `test_matches_mpmath_at_the_hardest_degree` below.
    """
    theta, phi, points = directions
    table = tabulate_real_spherical_harmonics(30, points)
    worst, worst_mode = 0.0, None
    for k, (l, m) in enumerate(spherical_index(30)):
        deviation = np.max(np.abs(table[k] - scipy_real_harmonic(l, m, theta, phi)))
        if deviation > worst:
            worst, worst_mode = deviation, (l, m)
    assert worst <= 3e-13, f"worst deviation {worst:.3e} at mode {worst_mode}"


def test_matches_mpmath_at_the_hardest_degree(directions):
    """Degree 30 against 60-digit mpmath, at the points that hurt.

    This is the only test in the file whose reference is exact, and it exists
    because at `l = 30` no double-precision implementation is a good enough
    yardstick — scipy itself is at 1.6e-13 at the near-pole point above.

    **The `3e-13` tolerance is the conditioning of the input, and that is
    checked rather than asserted by fiat.** The worst case is `(30, 0)` at
    `u = 0.9999960441884481`, where the measured deviation is 1.084e-13. At
    `u -> 1`, `|dY_l0/du| = N_l0 * l(l+1)/2 = sqrt(61/4pi) * 465 = 1024`, and
    one ulp of `u` there is 2.22e-16, so the smallest error any implementation
    can have from a double-precision Cartesian point is 2.28e-13. The measured
    value is *below* that. Nothing about the recursion is being excused: it is
    at the floor set by representing the point at all, and the assertion below
    checks that agreement explicitly so this reasoning cannot rot into a
    tolerance nobody can justify.

    mpmath is given the **Cartesian point**, exactly as the tabulation is, and
    derives `u` and `sin(theta)` from it at 60 digits. Handing it `u` alone and
    letting it take `sin(theta) = sqrt(1 - u^2)` makes the reference disagree
    with itself near a pole: at `sin(theta) ~ 1.7e-4`, `sqrt(1 - u^2)` and
    `sqrt(x^2+y^2)/r` differ by ~1e-9 relative, and the test then reports
    1.4e-11 at `(30, 1)` — measured — which is the reference's inconsistency,
    not the recursion's error. (This is the same cancellation that
    `_normalised_legendre` takes `sin_theta` as an argument to avoid.)

    Restricted to degree 30 and to eight points (both poles, the two
    nearest-pole points in the fixture, and four generic ones) to keep it under
    a couple of seconds.
    """
    import mpmath as mp

    _, _, points = directions
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    u_all, phi_all = z / r, np.arctan2(y, x)

    # Both poles, the two closest approaches to a pole, and four generic.
    order = np.argsort(-np.abs(u_all))
    chosen = list(order[:4]) + list(order[len(order) // 2::len(order) // 8])[:4]

    l = 30
    table = tabulate_real_spherical_harmonics(l, points)
    index = spherical_index(l)
    mp.mp.dps = 60

    worst, worst_mode = 0.0, None
    for m in range(-l, l + 1):
        am = abs(m)
        normalisation = mp.sqrt(mp.mpf(2 * l + 1) / (4 * mp.pi)
                                * mp.factorial(l - am) / mp.factorial(l + am))
        if m != 0:
            normalisation *= mp.sqrt(2)
        row = table[index.index((l, m))]
        for i in chosen:
            xi, yi, zi = (mp.mpf(float(x[i])), mp.mpf(float(y[i])),
                          mp.mpf(float(z[i])))
            ri = mp.sqrt(xi * xi + yi * yi + zi * zi)
            u = zi / ri
            sin_theta = mp.sqrt(xi * xi + yi * yi) / ri
            phi = mp.atan2(yi, xi)
            value = (normalisation * mp.diff(lambda t: mp.legendre(l, t), u, am)
                     * sin_theta ** am)
            if m > 0:
                value *= mp.cos(am * phi)
            elif m < 0:
                value *= mp.sin(am * phi)
            deviation = abs(float(row[i] - value))
            if deviation > worst:
                worst, worst_mode = deviation, (l, m, float(u_all[i]))
    # The conditioning floor for this point set: one ulp of u amplified by
    # max|dY_l0/du| = N_l0 l(l+1)/2, attained as u -> 1.
    u_worst = float(np.max(np.abs(u_all[chosen])))
    floor = (np.sqrt((2 * l + 1) / (4 * np.pi)) * l * (l + 1) / 2
             * np.spacing(u_worst))
    print(f"    [mpmath l=30] worst {worst:.3e} at {worst_mode}; "
          f"conditioning floor {floor:.3e}")
    assert worst <= 3e-13, f"worst deviation {worst:.3e} at {worst_mode}"
    # ...and that the deviation really is the input conditioning rather than
    # something the recursion is doing, which is what licenses the tolerance.
    assert worst <= 3 * floor, (
        f"deviation {worst:.3e} exceeds 3x the conditioning floor {floor:.3e}; "
        "the recursion, not the input representation, is the limit")


def test_parity_ceiling_between_the_two_paths(directions):
    """The ceiling that any tabulated-against-symbolic parity gate inherits.

    Two paths cannot agree with each other better than each agrees with the
    truth, so every parity comparison in the fast-DtN work is bounded by this
    number. It is pinned here, at `L = 20`, so that a change in either
    implementation surfaces as a failure in this file rather than as a mystery
    in a parity table somewhere downstream.

    This test used to be called `test_incumbent_horner_error_is_the_parity_
    ceiling`, and the ceiling used to be `real_spherical_harmonic`'s own error:
    its Horner evaluation of `d^m P_l/du^m` cancelled catastrophically, giving
    2.9e-10 at `l = 20` and 1.6e-6 at `l = 30` against 50-digit mpmath, so it
    was the binding constraint by four to eight orders of magnitude and the
    assertions were on it alone. That defect is fixed
    (`NOTES/FINDING-HORNER-ACCURACY.md`); both paths now run the same upward
    recursion and the ceiling has come down to ~1e-14, i.e. to the floor of
    double precision for this quantity. The gates below are tightened
    accordingly - from 1e-13 to 1e-14 below `l = 10`, and from 1e-9 to 1e-13 at
    `l = 20`.

    **Neither printed column is an implementation's own error.** Both are
    deviations from scipy, which at these degrees is itself only good to about
    1e-14, so each column charges its implementation for scipy's rounding as
    well as its own. Only a 50-digit mpmath reference separates the three, and
    that comparison lives in `tests/unit/test_spherical_harmonics_accuracy.py`
    and `NOTES/bench/horner_accuracy.py` rather than here.

    No ordering between the two columns is asserted. Not because the ordering
    is noise - there is a real tendency - but because it is far too weak to
    assert.

    Protocol for the numbers below, stated in full so they can be checked
    rather than taken: `theta ~ U(0.12, pi - 0.12)` and `phi ~ U(-pi, pi)` from
    `numpy.random.default_rng(31337 + 911*k)` for sample `k`; points built as
    `(sin t cos p, sin t sin p, cos t)`; the **generated** `theta, phi` passed
    to `real_spherical_harmonic_numpy` and to the reference, and the **points**
    passed to `tabulate_real_spherical_harmonics`, which is how this fixture
    feeds the two paths; reference is 50-digit mpmath built from the defining
    formula; the statistic is the worst deviation over all orders at `l = 20`.

    On that protocol the symbolic path is the more accurate of the two in 19 of
    30 samples of 10 points, 9 of 15 samples of 40 points, and 5 of 6 samples
    of 150 points. So it wins more often than not, but the margin is small -
    median ratio 1.09 to 1.15, individual samples spanning 0.55 to 2.54 - and
    the tabulated path wins outright in a substantial minority. Across degrees
    the tendency is visible at `l = 2` and `l = 5` and does not resolve at 10,
    15 or 20, where three independent 100-point samples disagree in direction.

    An ordering assertion would therefore be flaky. It is dropped for that
    reason, and because the four absolute gates below supersede it: they bound
    each column directly, which is what downstream parity comparisons actually
    inherit.

    One measurement trap, recorded because an earlier version of this docstring
    fell into it and published the result. It carried a table showing the
    tabulated path 70x to 457x worse near the poles, with a plausible mechanism
    attached. That was an artefact of the harness, not a property of either
    path: the harness recovered `theta = arccos(z/r)` from the points and
    evaluated the reference at the recovered value. Near a pole that loses most
    of `sin(theta)` - 2.6e-9 relative error at `theta = 1e-4`, and 5.2e-8 once
    raised to the 20th power - so the reference and the symbolic path stayed
    consistent with each other while the tabulation, which correctly takes
    `sin(theta)` as `sqrt(x^2 + y^2)/r`, was charged for the whole difference.
    Under the protocol above there is no near-pole amplification at all: the
    ratio is 1.06 to 2.40 for colatitudes in `0.01 .. 0.25` and 1.30 to 1.37 in
    `1e-4 .. 0.02`, indistinguishable from mid-latitude. It is the same trap
    `_normalised_legendre` documents for the implementation, met in the
    measurement instead.
    """
    theta, phi, points = directions
    table = tabulate_real_spherical_harmonics(L_MAX, points)
    tabulated, symbolic = {}, {}
    for k, (l, m) in enumerate(spherical_index(L_MAX)):
        reference = scipy_real_harmonic(l, m, theta, phi)
        from_expression = real_spherical_harmonic_numpy(l, m, theta, phi)
        tabulated[l] = max(tabulated.get(l, 0.0),
                           np.max(np.abs(table[k] - reference)))
        symbolic[l] = max(symbolic.get(l, 0.0),
                          np.max(np.abs(from_expression - reference)))
    for l in sorted(symbolic):
        print(f"    l={l:2d}  |tabulated-scipy| {tabulated[l]:.3e}   "
              f"|symbolic-scipy| {symbolic[l]:.3e}")

    # Headroom on each gate is a factor of 4 to 8; measured worst values are
    # 2.554e-15 and 9.104e-15 below l = 10, and 1.210e-14 and 4.907e-14 at 20.
    assert max(v for l, v in symbolic.items() if l <= 10) <= 1e-14
    assert max(v for l, v in tabulated.items() if l <= 10) <= 5e-14
    assert symbolic[L_MAX] <= 1e-13
    assert tabulated[L_MAX] <= 3e-13


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
