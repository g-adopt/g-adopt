r"""High-degree accuracy of `gadopt.spherical_harmonics`, against 50-digit mpmath.

`test_spherical_harmonics.py` pins the convention and runs to `L_MAX = 15`
against scipy. This module pins the thing that convention tests cannot see: how
the error *grows with degree*, referenced to arithmetic good enough to charge
each implementation only for its own rounding.

Both matter separately. The evaluation this module replaced agreed with every
convention test ever written - it had the right signs, the right `sqrt(2)` and
the right Condon-Shortley cancellation - and was still wrong by 1.6e-6 at
`l = 30`, because it summed monomial coefficients that grow like `2.3^l` to
make a function of size 1. A test that asks "is this the right harmonic?" at
low degree cannot distinguish that from a correct implementation. A test that
asks "how fast does the error grow?" can, and that is the whole content here.

Kept separate from `test_spherical_harmonics.py` for two reasons: it needs
`mpmath`, which is not a declared dependency of the package and is skipped if
absent, and the compiled-kernel test costs a TSFC compilation per mode.
"""

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from gadopt.spherical_harmonics import (  # noqa: E402
    real_spherical_harmonic,
    real_spherical_harmonic_numpy,
)

DPS = 50
DEGREE = 30


@pytest.fixture(scope="module")
def directions():
    """Random directions, well away from the poles, plus their angles."""
    rng = np.random.default_rng(20260728)
    theta = rng.uniform(0.15, np.pi - 0.15, 8)
    phi = rng.uniform(-np.pi, np.pi, 8)
    points = np.stack([np.sin(theta) * np.cos(phi),
                       np.sin(theta) * np.sin(phi),
                       np.cos(theta)], axis=1)
    return theta, phi, points


def mpmath_Y(l, m, theta, phi):
    r"""$Y_{lm}$ to 50 digits, from the construction the module documents.

    Built from `N_lm d^|m|P_l/du^|m| sin^|m|(theta) {cos,sin}(|m| phi)`
    directly, so it shares no arithmetic with either thing under test - not the
    recursion, not the constants in `associated_legendre.py`, and not scipy.
    """
    am = abs(m)
    with mp.workdps(DPS):
        N = mp.sqrt(mp.mpf(2 * l + 1) / (4 * mp.pi)
                    * mp.factorial(l - am) / mp.factorial(l + am))
        if m != 0:
            N *= mp.sqrt(2)
        out = []
        for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi)):
            u = mp.cos(mp.mpf(t))
            v = N * mp.diff(lambda x: mp.legendre(l, x), u, am)
            v *= mp.sin(mp.mpf(t)) ** am
            if m > 0:
                v *= mp.cos(am * mp.mpf(p))
            elif m < 0:
                v *= mp.sin(am * mp.mpf(p))
            out.append(v)
        return out


def worst_deviation(values, l, m, theta, phi):
    truth = mpmath_Y(l, m, theta, phi)
    return max(abs(float(values[i] - truth[i])) for i in range(len(truth)))


def test_numpy_path_at_degree_30(directions):
    """Every order at l = 30, within 1e-13 of 50-digit mpmath.

    The evaluation this replaced was at 1.6e-06 here.
    """
    theta, phi, _ = directions
    worst, worst_mode = 0.0, None
    for m in range(-DEGREE, DEGREE + 1):
        d = worst_deviation(
            real_spherical_harmonic_numpy(DEGREE, m, theta, phi),
            DEGREE, m, theta, phi)
        if d > worst:
            worst, worst_mode = d, (DEGREE, m)
    print(f"    worst at l={DEGREE}: {worst:.3e} at {worst_mode}")
    assert worst <= 1e-13, f"{worst:.3e} at {worst_mode}"


def test_error_does_not_grow_with_degree(directions):
    """The regression guard proper: no ~0.4 digits lost per degree.

    This is the test that would have caught the original defect, and the one
    that will catch its return. The absolute gate at any single degree is
    almost incidental - what is pinned is that the error at `l = 40` is not
    orders of magnitude worse than at `l = 8`. For reference, the coefficient
    evaluation gave 7.6e-15, 2.9e-10 and 1.6e-06 at degrees 8, 20 and 30, a
    growth of eleven orders of magnitude across the range this checks.
    """
    theta, phi, _ = directions
    worst = {}
    for l in (2, 8, 16, 24, 32, 40):
        # A spread of orders: zonal, tesseral, and the sectoral |m| = l corner.
        orders = sorted({0, 1, -1, l // 3, -(l // 2), l, -l})
        worst[l] = max(
            worst_deviation(real_spherical_harmonic_numpy(l, m, theta, phi),
                            l, m, theta, phi)
            for m in orders if abs(m) <= l)
        print(f"    l={l:2d}  {worst[l]:.3e}")

    assert max(worst.values()) <= 1e-13

    # The anti-exponential gate. Measured: 9.609e-16 at l = 8 and 1.928e-14 at
    # l = 40, a ratio of 20, so 1000 leaves a factor of 50 of headroom. The
    # coefficient path was at 7.6e-15 by l = 8 and, continuing to lose 0.4
    # digits per degree, would be near 1e-2 at l = 40 - a ratio of order 1e12,
    # missing this by nine orders of magnitude.
    #
    # l = 8 is the base rather than l = 2 because the l = 2 figure (1.669e-16)
    # is about one ulp of the values involved, so a ratio taken against it
    # measures quantisation rather than growth. The sequence is not monotone
    # either - 9.815e-15 at l = 24 against 4.968e-15 at l = 32 - which is
    # itself the signature of a quantity sitting on its floor and sampled at a
    # handful of orders, rather than one that is growing.
    assert worst[40] <= 1000.0 * worst[8]


def test_compiled_kernel_at_degree_30():
    """Criterion for this work: 1e-13 of mpmath at l = 30, through TSFC.

    The numpy twin and the UFL expression share a recursion but not a backend:
    this one is compiled to C by TSFC and evaluated by the resulting kernel,
    with GEM free to re-associate the arithmetic on the way. That freedom is
    precisely why the fix had to be a recursion whose stability does not depend
    on evaluation order, so the compiled path is the one that has to be
    measured - a numpy-only check would not have tested the property that made
    the approach viable.

    The mesh radius is 2.22, not 1: Y_lm is a function of direction only.
    """
    import firedrake as fd

    mesh = fd.CubedSphereMesh(radius=2.22, refinement_level=1, degree=2)
    V = fd.FunctionSpace(mesh, "CG", 2)
    X = fd.SpatialCoordinate(mesh)
    xyz = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(X)
    xyz = xyz.dat.data_ro

    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    worst, worst_mode = 0.0, None
    for m in (-DEGREE, -21, -7, 0, 7, 21, DEGREE):
        f = fd.Function(V).interpolate(real_spherical_harmonic(DEGREE, m, X))
        d = worst_deviation(f.dat.data_ro, DEGREE, m, theta, phi)
        if d > worst:
            worst, worst_mode = d, (DEGREE, m)
    print(f"    worst through compiled kernel: {worst:.3e} at {worst_mode}")
    assert worst <= 1e-13, f"{worst:.3e} at {worst_mode}"


def test_no_overflow_at_earth_radius():
    """High orders survive a mesh whose coordinates are in metres.

    The previous factorisation ran the azimuthal recurrence on `x, y` and
    divided by `r**|m|` at the end. On an Earth-radius mesh that power
    overflows double at `|m| ~ 46` - demonstrated in the first assertion below,
    which is the actual arithmetic that used to be performed. Running the
    recurrence on `x/r, y/r` instead costs one division rather than a power and
    cannot overflow at any order, since `|x/r| <= 1`.
    """
    from gadopt.dtn_tabulate import (
        spherical_index,
        tabulate_real_spherical_harmonics,
    )
    import firedrake as fd

    radius = 6.371e6
    with np.errstate(over="ignore"):  # the overflow is the point
        assert np.isinf(np.float64(radius) ** 46), (
            "the overflow this test guards against no longer reproduces")

    mesh = fd.CubedSphereMesh(radius=radius, refinement_level=1, degree=2)
    V = fd.FunctionSpace(mesh, "CG", 2)
    X = fd.SpatialCoordinate(mesh)
    xyz = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(X)
    xyz = xyz.dat.data_ro

    degree = 50
    table = tabulate_real_spherical_harmonics(degree, xyz)
    index = spherical_index(degree)

    for m in (-degree, -47, 47, degree):
        f = fd.Function(V).interpolate(real_spherical_harmonic(degree, m, X))
        got = f.dat.data_ro
        assert np.all(np.isfinite(got)), f"non-finite at m={m}"
        np.testing.assert_allclose(
            got, table[index.index((degree, m))], rtol=0, atol=1e-13,
            err_msg=f"(l, m) = ({degree}, {m}) at r = {radius}")
