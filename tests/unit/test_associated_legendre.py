r"""Tests for `gadopt.associated_legendre`, the shared recursion constants.

This module is small, but it is the one place where a mistake would be both
silent and total: `dtn_tabulate.py` and `spherical_harmonics.py` now take their
normalisation convention from it, so an error here moves *both* paths by the
same amount and every test that compares them to each other still passes. The
tests below therefore check the constants against an outside reference
(50-digit mpmath), never against either consumer.
"""

import numpy as np
import pytest

from gadopt.associated_legendre import (
    first_step,
    recurrence_a,
    recurrence_b,
    sectoral_ratio,
    sectoral_seed,
)

mp = pytest.importorskip("mpmath")

DPS = 50


def mpmath_Q(l: int, m: int, u: float) -> float:
    r"""$Q_l^m(u) = N_{lm} d^m P_l/du^m(u)$ to 50 digits, from the definition.

    This is the quantity the recursion claims to compute, built from the
    defining formula rather than from any recursion, so it shares no arithmetic
    with the thing under test.
    """
    with mp.workdps(DPS):
        N = mp.sqrt(mp.mpf(2 * l + 1) / (4 * mp.pi)
                    * mp.factorial(l - m) / mp.factorial(l + m))
        d = mp.diff(lambda x: mp.legendre(l, x), mp.mpf(u), m)
        return N * d


def recursion_Q(l: int, m: int, u: float) -> float:
    """Drive the sin-stripped recursion using only the module's constants.

    Deliberately not `spherical_harmonics._Q`: the point is to test the
    constants, so the loop is written out here. If this and `_Q` ever disagree
    the fault is in the loop, not in the convention.
    """
    q = sectoral_seed(m)
    if l == m:
        return q
    q_prev, q = q, first_step(m) * u * q
    for k in range(m + 2, l + 1):
        q_prev, q = q, recurrence_a(k, m) * u * q - recurrence_b(k, m) * q_prev
    return q


@pytest.mark.parametrize("l", [0, 1, 2, 5, 12, 30, 60])
def test_constants_reproduce_the_definition(l):
    """The recursion built from these constants is N_lm d^m P_l/du^m."""
    for u in (-0.9, -0.3, 0.0, 0.25, 0.8):
        for m in range(l + 1):
            got = recursion_Q(l, m, u)
            want = mpmath_Q(l, m, u)
            assert abs(got - want) <= 1e-13 + 1e-13 * abs(want), (
                f"Q({l},{m},{u}) = {got!r}, want {float(want)!r}")


def test_seed_matches_its_closed_form():
    """`sectoral_seed` is the accumulated `sectoral_ratio`, to rounding.

    The two are separate entry points and the tabulated path uses the ratio
    while the symbolic path uses the seed, so they have to agree: this is the
    exact divergence the shared module exists to make impossible.
    """
    for m in range(0, 121):
        with mp.workdps(DPS):
            exact = 1 / mp.sqrt(4 * mp.pi)
            for j in range(1, m + 1):
                exact *= mp.sqrt(mp.mpf(2 * j + 1) / (2 * j))
        assert abs(sectoral_seed(m) - exact) <= 1e-14 * abs(exact), f"m={m}"


def test_seed_does_not_overflow_or_underflow():
    """It grows like sqrt(m), so there is no `l + m ~ 170` limit here.

    Running the recursion on unnormalised `P_l^m` and normalising afterwards
    would overflow in the factorials; this form is why that limit is absent.
    """
    values = [sectoral_seed(m) for m in range(0, 401)]
    assert all(np.isfinite(v) and v > 0.0 for v in values)
    assert values[400] < 2.0, values[400]
    assert values == sorted(values)


@pytest.mark.parametrize("bad", [(0, 0), (3, 3), (2, 5), (1, -1)])
def test_recurrence_a_rejects_outside_its_domain(bad):
    with pytest.raises(ValueError, match="l > m >= 0"):
        recurrence_a(*bad)


@pytest.mark.parametrize("bad", [(0, 0), (3, 3), (4, 3), (2, -1)])
def test_recurrence_b_rejects_outside_its_domain(bad):
    with pytest.raises(ValueError, match=r"l >= m \+ 2"):
        recurrence_b(*bad)


def test_sectoral_ratio_and_seed_reject_outside_their_domain():
    with pytest.raises(ValueError, match="m >= 1"):
        sectoral_ratio(0)
    with pytest.raises(ValueError, match="m >= 1"):
        sectoral_ratio(-1)
    with pytest.raises(ValueError, match="m >= 0"):
        sectoral_seed(-1)
    with pytest.raises(ValueError, match="m >= 0"):
        first_step(-1)


def test_first_step_is_the_general_step_with_b_zero():
    """`first_step(m)` and `recurrence_a(m + 1, m)` are the same float.

    I wrote this test the other way round first, asserting they differ,
    and it failed - which is the useful outcome and the reason it is kept.
    Substituting `l = m + 1` into `recurrence_a` gives
    `sqrt((2m+3)(2m+1) / (1 * (2m+1)))`, and the `(2m+1)` cancels exactly, in
    floating point as well as in exact arithmetic. The `P_{l-2}^m` term
    disappears at the same time, because `recurrence_b`'s `(l - m - 1)` factor
    is zero there.

    So the seed step is the general step, and `first_step` is a name for a
    special case that is not special. Pinning that here means a future
    simplification that inlines it can be checked rather than argued about.
    """
    for m in range(0, 400):
        assert first_step(m) == recurrence_a(m + 1, m), f"m={m}"
