r"""The scalar constants of the normalised associated Legendre recursion.

Two places in this package evaluate the associated Legendre functions by the
standard upward recursion in $l$ at fixed $m$: `dtn_tabulate.py`, which fuses
every mode of a truncation into one $O(L^2)$ numpy pass, and
`spherical_harmonics.py`, which builds one UFL expression per mode. Their loop
structures are necessarily different and must stay that way - routing the
tabulation through a single-mode function would turn its $O(L^2)$ into
$O(L^3)$ - but the *numbers* they run on are the same four scalars.

This module owns those four scalars so that there is exactly one definition of
them. That is the whole reason it exists: a normalisation convention that
disagrees between the tabulated path and the symbolic path would be a silent,
untestable divergence, because each path would be internally consistent and
each would pass its own tests. Anything that changes here changes both.

The recursion, on the normalised functions

$$
  \bar P_l^m(u) = N_{lm} \sin^m\theta \frac{d^m P_l}{du^m}(u),
  \qquad N_{lm} = \sqrt{\tfrac{2l+1}{4\pi}\tfrac{(l-m)!}{(l+m)!}},
$$

is

    P[0,0] = 1/sqrt(4 pi)
    P[m,m] = sectoral_ratio(m) sin(theta) P[m-1,m-1]
    P[m+1,m] = first_step(m) u P[m,m]
    P[l,m] = recurrence_a(l,m) u P[l-1,m] - recurrence_b(l,m) P[l-2,m]

Running it on $P_l^m$ and normalising afterwards would overflow near
$l + m \approx 170$ (the factorials in $N_{lm}$); in this form nothing
overflows and nothing cancels, because every intermediate is a value of the
same $O(1)$ family as the answer.

The $\sin^m\theta$ factor is independent of $l$, so dividing it out leaves the
three-term recursion in $l$ untouched and changes only the seeds. The
sin-stripped functions

$$
  Q_l^m(u) = N_{lm} \frac{d^m P_l}{du^m}(u) = \bar P_l^m(u) / \sin^m\theta
$$

therefore obey the same recursion with `P[m,m]` replaced by the *constant*
`sectoral_seed(m)`, which is what makes the recursion expressible in UFL: a
UFL builder cannot form $\sin^m\theta$ from $z/r$ without reintroducing the
cancellation that $\sqrt{1 - u^2}$ carries near a pole, but it does not need
to, because the caller supplies that factor as $A_m/r^m$ or $B_m/r^m$.

Every function here returns a plain Python `float` and uses nothing but
`math`, so the recursion body written against these constants contains only
`+`, `-`, `*` and float literals and runs unchanged on numpy arrays and on UFL
expressions.

Each function states and enforces its valid domain. Outside it the closed
forms divide by zero, and the boundaries are not arbitrary: they are where one
step of the recursion stops existing and another takes over, so a
`ZeroDivisionError` there would be reporting a caller that has confused the
seed, the first step and the general step. They raise `ValueError` naming the
domain instead.
"""

import math
from functools import lru_cache

__all__ = [
    "recurrence_a", "recurrence_b", "first_step",
    "sectoral_ratio", "sectoral_seed",
]


def recurrence_a(l: int, m: int) -> float:
    r"""$\sqrt{(2l+1)(2l-1) / ((l-m)(l+m))}$, the $u\,P_{l-1}^m$ coefficient.

    Domain: `l > m >= 0`. The recursion itself only ever calls it with
    `l >= m + 2`; at `l == m` the denominator vanishes, which is not a
    degenerate case to be handled but the statement that the three-term
    recursion has no first step - use `sectoral_seed` and `first_step` there.
    """
    if m < 0 or l <= m:
        raise ValueError(f"Require l > m >= 0, got l={l}, m={m}")
    return math.sqrt((2 * l + 1) * (2 * l - 1) / ((l - m) * (l + m)))


def recurrence_b(l: int, m: int) -> float:
    r"""$\sqrt{(2l+1)(l+m-1)(l-m-1) / ((2l-3)(l-m)(l+m))}$, the $P_{l-2}^m$ one.

    Domain: `l >= m + 2 >= 2`. Below that there is no $P_{l-2}^m$ to
    subtract; at `l == m + 1` the numerator vanishes anyway, but the caller
    should be using `first_step`, so this refuses rather than returning 0.
    """
    if m < 0 or l < m + 2:
        raise ValueError(f"Require l >= m + 2, m >= 0, got l={l}, m={m}")
    return math.sqrt((2 * l + 1) * (l + m - 1) * (l - m - 1)
                     / ((2 * l - 3) * (l - m) * (l + m)))


def first_step(m: int) -> float:
    r"""$\sqrt{2m+3}$, taking $P_m^m$ to $P_{m+1}^m$. Domain: `m >= 0`.

    This is *not* a special case, appearances notwithstanding. Putting
    $l = m+1$ into `recurrence_b` gives a factor $(l - m - 1) = 0$, so the
    general step degenerates to $a\,u\,P_m^m$ with nothing to subtract; and
    putting $l = m+1$ into `recurrence_a` gives
    $\sqrt{(2m+3)(2m+1) / (1 \cdot (2m+1))}$, whose $(2m+1)$ cancels to leave
    exactly $\sqrt{2m+3}$. Verified bit-for-bit against
    `recurrence_a(m + 1, m)` for `m` up to 400 in
    `tests/unit/test_associated_legendre.py`.

    It is kept as its own name because the alternative is to call
    `recurrence_b(m + 1, m)` and rely on multiplying by a zero that the domain
    guard now refuses to compute - a step whose correctness rests on a term
    vanishing is worth naming rather than inlining.
    """
    if m < 0:
        raise ValueError(f"Require m >= 0, got m={m}")
    return math.sqrt(2 * m + 3)


def sectoral_ratio(m: int) -> float:
    r"""$\sqrt{(2m+1)/(2m)}$, taking $\bar P_{m-1}^{m-1}$ to $\bar P_m^m$.

    In the tabulated path this multiplies a $\sin\theta$; in the sin-stripped
    path it is the only thing left, and accumulating it gives `sectoral_seed`.

    Domain: `m >= 1`. There is no step into $\bar P_0^0$, which is the seed
    `sectoral_seed(0)` and not the product of anything.
    """
    if m < 1:
        raise ValueError(f"Require m >= 1, got m={m}")
    return math.sqrt((2 * m + 1) / (2 * m))


@lru_cache(maxsize=None)
def sectoral_seed(m: int) -> float:
    r"""$Q_m^m = \frac{1}{\sqrt{4\pi}} \prod_{j=1}^{m} \sqrt{\frac{2j+1}{2j}}$.

    The seed of the sin-stripped recursion, and a pure constant - all of the
    $u$ dependence of $\bar P_m^m$ sits in the $\sin^m\theta$ that has been
    divided out.

    It is accumulated by repeated multiplication by `sectoral_ratio`, i.e. by
    exactly the recurrence the tabulated path runs, rather than evaluated in
    closed form. That is deliberate: the two paths then agree bit for bit on
    this constant by construction and not by coincidence of rounding. The cost
    is `m` roundings, a relative error of order `m * eps` (measured against
    50-digit mpmath: 2.2e-16 at `m = 30`, 2.7e-15 at `m = 200`), which is well
    inside the 1e-13 the callers are gated at.

    Nothing overflows on the way: $\prod (2j+1)/(2j)$ grows only like
    $\sqrt{m}$, so the seed itself is still 1.128 at `m = 200`.
    """
    if m < 0:
        raise ValueError(f"Require m >= 0, got m={m}")
    value = 1.0 / math.sqrt(4.0 * math.pi)
    for j in range(1, m + 1):
        value *= sectoral_ratio(j)
    return value
