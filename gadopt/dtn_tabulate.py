r"""Numerical tabulation of the DtN boundary eigenfunctions.

`gadopt.spherical_harmonics.real_spherical_harmonic` builds one UFL expression
per $(l, m)$, each a Horner evaluation of $d^{|m|} P_l / du^{|m|}$ whose
coefficients come from sympy. That is the right representation for a form that
has to be differentiated and taped, and the wrong one for evaluating every mode
of a truncation at every boundary quadrature point: it costs $O(L^2)$ separate
polynomial evaluations of average degree $2L/3$, i.e. $O(L^3)$ operations per
point.

This module evaluates *all* modes up to a truncation at once, in plain numpy,
by the stable upward recursion in $l$ at fixed $m$ used throughout geodesy (and
by Yu, Myhill & Al-Attar 2026 §2.3.2). Every mode at a point costs $O(L^2)$
operations in total - one fused pass rather than $O(L^2)$ independent Horner
evaluations - and nothing overflows, because the recursion runs on the
*normalised* functions rather than on $P_l^m$ and its normalisation separately.

The convention is `spherical_harmonics.py`'s, bit for bit:

$$
  Y_{l0} = N_{l0} P_l(u), \qquad
  Y_{lm} = \sqrt{2} N_{l|m|} \frac{d^{|m|} P_l}{du^{|m|}}(u)
           \sin^{|m|}\theta \begin{cases} \cos(|m|\varphi) & m > 0 \\
                                          \sin(|m|\varphi) & m < 0 \end{cases}
$$

with $u = \cos\theta = z/r$ and
$N_{lm} = \sqrt{\tfrac{2l+1}{4\pi}\tfrac{(l-m)!}{(l+m)!}}$, normalised so that
$\int_{S^2} Y_{lm}^2 \, d\Omega = 1$. Written this way the Condon-Shortley phase
of the associated Legendre functions cancels against the phase of the real
combination, so there is no $(-1)^m$ anywhere below; that cancellation is the
one thing about this convention that a recursion can silently get wrong, and
`tests/unit/test_dtn_tabulate.py` pins it against both
`real_spherical_harmonic_numpy` and `scipy.special.sph_harm_y`.

The 2-D counterpart is the same exercise on $\{\cos m\varphi, \sin m\varphi\}$
by the complex-exponential (Chebyshev) recursion, and is included because the
2-D annulus is the fast iteration loop for anything built on top of this.

Mode ordering is `(l, m)` with `l` ascending and, within a degree, `m` running
from `-l` to `+l`; that is exactly the order `SphericalDtN.modes` builds its
table in, so row `k` of the returned array is mode `k` of the table. The 2-D
ordering is `cos 1, sin 1, cos 2, sin 2, ...`, matching `CylindricalDtN.modes`
after its optional leading `mean` mode.
"""

import numpy as np

__all__ = [
    "spherical_index", "tabulate_real_spherical_harmonics",
    "cylindrical_index", "tabulate_azimuthal_modes",
]


def spherical_index(L: int) -> list[tuple[int, int]]:
    """The `(l, m)` pairs up to degree `L`, in `SphericalDtN.modes` order."""
    return [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]


def cylindrical_index(M: int) -> list[tuple[int, str]]:
    """The `(m, parity)` pairs up to order `M`, in `CylindricalDtN.modes` order.

    `parity` is `"cos"` or `"sin"`; the `m = 0` mean mode of an interior
    boundary is not included, since it is the constant 1 and needs no
    tabulation.
    """
    return [(m, parity) for m in range(1, M + 1) for parity in ("cos", "sin")]


def _normalised_legendre(
    L: int, u: np.ndarray, sin_theta: np.ndarray
) -> np.ndarray:
    r"""Normalised associated Legendre functions without the Condon-Shortley phase.

    `sin_theta` is passed in rather than computed as `sqrt(1 - u**2)`: near a
    pole that expression loses half its significant digits to cancellation
    (`1 - u**2` is `O(theta**2)` against `u = O(1)`), and the caller has
    `sqrt(x**2 + y**2)/r` to hand, which does not. Measured, the difference is
    the whole error budget - `sqrt(1 - u**2)` costs about `4e-14` absolute at
    `l = 1`, against `1e-16` for the accurate form.

    Returns an array `P` of shape `(L + 1, L + 1, n_points)` whose entry
    `P[l, m]` is

    $$
      \bar P_l^m(u) = N_{lm} \, \sin^m\theta \, \frac{d^m P_l}{du^m}(u),
      \qquad N_{lm} = \sqrt{\tfrac{2l+1}{4\pi}\tfrac{(l-m)!}{(l+m)!}},
    $$

    for `0 <= m <= l`, and zero for `m > l`. Note that the $\sin^m\theta$
    factor lives *inside* $\bar P$: that is what keeps the recursion bounded
    (each entry is $O(1)$ for every $l$ and $m$), and it is why the caller must
    multiply by a bare $\cos m\varphi$ rather than by $A_m/r^m$.

    The recursion is the standard one, run entirely on the normalised
    functions:

        P[0,0] = 1/sqrt(4 pi)
        P[m,m] = sqrt((2m+1)/(2m)) sin(theta) P[m-1,m-1]
        P[m+1,m] = sqrt(2m+3) u P[m,m]
        P[l,m] = a u P[l-1,m] - b P[l-2,m],
            a = sqrt((2l+1)(2l-1) / ((l-m)(l+m)))
            b = sqrt((2l+1)(l+m-1)(l-m-1) / ((2l-3)(l-m)(l+m)))

    Running it on $P_l^m$ and normalising afterwards would overflow near
    $l + m \approx 170$ (the factorials in $N_{lm}$); this form has no such
    limit.
    """
    u = np.asarray(u, dtype=float)
    sin_theta = np.asarray(sin_theta, dtype=float)
    P = np.zeros((L + 1, L + 1) + u.shape)

    P[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
    for m in range(1, L + 1):
        P[m, m] = np.sqrt((2 * m + 1) / (2 * m)) * sin_theta * P[m - 1, m - 1]
    for m in range(L):
        P[m + 1, m] = np.sqrt(2 * m + 3) * u * P[m, m]
    for m in range(L - 1):
        for l in range(m + 2, L + 1):
            a = np.sqrt((2 * l + 1) * (2 * l - 1) / ((l - m) * (l + m)))
            b = np.sqrt((2 * l + 1) * (l + m - 1) * (l - m - 1)
                        / ((2 * l - 3) * (l - m) * (l + m)))
            P[l, m] = a * u * P[l - 1, m] - b * P[l - 2, m]
    return P


def _azimuthal(M: int, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`cos(m phi)` and `sin(m phi)` for `m = 0 .. M`, by complex recursion.

    `e^{i m phi} = e^{i (m-1) phi} e^{i phi}` preserves unit modulus to
    rounding, so the error grows like `m * eps` rather than compounding.
    """
    phi = np.asarray(phi, dtype=float)
    cos = np.empty((M + 1,) + phi.shape)
    sin = np.empty((M + 1,) + phi.shape)
    cos[0], sin[0] = 1.0, 0.0
    if M >= 1:
        cos[1], sin[1] = np.cos(phi), np.sin(phi)
    for m in range(2, M + 1):
        cos[m] = cos[m - 1] * cos[1] - sin[m - 1] * sin[1]
        sin[m] = sin[m - 1] * cos[1] + cos[m - 1] * sin[1]
    return cos, sin


def tabulate_real_spherical_harmonics(L: int, points: np.ndarray) -> np.ndarray:
    """Every real orthonormal `Y_lm` up to degree `L` at the given points.

    Arguments:
      L: Degree truncation (`L >= 0`).
      points: Array of shape `(n_points, 3)` of Cartesian coordinates. Only the
        direction matters - the harmonics are homogeneous of degree zero - but
        the points need not lie on a sphere.

    Returns:
      Array of shape `((L + 1)**2, n_points)`; row `k` is mode
      `spherical_index(L)[k]`, i.e. the same order as
      `SphericalDtN.modes`.

    At a pole the longitude is degenerate; `arctan2` returns 0 there and every
    mode with `m != 0` carries a `sin^|m| theta` factor that vanishes, so the
    result is 0 - which is what the UFL expression gives too, since its
    `A_m/r^m` factor vanishes at the pole for the same reason.
    """
    if L < 0:
        raise ValueError(f"Require L >= 0, got L={L}")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (n_points, 3), got {points.shape}")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    u = z / r
    phi = np.arctan2(y, x)

    P = _normalised_legendre(L, u, np.sqrt(x * x + y * y) / r)
    cos, sin = _azimuthal(L, phi)
    root2 = np.sqrt(2.0)

    out = np.empty(((L + 1) ** 2, points.shape[0]))
    for k, (l, m) in enumerate(spherical_index(L)):
        if m == 0:
            out[k] = P[l, 0]
        elif m > 0:
            out[k] = root2 * P[l, m] * cos[m]
        else:
            out[k] = root2 * P[l, -m] * sin[-m]
    return out


def tabulate_azimuthal_modes(M: int, points: np.ndarray) -> np.ndarray:
    """Every `cos(m phi)`, `sin(m phi)` for `m = 1 .. M` at the given points.

    Arguments:
      M: Order truncation (`M >= 0`; `M = 0` gives an empty tabulation).
      points: Array of shape `(n_points, 2)` of Cartesian coordinates.

    Returns:
      Array of shape `(2 * M, n_points)`; row `k` is mode
      `cylindrical_index(M)[k]`, i.e. `cos 1, sin 1, cos 2, ...`, matching
      `CylindricalDtN.modes` after its optional leading `mean` mode.
    """
    if M < 0:
        raise ValueError(f"Require M >= 0, got M={M}")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"points must have shape (n_points, 2), got {points.shape}")

    phi = np.arctan2(points[:, 1], points[:, 0])
    cos, sin = _azimuthal(M, phi)

    out = np.empty((2 * M, points.shape[0]))
    for k, (m, parity) in enumerate(cylindrical_index(M)):
        out[k] = cos[m] if parity == "cos" else sin[m]
    return out
