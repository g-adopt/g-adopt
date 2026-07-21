r"""Real orthonormal spherical harmonics as UFL expressions.

The real orthonormal spherical harmonics $Y_{lm}$ ($l \ge 0$, $-l \le m \le l$)
are the boundary eigenfunctions of the spherical Dirichlet-to-Neumann maps used
by the gravity solver. They satisfy

$$
  \int_{S^2} Y_{lm} \, Y_{l'm'} \, d\Omega = \delta_{ll'} \delta_{mm'},
$$

so on a sphere of radius $R$ the surface integral of $Y_{lm}^2$ is $R^2$.

Construction: with $u = z/r$ and the solid-harmonic factors
$A_m + i B_m = (x + i y)^m$,

$$
  Y_{l0} = N_{l0} P_l(u), \qquad
  Y_{lm} = \sqrt{2} N_{l|m|} \frac{d^{|m|} P_l}{du^{|m|}}(u)
           \frac{A_{|m|} \text{ or } B_{|m|}}{r^{|m|}},
$$

($A$ for $m > 0$, $B$ for $m < 0$), where $P_l$ is the Legendre polynomial and
$N_{lm} = \sqrt{\tfrac{2l+1}{4\pi} \tfrac{(l-m)!}{(l+m)!}}$. Written this way
the Condon-Shortley phases of the associated Legendre functions cancel against
the real-combination phase, and every factor is polynomial in $(x, y, z)$ and
$1/r$, i.e. directly expressible in UFL. The convention matches
`scipy.special.sph_harm_y`:
$Y_{lm} = \sqrt{2} (-1)^m \mathrm{Re}[Y_l^m]$ for $m > 0$,
$Y_{l0} = Y_l^0$, and
$Y_{lm} = \sqrt{2} (-1)^m \mathrm{Im}[Y_l^{|m|}]$ for $m < 0$.

The Legendre-derivative coefficients are generated once per $(l, m)$ with
sympy and cached; the returned expressions are plain UFL, so they can be used
in forms, differentiated and taped as usual.
"""

from functools import lru_cache
from math import factorial

import numpy as np
import sympy
from firedrake import dot, sqrt

__all__ = ["real_spherical_harmonic", "real_spherical_harmonic_numpy"]


@lru_cache(maxsize=None)
def _legendre_derivative_coefficients(l: int, m: int) -> tuple[float, ...]:
    """Coefficients of d^m P_l / du^m in ascending powers of u."""
    u = sympy.Symbol("u")
    poly = sympy.Poly(sympy.diff(sympy.legendre(l, u), u, m), u)
    return tuple(float(c) for c in reversed(poly.all_coeffs()))


def _normalisation(l: int, m: int) -> float:
    """Orthonormal prefactor: N_lm for m = 0, sqrt(2) N_lm otherwise."""
    am = abs(m)
    N = np.sqrt((2 * l + 1) / (4 * np.pi)
                * factorial(l - am) / factorial(l + am))
    return N if m == 0 else np.sqrt(2.0) * N


def _validate_mode(l: int, m: int):
    if l < 0:
        raise ValueError(f"Require l >= 0, got l={l}")
    if abs(m) > l:
        raise ValueError(f"Require |m| <= l, got l={l}, m={m}")


def real_spherical_harmonic(l: int, m: int, X):
    """Real orthonormal spherical harmonic Y_lm as a UFL expression.

    Arguments:
      l: Spherical harmonic degree (l >= 0).
      m: Spherical harmonic order (-l <= m <= l).
      X: Coordinate vector (e.g. `SpatialCoordinate(mesh)`) of a 3-D mesh.

    Returns:
      UFL expression for Y_lm evaluated at X, homogeneous of degree zero in X
      (i.e. a function of direction only).
    """
    _validate_mode(l, m)
    am = abs(m)
    r = sqrt(dot(X, X))
    u = X[2] / r

    # Horner evaluation of d^|m| P_l / du^|m| at u.
    coeffs = _legendre_derivative_coefficients(l, am)
    G = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        G = G * u + c

    if m == 0:
        return _normalisation(l, m) * G

    # A_m + i B_m = (x + i y)^m by recurrence, divided by r^m.
    A, B = 1.0, 0.0
    for _ in range(am):
        A, B = X[0] * A - X[1] * B, X[0] * B + X[1] * A
    azimuthal = (A if m > 0 else B) / r**am

    return _normalisation(l, m) * G * azimuthal


def real_spherical_harmonic_numpy(l: int, m: int, theta, phi):
    """Numpy twin of `real_spherical_harmonic` in spherical angles.

    Evaluates the same construction at colatitude theta and longitude phi
    (unit sphere). Used to validate the UFL expressions against
    scipy.special.sph_harm_y and in error assessments against passess.

    Arguments:
      l: Spherical harmonic degree (l >= 0).
      m: Spherical harmonic order (-l <= m <= l).
      theta: Colatitude(s) in [0, pi].
      phi: Longitude(s) in [0, 2 pi].

    Returns:
      Y_lm(theta, phi) as a float or ndarray.
    """
    _validate_mode(l, m)
    am = abs(m)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    u = np.cos(theta)

    G = np.polynomial.polynomial.polyval(
        u, _legendre_derivative_coefficients(l, am))

    if m == 0:
        result = _normalisation(l, m) * G
    else:
        # (sin theta)^m e^{i m phi} = ((x + i y)/r)^m
        azimuthal = np.sin(theta)**am * (np.cos(am * phi) if m > 0
                                         else np.sin(am * phi))
        result = _normalisation(l, m) * G * azimuthal
    return result.item() if result.ndim == 0 else result
