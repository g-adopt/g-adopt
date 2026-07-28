r"""Real orthonormal spherical harmonics as UFL expressions.

The real orthonormal spherical harmonics $Y_{lm}$ ($l \ge 0$, $-l \le m \le l$)
are the boundary eigenfunctions of the spherical Dirichlet-to-Neumann maps used
by the gravity solver. They satisfy

$$
  \int_{S^2} Y_{lm} \, Y_{l'm'} \, d\Omega = \delta_{ll'} \delta_{mm'},
$$

so on a sphere of radius $R$ the surface integral of $Y_{lm}^2$ is $R^2$.

Construction: with $u = z/r$ and the normalised solid-harmonic factors
$A_m + i B_m = ((x + i y)/r)^m$,

$$
  Y_{l0} = Q_l^0(u), \qquad
  Y_{lm} = \sqrt{2} \, Q_l^{|m|}(u) \, (A_{|m|} \text{ or } B_{|m|}),
$$

($A$ for $m > 0$, $B$ for $m < 0$), where $P_l$ is the Legendre polynomial and

$$
  Q_l^m(u) = N_{lm} \frac{d^m P_l}{du^m}(u), \qquad
  N_{lm} = \sqrt{\tfrac{2l+1}{4\pi} \tfrac{(l-m)!}{(l+m)!}}.
$$

Written this way the Condon-Shortley phases of the associated Legendre
functions cancel against the real-combination phase, and every factor is
polynomial in $(x, y, z)$ and $1/r$, i.e. directly expressible in UFL. The
convention matches `scipy.special.sph_harm_y`:
$Y_{lm} = \sqrt{2} (-1)^m \mathrm{Re}[Y_l^m]$ for $m > 0$,
$Y_{l0} = Y_l^0$, and
$Y_{lm} = \sqrt{2} (-1)^m \mathrm{Im}[Y_l^{|m|}]$ for $m < 0$.

$Q_l^m$ is evaluated by upward recursion in $l$ (see `_Q`), and *not* from the
monomial coefficients of $d^m P_l/du^m$, which is what this module used to do.
Those coefficients alternate in sign and grow by a factor of about 2.3 per
degree while the function they sum to stays $O(1)$, so the Horner evaluation
cancelled catastrophically: it lost roughly 0.4 digits per degree above
$l = 7$ and was wrong by 1.6e-06 at $l = 30$, against ~1e-14 for a stable
evaluation (`NOTES/FINDING-HORNER-ACCURACY.md`). That capped the truncation
this path could represent at all, independently of any solver or tolerance.

The recursion is not merely more accurate, it is accurate for a reason that
survives compilation: every intermediate is a value of the same $O(1)$ family
as the answer, so the stability does not depend on the order the operations are
evaluated in. That matters here specifically, because GEM is free to
re-associate what TSFC compiles - a compensated-Horner scheme, which buys its
accuracy from a particular summation order, would not have been safe in a UFL
form. Measured against 50-digit mpmath at $l = 30$, through a compiled TSFC
kernel: 2.6e-14, where the coefficient path gave 9.9e-07. (That is
`NOTES/bench/ufl_recursion_check2_compiled.py`'s sampling; the permanent test
in `tests/unit/test_spherical_harmonics_accuracy.py` measures 6.6e-15 for the
same quantity on its own coarser mesh and point set. Both are real - the
quantity is a maximum over sampled points, so it moves with the sample.)

The returned expressions are plain UFL, so they can be used in forms,
differentiated and taped as usual. The scalar constants of the recursion come
from `associated_legendre.py`, shared with the numerical tabulation in
`dtn_tabulate.py` so that the two cannot disagree about the convention.
"""

from math import sqrt as _math_sqrt

import numpy as np
from firedrake import dot, sqrt

from .associated_legendre import (
    first_step,
    recurrence_a,
    recurrence_b,
    sectoral_seed,
)

__all__ = ["real_spherical_harmonic", "real_spherical_harmonic_numpy"]

_ROOT2 = _math_sqrt(2.0)


def _Q(l: int, m: int, u):
    r"""$Q_l^m(u) = N_{lm} d^m P_l/du^m (u)$, by upward recursion in $l$.

    `u` may be a float, a numpy array or a UFL expression: the body uses only
    `+`, `-`, `*` and float constants, so one implementation serves the
    symbolic and numerical paths. Returns a bare float when `l == m`, since
    $Q_m^m$ is a constant - all of the $u$ dependence of the sectoral harmonic
    lives in the azimuthal factor the caller supplies.

    This is `dtn_tabulate`'s $\bar P_l^m$ with the $\sin^m\theta$ factor
    divided out. That factor is independent of $l$, so dividing it out leaves
    the three-term recursion in $l$ untouched and changes only the seed, which
    becomes the constant `sectoral_seed(m)`. Stripping it is what makes the
    recursion expressible in UFL at all: a form has $z/r$, and recovering
    $\sin\theta$ from it as $\sqrt{1 - u^2}$ would lose half the significant
    digits near a pole - but nothing needs to, because the caller multiplies by
    $A_m$ or $B_m$, which carries exactly that factor and is computed from
    $x/r$ and $y/r$ without cancellation.

    A trap worth knowing about, in UFL specifically. `Q[l]` references both
    `Q[l-1]` and `Q[l-2]`, so the expression is a DAG with `O(l)` distinct
    nodes whose unfolding as a *tree* is exponentially large - the number of
    paths back to the seed grows like the Fibonacci numbers. Traversals that
    memoise see the DAG; traversals that do not see the tree.

    Where it counts everything memoises, and this costs nothing: measured at
    `l = 30`, `compute_form_data` - the entire compile path - takes 0.01 s, and
    the form signature that keys the assembly cache takes under 0.005 s. Form
    compile-plus-assemble at `(l, m) = (30, 15)` is 1.19 s against 1.21 s for
    the old coefficient expression, i.e. a wash. Both figures are COLD - a
    never-compiled mode pair in a fresh process. With TSFC's on-disk cache warm
    the same measurement reads 0.10 s against 0.09 s, a factor of 13 lower; the
    ratio, and so the conclusion, is unchanged either way. What is slow is unmemoised
    traversal for its own sake: `ufl.corealg.traversal.pre_traversal` takes
    1.9 s and `str(expr)` takes 6.8 s at `l = 30`, against roughly 0.00 s each
    before. So printing one of these expressions in a debugger, or writing a
    pass over `pre_traversal`, will appear to hang - for reasons that have
    nothing to do with the solver and do not affect it.
    """
    q_curr = sectoral_seed(m)
    if l == m:
        return q_curr
    q_prev, q_curr = q_curr, first_step(m) * u * q_curr
    for k in range(m + 2, l + 1):
        q_prev, q_curr = (
            q_curr,
            recurrence_a(k, m) * u * q_curr - recurrence_b(k, m) * q_prev,
        )
    return q_curr


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
    Q = _Q(l, am, X[2] / r)

    if m == 0:
        return Q

    # A_m + i B_m = ((x + i y)/r)^m by recurrence. Running this on the
    # normalised coordinates rather than on x, y with a trailing division by
    # r**m costs one division instead of a power, and removes an overflow: on
    # an Earth-radius mesh (r ~ 6.4e6) the un-normalised r**m overflows double
    # at |m| ~ 46, while |x/r| <= 1 cannot overflow at any m.
    xh, yh = X[0] / r, X[1] / r
    A, B = 1.0, 0.0
    for _ in range(am):
        A, B = xh * A - yh * B, xh * B + yh * A

    return _ROOT2 * Q * (A if m > 0 else B)


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

    Q = _Q(l, am, np.cos(theta))

    if m == 0:
        # `Q` is a bare float when l == 0; broadcast it to theta's shape.
        result = np.asarray(Q) * np.ones_like(theta)
    else:
        # (sin theta)^m e^{i m phi} = ((x + i y)/r)^m
        azimuthal = np.sin(theta)**am * (np.cos(am * phi) if m > 0
                                         else np.sin(am * phi))
        result = np.asarray(_ROOT2 * Q * azimuthal)
    return result.item() if result.ndim == 0 else result
