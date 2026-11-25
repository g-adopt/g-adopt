r"""Closed-form potential of the study's checkerboard source, for error assessment.

The scaling model drives the shell with a two-bump spherical-harmonic
checkerboard density (`gravity_cubed_sphere.build_checkerboard_source`),

$$
\rho(r, \theta, \phi) = g_{out}(r) \sum_{l \le L} a_l \sum_m Y_{lm}
                      + g_{in}(r)  \sum_{l \le L} b_l \sum_m Y_{lm},
$$

with Gaussian radial bumps `g_out`, `g_in`. Because each term is a radial
profile times a surface harmonic, the potential separates mode by mode: for
$\rho = f(r) Y(\theta, \phi)$ with $Y$ any surface harmonic of degree $l$,
the solution of $\nabla^2 \psi = -4 \pi G \rho$ that is regular at the origin
and decays at infinity is $\psi = G \, U_l[f](r) \, Y(\theta, \phi)$ with

$$
U_l[f](r) = \frac{4\pi}{2l+1} \left[
    \int_{r_{min}}^{r} f(r') r' \left(\frac{r'}{r}\right)^{l+1} dr'
  + \int_{r}^{r_{max}} f(r') r' \left(\frac{r}{r'}\right)^{l} dr'
\right].
$$

That is the radial Green's function $g_l = r_<^l / ((2l+1) r_>^{l+1})$ of
`passess.spherical.PoissonSpherical3D`, integrated against an arbitrary radial
profile by quadrature instead of in closed form against a constant one. Two
things make this reference trustworthy where a hand-rolled solution would not
be. The bracket is written with the ratios `(r'/r)` and `(r/r')`, each bounded
by one over its own integration range, so nothing overflows or cancels at the
degrees this study reaches: the naive `r**(-(l+1)) * integral(r'**(l+2))`
form carries factors of `1e7` and `1e-7` separately at `l = 20`. And the
radial factor depends on `l` alone, never on `m` and never on the
normalisation of the harmonics, so multiplying it by G-ADOPT's own real
orthonormal harmonics cannot introduce the normalisation mismatch that
comparing against a complex-harmonic package invites.

`check_analytic_vs_passess.py` pins the quadrature against passess on
constant-density shells, where passess is exact. This module deliberately
carries no passess dependency, so the reference is available on a deployment
that has only G-ADOPT and Firedrake.

Since the source is band-limited to degree `L` and the DtN maps are exact for
every mode they treat, this is the exact solution of the *continuous* problem
the model discretises. The difference against the computed potential is
therefore discretisation error alone, with no truncation-error contribution -
which is what makes it usable both as a correctness check and as the measure of
how high a truncation a given mesh can carry.
"""

import numpy as np

from gadopt.spherical_harmonics import real_spherical_harmonic_numpy

# Gauss-Legendre nodes for the radial integrals. The integrands are a Gaussian
# times a power of r on an interval of width one, so this is far past the point
# of diminishing returns; the cost is negligible beside a single solve.
N_QUAD = 256


def _gauss_legendre(a, b, n=N_QUAD):
    """Nodes and weights of an `n`-point Gauss-Legendre rule on `[a, b]`.

    `a` may be an array (one interval per entry), in which case the returned
    arrays have shape `(len(a), n)`.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    half = 0.5 * (b - a)[:, None]
    mid = 0.5 * (b + a)[:, None]
    return mid + half * x[None, :], half * w[None, :]


def radial_potential_factor(degree, profile, radii, rmin, rmax, n_quad=N_QUAD):
    r"""Evaluate `U_l[f](r)` for one degree at the given radii.

    Args:
      degree: Spherical harmonic degree `l`.
      profile: Callable giving the radial density profile `f(r')`, vectorised
        over numpy arrays.
      radii: Radii at which to evaluate, array-like.
      rmin, rmax: Radial extent of the density support.
      n_quad: Points per Gauss-Legendre panel.

    Returns:
      Array of `U_l[f]` values, one per entry of `radii`.
    """
    r = np.atleast_1d(np.asarray(radii, dtype=float))
    if np.any(r <= 0.0):
        raise ValueError("radii must be positive")

    # Inner integral: r' runs from rmin up to r, so (r'/r) <= 1 throughout.
    lo, wlo = _gauss_legendre(np.full_like(r, rmin), r, n_quad)
    inner = np.sum(
        wlo * profile(lo) * lo * (lo / r[:, None]) ** (degree + 1), axis=1)

    # Outer integral: r' runs from r up to rmax, so (r/r') <= 1 throughout.
    hi, whi = _gauss_legendre(r, np.full_like(r, rmax), n_quad)
    outer = np.sum(
        whi * profile(hi) * hi * (r[:, None] / hi) ** degree, axis=1)

    return 4.0 * np.pi / (2.0 * degree + 1.0) * (inner + outer)


def checkerboard_mode_sums(lmax, theta, phi):
    """Per-degree sums of the real harmonics, `S_l = sum_m Y_lm`.

    The checkerboard source weights every order of a degree equally, so both
    the density and the potential are built from these sums. Returning them
    once keeps the source and its reference solution built from literally the
    same angular fields, which removes the most obvious way for a reference to
    disagree with the thing it is checking.

    Args:
      lmax: Maximum degree.
      theta: Colatitudes, array-like.
      phi: Longitudes, array-like.

    Returns:
      List of arrays `S_l`, indexed by degree.
    """
    return [
        sum(real_spherical_harmonic_numpy(l, m, theta, phi)
            for m in range(-l, l + 1))
        for l in range(lmax + 1)
    ]


def analytic_potential(xyz, lmax, rmin, rmax, r_out, r_in, width,
                       weights_out, weights_in, gravitational_constant=1.0,
                       mode_sums=None):
    """Potential of the checkerboard source at the given points.

    Args:
      xyz: `(npoints, 3)` array of Cartesian coordinates.
      lmax: Maximum degree of the source.
      rmin, rmax: Radial extent of the density support.
      r_out, r_in: Centres of the outer and inner Gaussian bumps.
      width: Gaussian width of both bumps.
      weights_out, weights_in: Per-degree weights `a_l`, `b_l`.
      gravitational_constant: `G`.
      mode_sums: Optional precomputed `S_l` from `checkerboard_mode_sums`.

    Returns:
      Array of potential values, one per point.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    if mode_sums is None:
        mode_sums = checkerboard_mode_sums(lmax, theta, phi)

    # Evaluating the radial factors once per distinct radius rather than once
    # per point: a radially extruded mesh puts every CG1 node on one of
    # `layers + 1` shells, so this is a factor of the horizontal DOF count.
    # Rounding guards against the last bit of the extrusion arithmetic
    # splitting a shell into two nearly-equal radii.
    unique_r, inverse = np.unique(np.round(r, 12), return_inverse=True)

    def bump(centre):
        return lambda rr: np.exp(-((rr - centre) / width) ** 2)

    psi = np.zeros_like(r)
    for l in range(lmax + 1):
        factor = (
            weights_out[l] * radial_potential_factor(
                l, bump(r_out), unique_r, rmin, rmax)
            + weights_in[l] * radial_potential_factor(
                l, bump(r_in), unique_r, rmin, rmax)
        )
        psi += factor[inverse] * mode_sums[l]

    return gravitational_constant * psi
