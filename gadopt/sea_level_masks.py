r"""Smooth ocean and grounded-ice masks for the sea-level equation.

The sea-level equation needs two 0/1 fields on the surface: the ocean function
`C`, which is 1 where the sea level `SL` is above the solid surface, and the
grounded-ice function `B`, which is 0 under ice that rests on the bed and 1
elsewhere. Both are switched by a sign change, and both enter a residual that
Newton and the adjoint differentiate. They are therefore written here as smooth
UFL expressions:

    C = H_k(SL),    B = 1 - H_k(I - (rho_w / rho_i) SL)

with `H_k(x) = 0.5 (1 + tanh(k x / 2)) = 1 / (1 + exp(-k x))`.

## The steepness

`k` has units of one over the length unit of `SL`. The transition of `H_k` is
about `4 / k` wide in `SL`. The masks are integrated with the facet quadrature
rule of the surface, so the transition must span at least about one quadrature
point per width. The calibration on the 2-D annulus
(`NOTES/PLAN-SEA-LEVEL-2026-09-15-C.md` section 4c) gives the rule
`k s h / p <= 0.7 q / p`, with `h` the facet size, `s` the surface slope of `SL`,
`p` the polynomial degree of `SL` and `q` the facet quadrature degree. The
default is `k = alpha p / h` with `alpha = 0.5` (no slope), which gives
`k s h / p = alpha s` and is inside that limit for `q >= 2 p` and any slope
below about 3. With a frozen slope `Function`, `k = alpha p / (h s)` and the
transition has a fixed width in arc length; see `mask_steepness`.

`h` is `FacetArea ** (1 / (dim - 1))`: the facet length in 2-D and the facet
side in 3-D. `CellDiameter` is not used because TSFC does not compile it on a
P2-curved mesh ("Cannot handle geometric quantity type").
"""

from firedrake import FacetArea, Function, max_value, min_value, sqrt, tanh

#: Clamp of the tanh argument. `tanh(350)` is 1 in double precision and
#: `cosh(350)^2` is about 1e302, so the derivative `sech^2` stays finite. Without
#: the clamp a large `k x` overflows in the derivative and the adjoint returns
#: NaN.
SMOOTH_STEP_CLAMP = 350.0

#: The default factor of the steepness, `k = alpha p / (h s)`.
DEFAULT_ALPHA_MASK = 0.5

#: The default floor of a frozen slope. A flat region has no shoreline, so the
#: floor only has to keep `k` finite there.
DEFAULT_GRAD_FLOOR = 1e-3


def smooth_step(x, k):
    """The smooth Heaviside step `0.5 (1 + tanh(k x / 2))`.

    The tanh form equals the logistic function `1 / (1 + exp(-k x))`. It has
    more headroom before overflow in the derivative, and the argument is also
    clamped at `+-SMOOTH_STEP_CLAMP`, so the value is exactly 0 or 1 far from
    the switch and the derivative is finite for any `k x`.

    Args:
      x: the UFL expression whose sign selects the side. Positive gives 1.
      k: the steepness, in one over the units of `x`.

    Returns:
      A UFL expression with values in `[0, 1]`.
    """
    z = 0.5 * k * x
    # Clamp the argument, not the result, so that the derivative of the clamped
    # region is the exact zero of the `max_value`/`min_value` branch.
    z = max_value(min_value(z, SMOOTH_STEP_CLAMP), -SMOOTH_STEP_CLAMP)
    return 0.5 * (1.0 + tanh(z))


def mask_steepness(mesh, degree, alpha=DEFAULT_ALPHA_MASK, slope=None, *,
                   grad_floor=DEFAULT_GRAD_FLOOR):
    """The steepness `k = alpha p / (h s)` of the masks on the surface facets.

    Args:
      mesh: the mesh whose facet measure integrates the masks.
      degree: `p`, the highest polynomial degree in the sea level (the
        displacement or the potential).
      alpha: the dimensionless factor. The calibration allows `alpha <= 0.7 q/p`
        for a facet quadrature degree `q`.
      slope: `None` for `s = 1`, which makes the transition a fixed width in
        sea level. Otherwise a `Function` that holds a frozen surface slope
        `|grad SL|`, which makes the transition a fixed width in arc length.
      grad_floor: the lower bound applied to `slope`, so that a flat region
        gives a finite `k`.

    Returns:
      A UFL expression. It contains `FacetArea`, so it is valid only inside a
      facet integral.

    Raises:
      TypeError: if `slope` is not `None` and not a `Function`. A live
        expression such as `sqrt(dot(grad(SL), grad(SL)))` on the tape adds a
        `1 / |grad SL|^2` term to the adjoint that is not physics.
    """
    dim = mesh.geometric_dimension
    # The facet size: the facet length in 2-D, the square root of the facet
    # area in 3-D. The 2-D case avoids a power with exponent 1.
    area = FacetArea(mesh)
    h = area if dim == 2 else (sqrt(area) if dim == 3
                               else area ** (1.0 / (dim - 1)))
    k = alpha * degree / h
    if slope is None:
        return k
    if not isinstance(slope, Function):
        raise TypeError(
            "mask_steepness: `slope` must be a frozen Function, got "
            f"{type(slope).__name__}. A live slope expression puts the "
            "derivative of 1/|grad SL| on the tape, which is not physics. "
            "Interpolate the slope into a Function first.")
    return k / max_value(slope, grad_floor)


def ocean_function(SL, k):
    """The ocean function `C = H_k(SL)`: 1 where the sea level is positive."""
    return smooth_step(SL, k)


def grounded_ice_function(I, SL, k, rho_w, rho_i):
    """The grounded-ice function `B = 1 - H_k(I - (rho_w / rho_i) SL)`.

    Ice floats when its thickness is below `(rho_w / rho_i)` times the water
    depth `SL`. Then the argument is negative and `B = 1`: the ice counts as
    water. Ice thicker than that rests on the bed and `B = 0`. On land `SL < 0`,
    so any ice thickness gives `B = 0` and no ice gives `B = 1`, where the
    ocean function `C = 0` removes the water.

    Args:
      I: the ice thickness, in the units of `SL`.
      SL: the sea level.
      k: the steepness.
      rho_w, rho_i: the water and ice densities.
    """
    return 1.0 - smooth_step(I - (rho_w / rho_i) * SL, k)
