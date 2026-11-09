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
(`NOTES/findings/FINDING-mask-steepness.md`) gives the rule
`k s h / p <= 0.7 q / p`, with `h` the facet size, `s` the surface slope of `SL`,
`p` the polynomial degree of `SL` and `q` the facet quadrature degree. That
rule is about three times too weak: the measurement of step S1
(`NOTES/findings/FINDING-alpha-floor-mass.md`) reads "clean" as a
grid-frequency artefact below the 1.7e-7 resolution of the measure and allows
`alpha <= 1` at the solver's own `q = 9` with `p = 2`, where the old rule gives
`0.7 q / p = 3.15`. The default is `k = alpha p / h` with `alpha = 1` (no
slope). With a frozen slope `Function`, `k = alpha p / (h max(s, grad_floor))`
and the transition has a fixed width in arc length; see `mask_steepness`.
`surface_slope` builds that frozen field from the initial sea level, on the
annulus and on the sphere alike.

`h` is `FacetArea ** (1 / (dim - 1))`: the facet length in 2-D and the facet
side in 3-D. `CellDiameter` is not used because TSFC does not compile it on a
P2-curved mesh ("Cannot handle geometric quantity type").

## The softplus sheet

The masks enter the solver only through the load of water and ice. With sharp
masks that load is `max(rho_w SL, rho_i I)`, so the second part of this module
writes the load directly as a softplus of the flotation argument
`f = I - (rho_w / rho_i) SL`, with a fixed width `eps` in units of ice
thickness (`softplus_column`). The steepness `k`, the slope and the clamp do
not enter that form. `SeaLevel.sheet = "softplus"` selects it. The masks stay
the default until the comparison in
`NOTES/PLAN-softplus-load-2026-11-09.md` is done.
"""

from firedrake import (FacetArea, Function, FunctionSpace, SpatialCoordinate,
                       conditional, dot, exp, grad, ln, max_value, min_value,
                       sqrt, tanh)
from pyadjoint.tape import stop_annotating

#: Clamp of the tanh argument. Two bounds fix it.
#:
#: The lower bound is 19. `tanh` reaches exactly 1.0 in double precision at an
#: argument of 19, so a smaller clamp would leave the saturated masks inexact,
#: and the saturated states of the tests depend on them being exact.
#:
#: The upper bound comes from the second derivative of the mask, which the
#: Jacobian of the centre-of-mass frame column holds. UFL writes the derivative
#: of `tanh(z)` as `(2 cosh(z) / (1 + cosh(2 z)))^2`, so differentiating that
#: again carries `cosh(2 z)^2`. Where a kernel forms that square, it overflows
#: in double precision above `z = 177.8`, and the product of the infinite value
#: with the exact zero derivative of the clamped branch is a NaN that stops the
#: solve. Where a kernel divides by `cosh(2 z)` twice in sequence, the overflow
#: comes only above `z = 355.2`. Which of the two a kernel does is a TSFC
#: grouping decision that this module does not control, so the clamp respects
#: the stricter bound of 177.8.
#:
#: 177.8 and 19 are derived. What is measured is the solver: at `grad_floor`
#: 7e-5 on the 78 km shelf with a deformable Earth, the clamp at 350 fails with
#: `ValueError: array must not contain infs or NaNs` inside
#: `DtNMultiplierDenseSchurPC._solve`, and the clamp at 170 converges in 3
#: Newton steps (`NOTES/findings/FINDING-alpha-floor-mass.md`).
#:
#: 150 sits inside both bounds. A third derivative of the mask would carry
#: `cosh(2 z)^3`, which overflows above `z = 118.6`, so a second-order adjoint
#: would need a clamp below that. Nothing here takes one today.
SMOOTH_STEP_CLAMP = 150.0

#: The default factor of the steepness, `k = alpha p / (h s)`.
#:
#: 1.0 on the measurement of step S1
#: (`NOTES/findings/FINDING-alpha-floor-mass.md`): at the solver's own facet
#: quadrature degree of 9 with `p = 2`, the grid-frequency artefact in the
#: gradient stays below the 1.7e-7 resolution of the measure up to `alpha = 1`,
#: and the quadrature error of that gradient is 1.7e-8 there. A larger `alpha`
#: is a narrower mask step and a smaller mass error, so the limit is what sets
#: the value. Newton costs the same between `alpha` 0.25 and 4: three steps on
#: a deformable Earth and two on a stiff one.
#:
#: The value holds at the calibrated quadrature degree only. At the refusal
#: floor `q = 2 p` the same measurement gives 0.5, and the check in
#: `SelfGravitatingGIASolver` tests `q >= 2 p` without knowing `alpha`. The
#: calibrated degree in 3-D is not the 2-D value of 9, so read the limit again
#: there.
DEFAULT_ALPHA_MASK = 1.0

#: The default floor of a frozen slope, `k = alpha p / (h max(s, grad_floor))`.
#: A flat region has no shoreline, so the floor only has to keep `k` finite
#: there.
#:
#: 1e-4 on the measurement of step S2
#: (`NOTES/findings/FINDING-alpha-floor-mass.md`). Above the floor the mask
#: step is `h / (alpha p)` wide in arc length on any slope; below it the width
#: grows as `grad_floor / s`, and the error of the shift grows with it. Real
#: continental shelves have slopes of 5e-4 to 2e-3, which the earlier floor of
#: 1e-3 sat inside: at `s = 5e-4` it cost a factor 3.8 in that error. At 1e-4
#: every one of those shelves is above the floor, so the floor changes nothing
#: there and the width is `h / (alpha p)`. What the floor still sets is ground
#: flatter than 1e-4, where it keeps `k` finite.
#:
#: Measured in S2: the error falls with the floor with an exponent between 1.5
#: and 2.2 while the floor is at least twice the shelf slope, and 1e-4 beats
#: 1e-3 at every slope measured, by factors of 3.8, 64 and 11.
#:
#: A floor this low needs `SMOOTH_STEP_CLAMP` below 177.8. At the earlier clamp
#: of 350 an ocean 5.5 km deep on 78 km facets puts the mask argument at the
#: clamp, and the solve fails with a NaN.
DEFAULT_GRAD_FLOOR = 1e-4


def smooth_step(x, k):
    """The smooth Heaviside step `0.5 (1 + tanh(k x / 2))`.

    The tanh form equals the logistic function `1 / (1 + exp(-k x))`. It has
    more headroom before overflow in the derivative, and the argument is also
    clamped at `+-SMOOTH_STEP_CLAMP`, so the value is exactly 0 or 1 far from
    the switch and both the first and the second derivative are finite for any
    `k x`. The second derivative is the term that sets the clamp; see
    `SMOOTH_STEP_CLAMP`.

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
      alpha: the dimensionless factor; see `DEFAULT_ALPHA_MASK` for the value
        and the measurement that fixes it. At the solver's own facet quadrature
        degree of 9 with `p = 2` the limit is `alpha <= 1`, and it falls to 0.5
        at the refusal floor `q = 2 p`. The older rule `alpha <= 0.7 q / p` is
        about three times too weak.
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


def surface_slope(SL_init, degree=None):
    """The frozen surface slope `|grad_t SL_init|` of an initial sea level.

    `mask_steepness` needs this field to set the mask width in arc length
    instead of in sea level. Without it the transition of the mask is a fixed
    width in `SL`, so on a gentle continental shelf the transition spreads over
    many facets, and on a steep one it collapses inside a single facet. The
    measured shelf slopes of the Martinec et al. (2018) basins are 2.51e-4
    (B1) and 1.26e-3 (B2), which is a factor of five between two coastlines of
    the same benchmark.

    The slope is the *tangential* part of `grad SL_init`, because only the
    change of the sea level along the surface converts a width in sea level
    into a width in arc length. The tangential part is taken against the radial
    direction `rhat = X / |X|` and not against `FacetNormal`, for two reasons.
    An interpolation into a cell-wise space has no facet, so `FacetNormal` does
    not compile there. And on the annulus and on the sphere the outer surface
    is a level set of the radius, so the radial direction is the surface normal
    up to the geometry error of the P2-curved mesh. That error is O(h^2) in the
    direction and not O(h^3): a P2 facet places the surface to O(h^3), and the
    normal is a derivative of the position, which loses one power of `h`.

    The result is a `Function`, and it is built inside `stop_annotating()`. The
    slope must not reach the tape: `mask_steepness` divides by it, so a live
    slope would add a `1 / |grad SL|^2` term to the adjoint that is not
    physics. `NOTES/DESIGN-SEA-LEVEL.md` section 6 records this: the gradient
    of a sea-level run ignores how the mask steepness depends on the initial
    topography. `mask_steepness` refuses anything that is not a `Function`, and
    this helper is how a caller produces one.

    The target space is discontinuous, because `grad SL_init` is already
    discontinuous between cells and a continuous space would average the two
    sides. A discontinuous field is two-valued on a facet. That is harmless
    here: the mechanics mesh lies on one side of the surface Re only, so the
    sea-level measure always reads the mantle-side value.

    Only `grad`, `dot` and the coordinates appear, so the same code runs on the
    2-D annulus and on the 3-D sphere.

    Args:
      SL_init: the initial sea level as a `Function` on the mechanics mesh, in
        the length unit of the model. Its own mesh and degree set the defaults
        below. The default degree needs a single number from
        `ufl_element().degree()`, which a Lagrange element gives and a
        tensor-product element does not; a caller on an extruded mesh must pass
        `degree` instead.
      degree: the polynomial degree of the discontinuous space that holds the
        result. The default is one less than the degree of `SL_init`. On a
        straight-sided cell that is exactly the degree of `grad SL_init`. The
        meshes of this solver are P2-curved, where the inverse Jacobian makes
        the gradient rational, and the magnitude and the projection are not
        polynomial in any case. The space therefore adds an interpolation
        error, of the same order as the error that the gradient already has.

    Returns:
      A `Function` on the mesh of `SL_init`, in the discontinuous space of that
      degree: `DG` on a simplex and `DQ` on a quadrilateral or a hexahedron. It
      holds the surface slope, which is dimensionless: a length per unit
      length.
    """
    space = SL_init.function_space()
    mesh = space.mesh()
    if degree is None:
        # On a straight-sided cell `grad` of a degree-p field is a
        # degree-(p - 1) field, so this is the smallest space that carries the
        # gradient. `max` keeps a degree-0 input (a piecewise-constant
        # topography, whose gradient is zero) from asking for a space of
        # degree -1.
        degree = max(space.ufl_element().degree() - 1, 0)

    # The discontinuous family of the cell. Firedrake converts "DG" into "DQ"
    # on a quadrilateral or a hexahedron by itself, but it writes a UserWarning
    # every time it does, and the 2-D annulus of this solver is a quadrilateral
    # mesh. Naming the family removes that noise and changes no result.
    # `cellname` is a property of `ufl.Cell` in this UFL, not a method.
    cell = mesh.ufl_cell().cellname
    family = "DQ" if cell in ("quadrilateral", "hexahedron") else "DG"

    X = SpatialCoordinate(mesh)
    # The radial direction. No origin guard is needed: every mesh of this
    # solver is a shell or an annulus whose inner radius is far from zero.
    rhat = X / sqrt(dot(X, X))
    gradient = grad(SL_init)
    # Remove the radial part, so that only the change along the surface is
    # left. On a radial surface this is the surface gradient.
    tangential = gradient - dot(gradient, rhat) * rhat

    slope = Function(FunctionSpace(mesh, family, degree), name="surface_slope")
    # Off the tape: see the docstring. `stop_annotating` also covers the case
    # where a driver builds the slope inside an annotated block.
    with stop_annotating():
        slope.interpolate(sqrt(dot(tangential, tangential)))
    return slope


def ocean_function(SL, k):
    """The ocean function `C = H_k(SL)`: 1 where the sea level is positive."""
    return smooth_step(SL, k)


def grounded_ice_function(I, SL, k, rho_w, rho_i):
    """The grounded-ice function `B = 1 - H_k(I - (rho_w / rho_i) SL)`.

    Ice floats when its thickness is below `(rho_w / rho_i)` times the water
    depth `SL`. Then the argument is negative and `B = 1`: the ice counts as
    water. Ice thicker than that rests on the bed and `B = 0`. Open water with
    no ice gives a negative argument and `B = 1`. On land `SL < 0`, so the
    argument is positive with or without ice and `B = 0`. So `B` alone is 1
    exactly on the ocean of Martinec eq. 23, open water and floating ice, and
    0 on land and under grounded ice.

    Where there is no ice, `B = H_k((rho_w / rho_i) SL)` is a second smooth
    step centred on the same coastline as `C = H_k(SL)`, steeper by
    `rho_w / rho_i`. The product `B C` is therefore narrower than either step
    at every ice-free coast, by about `1 / k` in width. That costs almost no
    water in the sheet, because `SL` is zero at the coast, but an area
    computed as `int B C` is low by that strip; compute the ocean area as
    `int B` instead.

    Args:
      I: the ice thickness, in the units of `SL`.
      SL: the sea level.
      k: the steepness.
      rho_w, rho_i: the water and ice densities.
    """
    return 1.0 - smooth_step(I - (rho_w / rho_i) * SL, k)


# -- The softplus sheet -------------------------------------------------------
#
# With sharp masks and a non-negative ice thickness, the load of water and ice
# at one point is a maximum of two columns:
#
#     rho_w B C SL + rho_i (1 - B) I = max(rho_w SL, rho_i I)
#
# Open ocean gives `rho_w SL`, land without ice gives `max(negative, 0) = 0`,
# grounded ice on land or below sea level gives `rho_i I`, and floating ice
# gives the water column `rho_w SL` that it displaces. The maximum is
# continuous at the coastline (`SL = 0`) and at the grounding line (the
# flotation condition `rho_i I = rho_w SL`), with a kink only. The functions
# below replace the kink by a softplus of a fixed width in metres of ice:
#
#     sigma_total = rho_w SL + rho_i eps softplus((I - r SL) / eps),
#     r = rho_w / rho_i
#
# which tends to `rho_w SL + rho_i max(I - r SL, 0) = max(rho_w SL, rho_i I)`
# as `eps -> 0`. The derivatives are
#
#     d sigma / d I  = rho_i H((I - r SL) / eps)
#     d sigma / d SL = rho_w (1 - H((I - r SL) / eps))
#
# with `H` the logistic function, so both lie in `[0, rho]` and a sensitivity
# never has the wrong sign. No slope, facet size, `alpha` or clamp enters, and
# every derivative of every order is continuous, so the Hessian exists. Away
# from the coastline and the grounding line the error is about
# `rho_i eps exp(-|I - r SL| / eps)`, which is zero to double precision in
# deep water, on high land and under thick grounded ice.


def _signed_parts(x):
    """`max(x, 0)` and `min(x, 0)` as UFL conditionals on the one test `x > 0`.

    The two parts are written with `conditional` and not with
    `max_value`/`min_value`, because UFL differentiates a conditional branch by
    branch, and that is what puts the derivative of the softplus right at
    `x = 0`: the negative part carries the derivative 1 there, so the slope of
    the softplus at zero is `0.5` exactly. With `min_value` the derivative at
    the tie is 0 and the slope at `x = 0` would come out as 0.

    Args:
      x: a UFL expression.

    Returns:
      `(positive, negative)` with `positive + negative = x` and
      `positive - negative = |x|`.
    """
    positive = conditional(x > 0, x, 0.0)
    negative = conditional(x > 0, 0.0, x)
    return positive, negative


def softplus(x):
    r"""`log(1 + exp(x))`, written so that no term overflows for any `x`.

        softplus(x) = max(x, 0) + log(1 + exp(-|x|))

    `exp(-|x|)` is at most 1, so the expression and every derivative that UFL
    forms from it are finite at any argument. Arguments of order 1e4 are normal
    here: 3 km of ice over a width of 1 m. The first derivative is the logistic
    function, the second its derivative, and both are exact at `x = 0`
    (`_signed_parts`).

    Args:
      x: a UFL expression, dimensionless.

    Returns:
      A UFL expression, positive, equal to `x` for large positive `x` and to 0
      for large negative `x` in double precision.
    """
    positive, negative = _signed_parts(x)
    # `negative - positive = -|x|`, so the exponential is at most 1.
    return positive + ln(1.0 + exp(negative - positive))


def logistic(x):
    r"""The logistic function `1 / (1 + exp(-x))`, the derivative of `softplus`.

    Written with `exp(-|x|)` like `softplus`, so it is finite for any `x`:

        logistic(x) = 1 / (1 + e)   for x > 0
                      e / (1 + e)   for x <= 0,   e = exp(-|x|)

    Args:
      x: a UFL expression, dimensionless.

    Returns:
      A UFL expression with values in `[0, 1]`, one half at zero.
    """
    positive, negative = _signed_parts(x)
    e = exp(negative - positive)
    return conditional(x > 0, 1.0 / (1.0 + e), e / (1.0 + e))


def nonnegative_ice(I):
    """The ice thickness with negative values set to zero.

    The maximum identity of the sheet needs `I >= 0`: with `I < 0` on land,
    `max(rho_w SL, rho_i I)` gives the negative water column `rho_w SL`. A
    higher-order interpolation of an ice field can undershoot below zero near
    an ice margin, so the sheet clips it.

    The clip is `conditional(I < 0, 0, I)` and not `max_value(I, 0)`. Ice-free
    regions have `I = 0` exactly, and UFL gives `max_value` a zero derivative
    at the tie, so the derivative of the load with respect to new ice on
    ice-free land would vanish. The conditional keeps the derivative 1 at
    `I = 0`, which is the right one-sided derivative for ice that grows.

    The load is therefore not differentiable at `I = 0` in a direction that
    thins the ice: the thinned ice is negative and the clip holds it at zero.
    A Taylor test whose direction makes ice-free ground negative falls to
    rate 1 for that reason. Directions that change the ice only where there
    is ice, or that only add ice, are smooth.

    Args:
      I: the ice thickness, a UFL expression.

    Returns:
      A UFL expression, `I` where `I >= 0` and 0 elsewhere.
    """
    return conditional(I < 0, 0.0, I)


def flotation_argument(I, SL, rho_w, rho_i):
    """`f = I_+ - (rho_w / rho_i) SL`: positive under grounded ice and on land.

    The ice thickness above the thickness at which the ice floats, in units of
    ice thickness. `f > 0` on land (`SL < 0`) and under grounded ice, `f < 0`
    in open water and under floating ice. `I_+` is `nonnegative_ice(I)`.

    Args:
      I: the ice thickness.
      SL: the sea level, in the units of `I`.
      rho_w, rho_i: the water and ice densities.

    Returns:
      A UFL expression in the units of `I`.
    """
    return nonnegative_ice(I) - (rho_w / rho_i) * SL


def softplus_column(I, SL, eps, rho_w, rho_i):
    r"""The mass per unit area of water and ice at one state.

        sigma_total = rho_w SL + rho_i eps softplus(f / eps),
        f = I_+ - (rho_w / rho_i) SL

    which tends to `max(rho_w SL, rho_i I_+)` as `eps -> 0` (see the comment
    above `_signed_parts`). On land without ice it is `rho_w SL + rho_w (-SL)
    = 0` up to the rounding near the coastline.

    Args:
      I: the ice thickness.
      SL: the sea level, in the units of `I`.
      eps: the width of the rounding, in the units of `I`. A constant.
      rho_w, rho_i: the water and ice densities.

    Returns:
      A UFL expression, in density times the length unit of `SL`.
    """
    f = flotation_argument(I, SL, rho_w, rho_i)
    return rho_w * SL + rho_i * eps * softplus(f / eps)


def water_fraction(I, SL, eps, rho_w, rho_i):
    """`1 - H(f / eps)`: 1 over open water and floating ice, 0 on land and grounded ice.

    The derivative `d sigma_total / d SL` divided by `rho_w`, and the sharp
    limit of `B C` of the masks. Its integral over the surface is the area that
    takes up water when the sea level rises, the ocean area of the barystatic
    value. With `I = 0` it is the ocean function of an ice-free Earth,
    `H((rho_w / rho_i) SL / eps)`.

    Args:
      I: the ice thickness.
      SL: the sea level, in the units of `I`.
      eps: the width of the rounding, in the units of `I`.
      rho_w, rho_i: the water and ice densities.

    Returns:
      A UFL expression with values in `[0, 1]`.
    """
    return 1.0 - logistic(flotation_argument(I, SL, rho_w, rho_i) / eps)
