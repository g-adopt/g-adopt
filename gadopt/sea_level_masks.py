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
"""

from firedrake import (FacetArea, Function, FunctionSpace, SpatialCoordinate,
                       dot, grad, max_value, min_value, sqrt, tanh)
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
