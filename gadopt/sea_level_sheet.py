r"""The sea-level sheet: the load of water and ice on the surface.

With sharp masks, the load of water and ice at one point of the surface is a
maximum of two columns, for a non-negative ice thickness `I` and the sea level
`SL` (the height of the sea surface above the solid surface, positive in the
ocean):

    rho_w B C SL + rho_i (1 - B) I = max(rho_w SL, rho_i I)

with `C` the ocean function and `B` the grounded-ice function of Martinec et
al. (2018). Open ocean gives `rho_w SL`, land without ice gives
`max(negative, 0) = 0`, grounded ice on land or below sea level gives
`rho_i I`, and floating ice gives the water column `rho_w SL` that it
displaces. The maximum is continuous at the coastline (`SL = 0`) and at the
grounding line (the flotation condition `rho_i I = rho_w SL`), with a kink
only.

This module rounds the kink by a softplus of a fixed width `eps` in units of
ice thickness:

    S(I, SL) = rho_w SL + rho_i eps softplus(f / eps),   f = I_+ - r SL,
    r = rho_w / rho_i

which tends to `rho_w SL + rho_i max(I_+ - r SL, 0) = max(rho_w SL, rho_i I_+)`
as `eps -> 0`. `I_+` is the ice thickness clipped at zero. The derivatives are

    d S / d I  = rho_i H(f / eps),     d S / d SL = rho_w (1 - H(f / eps))

with `H` the logistic function, so both lie in `[0, rho]` and a sensitivity
never has the wrong sign. Every derivative of every order is continuous, so the
Hessian exists. Away from the coastline and the grounding line the rounding is
about `rho_i eps exp(-|f| / eps)`, which is zero in double precision in deep
water, on high land and under thick grounded ice, on any slope and any mesh.

The width sets one more thing. The second derivative of `S` is nonzero in a
band about `eps / (r s)` wide in arc length along a coast of slope `s`. The
residual and the first derivative are accurate whatever that band is, but a
Hessian, or the Jacobian of the centre-of-mass frame column (which carries the
second derivative), is sampled by the surface quadrature only where the band is
at least as wide as the spacing of the quadrature points.

`SelfGravitatingGIASolver` reads `S` through `SeaLevel.softplus_width` and
subtracts `S(I_init, SL_init)`, so that an unchanged state carries no load.
`water_fraction` and `ocean_fraction` are the diagnostic fields: the area that
takes up water when the sea rises, and the area where the sea level is
positive.
"""

from firedrake import conditional, exp, ln


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

    which tends to `max(rho_w SL, rho_i I_+)` as `eps -> 0` (module
    docstring). On land without ice it is `rho_w SL + rho_w (-SL) = 0` up to
    the rounding near the coastline.

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

    The derivative `d sigma_total / d SL` divided by `rho_w`, and the rounded
    form of the product `B C` of the ocean function and the grounded-ice
    function. Its integral over the surface is the area that takes up water
    when the sea level rises, the ocean area of the barystatic value. With
    `I = 0` it is `ocean_fraction`.

    Args:
      I: the ice thickness.
      SL: the sea level, in the units of `I`.
      eps: the width of the rounding, in the units of `I`.
      rho_w, rho_i: the water and ice densities.

    Returns:
      A UFL expression with values in `[0, 1]`.
    """
    return 1.0 - logistic(flotation_argument(I, SL, rho_w, rho_i) / eps)


def ocean_fraction(SL, eps, rho_w, rho_i):
    """`H(r SL / eps)`: 1 where the sea level is positive, 0 where it is negative.

    The rounded ocean function `C`, which counts the ocean under grounded ice
    as well: `water_fraction` without ice. The rounding is the one of the sheet
    at a coast without ice, a step `eps / r` wide in sea level. The fixed
    coastline of `SeaLevel(fixed_ocean=True)` uses it at the reference sea
    level, and the drivers write it as the ocean function of an output.

    Args:
      SL: the sea level, in the units of `eps`.
      eps: the width of the rounding, in units of ice thickness.
      rho_w, rho_i: the water and ice densities.

    Returns:
      A UFL expression with values in `[0, 1]`.
    """
    return logistic((rho_w / rho_i) * SL / eps)
