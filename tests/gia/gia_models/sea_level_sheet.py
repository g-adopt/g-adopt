"""Helpers of `tests/unit/test_sea_level_sheet.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_sea_level_sheet.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
from gia_models.sea_level import EPS, PHASE, RHO_I, RHO_W, polar_angle


R = RHO_W / RHO_I


def width(eps=EPS):
    """`sea_level_overrides` that set the width of the sheet."""
    return dict(softplus_width=eps)


def marine_fields(sub):
    """A marine ice sheet with a grounding line that retreats when it thins.

    A uniform ocean of depth 1 (in units of D) with ice everywhere. The ice is
    `r (1 + 0.6 cos(phi - PHASE))` thick at the reference state, so it is
    grounded where `cos(phi - PHASE) > 0` and floats elsewhere. The current
    ice is 0.9 of that, so the flotation line moves from `cos = 0` to
    `cos = 0.185`, about 10.7 degrees on each side, which is one facet of the
    32-cell annulus. The ice is CG2 and positive everywhere (at least
    `0.4 r`), so the clip does not act.

    Returns:
      `SeaLevel` keywords, as `surface_fields`.
    """
    V = fd.FunctionSpace(sub, "CG", 2)
    phi = polar_angle(sub)
    zero = fd.Function(V)
    SL_init = fd.Function(V).assign(1.0)
    I_init = fd.Function(V).interpolate(R * (1.0 + 0.6 * fd.cos(phi - PHASE)))
    I = fd.Function(V).interpolate(0.9 * I_init)
    return dict(SL_init=SL_init, I=I, I_init=I_init, N_init=zero.copy(True),
                ur_init=zero.copy(True))
