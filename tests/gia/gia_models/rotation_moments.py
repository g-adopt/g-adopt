"""Helpers of `tests/unit/test_gia_rotation_moments.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gia_rotation_moments.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

from gadopt.gia_couplings import Rotation
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.utility import scalar_value
from gia_models.annulus import LAMBDA, build, coupling, polar_moment


#: The secular (fluid-limit) tidal Love number of M3-L70-V01, from the TABOO
#: transfer function the Spada et al. (2011) benchmark used. It is the number
#: `selfgrav_common.C_MINUS_A`'s comment names as the origin of the consistent
#: `C - A = 2.6952e35 kg m^2`, and it is dimensionless.
K_S_SPADA = 0.96672389


def build_rotation(meshes, *, lambda_value=LAMBDA, **rotation_kwargs):  # noqa: F811  (the fixture's value, passed in)
    """Build the 2-D annulus solver with one `Rotation` coupling.

    `test_gia_gravity.build` with its own rotation switched off and this
    `Rotation` passed as the only extra coupling: the same annulus, the same
    rigid core, the same `cos 2 phi` `SurfaceLoad` on Re, and the assembled
    polar moment `C` unless the caller gives one.

    Args:
      meshes: the `(parent, sub)` pair of the module fixture.
      lambda_value: the self-gravity number of the approximation, which the
        solver reads as `Lambda`.
      **rotation_kwargs: keywords of `Rotation`. `C` defaults to the mantle's
        polar moment and `Omega_sq` to `OMEGA_SQ_EARTH`.

    Returns:
      `(solver, rotation)`. Nothing is solved.
    """
    rotation_kwargs.setdefault("C", polar_moment(meshes))
    rotation_kwargs.setdefault("Omega_sq", OMEGA_SQ_EARTH)
    rotation = Rotation(**rotation_kwargs)
    solver, _, _ = build(
        meshes, rotation=False, couplings=[rotation],
        approximation_kwargs={"self_gravity_number": lambda_value})
    assert coupling(solver, Rotation) is rotation
    return solver, rotation


def closure(rotation, i):
    """The closure constant `K_i` of row `i` as a float (UFL on the `k_s` route)."""
    return scalar_value(rotation.closure_constants[i])
