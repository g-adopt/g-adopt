r"""The two constants of the rotational closure: `C - A` and `k_s`.

What this file protects
-----------------------

The `Rotation` coupling of `SelfGravitatingGIASolver` closes the polar-motion
rows as

    [(C - A) - Q k_T(t)] m = dI_direct,

because the solver's own degree-2 response to the centrifugal perturbation is
`Q k_T(t) m` with `Q = a^5 Omega^2 / (3 G)` from MacCullagh's relation. `Q` is
an identity of the exterior degree-2 field and not a model parameter, so the
classical secular form `(C - A)(1 - k_T/k_s) m = dI_direct` is the *same*
equation only when

    C - A = Q k_s.

In the solver's non-dimensional variables (lengths by `L`, densities by
`rho_bar`, accelerations by `g_bar`, moments by `rho_bar L^5`) the moment scale
divides out and

    Qhat = 4 pi ahat^5 Omega_sq / (3 Lambda),   ahat = a / L.

`Rotation.check` is what stops a caller stating a `C - A` and a `k_s` that
disagree. These tests cover the three spellings it accepts (`C_minus_A` alone,
unchecked; `k_s` with `surface_radius`, which derives `C - A`; both, which
must agree), the tolerances of the two checks it makes, the keywords it
refuses, and the calibration of `Qhat` against the Spada M3-L70-V01
benchmark's own constants.

Where each check runs
---------------------

`Rotation.check` runs inside `SelfGravitatingGIASolver.__init__`, after the
coupling is bound to the solver's context. Its two parts depend on the
dimension:

* The radius checks (`k_s` without `surface_radius`, a radius in metres, a
  radius at the core-mantle boundary) and the polar moment `C` apply in every
  dimension. These tests build the 2-D annulus solver of `test_gia_gravity`
  (its fixture and its `build`), so they also cover the call from the
  constructor. A refusal costs one constructor up to the coupling stage and no
  solve.
* `C - A` closes the polar-wander rows `m_1`, `m_2`, which only a 3-D body
  has. A 2-D disc has the `m_3` row only, and the coupling does not read
  `C_minus_A` or derive it there. The tests of the `C - A` resolution
  therefore call `Rotation.check` on a context that holds a 3-D mesh, an
  icosahedral sphere of radius `RE` (the mechanics mesh's outer radius that
  `surface_radius` is checked against), and the self-gravity number of the
  approximation. This is the code path of the constructor without the 3-D
  forms, so nothing is assembled, compiled or solved.

Nothing here solves, and nothing here is marked `slow`. The radius check on
two ranks, where the mesh radius is a maximum over the ranks, is the entry
`rotation-radius-np2` of the `parallel` case in `tests/gia`, which uses
`build_rotation`, `closure` and `K_S_SPADA` of this file.
"""

import re
from types import SimpleNamespace

import gadopt  # noqa: F401  (imported before firedrake, as G-ADOPT requires)
import firedrake as fd
import numpy as np
import pytest

from gadopt.gia_couplings import Rotation
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.utility import scalar_value

from test_gia_gravity import (  # noqa: E402  (module-level helpers)
    LAMBDA,
    RC,
    RE,
    approximation,
    build,
    coupling,
    polar_moment,
)
# The annulus fixture, requested by name by every 2-D test below. Imported on
# its own line because pyflakes reports the whole `from` statement as unused
# when the only unused name is in it.
from test_gia_gravity import meshes  # noqa: E402,F401  (pytest fixture)


class refstate:
    """The reference state of the Spada et al. (2011) benchmark drivers.

    These are the scales of `selfgrav_common` of the benchmark drivers (now in
    `examples/common` of the giamip repository), copied here so that this
    calibration test does not depend on that repository. The numbers are the
    ones the benchmark prescribes.
    """

    #: Newton's constant, prescribed by the benchmark, m^3 kg^-1 s^-2.
    G_NEWTON = 6.6732e-11
    #: The Earth radius, m.
    A_EARTH = 6.371e6
    #: The length scale D = Re - Rc, m.
    D_SCALE = 2.891e6
    #: The density scale, the mean density of M3-L70-V01, kg m^-3.
    RHO_BAR = 5511.68
    #: The gravity scale, m s^-2.
    G_BAR = 9.81555
    #: The self-gravity number 4 pi G rho_bar D / g_bar (1.361324).
    LAMBDA = 4 * np.pi * G_NEWTON * RHO_BAR * D_SCALE / G_BAR
    #: C - A of the hydrostatic figure with the secular Love number K_S_SPADA,
    #: 2.6952e35 kg m^2, in units of rho_bar D^5 (0.24214001).
    C_MINUS_A = {"ks": 2.6952e35 / (RHO_BAR * D_SCALE**5)}
    #: The rotation rate squared, Omega = 7.292115e-5 rad/s, in units of
    #: g_bar / D (1.5661757e-03).
    OMEGA_SQ = 7.292115e-5**2 * D_SCALE / G_BAR


#: The secular (fluid-limit) tidal Love number of M3-L70-V01, from the TABOO
#: transfer function the Spada et al. (2011) benchmark used. It is the number
#: `selfgrav_common.C_MINUS_A`'s comment names as the origin of the consistent
#: `C - A = 2.6952e35 kg m^2`, and it is dimensionless.
K_S_SPADA = 0.96672389

#: `Qhat` on the *test* annulus, whose `Lambda` and radius are not the
#: benchmark's. Computed here from the closed form so that the tests compare
#: the coupling against an independent evaluation of
#: `Qhat = 4 pi ahat^5 Omega_sq / (3 Lambda)`, not against itself.
Q_ANNULUS = 4 * np.pi * RE ** 5 * OMEGA_SQ_EARTH / (3 * LAMBDA)

#: Floating-point tolerance for "the coupling computed the same closed form I
#: did". Both sides are a handful of multiplications and divisions of the same
#: double-precision inputs, so the difference is a few units in the last place;
#: 1e-13 is that with three decades of margin, and it is tight enough that any
#: change to the *formula* (a factor of 4 pi, a power of the radius) fails.
FORMULA_RTOL = 1e-13


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


@pytest.fixture(scope="module")
def sphere():
    """A 3-D surface mesh of outer radius `RE`, for the 3-D `C - A` resolution.

    The icosahedron (refinement 0) has its twelve vertices on the sphere of
    radius `RE`, so `Rotation.check` measures the same outer radius as on the
    annulus. Only the geometric dimension and the vertex radius are read.
    """
    return fd.IcosahedralSphereMesh(RE, refinement_level=0)


def check_in_3d(sphere, *, lambda_value=LAMBDA, **rotation_kwargs):
    """Bind a `Rotation` to a 3-D context and run its `check`.

    The context carries the two things the check reads: the mechanics mesh
    (here `sphere`) and `Lambda`, taken from the approximation exactly as
    `SelfGravitatingGIASolver` takes it (`approximation.self_gravity_number`).
    The coupling is bound with no solver, because the check reads no solver
    state.

    Args:
      sphere: the 3-D mesh fixture.
      lambda_value: the self-gravity number of the approximation.
      **rotation_kwargs: keywords of `Rotation`. `C` defaults to 1, a value no
        test here reads; `Omega_sq` defaults to `OMEGA_SQ_EARTH`.

    Returns:
      The checked `Rotation`, with `closure_constants` set.
    """
    rotation_kwargs.setdefault("C", 1.0)
    rotation_kwargs.setdefault("Omega_sq", OMEGA_SQ_EARTH)
    rotation = Rotation(**rotation_kwargs)
    context = SimpleNamespace(
        mesh=sphere,
        Lambda=approximation(self_gravity_number=lambda_value).self_gravity_number)
    rotation.bind(None, context)
    rotation.check([rotation])
    return rotation


def closure(rotation, i):
    """The closure constant `K_i` of row `i` as a float (UFL on the `k_s` route)."""
    return scalar_value(rotation.closure_constants[i])


def numbers_in(message):
    """Every decimal number in a string, as floats.

    The refusal messages are asserted on by *value* and not by substring: a
    repr like `0.9433377228777087` does not contain the substring `0.94334`,
    so a substring test for the implied `k_s` would fail on a correct message.
    """
    pattern = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?"
    return [float(m) for m in re.findall(pattern, message)]


@pytest.fixture(scope="module")
def ks_rotation(sphere):
    """One coupling checked the consistent way: `k_s` alone, with `surface_radius`.

    Module-scoped because several tests read it and none of them changes it;
    the one test that runs `check` again (idempotence) asserts that the second
    call changes nothing.
    """
    return check_in_3d(sphere, k_s=K_S_SPADA, surface_radius=RE)


# -- Case 1: the identity, and what `k_s` alone derives ----------------------


def test_tidal_inertia_factor_is_the_macullagh_identity(sphere):
    """`Qhat = 4 pi ahat^5 Omega_sq / (3 Lambda)`, computed independently.

    Protects the non-dimensionalisation of `Q = a^5 Omega^2 / (3 G)`: with
    `a = ahat L`, `Omega^2 = Omega_sq g_bar / L` and
    `G = Lambda g_bar / (4 pi rho_bar L)`, the moment scale `rho_bar L^5`
    divides out and leaves the closed form above. A dropped `4 pi`, a wrong
    power of the radius or an inverted `Lambda` all fail here.

    The coupling has no separate `Q`; with `k_s = 1` its derived `C - A` is
    `Qhat` itself. Dimensionless quantity of order 0.3 on this annulus.
    """
    rotation = check_in_3d(sphere, k_s=1.0, surface_radius=RE)
    assert np.isclose(closure(rotation, 0), Q_ANNULUS,
                      rtol=FORMULA_RTOL, atol=0.0)


def test_k_s_alone_derives_the_consistent_C_minus_A(ks_rotation):
    """`k_s` with `surface_radius` fills in `C - A = Qhat k_s` in both rows.

    This is the route that cannot be inconsistent: the caller states the
    fluid-limit tidal Love number, and the dynamical ellipticity the closure
    needs is computed from the same `Q` the solver's own feedback carries.
    `closure_constants` is what `Rotation.residual` multiplies `m_i` by, so
    this is the value the polar-motion rows are built with.
    """
    for i in (0, 1):
        assert np.isclose(closure(ks_rotation, i), Q_ANNULUS * K_S_SPADA,
                          rtol=FORMULA_RTOL, atol=0.0)


def test_k_s_is_kept_in_the_dictionary(ks_rotation):
    """The coupling records what the caller asked for.

    `k_s` stays on the coupling alongside the derived closure constant, which
    is what makes the check idempotent (a second pass recomputes the same
    number) and what lets a later reader see which of the two spellings the
    caller used. No `C_minus_A` was given, and none is recorded as given.
    """
    assert ks_rotation.k_s == K_S_SPADA
    assert ks_rotation.C_minus_A_given is None


# -- Case 2: calibration against the benchmark's own constants ---------------


def test_Q_k_s_reproduces_the_benchmark_C_minus_A(sphere):
    r"""`Qhat k_s = 0.24214` against the driver's `C_MINUS_A["ks"]`.

    The one test that ties the coupling's `Q` to physical constants. With the
    benchmark's own `Lambda = 4 pi G rho_bar D / g_bar`, its
    `Omega_sq = Omega^2 D / g_bar` and `ahat = a / D = 6371/2891`, the identity
    gives `Qhat k_s = 0.24214062` for `k_s = 0.96672389`, against the driver's
    `C - A = 2.6952e35 kg m^2 / (rho_bar D^5) = 0.24214000`.

    Tolerance 1e-5 relative: the two agree to 2.5e-6, which is the rounding of
    the driver's five-digit `2.6952e35`, and 1e-5 is that with a factor of four
    of margin. It is ten times *tighter* than
    `Rotation.CONSISTENCY_RTOL = 1e-4`, so this test is what would catch an
    edit to the non-dimensionalisation that the consistency check itself still
    tolerates.

    The sphere of radius `RE` stands in for the benchmark's mesh: the mesh
    enters only through the 2 percent radius cross-check, and `RE = 2.2037`
    is 3.6e-6 from the benchmark's `ahat = 2.2037357`.
    """
    ahat = refstate.A_EARTH / refstate.D_SCALE
    rotation = check_in_3d(
        sphere, k_s=K_S_SPADA, lambda_value=refstate.LAMBDA,
        Omega_sq=refstate.OMEGA_SQ, surface_radius=ahat)

    assert np.isclose(closure(rotation, 0), refstate.C_MINUS_A["ks"],
                      rtol=1e-5, atol=0.0)


# -- Case 3: the tolerance boundary ------------------------------------------


def test_a_pair_just_inside_the_tolerance_is_accepted(sphere):
    """A `C - A` off by 0.99e-4 from `Qhat k_s` passes the check.

    The accepting side of `Rotation.CONSISTENCY_RTOL = 1e-4`. The tolerance
    must admit the four-to-five-digit rounding a caller does when it writes a
    constant down; asserting at 0.99 of it, and not at some value far inside,
    is what makes this a test of the boundary rather than of the idea.

    The given value is recorded as given. The rows are closed with the derived
    `Qhat k_s`, which is UFL in `Lambda` and `Omega_sq`, so that a control on
    either reaches the closure on the tape; the two differ by less than the
    tolerance.
    """
    given = Q_ANNULUS * K_S_SPADA * (1 + 0.99e-4)
    rotation = check_in_3d(sphere, k_s=K_S_SPADA, C_minus_A=given,
                           surface_radius=RE)

    assert rotation.C_minus_A_given == given
    assert np.isclose(closure(rotation, 0), Q_ANNULUS * K_S_SPADA,
                      rtol=FORMULA_RTOL, atol=0.0)


def test_a_pair_just_outside_the_tolerance_is_refused(sphere):
    """A `C - A` off by 1.01e-4 from `Qhat k_s` raises.

    The refusing side of the same boundary. The tolerance is set by how far the
    error travels: eliminating the displacement gives
    `m = (1 + k_L) dI_direct / [(C - A) - Q k_T(t)]`, so a fractional error
    `eps` in `C - A` moves `m` by `eps (C - A) / [(C - A) - Q k_T(t)]`, an
    amplification of about 100 near the fluid limit. 1e-4 therefore bounds the
    polar-motion error at about 1 percent.
    """
    given = Q_ANNULUS * K_S_SPADA * (1 + 1.01e-4)
    with pytest.raises(ValueError, match="disagree"):
        check_in_3d(sphere, k_s=K_S_SPADA, C_minus_A=given, surface_radius=RE)


# -- Case 4: the benchmark's own inconsistent pair ---------------------------

#: The Spada benchmark's two values of `C - A`: `2.63e35 kg m^2` prescribed in
#: its load excitation against the `2.6952e35 kg m^2` implied by the
#: `k_s = 0.96672389` of its transfer function. Their ratio is the
#: inconsistency, 2.42 percent, and it is scale free, so it applies on the test
#: annulus exactly as it does on the benchmark.
BENCHMARK_RATIO = 2.63 / 2.6952


def test_the_benchmark_inconsistent_pair_is_refused(sphere):
    """The 2.42 percent inconsistency of Spada et al. (2011) cannot be stated.

    This is the failure the check exists for: the pair is 240 times outside
    `Rotation.CONSISTENCY_RTOL`, and it shows as 3.6 percent in `|m|` at
    `t = 0` and more at later epochs, so accepting it would put an
    unattributable error into every polar-motion comparison.
    """
    given = Q_ANNULUS * K_S_SPADA * BENCHMARK_RATIO
    with pytest.raises(ValueError, match="disagree"):
        check_in_3d(sphere, k_s=K_S_SPADA, C_minus_A=given, surface_radius=RE)


def test_the_refusal_message_carries_both_values_and_the_implied_k_s(sphere):
    """The message must let the caller decide which of the two it meant.

    Which value is wrong is not knowable inside the solver, so the message has
    to report the given `C - A`, the given `k_s`, the `C - A = Qhat k_s` they
    imply and the `k_s = (C - A) / Qhat` the given ellipticity implies. On the
    benchmark's pair that last number is 0.94334, the value
    `selfgrav_common.C_MINUS_A`'s comment names for the prescribed
    `2.63e35 kg m^2`, and reproducing it here is an independent check of the
    arithmetic as well as of the message.

    The numbers are compared by value and not by substring, for the reason in
    `numbers_in`. The implied `k_s` is checked to 1e-5 absolute, which is the
    five-digit form quoted in `selfgrav_common.py`.
    """
    given = Q_ANNULUS * K_S_SPADA * BENCHMARK_RATIO
    with pytest.raises(ValueError) as excinfo:
        check_in_3d(sphere, k_s=K_S_SPADA, C_minus_A=given, surface_radius=RE)
    reported = numbers_in(str(excinfo.value))

    def present(target, tol):
        return any(abs(value - target) <= tol for value in reported)

    # The given C - A and the given k_s, to the precision a caller would use to
    # recognise its own input.
    assert present(given, 1e-9 * given)
    assert present(K_S_SPADA, 1e-9)
    # The consistent C - A the given k_s implies.
    assert present(Q_ANNULUS * K_S_SPADA, 1e-9 * Q_ANNULUS)
    # The k_s the given C - A implies: 0.94334, the benchmark's prescribed
    # value expressed as a Love number.
    assert present(0.94334, 1e-5)


# -- Case 5: `k_s` needs a radius --------------------------------------------


def test_k_s_without_surface_radius_is_refused(meshes):  # noqa: F811
    """`Q ~ ahat^5`, and `ahat` is the one factor the solver cannot infer.

    `Lambda` and `Omega_sq` the solver already holds; the free-surface radius
    is not stated by either mesh, because the parent runs past the surface into
    the DtN stand-off buffer and the mechanics submesh stops at the surface
    only by convention. Guessing it would make `C - A` wrong by the fifth
    power of the guess, so the check refuses instead, in every dimension.
    """
    with pytest.raises(ValueError, match="surface_radius"):
        build_rotation(meshes, k_s=K_S_SPADA)


# -- Case 6: the radius cross-check ------------------------------------------


@pytest.mark.parametrize(
    "radius, label",
    [(6.371e6, "metres"),           # a dimensional radius, 2.9e6 times too big
     (RC, "core-mantle boundary")])  # 1.2037 against 2.2037, 45 percent short
def test_a_surface_radius_that_is_not_the_mesh_radius_is_refused(meshes, radius, label):  # noqa: F811
    """A radius in metres, or the CMB radius, must not reach `Q`.

    `Rotation.SURFACE_RADIUS_RTOL = 2e-2` is a check on the *meaning* of the
    argument, not on the mesh: it must catch these two mistakes, and it must
    not grade a coarse annulus whose straight facets already cost 4e-4 of
    radius. Both cases here are refused by many orders of magnitude, which is
    the point - the check is a type check in disguise.
    """
    with pytest.raises(ValueError, match="outer radius"):
        build_rotation(meshes, k_s=K_S_SPADA, surface_radius=radius)


def test_a_surface_radius_inside_two_percent_is_accepted(sphere):
    """A radius 1.9 percent off the mesh still passes, and `Q` follows it.

    The accepting side of `Rotation.SURFACE_RADIUS_RTOL = 2e-2`, and the
    statement that the mesh check does *not* bound the accuracy of `Q`: the
    derived `C - A` is then about 10 percent wrong, because `Q ~ ahat^5` and
    `1.019^5 = 1.099`. The caller's `surface_radius`, not the mesh, is what
    sets the accuracy of the `k_s` route, and this test records that as
    intended behaviour rather than as a bound anybody can rely on.
    """
    ahat = RE * 1.019
    rotation = check_in_3d(sphere, k_s=K_S_SPADA, surface_radius=ahat)

    expected = 4 * np.pi * ahat ** 5 * OMEGA_SQ_EARTH / (3 * LAMBDA) * K_S_SPADA
    assert np.isclose(closure(rotation, 0), expected,
                      rtol=FORMULA_RTOL, atol=0.0)


# -- Case 7: the keyword whitelist -------------------------------------------


@pytest.mark.parametrize("key", ["ks", "kS", "k_S", "C_minusA"])
def test_an_unknown_rotation_moments_key_is_refused(key):
    """A misspelt keyword must not be ignored.

    Ignoring it would restore in silence exactly the unchecked `C - A` this
    machinery exists to remove: a caller who writes `ks` believes it has asked
    for the consistent route and gets no check at all. The four spellings here
    are the ones a hand or an editor produces. `Rotation` takes its constants
    as keyword-only arguments, so Python refuses every other name when the
    coupling is created, before any mesh is involved.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Rotation(C=1.0, Omega_sq=OMEGA_SQ_EARTH, surface_radius=RE,
                 **{key: K_S_SPADA})


# -- Case 8: the existing spelling is untouched ------------------------------


def test_C_minus_A_alone_passes_through_unchanged(sphere):
    """The `C_minus_A` spelling is taken as given, unchecked.

    No `k_s` is given, so the coupling has no second opinion: `C - A` reaches
    the closure exactly as written, and the toy values the structural tests
    use (`C_minus_A = 0.1 C`, `3.2737e-3 C`), which are the ellipticity of no
    real figure, must keep working. This is the test that fails if the
    consistency check is ever made unconditional.
    """
    rotation = check_in_3d(sphere, C_minus_A=0.1)

    assert closure(rotation, 0) == 0.1
    assert closure(rotation, 1) == 0.1
    # Nothing is invented on this route: no `k_s` appears, and no radius is
    # required or consulted.
    assert rotation.k_s is None
    assert rotation.surface_radius is None


def test_C_alone_still_builds_and_keeps_its_polar_moment(meshes):  # noqa: F811
    """`C` alone, the 2-D configuration of most tests, is untouched.

    `C` closes the `m_3` row and no Love number constrains it, so the check
    must leave it alone and must not demand a `C - A` that a 2-D disc does not
    need.
    """
    expected = polar_moment(meshes)

    _, rotation = build_rotation(meshes)

    # Assembled twice from the same form on the same mesh, so the two agree to
    # round-off; 1e-13 is that with margin.
    assert np.isclose(closure(rotation, 2), expected,
                      rtol=FORMULA_RTOL, atol=0.0)
    # The disc has no polar-wander rows, so no `C - A` slot exists.
    assert set(rotation.closure_constants) == {2}


# -- Case 9: idempotence -----------------------------------------------------


def test_resolving_twice_gives_the_same_C_minus_A(ks_rotation):
    """A second check must not move the derived constant.

    `k_s` is kept on the coupling and not consumed, so the second pass derives
    `Qhat k_s` again from the same inputs. It must agree exactly (relative
    difference 0); a check that multiplied by `k_s` again, or that consumed
    `k_s` and then found nothing to derive, would fail here.
    """
    before = closure(ks_rotation, 0)

    ks_rotation.check([ks_rotation])

    assert closure(ks_rotation, 0) == before
    assert ks_rotation.k_s == K_S_SPADA
