r"""The two constants of the rotational closure: `C - A` and `k_s`.

What this file protects
-----------------------

`SelfGravitatingGIASolver` closes the polar-motion rows as

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

`SelfGravitatingGIASolver._resolve_rotation_moments` is what stops a caller
stating a `C - A` and a `k_s` that disagree. These tests cover the three
spellings it accepts (`C_minus_A` alone, unchecked; `k_s` with
`surface_radius`, which derives `C - A`; both, which must agree), the
tolerances of the two checks it makes, the keys it refuses, and the calibration
of `Qhat` against the Spada M3-L70-V01 benchmark's own constants.

Why a separate file, and why a mesh at all
------------------------------------------

The tests live here and not in `test_gia_gravity.py` because that file is about
the *structure* of the mixed space and the forms built on it, and it already
runs for about fifteen minutes; these tests are about the validation of two
scalar inputs and they are cheap. They import that file's annulus fixture and
its helpers (`approximation`, `gravity_bcs`, `mechanics_bcs`, `LAMBDA`, `RE`,
`RC`) instead of building a second mesh, as `test_gia_nested_condensation.py`
and `test_gia_condensed_block0.py` already do.

A mesh is still needed, because every path here runs inside
`SelfGravitatingGIASolver.__init__` and the constructor needs a space. The
distinction that matters for the run time is a different one:

* The refusals (`k_s` without `surface_radius`, a radius in metres, a radius at
  the core-mantle boundary, an unknown key, a pair outside the tolerance) raise
  from `_resolve_rotation_moments`, which `__init__` calls before
  `set_measures`, before the base-class residual and before `check_geometry`.
  They cost one space construction and no assembly: about 0.02 s each.
* The acceptances (`k_s` alone, a pair inside the tolerance, `C_minus_A` alone,
  the calibration case) run the whole constructor, which builds the residual
  and the solver but solves nothing: about 0.4 s each once the mesh is cached.

Nothing here solves, and nothing here is marked `slow`. The 2-rank test at the
end is a module-level function, which `mpi-pytest` requires: the plugin drops
the class from a marked node id, so a parallel test inside a class never runs
(see the comment at `test_gravity_solver.py:622`).
"""

import re
import sys
from pathlib import Path

import firedrake as fd
import numpy as np
import pytest

from gadopt.gia_gravity import (
    OMEGA_SQ_EARTH,
    SelfGravitatingGIASolver,
    self_gravitating_gia_space,
)

from test_gia_gravity import (  # noqa: E402  (module-level helpers)
    LAMBDA,
    RC,
    RE,
    approximation,
    gravity_bcs,
    mechanics_bcs,
)
# The annulus fixture, requested by name by every test below. Imported on its
# own line because pyflakes reports the whole `from` statement as unused when
# the only unused name is in it.
from test_gia_gravity import meshes  # noqa: E402,F401  (pytest fixture)

# The benchmark driver's reference state, for the one calibration test. It is a
# demo module, so the demo directory goes on the path the same way
# `test_gia_gravity.meshes` puts `demos/gravity` there for the mesh generator.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos"
                       / "glacial_isostatic_adjustment" / "3d_spada_selfgrav"))
import reference_state as refstate  # noqa: E402

#: The secular (fluid-limit) tidal Love number of M3-L70-V01, from the TABOO
#: transfer function the Spada et al. (2011) benchmark used. It is the number
#: `reference_state.C_MINUS_A`'s comment names as the origin of the consistent
#: `C - A = 2.6952e35 kg m^2`, and it is dimensionless.
K_S_SPADA = 0.96672389

#: `Qhat k_s` on the *test* annulus, whose `Lambda` and radius are not the
#: benchmark's. Computed here from the closed form so that the tests compare
#: the solver against an independent evaluation of
#: `Qhat = 4 pi ahat^5 Omega_sq / (3 Lambda)`, not against itself.
Q_ANNULUS = 4 * np.pi * RE ** 5 * OMEGA_SQ_EARTH / (3 * LAMBDA)

#: Floating-point tolerance for "the solver computed the same closed form I
#: did". Both sides are a handful of multiplications and divisions of the same
#: double-precision inputs, so the difference is a few units in the last place;
#: 1e-13 is that with three decades of margin, and it is tight enough that any
#: change to the *formula* (a factor of 4 pi, a power of the radius) fails.
FORMULA_RTOL = 1e-13


def build_rotation(meshes, rotation_moments, *, lambda_value=LAMBDA, **kwargs):  # noqa: F811  (the fixture's value, passed in)
    """Build the annulus self-gravity solver with a given `rotation_moments`.

    A trimmed copy of `test_gia_gravity.build`: the same annulus, the same
    boundary conditions and the same assembled polar moment `C`, with
    `rotation_moments` and the solver keywords left to the caller, because the
    inputs under test are exactly those.

    Args:
      meshes: the `(parent, sub)` pair of the module fixture.
      rotation_moments: entries added to `{"C": <assembled>}`. A key given here
        overrides the assembled `C`.
      lambda_value: the self-gravity number, given to both the space factory
        and the approximation because `_check_self_gravity_number` requires
        them to agree. Only the calibration test moves it off `LAMBDA`.
      **kwargs: forwarded to `SelfGravitatingGIASolver`, which is how
        `surface_radius` and `Omega_sq` reach it.

    Returns:
      The constructed `SelfGravitatingGIASolver`. Nothing is solved.
    """
    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
        self_gravity_number=lambda_value)
    z = fd.Function(Z)
    # The disc's polar second moment `int rho_0 r^2 dV` with `rho_0 = 1`. It
    # closes the `m_3` row and no Love number constrains it, so every case here
    # carries it unchanged.
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    moments = {"C": fd.assemble(fd.dot(Xm, Xm) * dx_m)}
    moments.update(rotation_moments)
    return SelfGravitatingGIASolver(
        z, approximation(self_gravity_number=lambda_value), layout=layout,
        dt=1.0, bcs=mechanics_bcs(sub), rotation_moments=moments, **kwargs)


def numbers_in(message):
    """Every decimal number in a string, as floats.

    The refusal messages are asserted on by *value* and not by substring: a
    repr like `0.9433377228777087` does not contain the substring `0.94334`,
    so a substring test for the implied `k_s` would fail on a correct message.
    """
    pattern = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?"
    return [float(m) for m in re.findall(pattern, message)]


@pytest.fixture(scope="module")
def ks_solver(meshes):  # noqa: F811  (the imported fixture, used by name)
    """One solver built the consistent way: `k_s` alone, with `surface_radius`.

    Module-scoped because four tests read it and none of them mutates the
    solver; the one test that does call `_resolve_rotation_moments` again
    (idempotence) is asserting that the call changes nothing.
    """
    return build_rotation(meshes, {"k_s": K_S_SPADA}, surface_radius=RE)


# -- Case 1: the identity, and what `k_s` alone derives ----------------------


def test_tidal_inertia_factor_is_the_macullagh_identity(ks_solver):
    """`Qhat = 4 pi ahat^5 Omega_sq / (3 Lambda)`, computed independently.

    Protects the non-dimensionalisation of `Q = a^5 Omega^2 / (3 G)`: with
    `a = ahat L`, `Omega^2 = Omega_sq g_bar / L` and
    `G = Lambda g_bar / (4 pi rho_bar L)`, the moment scale `rho_bar L^5`
    divides out and leaves the closed form above. A dropped `4 pi`, a wrong
    power of the radius or an inverted `Lambda` all fail here.

    Dimensionless quantity of order 0.3 on this annulus.
    """
    assert np.isclose(ks_solver.tidal_inertia_factor, Q_ANNULUS,
                      rtol=FORMULA_RTOL, atol=0.0)


def test_k_s_alone_derives_the_consistent_C_minus_A(ks_solver):
    """`k_s` with `surface_radius` fills in `C - A = Qhat k_s`.

    This is the route that cannot be inconsistent: the caller states the
    fluid-limit tidal Love number, and the dynamical ellipticity the closure
    needs is computed from the same `Q` the solver's own feedback carries.
    """
    assert np.isclose(ks_solver.rotation_moments["C_minus_A"],
                      Q_ANNULUS * K_S_SPADA, rtol=FORMULA_RTOL, atol=0.0)


def test_the_derived_C_minus_A_reaches_the_rotation_closure(ks_solver):
    """The derived value is what the polar-motion rows are built with.

    Reads `_closure_constant(0)`, which is private, because it is the only path
    from `rotation_moments` into the assembled rotation row: a resolution that
    wrote the dictionary but left the form on a stale constant would pass the
    test above and still be wrong.
    """
    assert np.isclose(float(ks_solver._closure_constant(0)),
                      Q_ANNULUS * K_S_SPADA, rtol=FORMULA_RTOL, atol=0.0)


def test_k_s_is_kept_in_the_dictionary(ks_solver):
    """The resolution records what the caller asked for.

    `k_s` stays alongside the derived `C_minus_A`, which is what makes the
    resolution idempotent (a second pass recomputes the same number and
    compares it against itself) and what lets a later reader see which of the
    two spellings the caller used.
    """
    assert ks_solver.rotation_moments["k_s"] == K_S_SPADA


# -- Case 2: calibration against the benchmark's own constants ---------------


def test_Q_k_s_reproduces_the_benchmark_C_minus_A(meshes):  # noqa: F811
    r"""`Qhat k_s = 0.24214` against the driver's `C_MINUS_A["ks"]`.

    The one test that ties the solver's `Q` to physical constants. With the
    benchmark's own `Lambda = 4 pi G rho_bar D / g_bar`, its
    `Omega_sq = Omega^2 D / g_bar` and `ahat = a / D = 6371/2891`, the identity
    gives `Qhat k_s = 0.24214062` for `k_s = 0.96672389`, against the driver's
    `C - A = 2.6952e35 kg m^2 / (rho_bar D^5) = 0.24214000`.

    Tolerance 1e-5 relative: the two agree to 2.5e-6, which is the rounding of
    the driver's five-digit `2.6952e35`, and 1e-5 is that with a factor of four
    of margin. It is ten times *tighter* than
    `ROTATION_CONSISTENCY_RTOL = 1e-4`, so this test is what would catch an
    edit to the non-dimensionalisation that the consistency check itself still
    tolerates.

    The annulus mesh is reused with the benchmark's `Lambda`: the mesh enters
    only through the 2 percent radius cross-check, and its outer radius
    `RE = 2.2037` is 3.6e-6 from the benchmark's `ahat = 2.2037357`.
    """
    ahat = refstate.A_EARTH / refstate.D_SCALE
    solver = build_rotation(
        meshes, {"k_s": K_S_SPADA}, lambda_value=refstate.LAMBDA,
        Omega_sq=refstate.OMEGA_SQ, surface_radius=ahat)

    assert np.isclose(solver.rotation_moments["C_minus_A"],
                      refstate.C_MINUS_A["ks"], rtol=1e-5, atol=0.0)


# -- Case 3: the tolerance boundary ------------------------------------------


def test_a_pair_just_inside_the_tolerance_is_accepted(meshes):  # noqa: F811
    """A `C - A` off by 0.99e-4 from `Qhat k_s` builds a solver.

    The accepting side of `ROTATION_CONSISTENCY_RTOL = 1e-4`. The tolerance
    must admit the four-to-five-digit rounding a caller does when it writes a
    constant down; asserting at 0.99 of it, and not at some value far inside,
    is what makes this a test of the boundary rather than of the idea.
    """
    given = Q_ANNULUS * K_S_SPADA * (1 + 0.99e-4)
    solver = build_rotation(meshes, {"k_s": K_S_SPADA, "C_minus_A": given},
                            surface_radius=RE)

    # The given value is kept; a pair that agrees is not silently replaced by
    # the derived one, because the caller's own constant is what it wrote down.
    assert solver.rotation_moments["C_minus_A"] == given


def test_a_pair_just_outside_the_tolerance_is_refused(meshes):  # noqa: F811
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
        build_rotation(meshes, {"k_s": K_S_SPADA, "C_minus_A": given},
                       surface_radius=RE)


# -- Case 4: the benchmark's own inconsistent pair ---------------------------

#: The Spada benchmark's two values of `C - A`: `2.63e35 kg m^2` prescribed in
#: its load excitation against the `2.6952e35 kg m^2` implied by the
#: `k_s = 0.96672389` of its transfer function. Their ratio is the
#: inconsistency, 2.42 percent, and it is scale free, so it applies on the test
#: annulus exactly as it does on the benchmark.
BENCHMARK_RATIO = 2.63 / 2.6952


def test_the_benchmark_inconsistent_pair_is_refused(meshes):  # noqa: F811
    """The 2.42 percent inconsistency of Spada et al. (2011) cannot be stated.

    This is the failure the check exists for: the pair is 240 times outside
    `ROTATION_CONSISTENCY_RTOL`, and it shows as 3.6 percent in `|m|` at
    `t = 0` and more at later epochs, so accepting it would put an
    unattributable error into every polar-motion comparison.
    """
    given = Q_ANNULUS * K_S_SPADA * BENCHMARK_RATIO
    with pytest.raises(ValueError, match="disagree"):
        build_rotation(meshes, {"k_s": K_S_SPADA, "C_minus_A": given},
                       surface_radius=RE)


def test_the_refusal_message_carries_both_values_and_the_implied_k_s(meshes):  # noqa: F811
    """The message must let the caller decide which of the two it meant.

    Which value is wrong is not knowable inside the solver, so the message has
    to report the given `C - A`, the given `k_s`, the `C - A = Qhat k_s` they
    imply and the `k_s = (C - A) / Qhat` the given ellipticity implies. On the
    benchmark's pair that last number is 0.94334, the value
    `reference_state.C_MINUS_A`'s comment names for the prescribed
    `2.63e35 kg m^2`, and reproducing it here is an independent check of the
    arithmetic as well as of the message.

    The numbers are compared by value and not by substring, for the reason in
    `numbers_in`. The implied `k_s` is checked to 1e-5 absolute, which is the
    five-digit form quoted in `reference_state.py`.
    """
    given = Q_ANNULUS * K_S_SPADA * BENCHMARK_RATIO
    with pytest.raises(ValueError) as excinfo:
        build_rotation(meshes, {"k_s": K_S_SPADA, "C_minus_A": given},
                       surface_radius=RE)
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
    power of the guess, so the resolution refuses instead.
    """
    with pytest.raises(ValueError, match="surface_radius"):
        build_rotation(meshes, {"k_s": K_S_SPADA})


# -- Case 6: the radius cross-check ------------------------------------------


@pytest.mark.parametrize(
    "radius, label",
    [(6.371e6, "metres"),           # a dimensional radius, 2.9e6 times too big
     (RC, "core-mantle boundary")])  # 1.2037 against 2.2037, 45 percent short
def test_a_surface_radius_that_is_not_the_mesh_radius_is_refused(meshes, radius, label):  # noqa: F811
    """A radius in metres, or the CMB radius, must not reach `Q`.

    `SURFACE_RADIUS_RTOL = 2e-2` is a check on the *meaning* of the argument,
    not on the mesh: it must catch these two mistakes, and it must not grade a
    coarse annulus whose straight facets already cost 4e-4 of radius
    (`check_geometry`). Both cases here are refused by many orders of
    magnitude, which is the point - the check is a type check in disguise.
    """
    with pytest.raises(ValueError, match="outer radius"):
        build_rotation(meshes, {"k_s": K_S_SPADA}, surface_radius=radius)


def test_a_surface_radius_inside_two_percent_is_accepted(meshes):  # noqa: F811
    """A radius 1.9 percent off the mesh still builds, and `Q` follows it.

    The accepting side of `SURFACE_RADIUS_RTOL = 2e-2`, and the statement that
    the mesh check does *not* bound the accuracy of `Q`: the derived `C - A` is
    then about 10 percent wrong, because `Q ~ ahat^5` and `1.019^5 = 1.099`.
    The caller's `surface_radius`, not the mesh, is what sets the accuracy of
    the `k_s` route, and this test records that as intended behaviour rather
    than as a bound anybody can rely on.
    """
    ahat = RE * 1.019
    solver = build_rotation(meshes, {"k_s": K_S_SPADA}, surface_radius=ahat)

    expected = 4 * np.pi * ahat ** 5 * OMEGA_SQ_EARTH / (3 * LAMBDA) * K_S_SPADA
    assert np.isclose(solver.rotation_moments["C_minus_A"], expected,
                      rtol=FORMULA_RTOL, atol=0.0)


# -- Case 7: the key whitelist -----------------------------------------------


@pytest.mark.parametrize("key", ["ks", "kS", "k_S", "C_minusA"])
def test_an_unknown_rotation_moments_key_is_refused(meshes, key):  # noqa: F811
    """A misspelt key must not be ignored.

    Ignoring it would restore in silence exactly the unchecked `C - A` this
    machinery exists to remove: a caller who writes `ks` believes it has asked
    for the consistent route and gets no check at all. The four spellings here
    are the ones a hand or an editor produces.
    """
    with pytest.raises(ValueError, match="Unknown rotation_moments key"):
        build_rotation(meshes, {key: K_S_SPADA}, surface_radius=RE)


# -- Case 8: the existing spelling is untouched ------------------------------


def test_C_minus_A_alone_passes_through_unchanged(meshes):  # noqa: F811
    """The spelling every current caller uses is taken as given, unchecked.

    No `k_s` is given, so the solver has no second opinion: `C - A` reaches the
    closure exactly as written, and the toy values the structural tests use
    (`C_minus_A = 0.1 C`, `3.2737e-3 C`), which are the ellipticity of no real
    figure, must keep working. This is the test that fails if the consistency
    check is ever made unconditional.
    """
    solver = build_rotation(meshes, {"C_minus_A": 0.1})

    assert float(solver._closure_constant(0)) == 0.1
    # Nothing is invented on this route: no `k_s` appears, and no radius is
    # required or consulted.
    assert "k_s" not in solver.rotation_moments
    assert solver.surface_radius is None


def test_C_alone_still_builds_and_keeps_its_polar_moment(meshes):  # noqa: F811
    """`{"C": ...}` alone, the 2-D configuration of most tests, is untouched.

    `C` closes the `m_3` row and no Love number constrains it, so the
    resolution must leave it alone and must not demand a `C - A` that a 2-D
    disc does not need.
    """
    parent, sub = meshes
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    expected = fd.assemble(fd.dot(Xm, Xm) * dx_m)

    solver = build_rotation(meshes, {})

    # Assembled twice from the same form on the same mesh, so the two agree to
    # round-off; 1e-13 is that with margin.
    assert np.isclose(float(solver._closure_constant(2)), expected,
                      rtol=FORMULA_RTOL, atol=0.0)


def test_tidal_inertia_factor_without_surface_radius_raises(meshes):  # noqa: F811
    """`Q` is unavailable when no radius was given, and says so.

    A solver on the `C_minus_A`-alone route has no `ahat`, so the property must
    refuse instead of returning a number built from a default radius.
    """
    solver = build_rotation(meshes, {"C_minus_A": 0.1})

    with pytest.raises(ValueError, match="surface_radius"):
        solver.tidal_inertia_factor


# -- Case 9: idempotence -----------------------------------------------------


def test_resolving_twice_gives_the_same_C_minus_A(ks_solver):
    """A second resolution must not move the derived constant.

    `k_s` is kept in the dictionary rather than consumed, so the second pass
    sees `C_minus_A` already present and compares it against a freshly computed
    `Qhat k_s`. It must agree exactly (relative difference 0) and leave the
    value alone; a resolution that multiplied by `k_s` again, or that consumed
    `k_s` and then found nothing to check, would fail here.
    """
    before = ks_solver.rotation_moments["C_minus_A"]

    ks_solver._resolve_rotation_moments()

    assert ks_solver.rotation_moments["C_minus_A"] == before
    assert ks_solver.rotation_moments["k_s"] == K_S_SPADA


# -- Case 10: the parallel maximum over the mesh radius ----------------------


@pytest.mark.parallel(nprocs=2)
def test_the_radius_check_agrees_across_ranks(meshes):  # noqa: F811
    """On 2 ranks the radius check reduces, so both ranks reach the same answer.

    The mesh radius is measured as the largest coordinate radius over the whole
    communicator, and this annulus really does split so that one rank misses
    the outer boundary: measured on 2 ranks, the local maxima are 2.2037 and
    **2.0037**, the second 9.1 percent below `RE` and so far outside
    `SURFACE_RADIUS_RTOL = 2e-2`. An un-reduced check would therefore refuse
    the correct `surface_radius` on rank 1 alone, which inside a collective
    constructor is a one-sided exception or a hang, not a clean failure. The
    claim is that the 2-rank build succeeds on both ranks and derives the same
    `Qhat k_s` the serial build does.

    A module-level function, not a method: `mpi-pytest` drops the class from a
    marked node id and the test would never run.
    """
    solver = build_rotation(meshes, {"k_s": K_S_SPADA}, surface_radius=RE)

    assert np.isclose(solver.rotation_moments["C_minus_A"],
                      Q_ANNULUS * K_S_SPADA, rtol=FORMULA_RTOL, atol=0.0)
