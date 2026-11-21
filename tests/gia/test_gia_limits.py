"""Tests of the `limits` case: each solve against its limit, closed form or twin.

`gia_limits.py` writes, for each row, the measured value and the value it is
checked against to `out/limits-<entry>.dat`. Each test here holds the
tolerance of the unit test that the row comes from, named in its docstring.
No value is recorded from an earlier run.

This file does not import Firedrake; it runs where the outputs are.
"""

import math

import pytest

import gia_entries as E


def values(entry):
    """The `name value` lines of `out/limits-<entry>.dat` as a dictionary."""
    path = E.CASE_DIR / E.limits_values(entry)
    out = {}
    with open(path) as f:
        for line in f:
            name, value = line.split()
            out[name] = float(value)
    return out


def assert_close(measured, expected, rtol, label):
    """`|measured - expected| <= rtol |expected|`, with both numbers in the message."""
    bound = rtol * abs(expected)
    assert abs(measured - expected) <= bound, (
        f"{label}: measured {measured!r}, expected {expected!r}, "
        f"difference {abs(measured - expected)!r} above {rtol!r} relative")


def assert_converged(v, *keys):
    """Every named convergence flag is 1."""
    for key in keys:
        assert v[key] == 1.0, f"{key}: the solve did not converge ({v[key]!r})"


# -- sea ----------------------------------------------------------------------------

@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
@pytest.mark.parametrize("label", ["eustatic", "eustatic_softplus"])
def test_the_shift_is_the_eustatic_value(label, representation):
    """`Shift = -(rho_i / rho_w) int dI dS / L_ocean` on a stiff, non-gravitating Earth.

    `test_sea_level.py::TestEustaticLimit::test_the_shift_is_the_eustatic_value`
    and, with the softplus width of that file, `test_sea_level_sheet.py::
    TestSolves::test_the_shift_is_the_eustatic_value`. Melting raises the sea.
    """
    v = values("sea")
    assert_converged(v, f"{label}_converged_{representation}")
    closed = v[f"{label}_closed_{representation}"]
    assert closed > 0.0, f"{label}: the closed form {closed!r} must be positive"
    assert_close(v[f"{label}_shift_{representation}"], closed, E.EUSTATIC_RTOL,
                 f"{label} {representation}")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_the_fixed_coastline_shift_is_the_eustatic_value(representation):
    """`Shift = -rho_i int (1 - C0) dI / (rho_w int C0)` on a rigid Earth.

    `test_sea_level.py::TestFixedOcean::test_the_shift_is_the_eustatic_value`.
    Ice that grows lowers the sea, so the closed form is negative.
    """
    v = values("sea")
    assert_converged(v, f"fixed_eustatic_converged_{representation}")
    closed = v[f"fixed_eustatic_closed_{representation}"]
    assert closed < 0.0, f"fixed eustatic: the closed form {closed!r} must be negative"
    assert_close(v[f"fixed_eustatic_shift_{representation}"], closed,
                 E.FIXED_EUSTATIC_RTOL, f"fixed eustatic {representation}")


def assert_same_problem(v, prefix):
    """Displacement, potential and `Shift` of two solves agree to `SAME_PROBLEM_RTOL`."""
    for name in ("displacement", "potential", "shift"):
        difference = v[f"{prefix}_{name}"]
        assert difference <= E.SAME_PROBLEM_RTOL, (
            f"{prefix} {name}: relative difference {difference!r}, "
            f"at most {E.SAME_PROBLEM_RTOL!r}")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_a_saturated_coastline_gives_the_live_solution(representation):
    """On a coastline that cannot move, the fixed and the live solves agree.

    `test_sea_level.py::TestFixedOcean::
    test_a_saturated_coastline_gives_the_live_solution` (1e-8 per field,
    rigid rotation removed from the displacement difference).
    """
    v = values("sea")
    assert_converged(v, f"saturated_converged_False_{representation}",
                     f"saturated_converged_True_{representation}")
    shift = v[f"saturated_live_shift_{representation}"]
    assert abs(shift) > E.SATURATED_SHIFT_FLOOR, (
        f"saturated shift {shift!r} must exceed {E.SATURATED_SHIFT_FLOOR!r}")
    assert_same_problem(
        {k.replace(f"_{representation}", ""): x for k, x in v.items()
         if k.startswith("saturated_") and k.endswith(representation)}, "saturated")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_ice_over_the_ocean_changes_nothing(representation):
    """Ice grown over the reference ocean does not load a fixed-coastline solve.

    `test_sea_level.py::TestFixedOcean::test_ice_over_the_ocean_changes_nothing`:
    the ocean ice is more than 100 times the land ice change, and the two
    solves agree to 1e-8 per field.
    """
    v = values("sea")
    rep = representation
    assert_converged(v, f"ocean_ice_converged_0_{rep}", f"ocean_ice_converged_100_{rep}")
    ocean, land = v[f"ocean_ice_ocean_mass_{rep}"], v[f"ocean_ice_land_mass_{rep}"]
    assert ocean > E.OCEAN_ICE_MASS_RATIO * land, (
        f"ocean ice {ocean!r} must exceed {E.OCEAN_ICE_MASS_RATIO!r} x land ice {land!r}")
    assert_same_problem(
        {k.replace(f"_{rep}", ""): x for k, x in v.items()
         if k.startswith("ocean_ice_") and k.endswith(rep)}, "ocean_ice")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_water_piles_up_towards_added_ice(representation):
    """The geoid bulges towards added ice, so the sea rises next to it.

    `test_sea_level.py::TestSignAgainstPhysics::
    test_water_piles_up_towards_added_ice_on_a_stiff_earth`: the mean
    sea-level change next to the caps exceeds the mean far from them by
    more than 1e-3 of the eustatic value. A flipped geoid sign reverses it.
    """
    v = values("sea")
    assert_converged(v, f"sign_converged_{representation}")
    gap = v[f"sign_near_minus_far_{representation}"]
    margin = E.SIGN_MARGIN * v[f"sign_eustatic_{representation}"]
    assert gap > margin, f"sign {representation}: near - far {gap!r}, must exceed {margin!r}"


def test_the_marine_grounding_line_moves():
    """The water fraction changes over more than half a facet of Re.

    `test_sea_level_sheet.py::TestSolves::test_the_marine_grounding_line_moves`
    (multiplier path), so the marine case is not a fixed grounding line.
    """
    v = values("sea")
    assert_converged(v, "grounding_converged")
    bound = E.GROUNDING_FACET_FRACTION * v["grounding_facet"]
    assert v["grounding_change"] > bound, (
        f"grounding line: integral of |W - W_ref| {v['grounding_change']!r}, "
        f"must exceed {bound!r}")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_a_second_solve_gives_the_same_shift_and_geoid(representation):
    """A second solve with nothing changed repeats `Shift` and the potential.

    `Shift` and the potential repeat to `REPEAT_RTOL` (1e-8 relative); a
    datum that lags one solve behind moves `Shift` by 17 percent. The second
    solve, with the rigid rotation declared, takes at most
    `REPEAT_NEWTON_MAX` Newton steps, so that the row does not depend on how
    Newton iterates on round-off.
    """
    v = values("sea")
    rep = representation
    assert_converged(v, f"repeat_converged_{rep}")
    newton = v[f"repeat_second_newton_{rep}"]
    assert newton <= E.REPEAT_NEWTON_MAX, (
        f"repeat {rep}: second solve took {newton!r} Newton steps, at most "
        f"{E.REPEAT_NEWTON_MAX!r}")
    shift = v[f"repeat_shift_{rep}"]
    assert abs(shift) > E.REPEAT_SHIFT_FLOOR, f"repeat shift {shift!r} at the floor"
    change = v[f"repeat_shift_change_{rep}"]
    assert change <= E.REPEAT_RTOL * abs(shift), (
        f"repeat {rep}: Shift changed by {change!r}, at most {E.REPEAT_RTOL!r} x {shift!r}")
    scale, psi_change = v[f"repeat_psi_scale_{rep}"], v[f"repeat_psi_change_{rep}"]
    assert scale > 0.0
    assert psi_change <= E.REPEAT_RTOL * scale, (
        f"repeat {rep}: potential changed by {psi_change!r}, "
        f"at most {E.REPEAT_RTOL!r} x {scale!r}")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_the_surface_sinks_under_added_ice(representation):
    """The mean radial displacement under growing ice is negative and not small.

    A row without a unit-test original: on the elastic Earth of
    `TestRepeatedSolve` with ice growing on the two caps, the mean of
    `u . n` over the caps is below `-SINK_RMS_FRACTION` times the root mean
    square of `u . n` over Re. A sea-level sheet whose weight enters the
    displacement row with the wrong sign lifts the caps instead.
    """
    v = values("sea")
    mean, rms = v[f"sink_caps_mean_{representation}"], v[f"sink_rms_{representation}"]
    bound = -E.SINK_RMS_FRACTION * rms
    assert mean < bound, (
        f"{representation}: mean u . n under the caps {mean!r}, must be below {bound!r}")


# -- mechanics ----------------------------------------------------------------------

def test_zero_B_mu_is_the_uncoupled_mechanics_with_the_core_constraint():
    """`B_mu = 0` with a fluid core equals the independently constrained mechanics.

    `test_gia_gravity.py::TestFluidCore::
    test_zero_B_mu_matches_independently_constrained_mechanics` and
    `TestNullCoupling::test_it_reproduces_the_uncoupled_solver`, on the
    Quadrature family: both CMB fluxes below 1e-12, the core pressure to
    1e-11 (absolute 1e-13), strain and history to 1e-10, the displacement to
    1e-11 after the rigid modes are projected out. The reference is
    `CoupledInternalVariableSolver`, which has no gravity.
    """
    v = values("mechanics")
    assert_converged(v, "null_converged")
    assert abs(v["null_response_flux"]) > 1e-12, "the core pressure must move the flux"
    for key in ("null_flux", "null_flux_reference"):
        assert abs(v[key]) < E.NULL_FLUX_ATOL, f"{key}: {v[key]!r}"
    p, p_ref = v["null_pressure"], v["null_pressure_reference"]
    bound = E.NULL_PRESSURE_RTOL * abs(p_ref) + E.NULL_PRESSURE_ATOL
    assert abs(p - p_ref) <= bound, (
        f"core pressure: measured {p!r}, reference {p_ref!r}, bound {bound!r}")
    for key in ("null_strain", "null_history"):
        assert v[key] < E.NULL_FIELD_RTOL, f"{key}: relative difference {v[key]!r}"
    assert v["null_u_norm"] > 0.0
    assert v["null_displacement"] < E.NULL_DISPLACEMENT_RTOL, (
        f"displacement: relative difference {v['null_displacement']!r}")


def test_the_fluid_core_keeps_its_volume():
    """`|int u . n dS_cmb| <= 1e-10 scale + 1e-13` after a solve.

    `test_gia_gravity.py::TestFluidCore::
    test_solve_enforces_zero_integrated_cmb_displacement`.
    """
    v = values("mechanics")
    assert_converged(v, "cmb_converged")
    assert v["cmb_u_norm"] > 0.0
    bound = E.CMB_FLUX_SCALE_RTOL * v["cmb_scale"] + E.CMB_FLUX_ATOL
    assert abs(v["cmb_flux"]) <= bound, f"cmb flux {v['cmb_flux']!r}, bound {bound!r}"


def test_the_answer_is_degree_two():
    """A `cos 2 phi` load excites the `cos 2 phi` coefficients and nothing else.

    `test_gia_gravity.py::test_the_answer_is_degree_two`: on both DtN
    boundaries `|cos2| > 1e-6` and every other coefficient below
    `1e-9 |cos2| + 1e-15`.
    """
    v = values("mechanics")
    assert v["degree_active_min"] > E.DEGREE_ACTIVE, (
        f"cos2 coefficient {v['degree_active_min']!r}, must exceed {E.DEGREE_ACTIVE!r}")
    assert v["degree_inactive_ratio_max"] < E.DEGREE_INACTIVE_RTOL, (
        f"largest (|c| - 1e-15) / |cos2| of another mode: "
        f"{v['degree_inactive_ratio_max']!r}, below {E.DEGREE_INACTIVE_RTOL!r}")


@pytest.mark.parametrize("representation", E.REPRESENTATIONS)
def test_the_centre_of_mass_holds_after_a_degree_one_load(representation):
    """`|D_i| < 1e-8 SIGMA_HAT pi Re^2` after a solve with a degree-1 sheet.

    `test_gia_gravity.py::TestCentreOfMassFrame::
    test_solve_holds_the_centre_of_mass`.
    """
    v = values("mechanics")
    rep = representation
    assert_converged(v, f"com_converged_{rep}")
    assert v[f"com_multipliers_finite_{rep}"] == 1.0
    bound = E.COM_LOAD_RTOL * v[f"com_load_moment_{rep}"]
    for i in range(2):
        D = v[f"com_dipole_{i}_{rep}"]
        assert abs(D) < bound, f"{rep} D_{i}: measured {D!r}, bound {bound!r}"


def test_the_declared_rigid_rotation_is_absent_from_the_answer():
    """The declared mode `(-y, x)` is projected out of the displacement.

    `test_gia_gravity.py::TestRigidRotationNullspace::
    test_declaring_it_removes_it_from_the_answer`: its content is below
    1e-14 and below 1e-3 of the content without the declaration.
    """
    v = values("mechanics")
    assert_converged(v, "rotation_converged_plain", "rotation_converged_declared")
    declared, plain = v["rotation_content_declared"], v["rotation_content_plain"]
    assert declared < E.ROTATION_CONTENT_DECLARED, f"declared content {declared!r}"
    assert declared < E.ROTATION_CONTENT_RATIO * plain, (
        f"declared content {declared!r}, plain {plain!r}")


@pytest.mark.parametrize("nullspace", ["plain", "declared"])
@pytest.mark.parametrize("field", ["history", "displacement", "potential"])
def test_the_accepted_state_is_the_solved_state(field, nullspace):
    """After a step the accepted history, `u` and `psi` are the solved state, bit for bit.

    `test_gia_gravity.py::TestAcceptedStateAfterAStep`, with and without the
    declared rigid rotation; the solved field is nonzero.
    """
    v = values("mechanics")
    assert v[f"accepted_{field}_norm_{nullspace}"] > 0.0
    difference = v[f"accepted_{field}_difference_{nullspace}"]
    assert difference == 0.0, f"{field} ({nullspace}): difference {difference!r}"


# -- rheology -----------------------------------------------------------------------

@pytest.mark.parametrize("field", ["u", "history", "psi"])
def test_exponent_one_with_power_law_parameters_is_the_newtonian_solve(field):
    """Exponent 1 makes the transition and background stresses inert (1e-12).

    `test_gia_gravity.py::TestNewtonianLimit::
    test_exponent_one_with_power_law_parameters_is_the_newtonian_solve`.
    """
    v = values("rheology")
    assert v["exponent_one_u_norm"] > 0.0 and v["exponent_one_history_norm"] > 0.0
    difference = v[f"exponent_one_{field}"]
    assert difference < E.EXPONENT_ONE_RTOL, f"{field}: relative difference {difference!r}"


def test_exponent_near_one_is_the_newtonian_solve_to_first_order():
    """`n = 1 + 1e-6` with a background stress is Newtonian to 1e-6; without, not.

    `test_gia_gravity.py::TestNewtonianLimit::
    test_exponent_near_one_is_the_newtonian_solve_to_first_order`: the
    zero-background control differs by more than 0.1, because the factor
    tends to 1/2 there.
    """
    v = values("rheology")
    assert_converged(v, "near_one_converged")
    assert v["near_one_newtonls"] == 1.0, "the outer method must be newtonls"
    assert v["near_one_u"] < E.NEAR_ONE_RTOL, f"near one: {v['near_one_u']!r}"
    assert v["zero_background_u"] > E.ZERO_BACKGROUND_GAP, (
        f"zero background: {v['zero_background_u']!r}")


def test_newton_converges_on_a_stiff_power_law():
    """Every step of the stiff power-law march converges in at most 10 Newton iterations.

    `test_gia_transient_power_law_gate.py::
    test_newton_converges_on_a_stiff_power_law`, on the Quadrature family.
    """
    v = values("rheology")
    for k in range(1, E.STIFF_STEPS + 1):
        assert f"stiff_reason_{k}" in v, f"the march stopped before step {k}"
        reason, newton = v[f"stiff_reason_{k}"], v[f"stiff_newton_{k}"]
        assert reason > 0, f"step {k}: converged reason {reason!r}"
        assert 1 <= newton <= E.STIFF_NEWTON_MAX, (
            f"step {k}: {newton!r} Newton iterations, at most {E.STIFF_NEWTON_MAX}")
        assert math.isfinite(v[f"stiff_min_power_factor_{k}"])
