"""The entries of the GIA cases, their tolerances and the lists the tests use.

One entry is one driver process for one configuration. `meta.py` turns each
entry of `ENTRIES` into one step of `doit run_case`, and the test files
parametrize over the same lists. A name of an entry, a tolerance or an output
file exists in this file only.

This module is plain Python with no Firedrake import, because the test files
import it and the weekly long-test run collects them on a login node.
"""

from pathlib import Path

#: The directory of the cases; every output path is relative to it.
CASE_DIR = Path(__file__).resolve().parent
OUT = "out"
EXPECTED = "expected"

# -- forward ------------------------------------------------------------------------

#: The number of steps after which `sea-direct` and `sea-direct-np2` write
#: their state, and from which the restart entries continue.
RESTART_STEP = 4

#: The march entries of the forward case, by name. Each value says how the
#: entry differs from the base configuration `base` (`SEA` or `LOAD` of
#: `gia_march.py`):
#:
#: - `changes`: `Case` fields;
#: - `preset`: the solver preset;
#: - `default_options`: True adds the tight tolerances of
#:   `gia_cases.build_new` (Newton `snes_rtol` 1e-11, `snes_atol` 1e-13);
#:   False keeps the tolerances of the preset. The entries with a fixed
#:   coastline or the prescribed load use the preset tolerances. With the
#:   tight ones the `LOAD` march stops at step 2 with
#:   `DIVERGED_LINEAR_SOLVE`: after the first Newton step the residual is
#:   4.3e-11, and all of it lies along the declared rigid rotation, which
#:   the curved facets annihilate only to 8e-6 relative. The projected
#:   linear solve cannot remove that part (its right-hand side is 2.2e-17),
#:   so Newton cannot reach `snes_atol` 1e-13. Without the declared
#:   nullspace the same step converges in one Newton step to 2.4e-17.
#:   `sea-fixed` takes 2 Newton steps per step with the tight options;
#: - `cores`: the number of MPI ranks;
#: - `parameters`: replaced parameters of `gia_cases.build_new`;
#: - `save_state`: write the state after `RESTART_STEP` steps;
#: - `restart_from`: read the state that this entry wrote and march the
#:   steps after `RESTART_STEP`;
#: - `local`: False for an entry that the development machine does not run
#:   (more than 2 ranks); it runs on CI only.
FORWARD = {
    "sea-direct": dict(base="SEA", preset="direct", save_state=True),
    "sea-iterative": dict(base="SEA", preset="iterative", default_options=False),
    "sea-multiplier": dict(base="SEA", preset="direct",
                           changes={"representation": "multiplier"}),
    "sea-fixed": dict(base="SEA", preset="direct", default_options=False,
                      changes={"load": "sea_fixed"}),
    "sea-fixed-iterative": dict(base="SEA", preset="iterative", default_options=False,
                                changes={"load": "sea_fixed"}),
    "load-direct": dict(base="LOAD", preset="direct", default_options=False),
    "load-iterative": dict(base="LOAD", preset="iterative", default_options=False),
    "power-direct": dict(base="SEA", preset="direct", changes={"model": "power"},
                         parameters={"approximation.transition_stress": 1e-3}),
    "sea-direct-np2": dict(base="SEA", preset="direct", cores=2, save_state=True),
    "sea-iterative-np3": dict(base="SEA", preset="iterative", default_options=False,
                              cores=3, local=False),
    "restart": dict(base="SEA", preset="direct", restart_from="sea-direct"),
    "restart-np2": dict(base="SEA", preset="direct", cores=2, restart_from="sea-direct"),
    "restart-from-np2": dict(base="SEA", preset="direct",
                             restart_from="sea-direct-np2"),
}

#: The entries of the forward case that are not a march of `FORWARD`, each
#: run by its own function of `gia_forward.py` and written as `name value`
#: lines to `out/forward-<entry>.dat`:
#:
#: - `unreduced`: the Quadrature solver against the unreduced mixed system
#:   of `gia_reference_mixed.py` (an independent system);
#: - `reference-quadrature`: three steps against the stored states of the
#:   earlier solver (`gadopt.gia_gravity` at `2f4b1397`);
#: - `convert-quadrature-bdf2`: a checkpoint file of the earlier solver
#:   restores this one, which then continues as a run from rest does.
FORWARD_VALUES = ["unreduced", "reference-quadrature", "convert-quadrature-bdf2"]

#: The 3-D entry of the forward case, run by `gia_forward.run_sphere`: the
#: 24-column radially extruded cubed sphere of
#: `tests/unit/test_gia_solver_gate_3d.py` (one mesh for the displacement
#: and the potential), the Quadrature history with BDF2, the low-rank DtN,
#: two Maxwell elements, a fluid core, the three rotation rows, the three
#: centre-of-mass rows and a prescribed load centred off the axis, under the
#: iterative preset on 2 ranks for `SPHERE_STEP_COUNT` steps. The
#: development machine does not run it (no 3-D solve there), so it has no
#: recorded values: its tests check only bounds and couplings that need no
#: recorded value. Its cost is read from the CI run.
SPHERE = "sphere-iterative"
SPHERE_CORES = 2
SPHERE_STEP_COUNT = 3

#: In the sphere entry each of the three rotation unknowns and the core
#: pressure is above `COUPLING_FLOOR * u_l2` at the last step: the load has
#: a mean (core pressure), a `z` part (centre of mass) and `x z`, `y z`
#: parts (the polar-wander pair `m1`, `m2`), and its weight on the
#: rotating Earth moves `m3`.
SPHERE_COUPLING_COLUMNS = ("m1", "m2", "m3", "core_pressure")

#: Serial march entries with recorded values in `expected/`. The restart
#: and multi-rank entries are compared with `sea-direct` instead.
FORWARD_RECORDED = ["sea-direct", "sea-iterative", "sea-multiplier", "sea-fixed",
                    "sea-fixed-iterative", "load-direct", "load-iterative",
                    "power-direct"]

#: The relations between entries: `(entry, reference, rtol, first row)`.
#: Every value column of `entry` agrees with the same column of `reference`
#: from `first row` on (0-based), with the rule of `SCALE_OF`. The restart
#: entries hold the rows after `RESTART_STEP`.
RELATIONS = [
    ("sea-iterative", "sea-direct", 1e-5, 0),
    ("sea-multiplier", "sea-direct", 1e-8, 0),
    ("sea-fixed-iterative", "sea-fixed", 1e-5, 0),
    ("load-iterative", "load-direct", 1e-5, 0),
    ("restart", "sea-direct", 1e-10, RESTART_STEP),
    ("sea-direct-np2", "sea-direct", 1e-8, 0),
    ("restart-np2", "sea-direct", 1e-8, RESTART_STEP),
    ("restart-from-np2", "sea-direct", 1e-8, RESTART_STEP),
    ("sea-iterative-np3", "sea-direct", 1e-5, 0),
]

#: Entries with the direct preset; the others are iterative.
FORWARD_DIRECT = [name for name, e in FORWARD.items() if e["preset"] == "direct"]

#: Relative tolerance of a recorded column: the direct entries solve to
#: round-off (Newton `snes_rtol` 1e-11), the iterative entries to the
#: tolerances of the iterative preset.
RECORDED_RTOL_DIRECT = 1e-8
RECORDED_RTOL_ITERATIVE = 1e-5

#: The size that each value column is compared against, in addition to its
#: own value: `|measured - expected| <= rtol |expected| + scale_rtol scale`,
#: with `scale_rtol = SCALE_RTOL_DIRECT` when both sides come from the direct
#: preset and `scale_rtol = rtol` otherwise. A
#: quantity that is small against the size it could have (the mean radial
#: displacement of a nearly incompressible mantle, the core pressure under a
#: load with no degree-0 part, the residual first moment) is then compared
#: at the round-off of that size, and not at the round-off of its own small
#: value, which another BLAS or factorisation order does not reproduce.
#: Each entry names how the scale is formed from the columns of the same
#: row: `("column", c)` is `|c|`; `("product", a, b)` is `|a b|`;
#: `("l2", c)` is `sqrt(re_area * c)`, the Cauchy-Schwarz bound of the
#: integral of a field whose square integrates to `c`. A column that is not
#: listed is compared relative to its own value only.
#: The direct entries reproduce the small columns (`core_pressure`, `m3`,
#: `ur_mean` at 1e-8 to 1e-10 of their scale) to at most 3e-15 of the scale
#: between rank counts, restarts and DtN representations; 1e-12 leaves room
#: for another BLAS and factorisation order and still fails a sign flip.
SCALE_RTOL_DIRECT = 1e-12

SCALE_OF = {
    "u_probe": ("product", "u_norm", "u_probe_norm"),
    "psi_probe": ("product", "psi_norm", "psi_probe_norm"),
    "history_probe": ("product", "history_norm", "history_probe_norm"),
    "ur_mean": ("l2", "ur_sq"),
    "geoid_mean": ("l2", "geoid_sq"),
    "sl_mean": ("l2", "sl_sq"),
    "sheet_mass": ("column", "sheet_abs_mass"),
    "mass_identity": ("column", "sheet_abs_mass"),
    "com_0": ("column", "u_l2"),
    "com_1": ("column", "u_l2"),
    "dipole_0": ("column", "sheet_moment_0"),
    "dipole_1": ("column", "sheet_moment_1"),
    "core_pressure": ("column", "u_l2"),
    "m3": ("column", "u_l2"),
    "cmb_flux": ("column", "u_l2"),
}

#: The suffixes of the cache counters of the preconditioner contexts
#: (`gadopt/preconditioners.py`); `gia_march.preconditioner_counters` writes
#: one column per counter found, and the tests compare them as counts.
COUNTER_SUFFIXES = ("build_count", "column_builds", "assembly_count",
                    "elimination_count")

#: Columns that are bounds derived from the solver tolerances, not results.
#: They are compared with the recorded values of the same entry, but not
#: between entries with other presets.
BOUND_COLUMNS = ("mass_bound",)

#: Columns made of final residual norms, which are round-off and do not
#: reproduce between runs; they are checked against bounds only, never
#: compared with a recorded value or another entry.
RESIDUAL_COLUMNS = ("mass_newton", "newton_reduction")

#: The net sheet mass may be at most this fraction of the absolute sheet
#: mass. The identity of the `Shift` row sets it to the order of the
#: centre-of-mass multipliers; measured 0.8 percent in `sea-direct`, whose
#: ice load has a degree-1 part that the multipliers hold.
MASS_LEAK_RTOL = 2e-2

#: The largest Newton count of a step with a moving coastline. The scenario
#: was measured at 3 to 5 iterations per step.
MAX_NEWTON_MOVING = 5

#: The power-law entry must be nonlinear in its rheology: the smallest
#: power-law factor `eta_eff / eta` of its last step is below this value.
#: With the transition stress 1e-3 (a third of the default of
#: `gia_cases.approximation`) the factor falls from 0.955 at step 1 to 0.766
#: at step 8, with at most 5 Newton iterations per step. With the default
#: 3e-3 it stays above 0.967, which is nearly the Newtonian march.
POWER_ENTRY = "power-direct"
POWER_FACTOR_BELOW = 0.9

#: Each coupling must do something: at the last step the named column is
#: above this fraction of `u_l2`, so a stored value of a quantity that sits
#: at round-off cannot pass for a check.
COUPLING_FLOOR = 1e-4
COUPLING_COLUMNS = {"sea-direct": ("shift", "com_0"),
                    "load-direct": ("core_pressure", "m3")}

#: The entries with a fixed coastline or a prescribed load: their residual
#: is linear in the solution, so one Newton step solves them to the
#: tolerance of the linear solve. The reduction of the residual norm in that
#: step (`newton_reduction`) is at most `LINEAR_REDUCTION[preset]`: ten times
#: the largest value measured.
LINEAR_ENTRIES = ["sea-fixed", "sea-fixed-iterative", "load-direct", "load-iterative"]
#: Measured largest reductions: 1.4e-7 (direct, `load-direct`) and 1.5e-7
#: (iterative, `load-iterative`). The direct entries stop on the absolute
#: Newton tolerance (`CONVERGED_FNORM_ABS`), the iterative ones on the
#: relative one (`CONVERGED_FNORM_RELATIVE`).
LINEAR_REDUCTION = {"direct": 1.4e-6, "iterative": 1.5e-6}

#: The rebuild rule of the cached operators, checked in the iterative
#: entries with a linear residual. Their solver publishes no operator version
#: (the Quadrature step marks its Jacobian as dependent on the solution), so
#: the caches follow the lagged branch of `gadopt.preconditioners.
#: _RebuildRule`, which rebuilds when the effective history step `h / alpha`
#: differs from the one of the last build. That step changes at step 1 (the
#: first build), step 2 (the first BDF2 step after the backward-Euler start),
#: step 5 (the new `dt`, with the BDF2 coefficients of unequal steps) and
#: step 6 (the coefficients of equal steps again), so each counter increases
#: by 1 at exactly those 1-based steps.
REBUILD_ENTRIES = ("sea-fixed-iterative", "load-iterative")
REBUILD_STEPS = (1, 2, 5, 6)
REBUILD_COUNTERS = ("pc_DtNTwoBlockSchurPC_ainvb_build_count",
                    "pc_inner_split0_ReducedGIABlockPC_assembly_count")

#: `|D_i| <= DIPOLE_RTOL * sheet_moment_i`: the centre of mass holds to this
#: fraction of the first moment of the sheets that the displacement must
#: cancel (the bound of `test_sea_level.py` for the frame). The iterative
#: preset solves the constraint row to its own tolerance: measured 2.4e-7 in
#: `sea-iterative`, against 2e-11 with the direct preset.
DIPOLE_RTOL = 1e-8
DIPOLE_RTOL_ITERATIVE = 1e-6

#: `|int u . n dS_cmb| <= CMB_FLUX_RTOL * u_l2`: the fluid core keeps the
#: volume of the core.
CMB_FLUX_RTOL = 1e-9


#: `unreduced`: the largest ratio of a nodal difference to the bound
#: `5e-7 |b| + 2e-9` of `test_direct_matches_coupled_quadrature` is at most 1
#: for every field and step; the fields compared must not be zero.
UNREDUCED_FIELDS = ("displacement", "potential", "history")
UNREDUCED_REALS = ("core_pressure", "m3", "centre_of_mass_0", "centre_of_mass_1",
                   "sea_level_shift")
UNREDUCED_STEP_COUNT = 4

#: `reference-quadrature`: the stored states of the earlier solver, the label
#: of the configuration in that file (`gia_cases.Case.label`), and the
#: bound of `test_gia_solver.py::test_three_steps_agree_with_the_reference`.
REFERENCE_FILE = CASE_DIR.parent / "unit" / "data" / "gia_forward_reference.json"
REFERENCE_LABEL = "Quadrature-bdf2-lowrank-maxwell2-fluid-com-surface"
REFERENCE_FIELDS = ("displacement", "potential", "history")
SOLVED_RTOL = 1e-10

#: `convert-quadrature-bdf2`: the restored and continued states agree with a
#: run from rest to this bound (the bound of `test_gia_solver.py::
#: _old_checkpoint_converts`), and the restored previous step is the second
#: step of the file, 0.13.
CONVERT_RTOL = 1e-11
CONVERT_PREVIOUS_DT = .13


def forward_log(entry):
    """The path of the per-step table of a forward entry, relative to `CASE_DIR`."""
    return f"{OUT}/forward-{entry}.log"


def forward_state(entry):
    """The path of the state file a forward entry writes for a restart."""
    return f"{OUT}/forward-{entry}.h5"


def forward_values(entry):
    """The path of the `name value` file of an entry of `FORWARD_VALUES`."""
    return f"{OUT}/forward-{entry}.dat"


def forward_expected(entry):
    """The path of the recorded values of a forward entry."""
    return f"{EXPECTED}/forward-{entry}.txt"


def _forward_entries():
    """The `ENTRIES` items of the forward case."""
    out = []
    for name, entry in FORWARD.items():
        outputs = [forward_log(name)]
        if entry.get("save_state"):
            outputs.append(forward_state(name))
        item = {"case": "forward", "entry": name, "driver": "gia_forward.py",
                "cores": entry.get("cores", 1), "outputs": outputs}
        if "restart_from" in entry:
            # The state file of the other entry, read where it is.
            item["deps"] = [{"case": "tests/gia", "link": False,
                             "artifact": forward_state(entry["restart_from"])}]
        out.append(item)
    for name in FORWARD_VALUES:
        out.append({"case": "forward", "entry": name, "driver": "gia_forward.py",
                    "cores": 1, "outputs": [forward_values(name)]})
    out.append({"case": "forward", "entry": SPHERE, "driver": "gia_forward.py",
                "cores": SPHERE_CORES, "outputs": [forward_log(SPHERE)]})
    return out


# -- limits -------------------------------------------------------------------------

#: The entries of the `limits` case (`gia_limits.py`), one process each, so
#: that the three groups compile and run in parallel. Each writes the
#: measured and the reference value of every row as `name value` lines to
#: `out/limits-<entry>.dat`. No row has a recorded value; the tolerances
#: below are those of the unit tests that the rows come from.
LIMITS = ("sea", "mechanics", "rheology")

#: The DtN representations of the rows that the original tests ran on both.
REPRESENTATIONS = ("multiplier", "lowrank")

#: Eustatic shift of a melting cap on a stiff, non-gravitating Earth against
#: `-(rho_i / rho_w) int dI / L_ocean` (`test_sea_level.py::
#: TestEustaticLimit`, `test_sea_level_sheet.py::TestSolves`).
EUSTATIC_RTOL = 1e-4
#: Fixed-coastline eustatic shift on a rigid Earth against
#: `-rho_i int (1 - C0) dI / (rho_w int C0)` (`TestFixedOcean::
#: test_the_shift_is_the_eustatic_value`).
FIXED_EUSTATIC_RTOL = 1e-6
#: Two solves of the same problem (saturated fixed against live coastline;
#: ocean ice under a fixed coastline against none) agree per field in the
#: max norm (`TestFixedOcean`).
SAME_PROBLEM_RTOL = 1e-8
#: Guards of those rows: the shift is above this value, so the relative
#: comparison has a scale; the ice grown over the ocean is more than this
#: multiple of the land ice change.
SATURATED_SHIFT_FLOOR = 1e-4
OCEAN_ICE_MASS_RATIO = 100.0
#: The mean sea-level change next to added ice exceeds the one far from it
#: by this fraction of the eustatic value (`TestSignAgainstPhysics`).
SIGN_MARGIN = 1e-3
#: The water fraction changes over more than this fraction of one facet of
#: Re (`TestSolves::test_the_marine_grounding_line_moves`).
GROUNDING_FACET_FRACTION = 0.5
#: A second solve with nothing changed repeats `Shift` and the potential to
#: this tolerance; the shift is above the floor (`TestRepeatedSolve`).
REPEAT_RTOL = 1e-8
REPEAT_SHIFT_FLOOR = 1e-3
#: The surface sinks under added ice on an elastic Earth: the mean radial
#: displacement under the caps is below `-SINK_RMS_FRACTION` times the root
#: mean square of the radial displacement over Re.
SINK_RMS_FRACTION = 0.1

#: `B_mu = 0` with a fluid core against the uncoupled mechanics with the
#: core constraint (`TestFluidCore::
#: test_zero_B_mu_matches_independently_constrained_mechanics`,
#: `TestNullCoupling::test_it_reproduces_the_uncoupled_solver`): both
#: fluxes below `NULL_FLUX_ATOL`; the core pressure to `NULL_PRESSURE_RTOL`
#: (absolute floor `NULL_PRESSURE_ATOL`); strain and history to
#: `NULL_FIELD_RTOL`; the displacement after the rigid modes are projected
#: out to `NULL_DISPLACEMENT_RTOL`.
NULL_FLUX_ATOL = 1e-12
NULL_PRESSURE_RTOL = 1e-11
NULL_PRESSURE_ATOL = 1e-13
NULL_FIELD_RTOL = 1e-10
NULL_DISPLACEMENT_RTOL = 1e-11
#: `|int u . n dS_cmb| <= CMB_FLUX_SCALE_RTOL scale + CMB_FLUX_ATOL`
#: (`TestFluidCore::test_solve_enforces_zero_integrated_cmb_displacement`).
CMB_FLUX_SCALE_RTOL = 1e-10
CMB_FLUX_ATOL = 1e-13
#: A `cos 2 phi` load: the `cos2` trace coefficient above `DEGREE_ACTIVE`,
#: every other one below `DEGREE_INACTIVE_RTOL |cos2| + 1e-15`
#: (`test_the_answer_is_degree_two`; the driver writes the largest
#: `(|c| - 1e-15) / |cos2|`).
DEGREE_ACTIVE = 1e-6
DEGREE_INACTIVE_RTOL = 1e-9
#: `|D_i| < COM_LOAD_RTOL * SIGMA_HAT pi Re^2` after a degree-1 load
#: (`TestCentreOfMassFrame::test_solve_holds_the_centre_of_mass`).
COM_LOAD_RTOL = 1e-8
#: The content of the declared rigid rotation in the answer, and its ratio
#: to the content without the declaration
#: (`TestRigidRotationNullspace::test_declaring_it_removes_it_from_the_answer`).
ROTATION_CONTENT_DECLARED = 1e-14
ROTATION_CONTENT_RATIO = 1e-3

#: `TestNewtonianLimit`: exponent 1 with power-law parameters against the
#: Newtonian solve; exponent `1 + 1e-6` with a background stress against
#: the Newtonian displacement; with zero background stress the gap is above
#: `ZERO_BACKGROUND_GAP`.
EXPONENT_ONE_RTOL = 1e-12
NEAR_ONE_RTOL = 1e-6
ZERO_BACKGROUND_GAP = 1e-1
#: The stiff power-law march: each of its steps converges in at most this
#: many Newton iterations (`test_newton_converges_on_a_stiff_power_law`).
STIFF_STEPS = 3
STIFF_NEWTON_MAX = 10


def limits_values(entry):
    """The path of the `name value` file of a `limits` entry."""
    return f"{OUT}/limits-{entry}.dat"


def _limits_entries():
    """The `ENTRIES` items of the `limits` case."""
    return [{"case": "limits", "entry": name, "driver": "gia_limits.py", "cores": 1,
             "outputs": [limits_values(name)]} for name in LIMITS]


# -- operators ----------------------------------------------------------------------

#: The entries of the `operators` case (`gia_operators.py`), one process each.
#: No entry solves: each builds its solvers, sets a coordinate-defined state
#: (`gia_cases.coordinate_state`) and measures the Jacobian through its
#: actions on coordinate-defined vectors. A `Real` row or column is read as
#: the action on a unit `Real` function, never from an assembled matrix
#: block (firedrakeproject/firedrake#5665). Each entry writes the measured
#: and the reference number of every row as `name value` lines to
#: `out/operators-<entry>.dat`. No row has a recorded value; the tolerances
#: below are those of the unit tests that the rows come from.
OPERATORS = ("sea-transposes", "frame", "block0", "block1", "schur", "schur-np2")

#: The core count of each entry; the entries not listed run on one core.
OPERATOR_CORES = {"schur-np2": 2}

#: The block pairs of `test_sea_level.py::TestTransposes`, by the short names
#: of that file (`u`, `psi`, `Shift`, `lambda_x`, `lambda_y`), on the two
#: surface states `cap` (every quadrature point far from the rounding of the
#: sheet) and `shoreline` (a live coastline inside facets).
SEA_STATES = ("cap", "shoreline")
SEA_PAIRS = (("u", "psi"), ("u", "Shift"), ("psi", "Shift"),
             ("u", "lambda_x"), ("u", "lambda_y"),
             ("psi", "lambda_x"), ("psi", "lambda_y"),
             ("Shift", "lambda_x"), ("Shift", "lambda_y"))
#: The rotation pairs of `TestRotationTransposes`, on the shoreline state
#: with the rotation row `rot` (the 2-D scalar `m3`).
ROTATION_PAIRS = (("u", "rot"), ("psi", "rot"), ("Shift", "rot"))

#: A pair `(a, b)` transposes when the two bilinear values
#: `x_a . J_ab x_b` and `x_b . J_ba x_a` agree to `TRANSPOSE_RTOL` of the
#: Cauchy-Schwarz scale `max(|x_a| |J_ab x_b|, |x_b| |J_ba x_a|)`, which is
#: the size the pairing could have. The unit tests held
#: `max|A_ab - A_ba^T| <= 1e-14 max(|A_ab|, |A_ba|)` on the dense blocks;
#: both sides are variations of one energy and differ by round-off only.
TRANSPOSE_RTOL = 1e-14
#: The pairing itself is above this fraction of its scale, so that the
#: comparison above is made between two numbers that carry the block and
#: not between two round-off values. A sign error in either block gives a
#: difference of twice the pairing.
PAIRING_FLOOR = 1e-6
#: `(Shift, m3)` and `(m3, Shift)` are both above this fraction of the
#: `(Shift, Shift)` diagonal (`test_the_shift_rotation_block_is_nonzero`):
#: the sheet is in `Rotation.inertia_form` and `psi_rot` is in the sea level.
SHIFT_ROTATION_FLOOR = 1e-10
#: The displacement row of the sea level at `u = psi = 0` equals the row of
#: a plain `normal_stress` of `B_mu g(Re) rho_w S`
#: (`TestSheetSigns::test_the_mechanics_row_matches_a_normal_stress`); the
#: two rows integrate at the boundary degrees 6 and 10, measured 4e-9 apart.
NORMAL_STRESS_RTOL = 1e-6

#: The centre-of-mass saddle pair (`TestCentreOfMassFrame::
#: test_blocks_form_an_exact_saddle_pair`, 1e-14 of the larger block) on
#: both representations, with and without rotation, and on the 24-column
#: sphere (`test_extruded_sphere_assembles_the_saddle_pair`). The
#: multiplier-multiplier entries are exactly zero, as in the unit test.
FRAME_CONFIGURATIONS = (("multiplier", False), ("multiplier", True),
                        ("lowrank", False), ("lowrank", True))
FRAME_RTOL = 1e-14
#: The translation moment of the sphere: `D[e_i] = M e_i` to 1e-2 of the
#: whole mass `M` (the core volume is read from its faceted surface) and
#: the other two moments below 1e-10 of it.
SPHERE_MOMENT_RTOL = 1e-2
SPHERE_MOMENT_CROSS_RTOL = 1e-10
#: `|int sigma (x^2 + y^2) dS|` of the `cos 2 phi` load is below this
#: absolute value (`TestInertiaTranspose::test_inertia_of_a_cos2_sheet_vanishes`).
INERTIA_COS2_ATOL = 1e-14

#: The eliminated Quadrature block 0 (`TestEliminatedBlockSymmetry`) and the
#: block 0 with a live coastline: the bilinear asymmetry of three
#: coordinate-defined directions with zero `Real` part, relative to the
#: largest pairing, is at most this.
BLOCK0_SYMMETRY_RTOL = 1e-12
#: The sea-level part of block 0 is above this fraction of the whole, so
#: that the live-coastline row measures a block that contains the sheet.
SEA_BLOCK0_FLOOR = 1e-6

#: The `Real` block against its derived diagonal
#: (`TestBlockOneIsTheDiagonalItClaims`, `TestBlockOneInThreeDimensions`):
#: `|d_k - A_kk| <= DIAG_TOL |A_kk|` per row with a nonzero derived entry,
#: `|A_kk| <= DIAG_TOL max|A_jj|` for a Lagrange multiplier (the core
#: pressure, derived zero), and per row `max_j |A_kj| / |A_kk| <=
#: OFFDIAG_TOL` (relative to `max|A_jj|` for the zero rows).
DIAG_TOL = 1e-12
OFFDIAG_TOL = 1e-12
BLOCK1_TRUNCATIONS = (3, 5)
#: The multiplier alignment (`check_multiplier_alignment`): a fired row
#: against its discrete prediction, a row the mode must not move against the
#: weakest fired row (`ALIGNMENT_SILENT_2D` on the annulus,
#: `ALIGNMENT_SILENT_SPHERE` on the sphere at `L = 1`), and the analytic
#: deviation, the boundary's resolution of the mode, below
#: `ALIGNMENT_ANALYTIC`.
ALIGNMENT_PREDICTED = 1e-12
ALIGNMENT_SILENT_2D = 1e-6
ALIGNMENT_SILENT_SPHERE = 1e-6
ALIGNMENT_ANALYTIC = 1e-2
#: The polar pair of the sphere (`test_the_polar_pair_is_uncoupled_and_carries_C_minus_A`):
#: `m1` and `m2` carry the same entry to `POLAR_PAIR_RTOL`, `m3` carries
#: `-C / (C - A) = -10` times it to `POLAR_RATIO_RTOL`.
POLAR_PAIR_RTOL = 1e-14
POLAR_RATIO = -10.0
POLAR_RATIO_RTOL = 1e-12

#: The Schur preconditioners after `pc.setUp()` (`test_dtn_schur_ainvb.py`,
#: `test_dtn_multiplier_dense_schur.py`). `P^{-1} A x = x` and
#: `P^{-T} A^T x = x` to `SCHUR_IDENTITY_FLOOR`; the complement against an
#: independent elimination, the adjoint identity and the dense solves to
#: `SCHUR_PARITY_FLOOR`; the forward and the transpose apply of the cached
#: path differ by more than `SCHUR_TRANSPOSE_MARGIN`, the forward and the
#: transpose solve of the dense complement by more than
#: `DENSE_TRANSPOSE_MARGIN`. The vectors and the errors have the rigid
#: rotation of the mantle removed (see `gia_operators.py`).
SCHUR_IDENTITY_FLOOR = 1e-7
SCHUR_PARITY_FLOOR = 1e-10
SCHUR_TRANSPOSE_MARGIN = 1e-5
DENSE_TRANSPOSE_MARGIN = 1e-3


def operators_values(entry):
    """The path of the `name value` file of an `operators` entry."""
    return f"{OUT}/operators-{entry}.dat"


def _operators_entries():
    """The `ENTRIES` items of the `operators` case."""
    return [{"case": "operators", "entry": name, "driver": "gia_operators.py",
             "cores": OPERATOR_CORES.get(name, 1),
             "outputs": [operators_values(name)]} for name in OPERATORS]


# -- time_convergence ---------------------------------------------------------------

#: The entries of the `time_convergence` case (`gia_time_convergence.py`), one
#: process each. An entry names how its configuration differs from `SEA` of
#: `gia_march.py` (`changes`: `Case` fields; `parameters`: replaced parameters
#: of `gia_cases.build_new`). Each entry marches a reference and every
#: scheme and level of `TC_SCHEMES` and `TC_LEVELS` from rest to
#: `TC_END_TIME` under the same ice load, constant in time, and writes the
#: errors against the reference as `name value` lines to
#: `out/time_convergence-<entry>.dat` (names from `tc_key`). No value is
#: recorded: the expected values are the theoretical orders of the schemes.
#:
#: `power` uses the transition stress 1e-3 of the forward entry
#: `power-direct`. The smallest power-law factor `eta_eff / eta` at the end
#: time is then 0.893, against 0.987 with the default 3e-3, under which the
#: march is nearly the Newtonian one of `maxwell` and the entry would not
#: test the order of the scheme on a nonlinear rheology. Both values
#: converge at 8 steps (step 0.08) with at most 5 Newton iterations per step.
TIME_CONVERGENCE = {
    "maxwell": dict(changes={}),
    "power": dict(changes={"model": "power"},
                  parameters={"approximation.transition_stress": 1e-3}),
}

#: The end time of every march, in units of the Maxwell time of the
#: approximation. The ice load is applied at `t = 0` and kept, so the
#: viscous relaxation is far from complete at this time and the history
#: still changes at the end.
TC_END_TIME = 0.64

#: The step count of the reference march (BDF2, equal steps). Its error is
#: `(64 / 256)^2 = 1/16` of the error of BDF2 at the finest level, 64 steps.
#: An error measured against it is `e(n) - e(256)`, so for an exact second
#: order the order measured between 32 and 64 steps is
#: `log2((64 - 1) / (16 - 1)) = 2.07` and not 2. Against backward Euler the
#: reference is exact to the measured digits.
TC_REFERENCE_STEPS = 256

#: The step counts of the levels; each level halves the mean step.
TC_LEVELS = (8, 16, 32, 64)

#: The schemes, by name: the `time_scheme` of the solver and the pattern of
#: the steps. `equal` steps are `TC_END_TIME / n`. `alternating` steps are
#: `TC_ALTERNATION` times the mean step `TC_END_TIME / n`, in turn, so that
#: the step ratio `h_new / h_old` is 1.5 and 2/3 in turn and the variable-step
#: BDF2 coefficients differ from the equal-step ones at every step after the
#: first. Both factors average to 1, so `n` even steps end at `TC_END_TIME`.
TC_SCHEMES = {
    "backward_euler": ("backward_euler", "equal"),
    "bdf2": ("bdf2", "equal"),
    "bdf2_alternating": ("bdf2", "alternating"),
}
TC_ALTERNATION = (0.8, 1.2)

#: The quantities whose errors are measured: the displacement and the
#: potential (`errornorm` against the reference divided by the norm of the
#: reference), and the eustatic shift of the sea level (the difference
#: divided by the reference shift).
TC_QUANTITIES = ("u", "psi", "shift")

#: Backward Euler is first order: the order measured between 32 and 64 steps
#: is in this band for the displacement and the potential. Measured on
#: `maxwell` at the feasibility check: 1.01 and 0.98.
TC_BE_ORDER_BAND = (0.9, 1.15)
TC_BE_ORDER_QUANTITIES = ("u", "psi")
TC_BE_ORDER_LEVELS = (32, 64)

#: BDF2 is second order on equal and on variable steps: the orders measured
#: between 16 and 32 and between 32 and 64 steps are at least this value
#: for every quantity of `TC_QUANTITIES`. Measured on `maxwell` with equal
#: steps: 1.99, 2.06 (displacement), 1.91, 2.03 (potential), 2.64, 2.39
#: (shift); with alternating steps: 1.98, 2.06; 1.90, 1.98; 1.82, 1.92.
#: The order between 8 and 16 steps is not asserted: there the step is too
#: long for the asymptotic order (1.28 for the shift with alternating
#: steps on `maxwell`). Constant-step coefficients on the alternating steps give an
#: inconsistent approximation of the time derivative, which this bound
#: fails.
TC_BDF2_ORDER_MIN = 1.8
TC_BDF2_ORDER_PAIRS = ((16, 32), (32, 64))

#: Both schemes converge to the same solution: the error of backward Euler
#: against the BDF2 reference at 64 steps is below this value for every
#: quantity of `TC_QUANTITIES`. Measured: 1.8e-4 to 3.8e-4 for the
#: displacement and the potential, 1e-4 for the shift.
TC_BE_LIMIT_ERROR = 1e-3
TC_BE_LIMIT_STEPS = 64


def tc_key(scheme, steps, quantity):
    """The name in the `.dat` file of one number of one scheme and level.

    Args:
      scheme: a name of `TC_SCHEMES`, or `"reference"`.
      steps: the step count of the march.
      quantity: `"u"`, `"psi"` or `"shift"` for an error; `"time"` for the
        end time reached; `"newton_max"` for the largest Newton count of a
        step; the reference also writes `"u_norm"`, `"psi_norm"`,
        `"shift_value"` and `"min_power_factor"`.
    """
    return f"{scheme}_{steps}_{quantity}"


def time_convergence_values(entry):
    """The path of the `name value` file of a `time_convergence` entry."""
    return f"{OUT}/time_convergence-{entry}.dat"


def _time_convergence_entries():
    """The `ENTRIES` items of the `time_convergence` case."""
    return [{"case": "time_convergence", "entry": name,
             "driver": "gia_time_convergence.py", "cores": 1,
             "outputs": [time_convergence_values(name)]} for name in TIME_CONVERGENCE]


# -- adjoint ------------------------------------------------------------------------

#: The entries of the `adjoint` case (`gia_adjoint.py`), one process each.
#: An entry is one taped march of `ADJOINT_STEPS` for one configuration and
#: one control. Each value says how the entry differs from its base
#: configuration (`SEA` or `LOAD` of `gia_march.py`):
#:
#: - `control`: the control, one of `ADJOINT_CONTROLS`;
#: - `preset`, `changes`, `parameters`, `default_options`, `cores`: as in
#:   `FORWARD`. The `LOAD` entries use the tolerances of the preset: with
#:   the tight ones their Newton solve stalls on the declared rigid rotation
#:   (see `FORWARD`);
#: - `checkpoint`: `"disk"` tapes the march under
#:   `SingleDiskStorageSchedule` with disk checkpointing;
#: - `counter_preset`, `counter_changes`: the preset and the `Case` changes
#:   of a second taped march that gives the build counters of the entry.
#:   The direct preset has no cached preconditioner and so no counter
#:   (block 0 is `AssembledPC` with MUMPS). With the iterative preset, a
#:   moving coastline makes the Jacobian depend on the state, and the
#:   Quadrature low-rank path then sets `gia_rebuild = always`
#:   (`SelfGravitatingGIASolver._adjust_options`): every cache rebuilds at
#:   every Newton iteration, whatever the time step. So `sea-direct-dt`
#:   measures its derivatives with the direct preset on `SEA`, and its
#:   counters with the iterative preset on `SEA` with a fixed coastline,
#:   whose caches follow the lagged rule of `gadopt.preconditioners.
#:   _RebuildRule` (rebuild when the effective step `dt / alpha` changes).
#:
#: Each entry writes the values of `ADJOINT_VALUES` to
#: `out/adjoint-<entry>.dat` and the counts (`adjoint_count_names`) to
#: `out/adjoint-<entry>-counts.dat`, which `gia_record_expected.py` copies
#: to `expected/adjoint-<entry>.txt`.
ADJOINT = {
    "sea-direct-viscosity": dict(base="SEA", preset="direct", control="viscosity"),
    "sea-iterative-viscosity": dict(base="SEA", preset="iterative", control="viscosity"),
    "sea-multiplier-viscosity": dict(base="SEA", preset="direct", control="viscosity",
                                     changes={"representation": "multiplier"}),
    "sea-direct-ice": dict(base="SEA", preset="direct", control="ice"),
    "sea-direct-dt": dict(base="SEA", preset="direct", control="dt",
                          counter_preset="iterative",
                          counter_changes={"load": "sea_fixed"}),
    "load-direct-Lambda": dict(base="LOAD", preset="direct", control="Lambda",
                               default_options=False),
    "load-direct-B_mu": dict(base="LOAD", preset="direct", control="B_mu",
                             default_options=False),
    "power-direct-sl_init": dict(base="SEA", preset="direct", control="sl_init",
                                 changes={"model": "power"},
                                 parameters={"approximation.transition_stress": 1e-3}),
    "sea-direct-viscosity-disk": dict(base="SEA", preset="direct", control="viscosity",
                                      checkpoint="disk"),
    "sea-direct-viscosity-np2": dict(base="SEA", preset="direct", control="viscosity",
                                     cores=2),
}

#: The controls, by name: a CG1 field on the mantle that scales every
#: viscosity by `exp(m)`; the shape of the ice thickness (CG2 on the
#: mantle), which the march multiplies by `1 + .2 k` before step `k`; a
#: `Real` factor on every time step (`dt = h m`); the numbers `Lambda` and
#: `B_mu` as `Real` Functions on the parent mesh; the initial sea level
#: `SL_init` (CG2 on the mantle).
ADJOINT_CONTROLS = ("viscosity", "ice", "dt", "Lambda", "B_mu", "sl_init")

#: The steps of the adjoint scenario, in Maxwell times. The change from 0.08
#: to 0.13 at step 3 (1-based) gives the variable-step BDF2 coefficients
#: and a rebuild of the cached operators inside the taped march. The ice
#: before step `k` (0-based) is `.002 (1 + .2 k) (1 - cos theta) / 2`.
ADJOINT_STEPS = (.08, .08, .13, .13)

#: The 1-based step at which the time step changes. The effective step
#: `dt / alpha` of the history differs at each of the four steps (backward
#: Euler start, equal-step BDF2, the variable-step coefficients at the new
#: `dt`, equal steps again), so the lagged caches of the counter march of
#: `sea-direct-dt` rebuild at every step; the test asserts the rebuild at
#: this step, where `dt` itself changes.
ADJOINT_DT_CHANGE_STEP = 3

#: The step of the second control value, `control + OTHER_STEP * direction`,
#: at which a replay is compared with a fresh march.
ADJOINT_OTHER_STEP = 0.05

#: The step of the central difference of two fresh marches,
#: `(J(m + eps h) - J(m - eps h)) / (2 eps)`. Its error is `O(eps^2)`:
#: measured 3.2e-9 relative for the viscosity control at the feasibility
#: check of the roadmap.
ADJOINT_FRESH_EPS = 1e-3

#: The step of the central difference of two replays, which the tangent is
#: compared with (`test_gia_step_adjoint.py::_central_difference`).
ADJOINT_REPLAY_EPS = 1e-4

#: The values each entry writes to `out/adjoint-<entry>.dat`:
#:
#: - `J`: the functional of the taped march;
#: - `J_replay`: the replay of the tape at the recorded control;
#: - `J_fresh`: a fresh, untaped march at the recorded control;
#: - `J_other_replay`, `J_other_fresh`: the same at the other control;
#: - `dJ`: the directional derivative `<dJ/dm, h>` from the adjoint;
#: - `dJ_fresh`: the central difference of two fresh marches;
#: - `taylor_rate`: the smallest rate of the first-order Taylor remainder
#:   `|J(m + eps h) - J(m) - eps dJ|` over `eps = 0.01 / 2^i`, `i = 0..3`
#:   (the definition of `pyadjoint.taylor_test`);
#: - `taylor_rate_no_gradient`: the same with `dJ` replaced by 0 (the
#:   negative control of `test_gravity_adjoint.assert_taylor_with_guards`);
#: - `tangent`: the tangent sweep `dJ/dm h` on the tape;
#: - `tangent_replay`: the central difference of two replays;
#: - `adjoint_refused_residual`: 0, or the true residual of a linear adjoint
#:   solve that the library refused (`gia_adjoint.adjoint_derivative`; `dJ`
#:   and the rate with the gradient are then NaN);
#: - `wall`: the wall time of the entry in seconds (not asserted).
ADJOINT_VALUES = ("J", "J_replay", "J_fresh", "J_other_replay", "J_other_fresh",
                  "dJ", "dJ_fresh", "taylor_rate", "taylor_rate_no_gradient",
                  "tangent", "tangent_replay", "adjoint_refused_residual", "wall")

#: Tolerances of the per-entry assertions (roadmap 4.6), relative to the
#: second value named: replay against the recording; fresh march against
#: the recording; replay against the fresh march at the other control; the
#: adjoint against the fresh central difference; the floor of `|dJ|` against
#: `|J|`; the tangent against the adjoint pairing and against the difference
#: of replays.
REPLAY_RTOL = 1e-9
FRESH_RTOL = 1e-9
OTHER_RTOL = 1e-8
FRESH_DIFFERENCE_RTOL = 1e-6
DERIVATIVE_FLOOR = 1e-6
TANGENT_PAIRING_RTOL = 1e-8
TANGENT_REPLAY_RTOL = 1e-6
#: The smallest Taylor rate with the gradient, and the band of the rate
#: without it.
TAYLOR_RATE_MIN = 1.95
NO_GRADIENT_BAND = (0.85, 1.15)

#: The relations between entries: `(entry, reference, rtol of J, rtol of
#: dJ)`. The multiplier DtN solves the same system as the low-rank one; the
#: iterative preset solves it to its own tolerances; the disk schedule and
#: two ranks change only the order of the operations.
ADJOINT_RELATIONS = [
    ("sea-multiplier-viscosity", "sea-direct-viscosity", 1e-8, 1e-6),
    ("sea-iterative-viscosity", "sea-direct-viscosity", 1e-5, 1e-5),
    ("sea-direct-viscosity-disk", "sea-direct-viscosity", 1e-8, 1e-6),
    ("sea-direct-viscosity-np2", "sea-direct-viscosity", 1e-8, 1e-6),
]

#: The phases of a taped march whose build counters are written: the taped
#: forward march, the first replay at the recorded control, and the adjoint
#: sweep that follows it. The value written for a phase, a counter and a
#: step is the increase of the counter during that step of the phase.
COUNT_PHASES = ("forward", "replay", "adjoint")

#: The `guards` entry: one process, one flag (1 or 0) per guard, and the
#: numbers behind each flag. Each guard is an existing unit test that needs a
#: taped solve:
#:
#: - `capped_derivative_raises`, `uncapped_derivative_passes`: an adjoint
#:   solve capped with `derivative_max_it=1` raises `ConvergenceError`, with
#:   400 it does not (`test_gia_solver.py::test_a_capped_derivative_solve_raises`);
#: - `hessian_refused`: the Hessian of the Quadrature step raises
#:   `NotImplementedError` (`test_gia_step_adjoint.py::
#:   test_the_quadrature_step_refuses_a_hessian`);
#: - `absent_control_refused`, `present_control_accepted`:
#:   `require_controls_reach_residual` raises for a Function that is not in
#:   the residual and accepts the viscosity control;
#: - `scaled_gradient_detected`: the gradient times 1.01 gives a Taylor rate
#:   of at most `SCALED_GRADIENT_RATE_MAX` (`scaled_gradient_rate`);
#: - `tape_structure`: each step records sweep marker, step block, sweep
#:   marker, kernel projection, in this order, with the projection reading
#:   the output of the step block (`test_gia_step_adjoint.py::
#:   test_every_step_kind_records_the_same_structure`);
#: - `projection_replay`, `projection_taylor`: with the rigid rotation
#:   declared, the replay of a functional of the tangential surface
#:   displacement, which a rigid rotation changes, repeats the recording and
#:   a fresh march to `PROJECTION_RTOL` (`projection_replay_error`,
#:   `projection_fresh_error`), and its Taylor rate is at least
#:   `TAYLOR_RATE_MIN` (`projection_rate`)
#:   (`test_gia_nullspace_projection_adjoint.py::test_gia_projection_is_recorded`);
#: - `collected_direct`, `collected_iterative`: no solver of a finished
#:   taped run stays alive once the tape is cleared
#:   (`test_gia_step_adjoint.py::test_a_finished_run_is_collected`);
#: - `restore_replay`, `restore_taylor`: a state copied and restored into a
#:   second solver after two steps, on the tape: replay to `REPLAY_RTOL`
#:   (`restore_replay_error`) and Taylor rate at least `TAYLOR_RATE_MIN`
#:   (`restore_rate`) (`test_gia_solver.py::
#:   test_taylor_through_a_restore_on_the_tape`).
GUARDS = "guards"
GUARD_FLAGS = ("capped_derivative_raises", "uncapped_derivative_passes",
               "hessian_refused", "absent_control_refused", "present_control_accepted",
               "scaled_gradient_detected", "tape_structure", "projection_replay",
               "projection_taylor", "collected_direct", "collected_iterative",
               "restore_replay", "restore_taylor")
#: A gradient scaled by 1.01 makes the first-order remainder `0.01 eps dJ`,
#: so its rate falls to about 1 (`test_gravity_adjoint.py` sabotage bound).
SCALED_GRADIENT_RATE_MAX = 1.15
#: The replay of the tangential functional against the recording and a
#: fresh march (`test_gia_nullspace_projection_adjoint.REPLAY_RTOL`).
PROJECTION_RTOL = 1e-10


def adjoint_values(entry):
    """The path of the `name value` file of an `adjoint` entry."""
    return f"{OUT}/adjoint-{entry}.dat"


def adjoint_counts(entry):
    """The path of the block counts and build counters of an `adjoint` entry."""
    return f"{OUT}/adjoint-{entry}-counts.dat"


def adjoint_expected(entry):
    """The path of the recorded counts of an `adjoint` entry."""
    return f"{EXPECTED}/adjoint-{entry}.txt"


def adjoint_count_name(phase, counter, step):
    """The name of the increase of `counter` during the 1-based `step` of `phase`."""
    return f"{phase}_{counter}_step_{step}"


def adjoint_blocks_name(step):
    """The name of the number of tape blocks that the 1-based `step` records."""
    return f"blocks_step_{step}"


def _adjoint_entries():
    """The `ENTRIES` items of the `adjoint` case."""
    out = [{"case": "adjoint", "entry": name, "driver": "gia_adjoint.py",
            "cores": entry.get("cores", 1),
            "outputs": [adjoint_values(name), adjoint_counts(name)]}
           for name, entry in ADJOINT.items()]
    out.append({"case": "adjoint", "entry": GUARDS, "driver": "gia_adjoint.py",
                "cores": 1, "outputs": [adjoint_values(GUARDS)]})
    return out


#: The files that `gia_record_expected.py` copies, by case: pairs of the
#: driver output and the expected file, both relative to `CASE_DIR`.
RECORDED = {
    "forward": {name: (forward_log(name), forward_expected(name))
                for name in FORWARD_RECORDED},
    "adjoint": {name: (adjoint_counts(name), adjoint_expected(name))
                for name in ADJOINT},
}

#: Every step of `meta.py`: `case`, `entry`, `driver`, `cores`, `outputs`,
#: and optionally `deps`, `hpc` and `launcher_args`.
ENTRIES = (_forward_entries() + _limits_entries() + _operators_entries()
           + _time_convergence_entries() + _adjoint_entries())
