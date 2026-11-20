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
#:   coastline or the prescribed load use the preset tolerances: with the
#:   tight ones the `LOAD` march stops at step 2 with
#:   `DIVERGED_LINEAR_SOLVE` (the second Newton step starts from a KSP
#:   right-hand side of 2.3e-17 and does not reduce it by 1e-11 in 10000
#:   iterations), and `sea-fixed` takes 2 Newton steps per step;
#: - `cores`: the number of MPI ranks;
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
    "power-direct": dict(base="SEA", preset="direct", changes={"model": "power"}),
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
    return out


#: The files that `gia_record_expected.py` copies, by case: pairs of the
#: driver output and the expected file, both relative to `CASE_DIR`.
RECORDED = {
    "forward": {name: (forward_log(name), forward_expected(name))
                for name in FORWARD_RECORDED},
}

#: Every step of `meta.py`: `case`, `entry`, `driver`, `cores`, `outputs`,
#: and optionally `deps`, `hpc` and `launcher_args`.
ENTRIES = _forward_entries()
