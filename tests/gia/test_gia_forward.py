"""Assertions of the `forward` case of the coupled GIA solver.

`gia_forward.py` marches each entry of `gia_entries.FORWARD` under
`doit run_case` and writes one row of diagnostics per step to
`out/forward-<entry>.log`. This file reads those tables and asserts:

- recorded values: every column against `expected/forward-<entry>.txt`,
  which detects a change of the results (not a wrong result);
- each coupling does something: the quantities of the couplings are above
  a floor, so that the other assertions do not compare round-off;
- conservation and constraints, from exact identities: the net sheet mass
  that the converged `Shift` row implies, the centre of mass, the volume of
  the fluid core, the Newton count.

This file does not import Firedrake; it runs where the outputs are.
"""

import math

import numpy as np
import pandas as pd
import pytest

import gia_entries as E


def load_log(entry):
    """The per-step table of a forward entry, one row per step."""
    return pd.read_csv(E.CASE_DIR / E.forward_log(entry), sep=r"\s+")


def load_expected(entry):
    """The recorded table of a forward entry, or `None` if it was not recorded."""
    path = E.CASE_DIR / E.forward_expected(entry)
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", comment="#")


def is_counter(column):
    """Whether `column` is a cache counter of a preconditioner context."""
    return column.endswith(tuple(f"_{name}" for name in E.COUNTER_SUFFIXES))


def is_count(column):
    """Whether `column` is a count (compared by its own rules, not as a value)."""
    return column in ("newton", "linear", "operator_version") or is_counter(column)


def is_direct(entry):
    """Whether `entry` uses the direct preset."""
    return E.FORWARD[entry]["preset"] == "direct"


def scale_rtol(rtol, *entries):
    """The tolerance on the scale of a column for a comparison of `entries`.

    `SCALE_RTOL_DIRECT` (or `rtol`, if smaller) when every entry uses the
    direct preset, whose small columns are deterministic; `rtol` otherwise.
    """
    if all(is_direct(entry) for entry in entries):
        return min(rtol, E.SCALE_RTOL_DIRECT)
    return rtol


def recorded_rtol(entry):
    """The relative tolerance of the recorded values of `entry`."""
    return (E.RECORDED_RTOL_DIRECT if entry in E.FORWARD_DIRECT
            else E.RECORDED_RTOL_ITERATIVE)


def scale_of(table, column):
    """The scale of `column` in each row of `table` (`gia_entries.SCALE_OF`)."""
    rule = E.SCALE_OF.get(column)
    if rule is None:
        return np.zeros(len(table))
    kind, *names = rule
    values = [table[name].to_numpy(float) for name in names]
    if kind == "column":
        return np.abs(values[0])
    if kind == "product":
        return np.abs(values[0] * values[1])
    if kind == "l2":
        return np.sqrt(table["re_area"].to_numpy(float) * np.abs(values[0]))
    raise ValueError(rule)


def compare_values(measured, expected, column, rtol, scale_rtol, label):
    """Assert that one value column of two tables agrees, row by row.

    `|m - e| <= rtol |e| + scale_rtol scale`, with the scale of the column
    taken from the expected table (`scale_of`). A NaN in the expected column
    must be NaN in the measured one: the entry has no such quantity.

    Args:
      measured, expected: two tables with the same rows.
      column: the column to compare.
      rtol: the tolerance relative to the expected value.
      scale_rtol: the tolerance relative to the scale of the column.
      label: the entries and the column, for the message.
    """
    m = measured[column].to_numpy(float)
    e = expected[column].to_numpy(float)
    scale = scale_of(expected, column)
    for k, (mk, ek, sk) in enumerate(zip(m, e, scale)):
        if math.isnan(ek):
            assert math.isnan(mk), f"{label} row {k}: measured {mk!r}, expected NaN"
            continue
        allowed = rtol * abs(ek) + scale_rtol * sk
        assert abs(mk - ek) <= allowed, (
            f"{label} row {k}: measured {mk!r}, expected {ek!r}, tolerance "
            f"{allowed!r} (rtol {rtol!r}, scale {sk!r}, scale_rtol {scale_rtol!r})")


@pytest.mark.parametrize("entry", E.FORWARD_RECORDED)
def test_recorded_values(entry):
    """Every column of every step agrees with the recorded values.

    Values: `recorded_rtol(entry)` against the value, and `scale_rtol`
    against the scale of the column (`compare_values`). Newton count: within 1 of the stored count. Linear
    count: at most the stored count plus the larger of 2 and 20 percent.
    Cache counters and `operator_version`: equal.
    """
    expected = load_expected(entry)
    if expected is None:
        pytest.skip(f"{E.forward_expected(entry)} is not recorded yet; "
                    f"measured table:\n{load_log(entry).to_string()}")
    measured = load_log(entry)
    assert list(measured.columns) == list(expected.columns), (
        f"{entry}: columns {list(measured.columns)!r}, "
        f"expected {list(expected.columns)!r}")
    assert len(measured) == len(expected), (
        f"{entry}: {len(measured)} steps, expected {len(expected)}")
    rtol = recorded_rtol(entry)
    for column in expected.columns:
        m, e = measured[column].to_numpy(), expected[column].to_numpy()
        label = f"{entry} {column}"
        if column == "newton":
            assert np.all(np.abs(m - e) <= 1), f"{label}: measured {m!r}, expected {e!r} +- 1"
        elif column == "linear":
            limit = e + np.maximum(2, 0.2 * e)
            assert np.all(m <= limit), f"{label}: measured {m!r}, at most {limit!r}"
        elif column in E.RESIDUAL_COLUMNS:
            continue
        elif is_count(column):
            assert np.array_equal(m, e, equal_nan=True), (
                f"{label}: measured {m!r}, expected {e!r}")
        else:
            compare_values(measured, expected, column, rtol, scale_rtol(rtol, entry), label)


@pytest.mark.parametrize("entry,column", [
    (entry, column) for entry, columns in E.COUPLING_COLUMNS.items()
    for column in columns])
def test_each_coupling_acts(entry, column):
    """At the last step the coupling quantity is above `COUPLING_FLOOR * u_l2`.

    A recorded value of a quantity at round-off would check nothing; this
    assertion makes sure that the quantity responds to the load.
    """
    last = load_log(entry).iloc[-1]
    floor = E.COUPLING_FLOOR * last["u_l2"]
    assert abs(last[column]) > floor, (
        f"{entry} {column}: measured {last[column]!r}, must exceed {floor!r}")


def test_the_power_law_is_nonlinear():
    """The power-law entry departs from the Newtonian viscosity.

    A power-law march whose stress stays far below the transition stress
    solves the Newtonian problem, and its recorded values would not show a
    wrong power-law factor or a wrong Jacobian of it.
    """
    last = load_log(E.POWER_ENTRY).iloc[-1]
    factor = last["min_power_factor"]
    assert factor < E.POWER_FACTOR_BELOW, (
        f"{E.POWER_ENTRY}: smallest power-law factor {factor!r}, "
        f"must be below {E.POWER_FACTOR_BELOW!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_net_sheet_mass_is_set_by_the_frame(entry):
    """The net sheet mass equals the value the converged `Shift` row implies.

    The `Shift` row with the centre-of-mass column gives the identity
    `int sigma dS = sum_i lambda_i int x_i dsigma/dShift dS / g_s`
    (`gia_march._mass_terms`). The difference must be at most the part of
    the row that Newton leaves (`mass_newton`) plus 1e-10 of the absolute
    sheet mass for the round-off of the integrals. The net mass itself is at
    most `MASS_LEAK_RTOL` of the absolute sheet mass. Entries without sea
    level have NaN in these columns and are skipped.
    """
    local(entry)
    table = load_log(entry)
    if table["sheet_mass"].isna().all():
        pytest.skip(f"{entry} has no sea level")
    for _, row in table.iterrows():
        gap = abs(row["sheet_mass"] - row["mass_identity"])
        allowed = row["mass_newton"] + 1e-10 * row["sheet_abs_mass"]
        assert gap <= allowed, (
            f"{entry} step {row['step']:.0f}: |int sigma - identity| = {gap!r}, "
            f"allowed {allowed!r} (int sigma {row['sheet_mass']!r}, "
            f"identity {row['mass_identity']!r})")
        leak = E.MASS_LEAK_RTOL * row["sheet_abs_mass"]
        assert abs(row["sheet_mass"]) <= leak, (
            f"{entry} step {row['step']:.0f}: |int sigma| = {abs(row['sheet_mass'])!r}, "
            f"at most {E.MASS_LEAK_RTOL!r} * int |sigma| = {leak!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_centre_of_mass_holds(entry):
    """`|D_i| <= DIPOLE_RTOL * sheet_moment_i` at every step, both directions.

    `DIPOLE_RTOL_ITERATIVE` for an entry with the iterative preset.

    `D_i` is the first mass moment of the mantle and of every sheet;
    `sheet_moment_i`, the integral of `|x_i sigma|` over the sheets, is the
    moment that the displacement has to cancel, and it must be nonzero.
    """
    local(entry)
    table = load_log(entry)
    if table["dipole_0"].isna().all():
        pytest.skip(f"{entry} has no centre-of-mass constraint")
    rtol = (E.DIPOLE_RTOL if E.FORWARD[entry]["preset"] == "direct"
            else E.DIPOLE_RTOL_ITERATIVE)
    for i in range(2):
        moment = table[f"sheet_moment_{i}"].to_numpy()
        dipole = table[f"dipole_{i}"].to_numpy()
        assert np.all(moment > 0), f"{entry} sheet_moment_{i}: {moment!r}"
        assert np.all(np.abs(dipole) <= rtol * moment), (
            f"{entry} dipole_{i}: measured {dipole!r}, bound {rtol!r} * {moment!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_core_volume_is_kept(entry):
    """`|int u . n dS_cmb| <= CMB_FLUX_RTOL * u_l2`: the fluid core keeps its volume."""
    local(entry)
    table = load_log(entry)
    flux, scale = table["cmb_flux"].to_numpy(), table["u_l2"].to_numpy()
    assert np.all(np.abs(flux) <= E.CMB_FLUX_RTOL * scale), (
        f"{entry} cmb_flux: measured {flux!r}, bound {E.CMB_FLUX_RTOL!r} * {scale!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_newton_count(entry):
    """At most `MAX_NEWTON_MOVING` Newton iterations per step."""
    local(entry)
    newton = load_log(entry)["newton"].to_numpy()
    assert np.all(newton <= E.MAX_NEWTON_MOVING), (
        f"{entry} newton: measured {newton!r}, at most {E.MAX_NEWTON_MOVING}")


def load_values(entry):
    """The `name value` file of an entry of `FORWARD_VALUES`, as a dictionary."""
    path = E.CASE_DIR / E.forward_values(entry)
    out = {}
    for line in path.read_text().splitlines():
        name, value = line.split()
        out[name] = float(value)
    return out


def local(entry):
    """Skip `entry` where it does not run (more than 2 ranks) and has no output."""
    if not E.FORWARD[entry].get("local", True) and not (
            E.CASE_DIR / E.forward_log(entry)).exists():
        pytest.skip(f"{entry} runs on CI only")


@pytest.mark.parametrize("entry,reference,rtol,first", E.RELATIONS,
                         ids=[f"{r[0]}-vs-{r[1]}" for r in E.RELATIONS])
def test_relations_between_entries(entry, reference, rtol, first):
    """Two entries that solve the same problem agree in every value column.

    The other preset, the other DtN representation, the restart from a
    checkpoint and other rank counts must not change the answer beyond the
    tolerance of the relation (`gia_entries.RELATIONS`). Counts and the
    bounds derived from the solver tolerances are not compared.
    """
    local(entry)
    measured = load_log(entry).reset_index(drop=True)
    expected = load_log(reference).iloc[first:].reset_index(drop=True)
    assert list(measured["step"]) == list(expected["step"]), (
        f"{entry} steps {list(measured['step'])!r}, {reference} steps "
        f"{list(expected['step'])!r}")
    for column in expected.columns:
        if is_count(column) or column in (*E.BOUND_COLUMNS, *E.RESIDUAL_COLUMNS):
            continue
        assert column in measured.columns, f"{entry} lacks the column {column!r}"
        compare_values(measured, expected, column, rtol, scale_rtol(rtol, entry, reference),
                       f"{entry} against {reference}: {column}")


@pytest.mark.parametrize("entry", E.LINEAR_ENTRIES)
def test_a_linear_residual_takes_one_newton_step(entry):
    """With a fixed coastline or a prescribed load, one Newton step solves.

    Newton takes one step, and that step reduces the residual norm by at
    least the factor `LINEAR_REDUCTION` of the preset.
    """
    table = load_log(entry)
    newton = table["newton"].to_numpy()
    assert np.all(newton == 1), f"{entry} newton: measured {newton!r}, expected 1"
    reduction = table["newton_reduction"].to_numpy()
    limit = E.LINEAR_REDUCTION[E.FORWARD[entry]["preset"]]
    assert np.all(reduction <= limit), (
        f"{entry} newton_reduction: measured {reduction!r}, at most {limit!r}")


@pytest.mark.parametrize("counter", E.REBUILD_COUNTERS)
@pytest.mark.parametrize("entry", E.REBUILD_ENTRIES)
def test_the_caches_rebuild_where_the_step_changes(entry, counter):
    """Each cache counter increases by 1 exactly at `REBUILD_STEPS`.

    The step operator of these entries depends only on the effective
    history step `h / alpha`, which changes at `REBUILD_STEPS`. A cache that
    does not rebuild there keeps a preconditioner of another operator; one
    that rebuilds elsewhere does not cache.
    """
    table = load_log(entry)
    counts = table[counter].to_numpy()
    steps = table["step"].to_numpy().astype(int)
    increase = np.diff(counts, prepend=0)
    expected = np.array([1 if s in E.REBUILD_STEPS else 0 for s in steps])
    assert np.array_equal(increase, expected), (
        f"{entry} {counter}: counts {counts!r}, increase per step {increase!r}, "
        f"expected {expected!r}")


def test_the_elimination_matches_the_unreduced_system():
    """The Quadrature solver agrees with the unreduced mixed system.

    Per step and field the largest ratio of the nodal difference to
    `5e-7 |b| + 2e-9` is at most 1, and the displacement, the potential and
    the history are not zero.
    """
    values = load_values("unreduced")
    ratios = {name: value for name, value in values.items() if name.endswith("_ratio")}
    for k in range(1, E.UNREDUCED_STEP_COUNT + 1):
        for field in (*E.UNREDUCED_FIELDS, *E.UNREDUCED_REALS):
            assert f"step{k}_{field}_ratio" in ratios, f"step {k}: no ratio of {field}"
    bad = {name: value for name, value in ratios.items() if not value <= 1.0}
    assert not bad, f"differences above the bound (ratio to bound): {bad!r}"
    for k in range(1, E.UNREDUCED_STEP_COUNT + 1):
        for field in E.UNREDUCED_FIELDS:
            size = values[f"step{k}_{field}_size"]
            assert size > 0, f"step {k} {field}: largest value {size!r}"


def test_three_steps_agree_with_the_earlier_solver():
    """Three steps of the Quadrature case agree with the stored earlier solver.

    The fingerprints of the displacement, the potential and the history are
    compared as in `gia_cases.fingerprint_differences`: the larger of the
    norm difference over the reference norm and the probe difference over
    the reference norm times the probe norm, at `SOLVED_RTOL`. Every `Real`
    field is compared against the largest `Real` value of the step, or, if
    all are round-off against the displacement, against 1e-12 of its size.
    """
    import json
    reference = json.loads(E.REFERENCE_FILE.read_text())[E.REFERENCE_LABEL]
    values = load_values("reference-quadrature")
    for k, stored in enumerate(reference, start=1):
        for field in E.REFERENCE_FIELDS:
            norm, probe, probe_norm = stored[field]
            new_norm = values[f"step{k}_{field}_norm"]
            new_probe = values[f"step{k}_{field}_probe"]
            scale = norm if norm > 0 else 1e-300
            difference = max(abs(new_norm - norm) / scale,
                             abs(new_probe - probe) / (scale * probe_norm))
            assert difference <= E.SOLVED_RTOL, (
                f"step {k} {field}: measured norm {new_norm!r} probe {new_probe!r}, "
                f"stored {norm!r} {probe!r}; difference {difference!r}, "
                f"tolerance {E.SOLVED_RTOL!r}")
        reals = stored["reals"]
        measured = {name[len(f"step{k}_real_"):]: value for name, value in values.items()
                    if name.startswith(f"step{k}_real_")}
        assert set(measured) == set(reals), (sorted(measured), sorted(reals))
        size = stored["displacement"][0]
        largest = max(abs(v) for v in reals.values())
        bound = (E.SOLVED_RTOL * largest if largest > 1e-12 * size else 1e-12 * size)
        for name, value in reals.items():
            assert abs(measured[name] - value) <= bound, (
                f"step {k} {name}: measured {measured[name]!r}, stored {value!r}, "
                f"tolerance {bound!r}")


def test_a_checkpoint_of_the_earlier_solver_converts():
    """A file of the earlier solver restores this one, which then continues.

    The restored state and the state after one more step agree with a run
    from rest to `CONVERT_RTOL` per field; the restored previous step is
    `CONVERT_PREVIOUS_DT`; a state with the wrong time order and a
    conversion without the number of ranks of the writer are refused.
    """
    values = load_values("convert-quadrature-bdf2")
    for when in ("restored", "continued"):
        for field in E.REFERENCE_FIELDS:
            relative = values[f"{when}_{field}_relative"]
            assert values[f"{when}_{field}_norm"] > 0, f"{when} {field} is zero"
            assert relative <= E.CONVERT_RTOL, (
                f"{when} {field}: relative difference {relative!r}, "
                f"tolerance {E.CONVERT_RTOL!r}")
    assert values["previous_dt"] == E.CONVERT_PREVIOUS_DT, (
        f"previous_dt {values['previous_dt']!r}, expected {E.CONVERT_PREVIOUS_DT!r}")
    assert values["refuses_time_order"] == 1.0, "a state of the wrong time order was accepted"
    assert values["refuses_missing_writer_ranks"] == 1.0, (
        "a conversion without the number of writer ranks was accepted")


# -- 3-D entry -----------------------------------------------------------------------

def sphere_table():
    """The table of the 3-D entry; skipped where that entry did not run.

    The development machine runs no 3-D solve, so the table exists only on
    CI. The entry has no recorded values: every check below is a bound or a
    coupling floor.
    """
    path = E.CASE_DIR / E.forward_log(E.SPHERE)
    if not path.exists():
        pytest.skip(f"{E.SPHERE} runs on CI only")
    return pd.read_csv(path, sep=r"\s+")


def test_the_sphere_ran_every_step():
    """The 3-D entry wrote one row per step and moved the Earth."""
    table = sphere_table()
    assert len(table) == E.SPHERE_STEP_COUNT, (
        f"{E.SPHERE}: {len(table)} rows, expected {E.SPHERE_STEP_COUNT}")
    assert np.all(table["u_l2"] > 0), f"{E.SPHERE} u_l2: {table['u_l2'].to_numpy()!r}"


@pytest.mark.parametrize("column", E.SPHERE_COUPLING_COLUMNS)
def test_each_coupling_acts_on_the_sphere(column):
    """At the last step each rotation unknown and the core pressure is above the floor.

    In 3-D the polar-wander pair `m1`, `m2` exists only here; a pair that the
    load does not reach would leave its rows unchecked.
    """
    last = sphere_table().iloc[-1]
    floor = E.COUPLING_FLOOR * last["u_l2"]
    assert abs(last[column]) > floor, (
        f"{E.SPHERE} {column}: measured {last[column]!r}, must exceed {floor!r}")


def test_the_centre_of_mass_holds_on_the_sphere():
    """`|D_i| <= DIPOLE_RTOL_ITERATIVE * sheet_moment_i` in all three directions."""
    table = sphere_table()
    for i in range(3):
        moment = table[f"sheet_moment_{i}"].to_numpy()
        dipole = table[f"dipole_{i}"].to_numpy()
        assert np.all(moment > 0), f"{E.SPHERE} sheet_moment_{i}: {moment!r}"
        bound = E.DIPOLE_RTOL_ITERATIVE * moment
        assert np.all(np.abs(dipole) <= bound), (
            f"{E.SPHERE} dipole_{i}: measured {dipole!r}, bound {bound!r}")


def test_the_core_volume_is_kept_on_the_sphere():
    """`|int u . n dS_cmb| <= CMB_FLUX_RTOL * u_l2` on the bottom of the extruded mesh."""
    table = sphere_table()
    flux, scale = table["cmb_flux"].to_numpy(), table["u_l2"].to_numpy()
    assert np.all(np.abs(flux) <= E.CMB_FLUX_RTOL * scale), (
        f"{E.SPHERE} cmb_flux: measured {flux!r}, bound {E.CMB_FLUX_RTOL!r} * {scale!r}")


def test_the_sphere_takes_one_newton_step():
    """A prescribed load on a Maxwell Earth is linear: one Newton step per time step."""
    newton = sphere_table()["newton"].to_numpy()
    assert np.all(newton == 1), f"{E.SPHERE} newton: measured {newton!r}, expected 1"
