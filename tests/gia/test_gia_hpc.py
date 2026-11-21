"""Assertions of the weekly HPC entry of the coupled GIA solver.

`gia_sphere_submesh.py` marches `SEA` on a tetrahedral sphere with a mantle
submesh under `doit run_case_hpc` (the weekly long-test run on Gadi) and
writes one row of diagnostics per step to `out/sphere-submesh.log`. This
file reads that table and asserts:

- each coupling does something: the sea-level shift, the three
  centre-of-mass multipliers and the three rotation unknowns are above a
  floor at the last step, so that the other assertions do not compare
  round-off;
- conservation and constraints, from exact identities: the net sheet mass
  that the converged `Shift` row implies, the centre of mass in all three
  directions, the volume of the fluid core, the Newton count;
- recorded values: every column against `expected/sphere-submesh.txt`,
  which detects a change of the results (not a wrong result). The file is
  recorded on Gadi; until it exists, that test skips.

A missing table fails every test: on the weekly run it means that the job
failed before its first step. On a machine that does not run the entry,
set `GADOPT_GIA_LOCAL=1` (`gia_entries.LOCAL_ENV`) and the tests skip. The
module is marked `longtest`, so PR CI deselects it. It imports no
Firedrake, because the weekly run collects it on a login node.
"""

import math

import numpy as np
import pandas as pd
import pytest

import gia_entries as E

pytestmark = pytest.mark.longtest


def load_log():
    """The per-step table of the weekly entry.

    A missing table fails the test, or skips it where
    `gia_entries.LOCAL_ENV` is 1 (`gia_entries.require_output`).
    """
    path = E.CASE_DIR / E.WEEKLY_LOG
    E.require_output(path, E.WEEKLY_ENTRY)
    return pd.read_csv(path, sep=r"\s+")


def mesh_record():
    """The mesh record of the run as text, for the messages; empty if missing."""
    path = E.CASE_DIR / E.WEEKLY_MESH_RECORD
    return path.read_text() if path.exists() else ""


def test_every_step_ran():
    """The entry wrote one row per step and moved the Earth.

    The driver writes the table after every step, so a job that fails after
    some steps leaves fewer rows, and this test names that case.
    """
    table = load_log()
    assert len(table) == E.WEEKLY_STEP_COUNT, (
        f"{E.WEEKLY_LOG}: {len(table)} rows, expected {E.WEEKLY_STEP_COUNT}")
    u_l2 = table["u_l2"].to_numpy()
    assert np.all(u_l2 > 0), f"u_l2: {u_l2!r}"


@pytest.mark.parametrize("column", E.WEEKLY_COUPLING_COLUMNS)
def test_each_coupling_acts(column):
    """At the last step the coupling quantity is above `COUPLING_FLOOR * u_l2`.

    A recorded value of a quantity at round-off would check nothing. The
    load is centred off every coordinate axis, so its first moment reaches
    each centre-of-mass row and its second moment each rotation row; the
    polar-wander pair `m1`, `m2` exists only in 3-D.
    """
    last = load_log().iloc[-1]
    floor = E.COUPLING_FLOOR * last["u_l2"]
    assert abs(last[column]) > floor, (
        f"{column}: measured {last[column]!r}, must exceed {floor!r} "
        f"({E.COUPLING_FLOOR!r} * u_l2)")


def test_net_sheet_mass_is_set_by_the_frame():
    """The net sheet mass equals the value the converged `Shift` row implies.

    The `Shift` row with the centre-of-mass column gives the identity
    `int sigma dS = sum_i lambda_i int x_i dsigma/dShift dS / g_s`
    (`gia_march._mass_terms`). The difference must be at most the part of
    the row that Newton leaves (`mass_newton`) plus 1e-10 of the absolute
    sheet mass for the round-off of the integrals. The net mass itself is at
    most `MASS_LEAK_RTOL` of the absolute sheet mass. The same assertions
    as `test_gia_forward.py::test_net_sheet_mass_is_set_by_the_frame`.
    """
    for _, row in load_log().iterrows():
        gap = abs(row["sheet_mass"] - row["mass_identity"])
        allowed = row["mass_newton"] + 1e-10 * row["sheet_abs_mass"]
        assert gap <= allowed, (
            f"step {row['step']:.0f}: |int sigma - identity| = {gap!r}, allowed "
            f"{allowed!r} (int sigma {row['sheet_mass']!r}, identity "
            f"{row['mass_identity']!r})")
        leak = E.MASS_LEAK_RTOL * row["sheet_abs_mass"]
        assert abs(row["sheet_mass"]) <= leak, (
            f"step {row['step']:.0f}: |int sigma| = {abs(row['sheet_mass'])!r}, at most "
            f"{E.MASS_LEAK_RTOL!r} * int |sigma| = {leak!r}")


def test_centre_of_mass_holds():
    """`|D_i| <= DIPOLE_RTOL_ITERATIVE * sheet_moment_i` at every step, three directions.

    `D_i` is the first mass moment of the mantle and of every sheet;
    `sheet_moment_i`, the integral of `|x_i sigma|` over the sheets, is the
    moment that the displacement has to cancel, and it must be nonzero.
    """
    table = load_log()
    for i in range(3):
        moment = table[f"sheet_moment_{i}"].to_numpy()
        dipole = table[f"dipole_{i}"].to_numpy()
        assert np.all(moment > 0), f"sheet_moment_{i}: {moment!r}"
        bound = E.DIPOLE_RTOL_ITERATIVE * moment
        assert np.all(np.abs(dipole) <= bound), (
            f"dipole_{i}: measured {dipole!r}, bound {bound!r} "
            f"({E.DIPOLE_RTOL_ITERATIVE!r} * sheet_moment_{i})")


def test_core_volume_is_kept():
    """`|int u . n dS_cmb| <= CMB_FLUX_RTOL * u_l2`: the fluid core keeps its volume."""
    table = load_log()
    flux, scale = table["cmb_flux"].to_numpy(), table["u_l2"].to_numpy()
    assert np.all(np.abs(flux) <= E.CMB_FLUX_RTOL * scale), (
        f"cmb_flux: measured {flux!r}, bound {E.CMB_FLUX_RTOL!r} * {scale!r}")


def test_newton_count():
    """At most `MAX_NEWTON_MOVING` Newton iterations per step (a moving coastline)."""
    newton = load_log()["newton"].to_numpy()
    assert np.all(newton <= E.MAX_NEWTON_MOVING), (
        f"newton: measured {newton!r}, at most {E.MAX_NEWTON_MOVING}")


# -- Recorded values ---------------------------------------------------------------

def is_count(column):
    """Whether `column` is a count, compared by its own rule and not as a value."""
    counters = tuple(f"_{name}" for name in E.COUNTER_SUFFIXES)
    return column in ("newton", "linear", "operator_version") or column.endswith(counters)


def scale_of(row, column):
    """The scale of `column` in one expected row (`gia_entries.SCALE_OF`), else 0.

    The rule of `test_gia_forward.scale_of`, for one row: a small quantity
    is compared against the size it could have, not only against its own
    value.
    """
    rule = E.SCALE_OF.get(column)
    if rule is None:
        return 0.0
    kind, *names = rule
    values = [float(row[name]) for name in names]
    if kind == "column":
        return abs(values[0])
    if kind == "product":
        return abs(values[0] * values[1])
    if kind == "l2":
        return math.sqrt(float(row["re_area"]) * abs(values[0]))
    raise ValueError(rule)


def test_recorded_values():
    """Every column of every step agrees with the recorded values.

    Values: `|m - e| <= WEEKLY_RTOL (|e| + scale)`, with the scale of the
    column (`scale_of`). Newton count: within 1 of the stored count. Linear
    count: at most the stored count plus the larger of 2 and 20 percent.
    Cache counters and `operator_version`: equal. The columns made of final
    residual norms (`RESIDUAL_COLUMNS`) and the bound that contains them
    (`BOUND_COLUMNS`) are not compared: they do not reproduce between runs
    of the iterative preset.

    The recorded values hold for one mesh. Another gmsh can write other
    tetrahedra for the same seed; the messages print the mesh record of the
    run (gmsh version, seed, MD5) to show such a change.
    """
    measured = load_log()
    path = E.CASE_DIR / E.WEEKLY_EXPECTED
    if not path.exists():
        pytest.skip(f"{E.WEEKLY_EXPECTED} is not recorded yet; measured table:\n"
                    f"{measured.to_string()}\nmesh:\n{mesh_record()}")
    expected = pd.read_csv(path, sep=r"\s+", comment="#")
    mesh = f"\nmesh of this run:\n{mesh_record()}"
    assert list(measured.columns) == list(expected.columns), (
        f"columns {list(measured.columns)!r}, expected {list(expected.columns)!r}{mesh}")
    assert len(measured) == len(expected), (
        f"{len(measured)} steps, expected {len(expected)}{mesh}")
    rtol = E.WEEKLY_RTOL
    for column in expected.columns:
        if column in (*E.RESIDUAL_COLUMNS, *E.BOUND_COLUMNS):
            continue
        m, e = measured[column].to_numpy(float), expected[column].to_numpy(float)
        if column == "newton":
            assert np.all(np.abs(m - e) <= 1), f"newton: measured {m!r}, expected {e!r} +- 1"
        elif column == "linear":
            limit = e + np.maximum(2, 0.2 * e)
            assert np.all(m <= limit), f"linear: measured {m!r}, at most {limit!r}"
        elif is_count(column):
            assert np.array_equal(m, e, equal_nan=True), (
                f"{column}: measured {m!r}, expected {e!r}{mesh}")
        else:
            # Plain floats, so that the messages print `0.4` and not
            # `np.float64(0.4)`.
            for k, (mk, ek) in enumerate(zip(m.tolist(), e.tolist())):
                if math.isnan(ek):
                    assert math.isnan(mk), f"{column} row {k}: measured {mk!r}, expected NaN"
                    continue
                scale = scale_of(expected.iloc[k], column)
                allowed = rtol * (abs(ek) + scale)
                assert abs(mk - ek) <= allowed, (
                    f"{column} row {k}: measured {mk!r}, expected {ek!r}, tolerance "
                    f"{allowed!r} (rtol {rtol!r}, scale {scale!r}){mesh}")
