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
    return column.endswith(tuple(f"_{name}" for name in
                                 ("build_count", "column_builds", "assembly_count",
                                  "elimination_count")))


def recorded_rtol(entry):
    """The relative tolerance of the recorded values of `entry`."""
    return (E.RECORDED_RTOL_DIRECT if entry in E.FORWARD_DIRECT
            else E.RECORDED_RTOL_ITERATIVE)


def compare_values(measured, expected, scale, rtol, label):
    """Assert that two value columns agree; return nothing.

    A NaN in the expected column must be NaN in the measured one (the entry
    has no such quantity). An expected value below `ROUNDOFF_FRACTION *
    scale` is round-off and is compared absolutely with that bound; every
    other value relatively with `rtol`.

    Args:
      measured, expected: arrays of one column, one value per step.
      scale: the displacement norm `u_l2` of each step.
      rtol: the relative tolerance.
      label: the entry and the column, for the message.
    """
    for k, (m, e, s) in enumerate(zip(measured, expected, scale)):
        if math.isnan(e):
            assert math.isnan(m), f"{label} step {k + 1}: measured {m!r}, expected NaN"
            continue
        floor = E.ROUNDOFF_FRACTION * s
        if abs(e) < floor:
            assert abs(m - e) <= floor, (
                f"{label} step {k + 1}: measured {m!r}, expected {e!r}, "
                f"absolute tolerance {floor!r}")
        else:
            assert abs(m - e) <= rtol * abs(e), (
                f"{label} step {k + 1}: measured {m!r}, expected {e!r}, "
                f"relative tolerance {rtol!r}")


@pytest.mark.parametrize("entry", E.FORWARD_RECORDED)
def test_recorded_values(entry):
    """Every column of every step agrees with the recorded values.

    Values: `recorded_rtol(entry)`, with the round-off floor of
    `compare_values`. Newton count: within 1 of the stored count. Linear
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
        elif column == "operator_version" or is_counter(column):
            assert np.array_equal(m, e, equal_nan=True), (
                f"{label}: measured {m!r}, expected {e!r}")
        else:
            compare_values(m, e, expected["u_l2"].to_numpy(), rtol, label)


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


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_net_sheet_mass_is_set_by_the_frame(entry):
    """The net sheet mass equals the value the converged `Shift` row implies.

    The `Shift` row with the centre-of-mass column gives the identity
    `int sigma dS = sum_i lambda_i int x_i dsigma/dShift dS / g_s`
    (`gia_march._mass_terms`). The difference must be at most the part of
    the row that Newton leaves (`mass_newton`) plus 1e-10 of the absolute
    sheet mass for the round-off of the integrals. The net mass is also
    within the bound of `test_sea_level.py::TestLiveCoastlineSolve::
    test_mass_is_conserved_to_order_lambda` (`mass_bound`). Entries without
    sea level have NaN in these columns and are skipped.
    """
    table = load_log(entry)
    if table["sheet_mass"].isna().all():
        pytest.skip(f"{entry} has no sea level")
    for _, row in table.iterrows():
        gap = abs(row["sheet_mass"] - row["mass_identity"])
        allowed = row["mass_newton"] + 1e-10 * row["sheet_abs_mass"]
        assert gap <= allowed, (
            f"{entry} step {row['step']:.0f}: |int sigma - identity| = {gap!r}, "
            f"allowed {allowed!r} (int sigma {row['sheet_mass']!r})")
        assert abs(row["sheet_mass"]) <= row["mass_bound"], (
            f"{entry} step {row['step']:.0f}: |int sigma| = {abs(row['sheet_mass'])!r}, "
            f"bound {row['mass_bound']!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_centre_of_mass_holds(entry):
    """`|D_i| <= DIPOLE_RTOL * sheet_moment_i` at every step, both directions.

    `D_i` is the first mass moment of the mantle and of every sheet;
    `sheet_moment_i`, the integral of `|x_i sigma|` over the sheets, is the
    moment that the displacement has to cancel, and it must be nonzero.
    """
    table = load_log(entry)
    if table["dipole_0"].isna().all():
        pytest.skip(f"{entry} has no centre-of-mass constraint")
    for i in range(2):
        moment = table[f"sheet_moment_{i}"].to_numpy()
        dipole = table[f"dipole_{i}"].to_numpy()
        assert np.all(moment > 0), f"{entry} sheet_moment_{i}: {moment!r}"
        assert np.all(np.abs(dipole) <= E.DIPOLE_RTOL * moment), (
            f"{entry} dipole_{i}: measured {dipole!r}, "
            f"bound {E.DIPOLE_RTOL!r} * {moment!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_core_volume_is_kept(entry):
    """`|int u . n dS_cmb| <= CMB_FLUX_RTOL * u_l2`: the fluid core keeps its volume."""
    table = load_log(entry)
    flux, scale = table["cmb_flux"].to_numpy(), table["u_l2"].to_numpy()
    assert np.all(np.abs(flux) <= E.CMB_FLUX_RTOL * scale), (
        f"{entry} cmb_flux: measured {flux!r}, bound {E.CMB_FLUX_RTOL!r} * {scale!r}")


@pytest.mark.parametrize("entry", list(E.FORWARD))
def test_newton_count(entry):
    """At most `MAX_NEWTON_MOVING` Newton iterations per step."""
    newton = load_log(entry)["newton"].to_numpy()
    assert np.all(newton <= E.MAX_NEWTON_MOVING), (
        f"{entry} newton: measured {newton!r}, at most {E.MAX_NEWTON_MOVING}")
