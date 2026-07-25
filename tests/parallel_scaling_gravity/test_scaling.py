r"""Longtest regression assertions for the gravity DtN scaling study.

Two invariants, mirroring the study's two axes:

* the coupled solve's ``fieldsplit_0`` (AMG) iteration count per invocation is
  flat across refinement levels at fixed L — the weak-scaling verdict;
* the capacitance diagnostic's offline-GMRES-on-S count and cond(S) match the
  recorded reference at each L — the L-cost verdict.

Reference values live in ``expected.csv`` (coupled) and ``expected_capacitance.csv``,
populated from the layer-one bare run.
"""

import pytest
import pandas as pd
from pathlib import Path

from scaling import COUPLED_L, CAPACITANCE_L, get_coupled, get_capacitance

COUPLED_LEVELS = [4, 5, 6, 7]
CAPACITANCE_LEVELS = [4, 5]


@pytest.mark.longtest
@pytest.mark.parametrize("level,lmax",
                         [(lv, L) for lv in COUPLED_LEVELS for L in COUPLED_L])
def test_coupled_fieldsplit0_flat(level, lmax):
    """fieldsplit_0 iterations/invocation match the reference (AMG weak-scaling)."""
    b = Path(__file__).parent.resolve()
    got = get_coupled(level, lmax, b)["fieldsplit_0_iterations"]

    expected_df = pd.read_csv(b / "expected.csv")
    row = expected_df[(expected_df.level == level) & (expected_df.lmax == lmax)]
    assert not row.empty, f"no expected row for level {level}, L {lmax}"
    expected = float(row["fieldsplit_0_iterations"].iloc[0])

    # AMG count is small; an absolute tolerance of 1 iteration is generous and
    # catches drift (the whole point of the weak-scaling axis).
    assert abs(got - expected) < 1.0


@pytest.mark.longtest
@pytest.mark.parametrize("level,lmax",
                         [(lv, L) for lv in CAPACITANCE_LEVELS for L in CAPACITANCE_L])
def test_capacitance_cost(level, lmax):
    """Offline GMRES count and cond(S) match the reference (L-cost verdict)."""
    b = Path(__file__).parent.resolve()
    got = get_capacitance(level, lmax, b)

    expected_df = pd.read_csv(b / "expected_capacitance.csv")
    row = expected_df[(expected_df.level == level) & (expected_df.lmax == lmax)]
    assert not row.empty, f"no expected row for level {level}, L {lmax}"

    assert abs(got["gmres_flat"] - float(row["gmres_flat"].iloc[0])) <= 2
    # cond(S) is O(10) and essentially flat; a 20% band catches real growth.
    expected_cond = float(row["cond_S"].iloc[0])
    assert abs(got["cond_S"] - expected_cond) / expected_cond < 0.2
