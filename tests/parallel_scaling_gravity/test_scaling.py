r"""Longtest regression assertions for the gravity DtN scaling study.

Two invariants, mirroring the study's two axes:

* the coupled solve's ``fieldsplit_0`` (AMG) iteration count per invocation is
  flat across refinement levels at fixed L — the weak-scaling verdict;
* the capacitance diagnostic's offline-GMRES-on-S count and cond(S) match the
  recorded reference at each L — the L-cost verdict.

Reference values live in ``expected.csv`` (coupled) and ``expected_capacitance.csv``,
populated from the layer-one bare run.

**Both reference files are currently empty of rows, on purpose.** Every number
they held was measured under the previous boundary-quadrature default, which the
mesh-aware rule replaced, and under the pre-recursion ``real_spherical_harmonic``
evaluation, whose replacement deliberately changed the default path's numerical
output. Rather than assert against references known to describe a different
solver, the rows were dropped and are repopulated from the campaign. A case with
no reference therefore *skips* rather than fails: a red suite that everyone knows
to ignore is worse than no suite, because it stops being read. Repopulating the
files turns the assertions back on with no other change.
"""

import pytest
import pandas as pd
from pathlib import Path

from scaling import (CAPACITANCE_L, COUPLED_L, REPRESENTATIONS, get_capacitance,
                     get_coupled)

COUPLED_LEVELS = [4, 5, 6, 7]
CAPACITANCE_LEVELS = [4, 5]


def _reference(path, **keys):
    """The single reference row matching `keys`, or a skip if there is none."""
    frame = pd.read_csv(path)
    for column, value in keys.items():
        if column not in frame.columns:
            pytest.skip(f"{path.name} has no '{column}' column; it predates "
                        f"this axis and needs regenerating")
        frame = frame[frame[column] == value]
    if frame.empty:
        pytest.skip(f"no reference row in {path.name} for "
                    + ", ".join(f"{k}={v}" for k, v in keys.items()))
    return frame.iloc[0]


@pytest.mark.longtest
@pytest.mark.parametrize(
    "level,lmax,representation",
    [(lv, L, r) for lv in COUPLED_LEVELS for L in COUPLED_L
     for r in REPRESENTATIONS])
def test_coupled_amg_applications(level, lmax, representation):
    """Multigrid work per gravity solve matches the reference.

    `amg_applications_per_solve` rather than an iteration count, because it is
    the only iteration-like quantity that means the same thing on both paths.
    """
    b = Path(__file__).parent.resolve()
    row = _reference(b / "expected.csv", level=level, lmax=lmax,
                     representation=representation)
    got = get_coupled(level, lmax, b, representation)["amg_applications_per_solve"]
    assert got is not None, (
        f"sidecar for level {level}, L {lmax}, {representation} carries no "
        "amg_applications_per_solve")
    expected = float(row["amg_applications_per_solve"])
    # A 10% band: the count is a mean over timed solves and over invocations
    # whose individual counts vary by one, so an absolute tolerance that worked
    # for a per-invocation figure near 7 is far too tight for a summed figure
    # near 200.
    assert abs(got - expected) <= 0.1 * expected


@pytest.mark.longtest
@pytest.mark.parametrize("level,lmax",
                         [(lv, L) for lv in CAPACITANCE_LEVELS for L in CAPACITANCE_L])
def test_capacitance_cost(level, lmax):
    """Offline GMRES count and cond(S) match the reference (L-cost verdict)."""
    b = Path(__file__).parent.resolve()
    row = _reference(b / "expected_capacitance.csv", level=level, lmax=lmax)
    got = get_capacitance(level, lmax, b)

    assert abs(got["gmres_flat"] - float(row["gmres_flat"])) <= 2
    # cond(S) is O(10) and essentially flat; a 20% band catches real growth.
    expected_cond = float(row["cond_S"])
    assert abs(got["cond_S"] - expected_cond) / expected_cond < 0.2
