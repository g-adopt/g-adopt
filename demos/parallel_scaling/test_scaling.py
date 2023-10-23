import pytest
import re
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

levels = [5, 6, 7]

iteration_components = {
    "energy": "ImplicitMidpoint-EnergyEquation_stage0_",
    "velocity": "Stokes_fieldsplit_0_",
    "pressure": "Stokes_fieldsplit_1_",
}


@pytest.mark.longtest
@pytest.mark.parametrize("level,component", product(levels, iteration_components.keys()))
def test_scaling_iterations(level, component):
    b = Path(__file__).parent.resolve()
    iteration_str = "Linear {} solve converged due to".format(iteration_components[component])

    with open(b / f"level_{level}_full.out", "r") as f:
        iterations = [l.split()[-1] for l in f if iteration_str in l]
        mean_iterations = np.mean(np.array(iterations, dtype=float))

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level][f"{component}_iterations"]

    assert abs(mean_iterations - expected) < 0.5


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_pc_setup_time(level):
    b = Path(__file__).parent.resolve()

    with open(b / f"profile_{level}.txt") as f:
        solve_lines = [l for l in f if re.match(r"PCSetUp\s", l)]

    stokes_pc_setup = float(solve_lines[0].split()[3])

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level]["pc_setup"]

    assert abs((expected - stokes_pc_setup) / expected) < 0.1


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_total_solve_time(level):
    b = Path(__file__).parent.resolve()
    with open(b / f"profile_{level}.txt") as f:
        solve_lines = [l for l in f if l.startswith("Time")]

    solve_time = float(solve_lines[0].split()[2])

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level]["solve_time"]

    assert abs((expected - solve_time) / expected) < 0.1
