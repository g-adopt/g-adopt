import pytest
import pandas as pd
from itertools import product
from pathlib import Path

from scaling import get_data

levels = [5, 6, 7]

iteration_components = {
    "energy": "ImplicitMidpoint-EnergyEquation_stage0_",
    "velocity": "Stokes_fieldsplit_0_",
    "pressure": "Stokes_fieldsplit_1_",
}


@pytest.mark.longtest
@pytest.mark.parametrize(
    "level,component", product(levels, iteration_components.keys())
)
def test_scaling_iterations(level, component):
    b = Path(__file__).parent.resolve()
    mean_iterations = get_data(level, b)[f"{component}_iterations"]

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level][f"{component}_iterations"]

    assert abs(mean_iterations - expected) < 0.5


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_pc_setup_time(level):
    b = Path(__file__).parent.resolve()
    stokes_pc_setup = get_data(level, b)["pc_setup"]

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level]["pc_setup"]

    assert abs((expected - stokes_pc_setup) / expected) < 0.1


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_total_solve_time(level):
    b = Path(__file__).parent.resolve()
    solve_time = get_data(level, b)["total_time"]

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level]["solve_time"]

    assert abs((expected - solve_time) / expected) < 0.1
