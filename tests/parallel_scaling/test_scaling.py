import pytest
import pandas as pd
from itertools import product
from pathlib import Path

try:
    from gadopt_hpc_helper import system as ghpc_system
except ImportError:
    # Need to be able to import this file to gather tests, but need the
    # tests to fail if gadopt_hpc_helper is not found
    ghpc_system = "GADOPT_HPC_HELPER_IS_MISSING"


from scaling import get_data

levels = [5, 6, 7]

iteration_components = {
    "energy": "ImplicitMidpoint-Equation_stage0_",
    "velocity": "Stokes_fieldsplit_0_",
    "pressure": "Stokes_fieldsplit_1_",
}


@pytest.mark.longtest
@pytest.mark.parametrize("level,component", product(levels, iteration_components.keys()))
def test_scaling_iterations(level, component):
    b = Path(__file__).parent.resolve()
    mean_iterations = get_data(level, b)[f"{component}_iterations"]

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level][f"{component}_iterations"]

    assert abs(mean_iterations - expected) < 0.5


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_pc_setup_time(level):
    assert not isinstance(ghpc_system, str), "Attempted to run longtest without gadopt_hpc_helper module"

    # The level 5 parallel scaling tests run on less than a full node, so can
    # be scheduled across a single CPU, or across multiple CPUs. This leads to
    # the level 5 tests having a much more variable runtime than the level 6 or
    # 7 tests. Therefore the level 5 tests have a higher timing tolerance.
    tol = 0.2 if level == 5 else 0.1

    b = Path(__file__).parent.resolve()
    stokes_pc_setup = get_data(level, b)["pc_setup"]

    expected_df = pd.read_csv(b / f"{ghpc_system.name}_expected.csv", index_col="level")
    expected = expected_df.loc[level]["pc_setup"]

    assert abs((expected - stokes_pc_setup) / expected) < tol


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_total_solve_time(level):
    assert not isinstance(ghpc_system, str), "Attempted to run longtest without gadopt_hpc_helper module"

    # The level 5 parallel scaling tests run on less than a full node, so can
    # be scheduled across a single CPU, or across multiple CPUs. This leads to
    # the level 5 tests having a much more variable runtime than the level 6 or
    # 7 tests. Therefore the level 5 tests have a higher timing tolerance.
    tol = 0.2 if level == 5 else 0.1

    b = Path(__file__).parent.resolve()
    solve_time = get_data(level, b)["total_time"]

    expected_df = pd.read_csv(b / f"{ghpc_system.name}_expected.csv", index_col="level")
    expected = expected_df.loc[level]["solve_time"]

    assert abs((expected - solve_time) / expected) < tol
