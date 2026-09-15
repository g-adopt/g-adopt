import pytest
import pandas as pd
from pathlib import Path
import numpy as np
import re

try:
    from gadopt_hpc_helper import system as ghpc_system
except ImportError:
    # Need to be able to import this file to gather tests, but need the
    # tests to fail if gadopt_hpc_helper is not found
    ghpc_system = "GADOPT_HPC_HELPER_IS_MISSING"


levels = [5, 6, 7]


def get_data(level, base_path=None):
    """Return the timing and iteration metrics for a given level"""

    base_path = base_path or Path()
    output_path = base_path / f"level_{level}_full.out"
    profile_path = base_path / f"profile_{level}.txt"

    if not (output_path.exists() and profile_path.exists()):
        raise FileNotFoundError(f"outputs for level {level} not found")

    # total time (profile)
    data = {}

    iterations = []

    with open(output_path, "r") as f:
        for line in f:
            if m := re.match(
                r"\s+Linear InternalVariable_ solve converged due to CONVERGED_RTOL iterations (\d+)",
                line,
            ):
                iterations.append(int(m.group(1)))

    data["iterations"] = np.mean(np.array(iterations))

    with open(profile_path, "r") as f:
        for line in f:
            if "stokes_solve:" in line:
                data["stokes_solve"] = float(line.split()[2])

            if "snes_function" not in data and line.startswith("SNESFunctionEval"):
                data["snes_function"] = float(line.split()[3])
            if "snes_jacobian" not in data and line.startswith("SNESJacobianEval"):
                data["snes_jacobian"] = float(line.split()[3])

            # space is important to avoid the PCSetup_GAMG+ entry
            if "pc_setup" not in data and line.startswith("PCSetUp "):
                data["pc_setup"] = float(line.split()[3])

            if line.startswith("Time"):
                data["total_time"] = float(line.split()[4])

    return data


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_iterations(level):
    b = Path(__file__).parent.resolve()
    mean_iterations = get_data(level, b)["iterations"]

    expected_df = pd.read_csv(b / "expected.csv", index_col="level")
    expected = expected_df.loc[level]["iterations"]

    assert abs(mean_iterations - expected) < 0.5


@pytest.mark.longtest
@pytest.mark.parametrize("level", levels)
def test_scaling_pc_setup_time(level):
    assert not isinstance(ghpc_system, str), (
        "Attempted to run longtest without gadopt_hpc_helper module"
    )

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
    assert not isinstance(ghpc_system, str), (
        "Attempted to run longtest without gadopt_hpc_helper module"
    )

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
