"""Long tests for the Burgers weak-scaling jobs on Gadi.

Every (level, configuration) job runs the substituted reference solver and one
coupled configuration on the same mesh. The tests check, per job, the GAMG
V-cycles per step of both solvers against ``expected.csv`` and the PCSetUp
and stage times against ``<system>_expected.csv``. The expected values are
from the 2026-09-09 campaign (jobs 178561569 to 178562020 on Gadi).

The jobs are launched by ``gadopt_hpc_helper`` from ``meta.py`` and
``run.template``. Run the checks with ``pytest -m longtest`` in this directory
once the outputs are in place.
"""

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

from burgers_scaling import get_data

levels = [5, 6, 7]
configs = ["multiplicative", "schur-a11", "schur-substituted", "static-condensation"]
solvers = ["substituted", "coupled"]


def _expected(name, level, config):
    b = Path(__file__).parent.resolve()
    df = pd.read_csv(b / name, index_col=["level", "config"])
    return df.loc[(level, config)]


def _iteration_tolerance(expected):
    # Half an iteration for the eliminated routes. The multiplicative preset
    # sums hundreds of inner solves over ten sweeps, so allow five percent.
    return max(0.5, 0.05 * expected)


@pytest.mark.longtest
@pytest.mark.parametrize("level,config,solver", product(levels, configs, solvers))
def test_scaling_iterations(level, config, solver):
    b = Path(__file__).parent.resolve()
    data = get_data(level, config, b)
    name = "substituted" if solver == "substituted" else config
    measured = data[f"{name}_displacement_iterations"]
    expected = _expected("expected.csv", level, config)[f"{solver}_iterations"]
    assert abs(measured - expected) < _iteration_tolerance(expected)


@pytest.mark.longtest
@pytest.mark.parametrize("level,config,solver", product(levels, configs, solvers))
def test_scaling_pc_setup_time(level, config, solver):
    assert not isinstance(ghpc_system, str), "Attempted to run longtest without gadopt_hpc_helper module"

    # Level 5 runs on one node and shares it with other jobs, so its timings
    # vary more than the multi-node levels.
    tol = 0.2 if level == 5 else 0.1

    b = Path(__file__).parent.resolve()
    data = get_data(level, config, b)
    name = "substituted" if solver == "substituted" else config
    measured = data[f"{name}_pc_setup"]
    expected = _expected(f"{ghpc_system.name}_expected.csv", level, config)[f"{solver}_pc_setup"]
    assert abs((expected - measured) / expected) < tol


@pytest.mark.longtest
@pytest.mark.parametrize("level,config,solver", product(levels, configs, solvers))
def test_scaling_total_solve_time(level, config, solver):
    assert not isinstance(ghpc_system, str), "Attempted to run longtest without gadopt_hpc_helper module"

    tol = 0.2 if level == 5 else 0.1

    b = Path(__file__).parent.resolve()
    data = get_data(level, config, b)
    name = "substituted" if solver == "substituted" else config
    measured = data[f"{name}_solve"]
    expected = _expected(f"{ghpc_system.name}_expected.csv", level, config)[f"{solver}_solve"]
    assert abs((expected - measured) / expected) < tol
