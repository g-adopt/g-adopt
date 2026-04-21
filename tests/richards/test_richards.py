"""Pytest tests for Richards equation benchmarks.

Follows the same pattern as tests/analytical_comparisons: reads
errors-{case}-nodes{N}-dq{p}.dat files produced by richards.py,
computes convergence rates across mesh refinements, and checks
against expected theoretical rates.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .richards import cases, get_case


# Expected convergence rates: DQ theory gives O(h^{p+1})
expected_rates = {
    0: {"rate": 1.0, "rtol": 0.1},
    1: {"rate": 2.0, "rtol": 0.1},
    2: {"rate": 3.0, "rtol": 0.1},
}

# Cases to test and whether they are longtests
enabled_cases = {
    "tracy_2d_specified_head_dg1": {},
    "tracy_2d_specified_head_dg2": {"marks": pytest.mark.longtest},
    "tracy_2d_no_flux_dg1": {},
}

param_list = []
for name, opts in enabled_cases.items():
    marks = opts.get("marks")
    if marks:
        param_list.append(pytest.param(name, marks=marks))
    else:
        param_list.append(name)


@pytest.mark.parametrize("case_name", param_list)
def test_convergence_rate(case_name):
    """Verify spatial convergence rate across mesh refinements."""
    b = Path(__file__).parent.resolve()
    config = get_case(cases, case_name)
    levels = config["levels"]
    degree = config["degree"]

    cols = ["l2error_h", "l2error_theta", "l2anal_h", "l2anal_theta"]

    dats = []
    for nodes in levels:
        datfile = b / f"errors-{case_name}-nodes{nodes}-dq{degree}.dat"
        if not datfile.exists():
            pytest.skip(f"Missing {datfile.name}. Run: python richards.py {case_name} {nodes} {degree} {config['bc_type']}")
        dats.append(pd.read_csv(datfile, sep=" ", header=None))

    dat = pd.concat(dats)
    dat.columns = cols
    dat.insert(0, "nodes", levels)
    dat = dat.set_index("nodes")

    # Compute relative errors
    rel_errors = dat["l2error_h"] / dat["l2anal_h"]

    # Compute convergence rates between successive levels
    rates = np.log2(rel_errors.values[:-1] / rel_errors.values[1:])

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Tracy 2D Convergence: {case_name}")
    print(f"{'=' * 60}")
    print(f"{'Nodes':>8} | {'L2 error':>12} | {'Relative':>12} | {'Rate':>8}")
    print("-" * 50)
    for i, nodes in enumerate(levels):
        rate_str = f"{rates[i-1]:8.2f}" if i > 0 else "       -"
        print(f"{nodes:>8} | {dat.loc[nodes, 'l2error_h']:>12.4e} | "
              f"{rel_errors.loc[nodes]:>12.4e} | {rate_str}")
    print(f"{'=' * 60}\n")

    # Check that the finest-mesh rate meets the theoretical expectation
    expected = expected_rates[degree]
    finest_rate = rates[-1]
    assert finest_rate >= expected["rate"] * (1 - expected["rtol"]), (
        f"{case_name}: finest convergence rate {finest_rate:.2f} below "
        f"expected {expected['rate']:.1f} (with {expected['rtol']*100:.0f}% tolerance)"
    )
