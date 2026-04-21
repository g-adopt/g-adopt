"""Parallel-scaling assertions for the Richards default iterative presets.

Collected by the g-adopt longtest runner on Gadi. Three metrics are
checked per (case, solver, level) triple:

* ``linear_iterations`` -- mean Krylov iteration count across all time
  steps, parsed from the driver's stdout.
* ``pc_setup`` -- wall time spent in ``PCSetUp`` according to PETSc's
  ``-log_view``, parsed from ``profile_<tag>.txt``.
* ``solve_time`` -- wall time reported by ``-log_view``.

Reference values are kept in two CSVs:

* ``expected.csv`` -- iteration count is hardware-independent.
* ``gadi_expected.csv`` -- timings are system-specific (Sapphire Rapids).

Rows whose expected value is ``nan`` are skipped rather than asserted,
so the test file is usable in the period between landing this scaffolding
and filling the CSVs from the first successful scaling run.

The unit test at the bottom exercises ``scaling.get_data`` against a
synthetic output pair, so parser regressions are caught without needing
an HPC run.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

try:
    from gadopt_hpc_helper import system as ghpc_system
except ImportError:
    # Keep import-time collection working even when the helper is missing;
    # longtests are skipped at run time in that case.
    ghpc_system = "GADOPT_HPC_HELPER_IS_MISSING"

from richards_scaling import all_triples, get_data


_HERE = Path(__file__).parent.resolve()

# Tolerances are inherited from tests/parallel_scaling/test_scaling.py and
# expanded slightly because Richards iteration counts are more variable
# than Stokes (nonlinear residual sensitivity).
_ITER_TOL = 1.0      # absolute count
_TIME_TOL = 0.20     # fractional


def _load_expected(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(_HERE / csv_name).set_index(["case", "level", "solver"])


def _expected_row(df: pd.DataFrame, csv_name: str,
                  case: str, level: int, solver: str):
    try:
        return df.loc[(case, level, solver)]
    except KeyError:
        pytest.skip(
            f"no expected row for ({case}, {level}, {solver}) in {csv_name}"
        )


@pytest.mark.longtest
def test_expected_csv_covers_all_triples():
    """Every triple enumerated by ``all_triples`` has an expected.csv row.

    Guards against a silent regression where a future rewrite of
    ``expected.csv`` drops rows -- the per-triple tests ``pytest.skip`` on
    ``KeyError``, which would otherwise leave CI green on incomplete data.
    """
    expected = _load_expected("expected.csv")
    missing = [t for t in all_triples() if t not in expected.index]
    assert not missing, f"expected.csv missing rows for {missing}"


@pytest.mark.longtest
def test_gadi_expected_csv_covers_all_triples():
    """Same coverage guard for the system-specific timing CSV."""
    if isinstance(ghpc_system, str):
        pytest.skip("gadopt_hpc_helper not available; no system CSV to check")
    expected = _load_expected(f"{ghpc_system.name}_expected.csv")
    missing = [t for t in all_triples() if t not in expected.index]
    assert not missing, (
        f"{ghpc_system.name}_expected.csv missing rows for {missing}"
    )


def _skip_if_nan(value: float, metric: str) -> None:
    if isinstance(value, float) and math.isnan(value):
        pytest.skip(
            f"expected {metric} is nan -- fill in after the first "
            "successful run on the target system"
        )


@pytest.mark.longtest
@pytest.mark.parametrize("case,level,solver", list(all_triples()))
def test_linear_iterations(case, level, solver):
    """Mean linear-iteration count stays within ±1 of the recorded value."""
    expected_df = _load_expected("expected.csv")
    expected = float(_expected_row(expected_df, "expected.csv",
                                   case, level, solver)["linear_iterations"])
    _skip_if_nan(expected, "linear_iterations")

    data = get_data(case, solver, level, _HERE)
    measured = data["linear_iterations"]
    assert not math.isnan(measured), (
        f"no linear-iteration data parsed for {case} / {solver} / l={level}"
    )
    assert abs(measured - expected) < _ITER_TOL, (
        f"{case}/{solver}/l={level}: mean iterations {measured:.2f} "
        f"vs expected {expected:.2f} (tol ±{_ITER_TOL})"
    )


@pytest.mark.longtest
@pytest.mark.parametrize("case,level,solver", list(all_triples()))
def test_pc_setup_time(case, level, solver):
    assert not isinstance(ghpc_system, str), (
        "attempted to run longtest without gadopt_hpc_helper module"
    )
    csv_name = f"{ghpc_system.name}_expected.csv"
    expected_df = _load_expected(csv_name)
    expected = float(_expected_row(expected_df, csv_name,
                                   case, level, solver)["pc_setup"])
    _skip_if_nan(expected, "pc_setup")

    data = get_data(case, solver, level, _HERE)
    measured = data["pc_setup"]
    assert not math.isnan(measured), (
        f"no pc_setup data parsed for {case} / {solver} / l={level}"
    )
    assert expected > 0, (
        f"expected timing for {case}/{solver}/l={level} is non-positive"
    )
    assert abs((measured - expected) / expected) < _TIME_TOL, (
        f"{case}/{solver}/l={level}: pc_setup {measured:.2f}s "
        f"vs expected {expected:.2f}s (tol {_TIME_TOL*100:.0f}%)"
    )


@pytest.mark.longtest
@pytest.mark.parametrize("case,level,solver", list(all_triples()))
def test_total_solve_time(case, level, solver):
    assert not isinstance(ghpc_system, str), (
        "attempted to run longtest without gadopt_hpc_helper module"
    )
    csv_name = f"{ghpc_system.name}_expected.csv"
    expected_df = _load_expected(csv_name)
    expected = float(_expected_row(expected_df, csv_name,
                                   case, level, solver)["solve_time"])
    _skip_if_nan(expected, "solve_time")

    data = get_data(case, solver, level, _HERE)
    measured = data["solve_time"]
    assert not math.isnan(measured), (
        f"no solve_time data parsed for {case} / {solver} / l={level}"
    )
    assert expected > 0, (
        f"expected timing for {case}/{solver}/l={level} is non-positive"
    )
    assert abs((measured - expected) / expected) < _TIME_TOL, (
        f"{case}/{solver}/l={level}: solve_time {measured:.2f}s "
        f"vs expected {expected:.2f}s (tol {_TIME_TOL*100:.0f}%)"
    )


# ---------------------------------------------------------------------------
# Parser unit test -- runs on the dev machine, no longtest marker.
# ---------------------------------------------------------------------------
_SAMPLE_OUT = """\
 DOFs: 2230272  (mesh 120x120x156, DQ1, preset=vlumping)
   0 SNES Function norm 2.345e+01
   0 KSP Residual norm 5.678e+00
   1 KSP Residual norm 3.012e-01
  Linear solve converged due to CONVERGED_RTOL iterations 9
   1 SNES Function norm 3.451e-02
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
 step 1/3 | t=300.0s | wall=5.12s | NL=1 | L=9
   0 SNES Function norm 1.123e-02
  Linear solve converged due to CONVERGED_RTOL iterations 11
   1 SNES Function norm 4.567e-04
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
 step 2/3 | t=600.0s | wall=4.98s | NL=1 | L=11
   0 SNES Function norm 6.789e-03
  Linear solve converged due to CONVERGED_RTOL iterations 10
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
 step 3/3 | t=900.0s | wall=5.03s | NL=1 | L=10
"""

# Realistic -log_view excerpt. The event columns follow PETSc's
# schema: name, count-max, count-ratio, time-max, time-ratio, ...
# The "Time (sec):" summary block sits at the top: max, ratio, avg.
_SAMPLE_PROFILE = """\
---------------------------------------------- PETSc Performance Summary ----------------------------------------------

./driver on a arch-linux-c-opt with 104 processors, by sg8812 2026-04-20
Using Petsc Release Version 3.22.0

                         Max       Max/Min     Avg       Total
Time (sec):           1.234e+02  1.000      1.230e+02  1.231e+02
Objects:              7.200e+02  1.000
Flops:                0.000e+00  0.000      0.000e+00  0.000e+00
MPI Reductions:       0.000e+00

------------------------------------------------------------------------------------------------------------------------

Event                Count     Count-Ratio  Time (Max)  Time-Ratio  Mflop/s   ...
------------------------------------------------------------------------------------------------------------------------
SNESSolve              3        1.00        1.0123e+02  1.00  0.00e+00   ...
KSPSolve               3        1.00        9.8765e+01  1.00  0.00e+00   ...
PCSetUp                3        1.00        8.7654e+00  1.00  0.00e+00   ...
PCApply               30        1.00        1.2345e+01  1.00  0.00e+00   ...
"""


@pytest.fixture
def synthetic_run(tmp_path):
    """Drop a stdout + profile pair into tmp_path and return the tag."""
    case, solver, level = "cockett", "vlumping", 1
    (tmp_path / f"{case}_{solver}_{level}.out").write_text(_SAMPLE_OUT)
    (tmp_path / f"profile_{case}_{solver}_{level}.txt").write_text(_SAMPLE_PROFILE)
    return case, solver, level, tmp_path


def test_get_data_parses_iterations(synthetic_run):
    case, solver, level, tmp_path = synthetic_run
    data = get_data(case, solver, level, tmp_path)
    # Sample has three linear solves reporting 9, 11, 10 -> mean 10.
    assert data["linear_iterations"] == pytest.approx(10.0)


def test_get_data_parses_profile(synthetic_run):
    case, solver, level, tmp_path = synthetic_run
    data = get_data(case, solver, level, tmp_path)
    assert data["pc_setup"] == pytest.approx(8.7654)
    assert data["solve_time"] == pytest.approx(1.230e2)


def test_get_data_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_data("cockett", "iterative", 1, tmp_path)


def test_get_data_partial_output(tmp_path):
    """Only the stdout exists; profile metrics should be nan."""
    (tmp_path / "cockett_iterative_1.out").write_text(_SAMPLE_OUT)
    data = get_data("cockett", "iterative", 1, tmp_path)
    assert data["linear_iterations"] == pytest.approx(10.0)
    assert math.isnan(data["pc_setup"])
    assert math.isnan(data["solve_time"])
