"""Pytest tests for Richards equation benchmarks.

Tests include:
1. Intermediate error threshold tests - verify max error stays below threshold
2. Convergence rate tests - verify expected spatial convergence rates

Results are printed to stdout for visibility in GitHub Actions logs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Output Formatting for CI Visibility
# ============================================================================

def print_convergence_summary(case_name, df):
    """Print a formatted convergence summary table for a test case.

    This output is designed for visibility in GitHub Actions logs.
    """
    print()
    print("=" * 60)
    print(f"Tracy 2D Convergence Results: {case_name}")
    print("=" * 60)

    # Compute relative errors and convergence rates
    df_sorted = df.sort_values('level')
    levels = df_sorted['level'].values
    rel_errors = (df_sorted['l2error_h'] / df_sorted['l2anal_h']).values

    # Compute convergence rates
    rates = [None]  # First level has no rate
    for i in range(1, len(levels)):
        if rel_errors[i-1] > 0 and rel_errors[i] > 0:
            rate = np.log2(rel_errors[i-1] / rel_errors[i])
            rates.append(rate)
        else:
            rates.append(None)

    # Print table header
    print(f"{'Level':^7} | {'Rel Error (h)':^15} | {'Conv. Rate':^12} | {'Converged':^10}")
    print("-" * 7 + "-+-" + "-" * 15 + "-+-" + "-" * 12 + "-+-" + "-" * 10)

    # Print data rows
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        level = int(row['level'])
        rel_error = rel_errors[i]
        rate = rates[i]
        converged = row.get('converged', True)

        rate_str = f"{rate:>10.2f}" if rate is not None else "     -"
        conv_str = "Yes" if converged else "No"

        print(f"{level:^7} | {rel_error:>13.4e} | {rate_str:>12} | {conv_str:^10}")

    print("=" * 60)
    print()


# ============================================================================
# Threshold Configuration
# ============================================================================
# These thresholds are set based on baseline runs and should be updated
# after running: python richards.py run <case>
# Thresholds are for max_intermediate_error_h (relative L2 error)

THRESHOLDS = {
    "tracy_2d_specified_head_dg1": {
        1: None,  # To be set after baseline run
        2: None,
        3: None,
        4: None,
    },
    "tracy_2d_specified_head_dg2": {
        1: None,
        2: None,
        3: None,
        4: None,
    },
    "tracy_2d_no_flux_dg1": {
        1: None,
        2: None,
        3: None,
        4: None,
    },
    "tracy_2d_no_flux_dg2": {
        1: None,
        2: None,
        3: None,
        4: None,
    },
}

# ============================================================================
# Convergence Rate Configuration
# ============================================================================
# Expected minimum convergence rates (computed between levels 3 and 4)
# DG theory: L2 error converges as O(h^(p+1)) where p is polynomial degree
#   DG1 (p=1): theoretical O(h^2) -> rate = 2.0
#   DG2 (p=2): theoretical O(h^3) -> rate = 3.0
# We allow ~5% below theoretical to account for pre-asymptotic effects

EXPECTED_CONVERGENCE_RATES = {
    "dg1": 1.9,  # Theoretical: 2.0, allowing 5% margin
    "dg2": 2.9,  # Theoretical: 3.0, allowing ~3% margin
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_results(case_name):
    """Load results CSV for a test case."""
    results_file = Path(__file__).parent / f"results_{case_name}.csv"
    if not results_file.exists():
        return None
    return pd.read_csv(results_file)


def compute_convergence_rate(errors, levels):
    """Compute convergence rate between successive refinement levels.

    For mesh refinement where h halves with each level:
    rate = log2(error_coarse / error_fine)
    """
    rates = []
    for i in range(len(levels) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            rate = np.log2(errors[i] / errors[i + 1])
            rates.append(rate)
        else:
            rates.append(np.nan)
    return rates


# ============================================================================
# Test Cases
# ============================================================================

# Tracy 2D test cases (DG1 and DG2, both BC types)
tracy_cases = [
    "tracy_2d_specified_head_dg1",
    "tracy_2d_specified_head_dg2",
    "tracy_2d_no_flux_dg1",
    "tracy_2d_no_flux_dg2",
]


class TestIntermediateErrors:
    """Test that intermediate errors stay below thresholds."""

    @pytest.mark.parametrize("case_name", tracy_cases)
    def test_intermediate_error_threshold(self, case_name):
        """Verify max intermediate error is below threshold for each level."""
        df = load_results(case_name)
        if df is None:
            pytest.skip(f"Results file not found for {case_name}. Run 'python richards.py run {case_name}' first.")

        thresholds = THRESHOLDS.get(case_name)
        if thresholds is None:
            pytest.skip(f"No thresholds defined for {case_name}")

        # Check if any thresholds are set
        if all(v is None for v in thresholds.values()):
            pytest.skip(f"Thresholds not yet configured for {case_name}. Run baseline tests first.")

        all_passed = True
        for _, row in df.iterrows():
            level = int(row['level'])
            max_error = row['max_intermediate_error_h']
            threshold = thresholds.get(level)

            if threshold is None:
                continue  # Skip levels without thresholds

            if max_error >= threshold:
                all_passed = False
                print(f"FAIL: {case_name} level {level}: max intermediate error {max_error:.6e} "
                      f"exceeds threshold {threshold:.6e}")
            else:
                print(f"PASS: {case_name} level {level}: max intermediate error {max_error:.6e} "
                      f"< threshold {threshold:.6e}")

        assert all_passed, f"{case_name}: some levels exceeded error thresholds"


class TestConvergenceRates:
    """Test spatial convergence rates."""

    @pytest.mark.parametrize("case_name", tracy_cases)
    def test_convergence_rate(self, case_name):
        """Verify convergence rate between levels 3 and 4."""
        df = load_results(case_name)
        if df is None:
            pytest.skip(f"Results file not found for {case_name}. Run 'python richards.py run {case_name}' first.")

        # Print full convergence summary for CI visibility
        print_convergence_summary(case_name, df)

        # Filter to levels 3 and 4
        df_conv = df[df['level'].isin([3, 4])].sort_values('level')
        if len(df_conv) < 2:
            pytest.skip("Need results for levels 3 and 4 to compute convergence rate")

        # Compute relative errors
        rel_errors = (df_conv['l2error_h'] / df_conv['l2anal_h']).values
        levels = df_conv['level'].values

        # Compute convergence rate
        rates = compute_convergence_rate(rel_errors, levels)
        if not rates or np.isnan(rates[0]):
            pytest.fail(f"Could not compute convergence rate for {case_name}")

        rate = rates[0]

        # Get expected minimum rate based on polynomial degree
        degree = df_conv['degree'].iloc[0]
        dg_key = f"dg{degree}"
        expected_min_rate = EXPECTED_CONVERGENCE_RATES.get(dg_key, 1.5)

        assert rate >= expected_min_rate, (
            f"{case_name}: convergence rate {rate:.2f} is below expected minimum {expected_min_rate:.2f}"
        )

        # Print summary for CI logs
        theoretical_rate = degree + 1  # DG theory: O(h^(p+1))
        print(f"PASS: {case_name}")
        print(f"  Measured rate:    {rate:.2f}")
        print(f"  Theoretical rate: {theoretical_rate:.1f} (DG{degree} -> O(h^{theoretical_rate}))")
        print(f"  Minimum allowed:  {expected_min_rate:.2f}")


class TestVauclin:
    """Test that Vauclin case runs and produces sensible results.

    The Vauclin (1979) benchmark simulates 2D water table recharge with:
    - Initial water table at z = 0.65 m (bottom third saturated)
    - Top infiltration at 14.8 cm/hr for x <= 0.5 m
    - Fixed water table on right boundary
    - No-flux on left and bottom
    """

    def test_vauclin_runs(self):
        """Test that Vauclin case produces valid results."""
        df = load_results("vauclin_2d")
        if df is None:
            pytest.skip("Results file not found for vauclin_2d. Run 'python richards.py run vauclin_2d' first.")

        for _, row in df.iterrows():
            # Basic sanity checks for water table recharge problem
            # Initial condition: h = 0.65 - 1.001*y, so:
            # - At y=0 (bottom): h ≈ 0.65 m (positive, saturated)
            # - At y=2 (top): h ≈ -1.35 m (negative, unsaturated)
            assert row['min_h'] < 0, "Minimum h should be negative (unsaturated zone at top)"
            assert row['max_h'] > 0, "Maximum h should be positive (saturated zone at bottom)"
            assert row['max_h'] <= 0.65, "Maximum h should not exceed water table height"


# ============================================================================
# Utility for Setting Thresholds
# ============================================================================

def print_baseline_thresholds():
    """Print current results to help set thresholds.

    Run this interactively after baseline runs to see what thresholds to set.
    Usage: python -c "from test_richards import print_baseline_thresholds; print_baseline_thresholds()"
    """
    print("=" * 70)
    print("Baseline Results for Setting Thresholds")
    print("=" * 70)

    for case_name in tracy_cases:
        df = load_results(case_name)
        if df is None:
            print(f"\n{case_name}: No results file found")
            continue

        print(f"\n{case_name}:")
        print("-" * 50)

        for _, row in df.iterrows():
            level = int(row['level'])
            max_error = row['max_intermediate_error_h']
            rel_error = row['l2error_h'] / row['l2anal_h']

            # Suggest threshold with 20% margin
            suggested_threshold = max_error * 1.2

            print(f"  Level {level}:")
            print(f"    max_intermediate_error_h = {max_error:.6e}")
            print(f"    relative_l2_error_h      = {rel_error:.6e}")
            print(f"    suggested_threshold      = {suggested_threshold:.6e}")

    print("\n" + "=" * 70)
    print("Convergence Rates (levels 3->4)")
    print("=" * 70)

    for case_name in tracy_cases:
        df = load_results(case_name)
        if df is None:
            continue

        df_conv = df[df['level'].isin([3, 4])].sort_values('level')
        if len(df_conv) < 2:
            print(f"\n{case_name}: Need levels 3 and 4 for convergence rate")
            continue

        rel_errors = (df_conv['l2error_h'] / df_conv['l2anal_h']).values
        levels = df_conv['level'].values
        rates = compute_convergence_rate(rel_errors, levels)

        if rates and not np.isnan(rates[0]):
            print(f"\n{case_name}: rate = {rates[0]:.3f}")


if __name__ == "__main__":
    # When run directly, print baseline thresholds
    print_baseline_thresholds()
