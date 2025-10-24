from pathlib import Path
import richards
import numpy as np
import pandas as pd
import pytest

# Expected convergence rates and tolerances
enabled_cases = {
    "tracy_2d_specified_head": {
        "convergence": (2.0, 2.0),  # Expected rates for h and theta
        "rtol": 2e-1,
    },
    "tracy_2d_no_flux": {
        "convergence": (2.0, 2.0),
        "rtol": 2e-1,
    },
}

# Generate test configurations
configs = []
for name, expected in enabled_cases.items():
    configs.append((name, expected))


def idfn(val):
    """Generate test ID from configuration."""
    if isinstance(val, dict):
        return "-".join([f"{k}{v}" for k, v in val.items()])
    return str(val)


@pytest.mark.parametrize("name,expected", configs, ids=idfn)
def test_richards(name, expected):
    """Test Richards equation convergence."""
    levels = richards.get_case(richards.cases, name)["levels"]

    b = Path(__file__).parent.resolve()

    # Read error data files
    dats = []
    for level in levels:
        dat_file = b / f"errors-{name}-levels{level}.dat"
        if not dat_file.exists():
            pytest.skip(f"Data file {dat_file} not found. Run 'make {name}' first.")
        dats.append(
            pd.read_csv(dat_file, sep=" ", header=None)
        )

    # Column names
    cols_err = ["l2error_h", "l2error_theta"]
    cols_anal = ["l2anal_h", "l2anal_theta"]

    dat = pd.concat(dats)
    dat.columns = cols_err + cols_anal
    dat.insert(0, "level", levels)
    dat = dat.set_index("level")

    # Use the highest resolution analytical solutions as the reference
    ref = dat.iloc[-1][cols_anal].rename(index=lambda s: s.replace("anal", "error"))
    errs = dat[cols_err] / ref
    errs = errs.reset_index(drop=True)  # drop resolution label

    # Compute convergence rates
    convergence = np.log2(errs.shift() / errs).drop(index=0)
    expected_convergence = pd.Series(expected["convergence"], index=cols_err)

    # Check convergence rates
    assert np.allclose(
        convergence,
        expected_convergence,
        rtol=expected.get("rtol", 1e-2)
    ), f"Convergence rates do not match expected values.\nGot:\n{convergence}\nExpected:\n{expected_convergence}"


def test_vauclin_runs():
    """Test that Vauclin case runs without errors."""
    levels = richards.get_case(richards.cases, "vauclin_2d")["levels"]

    b = Path(__file__).parent.resolve()

    for level in levels:
        dat_file = b / f"errors-vauclin_2d-levels{level}.dat"
        if not dat_file.exists():
            pytest.skip(f"Data file {dat_file} not found. Run 'make vauclin_2d' first.")

        # Read results
        dat = pd.read_csv(dat_file, sep=" ", header=None)
        dat.columns = ["min_h", "max_h", "total_infiltration"]

        # Basic sanity checks
        assert dat["min_h"].iloc[0] < 0, "Minimum h should be negative (unsaturated)"
        assert dat["max_h"].iloc[0] < 0, "Maximum h should be negative (unsaturated)"
        assert dat["total_infiltration"].iloc[0] > 0, "Total infiltration should be positive"
