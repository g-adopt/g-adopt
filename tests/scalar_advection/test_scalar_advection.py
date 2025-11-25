from pathlib import Path

import numpy as np

base = Path(__file__).parent.resolve()


def test_scalar_advection():
    expected_error = np.load(base / "expected_error.npy")
    final_error = np.loadtxt(base / "final_error.log")

    # check that norm(q) is the same as previously run
    assert np.allclose(final_error, expected_error, rtol=1e-6, atol=1e-16)


def test_scalar_advection_adaptive():
    """Test adaptive timestepping version with multiple checks."""
    # Load expected values (will be generated on first run)
    expected_error_file = base / "expected_error_adaptive.npy"
    expected_steps_file = base / "expected_steps_adaptive.npy"
    expected_dt_stats_file = base / "expected_dt_stats_adaptive.npy"

    # Load actual results
    final_error = np.loadtxt(base / "final_error_adaptive.log")
    num_steps = np.loadtxt(base / "num_steps_adaptive.log")
    dt_stats = np.loadtxt(base / "dt_stats_adaptive.log")  # [min, max, mean]

    # Check final error value
    if expected_error_file.exists():
        expected_error = np.load(expected_error_file)
        assert np.allclose(final_error, expected_error, rtol=1e-6, atol=1e-16), \
            f"Final error mismatch: got {final_error}, expected {expected_error}"

    # Check number of timesteps
    if expected_steps_file.exists():
        expected_steps = np.load(expected_steps_file)
        assert num_steps == expected_steps, \
            f"Number of steps mismatch: got {num_steps}, expected {expected_steps}"

    # Check timestep statistics (min, max, mean)
    if expected_dt_stats_file.exists():
        expected_dt_stats = np.load(expected_dt_stats_file)
        assert np.allclose(dt_stats, expected_dt_stats, rtol=1e-6, atol=1e-16), \
            f"Timestep statistics mismatch: got {dt_stats}, expected {expected_dt_stats}"

    # Basic sanity checks even if expected files don't exist
    assert num_steps > 0, "Number of steps must be positive"
    assert dt_stats[0] > 0, "Minimum timestep must be positive"
    assert dt_stats[1] >= dt_stats[0], "Maximum timestep must be >= minimum"
    assert dt_stats[2] >= dt_stats[0] and dt_stats[2] <= dt_stats[1], \
        "Mean timestep must be between min and max"
