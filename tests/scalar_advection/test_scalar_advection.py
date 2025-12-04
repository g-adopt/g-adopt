from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

base = Path(__file__).parent.resolve()


def test_scalar_advection():
    expected_data = np.load(base / "expected.npz")
    expected_error = expected_data['error']
    final_error = np.loadtxt(base / "final_error.log")

    # check that norm(q) is the same as previously run
    assert_allclose(final_error, expected_error, rtol=1e-6, atol=1e-16)


def test_scalar_advection_adaptive():
    """Test adaptive timestepping version with multiple checks."""
    # Load expected values (will be generated on first run)
    expected_adaptive_file = base / "expected_adaptive.npz"

    # Load actual results
    final_error = np.loadtxt(base / "final_error_adaptive.log")
    num_steps = np.loadtxt(base / "num_steps_adaptive.log")
    dt_stats = np.loadtxt(base / "dt_stats_adaptive.log")  # [min, max, mean]

    # Check expected values if file exists
    if expected_adaptive_file.exists():
        expected_data = np.load(expected_adaptive_file)
        expected_error = expected_data['error']
        expected_steps = expected_data['steps']
        expected_dt_stats = expected_data['dt_stats']

        # Check final error value
        assert_allclose(final_error, expected_error, rtol=1e-6, atol=1e-16)

        # Check number of timesteps
        assert_allclose(num_steps, expected_steps, rtol=0, atol=0)

        # Check timestep statistics (min, max, mean)
        assert_allclose(dt_stats, expected_dt_stats, rtol=1e-6, atol=1e-16)

    # Basic sanity checks even if expected files don't exist
    assert num_steps > 0, "Number of steps must be positive"
    assert dt_stats[0] > 0, "Minimum timestep must be positive"
    assert dt_stats[1] >= dt_stats[0], "Maximum timestep must be >= minimum"
    assert dt_stats[2] >= dt_stats[0] and dt_stats[2] <= dt_stats[1], \
        "Mean timestep must be between min and max"
