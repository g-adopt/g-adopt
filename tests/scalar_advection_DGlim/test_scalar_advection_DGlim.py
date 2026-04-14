from pathlib import Path

import numpy as np

base = Path(__file__).parent.resolve()


def test_scalar_advection():
    expected_error = np.load(base / "expected_error.npy")
    final_error = np.loadtxt(base / "final_error.log")

    # check that norm(q) is the same as previously run
    assert np.allclose(final_error, expected_error, rtol=1e-6, atol=1e-16)
