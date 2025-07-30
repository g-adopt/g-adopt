from pathlib import Path

import numpy as np
import pytest
base = Path(__file__).parent.resolve()


@pytest.fixture
def expected_errors():
    return np.load(base / "expected_errors.npz")


cases = [
    ("viscoelastic-compressible", [0.96, 0.98]),  # Higher dx converges O(1)
    ("viscoelastic-compressible-burgers", [0.96, 0.98]),  # Higher dx converges O(1)
    ("elastic-compressible", [1.0, 1.0]),
]


@pytest.mark.parametrize("case_name,expected_convergence", cases)
def test_viscoelastic_free_surface(case_name, expected_convergence, expected_errors):
    expected_errors = expected_errors[case_name]
    errors = np.loadtxt(base / f"errors-{case_name}-free-surface.dat")

    # check that norm(q) is the same as previously run
    assert np.allclose(errors, expected_errors, rtol=1e-6, atol=1e-16)

    # use the highest resolution analytical solutions as the reference in scaling
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    assert np.allclose(convergence, expected_convergence, rtol=1e-1)
