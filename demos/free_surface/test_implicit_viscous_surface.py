import numpy as np
from pathlib import Path
import pytest

base = Path(__file__).parent.resolve()


@pytest.fixture
def expected_errors():
    return np.loadtxt(base / "expected-errors-implicit-free-surface-coupling.dat")


def test_scalar_advection_diffusion_DH27(expected_errors):

    expected_errors = expected_errors
    errors = np.loadtxt(base / "errors-implicit-free-surface-coupling.dat")

    # check that norm(q) is the same as previously run
    assert np.allclose(errors, expected_errors, rtol=1e-6)

    # use the highest resolution analytical solutions as the reference in scaling
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    expected_convergence = [2.0, 2.0, 2.0]

    assert np.allclose(convergence, expected_convergence, rtol=1e-1)


if __name__ == "__main__":
    print("Implicit-freesurface-coupling_")
