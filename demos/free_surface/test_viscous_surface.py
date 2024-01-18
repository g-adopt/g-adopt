import numpy as np
from pathlib import Path
import pytest

base = Path(__file__).parent.resolve()


@pytest.fixture
def expected_errors(coupling):
    return np.loadtxt(base / f"expected-errors-{coupling}-free-surface-coupling.dat")


@pytest.mark.parametrize("coupling,expected_convergence", [("explicit", [1.4, 1.1, 1.0]), ("implicit", [2.0, 2.0, 2.0])])
def test_scalar_advection_diffusion_DH27(coupling, expected_convergence, expected_errors):

    expected_errors = expected_errors
    errors = np.loadtxt(base / f"errors-{coupling}-free-surface-coupling.dat")

    # check that norm(q) is the same as previously run
    assert np.allclose(errors, expected_errors, rtol=1e-6)

    # use the highest resolution analytical solutions as the reference in scaling
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])

    assert np.allclose(convergence, expected_convergence, rtol=1e-1)


if __name__ == "__main__":
    print("Implicit-freesurface-coupling_")
