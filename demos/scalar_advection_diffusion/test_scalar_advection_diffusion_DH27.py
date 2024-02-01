# Test based on scalar advection-diffusion problem from Figure 2.7 in
# Chapter 2 Steady transport problems from Finite element Methods
# for Flow problems - Donea and Huerta, 2003
# Tests for second order convergence cf with the analytical solution
# (when grid Peclet < 1) and for regression testing based on solution norm.

import itertools
import numpy as np
from pathlib import Path
import pytest

# Test each grid Peclet number with SU enabled and disabled
conf = {
    "Pe": [0.25, 0.9, 50.0],
    "SU": [True, False],
}

# Only test the analytical convergence of the Pe=0.25 cases
convergence = [2.0, 2.0, None, None, None, None]

param_sets = zip(itertools.product(*conf.values()), convergence)

base = Path(__file__).parent.resolve()


@pytest.fixture
def expected_errors():
    return np.load(base / "expected_errors.npz")


@pytest.mark.parametrize("params,expected_convergence", param_sets)
def test_scalar_advection_diffusion_DH27(params, expected_convergence, expected_errors):
    Pe, SU = params
    param_str = "_".join(f"{p[0]}{p[1]}" for p in zip(conf.keys(), params))

    expected_errors = expected_errors[param_str]
    errors = np.loadtxt(base / f"errors-{param_str}.dat")

    # check that norm(q) is the same as previously run
    assert np.allclose(errors[:, 2], expected_errors[:, 2], rtol=1e-6)

    # use the highest resolution analytical solutions as the reference in scaling
    ref = errors[-1, 1]
    relative_errors = errors[:, 0] / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])

    # check second order convergence for Pe = 0.25 with and without SU (diffusion dominates
    # so in asymptotic limit for P1). For higher peclet numbers adding SU changes diffusivity
    # so not solving the same problem so probably don't expect to be in the asymptotic limit.
    # with Pe = 50 the convergence seems to be 1/2... see plot_convergence.py for a visual plot.
    if expected_convergence is not None:
        assert np.allclose(convergence, expected_convergence, rtol=1e-2)


if __name__ == "__main__":
    for c in param_sets:
        print("_".join([str(x) for x in c[0]]))
