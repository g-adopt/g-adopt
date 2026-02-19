from pathlib import Path
import numpy as np


def test_adjoint_taylortest():
    with open(Path(__file__).parent.resolve() / "taylor_test_minconv.txt", "r") as f:
        tt_minconv = [float(x) for x in f.readlines()]

    np.testing.assert_allclose(tt_minconv[-1], 2, rtol=2e-2)


def test_adjoint_optimisation():
    with open(Path(__file__).parent.resolve() / "functional.txt", "r") as f:
        functional_values = [float(x) for x in f.readlines()]

    np.testing.assert_allclose(functional_values[-1], 0.000629, rtol=1e-2)
