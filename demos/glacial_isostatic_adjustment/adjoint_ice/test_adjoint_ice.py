from pathlib import Path
import math


def test_adjoint_taylortest():
    with open(Path(__file__).parent.resolve() / "taylor_test_minconv.txt", "r") as f:
        tt_minconv = [float(x) for x in f.readlines()]

    assert math.isclose(tt_minconv[-1], 2, rel_tol=2e-2)


def test_adjoint_optimisation():
    with open(Path(__file__).parent.resolve() / "functional.txt", "r") as f:
        functional_values = [float(x) for x in f.readlines()]

    assert math.isclose(functional_values[-1], 0.000726, rel_tol=1e-2)
