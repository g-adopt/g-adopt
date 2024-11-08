from pathlib import Path


def test_adjoint_optimisation():
    with open(Path(__file__).parent.resolve() / "functional.txt", "r") as f:
        functional_values = [float(x) for x in f.readlines()]

    assert functional_values[-1] < 2e-4
