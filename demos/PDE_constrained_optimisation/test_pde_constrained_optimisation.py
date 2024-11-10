from pathlib import Path

import pytest

cases = [("boundary", 2e-3), ("field", 7e-3)]


@pytest.mark.parametrize("name,functional_tolerance", cases)
def test_pde_constrained_optimisation(name, functional_tolerance):
    with open(Path(__file__).parent.resolve() / f"functional_{name}.txt", "r") as f:
        functional_values = [float(x) for x in f.readlines()]

    assert functional_values[-1] < functional_tolerance
