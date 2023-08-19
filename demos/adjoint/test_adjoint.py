import pytest
import numpy as np
from pathlib import Path
import taylor_test


@pytest.mark.parametrize("case_name", taylor_test.cases)
def test_rectangular_taylor_test(case_name):
    with open(Path(__file__).parent.resolve() / f"{case_name}.conv", "r") as f:
        minconv = float(f.read())

    assert minconv > 1.9


def test_checkpointing():
    base = Path(__file__).parent.resolve()

    full_optimisation = np.loadtxt(base / "full_optimisation.dat")
    restored_optimisation = np.loadtxt(base / "restored_optimisation.dat")

    restored_steps = restored_optimisation.size

    assert np.allclose(full_optimisation[-restored_steps:], restored_optimisation)
