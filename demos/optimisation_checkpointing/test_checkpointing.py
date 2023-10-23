import numpy as np
from pathlib import Path


def test_checkpointing():
    base = Path(__file__).parent.resolve()

    full_optimisation = np.loadtxt(base / "full_optimisation.dat")
    restored_optimisation = np.loadtxt(base / "restored_optimisation_from_it_5.dat")

    restored_steps = restored_optimisation.size

    assert np.allclose(full_optimisation[-restored_steps:], restored_optimisation)

    restored_optimisation_last_it = np.loadtxt(base / "restored_optimisation_from_last_it.dat")

    assert full_optimisation[-1] > restored_optimisation_last_it[0]
