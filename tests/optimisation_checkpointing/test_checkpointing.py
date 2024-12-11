from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("procs", [1, 2])
def test_checkpointing(procs):
    base = Path(__file__).parent.resolve()

    full_optimisation = np.loadtxt(base / f"full_optimisation_np{procs}.dat")
    restored_optimisation = np.loadtxt(
        base / f"restored_optimisation_from_it_5_np{procs}.dat"
    )

    restored_steps = restored_optimisation.size

    assert np.allclose(full_optimisation[-restored_steps:], restored_optimisation)

    restored_optimisation_last_it = np.loadtxt(
        base / f"restored_optimisation_from_last_it_np{procs}.dat"
    )

    assert full_optimisation[-1] > restored_optimisation_last_it[0]
