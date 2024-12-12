from pathlib import Path

import numpy as np
import pytest

base = Path(__file__).parent.resolve()

case_name = ["integrated_q", "integrated_q_DH219"]

@pytest.mark.parametrize("case_name", case_name)
def test_scalar_advection_diffusion(case_name):
    expected_intq = np.load(base / f"expected_{case_name}.npy")
    intq = np.loadtxt(base / f"{case_name}.log")

    # check that integrated is the same as previously run
    assert np.allclose(expected_intq, intq, rtol=1e-6, atol=1e-16)
