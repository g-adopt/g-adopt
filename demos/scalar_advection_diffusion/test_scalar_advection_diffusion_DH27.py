import pandas as pd
import numpy as np
import pytest

# Su, grid Peclet number, expected convergence
config = [(True, 0.25, [2.0, 2.0, 2.0]),
          (False, 0.25, [2.0, 2.0, 2.0]),
          (True, 0.9, None),
          (False, 0.9, None),
          (True, 50, [0.5, 0.5, 0.5]),  # don't actually test for this convergence as not sure if it is meangingul
          (False, 50, None),
          ]


@pytest.mark.parametrize("su,pe,expected_convergence", config)
def test_scalar_advection_diffusion_DH27(su, pe, expected_convergence):

    dat = pd.read_csv(f"errors-su{su}_Pe{pe}.dat", sep=" ", header=None)
    dat.columns = ["l2error_q", "l2anal_q", "l2q"]

    expected_dat = pd.read_csv(f"expected_errors-su{su}_Pe{pe}.dat", sep=" ", header=None)
    expected_dat.columns = ["l2error_q", "l2anal_q", "l2q"]

    # check that norm(q) is the same as previously run
    assert np.allclose(dat["l2q"], expected_dat["l2q"], rtol=1e-6)

    # use the highest resolution analytical solutions as the reference in scaling
    ref = dat.iloc[-1][["l2anal_q"]].rename(index=lambda s: s.replace("anal", "error"))
    errs = dat[["l2error_q"]] / ref
    convergence = np.log2(errs.shift() / errs).drop(index=0)

    # check second order convergence for Pe = 0.25 with and without SU (diffusion dominates
    # so in asymptotic limit for P1). For higher peclet numbers adding SU changes diffusivity
    # so not solving the same problem so probably don't expect to be in the asymptotic limit.
    # with Pe = 50 the convergence seems to be 1/2... see plot_convergence.py for a visual plot.
    if pe == 0.25:
        assert np.allclose(convergence, expected_convergence, rtol=1e-2)
