import itertools
import numpy as np

from test_scalar_advection_diffusion_DH27 import conf


results = {}
for param_set in itertools.product(*conf.values()):
    param_str = "_".join(f"{p[0]}{p[1]}" for p in zip(conf.keys(), param_set))
    errors = np.loadtxt(f"errors-{param_str}.dat")
    results[param_str] = errors

np.savez("expected_errors.npz", **results)
