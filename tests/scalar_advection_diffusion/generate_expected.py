import itertools

import numpy as np
from test_scalar_advection_diffusion_DH27 import conf

# cases for analytical comparison DH27
results = {}
for param_set in itertools.product(*conf.values()):
    param_str = "_".join(f"{p[0]}{p[1]}" for p in zip(conf.keys(), param_set))
    errors = np.loadtxt(f"errors-{param_str}.dat")
    results[param_str] = errors

np.savez("expected_errors.npz", **results)

# cases for regression testing
int_q = np.loadtxt("integrated_q.log")
np.save("expected_integrated_q.npy", [int_q])

int_q_DH219 = np.loadtxt("integrated_q_DH219.log")
np.save("expected_integrated_q_DH219.npy", [int_q_DH219])
