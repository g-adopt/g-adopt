import numpy as np
from test_viscous_surface import cases

results = {}
for c in cases:
    case_name = c[0]
    errors = np.loadtxt(f"errors-{case_name}-free-surface-coupling.dat")
    results[case_name] = errors

np.savez("expected_errors.npz", **results)
