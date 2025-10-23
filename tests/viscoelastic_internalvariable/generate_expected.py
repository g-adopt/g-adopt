import numpy as np
from test_viscoelastic_free_surface import cases

results = {}
for case_name, _ in cases:
    errors = np.loadtxt(f"errors-{case_name}-free-surface.dat")
    results[case_name] = errors

np.savez("expected_errors.npz", **results)
