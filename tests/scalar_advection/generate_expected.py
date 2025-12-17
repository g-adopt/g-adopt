import numpy as np

# Generate expected values for non-adaptive test
final_error = np.loadtxt("final_error.log")
np.savez_compressed("expected.npz", error=[final_error])

# Generate expected values for adaptive test
try:
    final_error_adaptive = np.loadtxt("final_error_adaptive.log")
    num_steps_adaptive = np.loadtxt("num_steps_adaptive.log")
    dt_stats_adaptive = np.loadtxt("dt_stats_adaptive.log")

    np.savez_compressed("expected_adaptive.npz",
                        error=[final_error_adaptive],
                        steps=[num_steps_adaptive],
                        dt_stats=dt_stats_adaptive)
except FileNotFoundError:
    print("Warning: Adaptive test outputs not found. Run scalar_advection_adaptive.py first.")
