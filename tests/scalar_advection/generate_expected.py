import numpy as np

# Generate expected values for non-adaptive test
final_error = np.loadtxt("final_error.log")
np.save("expected_error.npy", [final_error])

# Generate expected values for adaptive test
try:
    final_error_adaptive = np.loadtxt("final_error_adaptive.log")
    num_steps_adaptive = np.loadtxt("num_steps_adaptive.log")
    dt_stats_adaptive = np.loadtxt("dt_stats_adaptive.log")

    np.save("expected_error_adaptive.npy", [final_error_adaptive])
    np.save("expected_steps_adaptive.npy", [num_steps_adaptive])
    np.save("expected_dt_stats_adaptive.npy", dt_stats_adaptive)
except FileNotFoundError:
    print("Warning: Adaptive test outputs not found. Run scalar_advection_adaptive.py first.")
