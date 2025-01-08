import numpy as np

final_error = np.loadtxt("final_error.log")
np.save("expected_error.npy", [final_error])
