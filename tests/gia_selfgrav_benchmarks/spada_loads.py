"""Spada (2011), Table 4 load coefficients, shared by solver and plots.

Only NumPy is required. Coefficients include degrees 0 and 1; the driver
removes those degrees when constructing the benchmark load.
"""

import numpy as np

CAP_THICKNESS_M = 1500.0
DISC_THICKNESS_M = 1000.0
CAP_RADIUS_DEG = 10.0  # Both finite-size loads have this angular radius.
RHO_ICE = 931.0
LOAD_SHAPES = ("cap", "disc")


def cap_load(nmax):
    """Parabolic cap coefficients sigma_0..sigma_nmax, in kg m^-2."""
    a = np.deg2rad(CAP_RADIUS_DEG)
    n = np.arange(nmax + 1)

    def T(k):
        return np.cos(k * a)

    return -RHO_ICE * CAP_THICKNESS_M / (4 * (1 - np.cos(a))) * (
        (T(n + 1) - T(n + 2)) / (n + 1.5) - (T(n - 1) - T(n)) / (n - 0.5))


def disc_load(nmax):
    """Uniform disc coefficients sigma_0..sigma_nmax, in kg m^-2.

    sigma_0 = rho h (1 - cos(alpha)) / 2;
    sigma_n = rho h [P_{n-1}(cos(alpha)) - P_{n+1}(cos(alpha))] / 2.
    """
    cosine = np.cos(np.deg2rad(CAP_RADIUS_DEG))
    P = np.polynomial.legendre.legvander(cosine, nmax + 1)[0]
    sigma = np.empty(nmax + 1)
    amplitude = RHO_ICE * DISC_THICKNESS_M
    sigma[0] = amplitude * (1 - cosine) / 2
    sigma[1:] = amplitude * (P[:nmax] - P[2:nmax + 2]) / 2
    return sigma


def load_coefficients(nmax, load_shape="cap"):
    """Return the Table 4 coefficients for the selected load geometry."""
    if load_shape == "cap":
        return cap_load(nmax)
    if load_shape == "disc":
        return disc_load(nmax)
    raise ValueError(f"Unknown load shape {load_shape!r}")


def polar_motion_scale(load_shape):
    """Scale the cap polar-motion series by sigma_2 (paper eq. 31).

    The Earth, load location and Heaviside history are shared, so the
    linear polar-motion response differs only by the degree-2 excitation.
    """
    return load_coefficients(2, load_shape)[2] / cap_load(2)[2]
