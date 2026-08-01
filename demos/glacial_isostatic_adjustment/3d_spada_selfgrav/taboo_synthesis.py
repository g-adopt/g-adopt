"""Synthesise the Spada et al. (2011) benchmark reference fields from the
spectral data in ``reference.npz``.

Equations (17)-(19) of the paper, for a Heaviside surface load:

    hbar(t) = h_e - sum_j h_j (1 - exp(s_j t)) / s_j
    kbar(t) = 1 + k_e - sum_j k_j (1 - exp(s_j t)) / s_j     <- the 1 is the direct term
    {U, V, N}(theta, t) = (3/rhobar) sum_n {hbar, lbar, kbar}_n(t)
                          sigma_n/(2n+1) {1, d_theta, 1} P_n(cos theta)

with rhobar = 5511.68 kg m^-3, the *model's own* mean density, not Earth's, and
s_j in kyr^-1 with t in kyr.

Dropping the 1 in kbar makes N about 60% low; using Earth's mean density makes
everything about 0.04% wrong; and ``spec_*.mod`` is tau_j in years rather than
s_j in kyr^-1, a factor of 1000 that looks like a physics error.

Rates come free by differentiating in time: d(hbar)/dt = sum_j h_j exp(s_j t).

Degrees 0 and 1 are absent by construction at n_min = 2, which is how the
reference is defined and disposes of the centre-of-mass frame question.
"""

import pathlib

import numpy as np

RHO_BAR = 5511.68     # kg m^-3, mean density of M3-L70-V01
A_EARTH = 6.371e6     # m
RHO_ICE = 931.0       # kg m^-3
DEFAULT_NPZ = pathlib.Path(__file__).resolve().parent / "reference.npz"


def legendre_and_dtheta(nmax, theta):
    """P_n(cos theta) and d/dtheta P_n(cos theta) for n = 0..nmax."""
    ct, st = np.cos(theta), np.sin(theta)
    P = np.zeros((nmax + 1, theta.size))
    P[0] = 1.0
    if nmax >= 1:
        P[1] = ct
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * ct * P[n] - n * P[n - 1]) / (n + 1)
    # (1-x^2) P_n'(x) = n (P_{n-1} - x P_n), so d_theta P_n = -n (P_{n-1} - x P_n)/sin.
    # The derivative vanishes at both poles; the zeros left there are correct.
    dP = np.zeros_like(P)
    ok = st > 1e-13
    for n in range(1, nmax + 1):
        dP[n, ok] = -n * (P[n - 1, ok] - ct[ok] * P[n, ok]) / st[ok]
    return P, dP


def cap_load(nmax, thickness=1500.0, alpha_deg=10.0, rho=RHO_ICE, nmin=0):
    """Table 4 coefficients of the parabolic cap,
    sigma(theta) = rho h sqrt((cos theta - cos alpha)/(1 - cos alpha))."""
    a = np.deg2rad(alpha_deg)
    n = np.arange(nmin, nmax + 1)
    T = lambda k: np.cos(k * a)  # noqa: E731
    return -rho * thickness / (4 * (1 - np.cos(a))) * (
        (T(n + 1) - T(n + 2)) / (n + 1.5) - (T(n - 1) - T(n)) / (n - 0.5)
    )


def disc_load(nmax, thickness=1000.0, alpha_deg=10.0, rho=RHO_ICE, nmin=0):
    """Coefficients of the uniform disc, from int_{cos alpha}^1 P_n dx."""
    a = np.deg2rad(alpha_deg)
    P, _ = legendre_and_dtheta(nmax + 1, np.array([a]))
    P = P[:, 0]
    n = np.arange(nmin, nmax + 1)
    below = np.where(n >= 1, P[np.maximum(n - 1, 0)], 1.0)  # P_{-1} := P_0 = 1
    return rho * thickness * (below - P[n + 1]) / 2.0


def point_load(nmax, mass=1e18, radius=A_EARTH, nmin=0):
    """Coefficients of a point mass at theta = 0."""
    n = np.arange(nmin, nmax + 1)
    return (2 * n + 1) * mass / (4 * np.pi * radius**2)


def load_mass(sigma_n, radius=A_EARTH):
    """Total mass of a load from its degree-zero coefficient."""
    return sigma_n[0] * 4 * np.pi * radius**2


class TabooReference:
    """The TABOO spectral reference, and the spatial fields it synthesises."""

    def __init__(self, npz_path=DEFAULT_NPZ):
        self.data = np.load(npz_path, allow_pickle=False)
        self.degrees = self.data["degrees"]
        self.nmin_available = int(self.degrees[0])
        self.nmax_available = int(self.degrees[-1])

    def love_time(self, t_kyr, nmax, fluid=False, rate=False):
        """hbar, lbar, kbar for degrees 2..nmax at epoch ``t_kyr`` (kyr).

        ``fluid`` returns the fluid limit instead, which is *not* the same as
        t = 1000 kyr: the slow n = 2 branches have tau = 2.76e4 kyr and are only
        a few percent relaxed there.  ``rate`` returns the time derivative, per
        kyr; there is no rate of the fluid limit.
        """
        if nmax > self.nmax_available:
            raise ValueError(f"nmax = {nmax} exceeds the table's {self.nmax_available}")
        if fluid and rate:
            raise ValueError("the fluid limit is stationary; it has no rate")
        sl = slice(0, nmax - self.nmin_available + 1)
        s = self.data["spectrum_s"][sl]
        out = []
        for symbol in ("h", "l", "k"):
            direct = 1.0 if symbol == "k" else 0.0
            residues = self.data[f"{symbol}_residues"][sl]
            if fluid:
                out.append(direct + self.data[f"{symbol}_fluid"][sl])
            elif rate:
                out.append((residues * np.exp(s * t_kyr)).sum(axis=1))
            else:
                shape = (1.0 - np.exp(s * t_kyr)) / s
                out.append(direct + self.data[f"{symbol}_elastic"][sl]
                           - (residues * shape).sum(axis=1))
        return tuple(out)

    def synthesise(self, t_kyr, theta, sigma_n, nmax=None, fluid=False, rate=False):
        """U, V, N (metres, or metres per kyr if ``rate``) on colatitudes
        ``theta`` (radians) for a load with coefficients ``sigma_n`` indexed
        from degree zero."""
        nmax = min(len(sigma_n) - 1, self.nmax_available) if nmax is None else nmax
        hbar, lbar, kbar = self.love_time(t_kyr, nmax, fluid=fluid, rate=rate)
        n = np.arange(self.nmin_available, nmax + 1)
        P, dP = legendre_and_dtheta(nmax, np.asarray(theta, dtype=float))
        c = 3.0 / RHO_BAR * sigma_n[self.nmin_available:nmax + 1] / (2 * n + 1)
        rows = slice(self.nmin_available, nmax + 1)
        return (c * hbar) @ P[rows], (c * lbar) @ dP[rows], (c * kbar) @ P[rows]
