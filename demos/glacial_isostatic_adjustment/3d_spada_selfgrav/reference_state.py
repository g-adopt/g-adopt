r"""The M3-L70-V01 reference state and the scales of the benchmark.

M3-L70-V01 is the Earth model of Spada et al. (2011): an incompressible,
self-gravitating Maxwell sphere with a 70 km elastic lithosphere, a two-layer
upper mantle, a lower mantle, and an inviscid homogeneous core. This module
holds its density layering, its closed-form reference gravity, and the
non-dimensional constants the driver needs, including the rotation moments.

## Non-dimensionalisation

Lengths are scaled by the mantle thickness D = Re - Rc = 2891 km, densities
by the model's own mean density rho_bar = 5511.68 kg m^-3, and accelerations
by the surface gravity g_bar = 9.81555 m s^-2. With psi = psi_hat g_bar D the
Poisson equation nabla^2 psi = -4 pi G rho becomes

    nabla_hat^2 psi_hat = -(4 pi G rho_bar D / g_bar) rho_hat = -Lambda rho_hat,

so the self-gravity number is Lambda = 4 pi G rho_bar D / g_bar = 1.3613.

The potential `psi` of the solver satisfies nabla^2 psi = -4 pi G rho, so psi
is minus the Newtonian potential and the gravitational acceleration is
g_0 = +grad(psi), pointing inward, with |g_0| = G M(<r) / r^2.

## The reference gravity

The mechanics uses the closed form |g_0|(r) = (Lambda / 4 pi) M(<r) / r^2 of
the layered density, `gravity_exact_ufl`. It is exact for the piecewise
constant layering, including the unmeshed core. The paper's gravity column,
`GRAVITY_COLUMN`, is the result of the same integral and is kept as a check.
"""
import numpy as np

import gadopt  # noqa: F401  (import gadopt before firedrake)
from firedrake import Constant, conditional

# Physical constants, prescribed by the benchmark.
G_NEWTON = 6.6732e-11          # m^3 kg^-1 s^-2
A_EARTH = 6.371e6              # m
D_SCALE = 2.891e6              # m, = Re - Rc
RHO_BAR = 5511.68              # kg m^-3, the model's own mean density
G_BAR = 9.81555                # m s^-2

#: Layers of M3-L70-V01, outermost first: (r_outer_km, r_inner_km, rho).
#: The core row runs to r = 0, so it covers both the meshed inner region and
#: the unmeshed ball.
LAYERS_KM = [(6371.0, 6301.0, 3037.0),
             (6301.0, 5951.0, 3438.0),
             (5951.0, 5701.0, 3871.0),
             (5701.0, 3480.0, 4978.0),
             (3480.0, 0.0, 10750.0)]

#: The paper's gravity column, in m s^-2, at the five radii above.
GRAVITY_COLUMN = {6371.0: 9.815, 6301.0: 9.854, 5951.0: 9.978,
                  5701.0: 10.024, 3480.0: 10.457}

#: The self-gravity number, dimensionless.
LAMBDA = 4 * np.pi * G_NEWTON * RHO_BAR * D_SCALE / G_BAR


# ---------------------------------------------------------------------------
# Rotation: the moments of the reference hydrostatic figure, and Omega^2.
# ---------------------------------------------------------------------------

#: Scale of the moments of inertia, rho_bar D^5, in kg m^2.
RHO_D5 = RHO_BAR * D_SCALE**5

#: The polar moment C, non-dimensional (72.2269). C enters the rotation-rate
#: change m_3 and not the polar motion m_1, m_2.
C_NONDIM = 8.0394e37 / RHO_D5

#: The dynamical ellipticity C - A, non-dimensional. A spherically symmetric
#: reference density gives C = A exactly, so C - A is an input.
#:
#: Two values exist and they differ by 2.4 percent. The benchmark prescribes
#: C - A = 2.63e35 kg m^2, but that value implies a secular Love number
#: k_s = 0.94334, while the benchmark's own rotation calculation used
#: k_s = 0.96672389. The value consistent with that k_s is 2.6952e35. Polar
#: motion carries the factor 1 / (1 - k_T / k_s), which amplifies the 2.4
#: percent to 3.6 percent at t = 0 and 7 to 12 percent by 20 kyr, so the
#: choice is visible in the comparison. The driver uses "ks".
C_MINUS_A = {"ks": 2.6952e35 / RHO_D5,              # 0.24214001
             "prescribed": 2.63e35 / RHO_D5}        # 0.23628236

#: The value the driver uses.
C_MINUS_A_PRIMARY = C_MINUS_A["ks"]

#: Omega^2 non-dimensionalised by g_bar / D, Omega = 7.292115e-5 rad/s.
OMEGA_SQ = 7.292115e-5**2 * D_SCALE / G_BAR          # 1.5661757e-03


def layers_nondim():
    """(r_outer, r_inner, rho) non-dimensional, outermost first."""
    return [(ro / (D_SCALE / 1e3), ri / (D_SCALE / 1e3), rho / RHO_BAR)
            for ro, ri, rho in LAYERS_KM]


def analytic_gravity(r_km):
    """|g| at radius `r_km` from the layered density, in m s^-2.

    The paper's gravity column is the result of this integral, so
    `analytic_gravity(r)` reproduces `GRAVITY_COLUMN[r]` to its printed digits.
    """
    m = 0.0
    for ro, ri, rho in sorted(LAYERS_KM, key=lambda x: x[1]):
        lo, hi = ri * 1e3, min(ro * 1e3, r_km * 1e3)
        if hi > lo:
            m += 4 / 3 * np.pi * (hi**3 - lo**3) * rho
    return G_NEWTON * m / (r_km * 1e3) ** 2


def total_mass():
    """Total mass of the model, in kg (5.97029e24)."""
    return sum(4 / 3 * np.pi * ((ro * 1e3)**3 - (ri * 1e3)**3) * rho
               for ro, ri, rho in LAYERS_KM)


def mean_density():
    """Mean density of the model, in kg m^-3 (equal to `RHO_BAR`)."""
    return total_mass() / (4 / 3 * np.pi * A_EARTH**3)


def enclosed_mass_ufl(r):
    """M(<r), non-dimensional, as UFL: piecewise cubic in r.

    Inside a layer with inner radius r_i and density rho, the mass is the mass
    below r_i plus (4/3) pi rho (r^3 - r_i^3). Above the outermost layer it is
    the total mass.
    """
    layers = sorted(layers_nondim(), key=lambda t: t[1])  # ascending r_inner
    acc = 0.0
    expr = Constant(0.0)
    for r_out, r_in, rho in layers:
        below = acc
        acc += 4 / 3 * np.pi * rho * (r_out**3 - r_in**3)
        here = Constant(below) + Constant(4 / 3 * np.pi * rho) * (
            r**3 - Constant(r_in**3))
        expr = conditional(r < r_out, conditional(r >= r_in, here, expr), expr)
    return conditional(r >= layers[-1][0], Constant(acc), expr)


def gravity_exact_ufl(r):
    """|g_0|(r) = (Lambda / 4 pi) M(<r) / r^2, non-dimensional, as UFL."""
    return (LAMBDA / (4 * np.pi)) * enclosed_mass_ufl(r) / r**2
