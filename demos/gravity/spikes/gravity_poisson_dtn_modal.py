"""Blind modal DtN on the truncated disc: the solver discovers the mode.

Same m=3 shell density and truncated (R = 2*rmax) curved mesh as the Robin
demonstration, but the boundary condition no longer knows the density is a
single mode: GravityPoissonSolver imposes the full modal DtN map for modes
1..M, with the trace Fourier coefficients as R-space unknowns.

Sweeping the truncation M with the density fixed at m=3:
  - M < 3: mode 3 is untreated (homogeneous Neumann), large error;
  - M >= 3: the error snaps to the discretisation floor of the Robin run
    and stays flat -- extra modes cost nothing and their coefficients come
    out at machine zero.

The solved c_3 must equal the analytical trace amplitude psi_m(R_grav).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import Mesh
from gadopt import log
from passess.polar import PoissonPolar2D
from gravity_dtn import GravityPoissonSolver
from gravity_poisson_robin import (
    get_mesh, curve_mesh, make_mantle_submesh, shell_density,
    mantle_relative_error, rmax, r1_shell, r2_shell, rho_m, gamma, degree,
    OUTER_ID)

m_mode = 3
R_grav_factor = 2
R_grav = R_grav_factor * rmax

analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                            r1=r1_shell, r2=r2_shell, gamma=gamma)
c3_exact = analytical.psi_m(R_grav).real

mesh_file = get_mesh(R_grav_factor, 0.2)
mesh, subm = make_mantle_submesh(curve_mesh(Mesh(mesh_file)))
rho = shell_density(subm, m_mode)

log(f"Density: m = {m_mode} shell; solver is NOT told the mode.")
log(f"R_grav = {R_grav:.2f}; analytical trace coefficient "
    f"c_{m_mode} = {c3_exact:.6e}")

log(f"\n{'M':>3s} | {'rel L2 (mantle)':>15s} | {'c_3 / exact':>12s} | "
    f"{'max |other c|':>13s}")
log("-" * 56)

for M in (1, 2, 3, 4, 5):
    solver = GravityPoissonSolver(mesh, rho, M,
                                  outer=(OUTER_ID, R_grav),
                                  gamma=gamma, degree=degree)
    solver.check_boundary_quadrature(rtol=1e-6)
    psi_h = solver.solve()
    rel_err = mantle_relative_error(psi_h, mesh, subm, m_mode, analytical)

    coeffs = solver.coefficients()["outer"]
    all_c = np.concatenate([coeffs["cos"], coeffs["sin"]])
    if M >= m_mode:
        c3 = coeffs["cos"][m_mode - 1]
        others = np.delete(all_c, m_mode - 1)  # c_3 is at index m-1 of the cos block
        c3_str = f"{c3 / c3_exact:12.6f}"
    else:
        others = all_c
        c3_str = f"{'--':>12s}"
    log(f"{M:3d} | {rel_err:15.6e} | {c3_str} | "
        f"{np.max(np.abs(others)):13.3e}")

log("\nDone.")
