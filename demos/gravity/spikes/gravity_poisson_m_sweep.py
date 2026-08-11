"""Dirichlet vs Robin on the truncated (R = 2*rmax) disc, swept over azimuthal mode m.

For each m the density is cos(m*phi) in the shell and the Robin coefficient
is m/R. The Dirichlet truncation error scales as (r_s/R)^m, so it grows as
m decreases; the Robin condition is exact for the mode present at any m,
so its error should stay at the discretisation floor throughout.

Uses solve_case from gravity_poisson_robin (same mesh, spaces, submesh
coupling and error assessment).
"""
import sys
import os

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from gadopt import log
from passess.polar import PoissonPolar2D
from gravity_poisson_robin import (
    solve_case, get_mesh, rmax, r1_shell, r2_shell, rho_m, gamma)

R_grav_factor = 2
lc_exterior = 0.2
R_grav = R_grav_factor * rmax

mesh_file = get_mesh(R_grav_factor, lc_exterior)

log(f"R_grav = {R_grav_factor}*rmax = {R_grav:.2f}")
log(f"\n{'m':>3s} | {'psi(R)/psi(rmax)':>16s} | {'Dirichlet':>12s} | "
    f"{'Robin':>12s} | {'ratio':>8s}")
log("-" * 65)

for m_mode in (1, 2, 3, 5, 10):
    analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                                r1=r1_shell, r2=r2_shell, gamma=gamma)
    trunc = (analytical.psi_m(R_grav) / analytical.psi_m(rmax)).real

    err_d, _ = solve_case(mesh_file, R_grav, False, m_mode, analytical)
    err_r, _ = solve_case(mesh_file, R_grav, True, m_mode, analytical)
    log(f"{m_mode:3d} | {trunc:16.3e} | {err_d:12.4e} | {err_r:12.4e} | "
        f"{err_d / err_r:8.1f}")

log("\nDone.")
