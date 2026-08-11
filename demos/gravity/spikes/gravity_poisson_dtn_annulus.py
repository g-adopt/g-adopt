"""Interior + exterior modal DtN on the structured annulus (no centre disc).

The domain is an annulus from rmin to R_grav = 2*rmax (structured
'extruded-style' quad mesh from generate_gravity_annulus.py): the disc
interior r < rmin is gone entirely, replaced by the interior DtN map
dpsi/dr - (m/rmin) psi = 0 at the inner boundary. The exterior modal DtN
acts at R_grav as validated before.

The interior map matters most at LOW m: the interior field grows as r^m,
so for m = 1, 2 a wrong inner treatment (naive Neumann or Dirichlet)
produces leading-order errors, while at high m the core is negligible.
Sweeping m in {1, 2, 3, 5} with truncation M = 5 throughout:

  - the mantle error should sit at the discretisation floor for every m,
    with no low-m degradation (the E3 pass criterion);
  - the solved trace coefficients at BOTH boundaries should match the
    analytical amplitudes psi_m(R_grav) and psi_m(rmin).

Density stays on the mantle Submesh (the reason this mesh is gmsh-built
rather than a Firedrake ExtrudedMesh, which Submesh cannot handle).
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
    curve_mesh, make_mantle_submesh, shell_density, mantle_relative_error,
    rmin, rmax, r1_shell, r2_shell, rho_m, gamma, degree, OUTER_ID)

INNER_ID = 2
M_trunc = 5
R_grav = 2 * rmax

mesh_file = "gravity_annulus.msh"
if not os.path.exists(mesh_file):
    log(f"Generating {mesh_file} ...")
    from generate_gravity_annulus import generate_annulus
    generate_annulus(rmin=rmin, rmax=rmax,
                     r1_shell=r1_shell, r2_shell=r2_shell,
                     R_grav_factor=2)

mesh, subm = make_mantle_submesh(curve_mesh(Mesh(mesh_file)))

log(f"Annulus [{rmin}, {R_grav:.2f}], no centre disc; "
    f"interior + exterior modal DtN, M = {M_trunc}")
log(f"\n{'m':>3s} | {'rel L2 (mantle)':>15s} | {'c_m(out)/exact':>14s} | "
    f"{'c_m(in)/exact':>13s} | {'max inactive':>12s}")
log("-" * 72)

for m_mode in (1, 2, 3, 5):
    analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                                r1=r1_shell, r2=r2_shell, gamma=gamma)
    rho = shell_density(subm, m_mode)

    solver = GravityPoissonSolver(mesh, rho, M_trunc,
                                  outer=(OUTER_ID, R_grav),
                                  inner=(INNER_ID, rmin),
                                  gamma=gamma, degree=degree)
    solver.check_boundary_quadrature(rtol=1e-6)
    psi_h = solver.solve()
    rel_err = mantle_relative_error(psi_h, mesh, subm, m_mode, analytical)

    coeffs = solver.coefficients()
    ratios = []
    for side, R_b in (("outer", R_grav), ("inner", rmin)):
        c_num = coeffs[side]["cos"][m_mode - 1]
        ratios.append(c_num / analytical.psi_m(R_b).real)
    inactive = np.concatenate(
        [np.delete(coeffs[s]["cos"], m_mode - 1) for s in ("outer", "inner")]
        + [coeffs[s]["sin"] for s in ("outer", "inner")]
        + [[coeffs["inner"]["mean"]]])
    log(f"{m_mode:3d} | {rel_err:15.6e} | {ratios[0]:14.6f} | "
        f"{ratios[1]:13.6f} | {np.max(np.abs(inactive)):12.3e}")

log("\nDone.")
