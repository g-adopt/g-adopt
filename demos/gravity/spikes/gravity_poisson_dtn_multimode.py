"""Modal DtN with a superposed multi-mode density on the truncated disc.

The density shell now carries three azimuthal modes at once,

    rho = [1.0*cos(2*phi) + 0.7*cos(3*phi) + 0.4*cos(5*phi)] * rho_shell(r),

so no single Robin coefficient can be exact and the modal machinery has to
do real work: discover all three trace amplitudes simultaneously. Because
the Poisson equation is linear, the analytical solution is still exact --
the sum of the passess single-mode solutions -- so this is a genuine
multi-mode validation, not a comparison against another numerical result.

Sweeping the truncation M, the error should drop stepwise as M crosses
each active mode (2, then 3, then 5) and sit at the discretisation floor
from M = 5 onward, with every solved coefficient matching its analytical
trace amplitude and all inactive coefficients at machine zero.

Mesh, submesh coupling, and spaces are shared with the single-mode tests
via gravity_poisson_robin.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import (
    Mesh, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    Measure, assemble, sqrt)
from gadopt import log
from passess.polar import PoissonPolar2D
from gravity_dtn import GravityPoissonSolver
from gravity_poisson_robin import (
    get_mesh, curve_mesh, make_mantle_submesh,
    rmin, rmax, r1_shell, r2_shell, gamma, degree, lc_mantle, OUTER_ID)

# Superposed density: {mode: amplitude}
modes = {2: 1.0, 3: 0.7, 5: 0.4}

R_grav_factor = 2
R_grav = R_grav_factor * rmax

analytical = {m: PoissonPolar2D(m=m, rho_m=amp, r1=r1_shell, r2=r2_shell,
                                gamma=gamma)
              for m, amp in modes.items()}
c_exact = {m: analytical[m].psi_m(R_grav).real for m in modes}

mesh_file = get_mesh(R_grav_factor, 0.2)
mesh, subm = make_mantle_submesh(curve_mesh(Mesh(mesh_file)))


def multimode_shell_density(subm):
    """DG0 superposed cos(m*phi) density in the mesh-conforming shell."""
    rho = Function(FunctionSpace(subm, "DG", 0), name="density")
    coords = Function(VectorFunctionSpace(subm, "DG", 0)).interpolate(
        SpatialCoordinate(subm))
    sc = coords.dat.data_ro
    r_sc = np.sqrt(sc[:, 0]**2 + sc[:, 1]**2)
    phi_sc = np.arctan2(sc[:, 1], sc[:, 0])
    mask = (r_sc >= r1_shell) & (r_sc <= r2_shell)
    vals = sum(amp * np.cos(m * phi_sc) for m, amp in modes.items())
    rho.dat.data[:] = np.where(mask, vals, 0.0)
    return rho


def multimode_mantle_error(psi_h):
    """Relative L2 error against the superposed analytical solution."""
    V = psi_h.function_space()
    deg = V.ufl_element().degree()
    coords = Function(VectorFunctionSpace(mesh, "CG", deg)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords.dat.data_ro
    r_gc = np.sqrt(gc[:, 0]**2 + gc[:, 1]**2)
    phi_gc = np.arctan2(gc[:, 1], gc[:, 0])

    tol = 2 * lc_mantle
    in_mantle = (r_gc >= rmin - tol) & (r_gc <= rmax + tol)
    vals = np.zeros(len(r_gc))
    for m, sol in analytical.items():
        vals[in_mantle] += np.array(
            [sol.psi_m(ri).real for ri in r_gc[in_mantle]]
        ) * np.cos(m * phi_gc[in_mantle])

    psi_exact = Function(V)
    psi_exact.dat.data[:] = vals

    dx_sub = Measure("dx", domain=subm,
                     intersect_measures=(Measure("dx", domain=mesh),))
    error = float(sqrt(assemble((psi_h - psi_exact)**2 * dx_sub)))
    norm = float(sqrt(assemble(psi_exact**2 * dx_sub)))
    return error / norm


rho = multimode_shell_density(subm)

log(f"Density: superposed modes {modes}; solver is NOT told the modes.")
log(f"R_grav = {R_grav:.2f}; analytical trace coefficients: "
    + ", ".join(f"c_{m} = {c:.4e}" for m, c in c_exact.items()))

log(f"\n{'M':>3s} | {'rel L2 (mantle)':>15s} | "
    + " | ".join(f"{f'c_{m}/exact':>10s}" for m in modes)
    + f" | {'max inactive':>12s}")
log("-" * 75)

for M in (1, 2, 3, 4, 5, 6):
    solver = GravityPoissonSolver(mesh, rho, M,
                                  outer=(OUTER_ID, R_grav),
                                  gamma=gamma, degree=degree)
    solver.check_boundary_quadrature(rtol=1e-6)
    psi_h = solver.solve()
    rel_err = multimode_mantle_error(psi_h)

    coeffs = solver.coefficients()["outer"]
    ratio_cols = []
    for m in modes:
        if M >= m:
            ratio_cols.append(f"{coeffs['cos'][m - 1] / c_exact[m]:10.6f}")
        else:
            ratio_cols.append(f"{'--':>10s}")
    active_idx = [m - 1 for m in modes if M >= m]
    inactive = np.delete(
        np.concatenate([coeffs["cos"], coeffs["sin"]]), active_idx)
    log(f"{M:3d} | {rel_err:15.6e} | " + " | ".join(ratio_cols)
        + f" | {np.max(np.abs(inactive)):12.3e}")

log("\nDone.")
