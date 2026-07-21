"""Surface mass sheet ON the DtN boundary: the inhomogeneous-DtN validation.

In the coupled problems (dynamic topography in mantle convection,
load-induced topography in GIA) mass sheets sit exactly on the domain
boundaries. Roadmap section 5.4: the sheet enters through the flux jump
[dpsi/dr] = -4 pi gamma sigma, so the DtN condition becomes inhomogeneous,

    exterior at rmax:  dpsi/dr + (m/R) psi = 4 pi gamma sigma,
    interior at rmin:  dpsi/dr - (m/R) psi = -4 pi gamma sigma,

and in the weak form BOTH reduce to the same extra right-hand-side term
-4 pi gamma * integral(sigma v ds) on the marked boundary, with the modal
DtN machinery itself completely unchanged. This script validates exactly
that: config-D annulus (no buffer), zero volume density, a single-mode
sheet sigma_m cos(m phi) placed on the outer (then the inner) DtN boundary,
blind modal DtN with M = 5, validated against passess SheetPolar2D.

Inside the domain the sheet potential is a pure harmonic mode
(~ r^m cos(m phi) below an outer sheet, ~ r^-m cos(m phi) above an inner
one), so with CG2 the m <= 2 cases are near-exact (geometry-limited) and
m = 3, 5 sit at the usual discretisation floor.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import (
    Mesh, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    TestFunctions, Constant, assemble, sqrt, dx, ds, cos, atan2, pi)
from gadopt import log
from passess.polar import SheetPolar2D
from gravity_dtn import GravityPoissonSolver
from gravity_poisson_robin import (
    curve_mesh, rmin, rmax, r1_shell, r2_shell, gamma, degree, OUTER_ID)
from generate_gravity_annulus import generate_annulus

INNER_ID = 2
M_trunc = 5
sigma_amp = 1.0
dr_mantle = 0.02


def confD_mesh(dr):
    """Config-D structured annulus (same construction as the convergence
    study), curved to P2 geometry."""
    n_azimuthal = int(round(2 * np.pi * r1_shell / dr))
    n_azimuthal += (-n_azimuthal) % 4
    n_shell = max(1, int(round((r2_shell - r1_shell) / dr)))
    fname = f"gravity_annulus_confD_dr{dr:.4f}.msh"
    if not os.path.exists(fname):
        generate_annulus(rmin=rmin, rmax=rmax, r1_shell=r1_shell,
                         r2_shell=r2_shell, R_grav_factor=1,
                         dr_mantle=dr, n_shell_layers=n_shell,
                         n_azimuthal=n_azimuthal, filename=fname)
    return curve_mesh(Mesh(fname))


def solve_with_sheet(mesh, m_mode, side):
    """Modal DtN solve (blind, M = M_trunc) with a single-mode sheet on one
    DtN boundary. The sheet is added as the extra RHS boundary term; the
    GravityPoissonSolver DtN machinery is untouched."""
    solver = GravityPoissonSolver(
        mesh, source_expr=Constant(0.0), M=M_trunc,
        outer=(OUTER_ID, rmax), inner=(INNER_ID, rmin),
        gamma=gamma, degree=degree)
    solver.check_boundary_quadrature(rtol=1e-6)

    X = SpatialCoordinate(mesh)
    phi = atan2(X[1], X[0])
    sigma = sigma_amp * cos(m_mode * phi)
    bid = OUTER_ID if side == "outer" else INNER_ID
    dss = ds(bid, domain=mesh, degree=solver.quad_degree)
    v = TestFunctions(solver.w.function_space())[0]
    solver.F -= 4 * pi * gamma * sigma * v * dss

    return solver, solver.solve()


def rel_error(psi_h, mesh, sheet, m_mode, ref_degree):
    Vh = FunctionSpace(mesh, "CG", ref_degree)
    coords = Function(VectorFunctionSpace(mesh, "CG", ref_degree)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords.dat.data_ro
    r = np.sqrt(gc[:, 0]**2 + gc[:, 1]**2)
    phi = np.arctan2(gc[:, 1], gc[:, 0])
    psi_ref = Function(Vh)
    psi_ref.dat.data[:] = np.array([sheet.psi_m(ri).real for ri in r]) \
        * np.cos(m_mode * phi)
    psi_num = Function(Vh).interpolate(psi_h)
    dxm = dx(domain=mesh, degree=2 * ref_degree)
    err = float(sqrt(assemble((psi_num - psi_ref)**2 * dxm)))
    nrm = float(sqrt(assemble(psi_ref**2 * dxm)))
    return err / nrm


mesh = confD_mesh(dr_mantle)
log(f"Config-D annulus [{rmin}, {rmax}], dr = {dr_mantle}, CG{degree}, "
    f"modal DtN M = {M_trunc}, sheet amplitude {sigma_amp}")

for side, a in (("outer", rmax), ("inner", rmin)):
    log(f"\n--- sheet on the {side} boundary (a = {a}) ---")
    log(f"{'m':>3s} | {'rel L2':>11s} | {'c_m({0})/exact':>15s} | "
        f"{'max inactive':>12s}".format(side))
    log("-" * 52)
    for m_mode in (1, 2, 3, 5):
        sheet = SheetPolar2D(m=m_mode, sigma_m=sigma_amp, a=a, gamma=gamma)
        solver, psi_h = solve_with_sheet(mesh, m_mode, side)
        err = rel_error(psi_h, mesh, sheet, m_mode, ref_degree=degree + 3)

        coeffs = solver.coefficients()
        c_num = coeffs[side]["cos"][m_mode - 1]
        ratio = c_num / sheet.psi_m(a).real
        inactive = np.concatenate(
            [np.delete(coeffs[s]["cos"], m_mode - 1) for s in ("outer", "inner")]
            + [coeffs[s]["sin"] for s in ("outer", "inner")]
            + [[coeffs["inner"]["mean"]]])
        log(f"{m_mode:3d} | {err:11.4e} | {ratio:15.8f} | "
            f"{np.max(np.abs(inactive)):12.3e}")

log("\nDone.")
