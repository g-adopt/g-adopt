"""Is the DG0 density the floor? Exact-source vs DG0 on the annulus.

A/B test isolating the *source representation* error. Everything is held
fixed -- same structured annulus, same interior+exterior modal DtN, same
solver -- and only the right-hand side differs:

  A (DG0):   rho = DG0 field, cos(m*phi) sampled at cell centroids in the
             mesh-conforming shell (what every previous script used).
  B (exact): the analytic source cos(m*phi) integrated directly over the
             shell subdomain with high quadrature -- no density field.

The DG0 source error is azimuthal: sampling cos(m*phi) as cell-wise
constants is O(h_phi) in L2, propagating to O(h_phi^2) in the potential,
whereas the exact source has NO azimuthal error. So we sweep the azimuthal
resolution n_azimuthal at fixed radial resolution: if DG0 was the floor,
its error falls with n_azimuthal while the exact-source error stays flat
(set by the radial CG2 discretisation). If they already agree, the density
was never the limiter.

Error is assessed against the analytic solution in a rich space (CG5, the
true potential evaluated at all nodes and integrated over the mantle
subdomains) so the reference-interpolation error does not contaminate the
measurement. No Submesh is used -- density lives on the full mesh, which
is also the config-D style.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import (
    Mesh, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    Constant, assemble, sqrt, dx, cos, atan2)
from gadopt import log
from passess.polar import PoissonPolar2D
from gravity_dtn import GravityPoissonSolver
from gravity_poisson_robin import (
    curve_mesh, rmin, rmax, r1_shell, r2_shell, rho_m, gamma, degree, OUTER_ID)
from generate_gravity_annulus import generate_annulus

INNER_ID = 2
SHELL_ID = 103                 # density_shell physical region in the annulus
MANTLE_IDS = (102, 103, 104)   # below-shell, shell, above-shell
M_trunc = 5
m_mode = 3
R_grav = 2 * rmax
REF_DEGREE = 5                 # rich space for the error norm

analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                            r1=r1_shell, r2=r2_shell, gamma=gamma)


def annulus_mesh(n_azimuthal, dr_mantle=0.02):
    fname = f"gravity_annulus_naz{n_azimuthal}.msh"
    if not os.path.exists(fname):
        generate_annulus(rmin=rmin, rmax=rmax, r1_shell=r1_shell,
                         r2_shell=r2_shell, R_grav_factor=2,
                         dr_mantle=dr_mantle, n_azimuthal=n_azimuthal,
                         filename=fname)
    return curve_mesh(Mesh(fname))


def dg0_density(mesh):
    """cos(m*phi) DG0 on the full mesh, nonzero only in the shell."""
    rho = Function(FunctionSpace(mesh, "DG", 0), name="density")
    coords = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
        SpatialCoordinate(mesh))
    sc = coords.dat.data_ro
    r_sc = np.sqrt(sc[:, 0]**2 + sc[:, 1]**2)
    phi_sc = np.arctan2(sc[:, 1], sc[:, 0])
    mask = (r_sc >= r1_shell) & (r_sc <= r2_shell)
    rho.dat.data[:] = np.where(mask, rho_m * np.cos(m_mode * phi_sc), 0.0)
    return rho


def highorder_rel_error(psi_h, mesh):
    """Relative L2 error over the mantle in a CG(REF_DEGREE) space.

    The analytic potential psi_m(r) cos(m phi) is valid at every radius, so
    it is evaluated at all nodes; the norm is taken over the mantle
    subdomains only, with the interpolation of psi_h into CG5 exact
    (CG2 subset CG5).
    """
    Vh = FunctionSpace(mesh, "CG", REF_DEGREE)
    coords = Function(VectorFunctionSpace(mesh, "CG", REF_DEGREE)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords.dat.data_ro
    r = np.sqrt(gc[:, 0]**2 + gc[:, 1]**2)
    phi = np.arctan2(gc[:, 1], gc[:, 0])

    psi_ref = Function(Vh)
    psi_ref.dat.data[:] = np.array([analytical.psi_m(ri).real for ri in r]) \
        * np.cos(m_mode * phi)
    psi_num = Function(Vh).interpolate(psi_h)

    dxm = dx(MANTLE_IDS, domain=mesh, degree=2 * REF_DEGREE)
    err = float(sqrt(assemble((psi_num - psi_ref)**2 * dxm)))
    nrm = float(sqrt(assemble(psi_ref**2 * dxm)))
    return err / nrm


log(f"Exact-source vs DG0, m = {m_mode}, M = {M_trunc}, annulus to {R_grav:.2f}")
log(f"Error assessed in CG{REF_DEGREE} over mantle regions {MANTLE_IDS}\n")
log(f"{'n_azim':>7s} | {'h_phi(shell)':>12s} | {'DG0 error':>12s} | "
    f"{'exact error':>12s} | {'DG0/exact':>10s}")
log("-" * 66)

for n_azimuthal in (256, 512, 1024):
    mesh = annulus_mesh(n_azimuthal)

    # sanity: the shell subdomain survived curving
    shell_area = assemble(Constant(1.0) * dx(SHELL_ID, domain=mesh))
    exact_area = np.pi * (r2_shell**2 - r1_shell**2)
    assert abs(shell_area / exact_area - 1) < 1e-3, \
        f"shell region {SHELL_ID} lost after curving ({shell_area:.3e})"

    common = dict(outer=(OUTER_ID, R_grav), inner=(INNER_ID, rmin),
                  gamma=gamma, degree=degree)

    # A: DG0 source
    rho = dg0_density(mesh)
    psi_a = GravityPoissonSolver(mesh, rho=rho, M=M_trunc, **common).solve()
    err_dg0 = highorder_rel_error(psi_a, mesh)

    # B: exact analytic source over the shell subdomain
    X = SpatialCoordinate(mesh)
    src = rho_m * cos(m_mode * atan2(X[1], X[0]))
    psi_b = GravityPoissonSolver(mesh, source_expr=src, source_id=SHELL_ID,
                                 source_degree=2 * m_mode + 8, M=M_trunc,
                                 **common).solve()
    err_exact = highorder_rel_error(psi_b, mesh)

    h_phi = 2 * np.pi * r1_shell / n_azimuthal
    log(f"{n_azimuthal:7d} | {h_phi:12.4e} | {err_dg0:12.4e} | "
        f"{err_exact:12.4e} | {err_dg0 / err_exact:10.2f}")

log("\nDone.")
