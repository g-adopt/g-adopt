"""Convergence-rate benchmark: truncated annulus, exact source, modal DtN (2D).

This is the study the baseline never had. With the analytic source
integrated exactly (no DG0 azimuthal cap, see gravity_poisson_exact_source.py)
and the DtN truncation held far below the discretisation error, the only
remaining error source is the finite-element discretisation itself, so the
mantle L2 error should fall at the theoretical rate:

    CG1 -> O(h^2),   CG2 -> O(h^3).

This is config D: the exterior DtN is placed DIRECTLY at rmax with no
buffer. An earlier version kept an extended-but-truncated exterior annulus
(rmax -> 2*rmax); a radial-band error decomposition showed the error lived
almost entirely in that coarse buffer (100-200x the mantle error) and bled
back through rmax, capping the mantle accuracy and destroying the
convergence rate. The whole point of DtN is that no buffer is needed --
the exterior of rmax is source-free, so the exterior DtN there is exact --
so we remove it. Domain = mantle only.

Setup held fixed while h shrinks:
  - structured annulus rmin -> rmax (no exterior buffer), curved to P2
    geometry so facet error does not cap CG2;
  - interior DtN at rmin and exterior DtN at rmax, both modal with
    M = 5 >> m = 3 (truncation error negligible);
  - source = cos(m*phi) integrated exactly over the shell subdomain.

Refinement is isotropic: dr_mantle, the azimuthal cell count, and the
number of shell layers all halve/double together. The error is measured
against the analytic solution in a CG(deg+3) space (reference-interpolation
error higher-order than the solution error, so it does not pollute the
rate).
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
    curve_mesh, rmin, rmax, r1_shell, r2_shell, rho_m, gamma, OUTER_ID)
from generate_gravity_annulus import generate_annulus

INNER_ID = 2
SHELL_ID = 103
MANTLE_IDS = (102, 103, 104)
m_mode = 3
M_trunc = 5
R_grav = rmax  # config D: exterior DtN at rmax, no buffer

analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                            r1=r1_shell, r2=r2_shell, gamma=gamma)


def refined_mesh(dr_mantle):
    """Structured annulus whose radial, azimuthal and shell resolution all
    scale with dr_mantle, curved to P2 geometry."""
    n_azimuthal = int(round(2 * np.pi * r1_shell / dr_mantle))
    n_azimuthal += (-n_azimuthal) % 4  # round up to a multiple of 4
    n_shell = max(1, int(round((r2_shell - r1_shell) / dr_mantle)))
    fname = f"gravity_annulus_confD_dr{dr_mantle:.4f}.msh"
    if not os.path.exists(fname):
        generate_annulus(rmin=rmin, rmax=rmax, r1_shell=r1_shell,
                         r2_shell=r2_shell, R_grav_factor=1,
                         dr_mantle=dr_mantle, n_shell_layers=n_shell,
                         n_azimuthal=n_azimuthal, filename=fname)
    return curve_mesh(Mesh(fname)), n_azimuthal


def rel_error(psi_h, mesh, ref_degree):
    Vh = FunctionSpace(mesh, "CG", ref_degree)
    coords = Function(VectorFunctionSpace(mesh, "CG", ref_degree)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords.dat.data_ro
    r = np.sqrt(gc[:, 0]**2 + gc[:, 1]**2)
    phi = np.arctan2(gc[:, 1], gc[:, 0])
    psi_ref = Function(Vh)
    psi_ref.dat.data[:] = np.array([analytical.psi_m(ri).real for ri in r]) \
        * np.cos(m_mode * phi)
    psi_num = Function(Vh).interpolate(psi_h)
    dxm = dx(MANTLE_IDS, domain=mesh, degree=2 * ref_degree)
    err = float(sqrt(assemble((psi_num - psi_ref)**2 * dxm)))
    nrm = float(sqrt(assemble(psi_ref**2 * dxm)))
    return err / nrm


dr_levels = [0.04, 0.02, 0.01]

for deg in (1, 2):
    log(f"\n=== CG{deg}  (expect order {deg + 1}) ===")
    log(f"{'dr':>7s} | {'n_azim':>7s} | {'dofs':>9s} | "
        f"{'rel L2 error':>13s} | {'order':>6s}")
    log("-" * 56)
    prev_err = prev_h = None
    for dr in dr_levels:
        mesh, n_azim = refined_mesh(dr)
        shell_area = assemble(Constant(1.0) * dx(SHELL_ID, domain=mesh))
        assert abs(shell_area / (np.pi * (r2_shell**2 - r1_shell**2)) - 1) < 1e-3

        X = SpatialCoordinate(mesh)
        src = rho_m * cos(m_mode * atan2(X[1], X[0]))
        solver = GravityPoissonSolver(
            mesh, source_expr=src, source_id=SHELL_ID,
            source_degree=2 * m_mode + 8, M=M_trunc,
            outer=(OUTER_ID, R_grav), inner=(INNER_ID, rmin),
            gamma=gamma, degree=deg)
        psi_h = solver.solve()
        err = rel_error(psi_h, mesh, ref_degree=deg + 3)

        if prev_err is None:
            order_str = f"{'--':>6s}"
        else:
            order = np.log(prev_err / err) / np.log(prev_h / dr)
            order_str = f"{order:6.2f}"
        log(f"{dr:7.4f} | {n_azim:7d} | {solver.V.dim():9d} | "
            f"{err:13.4e} | {order_str}")
        prev_err, prev_h = err, dr

log("\nDone.")
