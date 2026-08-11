"""Dirichlet vs Robin (single-mode DtN) boundary treatment for the gravity Poisson equation.

Same setup as gravity_poisson_test.py (density shell on a mantle submesh,
validated against passess), but run as a four-case matrix at m=3:

    R_grav = 10*rmax  x  {Dirichlet psi=0, Robin dpsi/dr + (m/R) psi = 0}
    R_grav =  2*rmax  x  {Dirichlet psi=0, Robin}

At m=3 the exterior potential decays as r^-3, so truncating the domain at
2*rmax makes the homogeneous Dirichlet condition wrong at the ~10% level,
while the Robin condition is exact for the mode present at any radius.
The Robin run at 10*rmax gives the pure discretisation floor.

Potential is CG2 on the full mesh; density is DG0 on the mantle submesh.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import *
from gadopt import log
from passess.polar import PoissonPolar2D
from generate_gravity_disc import generate_mesh

# Non-dimensional parameters (as in gravity_poisson_test.py)
rmin, rmax = 1.22, 2.22
D_km = 2891.0

depth_km = 500.0
thickness_km = 50.0
r_center = rmax - depth_km / D_km
r1_shell = r_center - thickness_km / (2 * D_km)
r2_shell = r_center + thickness_km / (2 * D_km)

rho_m = 1.0
gamma = 1.0
degree = 2
lc_mantle = 0.04

OUTER_ID = 1


def get_mesh(R_grav_factor, lc_exterior):
    """Generate (once) and return the mesh file for a given outer radius."""
    fname = f"gravity_disc_R{R_grav_factor}.msh"
    if not os.path.exists(fname):
        log(f"Generating {fname} ...")
        generate_mesh(rmin=rmin, rmax=rmax,
                      r1_shell=r1_shell, r2_shell=r2_shell,
                      R_grav_factor=R_grav_factor,
                      lc_mantle=lc_mantle, lc_exterior=lc_exterior,
                      filename=fname)
    return fname


def curve_mesh(linear_mesh):
    """Remap to quadratic (P2) coordinates so mesh circles become curved.

    New midpoint nodes are pushed radially to the linear interpolant of the
    vertex radii, so edges whose vertices lie on a circle become quadratic
    arcs on that circle (trick from the cylindrical Stokes benchmarks).
    The conditional guards the coordinate singularity at the disc centre.
    """
    X = SpatialCoordinate(linear_mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        conditional(r > 1e-12, r_p1 / r, 1.0) * X)
    return Mesh(X_p2)


def make_mantle_submesh(full_mesh):
    """Relabel mantle cells and extract them as a Submesh."""
    DG0_full = FunctionSpace(full_mesh, "DG", 0)
    X = SpatialCoordinate(full_mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    F_mantle = Function(DG0_full).interpolate(
        conditional(And(r >= rmin, r <= rmax), 1, 0))
    F_all = Function(DG0_full).interpolate(conditional(r >= 0, 1, 0))
    mesh = RelabeledMesh(full_mesh, [F_mantle, F_all], [98, 99])
    subm = Submesh(mesh, 2, 98)
    return mesh, subm


def shell_density(subm, m_mode):
    """DG0 cos(m*phi) density in the (mesh-conforming) shell, zero elsewhere."""
    rho = Function(FunctionSpace(subm, "DG", 0), name="density")
    coords_sub = Function(VectorFunctionSpace(subm, "DG", 0)).interpolate(
        SpatialCoordinate(subm))
    sc = coords_sub.dat.data_ro
    r_sc = np.sqrt(sc[:, 0]**2 + sc[:, 1]**2)
    phi_sc = np.arctan2(sc[:, 1], sc[:, 0])
    mask = (r_sc >= r1_shell) & (r_sc <= r2_shell)
    rho.dat.data[:] = np.where(mask, rho_m * np.cos(m_mode * phi_sc), 0.0)
    return rho


def mantle_relative_error(psi_h, mesh, subm, m_mode, analytical):
    """Relative L2 error of psi_h against the analytical mode, mantle only."""
    V = psi_h.function_space()
    deg = V.ufl_element().degree()
    coords_V = Function(VectorFunctionSpace(mesh, "CG", deg)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords_V.dat.data_ro
    r_gc = np.sqrt(gc[:, 0]**2 + gc[:, 1]**2)
    phi_gc = np.arctan2(gc[:, 1], gc[:, 0])

    psi_exact = Function(V)
    tol = 2 * lc_mantle
    in_mantle = (r_gc >= rmin - tol) & (r_gc <= rmax + tol)
    vals = np.zeros(len(r_gc))
    vals[in_mantle] = np.array([
        analytical.psi_m(ri).real * np.cos(m_mode * phi_i)
        for ri, phi_i in zip(r_gc[in_mantle], phi_gc[in_mantle])])
    psi_exact.dat.data[:] = vals

    dx_sub = Measure("dx", domain=subm,
                     intersect_measures=(Measure("dx", domain=mesh),))
    error = float(sqrt(assemble((psi_h - psi_exact)**2 * dx_sub)))
    norm = float(sqrt(assemble(psi_exact**2 * dx_sub)))
    return error / norm


def solve_case(mesh_file, R_grav, robin, m_mode, analytical, curved=True):
    full_mesh = Mesh(mesh_file)
    if curved:
        full_mesh = curve_mesh(full_mesh)
    mesh, subm = make_mantle_submesh(full_mesh)

    V = FunctionSpace(mesh, "CG", degree)
    rho = shell_density(subm, m_mode)

    dx_full = Measure("dx", domain=mesh,
                      intersect_measures=(Measure("dx", domain=subm),))
    dx_sub = Measure("dx", domain=subm,
                     intersect_measures=(Measure("dx", domain=mesh),))

    V_dummy = FunctionSpace(subm, "DG", 0)
    W = V * V_dummy
    w = Function(W)
    psi, lam = split(w)
    v, mu = TestFunctions(W)

    F = (
        inner(grad(psi), grad(v)) * dx_full
        - 4 * np.pi * gamma * rho * v * dx_sub
        + inner(lam, mu) * Measure("dx", domain=subm)
    )

    if robin:
        # Exterior DtN for a single mode m: dpsi/dr + (m/R) psi = 0 at r = R
        F += (m_mode / R_grav) * psi * v * ds(OUTER_ID, domain=mesh)
        bcs = []
    else:
        bcs = [DirichletBC(W.sub(0), 0.0, OUTER_ID)]

    solve(F == 0, w, bcs=bcs,
          solver_parameters={
              "mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps",
          })

    psi_h = w.subfunctions[0]
    rel_err = mantle_relative_error(psi_h, mesh, subm, m_mode, analytical)
    return rel_err, V.dim()


if __name__ == "__main__":
    m_mode = 3
    analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                                r1=r1_shell, r2=r2_shell, gamma=gamma)

    cases = [
        # (label, R_grav_factor, lc_exterior, robin)
        ("R=10*rmax  Dirichlet", 10, 2.0, False),
        ("R=10*rmax  Robin    ", 10, 2.0, True),
        ("R= 2*rmax  Dirichlet", 2, 0.2, False),
        ("R= 2*rmax  Robin    ", 2, 0.2, True),
    ]

    log(f"m = {m_mode}, CG{degree}, lc_mantle = {lc_mantle}")
    log(f"Shell: [{r1_shell:.4f}, {r2_shell:.4f}]")
    for factor in (10, 2):
        R = factor * rmax
        trunc = (analytical.psi_m(R) / analytical.psi_m(rmax)).real
        log(f"psi(R={R:.2f}) / psi(rmax) = {trunc:.3e}  "
                        f"(Dirichlet truncation scale)")

    log(f"\n{'case':>22s} | {'rel L2 (mantle)':>15s} | {'DOFs (psi)':>10s}")
    log("-" * 55)
    for label, factor, lc_ext, robin in cases:
        mesh_file = get_mesh(factor, lc_ext)
        rel_err, ndofs = solve_case(mesh_file, factor * rmax, robin,
                                    m_mode, analytical)
        log(f"{label:>22s} | {rel_err:15.6e} | {ndofs:10d}")

    log("\nDone.")
