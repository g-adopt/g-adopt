"""Gravity Poisson equation on a disc mesh with submesh coupling.

Solves -nabla^2 psi = 4 pi gamma rho on a full disc (r=0 to R_grav),
where rho is a density anomaly defined on the mantle submesh.

The density shell [r1, r2] has mesh-conforming boundaries (element edges
align with the shell radii). Density is DG0 on the submesh: cos(m*phi)
inside the shell, zero elsewhere. Potential psi is CG1 on the full mesh.

Error is assessed only within the mantle region where the solution matters.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import *
from passess.polar import PoissonPolar2D

# Non-dimensional parameters
rmin, rmax = 1.22, 2.22
D_km = 2891.0
R_grav = 10 * rmax

# Density shell: 50 km thick at 500 km depth
depth_km = 500.0
thickness_km = 50.0
r_center = rmax - depth_km / D_km
r1_shell = r_center - thickness_km / (2 * D_km)
r2_shell = r_center + thickness_km / (2 * D_km)

m_mode = 10
rho_m = 1.0
gamma = 1.0

PETSc.Sys.Print(f"Shell: r1={r1_shell:.6f}, r2={r2_shell:.6f}, "
                f"width={r2_shell - r1_shell:.6f}")
PETSc.Sys.Print(f"Mode m={m_mode}, R_grav={R_grav:.1f}")

# Analytical solution
analytical = PoissonPolar2D(m=m_mode, rho_m=rho_m,
                            r1=r1_shell, r2=r2_shell, gamma=gamma)

PETSc.Sys.Print(f"psi at R_grav: {analytical.psi_m(R_grav):.2e} (truncation error)")

# Generate mesh with shell-conforming boundaries
mesh_file = "gravity_disc.msh"
if not os.path.exists(mesh_file):
    PETSc.Sys.Print("Generating mesh...")
    from generate_gravity_disc import generate_mesh
    generate_mesh(rmin=rmin, rmax=rmax,
                  r1_shell=r1_shell, r2_shell=r2_shell,
                  lc_mantle=0.04, lc_exterior=50.0)

# Load mesh, create mantle submesh
full_mesh = Mesh(mesh_file)
DG0_full = FunctionSpace(full_mesh, "DG", 0)
X = SpatialCoordinate(full_mesh)
r = sqrt(X[0]**2 + X[1]**2)

F_mantle = Function(DG0_full).interpolate(
    conditional(And(r >= rmin, r <= rmax), 1, 0))
F_all = Function(DG0_full).interpolate(conditional(r >= 0, 1, 0))
mesh = RelabeledMesh(full_mesh, [F_mantle, F_all], [98, 99])
subm = Submesh(mesh, 2, 98)

PETSc.Sys.Print(f"Mantle area: {assemble(Constant(1) * dx(domain=subm)):.4f} "
                f"(expected {np.pi * (rmax**2 - rmin**2):.4f})")

# Function spaces
V = FunctionSpace(mesh, "CG", 1)
DG0_sub = FunctionSpace(subm, "DG", 0)

# Density on submesh via dat.data
rho = Function(DG0_sub, name="density")
coords_sub = Function(VectorFunctionSpace(subm, "DG", 0)).interpolate(
    SpatialCoordinate(subm))
sc = coords_sub.dat.data_ro
r_sc = np.sqrt(sc[:, 0]**2 + sc[:, 1]**2)
phi_sc = np.arctan2(sc[:, 1], sc[:, 0])

# Shell boundaries are mesh-conforming, so the mask is exact for DG0
mask = (r_sc >= r1_shell) & (r_sc <= r2_shell)
rho.dat.data[:] = np.where(mask, rho_m * np.cos(m_mode * phi_sc), 0.0)

n_shell = np.count_nonzero(mask)
PETSc.Sys.Print(f"Shell cells: {n_shell} / {len(mask)}")

# Check m=0 leakage
integral_rho = assemble(rho * Measure("dx", domain=subm))
PETSc.Sys.Print(f"integral(rho): {integral_rho:.6e} (should be ~0 for m!=0)")

# Measures for cross-mesh coupling
dx_full = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", domain=subm),))
dx_sub = Measure("dx", domain=subm, intersect_measures=(Measure("dx", domain=mesh),))

# Variational form via MixedFunctionSpace
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

bc = DirichletBC(W.sub(0), 0.0, 1)
solve(F == 0, w, bcs=[bc],
      solver_parameters={
          "mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
      })

psi_h = w.subfunctions[0]
psi_h.rename("gravitational_potential")

# Analytical potential on full mesh (for comparison within mantle)
coords_V = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
    SpatialCoordinate(mesh))
gc = coords_V.dat.data_ro
r_gc = np.sqrt(gc[:, 0]**2 + gc[:, 1]**2)
phi_gc = np.arctan2(gc[:, 1], gc[:, 0])

psi_exact_vals = np.array([
    analytical.psi_m(ri).real * np.cos(m_mode * phi_i)
    for ri, phi_i in zip(r_gc, phi_gc)])

psi_exact = Function(V, name="psi_exact")
psi_exact.dat.data[:] = psi_exact_vals

# Error only within the mantle
error_mantle = sqrt(assemble((psi_h - psi_exact)**2 * dx_sub))
norm_mantle = sqrt(assemble(psi_exact**2 * dx_sub))
PETSc.Sys.Print(f"\nMantle L2 error:    {float(error_mantle):.6e}")
PETSc.Sys.Print(f"Mantle L2 norm:     {float(norm_mantle):.6e}")
PETSc.Sys.Print(f"Mantle relative:    {float(error_mantle / norm_mantle):.6e}")

# Point comparison along phi=0 (within mantle only)
near_axis = np.abs(phi_gc) < 0.03
in_mantle = (r_gc >= rmin) & (r_gc <= rmax)
idx = np.where(near_axis & in_mantle)[0]
r_ax = r_gc[idx]
order = np.argsort(r_ax)
r_ax = r_ax[order]
idx = idx[order]

psi_exact_ax = np.array([analytical.psi_m(ri).real for ri in r_ax])

PETSc.Sys.Print(f"\nPoint comparison along phi=0 (mantle):")
PETSc.Sys.Print(f"  {'r':>7s} | {'numerical':>11s} | {'analytical':>11s} | {'ratio':>7s}")
PETSc.Sys.Print("  " + "-" * 47)
step = max(1, len(r_ax) // 12)
for i in range(0, len(r_ax), step):
    num_val = psi_h.dat.data_ro[idx[i]]
    exact_val = psi_exact_ax[i]
    ratio = num_val / exact_val if abs(exact_val) > 1e-15 else float('nan')
    PETSc.Sys.Print(f"  {r_ax[i]:7.4f} | {num_val:11.6e} | {exact_val:11.6e} | {ratio:7.4f}")

# Output
VTKFile("gravity_submesh.pvd").write(psi_h, psi_exact)
VTKFile("gravity_density.pvd").write(rho)
PETSc.Sys.Print("\nDone.")
