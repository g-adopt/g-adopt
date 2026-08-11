"""3D spherical-shell gravity Poisson with single-(l,m) Robin DtN (config D).

The 3D analogue of the 2D config-D convergence study. Domain is an extruded
cubed-sphere shell rmin -> rmax (no exterior buffer); the source is a single
real spherical harmonic Y_lm in a radial shell; the far-field and core are
imposed by Robin conditions that are exact for that single mode:

    exterior at rmax:  dpsi/dr + ((l+1)/rmax) psi = 0   (exterior ~ r^-(l+1))
    interior at rmin:  dpsi/dr -  (l/rmin)    psi = 0   (interior ~ r^+l)

Both enter the weak form with a positive sign, are SPD for l >= 1 (no
nullspace), and need no R-space multipliers -- this is the 3D entry point
(roadmap E7). Multi-mode modal DtN with Y_lm multipliers is E8.

Mesh mechanics validated in the smoke test: CubedSphereMesh + radial
ExtrudedMesh with variable layer heights conforming to the shell radii,
ds_t/ds_b for the boundaries, exact source over the shell via a radial
indicator (cells are entirely in/out because layers conform). No Submesh
-- which is what makes the extruded production mesh usable.

Validated against passess.spherical.PoissonSpherical3D; convergence in
(lateral refinement_level, radial dr) refined together, for CG1 and CG2.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

from firedrake import (
    CubedSphereMesh, ExtrudedMesh, Function, FunctionSpace,
    VectorFunctionSpace, SpatialCoordinate, Constant, TestFunction,
    TrialFunction, dot, grad, dx, ds_t, ds_b, sqrt, conditional, And,
    assemble, solve, pi)
from gadopt import log
from passess.spherical import PoissonSpherical3D

# Non-dimensional shell (same as 2D)
rmin, rmax = 1.22, 2.22
D_km = 2891.0
r_c = rmax - 500.0 / D_km
r1_shell = r_c - 50.0 / (2 * D_km)
r2_shell = r_c + 50.0 / (2 * D_km)

l_mode, m_mode = 2, 0
rho_lm = 1.0
G_grav = 1.0

analytical = PoissonSpherical3D(l=l_mode, m=m_mode, rho_lm=rho_lm,
                               r1=r1_shell, r2=r2_shell, G_grav=G_grav)

# Real orthonormal Y_20 = sqrt(5/(16 pi)) (3 cos^2(theta) - 1), cos(theta)=z/r,
# matching scipy sph_harm_y(2, 0, theta, phi) used by passess.
Y20_norm = np.sqrt(5.0 / (16.0 * np.pi))


def Y20(x):
    r2 = dot(x, x)
    return Y20_norm * (3.0 * x[2] * x[2] / r2 - 1.0)


def radial_nodes(dr):
    segs = [(rmin, r1_shell, max(1, round((r1_shell - rmin) / dr))),
            (r1_shell, r2_shell, 1),
            (r2_shell, rmax, max(1, round((rmax - r2_shell) / dr)))]
    nodes = [rmin]
    for a, b, n in segs:
        nodes += list(np.linspace(a, b, n + 1))[1:]
    return np.array(nodes)


def shell_mesh(refinement_level, dr):
    nodes = radial_nodes(dr)
    base = CubedSphereMesh(radius=rmin, refinement_level=refinement_level,
                           degree=2)
    heights = list(np.diff(nodes))
    return ExtrudedMesh(base, layers=len(heights), layer_height=heights,
                        extrusion_type="radial")


def solve_shell(mesh, degree):
    V = FunctionSpace(mesh, "CG", degree)
    x = SpatialCoordinate(mesh)
    r = sqrt(dot(x, x))
    psi = TrialFunction(V)
    v = TestFunction(V)

    shell = conditional(And(r >= r1_shell, r <= r2_shell), 1.0, 0.0)
    src = rho_lm * Y20(x) * shell

    a = dot(grad(psi), grad(v)) * dx
    a += ((l_mode + 1) / rmax) * psi * v * ds_t     # exterior DtN
    a += (l_mode / rmin) * psi * v * ds_b           # interior DtN
    L = 4 * pi * G_grav * src * v * dx(degree=2 * l_mode + 8)

    psi_h = Function(V, name="potential")
    solve(a == L, psi_h, solver_parameters={
        "mat_type": "aij", "ksp_type": "preonly",
        "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    return psi_h


def rel_error(psi_h, mesh, ref_degree):
    Vh = FunctionSpace(mesh, "CG", ref_degree)
    coords = Function(VectorFunctionSpace(mesh, "CG", ref_degree)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords.dat.data_ro
    r = np.sqrt(np.sum(gc**2, axis=1))
    # Y20 at nodes (same normalisation as the UFL source)
    y = Y20_norm * (3.0 * gc[:, 2]**2 / r**2 - 1.0)
    psi_ref = Function(Vh)
    psi_ref.dat.data[:] = np.array([analytical.psi_lm(ri).real for ri in r]) * y
    psi_num = Function(Vh).interpolate(psi_h)
    err = float(sqrt(assemble((psi_num - psi_ref)**2 * dx(domain=mesh, degree=2 * ref_degree))))
    nrm = float(sqrt(assemble(psi_ref**2 * dx(domain=mesh, degree=2 * ref_degree))))
    return err / nrm


# Normalisation self-check: passess to_spatial / psi_lm should equal Y20.
_r, _th, _ph = 1.7, 0.9, 0.3
_z = _r * np.cos(_th)
_ratio = (analytical.to_spatial(_r, _th, _ph).real / analytical.psi_lm(_r).real) \
    / (Y20_norm * (3 * np.cos(_th)**2 - 1))
assert abs(_ratio - 1) < 1e-12, f"Y20 normalisation mismatch: {_ratio}"

levels = [(2, 0.1), (3, 0.05), (4, 0.025)]

for deg in (1, 2):
    log(f"\n=== 3D CG{deg}  (l={l_mode}, expect order {deg + 1}) ===")
    log(f"{'ref':>4s} | {'dr':>7s} | {'dofs':>9s} | {'rel L2 error':>13s} | {'order':>6s}")
    log("-" * 54)
    prev_err = prev_h = None
    for ref_level, dr in levels:
        mesh = shell_mesh(ref_level, dr)
        psi_h = solve_shell(mesh, deg)
        err = rel_error(psi_h, mesh, ref_degree=deg + 2)
        if prev_err is None:
            order_str = f"{'--':>6s}"
        else:
            order_str = f"{np.log(prev_err / err) / np.log(prev_h / dr):6.2f}"
        log(f"{ref_level:4d} | {dr:7.4f} | {psi_h.function_space().dim():9d} | "
            f"{err:13.4e} | {order_str}")
        prev_err, prev_h = err, dr

log("\nDone.")
