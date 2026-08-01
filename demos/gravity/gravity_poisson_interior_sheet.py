"""Surface mass sheet on a tagged INTERIOR facet, against passess SheetPolar2D.

`gravity_poisson_sheet.py` validates the sheet sitting *on* a DtN boundary.
This script validates the other case, the one the self-gravitating GIA geometry
needs: the DtN boundaries are stood off from the sources, so the surface the
load sits on is interior to the mesh and the sheet term is an integral over a
tagged interior facet, `-4 pi G avg(sigma v) dS(id)`, rather than over `ds`.

Written the shipped way - `sigma` on an exterior-facet measure - the same
configuration assembles to nothing at all, with a `WARNING Subdomain (7,) is
empty` and a converged solve returning the zero field. The script prints that
outcome next to the correct one, because the ratio between them is the point:
this is not a small error, it is the whole term.

Geometry: an annulus RIN -> ROUT with a conforming circle at RSHEET tagged as
its own physical curve, exterior DtN at ROUT and interior DtN at RIN, no
volume density. Straight facets, so the errors here are the polygon-versus-
circle ones and improve with n_azimuthal.
"""
import os
import sys

import gadopt  # noqa: F401  - before firedrake, see spikes/SPIKE-RESULTS.md
import numpy as np
from firedrake import (
    Constant, Function, FunctionSpace, Mesh, SpatialCoordinate,
    VectorFunctionSpace, assemble, atan2, avg, cos, dS, ds, dx, dot, sqrt)

from gadopt import CylindricalDtN, GravitySolver, log

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))
from passess.polar import SheetPolar2D  # noqa: E402

RIN, RSHEET, ROUT = 1.0, 1.6, 2.4
OUTER_ID, INNER_ID, SHEET_ID = 1, 2, 7
M_TRUNC = 4
DEGREE = 2


def generate(filename, n_azimuthal=128, layers=(4, 6)):
    """Two transfinite rings with the circle between them a physical curve."""
    import gmsh

    radii = [RIN, RSHEET, ROUT]
    curve_tags = [INNER_ID, SHEET_ID, OUTER_ID]

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("interior_sheet_annulus")
    geo = gmsh.model.geo

    centre = geo.addPoint(0, 0, 0)
    quadrants = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    points = [[geo.addPoint(R * cx, R * cy, 0) for cx, cy in quadrants]
              for R in radii]
    arcs = [[geo.addCircleArc(pts[j], centre, pts[(j + 1) % 4])
             for j in range(4)] for pts in points]
    radial = [[geo.addLine(points[i][j], points[i + 1][j]) for j in range(4)]
              for i in range(len(radii) - 1)]

    surfaces = []
    for i in range(len(radii) - 1):
        ring = []
        for j in range(4):
            loop = geo.addCurveLoop([arcs[i][j], radial[i][(j + 1) % 4],
                                     -arcs[i + 1][j], -radial[i][j]])
            ring.append(geo.addPlaneSurface([loop]))
        surfaces.append(ring)

    for ring in arcs:
        for arc in ring:
            geo.mesh.setTransfiniteCurve(arc, n_azimuthal // 4 + 1)
    for i, n_layers in enumerate(layers):
        for line in radial[i]:
            geo.mesh.setTransfiniteCurve(line, n_layers + 1)
    for ring in surfaces:
        for surface in ring:
            geo.mesh.setTransfiniteSurface(surface)

    geo.synchronize()
    for tag, ring in zip(curve_tags, arcs):
        gmsh.model.addPhysicalGroup(1, ring, tag)
    gmsh.model.addPhysicalGroup(2, [s for ring in surfaces for s in ring], 101)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    return filename


def solve_with_sheet(mesh, sigma, key):
    """`key` is 'interior_sigma' (correct here) or 'sigma' (the silent one)."""
    psi = Function(FunctionSpace(mesh, "CG", DEGREE))
    solver = GravitySolver(
        psi, 0.0,
        bcs={OUTER_ID: {"dtn": CylindricalDtN(M=M_TRUNC)},
             INNER_ID: {"dtn": CylindricalDtN(M=M_TRUNC)},
             SHEET_ID: {key: sigma}},
        solver_parameters="direct")
    solver.solve()
    return psi, solver


def relative_error_against_passess(psi, mesh, sheet, m_mode, ref_degree):
    """L2 over the whole annulus, against psi_m(r) cos(m phi) node by node."""
    space = FunctionSpace(mesh, "CG", ref_degree)
    coords = Function(VectorFunctionSpace(mesh, "CG", ref_degree)).interpolate(
        SpatialCoordinate(mesh))
    gc = coords.dat.data_ro
    r = np.sqrt(gc[:, 0] ** 2 + gc[:, 1] ** 2)
    phi = np.arctan2(gc[:, 1], gc[:, 0])
    reference = Function(space)
    reference.dat.data[:] = np.array(
        [sheet.psi_m(ri).real for ri in r]) * np.cos(m_mode * phi)
    numerical = Function(space).interpolate(psi)
    dxq = dx(domain=mesh, degree=2 * ref_degree)
    error = float(sqrt(assemble((numerical - reference) ** 2 * dxq)))
    norm = float(sqrt(assemble(reference**2 * dxq)))
    return error / norm


mesh_file = "gravity_interior_sheet_annulus.msh"
if not os.path.exists(mesh_file):
    generate(mesh_file)
mesh = Mesh(mesh_file)

log(f"annulus [{RIN}, {ROUT}], sheet at {RSHEET}, CG{DEGREE}, "
    f"modal DtN M = {M_TRUNC}")
length = assemble(avg(Constant(1.0)) * dS(SHEET_ID, domain=mesh))
log(f"V10-assembly: dS({SHEET_ID}) = {length:.8f} against 2 pi a = "
    f"{2 * np.pi * RSHEET:.8f}, relative "
    f"{length / (2 * np.pi * RSHEET) - 1:+.2e}")
log(f"              ds({SHEET_ID}) = "
    f"{assemble(Constant(1.0) * ds(SHEET_ID, domain=mesh)):.8f}  "
    f"[the shipped spelling; this is the failure]")

log("\nV10-end-to-end, against passess SheetPolar2D")
log(f"{'m':>3s} | {'rel L2 (interior_sigma)':>23s} | {'c_m(out)/exact':>15s} | "
    f"{'c_m(in)/exact':>14s}")
log("-" * 64)

X = SpatialCoordinate(mesh)
r = sqrt(dot(X, X))
phi = atan2(X[1], X[0])

for m_mode in (1, 2, 3):
    sheet = SheetPolar2D(m=m_mode, sigma_m=1.0, a=RSHEET, gamma=1.0)
    psi, solver = solve_with_sheet(mesh, cos(m_mode * phi), "interior_sigma")
    err = relative_error_against_passess(psi, mesh, sheet, m_mode,
                                         ref_degree=DEGREE + 3)
    coefficients = solver.coefficients()
    outer = coefficients[OUTER_ID][f"cos{m_mode}"] / sheet.psi_m(ROUT).real
    inner = coefficients[INNER_ID][f"cos{m_mode}"] / sheet.psi_m(RIN).real
    log(f"{m_mode:3d} | {err:23.4e} | {outer:15.8f} | {inner:14.8f}")

log("\nThe same configuration written the shipped way, m = 2:")
try:
    psi_wrong, _ = solve_with_sheet(mesh, cos(2 * phi), "sigma")
    norm = float(sqrt(assemble(psi_wrong**2 * dx(domain=mesh))))
    log(f"  ||psi|| = {norm:.6e}  (the sheet contributed nothing)")
except ValueError as exc:
    log(f"  refused at construction: {exc}")

log("\nAnd the closed form the unit tests write out inline, for the record:")
for m_mode in (1, 2):
    sheet = SheetPolar2D(m=m_mode, sigma_m=1.0, a=RSHEET, gamma=1.0)
    inline = (2 * np.pi * RSHEET / m_mode) * (RIN / RSHEET) ** m_mode
    log(f"  m = {m_mode}: inline {inline:.12f} vs passess "
        f"{sheet.psi_m(RIN).real:.12f}")
