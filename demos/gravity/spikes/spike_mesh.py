"""Throwaway three-region annulus for the Phase 0 spikes.

Adapted from `generate_gravity_annulus.py` in the `sghelichkhani/poisson`
worktree (copied, not modified in place).  The difference that matters here is
that the *interior* interface circles are given physical curve groups of their
own, because spike S1 is exactly the question of whether such a tagged interior
curve survives Firedrake's gmsh reader.

Radial zones, tagged following the convention of ROADMAP-GIA-SELFGRAV §1.5:

    0.5 Rc ---- inner (102) ---- Rc ---- mantle (101) ---- Re ---- buffer (103) ---- 2 Re

Curve physical IDs:

    2 = Re      (interior facet of the parent, boundary facet of the submesh)
    3 = Rc      (interior facet of the parent, boundary facet of the submesh)
    4 = 2 Re    (exterior boundary, would carry the exterior DtN)
    5 = 0.5 Rc  (interior boundary of the parent, would carry the interior DtN)

Cell physical IDs: 101 mantle, 102 inner, 103 buffer.
"""
import numpy as np
import gmsh

# Non-dimensionalised by D = Re - Rc = 2891 km, as in the plan.
RC = 1.2037
RE = 2.2037
R_INNER = 0.5 * RC
R_OUTER = 2.0 * RE

CELL_MANTLE, CELL_INNER, CELL_BUFFER = 101, 102, 103
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5


def generate(filename="spike_annulus.msh", n_azimuthal=64, n_inner=4,
             n_mantle=8, n_buffer=6, quads=False, verbose=False):
    """Writes the three-region annulus and returns the filename."""
    assert n_azimuthal % 4 == 0, "n_azimuthal must be divisible by 4"

    radii = [R_INNER, RC, RE, R_OUTER]
    layers = [n_inner, n_mantle, n_buffer]
    cell_tags = [CELL_INNER, CELL_MANTLE, CELL_BUFFER]
    curve_tags = [CURVE_INNER, CURVE_RC, CURVE_RE, CURVE_OUTER]

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("spike_annulus")
    geo = gmsh.model.geo

    centre = geo.addPoint(0, 0, 0)
    angles = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    points = [[geo.addPoint(R * cx, R * cy, 0) for cx, cy in angles]
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

    n_quarter = n_azimuthal // 4
    for ring in arcs:
        for a in ring:
            geo.mesh.setTransfiniteCurve(a, n_quarter + 1)
    for i, n_layers in enumerate(layers):
        for line in radial[i]:
            geo.mesh.setTransfiniteCurve(line, n_layers + 1)
    for ring in surfaces:
        for s in ring:
            geo.mesh.setTransfiniteSurface(s)
            if quads:
                geo.mesh.setRecombine(2, s)

    geo.synchronize()

    # Every concentric circle gets a physical group, including the two that
    # are interior to the parent mesh.  This is the point of the spike.
    for tag, ring in zip(curve_tags, arcs):
        gmsh.model.addPhysicalGroup(1, ring, tag)
    for tag, ring in zip(cell_tags, surfaces):
        gmsh.model.addPhysicalGroup(2, ring, tag)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    return filename


def analytic():
    """Reference areas and circumferences for the validation prints."""
    return {
        "area_inner": np.pi * (RC**2 - R_INNER**2),
        "area_mantle": np.pi * (RE**2 - RC**2),
        "area_buffer": np.pi * (R_OUTER**2 - RE**2),
        "len_Re": 2 * np.pi * RE,
        "len_Rc": 2 * np.pi * RC,
        "len_outer": 2 * np.pi * R_OUTER,
        "len_inner": 2 * np.pi * R_INNER,
    }


if __name__ == "__main__":
    print(generate(verbose=True))
