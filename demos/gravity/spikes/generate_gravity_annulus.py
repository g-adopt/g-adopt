"""Structured annulus mesh for the interior+exterior DtN gravity tests.

Builds an 'extruded-style' annulus (rmin to R_grav) in gmsh: transfinite
quadrilateral layers between explicit concentric radii, i.e. the layered
structure of a radially extruded mesh, but written as an ordinary 2D mesh
so that Firedrake's Submesh works (Submesh raises NotImplementedError on
actual ExtrudedMesh objects, firedrake/mesh.py).

Radial zones (interfaces are exact mesh circles, so DG0 density fields
conform to the shell):

    rmin -> r1_shell     uniform fine layers (~dr_mantle)
    r1_shell -> r2_shell n_shell_layers layers (the density shell)
    r2_shell -> rmax     uniform fine layers (~dr_mantle)
    rmax -> R_grav       geometrically coarsening layers (growth ratio)

Boundary physical IDs: 1 = outer (R_grav), 2 = inner (rmin) --
matching OUTER_ID in the disc scripts.
"""
import numpy as np
import gmsh


def generate_annulus(rmin=1.22, rmax=2.22, r1_shell=None, r2_shell=None,
                     R_grav_factor=2, dr_mantle=0.02, n_shell_layers=1,
                     n_azimuthal=512, growth_exterior=1.3,
                     quads=True, filename="gravity_annulus.msh"):
    R_grav = R_grav_factor * rmax
    assert n_azimuthal % 4 == 0, "n_azimuthal must be divisible by 4"
    exterior = R_grav_factor > 1.0  # config D (DtN at rmax) has no buffer

    # Radial interface radii and per-zone (layer count, progression coef)
    zones = []
    for r_in, r_out in [(rmin, r1_shell), (r2_shell, rmax)]:
        zones.append((r_in, r_out, max(1, round((r_out - r_in) / dr_mantle)), 1.0))
    zones.insert(1, (r1_shell, r2_shell, n_shell_layers, 1.0))
    if exterior:
        # Exterior: first layer ~dr_mantle, geometric growth to R_grav
        g = growth_exterior
        width = R_grav - rmax
        n_ext = max(1, int(np.ceil(np.log1p(width * (g - 1) / dr_mantle) / np.log(g))))
        zones.append((rmax, R_grav, n_ext, g))

    radii = [zones[0][0]] + [z[1] for z in zones]

    gmsh.initialize()
    gmsh.model.add("gravity_annulus")
    geo = gmsh.model.geo

    centre = geo.addPoint(0, 0, 0)
    angles = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # points[i][j]: radius i, quadrant corner j
    points = [[geo.addPoint(R * cx, R * cy, 0) for cx, cy in angles]
              for R in radii]

    # arcs[i][j]: at radius i from corner j to j+1
    arcs = [[geo.addCircleArc(pts[j], centre, pts[(j + 1) % 4])
             for j in range(4)] for pts in points]
    # radial[i][j]: from radius i to i+1 at corner j
    radial = [[geo.addLine(points[i][j], points[i + 1][j]) for j in range(4)]
              for i in range(len(radii) - 1)]

    surfaces = []
    for i in range(len(radii) - 1):
        for j in range(4):
            loop = geo.addCurveLoop([arcs[i][j], radial[i][(j + 1) % 4],
                                     -arcs[i + 1][j], -radial[i][j]])
            surfaces.append(geo.addPlaneSurface([loop]))

    # Transfinite structure
    n_quarter = n_azimuthal // 4
    for ring in arcs:
        for a in ring:
            geo.mesh.setTransfiniteCurve(a, n_quarter + 1)
    for i, (_, _, n_layers, coef) in enumerate(zones):
        for line in radial[i]:
            geo.mesh.setTransfiniteCurve(line, n_layers + 1, "Progression", coef)
    for s in surfaces:
        geo.mesh.setTransfiniteSurface(s)
        if quads:
            geo.mesh.setRecombine(2, s)

    geo.synchronize()

    gmsh.model.addPhysicalGroup(1, arcs[-1], 1, name="outer_boundary")
    gmsh.model.addPhysicalGroup(1, arcs[0], 2, name="inner_boundary")
    n_zone_names = ["mantle_below_shell", "density_shell", "mantle_above_shell"]
    if exterior:
        n_zone_names.append("exterior")
    for i, name in enumerate(n_zone_names):
        gmsh.model.addPhysicalGroup(2, surfaces[4 * i:4 * i + 4], 102 + i,
                                    name=name)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    return filename


if __name__ == "__main__":
    D_km = 2891.0
    rmin, rmax = 1.22, 2.22
    r_center = rmax - 500.0 / D_km
    r1 = r_center - 50.0 / (2 * D_km)
    r2 = r_center + 50.0 / (2 * D_km)
    generate_annulus(rmin=rmin, rmax=rmax, r1_shell=r1, r2_shell=r2)
