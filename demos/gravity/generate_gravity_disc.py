"""Generate a full disc mesh for gravity Poisson equation testing.

Creates a disc from r=0 to R_grav with five concentric regions, where
the density shell [r1_shell, r2_shell] has its boundaries as explicit
mesh-conforming circles. This ensures DG0 density fields have no
elements straddling the shell boundary.

Regions:
  - Inner disc:           0        to rmin      (interior, very coarse)
  - Sub-shell mantle:     rmin     to r1_shell  (fine, lc_mantle)
  - Density shell:        r1_shell to r2_shell  (fine, lc_mantle)
  - Supra-shell mantle:   r2_shell to rmax      (fine, lc_mantle)
  - Exterior annulus:     rmax     to R_grav     (very coarse)

Resolution strategy: uniform lc_mantle throughout the mantle (rmin to rmax),
with rapid coarsening both inward (toward r=0) and outward (toward R_grav).
"""
import gmsh


def generate_mesh(rmin=1.22, rmax=2.22, r1_shell=None, r2_shell=None,
                  R_grav_factor=10, lc_mantle=0.04,
                  lc_exterior=50.0, filename="gravity_disc.msh"):
    R_grav = R_grav_factor * rmax

    gmsh.initialize()
    gmsh.model.add("gravity_disc")

    p0 = gmsh.model.geo.addPoint(0, 0, 0, lc_exterior)

    def add_circle(radius, lc):
        pts = []
        for x, y in [(radius, 0), (0, radius), (-radius, 0), (0, -radius)]:
            pts.append(gmsh.model.geo.addPoint(x, y, 0, lc))
        arcs = []
        for i in range(4):
            arcs.append(gmsh.model.geo.addCircleArc(pts[i], p0, pts[(i + 1) % 4]))
        loop = gmsh.model.geo.addCurveLoop(arcs)
        return pts, arcs, loop

    # Five concentric circles (inside out)
    # rmin, rmax boundaries get lc_mantle; shell boundaries too;
    # R_grav gets lc_exterior
    circles = []
    for rad, lc in [
        (rmin, lc_mantle),
        (r1_shell, lc_mantle),
        (r2_shell, lc_mantle),
        (rmax, lc_mantle),
        (R_grav, lc_exterior),
    ]:
        circles.append(add_circle(rad, lc))

    # Surfaces
    surf_inner = gmsh.model.geo.addPlaneSurface([circles[0][2]])
    surf_mantle_below = gmsh.model.geo.addPlaneSurface([circles[0][2], circles[1][2]])
    surf_shell = gmsh.model.geo.addPlaneSurface([circles[1][2], circles[2][2]])
    surf_mantle_above = gmsh.model.geo.addPlaneSurface([circles[2][2], circles[3][2]])
    surf_exterior = gmsh.model.geo.addPlaneSurface([circles[3][2], circles[4][2]])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, circles[4][1], 1, name="outer_boundary")
    gmsh.model.addPhysicalGroup(2, [surf_inner], 101, name="interior")
    gmsh.model.addPhysicalGroup(2, [surf_mantle_below], 102, name="mantle_below_shell")
    gmsh.model.addPhysicalGroup(2, [surf_shell], 103, name="density_shell")
    gmsh.model.addPhysicalGroup(2, [surf_mantle_above], 104, name="mantle_above_shell")
    gmsh.model.addPhysicalGroup(2, [surf_exterior], 105, name="exterior")

    # Size field strategy:
    # - Uniform lc_mantle throughout the mantle (rmin to rmax).
    # - Buffer zone outside rmax: keeps lc_mantle for ~0.5 beyond rmax,
    #   then ramps to lc_exterior. This ensures CG1 DOFs at the mantle
    #   surface can resolve cos(m*phi) without pollution from coarse exterior.
    # - Buffer zone inside rmin: same idea toward the centre.
    # We use SEPARATE distance fields for rmin and rmax curves
    # so they don't interfere.

    # Outward transition: distance from rmax curves
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", circles[3][1])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_mantle)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_exterior)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)  # fine out to rmax+0.5
    gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0)  # ramp over next 2.5

    # Inward transition: distance from rmin curves
    gmsh.model.mesh.field.add("Distance", 3)
    gmsh.model.mesh.field.setNumbers(3, "CurvesList", circles[0][1])
    gmsh.model.mesh.field.setNumber(3, "Sampling", 200)

    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", lc_mantle)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", lc_exterior)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 0.5)  # fine to rmin-0.5
    gmsh.model.mesh.field.setNumber(4, "DistMax", 1.5)  # ramp over next 1.0

    # Take the minimum (finest) of both fields
    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    # Let boundary mesh sizes extend into the 2D mesh so the fine
    # 1D mesh on mantle circles creates a buffer of fine elements
    # just inside/outside the mantle boundaries.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()


if __name__ == "__main__":
    D_km = 2891.0
    rmin, rmax = 1.22, 2.22
    depth_km = 500.0
    thickness_km = 50.0
    r_center = rmax - depth_km / D_km
    r1 = r_center - thickness_km / (2 * D_km)
    r2 = r_center + thickness_km / (2 * D_km)

    generate_mesh(rmin=rmin, rmax=rmax, r1_shell=r1, r2_shell=r2,
                  lc_mantle=0.04, lc_exterior=50.0)
