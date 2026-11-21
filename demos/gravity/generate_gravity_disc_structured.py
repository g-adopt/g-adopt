"""Generate an all-quad annular mesh with structured mantle for gravity Poisson testing.

The mesh is a pure quadrilateral annulus from r_core (slightly below rmin)
to R_grav (far exterior). No origin point is needed because neither boundary
carries a Dirichlet condition on the inner side — only the outer boundary
at R_grav has psi=0.

The mantle (rmin to rmax) gets uniform structured quads. The density shell
boundaries [r1_shell, r2_shell] are explicit geometric circles, so every
quad cell is entirely inside or outside the shell. The interior buffer
(r_core to rmin) and exterior buffer (rmax to R_grav) coarsen radially
via geometric progression.

Geometry decomposition:
  Four quadrants (split at x and y axes), each containing one 4-sided
  transfinite quad patch per radial band:
  - interior buffer: r_core to rmin (coarsening inward)
  - mantle below shell: rmin to r1_shell
  - density shell: r1_shell to r2_shell
  - mantle above shell: r2_shell to rmax
  - exterior buffer: rmax to R_grav (coarsening outward)
  Total: 4 quadrants x 5 bands = 20 transfinite quad surfaces.
"""
import gmsh


def generate_mesh(rmin=1.22, rmax=2.22, r1_shell=None, r2_shell=None,
                  R_grav_factor=10, r_core_factor=0.5,
                  N_phi_quad=32,
                  N_r_interior=6, interior_progression=1.3,
                  N_r_below=10, N_r_shell=4, N_r_above=10,
                  N_r_buffer=6, buffer_width=0.5,
                  N_r_exterior=8, exterior_progression=1.3,
                  filename="gravity_disc.msh"):
    """Generate an all-quad annular mesh with structured mantle.

    Args:
        rmin, rmax: inner and outer mantle radii.
        r1_shell, r2_shell: density shell inner and outer radii.
        R_grav_factor: R_grav = R_grav_factor * rmax.
        r_core_factor: r_core = r_core_factor * rmin (inner buffer radius).
        N_phi_quad: angular intervals per quadrant (total = 4x).
        N_r_interior: radial intervals from r_core to rmin.
        interior_progression: geometric coarsening toward r_core.
        N_r_below: radial intervals from rmin to r1_shell.
        N_r_shell: radial intervals across the density shell.
        N_r_above: radial intervals from r2_shell to rmax.
        N_r_buffer: radial intervals in the fine buffer zone beyond rmax.
        buffer_width: width of the fine buffer zone beyond rmax.
        N_r_exterior: radial intervals from r_buffer to R_grav.
        exterior_progression: geometric coarsening toward R_grav.
        filename: output mesh file.
    """
    R_grav = R_grav_factor * rmax
    r_core = r_core_factor * rmin
    r_buffer = rmax + buffer_width

    gmsh.initialize()
    gmsh.model.add("gravity_disc_structured")
    geo = gmsh.model.geo

    # Center point (only used as arc center, not a mesh vertex)
    p0 = geo.addPoint(0, 0, 0)

    # Concentric radii from inside out
    radii = [r_core, rmin, r1_shell, r2_shell, rmax, r_buffer, R_grav]

    # Points on each circle at four cardinal directions
    circle_pts = {}
    for rad in radii:
        pts = []
        for x, y in [(rad, 0), (0, rad), (-rad, 0), (0, -rad)]:
            pts.append(geo.addPoint(x, y, 0))
        circle_pts[rad] = pts

    # Quarter-circle arcs: arcs[rad][q] for quadrant q
    arcs = {}
    for rad in radii:
        pts = circle_pts[rad]
        qarcs = []
        for q in range(4):
            qarcs.append(geo.addCircleArc(pts[q], p0, pts[(q + 1) % 4]))
        arcs[rad] = qarcs

    # Radial lines between consecutive circles
    radii_pairs = list(zip(radii[:-1], radii[1:]))
    radial_lines = {}
    for r_in, r_out in radii_pairs:
        lines = []
        for c in range(4):
            lines.append(geo.addLine(circle_pts[r_in][c],
                                     circle_pts[r_out][c]))
        radial_lines[(r_in, r_out)] = lines

    # --- Transfinite constraints ---

    # Angular: all arcs get the same node count per quadrant
    N_phi_nodes = N_phi_quad + 1
    for rad in radii:
        for q in range(4):
            geo.mesh.setTransfiniteCurve(arcs[rad][q], N_phi_nodes)

    # Radial: per-band node count and optional progression.
    # Negative progression means grading toward the start of the line
    # (i.e., toward the inner radius).
    band_spec = {
        (r_core, rmin): (N_r_interior, -interior_progression),
        (rmin, r1_shell): (N_r_below, 1.0),
        (r1_shell, r2_shell): (N_r_shell, 1.0),
        (r2_shell, rmax): (N_r_above, 1.0),
        (rmax, r_buffer): (N_r_buffer, 1.0),
        (r_buffer, R_grav): (N_r_exterior, exterior_progression),
    }
    for (r_in, r_out), (n_intervals, prog) in band_spec.items():
        for c in range(4):
            geo.mesh.setTransfiniteCurve(
                radial_lines[(r_in, r_out)][c], n_intervals + 1,
                "Progression", prog)

    # --- Build the 20 quad surfaces (4 quadrants x 5 bands) ---

    band_surfaces = {bp: [] for bp in radii_pairs}

    for q in range(4):
        c_start = q
        c_end = (q + 1) % 4

        for r_in, r_out in radii_pairs:
            arc_in = arcs[r_in][q]
            line_end = radial_lines[(r_in, r_out)][c_end]
            arc_out = arcs[r_out][q]
            line_start = radial_lines[(r_in, r_out)][c_start]

            loop = geo.addCurveLoop([arc_in, line_end, -arc_out, -line_start])
            surf = geo.addPlaneSurface([loop])
            corners = [
                circle_pts[r_in][c_start],
                circle_pts[r_in][c_end],
                circle_pts[r_out][c_end],
                circle_pts[r_out][c_start],
            ]
            geo.mesh.setTransfiniteSurface(surf, "Left", corners)
            geo.mesh.setRecombine(2, surf)
            band_surfaces[(r_in, r_out)].append(surf)

    geo.synchronize()

    # --- Physical groups ---
    gmsh.model.addPhysicalGroup(1, arcs[R_grav], 1, name="outer_boundary")
    gmsh.model.addPhysicalGroup(
        2, band_surfaces[(r_core, rmin)], 101, name="interior")
    gmsh.model.addPhysicalGroup(
        2, band_surfaces[(rmin, r1_shell)], 102, name="mantle_below_shell")
    gmsh.model.addPhysicalGroup(
        2, band_surfaces[(r1_shell, r2_shell)], 103, name="density_shell")
    gmsh.model.addPhysicalGroup(
        2, band_surfaces[(r2_shell, rmax)], 104, name="mantle_above_shell")
    gmsh.model.addPhysicalGroup(
        2, band_surfaces[(rmax, r_buffer)] + band_surfaces[(r_buffer, R_grav)],
        105, name="exterior")

    gmsh.option.setNumber("Mesh.Smoothing", 10)

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
                  N_phi_quad=32,
                  N_r_interior=6, interior_progression=1.3,
                  N_r_below=10, N_r_shell=4, N_r_above=10,
                  N_r_exterior=8, exterior_progression=1.3)
