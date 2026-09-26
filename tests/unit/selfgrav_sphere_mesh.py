"""The four-region 3-D sphere of the self-gravitating Spada benchmark.

This module is a helper of `test_sea_level_masks_slope.py`, which builds a
very coarse sphere with it. It is the mesh generator of the former demo
`demos/glacial_isostatic_adjustment/3d_spada_selfgrav`, kept unchanged so
that the test keeps the mesh it was written against.

The gravitational potential lives on the whole mesh (the "parent"). The
mechanics lives on a `Submesh` of the mantle cells. The two outer regions are
buffers that carry the potential away from its sources, so that the exterior
gravity condition (a Dirichlet-to-Neumann map) sits at a distance from the
load and from the core-mantle boundary:

    0.5 Rc ---- inner (102) ---- Rc ---- mantle (101) ---- Re ---- buffer (103) ---- 2 Re
       |                          |                        |                          |
   surface 5                  surface 3                surface 2                  surface 4
  interior DtN            interface, INTERIOR      interface, INTERIOR        exterior DtN
                          facet of the parent      facet of the parent

Lengths are non-dimensionalised by the mantle thickness D = Re - Rc =
2891 km, so Rc = 1.203736, Re = 2.203736, 0.5 Rc = 0.601868 and
2 Re = 4.407471. The three density interfaces of the M3-L70-V01 Earth model
(6301, 5951 and 5701 km radius) are also surfaces of the mesh, with tags 6, 7
and 8. The core ball r < 0.5 Rc is not meshed; its only effect on the domain
is a degree-0 exterior field.

## Construction

Every sphere of the radius ladder is an OpenCASCADE solid, and
`occ.fragment` cuts them into nested shells that share their bounding
surfaces exactly. The core ball inside 0.5 Rc is then removed without its
boundary, so the sphere at 0.5 Rc stays as the inner boundary of the domain.

Two consequences follow. Every density interface is a geometric entity, so
the mesh conforms to it and a piecewise-constant (DG0) density is exact on
every cell. And each entity identifies itself by its measure: a surface
radius is sqrt(A / 4 pi), and a shell is known by the radii of its two
bounding surfaces. Nothing depends on the order in which gmsh returns
entities.

## Radial and lateral resolution are independent

The 70 km lithosphere must carry mesh-conforming radial layers whatever the
lateral spacing is. An isotropic size field cannot give that cheaply: 35 km
is 0.0121 in these units, and isotropic cells of that size through the
lithosphere cost millions of cells before the rest of the mantle is meshed.

The generator instead adds "subdivision spheres" inside the lithosphere. They
carry no physical group and no density jump; they only force gmsh to place
nodes on them. Each sub-shell is then filled with one layer of flat
tetrahedra at the lateral size. `litho_layers` is the number of sub-shells.
The cells are anisotropic by design, with an aspect ratio of about
h_lateral / h_radial.

The design aspect ratio matters for the solver: the algebraic multigrid on
the displacement block converges much more slowly on the highly anisotropic
lithosphere cells. The benchmark mesh `b2_coarse_ar7.msh` uses the `coarse`
lateral spacing (500 km) with ONE lithosphere layer (70 km), which gives a
design aspect ratio of about 7:

    python selfgrav_sphere_mesh.py --configuration coarse \\
        --litho-layers 1 --min-cells 32 --output b2_coarse_ar7.msh

gmsh tetrahedralisation is not bit-reproducible across gmsh versions, so a
regenerated mesh has cell counts close to, but not equal to, those of the
mesh of the validated run.

## The size field

The target cell size grows linearly away from the mantle in both directions,

    lc(r) = h (1 + grade * max(0, r - Re) + grade * max(0, Rc - r)),

so the buffers are coarse where the potential is smooth. A second field caps
the size at 2 pi r / N, so that every sphere carries at least N cells around
a great circle. Without that cap the small inner DtN sphere at 0.5 Rc, where
the grading makes cells coarse, becomes a polyhedron of about 70 triangles
with an area error of order 10 percent. The cap matters because that sphere
carries a DtN map, which must resolve the truncation degree L: N cells
around a great circle give N / L cells per wavelength of degree L.

## Curving

gmsh writes straight-sided tetrahedra. `curve_mesh` maps the coordinates to
a P2 (quadratic) field whose edge midpoints lie on the sphere through their
end vertices, so every mesh sphere becomes a piecewise-quadratic surface.
This reduces the geometric error of surface integrals on Re, Rc and the DtN
spheres from O(h^2) to O(h^3).
"""
import numpy as np

#: The length scale, the mantle thickness Re - Rc, in km.
D_KM = 2891.0
#: The core-mantle boundary radius, 3480 km / D.
RC = 1.203736
#: The Earth radius, 6371 km / D.
RE = 2.203736
#: The inner DtN sphere, at half the core radius.
R_INNER = 0.5 * RC
#: The outer DtN sphere, at twice the Earth radius.
R_OUTER = 2.0 * RE

#: Density interfaces of M3-L70-V01, strictly inside the mantle, ascending:
#: 5701 km, 5951 km and 6301 km divided by D.
DENSITY_INTERFACES = (1.971982, 2.058457, 2.179523)

#: Extra spheres in the outer buffer, so the size field has somewhere to grade.
BUFFER_SPHERES = (2.75, 3.30)

#: The base of the 70 km lithosphere.
R_LITHO = DENSITY_INTERFACES[-1]

# Cell groups and surface groups.
CELL_MANTLE, CELL_INNER, CELL_BUFFER = 101, 102, 103
SURF_RE, SURF_RC, SURF_OUTER, SURF_INNER = 2, 3, 4, 5

#: The three density interfaces as interior facet groups. The physics does not
#: need them: the divergence-form Poisson source produces the interface mass
#: sheets automatically once the density carries the jumps and the mesh
#: conforms. They exist so that a surface average of |g_0| can be taken on
#: them, which is a better check of the reference gravity than a point value of
#: a gradient that is discontinuous across the interface.
SURF_D1, SURF_D2, SURF_D3 = 6, 7, 8
DENSITY_INTERFACE_TAGS = {DENSITY_INTERFACES[0]: SURF_D1,
                          DENSITY_INTERFACES[1]: SURF_D2,
                          DENSITY_INTERFACES[2]: SURF_D3}

#: Named lateral spacings, in km.
CONFIGURATIONS = {"coarse": 500.0, "medium": 250.0, "fine": 120.0,
                  "production": 78.0}

#: `(litho_layers, min_cells_per_great_circle)` for each configuration. Both
#: scale with the lateral spacing, so that the lithosphere aspect ratio stays
#: at about 14 down the ladder and a refinement pair differs in resolution
#: only. A lower-anisotropy mesh is requested explicitly with a smaller
#: `litho_layers`, as for `b2_coarse_ar7.msh`.
RESOLUTION_LADDER = {"coarse": (2, 32), "medium": (4, 64),
                     "fine": (8, 128), "production": (12, 192)}


def lateral_spacing(configuration):
    """Non-dimensional lateral cell size for a named configuration."""
    return CONFIGURATIONS[configuration] / D_KM


def resolution_defaults(configuration):
    """`(litho_layers, min_cells_per_great_circle)` for a configuration."""
    return RESOLUTION_LADDER[configuration]


def design_aspect_ratio(h, litho_layers):
    """The lithosphere aspect ratio h_lateral / h_radial that a mesh is built for.

    `litho_layers` sub-shells are placed across the lithosphere RE - R_LITHO,
    so the radial spacing there is known before the mesh exists.

    Args:
      h: non-dimensional lateral spacing.
      litho_layers: number of lithosphere sub-shells.

    Returns:
      The design aspect ratio, dimensionless.
    """
    return h / ((RE - R_LITHO) / litho_layers)


def sphere_radii(litho_layers=2):
    """Every sphere in the ladder, ascending, with the subdivision spheres."""
    radii = [R_INNER, RC, *DENSITY_INTERFACES, RE, *BUFFER_SPHERES, R_OUTER]
    for k in range(1, litho_layers):
        radii.append(R_LITHO + k * (RE - R_LITHO) / litho_layers)
    return sorted(radii)


def shells(litho_layers=2):
    """The shells as (r_in, r_out, cell_tag), ascending."""
    radii = sphere_radii(litho_layers)
    out = []
    for r_in, r_out in zip(radii[:-1], radii[1:]):
        mid = 0.5 * (r_in + r_out)
        tag = (CELL_INNER if mid < RC
               else CELL_MANTLE if mid < RE else CELL_BUFFER)
        out.append((r_in, r_out, tag))
    return out


def analytic(litho_layers=2):
    """Exact volumes of the three regions and areas of the four tagged spheres."""
    ball = lambda r: 4 / 3 * np.pi * r**3  # noqa: E731
    vol = {CELL_MANTLE: 0.0, CELL_INNER: 0.0, CELL_BUFFER: 0.0}
    for r_in, r_out, tag in shells(litho_layers):
        vol[tag] += ball(r_out) - ball(r_in)
    return {
        "vol_mantle": vol[CELL_MANTLE],
        "vol_inner": vol[CELL_INNER],
        "vol_buffer": vol[CELL_BUFFER],
        "area_inner": 4 * np.pi * R_INNER**2,
        "area_Rc": 4 * np.pi * RC**2,
        "area_Re": 4 * np.pi * RE**2,
        "area_outer": 4 * np.pi * R_OUTER**2,
    }


def radius_of_surface(tag):
    """A sphere's radius from its own area. Exact for an OCC sphere."""
    # gmsh is imported here, not at module level, so that the benchmark
    # driver can import the constants and `curve_mesh` of this module in a
    # Firedrake installation without the gmsh Python module.
    import gmsh

    return np.sqrt(gmsh.model.occ.getMass(2, tag) / (4 * np.pi))


def generate(filename="selfgrav_sphere.msh", configuration="coarse",
             h=None, grade=2.0, litho_layers=None,
             min_cells_per_great_circle=None,
             verbose=False, quality=False, extra_size_fields=None,
             mesh_options=None):
    """Write the four-region sphere to a gmsh file.

    Args:
      filename: output `.msh` path.
      configuration: a key of `CONFIGURATIONS`. It sets the lateral spacing
        and the defaults of `litho_layers` and `min_cells_per_great_circle`.
      h: explicit non-dimensional lateral spacing, overriding the
        configuration's.
      grade: linear growth rate of the cell size away from the mantle.
      litho_layers: number of lithosphere sub-shells.
      min_cells_per_great_circle: lower bound on the number of cells around
        any sphere of the ladder; 0 or None after the ladder default disables
        the cap.
      verbose: print gmsh's own log.
      quality: compute per-shell tetrahedron quality.
      extra_size_fields: an optional callable `extra_size_fields(gmsh)`. It
        is called after the graded and capped size fields exist and before
        they are merged. It adds its own gmsh fields and returns a list of
        their tags. The tags join the `Min` list after the graded and capped
        fields, so the cell size is the smallest of all of them. A refined
        mesh uses this hook for local refinement without a copy of the
        geometry and the physical groups. With `None` (the default) the
        size field and therefore the mesh are the same as without the hook.
      mesh_options: an optional dictionary of numeric gmsh options, for
        example `{"Mesh.Algorithm3D": 10}`. The options are set after the
        size-field options of this function and before the mesh is
        generated. With `None` (the default) no option is set, and gmsh
        uses its own defaults, which are the options of the Spada meshes.

    Returns:
      `(filename, shells, stats)`: the path, the `(r_in, r_out, tag)` list of
      shells, and a dictionary with the cell count of every shell and the
      parameters used.
    """
    # gmsh is imported here, not at module level, so that the benchmark
    # driver can import the constants and `curve_mesh` of this module in a
    # Firedrake installation without the gmsh Python module.
    import gmsh

    h = lateral_spacing(configuration) if h is None else h
    ladder_layers, ladder_cells = resolution_defaults(configuration)
    if litho_layers is None:
        litho_layers = ladder_layers
    if min_cells_per_great_circle is None:
        min_cells_per_great_circle = ladder_cells
    radii = sphere_radii(litho_layers)

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("selfgrav_sphere")
    occ = gmsh.model.occ

    # One solid ball per radius; the fragment cuts them into nested shells
    # that share their bounding surfaces.
    balls = [occ.addSphere(0, 0, 0, R) for R in radii]
    occ.fragment([(3, balls[0])], [(3, b) for b in balls[1:]])
    occ.synchronize()

    # Remove the unmeshed core ball, keeping its bounding surface: that sphere
    # is the inner boundary of the domain and carries the interior DtN.
    core = [t for _, t in gmsh.model.getEntities(3)
            if abs(occ.getMass(3, t) - 4 / 3 * np.pi * radii[0]**3) < 1e-8]
    assert len(core) == 1, f"expected one core ball, found {core}"
    occ.remove([(3, core[0])], recursive=False)
    occ.synchronize()

    # Identify every entity by its own measure, never by gmsh's ordering. A
    # radius recovered from an area is never bit-identical to the ladder value
    # that produced it, so it is snapped back to the ladder before use as a key.
    def snap(R):
        match = [r for r in radii if abs(r - R) < 1e-6 * r]
        assert len(match) == 1, f"radius {R} is not in the ladder {radii}"
        return match[0]

    surface_of_radius = {}
    for _, t in gmsh.model.getEntities(2):
        surface_of_radius.setdefault(snap(radius_of_surface(t)), []).append(t)

    volume_of_shell = {}
    for _, t in gmsh.model.getEntities(3):
        bounding = sorted(snap(radius_of_surface(abs(s))) for _, s
                          in gmsh.model.getBoundary([(3, t)], oriented=True))
        volume_of_shell[(bounding[0], bounding[-1])] = t

    # Surface groups: the two DtN spheres, the two mechanics boundaries, and
    # the three density interfaces. The buffer and subdivision spheres get no
    # group: conformity only needs mesh nodes on them, which the fragment
    # supplies whether or not the surface is written to file.
    tagged = [(R_INNER, SURF_INNER, "inner_dtn"), (RC, SURF_RC, "Rc"),
              (RE, SURF_RE, "Re"), (R_OUTER, SURF_OUTER, "outer_dtn")]
    tagged += [(R, DENSITY_INTERFACE_TAGS[R], f"rho_interface_{i + 1}")
               for i, R in enumerate(DENSITY_INTERFACES)]
    for R, tag, name in tagged:
        gmsh.model.addPhysicalGroup(2, surface_of_radius[R], tag, name=name)

    # Cell groups: every mantle sub-shell joins 101.
    layout = shells(litho_layers)
    for tag, name in [(CELL_MANTLE, "mantle"), (CELL_INNER, "inner"),
                      (CELL_BUFFER, "buffer")]:
        members = [volume_of_shell[(r_in, r_out)]
                   for r_in, r_out, t in layout if t == tag]
        gmsh.model.addPhysicalGroup(3, members, tag, name=name)

    # The graded size field. `Min` merges it with the angular cap below.
    r = "sqrt(x*x+y*y+z*z)"
    graded = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(
        graded, "F", f"{h}*(1 + {grade}*max(0, {r} - {RE})"
                     f" + {grade}*max(0, {RC} - {r}))")
    # The angular cap lc <= 2 pi r / N: every sphere of radius r carries at
    # least N cells around a great circle, so both DtN spheres resolve the
    # truncation degree even where the grading has made the cells coarse.
    fields = [graded]
    if min_cells_per_great_circle:
        capped = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            capped, "F", f"{2 * np.pi / min_cells_per_great_circle}*{r}")
        fields.append(capped)
    # Local refinement from the caller. Each returned field is an upper bound
    # on the cell size in the region it describes; `Min` makes the smallest
    # bound win, so the extra fields can only make cells smaller. With no
    # hook the list is unchanged, and so is the mesh.
    if extra_size_fields is not None:
        # A Python exception in the hook must finalize gmsh, for the same
        # reason as a meshing failure below.
        try:
            fields.extend(extra_size_fields(gmsh))
        except Exception:
            gmsh.finalize()
            raise
    merged = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(merged, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(merged)
    # Without these three options, gmsh's own curvature and point heuristics
    # override the field near the small inner sphere and refine it by an order.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # A meshing failure must still finalize gmsh. Otherwise gmsh stays
    # initialised with a dirty model, and the next `generate` in the same
    # process returns an empty mesh instead of raising.
    try:
        # Caller options, for example another 3-D algorithm. They come last,
        # so that a caller can also override the three options above. They
        # are inside the `try`, because gmsh raises for an unknown option
        # name, and that error must finalize gmsh too.
        for key, value in (mesh_options or {}).items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(3)
    except Exception:
        gmsh.finalize()
        raise

    stats = {"cells": {}, "h": h, "grade": grade, "litho_layers": litho_layers,
             "min_cells_per_great_circle": min_cells_per_great_circle,
             "configuration": configuration}
    for r_in, r_out, tag in layout:
        vol = volume_of_shell[(r_in, r_out)]
        _, etags, _ = gmsh.model.mesh.getElements(3, vol)
        stats["cells"][(r_in, r_out)] = sum(len(e) for e in etags)
    if quality:
        stats["quality"] = _quality(layout, volume_of_shell)

    gmsh.write(filename)
    gmsh.finalize()
    return filename, layout, stats


def _quality(layout, volume_of_shell):
    """Per-shell tetrahedron quality, with gmsh's own measures.

    `gamma` is the inscribed/circumscribed radius ratio, scaled so that an
    equilateral tetrahedron is 1 and a degenerate one 0. `minSICN` is the
    signed inverse condition number, negative for an inverted element. The
    subdivision spheres make the lithosphere cells anisotropic by design,
    which lowers `gamma` and leaves `minSICN` positive; only an inverted
    element is a defect.
    """
    # gmsh is imported here, not at module level, so that the benchmark
    # driver can import the constants and `curve_mesh` of this module in a
    # Firedrake installation without the gmsh Python module.
    import gmsh

    out = {}
    for r_in, r_out, _ in layout:
        vol = volume_of_shell[(r_in, r_out)]
        _, etags, _ = gmsh.model.mesh.getElements(3, vol)
        tags = np.concatenate(etags) if len(etags) else np.array([], dtype=int)
        if tags.size == 0:
            continue
        gamma = np.asarray(gmsh.model.mesh.getElementQualities(tags, "gamma"))
        sicn = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN"))
        out[(r_in, r_out)] = {
            "n": tags.size, "gamma_min": gamma.min(), "gamma_mean": gamma.mean(),
            "sicn_min": sicn.min(), "n_inverted": int((sicn <= 0).sum()),
        }
    return out


def curve_mesh(linear_mesh, name=None):
    """Remap a straight-sided mesh to P2 coordinates, so mesh spheres are curved.

    Each P2 node is pushed radially onto the radius that the linear (P1)
    interpolant of the vertex radii takes there, X_p2 = (r_p1 / r) X. An edge
    whose two vertices lie on a sphere of radius R has r_p1 = R along its
    length, so its midpoint moves onto that sphere and the edge becomes a
    quadratic arc on it. No origin guard is needed, because r >= 0.5 Rc on
    this mesh.

    `Submesh` does not inherit the parent's coordinate field, so the driver
    applies this to the parent and to the mantle submesh separately.

    G-ADOPT, and with it Firedrake, is imported here and not at module level,
    so that the mesh generator runs in a Python environment that has gmsh and
    no Firedrake. The names come from `gadopt`, which re-exports the Firedrake
    namespace, so that gadopt imports before firedrake as the library
    requires.

    Args:
      linear_mesh: a Firedrake mesh with P1 coordinates.
      name: the name of the new mesh. A `CheckpointFile` that holds two
        meshes needs two distinct names.

    Returns:
      A new Firedrake mesh with P2 coordinates.
    """
    from gadopt import (Function, FunctionSpace, Mesh,  # noqa: PLC0415
                        SpatialCoordinate, VectorFunctionSpace, dot, sqrt)

    X = SpatialCoordinate(linear_mesh)
    r = sqrt(dot(X, X))
    # The vertex radii, linearly interpolated inside each cell.
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        (r_p1 / r) * X)
    if name is None:
        return Mesh(X_p2)
    return Mesh(X_p2, name=name)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configuration", default="coarse",
                    choices=list(CONFIGURATIONS),
                    help="named lateral spacing: coarse 500 km, medium 250 km, "
                         "fine 120 km, production 78 km")
    ap.add_argument("--h", type=float, default=None,
                    help="explicit non-dimensional lateral spacing; overrides "
                         "the configuration's spacing")
    ap.add_argument("--grade", type=float, default=2.0,
                    help="linear growth rate of the cell size away from the "
                         "mantle")
    ap.add_argument("--litho-layers", type=int, default=None,
                    help="lithosphere sub-shells; default from the "
                         "configuration (2 for coarse). Use 1 with coarse for "
                         "the benchmark mesh b2_coarse_ar7.msh")
    ap.add_argument("--min-cells", type=int, default=None,
                    help="minimum cells around a great circle of any sphere; "
                         "default from the configuration (32 for coarse)")
    ap.add_argument("--output", default="selfgrav_sphere.msh")
    args = ap.parse_args()

    name, layout, stats = generate(
        args.output, args.configuration, args.h, args.grade,
        args.litho_layers, args.min_cells, verbose=True, quality=True)
    ref = analytic(stats["litho_layers"])
    print(f"\n{name}: h = {stats['h']:.6f} "
          f"({stats['h'] * D_KM:.0f} km), grade = {stats['grade']}, "
          f"litho_layers = {stats['litho_layers']}, "
          f"min_cells = {stats['min_cells_per_great_circle']}, "
          f"design aspect ratio "
          f"{design_aspect_ratio(stats['h'], stats['litho_layers']):.1f}")
    print(f"{'r_in':>9} {'r_out':>9} {'tag':>5} {'cells':>9} "
          f"{'gamma_min':>10} {'gamma_avg':>10} {'sicn_min':>10} {'inverted':>9}")
    for r_in, r_out, tag in layout:
        q = stats.get("quality", {}).get((r_in, r_out), {})
        print(f"{r_in:9.6f} {r_out:9.6f} {tag:5d} "
              f"{stats['cells'][(r_in, r_out)]:9d} "
              f"{q.get('gamma_min', float('nan')):10.4f} "
              f"{q.get('gamma_mean', float('nan')):10.4f} "
              f"{q.get('sicn_min', float('nan')):10.4f} "
              f"{q.get('n_inverted', -1):9d}")
    tag_of_shell = {(a, b): t for a, b, t in layout}
    per = {t: sum(n for shell, n in stats["cells"].items()
                  if tag_of_shell[shell] == t)
           for t in (CELL_MANTLE, CELL_INNER, CELL_BUFFER)}
    print(f"\n  mantle {per[CELL_MANTLE]:8d}")
    print(f"  inner  {per[CELL_INNER]:8d}  "
          f"({per[CELL_INNER] / per[CELL_MANTLE]:.3f} of mantle)")
    print(f"  buffer {per[CELL_BUFFER]:8d}  "
          f"({per[CELL_BUFFER] / per[CELL_MANTLE]:.3f} of mantle)")
    print(f"  total  {sum(per.values()):8d}")
    print(f"\n  mantle volume {ref['vol_mantle']:.6f}, "
          f"buffer volume {ref['vol_buffer']:.6f}")
