"""The four-region sphere of the Martinec et al. (2018) sea-level benchmark.

This generator writes the same four-region tetrahedral sphere as
`../3d_spada_selfgrav/generate_selfgrav_sphere.py` (inner buffer, mantle,
outer buffer, the density interfaces and the two DtN spheres, with the same
physical groups), and adds local refinement in three regions of the Earth
surface Re:

- the ice cap of loads L1 and L2, centred at colatitude 25, longitude 75
  degrees, refined out to `cap_deg` (12 degrees; the cap radius is 10);
- the B1 coastline band, `|psi_B1 - 24.85| <= band_deg` around the basin
  centre (colatitude 100, longitude 320), used by case B;
- the B2 coastline band, `|psi_B2 - 24.85| <= band_deg` around the basin
  centre (colatitude 35, longitude 25), used by cases C and D.

One mesh with all three regions serves B, C and D. The geometry, the base
grading and the physical groups come unchanged from the Spada generator.
This file only builds extra gmsh size fields and passes them in through the
`extra_size_fields` hook of `generate`.

## The size field of one region

Lengths are non-dimensional, in units of the mantle thickness D = 2891 km,
as in the Spada generator. For a region centre with unit vector c, the
angular distance of a point X = (x, y, z) from the centre is

    psi = acos(clamp((c . X) / max(r, r_floor), -1, 1)),     r = |X|.

`r_floor` is 1e-12. It changes nothing at any point of the mesh, where r is
at least half of Rc, and it keeps the expression finite if gmsh evaluates the
field at the origin.

The angular distance outside the region, `d_ang` (radians, zero inside), is
`max(0, psi - cap)` for the cap and `max(0, |psi - psi0| - band)` for a
coastline band. The lateral size, in Will Scott's form
(`~/Workplace/sl_testing/globe_test/meshes/unstructured_sphere.py`), grows
linearly with the arc length `r d_ang` outside the region:

    h_region = h_fine + grade_lateral * r * d_ang.

The refinement fades with the distance from Re over the length `L_depth`:

    t = tanh(|r - Re| / L_depth),
    h = (1 - t) h_region + t h_graded.

`h_graded` is the base size field of the Spada generator,
`h_base (1 + grade max(0, r - Re) + grade max(0, Rc - r))`. Inside the mantle
it is the constant `h_base`, so there the fade is exactly the form
`(1 - t) h_region + t h_base` of the design. The buffers are the reason for
`h_graded` instead of `h_base`: the base grading makes the buffer cells much
larger than `h_base`, and a fade towards `h_base` would refine the whole of
both buffers to `h_base` under the `Min` merge. With `h_graded` the extra
field is never smaller than the base field far from Re or far from the
region, so it refines only near the region on Re.

## Resolution of the rest of the mesh

The base mesh is the benchmark mesh `b2_coarse_ar7.msh`: `h_base` 500 km,
grade 2, one lithosphere layer (70 km), at least 32 cells around a great
circle. At 78 km lateral spacing the refined lithosphere cells have an
aspect ratio near 1, and the coarse cells keep about 7.

## The 3-D algorithm and flat tetrahedra

The driver curves the mesh with `curve_mesh`, which moves every P2 node
radially onto the linear interpolant of the vertex radii. A tetrahedron with
all four vertices on one mesh sphere ("flat cell") gets all ten P2 nodes on
that sphere. Such a cell has almost no volume, and after the curving its
Jacobian determinant can change sign inside the cell (a folded cell). With
the gmsh default 3-D algorithm (Delaunay, `Mesh.Algorithm3D = 1`) this
refined mesh got two flat cells, one on Re and one on the inner DtN sphere,
and both folded after `curve_mesh`. Their positions change at random with
small changes of the size field.

The flat cells come from the random point insertion of the Delaunay
algorithm, so the gmsh option `Mesh.RandomSeed` moves or removes them. The
default of this generator is the Delaunay algorithm with seed 7. On phoenix
with gmsh 4.15.2, seeds 1 (the gmsh default) to 4 and 6 give one or two flat
cells, and seeds 5, 7 and 8 give none. Seed 7 also gives the largest smallest
`gamma` of the three.

The HXT algorithm also gives no flat cell. The seed option applies to HXT
too, so `--algorithm3d 10` alone uses seed 7 (153 174 cells), and
`--algorithm3d 10 --random-seed 1` gives the gmsh default seed (153 570
cells). With either seed, the mean edge of the HXT volume cells in the shells
above the refined regions is 0.96 to 1.51 times the size field. For the
Delaunay cells with seed 7 the ratio is 0.85 to 1.02. The measurement is in
`review-round2.md` of `NOTES/team/w6-refined-mesh/`, section 1.

A seed is not portable between gmsh versions. After the generator writes the
file, it therefore counts the flat cells and stops with a non-zero exit
status if there is one. Then use another `--random-seed`. The folded-cell
count needs Firedrake and is in the checks script
`NOTES/mesh/check_martinec_mesh.py`.

## Usage

    python generate_martinec_sphere.py                       # 78 km, the default
    python generate_martinec_sphere.py --h-fine-km 120       # the fallback

The output file name carries the parameters, for example
`martinec_h78_cap12_band4_gl0.5_depth500_base500_grade2_ll1_mc32_alg1_seed7.msh`. The generator needs the
gmsh Python module and not Firedrake.
"""
import argparse
import os
import sys

import numpy as np

# The Spada generator is a sibling directory, not a package. Put it on the path
# so that this file uses its geometry, its physical groups and its constants,
# instead of a copy that can drift out of step with the Spada meshes.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "3d_spada_selfgrav"))
import generate_selfgrav_sphere as gen  # noqa: E402

#: The ice cap centre of loads L1 and L2, (colatitude, longitude) in degrees.
CAP_CENTRE = (25.0, 75.0)
#: The basin centre of B1 (case B), (colatitude, longitude) in degrees.
B1_CENTRE = (100.0, 320.0)
#: The basin centre of B2 (cases C and D), (colatitude, longitude) in degrees.
B2_CENTRE = (35.0, 25.0)
#: The angular radius of the initial coastline around both basin centres, in
#: degrees: the zero of zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2)).
COAST_PSI_DEG = 24.85

#: The floor of the radius in the division `(c . X) / r` of the angular
#: distance, non-dimensional. It keeps the expression finite at the origin and
#: changes no value at a mesh point, where r is at least 0.5 Rc.
R_FLOOR = 1e-12

#: Default parameters of the refinement. Lengths in km, angles in degrees.
DEFAULTS = {"h_fine_km": 78.0, "cap_deg": 12.0, "band_deg": 4.0,
            "grade_lateral": 0.5, "depth_km": 500.0, "h_base_km": 500.0,
            "grade": 2.0, "litho_layers": 1, "min_cells": 32,
            "algorithm3d": 1, "random_seed": 7}


def unit_vector(colatitude_deg, longitude_deg):
    """The Cartesian unit vector of a point given in geographic degrees.

    Args:
      colatitude_deg: colatitude, degrees from the north pole (the +z axis).
      longitude_deg: longitude, degrees east from the +x axis.

    Returns:
      A NumPy array of shape (3,) with unit length.
    """
    theta = np.radians(colatitude_deg)
    phi = np.radians(longitude_deg)
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def regions(cap_deg=DEFAULTS["cap_deg"], band_deg=DEFAULTS["band_deg"]):
    """The three refined regions as plain data.

    Each region is a dictionary with the keys `name`, `centre` (unit vector),
    `kind` ("cap" or "band"), `radius` (the cap radius or the band centre
    line, radians) and `half_width` (zero for a cap, the band half width for
    a band, radians). The same description builds the gmsh fields and
    classifies the surface triangles in the checks, so the two cannot differ.

    Args:
      cap_deg: the angular radius of the refined cap, degrees.
      band_deg: the half width of each coastline band, degrees.

    Returns:
      A list of three dictionaries.
    """
    return [
        {"name": "cap", "centre": unit_vector(*CAP_CENTRE), "kind": "cap",
         "radius": np.radians(cap_deg), "half_width": 0.0},
        {"name": "B1_coast", "centre": unit_vector(*B1_CENTRE), "kind": "band",
         "radius": np.radians(COAST_PSI_DEG),
         "half_width": np.radians(band_deg)},
        {"name": "B2_coast", "centre": unit_vector(*B2_CENTRE), "kind": "band",
         "radius": np.radians(COAST_PSI_DEG),
         "half_width": np.radians(band_deg)},
    ]


def angular_distance_outside(region, points):
    """The angular distance of points outside a region, zero inside, NumPy.

    This is the NumPy twin of the gmsh expression that `region_size_expression`
    writes, for the checks of a written mesh.

    Args:
      region: one entry of `regions()`.
      points: array of shape (n, 3) of Cartesian points, any radius.

    Returns:
      Array of shape (n,), radians.
    """
    # The same floor on the divisor as in the gmsh expression.
    r = np.maximum(np.linalg.norm(points, axis=1), R_FLOOR)
    cospsi = np.clip(points @ region["centre"] / r, -1.0, 1.0)
    psi = np.arccos(cospsi)
    if region["kind"] == "cap":
        return np.maximum(0.0, psi - region["radius"])
    return np.maximum(0.0, np.abs(psi - region["radius"]) - region["half_width"])


def area_on_sphere(region, radius):
    """The exact area of a region on a sphere of the given radius.

    A cap of angular radius a has area 2 pi R^2 (1 - cos a). A band between
    the angular radii a1 < a2 around a centre has area
    2 pi R^2 (cos a1 - cos a2).

    Args:
      region: one entry of `regions()`.
      radius: sphere radius, in the units of the returned area's square root.

    Returns:
      The area.
    """
    if region["kind"] == "cap":
        return 2 * np.pi * radius**2 * (1 - np.cos(region["radius"]))
    a1 = region["radius"] - region["half_width"]
    a2 = region["radius"] + region["half_width"]
    return 2 * np.pi * radius**2 * (np.cos(a1) - np.cos(a2))


def _num(value):
    """A number as a plain decimal string for a gmsh `MathEval` expression.

    Fixed-point notation with 15 decimals keeps the non-dimensional values
    (all between 1e-3 and 5 here) to about 1e-15 absolute. It never writes
    an exponent, so the gmsh expression parser reads only plain decimals. A
    negative number is put in parentheses, because the parser rejects a
    unary minus after a binary operator, as in `a + -0.2*x`.
    """
    text = f"{float(value):.15f}"
    return f"({text})" if text.startswith("-") else text


def region_size_expression(region, h_fine, grade_lateral, depth, h_base,
                           grade):
    """The gmsh `MathEval` string of the size field of one region.

    All quantities are non-dimensional (units of D). The expression is

        h = (1 - t) (h_fine + grade_lateral r d_ang) + t h_graded,
        t = tanh(|r - Re| / depth),

    with `d_ang` the angular distance outside the region and `h_graded` the
    base size field of the Spada generator (see the module docstring for why
    the fade goes to `h_graded` and not to `h_base`).

    The argument of `acos` is clamped to [-1, 1]. Rounding in `(c . X) / r`
    can give a value just above 1 on the region axis, where `acos` would
    return NaN and gmsh would take the size as undefined.

    The divisor is `max(r, R_FLOOR)`. At every point of the mesh `r` is at
    least half of Rc, far above `R_FLOOR`, so the value of the field there is
    the same as with the plain `r`. At the origin the plain division is
    0 / 0, and gmsh aborts the process with a `mathex` error that no Python
    `try` can catch. Only a mesher that evaluates the field at the origin
    reaches this case. `Mesh.OptimizeNetgen` is one example, and it then
    still fails on the base size field `2 pi r / N`, which is zero there.

    Args:
      region: one entry of `regions()`.
      h_fine: the lateral size inside the region.
      grade_lateral: the growth of the size per unit arc length outside the
        region, dimensionless.
      depth: the radial fade length.
      h_base: the base lateral size of the Spada grading.
      grade: the radial growth rate of the Spada grading.

    Returns:
      The expression string, in the gmsh variables x, y, z.
    """
    # Every number goes into the string through `_num`, a plain decimal. The
    # `repr` of a NumPy scalar is `np.float64(...)`, which the gmsh expression
    # parser rejects.
    c0, c1, c2 = (_num(v) for v in region["centre"])
    r = "sqrt(x*x+y*y+z*z)"
    # The angular distance from the centre, radians.
    # The divisor is floored at `R_FLOOR` (see the docstring).
    psi = (f"acos(min(1, max(-1, ({c0}*x + {c1}*y + {c2}*z)"
           f"/max({r}, {_num(R_FLOOR)}))))")
    if region["kind"] == "cap":
        d_ang = f"max(0, {psi} - {_num(region['radius'])})"
    else:
        d_ang = (f"max(0, abs({psi} - {_num(region['radius'])})"
                 f" - {_num(region['half_width'])})")
    # The lateral size: fine inside, growing linearly with arc length outside.
    h_region = f"({_num(h_fine)} + {_num(grade_lateral)}*{r}*{d_ang})"
    # The base size field of the Spada generator, repeated here in the same
    # form so that the fade reaches exactly the base size away from Re.
    h_graded = (f"({_num(h_base)}*(1"
                f" + {_num(grade)}*max(0, {r} - {_num(gen.RE)})"
                f" + {_num(grade)}*max(0, {_num(gen.RC)} - {r})))")
    # The radial fade weight: 0 on Re, towards 1 at depth >> `depth`.
    t = f"tanh(abs({r} - {_num(gen.RE)})/{_num(depth)})"
    return f"(1 - {t})*{h_region} + {t}*{h_graded}"


def martinec_size_fields(h_fine, cap_deg, band_deg, grade_lateral, depth,
                         h_base, grade):
    """A hook for `generate_selfgrav_sphere.generate(extra_size_fields=...)`.

    Args:
      h_fine, grade_lateral, depth, h_base, grade: as in
        `region_size_expression`, non-dimensional.
      cap_deg, band_deg: as in `regions`, degrees.

    Returns:
      A callable that receives the gmsh module, adds one `MathEval` field per
      region, and returns the list of their tags.
    """
    def hook(gmsh):
        tags = []
        for region in regions(cap_deg, band_deg):
            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(
                field, "F", region_size_expression(
                    region, h_fine, grade_lateral, depth, h_base, grade))
            tags.append(field)
        return tags
    return hook


def default_filename(h_fine_km, cap_deg, band_deg, grade_lateral, depth_km,
                     h_base_km, grade, litho_layers, min_cells, algorithm3d,
                     random_seed=None):
    """The mesh file name, which records every parameter of the mesh.

    Two meshes that differ in any parameter get different names, so that one
    file never overwrites another. `%g` gives the shortest exact form of
    each value, so 78.0 becomes `78` and 0.5 stays `0.5`. The random seed
    goes into the name only when it is set.
    """
    name = (f"martinec_h{h_fine_km:g}_cap{cap_deg:g}_band{band_deg:g}"
            f"_gl{grade_lateral:g}_depth{depth_km:g}_base{h_base_km:g}"
            f"_grade{grade:g}_ll{litho_layers}_mc{min_cells}"
            f"_alg{algorithm3d}")
    if random_seed is not None:
        name += f"_seed{random_seed}"
    return name + ".msh"


def flat_cells(path):
    """The tetrahedra with all four vertices on one sphere, read from a file.

    `curve_mesh` puts all ten P2 nodes of such a cell on that sphere, and the
    curved cell can fold (see the module docstring). The test is the spread
    of the four vertex radii against the mean radius, with a relative
    tolerance of 1e-6. The smallest radial gap between two mesh spheres is
    the thickness of one lithosphere sub-shell, 70 km / `litho_layers`: 0.024
    D for one layer, 0.0121 D for two and 0.0020 D for twelve. The tolerance
    is at most 1e-6 times the largest radius, below 1e-5 D, so a real cell
    between two spheres never passes it, and a flat cell passes with a spread
    near 1e-15.

    gmsh must not be initialised when this function is called. It
    initialises and finalises gmsh itself.

    Args:
      path: the .msh file.

    Returns:
      A list with one dictionary per flat cell: `group` (the physical volume
      tag), `radius`, `colatitude` and `longitude` of the centroid in
      degrees.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        # All nodes, indexed by gmsh node tag.
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.zeros((int(ntags.max()) + 1, 3))
        xyz[ntags.astype(int)] = coords.reshape(-1, 3)
        rad = np.linalg.norm(xyz, axis=1)
        found = []
        for tag in (gen.CELL_MANTLE, gen.CELL_INNER, gen.CELL_BUFFER):
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                types, _, enodes = gmsh.model.mesh.getElements(3, ent)
                for etype, en in zip(types, enodes):
                    # Type 4 is the linear tetrahedron, the only 3-D element
                    # this generator writes.
                    assert etype == 4, f"unexpected volume element type {etype}"
                    tets = en.reshape(-1, 4).astype(int)
                    r4 = rad[tets]
                    flat = np.ptp(r4, axis=1) < 1e-6 * r4.mean(axis=1)
                    for cell in tets[flat]:
                        c = xyz[cell].mean(axis=0)
                        found.append({
                            "group": tag, "radius": float(rad[cell].mean()),
                            "colatitude": float(np.degrees(
                                np.arccos(c[2] / np.linalg.norm(c)))),
                            "longitude": float(np.degrees(
                                np.arctan2(c[1], c[0])) % 360)})
    finally:
        gmsh.finalize()
    return found


def generate(output=".", h_fine_km=DEFAULTS["h_fine_km"],
             cap_deg=DEFAULTS["cap_deg"], band_deg=DEFAULTS["band_deg"],
             grade_lateral=DEFAULTS["grade_lateral"],
             depth_km=DEFAULTS["depth_km"], h_base_km=DEFAULTS["h_base_km"],
             grade=DEFAULTS["grade"], litho_layers=DEFAULTS["litho_layers"],
             min_cells_per_great_circle=DEFAULTS["min_cells"],
             algorithm3d=DEFAULTS["algorithm3d"],
             random_seed=DEFAULTS["random_seed"],
             verbose=False, quality=False):
    """Write the refined Martinec sphere.

    Args:
      output: a directory for the default file name, or a path ending in
        `.msh`.
      h_fine_km: lateral size in the refined regions, km.
      cap_deg: angular radius of the refined cap, degrees.
      band_deg: half width of each coastline band, degrees.
      grade_lateral: growth of the size per unit arc length outside a region.
      depth_km: radial fade length of the refinement, km.
      h_base_km: lateral size of the base mesh, km.
      grade: radial growth rate of the base size in the buffers.
      litho_layers: lithosphere sub-shells of the base mesh.
      min_cells_per_great_circle: angular cap of the base mesh.
      algorithm3d: the gmsh `Mesh.Algorithm3D`, 1 (Delaunay, the default)
        or 10 (HXT).
      random_seed: the gmsh `Mesh.RandomSeed`, or `None` for the gmsh
        default (1). The default 7 gives no flat cell on the 78 km mesh (see
        the module docstring).
      verbose: print gmsh's own log.
      quality: compute per-shell tetrahedron quality.

    Returns:
      `(filename, shells, stats)` as `generate_selfgrav_sphere.generate`, with
      the refinement parameters added to `stats`.
    """
    if output.endswith(".msh"):
        filename = output
    else:
        filename = os.path.join(output, default_filename(
            h_fine_km, cap_deg, band_deg, grade_lateral, depth_km, h_base_km,
            grade, litho_layers, min_cells_per_great_circle, algorithm3d,
            random_seed))
    # Every length goes to gmsh non-dimensional, in units of D.
    h_fine = h_fine_km / gen.D_KM
    depth = depth_km / gen.D_KM
    h_base = h_base_km / gen.D_KM
    hook = martinec_size_fields(h_fine, cap_deg, band_deg, grade_lateral,
                                depth, h_base, grade)
    # The 3-D algorithm, and the seed if one is set. The Spada generator sets
    # these after its own size-field options.
    options = {"Mesh.Algorithm3D": algorithm3d}
    if random_seed is not None:
        options["Mesh.RandomSeed"] = random_seed
    name, layout, stats = gen.generate(
        filename, configuration="coarse", h=h_base, grade=grade,
        litho_layers=litho_layers,
        min_cells_per_great_circle=min_cells_per_great_circle,
        verbose=verbose, quality=quality, extra_size_fields=hook,
        mesh_options=options)
    stats.update({"h_fine_km": h_fine_km, "cap_deg": cap_deg,
                  "band_deg": band_deg, "grade_lateral": grade_lateral,
                  "depth_km": depth_km, "h_base_km": h_base_km,
                  "algorithm3d": algorithm3d, "random_seed": random_seed})
    return name, layout, stats


def main():
    """Command line: build the mesh and print the parameters and cell counts."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h-fine-km", type=float, default=DEFAULTS["h_fine_km"],
                    help="lateral cell size in the refined regions, km "
                         "(78 production, 120 the fallback)")
    ap.add_argument("--cap-deg", type=float, default=DEFAULTS["cap_deg"],
                    help="angular radius of the refined cap, degrees")
    ap.add_argument("--band-deg", type=float, default=DEFAULTS["band_deg"],
                    help="half width of each coastline band, degrees")
    ap.add_argument("--grade-lateral", type=float,
                    default=DEFAULTS["grade_lateral"],
                    help="growth of the cell size per unit arc length "
                         "outside a region")
    ap.add_argument("--depth-km", type=float, default=DEFAULTS["depth_km"],
                    help="radial fade length of the refinement, km")
    ap.add_argument("--h-base-km", type=float, default=DEFAULTS["h_base_km"],
                    help="lateral cell size of the base mesh, km")
    ap.add_argument("--grade", type=float, default=DEFAULTS["grade"],
                    help="radial growth rate of the base size in the buffers")
    ap.add_argument("--litho-layers", type=int,
                    default=DEFAULTS["litho_layers"],
                    help="lithosphere sub-shells of the base mesh")
    ap.add_argument("--min-cells", type=int, default=DEFAULTS["min_cells"],
                    help="minimum cells around a great circle of any sphere")
    ap.add_argument("--algorithm3d", type=int, default=DEFAULTS["algorithm3d"],
                    help="gmsh Mesh.Algorithm3D: 1 Delaunay (default), "
                         "10 HXT; --random-seed applies to both")
    ap.add_argument("--random-seed", type=int,
                    default=DEFAULTS["random_seed"],
                    help="gmsh Mesh.RandomSeed (default 7, which gives no "
                         "flat cell on the 78 km mesh); change it if the "
                         "generator reports a flat cell")
    ap.add_argument("--output", default=".",
                    help="a directory for the default file name, which "
                         "records the parameters, or a path ending in .msh")
    ap.add_argument("--verbose", action="store_true",
                    help="print gmsh's own log")
    args = ap.parse_args()

    print("Refinement parameters: "
          f"h_fine {args.h_fine_km:g} km, cap {args.cap_deg:g} deg, "
          f"band {args.band_deg:g} deg, grade_lateral {args.grade_lateral:g}, "
          f"depth {args.depth_km:g} km, h_base {args.h_base_km:g} km, "
          f"grade {args.grade:g}, litho_layers {args.litho_layers}, "
          f"min_cells {args.min_cells}, algorithm3d {args.algorithm3d}, "
          f"random_seed {args.random_seed}", flush=True)
    name, layout, stats = generate(
        args.output, args.h_fine_km, args.cap_deg, args.band_deg,
        args.grade_lateral, args.depth_km, args.h_base_km, grade=args.grade,
        litho_layers=args.litho_layers,
        min_cells_per_great_circle=args.min_cells,
        algorithm3d=args.algorithm3d, random_seed=args.random_seed,
        verbose=args.verbose, quality=True)

    print(f"\n{name}")
    print(f"{'r_in':>9} {'r_out':>9} {'tag':>5} {'cells':>9} "
          f"{'gamma_min':>10} {'sicn_min':>10} {'inverted':>9}")
    for r_in, r_out, tag in layout:
        q = stats.get("quality", {}).get((r_in, r_out), {})
        print(f"{r_in:9.6f} {r_out:9.6f} {tag:5d} "
              f"{stats['cells'][(r_in, r_out)]:9d} "
              f"{q.get('gamma_min', float('nan')):10.4f} "
              f"{q.get('sicn_min', float('nan')):10.4f} "
              f"{q.get('n_inverted', -1):9d}")
    # Cells per region, summed over the shells of each cell group.
    tag_of_shell = {(a, b): t for a, b, t in layout}
    per = {t: sum(n for shell, n in stats["cells"].items()
                  if tag_of_shell[shell] == t)
           for t in (gen.CELL_MANTLE, gen.CELL_INNER, gen.CELL_BUFFER)}
    print(f"\n  mantle {per[gen.CELL_MANTLE]:9d}")
    print(f"  inner  {per[gen.CELL_INNER]:9d}")
    print(f"  buffer {per[gen.CELL_BUFFER]:9d}")
    print(f"  total  {sum(per.values()):9d}")

    # Flat cells fold under `curve_mesh`. A mesh with one is not usable
    # without a decision, so the command fails and names the cells.
    flat = flat_cells(name)
    print(f"\n  flat cells (all four vertices on one sphere): {len(flat)}")
    for cell in flat:
        print(f"    group {cell['group']}, radius {cell['radius']:.6f}, "
              f"colatitude {cell['colatitude']:.2f}, "
              f"longitude {cell['longitude']:.2f}")
    if flat:
        sys.exit("The mesh has flat cells. Use another --algorithm3d or "
                 "--random-seed.")


if __name__ == "__main__":
    main()
