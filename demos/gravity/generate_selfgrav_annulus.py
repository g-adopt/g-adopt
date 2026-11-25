"""Four-region parent annulus for the self-gravitating GIA prototype.

Phase 2 of `NOTES/PLAN-MONOLITHIC-SELFGRAV.md`, geometry from
`ROADMAP-GIA-SELFGRAV.md` §1.1, tags from §1.5.  This is the Stack B parent:
the potential lives on the whole thing, the mechanics on a `Submesh` of the
mantle, and the two DtN boundaries are stood off from the sources by a factor
of two at each end.

    0.5 Rc ---- inner (102) ---- Rc ---- mantle (101) ---- Re ---- buffer (103) ---- 2 Re
       |                          |                        |                          |
   curve 5                    curve 3                  curve 2                    curve 4
  interior DtN            interface, INTERIOR      interface, INTERIOR        exterior DtN
                          facet of the parent      facet of the parent

Non-dimensionalised by D = Re - Rc = 2891 km, so Rc = 1.2037, Re = 2.2037,
0.5 Rc = 0.6019 and 2 Re = 4.4074.

Curves 2 and 3 are *interior* facets of the parent and *boundary* facets of
the mantle submesh; that is the entire point of the geometry, and it is why
they get physical groups of their own even though gmsh has no need of them.
Spike S1 established that a tagged interior curve survives Firedrake's reader
and that `dS(2)` gives 2 pi Re — and, more importantly, that asking for the
wrong kind of tag gives **zero and a warning, not an error**.  Anything built
on these tags must check its measure against 2 pi R.

This is a *new* generator.  `generate_gravity_annulus.py` (in the `poisson`
worktree, whence this lineage comes) stays as it is: it builds the validated
two-region mesh the standalone gravity convergence studies run on, and those
studies are config D — the exterior DtN sits *on* the sources with no buffer
at all, which is the opposite geometry from this one and is right for what it
does.

## Why the layering is graded rather than uniform

Road map §1.2's second proviso: E5 measured a coarse buffer concentrating
100-200x the mantle error and bleeding it back inward through Re, which is
how the earlier extended-domain configuration ended up capped at ~1.5e-6.  A
buffer four times the mantle's area at the mantle's resolution would also cost
more cells than the mantle, which defeats the point.  So both stand-off
regions grade geometrically away from the mantle with ratio ~1.3, starting at
the mantle's own radial spacing so nothing jumps across Rc or Re.  The layer
count follows from inverting the geometric series,

    W = h0 (g^n - 1) / (g - 1)   ->   n = ceil( log(1 + W (g-1) / h0) / log g )

with W the region's radial width and h0 = `dr_mantle`.  gmsh then normalises
the progression so the layers exactly fill W, so h0 is a target rather than a
guarantee; at the ratios used here it is met to a few percent.

The inner region grades *inward*.  Its radial lines are drawn outward, from
0.5 Rc to Rc, so the coefficient that coarsens away from the mantle is 1/g and
not g.  Getting that backwards produces a mesh that is finest where there is
nothing to resolve and coarsest at the interface, and it does not look wrong
in a plot.

## Density interfaces

`interfaces` is a list of radii strictly inside the mantle that layer
boundaries must land on, so a DG0 density supported between two of them is
represented exactly and leaks no m = 0 content (the finding recorded under
"mesh-conforming shell boundaries" in `demos/gravity/CLAUDE.md`, where
non-conforming shells produced 230 % error).  Each mantle sub-annulus is
meshed uniformly at ~`dr_mantle` and all of them go into cell group 101, so
the tag set does not depend on how many interfaces there are.  They get no
physical curve group: what DG0 conformity needs is mesh *nodes* on the circle,
which the transfinite structure supplies whether or not the curve is written
to file.

## Resolution knobs

`dr_mantle` and `n_azimuthal` are the two, and both refine the mantle while
leaving the graded regions' *relative* structure alone (their layer counts
follow dr_mantle through the formula above).  That is what a convergence study
needs: a single family in which the buffer does not become relatively coarser
as the mantle is refined.

## Measured

`validate_selfgrav_annulus.py` is the acceptance gate; numbers as of
2026-07-30.  At the defaults (dr_mantle = 0.05, n_azimuthal = 128, quads) the
straight-facet areas and circumferences carry the polygon error, 4.0e-04 and
1.0e-04; `dS(2)` on the parent equals `ds(2)` on the mantle submesh bit for
bit; and `curve_mesh` takes both down to 1.2e-08 and 6.0e-09.

That last pair is **not** the "~1e-11" the earlier record quotes, and neither
number is wrong: a P2 arc through three points of a circle deviates from it at
O(theta^4) in the azimuthal cell angle, so the residual is a property of
n_azimuthal and not of the method.  The measured order is exactly 4.00 across
64/128/256/512, reaching 4.7e-11 at 512 — which is the resolution the E3
record was written at.

Cost.  The buffer/mantle cell-count ratio depends only on the radial ladder
(both regions carry the same azimuthal count), and road map §1.3's "under a
third" holds only once the mantle is resolved: 0.80 at dr_mantle = 0.1, 0.55
at 0.05, 0.325 at 0.025, 0.20 at 0.0125, 0.11 at 0.00625.  A geometrically
graded region's layer count grows like log(1/dr) while a uniform one grows
like 1/dr, so the buffer gets relatively cheaper the more the mantle is
refined — which is the right way round, but it means a coarse prototype pays
more for the stand-off than the road map's figure suggests.
"""
import numpy as np
import gmsh

# Non-dimensionalised by D = Re - Rc = 2891 km.
RC = 1.2037
RE = 2.2037
R_INNER = 0.5 * RC
R_OUTER = 2.0 * RE

# Cell groups (road map §1.5: the 3-D mesh uses the same numbers, so nothing
# downstream is renamed when the prototype is promoted).
CELL_MANTLE, CELL_INNER, CELL_BUFFER = 101, 102, 103

# Curve groups.  2 and 3 are interior to the parent; 4 and 5 are its two
# actual boundaries and carry the DtN maps.
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5


def graded_layers(width, dr, growth):
    """Layer count for a geometric region of `width` starting at spacing `dr`.

    Inverts W = dr (g^n - 1)/(g - 1).  At least one layer, always.
    """
    if growth == 1.0:
        return max(1, round(width / dr))
    return max(1, int(np.ceil(np.log1p(width * (growth - 1) / dr)
                              / np.log(growth))))


def radial_zones(dr_mantle=0.05, growth=1.3, interfaces=()):
    """The radial ladder as (r_in, r_out, n_layers, progression, cell_tag).

    Ordered outward from 0.5 Rc.  Returned separately from `generate` so the
    validation script can predict cell counts without reading the mesh.
    """
    interfaces = tuple(sorted(interfaces))
    assert all(RC < r < RE for r in interfaces), \
        f"density interfaces must lie strictly inside the mantle: {interfaces}"

    zones = []

    # Inner: grades inward, so the progression on an outward-drawn line is 1/g
    # and the fine end is at Rc.
    zones.append((R_INNER, RC, graded_layers(RC - R_INNER, dr_mantle, growth),
                  1.0 / growth, CELL_INNER))

    # Mantle: uniform, cut at every density interface.
    edges = (RC,) + interfaces + (RE,)
    for r_in, r_out in zip(edges[:-1], edges[1:]):
        zones.append((r_in, r_out, max(1, round((r_out - r_in) / dr_mantle)),
                      1.0, CELL_MANTLE))

    # Buffer: grades outward.
    zones.append((RE, R_OUTER, graded_layers(R_OUTER - RE, dr_mantle, growth),
                  growth, CELL_BUFFER))

    return zones


def generate(filename="selfgrav_annulus.msh", dr_mantle=0.05, n_azimuthal=128,
             growth=1.3, interfaces=(), quads=True, verbose=False):
    """Writes the four-region annulus and returns (filename, zones)."""
    assert n_azimuthal % 4 == 0, "n_azimuthal must be divisible by 4"

    zones = radial_zones(dr_mantle, growth, interfaces)
    radii = [zones[0][0]] + [z[1] for z in zones]

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("selfgrav_annulus")
    geo = gmsh.model.geo

    centre = geo.addPoint(0, 0, 0)
    angles = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # points[i][j]: radius i, quadrant corner j.  Four arcs per circle keeps
    # every gmsh circle arc under pi, which addCircleArc requires.
    points = [[geo.addPoint(R * cx, R * cy, 0) for cx, cy in angles]
              for R in radii]
    arcs = [[geo.addCircleArc(pts[j], centre, pts[(j + 1) % 4])
             for j in range(4)] for pts in points]
    radial = [[geo.addLine(points[i][j], points[i + 1][j]) for j in range(4)]
              for i in range(len(radii) - 1)]

    surfaces = []
    for i in range(len(zones)):
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
    for i, (_, _, n_layers, coef, _) in enumerate(zones):
        for line in radial[i]:
            geo.mesh.setTransfiniteCurve(line, n_layers + 1,
                                         "Progression", coef)
    for ring in surfaces:
        for s in ring:
            geo.mesh.setTransfiniteSurface(s)
            if quads:
                geo.mesh.setRecombine(2, s)

    geo.synchronize()

    # The four concentric circles that carry a tag.  Interfaces interior to
    # the mantle deliberately get none.
    tagged = {R_INNER: (CURVE_INNER, "inner_dtn"), RC: (CURVE_RC, "Rc"),
              RE: (CURVE_RE, "Re"), R_OUTER: (CURVE_OUTER, "outer_dtn")}
    for R, ring in zip(radii, arcs):
        if R in tagged:
            tag, name = tagged[R]
            gmsh.model.addPhysicalGroup(1, ring, tag, name=name)

    # Cell groups: every mantle sub-annulus joins 101.
    for tag, name in [(CELL_MANTLE, "mantle"), (CELL_INNER, "inner"),
                      (CELL_BUFFER, "buffer")]:
        members = [s for ring, zone in zip(surfaces, zones)
                   if zone[4] == tag for s in ring]
        gmsh.model.addPhysicalGroup(2, members, tag, name=name)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    return filename, zones


def analytic():
    """Reference areas and circumferences, for the validation prints."""
    return {
        "area_inner": np.pi * (RC**2 - R_INNER**2),
        "area_mantle": np.pi * (RE**2 - RC**2),
        "area_buffer": np.pi * (R_OUTER**2 - RE**2),
        "len_inner": 2 * np.pi * R_INNER,
        "len_Rc": 2 * np.pi * RC,
        "len_Re": 2 * np.pi * RE,
        "len_outer": 2 * np.pi * R_OUTER,
    }


if __name__ == "__main__":
    name, zones = generate(verbose=True)
    print(name)
    for r_in, r_out, n, coef, tag in zones:
        print(f"  {r_in:7.4f} -> {r_out:7.4f}  {n:3d} layers  "
              f"progression {coef:5.3f}  cell tag {tag}")
