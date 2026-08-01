"""Four-region 3-D parent sphere for the self-gravitating Spada benchmark.

The 3-D analogue of `demos/gravity/generate_selfgrav_annulus.py`, geometry from
`HANDOFF-SPADA-BENCHMARK.md` §4 and radii from §2.2.  This is the Stack B
parent: the potential lives on the whole thing, the mechanics on a `Submesh` of
the mantle, and the two DtN boundaries are stood off from the sources by a
factor of two at each end.

    0.5 Rc ---- inner (102) ---- Rc ---- mantle (101) ---- Re ---- buffer (103) ---- 2 Re
       |                          |                        |                          |
   surface 5                  surface 3                surface 2                  surface 4
  interior DtN            interface, INTERIOR      interface, INTERIOR        exterior DtN
                          facet of the parent      facet of the parent

Non-dimensionalised by D = Re - Rc = 2891 km, so Rc = 1.203736,
Re = 2.203736, 0.5 Rc = 0.601868 and 2 Re = 4.407471.  Tags are the same
integers as the 2-D generator, deliberately, so nothing downstream is renamed.

## Construction: OCC fragment, not transfinite

The 2-D generator is transfinite quads, which has no useful 3-D analogue on a
sphere.  Here every sphere in the ladder is an OCC solid and
`occ.fragment` cuts them into nested shells that share their bounding surfaces
exactly.  The core ball inside 0.5 Rc is then removed *non-recursively*, which
keeps the sphere at 0.5 Rc as a surface (it becomes the domain's inner
boundary) while deleting the volume behind it.

Two consequences worth stating, because they are what make the construction
worth using.  Every interface is a genuine geometric entity, so the mesh
conforms to it exactly and the DG0 density requirement of §2.3 is satisfied by
construction rather than by a resolution argument -- there is no way to get a
non-conforming shell out of this generator.  And every entity identifies
*itself*: a surface's radius is recovered from its area as sqrt(A/4pi) and a
shell from the radii of its two bounding surfaces, so nothing depends on the
order gmsh happens to return entities in.

## Radial resolution is decoupled from lateral resolution

§4's hard requirement is that the 70 km lithosphere carries at least two
mesh-conforming layers, so radial spacing near Re must be <= 35 km *whatever
the lateral spacing is*.  An isotropic size field cannot do that: 35 km is
0.0121 non-dimensional, and asking for isotropic 0.0121 elements through the
lithosphere costs some 7e6 cells before the mantle has been meshed at all.

The mechanism used instead is a **subdivision sphere**: an extra sphere at
mid-lithosphere, carrying no physical group and no density jump, purely so that
gmsh must place nodes on it.  Each of the two sub-shells is then filled with a
single layer of flat tetrahedra at the *lateral* size, and two radial layers
cost 15 000 cells at `--coarse` rather than seven million.  `litho_layers`
generalises it.  The elements are anisotropic by design, with aspect ratio
about h_lateral / h_radial; that is the price and §4 asks for it explicitly.

The same effect appears uninvited in the density shells.  The shell from
1.971982 to 2.058457 is 250 km thick, thinner than a 500 km lateral cell, so
gmsh must flatten elements there too and the shell costs about 1.75x what its
volume would suggest.  That is why the measured `--coarse` mantle count is
~8e4 rather than §4's estimated 5e4: the estimate is a volume/h^3 argument and
the thin shells do not obey it.

## The size field

A `MathEval` growing linearly away from the mantle in both directions,

    lc(r) = h (1 + grade * max(0, r - Re) + grade * max(0, Rc - r))

merged into a `Min`, and a second field capping it so that no sphere is
coarser than `min_cells_per_great_circle` cells around.

§4 suggests grade ~ 0.6; measured, that is far too weak.  Buffer cell counts at
`--coarse`, against a mantle of 79 588:

    grade      0.6      1.0      2.0      4.0
    buffer   76 054   48 143   26 560   16 206

§4 wants the buffer near 20 000, so the default here is **2.0**, not 0.6.  Even
at grade 4 the buffer's first sub-shell costs 13 500 cells on its own, because
it abuts Re and must match the mantle's resolution there; that floor is real
and no grading removes it.  Note the countervailing 2-D lesson in
`demos/gravity/CLAUDE.md` (E5): an under-resolved buffer concentrated 100-200x
the mantle error and bled it back inward through Re.  That was a
Dirichlet-truncated configuration and this one carries a DtN at 2 Re, but the
grade is exposed as a knob rather than baked in for exactly that reason.

**The angular cap is not optional, and the grading is why.**  Coarsening
inward makes the inner DtN sphere at 0.5 Rc a ~70-triangle polyhedron: its
area is 8.3% wrong before curving and still 2.0e-03 wrong after, an order
worse than any other surface on the mesh, and it is a surface that carries a
DtN map.  Capping lc at 2 pi r / N fixes it cheaply, because the refinement is
confined to where the cap binds.  Measured at `--coarse`, relative area error
of the two DtN spheres against total cell count:

    N            none        24        32        48
    ds(5) straight  8.3e-02   1.3e-02   7.7e-03   3.5e-03
    ds(5) curved    2.0e-03   5.2e-05   1.6e-05   3.2e-06
    ds(4) curved    2.1e-05   2.2e-05   1.6e-05   3.2e-06
    total cells    109 802   111 228   113 653   132 434

The default is **32**: it puts the inner sphere on terms with everything else
for 3.5% more cells.  N = 48 buys one further digit for 21% more cells, which
nothing downstream has asked for.  N counts cells around a great circle, so at
degree L there are N/L cells per wavelength -- 6.4 at the L = 5 the DtN
truncation uses.

## Resolution ladder

§4's table, in kilometres of lateral spacing, converted by D = 2891 km.
Only `--coarse` and `--medium` are exercised by the validator.
"""
import numpy as np
import gmsh

# Non-dimensionalised by D = Re - Rc = 2891 km (handoff §2.2).
D_KM = 2891.0
RC = 1.203736
RE = 2.203736
R_INNER = 0.5 * RC
R_OUTER = 2.0 * RE

#: Density interfaces, strictly inside the mantle.  rho_0 is a layered DG0
#: field and rho_0 and Phi_0 must share level surfaces (§2.3); a non-conforming
#: shell produced 230% error in the 2-D record.
DENSITY_INTERFACES = (1.971982, 2.058457, 2.179523)

#: Extra spheres in the buffer, so the size field has somewhere to grade (§4).
BUFFER_SPHERES = (2.75, 3.30)

#: The lithosphere, whose 70 km must carry at least two layers.
R_LITHO = DENSITY_INTERFACES[-1]

# Cell groups and surface groups: the same integers as the 2-D generator.
CELL_MANTLE, CELL_INNER, CELL_BUFFER = 101, 102, 103
SURF_RE, SURF_RC, SURF_OUTER, SURF_INNER = 2, 3, 4, 5

#: The three density interfaces, ascending, as *interior* facet groups.  These
#: are additive: 2, 3, 4 and 5 keep the meanings the 2-D generator gave them.
#: Nothing in the physics needs them -- the divergence-form Poisson source
#: produces the interface sheets automatically once rho_0 carries the jumps and
#: the mesh conforms -- but the reference-state gate (A3, handoff §2.3) has to
#: evaluate |g_0| *on* them, and a surface average over a tagged facet set is
#: both exact and far better conditioned than point-evaluating a discontinuous
#: grad(psi) at a density jump.
SURF_D1, SURF_D2, SURF_D3 = 6, 7, 8
DENSITY_INTERFACE_TAGS = {DENSITY_INTERFACES[0]: SURF_D1,
                          DENSITY_INTERFACES[1]: SURF_D2,
                          DENSITY_INTERFACES[2]: SURF_D3}

#: §4's ladder, as lateral spacing in km.
CONFIGURATIONS = {"coarse": 500.0, "medium": 250.0, "fine": 120.0,
                  "production": 78.0}

#: Radial and angular resolution **scale with the configuration**, so that the
#: lithosphere's aspect ratio is roughly held across the ladder instead of
#: improving as the mesh refines.
#:
#: This was a defect, and a controlled 2x2 identified it rather than an
#: inference.  With `litho_layers` and `min_cells_per_great_circle` fixed, only
#: lateral `h` changed down the ladder, so the lithosphere stayed pinned at
#: ~35 km radially and the realised aspect ratio *fell*: p95/max **9.02/24.6**
#: at `--coarse`, 6.06/11.2 at `--medium`, 3.40/9.8 at `--fine`.
#: Displacement-block AMG iterations to rtol 1e-10 tracked it: **>600, 34, 37**.
#: The control: a mesh at *medium* lateral spacing with `litho_layers=4` --
#: coarse's anisotropy at 740 470 cells, **more than plain medium** -- behaved
#: like coarse (inner `[u, psi]` 318-693 against plain medium's 8-19).  **It is
#: the aspect ratio, not h and not problem size.**
#:
#: One generator flaw, three symptoms: the `--coarse` -> `--medium` refinement
#: pair froze three error components and biased every convergence order
#: downward (it nearly manufactured a false locking positive); the `[u, psi]`
#: solver counts were non-monotone; and `--coarse` was *pathological* rather
#: than a lower bound, so every solver cost measured there overstated
#: production difficulty and every coarse->medium improvement read as scaling
#: when it was largely anisotropy relaxing.
#:
#: The old values stay reachable by passing `litho_layers` and
#: `min_cells_per_great_circle` explicitly, because several measurements were
#: taken on them and will need reproducing.
RESOLUTION_LADDER = {"coarse": (2, 32), "medium": (4, 64),
                     "fine": (8, 128), "production": (12, 192)}


def lateral_spacing(configuration):
    """Non-dimensional lateral cell size for a named configuration."""
    return CONFIGURATIONS[configuration] / D_KM


def resolution_defaults(configuration):
    """`(litho_layers, min_cells_per_great_circle)` for a configuration.

    See `RESOLUTION_LADDER` for why these scale rather than staying fixed.
    """
    return RESOLUTION_LADDER[configuration]


def sphere_radii(litho_layers=2):
    """Every sphere in the ladder, ascending, with the subdivision spheres.

    Returned separately from `generate` so the validator can predict shell
    volumes without reading the mesh.
    """
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
    """Reference volumes and areas, for the validation prints."""
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
    """A sphere's radius from its own area.  Exact for an OCC sphere."""
    return np.sqrt(gmsh.model.occ.getMass(2, tag) / (4 * np.pi))


def generate(filename="selfgrav_sphere.msh", configuration="coarse",
             h=None, grade=2.0, litho_layers=None,
             min_cells_per_great_circle=None,
             verbose=False, quality=False):
    """Writes the four-region sphere and returns (filename, shells, stats)."""
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

    balls = [occ.addSphere(0, 0, 0, R) for R in radii]
    occ.fragment([(3, balls[0])], [(3, b) for b in balls[1:]])
    occ.synchronize()

    # Remove the unmeshed core, keeping its bounding surface: that sphere is
    # the domain's inner boundary and carries the interior DtN.
    core = [t for _, t in gmsh.model.getEntities(3)
            if abs(occ.getMass(3, t) - 4 / 3 * np.pi * radii[0]**3) < 1e-8]
    assert len(core) == 1, f"expected one core ball, found {core}"
    occ.remove([(3, core[0])], recursive=False)
    occ.synchronize()

    # Identify every entity by its own measure, never by gmsh's ordering.
    # A radius recovered from an area is never bit-identical to the ladder
    # value that produced it, so it is snapped back before being used as a key.
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

    # Surface groups: the four DtN/interface spheres, plus the three density
    # interfaces (A3 evaluates |g_0| on those; see DENSITY_INTERFACE_TAGS).
    # The **buffer and subdivision spheres deliberately get none**, and nothing
    # is lost by that: what DG0 conformity needs is mesh *nodes* on the sphere,
    # which the fragment supplies whether or not the surface is written to
    # file.  A physical group is only needed to integrate over the surface.
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

    # Size field.  `Min` is a merge point: further fields go in the list.
    r = "sqrt(x*x+y*y+z*z)"
    graded = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(
        graded, "F", f"{h}*(1 + {grade}*max(0, {r} - {RE})"
                     f" + {grade}*max(0, {RC} - {r}))")
    # Angular cap, and it is not optional: both DtN spheres must resolve the
    # truncation degree.  The grading alone leaves the inner sphere at 0.5 Rc a
    # ~70-triangle polyhedron with 8% area error, because it is small and the
    # field coarsens towards it; the outer sphere at 2 Re is coarsened for the
    # same reason but starts large enough to survive it.  Requiring
    # `min_cells_per_great_circle` cells around any sphere of radius r means
    # lc <= 2 pi r / N, which for N = 8 (L + 1) at L = 5 is eight cells per
    # wavelength of the highest treated degree.
    fields = [graded]
    if min_cells_per_great_circle:
        capped = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            capped, "F", f"{2 * np.pi / min_cells_per_great_circle}*{r}")
        fields.append(capped)
    merged = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(merged, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(merged)
    # Without these three, gmsh's own curvature and point heuristics override
    # the field near the small inner sphere and refine it by an order.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # A meshing failure must still finalize, or gmsh stays initialised with a
    # dirty model and the *next* `generate` silently returns an empty mesh
    # instead of raising -- which reads as a generator bug rather than as the
    # earlier failure it actually is.
    try:
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
    """Per-shell tetrahedron quality, gmsh's own measures.

    `gamma` is the inscribed/circumscribed radius ratio scaled so that an
    equilateral tetrahedron is 1 and a degenerate one 0; `minSICN` is the
    signed inverse condition number, negative for an inverted element.  Both
    are reported because they say different things: the subdivision spheres
    make the lithosphere deliberately anisotropic, which ruins `gamma` while
    leaving `minSICN` respectable, and only an inverted element is a defect.
    """
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", default="coarse",
                    choices=list(CONFIGURATIONS))
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--litho-layers", type=int, default=2)
    ap.add_argument("--output", default="selfgrav_sphere.msh")
    args = ap.parse_args()

    name, layout, stats = generate(
        args.output, args.configuration, args.h, args.grade,
        args.litho_layers, verbose=True, quality=True)
    ref = analytic(args.litho_layers)
    print(f"\n{name}: h = {stats['h']:.6f} "
          f"({stats['h'] * D_KM:.0f} km), grade = {stats['grade']}")
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
    per = {t: sum(n for (a, b), n in stats["cells"].items()
                  if dict(((x, y), z) for x, y, z in layout)[(a, b)] == t)
           for t in (CELL_MANTLE, CELL_INNER, CELL_BUFFER)}
    print(f"\n  mantle {per[CELL_MANTLE]:8d}")
    print(f"  inner  {per[CELL_INNER]:8d}  "
          f"({per[CELL_INNER] / per[CELL_MANTLE]:.3f} of mantle)")
    print(f"  buffer {per[CELL_BUFFER]:8d}  "
          f"({per[CELL_BUFFER] / per[CELL_MANTLE]:.3f} of mantle)")
    print(f"  total  {sum(per.values()):8d}")
    print(f"\n  mantle volume {ref['vol_mantle']:.6f}, "
          f"buffer volume {ref['vol_buffer']:.6f}")
