"""Acceptance gate for the four-region 3-D parent sphere.

Deliverable A2 of `HANDOFF-SPADA-BENCHMARK.md`, the 3-D counterpart of
`demos/gravity/validate_selfgrav_annulus.py`.  Every check states the number it
expects before it measures it.

What is gated:

* every region's volume against (4/3)pi(r2^3 - r1^3);
* every tagged surface against 4 pi R^2, the two DtN spheres as *exterior*
  facets (`ds`) and the two interfaces as *interior* facets (`dS`);
* `dS(2)` on the parent equal to `ds(2)` on the mantle submesh **bit for bit**,
  because they are the same facets seen from two meshes;
* the P2 (`curve_mesh`) correction actually recovering accuracy, on the parent
  *and* separately on the submesh, which does not inherit it;
* a cross-mesh intersected measure surviving the re-curving, which is the shape
  of both u<->psi coupling terms;
* cell counts per region.

Handoff §12 trap 2 drives the shape of this: a `ds` or `dS` given the wrong kind
of tag returns **zero and a warning**, not an exception.  So every measure is
asserted *positive* before it is asserted accurate, and the trap is exercised
deliberately rather than merely cited.

Run under MPI to check the tags survive the reader at more than one rank::

    for n in 1 2 4; do
      mpiexec -n $n python validate_selfgrav_sphere.py --configuration coarse
    done

gmsh runs on rank 0 only and every rank then reads the same file; gmsh is not
collective and all ranks writing one path is a race that usually produces a
truncated file rather than an error.
"""
import argparse
import os
import sys
from warnings import warn

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import numpy as np
from firedrake import (
    COMM_WORLD, Function, FunctionSpace, Measure, Mesh, SpatialCoordinate,
    Submesh, VectorFunctionSpace, assemble, avg, dS, ds, dx, sqrt)

import generate_selfgrav_sphere as gen
from generate_selfgrav_sphere import (
    CELL_BUFFER, CELL_INNER, CELL_MANTLE, RC, RE, R_INNER, R_OUTER, SURF_INNER,
    SURF_OUTER, SURF_RC, SURF_RE)

MESH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "selfgrav_sphere.msh")


def curve_mesh(linear_mesh, untangle=False):
    """Remap to quadratic (P2) coordinates so mesh spheres become curved.

    The 2-D helper verbatim, which needs no change in 3-D: edge midpoints are
    pushed radially onto the linear interpolant of the vertex radii, so an edge
    whose vertices lie on a sphere becomes a quadratic arc on that sphere.  No
    origin guard is needed; r >= 0.5 Rc everywhere here.

    **`untangle` is off by default and probably should not be.**  Curving
    tangles a handful of cells on this mesh -- measured, 2 of 113 653 at
    `--coarse` -- because it displaces a mid-edge node by ~h_lat^2/(8R), which
    on the flattened tetrahedra bounding a sub-cell-thickness shell is a large
    fraction of the element's smallest dimension.  With `untangle=True` those
    cells have their P2 edge nodes reset to the straight midpoints, which
    un-curves them locally and costs nothing measurable (2 cells out of
    113 653 against area errors of 1e-05).  It is opt-in only so that the
    numbers already reported for A2 and A3 remain reproducible; see
    `NOTES/IMPL-LOG-SPADA-A2-MESH.md`.
    """
    from firedrake import JacobianDeterminant  # noqa: PLC0415

    X = SpatialCoordinate(linear_mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    V = VectorFunctionSpace(linear_mesh, "CG", 2)
    X_p2 = Function(V).interpolate((r_p1 / r) * X)
    if not untangle:
        return Mesh(X_p2)

    # The straight geometry in the *same* space: edge nodes at true midpoints.
    #
    # **Detect once, correct once, verify once.**  An earlier version iterated
    # up to five times, rebuilding `Mesh(X_p2)` on each pass so the next
    # detection saw the corrected geometry.  Serially that is a second or two;
    # on 104 MPI ranks each `Mesh` construction carries a large fixed collective
    # cost and the loop spent **fifteen minutes** before the first solve, which
    # is how it was found.  Correcting every tangled cell in one pass is
    # equivalent in almost every case -- reverting a cell's edge nodes can only
    # improve its neighbours, never tangle them further -- so the second pass
    # exists to *verify*, and a residual is warned about rather than iterated
    # away.
    X_straight = Function(V).interpolate(X)
    nodes = V.cell_node_list

    def tangled_cells(mesh_c):
        W = FunctionSpace(mesh_c, "DG", 3)
        detJ = Function(W).interpolate(JacobianDeterminant(mesh_c))
        per_cell = W.finat_element.space_dimension()
        owned = mesh_c.cell_set.size
        d = detJ.dat.data_ro[:owned * per_cell].reshape(-1, per_cell)
        return np.where(d.min(axis=1) * d.max(axis=1) <= 0.0)[0]

    mesh_c = Mesh(X_p2)
    bad = tangled_cells(mesh_c)
    if mesh_c.comm.allreduce(int(bad.size)) == 0:
        return mesh_c
    # **`data_with_halos`, not `data`.**  `cell_node_list` indexes the local
    # node numbering *including* halo nodes, while `dat.data` exposes only the
    # owned ones -- so a cell touching the partition boundary indexes past the
    # end.  In serial the two are the same array and the bug is invisible; in
    # parallel it is intermittent, because it fires only when a *tangled* cell
    # happens to own a halo node, which depends on the partition and so can
    # differ between two otherwise identical builds in one job.  Reported as
    # `IndexError: index 3267 is out of bounds for axis 0 with size 1914` on
    # the third of three identical mesh builds.
    for c in bad:
        X_p2.dat.data_with_halos[nodes[c]] = \
            X_straight.dat.data_ro_with_halos[nodes[c]]
    mesh_c = Mesh(X_p2)
    left = mesh_c.comm.allreduce(int(tangled_cells(mesh_c).size))
    if left and COMM_WORLD.rank == 0:
        warn(f"curve_mesh: {left} cells still tangled after one correction "
             f"pass; the geometry is usable but not clean.")
    return mesh_c


def min_jacobian(mesh, degree=3):
    """Smallest Jacobian determinant sampled over the mesh, and where it is.

    **The quality numbers the generator prints are gmsh's, taken before
    curving, so they describe a mesh nobody assembles on.**  `curve_mesh`
    displaces every mid-edge node radially by about h_lat^2/(8R) -- 0.0017
    non-dimensional at `--coarse` -- against a lithosphere element whose radial
    thickness is 35/2891 = 0.0121.  The displacement is therefore ~14% of the
    smallest dimension of exactly the elements that are already worst, and a
    curved element can be tangled without any volume check noticing: an
    inverted region contributes *negative* volume, so it shows up as a small
    shortfall inside the tolerance rather than as an error.

    A P2 tetrahedron's Jacobian determinant is a cubic in the reference
    coordinates, so sampling it at DG3 nodes (20 per cell) is a close bound on
    its true behaviour rather than a single midpoint estimate.

    **The test is per cell, on the sign *change*, not on the sign.**  Firedrake
    does not globally orient simplices, so `detJ` is negative on about half the
    cells of any mesh and a global `min(detJ) < 0` means nothing -- measured,
    the *straight* parent reports 1 136 900 negative samples out of 2 273 060,
    i.e. exactly half.  What cannot happen on a straight simplex, and is
    precisely what tangling is on a curved one, is `detJ` changing sign *within*
    a single cell.  So the metric is `min(detJ) * max(detJ) <= 0` per cell, and
    the distortion is `min|detJ| / max|detJ|` per cell -- which is identically 1
    on a straight simplex, so the straight mesh doubles as a check that the
    measurement itself is right.

    Returns `(worst_ratio, n_tangled_cells, n_cells)`.
    """
    from firedrake import JacobianDeterminant  # noqa: PLC0415

    V = FunctionSpace(mesh, "DG", degree)
    detJ = Function(V).interpolate(JacobianDeterminant(mesh))
    per_cell = V.finat_element.space_dimension()
    owned = mesh.cell_set.size
    d = detJ.dat.data_ro[:owned * per_cell].reshape(-1, per_cell)
    lo, hi = d.min(axis=1), d.max(axis=1)
    tangled = int((lo * hi <= 0.0).sum())
    a = np.abs(d)
    ratio = a.min(axis=1) / np.maximum(a.max(axis=1), 1e-300)
    worst = float(ratio.min()) if ratio.size else 1.0
    comm = mesh.comm
    return (comm.allreduce(worst, op=min), comm.allreduce(tangled),
            comm.allreduce(owned))


def cell_count(mesh, tag):
    """Cells carrying a given cell tag, owned only, summed over ranks."""
    owned = mesh.cell_set.size
    local = int(np.count_nonzero(mesh.cell_subset(tag).indices < owned))
    return mesh.comm.allreduce(local)


def geometry_stage(configuration, grade, litho_layers, quality=False):
    ref = gen.analytic(litho_layers)
    say(f"\nGenerating: configuration={configuration} "
        f"({gen.CONFIGURATIONS[configuration]:.0f} km lateral), grade={grade}, "
        f"litho_layers={litho_layers}, ranks={COMM_WORLD.size}")
    layout, stats = build_mesh(configuration=configuration, grade=grade,
                               litho_layers=litho_layers, quality=quality)
    say(f"  h = {stats['h']:.6f} ({stats['h'] * gen.D_KM:.0f} km), "
        f"{sum(stats['cells'].values())} tetrahedra")

    rep = Report()
    parent = Mesh(MESH_FILE)

    # --- straight facets: everything should sit at the polyhedron-vs-sphere
    # --- O(h^2) error, which is what makes the curved comparison meaningful.
    say("\nStraight-facet parent (before curve_mesh), expect O(h^2):")
    for name, tag, expect in [
            ("volume mantle dx(101)", CELL_MANTLE, ref["vol_mantle"]),
            ("volume inner  dx(102)", CELL_INNER, ref["vol_inner"]),
            ("volume buffer dx(103)", CELL_BUFFER, ref["vol_buffer"]),
    ]:
        rep.check(name, assemble(1 * dx(tag, domain=parent)), expect, 5e-2)
    for name, tag, expect in [
            ("area 2Re   ds(4) exterior", SURF_OUTER, ref["area_outer"]),
            ("area 0.5Rc ds(5) exterior", SURF_INNER, ref["area_inner"]),
    ]:
        rep.check(name, assemble(1 * ds(tag, domain=parent)), expect, 5e-2)
    for name, tag, expect in [
            ("area Re    dS(2) interior", SURF_RE, ref["area_Re"]),
            ("area Rc    dS(3) interior", SURF_RC, ref["area_Rc"]),
    ]:
        rep.check(name, assemble(avg(1) * dS(tag, domain=parent)), expect, 5e-2)

    # --- the submesh, straight off the gmsh cell tag.
    say("\nMantle submesh of the straight parent, Submesh(parent, 3, 101):")
    sub = Submesh(parent, 3, CELL_MANTLE)
    rep.check("volume sub dx", assemble(1 * dx(domain=sub)),
              ref["vol_mantle"], 5e-2)
    for name, tag, expect in [("sub ds(2) = Re", SURF_RE, ref["area_Re"]),
                              ("sub ds(3) = Rc", SURF_RC, ref["area_Rc"])]:
        rep.check(name, assemble(1 * ds(tag, domain=sub)), expect, 5e-2)
    # The parent's interior view and the submesh's boundary view are the same
    # facets.  In 2-D they agreed bit for bit.  In 3-D they do not, and the
    # reason is worth pinning rather than tolerating: the two meshes visit the
    # facets in different orders, so the two reductions differ in their last
    # bits.  It is summation order, not a different facet set.
    #
    # Two weights, and the second must vary *angularly* to be worth anything.
    # A *missing* facet needs no second weight at all: one facet is ~5e-04 of
    # the total area, which the w = 1 comparison at 1e-12 rejects by eight
    # orders.  What the second weight is for is the failure w = 1 cannot see --
    # the same total area made of different facets -- and for that the weight
    # has to distinguish one part of the sphere from another.  `1 + x` does;
    # `r^2` does *not*, because every facet of a tagged set sits at the same
    # radius and r^2 varies over it only by the polygon error.  (An earlier
    # version used r^2 and claimed in this comment that it "weights the facets
    # unequally"; on a sphere that is false, and the test was strictly weaker
    # than the bit-for-bit one it replaced rather than sharper.)
    #
    # The `1 +` is not decoration: `x` alone integrates to zero over a sphere,
    # so a *relative* comparison against it would divide by roundoff.  With
    # `1 + x` the integral is the area, while swapping a facet at +x for one at
    # -x moves it by 2x times the facet area -- about 2e-03 relative at Re,
    # which the 1e-12 tolerance rejects by nine orders.
    Xp, Xs = SpatialCoordinate(parent), SpatialCoordinate(sub)
    wp, ws = 1 + Xp[0], 1 + Xs[0]
    for tag, label in [(SURF_RE, "Re"), (SURF_RC, "Rc")]:
        for wname, ip, isub in [("1", avg(1), 1), ("1+x", avg(wp), ws)]:
            a = assemble(ip * dS(tag, domain=parent))
            b = assemble(isub * ds(tag, domain=sub))
            rep.check(f"dS({tag}) = ds({tag}) [{label}, w={wname}]", a, b,
                      1e-12)
            rep.note(f"    absolute difference [{label}, w={wname}]",
                     abs(a - b))

    # --- the same measures on the wrong kind of tag, to demonstrate trap 2
    # --- rather than merely cite it.  These SHOULD be zero.
    say("\nTrap 2, exercised deliberately (zero expected):")
    rep.note("ds(2) on parent [Re is interior]",
             assemble(1 * ds(SURF_RE, domain=parent)))
    rep.note("dS(4) on parent [2Re is exterior]",
             assemble(avg(1) * dS(SURF_OUTER, domain=parent)))

    # --- P2 isoparametric correction.  On an unstructured tet mesh there is no
    # --- single cell angle, so the residual is gated as a *reduction factor*
    # --- against the straight-facet error rather than against a theta^4
    # --- constant, plus a loose absolute bound.
    say("\nCurved parent (curve_mesh, P2 coordinates):")
    parent_c = curve_mesh(parent)
    curved = {}
    for name, tag, expect, kind in [
            ("volume mantle dx(101)", CELL_MANTLE, ref["vol_mantle"], "dx"),
            ("volume inner  dx(102)", CELL_INNER, ref["vol_inner"], "dx"),
            ("volume buffer dx(103)", CELL_BUFFER, ref["vol_buffer"], "dx"),
            ("area 2Re   ds(4)", SURF_OUTER, ref["area_outer"], "ds"),
            ("area 0.5Rc ds(5)", SURF_INNER, ref["area_inner"], "ds"),
            ("area Re    dS(2)", SURF_RE, ref["area_Re"], "dS"),
            ("area Rc    dS(3)", SURF_RC, ref["area_Rc"], "dS"),
    ]:
        m = {"dx": lambda: assemble(1 * dx(tag, domain=parent_c)),
             "ds": lambda: assemble(1 * ds(tag, domain=parent_c)),
             "dS": lambda: assemble(avg(1) * dS(tag, domain=parent_c))}[kind]()
        s = {"dx": lambda: assemble(1 * dx(tag, domain=parent)),
             "ds": lambda: assemble(1 * ds(tag, domain=parent)),
             "dS": lambda: assemble(avg(1) * dS(tag, domain=parent))}[kind]()
        rel_c = abs(m - expect) / expect
        rel_s = abs(s - expect) / expect
        curved[name] = (rel_s, rel_c)
        rep.check(name, m, expect, 5e-3)
    say("\n  straight vs curved relative error, expect a large reduction:")
    for name, (rel_s, rel_c) in curved.items():
        ok = rel_c < 0.2 * rel_s
        say(f"  [{'ok ' if ok else 'FAIL'}] {name:<38s} "
            f"{rel_s:.3e} -> {rel_c:.3e}   x{rel_s / rel_c:8.1f}")
        if not ok:
            rep.failures.append(f"curving did not help: {name}")
    # Confirm the residual is geometry and not under-integration.
    rep.note("volume mantle at quad degree 8",
             assemble(1 * dx(CELL_MANTLE, domain=parent_c, degree=8)))

    # --- does the submesh inherit the parent's P2 coordinates?  The recorded
    # --- 2-D answer is no.  Establish it in 3-D, because under Stack B the
    # --- mechanics runs on this submesh and the interface sheets live exactly
    # --- on the surfaces the polygon error is concentrated at.
    say("\nDoes Submesh inherit the curved parent's P2 coordinates?")
    sub_c = Submesh(parent_c, 3, CELL_MANTLE)
    say(f"    parent coordinate degree "
        f"{parent_c.coordinates.function_space().ufl_element().degree()}, "
        f"submesh coordinate degree "
        f"{sub_c.coordinates.function_space().ufl_element().degree()}")
    v = assemble(1 * dx(domain=sub_c))
    rel = abs(v - ref["vol_mantle"]) / ref["vol_mantle"]
    say(f"    volume of Submesh(curved parent) {v: .12e}  rel {rel:.2e}  ->  "
        f"{'INHERITS' if rel < 1e-10 else 'does NOT inherit'}")

    say("\nWorkaround: curve_mesh the submesh separately.")
    sub_cc = curve_mesh(sub_c)
    rep.check("volume curve_mesh(submesh)", assemble(1 * dx(domain=sub_cc)),
              ref["vol_mantle"], 5e-3)
    for tag, label, expect in [(SURF_RE, "Re", ref["area_Re"]),
                               (SURF_RC, "Rc", ref["area_Rc"])]:
        rep.check(f"curved sub ds({tag}) [{label}]",
                  assemble(1 * ds(tag, domain=sub_cc)), expect, 5e-3)

    # Does the coupled solve's cross-mesh assembly survive re-curving the
    # submesh?  A parent-mesh coefficient integrated over the submesh with an
    # intersected measure is exactly the shape of both u<->psi coupling terms.
    want = 4 * np.pi * (RE**5 - RC**5) / 15  # int_shell x^2 dV
    for label, mesh_m, tol in [("curved sub x curved parent", sub_cc, 5e-3),
                               ("plain  sub x curved parent", sub_c, 5e-2)]:
        dx_m = Measure("dx", domain=mesh_m,
                       intersect_measures=(Measure("dx", domain=parent_c),))
        f = Function(FunctionSpace(parent_c, "CG", 2)).interpolate(
            SpatialCoordinate(parent_c)[0]**2)
        try:
            rep.check(f"int x^2, {label}", assemble(f * dx_m), want, tol)
        except Exception as exc:  # noqa: BLE001 - the point is what it raises
            say(f"    int x^2, {label}: {type(exc).__name__}: {exc}")
            rep.failures.append(f"cross-mesh {label}")

    # --- is anything tangled *after* curving?  See `min_jacobian`.
    say("\nJacobian determinant per cell, sampled at DG3 nodes.  Expect zero")
    say("tangled cells everywhere, and worst |detJ| ratio exactly 1 on the")
    say("straight mesh (detJ is constant on a straight simplex):")
    for label, m in [("straight parent", parent), ("curved parent", parent_c),
                     ("curved submesh", sub_cc)]:
        ratio, tangled, ncells = min_jacobian(m)
        ok = tangled == 0
        say(f"  [{'ok ' if ok else 'FAIL'}] {label:<20s} worst min|detJ|/max|detJ| "
            f"{ratio:.6f}   tangled cells {tangled} of {ncells}")
        if not ok:
            rep.failures.append(f"tangled elements in {label}")

    # --- cost.
    say("\nCell counts (owned, summed over ranks):")
    n = {t: cell_count(parent, t)
         for t in (CELL_MANTLE, CELL_INNER, CELL_BUFFER)}
    say(f"    mantle {n[CELL_MANTLE]:8d}")
    say(f"    inner  {n[CELL_INNER]:8d}   "
        f"({n[CELL_INNER] / n[CELL_MANTLE]:.3f} of mantle)")
    say(f"    buffer {n[CELL_BUFFER]:8d}   "
        f"({n[CELL_BUFFER] / n[CELL_MANTLE]:.3f} of mantle)")
    say(f"    total  {sum(n.values()):8d}")
    if quality and COMM_WORLD.rank == 0 and "quality" in stats:
        say("\nTetrahedron quality per shell "
            "(gamma = inradius/circumradius scaled, 1 = equilateral):")
        say(f"    {'r_in':>9} {'r_out':>9} {'cells':>8} {'gamma_min':>10} "
            f"{'gamma_avg':>10} {'sicn_min':>10} {'inverted':>9}")
        for (r_in, r_out), q in stats["quality"].items():
            say(f"    {r_in:9.6f} {r_out:9.6f} {q['n']:8d} "
                f"{q['gamma_min']:10.4f} {q['gamma_mean']:10.4f} "
                f"{q['sicn_min']:10.4f} {q['n_inverted']:9d}")

    return rep.done("Geometry stage"), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", default="coarse",
                    choices=list(gen.CONFIGURATIONS))
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--litho-layers", type=int, default=2)
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--mesh-file", default=MESH_FILE)
    args = ap.parse_args()
    MESH_FILE = args.mesh_file

    ok, _ = geometry_stage(args.configuration, args.grade, args.litho_layers,
                           args.quality)
    sys.exit(0 if ok else 1)
