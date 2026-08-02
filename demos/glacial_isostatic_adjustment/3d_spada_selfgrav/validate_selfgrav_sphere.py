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
import hashlib
import json
import os
import subprocess
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

MESH_FILE = os.path.join(HERE, "selfgrav_sphere.msh")


def say(*a):
    """Rank-0 print.  Everything this module prints is a collective quantity."""
    if COMM_WORLD.rank == 0:
        print(*a, flush=True)


# `Report` is IMPORTED, not copied: `demos/gravity/validate_selfgrav_annulus.py`
# is where it was written and where its own gate exercises it, and two copies
# of a pass/fail bookkeeper is exactly the drift this file exists to catch.
#
# **The import is not free.** That module runs `sys.path.insert` for `passess`
# and imports `scipy.integrate` at module scope, so this 3-D geometry gate now
# depends on both even though it uses neither.  If that ever becomes a problem
# on a machine without `passess`, move `Report` to a module of its own rather
# than copying it back.
sys.path.insert(0, os.path.join(REPO, "demos", "gravity"))
import validate_selfgrav_annulus as _annulus  # noqa: E402
from validate_selfgrav_annulus import Report  # noqa: E402,F401

# `Report.check` resolves `print` in ITS OWN module's globals, so subclassing
# cannot make it rank-0 and the report would otherwise repeat itself once per
# rank -- 4 interleaved copies of a table whose every number is collective.
# One rebinding, here, where the reason for it is visible.
_annulus.print = say


def provenance(label=""):
    """Rank-0 banner naming the code that is about to run.

    Repo HEAD, `git status --porcelain`, and `gadopt.__file__`.  Two of the
    last three production failures were stale code on a remote machine that
    nobody noticed, and a log that identifies its own code state ends that
    class of failure.  `gadopt.__file__` is the one that actually bites in this
    project: the editable install points at a *different worktree on a
    different branch*, so `import gadopt` silently loads the wrong code unless
    `PYTHONPATH` wins.

    Wrapped end to end, `diagnostic`-style: a provenance banner that raises
    would take down the run it exists to document.
    """
    if COMM_WORLD.rank != 0:
        return
    say(f"\n  provenance {label}".rstrip())
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                               capture_output=True, text=True,
                               timeout=30).stdout.rstrip()
        say(f"    HEAD      {head}  ({branch})")
        if dirty:
            say("    git status --porcelain:")
            for line in dirty.splitlines():
                say(f"      {line}")
        else:
            say("    git status --porcelain: clean")
    except Exception as exc:  # noqa: BLE001 - a banner may not kill a run
        say(f"    git UNAVAILABLE ({type(exc).__name__}: {exc})")
    try:
        say(f"    gadopt    {gadopt.__file__}")
    except Exception as exc:  # noqa: BLE001
        say(f"    gadopt    UNAVAILABLE ({type(exc).__name__}: {exc})")
    say(f"    ranks     {COMM_WORLD.size}")


def build_mesh(**kwargs):
    """gmsh on rank 0 only; every rank then reads the same file.

    gmsh is not collective and all ranks writing one path is a race that
    usually produces a truncated file rather than an error.

    **Same pattern as `gate_refstate.build_mesh`, deliberately not the same
    return.**  That one hands back a curved `Mesh`; `geometry_stage` needs the
    *straight* mesh and the curved one separately, and it needs the generator's
    `(layout, stats)`, so those are broadcast instead.  If the two ever want to
    be one function, this is the signature to converge on.

    `reuse=True` skips regeneration when the file already exists.  gmsh's
    tetrahedralisation is not reproducible across builds, so any comparison
    between two runs -- rank counts, repaired versus not -- must be against one
    file, not one configuration.
    """
    reuse = kwargs.pop("reuse", False)
    out = None
    if COMM_WORLD.rank == 0:
        if reuse and os.path.exists(MESH_FILE):
            out = (None, None)      # nothing regenerated, so no stats to hand back
        else:
            _, layout, stats = gen.generate(MESH_FILE, **kwargs)
            out = (layout, stats)
    COMM_WORLD.barrier()
    return COMM_WORLD.bcast(out, root=0)


def curve_mesh(linear_mesh, untangle=False, max_passes=8):
    """Remap to quadratic (P2) coordinates so mesh spheres become curved.

    The 2-D helper verbatim, which needs no change in 3-D: edge midpoints are
    pushed radially onto the linear interpolant of the vertex radii, so an edge
    whose vertices lie on a sphere becomes a quadratic arc on that sphere.  No
    origin guard is needed; r >= 0.5 Rc everywhere here.

    **`untangle` is off by default.**  Curving tangles a handful of cells on
    this mesh -- measured, 2 of 113 653 at `--coarse` -- because it displaces a
    mid-edge node by ~h_lat^2/(8R), which on the flattened tetrahedra bounding
    a sub-cell-thickness shell is a large fraction of the element's smallest
    dimension.  **The mesh as read from gmsh is clean**: the census measures
    zero tangled cells on the straight parent, so every fold here is made by
    this function.  It is opt-in so that the numbers already reported for A2
    and A3 remain reproducible; see `NOTES/IMPL-LOG-SPADA-A2-MESH.md`.

    ## What `untangle=True` does now, and the guarantee it carries

    It marks the P2 nodes of every tangled cell, reduces the marks over the dof
    star-forest so that every rank agrees on the marked set, resets the marked
    nodes to the straight geometry, and **repeats to a fixed point**.

    Two things make that terminate and terminate clean, and both are worth
    stating because the version this replaces claimed a guarantee it did not
    have ("reverting a cell's edge nodes can only improve its neighbours, never
    tangle them further" -- measured false; the single pass straightened its
    two targets and folded two neighbours ~400x worse in |detJ|):

    1. **Termination.**  A marked node is never unmarked -- `mark` persists
       across passes and the reset is re-applied in full each time -- so the
       straightened set grows monotonically and is bounded by the node count.
       Any pass that finds a tangled cell must mark at least one *new* node,
       because a cell all of whose nodes are already straight cannot be tangled
       (see 2).  So the marked set grows strictly while work remains.
    2. **The fixed point is tangle-free by construction.**  At the fixed point
       every tangled cell would have all nodes straight; an all-straight
       tetrahedron is an affine map with constant `detJ`, so it is tangled only
       if the straight mesh was.  That premise is *checked*, not assumed:
       the straight geometry is censused first and this raises if it is dirty,
       because the guarantee dies with it.

    Measured cost: 2 to 4 passes, each spreading one ring outward.  A pass is
    one `Mesh` rebuild plus one DG3 `detJ` interpolation, which is affordable at
    build time now that the per-cell collective-call defect is gone.

    **A residual tangle is a hard error, never a warning.**  A warned residual
    is exactly how "untangled" came to mean "differently tangled" across this
    project's record.

    **This is a mitigation, not the endpoint.**  It produces straight-sided
    patches, and it produces them at exactly the layer interfaces where |g_0|
    jumps -- measured, the folds sit at r = 1.971979 (the 1.971982 density
    interface) and at r = 2.191628 (inside the lithosphere).  Straightening
    there trades an inverted cell for an O(h^2) geometry error on the surface
    that carries the density contrast.  The route that does not make that trade
    is to stop the generator producing foldable cells in the first place.
    """
    X = SpatialCoordinate(linear_mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    V = VectorFunctionSpace(linear_mesh, "CG", 2)
    X_p2 = Function(V).interpolate((r_p1 / r) * X)
    if not untangle:
        return Mesh(X_p2)

    from pyop2 import op2  # noqa: PLC0415

    # The straight geometry in the *same* space: edge nodes at true midpoints.
    X_straight = Function(V).interpolate(X)

    # The premise, checked rather than assumed.  Everything below rests on an
    # all-straight cell not being tangled; if the file itself is folded that is
    # false and the iteration would spend `max_passes` straightening the whole
    # mesh and then raise anyway, with a misleading message.
    straight_bad, straight_owned = tangled_cells(Mesh(X_straight))
    n_straight = linear_mesh.comm.allreduce(
        int((straight_bad < straight_owned).sum()))
    if n_straight:
        raise RuntimeError(
            f"curve_mesh(untangle=True): the STRAIGHT geometry already has "
            f"{n_straight} tangled cells. The fixed-point argument requires "
            "an all-straight cell to be untangled, so it does not hold on this "
            "mesh and no amount of straightening will converge. Fix the mesh.")

    # `mark` is scalar CG2 on the same mesh, so it shares the vector space's
    # node numbering exactly -- verified: `cell_node_list` is identical and
    # `dof_dset.total_size` matches.  One scalar per node, 0 or 1, monotone.
    Vs = FunctionSpace(linear_mesh, "CG", 2)
    mark = Function(Vs)
    mark.dat.data_with_halos[:] = 0.0
    nodes = Vs.cell_node_list

    for npass in range(1, max_passes + 1):
        mesh_c = Mesh(X_p2)
        bad, owned = tangled_cells(mesh_c)
        n_tangled = mesh_c.comm.allreduce(int((bad < owned).sum()))
        n_marked = mesh_c.comm.allreduce(
            int((mark.dat.data_ro_with_halos[:Vs.dof_dset.size] > 0.5).sum()))
        say(f"  curve_mesh(untangle=True): pass {npass}: {n_tangled} tangled "
            f"cells, {n_marked} nodes already straight  "
            f"[{mesh_c.comm.size} ranks]")
        if n_tangled == 0:
            return mesh_c

        # Mark the nodes of the tangled cells this rank OWNS.  Owned-only is
        # enough: every cell is owned by exactly one rank, and the reduction
        # below carries the mark to whichever rank owns each node -- including
        # a rank that does not hold the cell at all, which is the case the
        # cell-halo approach could not reach.  Measured there: 3 of 20 nodes of
        # the two folded cells were owned by a rank holding neither cell, so
        # the next exchange restored them to the curved value.
        mine = bad[bad < owned]
        m_local = mark.dat.data_with_halos          # collective: unconditional
        if mine.size:
            m_local[nodes[mine].ravel()] = 1.0
        del m_local
        # Ghost marks -> owners, then owners -> ghosts.  `local_to_global`
        # supports INC, MIN and MAX in this Firedrake (`firedrake/halo.py`
        # asserts exactly that set); MAX is the natural op for a 0/1 mark and
        # is idempotent, so a node marked twice is marked once.  INC on the
        # same marks with a `> 0` test would be equivalent.
        mark.dat.local_to_global_begin(op2.MAX)
        mark.dat.local_to_global_end(op2.MAX)
        sel = mark.dat.data_ro_with_halos > 0.5    # collective: broadcasts back

        # Re-applied in FULL every pass, not incrementally: that is what makes
        # a straightened node stay straight, which is what makes the marked set
        # monotone, which is what makes this terminate.
        x_write = X_p2.dat.data_with_halos
        x_read = X_straight.dat.data_ro_with_halos
        x_write[sel] = x_read[sel]
        del x_write, x_read

    raise RuntimeError(
        f"curve_mesh(untangle=True): still tangled after {max_passes} passes. "
        "Expected 2 to 4. Either the mesh is far worse than the ladder's, or "
        "the fixed-point argument has been broken by an edit -- check that the "
        "reset is re-applied in full each pass and that `mark` is never "
        "cleared. This is deliberately an error and not a warning: a warned "
        "residual is how 'untangled' came to mean 'differently tangled'.")


def tangled_cells(mesh_c, degree=3):
    """Sign-changing cells of `mesh_c`, over ALL local cells INCLUDING halos.

    Returns `(bad, owned)`: `bad` indexes `mesh_c.cell_set`'s local cell
    numbering over its `total_size` cells, and `owned = mesh_c.cell_set.size`,
    so `bad[bad < owned]` is the owned subset and `(bad < owned).sum()` summed
    over ranks is the global count with no double counting.

    **Reading detJ with halos is what makes a repair built on this
    rank-consistent.**  `data_ro_with_halos` performs the forward halo exchange
    (`pyop2/types/dat.py`: `global_to_local_begin/end` fire whenever
    `halo_valid` is false), so a halo cell is judged on the *owner's* samples
    rather than on a locally recomputed copy, and every rank holding a cell
    reaches the same verdict from the same numbers.

    See `min_jacobian` for why the test is `min(detJ) * max(detJ) <= 0` per
    cell and not a sign test: Firedrake does not globally orient simplices, so
    half of any mesh has negative detJ and a global minimum means nothing.
    """
    from firedrake import JacobianDeterminant  # noqa: PLC0415

    W = FunctionSpace(mesh_c, "DG", degree)
    detJ = Function(W).interpolate(JacobianDeterminant(mesh_c))
    # Indexed through the map rather than by slicing the flat array, so the
    # rows for halo cells are found wherever the DG numbering actually puts
    # them instead of being assumed contiguous with the owned ones.
    d = detJ.dat.data_ro_with_halos[W.cell_node_list]
    return (np.where(d.min(axis=1) * d.max(axis=1) <= 0.0)[0],
            mesh_c.cell_set.size)


def illconditioned_on_tagged_facets(mesh, threshold=None, degree=3):
    """Do the ill-conditioned cells actually own a facet on a tagged surface?

    **This decides whether the distorted ring contaminates the sheet terms or
    merely sits beside them, and it is measurable rather than arguable.**  The
    two worst cells after untangling sit at r = 2.1810 and 2.1818, close to the
    interfaces the mass sheets live on -- but "close to" is not "on", and the
    sheet integrals are facet integrals.  A cell that owns no tagged facet
    contributes to `dx` only.

    Returns `(n_illconditioned, n_of_those_on_a_tagged_facet, {tag: count})`.
    """
    from firedrake import JacobianDeterminant  # noqa: PLC0415

    # Resolved at call time, not as a default argument: the threshold is
    # defined below with the panel it belongs to, and a default argument would
    # bind it at import.
    threshold = (MIN_JAC_RATIO_FOR_GATING if threshold is None else threshold)
    W = FunctionSpace(mesh, "DG", degree)
    d = Function(W).interpolate(
        JacobianDeterminant(mesh)).dat.data_ro_with_halos[W.cell_node_list]
    a = np.abs(d)
    ratio = a.min(axis=1) / np.maximum(a.max(axis=1), 1e-300)
    owned = mesh.cell_set.size
    ill = set(int(c) for c in np.where(ratio < threshold)[0] if c < owned)

    per_tag, on_tag = {}, set()
    for kind in ("exterior_facets", "interior_facets"):
        fs = getattr(mesh, kind, None)
        if fs is None:
            continue
        try:
            markers = np.asarray(fs.markers)
        except Exception:  # noqa: BLE001 - a mesh may carry neither
            continue
        for tag in (SURF_RE, SURF_RC, SURF_OUTER, SURF_INNER):
            if tag not in markers:
                continue
            try:
                idx = fs.subset(int(tag)).indices
                cells = np.atleast_2d(fs.facet_cell[idx]).ravel()
            except Exception:  # noqa: BLE001
                continue
            hit = ill.intersection(int(c) for c in cells if c < owned)
            if hit:
                per_tag[int(tag)] = per_tag.get(int(tag), 0) + len(hit)
                on_tag |= hit
    comm = mesh.comm
    return (comm.allreduce(len(ill)), comm.allreduce(len(on_tag)),
            {t: comm.allreduce(n) for t, n in per_tag.items()})


def tangle_census(mesh, label, degree=3):
    """Print how many cells are tangled and at what radii, owned only.

    The repaired arm of a mesh A/B must show zero and the unrepaired arm at
    least one.  If both show the same number the A/B is void, and the log
    should say so without anyone having to reason about which flag reached
    which call.

    Returns `(n_tangled, radii)` with `radii` the tangled cells' centroid radii
    gathered across ranks (identical on every rank).
    """
    bad, owned = tangled_cells(mesh, degree=degree)
    mine = bad[bad < owned]
    X = SpatialCoordinate(mesh)
    R0 = FunctionSpace(mesh, "DG", 0)
    r = Function(R0).interpolate(sqrt(X[0]**2 + X[1]**2 + X[2]**2))
    # `data_ro_with_halos` is `@mpi.collective`, so it is fetched here rather
    # than inside the conditional below.  Reaching for it only on the ranks
    # that have a tangled cell deadlocks -- which is what the first version of
    # this function did, and it hung a 4-rank run for seven minutes before it
    # was noticed.  Same hazard as the loop in `curve_mesh`; it is easy to
    # write and gives no error when it fires.
    rdat = r.dat.data_ro_with_halos
    local = (rdat[R0.cell_node_list[mine].ravel()]
             if mine.size else np.zeros(0))
    radii = np.concatenate([np.asarray(a, dtype=float)
                            for a in mesh.comm.allgather(local)])
    n = int(radii.size)
    if n == 0:
        say(f"  tangle census [{label}]: 0 tangled cells "
            f"of {mesh.comm.allreduce(owned)}")
    else:
        u = np.sort(radii)
        say(f"  tangle census [{label}]: {n} tangled cells of "
            f"{mesh.comm.allreduce(owned)}; centroid radii "
            + ", ".join(f"{v:.6f}" for v in u[:12])
            + (" ..." if n > 12 else "")
            + f"   [min {u.min():.6f} max {u.max():.6f}]")
    return n, radii


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


def geometry_stage(configuration, grade, litho_layers, quality=False,
                   reuse=False, allow_tangled=False):
    ref = gen.analytic(litho_layers)
    say(f"\nGenerating: configuration={configuration} "
        f"({gen.CONFIGURATIONS[configuration]:.0f} km lateral), grade={grade}, "
        f"litho_layers={litho_layers}, ranks={COMM_WORLD.size}")
    # `reuse` matters for the multi-rank run and is not a convenience: gmsh's
    # tetrahedralisation is not reproducible across builds, so regenerating
    # between rank counts compares two meshes and calls the difference a
    # partition effect.
    layout, stats = build_mesh(configuration=configuration, grade=grade,
                               litho_layers=litho_layers, quality=quality,
                               reuse=reuse)
    if stats is None:
        say(f"  reusing {MESH_FILE}; no generator statistics this run")
    else:
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
        # `allow_tangled` demotes this ONE check so that a deliberately-folded
        # stage-1 arm can exercise the other twenty. It does not change what is
        # measured or printed, and it is not the default: an unflagged run
        # still exits 1 on a folded mesh, which is the behaviour that made this
        # gate worth restoring.
        flag = "ok " if ok else ("note" if allow_tangled else "FAIL")
        say(f"  [{flag}] {label:<20s} worst min|detJ|/max|detJ| "
            f"{ratio:.6f}   tangled cells {tangled} of {ncells}")
        if not ok and not allow_tangled:
            rep.failures.append(f"tangled elements in {label}")

    # Where they sit, not just how many: a tangled cell at the deep interface
    # and one in the lithosphere have different consequences for a per-degree
    # error profile, and the count alone cannot tell them apart.
    say("")
    for label, m in [("curved parent", parent_c), ("curved submesh", sub_cc)]:
        tangle_census(m, label)

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
    if quality and COMM_WORLD.rank == 0 and stats and "quality" in stats:
        say("\nTetrahedron quality per shell "
            "(gamma = inradius/circumradius scaled, 1 = equilateral):")
        say(f"    {'r_in':>9} {'r_out':>9} {'cells':>8} {'gamma_min':>10} "
            f"{'gamma_avg':>10} {'sicn_min':>10} {'inverted':>9}")
        for (r_in, r_out), q in stats["quality"].items():
            say(f"    {r_in:9.6f} {r_out:9.6f} {q['n']:8d} "
                f"{q['gamma_min']:10.4f} {q['gamma_mean']:10.4f} "
                f"{q['sicn_min']:10.4f} {q['n_inverted']:9d}")

    return rep.done("Geometry stage"), n


def coordinate_rows(mesh):
    """Owned coordinate rows, gathered and sorted lexicographically on rank 0.

    **The geometry, measured directly instead of through an operator.**  The
    round-one panel asserted integrals of `|detJ|` to 1e-12 and could not be
    met: on a cell whose `detJ` changes sign the integrand is non-smooth, so
    the integral pins the *position of the sign crossing* to roundoff, which
    nothing guarantees.  The repair, by contrast, is a deterministic function
    of the input coordinates, so the rank count must not change the output
    coordinates **at all** -- and the coordinate multiset is partition-
    independent by construction, since every node is owned exactly once.

    Sorted lexicographically because the *numbering* is partition-dependent
    while the multiset is not.  Ties are broken by the later columns, and exact
    duplicate rows are impossible in a valid mesh.

    Returns the `(nnodes, 3)` sorted array on rank 0 and `None` elsewhere.
    """
    c = mesh.coordinates
    local = c.dat.data_ro[:c.function_space().dof_dset.size]
    gathered = mesh.comm.gather(np.asarray(local, dtype=float), root=0)
    if mesh.comm.rank != 0:
        return None
    allr = np.vstack(gathered)
    return allr[np.lexsort((allr[:, 2], allr[:, 1], allr[:, 0]))]


def coordinate_digest(mesh):
    """`(sha256 of the sorted owned coordinates, n_rows)`, identical on all ranks.

    The hash IS the bitwise test.  When it differs, `--coords-out` has the
    arrays themselves so the disagreement can be quantified rather than merely
    reported.
    """
    rows = coordinate_rows(mesh)
    out = None
    if mesh.comm.rank == 0:
        out = (hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
               int(rows.shape[0]))
    return mesh.comm.bcast(out, root=0)


#: Panel entries that integrate `|detJ|` over cells and are therefore only as
#: reproducible as the worst-conditioned cell in the mesh.  **Report-only when
#: the mesh cannot support them, fully gated when it can**, which is a demotion
#: and not a deletion: the checks are right, their precondition is not always
#: met.
CONDITIONING_SENSITIVE = ("vol_mantle", "sub.vol", "min_jac_ratio")

#: The predicate, and it is measured rather than guessed.  On `b4_sphere.msh`
#: after untangling, the coordinates are bitwise identical across rank counts
#: (the digests match) and the census is zero -- yet `vol_mantle` still moves by
#: 2.91e-11 between 1 and 4 ranks.  Localised: of 79447 mantle cells, 47730
#: differ bitwise between the two runs but all except **two** differ only in the
#: last bit (largest 3.7e-18).  Those two carry 1.0928e-09 of the 1.0928e-09
#: total.  They are cells 107270 and 108290, `min|detJ|/max|detJ|` = 0.0810 and
#: 0.0582, volumes 2.25e-05 and 2.98e-05, at r = 2.1818 and 2.1810 -- the ring
#: immediately outside the straightened patch, where straight nodes meet curved
#: ones.  A tetrahedron that distorted computes its own volume to about five
#: reproducible digits, because the local vertex ordering is partition-dependent
#: and the evaluation cancels.  The comparison in the buffer, whose worst cell
#: is at 0.97, is 1.8e-16 relative.
#:
#: So the demotion is triggered by the mesh's *conditioning*, not by tangling:
#: 0.5 separates the two bad cells (0.058, 0.081) from the next worst (0.520)
#: here by an order of magnitude.  A mesh that clears both conditions gets the
#: full panel gated, which is the point of keeping the checks.
MIN_JAC_RATIO_FOR_GATING = 0.5

#: **The ring's irreproducibility is acceptable, and it is a floor, not an
#: error bar.**  Recorded so nobody re-derives it: 1.09e-09 absolute on a
#: mantle volume of 37.5, i.e. 2.91e-11 relative, against gate tolerances of
#: 1e-02 to 1e-03 relative -- three or more orders below anything this project
#: gates on.  What that number is *not* is the discretisation error of those
#: cells.  A cell at min|detJ|/max|detJ| = 0.058 carries a genuine local
#: geometry error far above its own run-to-run jitter; the jitter only bounds
#: how reproducibly that error is committed.
#:
#: This is the strongest argument that the **generator predicate is the
#: endpoint and the fixed-point untangler is the interim**: the untangler
#: removes the folds and leaves the ring, while a generator that never emits a
#: foldable cell leaves neither.


def geometry_panel(untangle=True, quad_degree=8):
    """A fixed panel for comparison ACROSS RANK COUNTS, on ONE mesh file.

    Three kinds of entry, and they are gated differently on purpose:

    * **the coordinate digest and the census** -- exact.  These are the real
      gate.  The repair is a deterministic function of the coordinates, so a
      differing digest means the ranks built different geometry, full stop; and
      the census (count *and* the multiset of centroid radii) is the cheapest
      statement of the same thing that a human can read.  The round-one failure
      -- 3 tangled cells at 4 ranks against 2 at 1 -- is caught head-on here
      rather than inferred from a volume.
    * **surface integrals and the non-mantle volumes** -- 1e-12.  These passed
      at 1e-15 in round one and cost nothing, so they stay.
    * **the mantle volume and the min-Jacobian ratio** -- see
      `TANGLE_SENSITIVE`.

    Both meshes are in it because the submesh is re-curved separately.
    """
    panel = {}
    parent = curve_mesh(Mesh(MESH_FILE), untangle=untangle)
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 3, CELL_MANTLE), untangle=untangle)
    sub.cartesian = False

    for label, m in [("parent", parent), ("sub", sub)]:
        digest, nrows = coordinate_digest(m)
        panel[f"{label}.coords_sha256"] = digest
        panel[f"{label}.coords_rows"] = float(nrows)
        n, radii = tangle_census(m, f"{label}, untangle={untangle}")
        panel[f"{label}.census_n"] = float(n)
        panel[f"{label}.census_radii"] = sorted(float(v) for v in radii)

    for name, tag in [("vol_mantle", CELL_MANTLE), ("vol_inner", CELL_INNER),
                      ("vol_buffer", CELL_BUFFER)]:
        panel[f"parent.{name}"] = assemble(
            1 * dx(tag, domain=parent, degree=quad_degree))
    panel["sub.vol"] = assemble(1 * dx(domain=sub, degree=quad_degree))

    Xp, Xs = SpatialCoordinate(parent), SpatialCoordinate(sub)
    for tag, label in [(SURF_OUTER, "2Re"), (SURF_INNER, "0.5Rc")]:
        panel[f"parent.ds{tag}_{label}"] = assemble(
            (1 + Xp[0]) * ds(tag, domain=parent, degree=quad_degree))
    for tag, label in [(SURF_RE, "Re"), (SURF_RC, "Rc")]:
        panel[f"parent.dS{tag}_{label}"] = assemble(
            avg(1 + Xp[0]) * dS(tag, domain=parent, degree=quad_degree))
        panel[f"sub.ds{tag}_{label}"] = assemble(
            (1 + Xs[0]) * ds(tag, domain=sub, degree=quad_degree))

    for label, m in [("parent", parent), ("sub", sub)]:
        ratio, tangled, ncells = min_jacobian(m)
        panel[f"{label}.min_jac_ratio"] = ratio
        panel[f"{label}.cells"] = float(ncells)
        n_ill, n_tag, per = illconditioned_on_tagged_facets(m)
        panel[f"{label}.illcond_n"] = float(n_ill)
        panel[f"{label}.illcond_on_tagged"] = float(n_tag)
        panel[f"{label}.illcond_tags"] = {str(t): float(v)
                                          for t, v in sorted(per.items())}
    return panel, parent, sub


def panel_report(panel, reference=None, rtol=1e-12):
    """Print the panel and check it against a reference panel from another run.

    Returns `(ok, n_report_only)`.  `ok` ignores the report-only entries by
    design; they are printed with a `[rep]` flag and a reason so that a demoted
    check cannot be mistaken for a passing one.
    """
    tangled = any(panel.get(f"{m}.census_n", 0.0) > 0 for m in ("parent", "sub"))
    worst = min([panel.get(f"{m}.min_jac_ratio", 1.0)
                 for m in ("parent", "sub")] or [1.0])
    demote = tangled or worst < MIN_JAC_RATIO_FOR_GATING
    why = ("census nonzero" if tangled else
           f"worst min|detJ|/max|detJ| = {worst:.4f} < "
           f"{MIN_JAC_RATIO_FOR_GATING}")
    say(f"\nGeometry panel ({COMM_WORLD.size} ranks)"
        + (f"   [{why}: the |detJ|-weighted entries are REPORT-ONLY, see "
           "CONDITIONING_SENSITIVE]" if demote else
           "   [mesh well conditioned: every entry gated]"))
    if demote:
        for label in ("parent", "sub"):
            k = f"{label}.illcond_on_tagged"
            if k in panel:
                n_ill, n_tag, per = (panel[f"{label}.illcond_n"],
                                     panel[k], panel.get(f"{label}.illcond_tags",
                                                         {}))
                say(f"    {label}: {int(n_ill)} cells below "
                    f"min|detJ|/max|detJ| {MIN_JAC_RATIO_FOR_GATING}, of which "
                    f"{int(n_tag)} own a facet on a tagged surface"
                    + (f" {per}" if per else "")
                    + ("  -> the ring sits BESIDE the sheets, not on them"
                       if not n_tag else
                       "  -> the ring TOUCHES a sheet surface"))
    ok, n_rep = True, 0
    for k in sorted(panel):
        v = panel[k]
        demoted = demote and any(t in k for t in CONDITIONING_SENSITIVE)
        # `dict` and `list` entries are compared exactly, not numerically, so
        # they must survive the formatter too. An earlier version formatted
        # everything that was not a str or a list with `{: .16e}` and died on
        # the per-tag dict with `TypeError: unsupported format string passed to
        # dict.__format__` -- after the panel had been computed and the verdict
        # printed. A report that cannot print its own result is not a gate.
        if isinstance(v, str):
            show = v
        elif isinstance(v, dict):
            show = "{}" if not v else ", ".join(f"{k}:{int(x)}"
                                                for k, x in sorted(v.items()))
        elif isinstance(v, list):
            show = f"[{len(v)}] " + ", ".join(f"{x:.6f}" for x in v[:6])
        else:
            show = f"{v: .16e}"
        if reference is None or k not in reference:
            say(f"  [ -- ] {k:<26s} {show}")
            continue
        b = reference[k]
        if isinstance(v, (str, list, dict)):
            good, rel = (v == b), 0.0 if v == b else float("nan")
        else:
            rel = abs(v - b) / max(abs(b), 1e-300)
            good = rel <= rtol
        if demoted:
            n_rep += 1
            say(f"  [rep] {k:<26s} {show}   rel {rel:.2e}   "
                f"(report-only: {why})")
            continue
        ok &= good
        say(f"  [{'ok ' if good else 'FAIL'}] {k:<26s} {show}   rel {rel:.2e}")
    if reference is not None:
        missing = sorted(set(reference) - set(panel))
        if missing:
            ok = False
            say(f"  [FAIL] entries missing from this run: {missing}")
        say(f"\nPanel vs reference at rtol {rtol:.0e}: "
            f"{'MATCH' if ok else 'MISMATCH'}"
            + (f"   ({n_rep} entries report-only)" if n_rep else ""))
    return ok, n_rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", default="coarse",
                    choices=list(gen.CONFIGURATIONS))
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--litho-layers", type=int, default=2)
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--mesh-file", default=MESH_FILE)
    ap.add_argument("--panel", action="store_true",
                    help="the rank-consistency panel instead of the full "
                         "geometry stage; run it at two rank counts against "
                         "ONE mesh file and compare with --panel-check")
    ap.add_argument("--no-untangle", action="store_true",
                    help="--panel arm without the tangling repair")
    ap.add_argument("--coords-out", default=None,
                    help="write the sorted owned coordinates as .npy, so a "
                         "digest mismatch can be quantified rather than only "
                         "reported")
    ap.add_argument("--allow-tangled", action="store_true",
                    help="the tangled-cell checks become notes rather than "
                         "failures, so a deliberately-folded stage-1 arm can "
                         "run the rest of the gate. Exit 1 stays the DEFAULT.")
    ap.add_argument("--panel-out", default=None)
    ap.add_argument("--panel-check", default=None)
    ap.add_argument("--panel-rtol", type=float, default=1e-12)
    ap.add_argument("--reuse-mesh", action="store_true",
                    help="read an existing --mesh-file instead of regenerating; "
                         "REQUIRED when comparing rank counts, because gmsh "
                         "does not reproduce its tetrahedralisation")
    args = ap.parse_args()
    MESH_FILE = args.mesh_file

    provenance(os.path.basename(__file__))

    if args.panel:
        if not os.path.exists(MESH_FILE):
            build_mesh(configuration=args.configuration, grade=args.grade,
                       litho_layers=args.litho_layers)
        panel, parent_m, sub_m = geometry_panel(
            untangle=not args.no_untangle)
        ref = None
        if args.panel_check:
            with open(args.panel_check) as fh:
                ref = json.load(fh)
        good, _ = panel_report(panel, ref, rtol=args.panel_rtol)
        if args.coords_out:
            for label, m in (("parent", parent_m), ("sub", sub_m)):
                rows = coordinate_rows(m)
                if COMM_WORLD.rank == 0:
                    np.save(f"{args.coords_out}.{label}.npy", rows)
        if args.panel_out and COMM_WORLD.rank == 0:
            with open(args.panel_out, "w") as fh:
                json.dump(panel, fh, indent=1)
        sys.exit(0 if good else 1)

    ok, _ = geometry_stage(args.configuration, args.grade, args.litho_layers,
                           args.quality, reuse=args.reuse_mesh,
                           allow_tangled=args.allow_tangled)
    sys.exit(0 if ok else 1)
