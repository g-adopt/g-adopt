r"""GIAMIP Exp02: move the inverted bed of one mesh level onto another level.

WHAT THIS SCRIPT DOES
    An inversion at a coarse level (for example L0) ends with its bed at the
    start of the run in its `state.h5`: the last `bed_start_parent_cg2_m`
    entry, a CG2 field on the parent mesh whose nodes on Re carry the bed.
    An inversion at a finer level (for example L2) starts from that bed
    through the existing restart path of `invert_exp02.py`. This script
    makes that possible without a change to `invert_exp02.py`:

    1. Load the meshes of the target input file (an `inputs_exp02.h5` of
       `prepare_exp02.py --refine n`, on the rank count of that file).
    2. Load the parent mesh and the last bed of the source state file. The
       source can have been written on another rank count; Firedrake then
       loads its mesh with its default partition.
    3. Evaluate the source bed at the CG2 nodes of the target mantle on Re
       (`SourceSurface`, below).
    4. Append the result to the target file as entry `idx = 0` of
       `bed_start_parent_cg2_m`, with `exp02_common.append_control`, and a
       JSON record of the transfer as the dataset
       `/giamip_inputs/bed_transfer`.

    After that, `invert_exp02.py --mode invert --restart_state <target>`
    copies the target file into its run directory as `state.h5`, loads the
    meshes and the transferred bed (`read_inputs(last_control=True)`), and
    starts ROL from it. Its own iterates go to `idx = 1, 2, ...`.

THE EVALUATION
    The source bed is the P2 interpolant of its nodal values on the curved
    source surface Re: each facet of Re is a P2 triangle with three vertex
    nodes and three edge nodes, and both the position and the bed on the
    facet are the P2 Lagrange interpolants of the six nodes. The target
    nodes on Re lie on the sphere of radius Re. The source surface meets
    the sphere at its nodes and lies off the sphere between them by
    O(h^3), so a target node is not on the source surface in general.

    The script evaluates the source bed at the point of the source surface
    on the ray from the centre through the target node. For each target
    node x and each candidate facet it solves

        X(xi, eta) = s x                                   (3 equations)

    for the reference point (xi, eta) of the facet and the ray factor s,
    by Newton's method from the flat-triangle solution, with X the P2
    position map of the facet. The facet that contains the point (all
    barycentric coordinates >= 0 up to round-off) gives the value
    sum_k N_k(xi, eta) b_k with the P2 basis N_k and the nodal bed b_k.

    Why not a point evaluation in the 3-D source mesh (`VertexOnlyMesh`,
    `Function.at`, cross-mesh `interpolate`): the bed on the parent is
    nonzero only at the nodes on Re, and drops to zero at the next nodes
    inside and outside. A target node that lies a distance delta off the
    source surface then gets the value b (1 - O(delta / h_r)) with the
    radial cell size h_r. That error is proportional to the bed itself
    (several km in the oceans), and it is not an interpolation error of
    the surface field. On the ray, the value is the surface interpolant,
    and the only error is the P2 interpolation error of the source field.
    In the tests with a smooth bed on a small shell mesh, the point
    evaluation had about 1.8 times the RMS error of the ray rule at two
    levels; both errors decreased by about 7 per level.

    Consequences:
    - a target node that coincides with a source node (a vertex, or at L1
      the vertices of the target on the source edge nodes, because giamsh
      snaps both onto the sphere along the same ray) gets the source nodal
      value to round-off. The record gives the RMS and the largest
      difference at these nodes;
    - a smooth bed is reproduced at the other nodes to the P2 interpolation
      error of the source mesh, O(h^3);
    - a target node that no source facet contains stops the script with an
      error. No node gets a zero value by default;
    - an owned target node on Re that its owner rank does not find through
      its local Re facets stops the script with an error (the indicator of
      the nodes on Re is MAX-reduced to the owners, as in
      `exp02_common.SurfaceNodeMap`).

THE SOURCE MESH ON ANOTHER RANK COUNT
    `CheckpointFile.load_mesh` loads the stored distribution only when the
    file was written on the same number of ranks; otherwise it partitions
    the mesh again with its default partitioner. The script uses the source
    mesh only to read the facets of Re and their nodal values, so any
    partition works. No `Submesh` is cut from the source mesh: the facets
    of Re are interior facets of the parent with the tag of Re, and the
    side of the mantle is the side whose cell centroid is nearer the
    centre. Each rank collects the facets of Re that it owns, and all
    ranks receive all facets (`MPI.Allgatherv`): 24 doubles per facet for
    the geometry and 6 per field, a few tens of MB at L0.

THE SOURCE FIELD
    The last `bed_start_parent_cg2_m` entry of the source, or, for a state
    file of an earlier version of the inversion, the last `bed_start_m`
    entry (the broken CG2 copy of `ParentCopy`). Both are read on the
    mantle side of each facet of Re, which is where both versions hold the
    bed. The observed bed of the source (`SL_obs`, a broken copy) is
    transferred the same way, for the record only.

WHAT IS WRITTEN (--start)
    `change` (default): the observed bed of the target plus the change of
    the source inversion, (source bed - source observed bed), both
    evaluated at the target nodes. `bed`: the source bed at the target
    nodes.
    The observed bed of a finer mesh resolves topography that the source
    mesh does not (L0 -> L1 of Exp01: 147 m RMS between the two observed
    beds against 12.3 m RMS change of the L0 inversion). With `bed` a
    finer inversion starts with the missing topography in its misfit; with
    `change` it starts from the observed topography of its own mesh plus
    the correction that the coarse inversion found. The nodes off Re are
    zero in both cases.

THE TARGET FILE
    The target is changed in place: one function entry and one JSON
    dataset are appended. The script refuses a target that already holds
    an entry of `bed_start_parent_cg2_m` or `bed_start_m`, because a second
    `idx = 0` entry cannot be written and a later index would be read as a
    bed of the target inversion. To transfer again, use a fresh copy of the
    prepare output. `run_transfer.slurm` copies the target to
    `<target>.bak` before the run and removes the copy only after a run
    that exits with 0.

    The timestepping information of the entry has the keys of the entries
    that `invert_exp02.py` appends (`J`, `rms_misfit_m`), because a
    `CheckpointFile` refuses entries of one function with different keys.
    The values are the J and the RMS misfit of the source iterate (NaN if
    the source has none). They are the values of the source run on the
    source mesh, not of the target mesh. The JSON dataset
    `/giamip_inputs/bed_transfer` marks the entry as transferred and names
    the source.

RUN
    srun python3 transfer_bed_exp02.py --target <L2>/inputs_exp02.h5 \
        --source <L0 run>/state.h5
    on the rank count of the target file.

OUTPUT (next to the target)
    <target stem>_transfer_bed.json   the record of the transfer
    transfer_bed_exp02.log            the log of rank 0
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# `gadopt` imports `firedrake.pyplot`, which imports matplotlib's font
# manager. Without a font cache every rank scans the fonts; the job
# environment names a cache built once on a login node.
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# The shared module imports gadopt before firedrake.
import exp02_common as common  # noqa: E402
from gadopt import (COMM_WORLD, CheckpointFile, Constant,  # noqa: E402
                    DirichletBC, Function, FunctionSpace, VectorFunctionSpace,
                    assemble, dx)
from mpi4py import MPI  # noqa: E402
from pyop2 import op2  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

D_SCALE = common.D_SCALE

#: The names of the bed in a state file: the exact CG2 copy, and the
#: broken copy of an earlier version of the inversion.
BED_NAMES = (common.STATE_CONTROL_NAME, common.LEGACY_STATE_CONTROL_NAME)
#: The HDF5 dataset of the transfer record in the target file.
TRANSFER_RECORD = f"{common.INPUT_GROUP}/bed_transfer"

#: A target node belongs to a facet when every barycentric coordinate of
#: its ray point is at least -INSIDE_TOL. Round-off of the Newton solve is
#: about 1e-15; a node on a shared edge has a coordinate of that size and
#: either facet gives the same value there.
INSIDE_TOL = 1.0e-9
#: The Newton iterations of the ray-facet intersection. The P2 map is a
#: small perturbation of the flat triangle (the facets are nearly flat), so
#: Newton from the flat-triangle solution converges in 3-4 iterations.
NEWTON_ITERATIONS = 12
#: The residual |X(xi, eta) - s x| below which a Newton solve has converged,
#: relative to |x| (coordinates are in units of D, |x| = Re about 2.2).
NEWTON_TOL = 1.0e-13
#: The number of nearest facets (by the direction of their centroid) tried
#: first for each target node. The nodes that none of them contains are
#: tried again against every facet within the largest facet radius.
CANDIDATES = 12
#: Two nodes coincide when they are this close, relative to Re. The giamsh
#: snap and the P2 curving both put a node on the sphere as x R / |x|, so a
#: shared node agrees to round-off (about 1e-16 relative).
COINCIDE_TOL = 1.0e-10


# --------------------------------------------------------------------------
# The P2 triangle
# --------------------------------------------------------------------------


def p2_basis(xi, eta):
    """The six P2 Lagrange basis functions of a triangle and their derivatives.

    The node order is (v0, v1, v2, e01, e12, e20): the three vertices, then
    the edge nodes between v0-v1, v1-v2 and v2-v0. The barycentric
    coordinates are l0 = 1 - xi - eta, l1 = xi, l2 = eta. A vertex basis
    function is l (2 l - 1), an edge basis function 4 l_i l_j.

    Args:
      xi, eta: arrays of one shape.

    Returns:
      `(N, dN_dxi, dN_deta)`, each of shape `xi.shape + (6,)`.
    """
    l0, l1, l2 = 1.0 - xi - eta, xi, eta
    N = np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                  4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0], axis=-1)
    # d l0 / d xi = -1, d l1 / d xi = 1, d l2 / d xi = 0.
    dxi = np.stack([-(4 * l0 - 1), 4 * l1 - 1, np.zeros_like(xi),
                    4 * (l0 - l1), 4 * l2, -4 * l2], axis=-1)
    # d l0 / d eta = -1, d l1 / d eta = 0, d l2 / d eta = 1.
    deta = np.stack([-(4 * l0 - 1), np.zeros_like(xi), 4 * l2 - 1,
                     -4 * l1, 4 * l1, 4 * (l0 - l2)], axis=-1)
    return N, dxi, deta


def ray_facet_intersection(nodes, x):
    """The point of each P2 facet on the ray from the centre through x.

    Solves X(xi, eta) = s x for (xi, eta, s) by Newton's method, for many
    (facet, point) pairs at once. X is the P2 position map of the facet.
    The start is the intersection of the ray with the flat triangle of the
    three vertices. The solve also runs for pairs whose facet does not
    contain the ray; the caller rejects those by their barycentric
    coordinates (and a pair that does not converge by its residual).

    Args:
      nodes: shape `(m, 6, 3)`, the node positions of the facets in the
        order of `p2_basis`.
      x: shape `(m, 3)`, the target points.

    Returns:
      `(xi, eta, s, residual)`, each of shape `(m,)`. `residual` is
      |X(xi, eta) - s x| / |x|; it is +inf where the solve broke down.
    """
    m = len(x)
    if m == 0:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    v0, v1, v2 = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    # Flat triangle: v0 + xi (v1 - v0) + eta (v2 - v0) - s x = 0.
    A = np.stack([v1 - v0, v2 - v0, -x], axis=-1)
    with np.errstate(all="ignore"):
        try:
            start = np.linalg.solve(A, -v0[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # A singular system (the ray parallel to a facet far away):
            # solve pair by pair and mark the singular ones.
            start = np.full((m, 3), np.nan)
            for i in range(m):
                try:
                    start[i] = np.linalg.solve(A[i], -v0[i])
                except np.linalg.LinAlgError:
                    pass
    xi, eta, s = start[:, 0], start[:, 1], start[:, 2]
    scale = np.linalg.norm(x, axis=1)
    for _ in range(NEWTON_ITERATIONS):
        N, dxi, deta = p2_basis(xi, eta)
        # The residual F = X(xi, eta) - s x and its Jacobian
        # [dX/dxi, dX/deta, -x].
        F = np.einsum("mk,mkd->md", N, nodes) - s[:, None] * x
        J = np.stack([np.einsum("mk,mkd->md", dxi, nodes),
                      np.einsum("mk,mkd->md", deta, nodes), -x], axis=-1)
        with np.errstate(all="ignore"):
            try:
                step = np.linalg.solve(J, -F[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = np.full((m, 3), np.nan)
                for i in range(m):
                    try:
                        step[i] = np.linalg.solve(J[i], -F[i])
                    except np.linalg.LinAlgError:
                        pass
        xi, eta, s = xi + step[:, 0], eta + step[:, 1], s + step[:, 2]
        if np.all(~np.isfinite(step) | (np.abs(step) < 1e-16)):
            break
    N, _, _ = p2_basis(xi, eta)
    F = np.einsum("mk,mkd->md", N, nodes) - s[:, None] * x
    residual = np.linalg.norm(F, axis=1) / scale
    residual[~np.isfinite(residual)] = np.inf
    return xi, eta, s, residual


# --------------------------------------------------------------------------
# The source surface
# --------------------------------------------------------------------------


def facet_node_order(finat_element, local_facet):
    """The six local nodes of a facet of a P2 tetrahedron, in `p2_basis` order.

    Firedrake's CG2 element does not number its nodes in the UFC order
    (vertex 1 is not local node 1), so the order is read from the entity
    numbering of the element: the vertices of the facet from the reference
    cell, the node of each vertex and of each edge from `entity_dofs`.

    Args:
      finat_element: the FInAT element of the scalar CG2 space.
      local_facet: the local number of the facet in the cell.

    Returns:
      A list of six local node numbers: v0, v1, v2, e01, e12, e20.
    """
    topology = finat_element.cell.get_topology()
    dofs = finat_element.entity_dofs()
    v = topology[2][local_facet]
    edge_of = {tuple(sorted(vertices)): edge
               for edge, vertices in topology[1].items()}
    order = [dofs[0][v[0]][0], dofs[0][v[1]][0], dofs[0][v[2]][0]]
    for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
        order.append(dofs[1][edge_of[tuple(sorted((a, b)))]][0])
    return order


def allgather_rows(local, comm):
    """All rows of a 2-D float array from every rank, in rank order.

    Uses `Allgatherv` on the raw buffer, so a rank without rows sends
    nothing and the result is the same on every rank.
    """
    local = np.ascontiguousarray(local, dtype=float)
    width = local.shape[1]
    counts = np.array(comm.allgather(local.size), dtype=np.int64)
    out = np.empty(int(counts.sum()), dtype=float)
    displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    comm.Allgatherv(local.ravel(), [out, counts, displs, MPI.DOUBLE])
    return out.reshape(-1, width)


class SourceSurface:
    """The facets of Re of a parent mesh, with nodal values, on every rank.

    Each rank reads the facets of Re that it owns: their six node
    positions (P2 geometry of the parent) and the six nodal values of each
    field on the mantle side. All ranks then hold all facets. The class
    evaluates the fields at any point set by the ray rule of the module
    docstring.

    Args:
      parent: the source parent mesh (P2 coordinates), any partition.
      fields: `{name: Function}` on the parent: CG2, or the broken CG2 copy
        of `exp02_common.ParentCopy`. The broken copy has the node order of
        CG2 within each cell.
      surface_tag: the facet tag of Re.

    Attributes:
      nodes: shape `(n_facets, 6, 3)`, node positions.
      values: `{name: array of shape (n_facets, 6)}`.
      n_facets: the number of facets of Re.
    """

    def __init__(self, parent, fields, surface_tag=common.SURF_RE):
        comm = parent.comm
        V = FunctionSpace(parent, "CG", 2)
        element = V.finat_element
        if element.degree != 2 or element.cell.get_spatial_dimension() != 3:
            raise RuntimeError("SourceSurface needs CG2 on tetrahedra")
        # The P2 positions of the CG2 nodes. The parent coordinates are P2
        # Lagrange, so this interpolation copies them.
        coords = Function(VectorFunctionSpace(parent, "CG", 2)).interpolate(
            parent.coordinates).dat.data_ro_with_halos
        facets = parent.interior_facets
        # The interior facets of Re that this rank owns. Owned entities come
        # before the halo in a PyOP2 set; each facet is owned by one rank.
        indices = np.asarray(facets.subset(surface_tag).indices)
        indices = indices[indices < facets.set.size]
        cg2_map = V.interior_facet_node_map().values_with_halo
        local_facet = facets.local_facet_dat.data_ro_with_halos
        n_cell = element.space_dimension()
        # The local node order of each local facet number, in p2_basis order.
        order = np.array([facet_node_order(element, k) for k in range(4)])

        # The mantle side of each facet: the cell whose centroid (mean of
        # its CG2 nodes) is nearer the centre. Re is a sphere, so the cell
        # below it is inside, whatever the orientation of the facet.
        cells = cg2_map[indices].reshape(len(indices), 2, n_cell)
        radius = np.linalg.norm(coords[cells].mean(axis=2), axis=-1)
        side = np.argmin(radius, axis=1) if len(indices) else np.zeros(0, int)
        rows = np.arange(len(indices))
        lf = local_facet[indices, side] if len(indices) else np.zeros(0, int)
        local_nodes = order[lf] if len(indices) else np.zeros((0, 6), int)
        # The CG2 nodes of the six facet nodes on the mantle side.
        cg2_nodes = cells[rows, side][rows[:, None], local_nodes] \
            if len(indices) else np.zeros((0, 6), int)
        local_xyz = coords[cg2_nodes].reshape(-1, 18)

        local_values = {}
        for name, f in fields.items():
            W = f.function_space()
            if W.finat_element.space_dimension() != n_cell:
                raise RuntimeError(f"{name}: not a (broken) CG2 field")
            fmap = W.interior_facet_node_map().values_with_halo
            fcells = fmap[indices].reshape(len(indices), 2, n_cell)
            fnodes = fcells[rows, side][rows[:, None], local_nodes] \
                if len(indices) else np.zeros((0, 6), int)
            local_values[name] = f.dat.data_ro_with_halos[fnodes].reshape(-1, 6)

        self.nodes = allgather_rows(local_xyz, comm).reshape(-1, 6, 3)
        self.values = {name: allgather_rows(v, comm)
                       for name, v in local_values.items()}
        self.n_facets = len(self.nodes)
        if self.n_facets == 0:
            raise RuntimeError(f"the source mesh has no facet with tag "
                               f"{surface_tag}")
        # The search structure: the direction of the centroid of the three
        # vertices of each facet, and the largest angular size of a facet
        # (as a chord length on the unit sphere) for the fallback search.
        centre = self.nodes[:, :3].mean(axis=1)
        self.direction = centre / np.linalg.norm(centre, axis=1)[:, None]
        unit = self.nodes / np.linalg.norm(self.nodes, axis=2)[..., None]
        self.facet_radius = float(np.max(np.linalg.norm(
            unit - self.direction[:, None, :], axis=2)))
        self.tree = cKDTree(self.direction)
        # The distinct source nodes (vertices and edge nodes), for the
        # comparison at the target nodes that coincide with them.
        self.vertex_xyz, vi = np.unique(
            self.nodes[:, :3].reshape(-1, 3), axis=0, return_index=True)
        self.vertex_values = {n: v[:, :3].reshape(-1)[vi]
                              for n, v in self.values.items()}
        self.edge_xyz, ei = np.unique(
            self.nodes[:, 3:].reshape(-1, 3), axis=0, return_index=True)
        self.edge_values = {n: v[:, 3:].reshape(-1)[ei]
                            for n, v in self.values.items()}

    def _solve(self, points, candidates):
        """Try one candidate facet per point; return the score and solution.

        The score is the smallest barycentric coordinate of the ray point;
        it is -inf where the Newton solve did not converge or the ray
        points away from the facet.
        """
        xi, eta, s, residual = ray_facet_intersection(self.nodes[candidates],
                                                      points)
        score = np.minimum(np.minimum(1.0 - xi - eta, xi), eta)
        bad = (residual > NEWTON_TOL) | ~(s > 0) | ~np.isfinite(score)
        score[bad] = -np.inf
        return score, xi, eta, s

    def locate(self, points):
        """The facet and the reference point of each target point.

        Args:
          points: shape `(n, 3)`.

        Returns:
          `(facet, xi, eta, s, score)`, each of shape `(n,)`. `score` is the
          smallest barycentric coordinate; a point is located when it is at
          least `-INSIDE_TOL`.
        """
        n = len(points)
        facet = np.full(n, -1, dtype=np.int64)
        best = np.full(n, -np.inf)
        xi, eta, s = np.zeros(n), np.zeros(n), np.zeros(n)
        if n == 0:
            return facet, xi, eta, s, best
        unit = points / np.linalg.norm(points, axis=1)[:, None]

        def consider(rows, cand):
            # One candidate facet per row (distinct rows); keep it where it
            # is better than the best so far.
            sc, a, b, t = self._solve(points[rows], cand)
            better = sc > best[rows]
            r = rows[better]
            facet[r], best[r] = cand[better], sc[better]
            xi[r], eta[r], s[r] = a[better], b[better], t[better]

        # First pass: the nearest facets by centroid direction.
        k = min(CANDIDATES, self.n_facets)
        _, near = self.tree.query(unit, k=k)
        near = near.reshape(n, k)
        rows = np.arange(n)
        for j in range(k):
            consider(rows, near[:, j])
        # Second pass for the points that no near facet contains: every
        # facet whose centroid direction is within the largest facet size.
        missing = np.flatnonzero(best < -INSIDE_TOL)
        if len(missing):
            balls = self.tree.query_ball_point(unit[missing],
                                               r=1.5 * self.facet_radius)
            for i, cand in zip(missing, balls):
                cand = np.asarray(cand, dtype=np.int64)
                if not len(cand):
                    continue
                # All candidates of this one point at once; the best one is
                # offered to `consider` as a single row.
                sc, _, _, _ = self._solve(np.repeat(points[i:i + 1],
                                                    len(cand), axis=0), cand)
                consider(np.array([i]), cand[[int(np.argmax(sc))]])
        return facet, xi, eta, s, best

    def evaluate(self, facet, xi, eta, name):
        """The P2 interpolant of field `name` at the located points."""
        N, _, _ = p2_basis(xi, eta)
        return np.einsum("nk,nk->n", N, self.values[name][facet])


# --------------------------------------------------------------------------
# The files
# --------------------------------------------------------------------------


def load_source(path, say):
    """The parent mesh, the last bed and the record of a source state file.

    Collective. The mesh loads with its stored distribution on the rank
    count that wrote it, and with Firedrake's default partition on any
    other rank count.

    Returns:
      `(parent, fields, info)`: `fields` holds the bed (`"bed"`, CG2 or the
      broken copy) and, if the file has it, the broken copy of `SL_obs`
      (`"SL_obs"`); `info` holds the entry name, index, J and RMS misfit of
      the bed and the record of the prepare run.
    """
    with CheckpointFile(path, "r") as chk:
        record = common.read_json_dataset(chk, f"{common.INPUT_GROUP}/record")
        parent = chk.load_mesh(common.PARENT_NAME)
        found = None
        for name in BED_NAMES:
            try:
                history = chk.get_timestepping_history(parent, name)
            except Exception:  # noqa: BLE001  (no entry of this name)
                continue
            if history and len(history.get("index", [])):
                found = (name, history)
                break
        if found is None:
            raise RuntimeError(
                f"{path} holds no {BED_NAMES[0]} and no {BED_NAMES[1]} entry: "
                "the source must be the state.h5 of an inversion")
        name, history = found
        position = int(np.argmax(history["index"]))
        index = int(history["index"][position])
        bed = chk.load_function(parent, name, idx=index)
        fields = {"bed": bed}
        try:
            fields["SL_obs"] = chk.load_function(parent, "SL_obs")
        except Exception as exc:  # noqa: BLE001  (record only)
            say(f"  source: no SL_obs ({exc!r}); the observed-bed check is "
                "skipped")
    info = {"name": name, "index": index, "record": record,
            "legacy": name != common.STATE_CONTROL_NAME,
            "comm_size_written": record.get("mesh", {}).get("ranks")}
    for key in ("J", "rms_misfit_m"):
        values = history.get(key)
        info[key] = (float(np.asarray(values)[position])
                     if values is not None else float("nan"))
    return parent, fields, info


def load_target(path):
    """The meshes, `SL_obs` and the record of a target input file.

    Collective. This is the mesh part of `exp02_common.read_inputs`: the
    same rank check, the parent with its stored distribution, the mantle
    cut and curved by `exp02_common.mantle_mesh`, and `SL_obs` moved onto
    the mantle by the same interpolation. It does not load the other
    earth fields and the ice of every step, which at L2 are most of the
    file and of the memory, and which the transfer does not need.

    Returns:
      `(parent, mantle, SL_obs, record, has_control)`: `has_control` is
      True when the file already holds a bed entry.

    Raises:
      RuntimeError: if the file was written on another number of ranks.
    """
    with CheckpointFile(path, "r") as chk:
        record = common.read_json_dataset(chk, f"{common.INPUT_GROUP}/record")
        ranks = record["mesh"]["ranks"]
        if ranks != COMM_WORLD.size:
            raise RuntimeError(
                f"{path} was written on {ranks} ranks; run on the same "
                f"number, not {COMM_WORLD.size}. On another number Firedrake "
                "partitions the mesh again with its default partitioner, "
                "which can leave ranks without a mantle cell.")
        parent = chk.load_mesh(common.PARENT_NAME)
        stored = chk.load_function(parent, "SL_obs")
        has_control = False
        for name in BED_NAMES:
            try:
                history = chk.get_timestepping_history(parent, name)
            except Exception:  # noqa: BLE001  (no entry of this name)
                continue
            if history and len(history.get("index", [])):
                has_control = True
    parent.cartesian = False
    mantle = common.mantle_mesh(parent, record["mesh"]["mantle_tags"])
    SL_obs = Function(FunctionSpace(mantle, *common.MANTLE_FIELDS["SL_obs"]),
                      name="SL_obs").interpolate(stored)
    return parent, mantle, SL_obs, record, has_control


def file_identity(path, with_sha256):
    """Size, modification time and (optionally) SHA-256 of a file (rank 0)."""
    stat = os.stat(path)
    out = {"path": os.path.realpath(path), "bytes": int(stat.st_size),
           "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                  time.localtime(stat.st_mtime))}
    if with_sha256:
        tic = time.time()
        out["sha256"] = common.sha256sum(path)
        out["sha256_s"] = time.time() - tic
    else:
        out["sha256"] = None
    return out


def rms_nodes(values, comm):
    """The RMS of the values that each rank holds (owned nodes), over all ranks."""
    total = comm.allreduce(float(np.sum(values ** 2)), op=MPI.SUM)
    count = comm.allreduce(int(np.size(values)), op=MPI.SUM)
    return float(np.sqrt(total / count)) if count else float("nan"), count


def coincident_check(points, values, ref_xyz, ref_values, owned, comm):
    """The target nodes that are source nodes, and the value differences there.

    Args:
      points, values: the located target nodes and their transferred values.
      ref_xyz, ref_values: the distinct source nodes and their values.
      owned: a mask of the target nodes that this rank owns.

    Returns:
      `{"nodes": count, "rms_m": .., "max_abs_m": ..}` over all ranks.
    """
    if len(points) and len(ref_xyz):
        distance, nearest = cKDTree(ref_xyz).query(points)
        tol = COINCIDE_TOL * float(np.linalg.norm(ref_xyz, axis=1).max())
        hit = (distance <= tol) & owned
        diff = values[hit] - ref_values[nearest[hit]]
    else:
        diff = np.zeros(0)
    rms, count = rms_nodes(diff, comm)
    largest = comm.allreduce(float(np.abs(diff).max()) if diff.size else 0.0,
                             op=MPI.MAX)
    return {"nodes": count, "rms_m": rms, "max_abs_m": largest}


# --------------------------------------------------------------------------
# The transfer
# --------------------------------------------------------------------------


def transfer(source_surface, mantle, names, surface_tag=common.SURF_RE):
    """Evaluate fields of the source surface at the target mantle nodes on Re.

    Collective (the checks for missing and unwritten nodes are
    reductions).

    Args:
      source_surface: the `SourceSurface`.
      mantle: the target mantle mesh (P2 coordinates).
      names: the fields of `source_surface` to evaluate.
      surface_tag: the facet tag of Re in the target mantle.

    Returns:
      `(out, geometry)`: `out` maps each name to a CG2 Function on the
      target mantle, equal to the source field at the nodes on Re and zero
      at every other node (in the units of the source field); `geometry`
      holds the target node positions on Re and the located facets, for the
      checks of the caller.

    Raises:
      RuntimeError: if a target node on Re has no source facet, or if an
        owned target node on Re is not found by its owner rank (it would
        keep a zero value).
    """
    comm = mantle.comm
    V = FunctionSpace(mantle, *common.CONTROL_SPACE)
    # The local mantle nodes on Re, owned and halo, and their positions on
    # the P2 geometry (interpolating the P2 coordinates into CG2 copies
    # them).
    on_re = np.asarray(DirichletBC(V, 0.0, surface_tag).nodes)
    xyz = Function(VectorFunctionSpace(mantle, "CG", 2)).interpolate(
        mantle.coordinates).dat.data_ro_with_halos[on_re]
    facet, xi, eta, s, score = source_surface.locate(xyz)
    missing = int(np.sum(score < -INSIDE_TOL))
    total_missing = comm.allreduce(missing, op=MPI.SUM)
    if total_missing:
        worst = np.argsort(score)[:5]
        detail = [(xyz[i].tolist(), float(score[i])) for i in worst
                  if score[i] < -INSIDE_TOL]
        raise RuntimeError(
            f"{total_missing} target nodes on Re have no source facet on "
            f"their ray (rank {comm.rank}: {missing}; examples: {detail})")
    out = {}
    for name in names:
        f = Function(V, name=name)
        # Every local node on Re, owned and halo, gets the value computed
        # from the same gathered facets and the same position, so the halo
        # copies equal the owned values bit for bit.
        f.dat.data_with_halos[on_re] = source_surface.evaluate(facet, xi, eta,
                                                               name)
        out[name] = f
    owned = on_re < V.dof_dset.size
    # Coverage of the owned nodes on Re. The values above are written only
    # at the nodes that a rank finds through its own local (owned and halo)
    # Re facets. An owned node whose Re facets all lie in cells that only
    # another rank holds would not be in `on_re` of its owner, and would
    # keep the value 0 after the next halo exchange (the owner wins). The
    # guard is that of `exp02_common.SurfaceNodeMap`: each rank marks every
    # local node on Re, owned or halo, and the halo marks are sent to the
    # owners with a MAX reduction. The owned entries of the indicator then
    # count each node on Re once over all ranks, independent of which rank
    # found its facets. That count must equal the count of the owned nodes
    # that the owners wrote themselves.
    indicator = Function(V)
    indicator.dat.data_wo_with_halos[on_re] = 1.0
    indicator.dat.local_to_global_begin(op2.MAX)
    indicator.dat.local_to_global_end(op2.MAX)
    n_expected = comm.allreduce(int(np.count_nonzero(indicator.dat.data_ro)),
                                op=MPI.SUM)
    n_written = comm.allreduce(int(owned.sum()), op=MPI.SUM)
    if n_written != n_expected:
        # Both counts are global, so every rank takes this branch and the
        # error is collective.
        raise RuntimeError(
            f"{n_expected - n_written} owned target nodes on Re have no "
            f"local Re facet on their owner rank and would keep a zero bed "
            f"({n_expected} nodes on Re, {n_written} written by their "
            f"owners)")
    geometry = {"xyz": xyz, "on_re": on_re, "owned": owned, "s": s,
                "score": score, "n_expected": n_expected}
    return out, geometry


def surface_rms(node_map, surface_dx, area, mantle_field):
    """The area-weighted RMS over Re of a CG2 mantle field, on the surface.

    The bed of the field goes onto the surface with the exact node map,
    as `invert_exp02.py` does for the restart bed, and the integral is the
    one of the RMS change in `invert_exp02.py` (P1 surface, `dx`).
    """
    b = node_map.restrict(mantle_field)
    return float(np.sqrt(float(assemble(b ** 2 * surface_dx)) / area))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", required=True,
                   help="the input file of prepare_exp02.py at the target "
                        "level; it is changed in place")
    p.add_argument("--source", required=True,
                   help="the state.h5 of the source inversion")
    p.add_argument("--output_path", default=None,
                   help="the directory of the log and the JSON record "
                        "(default: the directory of the target)")
    p.add_argument("--source_sha256", choices=["yes", "no"], default="yes",
                   help="hash the source file (L0 state: about 1.5 GB)")
    p.add_argument("--target_sha256", choices=["yes", "no"], default="no",
                   help="hash the target file before and after the append "
                        "(slow for L2: about 100 GB); size and mtime are "
                        "always recorded")
    p.add_argument("--start", choices=["bed", "change"], default="change",
                   help="change (default): write the "
                        "target observed bed plus the change of the source "
                        "bed from the source observed bed (keeps the "
                        "topography that only the target mesh resolves). "
                        "bed: write the source bed")
    p.add_argument("--allow_other_experiment", action="store_true",
                   help="allow a source of another experiment")
    return p.parse_args()


def main():
    """Transfer the bed and write the record."""
    args = parse_args()
    target = os.path.realpath(args.target)
    source = os.path.realpath(args.source)
    out_dir = args.output_path or os.path.dirname(target)
    if COMM_WORLD.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    COMM_WORLD.barrier()
    say = common.RunLog(os.path.join(out_dir, "transfer_bed_exp02.log"))
    tic_run = time.time()
    timings = {}
    say("=" * 78)
    say(f"GIAMIP bed transfer, {COMM_WORLD.size} ranks")
    say(f"  target {target}")
    say(f"  source {source}")
    say("=" * 78)

    # The identity of both files before anything is changed. The target
    # changes by the append, so its hash (when asked for) is taken before
    # and after.
    tic = time.time()
    identity = None
    if COMM_WORLD.rank == 0:
        identity = {
            "source": file_identity(source, args.source_sha256 == "yes"),
            "target_before": file_identity(target,
                                           args.target_sha256 == "yes")}
    identity = COMM_WORLD.bcast(identity, root=0)
    timings["identity_s"] = elapsed = time.time() - tic
    say(f"  source file {identity['source']}")
    say(f"  target file before {identity['target_before']} ({elapsed:.1f} s)")

    # The target meshes.
    tic = time.time()
    parent, mantle, SL_obs, record, has_control = load_target(target)
    timings["load_target_s"] = time.time() - tic
    if has_control:
        raise RuntimeError(
            f"{target} already holds a bed entry; transfer into a fresh copy "
            "of the prepare output")
    say(f"  target: experiment {record['experiment']}, refine "
        f"{record['mesh']['refine_levels']}, {record['mesh']['cells_mantle']} "
        f"mantle cells, {record['mesh']['ranks']} ranks "
        f"({timings['load_target_s']:.1f} s)")
    common.memory_now("after the target meshes", say)

    # The source mesh and bed.
    tic = time.time()
    source_parent, source_fields, source_info = load_source(source, say)
    timings["load_source_s"] = time.time() - tic
    source_record = source_info.pop("record")
    if (source_record["experiment"] != record["experiment"]
            and not args.allow_other_experiment):
        raise RuntimeError(
            f"source experiment {source_record['experiment']} against target "
            f"{record['experiment']}; pass --allow_other_experiment to allow")
    same_coarse = source_record["mesh"]["md5"] == record["mesh"]["md5"]
    if not same_coarse:
        say(f"  WARNING: the coarse meshes differ (MD5 "
            f"{source_record['mesh']['md5']} against {record['mesh']['md5']})")
    say(f"  source: {source_info['name']} idx {source_info['index']} "
        f"(J {source_info['J']:.10e}, RMS misfit "
        f"{source_info['rms_misfit_m']:.4g} m), refine "
        f"{source_record['mesh']['refine_levels']}, written on "
        f"{source_info['comm_size_written']} ranks, loaded on "
        f"{COMM_WORLD.size} ({timings['load_source_s']:.1f} s)")
    if source_info["legacy"]:
        say("  source: broken copy of an earlier version of the inversion; "
            "read on the mantle side of each facet of Re")

    # The facets of Re of the source, on every rank.
    tic = time.time()
    surface = SourceSurface(source_parent, source_fields)
    timings["gather_source_s"] = time.time() - tic
    say(f"  source surface: {surface.n_facets} facets of Re on every rank, "
        f"{surface.vertex_xyz.shape[0]} vertices, {surface.edge_xyz.shape[0]} "
        f"edge nodes ({timings['gather_source_s']:.1f} s)")
    common.memory_now("after the source surface", say)

    # The evaluation at the target nodes on Re.
    tic = time.time()
    names = list(source_fields)
    out, geom = transfer(surface, mantle, names)
    timings["evaluate_s"] = time.time() - tic
    bed = out["bed"]
    bed.rename(common.CONTROL_NAME)
    owned = geom["owned"]
    values = bed.dat.data_ro_with_halos[geom["on_re"]]
    # The number of owned target nodes on Re, summed over the ranks: the
    # count of CG2 nodes on Re of the target mesh (vertices plus edge nodes,
    # for Exp01 530 322 at L1 and 2 121 282 at L2). `transfer` has already
    # checked it against the MAX-reduced indicator of the nodes on Re.
    n_target = COMM_WORLD.allreduce(int(owned.sum()), op=MPI.SUM)
    lo = COMM_WORLD.allreduce(float(values[owned].min()) if owned.any()
                              else np.inf, op=MPI.MIN)
    hi = COMM_WORLD.allreduce(float(values[owned].max()) if owned.any()
                              else -np.inf, op=MPI.MAX)
    rms_nodal, _ = rms_nodes(values[owned], COMM_WORLD)
    # The radial position of the source surface at the target nodes:
    # s = |X| / |x| - 1 along the ray, the distance of the target node from
    # the source surface relative to Re.
    gap = np.abs(geom["s"][owned] - 1.0)
    gap_max = COMM_WORLD.allreduce(float(gap.max()) if gap.size else 0.0,
                                   op=MPI.MAX)
    score_min = COMM_WORLD.allreduce(
        float(geom["score"][owned].min()) if owned.any() else np.inf,
        op=MPI.MIN)
    say(f"  coverage: {n_target} owned target nodes on Re written by their "
        f"owners, {geom['n_expected']} nodes on Re in the MAX-reduced "
        f"indicator")
    say(f"  evaluated at {n_target} target nodes on Re: bed min {lo:.3f} m, "
        f"max {hi:.3f} m, nodal RMS {rms_nodal:.3f} m; largest distance of "
        f"a target node from the source surface {gap_max:.3e} Re; smallest "
        f"barycentric coordinate {score_min:.2e} "
        f"({timings['evaluate_s']:.1f} s)")

    # The values at the target nodes that are source nodes: equal to the
    # source nodal values up to round-off.
    coincident = {
        "vertices": coincident_check(geom["xyz"], values, surface.vertex_xyz,
                                     surface.vertex_values["bed"], owned,
                                     COMM_WORLD),
        "edge_nodes": coincident_check(geom["xyz"], values, surface.edge_xyz,
                                       surface.edge_values["bed"], owned,
                                       COMM_WORLD)}
    say(f"  at target nodes on source vertices: {coincident['vertices']}")
    say(f"  at target nodes on source edge nodes: {coincident['edge_nodes']}")

    # The bed against the observed bed of the target. The fields go onto
    # the surface with the exact node map of the inversion (the restart
    # path), and the RMS is the area-weighted RMS of the inversion log.
    tic = time.time()
    surface_mesh = common.surface_mesh(mantle)
    V_bed = FunctionSpace(surface_mesh, *common.CONTROL_SPACE)
    node_map = common.SurfaceNodeMap(V_bed, bed.function_space(),
                                     common.SURF_RE)
    surface_dx = dx(domain=surface_mesh)
    area = float(assemble(Constant(1.0) * surface_dx))
    bed_obs = Function(bed.function_space()).assign(-D_SCALE * SL_obs)
    diff = Function(bed.function_space())
    rms = {"bed_m": surface_rms(node_map, surface_dx, area, bed),
           "bed_minus_target_observed_m": surface_rms(
               node_map, surface_dx, area, diff.assign(bed - bed_obs))}
    if "SL_obs" in out:
        # The observed bed of the source on the target nodes. Its difference
        # from the bed carries the change of the source inversion over to
        # the target nodes; its difference from the target observed bed is
        # the part of the observed bed that the source mesh does not
        # resolve.
        source_obs = Function(bed.function_space()).assign(
            -D_SCALE * out["SL_obs"])
        rms["bed_minus_source_observed_m"] = surface_rms(
            node_map, surface_dx, area, diff.assign(bed - source_obs))
        rms["source_observed_minus_target_observed_m"] = surface_rms(
            node_map, surface_dx, area, diff.assign(source_obs - bed_obs))
    timings["checks_s"] = time.time() - tic
    say(f"  RMS over Re (area-weighted): {rms}")

    # The field that goes into the target file. `bed`: the source bed
    # itself. `change`: the observed bed of the target plus the change that
    # the source inversion made to its own observed bed, both evaluated at
    # the target nodes. The second keeps the topography that the target
    # mesh resolves and the source mesh does not.
    if args.start == "bed":
        written = bed
    else:
        if "SL_obs" not in out:
            raise RuntimeError("--start change needs SL_obs in the source")
        written = Function(bed.function_space(), name=common.CONTROL_NAME)
        on_re = geom["on_re"]
        # Only the nodes on Re: every other node stays zero, as the
        # inversion writes the bed (`SurfaceNodeMap.extend`).
        written.dat.data_with_halos[on_re] = (
            bed_obs.dat.data_ro_with_halos[on_re]
            + bed.dat.data_ro_with_halos[on_re]
            - source_obs.dat.data_ro_with_halos[on_re])
        rms["written_minus_target_observed_m"] = surface_rms(
            node_map, surface_dx, area, diff.assign(written - bed_obs))
        rms["written_minus_transferred_bed_m"] = surface_rms(
            node_map, surface_dx, area, diff.assign(written - bed))
        say(f"  start {args.start}: RMS of the written bed against the "
            f"target observed bed "
            f"{rms['written_minus_target_observed_m']:.3f} m, against the "
            f"transferred source bed "
            f"{rms['written_minus_transferred_bed_m']:.3f} m")

    # The append, as `invert_exp02.py` appends its iterates, and the record
    # of the transfer in the target file.
    transfer_record = {
        "transferred": True,
        "start": args.start,
        "script": os.path.abspath(__file__),
        "source": identity["source"],
        "source_entry": source_info,
        "source_mesh": {k: source_record["mesh"].get(k) for k in
                        ("md5", "refine_levels", "ranks", "cells_mantle")},
        "target_mesh": {k: record["mesh"].get(k) for k in
                        ("md5", "refine_levels", "ranks", "cells_mantle")},
        "same_coarse_mesh": same_coarse,
        "target_entry": {"name": common.STATE_CONTROL_NAME, "idx": 0,
                         "timestepping_info": "J and rms_misfit_m of the "
                         "source entry (source mesh)"},
        "method": "P2 interpolant on the source surface Re at the ray "
                  "through each target node",
        "ranks": COMM_WORLD.size,
        "target_nodes_on_re": n_target,
        "target_nodes_on_re_indicator": geom["n_expected"],
        "source_facets_on_re": surface.n_facets,
        "bed_min_m": lo, "bed_max_m": hi, "bed_nodal_rms_m": rms_nodal,
        "largest_gap_to_source_surface_over_Re": gap_max,
        "smallest_barycentric": score_min,
        "coincident": coincident,
        "rms": rms,
    }
    tic = time.time()
    common.append_control(target, parent, written, 0,
                          {"J": source_info["J"],
                           "rms_misfit_m": source_info["rms_misfit_m"]})
    with CheckpointFile(target, "a") as chk:
        common.write_json_dataset(chk, TRANSFER_RECORD, transfer_record)
    timings["append_s"] = time.time() - tic
    say(f"  appended {common.STATE_CONTROL_NAME} idx 0 and "
        f"{TRANSFER_RECORD} to the target ({timings['append_s']:.1f} s)")

    # The entry must load back as the bed: the step of `read_inputs(...,
    # last_control=True)` that reads it (parent field, `assign` to the
    # mantle), compared at every mantle node.
    tic = time.time()
    with CheckpointFile(target, "r") as chk:
        history = chk.get_timestepping_history(parent,
                                               common.STATE_CONTROL_NAME)
        stored = chk.load_function(parent, common.STATE_CONTROL_NAME,
                                   idx=int(max(history["index"])))
    back = Function(written.function_space()).assign(stored)
    reload_diff = COMM_WORLD.allreduce(
        float(np.abs(back.dat.data_ro - written.dat.data_ro).max())
        if written.dat.data_ro.size else 0.0, op=MPI.MAX)
    timings["verify_s"] = time.time() - tic
    if reload_diff != 0.0:
        raise RuntimeError(f"the appended bed loads back with a largest "
                           f"difference of {reload_diff} m")
    say(f"  reload: the appended entry is the bed bit for bit "
        f"({timings['verify_s']:.1f} s)")

    identity_after = None
    if COMM_WORLD.rank == 0:
        identity_after = file_identity(target, args.target_sha256 == "yes")
    identity_after = COMM_WORLD.bcast(identity_after, root=0)
    timings["total_s"] = time.time() - tic_run
    timings["peak_memory_gb"] = common.peak_memory_gb()
    transfer_record.update({
        "target_before": identity["target_before"],
        "target_after": identity_after,
        "reload_largest_difference_m": reload_diff,
        "timings": timings,
    })
    stem = os.path.splitext(os.path.basename(target))[0]
    json_path = os.path.join(out_dir, f"{stem}_transfer_bed.json")
    common.write_json(json_path, transfer_record)
    say(f"  record {json_path}")
    say(f"  total {timings['total_s']:.1f} s, peak memory "
        f"{timings['peak_memory_gb']:.2f} GB per rank")


if __name__ == "__main__":
    main()
