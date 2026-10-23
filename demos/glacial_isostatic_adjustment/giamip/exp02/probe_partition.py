r"""What each part of the mantle-balanced partition of a GIAMIP mesh holds.

THE QUESTION
    The 8 Re mesh after 4 refinement rounds (`giamip_8re_r4`) ran out of
    memory on one rank per run: rank 1461 of 1664 and rank 1826 of 2080,
    both at 0.878 of the rank range, held 8.5 GB against a rank mean of
    2.5-2.6 GB. This probe tells what the parts of the partition hold, so
    that a heavy rank of a run can be put next to its part.

WHAT IT DOES
    It loads the P1 mesh file as `exp02_common.load_checkpoint_meshes` does
    (the DMPlex topology, labels and coordinates, spread over the ranks of
    this job in contiguous pieces of the file, no Firedrake mesh), and runs
    `exp02_common.checkpoint_balanced_partition` for each requested number of
    parts. The number of parts need not equal the number of ranks of this
    job: the partition depends only on the cell order of the file and on the
    Hilbert keys of the cell directions, not on the ranks that compute it.
    So one node can probe the partition of a run on 16 or 20 nodes.

    For each part it counts:

    - the cells by region: `mantle`, `buffer` (between the surface and the
      outer boundary), `inner` (the shell between the inner sphere and the
      CMB), and `other` (every cell that is not mantle);
    - the cells with a face on the surface (`surface_cells`, facet tag 2),
      on the CMB (`cmb_cells`, 3) and on the outer boundary (`outer_cells`,
      4, buffer side only);
    - the mean direction of its mantle cells (`lat`, `lon`, degrees) and the
      spread `1 - |sum of unit directions| / count` (0 for a point, near 1 for
      cells all over the sphere);
    - the smallest and largest mantle cell volume (`vol_min`, `vol_max`,
      from the P1 vertices, in the units of the mesh);
    - with `--dofs`: the P2 nodes (vertices and edges) in the closure of its
      cells, of all cells (`p2_all`, the gravitational potential lives on
      the whole mesh) and of its mantle cells (`p2_mantle`, the
      displacement), each node counted once per part. The degrees of freedom
      of a run are about `p2_all + 3 p2_mantle + 24 mantle` (the DG1 internal
      variables are a symmetric 3x3 tensor, 6 components at 4 nodes: 24
      values per cell, as the owned dofs of the L0 run show), in
      `closure_dofs`. This
      counts the nodes that a part shares with its neighbours once for each
      part, so it is an upper bound of the owned degrees of freedom; the
      owned share of a node is decided by the distribution of the run.

    The partition and the counts are collective. All per-part sums use
    buffer reductions (`Allreduce` on NumPy arrays): a pickled `allreduce`
    of arrays fails with `MPI.MIN` and `MPI.MAX`, because mpi4py compares the
    arrays with Python's `min` and `max`.

OUTPUT
    The log lines on rank 0 (stdout): per number of parts the min, mean and
    max of each count, the parts with the most `other` cells and the most
    closure dofs, and the parts named with `--show`, each against the mean
    part. `--output` (default `probe_partition.json` in the working
    directory) gets every count of every part.

RUN (one node of Gadi normalsr; about 2 SU for the 22 M cell mesh)
    mpiexec -np 104 python3 probe_partition.py \
        --mesh <dir>/giamip_8re_r4_linear.h5 --mesh_name giamip_8re_r4_linear \
        --sidecar <dir>/giamip_8re_r4.json --parts 1664,2080 \
        --show 1664:1461,2080:1826 --dofs
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# The shared module imports gadopt before firedrake.
import exp02_common as common  # noqa: E402
from giamsh.radial import RadialStructure  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

COMM = MPI.COMM_WORLD


def say(text):
    """Print on rank 0, flushed (the job output is read while it runs)."""
    if COMM.rank == 0:
        print(text, flush=True)


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="the P1 mesh file (.h5)")
    p.add_argument("--mesh_name", required=True,
                   help="the name of the mesh in the file")
    p.add_argument("--sidecar", required=True,
                   help="the JSON sidecar of the mesh (with 'radial')")
    p.add_argument("--parts", required=True,
                   help="comma-separated numbers of parts, e.g. 1664,2080")
    p.add_argument("--show", default="",
                   help="parts to print in full, as nparts:part,... "
                        "(default: the part at 0.878 of the range)")
    p.add_argument("--dofs", action="store_true",
                   help="also count the P2 nodes in the closure of each part")
    p.add_argument("--output", default="probe_partition.json")
    return p.parse_args()


def load_plex(path, mesh_name):
    """The DMPlex of the file, as `load_checkpoint_meshes` loads it (collective)."""
    topology_name = common.checkpoint_topology_name(path, mesh_name)
    viewer = PETSc.ViewerHDF5().create(path, mode="r", comm=COMM)
    plex = PETSc.DMPlex().create(comm=COMM)
    plex.setName(topology_name)
    viewer.pushFormat(format=PETSc.Viewer.Format.HDF5_PETSC)
    plex.distributionSetName(None)
    sfXB = plex.topologyLoad(viewer)
    plex.labelsLoad(viewer, sfXB)
    plex.coordinatesLoad(viewer, sfXB)
    viewer.popFormat()
    viewer.destroy()
    return plex


def cell_mask(plex, label, tags):
    """A mask of the local cells that carry one of `tags` in `label`."""
    cstart, cend = plex.getHeightStratum(0)
    mask = np.zeros(cend - cstart, dtype=bool)
    for tag in tags:
        if plex.getStratumSize(label, tag):
            points = np.asarray(plex.getStratumIS(label, tag).getIndices())
            # The refined mesh tags faces, edges and vertices too: keep cells.
            points = points[(points >= cstart) & (points < cend)]
            mask[points - cstart] = True
    return mask


def touching(plex, tag):
    """A mask of the local cells with a face in `Face Sets` = `tag`."""
    cstart, cend = plex.getHeightStratum(0)
    fstart, fend = plex.getHeightStratum(1)
    mask = np.zeros(cend - cstart, dtype=bool)
    if plex.getStratumSize("Face Sets", tag):
        for face in plex.getStratumIS("Face Sets", tag).getIndices():
            if not fstart <= face < fend:
                continue
            for cell in plex.getSupport(face):
                if cstart <= cell < cend:
                    mask[cell - cstart] = True
    return mask


def summed(values, op=MPI.SUM):
    """A buffer reduction of a float array over all ranks, in place, returned."""
    values = np.ascontiguousarray(values, dtype=float)
    COMM.Allreduce(MPI.IN_PLACE, values, op=op)
    return values


def closure_nodes(plex):
    """The global numbers of the P2 nodes (vertices, edges) of each local cell.

    Returns:
      An int64 array of shape (cells, 10) for tetrahedra: 4 vertices and 6
      edges, as global point numbers of the DMPlex.
    """
    cstart, cend = plex.getHeightStratum(0)
    vstart, vend = plex.getDepthStratum(0)
    estart, eend = plex.getDepthStratum(1)
    numbering = np.asarray(plex.createPointNumbering().getIndices(),
                           dtype=np.int64)
    # A point owned by another rank has the number -(g + 1).
    numbering = np.where(numbering < 0, -numbering - 1, numbering)
    pstart, _ = plex.getChart()
    rows = []
    for cell in range(cstart, cend):
        points = plex.getTransitiveClosure(cell)[0]
        keep = points[((points >= vstart) & (points < vend))
                      | ((points >= estart) & (points < eend))]
        rows.append(keep)
    if not rows:
        return np.zeros((0, 10), dtype=np.int64)
    local = np.array(rows, dtype=np.int64)
    return numbering[local - pstart]


def count_unique_per_part(part, nodes, nparts):
    """The number of distinct nodes per part, over all ranks.

    Each (part, node) pair is sent to the rank `part % size`, which counts
    the distinct pairs of its parts. Collective.

    Args:
      part: the part of each local cell.
      nodes: the global node numbers of each local cell, (cells, k).
      nparts: the number of parts.

    Returns:
      A float array of `nparts` counts (on every rank).
    """
    size = COMM.size
    top = int(COMM.allreduce(int(nodes.max()) + 1 if nodes.size else 0,
                             op=MPI.MAX))
    keys = (np.repeat(part, nodes.shape[1]).astype(np.int64) * top
            + nodes.ravel())
    keys = np.unique(keys)
    destination = (keys // top) % size
    order = np.argsort(destination, kind="stable")
    keys = keys[order]
    send_counts = np.bincount(destination, minlength=size).astype(np.int64)
    recv_counts = np.empty(size, dtype=np.int64)
    COMM.Alltoall(send_counts, recv_counts)
    send_displ = np.concatenate([[0], np.cumsum(send_counts)[:-1]])
    recv_displ = np.concatenate([[0], np.cumsum(recv_counts)[:-1]])
    received = np.empty(int(recv_counts.sum()), dtype=np.int64)
    COMM.Alltoallv([keys, send_counts, send_displ, MPI.INT64_T],
                   [received, recv_counts, recv_displ, MPI.INT64_T])
    received = np.unique(received)
    counts = np.bincount(received // top, minlength=nparts).astype(float)
    return summed(counts)


def main():
    """Load the mesh, partition it for each number of parts, count, report."""
    args = parse_args()
    tic = time.time()
    sidecar = json.load(open(args.sidecar))
    radial = RadialStructure.from_dict(sidecar["radial"])
    # The cell tags of each region; a mesh without a buffer or an inner
    # shell has no such region.
    regions = {}
    for name in ("mantle", "buffer", "inner"):
        try:
            regions[name] = list(radial.region_tags(name))
        except KeyError:
            pass
    plex = load_plex(args.mesh, args.mesh_name)
    cstart, cend = plex.getHeightStratum(0)
    n = cend - cstart
    say(f"loaded {args.mesh} on {COMM.size} ranks: "
        f"{COMM.allreduce(n, op=MPI.SUM)} cells ({time.time() - tic:.1f} s)")

    # The vertices of each cell: centroid and volume (P1, flat cells).
    section = plex.getCoordinateSection()
    coordinates = plex.getCoordinatesLocal()
    corners = np.array([plex.vecGetClosure(section, coordinates, c).reshape(-1, 3)
                        for c in range(cstart, cend)]).reshape(n, 4, 3)
    centroid = corners.mean(axis=1)
    volume = np.abs(np.einsum(
        "ij,ij->i", corners[:, 1] - corners[:, 0],
        np.cross(corners[:, 2] - corners[:, 0],
                 corners[:, 3] - corners[:, 0]))) / 6.0
    direction = centroid / np.linalg.norm(centroid, axis=1)[:, None]
    masks = {name: cell_mask(plex, "Cell Sets", tags)
             for name, tags in regions.items()}
    mantle = masks["mantle"]
    # The facet tags of the experiment: surface Re, CMB, outer boundary.
    surface_cells = touching(plex, common.SURF_RE)
    cmb_cells = touching(plex, common.SURF_RC)
    outer_cells = touching(plex, common.SURF_OUTER) & ~mantle
    nodes = closure_nodes(plex) if args.dofs else None
    say(f"cell masks{' and closures' if args.dofs else ''} "
        f"({time.time() - tic:.1f} s)")

    show = {}
    for item in filter(None, args.show.split(",")):
        nparts, part = (int(v) for v in item.split(":"))
        show.setdefault(nparts, []).append(part)

    out = {}
    for nparts in [int(v) for v in args.parts.split(",")]:
        sizes, points = common.checkpoint_balanced_partition(
            plex, nparts, regions["mantle"])
        # The part of each local cell: `points` lists the local cells sorted
        # by part, `sizes` the count of each part.
        owner = np.empty(n, dtype=np.int64)
        owner[np.asarray(points)] = np.repeat(np.arange(nparts), sizes)

        def per(mask, weights=None):
            w = np.ones(n) if weights is None else weights
            return summed(np.bincount(owner[mask], weights=w[mask],
                                      minlength=nparts))

        rows = {"mantle": per(mantle), "other": per(~mantle)}
        for name in ("buffer", "inner"):
            if name in masks:
                rows[name] = per(masks[name])
        rows.update(surface_cells=per(surface_cells), cmb_cells=per(cmb_cells),
                    outer_cells=per(outer_cells))
        vmin = np.full(nparts, np.inf)
        vmax = np.zeros(nparts)
        np.minimum.at(vmin, owner[mantle], volume[mantle])
        np.maximum.at(vmax, owner[mantle], volume[mantle])
        vmin, vmax = summed(vmin, MPI.MIN), summed(vmax, MPI.MAX)
        dsum = np.stack([per(mantle, direction[:, i]) for i in range(3)], axis=1)
        norm = np.linalg.norm(dsum, axis=1)
        lat = np.degrees(np.arcsin(dsum[:, 2] / np.maximum(norm, 1e-30)))
        lon = np.degrees(np.arctan2(dsum[:, 1], dsum[:, 0]))
        spread = 1.0 - norm / np.maximum(rows["mantle"], 1.0)
        if nodes is not None:
            rows["p2_all"] = count_unique_per_part(owner, nodes, nparts)
            rows["p2_mantle"] = count_unique_per_part(owner[mantle],
                                                      nodes[mantle], nparts)
            rows["closure_dofs"] = (rows["p2_all"] + 3.0 * rows["p2_mantle"]
                                    + 24.0 * rows["mantle"])
        if COMM.rank != 0:
            continue
        out[nparts] = {k: v.tolist() for k, v in rows.items()}
        out[nparts].update(vol_min=vmin.tolist(), vol_max=vmax.tolist(),
                           lat=lat.tolist(), lon=lon.tolist(),
                           spread=spread.tolist())
        say(f"\nnparts {nparts} ({time.time() - tic:.1f} s): min / mean / max")
        for key, values in rows.items():
            say(f"  {key:>13s}: {values.min():12.0f} {values.mean():12.0f} "
                f"{values.max():12.0f} (part {int(values.argmax())})")
        say(f"  {'vol_min':>13s}: {vmin.min():.3e} .. {vmin.max():.3e}; "
            f"vol_max {vmax.min():.3e} .. {vmax.max():.3e}")
        picks = set(show.get(nparts, [int(0.878 * nparts)]))
        picks.add(int(rows["other"].argmax()))
        if "closure_dofs" in rows:
            picks.add(int(rows["closure_dofs"].argmax()))
        for p in sorted(picks):
            ratio = ", ".join(
                f"{k} {rows[k][p]:.0f} ({rows[k][p] / max(rows[k].mean(), 1e-30):.2f}x)"
                for k in rows)
            say(f"  part {p}: {ratio}; lat {lat[p]:.1f} lon {lon[p]:.1f} "
                f"spread {spread[p]:.3e} vol {vmin[p]:.2e}..{vmax[p]:.2e}")
    if COMM.rank == 0:
        with open(args.output, "w") as handle:
            json.dump(out, handle)
        say(f"\nwrote {args.output} ({time.time() - tic:.1f} s)")


if __name__ == "__main__":
    main()
