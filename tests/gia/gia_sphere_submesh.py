r"""Driver of the weekly HPC entry: `SEA` on a tetrahedral sphere with a mantle submesh.

Usage: `gadopt_hpcrun -n 48 ... python3 gia_sphere_submesh.py --entry
sphere-submesh`, which `doit run_case_hpc` submits (`gia_entries.
_weekly_entries`); by hand, `mpiexec -n <ranks> python3 gia_sphere_submesh.py
--entry sphere-submesh` from this directory.

The entry is the 3-D counterpart of the forward entry `sea-iterative`: the
configuration `gia_march.SEA` (the Quadrature history with BDF2, the
low-rank DtN, two Maxwell elements, a fluid core, rotation, the
centre-of-mass rows and a sea level with a moving coastline) under the
iterative preset with its own tolerances, on the geometry of the GIAMIP
drivers:

    0.5 Rc --- inner (102) --- Rc --- mantle (101) --- Re --- buffer (103) --- 2 Re
    surface 5              surface 3              surface 2               surface 4
    interior DtN           interior facet         interior facet          exterior DtN

The potential lives on the whole mesh (the parent), the displacement on the
`Submesh` of the mantle cells. The radii are those of the 2-D annulus of
`gia_cases` (non-dimensional by the mantle depth, `Rc = 3480 / 2891`), and
the tags are the same numbers, so `gia_cases.SeaFields.kwargs` and
`gia_march.diagnostics` read the sphere with no change.

Rank 0 writes the mesh with gmsh in the job (`write_sphere`), with a coarse
graded size field and the next gmsh seed while a seed gives a flat cell
(`flat_cell_count`). The plex is distributed with `balanced_partition`,
which gives every rank the same number of mantle cells: the default
partitioner gives some of 48 ranks buffer cells only, and Firedrake fails on
a rank with an empty piece of the submesh. Both meshes are curved to P2 and
checked for folded cells. The mesh code follows
`examples/common/selfgrav_common.py` of the `gia-mip` repository, reduced to
the four spheres of the layout above.

In 3-D the `Rotation` coupling has three rows: the polar-wander pair `m1`,
`m2` and the rotation-rate change `m3`. The ice cap and the reference sea
level of the 2-D `SEA` scenario are turned onto the axis `LOAD_AXIS`, which
lies off every coordinate axis and coordinate plane, so that every row of
the frame and of the rotation is loaded (`gia_entries.WEEKLY_COUPLING_
COLUMNS`). The march is the first `WEEKLY_STEP_COUNT` steps of
`gia_march.FORWARD_STEPS` with the ice history of `gia_march.ice_at` on that
axis.

Outputs: one row of `gia_march.diagnostics` per step, with the columns `m1`
and `m2` added, in `out/sphere-submesh.log` (integrals, norms, `Real`
values and counts only, so the row does not depend on the partition), and
the record of the mesh in `out/sphere-submesh-mesh.dat`. `test_gia_hpc.py`
reads them; this script asserts nothing.
"""

import argparse
import hashlib
import time

import gadopt  # imported before firedrake
import firedrake as fd
import numpy as np
from firedrake.adjoint import stop_annotating
from gadopt import gia_couplings as cp
from mpi4py import MPI
from petsc4py import PETSc

import gia_cases as C
import gia_march as M
from gia_entries import (CASE_DIR, WEEKLY_ENTRY, WEEKLY_LOG, WEEKLY_MESH_FILE,
                         WEEKLY_MESH_RECORD, WEEKLY_STEP_COUNT)
from gia_forward import SPHERE_K_S, SPHERE_OMEGA_SQ

# -- Geometry ---------------------------------------------------------------------

#: The core-mantle boundary and the Earth radius, 3480 km and 6371 km over
#: the mantle depth of 2891 km: the radii of the 2-D annulus of `gia_cases`
#: and of the GIAMIP meshes.
RC = 3480.0 / 2891.0
RE = 6371.0 / 2891.0
#: The two DtN spheres: half the core radius inside, twice the Earth radius
#: outside, so that the exterior gravity condition sits at a distance from
#: the load and from the CMB.
R_INNER = 0.5 * RC
R_OUTER = 2.0 * RE
#: The cell groups of the three shells. The mantle tag is the one of the
#: annulus (`gia_cases.CELL_MANTLE`).
CELL_MANTLE, CELL_INNER, CELL_BUFFER = C.CELL_MANTLE, 102, 103

#: The lateral cell size in the mantle, in units of the mantle depth (about
#: 870 km). The mantle shell then holds of the order of 10 000 cells (an
#: estimate from the volume of a tetrahedron of this size; the mesh record
#: gives the count), a few hundred per rank on one 48-core node: a coarse
#: mesh on which every rank still owns mantle cells.
H_MANTLE = 0.3
#: The growth of the cell size per unit distance away from the mantle, in
#: the buffer and in the inner shell, as in the GIAMIP meshes.
GRADE = 2.0
#: The smallest number of cells around a great circle of any sphere: the cap
#: `2 pi r / MIN_CELLS` on the size keeps the small inner DtN sphere
#: resolved, where the grading alone would make it coarse.
MIN_CELLS = 16
#: The first gmsh seed and the number of seeds tried. The 3-D Delaunay
#: algorithm of gmsh (`Mesh.Algorithm3D` 1) places points at random, and some
#: seeds give a flat cell (`flat_cell_count`). A seed is not portable between
#: machines, so the job tries the next seeds until one gives none; the mesh
#: record names the seed it used.
FIRST_SEED, SEED_TRIES = 2, 40
#: The truncation degree of the DtN map on both spheres. The load has
#: harmonic degrees 1 and 2, which reach the exterior sphere; a degree-4
#: field there is smaller than at Re by `(Re / R_OUTER)^5 = 1/32`.
DTN_DEGREE = 3

#: The axis of the ice cap and of the reference sea level: colatitude 45
#: degrees, longitude 30 degrees. All three components are nonzero, so the
#: first moment of the load has a part along each coordinate axis (the three
#: centre-of-mass rows), and the second moment, proportional to
#: `a a^T - I / 3`, has nonzero `x z`, `y z` parts (the polar-wander rows
#: `m1`, `m2`) and a nonzero zonal part `a_z^2 - 1/3` (the rate row `m3`).
LOAD_AXIS = (np.sin(np.pi / 4) * np.cos(np.pi / 6),
             np.sin(np.pi / 4) * np.sin(np.pi / 6),
             np.cos(np.pi / 4))


# -- Mesh --------------------------------------------------------------------------

def write_sphere(path, seed):
    """Write the four-region sphere to the gmsh file `path` (serial, rank 0).

    Each sphere of `0.5 Rc, Rc, Re, 2 Re` is an OpenCASCADE ball, and
    `occ.fragment` cuts them into nested shells that share their bounding
    surfaces. The core ball inside `0.5 Rc` is removed without its boundary,
    which stays as the inner DtN surface. A surface is identified by its
    radius `sqrt(A / 4 pi)` and a shell by the radii of its boundary, so
    nothing depends on the order in which gmsh returns entities.

    The target size grows linearly away from the mantle,
    `h(r) = H (1 + GRADE max(0, r - Re) + GRADE max(0, Rc - r))`, and a
    second field caps it at `2 pi r / MIN_CELLS`; gmsh takes the smaller.

    Args:
      path: the output `.msh` file.
      seed: the gmsh random seed of the 3-D Delaunay algorithm.

    Returns:
      `{cell tag: cell count}` of the three shells.
    """
    import gmsh

    radii = [R_INNER, RC, RE, R_OUTER]
    gmsh.initialize()
    # gmsh stays initialised with a dirty model after an error, and the
    # next try in the same process writes an empty mesh; finalize always.
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gia_sphere_submesh")
        occ = gmsh.model.occ
        balls = [occ.addSphere(0, 0, 0, R) for R in radii]
        occ.fragment([(3, balls[0])], [(3, b) for b in balls[1:]])
        occ.synchronize()

        def snap(R):
            """The radius of `radii` that a radius recovered from a measure is."""
            match = [r for r in radii if abs(r - R) < 1e-6 * r]
            assert len(match) == 1, f"radius {R} is not one of {radii}"
            return match[0]

        def surface_radius(tag):
            return snap(np.sqrt(occ.getMass(2, tag) / (4 * np.pi)))

        # The core ball: the volume whose measure is that of the inner ball.
        core = [t for _, t in gmsh.model.getEntities(3)
                if abs(occ.getMass(3, t) - 4 / 3 * np.pi * R_INNER ** 3) < 1e-8]
        assert len(core) == 1, f"expected one core ball, found {core}"
        occ.remove([(3, core[0])], recursive=False)
        occ.synchronize()

        surfaces = {}
        for _, t in gmsh.model.getEntities(2):
            surfaces.setdefault(surface_radius(t), []).append(t)
        shells = {}
        for _, t in gmsh.model.getEntities(3):
            bounding = sorted(surface_radius(abs(s)) for _, s
                              in gmsh.model.getBoundary([(3, t)], oriented=True))
            shells[(bounding[0], bounding[-1])] = t
        # The facet tags of the annulus: Re 2, Rc 3, outer DtN 4, inner DtN 5.
        for R, tag in ((RE, C.CURVE_RE), (RC, C.CURVE_RC),
                       (R_OUTER, C.CURVE_OUTER), (R_INNER, C.CURVE_INNER)):
            gmsh.model.addPhysicalGroup(2, surfaces[R], tag)
        for (r_in, r_out), tag in (((R_INNER, RC), CELL_INNER), ((RC, RE), CELL_MANTLE),
                                   ((RE, R_OUTER), CELL_BUFFER)):
            gmsh.model.addPhysicalGroup(3, [shells[(r_in, r_out)]], tag)

        # The graded size and the angular cap, merged by `Min`.
        r = "sqrt(x*x+y*y+z*z)"
        graded = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            graded, "F", f"{H_MANTLE}*(1 + {GRADE}*max(0, {r} - {RE!r})"
                         f" + {GRADE}*max(0, {RC!r} - {r}))")
        capped = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(capped, "F", f"{2 * np.pi / MIN_CELLS!r}*{r}")
        merged = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(merged, "FieldsList", [graded, capped])
        gmsh.model.mesh.field.setAsBackgroundMesh(merged)
        # Without these, the curvature and point rules of gmsh override the
        # size field near the small inner sphere and refine it many times.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", seed)
        gmsh.model.mesh.generate(3)

        cells = {}
        for tag in (CELL_MANTLE, CELL_INNER, CELL_BUFFER):
            cells[tag] = 0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                _, element_tags, _ = gmsh.model.mesh.getElements(3, entity)
                cells[tag] += sum(len(e) for e in element_tags)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return cells


def flat_cell_count(path):
    """The number of tetrahedra in `path` with all four vertices on one sphere.

    The P2 curving (`curve_sphere`) puts every node of such a cell on that
    sphere. The cell has almost no volume, and its Jacobian determinant can
    change sign inside it, which gives a wrong answer without an error. A
    cell is flat when the spread of its four vertex radii is below 1e-6 of
    their mean; the smallest gap between two spheres of this mesh is 0.6, so
    no other cell passes.

    Args:
      path: the `.msh` file (serial, rank 0).
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        xyz = np.zeros((int(node_tags.max()) + 1, 3))
        xyz[node_tags.astype(int)] = coordinates.reshape(-1, 3)
        radius = np.linalg.norm(xyz, axis=1)
        count = 0
        for tag in (CELL_MANTLE, CELL_INNER, CELL_BUFFER):
            for entity in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                _, _, element_nodes = gmsh.model.mesh.getElements(3, entity)
                for nodes in element_nodes:
                    r4 = radius[nodes.reshape(-1, 4).astype(int)]
                    count += int(np.count_nonzero(
                        np.ptp(r4, axis=1) < 1e-6 * r4.mean(axis=1)))
    finally:
        gmsh.finalize()
    return count


#: The bits per axis of the Hilbert curve that orders the cell directions in
#: `balanced_partition`: 1024 steps across the unit cube, far finer than the
#: cells, so almost no two mantle cells share a key.
HILBERT_BITS = 10


def hilbert_keys(ijk, bits):
    """The index of integer points along a 3-D Hilbert curve.

    The transpose form of Skilling (2004), "Programming the Hilbert curve",
    AIP Conf. Proc. 707, 381, applied to all points at once: undo the excess
    work, Gray-code the result, and interleave the bits of the three axes
    into one key. Points close along the curve are close in space, so an
    equal split of the sorted keys gives compact pieces.

    Args:
      ijk: an integer array of shape `(n, 3)` with values in `[0, 2**bits)`.
      bits: the bits per axis, at most 21 so that a key fits in int64.

    Returns:
      An int64 array of `n` keys.
    """
    X = np.array(ijk, dtype=np.int64)
    M_top = np.int64(1) << (bits - 1)
    # Inverse undo of the excess work, from the top bit down.
    Q = M_top
    while Q > 1:
        P = Q - 1
        for i in range(3):
            high = (X[:, i] & Q) != 0
            # Where the bit is set, invert the low bits of the first axis.
            X[high, 0] ^= P
            # Elsewhere, exchange the low bits of the first axis and axis i.
            low = ~high
            t = (X[low, 0] ^ X[low, i]) & P
            X[low, 0] ^= t
            X[low, i] ^= t
        Q >>= 1
    # Gray encode.
    for i in range(1, 3):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(len(X), dtype=np.int64)
    Q = M_top
    while Q > 1:
        t[(X[:, 2] & Q) != 0] ^= Q - 1
        Q >>= 1
    X ^= t[:, None]
    # Interleave: the top bit of each axis first, axis 0 before axis 1.
    key = np.zeros(len(X), dtype=np.int64)
    for b in range(bits - 1, -1, -1):
        for i in range(3):
            key = (key << 1) | ((X[:, i] >> b) & 1)
    return key


def balanced_partition(plex, nparts):
    """A partition of the cells that gives every rank the same number of mantle cells.

    Each cell is ordered by the Hilbert key of the direction of its
    centroid. The mantle cells, in that order, are split into `nparts`
    pieces of equal count, and each inner or buffer cell goes to the rank
    whose mantle piece covers its key. Every rank then owns a solid-angle
    sector through all three shells: the mantle work, most of the cost, is
    balanced to one cell, and no rank has an empty piece of the submesh.

    Collective. Only rank 0 holds cells, because the gmsh reader loads the
    whole file there.

    Args:
      plex: the serial `DMPlex` read from the gmsh file, on `COMM_WORLD`.
      nparts: the number of ranks.

    Returns:
      `(sizes, points)` for `distribution_parameters["partition"]`: the
      number of cells of each rank and the cell points ordered by rank; all
      zero and empty on a rank that holds no cells.
    """
    cstart, cend = plex.getHeightStratum(0)
    if cend == cstart:
        return (np.zeros(nparts, dtype=PETSc.IntType), np.zeros(0, dtype=PETSc.IntType))
    # The centroid of each tetrahedron: the mean of its four vertices.
    section = plex.getCoordinateSection()
    coordinates = plex.getCoordinatesLocal()
    centroid = np.array([plex.vecGetClosure(section, coordinates, c).reshape(-1, 3).mean(axis=0)
                         for c in range(cstart, cend)])
    # The direction as integer coordinates in [0, 2^bits) on each axis.
    direction = centroid / np.linalg.norm(centroid, axis=1)[:, None]
    scale = (1 << HILBERT_BITS) - 1
    key = hilbert_keys(np.rint((direction + 1.0) * 0.5 * scale), HILBERT_BITS)
    mantle = np.zeros(cend - cstart, dtype=bool)
    mantle[np.asarray(plex.getStratumIS("Cell Sets", CELL_MANTLE).getIndices()) - cstart] = True
    # Split the mantle cells, in key order, into pieces of equal count; the
    # stable sort keeps equal keys in cell order, so the split is
    # deterministic.
    part = np.empty(cend - cstart, dtype=np.int64)
    mantle_index = np.flatnonzero(mantle)
    order = mantle_index[np.argsort(key[mantle_index], kind="stable")]
    part[order] = (np.arange(len(order)) * nparts) // len(order)
    # The key at which each mantle piece starts; an inner or buffer cell
    # goes to the last piece that starts at or below its key.
    first = np.searchsorted(part[order], np.arange(nparts), side="left")
    starts = key[order][first]
    other = np.flatnonzero(~mantle)
    part[other] = np.clip(np.searchsorted(starts, key[other], side="right") - 1, 0, nparts - 1)
    sizes = np.bincount(part, minlength=nparts).astype(PETSc.IntType)
    points = (np.argsort(part, kind="stable") + cstart).astype(PETSc.IntType)
    return sizes, points


def curve_sphere(linear_mesh, name):
    """Remap a straight-sided mesh to P2 coordinates, so its spheres are curved.

    Each P2 node moves radially onto the radius of the linear interpolant of
    the vertex radii, `X_p2 = (r_p1 / r) X`: an edge with both vertices on a
    sphere becomes a quadratic arc on it. The 3-D form of
    `gia_cases.curve_mesh`, which uses the radius in the plane.

    Args:
      linear_mesh: a mesh with P1 coordinates and no vertex at the origin.
      name: the name of the new mesh.
    """
    X = fd.SpatialCoordinate(linear_mesh)
    r = fd.sqrt(fd.dot(X, X))
    r_p1 = fd.Function(fd.FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = fd.Function(fd.VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate((r_p1 / r) * X)
    return fd.Mesh(X_p2, name=name)


def folded_cell_count(mesh):
    """The number of curved cells whose Jacobian determinant is not of one sign.

    On a P2 tetrahedron the Jacobian determinant is a cubic, so its DG3
    interpolant is exact, and the 20 DG3 nodes of a cell are 20 samples of
    it. A valid cell has one nonzero sign at all of them. Collective.

    Args:
      mesh: a mesh with P2 coordinates.
    """
    space = fd.FunctionSpace(mesh, "DG", 3)
    det = fd.Function(space).interpolate(fd.JacobianDeterminant(mesh))
    # The DG3 nodes of the owned cells, which come first in the cell-node
    # table; a DG node belongs to one cell only.
    owned = mesh.cells.owned.local_size
    samples = det.dat.data_ro_with_halos[space.cell_node_list[:owned]]
    local = int(np.count_nonzero(samples.min(axis=1) * samples.max(axis=1) <= 0.0))
    return mesh.comm.allreduce(local, op=MPI.SUM)


def _md5(path):
    """The MD5 of a file, as a hexadecimal string."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sphere_meshes():
    """Write, check, read, distribute and curve the sphere; return the two meshes.

    Rank 0 writes the gmsh file with the seeds from `FIRST_SEED` until one
    gives no flat cell, while the other ranks wait in the broadcast. Rank 0
    then reads the file into a plex, which `Mesh` distributes with
    `balanced_partition`. The mantle `Submesh` is cut from the curved parent
    and curved itself, because `Submesh` does not inherit the P2
    coordinates; the two coordinate fields agree on the shared facets.

    Returns:
      `(parent, mantle, record)`, with `record` the mesh record that
      `write_mesh_record` writes.

    Raises:
      RuntimeError: if every seed gives a flat cell or a curved cell folds.
    """
    comm = fd.COMM_WORLD
    path = CASE_DIR / WEEKLY_MESH_FILE
    record, error = {}, None
    if comm.rank == 0:
        import gmsh
        start = time.time()
        # An error on rank 0 must reach the other ranks, or they wait in the
        # broadcast below for ever.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            flat = {}
            for seed in range(FIRST_SEED, FIRST_SEED + SEED_TRIES):
                cells = write_sphere(path, seed)
                flat[seed] = flat_cell_count(path)
                if flat[seed] == 0:
                    break
            record = {"gmsh_version": gmsh.__version__, "random_seed": seed,
                      "seeds_tried": len(flat), "flat_cells": flat[seed],
                      "md5": _md5(path), "cells_mantle": cells[CELL_MANTLE],
                      "cells_inner": cells[CELL_INNER], "cells_buffer": cells[CELL_BUFFER],
                      "generation_s": round(time.time() - start, 1)}
        except Exception as exc:  # noqa: BLE001  (raised again on every rank)
            error = f"mesh generation failed on rank 0: {exc!r}"
    record, error = comm.bcast((record, error), root=0)
    if error is not None:
        raise RuntimeError(error)
    if record["flat_cells"]:
        raise RuntimeError(f"{path}: every one of {SEED_TRIES} gmsh seeds gives a flat cell")

    plex = PETSc.DMPlex().createFromFile(str(path), comm=comm)
    plex.setName("sphere_submesh_topology")
    sizes, points = balanced_partition(plex, comm.size)
    # The distribution gets a name of its own; without one, Firedrake names
    # it after the text of the partition arrays.
    linear = fd.Mesh(plex, distribution_parameters={"partition": (sizes, points)},
                     distribution_name=f"balanced_partition_{comm.size}")
    parent = curve_sphere(linear, "sphere_parent")
    # The geometry is a sphere: the vertical of the G-ADOPT terms is radial.
    parent.cartesian = False
    mantle = curve_sphere(fd.Submesh(parent, 3, CELL_MANTLE), "sphere_mantle")
    mantle.cartesian = False
    record["folded_cells_parent"] = folded_cell_count(parent)
    record["folded_cells_mantle"] = folded_cell_count(mantle)
    if record["folded_cells_parent"] or record["folded_cells_mantle"]:
        raise RuntimeError(f"{path}: curved cells fold: {record}")
    counts = comm.allgather(int(mantle.cells.owned.local_size))
    record.update({"ranks": comm.size, "mantle_cells_per_rank_min": min(counts),
                   "mantle_cells_per_rank_max": max(counts)})
    return parent, mantle, record


def write_mesh_record(record):
    """Write the mesh record as `name value` lines (rank 0) and print it."""
    PETSc.Sys.Print(f"mesh: {record}")
    if fd.COMM_WORLD.rank != 0:
        return
    path = CASE_DIR / WEEKLY_MESH_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name} {value}\n" for name, value in record.items()))


# -- Solver and load ----------------------------------------------------------------

def along_axis(X):
    """`cos` of the angle between the position `X` and `LOAD_AXIS`, as UFL."""
    a = fd.as_vector(LOAD_AXIS)
    return fd.dot(X, a) / fd.sqrt(fd.dot(X, X))


class SphereSeaFields(C.SeaFields):
    """The prescribed sea-level fields of `SEA`, turned onto `LOAD_AXIS`.

    The 2-D fields of `gia_cases.SeaFields` with `cos theta` replaced by the
    cosine of the angle from `LOAD_AXIS`: the reference sea level
    `.005 cos` changes sign on the great circle normal to the axis, so the
    coastline crosses the surface, and the ice (`sphere_ice_at`) sits on the
    opposite side, on land, with a grounding line. The coupling arguments
    are those of `SeaFields.kwargs`, on the facet tag `CURVE_RE` that the
    sphere shares with the annulus.
    """

    def __init__(self, sub):
        Q = fd.FunctionSpace(sub, "CG", 2)
        X = fd.SpatialCoordinate(sub)
        self.SL_init = fd.Function(Q).interpolate(.005 * along_axis(X))
        self.I_init = fd.Function(Q)
        self.I = fd.Function(Q)


def sphere_ice_at(sea, k):
    """Set the ice before step `k` (0-based): `gia_march.ice_at` on `LOAD_AXIS`.

    `.002 (1 + .2 min(k, 3)) (1 - cos) / 2`, with the angle measured from
    `LOAD_AXIS`: a cap centred on `-LOAD_AXIS`, growing by 20 percent per
    step up to step 3 and constant afterwards, as in the 2-D scenario.

    Args:
      sea: the `SphereSeaFields` of the run.
      k: the 0-based index of the step about to be solved.
    """
    X = fd.SpatialCoordinate(sea.I.function_space().mesh())
    sea.I.interpolate(.002 * (1 + .2 * min(k, 3)) * (1 - along_axis(X)) / 2)


def build(parent, sub):
    """The solver of `gia_march.SEA` on the sphere, under the iterative preset.

    The couplings are those of `gia_cases.build_new` for `SEA`: a fluid core
    on the CMB (core density 1.2, CMB gravity 1), rotation, the centre of
    mass and a sea level with a moving coastline. In 3-D the `Rotation`
    coupling needs the dynamical ellipticity `C - A` for the polar-wander
    rows; it gets it from the secular Love number `SPHERE_K_S` and the
    surface radius (`C - A = Q k_s`), as the 3-D forward entry does. `C` is
    the polar moment `int rho_0 (x^2 + y^2) dV` of the mantle (density 1).
    The rigid rotations are declared as the nullspace, as in the GIAMIP
    drivers: a rigid rotation of a spherically symmetric mantle changes no
    mass moment, moves no water and leaves the inertia tensor unchanged, so
    it stays in the kernel with the sea level, the frame and the rotation
    rows on. The preset keeps its own tolerances, as in `sea-iterative`.

    Args:
      parent: the curved parent mesh.
      sub: the curved mantle submesh.

    Returns:
      A `gia_march.Run`.
    """
    case = M.SEA
    # The couplings below are those of `SEA`; a change of `SEA` must reach
    # this builder.
    assert (case.core, case.rotation, case.centre, case.load) == (
        "fluid", True, True, "sea_moving"), case
    dt = C.dt_function(sub)
    sea = SphereSeaFields(sub)
    X = fd.SpatialCoordinate(sub)
    polar_moment = fd.assemble((X[0] ** 2 + X[1] ** 2) * fd.dx(domain=sub))
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    couplings = [cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.),
                 cp.Rotation(C=polar_moment, k_s=SPHERE_K_S, surface_radius=RE,
                             Omega_sq=SPHERE_OMEGA_SQ),
                 cp.CentreOfMass(),
                 cp.SeaLevel(**sea.kwargs(case))]
    gravity_bcs = {C.CURVE_OUTER: {"dtn": gadopt.SphericalDtN(DTN_DEGREE)},
                   C.CURVE_INNER: {"dtn": gadopt.SphericalDtN(DTN_DEGREE)}}
    with stop_annotating():
        solver = gadopt.SelfGravitatingGIASolver(
            u, psi, C.approximation(case), dt=dt, time_scheme=case.scheme,
            internal_variable_family=case.family, quad_degree=C.QUAD,
            gravity_bcs=gravity_bcs, couplings=couplings,
            dtn_representation=case.representation, nullspace=M.NULLSPACE,
            solver_parameters="iterative")
    return M.Run(case=case, solver=solver, u=u, psi=psi, sea=sea, dt=dt)


def diagnostics(run, k):
    """The row of step `k` (0-based): `gia_march.diagnostics` and `m1`, `m2`.

    `gia_march.diagnostics` reads the sphere as it reads the annulus (the
    facet tags are the same, the frame columns follow the dimension and give
    `com_2`, `dipole_2`, `sheet_moment_2`); it writes the rotation-rate
    change `m3` only, so the polar-wander pair is added here.
    """
    row = M.diagnostics(run, k)
    for name in ("m1", "m2"):
        row[name] = M._real(run.solver, name)
    return row


def run_weekly():
    """Build the mesh and the solver, march `WEEKLY_STEP_COUNT` steps, write the table."""
    start = time.time()
    parent, sub, record = sphere_meshes()
    write_mesh_record(record)
    run = build(parent, sub)
    M.enable_residual_history(run.solver)
    PETSc.Sys.Print(f"built in {time.time() - start:.1f} s")
    rows = []

    def on_step(k, run):
        rows.append(diagnostics(run, k))
        # The wall time goes to the job log only: it is not a reproducible
        # number, so it is not a column.
        PETSc.Sys.Print(f"step {k + 1}: newton {rows[-1]['newton']}, linear "
                        f"{rows[-1]['linear']}, wall {time.time() - start:.1f} s")
        # Write after every step, so that a job that fails later still
        # leaves the rows it solved.
        M.write_log(CASE_DIR / WEEKLY_LOG, rows)

    M.advance(run, M.FORWARD_STEPS[:WEEKLY_STEP_COUNT], on_step=on_step, ice=sphere_ice_at)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=[WEEKLY_ENTRY])
    parser.parse_args()
    run_weekly()
