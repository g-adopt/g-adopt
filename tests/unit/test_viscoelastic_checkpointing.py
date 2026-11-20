"""Round trips of `gadopt.viscoelastic_checkpointing.CheckpointField`.

`CheckpointField` stores a field on a parent mesh so that `CheckpointFile`
can write it. A Quadrature field has no polynomial extension, so the class
copies the values of each cell into one DG0 vector of that cell, through the
cell-to-node table of the quadrature space. The tests encode a field, set it
to zero, restore it, and require the original values exactly. They cover a
tensor and a scalar Quadrature field and a `Real` field, on the parent mesh
itself, on a cell submesh, and on an extruded mesh, in serial and on two
ranks. On two ranks the submesh is built so that rank 0 owns none of its
cells, which is the case where an empty array has to be reshaped.

The values of a Quadrature field are stored in the local node order of each
cell, which changes with the partition. The file tests write a file on one
rank and read it on two, and the other way round, inside one two-rank
process (one rank alone works on `COMM_SELF`). `CheckpointField.save`
stores the quadrature point coordinates and `load` pairs the values by
position; a file without the coordinates is refused on another number of
ranks; on the writer's number of ranks the values come back bit for bit.
A parent and a submesh of one mesh name keep a point set each.
"""
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

import gadopt  # noqa: F401  (gadopt before firedrake)
import firedrake as fd
from finat.ufl import FiniteElement, TensorElement
from gadopt.viscoelastic_checkpointing import (CheckpointField, match_points,
                                               written_rank_count)
from parallel_ranks import assert_marked_rank_count

#: Degree of the quadrature rule of the history fields.
QUADRATURE_DEGREE = 2
#: Cell tag of the submesh.
SUBMESH_TAG = 7


def quadrature_element(cell, shape=None):
    """A Quadrature element of `QUADRATURE_DEGREE`, scalar or of a tensor shape."""
    element = FiniteElement("Quadrature", cell, QUADRATURE_DEGREE, quad_scheme="default")
    return element if shape is None else TensorElement(element, shape=shape)


def parent_and_submesh(comm=MPI.COMM_WORLD, split_by_rank=True):
    """A labelled unit square and a cell submesh of it.

    In serial, or with `split_by_rank=False`, the submesh is the left half.
    Otherwise it is the set of cells that the last rank owns, so that every
    other rank owns no cell of the submesh. The marker is written on owned
    cells only; `RelabeledMesh` reads it with ghosts, which brings the ghost
    values up to date.

    Args:
      comm: the communicator of the mesh.
      split_by_rank: whether the submesh is the last rank's cells.

    Returns:
      `(labelled, sub)`: the labelled parent and the submesh.
    """
    parent = fd.UnitSquareMesh(6, 6, comm=comm)
    marker = fd.Function(fd.FunctionSpace(parent, "DG", 0))
    if comm.size == 1 or not split_by_rank:
        x, _ = fd.SpatialCoordinate(parent)
        marker.interpolate(fd.conditional(x < 0.5, 1, 0))
    else:
        marker.dat.data_wo[...] = 1.0 if comm.rank == comm.size - 1 else 0.0
    labelled = fd.RelabeledMesh(parent, [marker], [SUBMESH_TAG])
    sub = fd.Submesh(labelled, labelled.topological_dimension, SUBMESH_TAG)
    return labelled, sub


def fields_on(mesh):
    """A tensor and a scalar Quadrature field and a `Real` field on `mesh`.

    The values are smooth functions of the coordinates, so that each node has
    its own value and a node written to the wrong place changes the result.
    """
    dim = mesh.geometric_dimension
    X = fd.SpatialCoordinate(mesh)
    tensor_space = fd.FunctionSpace(mesh, quadrature_element(mesh.ufl_cell(), (dim, dim)))
    scalar_space = fd.FunctionSpace(mesh, quadrature_element(mesh.ufl_cell()))
    tensor = fd.Function(tensor_space).interpolate(
        fd.outer(X, X) + fd.sin(3 * X[0]) * fd.Identity(dim))
    scalar = fd.Function(scalar_space).interpolate(fd.cos(2 * X[1]) + X[0] + X[dim - 1] ** 2)
    real = fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(4.25)
    return {"tensor": tensor, "scalar": scalar, "real": real}


def round_trip_error(name, field, parent):
    """Encode, zero, restore; return the largest error over all ranks.

    Args:
      name: the name of the stored field.
      field: the field to store, changed in place.
      parent: the mesh the stored field lives on.

    Returns:
      `max |restored - original|` over every owned entry on every rank.
    """
    reference = field.copy(deepcopy=True)
    checkpoint = CheckpointField(name, field, parent)
    stored = checkpoint.encode().copy(deepcopy=True)
    field.assign(0.0)
    checkpoint.restore(stored)
    local = float(np.max(np.abs(field.dat.data_ro - reference.dat.data_ro), initial=0.0))
    return field.comm.allreduce(local, op=MPI.MAX)


def check_all_round_trips():
    """Every field on the parent, on the submesh and on an extruded mesh."""
    labelled, sub = parent_and_submesh()
    comm = labelled.comm
    if comm.size > 1:
        # The case this layout exists for: a rank without submesh cells.
        assert comm.allreduce(sub.cells.owned.local_size == 0, op=MPI.LOR)
    extruded = fd.ExtrudedMesh(fd.UnitSquareMesh(3, 3), 3)
    for label, mesh, parent in (("parent", labelled, labelled),
                                ("submesh", sub, labelled),
                                ("extruded", extruded, extruded)):
        for name, field in fields_on(mesh).items():
            error = round_trip_error(name, field, parent)
            assert error == 0.0, f"{label} {name}: max error {error}"


def test_round_trip_serial():
    check_all_round_trips()


@pytest.mark.parallel(nprocs=2)
def test_round_trip_two_ranks(request):
    assert_marked_rank_count(request)
    check_all_round_trips()


# -- Files read on another number of ranks ----------------------------------

#: The meshes of the file tests: label, and whether the field lives on the
#: submesh (`True`) or on the parent itself.
FILE_MESHES = (("parent", False), ("submesh", True))


def write_file(path, comm, *, with_points=True):
    """Write the parent and every field of the parent and the submesh.

    Args:
      path: the file.
      comm: the communicator of the writer (`COMM_SELF` for one rank).
      with_points: store through `CheckpointField.save` (with the point
        coordinates), or through `encode` and `save_function` alone, the
        layout of a file without coordinates.
    """
    labelled, sub = parent_and_submesh(comm, split_by_rank=False)
    with fd.CheckpointFile(str(path), "w", comm=comm) as chk:
        chk.save_mesh(labelled)
        for label, on_sub in FILE_MESHES:
            for name, field in fields_on(sub if on_sub else labelled).items():
                checkpoint = CheckpointField(f"{label}_{name}", field, labelled)
                if with_points:
                    checkpoint.save(chk)
                else:
                    chk.save_function(checkpoint.encode())


def read_file(path, comm, *, direct=False):
    """Read every field of `write_file` on `comm`; return the largest errors.

    The reference is each field computed again on the loaded mesh. Its
    values differ from the written ones by round-off, because a quadrature
    point computed from another vertex order of the cell is another
    floating-point number.

    Args:
      path: the file.
      comm: the communicator of the reader.
      direct: restore the values in the stored node order, without the
        pairing by position (the control of the test).

    Returns:
      `(error, exact)`: the largest relative error of any field against the
      recomputed values, and whether every field re-encodes to the stored
      encoding bit for bit.
    """
    error, exact = 0.0, True
    with fd.CheckpointFile(str(path), "r", comm=comm) as chk:
        labelled = chk.load_mesh(fd.mesh.DEFAULT_MESH_NAME)
        sub = fd.Submesh(labelled, labelled.topological_dimension, SUBMESH_TAG)
        for label, on_sub in FILE_MESHES:
            for name, reference in fields_on(sub if on_sub else labelled).items():
                field = fd.Function(reference.function_space())
                checkpoint = CheckpointField(f"{label}_{name}", field, labelled)
                if direct:
                    checkpoint.restore(chk.load_function(labelled, checkpoint.name))
                else:
                    checkpoint.load(chk)
                difference = np.max(np.abs(field.dat.data_ro - reference.dat.data_ro),
                                    initial=0.0)
                scale = np.max(np.abs(reference.dat.data_ro), initial=0.0)
                error = max(error, comm.allreduce(float(difference), op=MPI.MAX)
                            / comm.allreduce(float(scale), op=MPI.MAX))
                stored = chk.load_function(labelled, checkpoint.name)
                same = np.array_equal(checkpoint.encode().dat.data_ro, stored.dat.data_ro)
                exact &= comm.allreduce(bool(same), op=MPI.LAND)
    return error, exact


def shared_directory(tmp_path):
    """Rank 0's `tmp_path`, the same directory on every rank."""
    return Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))


def on_rank_zero(function, *args, **kwargs):
    """Run `function` on rank 0 alone; give its result, or its error, to every rank.

    The work of one rank is done on `COMM_SELF`. An exception there is
    passed to the other ranks too, so that every rank fails the test in
    the same way and none waits for a rank that left.
    """
    comm = MPI.COMM_WORLD
    result = None
    if comm.rank == 0:
        try:
            result = ("value", function(*args, **kwargs))
        except Exception as error:  # noqa: BLE001  (re-raised on every rank)
            result = ("error", error)
    kind, value = comm.bcast(result, root=0)
    if kind == "error":
        raise value
    return value


#: The largest relative error accepted for a field read on another partition:
#: round-off of the recomputed reference, measured below 1e-15.
CROSS_PARTITION_RTOL = 1e-13


@pytest.mark.parallel(nprocs=2)
def test_quadrature_fields_read_on_another_number_of_ranks(request, tmp_path):
    """Written on one rank and read on two, written on two and read on one.

    Covers the parent and the submesh. The control reads the file of one
    rank on two without the pairing and must see a large error, which shows
    that the partitions differ and that the test is not vacuous.
    """
    assert_marked_rank_count(request)
    directory = shared_directory(tmp_path)
    comm = MPI.COMM_WORLD
    one = directory / "written_on_one.h5"
    on_rank_zero(write_file, one, MPI.COMM_SELF)
    error, _ = read_file(one, comm)
    assert error <= CROSS_PARTITION_RTOL, error
    control, _ = read_file(one, comm, direct=True)
    assert control > 1e-3, control

    two = directory / "written_on_two.h5"
    write_file(two, comm)
    comm.barrier()
    error, _ = on_rank_zero(read_file, two, MPI.COMM_SELF)
    assert error <= CROSS_PARTITION_RTOL, error
    # On the writer's number of ranks the values come back bit for bit.
    _, exact = read_file(two, comm)
    assert exact


@pytest.mark.parallel(nprocs=2)
def test_a_file_without_point_coordinates_is_refused_on_another_number_of_ranks(request, tmp_path):
    """A file without coordinates: refused on one rank, bit for bit on two.

    The number of ranks of the writer is the one Firedrake stores with the
    distribution of the mesh, because the file has no record of `save`.
    """
    assert_marked_rank_count(request)
    directory = shared_directory(tmp_path)
    comm = MPI.COMM_WORLD
    path = directory / "without_points.h5"
    write_file(path, comm, with_points=False)
    comm.barrier()

    def writer_ranks():
        with fd.CheckpointFile(str(path), "r", comm=MPI.COMM_SELF) as chk:
            return written_rank_count(chk)

    assert on_rank_zero(writer_ranks) == 2
    with pytest.raises(ValueError, match=r"written on 2 rank\(s\)"):
        on_rank_zero(read_file, path, MPI.COMM_SELF)
    _, exact = read_file(path, comm)
    assert exact


def test_a_parent_and_a_submesh_of_one_name_keep_their_own_points(tmp_path):
    """Each Quadrature field of a file is read with the points of its own mesh.

    The parent and the submesh carry one mesh name here, as a parent and a
    submesh do when each is curved by `Mesh(coordinate_function)`
    (`gia_cases.annulus_meshes`). Their Quadrature fields of one rule
    are written to one file, the submesh field first, and read back on the
    writer's partition; both must come back bit for bit. This needs a
    stored point set per field: the points of the submesh cover the submesh
    cells only, so a parent field paired with them is refused, and a
    submesh field paired with the points of the parent depends on both
    meshes ordering the points of a shared cell in the same way.
    """
    comm = MPI.COMM_WORLD
    path = shared_directory(tmp_path) / "one_mesh_name.h5"

    def named_submesh(labelled):
        """The submesh of `labelled`, given the parent's mesh name."""
        return fd.Submesh(labelled, labelled.topological_dimension, SUBMESH_TAG,
                          name=labelled.name)

    labelled, _ = parent_and_submesh(comm, split_by_rank=False)
    sub = named_submesh(labelled)
    assert sub.name == labelled.name
    with fd.CheckpointFile(str(path), "w", comm=comm) as chk:
        chk.save_mesh(labelled)
        # The submesh field first: its point set is the one a key by mesh
        # name and rule would keep for both fields.
        for label, mesh in (("submesh", sub), ("parent", labelled)):
            field = fields_on(mesh)["scalar"]
            CheckpointField(f"{label}_scalar", field, labelled).save(chk)
    with fd.CheckpointFile(str(path), "r", comm=comm) as chk:
        labelled = chk.load_mesh(fd.mesh.DEFAULT_MESH_NAME)
        sub = named_submesh(labelled)
        for label, mesh in (("submesh", sub), ("parent", labelled)):
            reference = fields_on(mesh)["scalar"]
            field = fd.Function(reference.function_space())
            CheckpointField(f"{label}_scalar", field, labelled).load(chk)
            same = np.array_equal(field.dat.data_ro, reference.dat.data_ro)
            assert comm.allreduce(bool(same), op=MPI.LAND), label


def test_a_point_without_a_match_is_refused():
    """`match_points` refuses a stored point moved by more than the tolerance.

    A permutation of the points is undone; the same points moved by 1e-4
    of the cell size, and two local points on one stored point, are
    refused.
    """
    current = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    size = np.array([1.0])
    order = match_points(current, current[:, [2, 0, 1]], size, comm=MPI.COMM_SELF)
    assert np.array_equal(order, [[1, 2, 0]])
    with pytest.raises(ValueError, match="do not match"):
        match_points(current, current + 1e-4, size, comm=MPI.COMM_SELF)
    with pytest.raises(ValueError, match="share one stored point"):
        match_points(current, current[:, [0, 0, 1]], size, comm=MPI.COMM_SELF)
