"""Round trips of `gadopt.viscoelastic_checkpointing.CheckpointField`.

`CheckpointField` stores a field on a parent mesh so that `CheckpointFile`
can write it. A Quadrature field has no polynomial extension, so the class
copies the values of each cell into one DG0 vector of that cell, through the
cell-to-node table of the quadrature space. The tests encode a field, set it
to zero, restore it, and require the original values exactly. They cover a
tensor and a scalar Quadrature field and a `Real` field, on the parent mesh
itself, on a cell submesh, and on an extruded mesh. A parent and a submesh
of one mesh name keep a point set each, and `match_points` refuses a point
without a match.

The same round trips on two ranks, with a submesh of which one rank owns no
cell, and the files written on one rank and read on two (and the other way
round) are entries of the `parallel` case in `tests/gia` (`gia_parallel.py`,
`test_gia_parallel.py`). That driver uses `parent_and_submesh`, `fields_on`
and `round_trip_error` of this file.
"""
import numpy as np
import pytest
from mpi4py import MPI

import gadopt  # noqa: F401  (gadopt before firedrake)
import firedrake as fd
from finat.ufl import FiniteElement, TensorElement
from gadopt.viscoelastic_checkpointing import CheckpointField, match_points

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
    extruded = fd.ExtrudedMesh(fd.UnitSquareMesh(3, 3), 3)
    for label, mesh, parent in (("parent", labelled, labelled),
                                ("submesh", sub, labelled),
                                ("extruded", extruded, extruded)):
        for name, field in fields_on(mesh).items():
            error = round_trip_error(name, field, parent)
            assert error == 0.0, f"{label} {name}: max error {error}"


def test_round_trip_serial():
    check_all_round_trips()


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
    path = tmp_path / "one_mesh_name.h5"

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
