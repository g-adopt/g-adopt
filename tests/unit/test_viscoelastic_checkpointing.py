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
"""
import numpy as np
import pytest
from mpi4py import MPI

import gadopt  # noqa: F401  (gadopt before firedrake)
import firedrake as fd
from finat.ufl import FiniteElement, TensorElement
from gadopt.viscoelastic_checkpointing import CheckpointField

#: Degree of the quadrature rule of the history fields.
QUADRATURE_DEGREE = 2
#: Cell tag of the submesh.
SUBMESH_TAG = 7


def quadrature_element(cell, shape=None):
    """A Quadrature element of `QUADRATURE_DEGREE`, scalar or of a tensor shape."""
    element = FiniteElement("Quadrature", cell, QUADRATURE_DEGREE, quad_scheme="default")
    return element if shape is None else TensorElement(element, shape=shape)


def parent_and_submesh():
    """A labelled unit square and a cell submesh of it.

    In serial the submesh is the left half. On more ranks it is the set of
    cells that the last rank owns, so that every other rank owns no cell of
    the submesh. The marker is written on owned cells only; `RelabeledMesh`
    reads it with ghosts, which brings the ghost values up to date.

    Returns:
      `(labelled, sub)`: the labelled parent and the submesh.
    """
    parent = fd.UnitSquareMesh(6, 6)
    comm = parent.comm
    marker = fd.Function(fd.FunctionSpace(parent, "DG", 0))
    if comm.size == 1:
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
def test_round_trip_two_ranks():
    check_all_round_trips()
