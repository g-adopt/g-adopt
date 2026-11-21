"""Helpers of `tests/unit/test_viscoelastic_checkpointing.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_viscoelastic_checkpointing.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import numpy as np
from mpi4py import MPI
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
