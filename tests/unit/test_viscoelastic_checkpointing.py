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
`test_gia_parallel.py`). `parent_and_submesh`, `fields_on` and
`round_trip_error` are in `tests/gia/gia_models/checkpointing.py`, which that
driver and this module share.
"""
import numpy as np
import pytest
from mpi4py import MPI

import gadopt  # noqa: F401  (gadopt before firedrake)
import firedrake as fd
from gadopt.viscoelastic_checkpointing import CheckpointField, match_points
from gia_models.checkpointing import (
    SUBMESH_TAG,
    fields_on,
    parent_and_submesh,
    round_trip_error,
)


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
