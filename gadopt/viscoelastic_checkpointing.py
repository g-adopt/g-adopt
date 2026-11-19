"""Lossless parent-mesh storage of GIA fields, including quadrature histories.

`CheckpointFile` projects unsupported elements into polynomial spaces, and it
stores a Function on its own mesh only, which a `Submesh` is not.
`CheckpointField` stores a field on the parent mesh instead, in an encoding
that `CheckpointFile` writes without loss:

- A field of a polynomial element: in its own element on the parent
  (`broken=False`, a copy of the degrees of freedom), or interpolated into
  the broken element (`broken=True`).
- A Quadrature field: its values have no polynomial extension, so the values
  of all quadrature points of a cell go into one DG0 vector of that cell.
  This is an encoding, not a DG0 constitutive approximation.
- A `Real` field: as itself.

## Quadrature fields on another partition

The DG0 vector of a cell holds the point values in the local order of the
quadrature nodes of that cell. That order follows the local vertex order of
the cell, which changes when the mesh is distributed in another way: on
another number of ranks, or with other `distribution_parameters` or
`reorder` given to `CheckpointFile.load_mesh`. `CheckpointFile` moves the
DG0 vector to the right cell, but the values inside it come back permuted.

`save` therefore also stores the physical coordinates of the quadrature
points, once per file and per stored field, in the same encoding. `load`
pairs each local point of a cell with the stored point of the same cell at
the same position, and copies the values in that order. The pairing accepts
a distance of at most `MATCH_TOLERANCE` times the cell size and refuses
the file otherwise. On the partition that wrote the file the pairing is the
identity, so the values come back bit for bit.

A file with no stored coordinates (written by `encode` and
`CheckpointFile.save_function` directly, as the earlier drivers did) can be
read on the number of ranks that wrote it only. `load` reads that number
from the file and refuses another one. It cannot see a new partition on the
same number of ranks, so such a file must also be read with
`load_mesh(name)` alone, which loads the stored partition.

Application checkpoint I/O is outside the differentiated march: nothing in
this module is recorded on the tape.
"""
import firedrake as fd
import numpy as np
from finat.ufl import BrokenElement, VectorElement
from firedrake.mesh import DEFAULT_MESH_NAME
from mpi4py import MPI

__all__ = ["CheckpointField", "MATCH_TOLERANCE", "match_points",
           "quadrature_points_name", "written_rank_count"]

#: The HDF5 group of the attributes that `CheckpointField.save` writes: the
#: number of ranks of the writer and one flag per stored set of point
#: coordinates.
GROUP = "/gadopt_checkpoint_field"

#: The largest distance, relative to the cell size (the `d`-th root of the
#: cell volume), between a local quadrature point and the stored point it is
#: paired with. The points of one cell are a fixed fraction of the cell size
#: apart (about 0.1 for the rules in use), and the same point computed from
#: another vertex order differs by round-off of the coordinate magnitude.
#: 1e-6 lies far between the two, also for an Earth-sized mesh with a
#: coordinate magnitude 1e3 times the cell size.
MATCH_TOLERANCE = 1e-6

#: The number of cells whose distance tables are held at one time in
#: `match_points`. A table has `points x points` entries per cell, so this
#: bounds the memory of the pairing on a large mesh.
MATCH_CHUNK = 4096


def quadrature_points_name(name: str) -> str:
    """The stored name of the point coordinates of the stored field `name`.

    Each Quadrature field has its own stored point set, keyed by the name
    of the field. A key built from the mesh and the rule would not be
    unique: a parent and a submesh can carry one mesh name (both are
    `firedrake_default` when each is curved by `Mesh(coordinate_function)`,
    as the GIA tests and drivers do), and their point sets of one rule
    differ. A point set per field also comes from
    the same encoding as the values of that field, so the pairing of
    `load` does not depend on two meshes ordering the points of a shared
    cell in the same way. The cost is one coordinate vector per point and
    per Quadrature field, which is small against a tensor history.

    Args:
      name: the name of the stored field, as given to `CheckpointField`.

    Returns:
      The name of the stored point set of that field.
    """
    return f"gadopt_quadrature_points_{name}"


def _scalar_element(element):
    """The scalar Quadrature element of a scalar, vector or tensor one."""
    sub_elements = element.sub_elements
    return sub_elements[0] if sub_elements else element


def _attribute(checkpoint, key):
    """An attribute of `GROUP` in an open `CheckpointFile`, or `None`."""
    if GROUP not in checkpoint.h5pyfile:
        return None
    if not checkpoint.has_attr(GROUP, key):
        return None
    value = checkpoint.get_attr(GROUP, key)
    return value.item() if hasattr(value, "item") else value


def written_rank_count(checkpoint, mesh_name=DEFAULT_MESH_NAME):
    """The number of ranks that wrote a checkpoint file, or `None`.

    `CheckpointField.save` records the number in the file. A file written
    without it (by `CheckpointFile.save_function` directly) still has the
    number that Firedrake stores with the distribution of each saved mesh;
    that record is read through `CheckpointFile`'s private path helpers,
    so a change of Firedrake's layout gives `None`, never a wrong number.

    Args:
      checkpoint: a `CheckpointFile` open for reading.
      mesh_name: the name of the mesh whose distribution record is read when
        the file has no record of `save`.

    Returns:
      The number of ranks as an int, or `None` when the file records none.
    """
    count = _attribute(checkpoint, "comm_size")
    if count is not None:
        return int(count)
    try:
        topology = checkpoint._get_mesh_name_topology_name_map()[mesh_name]
        path = checkpoint._path_to_distributions(topology)
        distribution, = checkpoint.h5pyfile[path].keys()
        path = checkpoint._path_to_distribution(topology, distribution)
        return int(checkpoint.get_attr(path, "comm_size"))
    except (AttributeError, KeyError, ValueError):
        return None


def match_points(current, stored, size, *, comm,
                 tolerance=MATCH_TOLERANCE, chunk=MATCH_CHUNK):
    """Pair the quadrature points of each cell with the stored points by position.

    Collective on `comm`: every rank raises, or none does.

    Args:
      current: `(cells, points, dim)` array, the physical coordinates of the
        quadrature points of each owned cell in the local node order.
      stored: the same array as the file stores it, in the node order of the
        partition that wrote the file.
      size: `(cells,)` array, the size of each cell (a length).
      comm: the communicator of the mesh.
      tolerance: the largest accepted distance relative to `size`.
      chunk: the number of cells whose distance tables are held at once.

    Returns:
      `(cells, points)` integer array `order`: local point `i` of cell `c`
      is stored point `order[c, i]`.

    Raises:
      ValueError: if a local point has no stored point within the tolerance
        in its cell, or if two local points are paired with one stored
        point.
    """
    cells, points = current.shape[:2]
    order = np.empty((cells, points), dtype=np.intp)
    worst = 0.0
    duplicate = False
    identity = np.arange(points)
    for start in range(0, cells, chunk):
        stop = min(start + chunk, cells)
        # Distance of every local point to every stored point of its cell:
        # (cells in the chunk, local point, stored point).
        distance = np.linalg.norm(
            current[start:stop, :, None, :] - stored[start:stop, None, :, :], axis=3)
        nearest = np.argmin(distance, axis=2)
        gap = np.take_along_axis(distance, nearest[..., None], axis=2)[..., 0]
        worst = max(worst, float(np.max(gap / size[start:stop, None], initial=0.0)))
        # Each stored point must be taken once: the pairing is a permutation.
        duplicate |= not np.array_equal(np.sort(nearest, axis=1),
                                        np.broadcast_to(identity, nearest.shape))
        order[start:stop] = nearest
    worst = comm.allreduce(worst, op=MPI.MAX)
    duplicate = comm.allreduce(duplicate, op=MPI.LOR)
    if worst > tolerance or duplicate:
        raise ValueError(
            "The stored quadrature points do not match the quadrature points "
            f"of this mesh: the largest distance to the nearest stored point is "
            f"{worst:.3e} cell sizes (accepted: {tolerance:.1e})"
            + (", and two points share one stored point" if duplicate else "")
            + ". The file was written on another mesh, another coordinate field "
            "(for example a mesh curved in another way) or another quadrature "
            "rule.")
    return order


class CheckpointField:
    """Encode a field on a parent mesh and restore it through the submesh map.

    Args:
      name: the name of the stored Function.
      field: the field, on the parent or on a submesh of it.
      parent: the mesh the stored Function lives on.
      broken: for a polynomial field of a submesh, `True` interpolates it into
        the broken element on the parent, `False` copies its degrees of
        freedom into its own element on the parent. Ignored for Quadrature
        and `Real` fields; a `Real` field is stored as a `Real` of the parent.
    """

    def __init__(self, name, field, parent, *, broken=True):
        self.name = name
        self.field = field
        self.parent = parent
        space = field.function_space()
        # A `Real` field is stored as a `Real` of the parent in every case.
        # Its broken encoding would hold the value in the cells of the
        # submesh and zero elsewhere, and a read on another partition could
        # then take its value from a cell outside the submesh.
        self.broken = broken and space.ufl_element().family() != "Real"
        self.quadrature = space.ufl_element().family() == "Quadrature"
        if self.quadrature:
            # Cell-to-node table of the quadrature space, one row per local
            # cell. The owned cells come first, so the first `self.cells`
            # rows are the cells this rank writes. Quadrature nodes sit inside
            # a cell, so the nodes of an owned cell are owned nodes.
            self.nodes = space.cell_node_list
            self.cells = space.mesh().cells.owned.local_size
            # One DG0 vector per cell holds every component at every
            # quadrature point of that cell.
            self.width = self.nodes.shape[1]*space.block_size
            self.local = fd.Function(fd.VectorFunctionSpace(space.mesh(), "DG", 0, dim=self.width))
            self.local_nodes = self.local.function_space().cell_node_list[:self.cells, 0]
            self.points_name = quadrature_points_name(name)
            element = self.local.ufl_element()
        elif space.mesh() is not parent and self.broken:
            element = BrokenElement(space.ufl_element())
            self.local = fd.Function(fd.FunctionSpace(space.mesh(), element))
        else:
            element = space.ufl_element()
            self.local = field
        self.stored = fd.Function(fd.FunctionSpace(parent, element), name=name)

    # -- Encoding --------------------------------------------------------------

    def encode(self):
        """The parent-mesh encoding of the present field, a Function named `name`."""
        if self.quadrature:
            data = self.field.dat.data_ro_with_halos[self.nodes[:self.cells]]
            # `data_rw` because the write is indexed: in pyop3 a write-only
            # access drops a pending halo reduction of the whole vector.
            # The width is explicit, because a rank that owns no cell of a
            # submesh has an empty array, and numpy cannot infer -1 from 0.
            self.local.dat.data_rw[self.local_nodes] = data.reshape(self.cells, self.width)
        elif self.local is not self.field:
            self.local.interpolate(self.field)
        if self.field.function_space().ufl_element().family() == "Real":
            # A `Real` of a submesh cannot be assigned to one of the parent
            # across the meshes; its value is one number, held by every rank
            # (owned on one, a ghost entry on the others).
            self.stored.assign(float(self.field.dat.data_ro_with_halos[0]))
        elif self.local.function_space().mesh() is self.parent:
            self.stored.assign(self.local)
        elif not self.broken and not self.quadrature:
            self.stored.assign(self.local, allow_missing_dofs=True)
        else:
            self.stored.interpolate(self.local, allow_missing_dofs=True)
        return self.stored

    def _coordinates(self):
        """The physical coordinates of the quadrature points, a vector Quadrature Function.

        Interpolated from the mesh's coordinate field, so a curved mesh gives
        the curved positions, the points at which the history is defined.
        """
        space = self.field.function_space()
        mesh = space.mesh()
        element = VectorElement(_scalar_element(space.ufl_element()),
                                dim=mesh.geometric_dimension)
        return fd.Function(fd.FunctionSpace(mesh, element)).interpolate(
            fd.SpatialCoordinate(mesh))

    def points(self):
        """The parent-mesh encoding of the quadrature point coordinates.

        The encoding of `encode`, applied to the coordinates, so that the
        stored point `j` of a cell is where the stored value `j` of the cell
        sits. Named `quadrature_points_name(name)`.
        """
        if not self.quadrature:
            raise ValueError(f"{self.name!r} is not a Quadrature field; it has "
                             "no quadrature points.")
        return CheckpointField(self.points_name, self._coordinates(), self.parent).encode()

    def _cell_values(self, encoding, local, width):
        """The values of each owned cell in stored order, `(cells, points, width)`.

        `width` is explicit, because a rank that owns no cell of a submesh
        has an empty array, from which numpy cannot infer it.
        """
        local.interpolate(encoding)
        return local.dat.data_ro[self.local_nodes].reshape(self.cells, self.nodes.shape[1], width)

    def _order(self, points):
        """The stored index of each local quadrature point (`match_points`)."""
        mesh = self.field.function_space().mesh()
        dim = mesh.geometric_dimension
        if points.ufl_shape != (self.nodes.shape[1]*dim,):
            raise ValueError(
                f"The stored quadrature points of {self.name!r} have shape "
                f"{points.ufl_shape}; this quadrature space needs "
                f"({self.nodes.shape[1]*dim},).")
        scratch = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=points.ufl_shape[0]))
        stored = self._cell_values(points, scratch, dim)
        current = self._coordinates().dat.data_ro_with_halos[self.nodes[:self.cells]]
        current = current.reshape(self.cells, self.nodes.shape[1], dim)
        # The cell size as the `d`-th root of the cell volume. `CellDiameter`
        # would serve as well but has no evaluation on a curved (P2) mesh.
        size = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(
            fd.CellVolume(mesh) ** (1.0 / mesh.topological_dimension))
        size = size.dat.data_ro[size.function_space().cell_node_list[:self.cells, 0]]
        return match_points(current, stored, size, comm=mesh.comm)

    def restore(self, stored, *, points=None, writer_ranks=None):
        """Write the encoding `stored` back into the field.

        Args:
          stored: the parent-mesh encoding, as `encode` made it (and as
            `CheckpointFile.load_function` returns it).
          points: Quadrature fields only: the stored encoding of the point
            coordinates (`points`). The values are then copied by position
            and the restore is valid on any partition.
          writer_ranks: Quadrature fields only, without `points`: the number
            of ranks that wrote `stored`. Another number than this mesh's is
            refused. With neither `points` nor `writer_ranks` the values are
            copied in the stored node order, which is right only on the
            partition that encoded them (an encoding kept in memory).

        Raises:
          ValueError: if the stored shape does not match, if `writer_ranks`
            differs from the number of ranks of the mesh, or if the stored
            points do not match (`match_points`).
        """
        if self.quadrature:
            if stored.ufl_shape != self.stored.ufl_shape:
                raise ValueError("Checkpoint quadrature storage shape does not match the material history")
            comm = self.field.function_space().mesh().comm
            if points is None and writer_ranks is not None and writer_ranks != comm.size:
                raise ValueError(
                    f"The quadrature field {self.name!r} was written on "
                    f"{writer_ranks} rank(s) and the file has no stored point "
                    f"coordinates; it cannot be read on {comm.size} rank(s). The "
                    "values of a cell are stored in the local order of its "
                    "quadrature points, and that order changes with the "
                    "partition, so the values would come back permuted inside "
                    f"each cell. Read the file on {writer_ranks} rank(s), with "
                    "load_mesh(name) alone, and write it again with "
                    "CheckpointField.save, which stores the coordinates.")
            values = self._cell_values(stored, self.local,
                                       self.width // self.nodes.shape[1])
            if points is not None:
                order = self._order(points)
                values = np.take_along_axis(values, order[..., None], axis=1)
            shape = (self.cells, self.nodes.shape[1], *self.field.dat.data_ro.shape[1:])
            self.field.dat.data_rw[self.nodes[:self.cells]] = values.reshape(shape)
        elif self.field.function_space().ufl_element().family() == "Real":
            # The array with ghosts: a `Real` value is owned by one rank
            # and is a ghost entry on the others.
            self.field.assign(float(stored.dat.data_ro_with_halos[0]))
        elif not self.broken:
            # A copy of the degrees of freedom, the inverse of `encode`. From
            # the parent onto a submesh every target degree of freedom has a
            # source, and the assignment needs the flag to map across meshes.
            if self.field.function_space().mesh() is self.parent:
                self.field.assign(stored)
            else:
                self.field.assign(stored, allow_missing_dofs=True)
        else:
            self.field.interpolate(stored)

    # -- File I/O --------------------------------------------------------------

    def save(self, checkpoint, **kwargs):
        """Store the field in an open `CheckpointFile`, with its point coordinates.

        For a Quadrature field the point coordinates are stored once per
        file and field name (without a timestepping index), and the
        number of ranks of the writer is recorded, so that `load` reads the
        field on any partition.

        Args:
          checkpoint: a `CheckpointFile` open for writing or appending.
          kwargs: passed to `CheckpointFile.save_function` (`idx`,
            `timestepping_info`).
        """
        checkpoint.save_function(self.encode(), **kwargs)
        checkpoint.require_group(GROUP)
        checkpoint.set_attr(GROUP, "comm_size", self.parent.comm.size)
        if self.quadrature and not _attribute(checkpoint, self.points_name):
            checkpoint.save_function(self.points())
            checkpoint.set_attr(GROUP, self.points_name, 1)

    def load(self, checkpoint, **kwargs):
        """Read the field from an open `CheckpointFile` and restore it.

        The parent must be the mesh that `checkpoint.load_mesh` returned. A
        Quadrature field is paired by position when the file has its point
        coordinates; otherwise it is read on the number of ranks that wrote
        it only (`written_rank_count`).

        Args:
          checkpoint: a `CheckpointFile` open for reading.
          kwargs: passed to `CheckpointFile.load_function` (`idx`).

        Returns:
          The field, restored.

        Raises:
          ValueError: for a Quadrature field without stored coordinates when
            the file records no number of ranks or another one, or when the
            stored points do not match (`restore`).
        """
        stored = checkpoint.load_function(self.parent, self.name, **kwargs)
        if not self.quadrature:
            self.restore(stored)
            return self.field
        if _attribute(checkpoint, self.points_name):
            points = checkpoint.load_function(self.parent, self.points_name)
            self.restore(stored, points=points)
            return self.field
        writer_ranks = written_rank_count(checkpoint, self.parent.name)
        if writer_ranks is None:
            raise ValueError(
                f"The quadrature field {self.name!r} has no stored point "
                "coordinates and the file records no number of ranks of its "
                "writer, so this read cannot be checked against a permutation "
                "of the values inside each cell.")
        self.restore(stored, writer_ranks=writer_ranks)
        return self.field
