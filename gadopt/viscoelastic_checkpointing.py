"""Lossless parent-mesh storage of GIA fields, including quadrature histories.

CheckpointFile projects unsupported elements into polynomial spaces. Quadrature
values have no polynomial extension, so store their independent components as
one DG0 vector per cell instead. This is an encoding, not a DG0 constitutive
approximation. Application checkpoint I/O is outside the differentiated march.
"""
import firedrake as fd
from finat.ufl import BrokenElement


class CheckpointField:
    """Encode a field on a parent mesh and restore it through the submesh map."""

    def __init__(self, name, field, parent, *, broken=True):
        self.field = field
        self.parent = parent
        self.broken = broken
        space = field.function_space()
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
            element = self.local.ufl_element()
        elif space.mesh() is not parent and broken:
            element = BrokenElement(space.ufl_element())
            self.local = fd.Function(fd.FunctionSpace(space.mesh(), element))
        else:
            element = space.ufl_element()
            self.local = field
        self.stored = fd.Function(fd.FunctionSpace(parent, element), name=name)

    def encode(self):
        if self.quadrature:
            data = self.field.dat.data_ro_with_halos[self.nodes[:self.cells]]
            # `data_rw` because the write is indexed: in pyop3 a write-only
            # access drops a pending halo reduction of the whole vector.
            # The width is explicit, because a rank that owns no cell of a
            # submesh has an empty array, and numpy cannot infer -1 from 0.
            self.local.dat.data_rw[self.local_nodes] = data.reshape(self.cells, self.width)
        elif self.local is not self.field:
            self.local.interpolate(self.field)
        if self.local.function_space().mesh() is self.parent:
            self.stored.assign(self.local)
        elif not self.broken and not self.quadrature:
            self.stored.assign(self.local, allow_missing_dofs=True)
        else:
            self.stored.interpolate(self.local, allow_missing_dofs=True)
        return self.stored

    def restore(self, stored):
        if self.quadrature:
            if stored.ufl_shape != self.stored.ufl_shape:
                raise ValueError("Checkpoint quadrature storage shape does not match the material history")
            self.local.interpolate(stored)
            shape = (self.cells, self.nodes.shape[1], *self.field.dat.data_ro.shape[1:])
            self.field.dat.data_rw[self.nodes[:self.cells]] = (
                self.local.dat.data_ro[self.local_nodes].reshape(shape))
        elif self.field.function_space().ufl_element().family() == "Real":
            # The array with ghosts: a `Real` value is owned by rank 0 only
            # and is a ghost entry on the other ranks.
            self.field.assign(float(stored.dat.data_ro_with_halos[0]))
        elif not self.broken:
            self.field.assign(stored)
        else:
            self.field.interpolate(stored)
