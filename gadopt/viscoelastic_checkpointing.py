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
            self.nodes = space.cell_node_map().values
            self.cells = space.mesh().cell_set.size
            width = self.nodes.shape[1]*field.dat.cdim
            self.local = fd.Function(fd.VectorFunctionSpace(space.mesh(), "DG", 0, dim=width))
            self.local_nodes = self.local.function_space().cell_node_map().values[:self.cells, 0]
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
            self.local.dat.data_wo[self.local_nodes] = data.reshape(self.cells, -1)
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
            self.field.dat.data_wo[self.nodes[:self.cells]] = (
                self.local.dat.data_ro[self.local_nodes].reshape(shape))
        elif self.field.function_space().ufl_element().family() == "Real":
            self.field.assign(float(stored.dat.data_ro[0]))
        elif not self.broken:
            self.field.assign(stored)
        else:
            self.field.interpolate(stored)
