import firedrake as fd
from firedrake import Constant, FacetNormal, assemble, dot, grad, sqrt

from .utility import CombinedSurfaceMeasure


def domain_volume(mesh):
    return fd.assemble(fd.Constant(1) * fd.dx(domain=mesh))


def entrainment(
    level_set,
    domain_length_x,
    material_interface_y,
    entrainment_height,
):
    mesh_coords = fd.SpatialCoordinate(level_set.function_space().mesh())
    return (
        fd.assemble(
            fd.conditional(
                mesh_coords[1] >= entrainment_height,
                fd.conditional(level_set > 0.5, 1, 0),
                0,
            )
            * fd.dx
        )
        / domain_length_x
        / material_interface_y
    )


def rms_velocity(velocity):
    return fd.norm(velocity) / fd.sqrt(domain_volume(velocity.ufl_domain()))


class GeodynamicalDiagnostics:
    def __init__(self, u, p, T, bottom_id, top_id, degree=4):
        mesh = u.ufl_domain()
        self.domain_volume = domain_volume(mesh)
        self.u = u
        self.p = p
        self.T = T
        self.dx = fd.dx(degree=degree)
        if T.function_space().extruded:
            ds = CombinedSurfaceMeasure(mesh, degree)
        else:
            ds = fd.ds(mesh)
        self.ds_t = ds(top_id)
        self.ds_b = ds(bottom_id)
        self.n = FacetNormal(mesh)

    def u_rms(self):
        return sqrt(assemble(dot(self.u, self.u) * self.dx)) * sqrt(
            1.0 / self.domain_volume
        )

    def u_rms_top(self):
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return (
            -1
            * assemble(dot(grad(self.T), self.n) * self.ds_t)
            * (1.0 / assemble(Constant(1) * self.ds_t))
        )

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self.n) * self.ds_b) * (
            1.0 / assemble(Constant(1) * self.ds_b)
        )

    def T_avg(self):
        return assemble(self.T * self.dx) / self.domain_volume
