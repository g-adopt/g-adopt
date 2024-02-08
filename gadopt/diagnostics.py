from firedrake import (
    Constant, FacetNormal, SpatialCoordinate,
    assemble, conditional, dot, ds, dx, grad, norm, sqrt,
)
from firedrake.ufl_expr import extract_unique_domain

from .utility import CombinedSurfaceMeasure


def domain_volume(mesh):
    return assemble(Constant(1)*dx(domain=mesh))


def entrainment(level_set, material_area, entrainment_height):
    mesh_coords = SpatialCoordinate(level_set.function_space().mesh())
    target_region = mesh_coords[1] >= entrainment_height
    material_entrained = conditional(level_set < 0.5, 1, 0)

    return (
        assemble(conditional(target_region, material_entrained, 0) * dx)
        / material_area
    )


def rms_velocity(velocity):
    return norm(velocity) / sqrt(domain_volume(extract_unique_domain(velocity)))


class GeodynamicalDiagnostics:

    def __init__(self, u, p, T, bottom_id, top_id, degree=4):
        mesh = extract_unique_domain(u)
        mesh = extract_unique_domain(u)
        self.domain_volume = domain_volume(mesh)
        self.u = u
        self.p = p
        self.T = T
        self.dx = dx(degree=degree)
        if T.function_space().extruded:
            self.ds = CombinedSurfaceMeasure(mesh, degree)
        else:
            self.ds = ds(mesh)
        self.ds_t = self.ds(top_id)
        self.ds_b = self.ds(bottom_id)
        self.n = FacetNormal(mesh)

    def u_rms(self):
        return sqrt(assemble(dot(self.u, self.u) * self.dx)) * sqrt(1./self.domain_volume)

    def u_rms_top(self):
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return -1 * assemble(dot(grad(self.T), self.n) * self.ds_t) * (1./assemble(Constant(1) * self.ds_t))

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self.n) * self.ds_b) * (1./assemble(Constant(1) * self.ds_b))

    def T_avg(self):
        return assemble(self.T * self.dx) / self.domain_volume
