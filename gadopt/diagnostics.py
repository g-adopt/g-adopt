from firedrake import (
    Constant, FacetNormal, Function, SpatialCoordinate,
    assemble, conditional, dot, ds, dx, grad, norm, sqrt,
)
from firedrake.ufl_expr import extract_unique_domain

from .utility import CombinedSurfaceMeasure


class GeodynamicalDiagnostics:
    """Typical simulation diagnostics used in geodynamical simulations.

    Arguments:
      u:         Firedrake function for the velocity
      p:         Firedrake function for the pressure
      T:         Firedrake function for the temperature
      bottom_id: bottom boundary identifier
      top_id:    top boundary identifier
      diag_vars: dictionary containing additional diagnostic-related functions
      degree:    degree of the polynomial approximation
      
    Note:
      All the diagnostics are returned as a float value.

    Functions:
      u_rms: Root-mean squared velocity
      u_rms_top: Root-mean squared velocity along the top boundary
      Nu_top: Nusselt number at the top boundary
      Nu_bottom: Nusselt number at the bottom boundary
      T_avg: Average temperature in the domain
      entrainment: Proportion of material entrained into a target region

    """

    def __init__(
        self,
        u: Function,
        p: Function,
        T: Function,
        bottom_id: int,
        top_id: int,
        diag_vars: dict = {},
        degree: int = 4
    ):
        mesh = extract_unique_domain(u)

        self.u = u
        self.p = p
        self.T = T
        self.diag_vars = diag_vars

        self.dx = dx(domain=mesh, degree=degree)
        self.ds = (
            CombinedSurfaceMeasure(mesh, degree)
            if T.function_space().extruded
            else ds(mesh)
        )
        self.ds_t = self.ds(top_id)
        self.ds_b = self.ds(bottom_id)

        self.n = FacetNormal(mesh)

        self.domain_volume = assemble(Constant(1) * self.dx)
        self.top_surface = assemble(Constant(1) * self.ds_t)
        self.bottom_surface = assemble(Constant(1) * self.ds_b)

    def u_rms(self):
        return norm(self.u) / sqrt(self.domain_volume)

    def u_rms_top(self) -> float:
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return -assemble(dot(grad(self.T), self.n) * self.ds_t) / self.top_surface

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self.n) * self.ds_b) / self.bottom_surface

    def T_avg(self):
        return assemble(self.T * self.dx) / self.domain_volume

    def entrainment(self, level_set_index, material_area, entrainment_height):
        level_set = self.diag_vars["level_set"][level_set_index]

        mesh_coords = SpatialCoordinate(level_set.function_space().mesh())
        target_region = mesh_coords[1] >= entrainment_height
        material_entrained = conditional(level_set < 0.5, 1, 0)

        return (
            assemble(conditional(target_region, material_entrained, 0) * self.dx)
            / material_area
        )
