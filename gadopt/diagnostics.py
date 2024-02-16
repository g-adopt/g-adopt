import firedrake
from firedrake import assemble, Constant, Function, sqrt, dot, grad, FacetNormal
from firedrake.ufl_expr import extract_unique_domain
from .utility import CombinedSurfaceMeasure


class GeodynamicalDiagnostics:
    """Typical simulation diagnostics used in geodynamical simulations.

    Arguments:
      u:         Firedrake function for the velocity
      p:         Firedrake function for the pressure
      T:         Firedrake function for the temperature
      bottom_id: bottom boundary identifier.
      top_id:    top boundary identifier.
      degree:    degree of the polynomial approximation.

    Note:
      All the diagnostics are returned as a float value.

    Functions:
      domain_volume: The numerical domain's volume
      u_rms: Root-mean squared velocity
      u_rms_top: Root-mean squared velocity along the top boundary
      Nu_top: Nusselt number at the top boundary
      Nu_bottom: Nusselt number at the bottom boundary
      T_avg: Average temperature in the domain

    """

    def __init__(
        self,
        u: Function,
        p: Function,
        T: Function,
        bottom_id: int,
        top_id: int,
        degree: int = 4
    ):
        self.mesh = extract_unique_domain(u)
        self.u = u
        self.p = p
        self.T = T

        self.dx = firedrake.dx(degree=degree)
        if T.function_space().extruded:
            ds = CombinedSurfaceMeasure(self.mesh, degree)
        else:
            ds = firedrake.ds(self.mesh)
        self.ds_t = ds(top_id)
        self.ds_b = ds(bottom_id)

        self.n = FacetNormal(self.mesh)

    def domain_volume(self) -> float:
        return assemble(Constant(1)*firedrake.dx(domain=self.mesh))

    def u_rms(self) -> float:
        return sqrt(assemble(dot(self.u, self.u) * self.dx)) * sqrt(1./self.domain_volume())

    def u_rms_top(self) -> float:
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self) -> float:
        return -1 * assemble(dot(grad(self.T), self.n) * self.ds_t) * (1./assemble(Constant(1) * self.ds_t))

    def Nu_bottom(self) -> float:
        return assemble(dot(grad(self.T), self.n) * self.ds_b) * (1./assemble(Constant(1) * self.ds_b))

    def T_avg(self) -> float:
        return assemble(self.T * self.dx) / self.domain_volume()
