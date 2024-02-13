import firedrake
from firedrake import Constant, FacetNormal, Function, assemble, dot, grad, sqrt, ufl
from firedrake.ufl_expr import extract_unique_domain

from .utility import CombinedSurfaceMeasure

__all__ = ["GeodynamicalDiagnostics"]


class GeodynamicalDiagnostics:
    """Typical simulation diagnostics used in geodynamical simulations."""

    def __init__(
        self,
        u: Function,
        p: Function,
        T: Function,
        bottom_id: int,
        top_id: int,
        degree: int = 4
    ):
        """Initialises the diagnostics instance from the simulation's state.

        Args:
          u:
            Firedrake function for the velocity.
          p:
            Firedrake function for the pressure.
          T:
            Firedrake function for the temperature.
          bottom_id:
            Integer for the bottom boundary identifier.
          top_id:
            Integer for the top boundary identifier.
          degree:
            Integer specifying the degree of the polynomial approximation.
        """
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

    def domain_volume(self) -> ufl.core.expr.Expr:
        """Numerical domain area or volume.

        Returns:
          A UFL expression to calculate the numerical domain's volume.
        """
        return assemble(Constant(1)*firedrake.dx(domain=self.mesh))

    def u_rms(self) -> ufl.core.expr.Expr:
        """Root-mean squared velocity.

        Returns:
          A UFL expression to calculate the root-mean squared velocity.
        """
        return sqrt(assemble(dot(self.u, self.u) * self.dx)) * sqrt(1./self.domain_volume())

    def u_rms_top(self) -> ufl.core.expr.Expr:
        """Root-mean squared velocity along the top boundary.

        Returns:
          A UFL expression to calculate the root-mean squared velocity along the
          domain's top boundary.
        """
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self) -> ufl.core.expr.Expr:
        """Nusselt number at the top boundary.

        Returns:
          A UFL expression to calculate the Nusselt number at the domain's top boundary.
        """
        return -1 * assemble(dot(grad(self.T), self.n) * self.ds_t) * (1./assemble(Constant(1) * self.ds_t))

    def Nu_bottom(self) -> ufl.core.expr.Expr:
        """Nusselt number at the bottom boundary.

        Returns:
          A UFL expression to calculate the Nusselt number at the domain's bottom
          boundary.
        """
        return assemble(dot(grad(self.T), self.n) * self.ds_b) * (1./assemble(Constant(1) * self.ds_b))

    def T_avg(self) -> ufl.core.expr.Expr:
        """Average temperature.

        Returns:
          A UFL expression to calculate the average temperature throughout the domain.
        """
        return assemble(self.T * self.dx) / self.domain_volume()
