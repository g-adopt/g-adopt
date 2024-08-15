from firedrake import FacetNormal, dot

from .equations import BaseEquation, BaseTerm
from .utility import vertical_component

r"""
This module contains the free surface terms and equations

NOTE: for all terms, the residual() method returns the residual as it would be on the RHS of the equation, i.e.:

  dq/dt = \sum term.residual()

This sign-convention is for compatibility with Thetis' timeintegrators. In general, however we like to think about
the terms as they are on the LHS. Therefore in the residual methods below we assemble in F as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end "return -F".
"""


class FreeSurfaceTerm(BaseTerm):
    r"""Free Surface term: u \dot n"""

    def residual(self, test, trial, trial_lagged, fields, bcs):
        u = fields["velocity"]
        free_surface_id = fields["free_surface_id"]
        prefactor = fields["prefactor"]

        # Note this term is already on the RHS
        F = prefactor * test * dot(u, self.n) * self.ds(free_surface_id)

        return F


class FreeSurfaceEquation(BaseEquation):
    """Free Surface Equation."""

    terms = [FreeSurfaceTerm]

    def __init__(
        self, test_space, trial_space, free_surface_id, prefactor, quad_degree=None
    ):
        self.free_surface_id = free_surface_id
        self.prefactor = prefactor
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        r"""Return the UFL for the mass term \int test * trial * ds for the free surface time derivative term integrated over the free surface."""
        n = FacetNormal(self.mesh)
        return (
            self.prefactor
            * vertical_component(n)
            * dot(test, trial)
            * self.ds(self.free_surface_id)
        )
