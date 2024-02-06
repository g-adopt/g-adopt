from .equations import BaseTerm, BaseEquation
from firedrake import dot, FacetNormal
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
    r"""
    Free Surface term: u \dot n
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):
        assert 'free_surface_id' in self.term_kwargs
        free_surface_id = self.term_kwargs['free_surface_id']

        # Multiply by constant factor to keep the block system symmetric for the implicit coupling case
        if 'theta' in self.term_kwargs:
            assert 'free_surface_dt' in self.term_kwargs
            free_surface_dt = self.term_kwargs['free_surface_dt']
            theta = self.term_kwargs['theta']
            prefactor = free_surface_dt * theta
        else:
            prefactor = 1

        u = fields['velocity']
        psi = test
        n = self.n

        F = prefactor * psi * dot(u, n) * self.ds(free_surface_id)  # Note this term is already on the RHS

        return F


class FreeSurfaceEquation(BaseEquation):
    """
    Free Surface Equation.
    """

    terms = [FreeSurfaceTerm]

    def mass_term(self, test, trial):
        r"""Return the UFL for the mass term \int test * trial * ds for the free surface time derivative term integrated over the free surface."""
        assert 'free_surface_id' in self.kwargs
        free_surface_id = self.kwargs['free_surface_id']

        assert 'k' in self.kwargs
        k = self.kwargs['k']
        n = FacetNormal(self.mesh)

        return dot(n, k) * dot(test, trial) * self.ds(free_surface_id)
