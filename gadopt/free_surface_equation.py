from .equations import BaseTerm, BaseEquation
from firedrake import dot
r"""
This module contains the scalar terms and equations (e.g. for temperature and salinity transport)

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
        u = fields['velocity']
        assert 'surface_id' in fields
        surface_id = fields['surface_id']
        psi = test
        n = self.n

#        print("eta u.. inside: ", u.dat.data[:])

        F = psi * dot(u, n) * self.ds(surface_id)  # Note this term is already on the RHS

        return F


class FreeSurfaceEquation(BaseEquation):
    """
    Free Surface Equation.
    """

    terms = [FreeSurfaceTerm]
