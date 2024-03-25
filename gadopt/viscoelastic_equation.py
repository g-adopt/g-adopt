from .equations import BaseTerm
from .momentum_equation import MomentumEquation, ContinuityEquation
from firedrake import nabla_grad, inner
r"""
This module contains the additional terms and equations necessary for viscoelasticity

NOTE: for all terms, the residual() method returns the residual as it would be on the RHS of the equation, i.e.:

  dq/dt = \sum term.residual()

This sign-convention is for compatibility with Thetis' timeintegrators. In general, however we like to think about
the terms as they are on the LHS. Therefore in the residual methods below we assemble in F as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end "return -F".
"""


class PreviousStressTerm(BaseTerm):

    r"""
    Previous stress term :math:`-\nabla \cdot (\mu \nabla u)`

    """
    def residual(self, test, trial, trial_lagged, fields, bcs):

        previous_stress = fields['previous_stress']
        phi = test
        grad_test = nabla_grad(phi)

        F = inner(grad_test, previous_stress)*self.dx

        return -F


class ViscoelasticEquation(MomentumEquation):
    """
    Viscoelastic Equation.
    """
    terms = []
    # Add terms from viscous momentum equation
    for term in MomentumEquation.terms:
        terms.append(term)

    terms.append(PreviousStressTerm)


def ViscoelasticEquations(test_space, trial_space, quad_degree=None, **kwargs):
    mom_eq = ViscoelasticEquation(test_space.sub(0), trial_space.sub(0), quad_degree=quad_degree, **kwargs)
    cty_eq = ContinuityEquation(test_space.sub(1), trial_space.sub(1), quad_degree=quad_degree, **kwargs)
    return [mom_eq, cty_eq]
