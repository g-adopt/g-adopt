r"""Scalar terms and equations (e.g. for temperature and salinity transport).

All terms are considered as if they were on the right-hand side of the equation, leading
to the following UFL expression returned by the `residual` method:

$$
  (dq)/dt = sum "term.residual()"
$$

This sign convention ensures compatibility with Thetis's time integrators. In general,
however, we like to think about the terms as they are on the left-hand side. Therefore,
in the residual methods below, we first sum the terms in the variable `F` as if they
were on the left-hand side, i.e.

$$
  (dq)/dt + F(q) = 0,
$$

and then return `-F`.
"""
from typing import Optional

import firedrake as fd
from firedrake import avg, div, dot, grad, inner, jump, min_value, sign
from firedrake import CellVolume, FacetArea, Identity

from .equations import BaseEquation, BaseTerm
from .utility import cell_edge_integral_ratio, is_continuous, normal_is_continuous


class ScalarAdvectionTerm(BaseTerm):
    r"""Scalar advection term (non-conservative): u \dot \div(q)."""
    def residual(
        self,
        test: fd.ufl_expr.Argument,
        trial: fd.ufl_expr.Argument | fd.Function,
        trial_lagged: Optional[fd.ufl_expr.Argument | fd.Function] = None,
        fields: Optional[dict[str, fd.Constant | fd.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> fd.ufl.core.expr.Expr:
        u = fields['velocity']
        phi = test
        n = self.n
        q = trial
        continuous_u_normal = normal_is_continuous(u)
        if 'advective_velocity_scaling' in fields:
            u = fields['advective_velocity_scaling'] * u

        if 'su_nubar' in fields:
            # SU(PG) ala Donea & Huerta 2003
            nubar = fields['su_nubar']
            phi = phi + nubar / (dot(u, u)+1e-12) * dot(u, grad(phi))

            F = phi * dot(u, grad(q)) * self.dx  # The advection term is not integrated by parts so there are no boundary terms

        else:
            F = -q*div(phi*u)*self.dx

            # integration by parts leads to boundary term
            F += q*dot(n, u)*phi*self.ds

            if not (is_continuous(self.trial_space) and continuous_u_normal):
                # s=0: u.n(-)<0  =>  flow goes from '+' to '-' => '+' is upwind
                # s=1: u.n(-)>0  =>  flow goes from '-' to '+' => '-' is upwind
                s = 0.5*(sign(dot(avg(u), n('-'))) + 1.0)
                q_up = q('-')*s + q('+')*(1-s)
                F += jump(phi*u, n) * q_up * self.dS

        # replace boundary value on incoming Dirichlet 'q' boundaries:
        for id, bc in bcs.items():
            if 'q' in bc:
                # on incoming boundaries, dot(u,n)<0, replace q with bc['q']
                F += phi*min_value(dot(u, n), 0)*(bc['q']-q) * self.ds(id)

        return -F


class ScalarDiffusionTerm(BaseTerm):
    r"""Scalar diffusion term $-nabla * (kappa grad q)$.

    Using the symmetric interior penalty method, the weak form becomes

    $$
    {:( -int_Omega nabla * (kappa grad q) phi dx , = , int_Omega kappa (grad phi) * (grad q) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(phi bb n) * "avg"(kappa grad q) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(q bb n) * "avg"(kappa grad phi) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(kappa) "jump"(q bb n) * "jump"(phi bb n) dS )
    :}
    $$

    where σ is a penalty parameter (see Epshteyn and Riviere, 2007).

    Epshteyn, Y., & Rivière, B. (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied Mathematics,
    206(2), 843-872.

    """
    def residual(
        self,
        test: fd.ufl_expr.Argument,
        trial: fd.ufl_expr.Argument | fd.Function,
        trial_lagged: Optional[fd.ufl_expr.Argument | fd.Function] = None,
        fields: Optional[dict[str, fd.Constant | fd.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> fd.ufl.core.expr.Expr:

        kappa = fields['diffusivity']
        if len(kappa.ufl_shape) == 2:
            diff_tensor = kappa
        else:
            diff_tensor = kappa * Identity(self.dim)

        phi = test
        n = self.n
        q = trial
        if 'reference_for_diffusion' in fields:
            q += fields['reference_for_diffusion']

        grad_test = grad(phi)
        diff_flux = dot(diff_tensor, grad(q))

        F = 0
        F += inner(grad_test, diff_flux)*self.dx

        # Interior Penalty method
        #
        # see https://www.researchgate.net/publication/260085826 for details
        # on the choice of sigma

        degree = self.trial_space.ufl_element().degree()
        if not isinstance(degree, int):
            degree = max(degree[0], degree[1])
        # safety factor: 1.0 is theoretical minimum
        alpha = fields.get('interior_penalty', 2.0)
        if degree == 0:
            # probably only works for orthog. quads and hexes
            sigma = 1.0
        else:
            nf = self.mesh.ufl_cell().num_facets()
            sigma = alpha * cell_edge_integral_ratio(self.mesh, degree-1) * nf

        if not is_continuous(self.trial_space):
            # we use (3.23) + (3.20) from https://www.researchgate.net/publication/260085826
            # instead of maximum over two adjacent cells + and -, we just sum (which is 2*avg())
            # and the for internal facets we have an extra 0.5:
            sigma_int = sigma * avg(FacetArea(self.mesh)/CellVolume(self.mesh))
            F += sigma_int*inner(jump(phi, n), dot(avg(diff_tensor), jump(q, n)))*self.dS
            F += -inner(avg(dot(diff_tensor, grad(phi))), jump(q, n))*self.dS
            F += -inner(jump(phi, n), avg(dot(diff_tensor, grad(q))))*self.dS

        for id, bc in bcs.items():
            if 'q' in bc:
                jump_q = q-bc['q']
                sigma_ext = sigma * fd.FacetArea(self.mesh)/fd.CellVolume(self.mesh)
                # this corresponds to the same 3 terms as the dS integrals for DG above:
                F += 2*sigma_ext*phi*inner(n, dot(diff_tensor, n))*jump_q*self.ds(id)
                F += -inner(dot(diff_tensor, grad(phi)), n)*jump_q*self.ds(id)
                F += -inner(phi*n, dot(diff_tensor, grad(q))) * self.ds(id)
                if 'flux' in bc:
                    raise ValueError("Cannot apply both `q` and `flux` bc on same boundary")
            elif 'flux' in bc:
                # here we need only the third term, because we assume jump_q=0 (q_ext=q)
                # the provided flux = kappa dq/dn = dot(n, dot(diff_tensor, grad(q))
                F += -phi*bc['flux']*self.ds(id)

        return -F


class ScalarSourceTerm(BaseTerm):
    r"""Scalar source term `s_T`."""

    def residual(
        self,
        test: fd.ufl_expr.Argument,
        trial: fd.ufl_expr.Argument | fd.Function,
        trial_lagged: Optional[fd.ufl_expr.Argument | fd.Function] = None,
        fields: Optional[dict[str, fd.Constant | fd.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> fd.ufl.core.expr.Expr:
        if 'source' not in fields:
            return 0
        phi = test
        source = fields['source']

        # NOTE, here source term F is already on the RHS
        F = dot(phi, source)*self.dx

        return F


class ScalarAbsorptionTerm(BaseTerm):
    r"""Scalar absorption term `\alpha_T T`."""

    def residual(
        self,
        test: fd.ufl_expr.Argument,
        trial: fd.ufl_expr.Argument | fd.Function,
        trial_lagged: Optional[fd.ufl_expr.Argument | fd.Function] = None,
        fields: Optional[dict[str, fd.Constant | fd.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> fd.ufl.core.expr.Expr:
        if 'absorption_coefficient' not in fields:
            return 0

        phi = test
        alpha = fields['absorption_coefficient']

        # The absorption term F is already on the right-hand side.
        # Implement absorption term implicitly at current time step.
        F = -dot(phi, alpha*trial)*self.dx

        return F


class ScalarAdvectionEquation(BaseEquation):
    """Scalar advection equation with source and absorption terms."""
    terms = [ScalarAdvectionTerm, ScalarSourceTerm, ScalarAbsorptionTerm]


class ScalarAdvectionDiffusionEquation(BaseEquation):
    """Scalar advection-diffusion equation with source and absorption terms."""
    terms = [ScalarAdvectionTerm, ScalarDiffusionTerm, ScalarSourceTerm, ScalarAbsorptionTerm]


class EnergyEquation(ScalarAdvectionDiffusionEquation):
    """Energy equation defined as an advection-diffusion equation."""
    def __init__(
        self,
        test_space: fd.functionspaceimpl.WithGeometry,
        trial_space: fd.functionspaceimpl.WithGeometry,
        rhocp: Optional[fd.ufl.core.expr.Expr] = None,
        quad_degree: Optional[int] = None,
    ):
        self.rhocp = rhocp
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(
        self,
        test: fd.ufl_expr.Argument,
        trial: fd.ufl_expr.Argument | fd.Function,
    ) -> fd.ufl.core.expr.Expr:
        """UFL expression for the mass term used in the time discretisation.

        Arguments:
          test: Firedrake test function
          trial: Firedrake trial function

        Returns:
          The UFL expression associated with the mass term of the equation.

        """
        return self.rhocp * dot(test, trial) * self.dx
