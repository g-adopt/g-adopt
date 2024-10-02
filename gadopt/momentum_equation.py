r"""Derived terms and associated equations for the Stokes system.

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

Users should not interact with these classes; instead, please use
the solver provided in the stokes_integrators module.

"""

from typing import Optional

import firedrake as fd
from firedrake import dot, inner, outer, transpose, grad, nabla_grad, div
from firedrake import avg, sym, Identity, jump
from firedrake import FacetArea, CellVolume

from .equations import BaseTerm, BaseEquation
from .utility import is_continuous, normal_is_continuous, tensor_jump, cell_edge_integral_ratio


class ViscosityTerm(BaseTerm):
    r"""Viscosity term $-nabla * (mu nabla u)$ in the momentum equation.

    Using the symmetric interior penalty method, the weak form becomes

    $$
    {:( -int_Omega nabla * (mu grad u) phi dx , = , int_Omega mu (grad phi) * (grad u) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(phi bb n) * "avg"(mu grad u) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(u bb n) * "avg"(mu grad phi) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(mu) "jump"(u bb n) * "jump"(phi bb n) dS )
    :}
    $$

    where σ is a penalty parameter (see Epshteyn and Riviere, 2007).

    Epshteyn, Y., & Rivière, B. (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied Mathematics,
    206(2), 843-872.

    """
    def residual(self, test, trial, trial_lagged, fields, bcs):

        mu = fields['viscosity']
        phi = test
        n = self.n
        u = trial
        u_lagged = trial_lagged
        compressible = self.term_kwargs['compressible']

        grad_test = nabla_grad(phi)
        stress = 2 * mu * sym(grad(u))
        if compressible:
            stress -= 2/3 * mu * Identity(self.dim) * div(u)

        F = 0
        F += inner(grad_test, stress)*self.dx

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
            sigma = alpha * cell_edge_integral_ratio(self.mesh, degree) * nf
        # we use (3.23) + (3.20) from https://www.researchgate.net/publication/260085826
        # instead of maximum over two adjacent cells + and -, we just sum (which is 2*avg())
        # and the for internal facets we have an extra 0.5:
        # WEIRDNESS: avg(1/CellVolume(mesh)) crashes TSFC - whereas it works in scalar diffusion! - instead just writing out explicitly
        sigma *= FacetArea(self.mesh)*(1/CellVolume(self.mesh)('-') + 1/CellVolume(self.mesh)('+'))/2

        if not is_continuous(self.trial_space):
            u_tensor_jump = tensor_jump(n, u) + tensor_jump(u, n)
            if compressible:
                u_tensor_jump -= 2/3 * Identity(self.dim) * jump(u, n)
            F += sigma*inner(tensor_jump(n, phi), avg(mu) * u_tensor_jump)*self.dS
            F += -inner(avg(mu * nabla_grad(phi)), u_tensor_jump)*self.dS
            F += -inner(tensor_jump(n, phi), avg(stress))*self.dS

        for id, bc in bcs.items():
            if 'u' in bc or 'un' in bc:
                if 'u' in bc:
                    u_tensor_jump = outer(n, u-bc['u'])
                    if compressible:
                        u_tensor_jump -= 2/3 * Identity(self.dim) * (dot(n, u) - dot(n, bc['u']))
                else:
                    u_tensor_jump = outer(n, n)*(dot(n, u)-bc['un'])
                    if compressible:
                        u_tensor_jump -= 2/3 * Identity(self.dim) * (dot(n, u) - bc['un'])
                u_tensor_jump += transpose(u_tensor_jump)
                # this corresponds to the same 3 terms as the dS integrals for DG above:
                F += 2*sigma*inner(outer(n, phi), mu * u_tensor_jump)*self.ds(id)
                F += -inner(mu * nabla_grad(phi), u_tensor_jump)*self.ds(id)
                if 'u' in bc:
                    F += -inner(outer(n, phi), stress) * self.ds(id)
                elif 'un' in bc:
                    # we only keep, the normal part of stress, the tangential
                    # part is assumed to be zero stress (i.e. free slip), or prescribed via 'stress'
                    F += -dot(n, phi)*dot(n, dot(stress, n)) * self.ds(id)
            if 'stress' in bc:  # a momentum flux, a.k.a. "force"
                # here we need only the third term, because we assume jump_u=0 (u_ext=u)
                # the provided stress = n.(mu.stress_tensor)
                F += dot(-phi, bc['stress']) * self.ds(id)
            if 'normal_stress' in bc:
                # add the external normal stress
                normal_stress = bc['normal_stress']
                F -= dot(-phi, normal_stress * n) * self.ds(id)

            if 'drag' in bc:  # (bottom) drag of the form tau = -C_D u |u|
                C_D = bc['drag']
                unorm = pow(dot(u_lagged, u_lagged) + 1e-6, 0.5)

                F += dot(-phi, -C_D*unorm*u) * self.ds(id)

            # NOTE 1: unspecified boundaries are equivalent to free stress (i.e. free in all directions)
            # NOTE 2: 'un' can be combined with 'stress' provided the stress force is tangential (e.g. no-normal flow with wind)

            if 'u' in bc and 'stress' in bc:
                raise ValueError("Cannot apply both 'u' and 'stress' bc on same boundary")
            if 'u' in bc and 'normal_stress' in bc:
                raise ValueError("Cannot apply both 'u' and 'normal_stress' bc on same boundary")
            if 'u' in bc and 'drag' in bc:
                raise ValueError("Cannot apply both 'u' and 'drag' bc on same boundary")
            if 'u' in bc and 'un' in bc:
                raise ValueError("Cannot apply both 'u' and 'un' bc on same boundary")
            if 'un' in bc and 'normal_stress' in bc:
                raise ValueError("Cannot apply both 'un' and 'normal_stress' bc on same boundary")

        return -F


class PressureGradientTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        phi = test
        n = self.n
        p = fields['pressure']

        assert normal_is_continuous(phi)
        F = -dot(div(phi), p)*self.dx

        # integration by parts gives natural condition on pressure
        # (as part of a normal stress condition), for boundaries where
        # the normal component of u is specified, we remove that condition
        for id, bc in bcs.items():
            if 'u' in bc or 'un' in bc:
                F += dot(phi, n)*p*self.ds(id)

        return -F


class DivergenceTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        psi = test
        n = self.n
        u = fields['velocity']
        rho = fields.get('rho_continuity', 1)

        assert normal_is_continuous(u)
        F = dot(psi, div(rho * u))*self.dx

        # add boundary integral for bcs that specify normal u-component
        for id, bc in bcs.items():
            if 'u' in bc:
                F += psi * rho * dot(n, bc['u']-u) * self.ds(id)
            elif 'un' in bc:
                F += psi * rho * (bc['un'] - dot(n, u)) * self.ds(id)

        return F


class MomentumSourceTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        if 'source' not in fields:
            return 0

        phi = test
        source = fields['source']

        # NOTE, here source term F is already on the RHS
        F = dot(phi, source)*self.dx

        return F


class MomentumEquation(BaseEquation):
    """Momentum equation with viscosity, pressure gradient, and source terms."""
    terms = [ViscosityTerm, PressureGradientTerm, MomentumSourceTerm]


class ContinuityEquation(BaseEquation):
    """Mass continuity equation with a single divergence term."""
    terms = [DivergenceTerm]


def StokesEquations(
    test_space: fd.functionspaceimpl.WithGeometry,
    trial_space: fd.functionspaceimpl.WithGeometry,
    quad_degree: Optional[int] = None,
    **kwargs,
) -> list[BaseEquation]:
    """Stokes system involving the momentum and mass continuity equations.

    Arguments:
      test_space: Firedrake function space of the test function
      trial_space: Firedrake function space of the trial function
      quad_degree: Quadrature degree. Default value is `2p + 1`, where
                     p is the polynomial degree of the trial space

    Returns:
      A list of equation instances for the Stokes system.

    """
    mom_eq = MomentumEquation(
        test_space.sub(0), trial_space.sub(0), quad_degree=quad_degree, **kwargs
    )
    cty_eq = ContinuityEquation(
        test_space.sub(1), trial_space.sub(1), quad_degree=quad_degree, **kwargs
    )
    return [mom_eq, cty_eq]
