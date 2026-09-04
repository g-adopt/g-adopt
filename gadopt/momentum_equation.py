r"""Derived terms and associated equations for the Stokes system.

All terms are considered as if they were on the left-hand side of the equation, leading
to the following UFL expression returned by `Equation`'s `residual` method:

$$
  dq / dt + F(q) = 0.
$$

"""

from firedrake import *
from ufl.algorithms import expand_derivatives, extract_coefficients
from ufl.indexed import Indexed

from .approximations import QuasiCompressibleInternalVariableApproximation
from .equations import Equation, interior_penalty_factor
from .utility import (
    depends_on,
    is_continuous,
    normal_is_continuous,
    tensor_jump,
    upward_normal,
)


def viscosity_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    r"""Viscosity term $-nabla * (mu nabla u)$ in the momentum equation.

    Using the symmetric interior penalty method (Epshteyn & Rivière, 2007), the weak
    form becomes

    $$
    {:( -int_Omega nabla * (mu grad u) phi dx , = , int_Omega mu (grad phi) * (grad u) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(phi bb n) * "avg"(mu grad u) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(u bb n) * "avg"(mu grad phi) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(mu) "jump"(u bb n) * "jump"(phi bb n) dS )
    :}
    $$

    where σ is a penalty parameter.

    Epshteyn, Y., & Rivière, B. (2007).
    Estimation of penalty parameters for symmetric interior penalty Galerkin methods.
    Journal of Computational and Applied Mathematics, 206(2), 843-872.
    """
    mu = eq.approximation.mu
    stress = eq.stress
    F = inner(nabla_grad(eq.test), stress) * eq.dx

    # Whether mu depends on the solution `trial` determines two things below:
    # the weak boundary terms always use the tangent stress (so the Jacobian of
    # the weak boundary conditions is symmetric), but the extra
    # penalty-derivative term is only needed when mu itself varies with `trial`.
    # For a viscosity independent of the solution (linear mu) that term is identically
    # zero. The test detects a dependence on any component of the mixed solution
    # that `trial` belongs to, not the velocity specifically: a pressure-
    # dependent mu(p), for example, also sets mu_nonlinear = True. Checking
    # against the coefficient(s) `trial` depends on, rather than its UFL
    # terminals directly, means a spatially varying but solution-independent
    # viscosity (e.g. mu = mu(x)) is correctly treated as linear.
    mu_nonlinear = any(depends_on(mu, c) for c in extract_coefficients(trial))

    sigma = interior_penalty_factor(eq)
    sigma *= FacetArea(eq.mesh) / avg(CellVolume(eq.mesh))
    if not is_continuous(eq.trial_space):
        if mu_nonlinear:
            raise NotImplementedError(
                "Symmetric SIPG interior-facet (dS) terms for a solution-dependent "
                "viscosity are not implemented for discontinuous velocity elements."
            )
        trial_tensor_jump = eq.approximation.stress_per_mu_from_grad(
            tensor_jump(eq.n, trial)
        )

        F += (
            sigma
            * inner(tensor_jump(eq.n, eq.test), avg(mu) * trial_tensor_jump)
            * eq.dS
        )
        F -= inner(avg(mu * nabla_grad(eq.test)), trial_tensor_jump) * eq.dS
        F -= inner(tensor_jump(eq.n, eq.test), avg(stress)) * eq.dS

    # The symmetrising term of a weak velocity boundary condition is the
    # transpose of the flux term, which makes it the derivative of the stress in
    # the direction of the test function, $D\sigma(u)[\phi]$. It is taken from
    # the stress expression this equation carries rather than rebuilt from the
    # approximation, because only that expression holds the state a particular
    # formulation puts into the stress: the internal variables of the
    # viscoelastic solvers, or the stress carried over from the previous step.
    # Differentiating it picks up the right shear coefficient in each case, and
    # for a solution-dependent viscosity it also picks up the $D\mu[\phi]$
    # contribution that makes the boundary Jacobian symmetric.
    weak_velocity_bcs = any(bc.keys() & {"u", "un"} for bc in eq.bcs.values())
    if weak_velocity_bcs:
        if not any(depends_on(stress, c) for c in extract_coefficients(trial)):
            raise ValueError(
                "The stress supplied to viscosity_term does not depend on the "
                "trial function, so the symmetrising term of the weak velocity "
                "boundary conditions would be identically zero and their "
                "Jacobian would not be symmetric. Build the stress from the "
                "same function the residual is evaluated at."
            )
        tangent_stress = expand_derivatives(derivative(stress, trial, eq.test))

    # NOTE: Unspecified boundaries result in free stress (i.e. free in all directions).
    # NOTE: "un" can be combined with "stress" provided the stress component is
    # tangential (e.g. no normal flow with wind)
    for bc_id, bc in eq.bcs.items():
        if "u" in bc and any(bc_type in bc for bc_type in ["stress", "un"]):
            raise ValueError(
                '"stress" or "un" cannot be specified if "u" is already given.'
            )
        if "normal_stress" in bc and any(bc_type in bc for bc_type in ["u", "un"]):
            raise ValueError(
                '"u" or "un" cannot be specified if "normal_stress" is already given.'
            )

        if "u" in bc:
            w = trial - bc["u"]
            jump_gradient = outer(eq.n, w)
            # Penalty term, similar to the above term for the DG dS integrals.
            # The approximation converts the boundary jump tensor into a stress
            # the same way it converts a velocity gradient, so the penalty
            # carries every part of the stress the boundary condition
            # constrains, volumetric part included.
            F += (
                2
                * sigma
                * inner(
                    outer(eq.n, eq.test),
                    eq.approximation.stress_from_grad(jump_gradient),
                )
                * eq.ds(bc_id)
            )
            # Symmetrising term, the transpose of the flux integration by parts.
            F -= dot(w, dot(tangent_stress, eq.n)) * eq.ds(bc_id)
            F -= inner(outer(eq.n, eq.test), stress) * eq.ds(bc_id)
            # Derivative of the penalty term through mu: the penalty functional
            # is $\sigma_{pen}\,\langle G, S(G)\rangle$ with
            # $G = n \otimes w$ (jump_gradient) and $S$ the stress from a
            # gradient-like tensor, so its exact first variation (the residual)
            # picks up this extra term whenever $\mu$ itself depends on the
            # trial. Only the deviatoric part of $S$ scales with $\mu$, so only
            # that part appears here. This makes the resulting Newton Jacobian
            # (the second variation) symmetric by construction.
            if mu_nonlinear:
                dmu = expand_derivatives(derivative(mu, trial, eq.test))
                jump_tensor = eq.approximation.stress_per_mu_from_grad(jump_gradient)
                F += sigma * dmu * inner(jump_gradient, jump_tensor) * eq.ds(bc_id)

        if "un" in bc:
            un_jump = dot(eq.n, trial) - bc["un"]
            w = un_jump * eq.n
            jump_gradient = outer(eq.n, w)
            # Penalty term, as in the "u" branch but with the jump restricted to
            # its normal component. The trace of the jump tensor is the normal
            # jump itself, so an approximation with a bulk modulus penalises the
            # normal jump volumetrically as well as deviatorically.
            F += (
                2
                * sigma
                * inner(
                    outer(eq.n, eq.test),
                    eq.approximation.stress_from_grad(jump_gradient),
                )
                * eq.ds(bc_id)
            )
            # Symmetrising term, as in the "u" branch but with the jump restricted
            # to its normal component (free-slip/free-stress tangential direction).
            F -= dot(w, dot(tangent_stress, eq.n)) * eq.ds(bc_id)
            # We only keep the normal part of stress; the tangential part is assumed to
            # be zero stress (i.e. free slip) or prescribed via "stress".
            F -= dot(eq.n, eq.test) * dot(eq.n, dot(stress, eq.n)) * eq.ds(bc_id)
            # Derivative of the penalty term through mu, as in the "u" branch
            # above, restricted to the normal component of the jump.
            if mu_nonlinear:
                dmu = expand_derivatives(derivative(mu, trial, eq.test))
                jump_tensor = eq.approximation.stress_per_mu_from_grad(jump_gradient)
                F += sigma * dmu * inner(jump_gradient, jump_tensor) * eq.ds(bc_id)

        if "stress" in bc:  # a momentum flux, a.k.a. "force"
            # Here we need only the third term because we assume jump_u = 0
            # (u_ext = trial) and stress = n . (mu . stress_tensor).
            F -= dot(eq.test, bc["stress"]) * eq.ds(bc_id)

        if "normal_stress" in bc:
            F += dot(eq.test, bc["normal_stress"] * eq.n) * eq.ds(bc_id)

    return F


def pressure_gradient_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    assert normal_is_continuous(eq.test)

    F = -dot(div(eq.test), eq.p) * eq.dx

    # Integration by parts gives a natural condition on pressure (as part of a normal
    # stress condition). For boundaries where the normal component of u is specified, we
    # remove that condition.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc or "un" in bc:
            F += dot(eq.test, eq.n) * eq.p * eq.ds(bc_id)

    return F


def divergence_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    assert normal_is_continuous(eq.u)

    rho = eq.rho_continuity
    F = -dot(eq.test, div(rho * eq.u)) * eq.dx

    # Add boundary integral for bcs that specify the normal component of u.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc:
            F -= eq.test * rho * dot(eq.n, bc["u"] - eq.u) * eq.ds(bc_id)
        elif "un" in bc:
            F -= eq.test * rho * (bc["un"] - dot(eq.n, eq.u)) * eq.ds(bc_id)

    return F


def momentum_source_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    return -dot(eq.test, eq.source) * eq.dx


def hydrostatic_prestress_advection_and_buoyancy_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    # The advection of hydrostatic prestress and buoyancy terms are combined
    # to form an explicitly symmetric term, following Eqs. B22-B29 in
    # Appendix B of Al-Attar et al. 2014 and Scott et al. 2026, the full references
    # are provided in `approximations.py`.

    # For the Cathles 2024 benchmark in `tests/viscoelastic_internal_variable`
    # we neglect these terms to be consistent with the analytical solution
    if isinstance(eq.approximation, QuasiCompressibleInternalVariableApproximation):
        return 0

    B_mu = eq.approximation.B_mu
    rho0 = eq.approximation.density
    g = eq.approximation.g
    grad_phi = g * upward_normal(eq.mesh)

    F = 0.5 * B_mu * rho0 * dot(grad(dot(trial, grad_phi)), eq.test) * eq.dx
    F += 0.5 * B_mu * rho0 * dot(trial, grad(dot(eq.test, grad_phi))) * eq.dx

    F -= 0.5 * B_mu * rho0 * dot(div(trial)*grad_phi, eq.test) * eq.dx
    F -= 0.5 * B_mu * rho0 * dot(grad_phi, trial) * div(eq.test) * eq.dx

    return F


viscosity_term.required_attrs = {"stress"}
viscosity_term.optional_attrs = {"interior_penalty"}
pressure_gradient_term.required_attrs = {"p"}
pressure_gradient_term.optional_attrs = set()
divergence_term.required_attrs = {"u", "rho_continuity"}
divergence_term.optional_attrs = set()
momentum_source_term.required_attrs = {"source"}
momentum_source_term.optional_attrs = set()
hydrostatic_prestress_advection_and_buoyancy_term.required_attrs = set()
hydrostatic_prestress_advection_and_buoyancy_term.optional_attrs = set()

momentum_terms = [momentum_source_term, pressure_gradient_term, viscosity_term]
mass_terms = divergence_term
stokes_terms = [momentum_terms, mass_terms]

compressible_viscoelastic_terms = [
    hydrostatic_prestress_advection_and_buoyancy_term,
    momentum_source_term,
    viscosity_term,
]
