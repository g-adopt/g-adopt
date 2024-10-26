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

"""

from firedrake import *

from .equations import Equation, interior_penalty_factor
from .utility import is_continuous, normal_is_continuous, tensor_jump, upward_normal


def viscosity_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    r"""Viscosity term $-nabla * (mu nabla u)$ in the momentum equation.

    Using the symmetric interior penalty method (Epshteyn & Rivière, 2007), the weak
    form becomes

    $$
    {:( -int_Omega nabla * (mu grad u) test dx , = , int_Omega mu (grad test) * (grad u) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(test bb n) * "avg"(mu grad u) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(u bb n) * "avg"(mu grad test) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(mu) "jump"(u bb n) * "jump"(test bb n) dS )
    :}
    $$

    where σ is a penalty parameter.

    Epshteyn, Y., & Rivière, B. (2007).
    Estimation of penalty parameters for symmetric interior penalty Galerkin methods.
    Journal of Computational and Applied Mathematics, 206(2), 843-872.
    """
    stress_old = getattr(eq, "stress_old", 0.0)
    stress = eq.approximation.stress(trial, stress_old)
    F = inner(nabla_grad(eq.test), stress) * eq.dx

    dim = eq.mesh.geometric_dimension()
    identity = Identity(dim)
    mu = eq.approximation.mu
    compressible_stress = "compressible_stress" in eq.approximation.momentum_components

    sigma = interior_penalty_factor(eq)
    sigma *= FacetArea(eq.mesh) / avg(CellVolume(eq.mesh))
    if not is_continuous(eq.trial_space):
        trial_tensor_jump = tensor_jump(eq.n, trial) + tensor_jump(trial, eq.n)
        if compressible_stress:
            trial_tensor_jump -= 2 / 3 * identity * jump(trial, eq.n)

        F += (
            sigma
            * inner(tensor_jump(eq.n, eq.test), avg(mu) * trial_tensor_jump)
            * eq.dS
        )
        F -= inner(avg(mu * nabla_grad(eq.test)), trial_tensor_jump) * eq.dS
        F -= inner(tensor_jump(eq.n, eq.test), avg(stress)) * eq.dS

    # NOTE: Unspecified boundaries are equivalent to free stress (i.e. free in all
    # directions).
    # NOTE: "un" can be combined with "stress" provided the stress force is tangential
    # (e.g. no normal flow with wind)
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
            trial_tensor_jump = outer(eq.n, trial - bc["u"])
            if compressible_stress:
                trial_tensor_jump -= (
                    2 / 3 * identity * (dot(eq.n, trial) - dot(eq.n, bc["u"]))
                )
            trial_tensor_jump += transpose(trial_tensor_jump)
            # Terms below are similar to the above terms for the DG dS integrals.
            F += (
                2
                * sigma
                * inner(outer(eq.n, eq.test), mu * trial_tensor_jump)
                * eq.ds(bc_id)
            )
            F -= inner(mu * nabla_grad(eq.test), trial_tensor_jump) * eq.ds(bc_id)
            F -= inner(outer(eq.n, eq.test), stress) * eq.ds(bc_id)

        if "un" in bc:
            trial_tensor_jump = outer(eq.n, eq.n) * (dot(eq.n, trial) - bc["un"])
            if compressible_stress:
                trial_tensor_jump -= 2 / 3 * identity * (dot(eq.n, trial) - bc["un"])
            trial_tensor_jump += transpose(trial_tensor_jump)
            # Terms below are similar to the above terms for the DG dS integrals.
            F += (
                2
                * sigma
                * inner(outer(eq.n, eq.test), mu * trial_tensor_jump)
                * eq.ds(bc_id)
            )
            F -= inner(mu * nabla_grad(eq.test), trial_tensor_jump) * eq.ds(bc_id)
            # We only keep the normal part of stress; the tangential part is assumed to
            # be zero stress (i.e. free slip) or prescribed via "stress".
            F -= dot(eq.n, eq.test) * dot(eq.n, dot(stress, eq.n)) * eq.ds(bc_id)

        if "stress" in bc:  # a momentum flux, a.k.a. "force"
            # Here we need only the third term because we assume jump_u = 0
            # (u_ext = trial) and stress = n . (mu . stress_tensor).
            F -= dot(eq.test, bc["stress"]) * eq.ds(bc_id)

        if "normal_stress" in bc:
            F += dot(eq.test, bc["normal_stress"] * eq.n) * eq.ds(bc_id)

    return -F


def pressure_gradient_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    assert normal_is_continuous(eq.test)

    F = -dot(div(eq.test), eq.p) * eq.dx

    # Integration by parts gives a natural condition on pressure (as part of a normal
    # stress condition). For boundaries where the normal component of u is specified, we
    # remove that condition.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc or "un" in bc:
            F += dot(eq.test, eq.n) * eq.p * eq.ds(bc_id)

    return -F


def momentum_source_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    p = getattr(eq, "p", 0.0)
    T = getattr(eq, "T", 0.0)
    displ = getattr(eq, "displ", 0.0)

    source = eq.approximation.buoyancy(p=p, T=T, displ=displ) * upward_normal(eq.mesh)

    F = -dot(eq.test, source) * eq.dx

    return -F


def divergence_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    assert normal_is_continuous(eq.u)

    rho = eq.approximation.rho_flux
    F = -dot(eq.test, div(rho * eq.u)) * eq.dx

    # Add boundary integral for bcs that specify the normal component of u.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc:
            F -= eq.test * rho * dot(eq.n, bc["u"] - eq.u) * eq.ds(bc_id)
        elif "un" in bc:
            F -= eq.test * rho * (bc["un"] - dot(eq.n, eq.u)) * eq.ds(bc_id)

    return -F


viscosity_term.required_attrs = set()
viscosity_term.optional_attrs = {"stress_old", "interior_penalty"}
pressure_gradient_term.required_attrs = {"p"}
pressure_gradient_term.optional_attrs = set()
momentum_source_term.required_attrs = set()
momentum_source_term.optional_attrs = {"p", "T", "displ"}
divergence_term.required_attrs = {"u"}
divergence_term.optional_attrs = set()

residual_terms_momentum = [momentum_source_term, pressure_gradient_term, viscosity_term]
residual_terms_mass = divergence_term
residual_terms_stokes = [residual_terms_momentum, residual_terms_mass]
