r"""Derived terms and associated equations for the Stokes system.

All terms are considered as if they were on the left-hand side of the equation, leading
to the following UFL expression returned by `Equation`'s `residual` method:

$$
  dq / dt + F(q) = 0.
$$

"""

from firedrake import *
from ufl.indexed import Indexed

from .approximations import QuasiCompressibleInternalVariableApproximation
from .equations import Equation, interior_penalty_factor
from .utility import (
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
    dim = eq.mesh.geometric_dimension
    identity = Identity(dim)
    compressible_stress = eq.approximation.compressible

    mu = eq.approximation.mu
    stress = eq.stress
    F = inner(nabla_grad(eq.test), stress) * eq.dx

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
            trial_tensor_jump = 2 * sym(outer(eq.n, trial - bc["u"]))
            if compressible_stress:
                trial_tensor_jump -= (
                    2 / 3 * identity * (dot(eq.n, trial) - dot(eq.n, bc["u"]))
                )
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
            trial_tensor_jump = 2 * outer(eq.n, eq.n) * (dot(eq.n, trial) - bc["un"])
            if compressible_stress:
                trial_tensor_jump -= 2 / 3 * identity * (dot(eq.n, trial) - bc["un"])
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

            if hasattr(eq.approximation, 'bulk_modulus'):
                trial_tensor_jump = identity * (dot(eq.n, trial) - bc["un"])
                bulk = eq.approximation.bulk_modulus * eq.approximation.bulk_shear_ratio
                # Terms below are similar to the above terms for the DG dS integrals.
                F += (
                    2
                    * sigma
                    * inner(outer(eq.n, eq.test), bulk * trial_tensor_jump)
                    * eq.ds(bc_id)
                )
                F -= inner(bulk * nabla_grad(eq.test), trial_tensor_jump) * eq.ds(bc_id)

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

def grad_div_term(eq: Equation, trial: Argument | Indexed | Function) -> Form:
    if not hasattr(eq, "gamma"):
        return 0

    F = eq.gamma * inner(cell_avg(div(eq.test)), div(trial)) * eq.dx(metadata={"mode": "vanilla"})
    return -F


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
grad_div_term.required_attrs = set()
grad_div_term.optional_attrs = {"gamma"}
momentum_source_term.required_attrs = {"source"}
momentum_source_term.optional_attrs = set()
hydrostatic_prestress_advection_and_buoyancy_term.required_attrs = set()
hydrostatic_prestress_advection_and_buoyancy_term.optional_attrs = set()

momentum_terms = [momentum_source_term, pressure_gradient_term, viscosity_term, grad_div_term]
mass_terms = divergence_term
stokes_terms = [momentum_terms, mass_terms]

compressible_viscoelastic_terms = [
    hydrostatic_prestress_advection_and_buoyancy_term,
    momentum_source_term,
    viscosity_term,
]
