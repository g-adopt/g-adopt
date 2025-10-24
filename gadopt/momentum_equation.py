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
from firedrake.mesh import ExtrudedMeshTopology

from .equations import Equation, interior_penalty_factor
from .utility import (
    is_continuous,
    normal_is_continuous,
    tensor_jump,
    vertical_component,
)


def viscosity_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
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
    dim = eq.mesh.geometric_dimension()
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

            if hasattr(eq.approximation, 'bulk_modulus'):
                trial_tensor_jump = identity * (dot(eq.n, trial) - bc["un"])
                trial_tensor_jump += transpose(trial_tensor_jump)
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


def divergence_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    assert normal_is_continuous(eq.u)

    rho = eq.rho_continuity
    F = -dot(eq.test, div(rho * eq.u)) * eq.dx

    # Add boundary integral for bcs that specify the normal component of u.
    for bc_id, bc in eq.bcs.items():
        if "u" in bc:
            F -= eq.test * rho * dot(eq.n, bc["u"] - eq.u) * eq.ds(bc_id)
        elif "un" in bc:
            F -= eq.test * rho * (bc["un"] - dot(eq.n, eq.u)) * eq.ds(bc_id)

    return -F


def momentum_source_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    F = -dot(eq.test, eq.source) * eq.dx

    return -F


def advection_hydrostatic_prestress_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    # Advection of background hydrostatic pressure used in linearised
    # GIA simulations. This method implements the body integral and
    # jump terms (if density space is discontinuous) after integration
    # by parts. The boundary integral on the Earth's surface is
    # applied through the `normal_stress` boundary condition tag of
    # the `viscosity_term`. This is turned on by specifiying a
    # `free_surface` in the boundary conditions of the solver.
    B_mu = eq.approximation.B_mu
    rho0 = eq.approximation.density
    g = eq.approximation.g
    u_r = vertical_component(trial)

    # Only include jump term for discontinuous density spaces
    if is_continuous(rho0.function_space()):
        F = 0
    else:
        # change surface measure for extruded mesh.
        # assumes mesh is aligned in the vertical so that jump
        # only occurs across horizontal layers
        if isinstance(eq.mesh, ExtrudedMeshTopology):
            dS = dS_h
        else:
            dS = eq.dS
        F = B_mu("+") * jump(rho0) * u_r("+") * g("+") * dot(eq.test("+"), eq.n("+")) * dS
    # Include body integral after i.b.p of hydrostatic prestress advection term
    # Analytical solution from Cathles 2024 Eq 2b doesn't include prestress
    # so we neglect this term but keep the free surface term that accounts for
    # viscous feedback at isostatic equibrium
    F -= div(eq.test) * eq.approximation.compressible_adv_hyd_pre(u_r) * eq.dx

    return -F


viscosity_term.required_attrs = {"stress"}
viscosity_term.optional_attrs = {"interior_penalty"}
pressure_gradient_term.required_attrs = {"p"}
pressure_gradient_term.optional_attrs = set()
divergence_term.required_attrs = {"u", "rho_continuity"}
divergence_term.optional_attrs = set()
momentum_source_term.required_attrs = {"source"}
momentum_source_term.optional_attrs = set()
advection_hydrostatic_prestress_term.required_attrs = set()
advection_hydrostatic_prestress_term.optional_attrs = set()

momentum_terms = [momentum_source_term, pressure_gradient_term, viscosity_term]
mass_terms = divergence_term
stokes_terms = [momentum_terms, mass_terms]

compressible_viscoelastic_terms = [
    advection_hydrostatic_prestress_term,
    momentum_source_term,
    viscosity_term,
]
