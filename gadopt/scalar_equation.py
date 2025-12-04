r"""Scalar terms (e.g. for temperature and salinity transport).

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
from .utility import is_continuous, normal_is_continuous
from irksome import Dt


def advection_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    r"""Scalar advection term (non-conservative): u \dot \div(q)."""
    advective_velocity_scaling = getattr(eq, "advective_velocity_scaling", 1)
    u = advective_velocity_scaling * eq.u

    if hasattr(eq, "su_nubar"):  # SU(PG) à la Donea & Huerta (2003)
        phi = eq.test + eq.su_nubar / (dot(u, u) + 1e-12) * dot(u, grad(eq.test))

        # The advection term is not integrated by parts so there are no boundary terms
        F = phi * dot(u, grad(trial)) * eq.dx
    else:
        F = -trial * div(eq.test * u) * eq.dx
        F += trial * dot(eq.n, u) * eq.test * eq.ds  # Boundary term in the weak form

        if not (is_continuous(eq.trial_space) and normal_is_continuous(eq.u)):
            # s = 0: u.n(-) < 0 => flow goes from '+' to '-' => '+' is upwind
            # s = 1: u.n(-) > 0 => flow goes from '-' to '+' => '-' is upwind
            s = 0.5 * (sign(dot(avg(u), eq.n("-"))) + 1.0)
            q_up = trial("-") * s + trial("+") * (1 - s)
            F += jump(eq.test * u, eq.n) * q_up * eq.dS

    for bc_id, bc in eq.bcs.items():
        if "q" in bc:
            # On incoming boundaries, where dot(u, n) < 0, replace trial with bc["q"].
            F += eq.test * min_value(dot(u, eq.n), 0) * (bc["q"] - trial) * eq.ds(bc_id)

    return -F


def diffusion_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    r"""Scalar diffusion term $-nabla * (kappa grad q)$.

    Using the symmetric interior penalty method, the weak form becomes

    $$
    {:( -int_Omega nabla * (kappa grad q) phi dx , = , int_Omega kappa (grad phi) * (grad q) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(phi bb n) * "avg"(kappa grad q) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(q bb n) * "avg"(kappa grad phi) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(kappa) "jump"(q bb n) * "jump"(phi bb n) dS )
    :}
    $$

    where σ is a penalty parameter.

    Epshteyn, Y., & Rivière, B. (2007).
    Estimation of penalty parameters for symmetric interior penalty Galerkin methods.
    Journal of Computational and Applied Mathematics, 206(2), 843-872.
    """
    kappa = eq.diffusivity
    dim = eq.mesh.geometric_dimension
    diff_tensor = kappa if len(kappa.ufl_shape) == 2 else kappa * Identity(dim)

    reference_for_diffusion = getattr(eq, "reference_for_diffusion", 0)
    q = trial + reference_for_diffusion

    F = inner(grad(eq.test), dot(diff_tensor, grad(q))) * eq.dx

    sigma = interior_penalty_factor(eq, shift=-1)
    if not is_continuous(eq.trial_space):
        sigma_int = sigma * avg(FacetArea(eq.mesh) / CellVolume(eq.mesh))
        F += (
            sigma_int
            * inner(jump(eq.test, eq.n), dot(avg(diff_tensor), jump(q, eq.n)))
            * eq.dS
        )
        F -= inner(avg(dot(diff_tensor, grad(eq.test))), jump(q, eq.n)) * eq.dS
        F -= inner(jump(eq.test, eq.n), avg(dot(diff_tensor, grad(q)))) * eq.dS

    for bc_id, bc in eq.bcs.items():
        if "q" in bc and "flux" in bc:
            raise ValueError("Cannot apply both `q` and `flux` on the same boundary.")

        if "q" in bc:
            jump_q = trial - bc["q"]
            sigma_ext = sigma * FacetArea(eq.mesh) / CellVolume(eq.mesh)
            # Terms below are similar to the above terms for the DG dS integrals.
            F += (
                2
                * sigma_ext
                * eq.test
                * inner(eq.n, dot(diff_tensor, eq.n))
                * jump_q
                * eq.ds(bc_id)
            )
            F -= inner(dot(diff_tensor, grad(eq.test)), eq.n) * jump_q * eq.ds(bc_id)
            F -= inner(eq.test * eq.n, dot(diff_tensor, grad(q))) * eq.ds(bc_id)
        elif "flux" in bc:
            # Here we need only the third term because we assume jump_q = 0
            # (q_ext = trial) and flux = kappa dq/dn = dot(n, dot(diff_tensor, grad(q)).
            F -= eq.test * bc["flux"] * eq.ds(bc_id)

    return -F


def source_term(eq: Equation, trial: Argument | ufl.indexed.Indexed | Function) -> Form:
    r"""Scalar source term `s_T`."""
    F = -dot(eq.test, eq.source) * eq.dx

    return -F


def sink_term(eq: Equation, trial: Argument | ufl.indexed.Indexed | Function) -> Form:
    r"""Scalar sink term `\alpha_T T`."""
    # Implement sink term implicitly at current time step.
    F = dot(eq.test, eq.sink_coeff * trial) * eq.dx

    return -F


def mass_term(eq: Equation, trial: Argument | ufl.indexed.Indexed | Function) -> Form:
    """UFL form for the mass term used in the time discretisation.

    Args:
        eq:
          G-ADOPT Equation.
        trial:
          Firedrake trial function.

    Returns:
        The UFL form associated with the mass term of the equation.

    """
    return dot(eq.test, Dt(trial)) * eq.dx


advection_term.required_attrs = {"u"}
advection_term.optional_attrs = {"advective_velocity_scaling", "su_nubar"}
diffusion_term.required_attrs = {"diffusivity"}
diffusion_term.optional_attrs = {"reference_for_diffusion", "interior_penalty"}
source_term.required_attrs = {"source"}
source_term.optional_attrs = set()
sink_term.required_attrs = {"sink_coeff"}
sink_term.optional_attrs = set()
