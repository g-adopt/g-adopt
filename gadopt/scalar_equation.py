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

from firedrake import *

from .equations import Equation, interior_penalty_factor
from .utility import is_continuous, normal_is_continuous


def scalar_advection_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    r"""Scalar advection term (non-conservative): u \dot \div(q)."""
    if hasattr(eq, "advective_velocity_scaling"):
        u = eq.advective_velocity_scaling * eq.u
    else:
        u = eq.u

    if hasattr(eq, "su_nubar"):  # SU(PG) à la Donea & Huerta (2003)
        phi = eq.test + eq.su_nubar / (dot(u, u) + 1e-12) * dot(u, grad(eq.test))

        # The advection term is not integrated by parts so there are no boundary terms
        F = phi * dot(u, grad(trial)) * eq.dx
    else:
        F = -trial * div(eq.test * u) * eq.dx
        F += trial * dot(eq.n, u) * eq.test * eq.ds  # Boundary term in the weak form

        if not all([is_continuous(eq.trial_space), normal_is_continuous(eq.u)]):
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


def scalar_diffusion_term(
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
    dim = eq.mesh.geometric_dimension()
    diff_tensor = kappa if len(kappa.ufl_shape) == 2 else kappa * Identity(dim)

    if hasattr(eq, "reference_for_diffusion"):
        q = trial + eq.reference_for_diffusion
    else:
        q = trial

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


def scalar_source_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    r"""Scalar source term `s_T`."""
    source = eq.source if hasattr(eq, "source") else 0
    F = -dot(eq.test, source) * eq.dx

    return -F


def scalar_absorption_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
) -> Form:
    r"""Scalar absorption term `\alpha_T T`."""
    sink = eq.sink if hasattr(eq, "sink") else 0
    # Implement absorption term implicitly at current time step.
    F = dot(eq.test, sink * trial) * eq.dx

    return -F


def mass_term(eq: Equation, trial: Argument | ufl.indexed.Indexed | Function) -> Form:
    """UFL form for the mass term used in the time discretisation.

    Arguments:
        eq:
          G-ADOPT Equation.
        trial:
          Firedrake trial function.

    Returns:
        The UFL form associated with the mass term of the equation.

    """
    return dot(eq.test, trial) * eq.dx
