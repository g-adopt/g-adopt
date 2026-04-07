r"""Richards equation terms for unsaturated flow.

This module implements the individual terms for the Richards equation for
variational unsaturated flow. The Richards equation describes movement of
the wetting phase in a two-phase flow system, if we ignore the
compressibility of the dry phase.

All terms are considered as if they were on the right-hand side of the equation, leading
to the following UFL expression returned by the `residual` method:

$$
  (dh)/dt = sum "term.residual()"
$$

This sign convention ensures compatibility with G-ADOPT's time integrators. In general,
however, we like to think about the terms as they are on the left-hand side. Therefore,
in the residual methods below, we first sum the terms in the variable `F` as if they
were on the left-hand side, i.e.

$$
  (dh)/dt + F(h) = 0,
$$

and then return `-F`.

The Richards equation in mixed form:

$$
  (S_s S + C) \\frac{\\partial h}{\\partial t} + \\nabla \\cdot (K \\nabla h) + \\nabla \\cdot (K \\nabla z) = 0
$$

where:
- $h$ is the pressure head
- $S_s$ is the specific storage coefficient
- $S(h) = (\\theta - \\theta_r)/(\\theta_s - \\theta_r)$ is the effective saturation
- $C(h) = d\\theta/dh$ is the specific moisture capacity
- $K(h)$ is the hydraulic conductivity
- $z$ is the vertical coordinate (gravity term)

"""

from firedrake import *
from irksome import Dt, ConservativeDt
from ufl.indexed import Indexed

from .equations import Equation, interior_penalty_factor
from .utility import is_continuous

__all__ = [
    "richards_mass_term",
    "richards_diffusion_term",
    "richards_source_term",
    "richards_gravity_term",
]


def richards_mass_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    r"""Richards equation mass term with nonlinear capacity.

    The mass term accounts for water storage changes:

    $$
    (S_s S + C) \\frac{\\partial h}{\\partial t}
    $$

    where:
    - $S_s$ is the specific storage coefficient (from soil curve)
    - $S(h) = (\\theta - \\theta_r)/(\\theta_s - \\theta_r)$ is effective saturation
    - $C(h) = d\\theta/dh$ is the specific moisture capacity
    - $h$ is the pressure head

    Args:
        eq: G-ADOPT Equation instance
        trial: Firedrake trial function for pressure head

    Returns:
        UFL form for the mass term
    """
    soil_curve = eq.soil_curve

    # Evaluate nonlinear coefficients
    theta = soil_curve.moisture_content(trial)
    C = soil_curve.water_retention(trial)

    # Effective saturation
    S = (theta - soil_curve.theta_r) / (soil_curve.theta_s - soil_curve.theta_r)

    # Mass coefficient
    mass_coeff = soil_curve.Ss * S

    F = inner(eq.test, ConservativeDt(theta)) * eq.dx + inner(eq.test, mass_coeff * Dt(trial)) * eq.dx

    return F



def richards_diffusion_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    r"""Richards diffusion term for pressure-driven flow using SIPG.

    The diffusion term represents pressure-driven water flow:

    $$
    \\nabla \\cdot (K \\nabla h)
    $$

    Using the symmetric interior penalty method (SIPG), the weak form becomes:

    $$
    \\begin{aligned}
    &\\int_\\Omega K (\\nabla \\phi) \\cdot (\\nabla h) dx \\\\
    &- \\int_{\\mathcal{I}} \\text{jump}(\\phi \\mathbf{n}) \\cdot \\text{avg}(K \\nabla h) dS \\\\
    &- \\int_{\\mathcal{I}} \\text{jump}(h \\mathbf{n}) \\cdot \\text{avg}(K \\nabla \\phi) dS \\\\
    &+ \\int_{\\mathcal{I}} \\sigma \\text{avg}(K) \\text{jump}(h \\mathbf{n}) \\cdot \\text{jump}(\\phi \\mathbf{n}) dS
    \\end{aligned}
    $$

    where $\\sigma$ is the penalty parameter and $K(h)$ is the hydraulic conductivity.

    Args:
        eq: G-ADOPT Equation instance
        trial: Firedrake trial function for pressure head

    Returns:
        UFL form for the diffusion term
    """
    v = eq.test
    grad_v = grad(v)
    bcs = eq.bcs

    relative_conductivity = eq.soil_curve.relative_conductivity
    K = relative_conductivity(trial)
    K_old = relative_conductivity(eq.solution_old)

    # Volume integral
    F = inner(grad_v, K * grad(trial)) * eq.dx

    # SIPG
    sigma = interior_penalty_factor(eq, shift=0)
    sigma_int = sigma * avg(FacetArea(eq.mesh) / CellVolume(eq.mesh))

    jump_v = jump(v, eq.n)
    jump_h = jump(trial, eq.n)
    avg_K  = avg(K_old)

    F += sigma_int * inner(jump_v, avg_K * jump_h) * eq.dS
    F -= inner(avg(K_old * grad_v), jump_h) * eq.dS
    F -= inner(jump_v, avg(K_old * grad(trial))) * eq.dS

    # Impose bcs within the weak formulation
    for bc_idx, bc_info in bcs.items():
        boundaryInfo = bc_info
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]
        if boundaryType == 'h':
            sigma_ext = sigma * FacetArea(eq.mesh) / CellVolume(eq.mesh)
            diff = trial - boundaryValue

            F += 2 * sigma_ext * v * K * diff * eq.ds(bc_idx)
            F -= inner(K * grad_v, eq.n) * diff * eq.ds(bc_idx)
            F -= inner(v * eq.n, K * grad(trial)) * eq.ds(bc_idx)
        elif boundaryType == 'flux':
            F -= boundaryValue * eq.test * eq.ds(bc_idx)
        else:
            raise ValueError("Unknown boundary type, must be 'h' or 'flux'")

    return F


def richards_source_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:

    F = -inner(eq.source_term, eq.test) * eq.dx

    return F


def richards_gravity_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    r"""Richards gravity term for gravity-driven flow with upwinding.

    The gravity term represents gravity-driven water flow:

    $$
    \\nabla \\cdot (K \\nabla z) = \\frac{\\partial K}{\\partial z}
    $$

    where $z$ is the vertical coordinate. For DG discretizations, we use upwinding
    on interior facets to ensure stability.

    The weak form is:

    $$
    \\int_\\Omega K \\frac{\\partial \\phi}{\\partial z} dx
    - \\int_{\\mathcal{I}} \\text{jump}(\\phi) \\cdot (q_n^+ - q_n^-) dS
    $$

    where $q_n = 0.5(q \\cdot n + |q \\cdot n|)$ is the upwinded flux and
    $q = K \\nabla z$ is the gravity-driven flux.

    Args:
        eq: G-ADOPT Equation instance
        trial: Firedrake trial function for pressure head

    Returns:
        UFL form for the gravity term
    """
    v = eq.test
    x = SpatialCoordinate(eq.mesh)

    K = eq.soil_curve.relative_conductivity(trial)
    e_z = grad(x[eq.mesh.topological_dimension() - 1])
    q = K * e_z

    # Conservative split: - ∫Ω q · ∇v
    F = -inner(q, grad(v)) * eq.dx

    # Interior upwind flux:  ∫F (q̂·n) [v]
    qn = 0.5 * (dot(q, eq.n) + abs(dot(q, eq.n)))  # one-sided “+”
    F += jump(v) * (qn('+') - qn('-')) * eq.dS

    return -F


# Set required and optional attributes for each term
richards_mass_term.required_attrs = {'soil_curve'}
richards_mass_term.optional_attrs = set()

richards_diffusion_term.required_attrs = {'soil_curve'}
richards_diffusion_term.optional_attrs = {'interior_penalty'}

richards_source_term.required_attrs = {'source_term'}
richards_source_term.optional_attrs = set()

richards_gravity_term.required_attrs = {'soil_curve'}
richards_gravity_term.optional_attrs = set()