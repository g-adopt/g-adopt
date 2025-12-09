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
from irksome import Dt

from .equations import Equation, interior_penalty_factor
from .utility import is_continuous

__all__ = [
    "richards_mass_term",
    "richards_diffusion_term",
    "richards_gravity_term",
]


def richards_mass_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
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
    h = trial

    # Evaluate nonlinear coefficients
    theta = soil_curve.moisture_content(h)
    C = soil_curve.water_retention(h)

    # Effective saturation
    S = (theta - soil_curve.theta_r) / (soil_curve.theta_s - soil_curve.theta_r)

    # Mass coefficient
    mass_coeff = soil_curve.Ss * S + C

    #return inner(eq.test, mass_coeff * Dt(h)) * eq.dx
    return inner(eq.test, 1 * Dt(theta)) * eq.dx


def richards_diffusion_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
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
    soil_curve = eq.soil_curve
    h = trial

    # Evaluate hydraulic conductivity at trial function
    K = soil_curve.relative_permeability(h)

    # Volume integral
    F = inner(grad(eq.test), K * grad(h)) * eq.dx

    # Interior penalty for DG
    sigma = interior_penalty_factor(eq, shift=0)
    if not is_continuous(eq.trial_space):
        sigma_int = sigma * avg(FacetArea(eq.mesh) / CellVolume(eq.mesh))

        # SIPG terms on interior facets
        F += (
            sigma_int
            * inner(jump(eq.test, eq.n), avg(K) * jump(h, eq.n))
            * eq.dS
        )
        F -= inner(avg(K * grad(eq.test)), jump(h, eq.n)) * eq.dS
        F -= inner(jump(eq.test, eq.n), avg(K * grad(h))) * eq.dS

    # Boundary conditions
    for bc_id, bc in eq.bcs.items():
        if 'h' in bc and 'flux' in bc:
            raise ValueError("Cannot apply both 'h' and 'flux' on the same boundary.")

        if 'h' in bc:
            # Dirichlet BC on pressure head
            jump_h = trial - bc['h']
            sigma_ext = sigma * FacetArea(eq.mesh) / CellVolume(eq.mesh)

            # SIPG boundary terms (similar to interior)
            F += (2 * sigma_ext * eq.test * K * jump_h * eq.ds(bc_id))
            F -= inner(K * grad(eq.test), eq.n) * jump_h * eq.ds(bc_id)
            F -= eq.test * inner(K * grad(h), eq.n) * eq.ds(bc_id)

        elif 'flux' in bc:
            # Neumann BC on flux
            # flux = -K * dh/dn, so we add the flux term
            F -= eq.test * bc['flux'] * eq.ds(bc_id)

    return -F


def richards_gravity_term(
    eq: Equation, trial: Argument | ufl.indexed.Indexed | Function
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
    soil_curve = eq.soil_curve

    # Evaluate hydraulic conductivity at trial function
    K = soil_curve.relative_permeability(trial)

    # Get mesh dimension and spatial coordinates
    dim = eq.mesh.geometric_dimension()
    x = SpatialCoordinate(eq.mesh)

    # Gradient in vertical direction (last coordinate)
    # grad(z) = e_z, where e_z is the unit vector in z-direction
    n_down = grad(x[dim - 1])

    # Volume integral: \int K * \partial v/\partial z dx = \int K * \grad z \cdot \grad v dx
    F = inner(K * n_down, grad(eq.test)) * eq.dx

    # Upwinding on interior facets for DG
    if not is_continuous(eq.trial_space):
        # Gravity-driven flux
        q = K * n_down

        # Upwinded flux: take the value from the upwind side
        # q\cdot n > 0 means flow from '-' to '+', so use '-' side (upwind)
        # q\cdot n < 0 means flow from '+' to '-', so use '+' side (upwind)
        q_n = 0.5 * (dot(q, eq.n) + abs(dot(q, eq.n)))

        # Add upwind flux term
        F -= jump(eq.test) * (q_n('+') - q_n('-')) * eq.dS

    return -F


# Set required and optional attributes for each term
richards_mass_term.required_attrs = {'soil_curve'}
richards_mass_term.optional_attrs = set()

richards_diffusion_term.required_attrs = {'soil_curve'}
richards_diffusion_term.optional_attrs = {'interior_penalty'}

richards_gravity_term.required_attrs = {'soil_curve'}
richards_gravity_term.optional_attrs = set()
