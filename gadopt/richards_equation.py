r"""Richards equation terms for unsaturated flow.

This module implements the individual terms for the Richards equation for
variational unsaturated flow. The Richards equation describes movement of
the wetting phase in a two-phase flow system, if we ignore the
compressibility of the dry phase.

All terms are considered as if they were on the left-hand side of the equation,
leading to the following UFL expression returned by the ``Equation``'s
``residual`` method:

$$
  \frac{\partial \theta}{\partial t} + F(h) = 0.
$$

The Richards equation in mixed form:

$$
  \frac{\partial \theta}{\partial t} + S_s S \frac{\partial h}{\partial t}
  - \nabla \cdot (K \nabla h) - \nabla \cdot (K \nabla z) = 0
$$

where:
- $h$ is the pressure head
- $S_s$ is the specific storage coefficient
- $S(h) = (\theta - \theta_r)/(\theta_s - \theta_r)$ is the effective saturation
- $C(h) = d\theta/dh$ is the specific moisture capacity
- $K(h)$ is the hydraulic conductivity
- $z$ is the vertical coordinate (gravity term)

The discretisation deliberately carries the mass term as the conservative
$\partial \theta / \partial t$ form rather than expanding it via the chain rule
into $C(h)\,\partial h/\partial t$. With the stage-value stepper this yields the
exact finite difference $(\theta_{new} - \theta_{old})/\Delta t$, which keeps mass
balance exact for DG. As a consequence $C(h) = d\theta/dh$ is the physical moisture
capacity listed above but never appears explicitly in the residual.

"""

import numbers

from firedrake import *
from irksome import Dt
from ufl.indexed import Indexed

from .equations import Equation
from .utility import is_continuous, upward_normal

__all__ = [
    "richards_mass_term",
    "richards_gravity_term",
]


def _is_scalar_zero(value) -> bool:
    """Return True only when ``value`` is provably the scalar zero.

    A plain Python number or a scalar ``Constant`` can be safely evaluated to a
    float and compared against zero. Anything else (most importantly a ``Function``
    or ``Function(R)``, which must never be float()'d) is treated as non-zero so
    that a spatially varying coefficient or an inversion control is never dropped.
    """
    if isinstance(value, numbers.Real):
        return float(value) == 0.0
    if isinstance(value, Constant):
        return float(value) == 0.0
    return False


def richards_mass_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    r"""Richards equation mass term for moisture storage.

    The mass term accounts for water storage changes and has two parts:

    $$
    \frac{\partial \theta}{\partial t} + S_s S \frac{\partial h}{\partial t}
    $$

    The moisture content change d(theta)/dt is expressed as Dt(theta(h)).
    When used with a StageValueTimeStepper, this is automatically discretised
    as a conservative finite difference (theta(h_new) - theta(h_old))/dt
    rather than the chain-rule expansion C(h)*dh/dt, preserving exact mass
    conservation for DG discretisations.

    The specific storage term Ss*S*dh/dt uses Dt(h) since it is already
    formulated directly in terms of h.

    Args:
        eq: G-ADOPT Equation instance
        trial: Firedrake trial function for pressure head

    Returns:
        UFL form for the mass term
    """
    soil_curve = eq.soil_curve

    # Moisture content change: d(theta)/dt
    F = inner(eq.test, Dt(soil_curve.moisture_content(trial))) * eq.dx

    # Specific storage: Ss * S * dh/dt (standard Dt, linear in h for given S).
    # We skip this term when Ss is identically zero, to avoid polluting the form
    # with a zero-coefficient TimeDerivative that can confuse the nonlinear solver.
    # That skip is only safe for values we can provably evaluate to the scalar
    # zero: a plain Python number or a scalar Constant. A Function or Function(R)
    # (e.g. a spatially varying storage, or an inversion control) must never be
    # float()'d, so we always include the term for those and never silently drop it.
    if not _is_scalar_zero(soil_curve.Ss):
        theta = soil_curve.moisture_content(trial)
        S = (theta - soil_curve.theta_r) / (soil_curve.theta_s - soil_curve.theta_r)
        F += inner(eq.test, soil_curve.Ss * S * Dt(trial)) * eq.dx

    return F


def richards_gravity_term(
    eq: Equation, trial: Argument | Indexed | Function
) -> Form:
    r"""Richards gravity term for gravity-driven flow with upwinding.

    The gravity term represents the gravity-driven water flux:

    $$
    \nabla \cdot (K \nabla z)
    $$

    where $z$ is the elevation (upward coordinate); $\nabla z$ is supplied by
    ``upward_normal``, so the term is correct both in a Cartesian box and on the
    sphere. (In a Cartesian box with $z$ the vertical coordinate this reduces to
    the familiar $\partial K/\partial z$.) For DG discretizations, we use upwinding
    on interior facets to ensure stability.

    The weak form is:

    $$
    \int_\Omega K \frac{\partial \phi}{\partial z} dx
    - \int_{\mathcal{I}} \text{jump}(\phi) \cdot (q_n^+ - q_n^-) dS
    $$

    where $q_n = 0.5(q \cdot n + |q \cdot n|)$ is the upwinded flux and
    $q = K \nabla z$ is the gravity-driven flux.

    Args:
        eq: G-ADOPT Equation instance
        trial: Firedrake trial function for pressure head

    Returns:
        UFL form for the gravity term
    """
    soil_curve = eq.soil_curve

    # Evaluate hydraulic conductivity at trial function
    K = soil_curve.hydraulic_conductivity(trial)

    # Upward unit vector grad(z) = e_z; upward_normal gives the last Cartesian
    # basis vector in a box and X/r on the sphere, so this also works on a sphere.
    k = upward_normal(eq.mesh)

    # Volume integral: \int K * \partial v/\partial z dx = \int K * \grad z \cdot \grad v dx
    F = inner(K * k, grad(eq.test)) * eq.dx

    # Upwinding on interior facets for DG
    if not is_continuous(eq.trial_space):
        # Gravity-driven flux
        q = K * k

        # n has no fixed direction: n('+') and n('-') each point outward from
        # their own cell. So q.n > 0 on a given side means q exits there, i.e.
        # that side is the upwind side, and only one of q_n('+'), q_n('-') is
        # nonzero. The minus sign on jump(eq.test) below is needed because jump
        # gives test_+ - test_-, whereas we want test_up - test_down; when '-'
        # is the upwind side that extra minus sign corrects the orientation.
        q_n = 0.5 * (dot(q, eq.n) + abs(dot(q, eq.n)))

        # Add upwind flux term
        F -= jump(eq.test) * (q_n('+') - q_n('-')) * eq.dS

    return F


# Set required and optional attributes for each term
richards_mass_term.required_attrs = {'soil_curve'}
richards_mass_term.optional_attrs = set()

richards_gravity_term.required_attrs = {'soil_curve'}
richards_gravity_term.optional_attrs = set()
