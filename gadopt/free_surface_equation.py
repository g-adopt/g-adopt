r"""This module contains the free surface terms.

All terms implement the UFL residual as it would be on the LHS of the equation:

$$
dq / dt + F(q) = 0.
$$

By default, the free surface is advanced with the linearised kinematic boundary
condition $d\eta / dt = u \dot n$, where $n$ denotes the reference (undeformed) boundary
normal. Two opt-in extensions are supported, enabled through the free-surface boundary
condition parameters:

- `exact_normal`: Replace the reference normal with the exact normal of the deformed
  surface, obtained by displacing the reference boundary by the free-surface deflection
  $\eta$ along the upward direction, consistent with the linearised terms. This upgrades
  the kinematic condition to its full form
  $d\eta/dt + u_t \dot \nabla_t \eta = u \dot n$, where $\nabla_t \eta$ is the
  tangential gradient of the deflection, so that tangential (horizontal) flow transports
  topography. The construction only involves the reference normal and tangential
  gradients, and is thus valid for Cartesian, cylindrical, and spherical domains, in 2-D
  and 3-D alike. It is required for open or periodic domains, where topography must be
  allowed to cross lateral boundaries, and redistributes topography in closed domains.

- `volume_multiplier`: Add a uniform correction to the surface's normal speed, i.e.
  $d\eta/dt \mathrel{-}= vm$ for a multiplier $vm$. For a closed domain, evaluating
  $vm = \int (u \dot n - u \dot \nabla_t \eta) ds / \int ds$ with the previous step's
  fields renders the mean deflection stationary, removing the uniform vertical drift
  that is dynamically inert (only deflection gradients load the interior) while
  preserving its physically relevant gradients. For open or periodic domains, the
  boundary flux of tangentially transported topography must additionally be accounted
  for when choosing the multiplier.

Note that the normal stress applied at the free surface remains evaluated with the
reference normal in either case; accounting for the tilt of the surface there is a
second-order effect in the deflection gradient.

With `exact_normal`, the kinematic condition becomes nonlinear in the deflection, and
the coupled system loses the block symmetry of the linearised formulation of Kramer
et al. (2012): the momentum equation couples to the deflection through the reference
normal, whereas the kinematic condition couples to the velocity through the
deflection-dependent exact normal. The resulting asymmetry is of the same order as
the tangential-gradient terms and vanishes as the surface gradient tends to zero.
The system must then be solved with Newton iteration rather than a single linear
step; `StokesSolver` selects Newton parameters automatically in this case.
"""

import firedrake as fd
from irksome import Dt
from ufl.core.operator import Operator
from ufl.indexed import Indexed

from .equations import Equation
from .utility import free_surface_normal, vertical_component


def _midpoint_deflection(eq: Equation, trial: fd.Argument | Indexed) -> Operator:
    if not hasattr(eq, "trial_old"):
        raise ValueError(
            "The exact free-surface normal requires the 'trial_old' equation attribute."
        )

    return 0.5 * (trial + eq.trial_old)


def surface_velocity_term(
    eq: Equation, trial: fd.Argument | Indexed | fd.Function
) -> fd.Form:
    r"""Normal kinematics, with an optional global volume multiplier."""
    if getattr(eq, "exact_normal", False):
        n = free_surface_normal(_midpoint_deflection(eq, trial), eq.n)
    else:
        n = eq.n

    residual = -fd.dot(eq.u, n)
    if hasattr(eq, "volume_multiplier"):
        # The multiplier is a uniform correction to the surface's normal speed.
        residual += eq.volume_multiplier * vertical_component(n)

    return eq.buoyancy_scale * eq.test * residual * eq.ds(eq.boundary_id)


def mass_term(eq: Equation, trial: fd.Argument | Indexed | fd.Function) -> fd.Form:
    r"""Mass term for the free surface time discretisation.

    Args:
      eq:
        G-ADOPT Equation.
      trial:
        Firedrake trial function.

    Returns:
      The UFL form associated with the mass term of the equation.

    """
    if getattr(eq, "exact_normal", False):
        n = free_surface_normal(_midpoint_deflection(eq, trial), eq.n)
    else:
        n = eq.n
    n_up = vertical_component(n)

    if getattr(eq, "use_irksome", False):
        dt_trial = Dt(trial)
    else:
        if not hasattr(eq, "dt") or not hasattr(eq, "trial_old"):
            raise ValueError(
                "free_surface_equation.mass_term requires 'dt' and 'trial_old' equation"
                " attributes when use_irksome=False."
            )
        dt_trial = (trial - eq.trial_old) / eq.dt

    return eq.buoyancy_scale * eq.test * dt_trial * n_up * eq.ds(eq.boundary_id)


# Options that can be passed alongside the other free-surface boundary condition
# parameters to govern the free-surface kinematics
free_surface_option_keys = ("exact_normal", "volume_multiplier")

mass_term.required_attrs = {"buoyancy_scale", "boundary_id"}
mass_term.optional_attrs = {"dt", "trial_old", "use_irksome", "exact_normal"}
surface_velocity_term.required_attrs = {"u", "buoyancy_scale", "boundary_id"}
surface_velocity_term.optional_attrs = {"exact_normal", "volume_multiplier"}

free_surface_terms = [mass_term, surface_velocity_term]
