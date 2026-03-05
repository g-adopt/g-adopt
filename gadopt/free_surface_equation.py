r"""This module contains the free surface terms.

All terms implement the UFL residual as it would be on the LHS of the equation:

$$
dq / dt + F(q) = 0.
$$

"""

import firedrake as fd
from ufl.indexed import Indexed

from .equations import Equation
from .utility import vertical_component


def surface_velocity_term(
    eq: Equation, trial: fd.Argument | Indexed | fd.Function
) -> fd.Form:
    r"""Term for the normal component of motion at the free surface: $-u \dot n$."""
    return -eq.buoyancy_scale * eq.test * fd.dot(eq.u, eq.n) * eq.ds(eq.boundary_id)


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
    from irksome import Dt

    n_up = vertical_component(eq.n)
    use_irksome = getattr(eq, "use_irksome", False)

    if use_irksome:
        dt_trial = Dt(trial)
    else:
        if not hasattr(eq, "dt") or not hasattr(eq, "trial_old"):
            raise ValueError(
                "free_surface_equation.mass_term requires 'dt' and 'trial_old' "
                "equation attributes when use_irksome=False."
            )
        dt_trial = (trial - eq.trial_old) / eq.dt

    return eq.buoyancy_scale * eq.test * dt_trial * n_up * eq.ds(eq.boundary_id)


mass_term.required_attrs = {"buoyancy_scale", "boundary_id"}
mass_term.optional_attrs = {"dt", "trial_old", "use_irksome"}
surface_velocity_term.required_attrs = {"u", "buoyancy_scale", "boundary_id"}
surface_velocity_term.optional_attrs = set()

free_surface_terms = [mass_term, surface_velocity_term]
