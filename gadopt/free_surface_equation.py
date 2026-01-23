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
    r"""Mass term for the free surface theta-scheme time discretisation.

    Note: This mass term does not use Irksome's `Dt` operator; `StokesSolver` manually
    implements the time discretisation: `(eta - eta_old) / dt`.

    Args:
        eq:
          G-ADOPT Equation.
        trial:
          Firedrake trial function.

    Returns:
        The UFL form associated with the mass term of the equation.

    """
    n_up = vertical_component(eq.n)

    return eq.buoyancy_scale * eq.test * trial * n_up * eq.ds(eq.boundary_id)


mass_term.required_attrs = {"buoyancy_scale", "boundary_id"}
mass_term.optional_attrs = set()
surface_velocity_term.required_attrs = {"u", "buoyancy_scale", "boundary_id"}
surface_velocity_term.optional_attrs = set()

free_surface_terms = [mass_term, surface_velocity_term]
