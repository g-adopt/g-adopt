r"""This module contains the free surface terms.

All terms implement the UFL residual as it would be on the RHS of the equation:

  dq/dt = \sum term

This sign-convention is for compatibility with Thetis's time integrators. In general,
however, we like to think about the terms as they are on the LHS. Therefore, in the
function below, we assemble in F as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end return "-F".
"""

import firedrake as fd

from .equations import Equation
from .utility import vertical_component


def free_surface_term(
    eq: Equation, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
) -> fd.Form:
    r"""Free Surface term: u \dot n"""
    F = -eq.test * fd.dot(eq.u, eq.n) * eq.ds(eq.free_surface_id)

    return -F


def mass_term(
    eq: Equation, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
) -> fd.Form:
    r"""Mass term \int test * trial * ds for the free surface time discretisation.

    Arguments:
        eq:
          G-ADOPT Equation.
        trial:
          Firedrake trial function.

    Returns:
        The UFL form associated with the mass term of the equation.

    """
    return fd.dot(eq.test, trial) * vertical_component(eq.n) * eq.ds(eq.free_surface_id)
