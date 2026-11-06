r"""This module contains the free surface terms.

All terms implement the UFL residual as it would be on the LHS of the equation:

$$
dq / dt + F(q) = 0.
$$

By default, the free surface is advanced with the linearised kinematic boundary
condition $d\eta / dt = u \dot n$, where $n$ denotes the reference (undeformed) boundary
normal. The following opt-in extensions are supported, enabled through the free-surface
boundary condition parameters:

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
  The deflection entering the exact normal is the theta-midpoint of the ongoing and old
  deflections in the default (manual) time discretisation, or the stage deflection that
  Irksome's machinery substitutes for the ongoing solution under `use_irksome`.

- `volume_multiplier`: Add a uniform correction to the surface's normal speed, i.e.
  $d\eta/dt \mathrel{-}= vm$ for a multiplier $vm$. For a closed domain, evaluating
  $vm = \int (u \dot n - u \dot \nabla_t \eta) ds / \int ds$ with the previous step's
  fields renders the mean deflection stationary, removing the uniform vertical drift
  that is dynamically inert (only deflection gradients load the interior) while
  preserving its physically relevant gradients. For open or periodic domains, the
  boundary flux of tangentially transported topography must additionally be accounted
  for when choosing the multiplier.

- `auto_volume_multiplier`: Consumed by `StokesSolver` rather than by the equation:
  request the `volume_multiplier` to be auto-updated before each solve, with the
  closed-domain value above computed from the previous state's fields. A multiplier
  `Constant` is created if none is supplied. For open or periodic domains, pass
  `volume_multiplier` alone and update it manually.

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

from collections.abc import Sequence

import firedrake as fd
from irksome import Dt
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.indexed import Indexed

from .equations import Equation
from .utility import CombinedSurfaceMeasure, vertical_component


def free_surface_normal(deflection: Expr, reference_normal: Expr) -> Operator:
    r"""Unit normal of the deformed free surface.

    The deformed surface is the reference boundary displaced by `deflection` along the
    upwards direction. Its (unnormalised) normal reads $n - \nabla_t \eta$, where $n$ is
    the reference (undeformed) boundary normal and $\nabla_t \eta$ the tangential
    gradient of the deflection along the reference normal. The construction is agnostic
    to the domain geometry (Cartesian, cylindrical, or spherical; 2-D or 3-D) and
    handles both top and bottom free surfaces.

    Args:
      deflection:
        UFL expression for the free-surface deflection.
      reference_normal:
        UFL expression for the reference (undeformed) boundary normal.

    Returns:
      UFL expression for the unit normal of the deformed free surface.

    """
    # The deflection is measured along the upward direction; converting it to a
    # deflection along the reference normal flips its sign at bottom free surfaces
    grad_deflection = fd.grad(deflection * vertical_component(reference_normal))
    tangential_gradient = (
        grad_deflection - fd.dot(grad_deflection, reference_normal) * reference_normal
    )
    normal = reference_normal - tangential_gradient

    return normal / fd.sqrt(fd.dot(normal, normal))


def free_surface_volume_multiplier(
    u: fd.Function,
    deflection: fd.Function,
    boundary_id: Sequence[int | str] | int | str,
    quad_degree: int = 4,
) -> float:
    r"""Mean tendency of the free-surface deflection over a boundary.

    Returns the boundary average of the local height tendency $u \cdot n / n_{up}$,
    where $n$ is the exact normal of the deformed free surface (see
    `free_surface_normal`) and $n_{up}$ its upward component. For a closed domain,
    assigning the returned value to the `volume_multiplier` free-surface boundary
    condition parameter before a solve, using the previous step's fields, renders the
    mean deflection stationary, removing the uniform vertical drift associated with the
    tangential transport of topography. For open or periodic domains, the boundary flux
    of tangentially transported topography must additionally be accounted for.

    Args:
      u:
        Firedrake function for the velocity.
      deflection:
        Firedrake function for the free-surface deflection.
      boundary_id:
        Boundary identifier of the free surface.
      quad_degree:
        Degree of polynomial quadrature approximation. Only used on extruded meshes,
        whose surface measure requires an explicit quadrature degree.

    Returns:
      The boundary-averaged height tendency.

    """
    mesh = fd.ufl_expr.extract_unique_domain(u)
    if mesh.extruded:
        ds_surf = CombinedSurfaceMeasure(mesh, quad_degree)(boundary_id)
    else:
        ds_surf = fd.ds(boundary_id, domain=mesh)

    n = free_surface_normal(deflection, fd.FacetNormal(mesh))

    return fd.assemble(fd.dot(u, n) / vertical_component(n) * ds_surf) / fd.assemble(
        fd.Constant(1.0) * ds_surf
    )


def _midpoint_deflection(
    eq: Equation, trial: fd.Argument | Indexed | fd.Function
) -> Expr:
    if getattr(eq, "use_irksome", False):
        # Irksome's machinery substitutes every occurrence of the ongoing solution
        # in the residual with the stage deflection at t_n + c_i dt, which is already
        # the temporally consistent point for the exact normal. No theta-midpoint
        # with a manually managed old deflection is needed; `trial` is returned alone.
        return trial

    if not hasattr(eq, "trial_old"):
        raise ValueError(
            "The exact free-surface normal requires the 'trial_old' equation "
            "attribute when use_irksome=False."
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
