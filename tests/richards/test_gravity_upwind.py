"""Regression tests for the interior-facet gravity upwind direction.

The gravity (elevation-head) term ``richards_gravity_term`` upwinds ``K`` on
interior facets. Gravity drives drainage downward, so the upstream side of a
facet is the cell *above* it and the numerical flux must select the *upper*
trace of ``K`` (the negative part of ``q.n``). Selecting the lower trace (the
positive part) is the classic sign bug: it is masked by every smooth benchmark
because the two traces coincide wherever ``jump(K) -> 0`` (well-resolved DG, or
a continuous ``K``), and only bites where ``K`` jumps sharply across a facet.

Two guards:

* ``test_gravity_upwind_selects_upper_trace`` -- an assembly-level check on a
  two-cell mesh with a prescribed ``K`` jump. It exercises the *installed*
  ``richards_gravity_term`` and asserts the facet flux equals the upper (not the
  lower) ``K`` trace, for both an h-induced jump and a material ``Ks`` jump.

* ``test_gravity_upwind_transient_drainage`` -- a DG0 solve in a two-layer
  column with uniform initial head (so the capillary flux ``K grad(h)`` is zero
  and transport at t=0 is pure gravity). The correct (upper-trace) upwind pushes
  more water down across the conductivity interface than the wrong (lower-trace)
  choice; the shipped solver must match the correct one.
"""
import math

import numpy as np
from firedrake import (
    RectangleMesh, FunctionSpace, Function, SpatialCoordinate, TestFunction,
    Constant, conditional, assemble, inner, grad, dot, jump, dx,
)

from gadopt import ExponentialCurve, RichardsSolver, BackwardEuler
from gadopt import richards_equation as richards_eq
from gadopt.equations import Equation
from gadopt.utility import upward_normal, is_continuous

# The installed (production) gravity term, captured before any monkeypatching.
_INSTALLED_GRAVITY_TERM = richards_eq.richards_gravity_term


def _make_gravity_term(sign):
    """Build a gravity term whose facet flux uses q_n = 0.5*(q.n + sign*|q.n|).

    ``sign = -1`` is the correct (upper-trace) upwind; ``sign = +1`` is the
    lower-trace sign bug. Used to construct explicit reference behaviours in the
    transient test.
    """
    def term(eq, trial):
        K = eq.soil_curve.hydraulic_conductivity(trial)
        k = upward_normal(eq.mesh)
        F = inner(K * k, grad(eq.test)) * eq.dx
        if not is_continuous(eq.trial_space):
            q = K * k
            q_n = 0.5 * (dot(q, eq.n) + sign * abs(dot(q, eq.n)))
            F -= jump(eq.test) * (q_n('+') - q_n('-')) * eq.dS
        return F

    term.required_attrs = {'soil_curve'}
    term.optional_attrs = set()
    return term


def _installed_facet_trace(V, h, soil):
    """Magnitude of the facet gravity flux produced by the *installed* term.

    On a DG0 space grad(test) = 0, so the volume integral vanishes and the term
    reduces to its interior-facet contribution -jump(test)*(q_n+ - q_n-)*dS.
    Assembled against the DG0 test basis this gives +/-(q_n+ - q_n-)*|facet| per
    cell, whose magnitude equals the selected K trace (|e_z.n| = 1, |facet| = 1).
    """
    test = TestFunction(V)
    eq = Equation(test, V, residual_terms=[richards_eq.richards_gravity_term],
                  eq_attrs={'soil_curve': soil})
    facet_flux = assemble(richards_eq.richards_gravity_term(eq, h))
    return float(np.abs(facet_flux.dat.data_ro).max())


def test_gravity_upwind_selects_upper_trace():
    # Two DG0 cells stacked in y, one interior horizontal facet at y = 1.
    mesh = RectangleMesh(1, 2, 1.0, 2.0, quadrilateral=True)
    mesh.cartesian = True
    _, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DQ", 0)

    Ks, alpha = 1.0, 1.0

    # (a) h-induced jump: wet (h=0, K=Ks) above, dry (h=-10, K~0) below.
    soil = ExponentialCurve(theta_r=0.15, theta_s=0.45, Ks=Ks, Ss=0.0, alpha=alpha)
    h = Function(V).interpolate(conditional(y > 1.0, 0.0, -10.0))
    K_above = Ks * math.exp(alpha * 0.0)     # wet cell, on top
    K_below = Ks * math.exp(alpha * -10.0)   # dry cell, below
    sel = _installed_facet_trace(V, h, soil)
    assert math.isclose(sel, K_above, rel_tol=1e-9), (
        f"gravity facet flux selected {sel:.3e}; expected upper trace "
        f"K_above={K_above:.3e} (lower trace K_below={K_below:.3e} is the bug)"
    )

    # (b) material Ks jump: uniform h, discontinuous DG0 Ks (top 1, bottom 0.01).
    Ks_field = Function(V).interpolate(conditional(y > 1.0, 1.0, 0.01))
    soil_h = ExponentialCurve(theta_r=0.15, theta_s=0.45, Ks=Ks_field, Ss=0.0,
                              alpha=alpha)
    h_uniform = Function(V).interpolate(Constant(-0.3))
    kr = math.exp(alpha * -0.3)
    sel_h = _installed_facet_trace(V, h_uniform, soil_h)
    assert math.isclose(sel_h, 1.0 * kr, rel_tol=1e-9), (
        f"material-jump facet flux selected {sel_h:.3e}; expected upper trace "
        f"Ks_top*kr={1.0 * kr:.3e} (lower trace {0.01 * kr:.3e} is the bug)"
    )


def _water_below_interface(gravity_term, *, z_iface=1.0, nsteps=3):
    """Water that has moved below the conductivity interface after ``nsteps``.

    Two-layer DG0 column, hard 100x Ks jump at ``z_iface`` (aligned to a facet),
    uniform initial head so the capillary flux is zero at t=0 and the early
    transport across the interface is pure gravity. ``gravity_term`` is swapped
    into the equation module for the duration of the run.
    """
    H, nz, alpha = 2.0, 40, 4.0
    Ks_top, Ks_bot, h0 = 1.0, 0.01, -0.2

    saved = richards_eq.richards_gravity_term
    richards_eq.richards_gravity_term = gravity_term
    try:
        mesh = RectangleMesh(3, nz, 0.15, H, quadrilateral=True)
        mesh.cartesian = True
        _, z = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "DQ", 0)
        Ks = Function(V).interpolate(conditional(z > z_iface, Ks_top, Ks_bot))
        soil = ExponentialCurve(theta_r=0.06, theta_s=0.40, Ks=Ks, Ss=0.0,
                                alpha=alpha)
        h = Function(V).interpolate(Constant(h0))       # uniform -> pure gravity
        theta = soil.moisture_content(h)
        below = conditional(z < z_iface, 1.0, 0.0)
        m0 = assemble(theta * below * dx)

        solver = RichardsSolver(
            h, soil, delta_t=Constant(0.01), timestepper=BackwardEuler,
            bcs=None, solver_parameters="direct", quad_degree=4,
            interior_penalty=2.0,
        )
        for _ in range(nsteps):
            solver.solve()
        return assemble(theta * below * dx) - m0
    finally:
        richards_eq.richards_gravity_term = saved


def test_gravity_upwind_transient_drainage():
    below_buggy = _water_below_interface(_make_gravity_term(+1.0))   # lower trace
    below_fixed = _water_below_interface(_make_gravity_term(-1.0))   # upper trace
    below_installed = _water_below_interface(_INSTALLED_GRAVITY_TERM)

    # Both upwind choices move some water down, but the correct (upper-trace)
    # one moves distinctly more across the conductivity interface.
    assert below_buggy > 0.0 and below_fixed > 0.0
    assert below_fixed > 1.3 * below_buggy, (
        f"upper-trace upwind moved {below_fixed:.3e} below the interface, only "
        f"{below_fixed / below_buggy:.2f}x the lower-trace {below_buggy:.3e}; "
        f"expected a clear (>1.3x) gravity-driven separation"
    )

    # The shipped solver must behave as the correct (upper-trace) upwind.
    assert math.isclose(below_installed, below_fixed, rel_tol=1e-6), (
        f"installed gravity term moved {below_installed:.3e} below the "
        f"interface but the correct upper-trace upwind moves {below_fixed:.3e}; "
        f"the shipped term is not using the upper (upstream) K trace"
    )
