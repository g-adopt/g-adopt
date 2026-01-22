"""Test that adaptive timestepping correctly syncs dt back to the user.

G-ADOPT philosophy: solver.solve() returns dt, user does delta_t.assign(dt),
next step uses that value. The returned dt should be the NEW recommended dt,
not the OLD dt that was just used.

Bug: The returned dt was always the old used value, causing the adaptive
recommendation to be ignored when users follow the standard pattern.
"""

from gadopt import *
from gadopt.time_stepper import IrksomeIntegrator
from gadopt.equations import Equation
from gadopt.scalar_equation import diffusion_term, mass_term
from irksome import RadauIIA as IrksomeRadauIIA


def test_adaptive_dt_recommendation_changes():
    """Test that adaptive timestepping returns changing dt recommendations.

    G-ADOPT usage pattern:
        delta_t = Constant(initial)
        solver = Solver(..., delta_t, ...)
        for step in range(n):
            error, dt = solver.solve()
            time += dt  # or use dt for time tracking
            delta_t.assign(dt)  # User updates delta_t for next step

    The returned dt should be the NEW recommended dt. Over several steps with
    a diffusing solution, the adaptive algorithm should recommend different
    dt values. If dt never changes, the recommendation is being ignored.
    """
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    x, y = SpatialCoordinate(mesh)
    u.interpolate(exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))

    test = TestFunction(V)
    eq_attrs = {"diffusivity": Constant(1.0)}
    equation = Equation(
        test, V, residual_terms=[diffusion_term], mass_term=mass_term, eq_attrs=eq_attrs
    )

    delta_t = Constant(0.0001)

    integrator = IrksomeIntegrator(
        equation,
        u,
        delta_t,
        IrksomeRadauIIA(2),
        adaptive_parameters={"tol": 1e-2, "dtmin": 1e-10, "dtmax": 1.0},
    )

    # Collect returned dt values over several steps (following G-ADOPT pattern)
    dt_values = []
    for _ in range(5):
        _, dt = integrator.advance()
        dt_values.append(dt)
        delta_t.assign(dt)  # User updates delta_t for next step

    # The adaptive algorithm should recommend different dt values
    # With the bug: all values are identical (the initial dt)
    all_same = all(abs(dt_values[i] - dt_values[0]) < 1e-14 for i in range(len(dt_values)))

    assert not all_same, (
        f"Adaptive dt should change between steps, but got constant values: {dt_values}. "
        "This suggests the returned dt is the OLD used value, not the NEW recommendation."
    )
