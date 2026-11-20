"""Test temporal convergence using a manufactured solution.

We prescribe a spatially uniform pressure head h(x, y, t) = h0(t) on a
unit square. The divergence of the flux is zero, reducing the Richards
equation to dtheta/dt = S. For the exponential soil model, the required
source term is

    S(t) = (theta_s - theta_r) * alpha * exp(alpha * h0(t)) * h0'(t).

The boundary conditions and source term depend on the time Constant
passed to Irksome, so they are evaluated at the appropriate Runge-Kutta
stage times.

The test checks the temporal convergence orders of backward Euler (1),
implicit midpoint (2) and two-stage Gauss-Legendre (4).
"""

import numpy as np
import pytest

from gadopt import *
from firedrake import (
    UnitSquareMesh, FunctionSpace, Function, Constant,
    sin, cos, exp, pi, errornorm,
)


def _run_one(scheme, scheme_kwargs, dt_value, t_final, mesh_n=4):
    mesh = UnitSquareMesh(mesh_n, mesh_n, quadrilateral=True)
    mesh.cartesian = True
    V = FunctionSpace(mesh, "CG", 1)

    soil_curve = ExponentialCurve(
        theta_r=0.05, theta_s=0.40, Ks=1e-2, alpha=1.0, Ss=0.0,
    )

    # Use the same time Constant for the boundary conditions,
    # source term and Runge-Kutta stage times.
    mc = MeshConstant(mesh)
    t = mc.Constant(0.0)

    # h0(t): smooth, stays strictly negative on [0, T_period].
    T_period = 1.0
    omega = 2 * pi / T_period
    amplitude = 0.3
    baseline = -1.0

    h0 = baseline + amplitude * sin(omega * t)
    h0_dot = amplitude * omega * cos(omega * t)

    source = (
        (soil_curve.theta_s - soil_curve.theta_r)
        * soil_curve.alpha
        * exp(soil_curve.alpha * h0)
        * h0_dot
    )

    boundary_ids = get_boundary_ids(mesh)
    bcs = {
        boundary_ids.left: {'h': h0},
        boundary_ids.right: {'h': h0},
        boundary_ids.bottom: {'h': h0},
        boundary_ids.top: {'h': h0},
    }

    h = Function(V).interpolate(Constant(baseline))

    dt = Constant(dt_value)
    kwargs = {'t': t}
    if scheme_kwargs:
        kwargs.update(scheme_kwargs)

    solver = RichardsSolver(
        h, soil_curve,
        delta_t=dt,
        timestepper=scheme,
        bcs=bcs,
        source_term=source,
        timestepper_kwargs=kwargs,
        solver_parameters_extra={'snes_atol': 1e-13, 'snes_rtol': 1e-12},
    )

    nsteps = int(round(t_final / dt_value))
    time_var = 0.0
    for _ in range(nsteps):
        solver.solve()
        time_var += dt_value
        t.assign(time_var)

    h_exact_val = baseline + amplitude * float(np.sin(omega * time_var))
    h_exact = Function(V).interpolate(Constant(h_exact_val))
    return errornorm(h_exact, h, norm_type='L2')


@pytest.mark.parametrize("scheme,scheme_kwargs,expected_order,label", [
    (BackwardEuler, None, 1, "BackwardEuler"),
    (ImplicitMidpoint, None, 2, "ImplicitMidpoint"),
    (GaussLegendre, {"tableau_parameter": 2}, 4, "GaussLegendre(2)"),
])
def test_temporal_convergence(scheme, scheme_kwargs, expected_order, label):
    t_final = 0.1
    dt_values = [t_final / n for n in (2, 4, 8, 16)]

    errors = []
    for dt_value in dt_values:
        err = _run_one(scheme, scheme_kwargs, dt_value, t_final)
        errors.append(err)
        PETSc.Sys.Print(
            f"  {label}: dt={dt_value:.5f}  L2 error={err:.3e}"
        )

    rates = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(2))
        for i in range(len(errors) - 1)
    ]
    PETSc.Sys.Print(
        f"{label}: observed rates {rates}, expected order {expected_order}"
    )

    # Check the coarsest-step convergence rate to avoid solver-tolerance
    # effects at finer time steps.
    floor = 0.85 * expected_order
    assert rates[0] >= floor, (
        f"{label}: coarsest-pair rate {rates[0]:.2f} below expected "
        f"{expected_order} (floor {floor:.2f}); full rate list {rates}"
    )
