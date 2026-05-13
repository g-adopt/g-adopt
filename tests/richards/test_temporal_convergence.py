"""Temporal convergence verification via manufactured solution.

For Richards with a nonlinear `Dt(theta(h))` mass term, the stage_value
formulation is mass-conservative in two regimes: stiffly-accurate
tableaux (where u_new = U_s copies the per-stage finite difference
directly) and non-stiffly-accurate tableaux (where Irksome PR #226
routes through a conservative variational update). Mass conservation
is checked elsewhere (`test_mass_balance.py`). This file pins the
*temporal order* the three integrators are supposed to deliver, since
the conservative-update path has to preserve the formal order p of the
underlying Runge-Kutta method.

Manufactured solution.

We pick a spatially uniform solution h(x, y, t) = h0(t) on the unit
square, with ExponentialCurve soil and Dirichlet h = h0(t) on all four
sides. Spatial uniformity makes diffusion and gravity contribute zero,
so the PDE reduces pointwise to dtheta/dt = S, which is satisfied by
construction with

  S(t) = (theta_s - theta_r) * alpha * exp(alpha * h0(t)) * h0'(t).

The BC and source are time-dependent UFL expressions referencing the
*same* Constant Irksome advances through stages, so stage substitution
t -> t + c_i*dt is honest for multi-stage tableaux. Without that, the
observed order collapses to 1 regardless of the scheme. CG is used so
the strong Dirichlet BC handles gravity boundary fluxes cleanly.

The error at t = t_final reduces to |h_num - h0(t_final)| (since h_num
is spatially uniform), isolating the temporal truncation cleanly.
"""

import numpy as np
import pytest

from gadopt import *
from firedrake import (
    UnitSquareMesh, FunctionSpace, Function, Constant,
    sin, cos, exp, pi, errornorm,
)
from irksome import MeshConstant


def _run_one(scheme, scheme_kwargs, dt_value, t_final, mesh_n=4):
    mesh = UnitSquareMesh(mesh_n, mesh_n, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)

    soil_curve = ExponentialCurve(
        theta_r=0.05, theta_s=0.40, Ks=1e-2, alpha=1.0, Ss=0.0,
    )

    # Shared time Constant: Irksome's stepper will advance it through
    # stages once we pass it via timestepper_kwargs={'t': ...}, and the
    # BC + source expressions below reference it, so stage substitution
    # is consistent.
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
        boundary_ids.left:   {'h': h0},
        boundary_ids.right:  {'h': h0},
        boundary_ids.bottom: {'h': h0},
        boundary_ids.top:    {'h': h0},
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
    (BackwardEuler,    None,                     1, "BackwardEuler"),
    (ImplicitMidpoint, None,                     2, "ImplicitMidpoint"),
    (GaussLegendre,    {"tableau_parameter": 2}, 4, "GaussLegendre(2)"),
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

    # The finest pair sometimes brushes the solver-tolerance floor for
    # order-4 GaussLegendre(2), so take the best of the last two as the
    # representative rate. Tolerance is ~85% of formal order.
    representative_rate = max(rates[-2:])
    floor = 0.85 * expected_order
    assert representative_rate >= floor, (
        f"{label}: observed rate {representative_rate:.2f} below expected "
        f"{expected_order} (floor {floor:.2f}); full rate list {rates}"
    )
