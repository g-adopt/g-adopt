"""Mass conservation verification for the Richards equation solver.

The stage-value formulation discretises the nonlinear mass term Dt(theta(h))
as the exact per-stage finite difference, so theta(h) should be conserved to
solver tolerance. We check that along two axes:

(1) Function space, at fixed BackwardEuler: DQ0/DQ1/DQ2 sit at solver
    tolerance, while CG1/CG2 pick up an O(1e-4) consistency error from the
    lack of element-wise conservation.

(2) Time-stepping scheme, at fixed DQ1: BackwardEuler, ImplicitMidpoint and
    GaussLegendre(2) should all conserve mass to solver tolerance.

The domain is a unit square with no-flux boundaries on left, right and bottom
and a prescribed inflow flux of 1e-6 m/s on top. The soil follows a Haverkamp
model with moderate nonlinearity, and specific storage is zero so theta(h) is
the only conserved quantity.
"""

import pytest
from gadopt import *


def compute_mass_balance(grid_points, time_step, t_final,
                         polynomial_degree, time_integrator,
                         function_space, timestepper_kwargs=None,
                         scheme_label=None):
    mesh = UnitSquareMesh(grid_points, grid_points, quadrilateral=True)
    mesh.cartesian = True
    V = FunctionSpace(mesh, function_space, polynomial_degree)

    soil_curve = HaverkampCurve(
        theta_r=0.05,
        theta_s=0.40,
        Ks=1e-5,
        alpha=0.5,
        beta=1.3,
        A=0.01,
        gamma=1.5,
        Ss=0.0,
    )

    boundary_ids = get_boundary_ids(mesh)
    inflow_rate = 1e-06
    richards_bcs = {
        boundary_ids.left: {'flux': 0},
        boundary_ids.right: {'flux': 0},
        boundary_ids.bottom: {'flux': 0},
        boundary_ids.top: {'flux': inflow_rate},
    }

    h = Function(V).assign(-1.0)
    moisture_content = soil_curve.moisture_content
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

    dt = Constant(time_step)
    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=time_integrator,
        solver_parameters_extra={'snes_atol': 1e-15},
        bcs=richards_bcs,
        timestepper_kwargs=timestepper_kwargs,
    )

    initial_mass = assemble(theta * dx)
    previous_mass = initial_mass
    mass_error = 0.0

    time_var = 0.0
    while time_var < t_final:
        time_var += float(dt)
        richards_solver.solve()

        theta.interpolate(moisture_content(h))
        # Mass entering over the step is the flux integrated over the inflow
        # boundary, times dt. The top edge has measure 1 on a unit square, so
        # this equals dt * inflow_rate here, but integrate it properly so the
        # check stays correct if the domain ever changes.
        inflow = float(dt) * assemble(Constant(inflow_rate) * ds(boundary_ids.top, domain=mesh))
        current_mass = assemble(theta * dx)

        mass_error += abs(abs(current_mass - previous_mass) - abs(inflow))
        previous_mass = current_mass

    PETSc.Sys.Print(
        f"Net mass loss = {mass_error:.2e} | "
        f"dx = {1/(grid_points-1):.4f} | "
        f"dt = {float(dt):.0f} | "
        f"Function space = {function_space}{polynomial_degree} | "
        f"Time integration = {scheme_label or time_integrator.__name__}"
    )

    return mass_error


@pytest.mark.parametrize("function_space,polynomial_degree", [
    ("CG", 1),
    ("CG", 2),
    ("DQ", 0),
    ("DQ", 1),
    ("DQ", 2),
])
def test_mass_balance(function_space, polynomial_degree):
    grid_points = 26
    dt = 100.0
    t_final = 2e05

    mass_error = compute_mass_balance(
        grid_points, dt, t_final,
        polynomial_degree, BackwardEuler, function_space
    )

    if function_space == 'DQ':
        # DQ spaces with stage_type="value" conserve mass to solver tolerance
        assert mass_error < 1e-10, \
            f"Mass imbalance too large for {function_space}{polynomial_degree}: {mass_error:.2e}"
    else:
        # CG spaces lack local conservation; expect O(1e-4) mass error
        assert mass_error < 1e-3, \
            f"Mass imbalance too large for {function_space}{polynomial_degree}: {mass_error:.2e}"


# Same mass-balance check at fixed DQ1, swept over the time-stepping schemes
# we support (BackwardEuler, ImplicitMidpoint, GaussLegendre(2)). All should
# conserve mass to solver tolerance.
@pytest.mark.parametrize("scheme,scheme_kwargs,scheme_label", [
    (BackwardEuler, None, "BackwardEuler"),
    (ImplicitMidpoint, None, "ImplicitMidpoint"),
    (GaussLegendre, {"tableau_parameter": 2}, "GaussLegendre(2)"),
])
def test_mass_balance_tableaux(scheme, scheme_kwargs, scheme_label):
    grid_points = 26
    dt = 100.0
    # A hundred steps is enough to distinguish working (solver tolerance,
    # ~1e-12) from broken (non-SA linear-combination defect at O(dt^2),
    # ~1e-4); we do not need to integrate to physical steady state to
    # make the point.
    t_final = 100 * dt

    mass_error = compute_mass_balance(
        grid_points, dt, t_final,
        polynomial_degree=1, time_integrator=scheme,
        function_space="DQ",
        timestepper_kwargs=scheme_kwargs,
        scheme_label=scheme_label,
    )

    assert mass_error < 1e-10, \
        f"Mass imbalance too large for {scheme_label}: {mass_error:.2e}"
