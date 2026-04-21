"""Mass conservation verification for the Richards equation solver.

Tests that the discrete mass balance holds across different function spaces
(CG1, CG2, DQ0, DQ1, DQ2) with BackwardEuler time integration. The solver
uses stage_type="value" by default, which discretises Dt(theta(h)) as an
exact finite difference rather than using the chain-rule capacity form.
This should give mass conservation to solver tolerance for DQ spaces.

The test domain is a unit square with no-flux boundaries on left, right,
and bottom, and a prescribed inflow flux of 1e-6 m/s on top. The soil
follows a Haverkamp model with parameters chosen to give moderate
nonlinearity.
"""

import pytest
from gadopt import *


def compute_mass_balance(grid_points, time_step, t_final,
                         polynomial_degree, time_integrator,
                         function_space):
    mesh = UnitSquareMesh(grid_points, grid_points, quadrilateral=True)
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
    )

    initial_mass = assemble(theta * dx)
    previous_mass = initial_mass
    mass_error = 0.0

    time_var = 0.0
    while time_var < t_final:
        time_var += float(dt)
        richards_solver.solve()

        theta.interpolate(moisture_content(h))
        inflow = float(dt) * inflow_rate
        current_mass = assemble(theta * dx)

        mass_error += abs(abs(current_mass - previous_mass) - abs(inflow))
        previous_mass = current_mass

    PETSc.Sys.Print(
        f"Net mass loss = {mass_error:.2e} | "
        f"dx = {1/(grid_points-1):.4f} | "
        f"dt = {float(dt):.0f} | "
        f"Function space = {function_space}{polynomial_degree} | "
        f"Time integration = {time_integrator.__name__}"
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
