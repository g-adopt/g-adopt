from gadopt import *

"""
Performs simple conservation of mass test for different polynomial degrees.
"""

def test_mass_balance_small_domain():

    PETSc.Sys.Print("")

    grid_points = 26
    dt = Constant(100)
    time_integrator = BackwardEuler
    time_final = 2e05

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing mass balance for different function spaces")
    PETSc.Sys.Print("="*60)

    for function_space in ['CG', 'DQ']:
        for polynomial_degree in [0, 1, 2]:
            if polynomial_degree == 0 and function_space == 'CG':
                continue  # This combo doesn't exist
            else:
                compute_mass_balance(grid_points,
                            dt,
                            time_final,
                            polynomial_degree,
                            time_integrator,
                            function_space)


def compute_mass_balance(grid_points: int,
                time_step: Constant = 10.0,
                t_final: float = 1.0e5,
                polynomial_degree: int = 1,
                time_integrator: str = BackwardEuler,
                function_space: str = 'DG'):
    
    mesh = UnitSquareMesh(grid_points, grid_points, quadrilateral=True)
    V    = FunctionSpace(mesh, function_space, polynomial_degree)

    # --- Simple soil model ---
    soil_curve = HaverkampCurve(
        theta_r = 0.05,
        theta_s = 0.40,
        Ks      = 1e-5,
        alpha   = 0.5,
        beta    = 1.3,
        A       = 0.01,
        gamma   = 1.5,
        Ss      = 0.0,
    )

    # Boundary conditions: no flux everywhere except top
    boundary_ids = get_boundary_ids(mesh)
    richards_bcs = {
        boundary_ids.left:   {'flux': 0},
        boundary_ids.right:  {'flux': 0},
        boundary_ids.bottom: {'flux': 0},
        boundary_ids.top:    {'flux': 1e-06},
    }

    # Initial condition
    h     = Function(V).assign(-1.0)
    h_old = Function(V).assign(-1.0)

    moisture_content = soil_curve.moisture_content
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

    # Tight solver tolerance for mass conservation test
    solver_parameters_extra = {'snes_atol': 1e-15}

#    Richards equation object
    time_var = Constant(0.0)
    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=time_step,
        timestepper=time_integrator,
        solver_parameters_extra=solver_parameters_extra,
        bcs=richards_bcs,
    )

    #dx_quad = dx(metadata={"quadrature_degree": 2*polynomial_degree+1})
    initial_mass  = assemble(theta * dx)
    previous_mass = initial_mass

    mass_error   = 0.0

    # Run a few timesteps
    while float(time_var) < t_final:

        h_old.assign(h)
        time_var.assign(time_var + float(time_step))
        richards_solver.solve()

        # Compute mass error
        theta.interpolate(moisture_content(h))
        inflow       = float(time_step) * 1e-06
        current_mass = assemble(theta * dx)

        # Net mass loss each timestep
        mass_error += abs(abs(current_mass - previous_mass) - abs(inflow))

        previous_mass = current_mass

    PETSc.Sys.Print(f"Net mass loss = {mass_error:.2e} | "
            f"dx = {1/(grid_points-1):.4f} | "
            f"dt = {float(time_step):.0f} | "
            f"Function space = {function_space} | "
            f"Polynomial degree = {polynomial_degree} | "
            f"Time integration = {time_integrator} | "
            )

    # Only test mass conservation for expect mass conservation method
    if function_space == 'DQ':
        assert np.isclose(mass_error, 0.0, atol=1e-6), \
            f"Mass imbalance too large: {mass_error}"
