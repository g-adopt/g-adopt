from gadopt import *

"""
Performs simple conservation of mass test for different polynomial degrees.
"""

def test_mass_balance_small_domain():

    PETSc.Sys.Print("")

    grid_points = 25
    dt = Constant(100)
    function_space  = "DQ"
    time_integrator = "ImplicitMidpoint"
    time_final = 2e05

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing mass balance for different function spaces")
    PETSc.Sys.Print("="*60)

    for polynomial_degree in range(3):
        compute_mass_balance(grid_points,
                    dt,
                    time_final,
                    polynomial_degree,
                    time_integrator,
                    function_space)
    
    PETSc.Sys.Print("")

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing mass balance for different time integration methods")
    PETSc.Sys.Print("="*60)

    polynomial_degree = 1

    for time_integrator in ['BackwardEuler', 'ImplicitMidpoint', 'CrankNicolson']:
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
                time_integrator: str = "BackwardEuler",
                function_space: str = 'DG'):
    
    mesh = UnitSquareMesh(grid_points, grid_points, quadrilateral=True)
    V    = FunctionSpace(mesh, function_space, polynomial_degree)

    # --- Simple soil model ---
    soil_curves = HaverkampCurve(
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
    bcs = {
        1: {"flux": 0}, # Left
        2: {"flux": 0}, # Right
        3: {"flux": 0}, # Bottom
        4: {"flux": 1e-06}, # Top
        }

    # Initial condition
    h     = Function(V).assign(-1.0)
    h_old = Function(V).assign(h)

    moisture_content = soil_curves.moisture_content
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

    time_var = Constant(0.0)

    # Richards equation object
    eq = RichardsEquation(
        V=V,
        soil_curves=soil_curves,
        bcs=bcs,
        time_integrator=time_integrator,
    )
    
    richards_solver = RichardsSolver(h, h_old, time_var, time_step, eq)

    initial_mass  = assemble(theta * eq.dx)
    previous_mass = initial_mass

    mass_error   = 0.0

    # Run a few timesteps
    while float(time_var) < t_final:

        time_var.assign(time_var + float(time_step))
        h_old.assign(h)
        richards_solver.solve()

        # Compute mass error
        theta.interpolate(moisture_content(h))
        inflow       = float(time_step) * 1e-06
        current_mass = assemble(theta * eq.dx)

        mass_error += abs(abs(current_mass - previous_mass) - abs(inflow))

        previous_mass = current_mass

    # Check mass/loss
    PETSc.Sys.Print("Mass loss/gain", {float(mass_error)})

    assert np.isclose(mass_error, 0.0, atol=1e-7), \
        f"Mass imbalance too large: {mass_error}"
