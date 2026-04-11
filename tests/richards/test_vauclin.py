from gadopt import *

"""
Convergence test of Vauclin benchmark
"""


def test_vauclin():

    t_final = 28800
    time_step = 200
    polynomial_degree = 2
    grid_space = 2

    PETSc.Sys.Print('Generating reference solution...')
    h_ref = vauclin_benchmark(t_final, time_step, grid_space, polynomial_degree)
    PETSc.Sys.Print('Done')

    # Function space to interpolate solutions onto
    V_comp = FunctionSpace(h_ref.function_space().mesh(), "DQ", polynomial_degree)
    h_interp = Function(V_comp)

    PETSc.Sys.Print("")

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing convergence check with DG0")
    PETSc.Sys.Print("="*60)

    polynomial_degree = 0

    polynomial_degree = 0
    dx_vec = np.array([4, 3.333, 2.5])
    error_vec = np.zeros(len(dx_vec), dtype=float)

    for index in range(len(dx_vec)):
        dx = dx_vec[index]
        h = vauclin_benchmark(t_final, time_step, dx, polynomial_degree=0)
        err = errornorm(h_ref, h_interp.interpolate(h), norm_type="L2", degree_rise=3)
        PETSc.Sys.Print(f"Error with dx = {dx} is {err:.2e}")
        error_vec[index] = err

    X = np.log10(dx_vec)
    Y = np.log10(error_vec)
    slope, intercept = np.polyfit(X, Y, 1)
    PETSc.Sys.Print(f"Convergence rate is {slope}")
    assert slope >= 0.9, "Optimal convergence rate not achieved."

    PETSc.Sys.Print("")

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing convergence check with DG1")
    PETSc.Sys.Print("="*60)

    polynomial_degree = 1
    dx_vec = np.array([5, 4, 3.333, 2.5])
    error_vec = np.zeros(len(dx_vec), dtype=float)

    for index in range(len(dx_vec)):
        dx = dx_vec[index]
        h = vauclin_benchmark(t_final, time_step, dx, polynomial_degree=1)
        err = errornorm(h_ref, h_interp.interpolate(h), norm_type="L2", degree_rise=3)
        PETSc.Sys.Print(f"Error with dx = {dx} is {err:.2e}")
        error_vec[index] = err

    X = np.log10(dx_vec)
    Y = np.log10(error_vec)
    slope, intercept = np.polyfit(X, Y, 1)
    PETSc.Sys.Print(f"Convergence rate is {slope}")
    assert slope >= 1.9, "Optimal convergence rate not achieved."


def vauclin_benchmark(t_final=28800,
                    time_step=50,
                    grid_space=1,
                    polynomial_degree=1,
                    time_integrator=BackwardEuler,
                    ):

    """
    Runs the Vauclin infiltration benchmark and returns the final pressure head.

    Parameters
    ----------
    t_final : float, Final simulation time in seconds.
    time_step : float, Time step size.
    grid_space : float, Spatial resolution (m).
    polynomial_degree : int, DG polynomial degree.
    time_integration : object, Time integration method (e.g. ImplicitMidpoint).

    Returns
    -------
    Function
        Final pressure head field.
    """

    Lx, Ly = 300, 200  # Domain length [cm]
    nodes_x, nodes_y = round(Lx/grid_space) + 1, round(Ly/grid_space) + 1

    dt = Constant(time_step)

    # Create rectangular mesh
    mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
    X = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "DQ", polynomial_degree)

    soil_curve = HaverkampCurve(
        theta_r=0.00,   # Residual water content [-]
        theta_s=0.30,   # Saturated water content [-]
        Ks=9.722e-03,   # Saturated hydraulic conductivity [cm/s]
        alpha=40000,    # Fitting parameter [cm]
        beta=2.90,      # Fitting parameter [-]
        A=2.99e6,       # Fitting parameter [cm]
        gamma=5.00,     # Fitting parameter [-]
        Ss=0.00,        # Specific storage coefficient [1/cm]
    )

    h_ic = Function(V, name="InitialCondition").interpolate(65 - X[1])

    h = Function(V, name="PressureHead").interpolate(h_ic)

    # Set up boundary conditions
    time_var = Constant(0.0)

    # Define the recharge region (0 <= x <= 0.5 m) using tanh smoothing
    recharge_rate = Constant(4.11e-03) # m/s  # m/s 
    left_edge = 0.5 * (1 + tanh(0.1 * (X[0] + 50))) 
    right_edge = 0.5 * (1 + tanh(0.1 * (X[0] - 50))) 
    recharge_region_indicator = left_edge - right_edge

    top_flux = tanh(0.000125 * time_var) * recharge_rate * recharge_region_indicator

    # Boundary conditions
    richards_bcs = {
        1: {'flux': 0.0},
        2: {'h': 65 - X[1]},
        3: {'flux': 0.0},
        4: {'flux': top_flux},
    }

    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=time_integrator,
        bcs=richards_bcs,
    )

    time = 0

    while time < t_final:

        time += float(dt)
        time_var.assign(time)
        richards_solver.solve()

        if float(time + dt) > t_final:
            dt.assign(time + dt - t_final)

    return h
