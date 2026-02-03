from gadopt import *

"""
Spatial convergence test of Vauclin benchmark
"""

def main():

    t_final = 28800
    time_step = 25
    polynomial_degree = 2
    grid_space = 0.020

    PETSc.Sys.Print('Generating reference solution...')
    h_ref = vauclin_benmark(t_final, time_step, grid_space, polynomial_degree)
    PETSc.Sys.Print('Done')

    # Function space to interpolate solutions onto
    V_comp = FunctionSpace(h_ref.function_space().mesh(), "DQ", 2)
    h_interp = Function(V_comp)

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing convergence check with DG0")
    PETSc.Sys.Print("="*60)

    polynomial_degree = 0

    dx_vec = np.array([0.066666, 0.05, 0.04, 0.03333, 0.025, 0.02, 0.01, 0.005])
    error_vec = dx_vec*0

    for index in range(len(dx_vec)):
        dx = dx_vec[index]
        h = vauclin_benmark(t_final, time_step, dx, polynomial_degree, BackwardEuler)
        err = errornorm(h_ref, h_interp.interpolate(h), norm_type="L2", degree_rise=3)
        PETSc.Sys.Print(f"Error with dx= {dx} is {err}")
        error_vec[index] = err

    X = np.log10(dx_vec)
    Y = np.log10(error_vec)
    slope, intercept = np.polyfit(X, Y, 1)
    PETSc.Sys.Print(f"Convergence rate is {slope}")

    polynomial_degree = 0

    dx_vec = np.array([0.066666, 0.05, 0.04, 0.03333, 0.025, 0.02, 0.01, 0.005])
    error_vec = dx_vec*0

    for index in range(len(dx_vec)):
        dx = dx_vec[index]
        h = vauclin_benmark(t_final, time_step, dx, polynomial_degree, BackwardEuler)
        err = errornorm(h_ref, h_interp.interpolate(h), norm_type="L2", degree_rise=3)
        PETSc.Sys.Print(f"Error with dx= {dx} is {err}")
        error_vec[index] = err

    X = np.log10(dx_vec)
    Y = np.log10(error_vec)
    slope, intercept = np.polyfit(X, Y, 1)
    PETSc.Sys.Print(f"Convergence rate is {slope}")


    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing convergence check with DG1")
    PETSc.Sys.Print("="*60)

    polynomial_degree = 1

    dx_vec = np.array([0.16666, 0.1, 0.083333, 0.066666, 0.05, 0.04, 0.03333, 0.025, 0.02])
    error_vec = dx_vec*0

    for index in range(len(dx_vec)):
        dx = dx_vec[index]
        h = vauclin_benmark(t_final, time_step, dx, polynomial_degree)
        err = errornorm(h_ref, h_interp.interpolate(h), norm_type="L2", degree_rise=3)
        PETSc.Sys.Print(f"Error with dx= {dx} is {err}")
        error_vec[index] = err

    X = np.log10(dx_vec)
    Y = np.log10(error_vec)
    slope, intercept = np.polyfit(X, Y, 1)
    PETSc.Sys.Print(f"Convergence rate is {slope}")


def vauclin_benmark(t_final=28800, 
                    time_step=50, 
                    grid_space=0.01,
                    polynomial_degree=1,
                    time_integration=ImplicitMidpoint,
                    ):

    Lx, Ly = 3.00, 2.00  # Domain length [m]
    nodes_x, nodes_y = round(Lx/grid_space) + 1, round(Ly/grid_space) + 1

    dt = Constant(time_step)

    # Create rectangular mesh
    mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
    X = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "DQ", polynomial_degree)
    W = VectorFunctionSpace(mesh, 'DQ', polynomial_degree)

    soil_curve = HaverkampCurve(
        theta_r=0.00,   # Residual water content [-]
        theta_s=0.37,   # Saturated water content [-]
        Ks=9.722e-05,   # Saturated hydraulic conductivity [m/s]
        alpha=0.44,     # Fitting parameter [m]
        beta=1.2924,    # Fitting parameter [-]
        A=0.0104,       # Fitting parameter [m]
        gamma=1.5722,   # Fitting parameter [-]
        Ss=0e-00,       # Specific storage coefficient [1/m]
    )

    moisture_content = soil_curve.moisture_content
    relative_permeability = soil_curve.relative_permeability

    h_ic = Function(V, name="InitialCondition").interpolate(0.65 - X[1])

    h = Function(V, name="PressureHead").interpolate(h_ic)
    h_old = Function(V, name="PreviousSolution").interpolate(h_ic)
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h_ic))
    q = Function(W, name='VolumetricFlux')
    K = Function(V, name='RelativeConductivity').interpolate(relative_permeability(h_ic))

    # Set up boundary conditions
    time_var = Constant(0.0)

    # Define the recharge region (0 <= x <= 0.5 m) using tanh smoothing
    recharge_rate = Constant(4.11e-05) # m/s  # m/s 
    left_edge = 0.5 * (1 + tanh(10 * (X[0] + 0.50))) 
    right_edge = 0.5 * (1 + tanh(10 * (X[0] - 0.50))) 
    recharge_region_indicator = left_edge - right_edge

    top_flux = tanh(0.000125 * time_var) * recharge_rate * recharge_region_indicator

    # Boundary conditions
    boundary_ids = get_boundary_ids(mesh)
    richards_bcs = {
        boundary_ids.left: {'flux': 0.0},
        boundary_ids.right: {'h': 0.65 - X[1]},
        boundary_ids.bottom: {'flux': 0.0},
        boundary_ids.top: {'flux': top_flux},
    }

    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=time_integration,
        bcs=richards_bcs,
        )

    time = 0
    external_flux = 0
    initial_mass = assemble(theta*dx)

    while time < t_final:

        h_old.assign(h)
        time_var.assign(time)
        richards_solver.solve()
        time += float(dt)

        theta.interpolate(moisture_content(h))
        K.interpolate(relative_permeability(h))
        q.interpolate(-K*grad(h + X[1]))

        external_flux += assemble(float(dt)*dot(q, -FacetNormal(mesh))*ds)

        snes = richards_solver.ts.stepper.solver.snes
        nonlinear_iterations = snes.getIterationNumber()
        if float(time + dt) > t_final:
            dt.assign(time + dt - t_final)

    return h

if __name__ == "__main__":
    main()
