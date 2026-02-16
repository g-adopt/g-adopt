"""Vauclin 2D water table recharge benchmark.

This test reproduces the benchmark presented in:
    Vauclin, M., Khanji, D., & Vachaud, G. (1979). Experimental and numerical study
    of a transient, two-dimensional unsaturated-saturated water table recharge problem.
    Water Resources Research, 15(5), 1089-1101.
    https://doi.org/10.1029/WR015i005p01089

The simulation is performed in a domain of 3 x 2 metres, with initial condition
such that the region z <= 0.65 m is fully saturated (h = z - 0.65). Boundary
conditions: bottom and left are no-flux, right has fixed water table height
(h = z - 0.65 m), top has water injection at 14.8 cm/hour for x <= 0.5 m.
The simulation runs for 8 hours.
"""
from gadopt import *
import gwassess


def model(level=0, do_write=False, t_final=None):
    """Run Vauclin 2D Richards equation benchmark.

    Args:
        level: Mesh refinement level (0=coarsest)
        do_write: Whether to output VTK files
        t_final: Simulation end time [s]. If None, uses paper value (8 hours).

    Returns:
        Tuple of (min_h, max_h, external_flux, mass_balance)
    """
    # Initialize reference solution from gwassess
    vauclin_solution = gwassess.VauclinRichardsSolution2D()

    # Domain parameters from gwassess
    Lx = vauclin_solution.Lx
    Ly = vauclin_solution.Ly

    # Mesh resolution scales with refinement level
    nodes_x = 91 * (2 ** level)
    nodes_y = 61 * (2 ** level)

    # Use paper value for final time if not specified
    if t_final is None:
        t_final = vauclin_solution.SIMULATION_DURATION

    # Create rectangular mesh with quadrilateral elements
    mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
    mesh.cartesian = True
    X = SpatialCoordinate(mesh)

    # Function spaces
    V = FunctionSpace(mesh, "DQ", 2)  # Discontinuous Galerkin for pressure head
    W = VectorFunctionSpace(mesh, "DQ", 2)  # For flux output

    # Get soil parameters from gwassess
    soil_params = vauclin_solution.get_soil_parameters()

    # Create Haverkamp soil curve with paper parameters
    soil_curve = HaverkampCurve(
        theta_r=soil_params['theta_r'],
        theta_s=soil_params['theta_s'],
        Ks=soil_params['Ks'],
        Ss=soil_params['Ss'],
        alpha=soil_params['alpha'],
        beta=soil_params['beta'],
        A=soil_params['A'],
        gamma=soil_params['gamma'],
    )

    moisture_content = soil_curve.moisture_content
    relative_permeability = soil_curve.relative_permeability

    # Initial condition: water table at z = 0.65 m
    water_table_height = vauclin_solution.WATER_TABLE_HEIGHT
    h_ic = Function(V, name="InitialCondition").interpolate(
        water_table_height - 1.001 * X[1]
    )
    h = Function(V, name="PressureHead").interpolate(h_ic)
    h_old = Function(V, name="PreviousSolution").interpolate(h_ic)

    # Diagnostic fields
    theta = Function(V, name="MoistureContent").interpolate(moisture_content(h))
    q = Function(W, name="VolumetricFlux")
    K = Function(V, name="RelativeConductivity").interpolate(relative_permeability(h))

    # Time-dependent top boundary flux
    time_var = Constant(0.0)
    infiltration_rate = vauclin_solution.INFILTRATION_RATE
    infiltration_width = vauclin_solution.INFILTRATION_WIDTH

    top_flux = tanh(0.000125 * time_var) * infiltration_rate * (
        0.5 * (1 + tanh(10 * (X[0] + infiltration_width)))
        - 0.5 * (1 + tanh(10 * (X[0] - infiltration_width)))
    )

    # Boundary conditions from paper:
    # - Left: no flux
    # - Right: fixed water table height
    # - Bottom: no flux
    # - Top: infiltration flux
    boundary_ids = get_boundary_ids(mesh)
    richards_bcs = {
        boundary_ids.left: {'flux': 0.0},
        boundary_ids.right: {'h': h_ic},  # Fixed water table
        boundary_ids.bottom: {'flux': 0.0},
        boundary_ids.top: {'flux': top_flux},
    }

    # Time stepping
    dt = Constant(10)

    # Create Richards solver
    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=ImplicitMidpoint,
        bcs=richards_bcs,
        solver_parameters="direct",
        quad_degree=5,
    )

    # Output setup
    if do_write:
        output = VTKFile("vauclin.pvd")
        output.write(h, theta, q)

    # Mass balance tracking
    time = 0
    external_flux = 0
    initial_mass = assemble(theta * dx)

    # Measures with appropriate quadrature
    ds_mesh = Measure("ds", domain=mesh, metadata={"quadrature_degree": 5})
    dx_mesh = Measure("dx", domain=mesh, metadata={"quadrature_degree": 5})

    plot_iteration = 0

    # Time loop
    while time < t_final:
        h_old.assign(h)
        time_var.assign(time)
        richards_solver.solve()
        time += float(dt)

        # Update diagnostic fields
        theta.interpolate(moisture_content(h))
        K.interpolate(relative_permeability((h + h_old) / 2))
        q.interpolate(-K * grad((h + h_old) / 2 + X[1]))

        # Track external flux through all boundaries
        external_flux += assemble(float(dt) * dot(q, -FacetNormal(mesh)) * ds_mesh)

        plot_iteration += 1
        if do_write and plot_iteration % 25 == 0:
            output.write(h, theta, q)

    # Final diagnostics
    final_mass = assemble(theta * dx_mesh)
    mass_balance = (final_mass - initial_mass) / external_flux if external_flux != 0 else 0

    min_h = h.dat.data.min()
    max_h = h.dat.data.max()

    return min_h, max_h, external_flux, mass_balance


if __name__ == "__main__":
    min_h, max_h, external_flux, mass_balance = model(level=0, do_write=True)
    print(f"Min pressure head: {min_h:.4f} m")
    print(f"Max pressure head: {max_h:.4f} m")
    print(f"External flux: {external_flux:.6f}")
    print(f"Mass balance: {mass_balance:.4f}")
