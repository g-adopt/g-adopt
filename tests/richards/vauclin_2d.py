from gadopt import *
import numpy
import gwassess


def model(level, do_write=False):
    """Vauclin 2D Richards equation reference benchmark.

    Args:
        level: refinement level
        do_write: whether to output the pressure head fields
    """

    # Domain parameters
    Lx = 3.0  # Domain length [m]
    Ly = 2.0  # Domain height [m]
    nodes_x = 50 * (2 ** level)
    nodes_y = 33 * (2 ** level)

    # Create rectangular mesh
    mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=False)
    mesh.cartesian = True
    X = SpatialCoordinate(mesh)

    # Function space for pressure head (P2)
    V = FunctionSpace(mesh, "CG", 2)

    # Initialize reference solution from gwassess
    vauclin_solution = gwassess.VauclinRichardsSolution2D(
        Lx=Lx, Ly=Ly
    )

    # Get soil parameters from gwassess
    soil_params = vauclin_solution.get_soil_parameters()

    # Create Haverkamp soil curve
    soil_curve = HaverkampCurve(
        theta_r=soil_params['theta_r'],
        theta_s=soil_params['theta_s'],
        Ks=soil_params['Ks'],
        Ss=soil_params['Ss'],
        alpha=soil_params['alpha'],
        beta=soil_params['beta'],
        A=soil_params['A'],
        gamma=soil_params['gamma']
    )

    # Initial condition
    h = Function(V, name="PressureHead")
    h.assign(-10.0)

    # Boundary conditions
    boundary_ids = get_boundary_ids(mesh)

    # Time-dependent top boundary flux
    time_var = Constant(0.0)
    top_flux = tanh(0.0005 * time_var) * 2e-05 * (
        0.5 * (1 + tanh(10 * (X[0] + 0.5)))
        - 0.5 * (1 + tanh(10 * (X[0] - 0.5)))
    )

    richards_bcs = {
        boundary_ids.left: {'flux': 0.0},
        boundary_ids.right: {'flux': 0.0},
        boundary_ids.bottom: {'flux': 0.0},
        boundary_ids.top: {'flux': top_flux},
    }

    # Time parameters
    t_final = 86400.0  # 1 day [s]
    dt = Constant(10.0)  # Match reference implementation

    # Create Richards solver with DIRK22 timestepper
    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=ImplicitMidpoint,
        bcs=richards_bcs,
        solver_parameters="direct",
        quad_degree=3
    )

    # Time integration
    time = 0.0
    total_flux = 0.0
    dt_val = float(dt)

    while time < t_final:
        time_var.assign(time)
        richards_solver.solve()
        time += dt_val

        # Compute total infiltration through top boundary
        ds_top = ds(boundary_ids.top)
        flux_top = assemble(top_flux * ds_top)
        total_flux += flux_top * dt_val

    # Compute final solution statistics
    min_h = h.dat.data.min()
    max_h = h.dat.data.max()

    if do_write:
        # Write output file in VTK format
        h_file = VTKFile("vauclin_h_{}.pvd".format(level))
        h_file.write(h)

    return min_h, max_h, total_flux
