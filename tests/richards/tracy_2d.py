from gadopt import *
import numpy
import gwassess


def model(level, bc_type='specified_head', do_write=False):
    """Tracy 2D Richards equation benchmark with analytical solution.

    Args:
        level: refinement level (nodes = 26 * 2^level)
        bc_type: 'specified_head' or 'no_flux'
        do_write: whether to output the pressure head fields
    """

    # Domain parameters
    L = 15.24  # Domain size [m]
    nodes = 26 * (2 ** level)

    # Create square mesh
    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=False)
    mesh.cartesian = True
    X = SpatialCoordinate(mesh)

    # Function space for pressure head (P2)
    V = FunctionSpace(mesh, "DG", 1)

    # Soil parameters (exponential model)
    alpha = 0.328
    hr = -L  # -15.24 m
    theta_r = 0.15
    theta_s = 0.45
    Ks = 1.0e-05
    Ss = 0.0

    # Initialise analytical solution from gwassess
    tracy_solution = gwassess.TracyRichardsSolution2D(
        alpha=alpha, hr=hr, L=L,
        theta_r=theta_r, theta_s=theta_s, Ks=Ks
    )
    # Create soil curve
    soil_curve = ExponentialCurve(
        theta_r=theta_r,
        theta_s=theta_s,
        Ks=Ks,
        Ss=Ss,
        alpha=alpha
    )

    # Set up boundary conditions
    boundary_ids = get_boundary_ids(mesh)
    h0_val = 1 - exp(alpha * hr)

    if bc_type == 'specified_head':
        # Boundary conditions: specified head on all boundaries
        richards_bcs = {
            boundary_ids.left: {'h': hr},
            boundary_ids.right: {'h': hr},
            boundary_ids.bottom: {'h': hr},
            boundary_ids.top: {'h': (1/alpha) * ln(exp(alpha*hr) + h0_val * sin(pi*X[0]/L))},
        }
    elif bc_type == 'no_flux':
        # Boundary conditions: no-flux on sides, specified head on top/bottom
        richards_bcs = {
            boundary_ids.left: {'flux': 0.0},
            boundary_ids.right: {'flux': 0.0},
            boundary_ids.bottom: {'h': hr},
            boundary_ids.top: {'h': (1/alpha) * ln(exp(alpha*hr) + (h0_val/2) * (1 - cos(2*pi*X[0]/L)))},
        }
    else:
        raise ValueError("bc_type must be 'specified_head' or 'no_flux'")

    # Initial condition from analytical solution at t=2000s
    t_initial = 2000.0
    h = Function(V, name="PressureHead")

    # Use gwassess to get proper initial condition
    # Create a temporary CG1 coordinate function to get node coordinates
    V_coords = VectorFunctionSpace(mesh, "DG", 1)
    coords = Function(V_coords).interpolate(as_vector([X[0], X[1]]))

    # Get coordinates for P2 function space by interpolating from CG1
    coords_p2 = Function(VectorFunctionSpace(mesh, "DG", 1))
    coords_p2.interpolate(coords)

    # Set initial condition using gwassess analytical solution
    h.dat.data[:] = [
        tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t_initial, bc_type=bc_type)
        for xy in coords_p2.dat.data
    ]
    # Time parameters
    t_final = 1.0e6
    dt = Constant(5000.0)  # Initial time step size [s]

    # Create Richards solver with RadauIIA adaptive timestepper
    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=RadauIIA,
        bcs=richards_bcs,
        solver_parameters="direct",
        quad_degree=3,
        timestepper_kwargs={
            "adaptive_parameters": {
                'tol': 1e-3,  # Error tolerance per step
                'dtmin': 1e-6,  # Minimum allowed dt
                'dtmax': t_final/100.0,  # Maximum allowed dt (reasonable fraction of total time)
                'KI': 1/15,  # Integration gain
                'KP': 0.13,  # Proportional gain
            }
        }
    )

    # Time integration with adaptive timestepping
    time = t_initial
    step = 0

    output = VTKFile("tracy_h_test.pvd")
    output.write(h)

    while time < t_final:
        result = richards_solver.solve(t=time)

        # Get actual dt used (adaptive timestepping)
        if result is not None:
            error, dt_used = result
            time += dt_used
        else:
            time += float(dt)

        step += 1
        if step % 10 == 0:  # Write output every 10 steps
            output.write(h)

    # Compute analytical solution at final time
    h_anal = Function(V, name="AnalyticalPressureHead")
    h_anal.dat.data[:] = [
        tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t_initial, bc_type=bc_type)
        for xy in coords_p2.dat.data
    ]

    # Compute moisture content
    theta = Function(V, name="MoistureContent")
    theta.interpolate(soil_curve.moisture_content(h))

    theta_anal = Function(V, name="AnalyticalMoistureContent")
    theta_anal.dat.data[:] = [
        tracy_solution.moisture_content(h_val)
        for h_val in h_anal.dat.data
    ]

    # Compute errors
    h_error = Function(V, name="PressureHeadError").assign(h - h_anal)
    theta_error = Function(V, name="MoistureContentError").assign(theta - theta_anal)

    if do_write:
        # Write output files in VTK format
        h_file = VTKFile("tracy_h_{}.pvd".format(level))
        theta_file = VTKFile("tracy_theta_{}.pvd".format(level))

        h_file.write(h, h_anal, h_error)
        theta_file.write(theta, theta_anal, theta_error)

    # Compute L2 norms
    dx_quad = dx(metadata={"quadrature_degree": 3})
    l2anal_h = numpy.sqrt(assemble(dot(h_anal, h_anal) * dx_quad))
    l2anal_theta = numpy.sqrt(assemble(dot(theta_anal, theta_anal) * dx_quad))
    l2error_h = numpy.sqrt(assemble(dot(h_error, h_error) * dx_quad))
    l2error_theta = numpy.sqrt(assemble(dot(theta_error, theta_error) * dx_quad))

    return l2error_h, l2error_theta, l2anal_h, l2anal_theta
