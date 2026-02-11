from gadopt import *
import numpy
import gwassess


def visualise_tracy_analytical(
    level=2,
    bc_type='specified_head',
    times=None,
    output_prefix='tracy_analytical',
):
    """Visualise Tracy 2D analytical solution at multiple times using VTK output.

    Args:
        level: Refinement level (nodes = 26 * 2^level)
        bc_type: 'specified_head' or 'no_flux'
        times: List of times [s] at which to evaluate the solution.
               Default: [0, 1000, 10000, 100000, 1000000]
        output_prefix: Prefix for output PVD file.

    Returns:
        dict: Dictionary containing the tracy_solution instance and parameters.
    """
    if times is None:
        # Note: t=0 causes math domain errors due to large transient terms
        times = [2000, 10000, 100000, 500000, 1000000]

    # Domain and soil parameters
    L = 15.24  # Domain size [m]
    nodes = 26 * (2 ** level)

    # Create mesh
    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=False)
    mesh.cartesian = True
    X = SpatialCoordinate(mesh)

    # Function space
    V = FunctionSpace(mesh, "DG", 2)

    # Soil parameters (exponential model)
    alpha = 0.328
    hr = -L  # -15.24 m
    theta_r = 0.15
    theta_s = 0.45
    Ks = 1.0e-05

    # Create Tracy analytical solution
    tracy_solution = gwassess.TracyRichardsSolution2D(
        alpha=alpha, hr=hr, L=L,
        theta_r=theta_r, theta_s=theta_s, Ks=Ks
    )

    # Get coordinates for evaluating analytical solution
    V_coords = VectorFunctionSpace(mesh, "DG", 2)
    coords = Function(V_coords).interpolate(as_vector([X[0], X[1]]))

    # Create functions for output
    h_anal = Function(V, name="AnalyticalPressureHead")
    theta_anal = Function(V, name="AnalyticalMoistureContent")

    # Create VTK output
    output = VTKFile(f"{output_prefix}_{bc_type}.pvd")

    # Write solution at each time
    for t in times:
        # Update pressure head
        h_anal.dat.data[:] = [
            tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t, bc_type=bc_type)
            for xy in coords.dat.data
        ]

        # Update moisture content
        theta_anal.dat.data[:] = [
            tracy_solution.moisture_content(h_val)
            for h_val in h_anal.dat.data
        ]

        # Write to VTK with time
        output.write(h_anal, theta_anal, time=t)
        print(f"Wrote t = {t:.1f} s")

    print(f"Output saved to: {output_prefix}_{bc_type}.pvd")

    return {
        'tracy_solution': tracy_solution,
        'params': {
            'L': L, 'alpha': alpha, 'hr': hr,
            'theta_r': theta_r, 'theta_s': theta_s, 'Ks': Ks,
        },
    }


def model(level, bc_type='specified_head', degree=1, do_write=False, space_type='DG',
          convergence_tol=1e-6, t_max=1.0e6):
    """Tracy 2D Richards equation benchmark with analytical solution.

    Args:
        level: Refinement level (nodes = 26 * 2^level)
        bc_type: 'specified_head' or 'no_flux'
        degree: Polynomial degree (1 or 2)
        do_write: Whether to output the pressure head fields
        space_type: 'DG' for discontinuous Galerkin or 'CG' for continuous Galerkin
        convergence_tol: Relative tolerance for steady-state convergence (default: 1e-6)
        t_max: Maximum simulation time if convergence not reached (default: 1e6)

    Returns:
        dict: Dictionary containing:
            - l2error_h: L2 error in pressure head at final time
            - l2error_theta: L2 error in moisture content at final time
            - l2anal_h: L2 norm of analytical pressure head
            - l2anal_theta: L2 norm of analytical moisture content
            - max_intermediate_error_h: Maximum relative L2 error across all output steps
            - final_time: Actual final simulation time reached
            - converged: Whether the solution converged to steady state
    """

    # Domain parameters
    L = 15.24  # Domain size [m]
    nodes = 26 * (2 ** level)

    # Create square mesh
    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=False)
    mesh.cartesian = True
    X = SpatialCoordinate(mesh)

    # Function space for pressure head
    V = FunctionSpace(mesh, space_type, degree)

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
    # Create coordinate function matching the function space
    V_coords = VectorFunctionSpace(mesh, space_type, degree)
    coords = Function(V_coords).interpolate(as_vector([X[0], X[1]]))

    # Set initial condition using gwassess analytical solution
    h.dat.data[:] = [
        tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t_initial, bc_type=bc_type)
        for xy in coords.dat.data
    ]
    # Time parameters
    # Note: Tracy solution valid for t >= ~2000s (transient terms cause issues at t=0)
    t_final = t_max
    adaptive_tol = 1e-2  # Truncation tolerance for Irksome
    dt = Constant(0.01)  # Initial time-step

    # Create Richards solver with RadauIIA timestepper
    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=RadauIIA,
        bcs=richards_bcs,
        solver_parameters="direct",
        solver_parameters_extra={
            "snes_atol": 1e-12,  # Absolute tolerance to avoid over-solving when residual is already small
        },
        quad_degree=3,
        timestepper_kwargs={
            "adaptive_parameters": {
                'tol': adaptive_tol,  # Error tolerance per step
                'dtmin': 1e-6,  # Minimum allowed dt
                'dtmax': t_final/100.0,  # Maximum allowed dt (reasonable fraction of total time)
                'KI': 1/15,  # Integration gain
                'KP': 0.13,  # Proportional gain
                'max_reject': 50,  # Maximum number that the truncation error can exceed 'tol'
            }
        },
        interior_penalty=0.5 if space_type == 'DG' else None,
    )

    # Create functions for analytical solution and error computation
    h_anal = Function(V, name="AnalyticalPressureHead")
    theta_anal = Function(V, name="AnalyticalMoistureContent")
    h_error = Function(V, name="PressureHeadError")
    theta_error = Function(V, name="MoistureContentError")
    h_prev = Function(V, name="PreviousPressureHead")
    h_prev.assign(h)  # Store initial solution
    dx_quad = dx(metadata={"quadrature_degree": 3})

    def update_analytical_solution(t):
        """Update analytical solution functions at time t."""
        h_anal.dat.data[:] = [
            tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t, bc_type=bc_type)
            for xy in coords.dat.data
        ]
        theta_anal.dat.data[:] = [
            tracy_solution.moisture_content(h_val)
            for h_val in h_anal.dat.data
        ]
        # Compute errors
        h_error.assign(h - h_anal)
        theta_interp = Function(V)
        theta_interp.interpolate(soil_curve.moisture_content(h))
        theta_error.assign(theta_interp - theta_anal)

    def compute_relative_error_h():
        """Compute relative L2 error in pressure head."""
        l2_anal = numpy.sqrt(assemble(dot(h_anal, h_anal) * dx_quad))
        l2_err = numpy.sqrt(assemble(dot(h_error, h_error) * dx_quad))
        return l2_err / l2_anal if l2_anal > 0 else l2_err

    def compute_relative_change():
        """Compute relative change in solution between consecutive timesteps."""
        diff = h - h_prev
        l2_diff = numpy.sqrt(assemble(dot(diff, diff) * dx_quad))
        l2_h = numpy.sqrt(assemble(dot(h, h) * dx_quad))
        return l2_diff / l2_h if l2_h > 0 else l2_diff

    # Time integration with adaptive timestepping
    # Note: Tracy solution requires t >= ~2000s to avoid math domain errors
    time = t_initial
    step = 0
    converged = False

    # Track intermediate errors
    intermediate_errors_h = []

    if do_write:
        output = VTKFile("tracy_h_{}{}_lv{}.pvd".format(space_type.lower(), degree, level))
        # Write initial state with analytical comparison
        update_analytical_solution(time)
        output.write(h, h_anal, h_error, time=time)
        intermediate_errors_h.append(compute_relative_error_h())

    while time < t_final:
        result = richards_solver.solve(t=time)

        # Get actual dt used (adaptive timestepping)
        if result is not None:
            error, dt_used = result
            time += dt_used
        else:
            time += float(dt)

        step += 1

        # Check for convergence every few steps (after initial transient)
        if step % 10 == 0:
            rel_change = compute_relative_change()
            if rel_change < convergence_tol:
                converged = True
                break
            # Update previous solution for next convergence check
            h_prev.assign(h)

        if step % 50 == 0:
            # Update analytical solution at current time
            update_analytical_solution(time)
            # Track intermediate error
            intermediate_errors_h.append(compute_relative_error_h())
            if do_write:
                output.write(h, h_anal, h_error, time=time)

    # Compute analytical solution at actual final time
    update_analytical_solution(time)

    # Compute final L2 norms
    l2anal_h = numpy.sqrt(assemble(dot(h_anal, h_anal) * dx_quad))
    l2anal_theta = numpy.sqrt(assemble(dot(theta_anal, theta_anal) * dx_quad))
    l2error_h = numpy.sqrt(assemble(dot(h_error, h_error) * dx_quad))
    l2error_theta = numpy.sqrt(assemble(dot(theta_error, theta_error) * dx_quad))

    # Compute max intermediate error (include final error)
    intermediate_errors_h.append(l2error_h / l2anal_h if l2anal_h > 0 else l2error_h)
    max_intermediate_error_h = max(intermediate_errors_h)

    return {
        'l2error_h': l2error_h,
        'l2error_theta': l2error_theta,
        'l2anal_h': l2anal_h,
        'l2anal_theta': l2anal_theta,
        'max_intermediate_error_h': max_intermediate_error_h,
        'final_time': time,
        'converged': converged,
    }
