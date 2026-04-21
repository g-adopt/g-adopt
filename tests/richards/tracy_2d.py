from gadopt import *
import numpy as np
import gwassess


def model(nodes, degree=1, bc_type='specified_head'):
    """Tracy 2D Richards equation benchmark with analytical solution.

    Solves to steady state on a square domain using BackwardEuler with
    dt ramp-up, then computes L2 error against Tracy (2006) analytical
    solution. Parameters match Morrow et al. (2026), Section 4.1.

    Args:
        nodes: Number of cells per spatial dimension.
        degree: DQ polynomial degree (0, 1, or 2).
        bc_type: 'specified_head' or 'no_flux'.

    Returns:
        Tuple of (l2error_h, l2error_theta, l2anal_h, l2anal_theta).
    """

    L = 15.24
    alpha = 0.25
    hr = -L
    theta_r = 0.15
    theta_s = 0.45
    Ks = 1.0e-05

    tracy_solution = gwassess.TracyRichardsSolution2D(
        alpha=alpha, hr=hr, L=L,
        theta_r=theta_r, theta_s=theta_s, Ks=Ks,
    )

    soil_curve = ExponentialCurve(
        theta_r=theta_r, theta_s=theta_s,
        Ks=Ks, Ss=0.0, alpha=alpha,
    )

    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=True)
    mesh.cartesian = True
    X = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "DQ", degree)

    # Boundary conditions
    h0_val = 1 - exp(alpha * hr)
    boundary_ids = get_boundary_ids(mesh)

    if bc_type == 'specified_head':
        richards_bcs = {
            boundary_ids.left: {'h': hr},
            boundary_ids.right: {'h': hr},
            boundary_ids.bottom: {'h': hr},
            boundary_ids.top: {'h': (1/alpha) * ln(exp(alpha*hr) + h0_val * sin(pi*X[0]/L))},
        }
    elif bc_type == 'no_flux':
        richards_bcs = {
            boundary_ids.left: {'flux': 0.0},
            boundary_ids.right: {'flux': 0.0},
            boundary_ids.bottom: {'h': hr},
            boundary_ids.top: {'h': (1/alpha) * ln(exp(alpha*hr) + (h0_val/2) * (1 - cos(2*pi*X[0]/L)))},
        }
    else:
        raise ValueError("bc_type must be 'specified_head' or 'no_flux'")

    # Initial condition from analytical solution at t_offset
    t_offset = 2000.0
    V_coords = VectorFunctionSpace(mesh, "DQ", degree)
    coords = Function(V_coords).interpolate(as_vector([X[0], X[1]]))

    h = Function(V, name="PressureHead")
    h.dat.data[:] = [
        tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t_offset, bc_type=bc_type)
        for xy in coords.dat.data
    ]

    # Solve to steady state with BackwardEuler and dt ramp-up
    t_final = 2.5e6
    dt = Constant(5e4)

    richards_solver = RichardsSolver(
        h, soil_curve, delta_t=dt,
        timestepper=BackwardEuler,
        bcs=richards_bcs,
        solver_parameters="direct",
        quad_degree=3,
        interior_penalty=0.5,
    )

    time = 0.0
    while time < t_final:
        richards_solver.solve()
        time += float(dt)
        dt.assign(min(float(dt) * 1.05, t_final / 10))

    # Compute L2 errors against analytical solution at t_offset + time
    dx_quad = dx(metadata={"quadrature_degree": 3})

    h_anal = Function(V, name="AnalyticalPressureHead")
    h_anal.dat.data[:] = [
        tracy_solution.pressure_head_cartesian([xy[0], xy[1]], t_offset + time, bc_type=bc_type)
        for xy in coords.dat.data
    ]

    theta_anal = Function(V, name="AnalyticalMoistureContent")
    theta_anal.dat.data[:] = [
        tracy_solution.moisture_content(h_val)
        for h_val in h_anal.dat.data
    ]

    theta_num = Function(V)
    theta_num.interpolate(soil_curve.moisture_content(h))

    l2error_h = np.sqrt(assemble((h - h_anal)**2 * dx_quad))
    l2error_theta = np.sqrt(assemble((theta_num - theta_anal)**2 * dx_quad))
    l2anal_h = np.sqrt(assemble(h_anal**2 * dx_quad))
    l2anal_theta = np.sqrt(assemble(theta_anal**2 * dx_quad))

    log(f"L2 error (h): {l2error_h:.4e} | L2 analytical (h): {l2anal_h:.4e} | "
        f"Relative: {l2error_h/l2anal_h:.4e} | "
        f"nodes = {nodes} | DQ{degree} | DOFs = {V.dim()}")

    return l2error_h, l2error_theta, l2anal_h, l2anal_theta
