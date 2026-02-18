from gadopt import *

"""
Convergence study of Tracy's two dimensional exact solution (steady-state and transient)
====================================================
"""


def test_tracy():

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing steady-state solution check with DG0")
    PETSc.Sys.Print("="*60)

    t_final, polynomial_degree, integration_method = 1.0e09, 0, BackwardEuler
    nodes_vector = np.array([51, 101, 201], dtype=float)
    timestep_vec = 1e09/2 * np.array([1, 1, 1, 1], dtype=float)

    convergence_rate = conduct_tests(t_final, polynomial_degree, integration_method, nodes_vector, timestep_vec)
    PETSc.Sys.Print(f"Convergence rate {round(convergence_rate, 2)} achieved with  DG0.")
    assert convergence_rate >= 0.9, "Optimal convergence rate not achieved."
    PETSc.Sys.Print("")

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing steady-state solution check with DG1")
    PETSc.Sys.Print("="*60)

    t_final, polynomial_degree, integration_method = 1.0e9, 1, BackwardEuler
    nodes_vector = np.array([76, 151, 301], dtype=float)
    timestep_vec = 1e09/2 * np.array([1, 1, 1], dtype=float)

    convergence_rate = conduct_tests(t_final, polynomial_degree, integration_method, nodes_vector, timestep_vec)
    PETSc.Sys.Print(f"Convergence rate {round(convergence_rate, 2)} achieved with  DG1.")
    assert convergence_rate >= 1.9, "Optimal convergence rate not achieved."
    PETSc.Sys.Print("")


def conduct_tests(t_final, 
                  polynomial_degree,
                  integration_method,
                  nodes_vector, 
                  timestep_vec,
                  ):
    
    error_vector = nodes_vector * 0
    convergencerate = nodes_vector * 0

    for index in range(len(nodes_vector)):
        nodes = nodes_vector[index]
        time_step = timestep_vec[index]
        L2_error = compute_error(round(nodes), time_step, t_final, polynomial_degree, integration_method)
        error_vector[index] = L2_error
        if index == 0:
            convergencerate[index] = 0
        else:
            convergencerate[index] = np.log2(error_vector[index-1]/error_vector[index])
    max_convergence_rate = np.max(convergencerate)
    return max_convergence_rate


def compute_error(nodes, 
                  time_step=5000, 
                  t_final=1e05, 
                  polynomial_degree=1,
                  integration_method=BackwardEuler):
    
    """ Runs a Richards simulation for a given mesh resolution and timestep, then computes the L2 error against Tracy's exact solution (2006). 
    Parameters
    ----------
    nodes : int,  Number of cells per spatial dimension. 
    time_step : float,  Time step size.
    t_final : float,  Final simulation time. 
    polynomial_degree : int,  DG polynomial degree. 
    integration_method : object,  Time integration class (BackwardEuler, ImplicitMidpoint, ...). 
       
    Returns
    ------- 
    float L2 error at final time.
    """

    # Set some global parameters
    L = 15.24              # Domain length [m]

    soil_curve = ExponentialCurve(
        theta_r=0.15,  # Residual water content [-]
        theta_s=0.45,  # Saturated water content [-]
        Ks=1.00e-05,   # Saturated hydraulic conductivity [m/s]
        alpha=0.25,    # Fitting parameter [1/m]
        Ss=0.00,       # Specific storage coefficient [1/m]
    )

    alpha = soil_curve.parameters["alpha"]
    hr = -L
    h0 = 1 - exp(alpha*hr)

    dt = Constant(time_step)

    mesh2D = RectangleMesh(nodes, nodes, L, L, quadrilateral=True)
    mesh   = ExtrudedMesh(mesh2D, nodes, layer_height=L/nodes, name="mesh")
    X = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "DQ", polynomial_degree)

    def exact_solution(X, t):

        # Exact solution from Tracy 2006 (https://doi.org/10.1029/2005WR004638)

        beta = sqrt(alpha**2/4 + (pi/L)**2 + (pi/L)**2)
        hss = h0*sin(pi*X[0]/L)*sin(pi*X[1]/L)*exp((alpha/2)*(L - X[2]))*sinh(beta*X[2])/sinh(beta*L)
        c = alpha*(soil_curve.parameters["theta_s"] - soil_curve.parameters["theta_r"])/soil_curve.parameters["Ks"]

        phi = 0
        for k in range(1, 200):
            lambdak = k*pi/L
            gamma = (beta**2 + lambdak**2)/c
            phi = phi + ((-1)**k)*(lambdak/gamma)*sin(lambdak*X[2])*exp(-gamma*t)
        phi = phi*((2*h0)/(L*c))*sin(pi*X[0]/L)*sin(pi*X[1]/L)*exp(alpha*(L-X[2])/2)

        hBar = hss + phi
        hExact = ((1/alpha)*ln(exp(alpha*hr) + hBar))

        return hExact

    offset = 2000
    h = Function(V, name="InitialCondition").interpolate(exact_solution(X, offset))

    # Boundary conditions
    top_bc = (1/alpha)*ln(exp(alpha*hr) + (h0)*(sin(pi*X[0]/L)*sin(pi*X[1]/L)))
    richards_bcs = {
        1: {'h': hr},
        2: {'h': hr},
        3: {'h': hr},
        4: {'h': hr},
        'bottom': {'h': hr},
        'top': {'h': top_bc},
    }

    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=integration_method,
        bcs=richards_bcs,
    )

    time = 0

    dx_quad = dx(metadata={"quadrature_degree": 3})
    
    while time < t_final:

        richards_solver.solve()
        time += float(dt)
        

    # Print outputs
    hExact = exact_solution(X, time+offset)
    L2_norm = sqrt(assemble((h-hExact)**2 * dx_quad))

    PETSc.Sys.Print(f"L2 error:  = {L2_norm:.2e} | "
                    f"dx = {L/(nodes-1):.4f} | "
                    f"dt = {float(dt):.0f} | "
                    f"Nodes = {nodes} | "
                    f"DoF = {V.dim()}"
                    )
    
    return L2_norm

