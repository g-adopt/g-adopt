from gadopt import *

"""
Convergence study of Tracy's two dimensional exact solution
====================================================
"""

def main():

    dg1()

    backward_euler()

    implicit_midpoint()


def dg1():

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing steady state solution check with DG1")
    PETSc.Sys.Print("="*60)

    t_final = 2.5e06
    polynomial_degree = 1
    time_step = 5000

    nodes_vector = np.array([51, 101, 201, 401], dtype=float)
    error_vector = nodes_vector * 0
    convergencerate = nodes_vector * 0

    for index in range(len(nodes_vector)):
        nodes = nodes_vector[index]
        L2_error = compute_error(round(nodes), time_step, t_final, polynomial_degree)
        error_vector[index] = L2_error
        if index == 0:
            convergencerate[index] = 0
        else:
            convergencerate[index] = np.log2(error_vector[index-1]/error_vector[index])
    max_convergence_rate = np.max(convergencerate)
    PETSc.Sys.Print(f"Convergence rate {round(max_convergence_rate, 2)} achieved with DG1.")

    PETSc.Sys.Print("")


def backward_euler():

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing transient solution check with Backward Euler")
    PETSc.Sys.Print("="*60)

    t_final = 1e05
    nodes = 201
    polynomial_degree = 2
    integration_method = BackwardEuler

    timestep_vec = np.array([5000, 2500, 1250, 625], dtype=float)
    error_vector = timestep_vec * 0
    convergencerate = timestep_vec * 0

    for index in range(len(timestep_vec)):
        time_step = timestep_vec[index]
        L2_error = compute_error(nodes, time_step, t_final, polynomial_degree, integration_method)
        error_vector[index] = L2_error
        if index == 0:
            convergencerate[index] = 0
        else:
            convergencerate[index] = np.log2(error_vector[index-1]/error_vector[index])
    max_convergence_rate = np.max(convergencerate)
    PETSc.Sys.Print(f"Convergence rate {round(max_convergence_rate, 2)} achieved with Backward Euler.")

    PETSc.Sys.Print("")


def implicit_midpoint():

    PETSc.Sys.Print("="*60)
    PETSc.Sys.Print("Performing transient solution check with Implicit Midpoint")
    PETSc.Sys.Print("="*60)

    t_final = 1e05
    nodes = 301
    polynomial_degree = 2
    integration_method = ImplicitMidpoint

    timestep_vec = np.array([5000, 2500, 1250, 625], dtype=float)
    error_vector = timestep_vec * 0
    convergencerate = timestep_vec * 0

    for index in range(len(timestep_vec)):
        time_step = timestep_vec[index]
        L2_error = compute_error(nodes, time_step, t_final, polynomial_degree, integration_method)
        error_vector[index] = L2_error
        if index == 0:
            convergencerate[index] = 0
        else:
            convergencerate[index] = np.log2(error_vector[index-1]/error_vector[index])
    max_convergence_rate = np.max(convergencerate)
    PETSc.Sys.Print(f"Convergence rate {round(max_convergence_rate, 2)} achieved with Implicit Midpoint.")

    PETSc.Sys.Print("")


def compute_error(nodes, 
                  time_step=5000, 
                  t_final=1e05, 
                  polynomial_degree=1,
                  integration_method=BackwardEuler):

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

    mesh = RectangleMesh(round(nodes), round(nodes), L, L, name="mesh", quadrilateral=True)
    X = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "DQ", polynomial_degree)

    def exact_solution(X, t):

        # Exact solution from Tracy 2006 (https://doi.org/10.1029/2005WR004638, page 4)

        beta = sqrt(alpha**2/4 + (pi/L)**2)
        hss = h0*sin(pi*X[0]/L)*exp((alpha/2)*(L - X[1]))*sinh(beta*X[1])/sinh(beta*L)
        c = alpha*(soil_curve.parameters["theta_s"] - soil_curve.parameters["theta_r"])/soil_curve.parameters["Ks"]

        phi = 0
        for k in range(1, 200):
            lambdak = k*pi/L
            gamma = (beta**2 + lambdak**2)/c
            phi = phi + ((-1)**k)*(lambdak/gamma)*sin(lambdak*X[1])*exp(-gamma*t)
        phi = phi*((2*h0)/(L*c))*sin(pi*X[0]/L)*exp(alpha*(L-X[1])/2)

        hBar = hss + phi

        hExact = ((1/alpha)*ln(exp(alpha*hr) + hBar))

        return hExact

    offset = 2000
    h = Function(V, name="InitialCondition").interpolate(exact_solution(X, offset))

    # Boundary conditions
    boundary_ids = get_boundary_ids(mesh)
    top_bc = (1/alpha)*ln(exp(alpha*hr) + (h0)*(sin(pi*X[0]/L)))
    richards_bcs = {
        boundary_ids.left: {'h': hr},
        boundary_ids.right: {'h': hr},
        boundary_ids.bottom: {'h': hr},
        boundary_ids.top: {'h': top_bc},
    }

    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=integration_method,
        bcs=richards_bcs,
    )

    time = 0
    snes_iterations = 0

    dx_quad = dx(metadata={"quadrature_degree": 3})

    outfile = VTKFile("output.pvd")
    
    while time < t_final:

        #outfile.write(h)

        richards_solver.solve()
        time += float(dt)

        snes = richards_solver.ts.stepper.solver.snes
        nonlinear_iterations = snes.getIterationNumber()
        #print(nonlinear_iterations)
        #if nonlinear_iterations < 3:
        #    dt.assign(1.01*dt)

        

    # Print outputs
    hExact = exact_solution(X, time+offset)
    L2_norm = sqrt(assemble((h-hExact)**2 * dx_quad))

    PETSc.Sys.Print(f"L2 error:  = {L2_norm:.6e} | "
                    f"dx = {L/(nodes-1):.4f} | "
                    f"dt = {float(dt):.0f} | "
                    f"Nodes = {nodes} | "
                    f"DoF = {V.dim()}"
                    )
    
    return L2_norm

if __name__ == "__main__":
    main()
