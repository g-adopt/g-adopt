from gadopt import *

"""
Comparison to Tracy's two dimensional exact solution
====================================================
Here we compare numerical solutions with the exact solution derived in
    Tracy, 2006, Water Resources Research, Clean two- and three-dimensional solutions of Richards' equation for testing numerical solvers
    https://doi.org/10.1029/2005WR004638
The simulation is performed on a cube of side length L = 15.24 metres. Dirichlet boundaries conditions are imposed on the bottom, left and right boundaries $h = -L$. For the top boundary, we have
    $$h(x,z=L,t) = (1/alpha)*ln(exp(alpha*h_r) + h_0*(sin(pi*x/L)))$$
where $\alpha=0.25$, $hr=-L$, and $h_0 =  1 - exp(alpha*h_r)$. For the initial condition, we use $h$ from Tracy's exact solution at $t=2000$. We compute the L2 norm of h_{numerical}-h_{exact}

"""

L = 15.24    # Domain length [m]
nodes = 26  # Number of grid points in each direction

dt = Constant(10000)
t_final = 1e05

mesh2D = RectangleMesh(nodes, nodes, L, L, quadrilateral=True)
mesh = ExtrudedMesh(mesh2D, nodes, layer_height=L/nodes)
X = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DQ", 2)

soil_curve = ExponentialCurve(
    theta_r=0.15,  # Residual water content [-]
    theta_s=0.45,  # Saturated water content [-]
    Ks=1.00e-05,   # Saturated hydraulic conductivity [m/s]
    alpha=0.25,    # Fitting parameter [1/m]
    Ss=0.00,       # Specific storage coefficient [1/m]
)

moisture_content = soil_curve.moisture_content
relative_permeability = soil_curve.relative_permeability

alpha = soil_curve.parameters["alpha"]
hr = -L
h0 = 1 - exp(alpha*hr)


def exact_solution(x, t):

    # Exact solution from Tracy 2006
    h0 = 1 - exp(alpha * hr)
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
boundary_ids = get_boundary_ids(mesh)
top_bc = (1/alpha)*ln(exp(alpha*hr) + (h0)*(sin(pi*X[0]/L)*sin(pi*X[1]/L)))
richards_bcs = {
    boundary_ids.left: {'h': -L},
    boundary_ids.right: {'h': -L},
    boundary_ids.back: {'h': -L},
    boundary_ids.front: {'h': -L},
    boundary_ids.bottom: {'h': -L},
    boundary_ids.top: {'h': top_bc},
}

richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=ImplicitMidpoint,
    bcs=richards_bcs,
    solver_parameters="iterative",
    quad_degree=5,
)

time = 0
snes_iterations = 0
while time < t_final:

    richards_solver.solve()
    time += float(dt)


# Compute L2 norm of error
hExact = exact_solution(X, t_final+offset)
print("L2 error: ", assemble(sqrt(dot((h - hExact), (h - hExact)))*dx))