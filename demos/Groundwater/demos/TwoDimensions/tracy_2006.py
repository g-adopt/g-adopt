import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from gadopt import *
from richards_equation import *
from richards_solver import *
from soil_curves import *
import ufl
import time


"""
Comparison to Tracy's two dimensional exact solution
====================================================
Here we compare numerical solutions with the exact solution derived in
    Tracy, 2006, Water Resources Research, Clean two- and three-dimensional solutions of Richards' equation for testing numerical solvers
    https://doi.org/10.1029/2005WR004638
The simulation is performed on a square of side length L = 15.24 metres. Dirichlet boundaries conditions are imposed on the bottom, left and right boundaries $h = -L$. For the top boundary, we have
    $$h(x,z=L,t) = (1/alpha)*ln(exp(alpha*h_r) + h_0*(sin(pi*x/L)))$$
where $\alpha=0.25$, $hr=-L$, and $h_0 =  1 - exp(alpha*h_r)$. For the initial condition, we use $h$ from Tracy's exact solution at $t=2000$. We compute the L2 norm of h_{numerical}-h_{exact}

"""
# Set up mesh
L, nodes = 15.24, 251
mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=True)
x = SpatialCoordinate(mesh)

t_final = 1e05
dt = Constant(5000)  # Time step size
time_integrator = "CrankNicolson"

V = FunctionSpace(mesh, "DQ", 2)  # Function space for pressure head
PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

# Gather info about mesh
eq = richards_equation(V, mesh, 2*V.ufl_element().degree()+1)  

# Specify the hydrological parameters
soil_curve = ExponentialCurve(
    theta_r=0.15,   # Residual water content [-]
    theta_s=0.45,   # Saturated water content [-]
    Ks=1e-05,       # Saturated hydraulic conductivity [m/s]
    alpha=0.25,     # Fitting parameter [1/m]
    Ss=0,           # Specific storage coefficient [1/m]
)

alpha = soil_curve.parameters["alpha"]
hr = -L
h0 = 1 - exp(alpha*hr)

# Set up boundary conditions
boundary_ids = get_boundary_ids(mesh)
top_bc = (1/alpha)*ln(exp(alpha*hr) + (h0)*(sin(pi*x[0]/L)))
richards_bcs = {
    boundary_ids.left: {'h': hr},
    boundary_ids.right: {'h': hr},
    boundary_ids.bottom: {'h': hr},
    boundary_ids.top: {'h': top_bc},
}


def exact_solution(x, t):

    # Exact solution from Tracy 2006 (https://doi.org/10.1029/2005WR004638, page 4)

    beta = sqrt(alpha**2/4 + (pi/L)**2)
    hss = h0*sin(pi*x[0]/L)*exp((alpha/2)*(L - x[1]))*sinh(beta*x[1])/sinh(beta*L)
    c = alpha*(soil_curve.parameters["theta_s"] - soil_curve.parameters["theta_r"])/soil_curve.parameters["Ks"]

    phi = 0
    for k in range(1, 200):
        lambdak = k*pi/L
        gamma = (beta**2 + lambdak**2)/c
        phi = phi + ((-1)**k)*(lambdak/gamma)*sin(lambdak*x[1])*exp(-gamma*t)
    phi = phi*((2*h0)/(L*c))*sin(pi*x[0]/L)*exp(alpha*(L-x[1])/2)

    hBar = hss + phi

    hExact = ((1/alpha)*ln(exp(alpha*hr) + hBar))

    return hExact


offset = 2000
h_initial = Function(V, name="InitialCondition")
h_initial.interpolate(exact_solution(x, offset))

h = Function(V, name="PressureHead").assign(h_initial)
h_old = Function(V, name="PreviousSolution").assign(h_initial)

time = 0.0
time_var = Constant(0)

# Initialise the solver
richards_solver = richardsSolver(
    h,
    h_old,
    time=time_var,
    time_step=dt,
    time_integrator=time_integrator,
    eq=eq,
    soil_curves=soil_curve,
    bcs=richards_bcs
)

while time < t_final:

    # Solve equation
    time_var.assign(time)
    h_old.assign(h)
    richards_solver.solve()
    time += float(dt)

# Compute L2 norm of error
hExact = exact_solution(x, t_final+offset)
print("L2 error: ", assemble(sqrt(dot((h - hExact), (h - hExact)))*eq.dx))

# Save file
h_file = VTKFile("tracy_2006.pvd")
h_file.write(h)
