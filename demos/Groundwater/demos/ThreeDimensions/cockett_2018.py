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
Three-dimensional infultration of water into a heterogeneous soil
=================================================================
Here we present an example of the infultration into a heterogeneous column of soil composed of a mixture of sand and loamy sand, as described by
    Cockett, Heagy, and Haber, Computers and Geosciences, 2018, Efficient 3D inversions using the Richards equation
    https://doi.org/10.1016/j.cageo.2018.04.006
Simulations are formed in a rectangular prism of side length 2.0 x 2.0 x 2.6 m. No flux is imposed on all the boundaries except the top where $h = -0.1$ m.
"""

# Set up mesh
nodesX = 10
nodesY, nodesZ = nodesX, round(1.3*nodesX)
Lx, Ly, Lz = 2, 2, 2.6

mesh2D = RectangleMesh(nodesX, nodesY, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(mesh2D, nodesZ, layer_height=Lz/nodesZ)
x = SpatialCoordinate(mesh)

dt = Constant(300.0)  # Time step size
t_final = float(dt)*250
time_integrator = "CrankNicolson"

V = FunctionSpace(mesh, "DQ", 2)

PETSc.Sys.Print("The time step is:", dt)
PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

# Construct the heterogeneuous soil
epsilon = 1/500
r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837, 0.1321, 0.7227, 0.1104, 0.1175, 0.6407]
I = sin(3*(x[0]-r[0])) + sin(3*(x[1]-r[1])) + sin(3*(x[2]-r[2])) + sin(3*(x[0]-r[3])) + sin(3*(x[1]-r[4])) + sin(3*(x[2]-r[5]))+sin(3*(x[0]-r[6])) + sin(3*(x[1]-r[7])) + sin(3*(x[2]-r[8]))
I = 0.5*(1 + tanh(I/epsilon))

eq = richards_equation(V, mesh, 5)  # Gather info about mesh

# Specify the hydrological parameters
soil_curve = VanGenuchtenCurve(
    theta_r=0.02*I + 0.035*(1-I),    # Residual water content [-]
    theta_s=0.417*I + 0.401*(1-I),   # Saturated water content [-]
    Ks=5.82e-05*I + 1.69e-05*(1-I),  # Saturated hydraulic conductivity [m/s]
    alpha=13.8*I + 11.5*(1-I),       # Related to inverse of air entry [1/m]
    n=1.592*I + 1.474*(1-I),         # Measure of pore distribution [-]
    Ss=0,                            # Specific storage coefficient [1/m]
)

# Set up boundary conditions
boundary_ids = get_boundary_ids(mesh)
top_bc = -0.1
bottom_bc = -0.3
richards_bcs = {
    boundary_ids.left: {'flux': 0},
    boundary_ids.right: {'flux': 0},
    boundary_ids.back: {'flux': 0},
    boundary_ids.front: {'flux': 0},
    boundary_ids.bottom: {'h': bottom_bc},
    boundary_ids.top: {'h': top_bc},
}

# Initial condition
h_initial = Function(V, name="InitialCondition")
h_initial.interpolate(bottom_bc + 0.2*exp(5*(x[2]-Lz)))

h = Function(V, name="PressureHead").assign(h_initial)
h_old = Function(V, name="PreviousSolution").assign(h_initial)
h.assign(h_initial)
h_old.assign(h_initial)

K = Function(V, name="HydraulicConductivity")
K.interpolate(soil_curve.relative_permeability(h))
iohutfy
current_time = 0.0
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

start = time.time()
while current_time < t_final:

    # Solve equation
    time_var.assign(current_time)
    h_old.assign(h)
    richards_solver.solve()
    current_time += float(dt)
end = time.time()
PETSc.Sys.Print(f"Simulation time: {end - start}")


# Save file
h_file = VTKFile("cockett_2018.pvd")
h_file.write(h)