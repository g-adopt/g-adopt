import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

from gadopt import *
from richards_equation import *
from richards_solver import *
from soil_curves import *
import random

"""
Recharge of a two-dimensional water table
=========================================
Here we reproduce the test case presented in:
    Vauclin, Khanju, and Vachaud, Water Resources Research, 1979
    Experimental and numerical study of a transient, two-dimensional unsaturated-saturated water table recharge problem
    https://doi.org/10.1029/WR015i005p01089
The simulation is performed in a domain of 3 x 2 metres, and the initial
condition is chosen such that the region z <= 0.65 m is fully satured ($\theta =
\theta_s$), $h(t=0) = z - 0.65$. For the boundary conditions, the bottom and left boundary are no flux ($q \cdot n = 0$), the right boundary fixed the height of the water table ($h = z - 0.65$ m). For the top boundary, water is injected at a rate of 14.8 cm/hour  in the region where x <= 0.5 m and 0 otherwise. The simulation is concluded after 8 hours
"""

# Set up mesh
nodesX, nodesZ = 91, 61
mesh = RectangleMesh(nodesX, nodesZ, 300, 200, name="mesh", quadrilateral=True)
x = SpatialCoordinate(mesh)

t_final = 32400.0    # 8 hours (in seconds)
dt = Constant(10)  # Time step size
time_integrator = "SemiImplicit"

degree = 2
V = FunctionSpace(mesh, "DQ", degree)  # Function space for pressure head
W = VectorFunctionSpace(mesh, "DQ", degree)  # Function space for volumetric flux (doesn't influence solution)
PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

eq = richards_equation(V, mesh, 5)

# Specify the hydrological parameters
soil_curve = HaverkampCurve(
    theta_r=0.00,   # Residual water content [-]
    theta_s=0.37,   # Saturated water content [-]
    Ks=9.722e-03,   # Saturated hydraulic conductivity [cm/s]
    alpha=40000,    # Fitting parameter [cm]
    beta=2.90,      # Fitting parameter [-]
    A=2.99e06,      # Fitting parameter [cm]
    gamma=5.0,      # Fitting parameter [-]
    Ss=1e-05,       # Specific storage coefficient [1/cm]
)

# Set up boundary conditions
boundary_ids = get_boundary_ids(mesh)
time_var = Constant(0.0)
top_flux = tanh(0.000125 * time_var) * 4.11e-03 * (
    0.5 * (1 + tanh(25 * (x[0] + 50)))
    - 0.5 * (1 + tanh(25 * (x[0] - 50)))
)

richards_bcs = {
    boundary_ids.left: {'flux': 0.0},
    boundary_ids.right: {'h': 65 - x[1]},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top: {'flux': top_flux},
}

# Initial condition
h_initial = Function(V, name="InitialCondition")
h_initial.interpolate(-x[1]+65)

h = Function(V, name="PressureHead")
h_old = Function(V, name="PreviousSolution")
h.assign(h_initial)
h_old.assign(h_initial)

K = Function(V, name="HydraulicConductivity")
theta = Function(V, name="MoistureContent")
q = Function(W, name='VolumetricFlux')

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

richards_solver_linear = richardsSolver(
    h,
    h_old,
    time=time_var,
    time_step=dt,
    time_integrator="SemiImplicit",
    eq=eq,
    soil_curves=soil_curve,
    bcs=richards_bcs
)

time = 0.0
exterior_flux = 0

theta.interpolate(soil_curve.moisture_content(h))
initial_mass = assemble(theta*eq.dx)

while time < t_final:

    # Solve equation
    time_var.assign(time)
    h_old.assign(h)
    if time_integrator != "SemiImplicit":
        richards_solver_linear.solve()
    richards_solver.solve()
    time += float(dt)

    # Update volumetric flux
    if time_integrator == "BackwardEuler":
        K.interpolate(soil_curve.relative_permeability(h))
        q.interpolate(project(as_vector(-K*grad(h + x[eq.dimen-1])), W))
    elif time_integrator == "SemiImplicit":
        K.interpolate(soil_curve.relative_permeability(h_old))
        q.interpolate(project(as_vector(-K*grad(h + x[eq.dimen-1])), W))
    elif time_integrator == "ImplicitMidpoint":
        K.interpolate(soil_curve.relative_permeability((h+h_old)/2))
        q.interpolate(project(as_vector(-K*grad(((h+h_old)/2) + x[eq.dimen-1])), W))
    else:
        K.interpolate((soil_curve.relative_permeability(h)+soil_curve.relative_permeability(h_old))/2)
        q.interpolate(project(as_vector(-K*grad(((h+h_old)/2) + x[eq.dimen-1])), W))

    # Water entering/leaving domain
    exterior_flux += assemble(dt*dot(q, -eq.n)*eq.ds)

theta.interpolate(soil_curve.moisture_content(h))
final_mass = assemble(theta*eq.dx)

PETSc.Sys.Print(f"Initial mass: {initial_mass}")
PETSc.Sys.Print(f"Final mass: {final_mass}")
PETSc.Sys.Print(f"Exterior flux: {exterior_flux}")

# Save file
h_file = VTKFile("vauclin_1979.pvd")
h_file.write(h, theta, q)

with CheckpointFile(f"{nodesX}x{nodesZ}.h5", 'w') as afile:
    afile.save_mesh(mesh) # Save the mesh (optional, but recommended)
    afile.save_function(h) # Save the function
