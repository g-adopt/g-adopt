import os, sys
sys.path.insert(0, '../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from gadopt import *
from firedrake.__future__ import interpolate
from utilities import *
from richards_equation import *
from richards_solver import *
from soil_curves import *
from surface_mesh import *

"""
A case study of the Lower Murrumbidgee basin
=================================================================
"""

# Function space of horizontal and vertical
horiz_elt = FiniteElement("DG", triangle, 1)
vert_elt = FiniteElement("DG", interval, 1)
horizontal_resolution = 3000
number_layers = 300

# Time stepping details
time_var = Constant(0.0)
t_final = 30*52*7*86400  # (in seconds)
dt = Constant(7*86400)  # Initial timestep
time_integrator = "BackwardEuler"
PETSc.Sys.Print(time_integrator)

# Details of mesh extrusion
surface_mesh(horizontal_resolution)
layer_height = 1/number_layers
mesh2D = Mesh('MurrumbidgeeMeshSurface.msh')
mesh = ExtrudedMesh(mesh2D, number_layers, layer_height=layer_height, extrusion_type='uniform', name='mesh')

W = VectorFunctionSpace(mesh, 'CG', 1)
X = assemble(interpolate(mesh.coordinates, W))
mesh_coords = X.dat.data_ro

# Transform the z coordinate such that top and bottom are the points given by elevation_data.csv and bedrock_data.csv
z = mesh_coords[:, 2]
bedrock = data_2_function(mesh_coords, 'bedrock_data.csv')
elevation = data_2_function(mesh_coords, 'elevation_data.csv')
mesh.coordinates.dat.data[:, 2] = bedrock*z + elevation - bedrock

X = assemble(interpolate(mesh.coordinates, W))
mesh_coords = X.dat.data_ro

Vcg = FunctionSpace(mesh, 'CG', 1)
elt = TensorProductElement(horiz_elt, vert_elt)
Vdq = FunctionSpace(mesh, elt)
Wdq = VectorFunctionSpace(mesh, elt)
x = SpatialCoordinate(mesh)

PETSc.Sys.Print("The number of degrees of freedom is:", Vdq.dim())
PETSc.Sys.Print("Horizontal resolution:", horizontal_resolution)
PETSc.Sys.Print("Number of layers:", number_layers)

# Construct some functions from external data
elevation_cg = Function(Vcg)
elevation_cg.dat.data[:] = data_2_function(mesh_coords, 'elevation_data.csv')
elevation = Function(Vdq, name='elevation').interpolate(elevation_cg)

shallowLayer_cg = Function(Vcg)
shallowLayer_cg.dat.data[:] = data_2_function(mesh_coords, 'shallow_layer.csv')
shallowLayer = Function(Vdq, name='ShallowLayer').interpolate(shallowLayer_cg)

lowerLayer_cg = Function(Vcg)
lowerLayer_cg.dat.data[:] = data_2_function(mesh_coords, 'lower_layer.csv')
lowerLayer = Function(Vdq, name='LowerLayer').interpolate(lowerLayer_cg)

bedrock_cg = Function(Vcg)
bedrock_cg.dat.data[:] = data_2_function(mesh_coords, 'bedrock_data.csv')
bedrock = Function(Vdq, name='Bedrock').interpolate(bedrock_cg)

watertable_cg = Function(Vcg)
watertable_cg.dat.data[:] = data_2_function(mesh_coords, 'water_table.csv')
watertable = Function(Vdq, name='WaterTable').interpolate(watertable_cg)

rainfall_cg = Function(Vcg)
rainfall_cg.dat.data[:] = data_2_function(mesh_coords, 'rainfall_data.csv')
rainfall = Function(Vdq, name='Rainfall').interpolate(watertable_cg)

depth = Function(Vdq, name='depth').interpolate(elevation - x[2])

# Indicator functions of where each layer is
delta = 1
I1 = 0.5*(1 + tanh(delta*(shallowLayer - depth)))
I2 = 0.5*(1 + tanh(delta*(lowerLayer - depth)))
I3 = 0.5*(1 + tanh(delta*(bedrock - depth)))

S_depth = 1/((1 + 0.000071*depth)**5.989)  # Depth dependent porosity
K_depth = (1 - depth / (58 + 1.02*depth))**3  # Depth dependent conductivity
Ks = Function(Vdq, name='SaturatedConductivity')
Ks.interpolate(K_depth*(2.5e-05*I1 + 1e-03*(1 - I1)*I2 + 5e-04*(1-I2)))

# Specify the hydrological parameters
soil_curve = HaverkampCurve(
    theta_r=0.025,         # Residual water content [-]
    theta_s=0.40*S_depth,  # Saturated water content [-]
    Ks=Ks,                 # Saturated hydraulic conductivity [m/s]
    alpha=0.44,
    beta=1.2924,
    A=0.0104,
    gamma=1.5722,
    Ss=0,              # Specific storage coefficient [1/m]
)

h         = Function(Vdq, name="PressureHead").interpolate(depth - watertable)
h_old     = Function(Vdq, name="PressureHead").assign(h)
h_initial = Function(Vdq, name="InitialPressureHead").assign(h)
h_star    = Function(Vdq, name="SolutionGuess").assign(h)
theta     = Function(Vdq, name="MoistureContent").interpolate(soil_curve.
moisture_content(h))
K         = Function(Vdq, name='RelativeConductivity')
q         = Function(Wdq, name='VolumetricFlux')

# Extraction points
spread = 50000000
Ind = 0
xPts = np.array([1.7e05, 2.2e05, 2.4e05, 2.0e05, 1.6e05, 1.7e05, 2.2e05, 2.5e05, 2.0e05, 1.5e05, 2.3e05, 1.0e05, 2.0e05, 1.9e05, 1.9e05, 1.9e05, 1.6e05, 2.6e05, 1.2e05, 2.5e05])
yPts = np.array([8.0e04, 4.3e04, 4.2e04, 7.3e04, 3.5e04, 9.3e04, 6.0e04, 6.5e04, 6.0e04, 5.0e04, 9.0e04, 6.5e04, 2.0e04, 1.0e05, 8.5e04, 4.4e04, 6.0e04, 5.5e04, 6.5e04, 2.2e04])
for ii in range(20):
    Ind = Ind + exp((-(x[0]-xPts[ii])**2-(x[1]-yPts[ii])**2)/spread)

boundary_ids = get_boundary_ids(mesh)
richards_bcs = {
    1: {'h': depth - watertable},
    boundary_ids.bottom: {'flux': 3e-9*Ind},
    boundary_ids.top: {'flux': 0.14*3.171e-11*rainfall},
}

eq = richards_equation(Vdq, mesh, 5)
richards_solver = richardsSolver(
    h,
    h_old,
    h_star,
    time=time_var,
    time_step=dt,
    time_integrator=time_integrator,
    eq=eq,
    soil_curves=soil_curve,
    bcs=richards_bcs
)

current_time = 0

time_steps = 0
nonlinear_iterations = 0
linear_iterations = 0

while current_time < t_final:

    # Solve equation
    time_var.assign(current_time)
    h_old.assign(h)

    if time_integrator == 'Picard':

        pressure_head_error = np.inf
        moisture_content_error = 0
        picard_iterations = 0
        picard_linear = 0
        while (pressure_head_error > 1e-0):
            h_star.assign(h)
            richards_solver.solve()

            snes = richards_solver.snes
            pressure_head_error = norm(h - h_star, "L2")
            PETSc.Sys.Print(pressure_head_error)
            picard_iterations += 1
            picard_linear += snes.getLinearSolveIterations()

    else:

        richards_solver.solve()
        snes = richards_solver.snes

#   h_star.assign(h)
#    richards_solver.solve()


    time_steps           += 1
    if time_integrator == 'Picard':
        nonlinear_iterations += picard_iterations
        linear_iterations += picard_linear
    else:
        nonlinear_iterations += snes.getIterationNumber()
        linear_iterations    += snes.getLinearSolveIterations()

    theta.interpolate(soil_curve.moisture_content(h))

    PETSc.Sys.Print(f'Current time: {float(current_time)/86400} days')
    #PETSc.Sys.Print(f'Total water volume: {assemble(theta*eq.dx)}')
    PETSc.Sys.Print(f'External flux: {assemble(dt*dot(q, -eq.n)*eq.ds)}')
    PETSc.Sys.Print(f'Time step size: {float(dt)}')
    if time_integrator == 'Picard':
        PETSc.Sys.Print(f"Nonlinear Iterations: {picard_iterations}")
        PETSc.Sys.Print(f"Total Linear (KSP) Iterations: {picard_linear}")
    else:
        PETSc.Sys.Print(f"Nonlinear Iterations: {snes.getIterationNumber()}")
        PETSc.Sys.Print(f"Total Linear (KSP) Iterations: {snes.getLinearSolveIterations()}")
    PETSc.Sys.Print()

    current_time += float(dt)



PETSc.Sys.Print(f'Number of time steps taken: {time_steps}')
PETSc.Sys.Print(f"Total Linear (KSP) Iterations: {linear_iterations}")
PETSc.Sys.Print(f"Nonlinear (Newton) Iterations: {nonlinear_iterations}")

K.interpolate(soil_curve.relative_permeability((h+h_old)/2))
q.interpolate(project(as_vector(-K*grad((h+h_old)/2 + x[eq.dimen-1])), Wdq)/theta)

with CheckpointFile("DG10_dx=3000_layers=300.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(h)
    afile.save_function(theta)
    afile.save_function(q)
    afile.save_function(depth)
