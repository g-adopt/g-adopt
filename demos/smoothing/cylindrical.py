"""
Test case for smoothing
"""

from gadopt import *

# Start with a previously-initialised temperature field
with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")

# Set up function spaces for the Q2Q1 pair
Q = FunctionSpace(mesh, "CG", 2)  # Pressure function space (scalar)
trial, test = TrialFunction(Q), TestFunction(Q)

solution = Function(Q, name="Solution")
with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    T = f.load_function(mesh, "Temperature")

bcs = DirichletBC(V=Q, g=T, sub_domain=("top", "bottom"))

T_avg = Function(Q, name='Layer_Averaged_Temp')


# Define the center of radial diffusion (e.g., center of the unit square)
center = Constant((0.5, 0.5))

# Define the radial and tangential conductivity values
kr = Constant(0.0)  # Radial conductivity
kt = Constant(1.0)  # Tangential conductivity

# Function to compute radial and tangential components of the conductivity tensor


def conductivity_tensor(x, y):
    # Compute radial vector components
    dx = x
    dy = y
    r = sqrt(dx**2 + dy**2)
    # Unit radial and tangential vectors
    er = as_vector((dx/r, dy/r))
    et = as_vector((-dy/r, dx/r))
    # Construct the anisotropic conductivity tensor
    return kr * outer(er, er) + kt * outer(et, et)


x, y = SpatialCoordinate(mesh)

# Use the tensor in the equation
K = conductivity_tensor(x, y)

# Compute layer average for initial stage:
averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# a = - inner(grad(test), grad(trial)) * dx + test * inner(grad(trial), FacetNormal(mesh)) * ds_tb
del_t = 0.01
a = test * trial * dx + del_t * inner(K * grad(test), grad(trial)) * dx - del_t * test * inner(grad(trial), FacetNormal(mesh)) * ds_tb
L = test * T * dx

solve(a == L, solution, bcs=bcs)
VTKFile("test.pvd").write(T.assign(T - T_avg), solution.assign(solution - T_avg))
