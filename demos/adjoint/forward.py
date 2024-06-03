"""
This runs the forward portion of the adjoint test case, to generate the reference
final condition, and synthetic forcing (surface velocity observations).
"""

from gadopt import *
import numpy as np

# Domain extents
x_max = 1.0
y_max = 1.0

# Number of intervals along x direction
disc_n = 150

# Interval mesh in x direction, to be extruded along y
mesh1d = IntervalMesh(disc_n, length_or_left=0.0, right=x_max)
mesh = ExtrudedMesh(
    mesh1d, layers=disc_n, layer_height=y_max / disc_n, extrusion_type="uniform"
)

# write out the mesh
with CheckpointFile("mesh.h5", mode="w") as fi:
    fi.save_mesh(mesh)

bottom_id, top_id, left_id, right_id = "bottom", "top", 1, 2

# Set up function spaces for the Q2Q1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Z = MixedFunctionSpace([V, W])

z = Function(Z)  # A field over the mixed function space Z
u, p = z.subfunctions  # Symbolic UFL expressions for u and p
u.rename("Velocity")
p.rename("Pressure")

T = Function(Q, name="Temperature")
X = SpatialCoordinate(mesh)
T.interpolate(
    0.5 * (erf((1 - X[1]) * 3.0) + erf(-X[1] * 3.0) + 1) +
    0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2))) / Constant(0.1)) ** 2)
)

Taverage = Function(Q1, name="Average Temperature")

# Calculate the layer average of the initial state
averager = LayerAveraging(
    mesh, np.linspace(0, 1.0, 150 * 2), cartesian=True, quad_degree=6
)
averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)
checkpoint_file.save_function(Taverage, name="Average Temperature", idx=0)
checkpoint_file.save_function(T, name="Temperature", idx=0)

Ra = Constant(1e6)  # Rayleigh number
approximation = BoussinesqApproximation(Ra, cartesian=True)

delta_t = Constant(4e-6)  # Constant time step
max_timesteps = 80

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Free-slip velocity boundary condition on all sides
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"uy": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
temp_bcs = {
    bottom_id: {"T": 1.0},
    top_id: {"T": 0.0},
}

energy_solver = EnergySolver(
    T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
)

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    constant_jacobian=True,
)

# Create output file and select output_frequency
output_file = VTKFile("vtu-files/output.pvd")
dump_period = 10

# Now perform the time loop:
for timestep in range(0, max_timesteps):
    stokes_solver.solve()
    energy_solver.solve()

    # Storing velocity to be used in the objective F
    checkpoint_file.save_function(u, name="Velocity", idx=timestep)

    if timestep % dump_period == 0 or timestep == max_timesteps - 1:
        output_file.write(u, p, T)

# Save the reference final temperature
checkpoint_file.save_function(T, name="Temperature", idx=max_timesteps - 1)
checkpoint_file.close()
