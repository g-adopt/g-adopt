# Spontaneous subduction
# ======================

# This example
# ------------

# Here, we consider the spontaneous subduction benchmark presented in [Schmeling et al.
# (2008)](https://).
# Inside a 2-D domain, a lithosphere sinks into the upper mantle under its own negative
# buoyancy. Here we consider the scenario where the top boundary is modelled as a free
# surface. We describe below how to implement this problem using G-ADOPT.

# As with all examples, the first step is to import the `gadopt` package, which
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
nx, ny = 256, 64  # Number of cells in x and y directions
lx, ly = 3e6, 7e5  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W, W])  # Stokes function space (mixed)
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space for time step

z = Function(Z)  # A field over the mixed function space Z
u, p, eta = split(z)  # Indexed expressions for velocity, pressure, and free surface
z.subfunctions[0].rename("Velocity")  # Associated Firedrake velocity function
z.subfunctions[1].rename("Pressure")  # Associated Firedrake pressure function
z.subfunctions[2].rename("Free surface")  # Associated Firedrake pressure function
psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now provide initial conditions for the level-set field. To this end, we use the
# `shapely` library to represent the initial location of the material interface and
# derive the signed-distance function. Finally, we apply the transformation to obtain a
# smooth step function profile.

# +
import shapely as sl  # noqa: E402
from mpi4py import MPI  # noqa: E402

# Shapely Polygon representing the initial lithosphere
slab = sl.Polygon(
    [
        (1e6, ly),
        (1e6, 5e5),
        (1.1e6, 5e5),
        (1.1e6, 6e5),
        (lx + 1e5, 6e5),
        (lx + 1e5, ly),
        (1e6, ly),
    ]
)
sl.prepare(slab)
# Shapely LineString representing the domain's free surface
free_surface = sl.LineString([(0, ly), (lx, ly)])
sl.prepare(free_surface)
# Shapely union of the lithosphere boundary and free surface
material_interface = sl.union(free_surface, slab.boundary)

# Determine to which material nodes belong and calculate distance to interface
node_relation_to_interface = [
    (slab.contains(sl.Point(x, y)), material_interface.distance(sl.Point(x, y)))
    for x, y in node_coordinates(psi)
]

# Define the signed-distance function and overwrite its value array
signed_dist_to_interface = Function(K)
signed_dist_to_interface.dat.data[:] = [
    dist if is_inside else -dist for is_inside, dist in node_relation_to_interface
]

# Define thickness of the hyperbolic tangent profile
local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

# Initialise level set as a smooth step function
psi.interpolate((1 + tanh(signed_dist_to_interface / 2 / epsilon)) / 2)
# -

# We next define the material fields and instantiate the approximation. Here, the system
# of equations is non-dimensional and only includes compositional buoyancy under the
# Boussinesq approximation. Moreover, physical parameters are constant through space
# apart from density. As a result, the system is fully defined by the compositional
# Rayleigh number. We use the `material_field` function to define its value throughout
# the domain (including the shape of the material interface transition) and provide it
# to our approximation, alongside other parameters already mentioned.

# +
# Material fields defined based on each material value and location
mu_slab = 1e23
mu_mantle = 1e21
mu = material_field(psi, [mu_mantle, mu_slab], interface="geometric")

rho_slab = 3300
rho_mantle = 3200
rho_material = material_field(psi, [rho_mantle, rho_slab], interface="sharp")

approximation = Approximation(
    "BA",
    dimensional=True,
    parameters={"g": 9.81, "mu": mu, "rho": rho_mantle, "rho_material": rho_material},
)
# -

# As with the previous examples, we set up an instance of the `TimestepAdaptor` class
# for controlling the time-step length (via a CFL criterion) whilst the simulation
# advances in time. We specify the initial time, initial time step $\Delta t$, and
# output frequency (in time units).

time_now = 0  # Initial time
delta_t = Function(R).assign(1e11)  # Initial time step
# Frequency (based on simulation time) at which to output
output_frequency = 8e5 * 365.25 * 8.64e4
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that should not exceed 0.6.

# Boundary conditions are specified next: a free surface at the top and free slip at all
# other boundaries. No boundary conditions are required for level set, as the numerical
# domain is closed.

stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"free_surface": {"eta_index": 0, "rho_ext": 0}},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

# We now set up our output. To do so, we create the output file as a ParaView Data file
# that uses the XML-based VTK file format. We also open a file for logging, instantiate
# G-ADOPT geodynamical diagnostic utility, and define some parameters specific to this
# problem.

# +
output_file = VTKFile("output.pvd")

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt slab_tip_depth")
# -

# Here, we set up the variational problem for the Stokes and level-set systems. The
# former depends on the approximation defined above, and the latter includes both
# advection and reinitialisation components. Subcycling is available for level-set
# advection and is mainly useful when the problem at hand involves multiple CFL
# conditions, with the CFL for level-set advection being the most restrictive.

# +
stokes_solver = StokesSolver(z, approximation, bcs=stokes_bcs, timestep_fs=delta_t)
stokes_solver.solve()

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

# +
from gadopt.level_set_tools import min_max_height  # noqa: E402

step = 0  # A counter to keep track of looping
output_counter = 0  # A counter to keep track of outputting
time_end = 6e7 * 365.25 * 8.64e4
while True:
    # Write output
    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, psi, time=time_now / 1e6 / 365.25 / 8.64e4)
        output_counter += 1

    # Update timestep
    if time_end is not None:
        t_adapt.maximum_timestep = min(output_frequency, time_end - time_now)
    t_adapt.update_timestep()

    # Advect level set
    level_set_solver.solve(step)

    # Solve Stokes sytem
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    # Log diagnostics
    plog.log_str(
        f"{step} {time_now} {float(delta_t)} "
        f"{(ly - min_max_height(psi, epsilon, 1, 'min')) / 1e3}"
    )

    # Check if simulation has completed
    if time_now >= time_end:
        log("Reached end of simulation -- exiting time-step loop")
        break
# -

# At the end of the simulation, we write the final state for visualisation, close our
# logging file, and checkpoint solution fields to disk. These can later be used to
# restart the simulation, if required.

# +
output_file.write(*z.subfunctions, psi, time=time_now / 1e6 / 365.25 / 8.64e4)
plog.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
# -

# We can visualise the final level-set field using Firedrake's built-in plotting
# functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# tricontour(psi, axes=axes, levels=[0.5])
# -
