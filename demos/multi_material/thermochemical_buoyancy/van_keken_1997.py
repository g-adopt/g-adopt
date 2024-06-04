# Rayleigh-Taylor instability
# ===========================
#
# Rationale
# ---------
#
# One may wish to simulate a geodynamical flow involving multiple physical phases. A
# possible approach is to approximate phases as immiscible and forming a single fluid
# whose dynamics can still be described as a single-phase Stokes flow. Under this
# approximation, it is common to refer to immiscible phases as materials and to the
# resulting simulations as multi-material. In such simulations, each material occupies
# part of the numerical domain and is characterised by its own physical properties,
# such as density and viscosity. Along material boundaries, physical properties are
# averaged according to a chosen mathematical scheme.
#
# Numerical approach
# ------------------
#
# To model the coexistence of multiple materials in the numerical domain, we employ an
# interface-capturing approach called the conservative level-set method. Level-set
# methods associate each material interface to a mathematical field representing a
# measure of distance from that interface. In the conservative level-set approach, the
# classic signed-distance function, $\phi$, employed in the level-set method is
# transformed into a smooth step function, $\psi$, according to
#
# $$\psi(\mathbf{x}, t) = \frac{1}{2} \left[
# \mathrm{tanh} \left( \frac{\phi(\mathbf{x}, t)}{2\epsilon} \right) + 1
# \right]$$
#
# Throughout the simulation, the level-set field is advected with the flow:
#
# $$\frac{\partial \psi}{\partial t} + \nabla \cdot \left( \mathbf{u}\psi \right) = 0$$
#
# Advection of the level set deteriorates the shape of the initial profile. To maintain
# the profile as the simulation proceeds, a reinitialisation procedure is employed. We
# choose the equation proposed in Parameswaran and Mandal (2023):
#
# $$\frac{\partial \psi}{\partial \tau_{n}} = \theta \left[
# -\psi \left( 1 - \psi \right) \left( 1 - 2\psi \right)
# + \epsilon \left( 1 - 2\psi \right) \lvert\grad\psi\rvert
# \right]$$
#
# This example
# ------------
#
# Here, we consider the entrainment of a thin, compositionally dense layer by thermal
# convection presented in van Keken et al. (1997). Inside a 2-D domain heated from
# below, a denser material sits at the bottom boundary beneath a lighter material.
# Whilst the compositional stratification is stable, heat transfer from the boundary
# generates positive buoyancy in the denser material, allowing thin layers of material
# to be entrained in the convective circulation.

# As with all examples, the first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
nx, ny = 80, 40  # Number of cells in x and y directions
lx, ly = 2, 1  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W])  # Stokes function space (mixed)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space for time step

z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Symbolic UFL expressions for velocity and pressure
z.subfunctions[0].rename("Velocity")  # Associated Firedrake velocity function
z.subfunctions[1].rename("Pressure")  # Associated Firedrake pressure function
T = Function(Q, name="Temperature")  # Firedrake function for temperature
psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now provide initial conditions for the level set field. To this end, we use the
# shapely library to represent the initial location of the material interface and
# derive the signed-distance function. Finally, we apply the transformation to obtain a
# smooth step function profile.

# +
import numpy as np
import shapely as sl


def straight_line(x, slope, intercept):
    """Straight line equation"""
    return slope * x + intercept


interface_slope = 0  # Slope of the interface
material_interface_y = 0.025  # Vertical shift of the interface along the y axis
# Group parameters defining the straight-line profile
isd_params = (interface_slope, material_interface_y)

# Shapely LineString representation of the material interface
interface_x = np.linspace(0, lx, 1000)
interface_y = straight_line(interface_x, *isd_params)
line_string = sl.LineString([*np.column_stack((interface_x, interface_y))])
sl.prepare(line_string)

# Extract node coordinates
node_coords_x, node_coords_y = node_coordinates(psi)
# Determine to which material nodes belong and calculate distance to interface
node_relation_to_curve = [
    (
        node_coord_y > straight_line(node_coord_x, *isd_params),
        line_string.distance(sl.Point(node_coord_x, node_coord_y)),
    )
    for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y)
]

# Define the signed-distance function and overwrite its value array
signed_dist_to_interface = Function(K)
signed_dist_to_interface.dat.data[:] = [
    dist if is_above else -dist for is_above, dist in node_relation_to_curve
]

# Define thickness of the hyperbolic tangent profile
min_mesh_edge_length = min(lx / nx, ly / ny)
epsilon = Constant(min_mesh_edge_length / 4)

# Initialise level set as a smooth step function
psi.interpolate((1 + tanh(signed_dist_to_interface / 2 / epsilon)) / 2)
# -

# We next define materials present in the simulation. Here, the problem is
# non-dimensionalised and can be described by the product of the expressions for the
# Rayleigh and buoyancy numbers, RaB, which is also referred to as compositional
# Rayleigh number. Therefore, we provide a value for thermal and compositional Rayleigh
# numbers to define our approximation.

# +
dense_material = Material(RaB=4.5e5)
reference_material = Material(RaB=0)
materials = [dense_material, reference_material]

Ra = 3e5  # Thermal Rayleigh number

RaB = field_interface(
    [psi], [material.RaB for material in materials], method="arithmetic"
)  # Compositional Rayleigh number, defined based on each material value and location

approximation = BoussinesqApproximation(Ra, RaB=RaB)
# -

# As with the previous examples, we set up a *Timestep Adaptor* for controlling the
# time-step length (via a CFL criterion) as the simulation advances in time. We specify
# the initial time, initial time step $\Delta t$, and output frequency (in time units).

# +
time_now = 0  # Initial time
delta_t = Function(R).assign(1e-6)  # Initial time step
output_frequency = 1e-4  # Frequency (based on simulation time) at which to output
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)
# -

# This problem has a constant pressure nullspace, handled identically to our previous
# tutorials.

# +
Z_nullspace = create_stokes_nullspace(Z)
# -

# Boundary conditions are specified next: free slip on all sides, heating from below,
# and cooling from above.

# +
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"uy": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
temp_bcs = {bottom_id: {"T": 1}, top_id: {"T": 0}}
# -

# We move on to initialising the temperature field.

# +
X = SpatialCoordinate(mesh)  # Extract UFL representation of spatial coordinates

# Calculate quantities linked to the temperature initial condition using UFL
u0 = lx ** (7 / 3) / (1 + lx**4) ** (2 / 3) * (Ra / 2 / sqrt(pi)) ** (2 / 3)
v0 = u0
Q_ic = 2 * sqrt(lx / pi / u0)
Tu = erf((1 - X[1]) / 2 * sqrt(u0 / X[0])) / 2
Tl = 1 - 1 / 2 * erf(X[1] / 2 * sqrt(u0 / (lx - X[0])))
Tr = 1 / 2 + Q_ic / 2 / sqrt(pi) * sqrt(v0 / (X[1] + 1)) * exp(
    -(X[0] ** 2) * v0 / (4 * X[1] + 4)
)
Ts = 1 / 2 - Q_ic / 2 / sqrt(pi) * sqrt(v0 / (2 - X[1])) * exp(
    -((lx - X[0]) ** 2) * v0 / (8 - 4 * X[1])
)

# Interpolate temperature initial condition and ensure boundary condition values
T.interpolate(max_value(min_value(Tu + Tl + Tr + Ts - 3 / 2, 1), 0))
DirichletBC(Q, 1, bottom_id).apply(T)
DirichletBC(Q, 0, top_id).apply(T)
# -

# We now set up our output. To do so, we create the output file as a ParaView Data file
# that uses the XML-based VTK file format. We also open a file for logging and
# instantiate G-ADOPT geodynamical diagnostic utility.

# +
output_file = VTKFile("output.pvd")

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms entrainment")

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id)
material_area = material_interface_y * lx  # Quantity of tracked material
entrainment_height = 0.2  # Height above which entrainment diagnostic is calculated
# -

# Here, we set up the variational problem for the Stokes, energy, and level-set
# systems. The Stokes and energy systems depend on the approximation defined above,
# and the level-set system includes both advection and reinitialisation.

# +
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
)

subcycles = 1  # Number of advection solves to perform within one time step
level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, subcycles, epsilon)
# Increase the reinitialisation time step to make up for the coarseness of the mesh
level_set_solver.reini_params["tstep"] *= 20
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

# +
step = 0  #  A counter to keep track of looping
output_counter = 0  # A counter to keep track of outputting
time_end = 0.05
while True:
    # Write output
    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, T, psi)
        output_counter += 1

    # Update timestep
    if time_end is not None:
        t_adapt.maximum_timestep = min(output_frequency, time_end - time_now)
    t_adapt.update_timestep()
    time_now += float(delta_t)
    step += 1

    # Solve Stokes sytem
    stokes_solver.solve()

    # Temperature system
    energy_solver.solve()

    # Advect level set
    level_set_solver.solve(step)

    # Calculate material entrainment
    buoy_entr = entrainment(psi, material_area, entrainment_height)

    # Log diagnostics
    plog.log_str(f"{step} {time_now} {float(delta_t)} {gd.u_rms()} {buoy_entr}")

    # Check if simulation has completed
    if time_now >= time_end:
        log("Reached end of simulation -- exiting time-step loop")
        break
# -

# At the end of the simulation, once a steady-state has been achieved, we close our
# logging file and checkpoint solution fields to disk. These can later be used to
# restart the simulation, if required.

# +
plog.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
# -

# We can visualise the final temperature field using Firedrake's built-in plotting
# functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(collection);
# -