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
# Advection of the level set modifies the shape of the initial profile. In other words,
# the signed-distance property underpinning the smooth step function is lost. To
# maintain the original profile as the simulation proceeds, a reinitialisation
# procedure is employed. We choose the equation proposed in [Parameswaran and Mandal
# (2023)](https://www.sciencedirect.com/science/article/pii/S0997754622001364):
#
# $$\frac{\partial \psi}{\partial \tau_{n}} = \theta \left[
# -\psi \left( 1 - \psi \right) \left( 1 - 2\psi \right)
# + \epsilon \left( 1 - 2\psi \right) \lvert\nabla\psi\rvert
# \right]$$
#
# This example
# ------------
#
# Here, we consider the isoviscous Rayleigh-Taylor instability presented in [van Keken
# et al. (1997)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JB01353).
# Inside a 2-D domain, a buoyant, lighter material sits beneath a denser material. The
# initial material interface promotes the development of a rising instability on the
# domain's left-hand side, and further smaller-scale convective dynamics take place
# throughout the remainder of the simulation. We describe below how to implement this
# problem using G-ADOPT.

# As with all examples, the first step is to import the `gadopt` package, which
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
nx, ny = 40, 40  # Number of cells in x and y directions
lx, ly = 0.9142, 1  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

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

# We now initialise the level-set field. All we have to provide to G-ADOPT is a
# mathematical description of the interface location and use the available API. In this
# case, the interface is a curve and can be geometrically represented as a cosine
# function. In general, specifying a mathematical function would warrant supplying a
# callable (e.g. a function) implementing the mathematical operations, but given the
# common use of cosines, G-ADOPT already provides it and only callable arguments are
# required here. Additional presets are available for usual scenarios, and the API is
# sufficiently flexible to generate most shapes. Under the hood, G-ADOPT uses the
# `Shapely` library to determine the signed-distance function associated with the
# interface. We use G-ADOPT's default strategy to obtain a smooth step function profile
# from the signed-distance function.

# +
from numpy import linspace  # noqa: E402

# Initialise the level-set field according to the conservative level-set approach.
# First, write out the mathematical description of the material-interface location.
# Here, only arguments to the G-ADOPT cosine function are required. Then, use the
# G-ADOPT API to generate the thickness of the hyperbolic tangent profile and update the
# level-set field values.
callable_args = (
    curve_parameter := linspace(0, lx, 1000),
    interface_deflection := 0.02,
    perturbation_wavelength := 2 * lx,
    interface_coord_y := 0.2,
)
boundary_coordinates = [(lx, ly), (0.0, ly), (0.0, interface_coord_y)]

epsilon = interface_thickness(K)
assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="curve",
    interface_callable="cosine",
    interface_args=callable_args,
    boundary_coordinates=boundary_coordinates,
)
# -

# Let us visualise the location of the material interface that we have just initialised.
# To this end, we use Firedrake's built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(psi, levels=linspace(0, 1, 11), axes=axes, cmap="PiYG")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level-set")
# -

# We next define materials present in the simulation using the `Material` class. Here,
# the problem is non-dimensionalised and can be described by the product of the
# expressions for the Rayleigh and buoyancy numbers, RaB, which is also referred to as
# compositional Rayleigh number. Therefore, we provide a value for thermal and
# compositional Rayleigh numbers to define our approximation. Material fields, such as
# RaB, are created using the `field_interface` function, which generates a unique field
# over the numerical domain based on the level-set field(s) and values or expressions
# associated with each material. At the interface between two materials, the transition
# between values or expressions can be represented as sharp or diffuse, with the latter
# using averaging schemes, such as arithmetic, geometric, and harmonic means.


# +
Ra = 0  # Thermal Rayleigh number
# Compositional Rayleigh number, defined based on each material value and location
RaB_buoyant = 0.0
RaB_dense = 1.0
RaB = material_field(psi, [RaB_buoyant, RaB_dense], interface="arithmetic")

approximation = BoussinesqApproximation(Ra, RaB=RaB)
# -

# As with the previous examples, we set up an instance of the `TimestepAdaptor` class
# for controlling the time-step length (via a CFL criterion) whilst the simulation
# advances in time. We specify the initial time, initial time step $\Delta t$, and
# output frequency (in time units).

time_now = 0  # Initial time
delta_t = Function(R).assign(1)  # Initial time step
output_frequency = 10  # Frequency (based on simulation time) at which to output
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that should not exceed 0.6.

# This problem setup has a constant pressure nullspace, which corresponds to the
# default case handled in G-ADOPT.

Z_nullspace = create_stokes_nullspace(Z)

# Boundary conditions are specified next: no slip at the top and bottom and free slip
# on the left and ride sides. No boundary conditions are required for level set, as the
# numerical domain is closed.

stokes_bcs = {
    boundary.bottom: {"u": 0},
    boundary.top: {"u": 0},
    boundary.left: {"ux": 0},
    boundary.right: {"ux": 0},
}

# We now set up our output. To do so, we create the output file as a ParaView Data file
# that uses the XML-based VTK file format. We also open a file for logging, instantiate
# G-ADOPT's geodynamical diagnostic utility, and define parameters to compute an
# additional diagnostic specific to multi-material simulations, namely material
# entrainment.

# +
output_file = VTKFile("output.pvd")

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms entrainment")

gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top)

material_area = interface_coord_y * lx  # Area of tracked material in the domain
entrainment_height = 0.2  # Height above which entrainment diagnostic is calculated
# -

# Here, we set up the variational problem for the Stokes and level-set systems. The
# former depends on the approximation defined above, and the latter includes both
# advection and reinitialisation components. Subcycling is available for level-set
# advection and is mainly useful when the problem at hand involves multiple CFL
# conditions, with the CFL for level-set advection being the most restrictive.

# +
stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

# Instantiate a solver object for level-set advection and reinitialisation. G-ADOPT
# provides default values for most arguments; we only provide those that do not have
# one. No boundary conditions are required, as the numerical domain is closed.
adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

# +
step = 0  # A counter to keep track of looping
output_counter = 0  # A counter to keep track of outputting
time_end = 2000
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

    # Advect level set
    level_set_solver.solve()

    # Calculate proportion of material entrained above a given height
    buoy_entr = material_entrainment(
        psi,
        material_size=material_area,
        entrainment_height=entrainment_height,
        side=0,
        direction="above",
    )

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

# We can visualise the final level-set field using Firedrake's built-in plotting
# functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# tricontour(psi, axes=axes, levels=[0.5])
# -
