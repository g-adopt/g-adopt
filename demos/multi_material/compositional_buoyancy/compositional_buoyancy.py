# # Rayleigh-Taylor instability
# ---

# Rationale
# -

# One may wish to simulate a geodynamical flow involving multiple physical phases. A
# possible approach is to approximate phases as immiscible, forming a single fluid that
# still behaves as a single-phase Stokes flow. Under this approximation, it is usual to
# refer to immiscible phases as materials and the resulting simulations as
# multi-material. In such simulations, each material occupies part of the numerical
# domain and has intrinsic physical properties, such as density and viscosity. Across a
# material interface, physical properties transition from one value to another, either
# sharply or smoothly. In the former case, physical property fields exhibit
# discontinuities, whilst in the latter case, an averaging scheme weights contributions
# from each material.

# Numerical approach
# -

# We employ an interface-capturing approach called the conservative level-set method
# ([Olsson and Kreiss, 2005](https://doi.org/10.1016/j.jcp.2005.04.007)) to model the
# coexistence of multiple materials in the numerical domain. Level-set methods associate
# each material interface to a mathematical field measuring the distance from that
# interface. In the conservative level-set approach, the classic signed-distance
# function, $\phi$, employed in the level-set method is transformed into a smooth step
# function, $\psi$:

# $$\psi(\mathbf{x}, t) = \frac{1}{2} \left[
# \mathrm{tanh} \left( \frac{\phi(\mathbf{x}, t)}{2\epsilon} \right) + 1
# \right]$$

# Throughout the simulation, the level-set field is advected with the flow:

# $$\frac{\partial \psi}{\partial t} + \nabla \cdot \left( \mathbf{u}\psi \right) = 0$$

# Advection of the level set modifies the shape of the initial profile. In other words,
# the signed-distance property underpinning the smooth step function is lost. To
# maintain the original profile whilst the simulation proceeds, a reinitialisation
# procedure is employed. We choose the equation proposed in [Parameswaran and Mandal
# (2023)](https://doi.org/10.1016/j.euromechflu.2022.11.001):

# $$\frac{\partial \psi}{\partial \tau_{n}} = \theta \left[
# -\psi \left( 1 - \psi \right) \left( 1 - 2\psi \right)
# + \epsilon \left( 1 - 2\psi \right) \lvert\nabla\psi\rvert
# \right]$$

# This example
# -

# Here, we consider the isoviscous Rayleigh-Taylor instability presented in [van Keken
# et al. (1997)](https://doi.org/10.1029/97JB01353). Inside a 2-D domain, a buoyant,
# lighter material sits beneath a denser material. The initial material interface
# promotes the development of a rising instability on the domain's left-hand side, and
# further convective dynamics occur throughout the remainder of the simulation. We
# describe below the implementation of this problem using G-ADOPT.

# As with all examples, the first step is to import the `gadopt` package, which also
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
mesh_elements = (64, 64)  # Number of cells in x and y directions
domain_dims = (0.9142, 1.0)  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(*mesh_elements, *domain_dims, quadrilateral=True)
mesh.cartesian = True  # Tag the mesh as Cartesian to inform other G-ADOPT objects
boundary = get_boundary_ids(mesh)  # Object holding references to mesh boundary IDs

V = VectorFunctionSpace(mesh, "Q", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W])  # Stokes function space (mixed)
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space (constants across the domain)

stokes = Function(Z)  # A field over the mixed function space Z
stokes.subfunctions[0].rename("Velocity")  # Firedrake function for velocity
stokes.subfunctions[1].rename("Pressure")  # Firedrake function for pressure
u = split(stokes)[0]  # Indexed expression for velocity in the mixed space
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
    curve_parameter := linspace(0.0, domain_dims[0], 1000),
    interface_deflection := 0.02,
    perturbation_wavelength := 2.0 * domain_dims[0],
    interface_coord_y := 0.2,
)
boundary_coordinates = [domain_dims, (0.0, domain_dims[1]), (0.0, interface_coord_y)]

epsilon = interface_thickness(K, min_cell_edge_length=True)
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
# contours = tricontourf(psi, levels=linspace(0.0, 1.0, 11), axes=axes, cmap="PiYG")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level set")
# -

# We next define the material fields and instantiate the approximation. Here, the system
# of equations is non-dimensional and only includes compositional buoyancy under the
# Boussinesq approximation. Moreover, physical parameters are constant through space
# apart from density. As a result, the system is fully defined by the compositional
# Rayleigh number. We use the `material_field` function to define its value throughout
# the domain (including the shape of the material interface transition) and provide it
# to our approximation.

# +
Ra = 0.0  # Thermal Rayleigh number
# Compositional Rayleigh number, defined based on each material value and location
RaB_buoyant = 0.0
RaB_dense = 1.0
RaB = material_field(psi, [RaB_buoyant, RaB_dense], interface="arithmetic")

approximation = BoussinesqApproximation(Ra, RaB=RaB)
# -

# Let us now verify that the material fields have been correctly initialised. We plot
# the compositional Rayleigh number across the domain.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(
#     Function(psi).interpolate(RaB), levels=linspace(0.0, 1.0, 11), axes=axes
# )
# fig.colorbar(contours, label="Compositional Rayleigh number")
# -

# As with the previous examples, we set up an instance of the `TimestepAdaptor` class
# for controlling the time-step length (via a CFL criterion) whilst the simulation
# advances in time. We specify the initial time, initial time step $\Delta t$, and
# output frequency (in time units).

time_now = 0.0  # Initial time
time_step = Function(R).assign(1.0)  # Initial time step
output_frequency = 10.0  # Frequency (based on simulation time) at which to output
t_adapt = TimestepAdaptor(
    time_step, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that should not exceed 0.6.

# Here, we set up the variational problem for the Stokes and level-set systems. The
# former depends on the approximation defined above, and the latter includes both
# advection and reinitialisation components.

# +
# This problem setup has a constant pressure nullspace, which corresponds to the default
# case handled in G-ADOPT.
stokes_nullspace = create_stokes_nullspace(Z)

# Boundary conditions for the Stokes system: no slip at the top and bottom and free slip
# on the left and right sides.
stokes_bcs = {
    boundary.bottom: {"u": 0.0},
    boundary.top: {"u": 0.0},
    boundary.left: {"ux": 0.0},
    boundary.right: {"ux": 0.0},
}
# Instantiate a solver object for the Stokes system and perform a solve to obtain
# initial pressure and velocity.
stokes_solver = StokesSolver(
    stokes,
    approximation,
    bcs=stokes_bcs,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)
stokes_solver.solve()

# Instantiate a solver object for level-set advection and reinitialisation. G-ADOPT
# defines default values for most arguments; here, we only provide those without one,
# namely the velocity field and time step (needed for advection) and the level-set
# interface thickness (needed for reinitialisation). No boundary conditions are
# required, as the numerical domain is closed.
adv_kwargs = {"u": u, "timestep": time_step}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
# -

# We now set up our output. To do so, we create the output file as a ParaView Data file
# that uses the XML-based VTK file format. We also open a file for logging, instantiate
# G-ADOPT geodynamical diagnostic utility, and define some parameters specific to this
# problem.

# +
output_file = VTKFile("output.pvd")
output_file.write(*stokes.subfunctions, psi, time=time_now)

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms entrainment")

gd = GeodynamicalDiagnostics(stokes, bottom_id=boundary.bottom, top_id=boundary.top)

# Area of tracked material in the domain
material_area = interface_coord_y * domain_dims[0]
entrainment_height = 0.2  # Height above which entrainment diagnostic is calculated
# -

# Finally, we initiate the time loop, which runs until the simulation end time.

# +
step = 0  # A counter to keep track of loop iterations
output_counter = 1  # A counter to keep track of outputting
time_end = 2000.0
while True:
    # Update timestep
    if time_end - time_now < output_frequency:
        t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    # Advect level set
    level_set_solver.solve()
    # Solve Stokes sytem
    stokes_solver.solve()

    # Increment iteration count and time
    step += 1
    time_now += float(time_step)

    # Calculate proportion of material entrained above a given height
    buoy_entr = material_entrainment(
        psi,
        material_size=material_area,
        entrainment_height=entrainment_height,
        side=0,
        direction="above",
    )

    # Log diagnostics
    plog.log_str(f"{step} {time_now} {float(time_step)} {gd.u_rms()} {buoy_entr}")

    # Write output
    if time_now >= output_counter * output_frequency:
        output_file.write(*stokes.subfunctions, psi, time=time_now)
        output_counter += 1

    # Check if simulation has completed
    if time_now >= time_end:
        plog.close()  # Close logging file

        # Checkpoint solution fields to disk
        with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
            final_checkpoint.save_mesh(mesh)
            final_checkpoint.save_function(stokes, name="Stokes")
            final_checkpoint.save_function(psi, name="Level set")

        log("Reached end of simulation -- exiting time-step loop")
        break
# -

# Let us finally examine the location of the material interface at the end of the
# simulation.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(psi, levels=linspace(0.0, 1.0, 11), axes=axes, cmap="PiYG")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level-set")
# -
