# Idealised 2-D mantle convection problem with compositional buoyancy inside an annulus
# =

# In this tutorial, we analyse mantle flow in a 2-D annulus domain. We define our domain by the radii
# of the inner ($r_{\text{min}}$) and outer ($r_{\text{max}}$) boundaries. These are chosen such that
# the non-dimensional depth of the mantle, $z = r_{\text{max}} - r_{\text{min}} = 1$, and the ratio of
# the inner and outer radii, $f=r_{\text{min}} / r_{\text{max}} = 0.55$, thus approximating the ratio
# between the radii of Earth's surface and core-mantle-boundary (CMB). Specifically, we set
# $r_{\text{min}} = 1.22$ and $r_{\text{max}} = 2.22$.

# This example focusses on differences between running simulations in a 2-D annulus and 2-D Cartesian domain. These can be summarised as follows:
# 1. The geometry of the problem - i.e. the computational mesh.
# 2. The radial direction of gravity (as opposed to the vertical direction in a Cartesian domain).
# 3. Initialisation of the temperature field in a different domain.
# 4. With free-slip boundary conditions on both boundaries, this case incorporates a (rotational) velocity nullspace, as well as a pressure nullspace.

# The example is configured at $Ra = 1e5$. Boundary conditions are free-slip at the surface and base of the domain.

# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.
# We also import pyvista, which is used for plotting vtk output.

from gadopt import *

# + tags=["active-ipynb"]
# import pyvista as pv
# -

# We next set up the mesh, function spaces, and specify functions to hold our solutions,
# as with our previous tutorials.

# We generate a circular manifold mesh (with 128 elements) and extrude in the radial
# direction (using the optional keyword argument `extrusion_type`) to produce 32 layers.
# To better represent the curvature of the domain and ensure accuracy of our quadratic
# velocity representation, we approximate the curved cylindrical shell domain
# quadratically, using the optional keyword argument `degree`$=2$. Because this problem
# is not formulated in a Cartesian geometry, we set the `mesh.cartesian` attribute to
# `False`. This ensures the gravity direction points radially inward.

# +
rmin, rmax = 1.22, 2.22  # Annulus radii
mesh1d = CircleManifoldMesh(ncells=128, radius=rmin, degree=2)  # Circle mesh
mesh = ExtrudedMesh(mesh1d, layers=32, extrusion_type="radial")  # Annulus mesh
mesh.cartesian = False
boundary = get_boundary_ids(mesh)

V = VectorFunctionSpace(mesh, "Q", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "Q", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space (constants across the domain)

stokes = Function(Z)  # A field over the mixed function space Z
stokes.subfunctions[0].rename("Velocity")  # Firedrake function for velocity
stokes.subfunctions[1].rename("Pressure")  # Firedrake function for pressure
u = split(stokes)[0]  # Indexed expression for velocity in the mixed space
psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We can now visualise the resulting mesh.

# + tags=["active-ipynb"]
# VTKFile("mesh.pvd").write(Function(V))
# mesh_data = pv.read("mesh/mesh_0.vtu")
# edges = mesh_data.extract_all_edges()
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(edges, color="black")
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
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

# Initialise the level-set field according to the conservative level-set approach.
# First, write out the mathematical description of the material-interface location.
# Here, only arguments to the G-ADOPT cosine function are required. Then, use the
# G-ADOPT API to generate the thickness of the hyperbolic tangent profile and update the
# level-set field values.
x, y = SpatialCoordinate(mesh)  # Extract UFL representation of spatial coordinates
r = sqrt(x**2 + y**2)  # Radial coordinate
interface_coord_r = rmin + (rmax - rmin) / 3  # Interface location
signed_distance = r - interface_coord_r  # Signed distance from the interface

epsilon = interface_thickness(K)
assign_level_set_values(psi, epsilon, signed_distance)

# Let us visualise the location of the material interface that we have just initialised.
# To this end, we use Firedrake's built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(psi, levels=linspace(0.0, 1.0, 11), cmap="PiYG", axes=axes)
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level set")
# -

# We next define the material fields and instantiate the approximation. Here, the system
# of equations is non-dimensional and includes compositional and thermal buoyancy terms
# under the Boussinesq approximation. Moreover, physical parameters are constant through
# space apart from density. As a result, the system is fully defined by the values of
# the thermal and compositional Rayleigh numbers. We use the `material_field` function
# to define the compositional Rayleigh number throughout the domain (including the shape
# of the material interface transition). Both non-dimensional numbers are provided to
# our approximation.

# +
Ra = 1e5  # Thermal Rayleigh number
RaB_buoyant, RaB_dense = 0.0, 3e4
# Compositional Rayleigh number, defined based on each material value and location
RaB = material_field(psi, [RaB_buoyant, RaB_dense], interface="arithmetic")
# Viscosity defined as a material field
mu = material_field(psi, [mu_buoyant := 1.0, mu_dense := 2.0], interface="arithmetic")

approximation = BoussinesqApproximation(Ra, RaB=RaB, mu=mu)
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
# As with the previous examples, we set up a *Timestep Adaptor*,
# for controlling the time-step length (via a CFL
# criterion) as the simulation advances in time. For the latter,
# we specify the initial time, initial timestep $\Delta t$, and number of
# timesteps. Given the low Rayleigh number, a steady-state tolerance is also specified,
# allowing the simulation to exit when a steady-state has been achieved.

time_now = 0.0  # Initial time
time_step = Function(R).assign(1e-7)  # Initial time step
output_frequency = 5e-4  # Frequency (based on simulation time) at which to output
t_adapt = TimestepAdaptor(
    time_step, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that does not exceed 0.6.

# We next set up and initialise our Temperature field.
# We choose the initial temperature distribution to trigger upwelling of 4 equidistant plumes.
# This initial temperature field is prescribed as:

# $$T(x,y) = (r_{\text{max}} - r) + A\cos(4 \; atan2\ (y,x))  \sin(r-r_{\text{min}}) \pi)$$

# where $A=0.02$ is the amplitude of the initial perturbation.

T = Function(Q, name="Temperature")
T.interpolate(rmax - r + 0.02 * cos(4.0 * atan2(y, x)) * sin((r - rmin) * pi))

# We can plot this initial temperature field:

# + tags=["active-ipynb"]
# VTKFile("temp.pvd").write(T)
# temp_data = pv.read("temp/temp_0.vtu")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(temp_data)
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# As noted above, with a free-slip boundary condition on both boundaries, one can add an arbitrary rotation
# of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the velocity solution (i.e. this case incorporates a velocity nullspace,
# as well as a pressure nullspace). These lead to null-modes (eigenvectors) for the linear system, rendering the resulting matrix singular.
# In preconditioned Krylov methods these null-modes must be subtracted from the approximate solution at every iteration. We do that below,
# setting up a nullspace object as we did in the previous tutorial, albeit speciying the `rotational` keyword argument to be True.
# This removes the requirement for a user to configure these options, further simplifying the task of setting up a (valid) geodynamical simulation.

stokes_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

# Boundary conditions are next specified. Boundary conditions for temperature are set to $T = 0$ at the surface ($r_{\text{max}}$) and $T = 1$
# at the base ($r_{\text{min}}$). For velocity, we specify free‐slip conditions on both boundaries. We incorporate these <b>weakly</b> through
# the <i>Nitsche</i> approximation. This illustrates a key advantage of the G-ADOPT framework: the user only specifies that the normal component
# of velocity is zero and all required changes are handled under the hood.
stokes_bcs = {boundary.bottom: {"un": 0.0}, boundary.top: {"un": 0.0}}
temp_bcs = {boundary.bottom: {"T": 1.0}, boundary.top: {"T": 0.0}}

# We can now setup and solve the variational problem, for both the energy and Stokes equations,
# passing in the approximation, nullspace and near-nullspace information configured above.

# +
# Instantiate a solver object for the energy conservation system.
energy_solver = EnergySolver(
    T, u, approximation, time_step, ImplicitMidpoint, bcs=temp_bcs
)
# Instantiate a solver object for the Stokes system and perform a solve to obtain
# initial pressure and velocity.
stokes_solver = StokesSolver(
    stokes,
    approximation,
    T,
    bcs=stokes_bcs,
    constant_jacobian=True,
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
output_file.write(*stokes.subfunctions, T, psi, time=time_now)

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms nu_base nu_top energy avg_t T_min T_max entrainment")

f_ratio = rmin / rmax
top_scaling = ln(f_ratio) / (f_ratio - 1.0)
bot_scaling = f_ratio * ln(f_ratio) / (f_ratio - 1.0)

gd = GeodynamicalDiagnostics(stokes, T, boundary.bottom, boundary.top, quad_degree=6)

# Area of tracked material in the domain
material_area = pi * (interface_coord_r**2 - rmin**2)
entrainment_height = interface_coord_r  # Height above which entrainment is calculated
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

step = 0  # A counter to keep track of looping
output_counter = 1  # A counter to keep track of outputting
time_end = 0.1
while True:
    # Update timestep
    if time_end - time_now < output_frequency:
        t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    # Advect level set
    level_set_solver.solve()
    # Solve energy system
    energy_solver.solve()
    # Solve Stokes sytem
    stokes_solver.solve()

    # Increment iteration count and time
    step += 1
    time_now += float(time_step)

    # Compute diagnostics:
    nusselt_number_top = gd.Nu_top() * top_scaling
    nusselt_number_base = gd.Nu_bottom() * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # # Calculate proportion of material entrained above a given height
    buoy_entr = material_entrainment(
        psi,
        material_size=material_area,
        entrainment_height=entrainment_height,
        side=0,
        direction="above",
        skip_material_size_check=True,
    )

    # Log diagnostics:
    plog.log_str(
        f"{step} {time_now} {float(time_step)} {gd.u_rms()} {nusselt_number_base} "
        f"{nusselt_number_top} {energy_conservation} {gd.T_avg()} {gd.T_min()} "
        f"{gd.T_max()} {buoy_entr}"
    )

    # Write output
    if time_now >= output_counter * output_frequency - 1e-16:
        output_file.write(*stokes.subfunctions, T, psi, time=time_now)
        output_counter += 1

    # Check if simulation has completed
    if time_now >= time_end:
        plog.close()  # Close logging file

        # Checkpoint solution fields to disk
        with CheckpointFile("final_state.h5", "w") as final_checkpoint:
            final_checkpoint.save_mesh(mesh)
            final_checkpoint.save_function(T, name="Temperature")
            final_checkpoint.save_function(stokes, name="Stokes")
            final_checkpoint.save_function(psi, name="Level set")

        log("Reached end of simulation -- exiting time-step loop")
        break

# Let us finally examine the location of the material interface and the temperature
# field at the end of the simulation.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(T, levels=linspace(0.0, 1.0, 11), axes=axes, cmap="inferno")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Temperature")
