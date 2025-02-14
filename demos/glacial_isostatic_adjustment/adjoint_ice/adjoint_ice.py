# Synthetic ice inversion using adjoints
# =======================================================
#
# In this tutorial, we will use G-ADOPT's adjoint capability to invert for a synthetic ice
# load in an annulus domain. We will be running a 'twin' experiment where we will try to
# recover the ice load that we used as part of the earlier 2d cylindrical tutorial,
# starting from a different initial guess of the ice load.
#
# This example focusses on setting up an adjoint problem. These can be summarised as follows:
# 1. Defining an objective function.
# 2. Verifying the accuracy of the gradients using a Taylor test.
# 3. Setting up and solving a gradient-based minimisation problem for a synthetic ice load.

# This example
# -------------
# Let's get started!
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.

from gadopt import *
from gadopt.utility import step_func, vertical_component, CombinedSurfaceMeasure

# To bring in G-ADOPT's adjoint functionality we need to start taping the forward problem,
# which we do below. It's also good practice to clear the tape, so that we are starting
# fresh each time.

from gadopt.inverse import *
tape = get_working_tape()
tape.clear_tape()

# In this tutorial we are going load the mesh created by the forward cylindrical demo in the
# previous tutorial. This makes it easier to load the synthetic data from the previous
# tutorial for our 'twin' experiment.

# Let's download a checkpoint file we made earlier. This is the same as the forward cylindrical
# we saw in a previous tutorial.

# + tags=["active-ipynb"]
# ![ ! -f forward-2d-cylindrical-disp-incdisp.h5 ] && wget https://data.gadopt.org/demos/forward-2d-cylindrical-disp-incdisp.h5
# -

# Set up geometry:
checkpoint_file = "forward-2d-cylindrical-disp-incdisp.h5"
with CheckpointFile(checkpoint_file, 'r') as afile:
    mesh = afile.load_mesh(name='surface_mesh_extruded')
bottom_id, top_id = "bottom", "top"
mesh.cartesian = False
D = 2891e3  # Depth of domain in m

# We next set up the function spaces, and specify functions to hold our solutions.

# +
# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "Q", 2)  # (Incremental) Displacement function space (vector)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
S = TensorFunctionSpace(mesh, "DQ", 2)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Incremental Displacement")
z.subfunctions[1].rename("Pressure")

displacement = Function(V, name="displacement").assign(0)
stress_old = Function(S, name="stress_old").assign(0)
# -

# Let's set up the background profiles for the material properties with the same values as before.

# +
X = SpatialCoordinate(mesh)

# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [2, -2, -2, -1.698970004]  # viscosity = 1e23 * 10**viscosity_values
# N.b. that we have modified the viscosity of the Lithosphere viscosity from
# Spada et al 2011 because we are using coarse grid resolution


def initialise_background_field(field, background_values, vertical_tanh_width=40e3):
    profile = background_values[0]
    sharpness = 1 / vertical_tanh_width
    depth = sqrt(X[0]**2 + X[1]**2)-radius_values[0]
    for i in range(1, len(background_values)):
        centre = radius_values[i] - radius_values[0]
        mag = background_values[i] - background_values[i-1]
        profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

    field.interpolate(profile)


density = Function(W, name="density")
initialise_background_field(density, density_values)

shear_modulus = Function(W, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values)


def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
    numerator = exp(-1/(2*(1-rho**2))*arg)
    if normalised_area:
        denominator = 2*pi*sigma_x*sigma_y*(1-rho**2)**0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(viscosity):
    heterogenous_viscosity_field = Function(viscosity.function_space(), name='viscosity')
    antarctica_x, antarctica_y = -2e6, -5.5e6

    low_viscosity_antarctica = bivariate_gaussian(X[0], X[1], antarctica_x, antarctica_y, 1.5e6, 0.5e6, -0.4)
    heterogenous_viscosity_field.interpolate(-3*low_viscosity_antarctica + viscosity * (1-low_viscosity_antarctica))

    llsvp1_x, llsvp1_y = 3.5e6, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6, 1e6, 0)
    heterogenous_viscosity_field.interpolate(-3*llsvp1 + heterogenous_viscosity_field * (1-llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6, 1e6, 0)
    heterogenous_viscosity_field.interpolate(-3*llsvp2 + heterogenous_viscosity_field * (1-llsvp2))

    slab_x, slab_y = 3e6, 4.5e6
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6, 0.35e6, 0.7)
    heterogenous_viscosity_field.interpolate(-1*slab + heterogenous_viscosity_field * (1-slab))

    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6
    high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6, 0.5e6, 0.2)
    heterogenous_viscosity_field.interpolate(-1*high_viscosity_craton + heterogenous_viscosity_field * (1-high_viscosity_craton))

    return heterogenous_viscosity_field


normalised_viscosity = Function(W, name="Normalised viscosity")
initialise_background_field(normalised_viscosity, viscosity_values)
normalised_viscosity = setup_heterogenous_viscosity(normalised_viscosity)

viscosity = Function(normalised_viscosity, name="viscosity").interpolate(1e23*10**normalised_viscosity)

# -

# Now let's setup the ice load. For this tutorial we will start with an ice thickness of zero
# everywhere, but our target ice load will be the same two synthetic ice sheets in the
# previous demo. An import step is to define our control, i.e. the thing that we are inverting
# for. In our case, this is the normalised ice thickness.

# +
rho_ice = 931
g = 9.8125

Hice1 = 1000
Hice2 = 2000
year_in_seconds = Constant(3600 * 24 * 365.25)
# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
disc_halfwidth2 = (2*pi/360) * 20  # Disk half width in radians
surface_dx = 200*1e3
ncells = 2*pi*radius_values[0] / surface_dx
surface_resolution_radians = 2*pi / ncells
colatitude = atan2(X[0], X[1])
disc1_centre = (2*pi/360) * 25  # centre of disc1
disc2_centre = pi  # centre of disc2
disc1 = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))
disc2 = 0.5*(1-tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians)))

target_normalised_ice_thickness = Function(W, name="target normalised ice thickness")
target_normalised_ice_thickness.interpolate(disc1 + (Hice2/Hice1)*disc2)

normalised_ice_thickness = Function(W, name="normalised ice thickness")

control = Control(normalised_ice_thickness)
ice_load = rho_ice * g * Hice1 * normalised_ice_thickness

adj_ice_file = VTKFile("adj_ice.pvd")
# Since we are calculating the sensitivity to a field that is only defined
# on the top boundary if we do the usual L2 projection (using the
# mass matrix) to account for the size of the mesh element then we
# will get spurious oscillating values in the output gradient in
# cells not connected to the boundary. Instead we do the projection using
# a surface integral, so that our output gradient accounts for the surface
# area of each cell.
converter = RieszL2BoundaryRepresentation(W, top_id)  # convert to surface L2 representation

# We add a diagnostic block to the tape which will output the gradient
# field every time the tape is replayed.
tape.add_block(DiagnosticBlock(adj_ice_file, normalised_ice_thickness, riesz_options={'riesz_representation': converter}))
# -


# Let's visualise the ice thickness using pyvista, by plotting a ring outside our synthetic Earth.

# + tags=["active-ipynb"]
# import pyvista as pv
# import matplotlib.pyplot as plt
# ice_cmap = plt.get_cmap("Blues", 25)
#
# # Make two points at the bounds of the mesh and one at the center to
# # construct a circular arc.
# rmin = 3480e3
# rmax = 6371e3
# D = rmax-rmin
# nz = 32
# dz = D / nz
#
# normal = [0, 0, 1]
# polar = [radius_values[0]-0.01*dz, 0, 0]
# center = [0, 0, 0]
# angle = 360.0
# arc = pv.CircularArcFromNormal(center, 10000, normal, polar, angle)
#
# # Stretch line by 20%
# transform_matrix = np.array(
#     [
#         [1.2, 0, 0, 0],
#         [0, 1.2, 0, 0],
#         [0, 0, 1.2, 0],
#         [0, 0, 0, 1],
#     ])
#
#
# def add_ice(p, m, scalar="normalised ice thickness", scalar_bar_args=None):
#
#     if scalar_bar_args is None:
#         scalar_bar_args = {
#             "title": 'Normalised ice thickness',
#             "position_x": 0.2,
#             "position_y": 0.8,
#             "vertical": False,
#             "title_font_size": 22,
#             "label_font_size": 18,
#             "fmt": "%.1f",
#             "font_family": "arial",
#             "n_labels": 5,
#         }
#     data = m.read()[0]  # MultiBlock mesh with only 1 block
#
#     # Extract boundary surface, remove inner surface and expand ring width
#     surf = data.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
#                                       feature_edges=False,manifold_edges=False)
#     sphere = pv.Sphere(radius=5e6)
#     clipped_surf = surf.clip_surface(sphere, invert=False)
#     transformed_surf = clipped_surf.transform(transform_matrix)
#
#     p.add_mesh(transformed_surf, scalars=scalar, line_width=10, clim=[0, 2], cmap=ice_cmap, scalar_bar_args=scalar_bar_args)
#
#
# visc_file = VTKFile('viscosity.pvd').write(normalised_viscosity)
# reader = pv.get_reader("viscosity.pvd")
# visc_data = reader.read()[0]  # MultiBlock mesh with only 1 block
# visc_cmap = plt.get_cmap("inferno_r", 25)
#
#
# def add_viscosity(p):
#     p.add_mesh(
#         visc_data,
#         component=None,
#         lighting=False,
#         show_edges=False,
#         cmap=visc_cmap,
#         clim=[-3, 2],
#         scalar_bar_args={
#             "title": 'Normalised viscosity',
#             "position_x": 0.2,
#             "position_y": 0.1,
#             "vertical": False,
#             "title_font_size": 22,
#             "label_font_size": 18,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         }
#     )
#
# # Read the PVD file
# updated_ice_file = VTKFile('ice.pvd').write(normalised_ice_thickness, target_normalised_ice_thickness)
# reader = pv.get_reader("ice.pvd")
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 2), border=False, notebook=True, off_screen=False)
#
# plotter.subplot(0, 0)
# add_ice(plotter, reader, 'target normalised ice thickness')
# add_viscosity(plotter)
# plotter.add_text("Target")
# plotter.camera_position = 'xy'
#
# plotter.subplot(0, 1)
# add_ice(plotter, reader, 'normalised ice thickness')
# add_viscosity(plotter)
# plotter.add_text("Initial Guess")
# plotter.camera_position = 'xy'
#
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()
# -

# Let's choose a timestep (and output frequency) of 1000 years.

# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 1000
dt = Constant(dt_years * year_in_seconds)
Tend_years = 10e3
Tend = Constant(Tend_years * year_in_seconds)
dt_out_years = 1e3
dt_out = Constant(dt_out_years * year_in_seconds)

max_timesteps = round((Tend - Tstart * year_in_seconds) / dt)
log("max timesteps: ", max_timesteps)

dump_period = round(dt_out / dt)
log("dump_period:", dump_period)
log(f"dt: {float(dt / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} years")
# -

# Similar to before, we setup the boundary conditions, this time using the normalised
# ice thickness to account for ice covered regions when calculating the density
# contrast across the free surface.

# Setup boundary conditions
stokes_bcs = {
    top_id: {
        'normal_stress': ice_load,
        'free_surface': {}
    },
    bottom_id: {'un': 0}
}


# We also need to specify a G-ADOPT approximation, nullspaces and finally the
# stokes solver as before.  For this tutorial we will use a direct solver for
# the matrix system, so we don't need to provide the near nullspace like before.


# +
approximation = SmallDisplacementViscoelasticApproximation(density, shear_modulus, viscosity, g=g)

Z_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True)

stokes_solver = ViscoelasticStokesSolver(z, stress_old, displacement, approximation,
                                         dt, bcs=stokes_bcs, constant_jacobian=True,
                                         nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                         solver_parameters="direct")
# -

# We next set up our output in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create a velocity function for plotting
velocity = Function(V, name="velocity")
velocity.interpolate(z.subfunctions[0]/dt)
# Create output file
output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, displacement, velocity)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
)

checkpoint_filename = "viscoelastic_loading-chk.h5"

gd = GeodynamicalDiagnostics(z, density, bottom_id, top_id)

# Initialise a (scalar!) function for logging vertical displacement
U = FunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (scalar)
vertical_displacement = Function(U, name="Vertical displacement")
# -

# Now is a good time to setup a helper function for defining the time integrated misfit that we need
# later as part of our overall objective function. This is going to be called at each timestep of
# the forward run to calculate the difference between the displacement and velocity at the surface
# compared our reference forward simulation.


# +
def integrated_time_misfit(timestep, velocity_misfit, displacement_misfit):
    with CheckpointFile(checkpoint_file, 'r') as afile:
        target_incremental_displacement = afile.load_function(mesh, name="Incremental Displacement", idx=timestep)
        target_displacement = afile.load_function(mesh, name="Displacement", idx=timestep)
    circumference = 2 * pi * radius_values[0]
    target_velocity = target_incremental_displacement/dt_years
    velocity.interpolate(z.subfunctions[0]/dt_years)
    velocity_error = velocity - target_velocity
    velocity_scale = 10/dt_years
    velocity_misfit += assemble(dot(velocity_error, velocity_error) / (circumference * velocity_scale**2) * ds(top_id))

    displacement_error = displacement - target_displacement
    displacement_scale = 50
    displacement_misfit += assemble(dot(displacement_error, displacement_error) / (circumference * displacement_scale**2) * ds(top_id))
    return velocity_misfit, displacement_misfit


# Overload surface integral measure for G-ADOPT's extruded meshes.
ds = CombinedSurfaceMeasure(mesh, degree=6)
# -

# Now let's run the simulation! This should be the same as before except we are calculating the surface
# misfit between our current simulation and the reference run at each timestep.

# +
velocity_misfit = 0
displacement_misfit = 0

for timestep in range(max_timesteps):

    stokes_solver.solve()
    velocity_misfit, displacement_misfit = integrated_time_misfit(timestep, velocity_misfit, displacement_misfit)
    time.assign(time+dt)

    if timestep % dump_period == 0:
        log("timestep", timestep)

        output_file.write(*z.subfunctions, displacement, velocity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Stokes")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(stress_old, name="Deviatoric stress")

    vertical_displacement.interpolate(vertical_component(displacement))

    # Log diagnostics:
    plog.log_str(
        f"{timestep} {float(time)} {float(dt)} "
        f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(top_id)} "
        f"{vertical_displacement.dat.data.min()} {vertical_displacement.dat.data.max()}"
    )


# -

# Let's create a helper function to warp the mesh based on the displacement and


# + tags=["active-ipynb"]
# def add_displacement(p, m, disp="Displacement", scalar_bar_args=None):
#     data = m.read()[0]  # MultiBlock mesh with only 1 block
#
#     # Make a colour map
#     boring_cmap = plt.get_cmap("inferno_r", 25)
#
#     # Artificially warp the output data by the displacement field
#     # Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors=disp, factor=1500)
#     if scalar_bar_args is None:
#         scalar_bar_args = {
#             "title": 'Displacement (m)',
#             "position_x": 0.2,
#             "position_y": 0.8,
#             "vertical": False,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         }
#
#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars=disp,
#         component=None,
#         lighting=False,
#         clim=[0, 600],
#         cmap=boring_cmap,
#         scalar_bar_args=scalar_bar_args,
#     )
# -

# As we can see from the plot below there is no displacement at the final time given there is no ice load!

# + tags=["active-ipynb"]
# # Read the PVD file
# reader = pv.get_reader("output.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
#
# # Plot displacement
#
# reader.set_active_time_point(10)  # Read last timestep
# disp_scalar_bar_args={
#         "title": 'Displacement (m)',
#         "position_x": 0.85,
#         "position_y": 0.3,
#         "vertical": True,
#         "title_font_size": 20,
#         "label_font_size": 16,
#         "fmt": "%.0f",
#         "font_family": "arial",
#     }
# add_displacement(plotter, reader, 'displacement', scalar_bar_args=disp_scalar_bar_args)
#
# # Plot ice ring
# reader = pv.get_reader("ice.pvd")
# ice_scalar_bar_args = {"title": 'Normalised ice thickness',
#                        "position_x": 0.1,
#                        "position_y": 0.3,
#                        "vertical": True,
#                        "title_font_size": 22,
#                        "label_font_size": 18,
#                        "fmt": "%.1f",
#                        "font_family": "arial",
#                        "n_labels": 5,
#                        }
# add_ice(plotter, reader, 'normalised ice thickness', scalar_bar_args=ice_scalar_bar_args)
#
# plotter.camera_position = 'xy'
# plotter.add_text("Time = 10 ka")
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()
# -

# The inverse problem
# ------------------------------
#
# Now we can define our overall objective function that we want to minimise.
# This includes the time integrated displacement and velocity misfit at the
# surface as we discussed above. It is also a good idea to add a smoothing
# and damping term to help regularise the inversion problem.

# +
circumference = 2 * pi * radius_values[0]

alpha_smoothing = 1
alpha_damping = 0.1
damping = assemble((normalised_ice_thickness) ** 2 / circumference * ds(top_id))
smoothing = assemble(dot(grad(normalised_ice_thickness), grad(normalised_ice_thickness)) / circumference * ds(top_id))

J = (displacement_misfit + velocity_misfit) / max_timesteps + alpha_damping * damping + alpha_smoothing * smoothing
log("J = ", J)
# -


# Let's also pause
# annotation as we are now done with the forward terms.

pause_annotation()

# Let's setup some call backs to help us keep track of the inversion.

# +
updated_ice_thickness = Function(normalised_ice_thickness, name="updated ice thickness")
updated_ice_thickness_file = VTKFile("updated_ice_thickness.pvd")
updated_displacement = Function(displacement, name="updated displacement")
updated_velocity = Function(z.subfunctions[0], name="updated velocity")
updated_out_file = VTKFile("updated_out.pvd")

with CheckpointFile(checkpoint_file, 'r') as afile:
    final_target_incremental_displacement = afile.load_function(mesh, name="Incremental Displacement", idx=9)
    final_target_displacement = afile.load_function(mesh, name="Displacement", idx=9)

final_target_velocity = Function(V, name="target velocity").interpolate(final_target_incremental_displacement / dt_years)
functional_values = []


def eval_cb(J, m):
    if functional_values:
        functional_values.append(min(J, min(functional_values)))
    else:
        functional_values.append(J)

    circumference = 2 * pi * radius_values[0]
    # Define the component terms of the overall objective functional
    log("displacement misfit", displacement_misfit.block_variable.checkpoint / max_timesteps)
    log("velocity misfit", velocity_misfit.block_variable.checkpoint / max_timesteps)

    damping = alpha_damping * assemble((normalised_ice_thickness.block_variable.checkpoint) ** 2 / circumference * ds(top_id))
    smoothing = alpha_smoothing * assemble(dot(grad(normalised_ice_thickness.block_variable.checkpoint), grad(normalised_ice_thickness.block_variable.checkpoint)) / circumference * ds(top_id))
    log("damping", damping)
    log("smoothing", smoothing)

    # Write out values of control and final forward model results
    updated_ice_thickness.assign(m)
    updated_ice_thickness_file.write(updated_ice_thickness, target_normalised_ice_thickness)
    updated_displacement.interpolate(displacement.block_variable.checkpoint)
    updated_velocity.interpolate(z.subfunctions[0].block_variable.checkpoint / dt)
    updated_out_file.write(updated_displacement, final_target_displacement, updated_velocity, final_target_velocity)


# -

# The next important step is to define the reduced functional. This is pyadjoint's way of
# associating our objective function with the control variable that we are trying to
# optimise. We can pass our call back function which will be called every time
# the functional is evaluated.

reduced_functional = ReducedFunctional(J, control, eval_cb_post=eval_cb)

# ### Verifying the forward tape
#
#
# A good check to see if the forward taping worked is to rerun the forward model based on
# the operations stored on the tape. We can do this by providing the control to the
# reducted functional and print out the answer - it is good to see they are the same!

log("J", J)
log("replay tape RF", reduced_functional(normalised_ice_thickness))

# ### Visualising the derivative
#
# We can now calculate the derivative of our objective function with respect to the
# ice thickness.  This is as simple as calling the `derivative()` method on  our
# reduced functional.

dJdm = reduced_functional.derivative()

# We can also plot the derivative using pyvista. First of all let's define another helper
# function to plot the sensitivity to the ice thickness as a ring outside the domain.


# + tags=["active-ipynb"]
# def add_sensitivity_ring(p, m, scalar_bar_args=None):
#     # Make a colour map
#     adj_cmap = plt.get_cmap("coolwarm", 25)
#     if scalar_bar_args is None:
#         scalar_bar_args = {
#             "title": 'Adjoint sensitivity',
#             "position_x": 0.2,
#             "position_y": 0.8,
#             "vertical": False,
#             "title_font_size": 22,
#             "label_font_size": 18,
#             "fmt": "%.1e",
#             "font_family": "arial",
#             "n_labels": 3,
#         }
#     data = m.read()[0]  # MultiBlock mesh with only 1 block
#
#     # Extract boundary surface, remove inner surface and expand ring width
#     surf = data.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
#                                       feature_edges=False, manifold_edges=False)
#     sphere = pv.Sphere(radius=5e6)
#     clipped_surf = surf.clip_surface(sphere, invert=False)
#
#     transform_matrix = np.array(
#         [
#             [1.1, 0, 0, 0],
#             [0, 1.1, 0, 0],
#             [0, 0, 1.1, 0],
#             [0, 0, 0, 1],
#         ])
#
#     transformed_surf = clipped_surf.transform(transform_matrix)
#     p.add_mesh(transformed_surf, line_width=8, scalar_bar_args=scalar_bar_args, clim=[-5e-7, 5e-7], cmap=adj_cmap)
# -

# Next we read in the file that was written out as part of the diagnostic callback
# added to the tape earlier. We can see there is a clear hemispherical pattern in
# the gradients. Red indicates that increasing the ice thickness here would increase
# out objective function and blue areas indicates that increasing the ice thickness
# here would decrease our objective function. In the 'southern' hemisphere
# where we have the biggest ice load the gradient is negative, which makes sense as
# we expect increasing the ice thickness here to reduce our surface misfit.

# + tags=["active-ipynb"]
# # Read the PVD file
# reader = pv.get_reader("ice.pvd")
# adj_reader = pv.get_reader("adj_ice.pvd")
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 2), border=False, notebook=True, off_screen=False)
# plotter.subplot(0, 0)
# add_ice(plotter, reader, 'target normalised ice thickness')
# add_viscosity(plotter)
# plotter.add_text("Target")
# plotter.camera_position = 'xy'
# plotter.subplot(0, 1)
# add_ice(plotter, reader, 'normalised ice thickness')
# add_viscosity(plotter)
#
# add_sensitivity_ring(plotter, adj_reader)
# plotter.camera_position = 'xy'
# plotter.add_text("Initial Guess")
# plotter.show(jupyter_backend="static", interactive=False)
# # Closes and finalizes movie
# plotter.close()
# -

# ### Verification of Gradients via a Taylor Test
#
# A good way to verify this the gradient is correct is to carry out a Taylor test. For the control, $I_h$,
# reduced functional, $J(I_h)$, and its derivative,
# $\frac{\mathrm{d} J}{\mathrm{d} I_h}$, the Taylor remainder convergence test can be expressed as:
#
# $$ \left| J(I_h + h \,\delta I_h) - J(I_h) - h\,\frac{\mathrm{d} J}{\mathrm{d} I_h} \cdot \delta I_h \right| \longrightarrow 0 \text{ at } O(h^2). $$
#
# The expression on the left-hand side is termed the second-order Taylor remainder. i
# This term's convergence rate of $O(h^2)$ is a robust indicator for
# verifying the computational implementation of the gradient calculation.
# Essentially, if you halve the value of $h$, the magnitude
# of the second-order Taylor remainder should decrease by a factor of 4.
#
# We employ these so-called *Taylor tests* to confirm the accuracy of the
# determined gradients. The theoretical convergence rate is
# $O(2.0)$, and achieving this rate indicates that the gradient information
# is accurate down to floating-point precision.
#
# ### Performing Taylor Tests
#
# In our implementation, we perform a second-order Taylor remainder test for each
# term of the objective functional. The test involves
# computing the functional and the associated gradient when randomly perturbing
# the initial temperature field, $T_{ic}$, and subsequently
# halving the perturbations at each level.
#
# Here is how you can perform a Taylor test in the code:

# +
h = Function(normalised_ice_thickness)
h.dat.data[:] = np.random.random(h.dat.data_ro.shape)
minconv = taylor_test(reduced_functional, normalised_ice_thickness, h)

with open("taylor_test_minconv.txt", "w") as f:
    f.write(str(minconv))
# -

# ### Setting up the inversion
#
# Now that we have verified our gradient is correct, let's start setting up an inversion.
# First of all we will define some bounds that we enforce the control to lie within.
# For this problem the lower bound of zero ice thickness is particularly important,
# as we do not want negative ice thicknesses!

# +
ice_thickness_lb = Function(normalised_ice_thickness.function_space(), name="Lower bound ice thickness")
ice_thickness_ub = Function(normalised_ice_thickness.function_space(), name="Upper bound ice thickness")
ice_thickness_lb.assign(0.0)
ice_thickness_ub.assign(5)

bounds = [ice_thickness_lb, ice_thickness_ub]
# -

# Next we setup a pyadjoint minimization problem. We tweak GADOPT's default minimisation
# parameters (found in `gadopt/inverse.py`) for our problem. We limit the number of
# iterations to 15 just so that the demo is quick to run. (N.b. 35 iterations gives a
# very accurate answer.) We also increase the size of the initial radius of the trust region
# so that the inversion gets going a bit quicker than the default setting.

# +
minimisation_problem = MinimizationProblem(reduced_functional, bounds=bounds)

minimisation_parameters["Status Test"]["Iteration Limit"] = 15
minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1e4

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir="optimisation_checkpoint",
)
# Restart file for optimisation...
updated_ice_thickness_file = VTKFile("updated_ice_thickness.pvd")
updated_out_file = VTKFile("updated_out.pvd")
functional_values = []
# -


# ### Running the inversion
#
# Now let's run the inversion!

optimiser.run()

# If we're performing mulitple successive optimisations, we want
# to ensure the annotations are switched back on for the next code
# to use them

continue_annotation()

# Let's plot the results of the inversion at the final iteration.

# + tags=["active-ipynb"]
# # Read the PVD file
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 2), border=False, notebook=True, off_screen=False)
# reader = pv.get_reader("updated_out.pvd")
# reader.set_active_time_point(15)
# plotter.subplot(0, 0)
# add_displacement(plotter, reader, "Displacement")
# plotter.add_text("Target")
#
# plotter.subplot(0, 1)
# add_displacement(plotter, reader, disp="updated displacement")
#
# ice_scalar_bar_args = {"title": 'Normalised ice thickness',
#                        "position_x": 0.2,
#                        "position_y": 0.1,
#                        "vertical": False,
#                        "title_font_size": 22,
#                        "label_font_size": 18,
#                        "fmt": "%.1f",
#                        "font_family": "arial",
#                        "n_labels": 5,
#                        }
#
# reader = pv.get_reader("updated_ice_thickness.pvd")
# reader.set_active_time_point(15)
# plotter.subplot(0, 0)
# add_ice(plotter, reader, 'target normalised ice thickness', scalar_bar_args=ice_scalar_bar_args)
# plotter.camera_position = 'xy'
# plotter.subplot(0, 1)
# add_ice(plotter, reader, 'updated ice thickness')
#
# adj_reader = pv.get_reader("adj_ice.pvd")
# adj_reader.set_active_time_point(15)
# add_sensitivity_ring(plotter, adj_reader)
# plotter.camera_position = 'xy'
# plotter.add_text("Optimised")
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()
# -

# We can see that we have been able to recover two ice sheets in the correct locations and
# the final displacement pattern is very similar to the target run. Also the magnitude of the gradient
# is much smaller than before implying we are close to a minimum value of the objective function.

# And we'll write the functional values to a file so that we can test them.

with open("functional.txt", "w") as f:
    f.write("\n".join(str(x) for x in functional_values))

# We can confirm that
# the surface misfit has reduced by plotting the objective function at each iteration.

# + tags=["active-ipynb"]
# plt.semilogy(functional_values)
# plt.xlabel("Iteration #")
# plt.ylabel("Functional value")
# plt.title("Convergence")
# -
