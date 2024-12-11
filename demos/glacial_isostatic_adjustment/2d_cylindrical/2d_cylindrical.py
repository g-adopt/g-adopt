# Idealised 2-D viscoelastic loading problem in an annulus
# =======================================================

# In this tutorial, we examine an idealised 2-D loading problem in an annulus domain.

# This example focusses on differences between running simulations in 2-D annulus and
# 2-D Cartesian domains. These can be summarised as follows:
# 1. The geometry of the problem - i.e. the computational mesh;
# 2. The radial direction of gravity (as opposed to the vertical direction in a
#    Cartesian domain);
# 3. Solving a problem with laterally varying viscosity.
# 4. Accounting for a (rotational) velocity nullspace.

# This example
# -------------
# Let's get started!
# The first step is to import the gadopt module, which provides access to Firedrake and
# associated functionality.

from gadopt import *
from gadopt.utility import step_func, vertical_component

# In this tutorial we are going to use a fully unstructured mesh, created using
# [Gmsh](https://gmsh.info/). We have used Gmsh to construct a triangular mesh with a
# target resolution of 200 km near the surface of the Earth coarsening to 500 km in the
# interior. Gmsh is a widely used open-source software for creating finite element
# meshes, and Firedrake has inbuilt functionality to read Gmsh '.msh' files. Take a look
# at the *annulus_unstructured.py* script in this folder, which creates the mesh using
# Gmsh's Python API.
# As this problem is not formulated in a Cartesian geometry we set the `mesh.cartesian`
# attribute to `False`. This ensures the correct configuration of a radially inward
# vertical direction.

# +
# Set up geometry:
rmin = 3480e3
rmax = 6371e3
D = rmax - rmin
nz = 32
ncells = 180
dz = D / nz

# Construct a circle mesh and then extrude into a cylinder:
surface_mesh = CircleManifoldMesh(ncells, radius=rmin, degree=2, name="surface_mesh")
mesh = ExtrudedMesh(surface_mesh, layers=nz, layer_height=dz, extrusion_type="radial")

bottom_id, top_id = "bottom", "top"
mesh.cartesian = False
# -

# We next set up the function spaces and specify functions to hold our solutions. As our
# mesh is now made up of triangles instead of quadrilaterals, the syntax for defining
# our finite elements changes slighty. We need to specify *Continuous Galerkin*
# elements, i.e. replace `Q` with `CG` instead.

# +
# Set up function spaces - currently using the bilinear Q2Q1 element pair:
# (Incremental) Displacement function space (vector)
V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
# (Discontinuous) Stress tensor function space (tensor)
S = TensorFunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

Z = MixedFunctionSpace([V, W])  # Mixed function space

z = Function(Z)  # a field over the mixed function space Z
z.subfunctions[0].rename("Incremental displacement")
z.subfunctions[1].rename("Pressure")
u, p = split(z)  # Returns indexed UFL expressions for u and p

displ = Function(V, name="Displacement")
tau_old = Function(S, name="Deviatoric stress (old)")
# -

# We can output function space information, for example, the number of degrees of
# freedom (DOF).

# Output function space information:
log(f"Number of Incremental displacement DOF: {V.dim()}")
log(f"Number of Pressure DOF: {W.dim()}")
log(f"Number of Incremental displacement and Pressure DOF: {V.dim() + W.dim()}")

# We can now visualise the resulting mesh. As you can see there is finer resolution near
# the surface compared with the lower mantle.

# + tags=["active-ipynb"]
# import pyvista as pv
# import matplotlib.pyplot as plt

# VTKFile("mesh.pvd").write(Function(V))
# mesh_data = pv.read("mesh/mesh_0.vtu")
# edges = mesh_data.extract_all_edges()
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(edges, color="black")
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# Let's start initialising some parameters. First of all let's get the (symbolic)
# spatial coordinates of the mesh.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties. In this case
# the density and shear modulus vary in the vertical direction. We will approximate the
# series of layers using a smooth hyperbolic tangent function with a width of 40 km.

# +
# Layer properties from Spada et al. (2011)
# N.B.We  have modified the viscosity of the Lithosphere viscosity from
# Spada et al. (2011) because we are using coarse grid resolution.
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
rho_values = [3037, 3438, 3871, 4978]
G_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
mu_values = [2, -2, -2, -1.698970004]  # mu = 1e23 * 10**mu_values
# N.B. We have modified the viscosity of the lithosphere from Spada et al. (2011)
# because we are using a coarse grid resolution.


def initialise_background_field(field, background_values, vertical_tanh_width=40e3):
    profile = background_values[0]
    depth = sqrt(X[0] ** 2 + X[1] ** 2) - radius_values[0]

    for i in range(len(background_values) - 1):
        centre = radius_values[i + 1] - radius_values[0]
        mag = background_values[i + 1] - background_values[i]
        profile += step_func(
            depth, centre, mag, increasing=False, sharpness=1 / vertical_tanh_width
        )

    field.interpolate(profile)


rho = Function(W, name="Density")
G = Function(W, name="Shear modulus")
initialise_background_field(rho, rho_values)
initialise_background_field(G, G_values)
# -

# Let's have a quick look at the density field using PyVista. We can see that the mesh
# is still quite coarse, but it is able to capture the layered structure.

# + tags=["active-ipynb"]
# # Read the PVD file
# VTKFile("density.pvd").write(rho)
# reader = pv.get_reader("density.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block

# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# # Make a colour map
# boring_cmap = plt.get_cmap("viridis", 25)
# # Add the warped displacement field to the frame
# plotter.add_mesh(
#     data,
#     component=None,
#     lighting=False,
#     show_edges=False,
#     cmap=boring_cmap,
#     scalar_bar_args={
#         "title": "Density (kg / m^3)",
#         "position_x": 0.8,
#         "position_y": 0.2,
#         "vertical": True,
#         "title_font_size": 20,
#         "label_font_size": 16,
#         "fmt": "%.0f",
#         "font_family": "arial",
#     },
# )
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()  # Closes and finalizes movie
# -

# Next let's initialise the viscosity field. In this tutorial we are going to make
# things a bit more interesting by using a laterally varying viscosity field. We will
# put some regions of low viscosity near the South Pole (inspired by West Antarctica) as
# well as in the lower mantle. We have also put some relatively higher-viscosity patches
# of mantle in the northern hemisphere to represent a downgoing slab.


# +
def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    x_factor = (x - mu_x) / sigma_x
    y_factor = (y - mu_y) / sigma_y
    exp_factor = (x_factor) ** 2 - 2 * rho * (x_factor) * (y_factor) + (y_factor) ** 2
    numerator = exp(-1 / (2 * (1 - rho**2)) * exp_factor)

    if normalised_area:
        denominator = 2 * pi * sigma_x * sigma_y * (1 - rho**2) ** 0.5
    else:
        denominator = 1

    return numerator / denominator


def set_heterogenous_viscosity(viscosity):
    mu_heterogenous = Function(
        viscosity.function_space(), name="Heterogeneous viscosity"
    )
    antarctica_x, antarctica_y = -2e6, -5.5e6

    low_viscosity_antarctica = bivariate_gaussian(
        X[0], X[1], antarctica_x, antarctica_y, 1.5e6, 0.5e6, -0.4
    )
    mu_heterogenous.interpolate(
        -3 * low_viscosity_antarctica + viscosity * (1 - low_viscosity_antarctica)
    )

    llsvp1_x, llsvp1_y = 3.5e6, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6, 1e6, 0)
    mu_heterogenous.interpolate(-3 * llsvp1 + mu_heterogenous * (1 - llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6, 1e6, 0)
    mu_heterogenous.interpolate(-3 * llsvp2 + mu_heterogenous * (1 - llsvp2))

    slab_x, slab_y = 3e6, 4.5e6
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6, 0.35e6, 0.7)
    mu_heterogenous.interpolate(-1 * slab + mu_heterogenous * (1 - slab))

    viscous_craton_x, viscous_craton_y = 0, 6.2e6
    viscous_craton = bivariate_gaussian(
        X[0], X[1], viscous_craton_x, viscous_craton_y, 1.5e6, 0.5e6, 0.2
    )
    mu_heterogenous.interpolate(
        -1 * viscous_craton + mu_heterogenous * (1 - viscous_craton)
    )

    return mu_heterogenous


mu_scaled = Function(W, name="Normalised viscosity")
initialise_background_field(mu_scaled, mu_values)
mu_scaled = set_heterogenous_viscosity(mu_scaled)

mu = Function(mu_scaled, name="Viscosity").interpolate(1e23 * 10**mu_scaled)
# -

# Now let's plot the normalised viscosity viscosity field on a log plot (we have divided
# the viscosity by 1x10$^{23}$ Pa s). Again although we are using a coarse mesh we are
# able to capture the key features of the viscosity field.

# + tags=["active-ipynb"]
# # Read the PVD file
# VTKFile("viscosity.pvd").write(mu_scaled)
# reader = pv.get_reader("viscosity.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block

# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# # Make a colour map
# boring_cmap = plt.get_cmap("inferno_r", 25)
# # Add the warped displacement field to the frame
# plotter.add_mesh(
#     data,
#     component=None,
#     lighting=False,
#     show_edges=False,
#     cmap=boring_cmap,
#     scalar_bar_args={
#         "title": "Normalised viscosity",
#         "position_x": 0.8,
#         "position_y": 0.2,
#         "vertical": True,
#         "title_font_size": 20,
#         "label_font_size": 16,
#         "fmt": "%.0f",
#         "font_family": "arial",
#     },
# )
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()  # Closes and finalizes movie
# -

# Now let's setup the ice load. For this tutorial we will have two synthetic ice sheets.
# Let's put a larger one over the South Pole, with a total horizontal extent of
# 40 $^\circ$ and a maximum thickness of 2 km, and a smaller one offset from the North
# Pole, with a width of 20 $^\circ$ and a maximum thickness of 1 km. To simplify things
# let's keep the ice load fixed in time.

# +
rho_ice = 931
g = 9.8125
Hice_1 = 1000
Hice_2 = 2000

# Disc ice load but with a smooth transition given by a hyperbolic tangent profile
colatitude = atan2(X[0], X[1])
surface_dx = 200e3
ncells = 2 * pi * radius_values[0] / surface_dx
surface_resolution_radians = 2 * pi / ncells

disc_centre = (2 * pi / 360) * 25  # centre of disc_1
disc_halfwidth = (2 * pi / 360) * 10  # Disk half width in radians
disc_dist = abs(colatitude - disc_centre) - disc_halfwidth
disc_1 = 0.5 * (1 - tanh(disc_dist / 2 / surface_resolution_radians))

disc_centre = pi  # centre of disc_2
disc_halfwidth = (2 * pi / 360) * 20  # Disk half width in radians
disc_dist = abs(abs(colatitude) - disc_centre) - disc_halfwidth
disc_2 = 0.5 * (1 - tanh(disc_dist / 2 / surface_resolution_radians))

ice_load = rho_ice * g * (Hice_1 * disc_1 + Hice_2 * disc_2)
# -

# Let's visualise the ice thickness using PyVista by plotting a ring outside our
# synthetic Earth.


# + tags=["active-ipynb"]
# def make_ice_ring(reader):
#     data = reader.read()[0]

#     normal = [0, 0, 1]
#     polar = [rmax - dz / 2, 0, 0]
#     center = [0, 0, 0]
#     angle = 360.0
#     res = 10000
#     arc = pv.CircularArcFromNormal(center, res, normal, polar, angle)

#     arc_data = arc.sample(data)

#     # Stretch line by 20%
#     transform_matrix = np.array(
#         [
#             [1.2, 0, 0, 0],
#             [0, 1.2, 0, 0],
#             [0, 0, 1.2, 0],
#             [0, 0, 0, 1],
#         ]
#     )
#     return arc_data.transform(transform_matrix)


# def plot_ice_ring(plotter, ice_ring, scalar="Ice thickness"):
#     ice_cmap = plt.get_cmap("Blues", 25)

#     plotter.add_mesh(
#         ice_ring,
#         scalars=scalar,
#         line_width=10,
#         cmap=ice_cmap,
#         clim=[0, 2000],
#         scalar_bar_args={
#             "title": "Ice thickness (m)",
#             "position_x": 0.05,
#             "position_y": 0.3,
#             "vertical": True,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         },
#     )


# # Write ice thicknesss .pvd file
# ice_thickness = Function(W, name="Ice thickness").interpolate(
#     Hice1 * disc1 + Hice2 * disc2
# )
# zero_ice_thickness = Function(W, name="zero").assign(0)  # Used for plotting later
# ice_thickness_file = VTKFile("ice.pvd").write(ice_thickness, zero_ice_thickness)

# ice_reader = pv.get_reader("ice.pvd")
# ice_ring = make_ice_ring(ice_reader)

# reader = pv.get_reader("viscosity.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block

# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# # Make a colour map
# boring_cmap = plt.get_cmap("inferno_r", 25)
# # Add the warped displacement field to the frame
# plotter.add_mesh(
#     data,
#     component=None,
#     scalars=None,
#     lighting=False,
#     show_edges=False,
#     cmap=boring_cmap,
#     scalar_bar_args={
#         "title": "Normalised viscosity",
#         "position_x": 0.8,
#         "position_y": 0.3,
#         "vertical": True,
#         "title_font_size": 20,
#         "label_font_size": 16,
#         "fmt": "%.0f",
#         "font_family": "arial",
#     },
# )

# plot_ice_ring(plotter, ice_ring)

# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()  # Closes and finalizes movie
# -

# Let's setup the timestepping parameters with a timestep of 200 years and an output frequency of 1000 years.

# +
# Timestepping parameters
year_in_seconds = 8.64e4 * 365.25
time_start_years = 0
time_start = time_start_years * year_in_seconds
time_end_years = 10e3
time_end = time_end_years * year_in_seconds
time = Function(R).assign(time_start)

dt_years = 250
dt = dt_years * year_in_seconds
dt_out_years = 1e3
dt_out = dt_out_years * year_in_seconds
dt_years = 200

max_timesteps = round((time_end - time_start) / dt)
output_frequency = round(dt_out / dt)
log(f"max timesteps: {max_timesteps}")
log(f"output_frequency: {output_frequency}")
log(f"dt: {dt / year_in_seconds} years")
log(f"Simulation start time: {time_start_years} years")
# -

# We can now define the boundary conditions to be used in this simulation. Let's set the
# bottom and side boundaries to be free slip with no normal flow
# $\textbf{u} \cdot \textbf{n} = 0$. By passing the string `ux` and `uy`, G-ADOPT knows
# to specify these as strong Dirichlet boundary conditions.

# For the top surface we need to specify a normal stress, i.e. the weight of the ice
# load, as well as indicating this is a free surface.

# The `rho_ext` parameter represents the exterior density along the solid Earth free
# surface, such as that of ice or air.

# Setup boundary conditions
rho_ext = rho_ice * (disc_1 + disc_2)
stokes_bcs = {
    top_id: {"normal_stress": ice_load, "free_surface": {"rho_ext": rho_ext}},
    bottom_id: {"un": 0},
}

# We also need to specify a G-ADOPT approximation which sets up the various parameters
# and fields needed for the viscoelastic loading problem.

approximation = Approximation(
    "SDVA", dimensional=True, parameters={"G": G, "g": g, "mu": mu, "rho": rho}
)

# As noted above, with a free-slip boundary condition on both boundaries, one can add an
# arbitrary rotation of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the velocity
# solution. These lead to null-modes (eigenvectors) for the linear system, rendering the
# resulting matrix singular. In preconditioned Krylov methods these null-modes must be
# subtracted from the approximate solution at every iteration. We do that below, setting
# up a nullspace object, specifying the `rotational` keyword argument to be True. Note
# that we do not include a pressure nullspace as the top surface of the model is open.

Z_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True)

# Given the increased computational expense (typically requiring more degrees of
# freedom) in a 2-D annulus domain, G-ADOPT defaults to iterative solver parameters. As
# noted in our previous 3-D Cartesian tutorial, G-ADOPT's iterative solver setup is
# configured to use the GAMG preconditioner for the velocity block of the Stokes system,
# to which we must provide near-nullspace information, which, in 2-D, consists of two
# rotational and two translational modes.

Z_near_nullspace = create_stokes_nullspace(
    Z, closed=False, rotational=True, translations=[0, 1]
)

# We finally come to solving the variational problem, with solver objects for the Stokes
# system created. We pass in the solution fields `z` and various fields needed for the
# solve along with the approximation, timestep and boundary conditions.

viscoelastic_solver = ViscoelasticSolver(
    z,
    displ,
    tau_old,
    approximation,
    dt,
    bcs=stokes_bcs,
    constant_jacobian=True,
    nullspace={
        "nullspace": Z_nullspace,
        "transpose_nullspace": Z_nullspace,
        "near_nullspace": Z_near_nullspace,
    },
)

# We next set up our output, in VTK format. This format can be read by programs like
# PyVista and ParaView.

# +
# Create a velocity function for plotting
velocity = Function(V, name="Velocity")
velocity.interpolate(u / dt)

# Create output file
output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, velocity, displ, tau_old, mu, rho, G)

plog = ParameterLog("params.log", mesh)
plog.log_str("timestep time dt u_rms u_rms_surf ux_max disp_min disp_max")

checkpoint_filename = "viscoelastic_loading-chk.h5"

gd = GeodynamicalDiagnostics(z, bottom_id=bottom_id, top_id=top_id)

# Initialise a (scalar!) function for logging vertical displacement
U = FunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (scalar)
displ_vert = Function(U, name="Vertical displacement")
# -

# Now let's run the simulation! At each step we call `solve` to calculate the
# incremental displacement and pressure fields. This will update the displacement at the
# surface and stress values accounting for the time dependent Maxwell consitutive
# equation.

for timestep in range(max_timesteps + 1):
    viscoelastic_solver.solve()

    time += dt

    if timestep % output_frequency == 0:
        # First output step is after one solve i.e. roughly elastic displacement
        # provided dt < Maxwell time.
        log(f"timestep: {timestep}")

        velocity.interpolate(u / dt)
        output_file.write(*z.subfunctions, velocity, displ, tau_old, mu, rho, G)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Viscoelastic")
            checkpoint.save_function(displ, name="Displacement")
            checkpoint.save_function(tau_old, name="Deviatoric stress (old)")

    displ_vert.interpolate(vertical_component(displ))

    # Log diagnostics:
    plog.log_str(
        f"{timestep} {float(time)} {float(dt)} "
        f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(top_id)} "
        f"{displ_vert.dat.data.min()} {displ_vert.dat.data.max()}"
    )

# Let's use the python package *PyVista* to plot the magnitude of the displacement field
# through time. We will use the calculated displacement to artificially scale the mesh.
# We have exaggerated the stretching by a factor of 1500, **BUT...** it is important to
# remember this is just for ease of visualisation - the mesh is not moving in reality!

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import pyvista as pv

# # Read the PVD file
# reader = pv.get_reader("output.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block

# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# # Open a gif
# plotter.open_gif("displacement_warp.gif")

# # Make a colour map
# boring_cmap = plt.get_cmap("inferno_r", 25)

# # Fix camera in default position otherwise mesh appears to jumpy around!
# # plotter.camera_position = 'xy'

# # Make a list of output times (non-uniform because also
# # outputing first (quasi-elastic) solve
# times = [0, dt_years]
# for i in range(len(reader.time_values)):
#     times.append((i + 1) * 1000)


# for i in range(len(reader.time_values)):
#     print("Step: ", i)
#     reader.set_active_time_point(i)
#     data = reader.read()[0]

#     # Artificially warp the output data in the vertical direction by the free surface
#     # height. Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors="displacement", factor=1500)
#     arrows = warped.glyph(
#         orient="velocity", scale="velocity", factor=1e14, tolerance=0.01
#     )
#     plotter.add_mesh(arrows, color="grey", lighting=False)

#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars="displacement",
#         component=None,
#         lighting=False,
#         # show_edges=True,
#         clim=[0, 600],
#         cmap=boring_cmap,
#         scalar_bar_args={
#             "title": "Displacement (m)",
#             "position_x": 0.85,
#             "position_y": 0.3,
#             "vertical": True,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         },
#     )

#     plotter.camera_position = [
#         (0, 0, radius_values[0] * 5),
#         (0.0, 0.0, 0.0),
#         (0.0, 1.0, 0.0),
#     ]

#     plotter.add_text(f"Time: {times[i]:6} years", name="time-label")

#     if i == 0:
#         add_ice(plotter, scalar="zero")
#         for j in range(10):
#             plotter.write_frame()

#     plot_ice_ring(plotter, ice_ring)

#     # Write end frame multiple times to give a pause before gif starts again!
#     for j in range(10):
#         plotter.write_frame()

#     if i == len(reader.time_values) - 1:
#         # Write end frame multiple times to give a pause before gif starts again!
#         for j in range(20):
#             plotter.write_frame()

#     plotter.clear()

# # Closes and finalizes movie
# plotter.close()
# -
# Looking at the animation, we can see that the weight of the ice load deforms the
# mantle, sinking beneath the ice load and pushing up material away from the ice load.
# This forebulge grows through the simulation and by 10,000 years is close to isostatic
# equilibrium. As the ice load is applied instantaneously the highest velocity occurs
# within the first timestep and gradually decays as the simulation goes on, though there
# is still a small amount of deformation ongoing after 10,000 years. We can also clearly
# see that the lateral viscosity variations give rise to assymetrical displacement
# patterns. This is especially true near the South Pole, where the low viscosity region
# has enabled the isostatic relaxation to happen much faster than the surrounding
# regions.

# ![SegmentLocal](displacement_warp.gif "segment")
