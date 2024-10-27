# Idealised 2-D viscoelastic loading problem in an annulus
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in an annulus domain. 
#
# This example focusses on differences between running simulations in a 2-D annulus and 2-D Cartesian domain. These can be summarised as follows:
# 1. The geometry of the problem - i.e. the computational mesh.
# 2. The radial direction of gravity (as opposed to the vertical direction in a Cartesian domain).
# 3. Solving a problem with laterally varying viscosity.
# 4. With no constraint on tangential flow on both boundaries, this case incorporates a (rotational) velocity nullspace.
#

# This example
# -------------
# Let's get started! 
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality. We also import pyvista, which is used for plotting vtk output.

from gadopt import *
from gadopt.utility import step_func
import pyvista as pv


# We generate a circular manifold mesh (with a target horizontal grid resolution of 250 km) and extrude in the radial direction,
# using the optional keyword argument `extrusion_type`, forming 80 layers. To better represent the
# curvature of the domain and ensure accuracy of our quadratic representation of velocity, we
# approximate the curved cylindrical shell domain quadratically, using the optional keyword argument `degree`$=2$.
# Because this problem is not formulated in a Cartesian geometry, we set the `mesh.cartesian`
# attribute to False. This ensures the correct configuration of a radially inward vertical direction.

# +
# Set up geometry:
L = 1500e3  # length of the domain in m
D = 2891e3  # Depth of domain in m
dx=250*1e3
nz=80

radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
# Set up geometry:
rmin = radius_values[-1]
rmax = radius_values[0]
# Construct a circle mesh and then extrude into a cylinder:
radius_earth = 6371e3
ncells = round(2 * pi * radius_earth / dx)
surface_dx = 2 * pi * radius_earth / ncells
log("target surface resolution = ", dx)
log("actual surface resolution = ", surface_dx)
dz = D / nz
bottom_id, top_id = 1, 2

surface_mesh = CircleManifoldMesh(ncells, radius=rmin, degree=2, name='surface_mesh')
mesh = Mesh("unstructured_annulus_refined_surface.msh") # ExtrudedMesh(surface_mesh, layers=nz, layer_height=dz, extrusion_type='radial')

mesh.cartesian = False


# -
# We next set up the function spaces, and specify functions to hold our solutions,
# as with our previous tutorials.

# +
# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
TP1 = TensorFunctionSpace(mesh, "DG", 2)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
u_, p_ = z.subfunctions
u_.rename("Incremental Displacement")
p_.rename("Pressure")

displacement = Function(V, name="displacement").assign(0)
stress_old = Function(TP1, name="stress_old").assign(0)
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

# We can now visualise the resulting mesh.

VTKFile("mesh.pvd").write(Function(V))
mesh_data = pv.read("mesh/mesh_0.vtu")
edges = mesh_data.extract_all_edges()
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(edges, color="black")
plotter.camera_position = "xy"
plotter.show(jupyter_backend="static", interactive=False)

# Let's start initialising some parameters. First of all Firedrake has a helpful function to give a symbolic representation of the mesh coordinates.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties. In this case the density, shear modulus and viscosity only vary in the vertical direction. We will approximate the series of layers using a smooth tanh function with a width of 20 km.


# +
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
        


# +
# layer properties from spada et al 2011

density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values =  [2, -2, -2, -1.698970004] #[1e25, 1e21, 1e21, 2e21]
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

DG0 = FunctionSpace(mesh, "CG", 1)

density = Function(DG0, name="density")
initialise_background_field(density, density_values)

shear_modulus = Function(DG0, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values)

viscosity = Function(DG0, name="viscosity")
initialise_background_field(viscosity, viscosity_values)

viscosity = setup_heterogenous_viscosity(viscosity)
print(shear_modulus.dat.data[:].min())
visc_file = VTKFile('viscosity.pvd').write(viscosity)

# +
import matplotlib.pyplot as plt
import pyvista as pv

# Read the PVD file
reader = pv.get_reader("viscosity.pvd")
data = reader.read()[0]  # MultiBlock mesh with only 1 block

print(data.get_array)
# Create a plotter object
plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# Make a colour map
boring_cmap = plt.get_cmap("viridis", 25)
# Add the warped displacement field to the frame
plotter.add_mesh(
    data,
  #  scalars="viscosity",
    component=None,
    lighting=False,
   # show_edges=True,
   # edge_color='white',
    #clim=[0, 70],
    cmap=boring_cmap,
    scalar_bar_args={
        "title": 'Viscosity',
        "position_x": 0.8,
        "position_y": 0.2,
        "vertical": True,
        "title_font_size": 20,
        "label_font_size": 16,
        "fmt": "%.0f",
        "font_family": "arial",
    }
)
plotter.camera_position = 'xy'
plotter.show()
# Closes and finalizes movie
plotter.close()
# -

# Next let's create a function to store our ice load. Following the long test from Weeredesteijn et al 2023, during the first 90 thousand years of the simulation the ice sheet will grow to a thickness of 1 km. The ice thickness will rapidly shrink to ice free conditons in the next 10 thousand years. Finally, the simulation will run for a further 10 thousand years to allow the system to relax towards isostatic equilibrium. This is approximately the length of an interglacial-glacial cycle. The width of the ice sheet is 100 km and we have used a tanh function again to smooth out the transition.

# +
rho_ice = 931
g = 9.8125

Hice1 = 1000
Hice2 = 2000
year_in_seconds = Constant(3600 * 24 * 365.25)
# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
disc_halfwidth2 = (2*pi/360) * 20  # Disk half width in radians
surface_resolution_radians = 2*pi / ncells
colatitude = atan2(X[0], X[1])
disc1_centre = (2*pi/360) * 25  # centre of disc1
disc2_centre = pi  # centre of disc2
disc1 = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))
disc2 = 0.5*(1-tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians)))
ramp = Constant(1)
ice_load = Function(W, name="Ice load")
ice_load.interpolate(ramp * rho_ice * g * (Hice1 * disc1 + Hice2 * disc2))

visc_file = VTKFile('ice.pvd').write(ice_load)


# +
import matplotlib.pyplot as plt
import pyvista as pv

# Read the PVD file
reader = pv.get_reader("ice.pvd")
data = reader.read()[0]  # MultiBlock mesh with only 1 block

print(data.get_array)
# Create a plotter object
plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# Make a colour map
boring_cmap = plt.get_cmap("viridis", 25)
# Add the warped displacement field to the frame
plotter.add_mesh(
    data,
  #  scalars="viscosity",
    component=None,
    lighting=False,
    show_edges=False,
    edge_color='white',
    #clim=[0, 70],
    cmap=boring_cmap,
    scalar_bar_args={
        "title": 'Viscosity',
        "position_x": 0.8,
        "position_y": 0.2,
        "vertical": True,
        "title_font_size": 20,
        "label_font_size": 16,
        "fmt": "%.0f",
        "font_family": "arial",
    }
)
plotter.camera_position = 'xy'
plotter.show()
# Closes and finalizes movie
plotter.close()
# -

# Next let's define the length of our time step. If we want to accurately resolve the elastic response we should choose a timestep lower than the Maxwell time, $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation to 'catch up' the initial, instantaneous elastic deformation.
#
# Let's print out the maxwell time for each layer

for layer_visc, layer_mu in zip(viscosity_values, shear_modulus_values):
    log(f"Maxwell time: {float(1e23*10**layer_visc/layer_mu/year_in_seconds):.0f} years")

# As we can see the shortest Maxwell time is given by the lower mantle and is about 280 years, i.e. it will take about 280 years for the viscous deformation in that layer to catch up any instantaneous elastic deformation. Conversely the top layer, our lithosphere, has a maxwell time of 6 million years. Given that our simulations only run for 110000 years the viscous deformation over the course of the simulation will always be negligible compared with the elastic deformation. For now let's choose a timestep of 250 years and an output time step of 2000 years.

# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 200
dt = Constant(dt_years * year_in_seconds)
Tend_years = 10e3
Tend = Constant(Tend_years * year_in_seconds)
dt_out_years = 2e3
dt_out = Constant(dt_out_years * year_in_seconds)

max_timesteps = round((Tend - Tstart * year_in_seconds) / dt)
log("max timesteps: ", max_timesteps)

dump_period = round(dt_out / dt)
log("dump_period:", dump_period)
log(f"dt: {float(dt / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} years")

do_write = True
# -

# We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and side boundaries to be free slip with no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing the string `ux` and `uy`, G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as indicating this is a free surface.
#
# The `delta_rho_fs` option accounts for the density contrast across the free surface whether there is ice or air (or in later examples ocean!) above a particular region of the mantle.

# +
# Setup boundary conditions
stokes_bcs = {
            top_id: {'normal_stress': ice_load, 'free_surface': {'delta_rho_fs': density - rho_ice*(disc1+disc2)}},
            bottom_id: {'un': 0}
        }


gd = GeodynamicalDiagnostics(z, density, bottom_id, top_id)
# -


# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields needed for the viscoelastic loading problem.


# +
actual_visc = Function(viscosity).interpolate(1e23*10**viscosity)

approximation = SmallDisplacementViscoelasticApproximation(density, shear_modulus, actual_visc, g=g)

log(float(np.min(Function(W).interpolate(approximation.maxwell_time).dat.data)/year_in_seconds))
# -

# As noted above, with a free-slip boundary condition on both boundaries, one can add an arbitrary rotation
# of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the velocity solution (i.e. this case incorporates a velocity nullspace). These lead to null-modes (eigenvectors) for the linear system, rendering the resulting matrix singular.
# In preconditioned Krylov methods these null-modes must be subtracted from the approximate solution at every iteration. We do that below speciying the `rotational` keyword argument to be `True`.
#

Z_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True)

# Given the increased computational expense (typically requiring more degrees of freedom) in a 2-D annulus domain, G-ADOPT defaults to iterative
# solver parameters. If you have seen some of G-ADOPT's mantle convection tutorials (e.g. 3-D Cartesian or 2-D Cylindrical), G-ADOPT's iterative solver setup is configured to use the GAMG preconditioner
# for the velocity block of the Stokes system, to which we must provide near-nullspace information. In 2-D this consists of two rotational and two
# translational modes.

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution fields `z` and various fields needed for the solve along with the approximation, timestep and boundary conditions.
#

stokes_solver = ViscoelasticStokesSolver(z, stress_old, displacement, approximation,
                                         dt, bcs=stokes_bcs,) # nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,)
                                         #near_nullspace=Z_near_nullspace )

# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
prefactor_prestress = Function(W, name='prefactor prestress').interpolate(approximation.prefactor_prestress(dt))
effective_viscosity = Function(W, name='effective viscosity').interpolate(approximation.effective_viscosity(dt))

if do_write:
    # Create output file
    output_file = VTKFile("output.pvd")
    output_file.write(u_, displacement, p_, stress_old, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity, ice_load)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
)

checkpoint_filename = "viscoelastic_loading-chk.h5"
# -

# Now let's run the simulation! We are going to control the ice thickness using the `ramp` parameter. At each step we call `solve` to calculate the incremental displacement and pressure fields. This will update the displacement at the surface and stress values accounting for the time dependent Maxwell consitutive equation.

for timestep in range(1, max_timesteps+1):

    stokes_solver.solve()

    time.assign(time+dt)

    if timestep % dump_period == 0:
        log("timestep", timestep)

        if do_write:
            output_file.write(u_, displacement, p_, stress_old, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity, ice_load)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u_, name="Incremental Displacement")
            checkpoint.save_function(p_, name="Pressure")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(stress_old, name="Deviatoric stress")

    # Log diagnostics:
    plog.log_str(
        f"{timestep} {float(time)} {float(dt)} "
        f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(top_id)} "
        f"{displacement.dat.data[:, 1].min()} {displacement.dat.data[:, 1].max()}"
    )

# Let's use the python package *PyVista* to plot the magnitude of the displacement field through time. We will use the calculated displacement to artifically scale the mesh in the vertical direction. We have exaggerated the vertical stretching by a factor of 1500, **BUT...** it is important to remember this is just for ease of visualisation - the mesh is not moving in reality!

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import pyvista as pv
#
# # Read the PVD file
# reader = pv.get_reader("output.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
#
# # Open a gif
# plotter.open_gif("displacement_warp.gif")
#
# # Make a colour map
# boring_cmap = plt.get_cmap("inferno", 25)
#
# # Fix camera in default position otherwise mesh appears to jumpy around!
# #plotter.camera_position = 'xy'
#
#
# for i in range(len(reader.time_values)):
#     reader.set_active_time_point(i)
#     data = reader.read()[0]
#     # Artificially warp the output data in the vertical direction by the free surface height
#     # Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors="displacement", factor=1500)
#     arrows = data.glyph(orient="Incremental Displacement", scale="Incremental Displacement", factor=200000, tolerance=0.05)
#   #  plotter.add_mesh(arrows, color="white", lighting=False)
#
#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars="viscosity",
#         component=None,
#         lighting=False,
#        # show_edges=True,
#         clim=[0, 600],
#         cmap=boring_cmap,
#         scalar_bar_args={
#             "title": 'Displacement (m)',
#             "position_x": 0.8,
#             "position_y": 0.2,
#             "vertical": True,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         }
#     )
#
#     
#     plotter.camera_position = [(0, 0, rmax*5),
#                                  (0.0, 0.0, 0.0),
#                                  (0.0, 1.0, 0.0)]
#     plotter.add_text(f"Time: {i*2000:6} years", name='time-label')
#     print(plotter.camera_position)
#     # Write end frame multiple times to give a pause before gif starts again!
#     for j in range(5):
#         plotter.write_frame()
#     
#     if i == len(reader.time_values)-1:
#         # Write end frame multiple times to give a pause before gif starts again!
#         for j in range(20):
#             plotter.write_frame()
#
#     plotter.clear()
#
# # Closes and finalizes movie
# plotter.close()
# -
# Looking at the animation, we can see that as the weight of the ice load builds up the mantle deforms, pushing up material away from the ice load. If we kept the ice load fixed this forebulge will eventually grow enough that it balances the weight of the ice, i.e the mantle is in isostatic isostatic equilbrium and the deformation due to the ice load stops. At 100 thousand years when the ice is removed the topographic highs associated with forebulges are now out of equilibrium so the flow of material in the mantle reverses back towards the previously glaciated region.

# ![SegmentLocal](displacement_warp.gif "segment")

# mesh_data = pv.read("output/output_0.vtu")
# edges = mesh_data.extract_all_edges()
# plotter = pv.Plotter(notebook=True)
#
# plotter.add_mesh(
#         mesh_data,
#         scalars="viscosity",
#         component=None,
#         lighting=False,
#         show_edges=True,
#     )
# plotter.add_mesh(edges, color="white", opacity=1)
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)

# +
mesh_data = pv.read("viscosity/viscosity_0.vtu")
print(mesh_data)

DG0 = FunctionSpace(mesh, "DG", 0)
# Output function space information:
log("P2:", V.dim())
log("P1:", W.dim())
log("DG0:", DG0.dim())

mesh_data = pv.read("output/output_0.vtu")
print(mesh_data)

DG0 = FunctionSpace(mesh, "DG", 0)
# Output function space information:
log("P2:", V.dim())
log("P1:", W.dim())
log("DG0:", DG0.dim())
log("DG2:", TP1.dim())
# -

# Add the warped displacement field to the frame
mesh_data = pv.read("output/output_0.vtu")
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(
        mesh_data,
        scalars="viscosity",
        component=None,
        lighting=False,
        show_edges=True,
    )
plotter.camera_position = "xy"
plotter.show(jupyter_backend="static", interactive=False)

# In this demo we have seen how to setup a simple viscoelastic loading problem. In the next demos we will start to look at more realistic cases in Earth-like geometry and with additional physics including gravity, transient rheology and sea level.

# References
# ----------
# Cathles L.M. (1975). *Viscosity of the Earth's Mantle*, Princeton University Press.
#
# Dahlen F. A. and Tromp J. (1998). *Theoretical Global Seismology*, Princeton University Press.
#
# Ranalli, G. (1995). Rheology of the Earth. Springer Science & Business Media.
#
# Weerdesteijn, M. F., Naliboff, J. B., Conrad, C. P., Reusen, J. M., Steffen, R., Heister, T., & Zhang, J. (2023). *Modeling viscoelastic solid earth deformation due to ice age and contemporary glacial mass changes in ASPECT*. Geochemistry, Geophysics, Geosystems.
#
# Wu P., Peltier W. R. (1982). *Viscous gravitational relaxation*, Geophysical Journal International.
#
# Zhong, S., Paulson, A., & Wahr, J. (2003). Three-dimensional finite-element modelling of Earth’s viscoelastic deformation: effects of lateral variations in lithospheric thickness. Geophysical Journal International.
