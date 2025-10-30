# Idealised 2-D viscoelastic loading problem in an annulus
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in an annulus domain.
#
# This example focuses on differences between running simulations in a 2-D annulus
# with laterally varying viscosity compared to our
# [previous tutorial](../2d_cylindrical), where the viscosity varied as a function
# of depth only.

# This example
# -------------
# Let's get started!
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.

from gadopt import *
from gadopt.utility import (
    vertical_component,
    extruded_layer_heights,
    initialise_background_field
)
from gadopt.gia_demo_utilities import ice_sheet_disc

# We also import some helper functions for plotting and making animations associated
# with this demo.

# + tags=["active-ipynb"]
# from gadopt.gia_demo_utilities import (
#     plot_ice_ring,
#     plot_viscosity,
#     plot_animation
# )
# -

# Similar to our [previous tutorial](../base_case) demo we create the mesh in two
# stages. First we create a surface mesh of 180 cells using one of `Firedrake`'s
# utility meshes `CircleManifoldMesh` and then we extrude this in the radial
# direction by choosing the optional keyword argument `extrusion_type`. As before,
# the layer properties specified are from
# [Spada et al. (2011)](https://doi.org/10.1111/j.1365-246X.2011.04952.x).
# We specify 5 cells per rheological layer so 20 layers in total. To better
# represent the curvature of the domain and ensure accuracy of our quadratic
# representation of displacement, we approximate the curved cylindrical shell
# domain quadratically, using the optional keyword argument `degree`$=2$.
#
# As this problem is not formulated in a Cartesian geometry we set the `mesh.cartesian`
# attribute to `False`. This ensures the correct configuration of a radially inward
# gravitational direction.

# +
# Set up geometry:
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
domain_depth = radius_values[0]-radius_values[-1]
radius_values_nondim = np.array(radius_values)/domain_depth

# Construct a circle mesh and then extrude into a cylinder:
ncells = 180
rmin = radius_values_nondim[-1]
surface_mesh = CircleManifoldMesh(ncells, radius=rmin, degree=2, name='surface_mesh')

# Ensure layers of extruded mesh coincide with rheological boundaries
layer_heights = extruded_layer_heights(5, radius_values_nondim)

mesh = ExtrudedMesh(
    surface_mesh,
    layers=len(layer_heights),
    layer_height=layer_heights,
    extrusion_type='radial'
)

mesh.cartesian = False
boundary = get_boundary_ids(mesh)
# -

# We next set up the function spaces, and specify functions to hold our solutions.

# +
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space
S = TensorFunctionSpace(mesh, "DQ", 1)  # Stress tensor function space
DG0 = FunctionSpace(mesh, "DQ", 0)  # Density and shear modulus function space
DG1 = FunctionSpace(mesh, "DQ", 1)  # Viscosity function space
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

u = Function(V, name='displacement')
m = Function(S, name="internal variable")
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

log("Number of Displacement DOF:", V.dim())
log("Number of Internal variable DOF:", S.dim())

# We can now visualise the resulting mesh.

# + tags=["active-ipynb"]
# import pyvista as pv
# import matplotlib.pyplot as plt
#
# VTKFile("mesh.pvd").write(Function(V))
# mesh_data = pv.read("mesh/mesh_0.vtu")
# edges = mesh_data.extract_all_edges()
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(edges, color="black")
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# Let's start initialising some parameters. First of all let's get the (symbolic)
# spatial coordinates of the mesh

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties.
# In this case the density and shear modulus vary in the radial direction.
# The layer properties specified are from
# [Spada et al. (2011)](https://doi.org/10.1111/j.1365-246X.2011.04952.x)

# +
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
bulk_shear_ratio = 1.94
viscosity_values = [1e40, 1e21, 1e21, 2e21]

density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale

density_values_nondim = np.array(density_values)/density_scale
shear_modulus_values_nondim = np.array(shear_modulus_values)/shear_modulus_scale
viscosity_values_nondim = np.array(viscosity_values)/viscosity_scale

density = Function(DG0, name="density")
initialise_background_field(
    density, density_values_nondim, X, radius_values_nondim)

shear_modulus = Function(DG0, name="shear modulus")
initialise_background_field(
    shear_modulus, shear_modulus_values_nondim, X, radius_values_nondim)

bulk_modulus = Function(DG0, name="bulk modulus")
initialise_background_field(
    bulk_modulus, shear_modulus_values_nondim, X, radius_values_nondim)
# -

# Next let's initialise the viscosity field. In this tutorial we are
# going to make things a bit more interesting by using a laterally
# varying viscosity field. We'll put some regions of low viscosity
# near the South Pole (inspired by West Antarctica) as well as in the lower mantle.
# We've also put some relatively higher viscosity patches of mantle in the
# northern hemisphere to represent a downgoing slab. To better represent the
# spatially varying viscosity field lets use a linear discontinuous galerkin
# space, i.e. the viscosity fields varies linearly within cells but can
# have jumps in between cells.

# +
background_viscosity = Function(DG1, name="background viscosity")
initialise_background_field(
    background_viscosity, viscosity_values_nondim, X, radius_values_nondim)


# Defined lateral viscosity regions
def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
    numerator = exp(-1/(2*(1-rho**2))*arg)
    if normalised_area:
        denominator = 2*pi*sigma_x*sigma_y*(1-rho**2)**0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(
        background_viscosity: Function,
        viscosity_scale: float = 1e21,
        r_lith: float = 6301e3,
        domain_depth: float = 2891e3,
) -> Function:
    '''Adds lateral variations to a background viscosity field in a 2D annulus

    The synthetic lateral viscosity variations consist of 5 'blobs'
    constructed from bivariate gaussian functions to represent interesting features
    in the mantle. We assume the background viscosity only varies in the
    radial direction and do not make modifications to the viscosity structure
    in the lithosphere i.e. for r > `r_lith' where r is the radial distance.

    Args:
      background_viscosity:
        Background radial viscosity field (N.b. this is not modified)
      viscosity_scale:
        Characteristc viscosity used for nondimensionalisation
      r_lith:
        Radius of the lithosphere-mantle boundary in m
      domain_depth:
        Domain depth in m used for nondimensionalisation

    Returns:
      heterogenous_viscosity_field
        A new field containing the updated lateral viscosity variations
    '''
    heterogenous_viscosity_field = Function(background_viscosity.function_space(),
                                            name='viscosity')

    # Set up magnitudes of low and high viscosity regions
    low_visc = 1e20/viscosity_scale
    high_visc = 1e22/viscosity_scale

    # Add a low viscosity region in the bottom left corner
    # of the domain, aiming to mimic the low viscosity zone under the West
    # Antarctic ice sheet
    southpole_x, southpole_y = -2e6/domain_depth, -5.5e6/domain_depth
    low_viscosity_southpole = bivariate_gaussian(X[0], X[1],
                                                 southpole_x, southpole_y,
                                                 1.5e6/domain_depth,
                                                 0.5e6/domain_depth,
                                                 -0.4)

    heterogenous_viscosity_field.interpolate(
        low_visc*low_viscosity_southpole + background_viscosity * (1-low_viscosity_southpole))

    # Add two symmetrical low viscosity zones near the core-mantle boundary, inspired by
    # Large Low-Shear-Velocity Provinces (referred to as `llsvp`) so that
    # we can investigate sensitivity in the lower mantle.
    llsvp1_x, llsvp1_y = 3.5e6/domain_depth, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6/domain_depth,
                                1e6/domain_depth, 0)

    heterogenous_viscosity_field.interpolate(low_visc*llsvp1 +
                                             heterogenous_viscosity_field * (1-llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6/domain_depth, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6/domain_depth,
                                1e6/domain_depth, 0)

    heterogenous_viscosity_field.interpolate(low_visc*llsvp2 +
                                             heterogenous_viscosity_field * (1-llsvp2))

    # Add an elongated high viscosity region in the top right corner of the domain
    # to represent a slab geometry
    slab_x, slab_y = 3e6/domain_depth, 4.5e6/domain_depth
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6/domain_depth,
                              0.35e6/domain_depth, 0.7)

    heterogenous_viscosity_field.interpolate(high_visc*slab +
                                             heterogenous_viscosity_field * (1-slab))

    # Add a high viscosity feature at the top of the domain representing a craton
    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6/domain_depth
    high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x,
                                               high_viscosity_craton_y,
                                               1.5e6/domain_depth,
                                               0.5e6/domain_depth, 0.2)

    heterogenous_viscosity_field.interpolate(
        high_visc*high_viscosity_craton +
        heterogenous_viscosity_field * (1-high_viscosity_craton)
    )

    # We usually assume the lithosphere is purely elastic for GIA simulations,
    # so we reset viscosity in the lithosphere to the original background viscosity
    # value, which is assumed to be an arbitarily high constant so that the Maxwell
    # time in this layer is much larger than the timestep and simulation duration.
    heterogenous_viscosity_field.interpolate(
        conditional(vertical_component(X) > r_lith/domain_depth,
                    background_viscosity,
                    heterogenous_viscosity_field))

    return heterogenous_viscosity_field


viscosity = setup_heterogenous_viscosity(background_viscosity)
# -

# We'll keep the same ice synthetic ice sheet configuration as in the previous tutorial.
# Let's put one a larger one over the South Pole, with a total horizontal
# extent of 40 $^\circ$ and a maximum thickness of 2 km, and a smaller one offset from the
# North Pole with a width of 20 $^\circ$ and a maximum thickness of 1 km. To simplify
# things we keep the ice load fixed in time.

# +
# Initialise ice loading
rho_ice = 931 / density_scale
g = 9.815
B_mu = Constant(density_scale * domain_depth * g / shear_modulus_scale)
log("Ratio of buoyancy/shear = rho g D / mu = ", float(B_mu))
Hice1 = 1000 / domain_depth
Hice2 = 2000 / domain_depth

# Setup a disc ice load but with a smooth transition given by a tanh profile
disc_centre1 = (2*pi/360) * 25  # Centre of disc 1 in radians
disc_centre2 = pi  # Centre of disc 2 in radians
disc_halfwidth1 = (2*pi/360) * 10  # Disc 1 half width in radians
disc_halfwidth2 = (2*pi/360) * 20  # Disc 2 half width in radians
disc1 = ice_sheet_disc(X, disc_centre1, disc_halfwidth1)
disc2 = ice_sheet_disc(X, disc_centre2, disc_halfwidth2)
ice_load = B_mu * rho_ice * (Hice1 * disc1 + Hice2 * disc2)
# -

# As before let's visualise the viscosity field and ice thickness using pyvista.

# + tags=["active-ipynb"]
# # Write ice thicknesss .pvd file
# P1 = FunctionSpace(mesh, "CG", 1)  # Continuous function space
# ice_thickness = Function(P1, name="Ice thickness").interpolate(Hice1 * disc1 + Hice2 * disc2)
# zero_ice_thickness = Function(P1, name="zero").assign(0)  # Used for plotting later
# ice_thickness_file = VTKFile('ice.pvd').write(ice_thickness, zero_ice_thickness)
# visc_file = VTKFile('viscosity.pvd').write(viscosity)
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
# plot_ice_ring(plotter)
# plot_viscosity(plotter)
# plotter.camera_position = 'xy'
# plotter.show(jupyter_backend="static", interactive=False)
# plotter.close()
# -

# Let's setup the timestepping parameters with a timestep of 250 years and an output
# frequency of 1000 years.

# +
# Timestepping parameters
Tstart = 0
year_in_seconds = 3600*24*365.25
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 250
dt = Constant(dt_years * year_in_seconds/characteristic_maxwell_time)
Tend_years = 10e3
Tend = Constant(Tend_years * year_in_seconds/characteristic_maxwell_time)
dt_out_years = 1e3
dt_out = Constant(dt_out_years * year_in_seconds/characteristic_maxwell_time)

max_timesteps = round((Tend - Tstart * year_in_seconds/characteristic_maxwell_time) / dt)

output_frequency = round(dt_out / dt)
# -

# We can now define the boundary conditions to be used in this simulation. For the
# top surface we need to specify a normal stress, i.e. the weight of the ice load,
# as well as indicating this is a free surface.

# Setup boundary conditions
stokes_bcs = {boundary.top: {'free_surface': {'normal_stress': ice_load}},
              boundary.bottom: {'un': 0}
              }

# We also need to specify a G-ADOPT approximation which sets up the various parameters
# and fields needed for the viscoelastic loading problem.

approximation = MaxwellApproximation(
    bulk_modulus=bulk_modulus,
    density=density,
    shear_modulus=shear_modulus,
    viscosity=viscosity,
    B_mu=B_mu,
    bulk_shear_ratio=bulk_shear_ratio)

# As noted above, with a free-slip boundary condition on both boundaries, one can add
# an arbitrary rotation of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the displacement
# solution. These lead to null-modes (eigenvectors) for the linear system, rendering
# the resulting matrix singular. In preconditioned Krylov methods these null-modes
# must be subtracted from the approximate solution at every iteration. We do that
# below, setting up a nullspace object, specifying the `rotational` keyword argument
# to be True.

V_nullspace = rigid_body_modes(V, rotational=True)

# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution field `u` and
# various fields needed for the solve along with the approximation, timestep,
# list of internal variables, boundary conditions and nullspaces.

stokes_solver = InternalVariableSolver(
    u,
    approximation,
    dt=dt,
    internal_variables=m,
    bcs=stokes_bcs,
    constant_jacobian=True,
    nullspace=V_nullspace,
)

# We next set up our output in VTK format and the the logging file using `GeodynamicalDiagnostics` as before.

# +
# Create a velocity function for plotting
velocity = Function(u, name="velocity")
disp_old = Function(u, name="old_disp").assign(u)
# Create output file
output_file = VTKFile("output.pvd")
output_file.write(u, m, velocity)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf uv_min"
)

checkpoint_filename = "viscoelastic_loading-chk.h5"

gd = GeodynamicalDiagnostics(u, bottom_id=boundary.bottom, top_id=boundary.top)
# -

# Now let's run the simulation! At each step we call `solve` to calculate the
# displacement pressure field and update the internal variable accounting for
# the stress relaxation in the time dependent viscoelastic constitutive equation.

# +
for timestep in range(1, max_timesteps+1):

    time.assign(time+dt)
    stokes_solver.solve()

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(dt)} {gd.u_rms()} "
                 f"{gd.u_rms_top()} {gd.ux_max(boundary.top)} "
                 f"{gd.uv_min(boundary.top)}"
                 )

    velocity.interpolate((u - disp_old)/dt)
    disp_old.assign(u)

    if timestep % output_frequency == 0:
        log("timestep", timestep)

        output_file.write(u, m, velocity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u, name="displacement")
            checkpoint.save_function(m, name="internal variable")

plog.close()
# -

# Let's use the python package *PyVista* to plot the magnitude of the displacement
# field through time. We will use the calculated displacement to artifically scale
# the mesh. We have exaggerated the stretching by a factor of 1500, **BUT...**
# it is important to remember this is just for ease of visualisation - the mesh
# is not moving in reality!

# + tags=["active-ipynb"]
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
# plot_animation(plotter)
# plotter.close()
# -

# Looking at the animation, we can see that the weight of the ice load deforms the
# mantle, sinking beneath the ice load and pushing up material away from the ice
# load. This forebulge grows through the simulation and by 10,000 years is close
# to isostatic equilibrium. As the ice load is applied instantaneously the highest
# velocity occurs within the first timestep and gradually decays as the simulation
# goes on, though there is still a small amount of deformation ongoing after
# 10,000 years. We can also clearly see that the lateral viscosity variations
# give rise to asymmetrical displacement patterns. This is especially true near
# the South Pole, where the low viscosity region has enabled the isostatic
# relaxation to happen faster than the surrounding regions.

# ![SegmentLocal](displacement_warp.gif "segment")
