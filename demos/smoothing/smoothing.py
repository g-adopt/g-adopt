# Isotropic and Anisotropic Smoothing in G-ADOPT
# ===============================================
#
# This tutorial demonstrates the application of **isotropic** and **anisotropic**
# diffusive smoothing on a cylindrical temperature field using G-ADOPT.
# The purpose is to illustrate how different diffusion properties can affect the
# smoothing behaviour, and how to potentially use them in geodynamic simulations.
#
# Smoothing is a critical technique in geodynamics for:
# - Sometimes removing numerical noise from data and maybe simulations.
# - Post-processing results for visualisation.
# - A self-consistent way of applying smooth properties on derivative information.
#
# We demonstrate two types of smoothing:
# 1. **Isotropic smoothing**: uniform diffusion in all directions
# 2. **Anisotropic smoothing**: direction-dependent diffusion
#
# The checkpoint file used in this tutorial contains a cylindrical temperature field
# from a 2D adjoint simulation. We use the same checkpoint file as an adjoint simulation
# to demonstrate smoothing techniques on a realistic geodynamic field.
# To download the checkpoint file if it doesn't already exist, execute the following command:

# + tags=["active-ipynb"]
# ![ ! -f adjoint-demo-checkpoint-state.h5 ] && wget https://data.gadopt.org/demos/smoothing-example.h5
# -

# How to do smoothing in G-ADOPT
# -------------------------
#
# The first step is to import the gadopt module, which provides access to
# Firedrake and associated functionality.

from gadopt import *

# We also import pyvista for doing file visualisations for this notebook.

# + tags=["active-ipynb"]
# import pyvista as pv
# -

# Next, load the cylindrical temperature field from a checkpoint file.
# This field serves as our test case for demonstrating smoothing techniques.
# If you are not familiar with these steps, the
# [base case for mantle convection](../mantle_convection/base_case) is a good place to start.

# +
input_file = "smoothing-example.h5"
with CheckpointFile(input_file, mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")

# Set mesh properties for cylindrical coordinates
mesh.cartesian = False
boundaries = get_boundary_ids(mesh)
# -

# Let's visualise the original temperature field to understand what we are working with.

# + tags=["active-ipynb"]
# VTKFile("original_temperature.pvd").write(T)
# temp_data = pv.read("original_temperature/original_temperature_0.vtu")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(temp_data, scalars="Temperature", cmap="coolwarm", scalar_bar_args={"width": 0.6, "height": 0.05, "position_x": 0.2, "position_y": 0.05})
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# The visualisation above shows the temperature field with sharp well-defined structures.
# Now to continue, we define boundary conditions for the temperature field during smoothing.
# These ensure that the smoothing operation respects the physical boundaries.
# Note that by removing these boundary conditions, the smooth properties would extend to
# the boundaries, however, the boundary values would not be conserved.
# For our field tags, we use 'g' to denote a general field. We typically use a
# Dirichlet boundary condition for smoothing, as other types like a Neumann
# boundary conditions are not necessary for smoothing.

temp_bcs = {
    boundaries.bottom: {'g': 1.0},  # Fixed temperature at the bottom
    boundaries.top: {'g': 0.0},     # Fixed temperature at the top
}

# Compute layer average of the temperature for comparison purposes.
# This helps visualise the deviation from the mean state before and after smoothing.

# +
T_avg = Function(T.function_space(), name='Temperature (Layer Averaged)')
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# Create a deviation field for better visualisation
T_deviation = Function(T.function_space(), name='Temperature (Deviation)')
T_deviation.assign(T - T_avg)
# -

# Isotropic Smoothing
# -------------------
# In isotropic smoothing, the diffusion coefficient is the same in all directions.
# This simplifies the diffusion tensor to a scalar value, providing a uniform
# smoothing across all spatial directions. In gadopt, the smoothing is performed by
# the DiffusiveSmoothingSolver class. Here by providing the solution function where we want to store
# the smoothed result, plus the wavelength that we want to smooth, we will have the DiffusiveSmoothingSolver.
# This DiffusiveSmoothingSolver will then be used to apply the smoothing to any scalar field.
# Below we use a wavelength of 0.2.

# +
smooth_solution_isotropic = Function(T.function_space(), name="Temperature (Isotropic Smoothed)")
smoother_isotropic = DiffusiveSmoothingSolver(
    smooth_solution_isotropic,
    wavelength=0.2,  # Smoothing wavelength parameter
    bcs=temp_bcs)

# Apply isotropic smoothing
smoother_isotropic.action(T)

# Compute the smoothed deviation for visualisation
smooth_deviation_isotropic = Function(T.function_space(), name='Temperature (Isotropic Smoothed Deviation)')
smooth_deviation_isotropic.assign(smooth_solution_isotropic - T_avg)
# -

# Now to see the result, we visualise the isotropic smoothing results

# + tags=["active-ipynb"]
# # Create visualisation comparing original vs isotropic smoothed deviation
# VTKFile("isotropic_results.pvd").write(T_deviation, smooth_solution_isotropic, smooth_deviation_isotropic)
#
# # Load the data for visualisation
# isotropic_data = pv.read("isotropic_results/isotropic_results_0.vtu")
#
# # Create a two-panel comparison: original vs isotropic smoothed deviation
# iso_plotter = pv.Plotter(shape=(1, 2), notebook=True)
#
# # Original temperature deviation
# iso_plotter.subplot(0, 0)
# mesh_copy_orig = isotropic_data.copy()
# iso_plotter.add_mesh(mesh_copy_orig, scalars="Temperature (Deviation)", cmap="RdBu_r", clim=[-0.4, 0.4], scalar_bar_args={"width": 0.6, "height": 0.05, "position_x": 0.2, "position_y": 0.05})
# iso_plotter.add_title("Original Deviation", font_size=10)
# iso_plotter.camera_position = "xy"
#
# # Isotropic smoothed deviation
# iso_plotter.subplot(0, 1)
# mesh_copy_iso = isotropic_data.copy()
# iso_plotter.add_mesh(mesh_copy_iso, scalars="Temperature (Isotropic Smoothed Deviation)", cmap="RdBu_r", clim=[-0.4, 0.4], scalar_bar_args={"width": 0.6, "height": 0.05, "position_x": 0.2, "position_y": 0.05})
# iso_plotter.add_title("Isotropic Smoothed Deviation", font_size=10)
# iso_plotter.camera_position = "xy"
#
# iso_plotter.show(jupyter_backend="static", interactive=False)
# -

# Anisotropic Smoothing
# ---------------------
# Anisotropic smoothing allows different diffusion rates in different directions. This
# is in particular useful for mantle studies where vertical and horizontal properties
# might have different resolutions.
# Here, we model zero radial diffusion and full tangential diffusion, which is
# physically relevant for layered structures in geodynamics.

# +
# Define the radial and tangential conductivity values
kr = Constant(0.0)  # Radial conductivity is zero, preventing diffusion in the radial direction
kt = Constant(1.0)  # Tangential conductivity is one, allowing diffusion in the tangential direction

# Compute radial vector components for cylindrical coordinates
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)
er = as_vector((X[0]/r, X[1]/r))  # Unit radial vector
et = as_vector((-X[1]/r, X[0]/r))  # Unit tangential vector

# Construct the anisotropic conductivity tensor
# K = kr * outer(er, er) + kt * outer(et, et)
# This defines how diffusion behaves differently in radial and tangential directions
K = kr * outer(er, er) + kt * outer(et, et)
# -

# +
smooth_solution_anisotropic = Function(T.function_space(), name="Temperature (Anisotropic Smoothed)")
smoother_anisotropic = DiffusiveSmoothingSolver(
    smooth_solution_anisotropic,
    wavelength=0.3,
    bcs=temp_bcs,
    K=K)

# Apply anisotropic smoothing
smoother_anisotropic.action(T)

# Compute the smoothed deviation for visualisation
smooth_deviation_anisotropic = Function(T.function_space(), name='Temperature (Anisotropic Smoothed Deviation)')
smooth_deviation_anisotropic.assign(smooth_solution_anisotropic - T_avg)
# -

# Comparison and Final visualisation
# ===================================
# Now we compare the effects of both smoothing approaches and visualise the results.

# +
# Write original and smoothed temperature fields for comparison
VTKFile("output.pvd").write(
    T, T_avg, T_deviation,
    smooth_solution_isotropic, smooth_deviation_isotropic,
    smooth_solution_anisotropic, smooth_deviation_anisotropic
)

# Create a parameter log to record key metrics
parameter_log = ParameterLog('params.log', mesh)
parameter_log.log_str("u_rms original_rms isotropic_rms anisotropic_rms")

# Calculate RMS values for comparison

original_rms = sqrt(assemble(T_deviation**2 * dx))
isotropic_rms = sqrt(assemble(smooth_deviation_isotropic**2 * dx))
anisotropic_rms = sqrt(assemble(smooth_deviation_anisotropic**2 * dx))

# Note: by default gadopts demos are testing for the u_rms values in the demos.
# So to keep this demo consistent with other demos, we set u_rms to 1.0 as a
# placeholder since smoothing doesn't involve velocity
parameter_log.log_str(f"{1.0:.6f} {original_rms:.6f} {isotropic_rms:.6f} {anisotropic_rms:.6f}")
parameter_log.close()

print("RMS Comparison:")
print(f"Original deviation RMS: {original_rms:.6f}")
print(f"Isotropic smoothed RMS: {isotropic_rms:.6f}")
print(f"Anisotropic smoothed RMS: {anisotropic_rms:.6f}")
# -

# Visualise the comprehensive comparison between all smoothing approaches

# + tags=["active-ipynb"]
# # Load the output data for visualisation
# output_data = pv.read("output/output_0.vtu")
#
# # Create a two-panel comparison: isotropic vs anisotropic smoothed deviations
# comparison_plotter = pv.Plotter(shape=(1, 2), notebook=True)
#
# # Isotropic smoothing deviation
# comparison_plotter.subplot(0, 0)
# mesh_copy_iso = output_data.copy()
# comparison_plotter.add_mesh(mesh_copy_iso, scalars="Temperature (Isotropic Smoothed Deviation)", cmap="RdBu_r", clim=[-0.4, 0.4], scalar_bar_args={"width": 0.6, "height": 0.05, "position_x": 0.2, "position_y": 0.05})
# comparison_plotter.add_title("Isotropic Smoothed Deviation", font_size=10)
# comparison_plotter.camera_position = "xy"
#
# # Anisotropic smoothing deviation
# comparison_plotter.subplot(0, 1)
# mesh_copy_aniso = output_data.copy()
# comparison_plotter.add_mesh(mesh_copy_aniso, scalars="Temperature (Anisotropic Smoothed Deviation)", cmap="RdBu_r", clim=[-0.4, 0.4], scalar_bar_args={"width": 0.6, "height": 0.05, "position_x": 0.2, "position_y": 0.05})
# comparison_plotter.add_title("Anisotropic Smoothed Deviation", font_size=10)
# comparison_plotter.camera_position = "xy"
#
# comparison_plotter.show(jupyter_backend="static", interactive=False)
# -

# This tutorial has demonstrated how to:
# - Load temperature fields from checkpoint files
# - Apply isotropic smoothing with uniform diffusion
# - Apply anisotropic smoothing with directional diffusion control
# - Compare the effects of different smoothing approaches
# - Visualise and quantify smoothing results
#
# The key difference between isotropic and tangential anisotropic smoothing in our example is evident:
# - Our isotropic smoothing reduced variations uniformly in all directions
# - Anisotropic smoothing, the way we used it only in the tangential direction,
#   preserves radial structure while smoothing tangentially
