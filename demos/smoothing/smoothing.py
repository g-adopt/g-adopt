# Isotropic and Anisotropic Smoothing in G-ADOPT
# ===============================================
#
# This tutorial demonstrates the application of **isotropic** and **anisotropic**
# diffusive smoothing techniques on a cylindrical temperature field using G-ADOPT.
# The purpose is to illustrate how different diffusion properties can affect the
# smoothing behavior in geodynamic simulations.
#
# Smoothing is a critical technique in geodynamics for:
# - Removing numerical noise from solutions
# - Stabilizing simulations with sharp gradients
# - Preparing initial conditions for time-dependent problems
# - Post-processing results for visualization
#
# We demonstrate two types of smoothing:
# 1. **Isotropic smoothing**: uniform diffusion in all directions
# 2. **Anisotropic smoothing**: directionally-dependent diffusion
#
# The checkpoint file used in this tutorial contains a cylindrical temperature field
# from a 2D adjoint simulation. We use the same checkpoint file as the adjoint tutorial
# to demonstrate smoothing techniques on a realistic geodynamic field.
# To download the checkpoint file if it doesn't already exist, execute the following command:

# + tags=["active-ipynb"]
# ![ ! -f adjoint-demo-checkpoint-state.h5 ] && wget https://data.gadopt.org/demos/adjoint-demo-checkpoint-state.h5
# -

# Implementation in G-ADOPT
# -------------------------
#
# The first step is to import the gadopt module, which provides access to
# Firedrake and associated functionality.

from gadopt import *

# We also import pyvista for doing file visualisations for this notebook.

# + tags=["active-ipynb"]
# import pyvista as pv
# -

# Load the cylindrical temperature field from a checkpoint file.
# This field serves as our test case for demonstrating smoothing techniques.

# +
input_file = "smoothing-example.h5"
with CheckpointFile(input_file, mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")

# Set mesh properties for cylindrical coordinates
mesh.cartesian = False
boundaries = get_boundary_ids(mesh)
# -

# Let's visualize the original temperature field to understand what we're working with.

# + tags=["active-ipynb"]
# VTKFile("original_temperature.pvd").write(T)
# temp_data = pv.read("original_temperature/original_temperature_0.vtu")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(temp_data, scalars="Temperature", cmap="coolwarm")
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# Define boundary conditions for the temperature field during smoothing.
# These ensure that the smoothing operation respects the physical boundaries.

temp_bcs = {
    boundaries.bottom: {'T': 1.0},  # Fixed temperature at the bottom
    boundaries.top: {'T': 0.0},     # Fixed temperature at the top
}

# Compute layer average of the temperature for comparison purposes.
# This helps visualize the deviation from the mean state before and after smoothing.

# +
T_avg = Function(T.function_space(), name='Layer_Averaged_Temp')
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# Create a deviation field for better visualization
T_deviation = Function(T.function_space(), name='Temperature_Deviation')
T_deviation.assign(T - T_avg)
# -

# Isotropic Smoothing
# -------------------
# In isotropic smoothing, the diffusion coefficient is the same in all directions.
# This simplifies the diffusion tensor to a scalar value, promoting uniform
# smoothing across all spatial directions.

# +
smooth_solution_isotropic = Function(T.function_space(), name="Smooth_Temperature_Isotropic")
smoother_isotropic = DiffusiveSmoothingSolver(
    function_space=T.function_space(),
    wavelength=0.1,  # Smoothing wavelength parameter
    bcs=temp_bcs)

# Apply isotropic smoothing
smooth_solution_isotropic.assign(smoother_isotropic.action(T))

# Compute the smoothed deviation for visualization
smooth_deviation_isotropic = Function(T.function_space(), name='Smooth_Deviation_Isotropic')
smooth_deviation_isotropic.assign(smooth_solution_isotropic - T_avg)
# -

# Anisotropic Smoothing
# ---------------------
# Anisotropic smoothing allows different diffusion rates in different directions.
# This is critical in scenarios where material properties vary spatially or by direction.
# Here, we model zero radial diffusion and full tangential diffusion, which is
# physically relevant for layered structures in geodynamics.

# +
# Define the radial and tangential conductivity values
kr = Constant(0.0)  # Radial conductivity is zero, preventing diffusion in the radial direction
kt = Constant(1.0)  # Tangential conductivity is one, allowing diffusion in the tangential direction

# Compute radial vector components for cylindrical coordinates
X = SpatialCoordinate(T.function_space().mesh())
r = sqrt(X[0]**2 + X[1]**2)
er = as_vector((X[0]/r, X[1]/r))  # Unit radial vector
et = as_vector((-X[1]/r, X[0]/r))  # Unit tangential vector

# Construct the anisotropic conductivity tensor
# K = kr * outer(er, er) + kt * outer(et, et)
# This defines how diffusion behaves differently in radial and tangential directions
K = kr * outer(er, er) + kt * outer(et, et)
# -

# +
smooth_solution_anisotropic = Function(T.function_space(), name="Smooth_Temperature_Anisotropic")
smoother_anisotropic = DiffusiveSmoothingSolver(
    function_space=T.function_space(),
    wavelength=0.1,
    bcs=temp_bcs,
    K=K)

# Apply anisotropic smoothing
smooth_solution_anisotropic.assign(smoother_anisotropic.action(T))

# Compute the smoothed deviation for visualization
smooth_deviation_anisotropic = Function(T.function_space(), name='Smooth_Deviation_Anisotropic')
smooth_deviation_anisotropic.assign(smooth_solution_anisotropic - T_avg)
# -

# Output results for visualization and analysis
# We write out multiple fields to compare the effects of different smoothing approaches

# +
# Write original and smoothed temperature fields
VTKFile("output.pvd").write(
    T, T_avg, T_deviation,
    smooth_solution_isotropic, smooth_deviation_isotropic,
    smooth_solution_anisotropic, smooth_deviation_anisotropic
)

# Create a parameter log to record key metrics
parameter_log = ParameterLog('params.log', mesh)
parameter_log.log_str("original_rms isotropic_rms anisotropic_rms")

# Calculate RMS values for comparison
original_rms = sqrt(assemble(T_deviation**2 * dx))
isotropic_rms = sqrt(assemble(smooth_deviation_isotropic**2 * dx))
anisotropic_rms = sqrt(assemble(smooth_deviation_anisotropic**2 * dx))

parameter_log.log_str(f"{original_rms} {isotropic_rms} {anisotropic_rms}")
parameter_log.close()
# -

# Visualize the comparison between original and smoothed fields

# + tags=["active-ipynb"]
# # Load the output data for visualization
# output_data = pv.read("output/output_0.vtu")
#
# # Create a comparison plot
# plotter = pv.Plotter(shape=(2, 2), notebook=True)
#
# # Original temperature deviation
# plotter.subplot(0, 0)
# plotter.add_mesh(output_data, scalars="Temperature_Deviation", cmap="RdBu_r", clim=[-0.5, 0.5])
# plotter.add_title("Original Temperature Deviation")
# plotter.camera_position = "xy"
#
# # Layer average
# plotter.subplot(0, 1)
# plotter.add_mesh(output_data, scalars="Layer_Averaged_Temp", cmap="coolwarm")
# plotter.add_title("Layer Average Temperature")
# plotter.camera_position = "xy"
#
# # Isotropic smoothing result
# plotter.subplot(1, 0)
# plotter.add_mesh(output_data, scalars="Smooth_Deviation_Isotropic", cmap="RdBu_r", clim=[-0.5, 0.5])
# plotter.add_title("Isotropic Smoothing")
# plotter.camera_position = "xy"
#
# # Anisotropic smoothing result
# plotter.subplot(1, 1)
# plotter.add_mesh(output_data, scalars="Smooth_Deviation_Anisotropic", cmap="RdBu_r", clim=[-0.5, 0.5])
# plotter.add_title("Anisotropic Smoothing")
# plotter.camera_position = "xy"
#
# plotter.show(jupyter_backend="static", interactive=False)
# -

# This tutorial has demonstrated how to:
# - Load temperature fields from checkpoint files
# - Apply isotropic smoothing with uniform diffusion
# - Apply anisotropic smoothing with directional diffusion control
# - Compare the effects of different smoothing approaches
# - Visualize and quantify smoothing results
#
# The key difference between isotropic and anisotropic smoothing is evident:
# - Isotropic smoothing reduces variations uniformly in all directions
# - Anisotropic smoothing preserves radial structure while smoothing tangentially
#
# These techniques are essential for preprocessing data, stabilizing numerical
# simulations, and post-processing results in geodynamic modeling.
