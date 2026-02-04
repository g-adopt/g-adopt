# Lithosphere Indicator Reconstruction Through Geological Time
# =============================================================
#
# This tutorial demonstrates how to create a time-dependent 3D lithosphere
# indicator field using G-ADOPT's integration with gtrack. The indicator is
# ~1 inside the lithosphere and ~0 in the mantle, with a smooth tanh transition.
#
# The lithosphere base combines:
#
# 1. **Oceanic lithosphere**: Ages tracked forward in time using gtrack's
#    SeafloorAgeTracker, then converted to thickness using a cooling model.
# 2. **Continental lithosphere**: Present-day thickness data (e.g., from
#    seismic tomography) back-rotated to past positions.
#
# This example focuses on:
# 1. Setting up a `pyGplatesConnector` with polygon files for lithosphere tracking
# 2. Creating a `LithosphereConnector` that computes smooth indicator fields
# 3. Using `GplatesScalarFunction` to get a Firedrake-compatible indicator field
# 4. Using the indicator to modify viscosity in mantle convection simulations
#
# Prerequisites:
# - Working pyGPlates installation
# - gtrack package installed
# - Plate reconstruction files (download using Makefile)

# +
import h5py
import numpy as np

from gadopt import *
from gadopt.gplates import *

# Mesh parameters - higher resolution to resolve lithosphere
rmin, rmax, ref_level, nlayers = 1.208, 2.208, 6, 16

# Create non-uniform layer heights: finer near surface (lithosphere)
# Use geometric spacing with smallest layers at top
layer_heights = np.geomspace(0.02, 0.2, nlayers)[::-1]  # Reverse: thin at top, thick at bottom
layer_heights = layer_heights / layer_heights.sum()  # Normalize to sum to 1

mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, layer_height=layer_heights, extrusion_type="radial")
mesh.cartesian = False
boundary = get_boundary_ids(mesh)

# Scalar function space for lithosphere indicator
Q = FunctionSpace(mesh, "CG", 2)
# -

# ## Age-to-Thickness Conversion
#
# Oceanic lithosphere thickness is derived from seafloor age using a
# thermal cooling model. The half-space cooling model gives:
#
# $$z_L = 2.32 \sqrt{\kappa \cdot t}$$
#
# where $\kappa$ is thermal diffusivity (~10^-6 m^2/s) and $t$ is age in seconds.
# This gives approximately 100 km thickness for 80 Myr old seafloor.

# +


def half_space_cooling(age_myr, kappa=1e-6):
    """Convert seafloor age (Myr) to lithospheric thickness (km).

    Uses the half-space cooling model.

    Args:
        age_myr: Seafloor age in million years.
        kappa: Thermal diffusivity in m^2/s. Default 1e-6.

    Returns:
        Lithospheric thickness in km.
    """
    # Convert Myr to seconds
    age_sec = np.maximum(age_myr, 0) * 3.15576e13  # Myr to seconds
    # Half-space cooling: z = 2.32 * sqrt(kappa * t)
    thickness_m = 2.32 * np.sqrt(kappa * age_sec)
    # Convert to km and cap at reasonable maximum
    thickness_km = np.minimum(thickness_m / 1e3, 150.0)
    return thickness_km
# -

# ## Setting Up the Plate Reconstruction Model
#
# We use the Muller et al. (2022) plate reconstruction model. The
# `pyGplatesConnector` is extended with `continental_polygons` and
# `static_polygons` parameters needed for lithosphere tracking.
#
# Run `make data` in this directory to download the required files.


# +
muller_2022_files = ensure_reconstruction("Muller 2022 SE v1.2", ".")
# Create plate model connector with polygon files for lithosphere
plate_model = pyGplatesConnector(
    rotation_filenames=muller_2022_files["rotation_filenames"],
    topology_filenames=muller_2022_files["topology_filenames"],
    oldest_age=200,  # Start from 200 Ma for this demo
    nseeds=1e4,  # Coarser for demo
    nneighbours=4,
    delta_t=1.0,
    scaling_factor=1000.,  # Scale for low-Ra simulation
    # Polygon files for lithosphere tracking
    continental_polygons=muller_2022_files.get("continental_polygons"),
    static_polygons=muller_2022_files.get("static_polygons"),
)
# -

# ## Creating the Lithosphere Connector
#
# The `LithosphereConnector` wraps gtrack's components:
# - `SeafloorAgeTracker` for oceanic lithosphere ages
# - `PointRotator` for continental data rotation
# - `PolygonFilter` for continental/oceanic separation
#
# It produces a smooth 3D indicator field: ~1 inside lithosphere, ~0 in mantle.
#
# For this demo, we'll load continental thickness data from an HDF5 file.
# In practice, this would come from seismic tomography models like SL2013sv.

# +
# Load real continental thickness data from HDF5 file
continental_data_file = "/Users/sghelichkhani/Workplace/gtrack/examples/output/lithospheric_thickness_icosahedral.h5"
with h5py.File(continental_data_file, 'r') as f:
    lonlat = f['lonlat'][:]  # (N, 2) with (lon, lat) order
    thickness_values = f['values'][:]  # thickness in km

# Convert from (lon, lat) to (lat, lon) order for LithosphereConnector
latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])

# Create data tuple (latlon, values in km)
continental_data = (latlon, thickness_values)
# -

# ## LithosphereConfig: Configuration Parameters
#
# All tunable parameters for `LithosphereConnector` are grouped into
# `LithosphereConfig`. To inspect the default values:
#
#     from gadopt.gplates import LithosphereConnectorDefault
#     print(LithosphereConnectorDefault)
#
# ### Ocean Tracker Parameters
# Control how seafloor ages are tracked through time using gtrack:
#
# - `time_step` (default: 1.0) — Time step for age tracker in Myr
# - `n_points` (default: 10000) — Number of points in ocean tracker mesh;
#   higher values give finer resolution but slower computation
# - `reinit_interval_myr` (default: 50.0) — Reinitialize tracker every N Myr
#   to prevent numerical drift accumulation
#
# ### Interpolation Parameters
# Control how thickness values are looked up at mesh coordinates:
#
# - `k_neighbors` (default: 50) — Number of nearest neighbors for
#   inverse-distance weighted interpolation
# - `distance_threshold` (default: 0.1) — Maximum angular distance in radians
#   for valid interpolation; ~0.1 rad ≈ 570 km at Earth's surface
# - `default_thickness` (default: 100.0) — Thickness in km assigned to points
#   with no nearby source data (gaps)
#
# ### Mesh Geometry Parameters
# Must match your simulation mesh geometry:
#
# - `r_outer` (default: 2.208) — Outer radius of mesh in non-dimensional units;
#   this is the radial coordinate of Earth's surface
# - `depth_scale` (default: 2890.0) — Physical depth in km corresponding to
#   1 non-dimensional unit; for Earth's mantle this is 2890 km
# - `transition_width` (default: 10.0) — Width of tanh transition at
#   lithosphere base in km; controls smoothness of the indicator field
#
# ### Data Parameters
#
# - `property_name` (default: "thickness") — Name of the thickness property
#   when reading from data files
#
# ## Using config_extra to Override Defaults
#
# You can override specific parameters using the `config_extra` dictionary,
# following the same pattern as G-ADOPT's `solver_parameters_extra`.
# Only specify parameters you want to change — everything else keeps its
# default value.

# +
# Create the lithosphere connector with custom overrides
# Note: We pass comm=mesh.comm for efficient parallel execution.
# This ensures rank 0 handles all I/O and gtrack computations,
# then broadcasts results to other ranks. Without this, each MPI
# rank would independently load data and compute (wasteful).
lithosphere_connector = LithosphereConnector(
    gplates_connector=plate_model,
    continental_data=continental_data,
    age_to_property=half_space_cooling,
    config_extra={
        "r_outer": rmax,            # Match our mesh outer radius
        "n_points": 40000,          # Higher resolution ocean tracking
        "k_neighbors": 20,          # Fewer neighbors for faster interpolation
        "distance_threshold": 0.2,  # Wider search for sparse data (~1140 km)
    },
    comm=mesh.comm,  # Enable MPI parallelization
)
# -

# ## Creating the Firedrake Indicator Function
#
# `GplatesScalarFunction` extends `firedrake.Function` to automatically
# update from plate reconstruction data. It produces a smooth indicator
# field that can be used to modify viscosity or other material properties.

# +
# Create Firedrake function for lithosphere indicator
lithosphere_indicator = GplatesScalarFunction(
    Q,
    indicator_connector=lithosphere_connector,
    name="Lithosphere_Indicator"
)
# -

# ## Creating the Craton Indicator
#
# In addition to the general lithosphere indicator, we can create a
# separate indicator for cratonic regions. Cratons are the ancient,
# stable cores of continents with thick, cold lithospheric roots
# (typically 200-300 km deep).
#
# The `CratonConnector` works similarly to `LithosphereConnector`:
# 1. Uses craton polygon shapefiles to filter data to cratonic regions
# 2. Back-rotates the thickness data to past positions
# 3. Computes a depth-based 3D indicator (~1 inside craton, ~0 outside)
#
# We use the same continental thickness data - the CratonConnector
# automatically filters to only points inside craton polygons.

# +
# Path to craton shapefile (included with Muller 2022 reconstruction)
craton_shapefile = "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2/shapes_cratons.shp"

# Create craton connector
# Uses same thickness data as lithosphere - filtered to craton polygons
craton_connector = CratonConnector(
    gplates_connector=plate_model,
    craton_polygons=craton_shapefile,
    craton_thickness_data=continental_data,  # Same data, filtered to cratons
    config_extra={
        "r_outer": rmax,             # Match our mesh outer radius
        "k_neighbors": 1,            # Use nearest point only for sharp boundaries
        "distance_threshold": 0.15, # Small search radius (~95 km) - sharp horizontal cutoff
        "transition_width": 0.05,     # Sharp vertical transition at craton base (km)
    },
    comm=mesh.comm,  # Enable MPI parallelization
)

# Create Firedrake function for craton indicator
craton_indicator = GplatesScalarFunction(
    Q,
    indicator_connector=craton_connector,
    name="Craton_Indicator"
)
# -

# ## Updating Through Geological Time
#
# The `update_plate_reconstruction()` method updates the field to a new
# non-dimensional time. Internally, it:
# 1. Steps the ocean tracker forward
# 2. Rotates continental data backward
# 3. Computes lithosphere base at each (lon, lat)
# 4. Returns smooth indicator based on radial position
#
# The method uses caching - if time hasn't changed significantly
# (within `delta_t`), it returns the cached result.

# +
# Create a depth function in real Earth terms (km)
# depth = (rmax - r) * depth_scale, where r = sqrt(x^2 + y^2 + z^2)
depth_scale = 2890.0  # km per non-dimensional unit
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
depth_nondim = rmax - r
depth_km_expr = depth_nondim * depth_scale

depth_km = Function(Q, name="Depth_km")
depth_km.interpolate(depth_km_expr)
# -

# +
# Update to different geological ages and visualize
output_file = VTKFile("lithosphere_output.pvd")

# Times to output (geological ages in Ma)
output_ages = np.arange(200, -1, -10)

for age in output_ages:
    # Convert geological age to non-dimensional time
    ndtime = plate_model.age2ndtime(age)

    # TEMPORARILY DISABLED: Update lithosphere indicator (has gtrack bug at certain ages)
    lithosphere_indicator.update_plate_reconstruction(ndtime)

    # Update craton indicator
    craton_indicator.update_plate_reconstruction(ndtime)

    # Output to VTK (include depth for visualization)
    output_file.write(lithosphere_indicator, craton_indicator, depth_km)
    log(f"Written output for {age} Ma")
# -

# ## Visualization
#
# The output files can be visualized in ParaView. The lithosphere
# indicator field shows:
# - Values ~1 at the surface (inside lithosphere)
# - Values ~0 at depth (in mantle)
# - Thicker lithosphere (deeper ~1 values) under continents
# - Thinner lithosphere (shallower transition) at mid-ocean ridges

# + tags=["active-ipynb"]
# import pyvista as pv
# import os
#
# dataset = pv.read("lithosphere_output/lithosphere_output_0.vtu")
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# plotter.add_mesh(dataset, scalars="Lithosphere_Indicator", cmap="viridis")
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Using with Mantle Convection Simulations
#
# The primary use case is modifying viscosity. The indicator can be used to
# create a high-viscosity lithosphere that moves with the plates:
#
# ```python
# # Viscosity from indicator: 10^(indicator * 3)
# # - indicator = 0 (mantle) -> viscosity = 1
# # - indicator = 1 (lithosphere) -> viscosity = 1000
# viscosity = 10 ** (lithosphere_indicator * 3)
#
# # Use in approximation
# approximation = BoussinesqApproximation(Ra, mu=viscosity)
# ```
#
# In a full simulation, you would update the indicator at each time step:
#
# ```python
# for timestep in range(timesteps):
#     time += dt
#
#     # Update plate-derived fields
#     gplates_velocities.update_plate_reconstruction(time)
#     lithosphere_indicator.update_plate_reconstruction(time)
#
#     # Viscosity automatically updated through indicator
#     stokes_solver.solve()
#     energy_solver.solve()
# ```

# +
log("Demo complete. Output written to lithosphere_output.pvd")
# -
