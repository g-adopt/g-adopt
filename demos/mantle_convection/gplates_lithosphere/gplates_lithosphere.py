# 3-D lithosphere and craton indicator fields from plate reconstructions
# ======================================================================
#
# This tutorial demonstrates how to create time-dependent 3-D indicator
# fields for the lithosphere and cratons using G-ADOPT's integration
# with [gtrack](https://pypi.org/project/gtrack/).  The indicator
# fields are smooth functions that are ~1 inside the relevant region
# and ~0 outside, connected by a tanh transition.
#
# The lithosphere indicator combines two components:
#
# 1. **Oceanic lithosphere**: Seafloor ages tracked forward through
#    geological time using gtrack's `SeafloorAgeTracker`, then
#    converted to thickness via a half-space cooling model.
# 2. **Continental lithosphere**: Present-day thickness observations
#    (e.g., from seismic tomography) back-rotated to past positions
#    using plate reconstruction Euler poles.
#
# The craton indicator identifies the ancient, stable cores of
# continents — regions with thick (~200–300 km), cold lithospheric
# roots — using polygon boundaries and the same thickness data.
#
# This tutorial builds on the [GPlates global
# demo](../gplates_global), which covers setting up `pyGplatesConnector`
# and working with plate reconstruction files.  You should follow that
# tutorial first.
#
# Prerequisites:
# - Working pyGPlates installation
# - gtrack package (`pip install gtrack`)
# - Data files (download via `make data` in this directory)

# +
import h5py
import numpy as np

from gadopt import *
from gadopt.gplates import *

rmin, rmax, ref_level, nlayers = 1.208, 2.208, 5, 16

# Non-uniform radial layers: finer near the surface to resolve the
# lithosphere and its base.  We use geometric spacing with the
# thinnest layers at the top.
layer_heights = np.geomspace(0.02, 0.2, nlayers)[::-1]
layer_heights = layer_heights / layer_heights.sum()

mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, layer_height=layer_heights, extrusion_type="radial")
mesh.cartesian = False
boundary = get_boundary_ids(mesh)

Q = FunctionSpace(mesh, "CG", 2)
# -

# ## Age-to-thickness conversion
#
# Oceanic lithosphere thickness is derived from seafloor age using
# the half-space cooling model:
#
# $$z_L = 2.32 \sqrt{\kappa \, t}$$
#
# where $\kappa$ is thermal diffusivity (~10$^{-6}$ m$^2$/s) and $t$
# is the seafloor age in seconds.  For 80 Myr old seafloor, this
# gives approximately 100 km of lithospheric thickness.  We cap the
# result at 150 km to avoid unrealistically large values.

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
    age_sec = np.maximum(age_myr, 0) * 3.15576e13  # Myr to seconds
    thickness_m = 2.32 * np.sqrt(kappa * age_sec)
    thickness_km = np.minimum(thickness_m / 1e3, 150.0)
    return thickness_km
# -


# ## Loading the plate reconstruction model
#
# We use the Müller et al. (2022) plate reconstruction model.  The
# `ensure_reconstruction` helper locates the required rotation and
# topology files, as well as `continental_polygons` and
# `static_polygons` that the `LithosphereConnector` needs to
# distinguish oceanic from continental crust.
#
# Run `make data` in this directory to download the reconstruction
# files before running this script.

# +
muller_2022_files = ensure_reconstruction("Muller 2022 SE v1.2", ".")

plate_model = pyGplatesConnector(
    rotation_filenames=muller_2022_files["rotation_filenames"],
    topology_filenames=muller_2022_files["topology_filenames"],
    oldest_age=200,
    nseeds=1e4,
    nneighbours=4,
    delta_t=1.0,
    scaling_factor=1000.,
    continental_polygons=muller_2022_files.get("continental_polygons"),
    static_polygons=muller_2022_files.get("static_polygons"),
)
# -

# ## Loading continental thickness data
#
# We load a continental lithospheric thickness dataset stored as an
# HDF5 file.  The file contains point coordinates in (longitude,
# latitude) order and corresponding thickness values in kilometres.
# The `LithosphereConnector` expects coordinates in (latitude,
# longitude) order, so we swap the columns after loading.
#
# This dataset is based on the work of Hoggard et al., which combines
# seismic tomography constraints with surface-wave data to estimate
# lithospheric thickness beneath the continents.

# +
with h5py.File("lithospheric_thickness_mesh.h5", "r") as f:
    lonlat = f["lonlat"][:]       # (N, 2) — (lon, lat)
    thickness_values = f["values"][:]  # thickness in km

# Convert (lon, lat) -> (lat, lon) for the connector
latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])
continental_data = (latlon, thickness_values)
# -

# ## Setting up the LithosphereConnector
#
# The `LithosphereConnector` wraps gtrack components that handle
# oceanic age tracking (`SeafloorAgeTracker`), continental data
# rotation (`PointRotator`), and ocean/continent separation
# (`PolygonFilter`).  It produces a smooth 3-D indicator field with
# values ~1 inside the lithosphere and ~0 in the underlying mantle.
#
# All tunable parameters are grouped in `LithosphereConfig`.  We
# override selected defaults using the `config_extra` dictionary —
# the same pattern used for `solver_parameters` elsewhere in
# G-ADOPT.  Passing `comm=mesh.comm` ensures that rank 0 handles
# all I/O and gtrack computations, then broadcasts results to the
# remaining MPI ranks.

# +
lithosphere_connector = LithosphereConnector(
    gplates_connector=plate_model,
    continental_data=continental_data,
    age_to_property=half_space_cooling,
    config_extra={
        "r_outer": rmax,
        "n_points": 10000,
        "k_neighbors": 20,
        "distance_threshold": 0.2,
        "reinit_interval_myr": 20,
    },
    comm=mesh.comm,
)
# -

# ## Creating the lithosphere indicator function
#
# `GplatesScalarFunction` extends `firedrake.Function` so that its
# values can be updated from plate-reconstruction data at each time
# step.  The resulting field can be used directly in UFL expressions
# — for instance, to define a depth-dependent viscosity.

# +
lithosphere_indicator = GplatesScalarFunction(
    Q,
    indicator_connector=lithosphere_connector,
    name="Lithosphere_Indicator",
)
# -

# ## Setting up the CratonConnector
#
# Cratons are the ancient, stable cores of continents.  Their thick
# (~200–300 km), cold lithospheric roots are thought to protect them
# from tectonic reworking over billions of years.  The
# `CratonConnector` identifies cratonic regions by filtering
# thickness data through craton polygon boundaries provided as a
# shapefile.  The resulting indicator is ~1 inside the craton root
# and ~0 elsewhere.
#
# We download craton boundary polygons from the [EarthByte Craton
# Boundaries repository](https://github.com/EarthByte/Craton_Boundaries_Inferred)
# (via `make data`).

# +
craton_connector = CratonConnector(
    gplates_connector=plate_model,
    craton_polygons="Craton_Boundaries_Inferred.shp",
    craton_thickness_data=continental_data,
    config_extra={
        "r_outer": rmax,
        "k_neighbors": 1,
        "distance_threshold": 0.15,
        "transition_width": 0.05,
    },
    comm=mesh.comm,
)

craton_indicator = GplatesScalarFunction(
    Q,
    indicator_connector=craton_connector,
    name="Craton_Indicator",
)
# -

# ## Indicator fields at 200 Ma
#
# We begin by computing the indicator fields at 200 Ma — the oldest
# age in our reconstruction.  The `update_plate_reconstruction`
# method advances the ocean age tracker, rotates the continental
# data, computes lithosphere-base depth at each surface point, and
# returns a smooth indicator based on radial position.

# +
output_file = VTKFile("lithosphere_output.pvd")

ndtime = plate_model.age2ndtime(200)
lithosphere_indicator.update_plate_reconstruction(ndtime)
craton_indicator.update_plate_reconstruction(ndtime)
output_file.write(lithosphere_indicator, craton_indicator)
log("Written output for 200 Ma")
# -

# We extract contour isosurfaces at a threshold of 0.8 for both
# indicators.  The lithosphere base (blue) marks where the indicator
# drops below 0.8, while the craton roots (red) highlight the deep
# keels beneath ancient continental cores.

# + tags=["active-ipynb"]
# import pyvista as pv
# import os
#
# reader = pv.get_reader("lithosphere_output.pvd")
# reader.set_active_time_point(0)
# dataset = reader.read()[0]
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# lith_base = dataset.contour(isosurfaces=[0.8], scalars="Lithosphere_Indicator")
# plotter.add_mesh(lith_base, color="steelblue", opacity=0.6, label="Lithosphere base")
#
# craton_base = dataset.contour(isosurfaces=[0.8], scalars="Craton_Indicator")
# plotter.add_mesh(craton_base, color="firebrick", opacity=0.8, label="Craton base")
#
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Indicator fields at 100 Ma
#
# Advancing to 100 Ma, the ocean age tracker has evolved the
# seafloor ages and the continental blocks have moved to their
# mid-Cretaceous positions.

# +
ndtime = plate_model.age2ndtime(100)
lithosphere_indicator.update_plate_reconstruction(ndtime)
craton_indicator.update_plate_reconstruction(ndtime)
output_file.write(lithosphere_indicator, craton_indicator)
log("Written output for 100 Ma")
# -

# + tags=["active-ipynb"]
# reader = pv.get_reader("lithosphere_output.pvd")
# reader.set_active_time_point(1)
# dataset = reader.read()[0]
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# lith_base = dataset.contour(isosurfaces=[0.8], scalars="Lithosphere_Indicator")
# plotter.add_mesh(lith_base, color="steelblue", opacity=0.6, label="Lithosphere base")
#
# craton_base = dataset.contour(isosurfaces=[0.8], scalars="Craton_Indicator")
# plotter.add_mesh(craton_base, color="firebrick", opacity=0.8, label="Craton base")
#
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Indicator fields at present day
#
# Finally, we compute the indicators at the present day (0 Ma).
# The oceanic lithosphere now reflects its current age distribution,
# and the continental blocks are in their observed positions.

# +
ndtime = plate_model.age2ndtime(0)
lithosphere_indicator.update_plate_reconstruction(ndtime)
craton_indicator.update_plate_reconstruction(ndtime)
output_file.write(lithosphere_indicator, craton_indicator)
log("Written output for 0 Ma")
# -

# + tags=["active-ipynb"]
# reader = pv.get_reader("lithosphere_output.pvd")
# reader.set_active_time_point(2)
# dataset = reader.read()[0]
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# lith_base = dataset.contour(isosurfaces=[0.8], scalars="Lithosphere_Indicator")
# plotter.add_mesh(lith_base, color="steelblue", opacity=0.6, label="Lithosphere base")
#
# craton_base = dataset.contour(isosurfaces=[0.8], scalars="Craton_Indicator")
# plotter.add_mesh(craton_base, color="firebrick", opacity=0.8, label="Craton base")
#
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Using indicators in mantle convection simulations
#
# The primary application of these indicator fields is modifying
# material properties — in particular, viscosity.  A lithospheric
# indicator that is ~1 at the surface and ~0 at depth allows
# straightforward construction of a high-viscosity lid that moves
# with the plates:
#
# ```python
# # Viscosity contrast of 10^3 between lithosphere and mantle
# viscosity = 10 ** (lithosphere_indicator * 3)
# approximation = BoussinesqApproximation(Ra, mu=viscosity)
# ```
#
# In a time-stepping simulation, you update the indicators together
# with the plate velocities:
#
# ```python
# for timestep in range(timesteps):
#     time += dt
#     gplates_velocities.update_plate_reconstruction(time)
#     lithosphere_indicator.update_plate_reconstruction(time)
#     craton_indicator.update_plate_reconstruction(time)
#
#     stokes_solver.solve()
#     energy_solver.solve()
# ```
#
# See the [GPlates global demo](../gplates_global) for the full
# simulation setup including boundary conditions, nullspaces, and
# time-step adaptation.
