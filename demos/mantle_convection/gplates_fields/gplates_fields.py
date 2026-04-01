# 3-D plate-reconstruction fields: indicators and thermal initial condition
# =========================================================================
#
# This tutorial demonstrates how to create time-dependent 3-D fields from
# plate reconstructions using G-ADOPT's integration with
# [gtrack](https://pypi.org/project/gtrack/).  It covers two capabilities
# in a single workflow:
#
# 1. **Indicator fields**: smooth functions (1 inside a region, 0
#    outside) for the lithosphere, continental crust, and cratons.
# 2. **Thermal initial condition**: a composed temperature field that
#    blends oceanic and continental geotherms using the indicators as
#    weights.
#
# The lithosphere indicator combines two components:
#
# - **Oceanic lithosphere**: Seafloor ages tracked forward through
#   geological time using [gtrack's](https://pypi.org/project/gtrack/)
#   `SeafloorAgeTracker`, then converted to thickness via a half-space
#   cooling model.
# - **Continental lithosphere**: Present-day thickness observations
#   (e.g., from seismic tomography) back-rotated to past positions
#   using plate reconstruction Euler poles.
#
# The craton indicator identifies the ancient, stable cores of
# continents; regions with thick (~200-300 km), cold lithospheric
# roots, using polygon boundaries and the same thickness data.
#
# Forward mantle convection models need a thermal initial condition,
# and the lithosphere is the one part of the mantle where we have
# direct observational constraints: seafloor ages from magnetic
# anomalies for the oceans, and seismic tomography for the
# continents.  Deeper mantle thermal structure is largely unknown
# and is typically set to a uniform or adiabatic background.  The
# approach here therefore builds the initial temperature from the
# lithosphere downward, using the indicator fields as blending
# weights between lithospheric geotherms and the mantle background.
#
# The temperature composition follows Rhodri's formula:
#
# $$T_{\text{litho}} = T_{\text{lin}} \, I_{\text{cont}}
#                     + T_{\text{erf}} \, (1 - I_{\text{cont}})$$
#
# $$T = T_s + (T_{\text{lab}} - T_s)
#       \bigl(T_{\text{litho}} \, I_{\text{lith}}
#            + T_{\text{bg,norm}} \, (1 - I_{\text{lith}})\bigr)$$
#
# where $I_{\text{lith}}$ is the lithosphere indicator, $I_{\text{cont}}$
# is the continental indicator, $T_{\text{erf}}$ is the oceanic erf
# geotherm, $T_{\text{lin}}$ is the continental linear geotherm, $T_s$ is
# the surface temperature, and $T_{\text{lab}}$ is the temperature at the
# base of the lithosphere.
#
# This tutorial builds on the [GPlates global
# demo](../gplates_global), which covers setting up `pyGplatesConnector`
# and working with plate reconstruction files.  You should follow that
# tutorial first.
#
# Prerequisites:
# - Working pyGPlates installation
# - gtrack package (`pip install gtrack`)
# - Data files: the Muller et al. (2022) plate reconstruction,
#   continental lithospheric thickness from Hoggard et al. (2020),
#   and craton boundary shapefiles from Shirmard et al. (2025).
#   Running `make data` in this directory downloads all three
#   automatically; see the individual sections below for manual
#   download links.
#
# The `h5py` and `numpy` imports below are part of G-ADOPT's
# dependency set and do not require separate installation.

# +
import h5py
import numpy as np

from gadopt import *
from gadopt.gplates import *

rmin, rmax, ref_level, nlayers = 1.208, 2.208, 5, 32
# -

# As in the [3-D spherical tutorial](../3d_spherical), radii are
# non-dimensionalised by the CMB radius $R_{\text{CMB}} \approx 3480$
# km, giving $r_{\text{min}} = 1.208$ (CMB) and $r_{\text{max}} =
# 2.208$ (Earth's surface), so the non-dimensional mantle depth is
# $r_{\text{max}} - r_{\text{min}} = 1$.  This depth corresponds to
# roughly 2891 km, which defines the *depth scale* used throughout:
# any quantity expressed in kilometres (lithospheric thickness,
# transition widths, etc.) is converted to non-dimensional radial
# coordinates by dividing by 2891.

# +
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
# the half-space cooling model.  The temperature at depth $d$ below
# the surface in a cooling half-space of age $t$ is
#
# $$T(d, t) = T_s + (T_m - T_s) \, \operatorname{erf}\!\left(
#     \frac{d}{2\sqrt{\kappa \, t}}\right)$$
#
# where $T_s$ is the surface temperature, $T_m$ the mantle potential
# temperature (~1450 degrees C), and $\kappa$ the thermal diffusivity
# (~10$^{-6}$ m$^2$/s).  The base of the lithosphere is
# conventionally defined by the ~1300 degrees C isotherm, which
# corresponds to roughly 90 % of $T_m$.  Inverting for the depth at
# which this isotherm is reached gives
#
# $$d_L = 2\,\operatorname{erf}^{-1}(0.9)\,\sqrt{\kappa \, t}
#        \approx 2.32\,\sqrt{\kappa \, t}$$
#
# For 80 Myr old seafloor this yields approximately 100 km of
# lithospheric thickness.  We limit the maximum lithospheric
# thickness to 150 km to avoid unrealistically large values for the
# oldest ocean floor.

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
# We use the Muller et al. (2022) plate reconstruction model,
# available from the
# [EarthByte data collection](https://earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2022_SE/).
# Download and unzip the
# `Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.zip` archive
# into this directory (or run `make data`, which does this
# automatically).  The `ensure_reconstruction` helper then locates
# the required rotation and topology files, as well as
# `continental_polygons` and `static_polygons` that the connectors
# need to distinguish oceanic from continental crust.

# +
muller_2022_files = ensure_reconstruction("Muller 2022 SE v1.2", ".")

plate_model = pyGplatesConnector(
    rotation_filenames=muller_2022_files["rotation_filenames"],
    topology_filenames=muller_2022_files["topology_filenames"],
    oldest_age=500,
    nseeds=1e3,
    nneighbours=4,
    delta_t=10.0,
    scaling_factor=1000.,
    continental_polygons=muller_2022_files.get("continental_polygons"),
    static_polygons=muller_2022_files.get("static_polygons"),
)
# -

# ## Loading continental thickness data
#
# We load a continental lithospheric thickness dataset stored as an
# HDF5 file.  The data come from
# [Hoggard et al. (2020)](https://doi.org/10.1038/s41561-020-0593-2),
# who mapped the thermal lithosphere-asthenosphere boundary (LAB)
# globally by converting shear-wave velocities from the SL2013sv
# tomography model to temperature using a calibrated anelasticity
# parameterisation.  The LAB is defined as a thermal isotherm from
# these converted temperatures.  The original gridded data is
# re-sampled onto a Fibonacci-spiral sphere mesh of ~40 000
# uniformly spaced points using the `interpolate_to_mesh.py` utility
# in the [gtrack examples](https://pypi.org/project/gtrack/), which
# avoids the pole-clustering inherent in regular lat/lon grids.
# The resulting file is hosted on the G-ADOPT data server and can be
# downloaded from
# `https://data.gadopt.org/demos/continental_lithospheric_thickness_mesh.h5`
# (or via `make data`).
#
# The HDF5 file contains point coordinates in (longitude, latitude)
# order and corresponding thickness values in kilometres.  The
# connectors expect coordinates in (latitude, longitude) order, so
# we swap the columns after loading.

# +
with h5py.File("continental_lithospheric_thickness_mesh.h5", "r") as f:
    lonlat = f["lonlat"][:]       # (N, 2) -- (lon, lat)
    thickness_values = f["values"][:]  # thickness in km

# Convert (lon, lat) -> (lat, lon) for the connector
latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])
continental_data = (latlon, thickness_values)
# -

# ## Shared configuration
#
# All indicator and geotherm connectors share the same interpolation
# parameters.  We define them once here.
#
# `r_outer` is the outer radius of the mesh in non-dimensional units
# (same as `rmax`).  `n_points` controls how many source points
# gtrack samples on the sphere -- more points give finer spatial
# resolution at linearly increasing cost; 20 000 is a reasonable
# choice for a refinement-level-5 cubed-sphere mesh.  `k_neighbors`
# is the number of nearest neighbours used in the inverse-distance
# weighted (IDW) interpolation from source points to mesh degrees of
# freedom; cost per target point scales linearly with this value, and
# 50 gives smooth results without excessive blurring.
#
# `distance_threshold` is the maximum angular distance, in radians
# on the unit sphere, at which a target point is still considered
# "close enough" to the source data.  Points farther away receive an
# a-priori thickness (for `LithosphereConnector`, coming from
# whatever distance from the ridge) or zero (for
# `PolygonConnector`).  As a rough guide, 0.15 rad is approximately
# 960 km on Earth's surface; smaller values produce sharper
# boundaries but may leave gaps if the source mesh is sparse.
#
# `transition_width` sets the width of the tanh smoothing at the
# base of the lithosphere, in kilometres.  This value is converted
# internally to non-dimensional units by dividing by the depth scale
# (2891 km).  A width of 10 km gives a sharp but numerically stable
# transition between the lithosphere indicator (~1) and the mantle
# (~0).
#
# `gtrack_config` is a pass-through dictionary that forwards
# parameters directly to gtrack's `TracerConfig`.  This gives access
# to the full set of ocean-tracker knobs without duplicating them in
# G-ADOPT: `ridge_sampling_degrees` controls how densely mid-ocean
# ridges are tessellated (default 0.5 degrees, ~50 km; larger values
# seed fewer points per timestep), `velocity_delta_threshold` and
# `distance_threshold_per_myr` tune collision detection at subduction
# zones, `continental_cache_size` sets how many timesteps of polygon
# queries are cached, and the `reinit_*` parameters control the
# periodic reinitialization that redistributes tracers evenly on the
# sphere.  Any parameter accepted by `TracerConfig` can be set here;
# see the [gtrack documentation](https://pypi.org/project/gtrack/)
# for the full list.
#
# **Ocean tracker checkpointing.**  When the `LithosphereConnector`
# first computes an indicator field, rank 0 must initialise the
# `SeafloorAgeTracker` at `oldest_age` and step it forward to the
# requested reconstruction age -- a sequential process during which
# every other MPI rank sits idle at the broadcast.  Setting
# `checkpoint_interval_myr` tells the connector to periodically save
# the tracker state (tracer positions and material ages) to `.npz`
# files.  On a subsequent run the connector scans the checkpoint
# directory, loads the file closest to the first requested age, and
# only steps forward from there.  The result is identical to a
# continuous run, but the wall-clock saving can be substantial on
# large MPI jobs.  Checkpoint files are config-agnostic -- they store
# tracer positions and ages, not the `LithosphereConfig` parameters
# -- so changing `n_points`, `k_neighbors`, or other settings between
# runs is fine.  A failed checkpoint write (e.g. permission error) is
# logged but never crashes the simulation.

# +
connector_config = {
    "r_outer": rmax,
    "n_points": 5000,
    "k_neighbors": 20,
    "distance_threshold": 0.02,
    "kernel": "idw",
    "transition_width": 10.0,
    "reinit_interval_myr": 10.0,
    "checkpoint_interval_myr": 10.0,
    "checkpoint_dir": "./ocean_checkpoints",
    "gtrack_config": {
        "time_step": 2.0,  # internal gtrack timestep (Myr)
        "earth_radius": 6.3781e6,  # Earth radius (m)
        "velocity_delta_threshold": 7.0,  # collision velocity threshold (km/Myr)
        "distance_threshold_per_myr": 10.0,  # collision distance threshold (km/Myr)
        "default_mesh_points": 5000,  # initial Fibonacci sphere mesh points
        "initial_ocean_mean_spreading_rate": 75.0,  # the spreading rate we use at initial age (mm/yr)
        "ridge_sampling_degrees": 2.0,  # ridge tessellation resolution (degrees)
        "spreading_offset_degrees": 0.01, # offset from ridge for new seeds (degrees)
        "reinit_k_neighbors": 5,  # KNN neighbours during reinitialisation
        "reinit_max_distance": 500e3,  # max interpolation distance (m)
    },
}
# -

# ## Part 1: Indicator fields
#
# We create three indicator fields that define *where* different
# regions are in the mantle domain.  Each field is ~1 inside the
# target region and ~0 outside, connected by a smooth tanh
# transition.
#
# The workflow uses two objects per field.  A *connector*
# (`LithosphereConnector`, `PolygonConnector`, etc.) talks to gtrack
# and the plate reconstruction to produce thickness or geotherm data
# on a set of source points at each reconstruction age.  A
# `GplatesScalarFunction` wraps that connector in a standard
# Firedrake `Function` -- just as `GplatesVelocityFunction` does for
# surface velocities in the [GPlates global demo](../gplates_global).
# For all practical purposes these are ordinary Firedrake functions
# that can be used directly in UFL expressions; they simply know how
# to refresh their values from plate-reconstruction data via
# `update_plate_reconstruction(ndtime)`.

# ### Lithosphere indicator
#
# The `LithosphereConnector` wraps gtrack components that handle
# oceanic age tracking (`SeafloorAgeTracker`), continental data
# rotation (`PointRotator`), and ocean/continent separation
# (`PolygonFilter`).  It produces a smooth 3-D indicator field with
# values ~1 inside the lithosphere and ~0 in the underlying mantle.

# +
I_lith_connector = LithosphereConnector(
    gplates_connector=plate_model,
    continental_data=continental_data,
    age_to_property=half_space_cooling,
    config_extra=connector_config,
    comm=mesh.comm,
)

I_lith = GplatesScalarFunction(Q, indicator_connector=I_lith_connector, name="I_lith")
# -

# ### Continental indicator
#
# The `PolygonConnector` accepts either a `(coords, values)` tuple
# for spatially varying thickness or a single scalar for uniform
# thickness.  For example, passing `thickness_data=50.0` would give
# a uniform 50 km crustal layer everywhere inside the polygons --
# useful for representing the density deficit of continental crust.
# Here we use the full `continental_data` so that the indicator depth
# varies with the observed lithospheric thickness, which is the
# version used later for geotherm blending.

# +
I_cont_connector = PolygonConnector(
    gplates_connector=plate_model,
    polygons=muller_2022_files.get("continental_polygons"),
    thickness_data=continental_data,
    config_extra=connector_config,
    comm=mesh.comm,
)

I_cont = GplatesScalarFunction(Q, indicator_connector=I_cont_connector, name="I_cont")
# -

# ### Continental crust indicator
#
# The continental crust (top ~50 km of continental regions) is less
# dense than the mantle (~2700 vs ~3200 kg/m^3).  We model this as a
# uniform-thickness layer identified by the plate model's continental
# polygons.  In a full simulation this indicator drives an upward
# buoyancy force that represents the density deficit of continental
# crust relative to the mantle.  Unlike the continental indicator
# above, which uses the full spatially varying thickness for geotherm
# blending, this one uses a constant 50 km for the crustal layer.

# +
I_crust_connector = PolygonConnector(
    gplates_connector=plate_model,
    polygons=muller_2022_files.get("continental_polygons"),
    thickness_data=50.0,
    config_extra=connector_config,
    comm=mesh.comm,
)

I_crust = GplatesScalarFunction(Q, indicator_connector=I_crust_connector, name="I_crust")
# -

# ### Craton indicator
#
# Cratons are the ancient, stable cores of continents.  Their thick
# (~200-300 km), cold lithospheric roots are thought to protect them
# from tectonic reworking over billions of years.  The craton
# boundary polygons used here come from
# [Shirmard et al. (2025)](https://doi.org/10.1016/j.gsf.2025.102176),
# who delineated craton boundaries by applying unsupervised machine
# learning (PCA and k-means clustering) to horizontal shear-wave
# velocities from the REVEAL full-waveform tomography model,
# combined with lithospheric thickness and tectonic age constraints.
# The shapefiles are available from the
# [EarthByte Craton_Boundaries repository](https://github.com/EarthByte/Craton_Boundaries)
# on GitHub (or via `make data`).
#
# The `PolygonConnector` identifies cratonic regions by filtering
# thickness data through these polygon boundaries.  The resulting
# indicator is ~1 inside the craton root and ~0 elsewhere.

# +
I_craton_connector = PolygonConnector(
    gplates_connector=plate_model,
    polygons="Craton_Boundaries_Inferred.shp",
    thickness_data=continental_data,
    config_extra=connector_config,
    comm=mesh.comm,
)

I_craton = GplatesScalarFunction(Q, indicator_connector=I_craton_connector, name="I_craton")
# -

# ## Part 2: Geotherm fields
#
# We now create two geotherm fields that produce normalized
# temperature profiles with values in [0, 1], where 0 corresponds to
# the surface temperature $T_s$ and 1 to the LAB temperature
# $T_{\text{lab}}$.  The composition in Part 3 converts these to
# physical temperature via $T = T_s + (T_{\text{lab}} - T_s) \times
# T_{\text{normalized}}$.  The oceanic erf profile depends on seafloor
# age -- younger ocean has a thinner thermal boundary layer -- while
# the continental linear profile simply increases linearly from
# surface to the base of the lithosphere.  Outside the lithosphere
# (where $I_{\text{lith}} \approx 0$), the normalized background is
# set to 1 ($= T_{\text{lab}}$).  In a production simulation you
# would typically replace this uniform background with an adiabatic
# mantle geotherm or a 3-D temperature field from a previous run.
#
# Each geotherm wraps an existing indicator connector rather than
# creating its own tracking and rotation infrastructure.
# `LithosphereGeotherm` reuses `I_lith_connector`'s ocean age tracker,
# and `PolygonGeotherm` reuses `I_cont_connector`'s polygon rotator.
# This avoids duplicating the expensive `SeafloorAgeTracker` and
# ensures that both the indicator and the geotherm see exactly the
# same reconstructed data.
#
# Because the ocean age tracker can only evolve forward in time
# (decreasing geological age toward the present), the indicator
# connector must be updated before the geotherm at each time step.
# If your workflow requires independent ordering or different
# interpolation parameters for the geotherm, create a separate
# `LithosphereConnector` instead.

# +
T_erf_connector = LithosphereGeotherm(
    I_lith_connector,
    geotherm=ocean_erf_normalized,
    kappa=1e-6,
)

T_erf = GplatesScalarFunction(Q, indicator_connector=T_erf_connector, name="T_erf")

T_lin_connector = PolygonGeotherm(
    I_cont_connector,
    geotherm=continental_linear,
)

T_lin = GplatesScalarFunction(Q, indicator_connector=T_lin_connector, name="T_lin")
# -

# ## Part 3: Temperature composition
#
# Physical temperature parameters.  In this demo the background
# mantle temperature equals the LAB temperature, so the normalized
# background value is simply 1.  The composition blends the
# continental linear profile with the oceanic erf profile according
# to the continental indicator, then blends the resulting
# lithospheric temperature with the mantle background according to
# the lithosphere indicator.

# +
Ts = 273.0     # Surface temperature (K)
Tlab = 1573.0  # LAB temperature (K)

T_litho = Function(Q, name="T_litho")
T = Function(Q, name="Temperature")
# -

# ## Fields at 200 Ma
#
# We begin by computing all fields at 200 Ma -- the oldest age in our
# reconstruction.  Each update call advances the ocean age tracker,
# rotates the continental data, and computes the field values at the
# appropriate palaeoposition.

# +
output_file = VTKFile("gplates_fields_output.pvd")

plog = ParameterLog("params.log", mesh)
plog.log_str("age I_lith_int I_cont_int I_crust_int I_craton_int T_avg T_min T_max")

ndtime = plate_model.age2ndtime(200)
I_lith.update_plate_reconstruction(ndtime)
I_cont.update_plate_reconstruction(ndtime)
I_crust.update_plate_reconstruction(ndtime)
I_craton.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
T_lin.update_plate_reconstruction(ndtime)

T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))

output_file.write(T, T_litho, I_lith, I_cont, I_crust, I_craton, T_erf, T_lin)

with T.dat.vec_ro as v:
    T_min, T_max = v.min()[1], v.max()[1]
plog.log_str(f"{plate_model.ndtime2age(ndtime)} {assemble(I_lith * dx)} "
             f"{assemble(I_cont * dx)} {assemble(I_crust * dx)} "
             f"{assemble(I_craton * dx)} {assemble(T * dx)} {T_min} {T_max}")
log("Written output for 200 Ma")
# -

# We extract the lithosphere base isosurface (indicator = 0.5) and
# colour it by its radial depth, so the shape of the lithosphere
# base is displayed as a surface on the sphere where the colour
# tells you how deep the base extends at each point.  Craton and
# continental crust isosurfaces at 0.8 are overlaid for context.

# + tags=["active-ipynb"]
# import pyvista as pv
# import os
#
# reader = pv.get_reader("gplates_fields_output.pvd")
# reader.set_active_time_point(0)
# dataset = reader.read()[0]
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# r_outer = rmax
# lith_iso = dataset.contour(isosurfaces=[0.5], scalars="I_lith")
# if lith_iso.n_points > 0:
#     pts = lith_iso.points
#     depth = r_outer - np.sqrt(np.sum(pts**2, axis=1))
#     lith_iso["Depth"] = depth
#     plotter.add_mesh(lith_iso, scalars="Depth", cmap="viridis",
#                      opacity=0.7, scalar_bar_args={"title": "Depth (non-dim)"})
#     contour_vals = np.arange(0.02, depth.max() + 0.02, 0.02)
#     if len(contour_vals) > 0:
#         contours = lith_iso.contour(isosurfaces=contour_vals.tolist(), scalars="Depth")
#         if contours.n_points > 0:
#             plotter.add_mesh(contours, color="black", line_width=2)
#
# craton_base = dataset.contour(isosurfaces=[0.8], scalars="I_craton")
# if craton_base.n_points > 0:
#     plotter.add_mesh(craton_base, color="firebrick", opacity=0.8, label="Craton")
#
# cont_base = dataset.contour(isosurfaces=[0.8], scalars="I_cont")
# if cont_base.n_points > 0:
#     plotter.add_mesh(cont_base, color="sandybrown", opacity=0.5, label="Continental")
#
# plotter.add_legend()
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Fields at 100 Ma
#
# Advancing to 100 Ma, the ocean age tracker has evolved the
# seafloor ages and the continental blocks have moved to their
# mid-Cretaceous positions.

# +
ndtime = plate_model.age2ndtime(100)
I_lith.update_plate_reconstruction(ndtime)
I_cont.update_plate_reconstruction(ndtime)
I_crust.update_plate_reconstruction(ndtime)
I_craton.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
T_lin.update_plate_reconstruction(ndtime)

T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))

output_file.write(T, T_litho, I_lith, I_cont, I_crust, I_craton, T_erf, T_lin)

with T.dat.vec_ro as v:
    T_min, T_max = v.min()[1], v.max()[1]
plog.log_str(f"{plate_model.ndtime2age(ndtime)} {assemble(I_lith * dx)} "
             f"{assemble(I_cont * dx)} {assemble(I_crust * dx)} "
             f"{assemble(I_craton * dx)} {assemble(T * dx)} {T_min} {T_max}")
log("Written output for 100 Ma")
# -

# + tags=["active-ipynb"]
# reader = pv.get_reader("gplates_fields_output.pvd")
# reader.set_active_time_point(1)
# dataset = reader.read()[0]
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# r_outer = rmax
# lith_iso = dataset.contour(isosurfaces=[0.5], scalars="I_lith")
# if lith_iso.n_points > 0:
#     pts = lith_iso.points
#     depth = r_outer - np.sqrt(np.sum(pts**2, axis=1))
#     lith_iso["Depth"] = depth
#     plotter.add_mesh(lith_iso, scalars="Depth", cmap="viridis",
#                      opacity=0.7, scalar_bar_args={"title": "Depth (non-dim)"})
#     contour_vals = np.arange(0.02, depth.max() + 0.02, 0.02)
#     if len(contour_vals) > 0:
#         contours = lith_iso.contour(isosurfaces=contour_vals.tolist(), scalars="Depth")
#         if contours.n_points > 0:
#             plotter.add_mesh(contours, color="black", line_width=2)
#
# craton_base = dataset.contour(isosurfaces=[0.8], scalars="I_craton")
# if craton_base.n_points > 0:
#     plotter.add_mesh(craton_base, color="firebrick", opacity=0.8, label="Craton")
#
# cont_base = dataset.contour(isosurfaces=[0.8], scalars="I_cont")
# if cont_base.n_points > 0:
#     plotter.add_mesh(cont_base, color="sandybrown", opacity=0.5, label="Continental")
#
# plotter.add_legend()
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Fields at present day
#
# Finally, we compute all fields at the present day (0 Ma).  The
# oceanic lithosphere now reflects its current age distribution, and
# the continental blocks are in their observed positions.

# +
ndtime = plate_model.age2ndtime(0)
I_lith.update_plate_reconstruction(ndtime)
I_cont.update_plate_reconstruction(ndtime)
I_crust.update_plate_reconstruction(ndtime)
I_craton.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
T_lin.update_plate_reconstruction(ndtime)

T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))

output_file.write(T, T_litho, I_lith, I_cont, I_crust, I_craton, T_erf, T_lin)

with T.dat.vec_ro as v:
    T_min, T_max = v.min()[1], v.max()[1]
plog.log_str(f"{plate_model.ndtime2age(ndtime)} {assemble(I_lith * dx)} "
             f"{assemble(I_cont * dx)} {assemble(I_crust * dx)} "
             f"{assemble(I_craton * dx)} {assemble(T * dx)} {T_min} {T_max}")
log("Written output for 0 Ma")
plog.close()
# -

# + tags=["active-ipynb"]
# reader = pv.get_reader("gplates_fields_output.pvd")
# reader.set_active_time_point(2)
# dataset = reader.read()[0]
#
# plotter = pv.Plotter()
# backend = None
# if os.environ.get("GADOPT_RENDER", "false").lower() == "true":
#     backend = "static"
#
# r_outer = rmax
# lith_iso = dataset.contour(isosurfaces=[0.5], scalars="I_lith")
# if lith_iso.n_points > 0:
#     pts = lith_iso.points
#     depth = r_outer - np.sqrt(np.sum(pts**2, axis=1))
#     lith_iso["Depth"] = depth
#     plotter.add_mesh(lith_iso, scalars="Depth", cmap="viridis",
#                      opacity=0.7, scalar_bar_args={"title": "Depth (non-dim)"})
#     contour_vals = np.arange(0.02, depth.max() + 0.02, 0.02)
#     if len(contour_vals) > 0:
#         contours = lith_iso.contour(isosurfaces=contour_vals.tolist(), scalars="Depth")
#         if contours.n_points > 0:
#             plotter.add_mesh(contours, color="black", line_width=2)
#
# craton_base = dataset.contour(isosurfaces=[0.8], scalars="I_craton")
# if craton_base.n_points > 0:
#     plotter.add_mesh(craton_base, color="firebrick", opacity=0.8, label="Craton")
#
# cont_base = dataset.contour(isosurfaces=[0.8], scalars="I_cont")
# if cont_base.n_points > 0:
#     plotter.add_mesh(cont_base, color="sandybrown", opacity=0.5, label="Continental")
#
# plotter.add_legend()
# plotter.camera_position = [(10.0, 10.0, 10.0), (0.0, 0.0, 0), (0, 1, 0)]
# plotter.show(jupyter_backend=backend)
# -

# ## Using these fields in mantle convection simulations
#
# The indicator fields modify material properties -- in particular,
# viscosity.  A lithospheric indicator that is ~1 at the surface and
# ~0 at depth allows straightforward construction of a high-viscosity
# lid that moves with the plates:
#
# ```python
# # Viscosity contrast of 10^3 between lithosphere and mantle
# viscosity = 10 ** (I_lith * 3)
# approximation = BoussinesqApproximation(Ra, mu=viscosity)
# ```
#
# The temperature field `T` can be used directly as the initial
# condition.  In a time-stepping simulation, you update everything
# together with the plate velocities:
#
# ```python
# for timestep in range(timesteps):
#     time += dt
#     gplates_velocities.update_plate_reconstruction(time)
#     I_lith.update_plate_reconstruction(time)
#     I_cont.update_plate_reconstruction(time)
#     I_crust.update_plate_reconstruction(time)
#     I_craton.update_plate_reconstruction(time)
#     T_erf.update_plate_reconstruction(time)
#     T_lin.update_plate_reconstruction(time)
#
#     T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
#     T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))
#
#     stokes_solver.solve()
#     energy_solver.solve()
# ```
#
# Because checkpointing is enabled, the `ocean_checkpoints/`
# directory now contains tracker snapshots at every 10 Myr from
# 500 Ma to the present.  If you restart this demo, or start a new
# simulation that begins at a different age, a fresh
# `LithosphereConnector` pointed at the same `checkpoint_dir` will
# automatically load the nearest checkpoint instead of stepping all
# the way from `oldest_age`, skipping the long serial spin-up.
#
# See the [GPlates global demo](../gplates_global) for the full
# simulation setup including boundary conditions, nullspaces, and
# time-step adaptation.
