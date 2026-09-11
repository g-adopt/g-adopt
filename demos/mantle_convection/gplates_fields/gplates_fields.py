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

from pathlib import Path

from gadopt import *
from gadopt.gplates import (
    BoundedLayerIndicator,
    BoundedLinearGeotherm,
    GplatesScalarFunction,
    GlobalLayerIndicator,
    HalfSpaceCoolingGeotherm,
    InterpolationConfig,
    MeshConfig,
    PlateModelFiles,
    PointCloudSource,
    PolygonConnectorFactory,
    ScalarFieldConnector,
    ensure_reconstruction,
    pyGplatesConnector,
)
from gtrack import (
    CheckpointPolicy,
    LithosphereCloudConfig,
    LithosphereCloudSource,
    PolygonIndicatorConfig,
    PolygonIndicatorSource,
    TracerConfig,
)

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
# `continental_polygons` and `static_polygons`. The polygon files
# are not needed by the velocity reconstruction itself; they belong
# to the lithosphere/craton Sources, so we collect them into a
# `PlateModelFiles` and hand that to the sources below.

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
)

plate_files = PlateModelFiles(
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
# A plate-reconstruction field is built from three composable pieces:
#
# * a **Source** owns the stateful gtrack machinery (the
#   `SeafloorAgeTracker`, the `PointRotator`, the `PolygonFilter`)
#   and exposes a single `prepare(age)` call returning a dict of
#   source-point arrays;
# * an **OutputStrategy** turns interpolated source values at target
#   mesh nodes into a scalar field (a quintic indicator, an erf geotherm,
#   or a linear geotherm);
# * a **ScalarFieldConnector** wires the two together, handles the
#   kNN interpolation between source points and mesh DoFs, and caches
#   results by geological age and target coordinate array.
#
# The key benefit of this split: two connectors that share the *same*
# `PointCloudSource` instance see a single coherent advance of the
# underlying forward-only ocean tracker per geological age.  The
# Source's per-age cache enforces this regardless of which connector
# is asked first, so the order of `update_plate_reconstruction` calls
# between a paired indicator and geotherm doesn't matter.  This
# pattern also makes the resource budget transparent at the call
# site: you can see at a glance that `lith_source` is constructed
# once and that both `I_lith` and `T_erf` hold a reference to it,
# rather than each carrying an invisible duplicate tracker.
#
# We collect the shared parameters into four small dataclasses:
#
# * `MeshConfig` -- the mesh's outer radius (`r_outer = rmax`) and
#   physical depth scale (2891 km).
# * `InterpolationConfig` -- the kNN kernel, neighbour count, and
#   angular distance cut-off for interpolating source points onto
#   mesh DoFs.  0.02 rad ~ 130 km is a fairly tight threshold;
#   loosen it if you see "holes" in the field.
# * `LithosphereCloudConfig` (from gtrack) -- ocean-tracker knobs.
#   Every tracker knob lives on the nested `TracerConfig`, so there is
#   exactly one home for each (no pass-through dict that can silently
#   override a field): ridge sampling, collision thresholds, the seed
#   count `tracker_point_count`, and the internal `tracker_step_myr`.
#   The config also sets the rebuild interval and checkpoint policy.
# * `PolygonIndicatorConfig` (from gtrack) -- much simpler since polygon
#   sources have no time-stepping state. `background_point_count` sets
#   the size of the new uniform grid for each age.
#
# **Ocean tracker checkpointing.**  When the lithosphere producer first
# steps the `SeafloorAgeTracker`, rank 0 must initialise it at
# the plate-model maximum age and step forward to the requested age
# -- a sequential process during which every other MPI rank sits
# idle at the broadcast.  A `CheckpointPolicy` tells the producer to
# periodically save the tracker state (tracer positions and material
# ages) to `.npz` files.  On a subsequent run, the producer scans the
# checkpoint directory, loads the file closest to the first requested
# age, and only steps forward from there.  The result is identical to a
# continuous run, but the wall-clock saving can be substantial on large
# MPI jobs.  gtrack does not create the directory, so we make it up
# front on rank 0 -- a mistyped path then fails at wiring time.

# +
mesh_cfg = MeshConfig(r_outer=rmax)
interp_cfg = InterpolationConfig(
    kernel="idw",
    neighbor_count=20,                  # Maximum source points for each target node.
    max_source_separation_rad=0.02,     # Maximum source separation in radians (about 127 km).
)

checkpoint_dir = Path("./ocean_checkpoints")
if mesh.comm.rank == 0:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

lith_source_cfg = LithosphereCloudConfig(
    tracer=TracerConfig(
        tracker_step_myr=2.0,
        earth_radius_m=6.3781e6,
        collision_velocity_difference_km_per_myr=7.0,
        collision_distance_rate_km_per_myr=10.0,
        tracker_point_count=5000,
        initial_spreading_rate_mm_per_yr=75.0,
        ridge_sampling_angle_deg=2.0,
        ridge_offset_angle_deg=0.01,
        tracker_rebuild_neighbor_count=5,
        tracker_rebuild_max_distance_m=500e3,
    ),
    tracker_rebuild_interval_myr=10.0,
    checkpoint=CheckpointPolicy(directory=checkpoint_dir, interval_myr=10.0),
)

# `background_point_count` sets the new background grid size.
# `scalar_input_point_count` sets the grid size for a scalar thickness.
# The gridded continental and craton data supply their own source points.
poly_source_cfg = PolygonIndicatorConfig(
    background_point_count=5000,
    scalar_input_point_count=5000,
)
# -

# ## Part 1: Lithosphere indicator + geotherm
#
# We build a single lithosphere source -- a gtrack
# `LithosphereCloudSource` producer wrapped in a gadopt
# `PointCloudSource` -- and then hand it to two separate
# `ScalarFieldConnector` instances -- one with a
# `GlobalLayerIndicator` (the smooth indicator field: exactly 1 from the
# surface down to the lithospheric base, decaying to exactly 0 over a
# one-sided quintic transition below it) and one with a
# `HalfSpaceCoolingGeotherm` (the half-space cooling temperature profile).
# The lithosphere covers the whole sphere, so the indicator has no
# lateral term at all: every column is inside the region and only its
# base depth varies.  The thickness channel never vanishes -- nodes
# with no nearby seed are filled with `fallback_thickness_km` -- so the
# surface legitimately reads 1 everywhere.  Part 2 covers what changes
# when the region is bounded instead.
#
# Because both connectors hold a reference to `lith_source`, the
# underlying `SeafloorAgeTracker` advances exactly once per call to
# `update_plate_reconstruction(ndtime)`, no matter which of the two
# `GplatesScalarFunction` wrappers is asked first.

# +
lith_producer = LithosphereCloudSource(
    rotation_files=plate_model.rotation_filenames,
    topology_files=plate_model.topology_filenames,
    continental_polygons=plate_files.continental_polygons,
    static_polygons=plate_files.static_polygons,
    continental_data=continental_data,
    oceanic_thickness_from_age=half_space_cooling,
    plate_model_max_age_ma=plate_model.oldest_age,
    config=lith_source_cfg,
)
lith_source = PointCloudSource(lith_producer, plate_model, comm=mesh.comm)

I_lith = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    lith_source,
    GlobalLayerIndicator(
        base_transition_width_km=10.0,
        fallback_thickness_km=100.0,
    ),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="I_lith")

T_erf = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    lith_source,
    HalfSpaceCoolingGeotherm(
        thermal_diffusivity_m2_per_s=1e-6,
        fallback_age_myr=500.0,
    ),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="T_erf")
# -

# ## Part 2: Continental indicator + geotherm
#
# Same idiom for the continental polygon-bounded fields: one polygon
# source -- a gtrack `PolygonIndicatorSource` wrapped in a
# `PointCloudSource` -- shared between an indicator and a linear
# geotherm. The indicator is `BoundedLayerIndicator` rather than the
# `GlobalLayerIndicator` used for the lithosphere. This difference is
# important because the continental region has a lateral boundary.
#
# A bounded region is two independent facts: *where* it is, and *how
# deep* it goes there.  A polygon source stores its depth data on
# seeds inside the polygons and zero everywhere else, so its depth
# channel holds those two facts multiplied together.  The
# interpolation then smooths the product rather than the factors, and
# at that point neither can be recovered from it: a blended 100 km
# could be a half-covered 200 km region or a fully covered 100 km one.
# Read as a depth it makes the layer base rise towards the surface
# across every region edge, which the data never said; read as a
# coverage it makes the map boundary depend on the depth numbers.
#
# A polygon source therefore publishes `membership` as its own channel
# -- 1 on the seeds, 0 on the background -- alongside the depth channel,
# which it names `masked_thickness` precisely because it is a masked
# product and not a plain depth. `BoundedLayerIndicator` consumes both:
# `membership` gives the lateral extent, and dividing it back out of the
# blended `masked_thickness` gives the depth the data actually holds,
# right up to the boundary.  There is no reference thickness to choose.
# And because the channel is not called `thickness`, a plain
# `GlobalLayerIndicator` -- which requires `thickness` -- fails the
# connector's `requires <= provides` check against a polygon source,
# rather than painting the oceans as continent. The geotherm is
# `BoundedLinearGeotherm` for the same reason. It
# deblends `masked_thickness` too, and reads mantle temperature outside
# the region instead of the surface value a plain linear geotherm would
# give there.

# +
cont_producer = PolygonIndicatorSource(
    rotation_files=plate_model.rotation_filenames,
    topology_files=plate_model.topology_filenames,
    polygons=muller_2022_files.get("continental_polygons"),
    static_polygons=plate_files.static_polygons,
    thickness_data=continental_data,
    config=poly_source_cfg,
)
cont_source = PointCloudSource(cont_producer, plate_model, comm=mesh.comm)

I_cont = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    cont_source,
    BoundedLayerIndicator(base_transition_width_km=10.0),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="I_cont")

T_lin = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    cont_source,
    BoundedLinearGeotherm(),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="T_lin")
# -

# ## Part 3: Solo polygon indicators
#
# The continental crust and craton fields are indicator-only -- they
# don't have a paired geotherm.  When you only need one field per
# source, a `PolygonConnectorFactory` keeps the call site short without
# losing the source and output details. Create the source and indicator
# on the factory, then get the connector from `.indicator`.

# ### Continental crust
#
# The continental crust (top ~50 km of continental regions) is less
# dense than the mantle (~2700 vs ~3200 kg/m^3).  We model it as a
# uniform-thickness layer identified by the plate model's continental
# polygons, driving an upward buoyancy force that represents the
# density deficit of continental crust relative to the mantle.

# +
crust_factory = PolygonConnectorFactory(mesh=mesh_cfg, interpolation=interp_cfg)
crust_producer = PolygonIndicatorSource(
    rotation_files=plate_model.rotation_filenames,
    topology_files=plate_model.topology_filenames,
    polygons=muller_2022_files.get("continental_polygons"),
    static_polygons=plate_files.static_polygons,
    thickness_data=50.0,
    config=poly_source_cfg,
)
crust_factory.create_source(crust_producer, plate_model, comm=mesh.comm)
# A constant 50 km everywhere inside the polygons: `thickness_data` is
# a scalar, so every seed carries the same depth and only the lateral
# extent varies.  Nothing extra to configure -- the factory builds a
# `BoundedLayerIndicator`, which reads the depth from the data and the
# extent from the membership channel whether the depth is constant or
# not.  A constant-thickness region is not a special case here, it is
# just a varying-thickness one whose variation happens to be zero.
crust_factory.create_indicator(base_transition_width_km=10.0)

I_crust = GplatesScalarFunction(
    Q, indicator_connector=crust_factory.indicator, name="I_crust"
)
# -

# ### Cratons
#
# Cratons are the ancient, stable cores of continents.  Their thick
# (~200-300 km), cold lithospheric roots are thought to protect them
# from tectonic reworking over billions of years.  The craton boundary
# polygons used here come from
# [Shirmard et al. (2025)](https://doi.org/10.1016/j.gsf.2025.102176),
# who delineated craton boundaries by applying unsupervised machine
# learning (PCA and k-means clustering) to horizontal shear-wave
# velocities from the REVEAL full-waveform tomography model, combined
# with lithospheric thickness and tectonic age constraints.  The
# shapefiles are available from the
# [EarthByte Craton_Boundaries repository](https://github.com/EarthByte/Craton_Boundaries)
# on GitHub (or via `make data`).

# +
craton_factory = PolygonConnectorFactory(mesh=mesh_cfg, interpolation=interp_cfg)
craton_producer = PolygonIndicatorSource(
    rotation_files=plate_model.rotation_filenames,
    topology_files=plate_model.topology_filenames,
    polygons="Craton_Boundaries_Inferred.shp",
    static_polygons=plate_files.static_polygons,
    thickness_data=continental_data,
    config=poly_source_cfg,
)
craton_factory.create_source(craton_producer, plate_model, comm=mesh.comm)
# Cratonic roots run thick and, unlike the crust above, genuinely
# varying: roughly 150-300 km across the set.  That range is why this
# field needs the membership channel rather than a single reference
# thickness.  A constant chosen to keep the deep interiors saturated
# would push the map boundary far outside the polygons, and one chosen
# to place the boundary correctly would leave the shallower cratons at
# a fraction of full amplitude in their own interiors, thousands of km
# from any edge.  Reading extent and depth from separate channels
# needs no such compromise, and the craton outlines stop depending on
# how deep the keels happen to be.
craton_factory.create_indicator(base_transition_width_km=10.0)

I_craton = GplatesScalarFunction(
    Q, indicator_connector=craton_factory.indicator, name="I_craton"
)
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
#
# One feature of that isosurface is worth expecting rather than
# discovering.  The one-sided quintic puts its 0.5 crossing half a
# transition width below the base, so with `base_transition_width_km=10` a column of
# zero thickness still crosses 0.5 about 5 km down.  Newly formed
# lithosphere at a ridge axis has essentially zero thickness, so the
# "base" isosurface picks up a thin sheet hugging the entire ridge
# system just below the surface.  That is the honest consequence of a
# zero-thickness lithosphere having its base at the surface, not a
# rendering fault, but it looks like one.  Contour above 0.5, or clip
# the top few km, if you want only the deep base.

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
# Read that as a sketch rather than a recipe.  `I_lith` is a depth
# mask, not a strength map: it is 1 everywhere at the surface, ridge
# axes included, because the lithosphere covers the whole sphere and
# the indicator only says whether a point is above the base.  Fed
# straight into the expression above, the weakest material in the
# domain comes out as strong as the oldest plate interior.  If
# viscosity should vary laterally, that variation has to come from
# something that carries lateral information -- the temperature field,
# the plate age, or a bounded region's own indicator -- rather than
# from `I_lith`.
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
# `LithosphereCloudSource` given a `CheckpointPolicy` pointed at the
# same directory will automatically load the nearest checkpoint
# instead of stepping all
# the way from the plate-model maximum age, which skips the long serial spin-up.
#
# See the [GPlates global demo](../gplates_global) for the full
# simulation setup including boundary conditions, nullspaces, and
# time-step adaptation.
