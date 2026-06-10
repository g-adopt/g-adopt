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
from gadopt.gplates import (
    GplatesScalarFunction,
    ScalarFieldConnector,
    InterpolationConfig,
    LithosphereSource,
    LithosphereSourceConfig,
    MeshConfig,
    PlateModelFiles,
    PolygonSource,
    PolygonSourceConfig,
    QuinticOutput,
    GeothermERFOutput,
    GeothermLinearOutput,
    ensure_reconstruction,
    PolygonConnectorFactory,
    pyGplatesConnector,
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
#   mesh nodes into a scalar field (a tanh indicator, an erf geotherm,
#   a linear geotherm);
# * an **ScalarFieldConnector** wires the two together, handles the
#   kNN interpolation between source points and mesh DoFs, and caches
#   results by `(age, coords_hash)`.
#
# The key benefit of this split: two connectors that share the *same*
# `LithosphereSource` instance see a single coherent advance of the
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
# * `LithosphereSourceConfig` -- ocean-tracker knobs: how many
#   tracers to seed, how often to reinitialise the tracer mesh, and
#   the pass-through `gtrack_config` dictionary which is forwarded
#   directly to gtrack's `TracerConfig` (ridge sampling, collision
#   thresholds, etc).
# * `PolygonSourceConfig` -- much simpler since polygon sources have
#   no time-stepping state.
#
# **Ocean tracker checkpointing.**  When `LithosphereSource` first
# steps the `SeafloorAgeTracker`, rank 0 must initialise it at
# `oldest_age` and step forward to the requested reconstruction age
# -- a sequential process during which every other MPI rank sits
# idle at the broadcast.  Setting `checkpoint_interval_myr` tells
# the source to periodically save the tracker state (tracer positions
# and material ages) to `.npz` files.  On a subsequent run, the source
# scans the checkpoint directory, loads the file closest to the first
# requested age, and only steps forward from there.  The result is
# identical to a continuous run, but the wall-clock saving can be
# substantial on large MPI jobs.

# +
mesh_cfg = MeshConfig(r_outer=rmax)
interp_cfg = InterpolationConfig(
    kernel="idw",
    k_neighbors=20,           # source seeds averaged per target node; higher is smoother but costlier
    distance_threshold=0.02,  # max angular reach on the unit sphere, in RADIANS (0.02 rad ~ 127 km); a target with no seed inside this reads as "outside"
)

lith_source_cfg = LithosphereSourceConfig(
    n_points=5000,
    reinit_interval_myr=10.0,
    checkpoint_interval_myr=10.0,
    checkpoint_dir="./ocean_checkpoints",
    gtrack_config={
        "time_step": 2.0,                            # internal gtrack timestep (Myr)
        "earth_radius": 6.3781e6,                    # Earth radius (m)
        "velocity_delta_threshold": 7.0,             # collision velocity threshold (km/Myr)
        "distance_threshold_per_myr": 10.0,          # collision distance threshold (km/Myr)
        "default_mesh_points": 5000,                 # initial Fibonacci sphere mesh points
        "initial_ocean_mean_spreading_rate": 75.0,   # spreading rate at initial age (mm/yr)
        "ridge_sampling_degrees": 2.0,               # ridge tessellation resolution (degrees)
        "spreading_offset_degrees": 0.01,            # offset from ridge for new seeds (degrees)
        "reinit_k_neighbors": 5,                     # kNN during reinitialisation
        "reinit_max_distance": 500e3,                # max interpolation distance (m)
    },
)

poly_source_cfg = PolygonSourceConfig(n_points=5000)
# -

# ## Part 1: Lithosphere indicator + geotherm
#
# We build a single `LithosphereSource` and then hand it to two
# separate `ScalarFieldConnector` instances -- one with a
# `QuinticOutput` (the smooth indicator field: exactly 1 from the
# surface down to the lithospheric base, decaying to exactly 0 over a
# one-sided quintic transition below it) and one with a
# `GeothermERFOutput` (the half-space cooling temperature profile).
# No lateral fade is needed here: the thickness channel never
# vanishes (uncovered nodes are filled with `default_thickness_km`),
# so the surface legitimately reads 1 everywhere.  Because both connectors hold a reference to
# `lith_source`, the underlying `SeafloorAgeTracker` advances exactly
# once per call to `update_plate_reconstruction(ndtime)`, no matter
# which of the two `GplatesScalarFunction` wrappers is asked first.

# +
lith_source = LithosphereSource(
    gplates_connector=plate_model,
    continental_data=continental_data,
    age_to_property=half_space_cooling,
    plate_files=plate_files,
    config=lith_source_cfg,
    comm=mesh.comm,
)

I_lith = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    lith_source,
    QuinticOutput(width_km=10.0, default_thickness_km=100.0),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="I_lith")

T_erf = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    lith_source,
    GeothermERFOutput(kappa=1e-6, too_far_age_myr=500.0),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="T_erf")
# -

# ## Part 2: Continental indicator + geotherm
#
# Same idiom for the continental polygon-bounded fields: one
# `PolygonSource` shared between an indicator and a linear geotherm.
# Note `default_thickness_km=0.0` for the continental indicator --
# the polygon source places zero-thickness halo seeds outside the
# continental polygons, so `too_far` target nodes should read as
# "outside the region" rather than getting filled with a default
# 100 km of continental material.
#
# Crucially, a polygon-bounded indicator must also set `fade_ref_km`.
# The one-sided quintic step reads exactly 1 at the surface wherever
# the region has *any* thickness -- and where the thickness is zero
# the base sits at the surface, so the surface node itself would read
# 1 too.  Without a lateral fade the indicator would paint the entire
# surface, oceans included, as continent.  The fade multiplies the
# radial step by `clip(thickness / fade_ref_km, 0, 1)`, taking the
# field to exactly 0 where the thickness vanishes; 100 km is the
# typical thickness scale of the continental lithosphere data.

# +
cont_source = PolygonSource(
    gplates_connector=plate_model,
    polygons=muller_2022_files.get("continental_polygons"),
    thickness_data=continental_data,
    plate_files=plate_files,
    config=poly_source_cfg,
    comm=mesh.comm,
)

I_cont = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    cont_source,
    QuinticOutput(width_km=10.0, fade_ref_km=100.0, default_thickness_km=0.0),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="I_cont")

T_lin = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    cont_source,
    GeothermLinearOutput(),
    mesh=mesh_cfg, interpolation=interp_cfg,
), name="T_lin")
# -

# ## Part 3: Solo polygon indicators
#
# The continental crust and craton fields are indicator-only -- they
# don't have a paired geotherm.  When you only need one field per
# source, a `PolygonConnectorFactory` keeps the call site short without
# losing any of the underlying machinery: construct the source and the
# output on the factory and pull the connector off `.indicator`.

# ### Continental crust
#
# The continental crust (top ~50 km of continental regions) is less
# dense than the mantle (~2700 vs ~3200 kg/m^3).  We model it as a
# uniform-thickness layer identified by the plate model's continental
# polygons, driving an upward buoyancy force that represents the
# density deficit of continental crust relative to the mantle.

# +
crust_factory = PolygonConnectorFactory(mesh=mesh_cfg, interpolation=interp_cfg)
crust_factory.construct_source(
    gplates_connector=plate_model,
    polygons=muller_2022_files.get("continental_polygons"),
    thickness_data=50.0,
    plate_files=plate_files,
    config=poly_source_cfg,
    comm=mesh.comm,
)
# fade_ref_km is a required argument on the polygon factory; with a
# constant 50 km crust the natural fade reference is that same 50 km.
crust_factory.construct_output(fade_ref_km=50.0, width_km=10.0)

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
craton_factory.construct_source(
    gplates_connector=plate_model,
    polygons="Craton_Boundaries_Inferred.shp",
    thickness_data=continental_data,
    plate_files=plate_files,
    config=poly_source_cfg,
    comm=mesh.comm,
)
# Cratonic roots run thick (~150-300 km); fading over 150 km keeps
# the craton interiors at full amplitude while the margins, where the
# thickness data thins out, taper smoothly to zero.
craton_factory.construct_output(fade_ref_km=150.0, width_km=10.0)

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
# `LithosphereSource` pointed at the same `checkpoint_dir` will
# automatically load the nearest checkpoint instead of stepping all
# the way from `oldest_age`, skipping the long serial spin-up.
#
# See the [GPlates global demo](../gplates_global) for the full
# simulation setup including boundary conditions, nullspaces, and
# time-step adaptation.
