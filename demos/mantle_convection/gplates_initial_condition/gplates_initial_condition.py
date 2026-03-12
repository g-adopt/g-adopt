# 3-D lithospheric thermal initial condition from plate reconstructions
# =====================================================================
#
# This tutorial demonstrates how to build a complete lithospheric
# thermal initial condition by combining indicator fields (blending
# weights) with geotherm connectors (temperature profiles).  The
# indicator connectors produce smooth 0/1 masks that define *where*
# the lithosphere and continents are, while the geotherm connectors
# produce normalized temperature profiles that define *how hot* the
# lithosphere is at each depth.
#
# The composition follows Rhodri's formula:
#
# $$T_{\text{litho}} = T_{\text{lin}} \, I_{\text{cont}}
#                     + T_{\text{erf}} \, (1 - I_{\text{cont}})$$
#
# $$T = T_s + (T_{\text{lab}} - T_s)
#       \bigl(T_{\text{litho}} \, I_{\text{lith}}
#            + T_{\text{bg,norm}} \, (1 - I_{\text{lith}})\bigr)$$
#
# where:
#
# - $I_{\text{lith}}$ is the lithosphere indicator (~1 inside, ~0 in
#   the mantle)
# - $I_{\text{cont}}$ is the continental indicator (~1 on continents,
#   ~0 in oceans)
# - $T_{\text{erf}}$ is the oceanic erf geotherm (age-dependent)
# - $T_{\text{lin}}$ is the continental linear geotherm
# - $T_s$ is the surface temperature and $T_{\text{lab}}$ is the
#   temperature at the base of the lithosphere
#
# This builds on the [lithosphere indicator demo](../gplates_lithosphere)
# which covers the indicator connectors in detail.  You should work
# through that tutorial first.
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
# is the seafloor age in seconds.  We cap the result at 150 km.

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
# We use the Muller et al. (2022) plate reconstruction model.  The
# `ensure_reconstruction` helper locates the required rotation and
# topology files, as well as `continental_polygons` and
# `static_polygons` that the connectors need.
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
# The connectors expect coordinates in (latitude, longitude) order,
# so we swap the columns after loading.

# +
with h5py.File("lithospheric_thickness_mesh.h5", "r") as f:
    lonlat = f["lonlat"][:]       # (N, 2) -- (lon, lat)
    thickness_values = f["values"][:]  # thickness in km

# Convert (lon, lat) -> (lat, lon) for the connector
latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])
continental_data = (latlon, thickness_values)
# -

# ## Step 1: Blending indicators
#
# We first create two indicator fields that will serve as blending
# weights in the temperature composition.  The lithosphere indicator
# is ~1 everywhere inside the lithosphere (oceanic + continental)
# and ~0 in the underlying mantle.  The continental indicator is ~1
# on continental regions and ~0 in oceanic regions.

# +
I_lith_connector = LithosphereConnector(
    gplates_connector=plate_model,
    continental_data=continental_data,
    age_to_property=half_space_cooling,
    config_extra={
        "r_outer": rmax,
        "n_points": 20000,
        "k_neighbors": 50,
        "distance_threshold": 0.15,
        "transition_width": 10.0,
    },
    comm=mesh.comm,
)

I_lith = GplatesScalarFunction(Q, indicator_connector=I_lith_connector, name="I_lith")

I_cont_connector = PolygonConnector(
    gplates_connector=plate_model,
    polygons=muller_2022_files.get("continental_polygons"),
    thickness_data=continental_data,
    config_extra={
        "r_outer": rmax,
        "n_points": 20000,
        "k_neighbors": 50,
        "distance_threshold": 0.15,
        "transition_width": 10.0,
    },
    comm=mesh.comm,
)

I_cont = GplatesScalarFunction(Q, indicator_connector=I_cont_connector, name="I_cont")
# -

# ## Step 2: Geotherm connectors
#
# Next we create two geotherm fields that produce normalized
# temperature profiles (values in [0, 1]).  The oceanic erf profile
# depends on seafloor age — younger ocean has a thinner thermal
# boundary layer — while the continental linear profile simply
# increases linearly from surface to the base of the lithosphere.

# +
T_erf_connector = LithosphereGeotherm(
    gplates_connector=plate_model,
    continental_data=continental_data,
    age_to_property=half_space_cooling,
    geotherm=ocean_erf_normalized,
    kappa=1e-6,
    config_extra={
        "r_outer": rmax,
        "n_points": 20000,
        "k_neighbors": 50,
        "distance_threshold": 0.15,
        "transition_width": 10.0,
    },
    comm=mesh.comm,
)

T_erf = GplatesScalarFunction(Q, indicator_connector=T_erf_connector, name="T_erf")

T_lin_connector = PolygonGeotherm(
    gplates_connector=plate_model,
    polygons=muller_2022_files.get("continental_polygons"),
    thickness_data=continental_data,
    geotherm=continental_linear,
    config_extra={
        "r_outer": rmax,
        "n_points": 20000,
        "k_neighbors": 50,
        "distance_threshold": 0.15,
        "transition_width": 10.0,
    },
    comm=mesh.comm,
)

T_lin = GplatesScalarFunction(Q, indicator_connector=T_lin_connector, name="T_lin")
# -

# ## Step 3: Temperature composition
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

# ## Initial condition at 200 Ma
#
# We begin by computing the temperature field at 200 Ma — the
# oldest age in our reconstruction.  Each update call advances
# the ocean age tracker and rotates the continental data to the
# appropriate palaeoposition before computing the field values.

# +
output_file = VTKFile("initial_condition_output.pvd")

plog = ParameterLog("params.log", mesh)
plog.log_str("age T_avg T_min T_max I_lith_int I_cont_int")

ndtime = plate_model.age2ndtime(200)
I_lith.update_plate_reconstruction(ndtime)
I_cont.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
T_lin.update_plate_reconstruction(ndtime)

T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))

output_file.write(T, T_litho, I_lith, I_cont, T_erf, T_lin)

with T.dat.vec_ro as v:
    T_min, T_max = v.min()[1], v.max()[1]
plog.log_str(f"{plate_model.ndtime2age(ndtime)} {assemble(T * dx)} "
             f"{T_min} {T_max} {assemble(I_lith * dx)} {assemble(I_cont * dx)}")
log("Written output for 200 Ma")
# -

# ## Initial condition at 100 Ma
#
# Advancing to 100 Ma, the ocean age tracker has evolved the
# seafloor ages and the continental blocks have moved to their
# mid-Cretaceous positions.

# +
ndtime = plate_model.age2ndtime(100)
I_lith.update_plate_reconstruction(ndtime)
I_cont.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
T_lin.update_plate_reconstruction(ndtime)

T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))

output_file.write(T, T_litho, I_lith, I_cont, T_erf, T_lin)

with T.dat.vec_ro as v:
    T_min, T_max = v.min()[1], v.max()[1]
plog.log_str(f"{plate_model.ndtime2age(ndtime)} {assemble(T * dx)} "
             f"{T_min} {T_max} {assemble(I_lith * dx)} {assemble(I_cont * dx)}")
log("Written output for 100 Ma")
# -

# ## Initial condition at present day
#
# Finally, we compute the temperature at the present day (0 Ma).
# The oceanic lithosphere now reflects its current age distribution
# and the continental blocks are in their observed positions.

# +
ndtime = plate_model.age2ndtime(0)
I_lith.update_plate_reconstruction(ndtime)
I_cont.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
T_lin.update_plate_reconstruction(ndtime)

T_litho.interpolate(T_lin * I_cont + T_erf * (1 - I_cont))
T.interpolate(Ts + (Tlab - Ts) * (T_litho * I_lith + 1.0 * (1 - I_lith)))

output_file.write(T, T_litho, I_lith, I_cont, T_erf, T_lin)

with T.dat.vec_ro as v:
    T_min, T_max = v.min()[1], v.max()[1]
plog.log_str(f"{plate_model.ndtime2age(ndtime)} {assemble(T * dx)} "
             f"{T_min} {T_max} {assemble(I_lith * dx)} {assemble(I_cont * dx)}")
log("Written output for 0 Ma")
plog.close()
# -

# ## Using the initial condition in a simulation
#
# The temperature field `T` produced above can be used directly as
# the initial condition for a mantle convection simulation.  In a
# time-stepping loop, you would update the indicator and geotherm
# fields alongside the plate velocities and recompose whenever the
# thermal structure needs refreshing:
#
# ```python
# for timestep in range(timesteps):
#     time += dt
#     gplates_velocities.update_plate_reconstruction(time)
#     I_lith.update_plate_reconstruction(time)
#     I_cont.update_plate_reconstruction(time)
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
# See the [GPlates global demo](../gplates_global) for the full
# simulation setup including boundary conditions, nullspaces, and
# time-step adaptation.
