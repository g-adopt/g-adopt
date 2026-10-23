r"""GIAMIP Exp11: make the input file of the bed inversion.

WHAT THIS SCRIPT DOES
    It makes every field of the inversion that comes from data, once, and
    writes them to one Firedrake `CheckpointFile` (`inputs_exp11.h5`). The
    inversion (`invert_exp11.py`) loads that file and reads no data file, so
    its tape holds only the operations from the control to the misfit.

    1. Read the coarse giamsh mesh, partition it so that every rank holds
       the same number of mantle cells, refine it `--refine` times, curve it
       to P2 and cut the mantle submesh (`exp11_common.build_meshes`). A
       mesh that is refined already and saved as a P1 `CheckpointFile`
       (`.h5`, for example the 8 Re mesh `giamip_8re_r4_linear`) is loaded
       with the same mantle-balanced partition and curved
       (`exp11_common.load_checkpoint_meshes`).
    2. Make the earth fields on the mantle: density, shear and bulk modulus
       (DG1) and gravity (CG2) from the PREM-stable profile of `giamip.earth`,
       and the viscosity (DG0) from the 3-D model of Lloyd et al. (2024),
       read from g-drift. All fields are non-dimensional.
    3. Make the forcing on the mantle from the GIAMIP files: the observed sea
       level `SL_obs = -bed_obs / D` (CG2), the frozen surface slope of the
       ocean mask (`surface_slope(SL_obs)`, DG1), and the ice of every step
       of the march from the start to year 0 (CG1, one entry per step).
    4. Write the input file with its JSON record, the same record as a JSON
       file next to it, and ParaView files of the earth fields and of the ice
       at three steps for the reviewer.

    The script tapes nothing: `firedrake.adjoint` is not imported.

THE EXPERIMENT
    GIAMIP protocol v4.0, experiment Exp11, as `giamip` defines it
    (`giamip.mip.inputs.GiamipExperiment("Exp11")`, case file
    `giamip/resources/cases/giamip/exp11.json`). Exp11 is Exp03 with a 3-D
    viscosity. The protocol leaves the earth model and the topography open;
    the values below are the G-ADOPT choices of the `gadopt` block of the
    case file (`giamip.cases.with_gadopt_choices`):

        ice          GLAC3c profile 3 (`iceHistory-GLAC3c_profile3.nc`)
        topography   Bedmap3 + BedMachine Greenland 5 + GEBCO
                     (`bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc`), the
                     G-ADOPT choice `bedmap3`. The file is the observed
                     present-day bed (year 0); the inversion finds the bed at
                     the start of the run.
        earth        PREM-stable (elastic structure and density) with the
                     3-D viscosity of Lloyd et al. (2024) (`lloyd2024`),
                     Maxwell rheology, compressible
        period       122 ka to year 0, in steps of `--step_years` (2000 yr:
                     61 steps)
        constants    ice 917, ocean water 1025 kg/m^3

    Every number of the experiment comes from `giamip`, never from this file.
    `GiamipExperiment("Exp11").earth()` raises `NotImplementedError` for the
    3-D viscosity and `GiamipExperiment("Exp11").topography` is None, because
    the case file leaves both open. This script therefore builds the elastic
    structure with `giamip.earth.ElasticProfile` from the G-ADOPT choice, and
    opens the topography file that `giamip.cases.with_gadopt_choices` gives
    for the G-ADOPT choice.

THE EARTH MODEL ON THE MESH
    `giamip.earth` gives every property of the elastic structure at any
    radius, linear in radius between the nodes of the model file. Each DG1
    node takes the value at its own radius. Gravity is a CG2 field.
    PREM-stable has jumps at 400 and 670 km (mesh spheres) and at 15, 24.4,
    80, 220, 600, 771 and 2741 km (not spheres); a cell across one of the
    latter carries the linear interpolant of its nodal values.

    The viscosity is DG0, the value at the cell centroid. It comes from the
    g-drift dataset that the giamip model file `lloyd2024.json` names
    (dataset, field and SHA-256), read with `gdrift.EarthModel3D`. The script
    reads the model file through `importlib.resources` and applies its rules:

    - The field is log10 of the viscosity in Pa s. g-drift interpolates the
      stored log10 values (inverse-distance weights on each of the two
      bracketing depth layers of the dataset, then linear in radius between
      them). The linear value is never interpolated, because the values span
      21 orders of magnitude.
    - The lower bound of the model file is applied to the log10 value at the
      centroid, by depth (1e20 Pa s from the surface to 670 km, 1e21 Pa s
      below). There is no upper bound: values up to about 1e39 Pa s stay and
      act as elastic.
    - No elastic lid is added. Every cell, also in the top 120 km, takes its
      viscosity from the model; the stiff top of the model acts as the
      lithosphere. The 60, 90 and 120 km spheres have no special role.

    The script stops if the model file asks for a rule that this code does
    not implement (an upper bound, an added lithosphere, bounds that are not
    on log10), if the SHA-256 of the dataset differs from the model file, or
    if the depth range of the dataset does not cover the mantle of the mesh.
    The dataset has 201 depth layers from the surface (6371 km radius) to
    2891 km (the CMB of PREM-stable), so every centroid lies between two
    layers and no value is extrapolated.

    The record gives, for each mesh layer, the min, max and volume mean of
    log10 viscosity before and after the bound, and the volume fraction of
    the cells at the lower bound (`lloyd_viscosity.by_layer`).

    The g-drift model is loaded on one rank per node. The other ranks of the
    node send their centroids to that rank and receive the values, so the
    dataset (about 420 MB in memory) is held once per node.

THE FORCING ON THE MESH
    The GIAMIP grid is a Gauss-Legendre grid (257 x 513). Every node of the
    mantle mesh takes the bilinear interpolant of the grid at its direction
    (`BilinearTransfer`, the rule of the mesh script). The shared mesh was
    refined for the coastline of Exp01 (ICE-6G_D at 122 ka on Bedmap2); an
    experiment with another bed or ice uses the same mesh. The fields are
    defined in the whole mantle and are read only on Re. The record holds,
    for every step, the ice volume on the mesh against the Gauss-Legendre
    volume of the grid.

RUN
    srun python3 prepare_exp11.py --mesh <file.msh> --output_path <dir>
    The number of ranks must be the one of the inversion job.

OUTPUT (in --output_path)
    inputs_exp11.h5          the input file of the inversion
    inputs_exp11.json        its record (the same JSON as inside the file)
    inputs_exp11_earth.pvd   ParaView: the earth fields in SI units
    inputs_exp11_ice.pvd     ParaView: the ice at the first, middle and last
                             step of the march, m
    prepare_exp11.log        the log
"""

import argparse
import dataclasses
import importlib.metadata
import importlib.resources
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# `gadopt` imports `firedrake.pyplot`, which imports matplotlib's font
# manager. Without a font cache every rank scans the fonts; the job
# environment names a cache built once on a login node.
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# The shared module imports gadopt before firedrake.
import exp11_common as common  # noqa: E402
from gadopt import (COMM_WORLD, Function, FunctionSpace,  # noqa: E402
                    SpatialCoordinate, VectorFunctionSpace, VTKFile, assemble,
                    dot, ds, dx, sqrt)
from gadopt.sea_level_masks import surface_slope  # noqa: E402
from mpi4py import MPI  # noqa: E402

import giamip  # noqa: E402
from giamip import cases, grids  # noqa: E402
from giamip.earth import ElasticProfile  # noqa: E402
from giamip.mip.inputs import GiamipExperiment, Topography  # noqa: E402
from giamsh.radial import RadialStructure  # noqa: E402

# g-drift reads the 3-D viscosity. Only this script imports it: the
# inversion and the shared module must not depend on a data reader.
import gdrift  # noqa: E402
from gdrift.datasetnames import DATASET_REGISTRY  # noqa: E402
from gdrift.io import fetch_dataset  # noqa: E402

#: The experiment of this directory.
EXPERIMENT = "Exp11"
#: The file stem of the outputs.
STEM = "inputs_exp11"
#: The MD5 of the shared coarse mesh `n2_giamip_all_o1.msh` (handout section
#: 3). The run stops if the mesh file differs.
MESH_MD5 = "a4da84206dc82f5b458be66e4fd33844"

D_SCALE = common.D_SCALE


# --------------------------------------------------------------------------
# The earth model of Exp11: PREM-stable and the 3-D viscosity of g-drift
# --------------------------------------------------------------------------

#: The prefix of the g-drift registry names of the 3-D datasets that
#: `gdrift.EarthModel3D.from_dataset` loads; that function takes the name
#: without it.
GDRIFT_PREFIX = "3d_earthmodel_"
#: The number of centroids per call of the g-drift interpolation. For each
#: point g-drift makes arrays of its 8 lateral neighbours on two depth
#: layers; chunks keep these temporary arrays below about 100 MB.
GDRIFT_CHUNK = 500_000
#: The tolerance, m, of the check that the dataset reaches the surface and
#: the CMB of the elastic structure. The dataset radii are exact multiples
#: of 1 km, so any real gap is far larger.
RADIUS_TOLERANCE_M = 1.0


def model_file(model_id):
    """The giamip earth model file `model_id`, as a dict.

    The file is read from the installed giamip package with
    `importlib.resources`, so the run uses the model file of the giamip
    version that the record names, not a copy of its values.

    Returns:
      `(model, path)`: the parsed JSON and the path of the file as text.
    """
    resource = (importlib.resources.files("giamip").joinpath("resources")
                .joinpath("models").joinpath(f"{model_id}.json"))
    with resource.open() as handle:
        return json.load(handle), str(resource)


class Viscosity3D:
    """The 3-D viscosity of a giamip model file, from its g-drift dataset.

    The model file (for Exp11 `lloyd2024.json`) names the g-drift dataset,
    its field and its SHA-256, and gives the G-ADOPT rules: a lower bound by
    depth, no upper bound, and no added lithosphere. This class reads these
    entries and refuses a rule that it does not implement, so a change of the
    model file cannot change the run without an error.

    The field is log10 of the viscosity in Pa s. g-drift interpolates the
    stored values directly, so the interpolation is on log10. The lower bound
    applies to the interpolated log10 value at each point.

    The dataset is loaded on one rank per node (`load`). `log10_at` is
    collective over the ranks of a node: they send their points to the
    loading rank and receive the values.

    Args:
      model_id: the id of the model file, from the `gadopt` block of the case
        file (`lloyd2024`).
      surface_m: the surface radius of the elastic structure, m. Depths of
        the bounds are measured from it.
      cmb_m: the CMB radius of the elastic structure, m.

    Attributes:
      record: the model file, the dataset, the bounds and the g-drift
        version, for the record of the run.

    Raises:
      NotImplementedError: if the model file asks for a rule that this code
        does not implement.
      ValueError: if the bounds of the model file have a gap or do not reach
        the CMB, or the SHA-256 of the model file and of the g-drift registry
        differ.
    """

    def __init__(self, model_id, surface_m, cmb_m):
        model, path = model_file(model_id)
        self.surface_m = float(surface_m)
        self.cmb_m = float(cmb_m)
        if model.get("kind") != "viscosity_3d":
            raise NotImplementedError(f"{model_id} is {model.get('kind')!r}, "
                                      "not a 3-D viscosity (viscosity_3d)")

        # The dataset. The registry of the installed g-drift must name the
        # same file content as the model file.
        grid = model["grid"]
        if grid.get("package") != "gdrift":
            raise NotImplementedError(f"{model_id}: the grid is in "
                                      f"{grid.get('package')!r}, not gdrift")
        self.dataset = grid["dataset"]
        if not self.dataset.startswith(GDRIFT_PREFIX):
            raise NotImplementedError(
                f"{model_id}: {self.dataset} is not a {GDRIFT_PREFIX}* dataset "
                "of gdrift.EarthModel3D.from_dataset")
        self.field = grid["field"]
        self.sha256 = grid["sha256"]
        registry_hash = DATASET_REGISTRY.get_dataset(self.dataset).file_hash
        if registry_hash != f"sha256:{self.sha256}":
            raise ValueError(f"{self.dataset}: the model file gives SHA-256 "
                             f"{self.sha256}, the g-drift registry "
                             f"{registry_hash}")

        # The rules. The bounds act on log10 of the viscosity, and the
        # dataset must store log10, so that the bound and the interpolation
        # are on the same quantity.
        bounds = model["bounds"]
        if not str(bounds.get("applied_to", "")).startswith("log10"):
            raise NotImplementedError(f"{model_id}: bounds applied to "
                                      f"{bounds.get('applied_to')!r}, not log10")
        if not self.field.startswith("log10"):
            raise NotImplementedError(f"{model_id}: the field {self.field} is "
                                      "not a log10 viscosity")
        if bounds.get("upper_pa_s") is not None:
            raise NotImplementedError(f"{model_id}: an upper bound "
                                      f"{bounds['upper_pa_s']}; this script "
                                      "applies none")
        lithosphere = model.get("lithosphere")
        if not (isinstance(lithosphere, str)
                and lithosphere.startswith("none")):
            raise NotImplementedError(f"{model_id}: lithosphere "
                                      f"{lithosphere!r}; this script adds no "
                                      "lithosphere")

        # The lower bound: layers in depth below the surface, from the top
        # down, without a gap. The deepest layer reaches the CMB of the
        # elastic structure, as in `giamip.earth`; its bottom in the file
        # must be within 1 km of that CMB.
        lower = bounds["lower_pa_s"]
        if lower[0]["top_depth_km"] != 0:
            raise ValueError(f"{model_id}: the lower bound starts at "
                             f"{lower[0]['top_depth_km']} km, not at the "
                             "surface")
        for upper_layer, lower_layer in zip(lower, lower[1:]):
            if upper_layer["bottom_depth_km"] != lower_layer["top_depth_km"]:
                raise ValueError(f"{model_id}: a gap or an overlap of the "
                                 f"lower bound at "
                                 f"{upper_layer['bottom_depth_km']} km")
        cmb_depth_km = (self.surface_m - self.cmb_m) / 1e3
        if abs(lower[-1]["bottom_depth_km"] - cmb_depth_km) > 1.0:
            raise ValueError(f"{model_id}: the lower bound ends at "
                             f"{lower[-1]['bottom_depth_km']} km, the CMB is "
                             f"at {cmb_depth_km} km")
        self.lower_top_km = np.array([layer["top_depth_km"]
                                      for layer in lower], dtype=float)
        self.lower_log10 = np.log10([float(layer["viscosity_pa_s"])
                                     for layer in lower])

        self.model = None
        self.node = None
        self.record = {
            "model_id": model_id, "model_file": path,
            "model_file_sha256": common.sha256sum(path),
            "model": model,
            "dataset": self.dataset, "field": self.field,
            "sha256": self.sha256,
            "lower_bound": [{"top_depth_km": layer["top_depth_km"],
                             "bottom_depth_km": layer["bottom_depth_km"],
                             "viscosity_pa_s": float(layer["viscosity_pa_s"]),
                             "log10": float(value)}
                            for layer, value in zip(lower, self.lower_log10)],
            "upper_bound": None,
            "lithosphere": "none added",
            "gdrift": gdrift_version(),
        }

    def load(self):
        """Load the dataset on one rank per node and check it. Collective.

        The loading rank checks the SHA-256 of the file against the model
        file and the depth range of the dataset against the mantle of the
        elastic structure. The run stops if the dataset does not reach the
        surface or the CMB: a value outside the dataset would be an
        extrapolation, and no rule for it is decided.

        Returns:
          The dataset part of the record (file, SHA-256, depth range).
        """
        self.node = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        info = None
        if self.node.rank == 0:
            # `fetch_dataset` returns a file whose SHA-256 is the one of the
            # registry, which the constructor compared with the model file.
            # The second sum below is the record of the file that was read.
            path = fetch_dataset(self.dataset)
            digest = common.sha256sum(path)
            if digest != self.sha256:
                raise RuntimeError(f"{path} has SHA-256 {digest}, not "
                                   f"{self.sha256}")
            self.model = gdrift.EarthModel3D.from_dataset(
                self.dataset.removeprefix(GDRIFT_PREFIX), labels=[self.field])
            radius = np.linalg.norm(self.model.coordinates, axis=1)
            layers_km = np.unique(np.round(radius / 1e3, 3))
            values = self.model.available_fields[self.field]
            info = {"file": str(path), "file_sha256": digest,
                    "points": int(radius.size),
                    "depth_layers": int(layers_km.size),
                    "points_per_layer": int(radius.size // layers_km.size),
                    "radius_min_m": float(radius.min()),
                    "radius_max_m": float(radius.max()),
                    "depth_min_km": (self.surface_m - float(radius.max())) / 1e3,
                    "depth_max_km": (self.surface_m - float(radius.min())) / 1e3,
                    "log10_min": float(np.nanmin(values)),
                    "log10_max": float(np.nanmax(values)),
                    "nan_values": int(np.isnan(values).sum())}
            del radius
        # Every node leader read the same file; rank 0 speaks for all.
        info = COMM_WORLD.bcast(info, root=0)
        if (info["radius_min_m"] > self.cmb_m + RADIUS_TOLERANCE_M
                or info["radius_max_m"] < self.surface_m - RADIUS_TOLERANCE_M):
            raise RuntimeError(
                f"{self.dataset} covers radii {info['radius_min_m']:.0f} to "
                f"{info['radius_max_m']:.0f} m, the mantle "
                f"{self.cmb_m:.0f} to {self.surface_m:.0f} m. No rule for "
                "the cells outside the dataset is decided.")
        self.record["dataset_file"] = info
        return info

    def log10_at(self, xyz_m):
        """The log10 viscosity of the dataset at points, before the bound.

        Collective over the ranks of a node. The points of every rank of the
        node go to the rank that holds the dataset, which interpolates them
        in chunks of `GDRIFT_CHUNK` and sends each rank its values.

        Args:
          xyz_m: Cartesian points (n, 3) in metres, z to the north pole and
            x to longitude 0, the frame of the mesh and of g-drift
            (`gdrift.geodetic_to_cartesian`).

        Returns:
          The interpolated log10 viscosity (Pa s), an array of n values.

        Raises:
          RuntimeError: if a value is NaN, which g-drift returns for a point
            farther than its search distance from the dataset.
        """
        xyz_m = np.ascontiguousarray(xyz_m, dtype=float)
        counts = self.node.gather(len(xyz_m), root=0)
        points = self.node.gather(xyz_m, root=0)
        parts = None
        if self.node.rank == 0:
            points = np.concatenate(points)
            values = np.empty(len(points))
            for start in range(0, len(points), GDRIFT_CHUNK):
                stop = min(start + GDRIFT_CHUNK, len(points))
                values[start:stop] = np.atleast_1d(
                    self.model.at(self.field, points[start:stop]))
            parts = np.split(values, np.cumsum(counts)[:-1])
        values = self.node.scatter(parts, root=0)
        if COMM_WORLD.allreduce(int(np.isnan(values).sum()), op=MPI.SUM):
            raise RuntimeError(f"{self.dataset}: NaN at a mantle point")
        return values

    def lower_bound_log10(self, depth_km):
        """The log10 of the lower bound at the given depths (km below Re).

        A depth on a bound boundary belongs to the deeper layer. No cell
        centroid lies on one, because the bound boundary (670 km) is a mesh
        sphere.
        """
        index = np.searchsorted(self.lower_top_km, depth_km, side="right") - 1
        return self.lower_log10[np.clip(index, 0, len(self.lower_log10) - 1)]

    def release(self):
        """Drop the dataset from memory."""
        self.model = None


def gdrift_version():
    """The version of the g-drift package that runs, for the record."""
    info = {"version": getattr(gdrift, "__version__", None),
            "path": os.path.dirname(gdrift.__file__)}
    try:
        info["metadata_version"] = importlib.metadata.version("gdrift")
    except Exception:  # noqa: BLE001  (a version is a record, not a need)
        info["metadata_version"] = None
    return info


@dataclasses.dataclass(frozen=True)
class Exp11Earth:
    """The earth model of Exp11.

    `giamip.earth.for_case` cannot build it, because its viscosity varies
    laterally. The elastic part is the `giamip.earth.ElasticProfile` of the
    G-ADOPT choice (PREM-stable), the viscosity the `Viscosity3D` of the
    G-ADOPT choice (`lloyd2024`).

    Attributes:
      elastic: the elastic structure and density.
      viscosity: the 3-D viscosity.
      choices: the G-ADOPT choices of the case file.
    """

    elastic: ElasticProfile
    viscosity: Viscosity3D
    choices: dict


def exp11_earth(experiment):
    """The earth model of the experiment, from the `gadopt` block of its case.

    Raises:
      NotImplementedError: for a rheology other than Maxwell or an
        incompressible earth, which the solver of this directory does not
        run.
    """
    case = cases.with_gadopt_choices(experiment.id)
    earth = case["earth"]
    if earth["rheology"] != "maxwell":
        raise NotImplementedError(f"rheology {earth['rheology']!r}")
    if earth["compressibility"] != "compressible":
        raise NotImplementedError(f"compressibility "
                                  f"{earth['compressibility']!r}")
    elastic = ElasticProfile.from_model(earth["elastic_structure"])
    viscosity = Viscosity3D(earth["viscosity"], elastic.surface_radius_m,
                            elastic.core_mantle_boundary_m)
    return Exp11Earth(elastic=elastic, viscosity=viscosity,
                      choices=cases.gadopt_choices(experiment.id))


def experiment_topography(experiment):
    """The topography file of the experiment.

    The case file of Exp11 leaves the topography open, so
    `GiamipExperiment.topography` is None. `giamip.cases.with_gadopt_choices`
    fills in the file of the G-ADOPT choice (`bedmap3`), taken from a case
    that the protocol gives that topography. For a case that prescribes a
    file this is the prescribed file.
    """
    relative_path = cases.with_gadopt_choices(experiment.id)["topography"]["file"]
    return Topography(relative_path)


# --------------------------------------------------------------------------
# The earth model on the mesh
# --------------------------------------------------------------------------


def node_radius_m(space):
    """The radius of every node of a scalar space, in metres, with halos."""
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)
    r = Function(space).interpolate(sqrt(dot(X, X)))
    return np.array(r.dat.data_ro_with_halos) * D_SCALE


def earth_fields(mantle, earth, scales):
    """The density, the moduli, the viscosity and gravity on the mantle mesh.

    Each DG1 node takes the value of the elastic structure at its own
    radius. The DG1 element of Firedrake on a tetrahedron has its four
    nodes at interior Gauss-Legendre points (`finat.spectral.GaussLegendre`),
    not at the vertices, so the field is the linear interpolant through four
    interior values, and no node lies on a mesh sphere. The side rule below
    is kept for an element with nodes on the cell boundary: a node on a
    sphere where a property jumps takes the value of the side where its cell
    lies, "above" when the node is below the cell centroid and "below" when
    it is above it. A DG node belongs to one cell only, so this is defined
    for every node.

    The viscosity of each cell is the log10 viscosity of the 3-D model at
    the cell centroid, raised to the lower bound of the model file at the
    depth of the centroid, then taken to the power 10 (see the module
    docstring). Every cell takes its value from the model; no lid is added.

    Args:
      mantle: the mantle mesh (P2 coordinates).
      earth: the `Exp11Earth` of the experiment; its viscosity must be
        loaded (`Viscosity3D.load`).
      scales: the `exp11_common.Scales`.

    Returns:
      `(fields, viscosity_fields)`. `fields` is a dictionary with `density`,
      `shear_modulus`, `bulk_modulus` (DG1), `viscosity` (DG0) and `gravity`
      (CG2), all non-dimensional. `viscosity_fields` holds DG0 fields for the
      record: `log10_viscosity_raw` (the model at the centroid),
      `log10_viscosity` (after the lower bound) and `at_lower_bound` (1 in a
      cell where the bound applies, else 0).
    """
    dg1 = FunctionSpace(mantle, "DG", 1)
    dg0 = FunctionSpace(mantle, "DG", 0)
    X = SpatialCoordinate(mantle)
    r_expr = sqrt(dot(X, X))
    # The radius of each DG1 node and of the centroid of its cell. The DG0
    # value is the value at the centroid, and its interpolation into DG1
    # copies it to the four nodes of the cell.
    r_node = node_radius_m(dg1)
    centroid = Function(dg0).interpolate(r_expr)
    r_cell = np.array(Function(dg1).interpolate(centroid)
                      .dat.data_ro_with_halos) * D_SCALE
    # A node can lie a rounding distance above the surface or below the CMB.
    # The earth model refuses radii outside [0, Re].
    r_node = np.clip(r_node, scales.Rc_m, scales.Re_m)
    above = earth.elastic(r_node, "above")
    below = earth.elastic(r_node, "below")
    use_above = r_node <= r_cell

    def dg1_field(column, scale, name):
        field = Function(dg1, name=name)
        field.dat.data_with_halos[:] = np.where(
            use_above, above[column], below[column]) / scale
        return field

    rho = dg1_field("rho_kg_m3", scales.rho_bar, "density")
    mu = dg1_field("mu_Pa", common.MU_BAR, "shear_modulus")
    kappa = dg1_field("kappa_Pa", common.MU_BAR, "bulk_modulus")

    # Viscosity at the centroid from the 3-D model. The DG0 interpolation of
    # the coordinates gives the centroid of each cell (the image of the
    # reference centroid under the P2 map, as for the radius above), in
    # units of D; g-drift takes metres. The halo cells are evaluated too, so
    # every rank holds the value of every cell it sees.
    xyz_m = np.array(Function(VectorFunctionSpace(mantle, "DG", 0))
                     .interpolate(X).dat.data_ro_with_halos) * D_SCALE
    log10_raw = earth.viscosity.log10_at(xyz_m)
    # The lower bound by the depth of the centroid below the surface Re. The
    # bound acts on log10, the quantity that was interpolated.
    depth_km = (scales.Re_m - np.linalg.norm(xyz_m, axis=1)) / 1e3
    floor = earth.viscosity.lower_bound_log10(depth_km)
    log10_eta = np.maximum(log10_raw, floor)
    if not np.isfinite(log10_eta).all():
        raise RuntimeError("a mantle cell has no viscosity")
    eta = Function(dg0, name="viscosity")
    eta.dat.data_with_halos[:] = 10.0**log10_eta / common.ETA_BAR

    def dg0_field(values, name):
        field = Function(dg0, name=name)
        field.dat.data_with_halos[:] = values
        return field

    viscosity_fields = {
        "log10_viscosity_raw": dg0_field(log10_raw, "log10_viscosity_raw"),
        "log10_viscosity": dg0_field(log10_eta, "log10_viscosity"),
        "at_lower_bound": dg0_field((log10_raw < floor).astype(float),
                                    "at_lower_bound")}

    # Gravity is continuous; the side does not matter.
    cg2 = FunctionSpace(mantle, "CG", 2)
    r_cg2 = np.clip(node_radius_m(cg2), scales.Rc_m, scales.Re_m)
    g = Function(cg2, name="gravity")
    g.dat.data_with_halos[:] = earth.elastic(r_cg2)["g_m_s2"] / scales.g_bar
    return ({"density": rho, "shear_modulus": mu, "bulk_modulus": kappa,
             "viscosity": eta, "gravity": g}, viscosity_fields)


def core_density(elastic, scales, rule):
    """The density of the homogeneous fluid core, kg/m^3.

    Args:
      elastic: the `giamip.earth.ElasticProfile`.
      scales: the `exp11_common.Scales`.
      rule: `"mean"`, the mean density of the core from its mass
        g(Rc) Rc^2 / G, which gives the core its mass, or `"cmb"`, the
        density just below the CMB, which makes the core too light.
    """
    if rule == "cmb":
        return float(elastic(scales.Rc_m, "below")["rho_kg_m3"])
    g_cmb = float(elastic(scales.Rc_m, "below")["g_m_s2"])
    mass = g_cmb * scales.Rc_m**2 / common.G_NEWTON
    return mass / (4.0 / 3.0 * np.pi * scales.Rc_m**3)


def solver_constants(experiment, earth, scales, rule):
    """The constants of the solver that come from the experiment.

    The inversion reads these numbers from the record and does not read the
    earth model or the case file.

    Args:
      experiment: the `GiamipExperiment`.
      earth: its `Exp11Earth`.
      scales: the `exp11_common.Scales`.
      rule: the core density rule (`--core_density`).

    Returns:
      A dict of plain numbers, SI and non-dimensional.
    """
    rho_core = core_density(earth.elastic, scales, rule)
    g_cmb = float(earth.elastic(scales.Rc_m, "below")["g_m_s2"])
    rho_w = float(experiment.constants["ocean_water_density_kg_m3"])
    rho_i = float(experiment.constants["ice_density_kg_m3"])
    # The observed polar moment, kg m^2, and in units of rho_bar D^5.
    C_kg_m2 = common.C_OVER_MRE2 * scales.earth_mass_kg * scales.Re_m**2
    return {"core_density_rule": rule, "rho_core_kg_m3": rho_core,
            "rho_core_nondim": rho_core / scales.rho_bar,
            "g_cmb_m_s2": g_cmb, "g_cmb_nondim": g_cmb / scales.g_bar,
            "rho_w_kg_m3": rho_w, "rho_w_nondim": rho_w / scales.rho_bar,
            "rho_i_kg_m3": rho_i, "rho_i_nondim": rho_i / scales.rho_bar,
            "C_over_MRe2": common.C_OVER_MRE2, "C_kg_m2": C_kg_m2,
            "C_nondim": C_kg_m2 / (scales.rho_bar * D_SCALE**5)}


# --------------------------------------------------------------------------
# Statistics per mantle layer
# --------------------------------------------------------------------------


def mantle_layers(radial):
    """The mantle layers of the mesh: `(name, r_bottom, r_top)`, nondim."""
    spheres = radial.to_dict()["spheres"]
    layers = radial.to_dict()["layers"]
    out = []
    for name in radial.to_dict()["regions"]["mantle"]:
        lower, upper = layers[name]
        out.append((name, spheres[lower], spheres[upper]))
    return sorted(out, key=lambda item: -item[2])


class LayerCells:
    """The owned mantle cells of each layer, by the radius of the centroid.

    A cell lies in one layer, because no cell crosses a sphere, so its
    centroid radius tells the layer.

    Args:
      mantle: the mantle mesh.
      layers: the list of `mantle_layers`.
    """

    def __init__(self, mantle, layers):
        self.mantle = mantle
        self.layers = layers
        self.dg0 = FunctionSpace(mantle, "DG", 0)
        X = SpatialCoordinate(mantle)
        r = Function(self.dg0).interpolate(sqrt(dot(X, X)))
        n_owned = mantle.cell_set.size
        # The DG0 node of each owned cell (DG0 has one node per cell).
        self.cell_nodes = self.dg0.cell_node_list[:n_owned, 0]
        r_cell = r.dat.data_ro_with_halos[self.cell_nodes]
        self.masks = {name: (r_cell > lo) & (r_cell < hi)
                      for name, lo, hi in layers}
        unassigned = n_owned - sum(int(m.sum()) for m in self.masks.values())
        if COMM_WORLD.allreduce(unassigned, op=MPI.SUM):
            raise RuntimeError("a mantle cell lies in no layer")
        # The indicator of each layer, for volume integrals.
        self.indicators = {}
        for name, mask in self.masks.items():
            indicator = Function(self.dg0)
            indicator.dat.data_with_halos[self.cell_nodes] = mask.astype(float)
            self.indicators[name] = indicator
        self.volume = {name: float(assemble(ind * dx(domain=mantle)))
                       for name, ind in self.indicators.items()}

    def statistics(self, f, scale):
        """Min, max and volume mean of `f` (times `scale`) in each layer.

        The min and max are over the nodal values of the owned cells of the
        layer, after an exact interpolation into DG2 with equispaced nodes
        (DG2 contains DG0, DG1 and CG2 on every cell, so the nodal values are
        values of the field). The equispaced nodes include the four vertices
        of each cell, where a linear field takes its extreme values; the
        default DG2 nodes are interior points and would miss them.
        """
        dg2 = FunctionSpace(self.mantle, "DG", 2, variant="equispaced")
        values = Function(dg2).interpolate(f).dat.data_ro_with_halos
        nodes = dg2.cell_node_list[:self.mantle.cell_set.size]
        out = {}
        for name, _, _ in self.layers:
            local = values[nodes[self.masks[name]]]
            lo = local.min() if local.size else np.inf
            hi = local.max() if local.size else -np.inf
            mean = float(assemble(self.indicators[name] * f
                                  * dx(domain=self.mantle))) / self.volume[name]
            out[name] = {"min": COMM_WORLD.allreduce(float(lo), op=MPI.MIN) * scale,
                         "max": COMM_WORLD.allreduce(float(hi), op=MPI.MAX) * scale,
                         "mean": mean * scale}
        return out

    def distinct_values(self, f, scale, limit=10):
        """The distinct DG0 values of `f` in each layer, with volume fractions.

        For a piecewise constant viscosity model this shows where a layer
        boundary that is not a mesh sphere ended up: a shell with two values
        and their volume fractions. A layer with more than `limit` distinct
        values (a 3-D model) is reported as `"continuous"`.
        """
        values = f.dat.data_ro_with_halos[self.cell_nodes]
        out = {}
        for name, _, _ in self.layers:
            local = np.unique(values[self.masks[name]])
            every = np.unique(np.concatenate(COMM_WORLD.allgather(local)))
            if every.size > limit:
                out[name] = "continuous"
                continue
            rows = []
            for value in every:
                # The volume of the cells of this layer that carry the value.
                pick = Function(self.dg0)
                pick.dat.data_with_halos[self.cell_nodes] = (
                    self.masks[name] & (values == value)).astype(float)
                fraction = float(assemble(pick * dx(domain=self.mantle))) \
                    / self.volume[name]
                rows.append({"value": float(value) * scale,
                             "volume_fraction": fraction})
            out[name] = rows
        return out


# --------------------------------------------------------------------------
# The grid and the forcing
# --------------------------------------------------------------------------


def lat_lon_of_points(xyz):
    """Latitude and longitude in degrees of Cartesian points (n, 3)."""
    r = np.linalg.norm(xyz, axis=1)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0)))
    lon = np.mod(np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), 360.0)
    return lat, lon


class BilinearTransfer:
    """Bilinear interpolation from the Gauss-Legendre grid to scattered points.

    The same rule as `bilinear_to_fine` of the mesh script
    (`meshing-gia/NOTES/feasibility/giamip/n2_giamip.py`), with which the
    shared mesh was refined for the Exp01 coastline: linear in
    longitude with the periodic wrap from 359.3 to 0 degrees, then linear in
    latitude between the real Gauss-Legendre node latitudes. North and south
    of the outermost rows (|lat| > 89.465 degrees) the value of the outermost
    row is held. The weights are computed once.
    """

    def __init__(self, lat_nodes, lon_nodes, lat, lon):
        nlon = lon_nodes.size
        dlon = 360.0 / nlon
        s = (np.mod(lon, 360.0) - lon_nodes[0]) / dlon
        self.j0 = np.floor(s).astype(int) % nlon
        self.j1 = (self.j0 + 1) % nlon
        self.wj = s - np.floor(s)
        lat_c = np.clip(lat, lat_nodes[0], lat_nodes[-1])
        self.i1 = np.clip(np.searchsorted(lat_nodes, lat_c, side="right"), 1,
                          lat_nodes.size - 1)
        self.i0 = self.i1 - 1
        self.wi = ((lat_c - lat_nodes[self.i0])
                   / (lat_nodes[self.i1] - lat_nodes[self.i0]))

    def __call__(self, field):
        """The field (nlat, nlon, south first) at the points."""
        f = np.asarray(field, dtype=float)
        lower = (1 - self.wj) * f[self.i0, self.j0] + self.wj * f[self.i0, self.j1]
        upper = (1 - self.wj) * f[self.i1, self.j0] + self.wj * f[self.i1, self.j1]
        return (1 - self.wi) * lower + self.wi * upper


def coarse_step_indices(years, step_years):
    """The indices of the model years that a uniform coarser step keeps.

    Keeps every year whose distance from the first year is a whole multiple
    of `step_years`, and the last year. With the Exp11 years (500 yr steps
    to 26 ka, then 100 yr steps) and `step_years = 2000` this keeps -122000,
    -120000, ..., 0 and 1000. The ice history is then read at its own time
    slices and never interpolated in time.

    Args:
      years: the model years of the protocol, ascending, in years.
      step_years: the uniform step, in years.

    Returns:
      A numpy array of indices into `years`.
    """
    start = int(round(years[0]))
    keep = [i for i, year in enumerate(years)
            if (int(round(year)) - start) % int(step_years) == 0]
    if keep[-1] != len(years) - 1:
        keep.append(len(years) - 1)
    return np.array(keep, dtype=int)


class SurfaceForcing:
    """The observed bed and the ice on the mantle mesh, from the GIAMIP files.

    `SL_obs = -bed / D` is CG2, because the mask width reads its slope. The
    ice is CG1 (see `exp11_common.ICE_SPACE`). Every node of the mantle mesh
    takes the value of its direction, so the fields are defined in the whole
    mantle and are read only on Re.

    Args:
      mantle: the mantle mesh.
      experiment: the `GiamipExperiment`.
      step_years: the uniform step of the march, years.
    """

    def __init__(self, mantle, experiment, step_years):
        self.experiment = experiment
        self.ice_history = experiment.ice_history
        # The case file leaves the topography open; the file is the one of
        # the G-ADOPT choice.
        self.topography = experiment_topography(experiment)
        lat_nodes = self.topography.latitudes()
        lon_nodes = grids.longitudes()
        # Both files must be on the GIAMIP Gauss-Legendre grid.
        grids.check(lat_nodes, np.asarray(self.topography._ds["lon"][:],
                                          dtype=float), name="topography")
        grids.check(self.ice_history.latitudes(),
                    np.asarray(self.ice_history._ds["lon"][:], dtype=float),
                    name="ice history")
        self.lat_nodes, self.lon_nodes = lat_nodes, lon_nodes

        cg2 = FunctionSpace(mantle, "CG", 2)
        cg1 = FunctionSpace(mantle, *common.ICE_SPACE)
        self._cg2_transfer = self._transfer(cg2)
        self._cg1_transfer = self._transfer(cg1)
        self.cg1 = cg1

        bed = self.topography.elevation()
        self.SL_obs = Function(cg2, name="SL_obs")
        self.SL_obs.dat.data_with_halos[:] = -self._cg2_transfer(bed) / D_SCALE

        # The steps of the march: a subset of the protocol years at a uniform
        # step, read at their own time slices.
        steps = experiment.forcing_steps()
        years = experiment.model_years()
        keep = coarse_step_indices(years, step_years)
        self.steps, self.years = steps[keep], years[keep]

    def _transfer(self, space):
        mesh = space.mesh()
        X = Function(VectorFunctionSpace(mesh, space.ufl_element().family(),
                                         space.ufl_element().degree()))
        X.interpolate(SpatialCoordinate(mesh))
        lat, lon = lat_lon_of_points(np.array(X.dat.data_ro_with_halos))
        return BilinearTransfer(self.lat_nodes, self.lon_nodes, lat, lon)

    def thickness_m(self, index):
        """The ice thickness of march step `index` on the grid, m."""
        return self.ice_history.thickness(int(self.steps[index]))

    def ice(self, index):
        """The ice of march step `index` as a new CG1 Function, units of D."""
        values = self._cg1_transfer(self.thickness_m(index)) / D_SCALE
        target = Function(self.cg1, name=common.ICE_NAME)
        # The bilinear interpolant of non-negative data is non-negative; the
        # clip only removes rounding.
        target.dat.data_with_halos[:] = np.maximum(values, 0.0)
        return target

    def grid_ice_volume_m3(self, index, radius_m):
        """The ice volume of step `index` on the grid (Gauss-Legendre), m^3."""
        return grids.integrate(self.thickness_m(index), radius_m)


def data_file_record(f):
    """The record of one GIAMIP data file: path, SHA-256, corrections."""
    return {"relative_path": f.relative_path, "sha256": f.sha256,
            "is_current_version": bool(f.is_current_version),
            "corrections": [c.get("id") for c in f.corrections]}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True,
                   help="the coarse giamsh .msh file, or a P1 mesh in a "
                        "CheckpointFile (.h5, refined already, for example "
                        "giamip_8re_r4_linear.h5); its .json sidecar must be "
                        "beside it unless --sidecar names it")
    p.add_argument("--mesh_name", default=None,
                   help="the name of the mesh in a .h5 mesh file")
    p.add_argument("--sidecar", default=None,
                   help="the .json sidecar of the mesh (default: the mesh "
                        "file with the suffix .json)")
    p.add_argument("--mesh_md5", default=MESH_MD5,
                   help="the expected MD5 of the mesh file ('' to skip)")
    p.add_argument("--refine", type=int, default=0,
                   help="uniform refinement levels (0 for L0; a .h5 mesh "
                        "takes 0 only)")
    p.add_argument("--output_path", default="./")
    p.add_argument("--ice_steps", type=int, default=None,
                   help="write the ice of steps 0 .. this step only (short "
                        "tests on a large mesh; the inversion then runs "
                        "with --steps at most this value)")
    p.add_argument("--step_years", type=int, default=2000,
                   help="the uniform step of the march, years")
    p.add_argument("--core_density", choices=["mean", "cmb"], default="mean")
    p.add_argument("--no_vtk", action="store_true",
                   help="do not write the ParaView files")
    return p.parse_args()


def main():
    """Make and write the input file."""
    args = parse_args()
    tic_run = time.time()
    if COMM_WORLD.rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    COMM_WORLD.barrier()
    say = common.RunLog(os.path.join(args.output_path, "prepare_exp11.log"))
    say("=" * 78)
    say(f"GIAMIP {EXPERIMENT}: input file of the bed inversion, "
        f"{COMM_WORLD.size} ranks")
    say("=" * 78)
    timings = {}

    # The experiment and its earth model, with the G-ADOPT choices for the
    # fields that the protocol leaves open.
    experiment = GiamipExperiment(EXPERIMENT)
    earth = exp11_earth(experiment)
    scales = common.Scales.from_elastic(earth.elastic)
    say(f"  G-ADOPT choices: {earth.choices}")
    say(f"  scales: {scales.as_dict()}")

    # The mesh, with a check that it is the shared mesh of the handout and
    # that its surface is the surface of the earth model.
    md5 = common.md5sum(args.mesh) if COMM_WORLD.rank == 0 else None
    md5 = COMM_WORLD.bcast(md5, root=0)
    if args.mesh_md5 and md5 != args.mesh_md5:
        raise RuntimeError(f"{args.mesh} has MD5 {md5}, not {args.mesh_md5}")
    sidecar = args.sidecar or os.path.splitext(args.mesh)[0] + ".json"
    with open(sidecar) as handle:
        sidecar_record = json.load(handle)
    radial = RadialStructure.from_dict(sidecar_record["radial"])
    if abs(radial.radius("surface") - scales.Re) > 1e-9 * scales.Re:
        raise RuntimeError(f"mesh surface radius {radial.radius('surface')} "
                           f"against the earth model {scales.Re}")
    tic = time.time()
    if args.mesh.endswith(".h5"):
        # A refined mesh in a CheckpointFile: loaded with the mantle-balanced
        # partition and curved here, never refined again.
        if args.refine:
            raise RuntimeError("a .h5 mesh is refined already; --refine "
                               "must be 0")
        if not args.mesh_name:
            raise RuntimeError("a .h5 mesh needs --mesh_name")
        parent, mantle, counts = common.load_checkpoint_meshes(
            args.mesh, args.mesh_name, radial)
        # The sidecar of a refined mesh records its cells per layer. The
        # loaded mesh and its mantle submesh must have the same counts; a
        # label that marks points other than cells would change them.
        final = sidecar_record.get("final", {})
        if "cells_per_layer" in final:
            expected = {
                "cells_parent": int(final["cells"]),
                "cells_mantle": int(sum(final["cells_per_layer"][name] for
                                        name in radial.to_dict()["regions"]
                                        ["mantle"]))}
            found = {key: int(counts[key]) for key in expected}
            if found != expected:
                raise RuntimeError(f"cell counts {found} against the sidecar "
                                   f"{expected}")
    else:
        parent, mantle, counts = common.build_meshes(args.mesh, radial,
                                                     args.refine)
    timings["mesh_s"] = time.time() - tic
    mesh_record = {"file": os.path.abspath(args.mesh), "md5": md5,
                   "mesh_name": args.mesh_name,
                   "sidecar": os.path.abspath(sidecar),
                   "sidecar_inputs": sidecar_record.get("inputs"),
                   "refine_levels": args.refine,
                   "distribution_name": common.distribution_name(),
                   "mantle_tags": list(radial.region_tags("mantle")),
                   **counts}
    say(f"  mesh: {mesh_record} ({timings['mesh_s']:.1f} s)")
    common.memory_now("after the mesh", say)

    # The earth fields. The 3-D viscosity is loaded from g-drift for this
    # step only and released afterwards.
    tic = time.time()
    dataset_info = earth.viscosity.load()
    say(f"  viscosity: {earth.viscosity.dataset} ({earth.viscosity.field}), "
        f"g-drift {earth.viscosity.record['gdrift']}, file {dataset_info}")
    say(f"  lower bound (log10 Pa s, by depth): "
        f"{earth.viscosity.record['lower_bound']}; no upper bound; no lid")
    common.memory_now("with the viscosity dataset", say)
    fields, viscosity_fields = earth_fields(mantle, earth, scales)
    earth.viscosity.release()
    timings["earth_s"] = time.time() - tic

    # The forcing: observed sea level, its frozen slope, and the ice.
    tic = time.time()
    forcing = SurfaceForcing(mantle, experiment, args.step_years)
    fields["SL_obs"] = forcing.SL_obs
    # The surface slope of the ocean mask, from the observed bed. It stays
    # fixed in the inversion: the mask width does not depend on the control.
    fields["slope"] = surface_slope(forcing.SL_obs)
    fields["slope"].rename("slope")
    years = np.asarray(forcing.years, dtype=float)
    year0 = int(np.flatnonzero(np.isclose(years, 0.0))[0])
    # The march of the inversion runs from the first year to year 0; the
    # file holds the ice of steps 0 .. year0, or of steps 0 .. --ice_steps
    # for a short test (the full ice of the 8 Re mesh is about 200 GB).
    n_ice = year0 + 1
    if args.ice_steps is not None:
        n_ice = min(n_ice, args.ice_steps + 1)
    say(f"  march: {year0} steps of {args.step_years} yr, years "
        f"{years[0]:.0f} .. {years[year0]:.0f}; ice of steps 0 .. "
        f"{n_ice - 1} in the file")

    # The ice of every step, made once and kept: the writer and the volume
    # check use it. At L0 one CG1 field is small.
    ice = [forcing.ice(index) for index in range(n_ice)]
    surface = ds(common.SURF_RE, domain=mantle)
    ice_check = []
    for index, f in enumerate(ice):
        mesh_m3 = float(assemble(f * surface)) * D_SCALE**3
        grid_m3 = forcing.grid_ice_volume_m3(index, scales.Re_m)
        ice_check.append({"step": index, "year": float(years[index]),
                          "protocol_step": int(forcing.steps[index]),
                          "mesh_volume_m3": mesh_m3,
                          "grid_volume_m3": grid_m3,
                          "relative_error": ((mesh_m3 - grid_m3) / grid_m3
                                             if grid_m3 else 0.0)})
    worst = max(abs(row["relative_error"]) for row in ice_check)
    say(f"  ice: {n_ice} steps, largest relative error of the volume on the "
        f"mesh against the grid {worst:.3e}")
    timings["forcing_s"] = time.time() - tic

    # The statistics of the earth fields per mantle layer, in SI units.
    tic = time.time()
    layers = mantle_layers(radial)
    cells = LayerCells(mantle, layers)
    # name -> (unit, scale to SI, name in the ParaView file).
    si = {"density": ("kg/m^3", scales.rho_bar, "density_kg_m3"),
          "shear_modulus": ("Pa", common.MU_BAR, "shear_modulus_pa"),
          "bulk_modulus": ("Pa", common.MU_BAR, "bulk_modulus_pa"),
          "viscosity": ("Pa s", common.ETA_BAR, "viscosity_pa_s"),
          "gravity": ("m/s^2", scales.g_bar, "gravity_m_s2")}
    layer_stats = {name: {"unit": unit,
                          "layers": cells.statistics(fields[name], scale)}
                   for name, (unit, scale, _) in si.items()}
    viscosity_by_layer = cells.distinct_values(fields["viscosity"],
                                               common.ETA_BAR)
    for name, _, _ in layers:
        rows = viscosity_by_layer[name]
        text = rows if isinstance(rows, str) else ", ".join(
            f"{row['value']:.3g} Pa s ({100 * row['volume_fraction']:.1f} %)"
            for row in rows)
        say(f"    viscosity {name}: {text}")
    # The 3-D viscosity per layer, in log10 Pa s: min, max and volume mean
    # before and after the lower bound, and the volume fraction of the
    # cells at the bound (the volume mean of the 0/1 indicator).
    raw_stats = cells.statistics(viscosity_fields["log10_viscosity_raw"], 1.0)
    log10_stats = cells.statistics(viscosity_fields["log10_viscosity"], 1.0)
    bound_stats = cells.statistics(viscosity_fields["at_lower_bound"], 1.0)
    log10_by_layer = {}
    for name, lo, hi in layers:
        log10_by_layer[name] = {
            "depth_km": [(scales.Re - hi) * D_SCALE / 1e3,
                         (scales.Re - lo) * D_SCALE / 1e3],
            "log10_viscosity": log10_stats[name],
            "log10_viscosity_raw": raw_stats[name],
            "lower_bound_volume_fraction": bound_stats[name]["mean"]}
        row = log10_stats[name]
        say(f"    log10 viscosity {name}: min {row['min']:.3f}, max "
            f"{row['max']:.3f}, mean {row['mean']:.3f} (raw min "
            f"{raw_stats[name]['min']:.3f}); at the lower bound "
            f"{100 * bound_stats[name]['mean']:.1f} % of the volume")
    mantle_volume = float(assemble(1.0 * dx(domain=mantle)))
    bound_fraction = float(assemble(viscosity_fields["at_lower_bound"]
                                    * dx(domain=mantle))) / mantle_volume
    say(f"    at the lower bound: {100 * bound_fraction:.1f} % of the mantle "
        "volume")
    earth.viscosity.record["by_layer"] = log10_by_layer
    earth.viscosity.record["lower_bound_volume_fraction_mantle"] = \
        bound_fraction
    timings["statistics_s"] = time.time() - tic

    record = {
        "experiment": EXPERIMENT,
        "case": experiment.definition,
        # The case as the G-ADOPT run uses it: the choices of the `gadopt`
        # block filled into the open fields.
        "case_gadopt": cases.with_gadopt_choices(experiment.id),
        "gadopt_choices": earth.choices,
        # The rheology and compressibility after the G-ADOPT choices, under
        # the same keys as in the records of the 1D experiments.
        "rheology": earth.choices["rheology"],
        "compressibility": earth.choices["compressibility"],
        "giamip_version": giamip.__version__,
        "giamip_path": os.path.dirname(giamip.__file__),
        "viscosity_3d": earth.viscosity.record,
        "inputs": {"ice_history": data_file_record(forcing.ice_history),
                   "topography": data_file_record(forcing.topography)},
        "mesh": mesh_record,
        "march": {"step_years": args.step_years, "years": years[:n_ice],
                  "protocol_steps": forcing.steps[:n_ice],
                  "ice_steps": n_ice, "year0_step": year0},
        "scales": scales.as_dict(),
        "solver_constants": solver_constants(experiment, earth, scales,
                                             args.core_density),
        "earth_by_layer": layer_stats,
        "viscosity_by_layer": viscosity_by_layer,
        "ice_volume_check": ice_check,
        # The numbers that the inversion computes again after it loads the
        # file (`exp11_common.compare_statistics`).
        "statistics": common.input_statistics(mantle, fields, ice),
        "fields": {name: list(space)
                   for name, space in common.MANTLE_FIELDS.items()},
        "timings": timings,
        "provenance": common.provenance(HERE) if COMM_WORLD.rank == 0 else None,
    }
    record["provenance"] = COMM_WORLD.bcast(record["provenance"], root=0)

    # The input file and its record.
    tic = time.time()
    path = os.path.join(args.output_path, f"{STEM}.h5")
    common.write_inputs(path, parent, fields, lambda index: ice[index],
                        record)
    timings["write_s"] = time.time() - tic
    common.write_json(os.path.join(args.output_path, f"{STEM}.json"), record)
    say(f"  wrote {path} ({timings['write_s']:.1f} s)")

    # ParaView files for the reviewer: the earth fields in SI units, and the
    # ice at the first, middle and last step of the march.
    if not args.no_vtk:
        tic = time.time()
        dg1 = FunctionSpace(mantle, "DG", 1)
        si_fields = [Function(dg1, name=vtk_name)
                     .interpolate(fields[name] * scale)
                     for name, (_, scale, vtk_name) in si.items()]
        VTKFile(os.path.join(args.output_path, f"{STEM}_earth.pvd")).write(
            *si_fields)
        ice_file = VTKFile(os.path.join(args.output_path, f"{STEM}_ice.pvd"))
        ice_m = Function(forcing.cg1, name="ice_m")
        for index in sorted({0, n_ice // 2, n_ice - 1}):
            ice_m.assign(ice[index] * D_SCALE)
            ice_file.write(ice_m, time=float(years[index]))
        timings["vtk_s"] = time.time() - tic

    record["timings"]["total_s"] = time.time() - tic_run
    record["timings"]["peak_memory_gb"] = common.peak_memory_gb()
    common.write_json(os.path.join(args.output_path, f"{STEM}.json"), record)
    say(f"  total {record['timings']['total_s']:.1f} s, peak memory "
        f"{record['timings']['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
