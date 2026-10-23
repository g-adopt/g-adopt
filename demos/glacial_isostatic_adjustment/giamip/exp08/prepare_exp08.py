r"""GIAMIP Exp08: make the input file of the bed inversion.

WHAT THIS SCRIPT DOES
    It makes every field of the inversion that comes from data, once, and
    writes them to one Firedrake `CheckpointFile` (`inputs_exp08.h5`). The
    inversion (`invert_exp08.py`) loads that file and reads no data file, so
    its tape holds only the operations from the control to the misfit.

    1. Read the coarse giamsh mesh, partition it so that every rank holds
       the same number of mantle cells, refine it `--refine` times, curve it
       to P2 and cut the mantle submesh (`exp08_common.build_meshes`). A
       mesh that is refined already and saved as a P1 `CheckpointFile`
       (`.h5`, for example the 8 Re mesh `giamip_8re_r4_linear`) is loaded
       with the same mantle-balanced partition and curved
       (`exp08_common.load_checkpoint_meshes`).
    2. Make the earth fields on the mantle from the earth model of the
       experiment (`giamip.earth` for the 1D part, g-drift for the GLAD-M25
       relative change of vs and vp): density, shear and bulk modulus (DG1),
       viscosity (DG0) and gravity (CG2), all non-dimensional.
    3. Make the forcing on the mantle from the GIAMIP files: the observed sea
       level `SL_obs = -bed_obs / D` (CG2), the frozen surface slope of the
       ocean mask (`surface_slope(SL_obs)`, DG1), and the ice of every step
       of the march from the start to year 0 (CG1, one entry per step).
    4. Write the input file with its JSON record, the same record as a JSON
       file next to it, and ParaView files of the earth fields and of the ice
       at three steps for the reviewer.

    The script tapes nothing: `firedrake.adjoint` is not imported.

THE EXPERIMENT
    GIAMIP protocol v4.0, experiment Exp08, as `giamip` defines it
    (`giamip.mip.inputs.GiamipExperiment("Exp08")`, case file
    `giamip/resources/cases/giamip/exp08.json`). Exp08 is Exp05 with a
    laterally varying elastic structure. The protocol leaves the earth and
    the topography open; the values below are the G-ADOPT choices of the
    `gadopt` block of the case file:

        ice          PaleoMIST_1a (`iceHistory-PaleoMIST_1a.nc`)
        topography   `bedmach3`: BedMachine Antarctica 3 + BedMachine
                     Greenland 5 + GEBCO
                     (`bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc`). The
                     case file names no file, so `topography_of` maps the
                     G-ADOPT choice to the file of the GIAMIP cases that
                     prescribe that topography (`giamip.cases`). The file is
                     the observed present-day bed (year 0); the inversion
                     finds the bed at the start of the run.
        earth        `prem-glad-m25`: PREM-stable density, PREM-stable shear
                     and bulk modulus scaled by the lateral relative change
                     of vs and vp of GLAD-M25 (see below), with the Gowan et
                     al. (2021) viscosity, Maxwell rheology, compressible
        period       80 ka to year 0, in steps of `--step_years` (2000 yr:
                     40 steps)
        constants    ice 917, ocean water 1025 kg/m^3

    Every number of the experiment comes from `giamip` or from the g-drift
    dataset that the giamip model file names, never from this file.

THE EARTH MODEL ON THE MESH
    `giamip.earth` raises `NotImplementedError` for `prem-glad-m25`, because
    giamip holds no laterally varying model. `radial_earth` therefore builds
    the 1D part from the giamip model files: the elastic structure of the
    radial base of `prem-glad-m25.json` (PREM-stable) and the viscosity of
    the case (`gowan2021`). `giamip.earth` gives every property of that 1D
    earth at any radius, linear in radius between the nodes of the model
    file. Each DG1 node takes the value at its own radius. Gravity is a CG2
    field. The viscosity is DG0, the value at the cell centroid; an elastic
    layer ("inf" in the model file) takes `ETA_LID` = 1e40 Pa s.

    The mesh `n2_giamip_all_o1` has spheres at 60, 90, 120, 400, 670, 980,
    1260, 1470, 1950 and 2400 km depth, the CMB and the surface. No cell
    crosses a sphere, so a jump of the earth model at a sphere is exact. For
    Gowan the jumps at 120 and 670 km are spheres, so the viscosity has no
    boundary inside a cell. The record gives the volume fraction of each
    viscosity value in each shell (`viscosity_by_layer`). PREM-stable has
    jumps at 400 and 670 km (spheres) and at 15, 24.4, 80, 220, 600, 771 and
    2741 km (not spheres); a cell across one of the latter carries the
    linear interpolant of its nodal values.

THE ELASTIC STRUCTURE: GLAD-M25
    The giamip model file `prem-glad-m25.json` names the g-drift dataset,
    its fields, its SHA-256 and the rule. The script reads them from the
    file (`tomography_model_file`) and stops if the rule texts of the file
    differ from the texts that the code implements (`IMPLEMENTED_RULES`), so
    a change of the model file cannot pass without a change of the code.

    g-drift stores GLAD-M25 on 331 depths from 6 to 2891 km, each on the
    same equal-area point set. `RelativeChange` forms, on each depth of the
    dataset,

        dlnvs = vs / <vs>(r) - 1,    dlnvp = vp / <vp>(r) - 1,

    with `<.>(r)` the plain mean over the points of that depth, which is the
    mean over the sphere. The relative change is then interpolated to the
    DG1 nodes with the layered rule of `gdrift.EarthModel3D` (inverse
    distance squared over the 8 nearest directions on each of the two depths
    that bracket the node, then linear in radius). Above 6 km the 6 km values
    hold. With P the PREM-stable value at the node,

        mu = mu_P (1 + dlnvs)^2
        K  = rho_P (vp_P^2 (1 + dlnvp)^2 - 4/3 vs_P^2 (1 + dlnvs)^2)

    and the density stays PREM-stable, so the reference state stays
    hydrostatic.

    The G-ADOPT choice of 2026-10-19 (Sia): mu and K in the 0-60 km shell
    are the values of the GLAD-M25 rule at the cell centroid, so that the
    linear DG1 interpolant does not extrapolate the steep crustal changes of
    GLAD-M25 to negative or near-zero values at the cell vertices. The DG1
    nodes are interior points of each cell, and the linear field of a cell
    extrapolates from them to its vertices. The written DG1 field is the
    interpolation of `conditional(r_centroid > R_60, f_dg0, f_dg1)`, with
    `f_dg0` the rule at the centroid, `f_dg1` the rule at the DG1 nodes, and
    `R_60` the radius of the 60 km mesh sphere from the sidecar. No cell
    crosses that sphere, so mu and K are constant per cell in 0-60 km and
    linear below. The script stops unless mu and K are positive at every
    centroid in 0-60 km, at every vertex below 60 km, and at every vertex of
    the written fields. The record holds the rule, R_60, the cells of each
    rule and the vertex min and max of mu and K per layer
    (`moduli_lid_rule`).

    The relative change is lateral at each depth, so the radial jumps of
    GLAD-M25 (650 km, and between 405 and 410 km) do not enter mu and K; the
    jumps of mu and K at 400 and 670 km are those of PREM-stable. The record
    holds the per-depth check of this near the PREM-stable jumps on mesh
    spheres (`tomography.sphere_check`).

    Only rank 0 reads the dataset (about 0.9 GB with the two fields). The
    other ranks send it the positions of their DG1 nodes one rank at a time
    and get the relative change back, so the memory of the dataset is used
    once per job and the message size is that of one rank.

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
    srun python3 prepare_exp08.py --mesh <file.msh> --output_path <dir>
    The number of ranks must be the one of the inversion job.

OUTPUT (in --output_path)
    inputs_exp08.h5          the input file of the inversion
    inputs_exp08.json        its record (the same JSON as inside the file)
    inputs_exp08_earth.pvd   ParaView: the earth fields in SI units, with
                             dlnvs and dlnvp
    inputs_exp08_ice.pvd     ParaView: the ice at the first, middle and last
                             step of the march, m
    prepare_exp08.log        the log
"""

import argparse
import hashlib
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
import exp08_common as common  # noqa: E402
from gadopt import (COMM_WORLD, Function, FunctionSpace,  # noqa: E402
                    SpatialCoordinate, VectorFunctionSpace, VTKFile, assemble,
                    conditional, dot, ds, dx, sqrt)
from gadopt.sea_level_masks import surface_slope  # noqa: E402
from mpi4py import MPI  # noqa: E402

import gdrift  # noqa: E402
import giamip  # noqa: E402
from gdrift.io import fetch_dataset  # noqa: E402
from gdrift.seismic import SEISMIC_MODEL_PREFIX  # noqa: E402
from giamip import cases, grids  # noqa: E402
from giamip.earth import (ElasticProfile, RadialEarth,  # noqa: E402
                          ViscosityProfile)
from giamip.mip.inputs import GiamipExperiment, Topography  # noqa: E402
from giamsh.radial import RadialStructure  # noqa: E402

#: The experiment of this directory.
EXPERIMENT = "Exp08"
#: The file stem of the outputs.
STEM = "inputs_exp08"
#: The MD5 of the shared coarse mesh `n2_giamip_all_o1.msh` (handout section
#: 3). The run stops if the mesh file differs.
MESH_MD5 = "a4da84206dc82f5b458be66e4fd33844"

D_SCALE = common.D_SCALE

#: The rule texts of the giamip model file that this script implements, word
#: for word (`perturbation` block of `prem-glad-m25.json`, gia-mip 0cf5f39).
#: The script stops if the model file says anything else: the formulas below
#: are code, and a new rule in the model file needs new code.
#:   relative_change   -> `RelativeChange` (layer means, relative change)
#:   shear_modulus     -> `glad_moduli`
#:   bulk_modulus      -> `glad_moduli`
#:   density           -> `earth_fields` keeps the PREM-stable density
#:   above_top_depth   -> g-drift holds the top layer above it; checked
#:                        against `TOP_DEPTH_KM` in `RelativeChange`
IMPLEMENTED_RULES = {
    "relative_change": "dlnvs = vs / <vs>(r) - 1 and dlnvp = vp / <vp>(r) - 1, "
                       "with <.>(r) the mean of GLAD-M25 over the sphere at "
                       "each depth of the dataset",
    "shear_modulus": "mu = mu_P (1 + dlnvs)^2",
    "bulk_modulus": "K = rho_P (vp_P^2 (1 + dlnvp)^2 - 4/3 vs_P^2 (1 + dlnvs)^2)",
    "density": "PREM-stable, unchanged",
    "above_top_depth": "the values at 6 km, the top depth of the dataset",
}
#: The two fields of the dataset that the rule uses, in the order of the
#: model file. The code names them, so the model file must name these two.
IMPLEMENTED_FIELDS = ["vs", "vp"]
#: The data package that the code reads.
IMPLEMENTED_PACKAGE = "gdrift"
#: The top depth of the dataset that the `above_top_depth` text names, km.
#: The script checks it against the shallowest depth of the dataset.
TOP_DEPTH_KM = 6.0
#: Half the width, km, of the depth window around each PREM-stable jump on a
#: mesh sphere in which the per-depth check lists the GLAD-M25 depths. 30 km
#: around 400 km takes in the GLAD-M25 jump between 405 and 410 km, and
#: around 670 km the jump at 650 km.
CHECK_WINDOW_KM = 30.0
#: The mesh sphere (a name of the sidecar) whose shell above it takes the
#: centroid values of the GLAD-M25 rule (G-ADOPT choice of 2026-10-19, Sia).
#: Its radius is read from the sidecar.
LID_SPHERE = "d60"
#: The number of DG1 node positions that rank 0 passes to g-drift at once.
#: It bounds the memory of the neighbour search (8 neighbours per point).
GDRIFT_CHUNK = 500_000


# --------------------------------------------------------------------------
# The Exp08 earth model: the 1D part from giamip, the lateral part from
# g-drift
# --------------------------------------------------------------------------


def tomography_model_file(experiment):
    """The giamip model file of the elastic structure of the experiment.

    The G-ADOPT choice of the case (`elastic_structure`, `prem-glad-m25` for
    Exp08) names the model file. It is read through `importlib.resources`
    from the installed giamip package, so the job reads the file that its
    giamip carries. The function stops if the file is not a laterally
    varying elastic structure from g-drift, or if its rule texts or fields
    differ from the ones the code implements (`IMPLEMENTED_RULES`,
    `IMPLEMENTED_FIELDS`).

    Args:
      experiment: the `GiamipExperiment`.

    Returns:
      `(model, sha256)`: the model file as a dict, and the SHA-256 of its
      text.
    """
    choice = cases.with_gadopt_choices(experiment.id)["earth"]
    model_id = choice["elastic_structure"]
    resource = importlib.resources.files("giamip").joinpath(
        "resources", "models", f"{model_id}.json")
    text = resource.read_text()
    model = json.loads(text)
    sha256 = hashlib.sha256(text.encode()).hexdigest()
    if model.get("kind") != "elastic_structure" or "tomography" not in model:
        raise RuntimeError(f"{model_id} is not a laterally varying elastic "
                           "structure")
    tomography = model["tomography"]
    if tomography.get("package") != IMPLEMENTED_PACKAGE:
        raise RuntimeError(f"{model_id} names the package "
                           f"{tomography.get('package')!r}, the code reads "
                           f"{IMPLEMENTED_PACKAGE!r}")
    if list(tomography.get("fields", [])) != IMPLEMENTED_FIELDS:
        raise RuntimeError(f"{model_id} names the fields "
                           f"{tomography.get('fields')}, the code uses "
                           f"{IMPLEMENTED_FIELDS}")
    rules = model.get("perturbation", {})
    if rules != IMPLEMENTED_RULES:
        # Name every rule that differs, so the change is easy to read.
        keys = sorted(set(rules) | set(IMPLEMENTED_RULES))
        diff = {key: {"model_file": rules.get(key),
                      "code": IMPLEMENTED_RULES.get(key)}
                for key in keys if rules.get(key) != IMPLEMENTED_RULES.get(key)}
        raise RuntimeError(f"the rule texts of {model_id}.json differ from the "
                           f"texts that this script implements: {diff}")
    return model, sha256


def radial_earth(experiment, model):
    """The 1D part of the earth of the experiment.

    `GiamipExperiment.earth()` raises `NotImplementedError` for a laterally
    varying elastic structure, so the 1D earth is built here from the giamip
    model files: the elastic structure of the radial base of the model file
    (`model["base"]`, PREM-stable) and the viscosity structure of the case
    with the G-ADOPT choices (`gowan2021` for Exp08). The rheology and the
    compressibility are recorded as the case gives them. This is the earth
    that `giamip.earth.for_case` returns for a case with the same viscosity
    and PREM-stable, for example Exp05.

    Args:
      experiment: the `GiamipExperiment`.
      model: the model file of the elastic structure
        (`tomography_model_file`).

    Returns:
      A `giamip.earth.RadialEarth`.
    """
    choice = cases.with_gadopt_choices(experiment.id)["earth"]
    elastic = ElasticProfile.from_model(model["base"])
    viscosity = ViscosityProfile.from_model(choice["viscosity"], elastic)
    return RadialEarth(elastic=elastic, viscosity=viscosity,
                       rheology=choice["rheology"],
                       compressibility=choice["compressibility"])


def glad_moduli(rho, vp, vs, mu_P, dlnvs, dlnvp):
    """The shear and bulk modulus of the GLAD-M25 rule, Pa.

    The rule of the model file (`IMPLEMENTED_RULES`):

        mu = mu_P (1 + dlnvs)^2
        K  = rho_P (vp_P^2 (1 + dlnvp)^2 - 4/3 vs_P^2 (1 + dlnvs)^2)

    `rho_P vs_P^2` is the PREM-stable shear modulus. `mu` scales the
    `mu_Pa` column of PREM-stable; the PREM-stable file rounds its columns,
    so `rho_P vs_P^2` and `mu_P` differ by up to about 2e-5 of the value.
    With no relative change, K equals `rho_P (vp_P^2 - 4/3 vs_P^2)`, which
    differs from the `kappa_Pa` column by the same rounding.

    Args:
      rho, vp, vs, mu_P: PREM-stable density (kg/m^3), P- and S-wave speed
        (m/s) and shear modulus (Pa) at the points.
      dlnvs, dlnvp: the relative change of vs and vp at the points.

    Returns:
      `(mu, K)` in Pa.
    """
    mu = mu_P * (1.0 + dlnvs) ** 2
    K = rho * (vp**2 * (1.0 + dlnvp) ** 2
               - 4.0 / 3.0 * vs**2 * (1.0 + dlnvs) ** 2)
    return mu, K


class RelativeChange:
    """The relative change of vs and vp of GLAD-M25 at each depth.

    Collective. Rank 0 reads the g-drift dataset that the model file names,
    checks its SHA-256 against the model file, and forms on each depth of
    the dataset

        dlnvs = vs / <vs>(r) - 1,    dlnvp = vp / <vp>(r) - 1,

    with `<.>(r)` the plain mean over the points of that depth. g-drift
    stores every depth on the same equal-area point set, so the plain mean
    is the mean over the sphere; the constructor checks that the depths
    share one point set and that the second moments of its directions are
    those of a uniform set. The relative change goes into a new
    `gdrift.EarthModel3D` on the coordinates of the dataset, whose layered
    interpolation (inverse distance squared over the nearest directions on
    the two bracketing depths, then linear in radius) gives it at any point.
    A point above the top depth gets the top depth values.

    Every other rank holds no data. `at_points` is collective: the ranks
    send their points to rank 0 one at a time.

    Args:
      model: the model file (`tomography_model_file`).
      model_sha256: the SHA-256 of the model file text, for the record.
      elastic: the PREM-stable `ElasticProfile`, for the sphere check.
      check_radii_m: the radii of the PREM-stable jumps on mesh spheres
        inside the mantle (`prem_jumps_on_spheres`), for the sphere check.

    Attributes:
      record: the record of the dataset and of the checks (on every rank).
    """

    def __init__(self, model, model_sha256, elastic, check_radii_m):
        tomography = model["tomography"]
        self.dataset = tomography["dataset"]
        self.fields = list(tomography["fields"])
        self.sha256 = tomography["sha256"]
        self._model = None
        record = None
        if COMM_WORLD.rank == 0:
            record = self._build(model, model_sha256, elastic, check_radii_m)
        self.record = COMM_WORLD.bcast(record, root=0)

    def _build(self, model, model_sha256, elastic, check_radii_m):
        """Read the dataset, form the relative change, check it (rank 0)."""
        # The dataset file, verified by g-drift against its manifest, and
        # its SHA-256 computed here against the model file.
        registry = gdrift.DATASET_REGISTRY.get_dataset(self.dataset)
        if registry is None:
            raise RuntimeError(f"g-drift has no dataset {self.dataset}")
        registry_sha256 = (registry.file_hash or "").split(":", 1)[-1]
        if registry_sha256 != self.sha256:
            raise RuntimeError(f"g-drift names {self.dataset} with SHA-256 "
                               f"{registry_sha256}, the model file with "
                               f"{self.sha256}")
        path = fetch_dataset(self.dataset)
        file_sha256 = common.sha256sum(path)
        if file_sha256 != self.sha256:
            raise RuntimeError(f"{path} has SHA-256 {file_sha256}, the model "
                               f"file names {self.sha256}")

        # The tomography with the two fields only (about 0.9 GB).
        name = self.dataset.removeprefix(SEISMIC_MODEL_PREFIX)
        source = gdrift.SeismicModel(name, labels=self.fields)
        coordinates = source.coordinates
        vs = source.available_fields["vs"]
        vp = source.available_fields["vp"]

        # The depths of the dataset: consecutive blocks of points of one
        # radius. Every block must have the same size and the same
        # directions, which is what the plain mean and the layered
        # interpolation of g-drift assume.
        radius = np.linalg.norm(coordinates, axis=1)
        breaks = np.flatnonzero(np.abs(np.diff(radius)) > 1.0) + 1
        sizes = np.diff(np.concatenate([[0], breaks, [radius.size]]))
        if sizes.size < 2 or np.any(sizes != sizes[0]):
            raise RuntimeError(f"{self.dataset}: the depths do not hold the "
                               "same number of points")
        n_layers, n_points = int(sizes.size), int(sizes[0])
        radius = radius.reshape(n_layers, n_points)
        spread = float(np.max(radius.max(axis=1) - radius.min(axis=1)))
        if spread > 1e-3:
            raise RuntimeError(f"{self.dataset}: the radius varies by {spread} "
                               "m within one depth")
        r_layer = radius.mean(axis=1)
        directions = coordinates[:n_points] / radius[0][:, None]
        worst = 0.0
        for k in range(1, n_layers):
            block = coordinates[k * n_points:(k + 1) * n_points] \
                / radius[k][:, None]
            worst = max(worst, float(np.max(np.abs(block - directions))))
        if worst > 1e-9:
            raise RuntimeError(f"{self.dataset}: the depths do not share one "
                               f"point set (largest difference {worst})")
        # For directions spread uniformly over the sphere the mean of each
        # component is 0 and the mean of its square is 1/3. A point set that
        # crowds some region fails this, and its plain mean is then not the
        # mean over the sphere.
        first = np.abs(directions.mean(axis=0))
        second = np.abs((directions**2).mean(axis=0) - 1.0 / 3.0)
        uniform = {"largest_mean_component": float(first.max()),
                   "largest_second_moment_error": float(second.max())}
        if first.max() > 1e-3 or second.max() > 1e-3:
            raise RuntimeError(f"{self.dataset}: the point set of a depth is "
                               f"not uniform over the sphere ({uniform})")

        # The top depth of the dataset, against the text of the model file.
        depth_km = (gdrift.R_earth - r_layer) / 1e3
        top = int(np.argmax(r_layer))
        if abs(depth_km[top] - TOP_DEPTH_KM) > 1e-6:
            raise RuntimeError(f"{self.dataset}: the top depth is "
                               f"{depth_km[top]} km, the model file says "
                               f"{TOP_DEPTH_KM} km")

        # The mean over the sphere at each depth, and the relative change.
        # The dataset stores the wave speeds in single precision; the means
        # and the relative change are formed in double precision, so the
        # mean of the relative change at each depth is zero to round-off.
        vs = vs.astype(np.float64).reshape(n_layers, n_points)
        vp = vp.astype(np.float64).reshape(n_layers, n_points)
        mean_vs = vs.mean(axis=1)
        mean_vp = vp.mean(axis=1)
        dlnvs = vs / mean_vs[:, None] - 1.0
        dlnvp = vp / mean_vp[:, None] - 1.0
        if not (np.isfinite(dlnvs).all() and np.isfinite(dlnvp).all()):
            raise RuntimeError(f"{self.dataset}: the relative change is not "
                               "finite")

        # The model of the relative change, with the interpolation settings
        # of the tomography model. `set_coordinates` copies the coordinates;
        # the tomography is released after that.
        self._model = gdrift.EarthModel3D(
            nearest_neighbours=source.nearest_neighbours,
            default_max_distance=source.default_max_distance)
        self._model.set_coordinates(coordinates)
        self._model.add_quantity("dlnvs", dlnvs.ravel())
        self._model.add_quantity("dlnvp", dlnvp.ravel())
        # The relative change is all that is kept of the tomography. The
        # wave speeds go now; dlnvs and dlnvp stay for the sphere check.
        del vs, vp, radius
        interpolation = {"nearest_neighbours": int(source.nearest_neighbours),
                         "default_max_distance_m":
                             float(source.default_max_distance),
                         "rule": "gdrift.EarthModel3D layered interpolation: "
                                 "inverse distance squared over the nearest "
                                 "directions on the two bracketing depths, "
                                 "then linear in radius; top depth values "
                                 "above the top depth"}
        del source, coordinates

        # Above the top depth the top values hold: the relative change at
        # the surface equals the one at the top depth, direction by
        # direction. The radius of the top depth is the mean radius of its
        # points, so the radial weight of g-drift there is 1 to round-off.
        surface = self.evaluate(directions[::50] * elastic.surface_radius_m)
        at_top = self.evaluate(directions[::50] * r_layer[top])
        hold_error = float(np.max(np.abs(surface - at_top)))
        if hold_error > 1e-9:
            raise RuntimeError(f"{self.dataset}: the value at the surface "
                               "differs from the top depth value by "
                               f"{hold_error}")

        sphere_check = self._sphere_check(
            elastic, check_radii_m, directions, depth_km, mean_vs, mean_vp,
            dlnvs, dlnvp)

        try:
            distribution = importlib.metadata.version("gdrift")
        except importlib.metadata.PackageNotFoundError:
            distribution = None
        order = np.argsort(depth_km)
        return {
            "model_file": {"id": model["id"], "sha256": model_sha256,
                           "base": model["base"],
                           "tomography": model["tomography"],
                           "perturbation": model["perturbation"]},
            "dataset": self.dataset, "fields": self.fields,
            "sha256_model_file": self.sha256,
            "sha256_registry": registry_sha256,
            "sha256_file": file_sha256, "path": str(path),
            "gdrift_version": gdrift.__version__,
            "gdrift_distribution_version": distribution,
            # The commit that the version file of the installed g-drift
            # names, if it names one.
            "gdrift_commit": getattr(getattr(gdrift, "_version", None),
                                     "commit_id", None),
            "gdrift_path": os.path.dirname(gdrift.__file__),
            "depths": n_layers, "points_per_depth": n_points,
            "top_depth_km": float(depth_km.min()),
            "bottom_depth_km": float(depth_km.max()),
            "uniform_check": uniform,
            "hold_above_top_error": hold_error,
            "interpolation": interpolation,
            # The mean over the sphere at each depth, from the top down.
            "layer_means": {"depth_km": depth_km[order],
                            "vs_m_s": mean_vs[order],
                            "vp_m_s": mean_vp[order]},
            "sphere_check": sphere_check,
        }

    def _sphere_check(self, elastic, check_radii_m, directions, depth_km,
                      mean_vs, mean_vp, dlnvs, dlnvp):
        """The relative change near each PREM-stable jump on a mesh sphere.

        For each radius of `check_radii_m` it lists the depths of GLAD-M25
        within `CHECK_WINDOW_KM`: the mean of vs and vp, and the min, max,
        mean and RMS of dlnvs and dlnvp. A jump of GLAD-M25 shows in the
        means and not in the relative change. It then evaluates the
        interpolated relative change 1 m above and 1 m below the sphere on
        every 10th direction of the dataset: the relative change is
        continuous there, and the jump of mu and K at the sphere is the jump
        of PREM-stable, which is listed too.
        """
        out = []
        for r_m in check_radii_m:
            depth = (elastic.surface_radius_m - r_m) / 1e3
            window = np.flatnonzero(np.abs(depth_km - depth)
                                    <= CHECK_WINDOW_KM)
            window = window[np.argsort(depth_km[window])]
            rows = []
            for k in window:
                rows.append({
                    "depth_km": float(depth_km[k]),
                    "mean_vs_m_s": float(mean_vs[k]),
                    "mean_vp_m_s": float(mean_vp[k]),
                    "dlnvs": {"min": float(dlnvs[k].min()),
                              "max": float(dlnvs[k].max()),
                              "mean": float(dlnvs[k].mean()),
                              "rms": float(np.sqrt(np.mean(dlnvs[k]**2)))},
                    "dlnvp": {"min": float(dlnvp[k].min()),
                              "max": float(dlnvp[k].max()),
                              "mean": float(dlnvp[k].mean()),
                              "rms": float(np.sqrt(np.mean(dlnvp[k]**2)))}})
            sample = directions[::10]
            above = self.evaluate(sample * (r_m + 1.0))
            below = self.evaluate(sample * (r_m - 1.0))
            prem_above = elastic(r_m, "above")
            prem_below = elastic(r_m, "below")
            out.append({
                "depth_km": float(depth),
                "glad_depths": rows,
                "directions": int(sample.shape[0]),
                "dlnvs_at_sphere": {"min": float(above[:, 0].min()),
                                    "max": float(above[:, 0].max()),
                                    "rms": float(np.sqrt(np.mean(
                                        above[:, 0]**2)))},
                "dlnvp_at_sphere": {"min": float(above[:, 1].min()),
                                    "max": float(above[:, 1].max()),
                                    "rms": float(np.sqrt(np.mean(
                                        above[:, 1]**2)))},
                "largest_dlnvs_change_across_2m": float(
                    np.max(np.abs(above[:, 0] - below[:, 0]))),
                "largest_dlnvp_change_across_2m": float(
                    np.max(np.abs(above[:, 1] - below[:, 1]))),
                "prem_stable_mu_pa": {"below": float(prem_below["mu_Pa"]),
                                      "above": float(prem_above["mu_Pa"])},
                "prem_stable_kappa_pa": {
                    "below": float(prem_below["kappa_Pa"]),
                    "above": float(prem_above["kappa_Pa"])}})
        return out

    def evaluate(self, xyz_m):
        """The relative change at points, on rank 0 only.

        Args:
          xyz_m: an `(n, 3)` array of Cartesian positions in metres.

        Returns:
          An `(n, 2)` array: dlnvs and dlnvp.
        """
        xyz_m = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
        out = np.empty((xyz_m.shape[0], 2))
        for start in range(0, xyz_m.shape[0], GDRIFT_CHUNK):
            stop = min(start + GDRIFT_CHUNK, xyz_m.shape[0])
            out[start:stop] = np.asarray(self._model.at(
                ["dlnvs", "dlnvp"], xyz_m[start:stop])).reshape(-1, 2)
        return out

    def at_points(self, xyz_m):
        """The relative change at the points of every rank. Collective.

        Each rank passes its own points. Rank 0 evaluates its own, then
        receives the points of rank 1, 2, ... in turn and sends each its
        values back. The memory of rank 0 holds one rank's points at a time.

        Args:
          xyz_m: an `(n, 3)` array of this rank's positions, metres.

        Returns:
          This rank's `(n, 2)` array of dlnvs and dlnvp.
        """
        comm = COMM_WORLD
        if comm.rank == 0:
            mine = self.evaluate(xyz_m)
            for source in range(1, comm.size):
                points = comm.recv(source=source, tag=801)
                comm.send(self.evaluate(points), dest=source, tag=802)
            return mine
        comm.send(np.asarray(xyz_m, dtype=float), dest=0, tag=801)
        return comm.recv(source=0, tag=802)


def prem_jumps_on_spheres(elastic, radial, scales):
    """The radii of the PREM-stable jumps that are mesh spheres, m.

    A jump of the elastic structure is a radius with two rows in the model
    file. The ones that are spheres of the mesh, strictly between the CMB and
    the surface, are where the moduli jump exactly at a cell face (400 and
    670 km for PREM-stable on `n2_giamip_all_o1`).
    """
    nodes = elastic.radius_m
    jumps = nodes[1:][np.diff(nodes) == 0.0]
    spheres = np.array(sorted(radial.to_dict()["spheres"].values())) * D_SCALE
    found = [float(r) for r in jumps
             if scales.Rc_m < r < scales.Re_m
             and np.min(np.abs(spheres - r)) < 1.0]
    return sorted(found, reverse=True)


# --------------------------------------------------------------------------
# The earth model on the mesh
# --------------------------------------------------------------------------


def node_radius_m(space):
    """The radius of every node of a scalar space, in metres, with halos."""
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)
    r = Function(space).interpolate(sqrt(dot(X, X)))
    return np.array(r.dat.data_ro_with_halos) * D_SCALE


def vertex_values(f):
    """The values of a DG1 field at the four vertices of every cell.

    The DG1 element of Firedrake on a tetrahedron has its nodes at interior
    points, so the vertex values come from an exact interpolation into the
    DG1 space with nodes at the vertices (`variant="equispaced"`).

    Args:
      f: a DG1 Function on the mantle.

    Returns:
      An array `(cells, 4)`, owned cells first, then halo cells.
    """
    mesh = f.function_space().mesh()
    space = FunctionSpace(mesh, "DG", 1, variant="equispaced")
    at_vertices = Function(space).interpolate(f)
    return np.array(at_vertices.dat.data_ro_with_halos)[space.cell_node_list]


def earth_fields(mantle, earth, scales, relative, radial):
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

    The g-drift step: every DG1 node takes the relative change of vs and vp
    at its position (`RelativeChange.at_points`), and the shear and bulk
    modulus follow from the rule of the model file (`glad_moduli`) with the
    PREM-stable values of the node. The density, the viscosity and gravity
    stay 1D.

    The G-ADOPT choice of 2026-10-19 (Sia) for the 0-60 km shell: mu and K
    in the 0-60 km shell are the values of the GLAD-M25 rule at the cell
    centroid, so that the linear DG1 interpolant does not extrapolate the
    steep crustal changes of GLAD-M25 to negative or near-zero values at
    the cell vertices. Each modulus is made twice: `f_dg0`, the rule at the
    centroid of every cell (the relative change sampled there with the same
    construction as at the nodes), and `f_dg1`, the rule at the DG1 nodes.
    The written DG1 field is the interpolation of
    `conditional(r_centroid > R_60, f_dg0, f_dg1)`, with `R_60` the radius
    of the 60 km mesh sphere from the sidecar (`radial.radius("d60")`). No
    cell crosses that sphere, so each cell takes one rule: constant in
    0-60 km, linear below. The function stops unless mu and K are positive
    at every centroid of the 0-60 km shell, at every vertex of every cell
    below it (`f_dg1`), and at every vertex of every cell of the written
    fields.

    Args:
      mantle: the mantle mesh (P2 coordinates).
      earth: the 1D `giamip.earth.RadialEarth` of the experiment
        (`radial_earth`).
      scales: the `exp08_common.Scales`.
      relative: the `RelativeChange` of GLAD-M25.
      radial: the `giamsh.radial.RadialStructure` of the mesh sidecar.

    Returns:
      `(fields, relative_fields, moduli_record)`: a dictionary with
      `density`, `shear_modulus`, `bulk_modulus` (DG1), `viscosity` (DG0)
      and `gravity` (CG2), all non-dimensional; a dictionary with the DG1
      fields `dlnvs` and `dlnvp` at the DG1 nodes, for the record and the
      ParaView file; and the record of the 0-60 km rule and its checks.
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

    # The g-drift step. The position of every DG1 node (halos included, so
    # every rank fills its halo nodes with the values of their owners), in
    # metres, and the relative change of GLAD-M25 there.
    X_dg1 = Function(VectorFunctionSpace(mantle, "DG", 1)).interpolate(X)
    xyz_m = np.array(X_dg1.dat.data_ro_with_halos) * D_SCALE
    change = relative.at_points(xyz_m)
    dlnvs_node, dlnvp_node = change[:, 0], change[:, 1]
    bad = int(np.count_nonzero(~np.isfinite(change)))
    if COMM_WORLD.allreduce(bad, op=MPI.SUM):
        raise RuntimeError("g-drift gives no relative change at some DG1 "
                           "nodes")

    def column(name):
        # The PREM-stable value of each DG1 node, from the side of its cell.
        return np.where(use_above, above[name], below[name])

    mu_pa, kappa_pa = glad_moduli(column("rho_kg_m3"), column("vp_m_s"),
                                  column("vs_m_s"), column("mu_Pa"),
                                  dlnvs_node, dlnvp_node)
    # The rule at the DG1 nodes: `f_dg1`.
    mu_dg1 = Function(dg1, name="shear_modulus_dg1")
    mu_dg1.dat.data_with_halos[:] = mu_pa / common.MU_BAR
    kappa_dg1 = Function(dg1, name="bulk_modulus_dg1")
    kappa_dg1.dat.data_with_halos[:] = kappa_pa / common.MU_BAR

    # The rule at the cell centroids: `f_dg0`. The centroid is the DG0 node
    # of the cell; the relative change there comes from the same g-drift
    # model as at the DG1 nodes, and the PREM-stable values from the radius
    # of the centroid. A centroid never lies on a mesh sphere, so the side
    # of a PREM-stable jump does not matter.
    X_dg0 = Function(VectorFunctionSpace(mantle, "DG", 0)).interpolate(X)
    change_c = relative.at_points(
        np.array(X_dg0.dat.data_ro_with_halos) * D_SCALE)
    bad = int(np.count_nonzero(~np.isfinite(change_c)))
    if COMM_WORLD.allreduce(bad, op=MPI.SUM):
        raise RuntimeError("g-drift gives no relative change at some cell "
                           "centroids")
    r_c = np.clip(np.array(centroid.dat.data_ro_with_halos) * D_SCALE,
                  scales.Rc_m, scales.Re_m)
    prem_c = earth.elastic(r_c)
    mu_c, kappa_c = glad_moduli(prem_c["rho_kg_m3"], prem_c["vp_m_s"],
                                prem_c["vs_m_s"], prem_c["mu_Pa"],
                                change_c[:, 0], change_c[:, 1])
    mu_dg0 = Function(dg0, name="shear_modulus_dg0")
    mu_dg0.dat.data_with_halos[:] = mu_c / common.MU_BAR
    kappa_dg0 = Function(dg0, name="bulk_modulus_dg0")
    kappa_dg0.dat.data_with_halos[:] = kappa_c / common.MU_BAR

    # The cells of each rule, from the radius of the centroid against the
    # 60 km sphere of the mesh (non-dimensional, as the mesh).
    R_60 = radial.radius(LID_SPHERE)
    n_owned = mantle.cell_set.size
    cell_dof = dg0.cell_node_list[:, 0]
    in_lid = np.array(centroid.dat.data_ro_with_halos)[cell_dof] > R_60
    lid_owned = in_lid[:n_owned]

    def positive_or_stop(values, what):
        # Stop unless every value is positive, over all ranks.
        local = values.min() if values.size else np.inf
        lowest = COMM_WORLD.allreduce(float(local), op=MPI.MIN)
        if not lowest > 0.0:
            raise RuntimeError(f"{what} is not positive (min "
                               f"{lowest * common.MU_BAR} Pa)")
        return lowest

    # Check 1: mu and K at every centroid of the 0-60 km shell.
    cell_mu = np.array(mu_dg0.dat.data_ro_with_halos)[cell_dof][:n_owned]
    cell_kappa = np.array(kappa_dg0.dat.data_ro_with_halos)[cell_dof][:n_owned]
    positive_or_stop(cell_mu[lid_owned], "mu at a centroid in 0-60 km")
    positive_or_stop(cell_kappa[lid_owned], "K at a centroid in 0-60 km")
    # Check 2: mu and K of the DG1 rule at every vertex below 60 km.
    positive_or_stop(vertex_values(mu_dg1)[:n_owned][~lid_owned],
                     "mu at a vertex below 60 km")
    positive_or_stop(vertex_values(kappa_dg1)[:n_owned][~lid_owned],
                     "K at a vertex below 60 km")

    # The written fields: one DG1 interpolation of the choice per cell.
    mu = Function(dg1, name="shear_modulus").interpolate(
        conditional(centroid > R_60, mu_dg0, mu_dg1))
    kappa = Function(dg1, name="bulk_modulus").interpolate(
        conditional(centroid > R_60, kappa_dg0, kappa_dg1))
    # Check 3: mu and K of the written fields at every vertex of every cell.
    # A linear field on a cell takes its smallest value at a vertex, so they
    # are then positive everywhere in the mantle.
    mu_v = vertex_values(mu)[:n_owned]
    kappa_v = vertex_values(kappa)[:n_owned]
    positive_or_stop(mu_v, "mu of the written field at a vertex")
    positive_or_stop(kappa_v, "K of the written field at a vertex")

    # The record: the rule, the sphere, the cells of each rule, and the
    # vertex values of the written fields per mesh layer.
    r_owned = np.array(centroid.dat.data_ro_with_halos)[cell_dof][:n_owned]
    per_layer = {}
    for name, r_bottom, r_top in mantle_layers(radial):
        in_layer = (r_owned > r_bottom) & (r_owned < r_top)
        row = {}
        for label, values in (("mu_pa", mu_v), ("K_pa", kappa_v)):
            local = values[in_layer]
            row[label] = {
                "min": COMM_WORLD.allreduce(
                    float(local.min()) if local.size else np.inf,
                    op=MPI.MIN) * common.MU_BAR,
                "max": COMM_WORLD.allreduce(
                    float(local.max()) if local.size else -np.inf,
                    op=MPI.MAX) * common.MU_BAR}
        per_layer[name] = row
    moduli_record = {
        "rule": "G-ADOPT choice of 2026-10-19 (Sia): mu and K in the 0-60 km "
                "shell are the values of the GLAD-M25 rule at the cell "
                "centroid, so that the linear DG1 interpolant does not "
                "extrapolate the steep crustal changes of GLAD-M25 to "
                "negative or near-zero values at the cell vertices; below "
                "60 km they are the values of the rule at the DG1 nodes",
        "sphere": LID_SPHERE,
        "R_nondim": float(R_60), "R_m": float(R_60 * D_SCALE),
        "depth_km": float((scales.Re - R_60) * D_SCALE / 1e3),
        "cells_centroid_rule": COMM_WORLD.allreduce(
            int(np.count_nonzero(lid_owned)), op=MPI.SUM),
        "cells_dg1_rule": COMM_WORLD.allreduce(
            int(np.count_nonzero(~lid_owned)), op=MPI.SUM),
        "lid_centroid_values": {
            "mu_pa_min": COMM_WORLD.allreduce(
                float(cell_mu[lid_owned].min()) if lid_owned.any()
                else np.inf, op=MPI.MIN) * common.MU_BAR,
            "K_pa_min": COMM_WORLD.allreduce(
                float(cell_kappa[lid_owned].min()) if lid_owned.any()
                else np.inf, op=MPI.MIN) * common.MU_BAR},
        "vertex_values_written": per_layer}

    dlnvs = Function(dg1, name="dlnvs")
    dlnvs.dat.data_with_halos[:] = dlnvs_node
    dlnvp = Function(dg1, name="dlnvp")
    dlnvp.dat.data_with_halos[:] = dlnvp_node

    # Viscosity at the centroid. A layer boundary of the viscosity model that
    # is a mesh sphere is exact; one that is not moves to the nearest cell
    # faces (see the module docstring).
    r_centroid = np.clip(np.array(centroid.dat.data_ro_with_halos) * D_SCALE,
                         scales.Rc_m, scales.Re_m)
    eta_pa_s = earth.viscosity(r_centroid)
    eta_pa_s = np.where(np.isinf(eta_pa_s), common.ETA_LID, eta_pa_s)
    if np.isnan(eta_pa_s).any():
        raise RuntimeError("a mantle cell has no viscosity")
    eta = Function(dg0, name="viscosity")
    eta.dat.data_with_halos[:] = eta_pa_s / common.ETA_BAR

    # Gravity is continuous; the side does not matter.
    cg2 = FunctionSpace(mantle, "CG", 2)
    r_cg2 = np.clip(node_radius_m(cg2), scales.Rc_m, scales.Re_m)
    g = Function(cg2, name="gravity")
    g.dat.data_with_halos[:] = earth.elastic(r_cg2)["g_m_s2"] / scales.g_bar
    return ({"density": rho, "shear_modulus": mu, "bulk_modulus": kappa,
             "viscosity": eta, "gravity": g},
            {"dlnvs": dlnvs, "dlnvp": dlnvp}, moduli_record)


def core_density(elastic, scales, rule):
    """The density of the homogeneous fluid core, kg/m^3.

    Args:
      elastic: the `giamip.earth.ElasticProfile`.
      scales: the `exp08_common.Scales`.
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
      earth: its `giamip.earth.RadialEarth`.
      scales: the `exp08_common.Scales`.
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
    of `step_years`, and the last year. With the Exp08 years (500 yr steps
    to 26 ka, then 100 yr steps) and `step_years = 2000` this keeps -80000,
    -78000, ..., 0 and 1000. The ice history is then read at its own time
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


def topography_of(experiment):
    """The topography file of the experiment, with the G-ADOPT choice.

    The case file of Exp08 names no topography file (the protocol leaves it
    open), so `GiamipExperiment.topography` is None. The G-ADOPT choice of
    the `gadopt` block (`bedmach3`) maps to the file of the GIAMIP cases
    that prescribe that topography (`giamip.cases.with_gadopt_choices`, which
    uses `giamip.cases.topography_files`):
    `bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc`.

    Args:
      experiment: the `GiamipExperiment`.

    Returns:
      A `giamip.mip.inputs.Topography`.
    """
    if experiment.topography is not None:
        return experiment.topography
    relative_path = cases.with_gadopt_choices(
        experiment.id)["topography"]["file"]
    return Topography(relative_path)


class SurfaceForcing:
    """The observed bed and the ice on the mantle mesh, from the GIAMIP files.

    `SL_obs = -bed / D` is CG2, because the mask width reads its slope. The
    ice is CG1 (see `exp08_common.ICE_SPACE`). Every node of the mantle mesh
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
        self.topography = topography_of(experiment)
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
    say = common.RunLog(os.path.join(args.output_path, "prepare_exp08.log"))
    say("=" * 78)
    say(f"GIAMIP {EXPERIMENT}: input file of the bed inversion, "
        f"{COMM_WORLD.size} ranks")
    say("=" * 78)
    timings = {}

    # The experiment and its earth model, with the G-ADOPT choices for the
    # fields that the protocol leaves open. The 1D part comes from the giamip
    # model files, because `experiment.earth()` refuses the laterally
    # varying elastic structure of Exp08.
    experiment = GiamipExperiment(EXPERIMENT)
    model, model_sha256 = tomography_model_file(experiment)
    earth = radial_earth(experiment, model)
    # The solver of this directory has one Maxwell element per cell. Stop
    # for any other rheology, so that a copy of this script for an
    # experiment with Burgers or power-law rheology cannot run as Maxwell.
    # `radial_earth` takes the rheology of the case with the G-ADOPT choices
    # (`giamip.cases.with_gadopt_choices`).
    if earth.rheology != "maxwell":
        raise RuntimeError(f"{EXPERIMENT} has rheology {earth.rheology!r}; "
                           "this script builds a Maxwell solver only")
    say(f"  earth: {model['id']} on {earth.elastic.model_id}, viscosity "
        f"{earth.viscosity.model_id}, {earth.rheology}, "
        f"{earth.compressibility}")
    scales = common.Scales.from_elastic(earth.elastic)
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

    # The relative change of GLAD-M25, read on rank 0, with the per-depth
    # check at the PREM-stable jumps on mesh spheres.
    tic = time.time()
    check_radii = prem_jumps_on_spheres(earth.elastic, radial, scales)
    relative = RelativeChange(model, model_sha256, earth.elastic, check_radii)
    timings["gdrift_s"] = time.time() - tic
    tomography = relative.record
    say(f"  g-drift {tomography['gdrift_version']} "
        f"({tomography['gdrift_distribution_version']}): "
        f"{tomography['dataset']} sha256 {tomography['sha256_file']}, "
        f"{tomography['depths']} depths of {tomography['points_per_depth']} "
        f"points, {tomography['top_depth_km']:g} to "
        f"{tomography['bottom_depth_km']:g} km ({timings['gdrift_s']:.1f} s)")
    for check in tomography["sphere_check"]:
        say(f"    sphere {check['depth_km']:.0f} km: dlnvs rms "
            f"{check['dlnvs_at_sphere']['rms']:.4f}, dlnvp rms "
            f"{check['dlnvp_at_sphere']['rms']:.4f}, largest change across "
            f"2 m {check['largest_dlnvs_change_across_2m']:.2e} (dlnvs), "
            f"{check['largest_dlnvp_change_across_2m']:.2e} (dlnvp)")
        for row in check["glad_depths"]:
            say(f"      GLAD-M25 {row['depth_km']:7.2f} km: <vs> "
                f"{row['mean_vs_m_s']:.1f}, <vp> {row['mean_vp_m_s']:.1f} m/s,"
                f" dlnvs [{row['dlnvs']['min']:+.4f}, "
                f"{row['dlnvs']['max']:+.4f}] rms {row['dlnvs']['rms']:.4f},"
                f" dlnvp [{row['dlnvp']['min']:+.4f}, "
                f"{row['dlnvp']['max']:+.4f}] rms {row['dlnvp']['rms']:.4f}")

    # The earth fields.
    tic = time.time()
    fields, relative_fields, moduli_record = earth_fields(
        mantle, earth, scales, relative, radial)
    timings["earth_s"] = time.time() - tic
    say(f"  moduli: centroid values above the {moduli_record['sphere']} "
        f"sphere (R = {moduli_record['R_m']:.1f} m) in "
        f"{moduli_record['cells_centroid_rule']} cells, DG1 values in "
        f"{moduli_record['cells_dg1_rule']} cells")
    for name, row in moduli_record["vertex_values_written"].items():
        say(f"    vertex values {name}: mu [{row['mu_pa']['min']:.4g}, "
            f"{row['mu_pa']['max']:.4g}] Pa, K [{row['K_pa']['min']:.4g}, "
            f"{row['K_pa']['max']:.4g}] Pa")

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
          "gravity": ("m/s^2", scales.g_bar, "gravity_m_s2"),
          # The relative change of GLAD-M25 at the DG1 nodes.
          "dlnvs": ("1", 1.0, "dlnvs"),
          "dlnvp": ("1", 1.0, "dlnvp")}
    # The earth fields and the relative change. The relative change is not
    # in the input file; it is only recorded and shown.
    earth_view = {**fields, **relative_fields}
    layer_stats = {name: {"unit": unit,
                          "layers": cells.statistics(earth_view[name], scale)}
                   for name, (unit, scale, _) in si.items()}
    for name in ("dlnvs", "dlnvp", "shear_modulus", "bulk_modulus"):
        for layer, _, _ in layers:
            row = layer_stats[name]["layers"][layer]
            say(f"    {name} {layer}: min {row['min']:.4g}, max "
                f"{row['max']:.4g}, mean {row['mean']:.4g} "
                f"{layer_stats[name]['unit']}")
    viscosity_by_layer = cells.distinct_values(fields["viscosity"],
                                               common.ETA_BAR)
    for name, _, _ in layers:
        rows = viscosity_by_layer[name]
        text = rows if isinstance(rows, str) else ", ".join(
            f"{row['value']:.3g} Pa s ({100 * row['volume_fraction']:.1f} %)"
            for row in rows)
        say(f"    viscosity {name}: {text}")
    timings["statistics_s"] = time.time() - tic

    record = {
        "experiment": EXPERIMENT,
        "case": experiment.definition,
        # The rheology and compressibility after the G-ADOPT choices and the
        # references between cases ("same as Exp02") are resolved.
        "rheology": earth.rheology,
        "compressibility": earth.compressibility,
        "giamip_version": giamip.__version__,
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
        # The GLAD-M25 dataset, the rule and its checks.
        "tomography": tomography,
        # The rule of the moduli in the 0-60 km shell, its checks, and the
        # vertex values of the written mu and K per layer.
        "moduli_lid_rule": moduli_record,
        "viscosity_by_layer": viscosity_by_layer,
        "ice_volume_check": ice_check,
        # The numbers that the inversion computes again after it loads the
        # file (`exp08_common.compare_statistics`).
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
                     .interpolate(earth_view[name] * scale)
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
