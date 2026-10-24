r"""GIAMIP Exp12: make the input file of the bed inversion.

WHAT THIS SCRIPT DOES
    It makes every field of the inversion that comes from data, once, and
    writes them to one Firedrake `CheckpointFile` (`inputs_exp12.h5`). The
    inversion (`invert_exp12.py`) and the forward run (`forward_exp12.py`)
    load that file and read no earth-model file, so the tape of the
    inversion holds only the operations from the control to the misfit.

    1. Read the coarse giamsh mesh, partition it so that every rank holds
       the same number of mantle cells, refine it `--refine` times, curve it
       to P2 and cut the mantle submesh (`exp12_common.build_meshes`). A
       mesh that is refined already and saved as a P1 `CheckpointFile`
       (`.h5`, for example the 8 Re mesh `giamip_8re_r4_linear`) is loaded
       with the same mantle-balanced partition and curved
       (`exp12_common.load_checkpoint_meshes`).
    2. Make the earth fields on the mantle: the viscosity eta_0 (DG0) from
       the 3-D model of Lloyd et al. (2024), read from g-drift (as Exp11);
       density and gravity of PREM-stable, and the shear and bulk modulus
       (DG1) of PREM-stable scaled by the GLAD-M25 relative change of vs and
       vp (`prem-glad-m25`, as Exp08). All fields are non-dimensional.
    3. Make the rheology of Exp12: the N = 3 Maxwell elements of the
       Extended Burgers (EB) fit below 90 km, with the elements that are much
       faster than the shortest time step replaced (THE DROP RULE, below),
       and one Maxwell element in N parts above 90 km; and the power-law
       exponent and transition stress (DG0).
    4. Make the forcing on the mantle from the GIAMIP files: the observed sea
       level `SL_obs = -bed_obs / D` (CG2), the frozen surface slope of the
       ocean mask (`surface_slope(SL_obs)`, DG1), and the ice of every step
       of the march from the start to year 0 (CG1, one entry per step).
    5. Write the input file with its JSON record, the same record as a JSON
       file next to it, and ParaView files of the earth fields and of the ice
       at three steps for the reviewer.

    With `--earth_only` the script stops after step 3 and writes only the
    record of the earth and rheology fields (`inputs_exp12_earth_only.json`).
    This is the serial check of the drop rule on the real fields.

    The script tapes nothing: `firedrake.adjoint` is not imported.

THE EXPERIMENT
    GIAMIP protocol v4.0, experiment Exp12, as `giamip` defines it
    (`giamip.mip.inputs.GiamipExperiment("Exp12")`, case file
    `giamip/resources/cases/giamip/exp12.json`). The protocol leaves the
    earth model and the topography open; the values below are the G-ADOPT
    choices of the `gadopt` block of the case file
    (`giamip.cases.with_gadopt_choices`):

        ice          ICE-6G_D (`iceHistory-ICE-6G_D.nc`)
        topography   Bedmap3 + BedMachine Greenland 5 + GEBCO
                     (`bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc`), the
                     G-ADOPT choice `bedmap3`. The file is the observed
                     present-day bed (year 0); the inversion finds the bed at
                     the start of the run.
        earth        `prem-glad-m25` (PREM-stable density, GLAD-M25 moduli,
                     as Exp08) with the 3-D viscosity of Lloyd et al. (2024)
                     (`lloyd2024`, as Exp11), rheology `burgers-power-law`,
                     compressible
        period       122 ka to year 0, in steps of `--step_years` (2000 yr:
                     61 steps), or with `--schedule windows` the steps of
                     schedule C (protocol steps plus the fast melt windows
                     of `exp12_common.SCHEDULE_C_WINDOWS`, design handout
                     8.3; S33, S34), or with `--schedule windows-coarse`
                     the steps of schedule D (1 kyr steps plus the windows
                     of schedule C at twice their step,
                     `exp12_common.schedule_d_years`; R12 of the r2
                     production), with the ice of every year of the
                     march: the protocol slice at a protocol year, the
                     linear combination of the two neighbouring slices
                     between them (`exp12_common.linear_ice_into`)
        constants    ice 917, ocean water 1025 kg/m^3

    Exp12 combines the EB set of Exp09 with the power law of Exp10 (design
    handout of 2026-10-20, `NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-
    2026-10-20.md`, S6, S21-S23, S31, S32). It has no Maxwell control run
    (S23).

THE VISCOSITY (as Exp11)
    The viscosity eta_0 is DG0, the value at the cell centroid, from the
    g-drift dataset that the giamip model file `lloyd2024.json` names. The
    stored log10 viscosity is interpolated, the lower bound of the model
    file is applied to log10 by the depth of the centroid (1e20 Pa s from
    the surface to 670 km, 1e21 Pa s below), there is no upper bound and no
    elastic lid. The record gives per mesh layer the min, max and volume
    mean of log10 viscosity before and after the bound
    (`viscosity_3d.by_layer`).

THE ELASTIC STRUCTURE (as Exp08)
    mu = mu_P (1 + dlnvs)^2 and K = rho_P (vp_P^2 (1 + dlnvp)^2 - 4/3 vs_P^2
    (1 + dlnvs)^2), with P the PREM-stable value at the DG1 node and dlnvs,
    dlnvp the GLAD-M25 relative change of vs and vp at each depth of the
    dataset. In the 0-60 km shell mu and K are the values of that rule at
    the cell centroid (G-ADOPT choice of 2026-10-19), below 60 km the values
    at the DG1 nodes. The density stays PREM-stable. The rule texts of
    `prem-glad-m25.json` must equal `IMPLEMENTED_RULES`.

THE EXTENDED BURGERS ELEMENTS
    The normalised fit `free_time_fits/3` of `../eb_fit/eb_fit.json` (S31:
    N = 3, free element times) gives N pairs (mu_bar_j, tau_bar_j), moduli
    in units of mu_inf and element times in units of the local Maxwell time
    tau_M = eta_0 / mu_inf, with sum_j mu_bar_j tau_bar_j = 1 (design
    handout section 5). Each element is a Maxwell element in parallel with
    the others: mu_j = mu_inf mu_bar_j (DG1, at every node), eta_j = eta_0
    mu_bar_j tau_bar_j (DG0). Its Maxwell time eta_j / mu_j is tau_bar_j
    times the local tau_M at every node.

    - Below 90 km (centroid radius below the `d90` mesh sphere; no cell
      crosses it): the EB elements with the drop rule (S22: one EB set in
      the whole mantle below 90 km).
    - Above 90 km (S10, no transient part in the lid): one Maxwell element
      with the local tau_M in N parts, mu_j = mu_inf w_j, eta_j = eta_0 w_j,
      w_j = mu_bar_j / sum mu_bar. Exp12 has no elastic lid (the Lloyd
      viscosity holds up to the surface), so these cells keep the Lloyd
      viscosity and are Maxwell cells, not elastic cells.

THE DROP RULE (S32)
    The gate test of the shared code (`NOTES/team/builder-shared-transient.md`,
    piece 1) found that Newton stalls with the power law when the fastest
    element is 5 times faster than the step (dt / tau of its fastest element
    = 5), and converges at 2.5 and 0.5. Elements are counted from 0 here and
    in the record. In Exp12 the fastest element of the free fit, element 0
    (tau_bar_0 = 0.0149), has a Maxwell time of about 0.7 yr where the
    Lloyd viscosity is at its lower bound of 1e20 Pa s, so dt / tau_0 is
    about 15 at the shortest step of 10 yr.

    The rule, cell by cell below 90 km:

    - The shortest Maxwell time of element j in the cell is
      tau_j = tau_bar_j eta_0 / max(mu_inf), with the largest mu_inf at the
      four vertices of the cell (the extreme of the linear DG1 field), so
      that no point of a kept element is faster than the limit.
    - Element j is DROPPED where dt_min / tau_j > `EB_DROP_MAX_RATIO` (2),
      with dt_min = `EB_DROP_DT_YR` (10 yr, the shortest step of every
      schedule, S18). That is tau_j < 5 yr. The limit is 2.5 times below the
      ratio at which Newton stalled in the gate and below the largest ratio
      at which it converged. The same fields serve the 2 kyr inversion and
      the 10 yr forward run, because the rule is set against the shortest
      step of any run of the experiment.
    - The slowest element (the largest tau_bar_j) is never dropped. A cell
      in which it is also faster than the limit keeps all its elements and
      is counted (`slowest_below_limit_cells`; none is expected, because it
      needs tau_M < 1.7 yr).

    A dropped element is fully relaxed: it carries no elastic stress over a
    step of 10 yr, which is the behaviour of an element with tau_j << dt
    (Lau et al. 2026, Text S1; design handout section 5). It still carries
    its viscous stress eta_j times the strain rate. That stress is kept by
    adding eta_j to the slowest element, so the sum constraint of the fit
    holds in every cell: sum_j eta_j = eta_0, and the long-term viscosity of
    Exp12 is the Lloyd viscosity everywhere, as in Exp11. Without this the
    long-term viscosity would drop by mu_bar_0 tau_bar_0 = 0.29 % where
    element 0 is dropped; the choice keeps the steady-state flow and the
    comparison with the Maxwell experiments on the same eta_0, and moves
    tau of the slowest element by the same 0.3 %.

    The number of elements is one field of shape (N, 3, 3) for the whole
    mesh, so a dropped element keeps its slot. The slot carries an equal
    share of the slowest element: with D the dropped elements of the cell
    and s the slowest,

        tau_bar_s'  = (mu_bar_s tau_bar_s + sum_{j in D} mu_bar_j tau_bar_j)
                      / mu_bar_s
        mu_bar_k'   = mu_bar_s / (|D| + 1),  tau_bar_k' = tau_bar_s'
                      for k = s and every k in D.

    Two Maxwell elements with the same Maxwell time and the same initial
    state evolve as one element with the sum of their moduli, also with the
    power law (the factor is one per point and scales both times alike).
    The cell then responds as the kept elements of the fit with the dropped
    ones removed, and no element has a tiny modulus (the internal-variable
    rows stay scaled as for the other elements) or a new short time. The
    instant modulus of a cell with a dropped element j is
    mu_inf (sum mu_bar - mu_bar_j), mu_EB at about the shortest step and
    not the seismic mu_inf; this is consistent with the model (design
    handout section 5).

    The record (`burgers.drop_rule`) gives the rule, the volume fraction of
    the mantle and of each shell where each element is dropped, the number
    of cells, and the instant modulus sum_j mu_j with and without the drop
    (volume means per shell and of the mantle).

THE POWER LAW (as Exp10, S21)
    exponent = 3.5 (P13) where the centroid depth is strictly between 90
    and 670 km and eta_0 < 1e22 Pa s; exactly 1.0 elsewhere. The transition
    stress is 1 MPa (P14) in every cell, also where n = 1, because
    `power_law_factor` divides by it. The background stress is 0 (P15). The
    stress measure is sqrt(J2) (P19). The factor scales the Maxwell time of
    every element in the power-law cells (design handout 3.3), so it shifts
    the whole EB spectrum there. In Exp12 the rule selects cells, not whole
    shells, because eta_0 is 3-D; the record gives the volume fraction of
    the power-law cells per shell.

THE FORCING ON THE MESH
    The GIAMIP grid is a Gauss-Legendre grid (257 x 513). Every node of the
    mantle mesh takes the bilinear interpolant of the grid at its direction
    (`BilinearTransfer`, the rule of the mesh script). The record holds, for
    every step, the ice volume on the mesh against the Gauss-Legendre volume
    of the grid.

RUN
    srun python3 prepare_exp12.py --mesh <file.msh> --output_path <dir>
    The number of ranks must be the one of the inversion and forward jobs
    (6 nodes, 768 ranks, S24).

OUTPUT (in --output_path)
    inputs_exp12.h5          the input file of the inversion
    inputs_exp12.json        its record (the same JSON as inside the file)
    inputs_exp12_earth.pvd   ParaView: the earth fields in SI units, with
                             dlnvs, dlnvp, the instant modulus ratio, the
                             number of dropped elements and the exponent
    inputs_exp12_ice.pvd     ParaView: the ice at the first, middle and last
                             step of the march, m
    prepare_exp12.log        the log
    inputs_exp12_earth_only.json  with --earth_only, instead of the above
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
import exp12_common as common  # noqa: E402
from gadopt import (COMM_WORLD, Function, FunctionSpace,  # noqa: E402
                    SpatialCoordinate, VectorFunctionSpace, VTKFile, assemble,
                    conditional, dot, ds, dx, lt, sqrt)
from gadopt.sea_level_masks import surface_slope  # noqa: E402
from mpi4py import MPI  # noqa: E402

import giamip  # noqa: E402
from giamip import cases, grids  # noqa: E402
from giamip.earth import ElasticProfile  # noqa: E402
from giamip.mip.inputs import GiamipExperiment, Topography  # noqa: E402
from giamsh.radial import RadialStructure  # noqa: E402

# g-drift reads the 3-D viscosity and the tomography. Only this script
# imports it: the inversion and the shared module must not depend on a data
# reader.
import gdrift  # noqa: E402
from gdrift.datasetnames import DATASET_REGISTRY  # noqa: E402
from gdrift.io import fetch_dataset  # noqa: E402
from gdrift.seismic import SEISMIC_MODEL_PREFIX  # noqa: E402

#: The experiment of this directory.
EXPERIMENT = "Exp12"
#: The file stem of the outputs.
STEM = "inputs_exp12"
#: The MD5 of the shared coarse mesh `n2_giamip_all_o1.msh` (handout section
#: 3). The run stops if the mesh file differs.
MESH_MD5 = "a4da84206dc82f5b458be66e4fd33844"

D_SCALE = common.D_SCALE

#: The rheology of the `gadopt` block of the case file that this script
#: builds. Any other value stops the run.
RHEOLOGY = "burgers-power-law"

#: The fit file of the Extended Burgers elements (`../eb_fit/eb_fit.py`).
EB_FIT_FILE = os.path.join(HERE, os.pardir, "eb_fit", "eb_fit.json")
#: The fit of decision S31: N = 3 with free element times
#: (`eb_fit.py --select_free`). The `selected` entry of the fit file is the
#: fixed-grid fit, so the key is named explicitly: `<group>/<N>`.
EB_FIT_KEY = "free_time_fits/3"
#: The EB parameters that the fit must have been made with (S22, P2, design
#: handout 6.2). The script stops if the fit file says otherwise.
EB_PARAMETERS = {"alpha": 0.274, "Delta": 1.9, "tau_H_over_tau_M": 1.0,
                 "tau_L_over_tau_M": 0.0}
#: The smallest normalised modulus of an element (design handout 6.3): a
#: zero modulus gives the Maxwell time 0/0 in the solver.
EB_MU_FLOOR = 1.0e-6
#: The tolerance of the constraint sum_j mu_bar_j tau_bar_j = 1 of the fit.
EB_SUM_TOLERANCE = 1.0e-10
#: The mesh sphere above which the cells have no transient part (S10): the
#: 90 km sphere of the sidecar. No cell crosses it.
EB_SPHERE = "d90"
#: The drop rule (S32): the step, yr, against which an element is tested.
#: 10 yr is the shortest step of every schedule of Exp12 (S18).
EB_DROP_DT_YR = 10.0
#: The drop rule (S32): element j is dropped in a cell where
#: EB_DROP_DT_YR / tau_j > EB_DROP_MAX_RATIO, with tau_j its shortest
#: Maxwell time in the cell. The gate of the shared code stalled at 5 and
#: converged at 2.5 (see THE DROP RULE in the module docstring).
EB_DROP_MAX_RATIO = 2.0
#: The step lengths (yr) against which the record gives the volume fraction
#: of each element whose Maxwell time is shorter: the shortest step of the
#: fine schedule (10 yr), its step before 26 ka (50 yr) and the inversion
#: step (2000 yr).
EB_STEP_CHECK_YR = (10.0, 50.0, 2000.0)

#: The power-law exponent n of the power-law cells (P13). The other cells
#: get exactly 1.0.
POWER_LAW_EXPONENT = 3.5
#: The transition stress sigma_T, Pa (P14), in every cell.
TRANSITION_STRESS_PA = 1.0e6
#: The background stress sigma_b, Pa (P15). Zero: the low-stress limit of
#: the power law is the linear EB rheology.
BACKGROUND_STRESS_PA = 0.0
#: The power-law region (S21): strictly between these depths, m, ...
POWER_LAW_DEPTH_RANGE_M = (90.0e3, 670.0e3)
#: ... and where the linear viscosity eta_0 is strictly below this, Pa s.
POWER_LAW_ETA_CAP_PA_S = 1.0e22


# --------------------------------------------------------------------------
# The 3-D viscosity of g-drift (as Exp11)
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

    The model file (for Exp12 `lloyd2024.json`) names the g-drift dataset,
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


# --------------------------------------------------------------------------
# The GLAD-M25 elastic structure (as Exp08)
# --------------------------------------------------------------------------

#: The rule texts of the giamip model file that this script implements, word
#: for word (`perturbation` block of `prem-glad-m25.json`, gia-mip 0cf5f39).
#: The script stops if the model file says anything else: the formulas below
#: are code, and a new rule in the model file needs new code.
#:   relative_change   -> `RelativeChange` (layer means, relative change)
#:   shear_modulus     -> `glad_moduli`
#:   bulk_modulus      -> `glad_moduli`
#:   density           -> `elastic_fields` keeps the PREM-stable density
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


def tomography_model_file(experiment):
    """The giamip model file of the elastic structure of the experiment.

    The G-ADOPT choice of the case (`elastic_structure`, `prem-glad-m25` for
    Exp12) names the model file. It is read through `importlib.resources`
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
# The earth model of Exp12
# --------------------------------------------------------------------------


class Exp12Earth:
    """The earth model of Exp12.

    `giamip.earth.for_case` cannot build it, because both its elastic
    structure and its viscosity vary laterally. The parts:

    Attributes:
      elastic: the radial base of the elastic structure (PREM-stable, the
        `base` of `prem-glad-m25.json`), a `giamip.earth.ElasticProfile`.
      viscosity: the 3-D viscosity of the G-ADOPT choice (`lloyd2024`), a
        `Viscosity3D`.
      model: the model file of the elastic structure (`prem-glad-m25`).
      model_sha256: the SHA-256 of the text of that model file.
      choices: the G-ADOPT choices of the case file.
    """

    def __init__(self, elastic, viscosity, model, model_sha256, choices):
        self.elastic = elastic
        self.viscosity = viscosity
        self.model = model
        self.model_sha256 = model_sha256
        self.choices = choices


def exp12_earth(experiment):
    """The earth model of the experiment, from the `gadopt` block of its case.

    Raises:
      NotImplementedError: for a rheology other than `RHEOLOGY` or an
        incompressible earth, which this script does not build.
    """
    case = cases.with_gadopt_choices(experiment.id)
    earth = case["earth"]
    if earth["rheology"] != RHEOLOGY:
        raise NotImplementedError(f"rheology {earth['rheology']!r}; this "
                                  f"script builds {RHEOLOGY!r}")
    if earth["compressibility"] != "compressible":
        raise NotImplementedError(f"compressibility "
                                  f"{earth['compressibility']!r}")
    model, model_sha256 = tomography_model_file(experiment)
    elastic = ElasticProfile.from_model(model["base"])
    viscosity = Viscosity3D(earth["viscosity"], elastic.surface_radius_m,
                            elastic.core_mantle_boundary_m)
    return Exp12Earth(elastic=elastic, viscosity=viscosity, model=model,
                      model_sha256=model_sha256,
                      choices=cases.gadopt_choices(experiment.id))


def experiment_topography(experiment):
    """The topography file of the experiment.

    The case file of Exp12 leaves the topography open, so
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


def vertex_values(f):
    """The values of a DG1 field at the four vertices of every cell.

    The DG1 element of Firedrake on a tetrahedron has its nodes at interior
    points, so the vertex values come from an exact interpolation into the
    DG1 space with nodes at the vertices (`variant="equispaced"`).

    Args:
      f: a DG1 Function on the mantle.

    Returns:
      An array `(cells, 4)`, owned cells first, then halo cells, in the cell
      order of `cell_node_list`.
    """
    mesh = f.function_space().mesh()
    space = FunctionSpace(mesh, "DG", 1, variant="equispaced")
    at_vertices = Function(space).interpolate(f)
    return np.array(at_vertices.dat.data_ro_with_halos)[space.cell_node_list]


def viscosity_fields(mantle, earth, scales):
    """The viscosity eta_0 of every cell from the 3-D model (as Exp11).

    The viscosity of each cell is the log10 viscosity of the 3-D model at
    the cell centroid, raised to the lower bound of the model file at the
    depth of the centroid, then taken to the power 10. Every cell takes its
    value from the model; no lid is added.

    Args:
      mantle: the mantle mesh (P2 coordinates).
      earth: the `Exp12Earth`; its viscosity must be loaded
        (`Viscosity3D.load`). Collective over the ranks of each node.
      scales: the `exp12_common.Scales`.

    Returns:
      `(eta, record_fields)`: the DG0 viscosity in units of eta_bar, and DG0
      fields for the record: `log10_viscosity_raw` (the model at the
      centroid), `log10_viscosity` (after the lower bound) and
      `at_lower_bound` (1 in a cell where the bound applies, else 0).
    """
    dg0 = FunctionSpace(mantle, "DG", 0)
    X = SpatialCoordinate(mantle)
    # The DG0 interpolation of the coordinates gives the centroid of each
    # cell (the image of the reference centroid under the P2 map), in units
    # of D; g-drift takes metres. The halo cells are evaluated too, so every
    # rank holds the value of every cell it sees.
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

    return eta, {
        "log10_viscosity_raw": dg0_field(log10_raw, "log10_viscosity_raw"),
        "log10_viscosity": dg0_field(log10_eta, "log10_viscosity"),
        "at_lower_bound": dg0_field((log10_raw < floor).astype(float),
                                    "at_lower_bound")}


def elastic_fields(mantle, earth, scales, relative, radial):
    """The density, the GLAD-M25 moduli and gravity on the mantle (as Exp08).

    Each DG1 node takes the value of PREM-stable at its own radius. The DG1
    element of Firedrake on a tetrahedron has its four nodes at interior
    Gauss-Legendre points, so no node lies on a mesh sphere. The side rule
    below is kept for an element with nodes on the cell boundary: a node on
    a sphere where a property jumps takes the value of the side where its
    cell lies.

    The g-drift step: every DG1 node takes the relative change of vs and vp
    at its position (`RelativeChange.at_points`), and the shear and bulk
    modulus follow from the rule of the model file (`glad_moduli`) with the
    PREM-stable values of the node. The density and gravity stay 1D.

    The G-ADOPT choice of 2026-10-19 (Sia) for the 0-60 km shell: mu and K
    there are the values of the GLAD-M25 rule at the cell centroid, so that
    the linear DG1 interpolant does not extrapolate the steep crustal
    changes of GLAD-M25 to negative or near-zero values at the cell
    vertices. The written DG1 field is the interpolation of
    `conditional(r_centroid > R_60, f_dg0, f_dg1)`. The function stops
    unless mu and K are positive at every centroid of the 0-60 km shell, at
    every vertex of every cell below it, and at every vertex of every cell
    of the written fields.

    Args:
      mantle: the mantle mesh (P2 coordinates).
      earth: the `Exp12Earth`.
      scales: the `exp12_common.Scales`.
      relative: the `RelativeChange` of GLAD-M25.
      radial: the `giamsh.radial.RadialStructure` of the mesh sidecar.

    Returns:
      `(fields, relative_fields, moduli_record)`: a dictionary with
      `density`, `shear_modulus`, `bulk_modulus` (DG1) and `gravity` (CG2),
      all non-dimensional; a dictionary with the DG1 fields `dlnvs` and
      `dlnvp`, for the record and the ParaView file; and the record of the
      0-60 km rule and its checks.
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

    # Gravity is continuous; the side does not matter.
    cg2 = FunctionSpace(mantle, "CG", 2)
    r_cg2 = np.clip(node_radius_m(cg2), scales.Rc_m, scales.Re_m)
    g = Function(cg2, name="gravity")
    g.dat.data_with_halos[:] = earth.elastic(r_cg2)["g_m_s2"] / scales.g_bar
    return ({"density": rho, "shear_modulus": mu, "bulk_modulus": kappa,
             "gravity": g},
            {"dlnvs": dlnvs, "dlnvp": dlnvp}, moduli_record)


# --------------------------------------------------------------------------
# The Extended Burgers elements and the drop rule
# --------------------------------------------------------------------------


def load_eb_fit(path, key):
    """The normalised Extended Burgers fit of `key` in the fit file `path`.

    The fit gives N pairs (mu_bar_j, tau_bar_j): the moduli in units of
    mu_inf and the element times in units of the local Maxwell time
    tau_M = eta_0 / mu_inf (design handout section 5). The function checks
    what the run depends on and stops otherwise:

    - the EB parameters of the file are those of `EB_PARAMETERS` (S22);
    - every mu_bar_j is at least `EB_MU_FLOOR` and every tau_bar_j is
      positive, so no element has the Maxwell time 0/0;
    - sum_j mu_bar_j tau_bar_j = 1 to `EB_SUM_TOLERANCE`, so the long-term
      viscosity of the elements is eta_0;
    - the two lists have the same length.

    Args:
      path: the fit file (`eb_fit.json`).
      key: `<group>/<N>`, for example `free_time_fits/3`.

    Returns:
      `(mu_bar, tau_bar, record)`: two numpy arrays of N values and the part
      of the record of the run that describes the fit.
    """
    with open(path) as handle:
        fit_file = json.load(handle)
    group, n_text = key.split("/")
    fit = fit_file[group][n_text]
    parameters = fit_file["parameters"]
    for name, value in EB_PARAMETERS.items():
        if abs(float(parameters[name]) - value) > 1e-12:
            raise RuntimeError(f"{path}: the fit has {name} = "
                               f"{parameters[name]}, the run needs {value} "
                               "(design handout S22)")
    mu_bar = np.asarray(fit["mu_bar"], dtype=float)
    tau_bar = np.asarray(fit["tau_bar"], dtype=float)
    if mu_bar.shape != tau_bar.shape or mu_bar.size != int(n_text):
        raise RuntimeError(f"{path} {key}: {mu_bar.size} moduli and "
                           f"{tau_bar.size} times, expected {n_text} each")
    if mu_bar.min() < EB_MU_FLOOR or tau_bar.min() <= 0.0:
        raise RuntimeError(f"{path} {key}: a modulus below {EB_MU_FLOOR} or "
                           f"a time not positive: {mu_bar}, {tau_bar}")
    long_term = float(np.sum(mu_bar * tau_bar))
    if abs(long_term - 1.0) > EB_SUM_TOLERANCE:
        raise RuntimeError(f"{path} {key}: sum mu_bar tau_bar = {long_term}, "
                           "not 1; the long-term viscosity would not be eta_0")
    # The slowest element must be unique: the drop rule folds the dropped
    # elements into it.
    if np.count_nonzero(tau_bar == tau_bar.max()) != 1:
        raise RuntimeError(f"{path} {key}: the slowest element time "
                           f"{tau_bar.max()} is not unique")
    record = {"fit_file": os.path.abspath(path),
              "fit_file_sha256": common.sha256sum(path),
              "fit_script_sha256": fit_file.get("script_sha256"),
              "fit_key": key, "parameters": parameters,
              "dimensionalise": fit_file.get("dimensionalise"),
              # The entry of the fit file as written by eb_fit.py. Its own
              # `dropped` key is the weight-floor rule of the fit (design
              # handout 6.3), not the drop rule of this run, which is in
              # `drop_rule` of the run record.
              "fit_entry": fit, "mu_bar": mu_bar, "tau_bar": tau_bar,
              "sum_mu_bar": float(mu_bar.sum()),
              "sum_mu_bar_tau_bar": long_term}
    return mu_bar, tau_bar, record


def drop_fast_elements(mu_bar, tau_bar, tau_m_min, dt_min, max_ratio):
    """The normalised elements of every cell after the drop rule (S32).

    Pure numpy, no mesh, so the rule can be tested alone. See THE DROP RULE
    in the module docstring.

    Element j of a cell is dropped when dt_min / (tau_bar_j tau_m_min) >
    max_ratio. The slowest element s (largest tau_bar) is never dropped. In
    a cell with dropped elements D, the slowest element takes their viscous
    part, mu_bar_s tau_bar_s' = mu_bar_s tau_bar_s + sum_{j in D} mu_bar_j
    tau_bar_j, and its modulus is shared equally between its own slot and
    the slots of D, all with the time tau_bar_s'. So in every cell:

    - sum_j mu_j' tau_j' = sum_j mu_bar_j tau_bar_j (= 1): the long-term
      viscosity is eta_0;
    - sum_j mu_j' = sum_j mu_bar_j - sum_{j in D} mu_bar_j: the instant
      modulus loses the dropped moduli;
    - the elements of the slots of D and of s have one time and act as one
      Maxwell element with the modulus mu_bar_s.

    Args:
      mu_bar, tau_bar: the normalised fit, arrays of N values.
      tau_m_min: the shortest local Maxwell time eta_0 / max(mu_inf) of
        every cell, an array of `cells` values, in the unit of `dt_min`.
      dt_min: the step against which the elements are tested.
      max_ratio: the largest allowed dt_min / tau_j of a kept element.

    Returns:
      `(mu, tau, dropped, slowest_fast)`: `mu` and `tau` arrays
      `(cells, N)` of the normalised modulus and element time of every slot;
      `dropped` a bool array `(cells, N)`; `slowest_fast` a bool array
      `(cells,)`, True where the slowest element is also faster than the
      limit (it is kept; the cell keeps every element).
    """
    mu_bar = np.asarray(mu_bar, dtype=float)
    tau_bar = np.asarray(tau_bar, dtype=float)
    tau_m_min = np.asarray(tau_m_min, dtype=float)
    n_cells, n = tau_m_min.size, mu_bar.size
    slow = int(np.argmax(tau_bar))
    # dt / tau_j of every element in every cell.
    ratio = dt_min / (tau_bar[None, :] * tau_m_min[:, None])
    dropped = ratio > max_ratio
    slowest_fast = dropped[:, slow].copy()
    # A cell whose slowest element is too fast keeps every element: there is
    # no slower element to carry the viscous part of the others.
    dropped[slowest_fast, :] = False
    dropped[:, slow] = False

    mu = np.broadcast_to(mu_bar, (n_cells, n)).copy()
    tau = np.broadcast_to(tau_bar, (n_cells, n)).copy()
    n_drop = dropped.sum(axis=1)
    rows = np.flatnonzero(n_drop)
    if rows.size:
        # The viscous part of the dropped elements of each cell.
        moved = (dropped[rows] * (mu_bar * tau_bar)[None, :]).sum(axis=1)
        tau_slow = (mu_bar[slow] * tau_bar[slow] + moved) / mu_bar[slow]
        share = mu_bar[slow] / (n_drop[rows] + 1)
        # The slots that carry the slowest element: its own and the dropped.
        carry = dropped[rows].copy()
        carry[:, slow] = True
        mu_rows = mu[rows]
        tau_rows = tau[rows]
        mu_rows[carry] = np.repeat(share, carry.sum(axis=1))
        tau_rows[carry] = np.repeat(tau_slow, carry.sum(axis=1))
        mu[rows] = mu_rows
        tau[rows] = tau_rows
    return mu, tau, dropped, slowest_fast


def burgers_elements(mantle, fields, scales, radial, mu_bar, tau_bar,
                     dt_min_yr, max_ratio):
    """The moduli and viscosities of the N Maxwell elements of Exp12.

    Below the `EB_SPHERE` (90 km) sphere: the EB fit with the drop rule
    (`drop_fast_elements`), mu_j = mu_inf mu_j' at each DG1 node and
    eta_j = eta_0 mu_j' tau_j' in each cell. Above it: one Maxwell element
    in N parts, mu_j = mu_inf w_j, eta_j = eta_0 w_j, w_j = mu_bar_j /
    sum mu_bar. The shortest local Maxwell time of a cell is eta_0 over the
    largest mu_inf at its four vertices.

    The DG1 vector space and the DG1 scalar space of `shear_modulus` share
    the node numbering, and so do the DG0 spaces, so each component is a
    nodal product and needs no interpolation. Every cell, owned or halo, is
    computed from the values of the cell, so the halos agree with their
    owners.

    Args:
      mantle: the mantle mesh.
      fields: `shear_modulus` (mu_inf, DG1) and `viscosity` (eta_0, DG0),
        non-dimensional.
      scales: the `exp12_common.Scales`.
      radial: the `RadialStructure` of the mesh sidecar.
      mu_bar, tau_bar: the normalised fit (`load_eb_fit`).
      dt_min_yr: the step of the drop rule, yr.
      max_ratio: the limit of the drop rule.

    Returns:
      `(shear_moduli, viscosities, diagnostics, counts, errors)`: the DG1
      and DG0 vector fields of N components (units of mu_bar and eta_bar);
      DG0 fields for the record and ParaView (`eb_cell`, 1 below 90 km;
      `instant_ratio` and `instant_ratio_undropped`, sum_j mu_j / mu_inf
      with and without the drop; `n_dropped`; `dropped_<j>` for every j);
      cell counts over all ranks; the largest relative errors of the sum
      checks.

    Raises:
      RuntimeError: if a sum check fails or a Maxwell time is not positive.
    """
    n = mu_bar.size
    weights = mu_bar / mu_bar.sum()
    dg0 = FunctionSpace(mantle, "DG", 0)
    dg1 = FunctionSpace(mantle, "DG", 1)
    X = SpatialCoordinate(mantle)
    # Cell order: the order of `cell_node_list`, owned cells first, then the
    # halo cells. `cell_dof` is the DG0 node of each cell, `nodes` the four
    # DG1 nodes of each cell.
    cell_dof = dg0.cell_node_list[:, 0]
    nodes = dg1.cell_node_list
    n_owned = mantle.cell_set.size
    centroid = Function(dg0).interpolate(sqrt(dot(X, X)))
    r_cell = np.array(centroid.dat.data_ro_with_halos)[cell_dof]
    eb_cell = r_cell < radial.radius(EB_SPHERE)

    eta0 = np.array(fields["viscosity"].dat.data_ro_with_halos)
    mu_inf = np.array(fields["shear_modulus"].dat.data_ro_with_halos)
    eta0_cell = eta0[cell_dof]
    # The largest mu_inf of each cell is at a vertex (a linear field).
    mu_max_cell = vertex_values(fields["shear_modulus"]).max(axis=1)
    if not COMM_WORLD.allreduce(float(mu_max_cell.min(initial=np.inf)),
                                op=MPI.MIN) > 0.0:
        raise RuntimeError("mu_inf is not positive at every vertex")
    tau_m_min = eta0_cell / mu_max_cell
    dt_min = dt_min_yr / common.T_BAR_YR

    mu_cell, tau_cell, dropped, slowest_fast = drop_fast_elements(
        mu_bar, tau_bar, tau_m_min, dt_min, max_ratio)
    # The undropped fit, for the record of the change.
    mu_ref = np.broadcast_to(mu_bar, mu_cell.shape).copy()
    # Above 90 km: one Maxwell element with the local tau_M in N parts.
    lid = ~eb_cell
    mu_cell[lid] = weights
    tau_cell[lid] = 1.0
    mu_ref[lid] = weights
    dropped[lid] = False
    slowest_fast[lid] = False

    # Every DG node belongs to one cell of `cell_node_list`, so the cell
    # loops below write every node; the check makes that explicit.
    if (np.unique(nodes).size != mu_inf.size
            or np.unique(cell_dof).size != eta0.size):
        raise RuntimeError("the cells of this rank do not cover every DG node")
    moduli = Function(VectorFunctionSpace(mantle, "DG", 1, dim=n),
                      name="shear_moduli")
    m = np.empty((mu_inf.size, n))
    m[nodes] = mu_inf[nodes][:, :, None] * mu_cell[:, None, :]
    moduli.dat.data_with_halos[:] = m
    viscosities = Function(VectorFunctionSpace(mantle, "DG", 0, dim=n),
                           name="viscosities")
    v = np.empty((eta0.size, n))
    v[cell_dof] = eta0_cell[:, None] * mu_cell * tau_cell
    viscosities.dat.data_with_halos[:] = v

    # The checks, on every cell of this rank: the long-term viscosity is
    # eta_0 everywhere (the sum constraint is kept); the modulus of a cell
    # above 90 km is mu_inf; the instant modulus below is mu_inf times the
    # sum of the kept moduli; every Maxwell time is positive.
    def worst(a, b):
        local = np.max(np.abs(a - b) / np.abs(b)) if a.size else 0.0
        return COMM_WORLD.allreduce(float(local), op=MPI.MAX)

    m_cells = m[nodes]                      # (cells, 4, N)
    errors = {
        "long_term_viscosity": worst(v[cell_dof].sum(axis=1), eta0_cell),
        "lid_modulus_sum": worst(m_cells[lid].sum(axis=2),
                                 mu_inf[nodes][lid]),
        "eb_instant_modulus": worst(
            m_cells[eb_cell].sum(axis=2),
            mu_inf[nodes][eb_cell] * mu_cell[eb_cell].sum(axis=1)[:, None]),
    }
    bad = {k: e for k, e in errors.items() if not e < 1e-12}
    if bad:
        raise RuntimeError(f"the element fields fail the sum checks: {bad}")
    if not COMM_WORLD.allreduce(float((v[cell_dof] / mu_max_cell[:, None])
                                      .min(initial=np.inf)), op=MPI.MIN) > 0:
        raise RuntimeError("an element has a Maxwell time that is not "
                           "positive")

    def dg0_field(values, name):
        field = Function(dg0, name=name)
        data = np.zeros(eta0.size)
        data[cell_dof] = values
        field.dat.data_with_halos[:] = data
        return field

    # The shortest Maxwell time of the kept elements of each cell, yr (the
    # linear value; the power law divides it by the factor f at run time).
    kept_tau = np.where(dropped, np.inf, tau_cell * tau_m_min[:, None])
    diagnostics = {
        "kept_min_tau_yr": dg0_field(kept_tau.min(axis=1) * common.T_BAR_YR,
                                     "kept_min_tau_yr"),
        "eb_cell": dg0_field(eb_cell.astype(float), "eb_cell"),
        "instant_ratio": dg0_field(mu_cell.sum(axis=1), "instant_ratio"),
        "instant_ratio_undropped": dg0_field(mu_ref.sum(axis=1),
                                             "instant_ratio_undropped"),
        "n_dropped": dg0_field(dropped.sum(axis=1).astype(float),
                               "n_dropped"),
    }
    for j in range(n):
        diagnostics[f"dropped_{j}"] = dg0_field(dropped[:, j].astype(float),
                                                f"dropped_{j}")

    def count(mask):
        return COMM_WORLD.allreduce(int(np.count_nonzero(mask[:n_owned])),
                                    op=MPI.SUM)

    counts = {"cells": count(np.ones(r_cell.size, dtype=bool)),
              "eb_cells": count(eb_cell),
              "lid_cells": count(lid),
              "cells_with_a_dropped_element": count(dropped.any(axis=1)),
              "dropped_cells_per_element": [count(dropped[:, j])
                                            for j in range(n)],
              "slowest_below_limit_cells": count(slowest_fast)}
    return moduli, viscosities, diagnostics, counts, errors


def element_columns(vector):
    """The scalar Functions of the components of a DG vector field.

    The same nodal copy as `exp12_common.element_fields`, for the record and
    the ParaView file of this script.
    """
    space = FunctionSpace(vector.function_space().mesh(),
                          vector.function_space().ufl_element().sub_elements[0])
    columns = []
    for j in range(vector.function_space().value_size):
        column = Function(space, name=f"{vector.name()}_{j}")
        column.dat.data_wo_with_halos[:] = vector.dat.data_ro_with_halos[:, j]
        columns.append(column)
    return columns


def burgers_by_layer(cells, layers, fields, moduli, viscosities, diagnostics):
    """The record of the EB elements and of the drop rule per mantle layer.

    For each layer: the instant modulus ratio sum_j mu_j / mu_inf with and
    without the drop (min, max, volume mean), the volume mean of the instant
    modulus sum_j mu_j in Pa with and without the drop and its relative
    change, the volume fraction of the cells where each element is dropped,
    and for each element j its Maxwell time eta_j / mu_j in years (min, max,
    volume mean over the nodal values) and the volume fraction of the layer
    where it is shorter than each step of `EB_STEP_CHECK_YR`. The fractions
    of the Maxwell times are integrals of an indicator at the quadrature
    points of each cell, so they resolve the radial change of mu_inf inside
    a cell.

    Returns:
      `{layer: {...}}`.
    """
    mu_j = element_columns(moduli)
    eta_j = element_columns(viscosities)
    mu_inf = fields["shear_modulus"]
    ratio = cells.statistics(diagnostics["instant_ratio"], 1.0)
    ratio_ref = cells.statistics(diagnostics["instant_ratio_undropped"], 1.0)
    # The volume means of the instant modulus in Pa, with and without the
    # drop (mu_inf times the DG0 ratio).
    instant = cells.statistics(mu_inf * diagnostics["instant_ratio"],
                               common.MU_BAR)
    instant_ref = cells.statistics(
        mu_inf * diagnostics["instant_ratio_undropped"], common.MU_BAR)
    n = len(mu_j)
    dropped = [cells.statistics(diagnostics[f"dropped_{j}"], 1.0)
               for j in range(n)]
    times = [cells.statistics(eta / mu, common.T_BAR_YR)
             for mu, eta in zip(mu_j, eta_j)]
    out = {}
    for name, _, _ in layers:
        elements = []
        for j, (mu, eta) in enumerate(zip(mu_j, eta_j)):
            tau_yr = eta / mu * common.T_BAR_YR
            fractions = {}
            for step in EB_STEP_CHECK_YR:
                below = float(assemble(cells.indicators[name]
                                       * conditional(lt(tau_yr, step), 1.0, 0.0)
                                       * dx(domain=cells.mantle)))
                fractions[f"below_{step:g}_yr"] = below / cells.volume[name]
            elements.append({"element": j,
                             "maxwell_time_yr": times[j][name],
                             "volume_fraction": fractions,
                             "dropped_volume_fraction":
                                 dropped[j][name]["mean"]})
        mean, mean_ref = instant[name]["mean"], instant_ref[name]["mean"]
        out[name] = {"instant_modulus_ratio": ratio[name],
                     "instant_modulus_ratio_undropped": ratio_ref[name],
                     "instant_modulus_mean_pa": mean,
                     "instant_modulus_mean_undropped_pa": mean_ref,
                     "instant_modulus_relative_change":
                         (mean - mean_ref) / mean_ref,
                     "elements": elements}
    return out


def power_law_fields(mantle, eta_pa_s, scales):
    """The DG0 exponent and transition stress of the power law (S21).

    The rule is tested at the cell centroid, as the viscosity is: a cell is
    a power-law cell when its centroid depth is strictly between 90 and
    670 km and its viscosity eta_0 is strictly below 1e22 Pa s. No cell of
    the mesh crosses the spheres at 90 and 670 km. The values are written
    into the data array, not interpolated, so they are exactly 1.0 and
    exactly `POWER_LAW_EXPONENT`.

    Args:
      mantle: the mantle mesh.
      eta_pa_s: the viscosity of every DG0 node with halos, Pa s, in the
        node order of the DG0 space of `mantle`.
      scales: the `exp12_common.Scales`.

    Returns:
      `(exponent, transition_stress)`: two DG0 Functions; the transition
      stress is in units of mu_bar (non-dimensional 1e-5 for 1 MPa).
    """
    dg0 = FunctionSpace(mantle, "DG", 0)
    X = SpatialCoordinate(mantle)
    # The centroid radius of every cell (DG0 takes the value at the
    # centroid), in metres, and the depth below the surface of the model.
    r_cell = np.array(Function(dg0).interpolate(sqrt(dot(X, X)))
                      .dat.data_ro_with_halos) * D_SCALE
    depth = scales.Re_m - r_cell
    top, bottom = POWER_LAW_DEPTH_RANGE_M
    power = ((depth > top) & (depth < bottom)
             & (np.asarray(eta_pa_s) < POWER_LAW_ETA_CAP_PA_S))
    exponent = Function(dg0, name="exponent")
    # Exact values: 1.0 for the linear cells, so `n - 1 == 0` holds there.
    exponent.dat.data_with_halos[:] = np.where(power, POWER_LAW_EXPONENT, 1.0)
    # The transition stress is positive in every cell (`power_law_factor`
    # divides by it also where n = 1).
    transition = Function(dg0, name="transition_stress")
    transition.dat.data_with_halos[:] = TRANSITION_STRESS_PA / common.MU_BAR
    return exponent, transition


def core_density(elastic, scales, rule):
    """The density of the homogeneous fluid core, kg/m^3.

    Args:
      elastic: the `giamip.earth.ElasticProfile`.
      scales: the `exp12_common.Scales`.
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
      earth: its `Exp12Earth`.
      scales: the `exp12_common.Scales`.
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
    of `step_years`, and the last year. With the Exp12 years (500 yr steps
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
    ice is CG1 (see `exp12_common.ICE_SPACE`). Every node of the mantle mesh
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
    p.add_argument("--schedule",
                   choices=["uniform", *common.WINDOW_SCHEDULES],
                   default="uniform",
                   help="the march of the input file: uniform steps of "
                        "--step_years (default), schedule C (windows: "
                        "the protocol steps plus the 25 yr and 10 yr steps "
                        "of the fast melt windows, decisions S33 and S34), "
                        "or schedule D (windows-coarse: 1 kyr steps plus "
                        "50 yr and 20 yr steps in the same windows, "
                        "decision R12), with the ice of every year of the "
                        "schedule, linear in time between the protocol "
                        "years; --step_years is then not used")
    p.add_argument("--core_density", choices=["mean", "cmb"], default="mean")
    p.add_argument("--eb_fit", default=EB_FIT_FILE,
                   help="the fit file of the EB elements (eb_fit.json)")
    p.add_argument("--eb_fit_key", default=EB_FIT_KEY,
                   help="the fit in the fit file, <group>/<N> (default "
                        "free_time_fits/3)")
    p.add_argument("--eb_drop_dt_years", type=float, default=EB_DROP_DT_YR,
                   help="the step of the drop rule, yr (default 10)")
    p.add_argument("--eb_drop_max_ratio", type=float,
                   default=EB_DROP_MAX_RATIO,
                   help="drop element j where dt / tau_j is larger (default "
                        "2); 'inf' keeps every element")
    p.add_argument("--earth_only", action="store_true",
                   help="stop after the earth and rheology fields and write "
                        "only their record (the serial check of the drop "
                        "rule)")
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
    say = common.RunLog(os.path.join(args.output_path, "prepare_exp12.log"))
    say("=" * 78)
    say(f"GIAMIP {EXPERIMENT}: input file of the bed inversion, "
        f"{COMM_WORLD.size} ranks")
    say("=" * 78)
    timings = {}

    # The experiment and its earth model, with the G-ADOPT choices for the
    # fields that the protocol leaves open. The script stops for any
    # rheology other than `RHEOLOGY` (`exp12_earth`).
    experiment = GiamipExperiment(EXPERIMENT)
    earth = exp12_earth(experiment)
    scales = common.Scales.from_elastic(earth.elastic)
    say(f"  G-ADOPT choices: {earth.choices}")
    say(f"  elastic structure {earth.model['id']} on {earth.model['base']}, "
        f"viscosity {earth.viscosity.record['model_id']}, rheology "
        f"{RHEOLOGY}")
    say(f"  scales: {scales.as_dict()}")

    # The EB fit, read and checked before any expensive step.
    mu_bar, tau_bar, burgers_record = load_eb_fit(args.eb_fit, args.eb_fit_key)
    say(f"  EB fit {args.eb_fit_key} of {args.eb_fit} (SHA-256 "
        f"{burgers_record['fit_file_sha256'][:16]}): tau_bar {tau_bar}, "
        f"mu_bar {mu_bar}, sum mu_bar {mu_bar.sum():.4f}, sum mu_bar tau_bar "
        f"{burgers_record['sum_mu_bar_tau_bar']:.15f}")
    say(f"  drop rule: element j dropped where {args.eb_drop_dt_years:g} yr "
        f"/ tau_j > {args.eb_drop_max_ratio:g}"
        + (f", i.e. tau_j < {args.eb_drop_dt_years / args.eb_drop_max_ratio:g}"
           " yr" if np.isfinite(args.eb_drop_max_ratio)
           else " (no element is dropped)"))

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
    # The 90 km sphere of the EB rule must be at 90 km below Re.
    eb_depth_m = (scales.Re - radial.radius(EB_SPHERE)) * D_SCALE
    if abs(eb_depth_m - POWER_LAW_DEPTH_RANGE_M[0]) > 1.0:
        raise RuntimeError(f"the {EB_SPHERE} sphere is {eb_depth_m:.1f} m "
                           "below Re, not 90 km")
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

    # The viscosity. The 3-D model is loaded from g-drift for this step only
    # (once per node) and released before the tomography is read.
    tic = time.time()
    dataset_info = earth.viscosity.load()
    say(f"  viscosity: {earth.viscosity.dataset} ({earth.viscosity.field}), "
        f"g-drift {earth.viscosity.record['gdrift']}, file {dataset_info}")
    say(f"  lower bound (log10 Pa s, by depth): "
        f"{earth.viscosity.record['lower_bound']}; no upper bound; no lid")
    common.memory_now("with the viscosity dataset", say)
    eta, log10_fields = viscosity_fields(mantle, earth, scales)
    earth.viscosity.release()
    timings["viscosity_s"] = time.time() - tic

    # The relative change of GLAD-M25, read on rank 0, with the per-depth
    # check at the PREM-stable jumps on mesh spheres, and the moduli.
    tic = time.time()
    check_radii = prem_jumps_on_spheres(earth.elastic, radial, scales)
    relative = RelativeChange(earth.model, earth.model_sha256, earth.elastic,
                              check_radii)
    tomography = relative.record
    say(f"  g-drift {tomography['gdrift_version']} "
        f"({tomography['gdrift_distribution_version']}): "
        f"{tomography['dataset']} sha256 {tomography['sha256_file']}, "
        f"{tomography['depths']} depths of {tomography['points_per_depth']} "
        f"points, {tomography['top_depth_km']:g} to "
        f"{tomography['bottom_depth_km']:g} km")
    fields, relative_fields, moduli_record = elastic_fields(
        mantle, earth, scales, relative, radial)
    # The tomography is not needed after this; rank 0 frees about 0.9 GB.
    del relative
    fields["viscosity"] = eta
    timings["moduli_s"] = time.time() - tic
    say(f"  moduli: centroid values above the {moduli_record['sphere']} "
        f"sphere in {moduli_record['cells_centroid_rule']} cells, DG1 values "
        f"in {moduli_record['cells_dg1_rule']} cells")
    common.memory_now("after the moduli", say)

    # The rheology: the EB elements with the drop rule, and the power law.
    tic = time.time()
    shear_moduli, viscosities, eb_diagnostics, eb_counts, eb_errors = \
        burgers_elements(mantle, fields, scales, radial, mu_bar, tau_bar,
                         args.eb_drop_dt_years, args.eb_drop_max_ratio)
    fields["shear_moduli"] = shear_moduli
    fields["viscosities"] = viscosities
    eta_pa_s = np.array(eta.dat.data_ro_with_halos) * common.ETA_BAR
    exponent, transition = power_law_fields(mantle, eta_pa_s, scales)
    fields["exponent"] = exponent
    fields["transition_stress"] = transition
    timings["rheology_s"] = time.time() - tic
    say(f"  EB cells {eb_counts['eb_cells']} of {eb_counts['cells']} (below "
        f"90 km), cells with a dropped element "
        f"{eb_counts['cells_with_a_dropped_element']}, per element "
        f"{eb_counts['dropped_cells_per_element']}, slowest element below "
        f"the limit in {eb_counts['slowest_below_limit_cells']} cells; sum "
        f"checks {eb_errors}")
    if eb_counts["slowest_below_limit_cells"]:
        say("  WARNING: cells whose slowest element is faster than the limit "
            "keep every element")

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
          "exponent": ("1", 1.0, "power_law_exponent"),
          "transition_stress": ("Pa", common.MU_BAR, "transition_stress_pa"),
          # The relative change of GLAD-M25 at the DG1 nodes and the EB
          # diagnostics. They are not in the input file; they are only
          # recorded and shown.
          "dlnvs": ("1", 1.0, "dlnvs"),
          "dlnvp": ("1", 1.0, "dlnvp"),
          "instant_ratio": ("1", 1.0, "instant_modulus_ratio"),
          "n_dropped": ("1", 1.0, "eb_dropped_elements")}
    earth_view = {**fields, **relative_fields, **eb_diagnostics}
    layer_stats = {name: {"unit": unit,
                          "layers": cells.statistics(earth_view[name], scale)}
                   for name, (unit, scale, _) in si.items()}
    for name in ("dlnvs", "dlnvp", "shear_modulus", "bulk_modulus"):
        for layer, _, _ in layers:
            row = layer_stats[name]["layers"][layer]
            say(f"    {name} {layer}: min {row['min']:.4g}, max "
                f"{row['max']:.4g}, mean {row['mean']:.4g} "
                f"{layer_stats[name]['unit']}")
    # The 3-D viscosity per layer, in log10 Pa s: min, max and volume mean
    # before and after the lower bound, and the volume fraction of the
    # cells at the bound (the volume mean of the 0/1 indicator).
    raw_stats = cells.statistics(log10_fields["log10_viscosity_raw"], 1.0)
    log10_stats = cells.statistics(log10_fields["log10_viscosity"], 1.0)
    bound_stats = cells.statistics(log10_fields["at_lower_bound"], 1.0)
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
            f"{row['max']:.3f}, mean {row['mean']:.3f}; at the lower bound "
            f"{100 * bound_stats[name]['mean']:.1f} % of the volume")
    mantle_volume = float(assemble(1.0 * dx(domain=mantle)))

    def mantle_mean(f):
        # The volume mean of a field over the mantle.
        return float(assemble(f * dx(domain=mantle))) / mantle_volume

    bound_fraction = mantle_mean(log10_fields["at_lower_bound"])
    earth.viscosity.record["by_layer"] = log10_by_layer
    earth.viscosity.record["lower_bound_volume_fraction_mantle"] = \
        bound_fraction

    # The EB elements and the drop rule per layer and for the mantle.
    burgers_layers = burgers_by_layer(cells, layers, fields, shear_moduli,
                                      viscosities, eb_diagnostics)
    for name, _, _ in layers:
        row = burgers_layers[name]
        dropped = ", ".join(f"{100 * e['dropped_volume_fraction']:.1f}"
                            for e in row["elements"])
        say(f"    EB {name}: dropped volume % per element [{dropped}], "
            f"instant modulus ratio mean "
            f"{row['instant_modulus_ratio']['mean']:.4f} (undropped "
            f"{row['instant_modulus_ratio_undropped']['mean']:.4f}), "
            f"instant modulus change "
            f"{100 * row['instant_modulus_relative_change']:+.2f} %")
    mu_inf = fields["shear_modulus"]
    instant = mantle_mean(mu_inf * eb_diagnostics["instant_ratio"])
    instant_ref = mantle_mean(mu_inf
                              * eb_diagnostics["instant_ratio_undropped"])
    eb_volume = mantle_mean(eb_diagnostics["eb_cell"])
    eb_instant = float(assemble(eb_diagnostics["eb_cell"] * mu_inf
                                * eb_diagnostics["instant_ratio"]
                                * dx(domain=mantle)))
    eb_instant_ref = float(assemble(
        eb_diagnostics["eb_cell"] * mu_inf
        * eb_diagnostics["instant_ratio_undropped"] * dx(domain=mantle)))
    drop_mantle = {
        "volume_fraction_mantle_any": mantle_mean(
            conditional(eb_diagnostics["n_dropped"] > 0.5, 1.0, 0.0)),
        "volume_fraction_mantle_per_element": [
            mantle_mean(eb_diagnostics[f"dropped_{j}"])
            for j in range(mu_bar.size)],
        "eb_volume_fraction_mantle": eb_volume,
        "instant_modulus_mantle_mean_pa": instant * common.MU_BAR,
        "instant_modulus_mantle_mean_undropped_pa": instant_ref * common.MU_BAR,
        "instant_modulus_mantle_relative_change":
            (instant - instant_ref) / instant_ref,
        "instant_modulus_eb_region_relative_change":
            (eb_instant - eb_instant_ref) / eb_instant_ref,
        "seismic_modulus_mantle_mean_pa": mantle_mean(mu_inf) * common.MU_BAR,
    }
    say(f"    drop rule: {100 * drop_mantle['volume_fraction_mantle_any']:.2f}"
        f" % of the mantle volume has a dropped element (per element "
        f"{[round(100 * f, 2) for f in drop_mantle['volume_fraction_mantle_per_element']]} %); "
        f"instant modulus change "
        f"{100 * drop_mantle['instant_modulus_mantle_relative_change']:+.2f} % "
        f"(mantle), "
        f"{100 * drop_mantle['instant_modulus_eb_region_relative_change']:+.2f}"
        f" % (below 90 km)")
    burgers_record.update({
        "n_elements": int(mu_bar.size),
        "eb_sphere": EB_SPHERE,
        "eb_depth_m": eb_depth_m,
        "lid_rule": "centroid above the d90 sphere: one Maxwell element in N "
                    "parts, mu_inf w_j, eta_0 w_j, w_j = mu_bar_j / sum "
                    "mu_bar (no elastic lid in Exp12); below: the EB "
                    "elements mu_inf mu_bar_j', eta_0 mu_bar_j' tau_bar_j' "
                    "after the drop rule",
        "drop_rule": {
            "decision": "S32 (design handout 2026-10-20, sixth round)",
            "dt_years": args.eb_drop_dt_years,
            "max_ratio": args.eb_drop_max_ratio,
            "time_limit_years": (args.eb_drop_dt_years
                                 / args.eb_drop_max_ratio),
            "rule": "below 90 km, element j is dropped in a cell where "
                    "dt / tau_j > max_ratio, tau_j = tau_bar_j eta_0 / "
                    "max over the cell vertices of mu_inf; the slowest "
                    "element is never dropped; the viscous part mu_bar_j "
                    "tau_bar_j of the dropped elements is added to the "
                    "slowest element (sum constraint kept: long-term "
                    "viscosity eta_0), whose modulus is shared equally "
                    "between its slot and the dropped slots at one time",
            "basis": "gate of the shared code: Newton stalled with the power "
                     "law at dt / tau of the fastest element = 5 and "
                     "converged at 2.5 and 0.5 "
                     "(NOTES/team/builder-shared-transient.md, piece 1)",
            "counts": eb_counts,
            "sum_check_errors": eb_errors,
            "mantle": drop_mantle,
        },
        "by_layer": burgers_layers,
    })
    exponent_by_layer = cells.distinct_values(exponent, 1.0)
    power_law_volume = {}
    for name, _, _ in layers:
        rows = exponent_by_layer[name]
        if isinstance(rows, str):
            raise RuntimeError(f"shell {name}: exponent values {rows}")
        power_law_volume[name] = sum(row["volume_fraction"] for row in rows
                                     if row["value"] == POWER_LAW_EXPONENT)
        say(f"    exponent {name}: power law in "
            f"{100 * power_law_volume[name]:.1f} % of the volume")
    power_law_mantle = mantle_mean(conditional(exponent > 1.5, 1.0, 0.0))
    if not power_law_mantle > 0.0:
        raise RuntimeError("no mantle cell has the power law")
    say(f"    power law in {100 * power_law_mantle:.2f} % of the mantle "
        "volume")
    # The prior for the Newton watch (S25): the rule tests the linear
    # Maxwell times, and the power law divides every time of a power-law
    # cell by the factor f at run time. For f = 2, 5 and 17 (handout 7.3:
    # 17 at 3 MPa with sigma_T = 1 MPa), the volume fraction of the mantle
    # where a power-law cell has a kept element faster than the limit of the
    # rule, and faster than dt / 5 (the ratio at which the gate stalled).
    kept_tau = eb_diagnostics["kept_min_tau_yr"]
    power_cell = conditional(exponent > 1.5, 1.0, 0.0)
    stall_yr = args.eb_drop_dt_years / 5.0
    limit_yr = args.eb_drop_dt_years / args.eb_drop_max_ratio
    effective = {}
    for factor in (2.0, 5.0, 17.0):
        effective[f"f_{factor:g}"] = {
            "below_limit_volume_fraction_mantle": mantle_mean(
                power_cell * conditional(lt(kept_tau / factor, limit_yr),
                                         1.0, 0.0)),
            "below_stall_volume_fraction_mantle": mantle_mean(
                power_cell * conditional(lt(kept_tau / factor, stall_yr),
                                         1.0, 0.0))}
        say(f"    power-law cells with a kept element below "
            f"{limit_yr:g} yr / {stall_yr:g} yr at f = {factor:g}: "
            f"{100 * effective[f'f_{factor:g}']['below_limit_volume_fraction_mantle']:.2f}"
            f" % / "
            f"{100 * effective[f'f_{factor:g}']['below_stall_volume_fraction_mantle']:.2f}"
            " % of the mantle volume")
    burgers_record["drop_rule"]["power_law_effective_times"] = {
        "meaning": "volume fraction of the mantle where a power-law cell has "
                   "a kept element whose linear Maxwell time divided by f "
                   "is below the limit of the rule, or below dt / 5 (the "
                   "gate stall ratio); the rule itself is static",
        "limit_yr": limit_yr, "stall_yr": stall_yr, **effective}
    power_law_record = {
        "exponent": POWER_LAW_EXPONENT,
        "background_stress": BACKGROUND_STRESS_PA / common.MU_BAR,
        "background_stress_MPa": BACKGROUND_STRESS_PA / 1e6,
        "transition_stress_MPa": TRANSITION_STRESS_PA / 1e6,
        "transition_stress_nondim": TRANSITION_STRESS_PA / common.MU_BAR,
        "stress_measure": "sqrt(J2) of the deviatoric stress",
        "form": "Kang et al. (2021) Eq. 12, gadopt power_law_factor; the "
                "factor scales the Maxwell time of every element",
        "rule": {"depth_above_km": POWER_LAW_DEPTH_RANGE_M[0] / 1e3,
                 "depth_below_km": POWER_LAW_DEPTH_RANGE_M[1] / 1e3,
                 "eta_below_pa_s": POWER_LAW_ETA_CAP_PA_S,
                 "tested_at": "cell centroid",
                 "elsewhere": "exponent exactly 1.0"},
        "power_law_volume_fraction_by_layer": power_law_volume,
        "power_law_volume_fraction_mantle": power_law_mantle,
        "decisions": "S21, S27 (P13 n, P14 sigma_T, P15 sigma_b, P19 "
                     "sqrt(J2)), design handout 2026-10-20",
    }
    timings["statistics_s"] = time.time() - tic

    # The part of the record that the earth fields give.
    earth_record = {
        "experiment": EXPERIMENT,
        "case": experiment.definition,
        "case_gadopt": cases.with_gadopt_choices(experiment.id),
        "gadopt_choices": earth.choices,
        "rheology": earth.choices["rheology"],
        "compressibility": earth.choices["compressibility"],
        "giamip_version": giamip.__version__,
        "giamip_path": os.path.dirname(giamip.__file__),
        "viscosity_3d": earth.viscosity.record,
        "tomography": tomography,
        "moduli_lid_rule": moduli_record,
        # The rheology entries that the common file reads: `n_elements`
        # (the EB element fields) and `power_law` (the exponent and the
        # transition stress), design handout 12.5.
        "n_elements": int(mu_bar.size),
        "burgers": burgers_record,
        "power_law": power_law_record,
        "exponent_by_layer": exponent_by_layer,
        "mesh": mesh_record,
        "scales": scales.as_dict(),
        "earth_by_layer": layer_stats,
    }
    if args.earth_only:
        earth_record["timings"] = timings
        earth_record["timings"]["total_s"] = time.time() - tic_run
        earth_record["timings"]["peak_memory_gb"] = common.peak_memory_gb()
        common.write_json(os.path.join(args.output_path,
                                       f"{STEM}_earth_only.json"),
                          earth_record)
        say(f"  earth only: wrote {STEM}_earth_only.json, total "
            f"{earth_record['timings']['total_s']:.1f} s, peak memory "
            f"{earth_record['timings']['peak_memory_gb']:.2f} GB")
        return

    # The forcing: observed sea level, its frozen slope, and the ice.
    tic = time.time()
    # Schedules C and D read every protocol year: `step_years = 1` keeps
    # them all in `coarse_step_indices` (every distance from the start is a
    # whole number of years), as in the forward script.
    windows_march = args.schedule in common.WINDOW_SCHEDULES
    # The name of the schedule in the log lines.
    label = {common.SCHEDULE_C: "schedule C",
             common.SCHEDULE_D: "schedule D"}.get(args.schedule)
    forcing = SurfaceForcing(mantle, experiment,
                             1 if windows_march else args.step_years)
    fields["SL_obs"] = forcing.SL_obs
    # The surface slope of the ocean mask, from the observed bed. It stays
    # fixed in the inversion: the mask width does not depend on the control.
    fields["slope"] = surface_slope(forcing.SL_obs)
    fields["slope"].rename("slope")
    # The years of the forcing slices: the march years of a uniform march,
    # every protocol year for schedules C and D.
    slice_years = np.asarray(forcing.years, dtype=float)
    window_info = None
    if windows_march:
        # Schedule C (design handout 8.3, S33, S34) or D (R12): the years
        # of `common.window_schedule`, the function of the forward script,
        # so the inversion on this file and the forward run march the same
        # years. `window_schedule` also checks the step count of a march
        # from 122 ka. The ice of a year between two protocol years is
        # linear in time between the two slices (`common.ice_bracket`,
        # `common.linear_ice_into`).
        ice_file = os.path.basename(str(forcing.ice_history.path))
        windows, all_years = common.window_schedule(
            args.schedule, ice_file, int(slice_years[0]), slice_years)
        years = all_years.astype(float)
        brackets = [common.ice_bracket(slice_years, y) for y in years]
        window_info = {"ice_history": ice_file, "windows": windows,
                       "steps_to_last_year": int(len(all_years) - 1)}
    else:
        years = slice_years
        # Every year of a uniform march is a slice: slice k, weight 0.
        brackets = [(index, 0.0) for index in range(len(years))]
    year0 = int(np.flatnonzero(np.isclose(years, 0.0))[0])
    # The march of the inversion runs from the first year to year 0; the
    # file holds the ice of steps 0 .. year0, or of steps 0 .. --ice_steps
    # for a short test (the full ice of the 8 Re mesh is about 200 GB).
    n_ice = year0 + 1
    if args.ice_steps is not None:
        n_ice = min(n_ice, args.ice_steps + 1)
    if windows_march:
        dts = np.diff(years[:year0 + 1])
        say(f"  march: {label}, {year0} steps to year 0 "
            f"({window_info['steps_to_last_year']} to +1000 yr), steps of "
            f"{sorted(set(dts.tolist()))} yr, windows {windows}, years "
            f"{years[0]:.0f} .. {years[year0]:.0f}; ice of steps 0 .. "
            f"{n_ice - 1} in the file")
    else:
        say(f"  march: {year0} steps of {args.step_years} yr, years "
            f"{years[0]:.0f} .. {years[year0]:.0f}; ice of steps 0 .. "
            f"{n_ice - 1} in the file")

    # The ice of every step, made once and kept: the writer and the volume
    # check use it. At L0 one CG1 field is small (schedule C: about 750
    # fields, a few tens of MB per rank). A uniform march reads slice k for
    # step k. Schedules C and D read each protocol slice they need once,
    # keep the two slices around the current year, and write the
    # combination of `common.linear_ice_into` (a copy of the slice at a
    # protocol year).
    if windows_march:
        ice = common.march_ice(forcing.ice, brackets[:n_ice], forcing.cg1)
        # The check of the forward script, here before the write: every
        # year between two protocol years holds the combination of the two
        # protocol slices of the march, bit for bit.
        window_info["linear_ice_check"] = common.check_linear_ice(
            years[:n_ice], ice, slice_years, Function(forcing.cg1))
        say(f"  ice of {label}: "
            f"{window_info['linear_ice_check']['years_checked']} years "
            f"between protocol years, largest difference to the linear "
            f"combination {window_info['linear_ice_check']['max_abs_diff']:.3e}"
            f" (must be 0)")
        if window_info["linear_ice_check"]["max_abs_diff"] != 0.0:
            raise RuntimeError(f"the ice of {label} is not linear between "
                               f"its protocol slices")
    else:
        ice = [forcing.ice(index) for index in range(n_ice)]
    # The protocol step of each year (the ice-history slice), -1 for a year
    # between two protocol years.
    march_steps = [int(forcing.steps[k]) if w == 0.0 else -1
                   for k, w in brackets[:n_ice]]
    surface = ds(common.SURF_RE, domain=mantle)
    ice_check = []
    grid_volumes = {}

    def grid_volume(k):
        """The grid ice volume of slice k, m^3, computed once."""
        if k not in grid_volumes:
            grid_volumes[k] = forcing.grid_ice_volume_m3(k, scales.Re_m)
        return grid_volumes[k]

    for index, f in enumerate(ice):
        mesh_m3 = float(assemble(f * surface)) * D_SCALE**3
        # The volume is linear in the thickness, so the grid volume of the
        # linear ice is the linear combination of the two grid volumes.
        k, w = brackets[index]
        grid_m3 = (grid_volume(k) if w == 0.0
                   else (1.0 - w) * grid_volume(k) + w * grid_volume(k + 1))
        ice_check.append({"step": index, "year": float(years[index]),
                          "protocol_step": march_steps[index],
                          "mesh_volume_m3": mesh_m3,
                          "grid_volume_m3": grid_m3,
                          "relative_error": ((mesh_m3 - grid_m3) / grid_m3
                                             if grid_m3 else 0.0)})
    worst = max(abs(row["relative_error"]) for row in ice_check)
    say(f"  ice: {n_ice} steps, largest relative error of the volume on the "
        f"mesh against the grid {worst:.3e}")
    timings["forcing_s"] = time.time() - tic

    # The input fields of the file: `MANTLE_FIELDS` and the optional fields
    # that the record asks for (`n_elements`, `power_law`).
    names = common.field_names(earth_record)
    file_fields = {name: fields[name] for name in names}
    record = {
        **earth_record,
        "inputs": {"ice_history": data_file_record(forcing.ice_history),
                   "topography": data_file_record(forcing.topography)},
        "march": ({"step_years": args.step_years, "years": years[:n_ice],
                   "protocol_steps": forcing.steps[:n_ice],
                   "ice_steps": n_ice, "year0_step": year0}
                  if not windows_march else
                  common.window_march_record(
                      args.schedule, years[:n_ice], march_steps, n_ice,
                      year0, brackets[:n_ice], **window_info)),
        "solver_constants": solver_constants(experiment, earth, scales,
                                             args.core_density),
        "ice_volume_check": ice_check,
        # The numbers that the inversion computes again after it loads the
        # file (`exp12_common.compare_statistics`).
        "statistics": common.input_statistics(mantle, file_fields, ice),
        "fields": {name: [str(item) for item in
                          (common.MANTLE_FIELDS[name]
                           if name in common.MANTLE_FIELDS
                           else common.OPTIONAL_FIELDS[name][:2])]
                   for name in names},
        "timings": timings,
        "provenance": common.provenance(HERE) if COMM_WORLD.rank == 0 else None,
    }
    record["provenance"] = COMM_WORLD.bcast(record["provenance"], root=0)

    # The input file and its record.
    tic = time.time()
    path = os.path.join(args.output_path, f"{STEM}.h5")
    common.write_inputs(path, parent, file_fields, lambda index: ice[index],
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
