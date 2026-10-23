r"""GIAMIP Exp10: make the input file of the bed inversion.

WHAT THIS SCRIPT DOES
    It makes every field of the inversion that comes from data, once, and
    writes them to one Firedrake `CheckpointFile` (`inputs_exp10.h5`). The
    inversion (`invert_exp10.py`) loads that file and reads no data file, so
    its tape holds only the operations from the control to the misfit.

    1. Read the coarse giamsh mesh, partition it so that every rank holds
       the same number of mantle cells, refine it `--refine` times, curve it
       to P2 and cut the mantle submesh (`exp10_common.build_meshes`). A
       mesh that is refined already and saved as a P1 `CheckpointFile`
       (`.h5`, for example the 8 Re mesh `giamip_8re_r4_linear`) is loaded
       with the same mantle-balanced partition and curved
       (`exp10_common.load_checkpoint_meshes`).
    2. Make the earth fields on the mantle from the earth model of the
       experiment (`giamip.earth`): density, shear and bulk modulus (DG1),
       viscosity (DG0) and gravity (CG2), all non-dimensional, and the two
       DG0 fields of the power law, `exponent` and `transition_stress`
       (see THE POWER LAW below).
    3. Make the forcing on the mantle from the GIAMIP files: the observed sea
       level `SL_obs = -bed_obs / D` (CG2), the frozen surface slope of the
       ocean mask (`surface_slope(SL_obs)`, DG1), and the ice of every step
       of the march from the start to year 0 (CG1, one entry per step).
    4. Write the input file with its JSON record, the same record as a JSON
       file next to it, and ParaView files of the earth fields and of the ice
       at three steps for the reviewer.

    The script tapes nothing: `firedrake.adjoint` is not imported.

THE EXPERIMENT
    GIAMIP protocol v4.0, experiment Exp10, as `giamip` defines it
    (`giamip.mip.inputs.GiamipExperiment("Exp10")`, case file
    `giamip/resources/cases/giamip/exp10.json`):

        ice          GLAC3c profile 2 (`iceHistory-GLAC3c_profile2.nc`)
        topography   Bedmap3 + BedMachine Greenland 5 + GEBCO
                     (`bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc`). The
                     case file leaves the topography open; the file is the
                     one of the G-ADOPT choice `bedmap3`, the file of Exp04
                     (`experiment_topography`). It is the observed
                     present-day bed (year 0); the inversion finds the bed
                     at the start of the run.
        earth        PREM-stable (elastic structure and density) with the
                     viscosity at the midpoint of the GIAMIP suggested
                     range (`giamip-open-range-midpoint`: elastic lid to
                     90 km, 10^20.5 Pa s to 670 km, 1e22 Pa s below),
                     power-law rheology with one Maxwell element,
                     compressible. The protocol leaves all of them open;
                     each is the G-ADOPT choice of the `gadopt` block of the
                     case file. Exp10 is Exp04 with the power law.
        period       122 ka to year 0, in steps of `--step_years` (2000 yr:
                     61 steps), or with `--schedule windows` the steps of
                     schedule C (protocol steps plus the fast melt windows
                     of `exp10_common.SCHEDULE_C_WINDOWS`, design handout
                     8.3; S33, S34), with the ice of every year of the
                     march: the protocol slice at a protocol year, the
                     linear combination of the two neighbouring slices
                     between them (`exp10_common.linear_ice_into`)
        constants    ice 917, ocean water 1025 kg/m^3

    Every number of the experiment comes from `giamip`, never from this file.

THE EARTH MODEL ON THE MESH
    `giamip.earth` gives every property of the earth model at any radius,
    linear in radius between the nodes of the model file. Each DG1 node takes
    the value at its own radius. Gravity is a CG2 field. The viscosity is
    DG0, the value at the cell centroid; an elastic layer ("inf" in the model
    file) takes `ETA_LID` = 1e40 Pa s.

    The mesh `n2_giamip_all_o1` has spheres at 60, 90, 120, 400, 670, 980,
    1260, 1470, 1950 and 2400 km depth, the CMB and the surface. No cell
    crosses a sphere, so a jump of the earth model at a sphere is exact. For
    the midpoint profile the base of the elastic lid at 90 km and the jump
    at 670 km are spheres, so every viscosity boundary is exact and each
    shell holds one viscosity value. The record gives the volume fraction
    of each viscosity value in each shell (`viscosity_by_layer`), which
    shows this.
    PREM-stable has jumps at 400 and 670 km (spheres) and at 15, 24.4, 80,
    220, 600, 771 and 2741 km (not spheres); a cell across one of the
    latter carries the linear interpolant of its nodal values.

THE POWER LAW
    `gadopt` (`InternalVariableApproximation.power_law_factor`, Kang et al.
    2021, Eq. 12) scales the Maxwell time of the element by

        f = (1 + (sigma_b / sigma_T)^(n-1)) / (1 + ((sigma + sigma_b) / sigma_T)^(n-1)),

    with sigma = sqrt(J2) of the deviatoric stress, sigma_T the transition
    stress and sigma_b the background stress. With sigma_b = 0 the
    effective viscosity is eta_0 / (1 + (sigma / sigma_T)^(n-1)): the
    Maxwell viscosity at low stress, lower where the stress of the load is
    large. The values (design handout of 2026-10-20, section 7.2,
    decisions S21 and S27):

        n          3.5 (P13; Kang et al. 2021, upper mantle), exactly 1.0
                   outside the power-law region
        sigma_T    1 MPa (P14), in every cell, also where n = 1:
                   `power_law_factor` divides by it before it chooses the
                   linear branch, so a zero would give NaN
        sigma_b    0 (P15), a plain number in the record
        region     90 km < depth < 670 km and eta_0 < 1e22 Pa s, tested at
                   the cell centroid (S21). For the midpoint profile this is
                   the whole upper mantle below the elastic lid, three
                   shells of the mesh (90-120, 120-400, 400-670 km). The
                   lid and the lower mantle are linear.

    The exponent is written as exactly 1.0 or exactly 3.5, so the test of
    `power_law_factor` for the linear branch, `n - 1 == 0`, holds bit for
    bit; `exp10_common.read_inputs` snaps the loaded values again. The
    record entry `power_law` makes the common file read the two fields and
    build the solver with the power law (`exp10_common.power_law_settings`).
    The record gives the volume fraction of each exponent value in each
    shell (`exponent_by_layer`); each shell must hold one value.

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
    srun python3 prepare_exp10.py --mesh <file.msh> --output_path <dir>
    The number of ranks must be the one of the inversion job.

OUTPUT (in --output_path)
    inputs_exp10.h5          the input file of the inversion
    inputs_exp10.json        its record (the same JSON as inside the file)
    inputs_exp10_earth.pvd   ParaView: the earth fields in SI units
    inputs_exp10_ice.pvd     ParaView: the ice at the first, middle and last
                             step of the march, m
    prepare_exp10.log        the log
"""

import argparse
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
import exp10_common as common  # noqa: E402
from gadopt import (COMM_WORLD, Function, FunctionSpace,  # noqa: E402
                    SpatialCoordinate, VectorFunctionSpace, VTKFile, assemble,
                    dot, ds, dx, sqrt)
from gadopt.sea_level_masks import surface_slope  # noqa: E402
from mpi4py import MPI  # noqa: E402

import giamip  # noqa: E402
from giamip import cases, grids  # noqa: E402
from giamip.mip.inputs import GiamipExperiment, Topography  # noqa: E402
from giamsh.radial import RadialStructure  # noqa: E402

#: The experiment of this directory.
EXPERIMENT = "Exp10"
#: The file stem of the outputs.
STEM = "inputs_exp10"
#: The MD5 of the shared coarse mesh `n2_giamip_all_o1.msh` (handout section
#: 3). The run stops if the mesh file differs.
MESH_MD5 = "a4da84206dc82f5b458be66e4fd33844"

D_SCALE = common.D_SCALE

#: The rheology of the `gadopt` block of the case file that this script
#: builds. Any other value stops the run.
RHEOLOGY = "power-law"
#: The power-law exponent n of the power-law cells (P13). The other cells
#: get exactly 1.0.
POWER_LAW_EXPONENT = 3.5
#: The transition stress sigma_T, Pa (P14), in every cell.
TRANSITION_STRESS_PA = 1.0e6
#: The background stress sigma_b, Pa (P15). Zero: the low-stress limit of
#: the power law is the Maxwell twin (Exp04).
BACKGROUND_STRESS_PA = 0.0
#: The power-law region (S21): strictly between these depths, m, ...
POWER_LAW_DEPTH_RANGE_M = (90.0e3, 670.0e3)
#: ... and where the linear viscosity eta_0 is strictly below this, Pa s.
POWER_LAW_ETA_CAP_PA_S = 1.0e22


def experiment_topography(experiment):
    """The topography file of the experiment.

    The case file of Exp10 leaves the topography open, so
    `GiamipExperiment.topography` is None. `giamip.cases.with_gadopt_choices`
    fills in the file of the G-ADOPT choice (`bedmap3`), the file that the
    protocol prescribes for Exp04. For a case that prescribes a file this is
    the prescribed file.
    """
    relative_path = cases.with_gadopt_choices(experiment.id)["topography"]["file"]
    return Topography(relative_path)


def power_law_fields(mantle, eta_pa_s, scales):
    """The DG0 exponent and transition stress of the power law (S21).

    The rule is tested at the cell centroid, as the viscosity is: a cell is
    a power-law cell when its centroid depth is strictly between 90 and
    670 km and its viscosity eta_0 is strictly below 1e22 Pa s. No cell of
    the mesh crosses the spheres at 90 and 670 km, so the rule selects whole
    shells. The values are written into the data array, not interpolated,
    so they are exactly 1.0 and exactly `POWER_LAW_EXPONENT`.

    Args:
      mantle: the mantle mesh.
      eta_pa_s: the viscosity of every DG0 node with halos, Pa s (the lid
        has `ETA_LID`), in the node order of the DG0 space of `mantle`.
      scales: the `exp10_common.Scales`.

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

    Args:
      mantle: the mantle mesh (P2 coordinates).
      earth: the `giamip.earth.RadialEarth` of the experiment.
      scales: the `exp10_common.Scales`.

    Returns:
      A dictionary with `density`, `shear_modulus`, `bulk_modulus` (DG1),
      `viscosity` (DG0), `gravity` (CG2), `exponent` and
      `transition_stress` (DG0, `power_law_fields`), all non-dimensional.
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

    # The power law, from the viscosity in Pa s of the same cells.
    exponent, transition = power_law_fields(mantle, eta_pa_s, scales)

    # Gravity is continuous; the side does not matter.
    cg2 = FunctionSpace(mantle, "CG", 2)
    r_cg2 = np.clip(node_radius_m(cg2), scales.Rc_m, scales.Re_m)
    g = Function(cg2, name="gravity")
    g.dat.data_with_halos[:] = earth.elastic(r_cg2)["g_m_s2"] / scales.g_bar
    return {"density": rho, "shear_modulus": mu, "bulk_modulus": kappa,
            "viscosity": eta, "gravity": g, "exponent": exponent,
            "transition_stress": transition}


def core_density(elastic, scales, rule):
    """The density of the homogeneous fluid core, kg/m^3.

    Args:
      elastic: the `giamip.earth.ElasticProfile`.
      scales: the `exp10_common.Scales`.
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
      scales: the `exp10_common.Scales`.
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
    of `step_years`, and the last year. With the Exp10 years (500 yr steps
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
    ice is CG1 (see `exp10_common.ICE_SPACE`). Every node of the mantle mesh
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
        # the G-ADOPT choice (`experiment_topography`).
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
    p.add_argument("--schedule", choices=["uniform", "windows"],
                   default="uniform",
                   help="the march of the input file: uniform steps of "
                        "--step_years (default), or schedule C (windows: "
                        "the protocol steps plus the 25 yr and 10 yr steps "
                        "of the fast melt windows, decisions S33 and S34), "
                        "with the ice of every year of schedule C, linear in "
                        "time between the protocol years; --step_years is "
                        "then not used")
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
    say = common.RunLog(os.path.join(args.output_path, "prepare_exp10.log"))
    say("=" * 78)
    say(f"GIAMIP {EXPERIMENT}: input file of the bed inversion, "
        f"{COMM_WORLD.size} ranks")
    say("=" * 78)
    timings = {}

    # The experiment and its earth model, with the G-ADOPT choices for the
    # fields that the protocol leaves open.
    experiment = GiamipExperiment(EXPERIMENT)
    earth = experiment.earth()
    # This script writes one Maxwell element per cell with the power law.
    # Stop for any other rheology, so that the input file cannot claim a
    # rheology that its fields do not hold.
    if earth.rheology != RHEOLOGY:
        raise RuntimeError(f"{EXPERIMENT} has rheology {earth.rheology!r}; "
                           f"this script builds the {RHEOLOGY!r} solver only")
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

    # The earth fields.
    tic = time.time()
    fields = earth_fields(mantle, earth, scales)
    timings["earth_s"] = time.time() - tic

    # The forcing: observed sea level, its frozen slope, and the ice.
    tic = time.time()
    # Schedule C reads every protocol year: `step_years = 1` keeps them all
    # in `coarse_step_indices` (every distance from the start is a whole
    # number of years), as in the forward script.
    windows_march = args.schedule == "windows"
    forcing = SurfaceForcing(mantle, experiment,
                             1 if windows_march else args.step_years)
    fields["SL_obs"] = forcing.SL_obs
    # The surface slope of the ocean mask, from the observed bed. It stays
    # fixed in the inversion: the mask width does not depend on the control.
    fields["slope"] = surface_slope(forcing.SL_obs)
    fields["slope"].rename("slope")
    # The years of the forcing slices: the march years of a uniform march,
    # every protocol year for schedule C.
    slice_years = np.asarray(forcing.years, dtype=float)
    schedule_c = None
    if windows_march:
        # Schedule C (design handout 8.3, S33, S34): the years of
        # `common.schedule_c_years`, the function of the forward script of
        # schedule C, so the inversion on this file and the forward run
        # march the same years. The ice of a year between two protocol
        # years is linear in time between the two slices
        # (`common.ice_bracket`, `common.linear_ice_into`).
        ice_file = os.path.basename(str(forcing.ice_history.path))
        windows = common.schedule_c_windows(ice_file)
        all_years = common.schedule_c_years(int(slice_years[0]), slice_years,
                                            windows)
        expected = (common.SCHEDULE_C_STEPS_122KA.get(ice_file)
                    if int(slice_years[0]) == -122000 else None)
        if expected is not None and len(all_years) - 1 != expected:
            raise RuntimeError(f"schedule C has {len(all_years) - 1} steps "
                               f"for {ice_file}, expected {expected}")
        years = all_years.astype(float)
        brackets = [common.ice_bracket(slice_years, y) for y in years]
        schedule_c = {"ice_history": ice_file, "windows": windows,
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
        say(f"  march: schedule C, {year0} steps to year 0 "
            f"({schedule_c['steps_to_last_year']} to +1000 yr), steps of "
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
    # step k. Schedule C reads each protocol slice once, keeps the two
    # slices around the current year, and writes the combination of
    # `common.linear_ice_into` (a copy of the slice at a protocol year).
    if windows_march:
        ice = common.march_ice(forcing.ice, brackets[:n_ice], forcing.cg1)
        # The check of the forward script, here before the write: every
        # year between two protocol years holds the combination of the two
        # protocol slices of the march, bit for bit.
        schedule_c["linear_ice_check"] = common.check_linear_ice(
            years[:n_ice], ice, slice_years, Function(forcing.cg1))
        say(f"  ice of schedule C: "
            f"{schedule_c['linear_ice_check']['years_checked']} years "
            f"between protocol years, largest difference to the linear "
            f"combination {schedule_c['linear_ice_check']['max_abs_diff']:.3e}"
            f" (must be 0)")
        if schedule_c["linear_ice_check"]["max_abs_diff"] != 0.0:
            raise RuntimeError("the ice of schedule C is not linear between "
                               "its protocol slices")
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
          "transition_stress": ("Pa", common.MU_BAR, "transition_stress_pa")}
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
    # The exponent per shell. No cell crosses a sphere of the mesh, so each
    # shell must hold one exponent value; a shell with two values means the
    # rule did not select whole shells, and the run stops.
    exponent_by_layer = cells.distinct_values(fields["exponent"], 1.0)
    power_law_volume = {}
    for name, _, _ in layers:
        rows = exponent_by_layer[name]
        if isinstance(rows, str) or len(rows) != 1:
            raise RuntimeError(f"shell {name} holds more than one exponent "
                               f"value: {rows}")
        power_law_volume[name] = float(rows[0]["value"] == POWER_LAW_EXPONENT)
        say(f"    exponent {name}: {rows[0]['value']:g}")
    if not any(power_law_volume.values()):
        raise RuntimeError("no mantle cell has the power law")
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
        "march": ({"step_years": args.step_years, "years": years[:n_ice],
                   "protocol_steps": forcing.steps[:n_ice],
                   "ice_steps": n_ice, "year0_step": year0}
                  if not windows_march else
                  # Schedule C: `protocol_steps` is -1 at a year between two
                  # protocol years; `ice_brackets` holds (slice, weight of the
                  # next slice) of every year.
                  {"schedule": "windows", "step_years": None,
                   "years": years[:n_ice], "protocol_steps": march_steps,
                   "ice_steps": n_ice, "year0_step": year0,
                   "ice_brackets": [[int(k), float(w)]
                                    for k, w in brackets[:n_ice]],
                   **schedule_c}),
        "scales": scales.as_dict(),
        "solver_constants": solver_constants(experiment, earth, scales,
                                             args.core_density),
        "earth_by_layer": layer_stats,
        "viscosity_by_layer": viscosity_by_layer,
        "exponent_by_layer": exponent_by_layer,
        # The power law (design handout of 2026-10-20, section 7.2). The
        # common file reads `exponent` and `background_stress` (units of
        # mu_bar); the other keys are the record for the reader.
        "power_law": {
            "exponent": POWER_LAW_EXPONENT,
            "background_stress": BACKGROUND_STRESS_PA / common.MU_BAR,
            "background_stress_MPa": BACKGROUND_STRESS_PA / 1e6,
            "transition_stress_MPa": TRANSITION_STRESS_PA / 1e6,
            "transition_stress_nondim": TRANSITION_STRESS_PA / common.MU_BAR,
            "stress_measure": "sqrt(J2) of the deviatoric stress",
            "form": "Kang et al. (2021) Eq. 12, gadopt power_law_factor",
            "rule": {"depth_above_km": POWER_LAW_DEPTH_RANGE_M[0] / 1e3,
                     "depth_below_km": POWER_LAW_DEPTH_RANGE_M[1] / 1e3,
                     "eta_below_pa_s": POWER_LAW_ETA_CAP_PA_S,
                     "tested_at": "cell centroid",
                     "elsewhere": "exponent exactly 1.0"},
            "power_law_volume_fraction_by_layer": power_law_volume,
            "decisions": "S21, S27 (P13 n, P14 sigma_T, P15 sigma_b, "
                         "P19 sqrt(J2)), design handout 2026-10-20",
        },
        "ice_volume_check": ice_check,
        # The numbers that the inversion computes again after it loads the
        # file (`exp10_common.compare_statistics`).
        "statistics": common.input_statistics(mantle, fields, ice),
        "fields": None,
        "timings": timings,
        "provenance": common.provenance(HERE) if COMM_WORLD.rank == 0 else None,
    }
    record["provenance"] = COMM_WORLD.bcast(record["provenance"], root=0)
    # The fields of the file: `MANTLE_FIELDS` and the power-law fields,
    # which the record asks for through its `power_law` entry.
    record["fields"] = {
        name: list(common.MANTLE_FIELDS[name] if name in common.MANTLE_FIELDS
                   else common.OPTIONAL_FIELDS[name][:2])
        for name in common.field_names(record)}
    # Every field that this script made must be a field of the file, and
    # the other way round, so that no power-law field is left out of it.
    if set(record["fields"]) != set(fields):
        raise RuntimeError(f"the record names the fields "
                           f"{sorted(record['fields'])}, the script made "
                           f"{sorted(fields)}")

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
