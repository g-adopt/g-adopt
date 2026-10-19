r"""GIAMIP experiment Exp01 with the coupled self-gravitating sea-level solver.

THE EXPERIMENT
    GIAMIP protocol v4.0 (2026-10-05), experiment Exp01, as `giamip` 0.2.0
    defines it (`giamip.mip.inputs.GiamipExperiment("Exp01")`):

        ice          ICE-6G_D, 257 x 513 Gauss-Legendre grid, 463 steps
        topography   Bedmap2 + BedMachine Greenland 5 + GEBCO. The file is
                     the observed present-day bed (2000 CE), although the
                     archive calls it "initial-topography". This driver
                     starts the run from it, which is the first iterate of
                     the topography iteration: the bed at 122 ka must be
                     found so that the run ends at this bed in year 0
                     (NOTES/EXP01-HANDOVER.md, section 17).
        earth        PREM-stable (elastic structure and density) with the
                     VM5a viscosity, Maxwell rheology, compressible (the
                     G-ADOPT choice for the open field)
        period       122 ka to 1000 yr after 2000 CE: 500 yr steps to 26 ka,
                     then 100 yr steps
        constants    ice 917, ocean water 1025 kg/m^3, Re 6371 km
        frame        centre of mass of the whole Earth system (CM)

    Every number of the experiment comes from `giamip`, never from this file.

WHAT THE RUN SOLVES
    One monolithic system per time step: the compressible viscoelastic
    mantle (P2 displacement, DG1 internal variables, one Maxwell element),
    the perturbed gravitational potential on the whole mesh (P2), the fluid
    core on the core-mantle boundary, the polar motion and the rotation-rate
    change (three `Real` rows), the centre-of-mass frame (three `Real`
    rows), and the sea-level equation with a moving coastline and floating
    ice (`Shift`, one `Real` row). With the core pressure the `Real` block
    has 8 rows. The time stepping is backward Euler. The coastline moves, so
    each step runs Newton.

    The sea level is

        SL    = SL_init + (N - N_init) - (u_r - ur_init) + Shift
        sigma = rho_w B C (SL - SL_init) + rho_i (1 - B) I - rho_i (1 - B0) I_init

    in the form that `gadopt.SeaLevel` documents, with `N = (psi + psi_rot) /
    g_s` the geoid with the rotational potential, `SL_init = -bed` and the
    ice of 122 ka as `I_init`. The run starts from the undeformed state with
    that ice on it, so the load is zero at the start and there is no elastic
    step at 122 ka.

SCALES
    length       D = 2891 km, the mantle thickness of PREM-stable (also the
                 unit of the giamsh mesh)
    gravity      g_bar = g(Re) of PREM-stable, so g_s = 1 in the scaled
                 equations
    density      rho_bar = 3 g_bar / (4 pi G Re), the mean density that gives
                 that surface gravity with G = 6.6743e-11 (the G of the
                 PREM-stable gravity column)
    stress       mu_bar = 1e11 Pa
    viscosity    eta_bar = 1e21 Pa s, time t_bar = eta_bar / mu_bar = 1e10 s

    Lambda = 4 pi G rho_bar D / g_bar = 3 D / Re, so the scaled Poisson
    equation and the scaled gravity of the file describe the same mass.

THE EARTH MODEL ON THE MESH
    `giamip.earth` gives every property of PREM-stable and VM5a at any
    radius, linear in radius between the nodes of the file. Density, shear
    modulus and bulk modulus are DG1 fields: each DG1 node (an interior
    Gauss-Legendre point of its cell) takes the value at its own radius. The
    jumps at 400 km, 670 km and the CMB are mesh spheres, so no cell crosses
    them and each side keeps its own values.
    The jumps at 15 and 24.4 km (density and moduli) and at 220 km (moduli
    only) are not mesh spheres; a cell across them carries the linear
    interpolant of the nodal values. Gravity is a CG2 field (it is continuous
    and the prestress terms need its gradient). The viscosity is DG0 at the
    cell centroid. The elastic lid of VM5a (viscosity "inf") takes 1e40 Pa s,
    a Maxwell time of about 2e19 t_bar.

    The core is a `gadopt.gia_gravity.FluidCore`: inviscid, homogeneous and
    incompressible, with one density. `--core_density mean` (the default)
    takes the mean density of the PREM-stable core, 10 986.5 kg/m^3, which
    gives the core its mass. `cmb` takes 9903 kg/m^3, the value just below
    the CMB, which makes the core 9.9 percent light.

ROTATION
    `rotation_moments = {"C": C, "k_s": k_s}` with `surface_radius = Re`. The
    solver computes C - A = Q k_s, with Q = Omega^2 Re^5 / (3 G). k_s must be
    the fluid Love number of the lovejx run that this run is compared with
    (`--k_s`, default 0.9422, the legacy lovejx value 0.9342 + 0.008). C is
    the observed polar moment of lovejx, 0.3307007 M Re^2.

MESH
    The coarse giamsh mesh of Exp01 (`n2_giamip_exp01_o1.msh` and its JSON
    sidecar). Rank 0 reads it, `mantle_balanced_partition` gives every rank
    the same number of mantle cells, `giamsh.refine.refine_uniform` refines
    it `--refine` times (0 for L0), and the result is curved to P2. The
    mantle submesh comes from `giamsh.refine.region_submesh`.

RUN
    mpiexec -np <N> python3 exp01.py --mesh <file.msh> --output_path <dir>
        [--steps 1] [--restart <checkpoint.h5>]

    `--step_years 2000` keeps only the model years at a uniform 2000 yr
    spacing from the start (and the last year), for the L0 runs of the
    topography iteration; a restart must use the same value.

    `--steps n` stops after n forcing steps. A run writes a restart
    checkpoint every `--checkpoint_every` steps and after its last step, one
    file per checkpoint, and keeps the newest two. `--wall_limit_hours`
    stops a job early, with a checkpoint, before the time limit. A restart
    needs the same number of ranks; `--restart
    <dir>/checkpoint_exp01_latest.h5` continues from the newest checkpoint.

OUTPUT (in --output_path); no VTK files, the files load on any number of
ranks for the grid values and ParaView files after the run
    summary_exp01.json      the settings, scales, mesh record, inputs and
                            software versions, and the run state
    params_exp01.log        one line per step (the rows are also in the
                            summary at the end of a job)
    surface_from<s>.h5      every step: displacement, potential, its radial
                            derivative and the ice on the surface Re, with
                            step, year, shift and m of each entry
                            (`FieldOutput`); <s> is the first step of the job
    mantle_from<s>.h5       every --mantle_every steps: displacement and
                            potential on the mantle; once: the earth model
    checkpoint_exp01_step<n>.h5, checkpoint_exp01_latest.h5
                            the restart checkpoints (Firedrake CheckpointFile)
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# `gadopt` imports `firedrake.pyplot`, which imports matplotlib's font
# manager. Without a font cache every rank scans the fonts and tries to write
# the cache under a lock; on 16 384 ranks this held a job for many minutes.
# The job environment names a cache directory that was built once on a
# login node with the same container (`EXP01_MPLCONFIGDIR`); every rank then
# only reads it. This must happen before `gadopt` is imported.
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
# The benchmark helpers (Hilbert keys, checkpoint file, JSON output).
sys.path.insert(0, os.path.join(HERE, "..", "gia_selfgrav_benchmarks"))

import numpy as np  # noqa: E402

# gadopt before firedrake: gadopt re-exports the Firedrake names.
from gadopt import (COMM_WORLD, CompressibleInternalVariableApproximation,  # noqa: E402
                    Constant, Function, FunctionSpace, Mesh, ParameterLog,
                    SpatialCoordinate, SphericalDtN, TrialFunction,
                    VectorFunctionSpace, assemble, dot, grad, interpolate, log,
                    max_value, sqrt)
from gadopt.gia_gravity import (FluidCore, SeaLevel,  # noqa: E402
                                SelfGravitatingGIASolver,
                                rigid_rotation_nullspace,
                                self_gravitating_gia_space,
                                selfgrav_dtn_iterative_solver_parameters)
from gadopt.sea_level_masks import (DEFAULT_ALPHA_MASK,  # noqa: E402
                                    DEFAULT_GRAD_FLOOR, grounded_ice_function,
                                    ocean_function, surface_slope)
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import giamip  # noqa: E402
from giamip import grids  # noqa: E402
from giamip.mip.inputs import GiamipExperiment  # noqa: E402
from giamsh.radial import RadialStructure  # noqa: E402
from giamsh import refine as giamsh_refine  # noqa: E402

import selfgrav_common as common  # noqa: E402

# --------------------------------------------------------------------------
# Constants that are not in the experiment definition
# --------------------------------------------------------------------------

#: Newton's constant of the PREM-stable gravity column
#: (`prem-stable/build_prem_stable.py`, CODATA 2018), m^3 kg^-1 s^-2. The
#: scales use it, so the scaled Poisson equation and the gravity of the file
#: belong to the same mass.
G_NEWTON = 6.6743e-11
#: The length scale D, the mantle thickness of PREM-stable, m. The giamsh
#: mesh is in this unit.
D_SCALE = 2.891e6
#: The stress scale, Pa.
MU_BAR = 1.0e11
#: The viscosity scale, Pa s.
ETA_BAR = 1.0e21
#: The time scale eta_bar / mu_bar = 1e10 s in Julian years.
T_BAR_YR = ETA_BAR / MU_BAR / (365.25 * 86400.0)
#: The viscosity of the elastic lid of VM5a, Pa s. "inf" in the file. 1e40
#: Pa s gives a Maxwell time of about 2e19 t_bar, far beyond the run.
ETA_LID = 1.0e40
#: The rotation rate of the Earth, rad/s (the value of the Spada benchmark
#: and of lovejx).
OMEGA = 7.292115e-5
#: The observed polar moment of inertia over M Re^2, as lovejx uses it
#: (`lovejx/sealevel/constants.py`).
C_OVER_MRE2 = 0.3307007

#: Facet tags of the giamsh layout that the solver needs: the surface Re, the
#: CMB, and the two DtN spheres.
SURF_RE, SURF_RC, SURF_OUTER, SURF_INNER = 2, 3, 4, 5
#: The degree of `SphericalDtN` on the two truncation spheres (as the
#: benchmarks).
DTN_DEGREE = 5
#: The tolerances of the outer FGMRES and of block 0 (as the benchmarks).
OUTER_RTOL = 1.0e-6
BLOCK0_RTOL = 1.0e-4


# --------------------------------------------------------------------------
# Scales and the earth model
# --------------------------------------------------------------------------


class Scales:
    """The non-dimensional scales of the run, from PREM-stable.

    Attributes:
      g_bar: g at the surface, m/s^2.
      rho_bar: the mean density that gives `g_bar`, kg/m^3.
      Re, Rc: the surface and CMB radii, non-dimensional.
      B_mu: rho_bar g_bar D / mu_bar.
      Lambda: 4 pi G rho_bar D / g_bar.
      Omega_sq: Omega^2 D / g_bar.
      earth_mass_kg: the mass that gives `g_bar`, kg.
    """

    def __init__(self, elastic):
        """Build the scales from a `giamip.earth.ElasticProfile`."""
        self.Re_m = elastic.surface_radius_m
        self.Rc_m = elastic.core_mantle_boundary_m
        # Gravity just below the surface; the surface is the last node.
        self.g_bar = float(elastic(self.Re_m, "below")["g_m_s2"])
        self.earth_mass_kg = self.g_bar * self.Re_m**2 / G_NEWTON
        self.rho_bar = self.earth_mass_kg / (4.0 / 3.0 * np.pi * self.Re_m**3)
        self.Re = self.Re_m / D_SCALE
        self.Rc = self.Rc_m / D_SCALE
        self.B_mu = self.rho_bar * self.g_bar * D_SCALE / MU_BAR
        self.Lambda = 4 * np.pi * G_NEWTON * self.rho_bar * D_SCALE / self.g_bar
        self.Omega_sq = OMEGA**2 * D_SCALE / self.g_bar

    def as_dict(self):
        """The scales as plain numbers, for the summary."""
        return {"D_m": D_SCALE, "g_bar": self.g_bar, "rho_bar": self.rho_bar,
                "mu_bar": MU_BAR, "eta_bar": ETA_BAR, "t_bar_yr": T_BAR_YR,
                "G": G_NEWTON, "Re": self.Re, "Rc": self.Rc,
                "B_mu": self.B_mu, "Lambda": self.Lambda,
                "Omega_sq": self.Omega_sq,
                "earth_mass_kg": self.earth_mass_kg}


def node_radius_m(space):
    """The radius of every node of a scalar space, in metres, with halos."""
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)
    r = Function(space).interpolate(sqrt(dot(X, X)))
    return np.array(r.dat.data_ro_with_halos) * D_SCALE


def earth_fields(mantle, earth, scales):
    """The density, the moduli, the viscosity and gravity on the mantle mesh.

    Each DG1 node takes the value of PREM-stable at its own radius. The DG1
    element of Firedrake on a tetrahedron has its four nodes at interior
    Gauss-Legendre points (`finat.spectral.GaussLegendre`), not at the
    vertices, so the field is the linear interpolant through four interior
    values, and no node lies on a mesh sphere. The side rule below is kept
    for an element with nodes on the cell boundary: a node on a sphere where
    a property jumps takes the value of the side where its cell lies,
    "above" when the node is below the cell centroid and "below" when it is
    above it. A DG node belongs to one cell only, so this is defined for
    every node.

    Args:
      mantle: the mantle mesh (P2 coordinates).
      earth: the `giamip.earth.RadialEarth` of the experiment.
      scales: the `Scales`.

    Returns:
      A dictionary with `density`, `shear_modulus`, `bulk_modulus` (DG1),
      `viscosity` (DG0) and `g` (CG2), all non-dimensional.
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
    mu = dg1_field("mu_Pa", MU_BAR, "shear_modulus")
    kappa = dg1_field("kappa_Pa", MU_BAR, "bulk_modulus")

    # Viscosity at the centroid, which lies inside one VM5a layer: the layer
    # boundaries at 60, 100, 670 and 1260 km are mesh spheres, and the one at
    # 420 km separates two equal values.
    r_centroid = np.clip(np.array(centroid.dat.data_ro_with_halos) * D_SCALE,
                         scales.Rc_m, scales.Re_m)
    eta_pa_s = earth.viscosity(r_centroid)
    eta_pa_s = np.where(np.isinf(eta_pa_s), ETA_LID, eta_pa_s)
    if np.isnan(eta_pa_s).any():
        raise RuntimeError("a mantle cell has no VM5a viscosity")
    eta = Function(dg0, name="viscosity")
    eta.dat.data_with_halos[:] = eta_pa_s / ETA_BAR

    # Gravity is continuous; the side does not matter.
    cg2 = FunctionSpace(mantle, "CG", 2)
    r_cg2 = np.clip(node_radius_m(cg2), scales.Rc_m, scales.Re_m)
    g = Function(cg2, name="gravity")
    g.dat.data_with_halos[:] = earth.elastic(r_cg2)["g_m_s2"] / scales.g_bar
    return {"density": rho, "shear_modulus": mu, "bulk_modulus": kappa,
            "viscosity": eta, "g": g}


def field_range(f):
    """(min, max) of a `Function` over all ranks."""
    data = f.dat.data_ro
    lo = data.min() if data.size else np.inf
    hi = data.max() if data.size else -np.inf
    return (COMM_WORLD.allreduce(float(lo), op=MPI.MIN),
            COMM_WORLD.allreduce(float(hi), op=MPI.MAX))


def core_density(elastic, scales, rule):
    """The density of the homogeneous `FluidCore`, kg/m^3.

    Args:
      elastic: the `giamip.earth.ElasticProfile`.
      scales: the `Scales`.
      rule: `"mean"`, the mean density of the core from its mass
        g(Rc) Rc^2 / G, or `"cmb"`, the density just below the CMB.
    """
    if rule == "cmb":
        return float(elastic(scales.Rc_m, "below")["rho_kg_m3"])
    g_cmb = float(elastic(scales.Rc_m, "below")["g_m_s2"])
    mass = g_cmb * scales.Rc_m**2 / G_NEWTON
    return mass / (4.0 / 3.0 * np.pi * scales.Rc_m**3)


# --------------------------------------------------------------------------
# The mesh
# --------------------------------------------------------------------------


def mantle_balanced_partition(plex, nparts, mantle_tags):
    """The partition of `selfgrav_common.balanced_partition` for several tags.

    The mantle of the giamsh mesh is six cell tags (one per layer). Each cell
    is ordered by the Hilbert key of the direction of its centroid. The
    mantle cells, in that order, are split into `nparts` pieces of equal
    count, and every other cell goes to the rank whose mantle piece covers
    its key. Every rank then owns a solid-angle sector through all shells,
    and the mantle cell counts differ by at most one.

    Collective. Only rank 0 holds cells, because the gmsh reader loads the
    whole file there.

    Args:
      plex: the serial `DMPlex` of the gmsh file, on `COMM_WORLD`.
      nparts: the number of ranks.
      mantle_tags: the cell tags of the mantle region.

    Returns:
      `(sizes, points)` for `distribution_parameters["partition"]`.
    """
    cstart, cend = plex.getHeightStratum(0)
    if cend == cstart:
        return (np.zeros(nparts, dtype=PETSc.IntType),
                np.zeros(0, dtype=PETSc.IntType))
    section = plex.getCoordinateSection()
    coordinates = plex.getCoordinatesLocal()
    centroid = np.array([plex.vecGetClosure(section, coordinates, c)
                         .reshape(-1, 3).mean(axis=0)
                         for c in range(cstart, cend)])
    direction = centroid / np.linalg.norm(centroid, axis=1)[:, None]
    scale = (1 << common.HILBERT_BITS) - 1
    key = common.hilbert_keys(np.rint((direction + 1.0) * 0.5 * scale),
                              common.HILBERT_BITS)
    mantle = np.zeros(cend - cstart, dtype=bool)
    for tag in mantle_tags:
        points = plex.getStratumIS("Cell Sets", tag).getIndices()
        mantle[np.asarray(points) - cstart] = True
    part = np.empty(cend - cstart, dtype=np.int64)
    mantle_index = np.flatnonzero(mantle)
    order = mantle_index[np.argsort(key[mantle_index], kind="stable")]
    part[order] = (np.arange(len(order)) * nparts) // len(order)
    first = np.searchsorted(part[order], np.arange(nparts), side="left")
    starts = key[order][first]
    other = np.flatnonzero(~mantle)
    part[other] = np.clip(
        np.searchsorted(starts, key[other], side="right") - 1, 0, nparts - 1)
    sizes = np.bincount(part, minlength=nparts).astype(PETSc.IntType)
    points = (np.argsort(part, kind="stable") + cstart).astype(PETSc.IntType)
    return sizes, points


def read_radial(sidecar):
    """The `RadialStructure` and the record of the mesh sidecar file."""
    with open(sidecar) as handle:
        record = json.load(handle)
    return RadialStructure.from_dict(record["radial"]), record


def build_meshes(msh, radial, n_levels):
    """Read, partition, refine and curve the parent mesh; cut the mantle.

    Args:
      msh: the coarse gmsh file.
      radial: the `RadialStructure` of its sidecar.
      n_levels: uniform refinement levels (0 for L0).

    Returns:
      `(parent, mantle, info)`.
    """
    mantle_tags = list(radial.region_tags("mantle"))
    plex = PETSc.DMPlex().createFromFile(msh, comm=COMM_WORLD)
    plex.setName("exp01_topology")
    sizes, points = mantle_balanced_partition(plex, COMM_WORLD.size,
                                              mantle_tags)
    # The distribution needs a name of its own; without one,
    # `CheckpointFile.save_mesh` hangs in `DMPlex.topologyView`
    # (`selfgrav_common.build_meshes`).
    linear = Mesh(plex, distribution_parameters={"partition": (sizes, points)},
                  distribution_name=f"mantle_balanced_{COMM_WORLD.size}",
                  name="exp01_linear")
    # The sag check of giamsh raises on this coarse mesh (cmb, buffer_2 and
    # outer), and the refinement of the mesh campaign ran without it.
    linear = giamsh_refine.refine_uniform(linear, radial, n_levels,
                                          curve=False, check=False)
    parent = giamsh_refine.curve_mesh(linear, name="exp01_parent")
    # `refine_uniform` refines with Firedrake's `MeshHierarchy`, which marks
    # the refined topology with multigrid level information. The mantle
    # submesh has none, and a mixed space over the parent and the mantle then
    # fails with "Found inconsistent component meshes". The solver uses
    # GAMG, not geometric multigrid, so the level information is removed.
    # It also holds the coarser meshes of the hierarchy, whose memory is then
    # released.
    for mesh in (linear, parent):
        if hasattr(mesh.topological, "__level_info__"):
            delattr(mesh.topological, "__level_info__")
    mantle = giamsh_refine.curve_mesh(
        _mantle_submesh(parent, mantle_tags), name="exp01_mantle")
    info = {"file": os.path.abspath(msh), "md5": common.md5sum(msh)
            if COMM_WORLD.rank == 0 else None, "refine_levels": n_levels}
    info["md5"] = COMM_WORLD.bcast(info["md5"], root=0)
    info.update(mesh_counts(parent, mantle))
    info["folded_cells_parent"] = giamsh_refine.folded_cells(parent)
    info["folded_cells_mantle"] = giamsh_refine.folded_cells(mantle)
    if info["folded_cells_parent"] or info["folded_cells_mantle"]:
        raise RuntimeError(f"folded cells after the curving: {info}")
    return parent, mantle, info


#: The DMPlex label that marks every mantle cell with the value 1.
MANTLE_LABEL = "exp01_mantle_cells"


def _mantle_submesh(parent, mantle_tags):
    """The `Submesh` of the mantle cells (P1 coordinates).

    The mantle is six cell tags, one per layer. `Submesh(parent, 3, tags)`
    with a list merges the cells of each tag with `IS.union`, and on a rank
    that holds cells of one tag and none of a later tag that merge works on
    an empty index set without a communicator and stops with "Invalid
    communicator" in `ISSorted` (Firedrake `dmcommon.submesh_create`). The
    thinnest layer has 2 533 coarse cells, so from about 2 000 ranks some
    ranks hold none of it. All mantle cells therefore go into one label of
    their own, and `Submesh` takes that one value: a single value needs no
    merge, and every rank holds mantle cells. Halo cells are labelled too,
    as `Cell Sets` labels them.
    """
    from firedrake import Submesh
    dm = parent.topology_dm
    if not dm.hasLabel(MANTLE_LABEL):
        dm.createLabel(MANTLE_LABEL)
    label = dm.getLabel(MANTLE_LABEL)
    for tag in mantle_tags:
        if dm.getStratumSize("Cell Sets", tag) > 0:
            for point in dm.getStratumIS("Cell Sets", tag).getIndices():
                label.setValue(point, 1)
    return Submesh(parent, 3, 1, label_name=MANTLE_LABEL)


def mesh_counts(parent, mantle):
    """Global cell counts and the spread of the mantle cells over the ranks."""
    def owned(mesh):
        return mesh.cell_set.size

    local_mantle = owned(mantle)
    counts = COMM_WORLD.allgather(local_mantle)
    return {"cells_parent": COMM_WORLD.allreduce(owned(parent), op=MPI.SUM),
            "cells_mantle": int(sum(counts)),
            "mantle_cells_per_rank_min": int(min(counts)),
            "mantle_cells_per_rank_max": int(max(counts)),
            "ranks": COMM_WORLD.size}


def rank_balance(parent, solver):
    """The spread over the ranks of the work that each rank owns.

    The partition balances the mantle cells only (`mantle_balanced_partition`).
    The cells of the inner sphere and of the buffer go to the rank whose
    mantle sector covers them, and the potential lives on all of them, so the
    parent cells and the unknowns per rank can differ more. The `Real` rows
    are left out: every rank holds a copy of them.

    Collective.

    Args:
      parent: the parent mesh.
      solver: the `SelfGravitatingGIASolver`.

    Returns:
      A dict from a quantity name to `(min, mean, max)` over the ranks.
    """
    owned_unknowns = sum(
        f.dat.data_ro.size for f in solver.solution.subfunctions
        if f.function_space().ufl_element().family() != "Real")
    quantities = {"parent cells": parent.cell_set.size,
                  "owned unknowns": owned_unknowns}
    result = {}
    for name, local in quantities.items():
        values = COMM_WORLD.allgather(int(local))
        result[name] = (min(values), sum(values) / len(values), max(values))
    return result


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
    (`meshing-gia/NOTES/feasibility/giamip/n2_giamip.py`), so that the
    coastline the solver sees is the one the mesh was refined for: linear in
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
    of `step_years`, and the last year, so the run still ends where the
    protocol ends. With the Exp01 years (500 yr steps to 26 ka, then 100 yr
    steps) and `step_years = 2000` this keeps -122000, -120000, ..., 0 and
    1000: 61 steps of 2000 yr and one last step of 1000 yr. Backward Euler is
    stable for any step; the larger step loses the accuracy of the mantle
    modes that relax faster than the step and samples the ice history more
    coarsely, which is acceptable for the L0 runs of the topography
    iteration and not for the production run.

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
    """The bed and the ice on the mantle mesh, from the GIAMIP files.

    `SL_init = -bed / D` is CG2, because the mask width reads its slope (as
    in the Martinec driver). The ice is CG1: a CG2 interpolant of a field
    that is zero next to a thick ice margin undershoots below zero between
    nodes, and a negative thickness is a negative load. Every node of the
    mantle mesh takes the value of its direction, so the fields are defined
    in the whole mantle and are read only on Re.
    """

    def __init__(self, mantle, experiment, step_years=None):
        self.experiment = experiment
        self.ice_history = experiment.ice_history
        self.topography = experiment.topography
        lat_nodes = self.topography.latitudes()
        lon_nodes = grids.longitudes()
        grids.check(lat_nodes, np.asarray(self.topography._ds["lon"][:],
                                          dtype=float), name="topography")
        grids.check(self.ice_history.latitudes(),
                    np.asarray(self.ice_history._ds["lon"][:], dtype=float),
                    name="ice history")
        self.lat_nodes, self.lon_nodes = lat_nodes, lon_nodes

        cg2 = FunctionSpace(mantle, "CG", 2)
        cg1 = FunctionSpace(mantle, "CG", 1)
        self._cg2_transfer = self._transfer(cg2)
        self._cg1_transfer = self._transfer(cg1)

        bed = self.topography.elevation()
        self.SL_init = Function(cg2, name="SL_init")
        self.SL_init.dat.data_with_halos[:] = -self._cg2_transfer(bed) / D_SCALE
        self.I = Function(cg1, name="ice")
        self.I_init = Function(cg1, name="ice_init")
        self.steps = experiment.forcing_steps()
        self.years = experiment.model_years()
        if step_years is not None:
            # A coarser uniform step: keep a subset of the protocol years,
            # so the ice history is read at its own time slices and never
            # interpolated in time.
            keep = coarse_step_indices(self.years, step_years)
            self.steps, self.years = self.steps[keep], self.years[keep]
        self.set_ice(self.I_init, 0)
        self.set_ice(self.I, 0)

    def _transfer(self, space):
        mesh = space.mesh()
        X = Function(VectorFunctionSpace(mesh, space.ufl_element().family(),
                                         space.ufl_element().degree()))
        X.interpolate(SpatialCoordinate(mesh))
        lat, lon = lat_lon_of_points(np.array(X.dat.data_ro_with_halos))
        return BilinearTransfer(self.lat_nodes, self.lon_nodes, lat, lon)

    def thickness_m(self, index):
        """The ice thickness of forcing step `index` on the grid, m."""
        return self.ice_history.thickness(int(self.steps[index]))

    def set_ice(self, target, index):
        """Write the ice of forcing step `index` into `target`, in units of D."""
        values = self._cg1_transfer(self.thickness_m(index)) / D_SCALE
        # The bilinear interpolant of non-negative data is non-negative; the
        # clip only removes rounding.
        target.dat.data_with_halos[:] = np.maximum(values, 0.0)

    def grid_ice_volume_m3(self, index, radius_m):
        """The ice volume of step `index` on the grid (Gauss-Legendre), m^3."""
        return grids.integrate(self.thickness_m(index), radius_m)


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


def build_solver(parent, mantle, earth, scales, forcing, dt, args):
    """The coupled solver of Exp01.

    Returns:
      `(solver, layout, approximation, record)`.
    """
    gravity_bcs = {SURF_OUTER: {"dtn": SphericalDtN(L=DTN_DEGREE)},
                   SURF_INNER: {"dtn": SphericalDtN(L=DTN_DEGREE)}}
    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=True,
        fluid_core=True, self_gravity_number=scales.Lambda,
        displacement_degree=2, internal_variable_degree=1,
        potential_degree=2, centre_of_mass=True, sea_level=True,
        dtn_representation=common.DTN_REPRESENTATION)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    fields = earth_fields(mantle, earth, scales)
    approximation = CompressibleInternalVariableApproximation(
        bulk_modulus=fields["bulk_modulus"], density=fields["density"],
        shear_modulus=[fields["shear_modulus"]],
        viscosity=[fields["viscosity"]], bulk_shear_ratio=1,
        g=fields["g"], B_mu=scales.B_mu, self_gravity_number=scales.Lambda)

    rho_core = core_density(earth.elastic, scales, args.core_density)
    g_cmb = float(earth.elastic(scales.Rc_m, "below")["g_m_s2"]) / scales.g_bar
    fluid_core = FluidCore(boundary=SURF_RC, rho_core=rho_core / scales.rho_bar,
                           g=Constant(g_cmb))

    constants = forcing.experiment.constants
    rho_w = float(constants["ocean_water_density_kg_m3"]) / scales.rho_bar
    rho_i = float(constants["ice_density_kg_m3"]) / scales.rho_bar
    # g_s = g_bar by the choice of the gravity scale.
    g_surface = 1.0
    sea_level = SeaLevel(
        boundary=SURF_RE, rho_w=rho_w, rho_i=rho_i, g_surface=g_surface,
        SL_init=forcing.SL_init, I=forcing.I, I_init=forcing.I_init,
        N_init=Constant(0.0), ur_init=Constant(0.0),
        alpha_mask=DEFAULT_ALPHA_MASK, slope=surface_slope(forcing.SL_init),
        grad_floor=DEFAULT_GRAD_FLOOR, fixed_ocean=False)

    C_kg_m2 = C_OVER_MRE2 * scales.earth_mass_kg * scales.Re_m**2
    C_nd = C_kg_m2 / (scales.rho_bar * D_SCALE**5)

    parameters = selfgrav_dtn_iterative_solver_parameters(
        condensed=layout.condensed, outer_rtol=OUTER_RTOL,
        block0_rtol=BLOCK0_RTOL, block0_max_it=args.block0_max_it,
        # The stopping rule of the outer FGMRES: an optional absolute floor
        # (`--outer_atol`, off by default) and a capped solve counted as
        # converged so Newton finishes the step. See the preset's docstring.
        outer_atol=args.outer_atol, outer_max_it=args.outer_max_it,
        outer_accept_capped=True,
        snes_type="newtonls",
        dtn_representation=common.DTN_REPRESENTATION,
        n_real=len(layout.real_fields))
    # The one rebuild rule of the preconditioner (block 0 and the Real-block
    # complement together). Without the options the library default holds:
    # `lagged`, every 5 nonlinear solves, at once when dt changes.
    if args.gia_rebuild is not None:
        parameters["gia_rebuild"] = args.gia_rebuild
    if args.gia_rebuild_every is not None:
        parameters["gia_rebuild_every"] = args.gia_rebuild_every
    if args.gia_rebuild_max_outer is not None:
        parameters["gia_rebuild_max_outer"] = args.gia_rebuild_max_outer

    # A rigid rotation of the mantle moves no water and has no mass moment
    # for a spherically symmetric density, so it stays in the kernel with sea
    # level and the frame rows on (as in the Martinec and Spada drivers).
    nullspace = rigid_rotation_nullspace(Z, layout)
    solver = SelfGravitatingGIASolver(
        z, approximation, layout=layout, dt=dt, bcs={},
        fluid_core=fluid_core, sea_level=sea_level,
        dtn_representation=common.DTN_REPRESENTATION,
        rotation_moments={"C": C_nd, "k_s": args.k_s},
        Omega_sq=scales.Omega_sq, surface_radius=scales.Re,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=parameters)
    record = {
        "rho_core_kg_m3": rho_core, "core_density_rule": args.core_density,
        "g_cmb_nondim": g_cmb, "rho_w_nondim": rho_w, "rho_i_nondim": rho_i,
        "C_kg_m2": C_kg_m2, "C_nondim": C_nd, "k_s": args.k_s,
        "C_minus_A_nondim": float(
            solver.rotation_moments.get("C_minus_A", np.nan)),
        "gia_rebuild": parameters.get("gia_rebuild", "lagged (default)"),
        "gia_rebuild_every": parameters.get("gia_rebuild_every", "default"),
        "gia_rebuild_max_outer": parameters.get("gia_rebuild_max_outer", 0.0),
        "block0_max_it": args.block0_max_it,
        "outer_atol": args.outer_atol, "outer_max_it": args.outer_max_it,
        "unknowns": solver.solution.function_space().dim(),
        "real_rows": len(layout.real_fields),
        "ranges": {name: field_range(f) for name, f in fields.items()},
        "solver_parameters": {k: str(v) for k, v in parameters.items()},
    }
    return solver, layout, approximation, record


# --------------------------------------------------------------------------
# Numbers of one solved state
# --------------------------------------------------------------------------


def state_row(solver, layout, forcing, scales, index, dt_yr, wall_s):
    """Every per-step number of one solved state. Collective."""
    dss = solver.sea_level_measure()(SURF_RE)
    k = solver._sea_level_steepness()
    p = solver.sea_level_parameters
    SL = solver.sea_level()
    C = ocean_function(SL, k)
    B = grounded_ice_function(forcing.I, SL, k, p.rho_w, p.rho_i)
    ice = forcing.I
    D3 = D_SCALE**3
    rho_i_kg = p.rho_i * scales.rho_bar
    rho_w_kg = p.rho_w * scales.rho_bar
    shift = float(solver.solution.subfunctions[layout.sea_level])
    rot = solver.rotation_values()
    newton, outer = common.iteration_counts(solver)
    sphere = 4.0 * np.pi * scales.Re**2
    sheet = solver.surface_load_sheet()
    dipole = solver.mass_dipole()
    load_moment = float(assemble(scales.Re * abs(sheet) * dss))
    abs_dipole = float(np.linalg.norm(dipole))
    ice_volume_mesh = float(assemble(ice * dss)) * D3
    ice_volume_grid = forcing.grid_ice_volume_m3(index, scales.Re_m)
    # maf: rho_i int max(0, I - (rho_w / rho_i) max(SL, 0)) dS, the ice mass
    # above flotation.
    maf = rho_i_kg * float(assemble(
        max_value(ice - (p.rho_w / p.rho_i) * max_value(SL, 0.0), 0.0)
        * dss)) * D3
    return {
        "step": int(index), "year": float(forcing.years[index]),
        "dt_yr": float(dt_yr), "wall_s": float(wall_s),
        "newton": newton, "outer": outer,
        "shift_m": shift * D_SCALE,
        "m1": rot.get("m1", 0.0), "m2": rot.get("m2", 0.0),
        "m3": rot.get("m3", 0.0),
        "ocean_area_B": float(assemble(B * dss)) / sphere,
        "ocean_area_C": float(assemble(C * dss)) / sphere,
        "grounded_ice_kg": rho_i_kg * float(assemble((1 - B) * ice * dss)) * D3,
        "total_ice_kg": rho_i_kg * ice_volume_mesh,
        "total_ice_grid_kg": rho_i_kg * ice_volume_grid,
        "ice_transfer_error": ((ice_volume_mesh - ice_volume_grid)
                               / ice_volume_grid if ice_volume_grid else 0.0),
        "maf_kg": maf,
        "net_sheet_mass": float(assemble(sheet * dss)),
        "dipole_rel": abs_dipole / load_moment if load_moment > 0 else 0.0,
        "water_density_kg_m3": rho_w_kg,
    }


ROW_KEYS = ("step", "year", "dt_yr", "wall_s", "newton", "outer", "shift_m",
            "m1", "m2", "m3", "ocean_area_B", "ocean_area_C",
            "grounded_ice_kg", "total_ice_kg", "total_ice_grid_kg",
            "ice_transfer_error", "maf_kg", "net_sheet_mass", "dipole_rel")


class FieldOutput:
    """The solved fields as two Firedrake `CheckpointFile`s (HDF5), no VTK.

    A `VTKFile` writes one file per rank at every write, which on thousands
    of ranks and 463 steps is millions of files. A `CheckpointFile` is one
    file whatever the number of ranks, and it loads on any number of ranks.
    The grid values (U, N and S on the 257 x 513 grid) and the ParaView files
    are made after the run from these files, on a few ranks.

    `surface_from<start>.h5`, every step (and the initial state of a new
    run). The mesh is the surface Re of the mantle, a codim-1 `Submesh`
    (`exp01_surface`). Fields, SI units:

    - `displacement_m`: the displacement, P2 vector, m.
    - `potential_m2_s2`: the perturbed gravitational potential, P2, m^2/s^2.
    - `dpotential_dr_m_s2`: its radial derivative on the mantle side of Re,
      DG1, m/s^2. The derivative jumps across Re by 4 pi G times the surface
      load (ice and ocean), so the outer side follows from the load.
    - `ice_m`: the ice thickness of the step, CG1, m.

    Each entry has `idx` = the step number and the timestepping values
    `step`, `year`, `shift_m` (the uniform sea-level shift) and `m1`, `m2`,
    `m3` (the rotation vector). The geoid is then
    N = (potential + rotational potential of m) / g(Re) and the sea surface
    is S = N + shift. Saved once: `coordinates_m`, the P2 coordinates of the
    mantle at the surface nodes. The submesh can carry flat (P1)
    coordinates; this field gives the curved positions of the P2 nodes.

    `mantle_from<start>.h5`, every `mantle_every` steps, the initial state
    and the last step of the job. The mesh is the mantle (`exp01_mantle`).
    Fields: `displacement_m` and `potential_m2_s2` (P2), same entries and
    timestepping values. Saved once: the earth model of the run in SI units
    (density, shear and bulk moduli, viscosity, gravity) and the initial sea
    level and ice, in DG1.

    Every job writes its own files (`<start>` is the first step of the job,
    four digits), so a restart never opens a file that an earlier job left
    behind. A file is opened and closed for each write, so the entries
    already written are complete on disk if a later write fails.

    The potential lives on the parent mesh. It goes to the mantle P2 space
    first (each mantle node is a parent node, so the copy is exact and
    follows the submesh map). From the mantle to the surface, Firedrake
    interpolates by locating every surface node in the mantle mesh. That
    search is cheap to repeat on a small mesh, but at L2 it holds about
    1.2 GB per rank for a moment. After a solve, with the Jacobian and the
    preconditioner still in memory, that made nodes of 128 ranks go over
    their limit. The interpolation from the mantle to the surface is
    therefore assembled once per element as a matrix (`_matrices`), here at
    setup where the memory is free, and every write only multiplies by it.
    A test on 1 and 2 ranks gave errors at round-off and no search after
    the setup.
    """

    def __init__(self, directory, parent, mantle, solver, layout,
                 approximation, forcing, scales, start, mantle_every):
        from firedrake import CheckpointFile, Submesh
        self._CheckpointFile = CheckpointFile
        self.solver, self.forcing, self.scales = solver, forcing, scales
        # The index of the `Real` sea-level shift in the mixed solution.
        self.sea_level_index = layout.sea_level
        self.mantle_every = int(mantle_every)
        self.mantle = mantle
        self.surface = Submesh(mantle, 2, SURF_RE, name="exp01_surface")
        self.surface_path = os.path.join(directory,
                                         f"surface_from{start:04d}.h5")
        self.mantle_path = os.path.join(directory, f"mantle_from{start:04d}.h5")
        # The mantle spaces of the potential and of its radial derivative.
        self._phi_mantle = Function(FunctionSpace(mantle, "CG", 2))
        Xm = SpatialCoordinate(mantle)
        self._dphi_dr = dot(grad(self._phi_mantle), Xm / sqrt(dot(Xm, Xm)))
        self._dphi_dr_mantle = Function(FunctionSpace(mantle, "DG", 1))
        # The surface fields of the file.
        surface = self.surface
        self.s_u = Function(VectorFunctionSpace(surface, "CG", 2),
                            name="displacement_m")
        self.s_phi = Function(FunctionSpace(surface, "CG", 2),
                              name="potential_m2_s2")
        self.s_dphi = Function(FunctionSpace(surface, "DG", 1),
                               name="dpotential_dr_m_s2")
        self.s_ice = Function(FunctionSpace(surface, "CG", 1), name="ice_m")
        # The mantle fields of the file.
        self.m_u = Function(VectorFunctionSpace(mantle, "CG", 2),
                            name="displacement_m")
        self.m_phi = Function(FunctionSpace(mantle, "CG", 2),
                              name="potential_m2_s2")
        # Potential: nondimensional potential times g_bar D gives m^2/s^2;
        # its radial derivative times g_bar gives m/s^2.
        self._phi_scale = scales.g_bar * D_SCALE
        self._dphi_scale = scales.g_bar
        # The mantle copies of the displacement and the ice, which the
        # interpolation matrices act on.
        self._u_mantle = Function(VectorFunctionSpace(mantle, "CG", 2))
        self._ice_mantle = Function(forcing.I.function_space())
        # One interpolation matrix from the mantle to the surface per element
        # (P2 vector, P2, DG1 and the element of the ice), assembled once.
        self._matrices = {}
        for source, target in ((self._u_mantle, self.s_u),
                               (self._phi_mantle, self.s_phi),
                               (self._dphi_dr_mantle, self.s_dphi),
                               (self._ice_mantle, self.s_ice)):
            self._matrix(source.function_space(), target.function_space())
        self._write_static(approximation)

    def _matrix(self, source_space, target_space):
        """The interpolation matrix from a mantle space to a surface space.

        Assembled on the first call for a pair of spaces and kept. Building
        it locates the surface nodes in the mantle mesh once. Collective.
        """
        key = (source_space.ufl_element(), target_space.ufl_element())
        if key not in self._matrices:
            self._matrices[key] = assemble(interpolate(
                TrialFunction(source_space), target_space)).petscmat
        return self._matrices[key]

    def _to_surface(self, target, source):
        """Apply the kept interpolation matrix: `target` = I `source`."""
        matrix = self._matrix(source.function_space(), target.function_space())
        with source.dat.vec_ro as x, target.dat.vec_wo as y:
            matrix.mult(x, y)

    def _to(self, target, source):
        """Interpolate `source` into `target` across the submesh maps.

        Used for the parent-to-mantle copy of the potential, which follows
        the submesh map without a search.
        """
        target.assign(assemble(interpolate(source, target.function_space())))

    def _write_static(self, approximation):
        """Create both files with their mesh and the fields of no step."""
        coordinates = Function(VectorFunctionSpace(self.surface, "CG", 2),
                               name="coordinates_m")
        self._u_mantle.interpolate(SpatialCoordinate(self.mantle))
        self._to_surface(coordinates, self._u_mantle)
        coordinates *= D_SCALE
        with self._CheckpointFile(self.surface_path, "w") as chk:
            chk.save_mesh(self.surface)
            chk.save_function(coordinates)
        dg1 = FunctionSpace(self.mantle, "DG", 1)
        earth = [
            ("density_kg_m3", approximation.density, self.scales.rho_bar),
            ("shear_modulus_pa", approximation.shear_modulus[0], MU_BAR),
            ("bulk_modulus_pa", approximation.bulk_modulus, MU_BAR),
            ("viscosity_pa_s", approximation.viscosity[0], ETA_BAR),
            ("gravity_m_s2", approximation.g, self.scales.g_bar),
            ("SL_init_m", self.forcing.SL_init, D_SCALE),
            ("ice_init_m", self.forcing.I_init, D_SCALE)]
        with self._CheckpointFile(self.mantle_path, "w") as chk:
            chk.save_mesh(self.mantle)
            for name, value, scale in earth:
                chk.save_function(
                    Function(dg1, name=name).interpolate(value * scale))

    def _state(self, step, year):
        """The timestepping values of one entry (floats, as h5py needs)."""
        values = self.solver.rotation_values()
        shift = float(self.solver.solution.subfunctions[self.sea_level_index])
        return {"step": float(step), "year": float(year),
                "shift_m": shift * D_SCALE,
                "m1": float(values.get("m1", 0.0)),
                "m2": float(values.get("m2", 0.0)),
                "m3": float(values.get("m3", 0.0))}

    def write(self, step, year, last=False):
        """Append the state of `step`; the mantle too when it is due.

        Collective. Returns the wall time of the write, s.
        """
        tic = time.time()
        solver = self.solver
        info = self._state(step, year)
        # The potential on the mantle, then its radial derivative there.
        self._to(self._phi_mantle, solver.potential)
        self._dphi_dr_mantle.interpolate(self._dphi_dr)
        self._u_mantle.assign(solver.displacement)
        self._to_surface(self.s_u, self._u_mantle)
        self.s_u *= D_SCALE
        self._to_surface(self.s_phi, self._phi_mantle)
        self.s_phi *= self._phi_scale
        self._to_surface(self.s_dphi, self._dphi_dr_mantle)
        self.s_dphi *= self._dphi_scale
        self._ice_mantle.assign(self.forcing.I)
        self._to_surface(self.s_ice, self._ice_mantle)
        self.s_ice *= D_SCALE
        with self._CheckpointFile(self.surface_path, "a") as chk:
            for f in (self.s_u, self.s_phi, self.s_dphi, self.s_ice):
                chk.save_function(f, idx=int(step), timestepping_info=info)
        if (step == 0 or last
                or (self.mantle_every > 0 and step % self.mantle_every == 0)):
            self.write_mantle(step, year, info)
        return time.time() - tic

    def write_mantle(self, step, year, info=None):
        """Append the mantle fields of `step`. Collective.

        `write` calls this when the mantle is due. A job that stops early
        calls it for its last step, whose surface entry already exists.
        """
        if info is None:
            info = self._state(step, year)
            self._to(self._phi_mantle, self.solver.potential)
        self.m_u.interpolate(self.solver.displacement * D_SCALE)
        self.m_phi.assign(self._phi_mantle)
        self.m_phi *= self._phi_scale
        with self._CheckpointFile(self.mantle_path, "a") as chk:
            for f in (self.m_u, self.m_phi):
                chk.save_function(f, idx=int(step), timestepping_info=info)


class RotatingCheckpoint(common.RunCheckpoint):
    """`RunCheckpoint` with one file per checkpoint, keeping the newest two.

    At L2 one checkpoint is about 25 GB, so the single growing file of
    `RunCheckpoint` would reach a terabyte. Each save writes
    `checkpoint_exp01_step<step>.h5` with one entry (mesh and fields), points
    the symbolic link `checkpoint_exp01_latest.h5` at it, and deletes the
    older step files except the newest `keep`. The link moves only after the
    file is closed, so it always names a complete checkpoint, and a restart
    with `--restart <dir>/checkpoint_exp01_latest.h5` continues from it.
    """

    LATEST = "checkpoint_exp01_latest.h5"

    def __init__(self, directory, parent, solver, every, mesh_info, record,
                 keep=2):
        self.directory = directory
        self.keep = int(keep)
        super().__init__(os.path.join(directory, self.LATEST), parent, solver,
                         every, mesh_info, record, restart=None)

    def _keep_previous_file(self):
        """No renaming: every checkpoint has its own file."""

    def save(self, step, t_kyr):
        """Write the checkpoint of `step` to its own file. Collective."""
        self.path = os.path.join(self.directory,
                                 f"checkpoint_exp01_step{step:04d}.h5")
        self.next_index = 0
        self.appending = False
        super().save(step, t_kyr)
        if COMM_WORLD.rank == 0:
            link = os.path.join(self.directory, self.LATEST)
            temporary = link + ".tmp"
            if os.path.lexists(temporary):
                os.remove(temporary)
            os.symlink(os.path.basename(self.path), temporary)
            os.replace(temporary, link)
            steps = sorted(
                int(name[len("checkpoint_exp01_step"):-len(".h5")])
                for name in os.listdir(self.directory)
                if name.startswith("checkpoint_exp01_step")
                and name.endswith(".h5"))
            for old in steps[:-self.keep]:
                os.remove(os.path.join(self.directory,
                                       f"checkpoint_exp01_step{old:04d}.h5"))
        COMM_WORLD.barrier()


# --------------------------------------------------------------------------
# Restart
# --------------------------------------------------------------------------


def load_restart(path, radial):
    """The meshes and the last state of a checkpoint file of this driver.

    `selfgrav_common.load_checkpoint` with the giamsh mesh: the parent is
    loaded from the file with its stored distribution, and the mantle is cut
    from it with the mantle tags of the sidecar and curved again.
    """
    from firedrake import CheckpointFile
    group = common._RESTART_GROUP
    with CheckpointFile(path, "r") as chk:
        mesh_info = common._read_json_dataset(chk, f"{group}/mesh_info")
        if mesh_info["ranks"] != COMM_WORLD.size:
            raise RuntimeError(
                f"{path} was written on {mesh_info['ranks']} ranks; restart "
                f"on the same number, not {COMM_WORLD.size}.")
        parent = chk.load_mesh("exp01_parent")
        names = mesh_info["fields"]
        histories = {name: chk.get_timestepping_history(parent, name)
                     for name in names}
        written = [{int(i) for i in histories[name]["index"]}
                   for name in names]
        complete = sorted(i for i in set.intersection(*written)
                          if f"{group}/{i}/record" in chk.h5pyfile)
        if not complete:
            raise RuntimeError(f"{path} holds no complete checkpoint")
        index = complete[-1]
        next_index = max(set.union(*written)) + 1
        first = histories[names[0]]
        position = [int(i) for i in first["index"]].index(index)
        step = int(round(first["step"][position]))
        t_kyr = float(first["t_kyr"][position])
        fields = {name: chk.load_function(parent, name, idx=index)
                  for name in names}
        record = common._read_json_dataset(chk, f"{group}/{index}/record")
    parent.cartesian = False
    mantle = giamsh_refine.curve_mesh(
        _mantle_submesh(parent, list(radial.region_tags("mantle"))),
        name="exp01_mantle")
    log(f"  restart: {path}, entry {index}, step {step}")
    state = common.RestartState(path, index, next_index, step, t_kyr, fields,
                                record)
    return parent, mantle, mesh_info, state


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def provenance():
    """Software versions and the state of the source tree (rank 0)."""
    info = {"giamip": giamip.__version__, "python": sys.version.split()[0],
            "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
            "petsc_int_bytes": int(np.dtype(PETSc.IntType).itemsize)}
    try:
        import importlib.metadata as md
        info["firedrake"] = md.version("firedrake")
    except Exception:  # noqa: BLE001  (a version is a record, not a need)
        info["firedrake"] = None
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        status = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                                capture_output=True, text=True, timeout=60)
        info["git_head"] = head.stdout.strip()
        # The text of `git diff` depends on the git version (2.55 on phoenix
        # and 2.51 on Setonix give different bytes for the same change), so
        # the record holds the SHA-256 of each changed or new file instead.
        info["git_status"] = status.stdout.splitlines()
        changed = {}
        for line in info["git_status"]:
            path = os.path.join(root, line[3:].strip())
            files = ([os.path.join(d, f) for d, _, fs in os.walk(path)
                      for f in fs] if os.path.isdir(path) else [path])
            for name in sorted(files):
                if os.path.isfile(name) and "__pycache__" not in name:
                    with open(name, "rb") as handle:
                        changed[os.path.relpath(name, root)] = (
                            hashlib.sha256(handle.read()).hexdigest())
        info["changed_files_sha256"] = changed
    except Exception as exc:  # noqa: BLE001
        info["git_error"] = repr(exc)
    with open(os.path.abspath(__file__), "rb") as handle:
        info["driver_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    return info


#: The ranks of one node, for the memory total that the node limit applies to.
_NODE_COMM = None


def memory_now(label):
    """Log the resident memory now: mean and largest per rank, largest node.

    Collective. The resident size comes from `/proc/self/statm` (Linux), so
    it is the memory in use at this moment, not the peak. The node total is
    the sum over the ranks of one node, which is what the Slurm memory limit
    of a node applies to. On a system without `/proc` it logs nothing.

    Args:
      label: the point of the run, for the log line.
    """
    global _NODE_COMM
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return
    if _NODE_COMM is None:
        _NODE_COMM = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    node = _NODE_COMM.allreduce(rss, op=MPI.SUM)
    mean = COMM_WORLD.allreduce(rss, op=MPI.SUM) / COMM_WORLD.size
    largest = COMM_WORLD.allreduce((rss, COMM_WORLD.rank), op=MPI.MAXLOC)
    node_max = COMM_WORLD.allreduce(node, op=MPI.MAX)
    log(f"  memory {label}: rank mean {mean:.2f} GB, largest {largest[0]:.2f} "
        f"GB (rank {largest[1]}), largest node {node_max:.0f} GB")


def block0_assembly_count(solver):
    """The number of block-0 rebuilds so far, or None if it cannot be read.

    Reads `assembly_count` of the `gadopt.CondensedBlockPC` in the first split
    of the outer field split. It shows whether a lagged preconditioner
    really skips the block-0 rebuild.
    """
    # Every type is checked before `getPythonContext`: asking a PC that is not
    # of type python for its context reads a pointer that is not a Python
    # object and crashes the process. The outer PC is a Python PC whose `pc`
    # is the inner field split; its first split runs `CondensedBlockPC`.
    outer = solver.solver.snes.ksp.pc
    if outer.getType() != "python":
        return None
    inner = getattr(outer.getPythonContext(), "pc", None)
    if inner is None or inner.getType() != "fieldsplit":
        return None
    block0 = inner.getFieldSplitSubKSP()[0].getPC()
    if block0.getType() != "python":
        return None
    count = getattr(block0.getPythonContext(), "assembly_count", None)
    return None if count is None else int(count)


def peak_memory_gb():
    """The largest peak resident memory over the ranks, GB (Linux kB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS bytes.
    gb = rss / 1e6 if sys.platform.startswith("linux") else rss / 1e9
    return COMM_WORLD.allreduce(gb, op=MPI.MAX)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True,
                   help="the coarse giamsh .msh file; its .json sidecar must "
                        "be beside it")
    p.add_argument("--refine", type=int, default=0,
                   help="uniform refinement levels (0 for L0)")
    p.add_argument("--output_path", default="./")
    p.add_argument("--step_years", type=int, default=None,
                   help="a uniform coarser step in years (for example 2000); "
                        "keeps a subset of the protocol years")
    p.add_argument("--steps", type=int, default=None,
                   help="stop after this many forcing steps")
    p.add_argument("--k_s", type=float, default=0.9422,
                   help="the fluid Love number of the bulge; it must be the "
                        "kf of the lovejx run of the comparison")
    p.add_argument("--core_density", choices=["mean", "cmb"], default="mean")
    # The rebuild rule of the preconditioner (solver options `gia_rebuild*`);
    # left out, the library default holds (lagged, every 5 solves).
    p.add_argument("--gia_rebuild", choices=["per_solve", "lagged", "always"],
                   default=None)
    p.add_argument("--gia_rebuild_every", type=int, default=None)
    p.add_argument("--gia_rebuild_max_outer", type=float, default=None)
    # The iteration cap of the block-0 FGMRES (also its restart length). On
    # the L0 mesh the capped block-0 solves stop decreasing after about 30
    # iterations, so a cap near 50 removes iterations that do not reduce the
    # residual. The outer FGMRES accepts an inexact block-0 solve.
    p.add_argument("--block0_max_it", type=int, default=200)
    # The stopping rule of the outer FGMRES (preset arguments `outer_*`).
    # 0 (off): the low-rank operator removes the declared rigid rotation from
    # its action, so the outer residual has no floor above round-off and
    # every solve converges on outer_rtol (verified on Setonix, steps 51-62
    # at L0, NOTES/EXP01-HANDOVER.md section 18.9).
    p.add_argument("--outer_atol", type=float, default=0.0,
                   help="absolute tolerance of the outer FGMRES (0: off)")
    p.add_argument("--outer_max_it", type=int, default=30,
                   help="iteration cap of the outer FGMRES; a solve at the "
                        "cap counts as converged")
    p.add_argument("--checkpoint_every", type=int, default=25,
                   help="steps between two restart checkpoints; the newest "
                        "two are kept, and the last step of a job is saved")
    p.add_argument("--restart", default=None, metavar="FILE")
    p.add_argument("--check_quadrature", action="store_true",
                   help="measure the quadrature error of the DtN boundary "
                        "rule at setup (minutes)")
    p.add_argument("--mantle_every", type=int, default=50,
                   help="steps between two writes of the mantle fields; the "
                        "surface fields are written every step")
    p.add_argument("--wall_limit_hours", type=float, default=None,
                   help="stop after the step that would leave too little "
                        "time, save a checkpoint and exit 0 (the job script "
                        "passes the Slurm time limit)")
    p.add_argument("--wall_margin_s", type=float, default=1200.0,
                   help="time kept free at the end of the job for the last "
                        "checkpoint and the summary, s")
    return p.parse_args()


def main():
    """Run Exp01."""
    args = parse_args()
    tic_run = time.time()
    if COMM_WORLD.rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    COMM_WORLD.barrier()
    timings = {}

    experiment = GiamipExperiment("Exp01")
    earth = experiment.earth()
    scales = Scales(earth.elastic)
    sidecar = os.path.splitext(args.mesh)[0] + ".json"
    radial, mesh_record = read_radial(sidecar)
    if abs(radial.radius("surface") - scales.Re) > 1e-9 * scales.Re:
        raise RuntimeError(f"mesh surface radius {radial.radius('surface')} "
                           f"against PREM-stable {scales.Re}")

    log("=" * 78)
    log("GIAMIP Exp01, G-ADOPT")
    log("=" * 78)
    log(f"  scales: {scales.as_dict()}")

    tic = time.time()
    restart = None
    if args.restart is None:
        parent, mantle, mesh_info = build_meshes(args.mesh, radial,
                                                 args.refine)
    else:
        # The restart path can be the link `checkpoint_exp01_latest.h5`;
        # the record keeps the file it points at.
        args.restart = os.path.realpath(args.restart)
        parent, mantle, mesh_info, restart = load_restart(args.restart, radial)
    timings["mesh_s"] = time.time() - tic
    log(f"  mesh: {mesh_info} ({timings['mesh_s']:.1f} s)")
    memory_now("after the mesh")

    tic = time.time()
    forcing = SurfaceForcing(mantle, experiment, step_years=args.step_years)
    timings["forcing_s"] = time.time() - tic
    dt = Constant(1.0)
    tic = time.time()
    solver, layout, approximation, solver_record = build_solver(
        parent, mantle, earth, scales, forcing, dt, args)
    timings["solver_build_s"] = time.time() - tic
    log(f"  solver: {solver_record['unknowns']} unknowns, "
        f"{solver_record['real_rows']} Real rows, core "
        f"{solver_record['rho_core_kg_m3']:.1f} kg/m^3, k_s {args.k_s}, "
        f"C - A {solver_record['C_minus_A_nondim']:.6g} "
        f"({timings['solver_build_s']:.1f} s)")
    for name, (lo, hi) in solver_record["ranges"].items():
        log(f"    {name}: {lo:.6g} .. {hi:.6g}")
    memory_now("after the solver build")
    balance = rank_balance(parent, solver)
    solver_record["rank_balance"] = balance
    for name, (lo, mean, hi) in balance.items():
        log(f"    per rank {name}: min {lo}, mean {mean:.0f}, max {hi}, "
            f"max/mean {hi / mean:.3f}")
    # The inner DtN sphere has about 200 facets, and the default boundary
    # rule is capped below the degree that `SphericalDtN(L=5)` asks for
    # there (a warning of `dtn_form`). This measures the quadrature error of
    # the rule on the DtN modes against a richer rule on the same mesh; it
    # only warns.
    # It costs minutes (176 s on 6 ranks on the 176k-cell smoke mesh, where
    # it gave 3.5e-14), so it runs on request only.
    if args.check_quadrature:
        tic = time.time()
        solver_record["dtn_quadrature_error"] = float(
            solver.check_boundary_quadrature(rtol=1e-6, action="warn"))
        log(f"  DtN boundary quadrature: worst relative error "
            f"{solver_record['dtn_quadrature_error']:.3e} "
            f"({time.time() - tic:.1f} s)")

    tic = time.time()
    start = 0 if restart is None else restart.step
    output = FieldOutput(args.output_path, parent, mantle, solver, layout,
                         approximation, forcing, scales, start,
                         args.mantle_every)
    timings["output_setup_s"] = time.time() - tic
    memory_now("after the output setup")

    summary_path = os.path.join(args.output_path, "summary_exp01.json")
    summary = {"experiment": experiment.id, "args": vars(args),
               "scales": scales.as_dict(), "mesh": mesh_info,
               "mesh_sidecar": mesh_record.get("inputs"),
               "solver": solver_record, "timings": timings,
               "outputs": {"surface": os.path.basename(output.surface_path),
                           "mantle": os.path.basename(output.mantle_path)},
               "provenance": provenance() if COMM_WORLD.rank == 0 else None}

    rows = [] if restart is None else restart.record["rows"]
    plog = ParameterLog(os.path.join(args.output_path, "params_exp01.log"),
                        parent)
    plog.log_str(" ".join(ROW_KEYS))
    for row in rows:
        plog.log_str(" ".join(str(row[key]) for key in ROW_KEYS))

    # The initial state: the ice of 122 ka on the undeformed Earth. The row
    # is written with the masks of the initial sea level.
    if restart is None:
        rows.append(state_row(solver, layout, forcing, scales, 0, 0.0, 0.0))
        plog.log_str(" ".join(str(rows[-1][key]) for key in ROW_KEYS))
        output.write(0, rows[-1]["year"])
        memory_now("after the step-0 output")
        log(f"  initial state: ocean area {rows[-1]['ocean_area_B']:.6f}, "
            f"ice {rows[-1]['total_ice_kg']:.6e} kg (grid "
            f"{rows[-1]['total_ice_grid_kg']:.6e} kg, transfer error "
            f"{rows[-1]['ice_transfer_error']:.3e})")

    def record():
        return {"rows": rows, "wall_s": time.time() - tic_run}

    checkpoint = RotatingCheckpoint(
        args.output_path, parent, solver, args.checkpoint_every,
        {**mesh_info, "kind": "giamip-exp01"}, record)
    if restart is not None:
        common.restore_checkpoint(restart, solver, parent)
        # The ice of the restored step, so that the next step starts from it.
        forcing.set_ice(forcing.I, start)

    n_steps = len(forcing.years) - 1
    last = n_steps if args.steps is None else min(n_steps, start + args.steps)
    previous_dt = None
    common.write_json(summary_path, {**summary, "state": "running"})
    # The longest step of this job (solve and output), for the time limit.
    longest_step_s = 0.0
    stopped_early = False
    for index in range(start + 1, last + 1):
        dt_yr = float(forcing.years[index] - forcing.years[index - 1])
        if dt_yr != previous_dt:
            dt.assign(dt_yr / T_BAR_YR)
            previous_dt = dt_yr
        # Backward Euler: the load is the one at the end of the step.
        forcing.set_ice(forcing.I, index)
        tic = time.time()
        solver.solve()
        wall = time.time() - tic
        memory_now(f"step {index} after the solve")
        tic = time.time()
        row = state_row(solver, layout, forcing, scales, index, dt_yr, wall)
        memory_now(f"step {index} after the step numbers")
        rows.append(row)
        plog.log_str(" ".join(str(row[key]) for key in ROW_KEYS))
        output_s = output.write(index, row["year"], last=(index == last))
        memory_now(f"step {index} after the output")
        log(f"  step {index:3d}  year {row['year']:9.0f}  dt {dt_yr:5.0f} yr  "
            f"newton {row['newton']}  outer {row['outer']:3d}  block-0 "
            f"builds {block0_assembly_count(solver)}  shift "
            f"{row['shift_m']:9.4f} m  m ({row['m1']:.3e}, {row['m2']:.3e}, "
            f"{row['m3']:.3e})  solve {wall:7.1f} s  output "
            f"{time.time() - tic:5.1f} s (files {output_s:.1f} s)  peak "
            f"memory {peak_memory_gb():.2f} GB")
        checkpoint(index, row["year"] / 1000.0)
        longest_step_s = max(longest_step_s, time.time() - tic + wall)
        # Stop when two more of the longest steps and the margin for the
        # last checkpoint do not fit into the time limit of the job. The
        # time is that of rank 0, broadcast so that every rank decides alike.
        if args.wall_limit_hours is not None and index < last:
            elapsed = COMM_WORLD.bcast(time.time() - tic_run, root=0)
            if (elapsed + 2.0 * longest_step_s + args.wall_margin_s
                    > 3600.0 * args.wall_limit_hours):
                log(f"  stopping after step {index}: {elapsed:.0f} s used "
                    f"of {3600.0 * args.wall_limit_hours:.0f} s")
                last = index
                stopped_early = True
                break
    # Save the last step of the job if the rule above did not save it, so
    # the next job of a chain restarts from here and no solved step is lost.
    if (last > start and args.checkpoint_every > 0
            and last % args.checkpoint_every != 0):
        checkpoint.save(last, rows[-1]["year"] / 1000.0)
    if stopped_early and last % max(args.mantle_every, 1) != 0:
        # The mantle fields of the last step of this job, as a job that
        # runs to its planned end writes them.
        output.write_mantle(last, rows[-1]["year"])
    plog.close()
    summary["timings"]["total_s"] = time.time() - tic_run
    summary["timings"]["peak_memory_gb"] = peak_memory_gb()
    state = "finished" if last == n_steps else "stopped"
    common.write_json(summary_path, {**summary, "state": state,
                                     "last_step": last, "rows": rows})


if __name__ == "__main__":
    main()
