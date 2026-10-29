r"""Code shared by `prepare_exp10.py` and `invert_exp10.py`.

This module holds everything that both scripts need and that does not read
the GIAMIP data: the scales of the run, the mesh partition and the mesh
cutting, the input file (write and read), the solver construction from
fields that are already on the mesh, and a few helpers for logs and records.

It must not import `giamip`, `gdrift`, `h5netcdf` or `netCDF4`. The
inversion imports this module, and the inversion must not depend on the
data readers: every field that comes from data is made once by the prepare
script and written to the input file.

The directory of this file is self-contained. It imports nothing from
another experiment directory and nothing from `tests/`. Shared library code
comes from `gadopt` and `giamsh` only.

THE INPUT FILE
    One Firedrake `CheckpointFile`, written by `write_inputs` and read by
    `read_inputs`. It holds:

    - the parent mesh (`giamip_parent`), refined and curved, saved with its
      distribution (`mantle_balanced_<ranks>`);
    - the earth fields and the forcing fields of the mantle (`MANTLE_FIELDS`)
      as copies on the parent mesh;
    - the ice of every step of the inversion march (`ice`, one entry per
      step, `idx` = step index), as copies on the parent mesh;
    - the JSON record of the prepare run (`/giamip_inputs/record`).

    `CheckpointFile` stores one mesh per function and knows nothing of
    `Submesh`, so a field on the mantle submesh goes into the file as its
    copy on the parent mesh, in the broken (cell-wise discontinuous) version
    of its element. The copy follows the submesh map cell by cell, so the
    round trip mantle -> parent -> mantle returns the same nodal values. The
    broken element matters for a continuous field: the parent nodes on the
    CMB and on Re are shared with cells outside the mantle, and a continuous
    parent space would take the value of whichever cell came last there.

    The state file of an inversion is a copy of the input file. The
    inversion appends the bed of each ROL iteration to it as a continuous
    CG2 field on the parent (`append_control`), an exact copy of the mantle
    nodes made with `Function.assign` (see there).

    A mesh saved with a custom partition loads with that partition only on
    the same number of ranks. The prepare job and the inversion job must
    therefore run on the same number of ranks; `read_inputs` stops with an
    error otherwise.
"""

import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

# gadopt before firedrake: gadopt re-exports the Firedrake names.
from gadopt import (COMM_WORLD, CheckpointFile,  # noqa: E402
                    CompressibleInternalVariableApproximation, Constant,
                    Cofunction, DirichletBC, Function, FunctionSpace, Mesh,
                    SpatialCoordinate, SphericalDtN, Submesh,
                    VectorFunctionSpace, assemble, ds, dx, interpolate, log)
from gadopt.gia_gravity import (FluidCore, SeaLevel,  # noqa: E402
                                SelfGravitatingGIASolver,
                                rigid_rotation_nullspace,
                                self_gravitating_gia_space,
                                selfgrav_dtn_iterative_solver_parameters)
from gadopt.sea_level_masks import (DEFAULT_ALPHA_MASK,  # noqa: E402
                                    DEFAULT_GRAD_FLOOR)
from finat.ufl import BrokenElement  # noqa: E402
from firedrake.adjoint_utils.checkpointing import \
    maybe_disk_checkpoint  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from pyadjoint import Block  # noqa: E402
from pyop2 import op2  # noqa: E402
from pyadjoint.tape import (annotate_tape, get_working_tape,  # noqa: E402
                            stop_annotating)
from scipy.spatial import cKDTree  # noqa: E402

from giamsh import refine as giamsh_refine  # noqa: E402

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
#: The viscosity that stands for an elastic layer ("inf" in the giamip
#: model files), Pa s. 1e40 Pa s gives a Maxwell time of about 2e19 t_bar,
#: far beyond any run.
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
#: The degree of `SphericalDtN` on the two truncation spheres.
DTN_DEGREE = 5
#: The representation of the DtN operators in the coupled solver.
DTN_REPRESENTATION = "lowrank"
#: The tolerances of the outer FGMRES and of block 0.
OUTER_RTOL = 1.0e-6
BLOCK0_RTOL = 1.0e-4

#: The names of the meshes. They are the same in every experiment
#: directory, so the input files of two experiments differ only in their
#: data.
PARENT_NAME = "giamip_parent"
MANTLE_NAME = "giamip_mantle"
SURFACE_NAME = "giamip_surface"

#: The HDF5 group of the record of the prepare run in the input file.
INPUT_GROUP = "/giamip_inputs"

#: The fields of the mantle in the input file: name -> (family, degree) of
#: the space on the mantle mesh, and the meaning of the field. All values
#: are non-dimensional (see `Scales`).
MANTLE_FIELDS = {
    # Density of PREM-stable, DG1, in units of rho_bar.
    "density": ("DG", 1),
    # Shear modulus, DG1, in units of mu_bar.
    "shear_modulus": ("DG", 1),
    # Bulk modulus, DG1, in units of mu_bar.
    "bulk_modulus": ("DG", 1),
    # Viscosity at the cell centroid, DG0, in units of eta_bar.
    "viscosity": ("DG", 0),
    # Gravity of PREM-stable, CG2, in units of g_bar (continuous, because
    # the prestress terms take its gradient).
    "gravity": ("CG", 2),
    # The observed sea level -bed_obs / D, CG2 (the mask width reads its
    # slope). Only its values on Re are used.
    "SL_obs": ("CG", 2),
    # The frozen surface slope of the ocean mask, |grad_t SL_obs|, DG1
    # (`gadopt.sea_level_masks.surface_slope`), dimensionless.
    "slope": ("DG", 1),
}
#: The ice of each step: thickness in units of D, CG1 on the mantle. CG1,
#: because a CG2 interpolant of a field that is zero next to a thick ice
#: margin undershoots below zero between nodes, and a negative thickness is
#: a negative load.
ICE_NAME = "ice"
ICE_SPACE = ("CG", 1)

#: The optional fields of the mantle: name -> (family, degree, record key).
#: A field is written and read only when the record of the input file has
#: its key, so an input file of a Maxwell experiment (no key) holds and
#: loads exactly `MANTLE_FIELDS`. The rheology of Exp09, Exp10 and Exp12
#: (design handout of 2026-10-20, sections 12.4 and 12.5):
#:
#: - `n_elements` (an integer N): the Maxwell elements in parallel that
#:   represent the Extended Burgers fit. `shear_moduli` is a DG1 vector
#:   field of N components (units of mu_bar) and `viscosities` a DG0 vector
#:   field of N components (units of eta_bar); component j is element j.
#:   `shear_modulus` and `viscosity` stay in the file as the Maxwell
#:   reference (the modulus of the Maxwell twin, mu_inf, and the long-term
#:   viscosity eta_0).
#: - `power_law` (a dict, see `power_law_settings`): the power-law exponent
#:   n, a DG0 field that is exactly 1.0 or exactly the exponent of the
#:   record in each cell, and the transition stress, a DG0 field in units
#:   of mu_bar, positive in every cell (`power_law_factor` divides by it
#:   also where n = 1).
OPTIONAL_FIELDS = {
    "shear_moduli": ("DG", 1, "n_elements"),
    "viscosities": ("DG", 0, "n_elements"),
    "exponent": ("DG", 0, "power_law"),
    "transition_stress": ("DG", 0, "power_law"),
}


def field_names(record):
    """The names of the mantle fields of an input file with this record.

    `MANTLE_FIELDS`, then every name of `OPTIONAL_FIELDS` whose record key
    is in `record`, in the order of `OPTIONAL_FIELDS`.
    """
    return list(MANTLE_FIELDS) + [name for name, (_, _, key)
                                  in OPTIONAL_FIELDS.items() if key in record]


def field_space(mantle, name, record):
    """The space of the mantle field `name` of an input file with `record`.

    A scalar space for the names of `MANTLE_FIELDS` and for the power-law
    fields; a vector space of `record["n_elements"]` components for
    `shear_moduli` and `viscosities`.
    """
    if name in MANTLE_FIELDS:
        return FunctionSpace(mantle, *MANTLE_FIELDS[name])
    family, degree, key = OPTIONAL_FIELDS[name]
    if key == "n_elements":
        return VectorFunctionSpace(mantle, family, degree,
                                   dim=int(record["n_elements"]))
    return FunctionSpace(mantle, family, degree)


def power_law_settings(record):
    """The power-law entry of a record, checked; None without a power law.

    The entry is written by the prepare scripts of Exp10 and Exp12:
    `{"exponent": 3.5, "background_stress": 0.0, ...}`, with the exponent
    of the power-law cells (the other cells have exactly 1.0) and the
    background stress in units of mu_bar. Further keys (the rule, the
    transition stress in MPa) are a record for the reader and are not used
    here.

    Raises:
      ValueError: if the exponent is missing or not above 1, or the
        background stress is negative.
    """
    settings = record.get("power_law")
    if settings is None:
        return None
    exponent = float(settings["exponent"])
    background = float(settings.get("background_stress", 0.0))
    if not exponent > 1.0 or background < 0.0:
        raise ValueError(f"power_law record {settings}: the exponent must be "
                         "above 1 and the background stress not negative")
    return {"exponent": exponent, "background_stress": background}


# --------------------------------------------------------------------------
# Scales
# --------------------------------------------------------------------------


class Scales:
    """The non-dimensional scales of the run.

    length       D = 2891 km, the mantle thickness of PREM-stable (also the
                 unit of the giamsh mesh)
    gravity      g_bar = g(Re) of PREM-stable, so g_s = 1 in the scaled
                 equations
    density      rho_bar = 3 g_bar / (4 pi G Re), the mean density that
                 gives that surface gravity
    stress       mu_bar = 1e11 Pa
    viscosity    eta_bar = 1e21 Pa s, time t_bar = eta_bar / mu_bar = 1e10 s

    Lambda = 4 pi G rho_bar D / g_bar = 3 D / Re, so the scaled Poisson
    equation and the scaled gravity of the file describe the same mass.

    The prepare script builds the scales from the elastic structure of the
    experiment (`from_elastic`) and writes `as_dict` into the record. The
    inversion builds them again from the record (`from_dict`), so it does
    not read the earth model.

    Args:
      g_bar: gravity just below the surface, m/s^2.
      Re_m: the surface radius, m.
      Rc_m: the core-mantle boundary radius, m.

    Attributes:
      g_bar: g at the surface, m/s^2.
      rho_bar: the mean density that gives `g_bar`, kg/m^3.
      Re, Rc: the surface and CMB radii, non-dimensional.
      B_mu: rho_bar g_bar D / mu_bar.
      Lambda: 4 pi G rho_bar D / g_bar.
      Omega_sq: Omega^2 D / g_bar.
      earth_mass_kg: the mass that gives `g_bar`, kg.
    """

    def __init__(self, g_bar, Re_m, Rc_m):
        self.Re_m = float(Re_m)
        self.Rc_m = float(Rc_m)
        self.g_bar = float(g_bar)
        # The mass inside Re that gives g_bar at Re, and its mean density.
        self.earth_mass_kg = self.g_bar * self.Re_m**2 / G_NEWTON
        self.rho_bar = self.earth_mass_kg / (4.0 / 3.0 * np.pi * self.Re_m**3)
        self.Re = self.Re_m / D_SCALE
        self.Rc = self.Rc_m / D_SCALE
        # The buoyancy number and the self-gravity number of the scaled
        # equations.
        self.B_mu = self.rho_bar * self.g_bar * D_SCALE / MU_BAR
        self.Lambda = 4 * np.pi * G_NEWTON * self.rho_bar * D_SCALE / self.g_bar
        # The centrifugal number of the rotational potential.
        self.Omega_sq = OMEGA**2 * D_SCALE / self.g_bar

    @classmethod
    def from_elastic(cls, elastic):
        """The scales of a `giamip.earth.ElasticProfile`.

        Gravity is taken just below the surface; the surface is the last
        node of the profile.
        """
        Re_m = elastic.surface_radius_m
        g_bar = float(elastic(Re_m, "below")["g_m_s2"])
        return cls(g_bar, Re_m, elastic.core_mantle_boundary_m)

    @classmethod
    def from_dict(cls, record):
        """The scales of the record written by `as_dict`."""
        return cls(record["g_bar"], record["Re_m"], record["Rc_m"])

    def as_dict(self):
        """The scales as plain numbers, for the record."""
        return {"D_m": D_SCALE, "g_bar": self.g_bar, "rho_bar": self.rho_bar,
                "mu_bar": MU_BAR, "eta_bar": ETA_BAR, "t_bar_yr": T_BAR_YR,
                "G": G_NEWTON, "Re_m": self.Re_m, "Rc_m": self.Rc_m,
                "Re": self.Re, "Rc": self.Rc, "B_mu": self.B_mu,
                "Lambda": self.Lambda, "Omega_sq": self.Omega_sq,
                "earth_mass_kg": self.earth_mass_kg}


# --------------------------------------------------------------------------
# The mesh
# --------------------------------------------------------------------------

#: The bits per axis of the Hilbert curve that orders the cell directions in
#: `mantle_balanced_partition`. 2^10 = 1024 steps across the unit cube give
#: a step of 2 / 1023 in each component of the direction, 12 km at Re, well
#: below the size of the coarse cells.
HILBERT_BITS = 10


def hilbert_keys(ijk, bits):
    """The index of integer points along a 3-D Hilbert curve.

    The algorithm is the transpose form of Skilling (2004), "Programming the
    Hilbert curve", AIP Conf. Proc. 707, 381, applied to all points at once.
    It undoes the excess work, Gray-codes the result, and interleaves the
    bits of the three axes into one key. Points that are close along the
    curve are close in space, so an equal split of the sorted keys gives
    compact pieces.

    Args:
      ijk: an integer array of shape `(n, 3)` with values in `[0, 2**bits)`.
      bits: the bits per axis. The key has `3 * bits` bits and must fit in
        int64, so `bits` is at most 21.

    Returns:
      An int64 array of `n` keys.
    """
    X = np.array(ijk, dtype=np.int64)
    M = np.int64(1) << (bits - 1)
    # Inverse undo of the excess work, from the top bit down.
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(3):
            high = (X[:, i] & Q) != 0
            # Where the bit is set, invert the low bits of the first axis.
            X[high, 0] ^= P
            # Elsewhere, exchange the low bits of the first axis and axis i.
            low = ~high
            t = (X[low, 0] ^ X[low, i]) & P
            X[low, 0] ^= t
            X[low, i] ^= t
        Q >>= 1
    # Gray encode.
    for i in range(1, 3):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(len(X), dtype=np.int64)
    Q = M
    while Q > 1:
        t[(X[:, 2] & Q) != 0] ^= Q - 1
        Q >>= 1
    X ^= t[:, None]
    # Interleave: the top bit of each axis first, axis 0 before axis 1.
    key = np.zeros(len(X), dtype=np.int64)
    for b in range(bits - 1, -1, -1):
        for i in range(3):
            key = (key << 1) | ((X[:, i] >> b) & 1)
    return key


def mantle_balanced_partition(plex, nparts, mantle_tags):
    """A partition that gives every rank the same number of mantle cells.

    The mantle of the giamsh mesh is several cell tags (one per layer). Each
    cell is ordered by the Hilbert key of the direction of its centroid. The
    mantle cells, in that order, are split into `nparts` pieces of equal
    count, and every other cell goes to the rank whose mantle piece covers
    its key. Every rank then owns a solid-angle sector through all shells,
    and the mantle cell counts differ by at most one. The mantle carries
    most of the work (displacement, internal variables and potential), so
    this balances the solver.

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
    # The direction of the centroid of every cell, from its vertices.
    section = plex.getCoordinateSection()
    coordinates = plex.getCoordinatesLocal()
    centroid = np.array([plex.vecGetClosure(section, coordinates, c)
                         .reshape(-1, 3).mean(axis=0)
                         for c in range(cstart, cend)])
    direction = centroid / np.linalg.norm(centroid, axis=1)[:, None]
    # Directions in [-1, 1]^3 mapped onto the integer grid of the curve.
    scale = (1 << HILBERT_BITS) - 1
    key = hilbert_keys(np.rint((direction + 1.0) * 0.5 * scale), HILBERT_BITS)
    mantle = np.zeros(cend - cstart, dtype=bool)
    for tag in mantle_tags:
        points = plex.getStratumIS("Cell Sets", tag).getIndices()
        mantle[np.asarray(points) - cstart] = True
    part = np.empty(cend - cstart, dtype=np.int64)
    # Equal counts of mantle cells along the curve.
    mantle_index = np.flatnonzero(mantle)
    order = mantle_index[np.argsort(key[mantle_index], kind="stable")]
    part[order] = (np.arange(len(order)) * nparts) // len(order)
    # Every other cell goes to the rank whose key range covers its key.
    first = np.searchsorted(part[order], np.arange(nparts), side="left")
    starts = key[order][first]
    other = np.flatnonzero(~mantle)
    part[other] = np.clip(
        np.searchsorted(starts, key[other], side="right") - 1, 0, nparts - 1)
    sizes = np.bincount(part, minlength=nparts).astype(PETSc.IntType)
    points = (np.argsort(part, kind="stable") + cstart).astype(PETSc.IntType)
    return sizes, points


def distribution_name():
    """The name of the mesh distribution on this number of ranks.

    `CheckpointFile.save_mesh` hangs in `DMPlex.topologyView` for a mesh
    built with a custom partition and no distribution name, so the name is
    always set. It names the rank count, because the partition belongs to it.
    """
    return f"mantle_balanced_{COMM_WORLD.size}"


def build_meshes(msh, radial, n_levels):
    """Read, partition, refine and curve the parent mesh; cut the mantle.

    Args:
      msh: the coarse gmsh file.
      radial: the `giamsh.radial.RadialStructure` of its sidecar.
      n_levels: uniform refinement levels (0 for L0).

    Returns:
      `(parent, mantle, counts)`, with `counts` from `mesh_counts`.
    """
    mantle_tags = list(radial.region_tags("mantle"))
    plex = PETSc.DMPlex().createFromFile(msh, comm=COMM_WORLD)
    plex.setName("giamip_topology")
    sizes, points = mantle_balanced_partition(plex, COMM_WORLD.size,
                                              mantle_tags)
    linear = Mesh(plex, distribution_parameters={"partition": (sizes, points)},
                  distribution_name=distribution_name(), name="giamip_linear")
    # The sag check of giamsh raises on this coarse mesh (cmb, buffer_2 and
    # outer), and the refinement of the mesh campaign ran without it.
    linear = giamsh_refine.refine_uniform(linear, radial, n_levels,
                                          curve=False, check=False)
    parent = giamsh_refine.curve_mesh(linear, name=PARENT_NAME)
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
    mantle = mantle_mesh(parent, mantle_tags)
    counts = mesh_counts(parent, mantle)
    counts["folded_cells_parent"] = giamsh_refine.folded_cells(parent)
    counts["folded_cells_mantle"] = giamsh_refine.folded_cells(mantle)
    if counts["folded_cells_parent"] or counts["folded_cells_mantle"]:
        raise RuntimeError(f"folded cells after the curving: {counts}")
    return parent, mantle, counts


#: The bits per axis of the Hilbert curve of `checkpoint_balanced_partition`.
#: The locally refined meshes have cells of 2.5 km at the coast, much finer
#: than the 12 km step of `HILBERT_BITS`. 2^16 steps give 0.2 km at Re, so
#: few cells share one key. The key has 48 bits, which fits in int64.
CHECKPOINT_HILBERT_BITS = 16

#: The DMPlex labels that a saved topology carries and that depend on the
#: distribution of the run that saved it. Firedrake's own `CheckpointFile`
#: load removes the `pyop2_*` labels; `firedrake_is_ghost` is the label of
#: the same kind that Firedrake with pyop3 writes (the 8 Re mesh was saved
#: by that version). Firedrake makes these labels again for the new
#: distribution.
DISTRIBUTION_LABELS = ("pyop2_core", "pyop2_owned", "pyop2_ghost",
                       "firedrake_is_ghost")


def checkpoint_topology_name(path, mesh_name):
    """The name of the DMPlex topology of mesh `mesh_name` in a `CheckpointFile`.

    Firedrake stores a pickled `{mesh name: topology name}` dict as the
    attribute `firedrake_mesh_name_topology_name_map` of the group
    `topologies`. Read on rank 0 with h5py and broadcast.

    Args:
      path: the `.h5` file of `CheckpointFile.save_mesh`.
      mesh_name: the name of the mesh in that file.

    Returns:
      The topology name, the name of the group under `topologies/`.
    """
    import pickle

    import h5py

    name = None
    if COMM_WORLD.rank == 0:
        with h5py.File(path, "r") as f:
            raw = f["topologies"].attrs["firedrake_mesh_name_topology_name_map"]
            name = pickle.loads(raw.tobytes())[mesh_name]
    return COMM_WORLD.bcast(name, root=0)


def checkpoint_balanced_partition(plex, nparts, mantle_tags):
    """The partition of `mantle_balanced_partition`, for a DMPlex loaded in parallel.

    `mantle_balanced_partition` needs the whole mesh on rank 0, as the gmsh
    reader gives it. A DMPlex loaded from a `CheckpointFile` with
    `topologyLoad` is spread over all ranks instead: each rank holds a
    contiguous piece of the cells of the file, with the vertices of its cells
    (this is the "naive" distribution that Firedrake's own load
    redistributes). This function makes the same partition from that state:

    1. Each rank computes the Hilbert key of the centroid direction of each
       of its cells (`CHECKPOINT_HILBERT_BITS` bits per axis) and marks the
       mantle cells.
    2. Rank 0 gathers the keys of all mantle cells (13.7 M int64 values for
       the 8 Re mesh, about 110 MB), sorts them and splits the sorted list
       into `nparts` pieces of equal count. A stable sort over the gathered
       order makes the split deterministic. Rank 0 sends each rank the part
       of each of its mantle cells, and sends every rank the first key of
       each part.
    3. Every other cell (buffer, inner shell) goes to the part whose key
       range covers its key, as in `mantle_balanced_partition`.

    Each part is then a solid-angle sector through all shells, and the
    mantle cell counts of the parts differ by at most one. The buffer cells
    are not balanced. They hold only the gravitational potential, so they
    cost much less per cell than the mantle cells.

    Collective.

    Args:
      plex: the DMPlex from `topologyLoad`, `labelsLoad` and
        `coordinatesLoad`, before any distribution.
      nparts: the number of ranks.
      mantle_tags: the cell tags of the mantle region.

    Returns:
      `(sizes, points)` for `distribution_parameters["partition"]`: on each
      rank, the number of its cells that go to each part and the local
      indices of its cells (0 for the first cell of the rank) sorted by
      part. PETSc's shell partitioner takes these per rank when the DMPlex
      is distributed.

    Raises:
      RuntimeError: if a rank holds a cell that it does not own. The shell
        partition must list every cell of the rank once.
    """
    cstart, cend = plex.getHeightStratum(0)
    ncells = cend - cstart
    # A cell that is a leaf of the point SF belongs to another rank. The
    # loaded DMPlex has none, but the shell partition would be wrong with
    # one, so check.
    _, leaves, _ = plex.getPointSF().getGraph()
    leaves = np.asarray(leaves if leaves is not None else [], dtype=np.int64)
    ghost_cells = int(np.count_nonzero((leaves >= cstart) & (leaves < cend)))
    if COMM_WORLD.allreduce(ghost_cells, op=MPI.SUM):
        raise RuntimeError("checkpoint_balanced_partition: the loaded DMPlex "
                           "has cells owned by another rank")
    # 1. The direction of the centroid of every cell, from its vertices.
    # The local coordinate vector holds the vertices of every local cell,
    # also those owned by another rank.
    section = plex.getCoordinateSection()
    coordinates = plex.getCoordinatesLocal()
    centroid = np.array([plex.vecGetClosure(section, coordinates, c)
                         .reshape(-1, 3).mean(axis=0)
                         for c in range(cstart, cend)]).reshape(ncells, 3)
    direction = centroid / np.linalg.norm(centroid, axis=1)[:, None]
    bits = CHECKPOINT_HILBERT_BITS
    scale = (1 << bits) - 1
    key = hilbert_keys(np.rint((direction + 1.0) * 0.5 * scale), bits)
    mantle = np.zeros(ncells, dtype=bool)
    for tag in mantle_tags:
        if plex.getStratumSize("Cell Sets", tag) > 0:
            points = np.asarray(plex.getStratumIS("Cell Sets", tag)
                                .getIndices())
            # The refinement of the 8 Re mesh passed the cell tags on to the
            # faces, edges and vertices of each cell (77 M labelled points
            # for 22 M cells), so only the points of the cell stratum count.
            points = points[(points >= cstart) & (points < cend)]
            mantle[points - cstart] = True
    mantle_index = np.flatnonzero(mantle)
    mantle_keys = np.ascontiguousarray(key[mantle_index], dtype=np.int64)

    # 2. The equal split of the sorted mantle keys, on rank 0.
    counts = np.array(COMM_WORLD.allgather(len(mantle_keys)), dtype=np.int64)
    total = int(counts.sum())
    displacements = np.concatenate([[0], np.cumsum(counts)[:-1]])
    gathered = (np.empty(total, dtype=np.int64) if COMM_WORLD.rank == 0
                else None)
    COMM_WORLD.Gatherv(mantle_keys,
                       [gathered, counts, displacements, MPI.INT64_T]
                       if COMM_WORLD.rank == 0 else None, root=0)
    if COMM_WORLD.rank == 0:
        order = np.argsort(gathered, kind="stable")
        part_all = np.empty(total, dtype=np.int64)
        part_all[order] = (np.arange(total, dtype=np.int64) * nparts) // total
        # The first key of each part, in key order.
        first = np.searchsorted(part_all[order], np.arange(nparts),
                                side="left")
        starts = gathered[order][first]
        del order
    else:
        part_all, starts = None, None
    part_mantle = np.empty(len(mantle_keys), dtype=np.int64)
    COMM_WORLD.Scatterv([part_all, counts, displacements, MPI.INT64_T]
                        if COMM_WORLD.rank == 0 else None,
                        part_mantle, root=0)
    starts = COMM_WORLD.bcast(starts, root=0)
    del part_all, gathered

    # 3. Every other cell goes to the part whose key range covers its key.
    part = np.empty(ncells, dtype=np.int64)
    part[mantle_index] = part_mantle
    other = np.flatnonzero(~mantle)
    part[other] = np.clip(
        np.searchsorted(starts, key[other], side="right") - 1, 0, nparts - 1)
    sizes = np.bincount(part, minlength=nparts).astype(PETSc.IntType)
    # PETSc maps the entries of a shell partition through the local cell
    # numbering (`PetscPartitionerDMPlexPartition`: entry i is the i-th owned
    # cell of the height stratum), so the entries count from 0 and not from
    # `cstart`. For the serial gmsh DMPlex of `mantle_balanced_partition`,
    # `cstart` is 0 and the two agree.
    points = np.argsort(part, kind="stable").astype(PETSc.IntType)
    return sizes, points


def load_checkpoint_meshes(path, mesh_name, radial):
    """Load a refined P1 mesh from a `CheckpointFile`, balance the mantle, curve it.

    The locally refined GIAMIP meshes (for example `giamip_8re_r4`, NOTES
    handout of 2026-10-23) are refined already and come as `CheckpointFile`
    files, not as gmsh files. `build_meshes` cannot read them. This function
    gives the same result as `build_meshes` with no refinement:

    1. Load the DMPlex topology, its labels and its vertex coordinates from
       the file, as `CheckpointFile._load_mesh_topology` does, but without
       the Firedrake mesh: the DMPlex is spread over the ranks in contiguous
       pieces of the file. The labels that belong to the distribution of the
       saving run are removed (`DISTRIBUTION_LABELS`).
    2. Make the mantle-balanced partition from that state
       (`checkpoint_balanced_partition`) and build the Firedrake mesh with
       it. Firedrake distributes the DMPlex with PETSc's shell partitioner.
       A default load would balance all cells, and the 8.4 M buffer cells of
       the 8 Re mesh would leave the ranks with very different mantle work.
    3. Curve the mesh with `giamsh.refine.curve_mesh`, cut and curve the
       mantle submesh (`mantle_mesh`), and check for folded cells.

    The vertex coordinates of the file must be snapped to the spheres (the
    P1 file of the mesh build is), because `curve_mesh` moves each P2 node
    to the radius of the P1 interpolant of the vertex radii. This is the
    rule that made the P2 file of the same build, so the curved parent
    equals the P2 file of the build at every node.

    Collective.

    Args:
      path: the P1 mesh file.
      mesh_name: the name of the mesh in the file.
      radial: the `giamsh.radial.RadialStructure` of the sidecar.

    Returns:
      `(parent, mantle, counts)` as `build_meshes`.
    """
    mantle_tags = list(radial.region_tags("mantle"))
    topology_name = checkpoint_topology_name(path, mesh_name)
    viewer = PETSc.ViewerHDF5().create(path, mode="r", comm=COMM_WORLD)
    plex = PETSc.DMPlex().create(comm=COMM_WORLD)
    plex.setName(topology_name)
    viewer.pushFormat(format=PETSc.Viewer.Format.HDF5_PETSC)
    plex.distributionSetName(None)
    sfXB = plex.topologyLoad(viewer)
    plex.labelsLoad(viewer, sfXB)
    plex.coordinatesLoad(viewer, sfXB)
    viewer.popFormat()
    viewer.destroy()
    for label in DISTRIBUTION_LABELS:
        if plex.hasLabel(label):
            plex.removeLabel(label)
    sizes, points = checkpoint_balanced_partition(plex, COMM_WORLD.size,
                                                  mantle_tags)
    linear = Mesh(plex, distribution_parameters={"partition": (sizes, points)},
                  distribution_name=distribution_name(), name="giamip_linear")
    parent = giamsh_refine.curve_mesh(linear, name=PARENT_NAME)
    mantle = mantle_mesh(parent, mantle_tags)
    counts = mesh_counts(parent, mantle)
    counts["folded_cells_parent"] = giamsh_refine.folded_cells(parent)
    counts["folded_cells_mantle"] = giamsh_refine.folded_cells(mantle)
    if counts["folded_cells_parent"] or counts["folded_cells_mantle"]:
        raise RuntimeError(f"folded cells after the curving: {counts}")
    return parent, mantle, counts


#: The DMPlex label that marks every mantle cell with the value 1.
MANTLE_LABEL = "giamip_mantle_cells"


def mantle_mesh(parent, mantle_tags):
    """The curved mantle submesh of the parent mesh.

    The mantle is several cell tags, one per layer. `Submesh(parent, 3,
    tags)` with a list merges the cells of each tag with `IS.union`, and on
    a rank that holds cells of one tag and none of a later tag that merge
    works on an empty index set without a communicator and stops with
    "Invalid communicator" in `ISSorted` (Firedrake `dmcommon.submesh_create`).
    All mantle cells therefore go into one label of their own, and `Submesh`
    takes that one value: a single value needs no merge. Halo cells are
    labelled too, as `Cell Sets` labels them.

    The submesh has straight-sided cells; `giamsh.refine.curve_mesh` curves
    it with the rule that curved the parent, so the two geometries agree at
    every node.

    Args:
      parent: the parent mesh, built by `build_meshes` or loaded from the
        input file.
      mantle_tags: the cell tags of the mantle region.

    Returns:
      The mantle mesh (P2 coordinates), named `MANTLE_NAME`.
    """
    dm = parent.topology_dm
    if not dm.hasLabel(MANTLE_LABEL):
        dm.createLabel(MANTLE_LABEL)
    label = dm.getLabel(MANTLE_LABEL)
    # Only cells go into the label. The gmsh meshes tag cells only; the
    # locally refined 8 Re mesh tags the faces, edges and vertices of each
    # cell too.
    cstart, cend = dm.getHeightStratum(0)
    for tag in mantle_tags:
        if dm.getStratumSize("Cell Sets", tag) > 0:
            for point in dm.getStratumIS("Cell Sets", tag).getIndices():
                if cstart <= point < cend:
                    label.setValue(point, 1)
    submesh = Submesh(parent, 3, 1, label_name=MANTLE_LABEL)
    return giamsh_refine.curve_mesh(submesh, name=MANTLE_NAME)


def mesh_counts(parent, mantle):
    """Global cell counts and the spread of the mantle and parent cells over the ranks."""
    local_mantle = mantle.cell_set.size
    counts = COMM_WORLD.allgather(local_mantle)
    parent_counts = COMM_WORLD.allgather(parent.cell_set.size)
    return {"cells_parent": int(sum(parent_counts)),
            "cells_mantle": int(sum(counts)),
            "mantle_cells_per_rank_min": int(min(counts)),
            "mantle_cells_per_rank_max": int(max(counts)),
            "parent_cells_per_rank_min": int(min(parent_counts)),
            "parent_cells_per_rank_max": int(max(parent_counts)),
            "ranks": COMM_WORLD.size}


def surface_mesh(mantle):
    """The surface Re as a codim-1 `Submesh` of the mantle.

    The submesh takes its coordinates from the DMPlex of the mantle, which
    holds the vertex positions only. Its geometry is therefore P1 (flat
    triangles), not the P2 geometry of the curved mantle: the vertices are
    the vertices of the mantle, but the edge midpoints of the surface lie on
    the chords, below the curved midpoint nodes of the mantle on Re. The
    control and its Riesz map live on this P1 surface. Fields move between
    the surface and the mantle only through `SurfaceNodeMap`, which matches
    the nodes and never evaluates a field at a point.
    """
    return Submesh(mantle, 2, SURF_RE, name=SURFACE_NAME)


# --------------------------------------------------------------------------
# The exact node map between the surface and the mantle
# --------------------------------------------------------------------------


class SurfaceNodeMap:
    r"""The exact map between CG2 on the surface submesh and CG2 on the mantle.

    WHY THIS EXISTS
        Firedrake interpolates between a mesh and its submesh across the
        submesh map only when both have the same topological dimension. The
        surface is a codim-1 submesh of the mantle, so `interpolate` between
        the two falls back to the cross-mesh interpolator: it locates every
        target node in the source mesh by a point search and evaluates the
        source field there. The surface has P1 geometry and the mantle P2
        geometry (see `surface_mesh`), so the curved midpoint nodes of the
        mantle lie outside the flat surface triangles, and the flat midpoint
        nodes of the surface lie inside the mantle cells. Both directions
        then return values that are not the nodal values (an extrapolation
        of the surface field, or a blend with the interior nodes of the
        mantle), and a point near a sharp turn of the surface can land in a
        wrong cell. The round trip mantle -> surface -> mantle is then not
        the identity, and the solver does not see the bed of the control.

    WHAT IT DOES
        The CG2 nodes of the surface are the CG2 nodes of the mantle on Re.
        This class pairs them once, by position, and stores the pairing as a
        sparse matrix `R` of zeros and ones in the global numbering of the
        two spaces: one row per surface node, with a single 1 in the column
        of its mantle node. Then

            restrict:  b = R m       (mantle -> surface, a copy of the
                                      values at the nodes on Re)
            extend:    m = R^T b     (surface -> mantle, a copy onto the
                                      nodes on Re, zero at every other node)

        Both are exact: the product of a value with 1.0 and a sum with zeros
        are exact in floating point, so `restrict(extend(b)) == b` bit for
        bit. `extend` is linear, so its adjoint is `R` applied to the
        coefficients of the adjoint cofunction (`SurfaceExtendBlock`).

    THE PAIRING
        The two meshes share the vertex positions, so the nodes are matched
        in the P1 geometry of both: the surface nodes from the surface
        coordinates, which are P1, and the mantle nodes from the P1
        interpolant of the mantle coordinates (the curving of
        `giamsh.refine.curve_mesh` does not move the vertices). The match
        is a nearest-neighbour search per rank among the mantle nodes on Re.
        Every local surface node must find a mantle node within a round-off
        distance, and no two surface nodes may share one; otherwise the
        constructor raises. A local surface facet is always a local mantle
        facet, so every surface node of a rank, owned or halo, is a node of
        the mantle on that rank. Each rank inserts the rows of the surface
        nodes it owns; the column may be a mantle node of another rank,
        which the global numbering handles.

    Args:
      V_surface: CG2 on the surface submesh.
      V_mantle: CG2 on the mantle mesh, the parent of the surface submesh.
      boundary: the facet tag of the surface in the mantle mesh.

    Raises:
      RuntimeError: if a surface node has no mantle node at its position,
        if two surface nodes share a mantle node, or if a mantle node on Re
        has no surface node.
    """

    def __init__(self, V_surface, V_mantle, boundary=SURF_RE):
        self.V_surface = V_surface
        self.V_mantle = V_mantle
        surface = V_surface.mesh()
        mantle = V_mantle.mesh()
        comm = mantle.comm

        # The P1 positions of the CG2 nodes of the surface: the surface
        # coordinates are P1, so the CG2 interpolant of the coordinates
        # holds the vertices and the chord midpoints.
        x_surface = Function(VectorFunctionSpace(surface, "CG", 2)).interpolate(
            SpatialCoordinate(surface)).dat.data_ro_with_halos
        # The P1 positions of the CG2 nodes of the mantle: the curved
        # coordinates evaluated at the vertices (CG1, exact at the vertices),
        # then the CG2 interpolant of that P1 field (vertices and chord
        # midpoints, as on the surface).
        vertices = Function(VectorFunctionSpace(mantle, "CG", 1)).interpolate(
            mantle.coordinates)
        x_mantle = Function(VectorFunctionSpace(mantle, "CG", 2)).interpolate(
            vertices).dat.data_ro_with_halos
        # The local mantle nodes on Re, owned and halo.
        on_boundary = np.asarray(DirichletBC(V_mantle, 0.0, boundary).nodes)

        n_surface = len(x_surface)
        bad = 0
        if n_surface:
            tree = cKDTree(x_mantle[on_boundary])
            distance, nearest = tree.query(x_surface)
            # Round-off distance: the coordinates are O(1) in units of D,
            # and the node spacing is many orders of magnitude larger.
            tolerance = 1e-9 * max(1.0, float(np.abs(x_surface).max()))
            bad += int(np.sum(distance > tolerance))
            bad += n_surface - len(np.unique(nearest))
            mantle_node = on_boundary[nearest]
        else:
            mantle_node = np.zeros(0, dtype=np.int64)
        bad = comm.allreduce(bad, op=MPI.SUM)
        if bad:
            raise RuntimeError(
                f"SurfaceNodeMap: {bad} surface nodes have no unique mantle "
                "node at their position")

        # The matrix R, rows = surface nodes, columns = mantle nodes, in
        # the global numbering of the PETSc vectors of the two spaces. A
        # scalar CG2 space has block size 1, so the local-to-global map of
        # the node set is the map of the vector entries. Firedrake numbers
        # the owned nodes of a rank before its halo nodes.
        n_owned = V_surface.dof_dset.size
        rows = V_surface.dof_dset.lgmap.apply(
            np.arange(n_owned, dtype=PETSc.IntType))
        cols = V_mantle.dof_dset.lgmap.apply(
            mantle_node[:n_owned].astype(PETSc.IntType))
        R = PETSc.Mat().createAIJ(
            size=((n_owned, PETSc.DETERMINE),
                  (V_mantle.dof_dset.size, PETSc.DETERMINE)),
            nnz=(1, 1), comm=comm)
        for row, col in zip(rows, cols):
            R.setValue(row, col, 1.0)
        R.assemble()
        self.R = R

        # Coverage: R^T 1 must be 1 at every mantle node on Re and 0
        # elsewhere. A value above 1 would mean two surface nodes on one
        # mantle node; a 0 on Re would mean a mantle node that no surface
        # node reaches (the solver would read a zero bed there).
        ones = Function(V_surface).assign(1.0)
        hits = Function(V_mantle)
        self._apply_transpose(ones, hits)
        # The indicator of the mantle nodes on Re. A rank finds the nodes on
        # Re through its local facets, and an owned node of a rank can lie
        # on Re facets that only another rank holds. Each rank therefore
        # marks every local node on Re, owned or halo, and the halo entries
        # are sent to the owners with a MAX reduction.
        expected = Function(V_mantle)
        expected.dat.data_wo_with_halos[on_boundary] = 1.0
        expected.dat.local_to_global_begin(op2.MAX)
        expected.dat.local_to_global_end(op2.MAX)
        with hits.dat.vec_ro as h, expected.dat.vec_ro as e:
            mismatch = (h - e).norm(PETSc.NormType.INFINITY)
        if mismatch != 0.0:
            raise RuntimeError(
                "SurfaceNodeMap: the surface nodes do not cover the mantle "
                f"nodes on Re one to one (largest mismatch {mismatch})")

    def _apply(self, source, out, scale=1.0):
        """`out = scale R source` on the coefficient vectors (no annotation).

        `source` lives on the mantle side (a Function in `V_mantle`, or a
        Cofunction on its dual), `out` on the surface side. The work is done
        on the PETSc vectors of the owned entries; Firedrake marks the halo
        of `out` as out of date and updates it when it is next read. The
        scaling is done on the PETSc vector too, so no halo value is touched
        before that update.
        """
        with source.dat.vec_ro as x, out.dat.vec_wo as y:
            self.R.mult(x, y)
            if scale != 1.0:
                y.scale(scale)
        return out

    def _apply_transpose(self, source, out, scale=1.0):
        """`out = scale R^T source` on the coefficient vectors (no annotation).

        `source` is a Function in `V_surface`, `out` a Function in
        `V_mantle`. See `_apply` for the halo.
        """
        with source.dat.vec_ro as x, out.dat.vec_wo as y:
            self.R.multTranspose(x, y)
            if scale != 1.0:
                y.scale(scale)
        return out

    def restrict(self, f_mantle, out=None):
        """The values of a mantle CG2 field at the surface nodes, exactly.

        Not annotated: the inversion uses it only with annotation paused
        (the observed bed and the restart bed).

        Args:
          f_mantle: a Function in `V_mantle`.
          out: a Function in `V_surface` to write into, or None for a new one.

        Returns:
          The Function in `V_surface`.
        """
        out = Function(self.V_surface) if out is None else out
        return self._apply(f_mantle, out)

    def extend(self, f_surface, out, scale=1.0):
        """Write `scale * f_surface` onto the mantle nodes on Re, exactly.

        Every other node of `out` becomes zero. The solver reads `out` only
        on the facets of Re, where only the nodes on Re matter. When the
        tape is annotating, one `SurfaceExtendBlock` goes on it, with
        `f_surface` as its dependency and `out` as its output.

        Args:
          f_surface: a Function in `V_surface`.
          out: a Function in `V_mantle`; it is overwritten.
          scale: a float factor (for example `-1 / D` from a bed in metres
            to a non-dimensional sea level). A float, not a `Constant`, so
            it is not a control of the tape.

        Returns:
          `out`.
        """
        scale = float(scale)
        annotate = annotate_tape()
        if annotate:
            block = SurfaceExtendBlock(self, f_surface, scale)
            get_working_tape().add_block(block)
        with stop_annotating():
            self._apply_transpose(f_surface, out, scale)
        if annotate:
            block.add_output(out.create_block_variable())
        return out


class SurfaceExtendBlock(Block):
    """The tape block of `SurfaceNodeMap.extend`: `m = scale R^T b`.

    The map is linear in `b` and does not depend on anything else, so

        tangent linear:  dm = scale R^T db
        adjoint:         b* = scale R m*    (on the coefficient vectors)
        Hessian:         the adjoint applied to the second-order input,
                         with no second-derivative term.

    The adjoint input `m*` is a Cofunction on the dual of the mantle space;
    `R` maps its coefficients to the coefficients of a Cofunction on the
    dual of the surface space. That is the transpose of the forward map on
    the coefficient vectors, which is the adjoint of a linear map between
    coefficient spaces.

    Args:
      node_map: the `SurfaceNodeMap`.
      f_surface: the Function in the surface space that the block reads.
      scale: the float factor of the map.
    """

    def __init__(self, node_map, f_surface, scale):
        super().__init__()
        self.node_map = node_map
        self.scale = float(scale)
        self.add_dependency(f_surface)

    def __str__(self):
        return f"surface_to_mantle_nodes(scale={self.scale:g})"

    def _adjoint(self, cofunction):
        """`scale R cofunction` as a Cofunction on the surface dual space."""
        if cofunction is None:
            return None
        out = Cofunction(self.node_map.V_surface.dual())
        return self.node_map._apply(cofunction, out, self.scale)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return self._adjoint(adj_inputs[0])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if tlm_inputs[0] is None:
            return None
        out = Function(self.node_map.V_mantle)
        return self.node_map._apply_transpose(tlm_inputs[0], out, self.scale)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear map: no second-derivative term.
        return self._adjoint(hessian_inputs[0])

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        # A new Function per replay; with disk checkpointing on it goes to
        # disk like the outputs of Firedrake's own blocks.
        out = Function(self.node_map.V_mantle)
        self.node_map._apply_transpose(inputs[0], out, self.scale)
        return maybe_disk_checkpoint(out)


# --------------------------------------------------------------------------
# The input file
# --------------------------------------------------------------------------


class ParentCopy:
    """Copies of mantle fields on the parent mesh, for `CheckpointFile`.

    A field on the mantle submesh is interpolated into the broken version
    of its element on the mantle, which is exact because the broken space
    contains the original one. That broken field then goes to the parent
    across the submesh map. `allow_missing_dofs` leaves the parent cells
    outside the mantle at zero; nothing reads them.

    One pair of work functions is kept per name, so a field that is written
    at many steps (the ice) reuses them.

    Args:
      parent: the parent mesh.
    """

    def __init__(self, parent):
        self.parent = parent
        self._copies = {}

    def __call__(self, name, f):
        """The copy of the mantle function `f` on the parent, named `name`."""
        space = f.function_space()
        if name not in self._copies:
            broken = BrokenElement(space.ufl_element())
            self._copies[name] = (
                Function(FunctionSpace(space.mesh(), broken)),
                Function(FunctionSpace(self.parent, broken), name=name))
        local, copy = self._copies[name]
        local.interpolate(f)
        copy.assign(assemble(interpolate(local, copy.function_space(),
                                         allow_missing_dofs=True)))
        return copy


def write_json_dataset(chk, path, data):
    """Write `data` as JSON text into the `uint8` dataset `path` of `chk`.

    Every rank must write the same bytes, because the parallel HDF5 file is
    written by all ranks. The text is made on rank 0 and broadcast. The text
    is a dataset, not an attribute, because an HDF5 attribute cannot hold
    more than 64 KiB.
    """
    text = json.dumps(plain_json(data)) if COMM_WORLD.rank == 0 else None
    text = COMM_WORLD.bcast(text, root=0)
    raw = np.frombuffer(text.encode(), dtype=np.uint8)
    group, name = os.path.split(path)
    grp = chk.require_group(group)
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=raw)


def read_json_dataset(chk, path):
    """Read the JSON text that `write_json_dataset` wrote at `path`."""
    return json.loads(bytes(chk.h5pyfile[path][...]).decode())


def write_inputs(path, parent, fields, ice, record):
    """Write the input file of the inversion.

    Collective.

    Args:
      path: the `.h5` file; an existing file is replaced.
      parent: the parent mesh.
      fields: `{name: Function on the mantle}` for every name of
        `field_names(record)`: `MANTLE_FIELDS` and the optional fields that
        the record asks for (`OPTIONAL_FIELDS`).
      ice: a callable `ice(index)` that returns the ice of step `index` as a
        Function on the mantle, and `record["march"]["ice_steps"]` steps.
      record: the record of the prepare run; it goes into the file as JSON.
    """
    names = field_names(record)
    missing = set(names) - set(fields)
    if missing:
        raise ValueError(f"write_inputs: no field for {sorted(missing)}")
    # The element fields must have one component per element of the record.
    for name in ("shear_moduli", "viscosities"):
        if name in names:
            size = fields[name].function_space().value_size
            if size != int(record["n_elements"]):
                raise ValueError(f"write_inputs: {name} has {size} "
                                 f"components, the record n_elements "
                                 f"{record['n_elements']}")
    copies = ParentCopy(parent)
    with CheckpointFile(path, "w") as chk:
        chk.save_mesh(parent)
        # A vector field goes through the broken copy like a scalar one
        # (`BrokenElement` of the vector element), which is exact.
        for name in names:
            chk.save_function(copies(name, fields[name]))
        # One entry per step of the march, `idx` = step index. The year is
        # stored with every entry, so the file can be read without the
        # record.
        years = record["march"]["years"]
        for index in range(record["march"]["ice_steps"]):
            chk.save_function(copies(ICE_NAME, ice(index)), idx=index,
                              timestepping_info={"year": float(years[index])})
        write_json_dataset(chk, f"{INPUT_GROUP}/record", record)


#: The name of the control (the bed at the start of the run, m): the
#: surface field of the inversion and of `controls.h5`, and the mantle field
#: that carries it on the nodes of Re.
CONTROL_NAME = "bed_start_m"
#: The space of the bed on the mantle and of its copy in the state file.
CONTROL_SPACE = ("CG", 2)
#: The name of the bed in the state file: a CG2 field on the parent mesh,
#: an exact copy of the mantle nodes (`append_control`).
STATE_CONTROL_NAME = "bed_start_parent_cg2_m"
#: The name under which earlier versions of the inversion stored the bed in
#: the state file: the broken CG2 copy of `ParentCopy`. `read_inputs` still
#: reads it, so a run of an earlier version can continue (see there).
LEGACY_STATE_CONTROL_NAME = CONTROL_NAME


def append_control(path, parent, bed_mantle, index, info):
    """Append one iterate of the control to the state file of an inversion.

    The state file is a copy of the input file, and the parent mesh in
    memory was loaded from it. A function on that mesh can be saved into
    the same file and loaded back onto a parent mesh loaded from it, without
    any interpolation between meshes. The bed lives on the surface submesh,
    which `CheckpointFile` cannot store, so the caller passes it as a CG2
    field on the mantle (the nodes on Re carry the bed; see
    `SurfaceNodeMap.extend`).

    The bed goes into the file as a CG2 field on the parent, written with
    `Function.assign` across the submesh map: an exact copy of the nodal
    values, no evaluation of basis functions. The input fields use the
    broken copy of `ParentCopy` instead, whose interpolations change the
    values by round-off (about 1e-14 relative); for the bed that round-off
    would make the J of a restart differ from the J of the earlier run. A
    continuous parent space is exact here because `assign` writes every
    parent node of a mantle cell with the value of the mantle node, so the
    nodes on Re, which the parent shares with the cells outside the mantle,
    take the mantle value; the parent nodes outside the mantle stay zero.

    Collective.

    Args:
      path: the state file.
      parent: the parent mesh, loaded from `path`.
      bed_mantle: the bed on the mantle, CG2, m.
      index: the ROL iteration, used as `idx`.
      info: plain numbers stored with the entry (J, RMS misfit).
    """
    copy = Function(FunctionSpace(parent, *CONTROL_SPACE),
                    name=STATE_CONTROL_NAME)
    copy.assign(bed_mantle, allow_missing_dofs=True)
    with CheckpointFile(path, "a") as chk:
        chk.save_function(copy, idx=int(index),
                          timestepping_info={k: float(v)
                                             for k, v in info.items()})


def read_inputs(path, last_control=False):
    """Load the meshes, the fields and the ice of an input file.

    Collective. The parent mesh comes from the file with its stored
    distribution; the mantle submesh is cut from it and curved as
    `build_meshes` does. Each field is interpolated from its copy on the
    parent into its space on the mantle (`MANTLE_FIELDS`), across the
    submesh map.

    Args:
      path: the input file of `write_inputs`, or the state file of an
        inversion (a copy of the input file with the iterates of the
        control appended).
      last_control: also load the last iterate of the control that
        `append_control` wrote (or that an earlier version of the inversion
        wrote as a broken copy, `LEGACY_STATE_CONTROL_NAME`).

    Returns:
      `(parent, mantle, fields, ice, record)`: `fields` maps each name of
      `field_names(record)` to a Function on the mantle (`MANTLE_FIELDS`
      and the optional fields the record names), `ice` is a list with one
      new Function on the mantle per step. With `last_control`, a sixth
      item `(bed_mantle, index)`: the bed on the mantle (CG2, m; only the
      nodes on Re carry it) and the ROL iteration it comes from.

    Raises:
      RuntimeError: if the file was written on another number of ranks, or
        if `last_control` is asked for and the file holds no control.
    """
    with CheckpointFile(path, "r") as chk:
        record = read_json_dataset(chk, f"{INPUT_GROUP}/record")
        ranks = record["mesh"]["ranks"]
        if ranks != COMM_WORLD.size:
            raise RuntimeError(
                f"{path} was written on {ranks} ranks; run on the same "
                f"number, not {COMM_WORLD.size}. On another number Firedrake "
                "partitions the mesh again with its default partitioner, "
                "which can leave ranks without a mantle cell.")
        parent = chk.load_mesh(PARENT_NAME)
        names = field_names(record)
        stored = {name: chk.load_function(parent, name) for name in names}
        stored_ice = [chk.load_function(parent, ICE_NAME, idx=index)
                      for index in range(record["march"]["ice_steps"])]
        stored_control = None
        if last_control:
            # The exact copy of `append_control`, or else the broken copy of
            # an earlier version of the inversion.
            for control_name in (STATE_CONTROL_NAME,
                                 LEGACY_STATE_CONTROL_NAME):
                try:
                    history = chk.get_timestepping_history(parent,
                                                           control_name)
                except Exception:  # noqa: BLE001  (no entry of this name)
                    continue
                control_index = int(max(history["index"]))
                stored_control = chk.load_function(parent, control_name,
                                                   idx=control_index)
                break
            else:
                raise RuntimeError(f"{path} holds no {STATE_CONTROL_NAME} "
                                   f"and no {LEGACY_STATE_CONTROL_NAME}")
    parent.cartesian = False
    mantle = mantle_mesh(parent, record["mesh"]["mantle_tags"])

    def on_mantle(source, space, name):
        # The interpolation from the broken parent copy into the mantle space
        # copies nodal values cell by cell. At a node shared by several
        # mantle cells every cell holds the same value, because the stored
        # field was continuous there.
        target = Function(space, name=name)
        target.interpolate(source)
        return target

    fields = {name: on_mantle(stored[name], field_space(mantle, name, record),
                              name)
              for name in names}
    if "exponent" in fields:
        snap_exponent(fields["exponent"], power_law_settings(record))
        # `power_law_factor` divides by the transition stress in every cell,
        # also where n = 1, so it must be positive everywhere.
        low = COMM_WORLD.allreduce(
            float(fields["transition_stress"].dat.data_ro.min(initial=np.inf)),
            op=MPI.MIN)
        if not low > 0.0:
            raise RuntimeError(f"{path}: the transition stress is not "
                               f"positive everywhere (min {low})")
    ice_space = FunctionSpace(mantle, *ICE_SPACE)
    ice = [on_mantle(f, ice_space, f"{ICE_NAME}_{index}")
           for index, f in enumerate(stored_ice)]
    if last_control:
        bed = Function(FunctionSpace(mantle, *CONTROL_SPACE),
                       name=CONTROL_NAME)
        if stored_control.name() == STATE_CONTROL_NAME:
            # The exact copy: parent nodes to mantle nodes across the
            # submesh map, with no evaluation of basis functions.
            bed.assign(stored_control)
        else:
            # The broken copy of an earlier version: the interpolation of
            # the input fields (exact to round-off). Its nodes on Re hold
            # the values that the earlier version wrote into `SL_init` (its
            # cross-mesh interpolation of the control onto the mantle), so
            # the exact restriction of this field is the bed that the solver
            # of that run used, and the restart continues from it.
            bed = on_mantle(stored_control,
                            FunctionSpace(mantle, *CONTROL_SPACE),
                            CONTROL_NAME)
        return parent, mantle, fields, ice, record, (bed, control_index)
    return parent, mantle, fields, ice, record


def snap_exponent(exponent, settings):
    """Make every value of the DG0 exponent exactly 1.0 or exactly n.

    `power_law_factor` tests `ufl.eq(n - 1, 0)` to choose the Newtonian
    branch, so a cell with n = 1 must hold 1.0 bit for bit. The prepare
    script writes the two values exactly; the copy through the parent mesh
    is an interpolation, which for DG0 multiplies by a basis value of 1 and
    is exact in practice. This makes it exact by construction: each value
    within 1e-12 of 1 or of n is set to that number, and any other value
    stops the run (the file does not hold the rule of its record).

    Collective (the count of bad values is reduced over the ranks).

    Args:
      exponent: the DG0 exponent on the mantle; changed in place.
      settings: `power_law_settings(record)`.

    Raises:
      RuntimeError: if a value is near neither 1 nor n.
    """
    n = settings["exponent"]
    # Read with the halos up to date (a read-only access exchanges them),
    # check the owned values only (each value is counted once), then write
    # the snapped values everywhere: every rank snaps a halo value as its
    # owner does, so the halos stay consistent.
    values = np.array(exponent.dat.data_ro_with_halos)
    owned = exponent.dat.data_ro.shape[0]
    near_one = np.abs(values - 1.0) < 1e-12
    near_n = np.abs(values - n) < 1e-12
    bad = COMM_WORLD.allreduce(
        int(np.sum(~(near_one | near_n)[:owned])), op=MPI.SUM)
    if bad:
        sample = values[:owned][~(near_one | near_n)[:owned]][:5]
        raise RuntimeError(f"{bad} exponent values are neither 1 nor {n} "
                           f"(on rank {COMM_WORLD.rank}: {sample})")
    values[near_one] = 1.0
    values[near_n] = n
    exponent.dat.data_wo_with_halos[:] = values


def element_fields(fields):
    """The shear moduli and viscosities of the Maxwell elements, as lists.

    With `shear_moduli` and `viscosities` in `fields` (an input file with
    `n_elements`), element j has the scalar fields of component j, copied
    from the vector data column by column (the vector and the scalar space
    of one element share the node numbering, so the copy is exact and
    needs no interpolation). Without them, the one Maxwell element of
    `shear_modulus` and `viscosity`.

    Returns:
      `(shear_moduli, viscosities)`, two lists of N Functions on the mantle.
    """
    if "shear_moduli" not in fields:
        return [fields["shear_modulus"]], [fields["viscosity"]]
    lists = []
    for name in ("shear_moduli", "viscosities"):
        vector = fields[name]
        space = FunctionSpace(vector.function_space().mesh(),
                              vector.function_space().ufl_element().sub_elements[0])
        n = vector.function_space().value_size
        columns = []
        for j in range(n):
            column = Function(space, name=f"{name}_{j}")
            column.dat.data_wo_with_halos[:] = \
                vector.dat.data_ro_with_halos[:, j]
            columns.append(column)
        lists.append(columns)
    if len(lists[0]) != len(lists[1]):
        raise ValueError("shear_moduli and viscosities have different "
                         "numbers of elements")
    return lists[0], lists[1]


# --------------------------------------------------------------------------
# Field statistics
# --------------------------------------------------------------------------


def field_statistics(f, measure):
    """Min, max and integral of a field over all ranks.

    The min and max are over the nodal values that this rank owns. The
    integral is over `measure` (`dx` of the mantle, or `ds` of Re for a
    surface field). The prepare script records these numbers and the
    inversion computes them again after it loads the file: a difference
    means the file did not come back as it was written.

    A vector field (the element fields of `OPTIONAL_FIELDS`) gives the min
    and max over all components and the integral of the sum of its
    components.

    Returns:
      `{"min": .., "max": .., "integral": ..}`.
    """
    data = f.dat.data_ro
    shape = f.ufl_shape
    integrand = f if not shape else sum(f[i] for i in range(shape[0]))
    lo = data.min() if data.size else np.inf
    hi = data.max() if data.size else -np.inf
    return {"min": COMM_WORLD.allreduce(float(lo), op=MPI.MIN),
            "max": COMM_WORLD.allreduce(float(hi), op=MPI.MAX),
            "integral": float(assemble(integrand * measure))}


def input_statistics(mantle, fields, ice):
    """The `field_statistics` of every field and of the ice of every step.

    The earth fields and the observed sea level are integrated over the
    mantle volume, the ice over Re (its volume in units of D^3).
    """
    volume = dx(domain=mantle)
    surface = ds(SURF_RE, domain=mantle)
    stats = {name: field_statistics(f, volume) for name, f in fields.items()}
    stats[ICE_NAME] = [field_statistics(f, surface) for f in ice]
    return stats


def compare_statistics(expected, found, rtol=1e-10):
    """The entries of two `input_statistics` results that differ.

    Two numbers of one field agree when they differ by at most `rtol` times
    the size of that field, the largest magnitude of its min, max and
    integral. The tolerance is relative to the field and not to each number,
    because a min of exactly zero (the ice outside the ice sheets) comes
    back as a round-off value near 1e-19: the interpolation from the stored
    copy into the mantle space evaluates the basis functions at the nodes,
    and that evaluation is exact only to round-off. The loaded mesh can also
    number its cells in another order than the written one, so the
    integrals can differ in the last bits.

    Returns:
      A list of `(key, expected, found)` for every entry that differs.
    """
    bad = []

    def compare_field(key, a, b):
        # The size of the field, from the expected numbers.
        size = max(abs(value) for value in a.values())
        for name in a:
            if abs(a[name] - b[name]) > rtol * size:
                bad.append((f"{key}.{name}", a[name], b[name]))

    def walk(key, a, b):
        if isinstance(a, list):
            if len(a) != len(b):
                bad.append((f"{key} length", len(a), len(b)))
            for i, (x, y) in enumerate(zip(a, b)):
                walk(f"{key}[{i}]", x, y)
        else:
            compare_field(key, a, b)

    for key in expected:
        walk(key, expected[key], found[key])
    return bad


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


def time_step(mantle, value=1.0):
    """The time step of the solver: a `Function` on the Real space of the mantle.

    A Real `Function` and not a `Constant` (decision S28 of the design
    handout of 2026-10-20). An assignment to it is a block on the tape, and
    the coupled solve block lists it as a dependency, so a taped march can
    change the time step (for example from 500 yr to 100 yr at 26 ka), the
    replay reads the value of each step from the tape, and the rebuild rule
    of the preconditioner sees the time step of the step being solved in the
    replay and in the adjoint (`gadopt.dtn_coupled_adjoint`, `gia_dt`). It
    lives on the mantle mesh, the mesh of the internal-variable equation,
    which is the only place the time step enters.

    Args:
      mantle: the mantle mesh.
      value: the initial value, units of t_bar. The scripts assign the value
        of each step before its solve.

    Returns:
      The Function, named `dt`.
    """
    return Function(FunctionSpace(mantle, "R", 0), name="dt").assign(value)


# --------------------------------------------------------------------------
# Schedule C: the protocol steps plus the fast melt windows
# --------------------------------------------------------------------------

#: The last year of every march with a step after year 0 (the end of the
#: GIAMIP protocol).
SCHEDULE_LAST_YEAR = 1000

#: The windows of schedule C (design handout 8.3, decisions S33 and S34), by
#: the file of the ice history, as `(first year, end year, step in yr)`. A
#: step that starts inside a window has the step length of that window; any
#: other step goes to the next protocol year (500 yr before year -26000,
#: 100 yr after). The windows follow the fast melt of each history after Lau
#: (2023), Sect. 3.1: 10 yr steps in the fastest melt, 25 yr steps in the
#: kyr before and after it. The handout lists ICE-6G_D (Exp12) and GLAC3c
#: profile 1 (Exp09) in full. For profile 2 (Exp10) it says "the same
#: pattern at 15-14 ka and 11-10 ka": the Exp09 pattern is 25 yr from 1 kyr
#: before the first 10 yr window to 1 kyr after the last one, with 25 yr
#: also between the two windows, so Exp10 has 25 yr from 14 to 11 ka.
SCHEDULE_C_WINDOWS = {
    # ICE-6G_D (Exp12): MWP-1A, 15 to 13 ka.
    "iceHistory-ICE-6G_D.nc": ((-16000, -15000, 25), (-15000, -13000, 10),
                               (-13000, -11000, 25)),
    # GLAC3c profile 1 (Exp09): 15 to 14 ka and 12 to 11 ka.
    "iceHistory-GLAC3c_profile1.nc": ((-16000, -15000, 25),
                                      (-15000, -14000, 10),
                                      (-14000, -12000, 25),
                                      (-12000, -11000, 10),
                                      (-11000, -10000, 25)),
    # GLAC3c profile 2 (Exp10): 15 to 14 ka and 11 to 10 ka.
    "iceHistory-GLAC3c_profile2.nc": ((-16000, -15000, 25),
                                      (-15000, -14000, 10),
                                      (-14000, -11000, 25),
                                      (-11000, -10000, 10),
                                      (-10000, -9000, 25)),
}

#: The number of steps of schedule C from 122 ka to +1000 yr (design
#: handout 8.3: 732 for Exp12 and 762 for Exp09; Exp10 by the reading of
#: `SCHEDULE_C_WINDOWS`). The protocol alone has 192 + 260 + 10 = 462
#: steps; each kyr at 25 yr adds 30 steps, each kyr at 10 yr adds 90.
SCHEDULE_C_STEPS_122KA = {"iceHistory-ICE-6G_D.nc": 732,
                          "iceHistory-GLAC3c_profile1.nc": 762,
                          "iceHistory-GLAC3c_profile2.nc": 792}


def schedule_c_windows(ice_file):
    """The windows of schedule C for the ice history `ice_file` (a basename).

    Raises:
      KeyError: if schedule C has no windows for that history.
    """
    if ice_file not in SCHEDULE_C_WINDOWS:
        raise KeyError(f"schedule C has no windows for the ice history "
                       f"{ice_file}")
    return SCHEDULE_C_WINDOWS[ice_file]


def _check_windows(protocol_years, windows):
    """Check the windows of schedule C or D against the protocol years.

    The rules that both schedules need, so that a step inside a window
    never jumps over a protocol year or over the end of the window: each
    window starts and ends at a protocol year, its step divides it, its step
    reaches every protocol year inside it, and the windows are in order and
    do not overlap.

    Args:
      protocol_years: the protocol years of giamip, ascending.
      windows: `(first year, end year, step)` triples.

    Returns:
      The protocol years as a list of Python integers.

    Raises:
      ValueError: if the protocol years are not whole years, or if a window
        breaks one of the rules.
    """
    protocol = [int(round(y)) for y in protocol_years]
    if any(float(y) != float(p) for y, p in zip(protocol_years, protocol)):
        raise ValueError("the protocol years are not whole years")
    pset = set(protocol)
    for first, end, step in windows:
        if (first not in pset or end not in pset or step <= 0
                or (end - first) % step != 0):
            raise ValueError(f"window {(first, end, step)}: its ends must be "
                             f"protocol years and its step must divide it")
        # A protocol year inside the window must be a year of the window.
        inside = [p for p in protocol if first < p < end]
        if any((p - first) % step != 0 for p in inside):
            raise ValueError(f"window {(first, end, step)}: its step does "
                             f"not reach every protocol year inside it")
    # The windows must be in order and must not overlap: a year belongs to
    # one window at most.
    for (_, end, _), (first, _, _) in zip(windows[:-1], windows[1:]):
        if end > first:
            raise ValueError(f"windows {windows}: not in order or overlapping")
    return protocol


def schedule_c_years(start, protocol_years, windows, last=SCHEDULE_LAST_YEAR):
    """Every year of schedule C, from `start` to `last`.

    Each step either has the length of the window it starts in, or goes to
    the next protocol year. The years are integers, built by adding whole
    step lengths, so no rounding error accumulates and every protocol year
    is a year of the schedule. A window must start and end at a protocol
    year and its step must divide the window: then a step inside a window
    never jumps over a protocol year or over the end of the window.

    Args:
      start: the first year (the start of the experiment), a protocol year.
      protocol_years: the protocol years of giamip, ascending.
      windows: `(first year, end year, step)` triples (`SCHEDULE_C_WINDOWS`).
      last: the last year of the schedule.

    Returns:
      A NumPy integer array of the years, `start` first, `last` last.

    Raises:
      ValueError: if a window or the schedule breaks one of these rules.
    """
    protocol = _check_windows(protocol_years, windows)
    pset = set(protocol)
    if int(start) not in pset or last not in pset:
        raise ValueError(f"start {start} or last {last} is not a protocol "
                         f"year")
    years = [int(start)]
    while years[-1] < last:
        year = years[-1]
        step = next((s for first, end, s in windows if first <= year < end),
                    None)
        if step is None:
            # The next protocol year.
            following = [p for p in protocol if p > year]
            years.append(following[0])
        else:
            years.append(year + step)
    years = np.asarray(years, dtype=np.int64)
    missing = sorted(p for p in pset
                     if int(start) <= p <= last and p not in set(years.tolist()))
    if years[-1] != last or missing:
        raise ValueError(f"schedule C from {start} ends at {years[-1]} or "
                         f"misses protocol years {missing[:5]}")
    return years


# --------------------------------------------------------------------------
# Schedule D: 1 kyr steps plus the fast melt windows at twice the C step
# --------------------------------------------------------------------------

#: The `--schedule` value of schedule C in the prepare and forward scripts,
#: and the `march.schedule` entry of its input file.
SCHEDULE_C = "windows"
#: The `--schedule` value of schedule D, and the `march.schedule` entry of
#: its input file.
SCHEDULE_D = "windows-coarse"
#: The two schedules with fast melt windows. Their input files hold the ice
#: of every year of the march, linear in time between protocol slices.
WINDOW_SCHEDULES = (SCHEDULE_C, SCHEDULE_D)

#: The step (yr) of schedule D outside the windows (Sia, 2026-10-24,
#: decision R12 of the r2 production): from the start to the first window,
#: from the last window to year 0, and from year 0 to the end of the
#: protocol. Schedule C on the r2 mesh needs more node disk for the tape
#: and more wall time per restart than a Gadi job has; the tape and the
#: time of a sweep scale with the number of steps.
SCHEDULE_D_STEP_YR = 1000
#: The factor between the window step of schedule D and that of schedule C
#: (R12: 25 yr becomes 50 yr, 10 yr becomes 20 yr). This is the reading of
#: "twice that" of the decision; Sia is to confirm it.
SCHEDULE_D_WINDOW_FACTOR = 2

#: The number of steps of schedule D from 122 ka to +1000 yr. Outside the
#: windows: one step per kyr, (122 - window span in kyr) to year 0 and one
#: after it. In the windows: a kyr at 50 yr adds 20 steps, a kyr at 20 yr
#: adds 50. ICE-6G_D: 117 + 1 + 20 + 100 + 40 = 278; GLAC3c profile 1:
#: 116 + 1 + 20 + 50 + 40 + 50 + 20 = 297; GLAC3c profile 2:
#: 115 + 1 + 20 + 50 + 60 + 50 + 20 = 316.
SCHEDULE_D_STEPS_122KA = {"iceHistory-ICE-6G_D.nc": 278,
                          "iceHistory-GLAC3c_profile1.nc": 297,
                          "iceHistory-GLAC3c_profile2.nc": 316}


def schedule_d_windows(ice_file):
    """The windows of schedule D for the ice history `ice_file` (a basename).

    The windows of schedule C (`SCHEDULE_C_WINDOWS`), same first and end
    years, with `SCHEDULE_D_WINDOW_FACTOR` times the step.

    Raises:
      KeyError: if schedule C has no windows for that history.
    """
    return tuple((first, end, SCHEDULE_D_WINDOW_FACTOR * step)
                 for first, end, step in schedule_c_windows(ice_file))


def schedule_d_years(start, protocol_years, windows, last=SCHEDULE_LAST_YEAR,
                     step=SCHEDULE_D_STEP_YR):
    """Every year of schedule D, from `start` to `last`.

    Each step either has the length of the window it starts in, or is
    `step` years long (1 kyr). The years are integers, built by adding whole
    step lengths, so no rounding error accumulates.

    Window edges and the 1 kyr steps must line up: the edges of every
    window, year 0 and `last` must be a whole number of `step` from
    `start`. The function refuses a window edge off that grid
    (`ValueError`); it never shortens a step to reach the edge. So the
    schedule has no fractional step, and the dt of the march takes only the
    values `step` and the window steps. The windows of all three histories
    have their edges on whole kyr and every experiment starts at a whole
    kyr, so the rule holds for them.

    Every year outside the windows must be a protocol year, so the ice of
    such a year is the protocol slice itself (weight 0 in `ice_bracket`).
    The years inside a window reach every protocol year there
    (`_check_windows`), so each of their bracketing protocol years is a
    year of the march, as `check_linear_ice` needs. Between two 1 kyr years
    the load is not sampled: the protocol slices in between (every 500 yr
    before year -26000, every 100 yr after) do not enter the march, and
    their years are not years of the schedule.

    Args:
      start: the first year (the start of the experiment), a protocol year.
      protocol_years: the protocol years of giamip, ascending.
      windows: `(first year, end year, step)` triples (`schedule_d_windows`).
      last: the last year of the schedule.
      step: the step outside the windows, years.

    Returns:
      A NumPy integer array of the years, `start` first, `last` last; year 0
      is one of them.

    Raises:
      ValueError: if a window or the schedule breaks one of these rules.
    """
    protocol = _check_windows(protocol_years, windows)
    pset = set(protocol)
    start = int(start)
    if start not in pset or last not in pset:
        raise ValueError(f"start {start} or last {last} is not a protocol "
                         f"year")
    # The grid of the 1 kyr steps: every window edge, year 0 and the last
    # year must be on it, else a 1 kyr step would end inside a window or
    # jump over year 0.
    edges = sorted({e for first, end, _ in windows for e in (first, end)}
                   | {0, int(last)})
    off_grid = [e for e in edges if (e - start) % step != 0]
    if off_grid:
        raise ValueError(f"schedule D from {start}: the years {off_grid} are "
                         f"not a whole number of {step} yr steps from the "
                         f"start")
    years = [start]
    while years[-1] < last:
        year = years[-1]
        dt = next((s for first, end, s in windows if first <= year < end),
                  step)
        years.append(year + dt)
    years = np.asarray(years, dtype=np.int64)
    # A year outside every window must carry a protocol slice of its own.
    outside = [int(y) for y in years
               if not any(first <= y < end for first, end, _ in windows)
               and int(y) not in pset]
    if years[-1] != last or 0 not in set(years.tolist()) or outside:
        raise ValueError(f"schedule D from {start} ends at {years[-1]}, "
                         f"misses year 0, or has years without a protocol "
                         f"slice {outside[:5]}")
    return years


# --------------------------------------------------------------------------
# Schedule D with the protocol years as extra step ends (the "b" forward run)
# --------------------------------------------------------------------------

#: The `--schedule` value of the forward run that marches schedule D with
#: every protocol year added as a step end (decision R14 of the r2
#: production, Sia 2026-10-29). Only the forward script knows it: the
#: inversion and its input file stay on schedule D, and this forward run
#: starts from the bed of a schedule-D state file.
SCHEDULE_DP = "windows-coarse-protocol"

#: The shortest step (yr) that the merge of schedule D and the protocol
#: years may make. A step piece shorter than this is removed by removing
#: its end that is a schedule-D year and not a protocol year (the protocol
#: years must stay, they carry the output). 10 yr is the shortest step of
#: every schedule of these experiments, and the drop rule of the Extended
#: Burgers elements (`prepare_expXX.EB_DROP_DT_YR`) assumes that no step is
#: shorter: an element with a Maxwell time below 5 yr is dropped because it
#: relaxes fully inside a 10 yr step, which a shorter step would not
#: guarantee. For the three histories the rule never fires: every protocol
#: year and every schedule-D year is a whole multiple of 10 yr, the
#: schedule-D windows already reach every protocol year inside them, and
#: outside the windows a 1 kyr step is cut into pieces of 500 yr or 100 yr.
SCHEDULE_DP_MIN_STEP_YR = 10

#: The number of steps of schedule D plus the protocol years from 122 ka to
#: +1000 yr. Outside the windows the merge gives the protocol steps: 192
#: steps of 500 yr to year -26000, then 100 yr steps from -26000 to the
#: first window, from the last window to year 0 and from year 0 to +1000.
#: The windows keep their schedule-D steps (they reach every protocol year
#: already). ICE-6G_D: 192 + 100 + (20 + 100 + 40) + 110 + 10 = 572;
#: GLAC3c profile 1: 192 + 100 + (20 + 50 + 40 + 50 + 20) + 100 + 10 = 582;
#: GLAC3c profile 2: 192 + 100 + (20 + 50 + 60 + 50 + 20) + 90 + 10 = 592.
SCHEDULE_DP_STEPS_122KA = {"iceHistory-ICE-6G_D.nc": 572,
                           "iceHistory-GLAC3c_profile1.nc": 582,
                           "iceHistory-GLAC3c_profile2.nc": 592}


def merge_protocol_years(base_years, protocol_years,
                         min_step=SCHEDULE_DP_MIN_STEP_YR):
    """A march with every protocol year of its span added as a step end.

    The years of the result are the union of `base_years` and the protocol
    years from the first to the last base year. Where a protocol year falls
    inside a base step, that step is split into two (or more, for several
    protocol years inside it); the base years themselves all stay. Every
    year is an integer, so no rounding error enters the step lengths.

    A piece shorter than `min_step` is removed by removing its end that is
    a base year and not a protocol year, which merges the piece into its
    neighbour. The first and the last year stay. A short piece whose two
    ends are both protocol years, or both base years that are not protocol
    years (a short step of the base march itself, not made by a split), is
    refused, since removing either end would change what the run outputs or
    what the base march means.

    Args:
      base_years: the years of the base march (schedule D), ascending
        integers.
      protocol_years: the protocol years of giamip, ascending.
      min_step: the shortest piece, yr (`SCHEDULE_DP_MIN_STEP_YR`).

    Returns:
      `(years, dropped)`: the NumPy integer array of the merged years and
      the list of base years that were removed for a short piece (empty
      for the three histories).

    Raises:
      ValueError: if a protocol year is not a whole year, or if a short
        piece cannot be removed by the rule above.
    """
    base = [int(y) for y in base_years]
    if any(float(y) != float(b) for y, b in zip(base_years, base)):
        raise ValueError("the base years are not whole years")
    first, last = base[0], base[-1]
    protocol = [int(round(p)) for p in protocol_years]
    if any(float(p) != float(q) for p, q in zip(protocol_years, protocol)):
        raise ValueError("the protocol years are not whole years")
    pset = {p for p in protocol if first <= p <= last}
    bset = set(base)
    years = sorted(bset | pset)
    dropped = []
    while True:
        # The first piece that is too short, if any.
        short = next((j for j in range(1, len(years))
                      if years[j] - years[j - 1] < min_step), None)
        if short is None:
            break
        ends = (years[short - 1], years[short])
        # The ends that may go: a base year, not a protocol year, and not
        # the first or last year of the march.
        removable = [y for y in ends
                     if y in bset and y not in pset and y not in (first, last)]
        if len(removable) != 1:
            raise ValueError(f"the piece {ends} is shorter than {min_step} yr "
                             f"and has no single schedule-D end that may be "
                             f"removed")
        years.remove(removable[0])
        dropped.append(removable[0])
    return np.asarray(years, dtype=np.int64), dropped


def window_schedule(name, ice_file, start, protocol_years,
                    last=SCHEDULE_LAST_YEAR):
    """The windows and the years of schedule C, D, or D plus the protocol years.

    The one call of the prepare script and of the forward script, so the
    input file and the forward run of a schedule march the same years.
    `SCHEDULE_DP` is a forward schedule only (its input file is one of
    schedule D, `input_schedule_of`).

    Args:
      name: `SCHEDULE_C`, `SCHEDULE_D` or `SCHEDULE_DP`.
      ice_file: the basename of the ice history.
      start: the first year (a protocol year).
      protocol_years: the protocol years of giamip, ascending.
      last: the last year of the schedule.

    Returns:
      `(windows, years)`: the `(first, end, step)` windows and the NumPy
      integer array of the years from `start` to `last`.

    Raises:
      ValueError: if `name` is not one of the three schedules, or if the
        schedule breaks one of its rules.
      RuntimeError: if a march from 122 ka does not have the step count of
        `SCHEDULE_C_STEPS_122KA`, `SCHEDULE_D_STEPS_122KA` or
        `SCHEDULE_DP_STEPS_122KA`.
    """
    if name == SCHEDULE_C:
        windows = schedule_c_windows(ice_file)
        years = schedule_c_years(int(start), protocol_years, windows, last)
        expected = SCHEDULE_C_STEPS_122KA.get(ice_file)
    elif name == SCHEDULE_D:
        windows = schedule_d_windows(ice_file)
        years = schedule_d_years(int(start), protocol_years, windows, last)
        expected = SCHEDULE_D_STEPS_122KA.get(ice_file)
    elif name == SCHEDULE_DP:
        # Schedule D, then every protocol year added as a step end. The
        # windows are those of schedule D: the merge adds no year inside
        # them (their steps reach every protocol year there).
        windows = schedule_d_windows(ice_file)
        d_years = schedule_d_years(int(start), protocol_years, windows, last)
        years, _ = merge_protocol_years(d_years, protocol_years)
        expected = SCHEDULE_DP_STEPS_122KA.get(ice_file)
    else:
        raise ValueError(f"unknown window schedule {name!r}")
    # The counts are for the full march of a 122 ka experiment.
    if (int(start) == -122000 and last == SCHEDULE_LAST_YEAR
            and expected is not None and len(years) - 1 != expected):
        raise RuntimeError(f"{name} has {len(years) - 1} steps for "
                           f"{ice_file}, expected {expected}")
    return windows, years


def input_schedule_of(name):
    """The schedule of the input file that a forward run of `name` reads.

    Schedule C and schedule D run from an input file (state file) of their
    own schedule. `SCHEDULE_DP` runs from a state file of schedule D: it
    marches the years of schedule D plus the protocol years, so the steps
    of the file are a part of its steps. Any other name has no window input
    file (None).
    """
    if name in WINDOW_SCHEDULES:
        return name
    if name == SCHEDULE_DP:
        return SCHEDULE_D
    return None


def window_march_record(name, years, protocol_steps, ice_steps, year0_step,
                        brackets, **extra):
    """The `march` entry of the record of an input file of schedule C or D.

    The inversion reads `years` (the step lengths and dt come from their
    differences), `ice_steps` and `year0_step`; the forward script also
    reads `schedule` and compares `years` with its own schedule. The record
    goes into the input file as JSON (`write_json_dataset`).

    Args:
      name: `SCHEDULE_C` or `SCHEDULE_D`.
      years: the years of the steps in the file (step 0 first).
      protocol_steps: the ice-history slice of each year, -1 at a year
        between two protocol years.
      ice_steps: the number of ice fields in the file.
      year0_step: the step of year 0.
      brackets: `(k, w)` of every year in the file (`ice_bracket`).
      **extra: further entries (the ice history, the windows, the step
        count to the last year, the linear ice check).

    Returns:
      A dict of plain values.
    """
    return {"schedule": name, "step_years": None, "years": years,
            "protocol_steps": protocol_steps, "ice_steps": ice_steps,
            "year0_step": year0_step,
            "ice_brackets": [[int(k), float(w)] for k, w in brackets],
            **extra}


def ice_bracket(protocol_years, year):
    """The two protocol slices around `year` and the weight of the second.

    Design handout P9 and S34: the ice at a year between two protocol years
    is linear in time between them, ice(t) = (1 - w) ice(k) + w ice(k + 1),
    w = (t - t_k) / (t_{k+1} - t_k), with t_k <= t < t_{k+1}. At a protocol
    year w is 0 and the ice is slice k itself. The prepare script (the ice of
    the input file) and the forward script (`LinearIce`) both take k and w
    from here, so the same year gives the same two numbers in both.

    Args:
      protocol_years: the protocol years, ascending, as floats.
      year: the year.

    Returns:
      `(k, w)`: the index of the first slice and the weight (a float) of
      slice k + 1; w is exactly 0.0 at a protocol year.

    Raises:
      ValueError: if `year` is outside the protocol years.
    """
    years = np.asarray(protocol_years, dtype=float)
    k = int(np.searchsorted(years, year, side="right")) - 1
    if k < 0 or year > years[-1]:
        raise ValueError(f"year {year} is outside the protocol years")
    if year == years[k]:
        return k, 0.0
    return k, float((year - years[k]) / (years[k + 1] - years[k]))


def linear_ice_into(out, a, w, b=None):
    """Write `(1 - w) a + w b` into `out`, node by node (all CG1 Functions).

    The one formula of the ice between two protocol slices, used by the
    prepare script and the forward script, so the ice of a year is the same
    bit for bit in the input file and in the forward run when both start
    from the same two slices. At w = 0 the result is a copy of `a`, bit for
    bit (no arithmetic). Owned and halo nodes alike: the halo values equal
    those of the owners, and the combination is node by node, so they stay
    equal. A UFL expression with the weight as a literal would compile a new
    kernel for every weight.

    Args:
      out: the target Function.
      a: slice k.
      w: the weight of slice k + 1 (`ice_bracket`).
      b: slice k + 1; not read when w = 0.

    Returns:
      `out`.
    """
    if w == 0.0:
        out.dat.data_wo_with_halos[:] = a.dat.data_ro_with_halos
        return out
    out.dat.data_wo_with_halos[:] = ((1.0 - w) * a.dat.data_ro_with_halos
                                     + w * b.dat.data_ro_with_halos)
    return out


def march_ice(read_slice, brackets, space):
    """The ice of every year of a march, linear in time between slices.

    The ice of the input file of schedule C (`prepare_expXX.py --schedule
    windows`). Each protocol slice is read once; the two slices around the
    current year are kept, and each year gets a new Function with the
    combination of `linear_ice_into` (a copy of the slice at a protocol
    year). The forward script makes the same numbers with `LinearIce` from
    the same slices.

    Args:
      read_slice: `read_slice(k)` returns protocol slice k as a CG1 Function
        (`SurfaceForcing.ice`).
      brackets: `(k, w)` of every year of the march (`ice_bracket`).
      space: the CG1 space of the ice.

    Returns:
      A list of Functions named `ICE_NAME`, one per year.
    """
    ice = []
    slices = {}
    for k, w in brackets:
        # Drop the slices that the march has passed.
        for old in [j for j in slices if j < k]:
            del slices[old]
        for j in ((k,) if w == 0.0 else (k, k + 1)):
            if j not in slices:
                slices[j] = read_slice(j)
        target = Function(space, name=ICE_NAME)
        linear_ice_into(target, slices[k], w,
                        slices[k + 1] if w != 0.0 else None)
        ice.append(target)
    return ice


def check_linear_ice(years, ice, protocol_years, scratch):
    """Check that the ice of a schedule-C march is linear between its slices.

    For every year of the march that is not a protocol year, the ice must
    equal `linear_ice_into` of the ice of the two bracketing protocol years
    of the same march, bit for bit. The protocol years of the march are
    themselves entries of `ice`. This is the check that the inversion (which
    reads `ice` from the input file) sees the same linear ice as the forward
    run of schedule C. Collective.

    In the prepare job (before the write) the difference is exactly 0. After
    a write and a read of the input file it is not: `read_inputs`
    interpolates each field from its broken copy on the parent mesh onto the
    mantle, which is exact only to round-off (about 3e-19 in units of D,
    1e-12 m, at L0). The forward run therefore compares the difference with
    a tolerance relative to the largest ice (`max_abs_ice`). The inversion
    and the forward run both read the same file, so they march the same ice
    either way.

    Args:
      years: the years of the march.
      ice: the ice of every year of the march (CG1 Functions on the mantle).
      protocol_years: the protocol years of giamip.
      scratch: a CG1 Function for the combination.

    Returns:
      `{"years_checked": n, "max_abs_diff": d, "max_abs_ice": m}` (d and m
      over all ranks, units of D; d is 0.0 when the ice is linear bit for
      bit; m is the largest ice of the march).

    Raises:
      ValueError: if a bracketing protocol year is not a year of the march.
    """
    position = {float(y): j for j, y in enumerate(years)}
    p_years = np.asarray(protocol_years, dtype=float)
    worst = 0.0
    largest = max((float(np.max(np.abs(f.dat.data_ro), initial=0.0))
                   for f in ice), default=0.0)
    n = 0
    for j, year in enumerate(years):
        k, w = ice_bracket(p_years, float(year))
        if w == 0.0:
            continue
        ja, jb = position.get(p_years[k]), position.get(p_years[k + 1])
        if ja is None or jb is None:
            raise ValueError(f"year {year}: the bracketing protocol years "
                             f"are not in the march")
        linear_ice_into(scratch, ice[ja], w, ice[jb])
        d = np.max(np.abs(scratch.dat.data_ro - ice[j].dat.data_ro),
                   initial=0.0)
        worst = max(worst, float(d))
        n += 1
    worst = COMM_WORLD.allreduce(worst, op=MPI.MAX)
    largest = COMM_WORLD.allreduce(largest, op=MPI.MAX)
    return {"years_checked": n, "max_abs_diff": worst,
            "max_abs_ice": largest}


def build_solver(parent, mantle, fields, scales, constants, forcing, dt,
                 options, record=None):
    """The coupled self-gravitating sea-level solver of the experiment.

    One monolithic system per time step: the compressible viscoelastic
    mantle (P2 displacement, DG1 internal variables, one Maxwell element),
    the perturbed gravitational potential on the whole mesh (P2), the fluid
    core on the CMB, the polar motion and the rotation-rate change (three
    `Real` rows), the centre-of-mass frame (three `Real` rows), and the
    sea-level equation with a moving coastline (`Shift`, one `Real` row).
    The time stepping is backward Euler. The coastline moves, so each step
    runs Newton.

    The sea level is

        SL    = SL_init + (N - N_init) - (u_r - ur_init) + Shift
        sigma = rho_w B C (SL - SL_init) + rho_i (1 - B) I - rho_i (1 - B0) I_init

    in the form that `gadopt.SeaLevel` documents, with `N = (psi + psi_rot) /
    g_s` the geoid with the rotational potential. The run starts from the
    undeformed state with the ice of the first step on it, so the load is
    zero at the start and there is no elastic step.

    Args:
      parent: the parent mesh.
      mantle: the mantle mesh.
      fields: the mantle fields of `read_inputs` (`density`,
        `shear_modulus`, `bulk_modulus`, `viscosity`, `gravity`, `slope`).
      scales: the `Scales`.
      constants: the solver constants of the record
        (`record["solver_constants"]`): core density and gravity at the CMB,
        ice and water density, polar moment.
      forcing: `{"SL_init": .., "I": .., "I_init": ..}`, the Functions that
        the sea-level equation reads (the caller keeps them and changes them).
      dt: the time step in units of t_bar, the Real `Function` of
        `time_step`.
      options: the command line of the caller (`k_s`, `block0_max_it`,
        `outer_atol`, `outer_max_it`, `gia_rebuild`, `gia_rebuild_every`,
        `gia_rebuild_max_outer`).
      record: the record of the input file. Required when `fields` holds
        the power-law fields: its `power_law` entry gives the background
        stress (`power_law_settings`).

    Returns:
      `(solver, layout, record)`.
    """
    gravity_bcs = {SURF_OUTER: {"dtn": SphericalDtN(L=DTN_DEGREE)},
                   SURF_INNER: {"dtn": SphericalDtN(L=DTN_DEGREE)}}
    # The Maxwell elements in parallel: one (the Maxwell experiments) or N
    # (the Extended Burgers fit of Exp09 and Exp12, `n_elements` in the
    # record). The internal variables are one DG1 field of shape (N, 3, 3).
    shear_moduli, viscosities = element_fields(fields)
    n_elements = len(shear_moduli)
    # The record states the rheology of the input file (handout 12.5); the
    # fields must agree with it, so that a mismatch stops here and does not
    # build a Maxwell solver for an EB or power-law input file.
    if record is not None:
        if int(record.get("n_elements", 1)) != n_elements:
            raise ValueError(
                f"build_solver: the record has n_elements "
                f"{record.get('n_elements', 1)}, the fields {n_elements}")
        if ("power_law" in record) != ("exponent" in fields):
            raise ValueError("build_solver: the record and the fields "
                             "disagree on the power law")
    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=True,
        fluid_core=True, self_gravity_number=scales.Lambda,
        n_internal_variables=n_elements,
        displacement_degree=2, internal_variable_degree=1,
        potential_degree=2, centre_of_mass=True, sea_level=True,
        dtn_representation=DTN_REPRESENTATION)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    # The power law of Exp10 and Exp12 (`power_law` in the record): the DG0
    # exponent (exactly 1.0 where the rheology is linear) and the DG0
    # transition stress. `gadopt` treats a `Function` exponent as a power
    # law, so Newton stays on and block 0 runs GMRES (the power-law operator
    # is not symmetric); the layout stays uncondensed, which is the layout
    # that supports a power law. Without the record entry the approximation
    # gets no power-law argument and the Maxwell experiments build the
    # solver they always built.
    power_law = {}
    if "exponent" in fields:
        settings = power_law_settings(record or {})
        if settings is None:
            raise ValueError("build_solver: the fields hold a power law, but "
                             "no record with a power_law entry was given")
        power_law = dict(exponent=fields["exponent"],
                         transition_stress=fields["transition_stress"],
                         background_stress=settings["background_stress"])
    approximation = CompressibleInternalVariableApproximation(
        bulk_modulus=fields["bulk_modulus"], density=fields["density"],
        shear_modulus=shear_moduli, viscosity=viscosities,
        bulk_shear_ratio=1, g=fields["gravity"], B_mu=scales.B_mu,
        self_gravity_number=scales.Lambda, **power_law)

    # The core: inviscid, homogeneous and incompressible, with one density.
    fluid_core = FluidCore(boundary=SURF_RC,
                           rho_core=constants["rho_core_nondim"],
                           g=Constant(constants["g_cmb_nondim"]))

    # g_s = g_bar by the choice of the gravity scale.
    sea_level = SeaLevel(
        boundary=SURF_RE, rho_w=constants["rho_w_nondim"],
        rho_i=constants["rho_i_nondim"], g_surface=1.0,
        SL_init=forcing["SL_init"], I=forcing["I"], I_init=forcing["I_init"],
        N_init=Constant(0.0), ur_init=Constant(0.0),
        alpha_mask=DEFAULT_ALPHA_MASK, slope=fields["slope"],
        grad_floor=DEFAULT_GRAD_FLOOR, fixed_ocean=False)

    parameters = selfgrav_dtn_iterative_solver_parameters(
        condensed=layout.condensed, outer_rtol=OUTER_RTOL,
        block0_rtol=BLOCK0_RTOL, block0_max_it=options.block0_max_it,
        # The stopping rule of the outer FGMRES: an optional absolute floor
        # (`--outer_atol`, off by default) and a capped solve counted as
        # converged so Newton finishes the step. See the preset's docstring.
        outer_atol=options.outer_atol, outer_max_it=options.outer_max_it,
        outer_accept_capped=True,
        snes_type="newtonls",
        dtn_representation=DTN_REPRESENTATION,
        n_real=len(layout.real_fields))
    # The one rebuild rule of the preconditioner (block 0 and the Real-block
    # complement together). Without the options the library default holds:
    # `lagged`, every 5 nonlinear solves, at once when dt changes.
    if options.gia_rebuild is not None:
        parameters["gia_rebuild"] = options.gia_rebuild
    if options.gia_rebuild_every is not None:
        parameters["gia_rebuild_every"] = options.gia_rebuild_every
    if options.gia_rebuild_max_outer is not None:
        parameters["gia_rebuild_max_outer"] = options.gia_rebuild_max_outer
    # `invert_exp10.py --diagnostics` records the iterations of every KSP of
    # every solve in its own table (`exp10_diagnostics`). The converged-
    # reason lines of the preset (two per block-0 iteration on the condensed
    # route) then only fill stdout, so they are removed. A converged-reason
    # key is a viewer and changes no solve. Every other caller passes options
    # without `diagnostics` and keeps the preset unchanged.
    if getattr(options, "diagnostics", False):
        for key in [k for k in parameters if k.endswith("ksp_converged_reason")]:
            del parameters[key]

    # A rigid rotation of the mantle moves no water and has no mass moment
    # for a spherically symmetric density, so it stays in the kernel with sea
    # level and the frame rows on. Declared as nullspace and transpose
    # nullspace: the low-rank operator then removes it from its action.
    nullspace = rigid_rotation_nullspace(Z, layout)
    solver = SelfGravitatingGIASolver(
        z, approximation, layout=layout, dt=dt, bcs={},
        fluid_core=fluid_core, sea_level=sea_level,
        dtn_representation=DTN_REPRESENTATION,
        rotation_moments={"C": constants["C_nondim"], "k_s": options.k_s},
        Omega_sq=scales.Omega_sq, surface_radius=scales.Re,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=parameters)
    record = {
        "n_elements": n_elements,
        "power_law_active": bool(power_law),
        "k_s": options.k_s,
        "C_minus_A_nondim": float(
            solver.rotation_moments.get("C_minus_A", np.nan)),
        "gia_rebuild": parameters.get("gia_rebuild", "lagged (default)"),
        "gia_rebuild_every": parameters.get("gia_rebuild_every", "default"),
        "gia_rebuild_max_outer": parameters.get("gia_rebuild_max_outer", 0.0),
        "block0_max_it": options.block0_max_it,
        "outer_atol": options.outer_atol,
        "outer_max_it": options.outer_max_it,
        "unknowns": solver.solution.function_space().dim(),
        "real_rows": len(layout.real_fields),
        "solver_parameters": {k: str(v) for k, v in parameters.items()},
    }
    return solver, layout, record


def iteration_counts(solver):
    """The Newton and outer Krylov iteration counts of the last solve."""
    snes = solver.solver.snes
    return int(snes.getIterationNumber()), int(snes.ksp.getIterationNumber())


# --------------------------------------------------------------------------
# Logs and records
# --------------------------------------------------------------------------


class RunLog:
    """Log lines to stdout and to a file, on rank 0.

    The file keeps the record of the run next to its output, so a run can be
    read later without the Slurm output file. The file is opened in append
    mode, so the legs of a restarted run go into one log.
    """

    def __init__(self, path):
        self.handle = open(path, "a") if COMM_WORLD.rank == 0 else None

    def __call__(self, text):
        log(text)
        if self.handle is not None:
            self.handle.write(text + "\n")
            self.handle.flush()


def plain_json(value):
    """`value` with NumPy scalars and arrays replaced by Python numbers and lists.

    `json` cannot write a NumPy type, and the records collect NumPy floats
    from `assemble` and from the grids.
    """
    if isinstance(value, dict):
        return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, data):
    """Write `data` as indented JSON on rank 0."""
    if COMM_WORLD.rank == 0:
        with open(path, "w") as handle:
            json.dump(plain_json(data), handle, indent=1)


def sha256sum(filename):
    """The SHA-256 of a file, as a hexadecimal string."""
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5sum(filename):
    """The MD5 of a file, as a hexadecimal string (the mesh records use it)."""
    digest = hashlib.md5()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(directory):
    """Software versions and the state of the source tree (rank 0 only).

    The record holds the git commit of the G-ADOPT tree, the list of changed
    files with the SHA-256 of each (the text of `git diff` depends on the
    git version), and the SHA-256 of every Python file of the experiment
    directory.

    Args:
      directory: the experiment directory.
    """
    info = {"python": sys.version.split()[0],
            "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
            "petsc_int_bytes": int(np.dtype(PETSc.IntType).itemsize)}
    try:
        import importlib.metadata as md
        info["firedrake"] = md.version("firedrake")
    except Exception:  # noqa: BLE001  (a version is a record, not a need)
        info["firedrake"] = None
    try:
        root = subprocess.run(["git", "-C", directory, "rev-parse",
                               "--show-toplevel"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        status = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                                capture_output=True, text=True, timeout=60)
        info["git_head"] = head.stdout.strip()
        info["git_status"] = status.stdout.splitlines()
        changed = {}
        for line in info["git_status"]:
            path = os.path.join(root, line[3:].strip())
            files = ([os.path.join(d, f) for d, _, fs in os.walk(path)
                      for f in fs] if os.path.isdir(path) else [path])
            for name in sorted(files):
                if os.path.isfile(name) and "__pycache__" not in name:
                    changed[os.path.relpath(name, root)] = sha256sum(name)
        info["changed_files_sha256"] = changed
    except Exception as exc:  # noqa: BLE001
        info["git_error"] = repr(exc)
    info["scripts_sha256"] = {
        name: sha256sum(os.path.join(directory, name))
        for name in sorted(os.listdir(directory)) if name.endswith(".py")}
    return info


#: The ranks of one node, for the memory total that the node limit applies to.
_NODE_COMM = None


def memory_now(label, say=log):
    """Log the resident memory now: mean and largest per rank, largest node.

    Collective. The resident size comes from `/proc/self/statm` (Linux), so
    it is the memory in use at this moment, not the peak. The node total is
    the sum over the ranks of one node, which is what the Slurm memory limit
    of a node applies to. On a system without `/proc` it logs nothing.

    Args:
      label: the point of the run, for the log line.
      say: the log function.

    Returns:
      The largest node total in GB, or None without `/proc`.
    """
    global _NODE_COMM
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return None
    if _NODE_COMM is None:
        _NODE_COMM = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    node = _NODE_COMM.allreduce(rss, op=MPI.SUM)
    mean = COMM_WORLD.allreduce(rss, op=MPI.SUM) / COMM_WORLD.size
    largest = COMM_WORLD.allreduce((rss, COMM_WORLD.rank), op=MPI.MAXLOC)
    node_max = COMM_WORLD.allreduce(node, op=MPI.MAX)
    say(f"  memory {label}: rank mean {mean:.2f} GB, largest {largest[0]:.2f} "
        f"GB (rank {largest[1]}), largest node {node_max:.0f} GB")
    return node_max


def peak_memory_gb():
    """The largest peak resident memory over the ranks, GB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS bytes.
    gb = rss / 1e6 if sys.platform.startswith("linux") else rss / 1e9
    return COMM_WORLD.allreduce(gb, op=MPI.MAX)


def elapsed(tic):
    """Seconds since `tic`, on rank 0, broadcast so every rank agrees."""
    return COMM_WORLD.bcast(time.time() - tic, root=0)
