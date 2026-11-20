"""Driver of the `parallel` case: the checks that need more than one rank.

Usage: `python gia_parallel.py --entry <name>`, under `mpiexec -n 2` for an
entry whose name ends in `-np2`, with the entry names of
`gia_entries.PARALLEL`.

The unit tests in `tests/unit` run on one rank. A property that only a run
on several ranks can show (a reduction that a rank skips, a collective that
one rank leaves early, a file read on another partition than the writer's)
is an entry of this case. Each entry writes its measured values, and 1 or 0
for a refusal or a flag, as `name value` lines to
`out/parallel-<entry>.dat`; rank 0 writes. A value that differs between
ranks is gathered first and written as a count or a largest difference. A
rank that raises or waits in a collective for ever leaves no file, and the
test of the entry then fails on the missing file. The test
`test_gia_parallel.py` holds the tolerance of the original unit test and
asserts; this script asserts nothing.

The entries reuse the configurations of the unit tests through their helper
functions, so that an entry measures what the original test measured. The
helpers come from these files of `tests/unit`, which this script puts on
`sys.path`:

- `test_viscoelastic_checkpointing.py`: `parent_and_submesh`, `fields_on`,
  `round_trip_error`, `SUBMESH_TAG`;
- `test_gia_nullspace_projection_adjoint.py`: `_small_case`,
  `_small_forward`, `_small_taped`, `_shifted`;
- `test_gia_rotation_moments.py`: `build_rotation`, `closure`, `K_S_SPADA`;
- `test_gia_gravity.py`: `polar_moment`, `RE`;
- `test_gravity_solver.py`: `annulus_mesh`, `annulus_source`,
  `solve_annulus`, `field_number_parameters`, `max_coefficient_difference`,
  `relative_l2_error`, `RMIN`, `RMAX`;
- `test_dtn_lowrank_parity.py`: `solve_both`;

and from the sibling driver `gia_limits.py`: `unit_meshes` (the gmsh annulus
of the unit fixture `test_gia_gravity.meshes`). When those unit files are
deleted, these helpers must move to `tests/gia` first.
"""

import argparse
import re
import sys

import gadopt  # imported before firedrake
import firedrake as fd
import numpy as np
from firedrake.adjoint import get_working_tape, stop_annotating, taylor_test
from gadopt import CylindricalDtN, GravitySolver, KernelProjectionBlock
from gadopt.dtn_lowrank import _assert_against_reference
from gadopt.gia_couplings import Rotation
from gadopt.viscoelastic_checkpointing import CheckpointField, written_rank_count
from mpi4py import MPI

import gia_march as M
from gia_entries import (CASE_DIR, CHECKPOINT_FIELDS, CHECKPOINT_FILES, CHECKPOINT_MESHES,
                         DTN_GUARD_MESSAGE, GRAVITY_MATFREE_APPLICATIONS,
                         GRAVITY_REPRESENTATIONS, PARALLEL, WRITTEN_RANKS_MESSAGE,
                         parallel_checkpoint, parallel_values)
from gia_limits import unit_meshes

sys.path.insert(0, str(CASE_DIR.parent / "unit"))
import test_dtn_lowrank_parity as TL  # noqa: E402  (after the path of tests/unit)
import test_gia_gravity as G  # noqa: E402
import test_gia_nullspace_projection_adjoint as TP  # noqa: E402
import test_gia_rotation_moments as TR  # noqa: E402
import test_gravity_solver as TG  # noqa: E402
import test_viscoelastic_checkpointing as TV  # noqa: E402


# -- dtn-guard-np2 ------------------------------------------------------------------

#: The annulus and the trace of the reference check: the outer boundary of a
#: radially extruded annulus, the exterior modes of `CylindricalDtN(3)`, the
#: quadrature degree of the trace integrals, and the tolerance of the check.
DTN_RMIN, DTN_RMAX = 1.22, 2.22
DTN_QUAD_DEGREE = 12
DTN_SIDE, DTN_RTOL = "exterior", 1e-13
DTN_TRUNCATION = 3


def dtn_annulus():
    """A radially extruded annulus; its partition does not matter here.

    The divergence between ranks that the entry tests is produced by
    zeroing the rows on one rank, not by a partition that leaves a rank
    without boundary facets. Extrusion partitions laterally, so every rank
    owns part of both boundaries, and the entry does not depend on the
    partitioner.
    """
    heights = list(np.diff(np.linspace(DTN_RMIN, DTN_RMAX, 5)))
    base = fd.CircleManifoldMesh(48, radius=DTN_RMIN, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=len(heights), layer_height=heights,
                           extrusion_type="radial")
    mesh.cartesian = False
    return mesh


def dtn_pieces(mesh):
    """The keyword arguments of `_assert_against_reference` with clean rows.

    The rows `int mode v ds_t` are assembled from the same symbolic modes as
    the reference inside the check, so a clean call finds no deviation, and
    a changed row is the only source of a deviation.

    Args:
      mesh: the annulus of `dtn_annulus`.

    Returns:
      A dictionary of the arguments; `rows` has one row per mode, over the
      local degrees of freedom of this rank.
    """
    descriptor = CylindricalDtN(DTN_TRUNCATION)
    V = fd.FunctionSpace(mesh, "CG", 1)
    v = fd.TestFunction(V)
    measure = fd.ds_t(domain=mesh, degree=DTN_QUAD_DEGREE)
    X = fd.SpatialCoordinate(mesh)
    keys = [mode.key for mode in descriptor.mode_metadata(DTN_SIDE, DTN_RMAX)]
    rows = np.array([
        np.asarray(fd.assemble(mode.expr * v * measure).dat.data_ro, dtype=float)
        for mode in descriptor.modes(DTN_SIDE, DTN_RMAX, X)])
    return dict(rows=rows, keys=keys, descriptor=descriptor, side=DTN_SIDE,
                radius=DTN_RMAX, v=v, measure=measure, mesh=mesh,
                trace_degree=DTN_QUAD_DEGREE // 2, rtol=DTN_RTOL)


def guarded_check(pieces):
    """Run the reference check; return the message of its `RuntimeError`, or `""`.

    Any other exception propagates, so that the entry leaves no file.
    """
    try:
        _assert_against_reference(**pieces)
    except RuntimeError as error:
        return str(error)
    return ""


def run_dtn_guard():
    """The reference check of the low-rank DtN takes the same branch on every rank.

    The check computes the scale of the rows and the worst mode. Both must be
    reductions over all ranks, because the check assembles the reference rows
    collectively after them: a rank-local early return or a rank-local raise
    leaves the other ranks waiting in a collective for ever. Three calls
    (`test_dtn_lowrank_parallel.py`):

    - clean rows: no rank raises (the positive control);
    - the rows of every rank but rank 0 zeroed, the state of a rank that owns
      no degree of freedom on the boundary: every rank enters the reference
      assemblies and raises the same `RuntimeError`, and every rank reaches
      the line after the call;
    - a different mode changed on each rank: every rank raises with the same
      message, so the worst mode is chosen from global values.

    Returns:
      `ranks`; `clean_raised_ranks` (expected 0); `empty_raised_ranks` (ranks
      whose message matches `DTN_GUARD_MESSAGE`) and `empty_reached_ranks`
      (the sum of 1 over the ranks after the call), both expected `ranks`;
      `mode_raised_ranks` (expected `ranks`) and `mode_messages_agree` (1 if
      every rank gave the message of rank 0, else 0).
    """
    comm = fd.COMM_WORLD
    mesh = dtn_annulus()
    out = {"ranks": comm.size}

    # The positive control: without it the poisoned calls prove nothing.
    message = guarded_check(dtn_pieces(mesh))
    out["clean_raised_ranks"] = comm.allreduce(int(bool(message)))

    # Zeroed rows on every rank but rank 0.
    pieces = dtn_pieces(mesh)
    if comm.rank != 0:
        pieces["rows"] = np.zeros_like(pieces["rows"])
    message = guarded_check(pieces)
    out["empty_raised_ranks"] = comm.allreduce(
        int(re.search(DTN_GUARD_MESSAGE, message) is not None))
    # Every rank reaches this line only if no rank took a rank-local return.
    out["empty_reached_ranks"] = comm.allreduce(1)

    # A different mode changed on each rank, so a rank-local worst mode
    # would give a different message on each rank.
    pieces = dtn_pieces(mesh)
    target = 0 if comm.rank == 0 else len(pieces["keys"]) - 1
    pieces["rows"] = pieces["rows"].copy()
    pieces["rows"][target] *= 1.0 + (comm.rank + 1) * 1e-3
    message = guarded_check(pieces)
    messages = comm.allgather(message)
    out["mode_raised_ranks"] = sum(1 for m in messages if m)
    out["mode_messages_agree"] = 1 if all(m == messages[0] for m in messages) else 0
    return out


# -- checkpoint entries -------------------------------------------------------------

#: The meshes of the files of the checkpoint entries: label, and whether the
#: field lives on the submesh (`True`) or on the parent itself.
FILE_MESHES = (("parent", False), ("submesh", True))


def write_file(path, comm, *, with_points=True):
    """Write the parent and every field of the parent and the submesh.

    Args:
      path: the file.
      comm: the communicator of the writer.
      with_points: store through `CheckpointField.save` (with the point
        coordinates), or through `encode` and `save_function` alone, the
        layout of a file without coordinates.
    """
    labelled, sub = TV.parent_and_submesh(comm, split_by_rank=False)
    with fd.CheckpointFile(str(path), "w", comm=comm) as chk:
        chk.save_mesh(labelled)
        for label, on_sub in FILE_MESHES:
            for name, field in TV.fields_on(sub if on_sub else labelled).items():
                checkpoint = CheckpointField(f"{label}_{name}", field, labelled)
                if with_points:
                    checkpoint.save(chk)
                else:
                    chk.save_function(checkpoint.encode())


def read_file(path, comm, *, direct=False):
    """Read every field of `write_file` on `comm`; return the largest errors.

    The reference is each field computed again on the loaded mesh. Its
    values differ from the written ones by round-off, because a quadrature
    point computed from another vertex order of the cell is another
    floating-point number.

    Args:
      path: the file.
      comm: the communicator of the reader.
      direct: restore the values in the stored node order, without the
        pairing by position (the control of the entry `checkpoint-read-np2`).

    Returns:
      `(error, exact)`: the largest relative error of any field against the
      recomputed values, and whether every field re-encodes to the stored
      encoding bit for bit.
    """
    error, exact = 0.0, True
    with fd.CheckpointFile(str(path), "r", comm=comm) as chk:
        labelled = chk.load_mesh(fd.mesh.DEFAULT_MESH_NAME)
        sub = fd.Submesh(labelled, labelled.topological_dimension, TV.SUBMESH_TAG)
        for label, on_sub in FILE_MESHES:
            for name, reference in TV.fields_on(sub if on_sub else labelled).items():
                field = fd.Function(reference.function_space())
                checkpoint = CheckpointField(f"{label}_{name}", field, labelled)
                if direct:
                    checkpoint.restore(chk.load_function(labelled, checkpoint.name))
                else:
                    checkpoint.load(chk)
                difference = np.max(np.abs(field.dat.data_ro - reference.dat.data_ro),
                                    initial=0.0)
                scale = np.max(np.abs(reference.dat.data_ro), initial=0.0)
                error = max(error, comm.allreduce(float(difference), op=MPI.MAX)
                            / comm.allreduce(float(scale), op=MPI.MAX))
                stored = chk.load_function(labelled, checkpoint.name)
                same = np.array_equal(checkpoint.encode().dat.data_ro, stored.dat.data_ro)
                exact &= comm.allreduce(bool(same), op=MPI.LAND)
    return error, exact


def run_checkpoint_round_trip():
    """`CheckpointField` round trips on the parent, a submesh and an extruded mesh.

    On two ranks the submesh is the set of cells of the last rank, so the
    other rank owns no cell of it and has to reshape an empty array
    (`test_viscoelastic_checkpointing.py::test_round_trip_two_ranks`). Each
    field is encoded, set to zero and restored; the error must be exactly 0.

    Returns:
      `ranks`; `empty_submesh_rank` (1 if some rank owns no cell of the
      submesh); `<mesh>_<field>_error`, the largest error over all ranks.
    """
    labelled, sub = TV.parent_and_submesh()
    comm = labelled.comm
    out = {"ranks": comm.size,
           "empty_submesh_rank": int(comm.allreduce(sub.cells.owned.local_size == 0,
                                                    op=MPI.LOR))}
    extruded = fd.ExtrudedMesh(fd.UnitSquareMesh(3, 3), 3)
    # The mesh of each field and the parent its stored field lives on, in the
    # order of `CHECKPOINT_MESHES`.
    layouts = {"parent": (labelled, labelled), "submesh": (sub, labelled),
               "extruded": (extruded, extruded)}
    for label in CHECKPOINT_MESHES:
        mesh, parent = layouts[label]
        fields = TV.fields_on(mesh)
        for name in CHECKPOINT_FIELDS:
            out[f"{label}_{name}_error"] = TV.round_trip_error(name, fields[name], parent)
    return out


def run_checkpoint_write(name):
    """Write the files of the entry `name` on the ranks of this process.

    `write_file` writes the parent and every
    field of the parent and of its left-half submesh, with the coordinates of
    the quadrature points (`with-points`) or without them
    (`without-points`).

    Returns:
      `ranks`, the number of ranks of the writer.
    """
    comm = fd.COMM_WORLD
    for label in CHECKPOINT_FILES[name]:
        path = CASE_DIR / parallel_checkpoint(label)
        write_file(path, comm, with_points=label.endswith("-with-points"))
    return {"ranks": comm.size}


def run_checkpoint_read_two_ranks():
    """Read on two ranks the file of one rank, and the files of two ranks.

    `test_quadrature_fields_read_on_another_number_of_ranks` and
    `test_a_file_without_point_coordinates_is_refused_on_another_number_of_ranks`:
    the fields of the one-rank file paired by position against the fields
    computed again on the loaded mesh; the same file without the pairing (the
    control: the partitions differ, so the values land on other points); the
    files of two ranks, with and without point coordinates, re-encode to the
    stored encoding bit for bit on the writer's number of ranks.

    Returns:
      `ranks`; `one_error` (largest relative error); `one_control` (the
      same without the pairing); `two_exact` and `two_without_points_exact`
      (1 if bit for bit, else 0).
    """
    comm = fd.COMM_WORLD
    out = {"ranks": comm.size}
    one = CASE_DIR / parallel_checkpoint("one-with-points")
    out["one_error"], _ = read_file(one, comm)
    out["one_control"], _ = read_file(one, comm, direct=True)
    _, exact = read_file(CASE_DIR / parallel_checkpoint("two-with-points"), comm)
    out["two_exact"] = int(exact)
    _, exact = read_file(CASE_DIR / parallel_checkpoint("two-without-points"), comm)
    out["two_without_points_exact"] = int(exact)
    return out


def run_checkpoint_read_one_rank():
    """Read on one rank the files of two ranks.

    The file with point coordinates is paired by position; the file without
    them names 2 as the number of ranks of its writer (the number Firedrake
    stores with the distribution of the mesh) and is refused with a
    `ValueError` that names it.

    Returns:
      `ranks`; `two_error` (largest relative error);
      `without_points_writer_ranks`; `without_points_refused` (1 if the read
      raised the `ValueError` that matches `WRITTEN_RANKS_MESSAGE`, else 0).
    """
    comm = fd.COMM_WORLD
    out = {"ranks": comm.size}
    out["two_error"], _ = read_file(CASE_DIR / parallel_checkpoint("two-with-points"), comm)
    path = CASE_DIR / parallel_checkpoint("two-without-points")
    with fd.CheckpointFile(str(path), "r", comm=comm) as chk:
        out["without_points_writer_ranks"] = written_rank_count(chk)
    try:
        read_file(path, comm)
        refused = 0
    except ValueError as error:
        refused = int(re.search(WRITTEN_RANKS_MESSAGE, str(error)) is not None)
    out["without_points_refused"] = refused
    return out


# -- projection-np2 -----------------------------------------------------------------

def run_projection():
    """The recorded kernel projection on two ranks.

    The small mixed problem of `test_gia_nullspace_projection_adjoint.py`
    (a unit square, a rotation basis on the vector block, a control on the
    scalar block), taped once. On two ranks the products with the basis are
    global, so a rank-local product in the forward, the replay or the
    adjoint of `KernelProjectionBlock` changes the numbers below
    (`test_small_parallel_replay_and_gradient`).

    Returns:
      `ranks`; `blocks` (the number of `KernelProjectionBlock` on the tape);
      `J` and `J_replay` (the replay at the recorded control); `J_fresh`,
      `J_replay_shifted` and `J_unprojected` at the shifted control (a fresh
      run, the replay, a fresh run without the projection); `dJ` (the
      directional derivative from the adjoint); `response` (`|J(m + 0.01 h)
      - J(m)|`); `taylor_rate` and `taylor_rate_no_gradient` (the smallest
      rates of `pyadjoint.taylor_test` with the adjoint derivative and with
      0).
    """
    Z, nullspace, control, direction = TP._small_case()
    J, rf = TP._small_taped(Z, nullspace, control)
    out = {"ranks": fd.COMM_WORLD.size, "J": float(J)}
    out["blocks"] = sum(1 for b in get_working_tape().get_blocks()
                        if isinstance(b, KernelProjectionBlock))
    out["J_replay"] = float(rf(control))
    shifted = TP._shifted(control, direction)
    with stop_annotating():
        out["J_fresh"] = float(TP._small_forward(Z, nullspace, shifted))
        out["J_unprojected"] = float(TP._small_forward(Z, nullspace, shifted,
                                                       project=False))
    out["J_replay_shifted"] = float(rf(shifted))

    # The guards of `test_gravity_adjoint.assert_taylor_with_guards`, with the
    # replay state set back to the recorded control before each sweep.
    rf(control)
    out["dJ"] = direction._ad_dot(rf.derivative())
    with stop_annotating():
        step = control._ad_add(direction._ad_mul(0.01))
    J0 = float(rf(control))
    out["response"] = abs(float(rf(step)) - J0)
    rf(control)
    out["taylor_rate"] = taylor_test(rf, control, direction)
    out["taylor_rate_no_gradient"] = taylor_test(rf, control, direction, dJdm=0.0)
    return out


# -- rotation-radius-np2 ------------------------------------------------------------

def run_rotation_radius():
    """The radius check of `Rotation` gives the same answer on every rank.

    `Rotation.check` compares `surface_radius` with the largest coordinate
    radius of the mechanics mesh over the whole communicator. On two ranks
    the annulus of the unit fixture splits so that one rank does not reach
    the outer boundary, so a check on the local radius would refuse the
    correct radius on that rank alone, inside a collective constructor
    (`test_gia_rotation_moments.py::test_the_radius_check_agrees_across_ranks`).
    The entry builds the 2-D solver with the `k_s` spelling and the radius
    `RE`, and reads the polar moment that closes the `m_3` row.

    Returns:
      `ranks`; `surface_radius` and `RE`; `closure_m3` and `polar_moment`
      (assembled from the mantle); `local_radius_min` and `local_radius_max`
      (the smallest and the largest over the ranks of the local largest
      coordinate radius, as the check computes it); `radius_rtol`
      (`Rotation.SURFACE_RADIUS_RTOL`).
    """
    meshes = unit_meshes()
    _, rotation = TR.build_rotation(meshes, k_s=TR.K_S_SPADA, surface_radius=G.RE)
    coords = meshes[1].coordinates.dat.data_ro
    local = float(np.sqrt((coords ** 2).sum(axis=1)).max()) if coords.size else 0.0
    radii = fd.COMM_WORLD.allgather(local)
    return {"ranks": fd.COMM_WORLD.size,
            "surface_radius": float(rotation.surface_radius), "RE": G.RE,
            "closure_m3": TR.closure(rotation, 2), "polar_moment": G.polar_moment(meshes),
            "local_radius_min": min(radii), "local_radius_max": max(radii),
            "radius_rtol": Rotation.SURFACE_RADIUS_RTOL}


# -- gravity-np2 --------------------------------------------------------------------

def gravity_parity(out):
    """Low-rank against multiplier DtN on two ranks (`test_parallel_parity`).

    The low-rank apply sums rank-local products of the mode vectors through
    one `Allreduce`; an ownership mistake there shows only on several ranks.
    The two representations discretise the same operator, so the potentials
    agree to round-off. The trace coefficients are a reduction too: every
    rank must hold the same values
    (`test_parallel_coefficients_are_rank_consistent`).
    """
    mesh = TG.annulus_mesh(n_azimuthal=96, dr=0.25)
    rho = TG.annulus_source(mesh)
    results = TL.solve_both(mesh, rho,
                            lambda: {"top": {"dtn": CylindricalDtN(M=5)},
                                     "bottom": {"dtn": CylindricalDtN(M=5)}})
    (psi_m, solver_m), (psi_l, solver_l) = results["multiplier"], results["lowrank"]
    out["parity_psi_error"] = TG.relative_l2_error(psi_l, psi_m, mesh)
    out["parity_coefficient_error"] = TG.max_coefficient_difference(
        solver_m.coefficients(), solver_l.coefficients())

    # Every rank reports the same trace spectrum of a low-rank solve.
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    solver = GravitySolver(psi, rho,
                           bcs={"top": {"dtn": CylindricalDtN(M=4)},
                                "bottom": {"dtn": CylindricalDtN(M=4)}},
                           dtn_representation="lowrank")
    solver.solve()
    local = solver.coefficients()
    gathered = fd.COMM_WORLD.allgather(local)
    out["coefficients_count"] = sum(len(values) for values in local.values())
    out["coefficients_rank_difference"] = max(
        TG.max_coefficient_difference(local, other) for other in gathered)


def gravity_monopole(out, representation):
    """Repeated solves of one solver keep the enclosed mass on two ranks.

    `test_parallel_repeated_monopole_solves_are_stable`: a cached linear
    solver whose right-hand side holds a facet integral into a `Real` space
    gave the right value on its first solve and wrong values afterwards on
    several ranks on an earlier Firedrake. A `sigma` sheet on the top and a
    `flux` condition at the bottom both route mass through facet integrals.
    The first mass is compared with the closed form `pi (RMAX^2 - RMIN^2) +
    2 pi RMAX sigma + cavity`; the later masses and the norms of the
    potential with the first ones.
    """
    sigma, cavity = 0.3, 2.0
    mesh = TG.annulus_mesh(n_azimuthal=64, dr=0.25)
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
    solver = GravitySolver(
        psi, 1.0,
        bcs={"top": {"dtn": CylindricalDtN(M=3), "sigma": fd.Constant(sigma)},
             "bottom": {"flux": fd.Constant(2 * cavity / TG.RMIN)}},
        solver_parameters="direct", dtn_representation=representation)
    masses, norms = [], []
    for _ in range(4):
        solver.mixed_solution.assign(0)
        solver.solve()
        masses.append(solver.total_enclosed_mass())
        norms.append(float(np.sqrt(fd.assemble(psi ** 2 * fd.dx))))
    prefix = f"monopole_{representation}"
    out[f"{prefix}_mass"] = masses[0]
    out[f"{prefix}_mass_expected"] = (np.pi * (TG.RMAX ** 2 - TG.RMIN ** 2)
                                      + 2 * np.pi * TG.RMAX * sigma + cavity)
    out[f"{prefix}_mass_drift"] = max(abs(m - masses[0]) for m in masses) / abs(masses[0])
    out[f"{prefix}_norm_drift"] = max(abs(n - norms[0]) for n in norms) / abs(norms[0])


def gravity_split(out):
    """The split by index sets against the split by field numbers on two ranks.

    `test_parallel_matches_the_field_number_split`: the `Real` multipliers all
    sit on rank 0, so their index sets are empty on the other rank. The
    iterative preset (index sets) must agree with the same Schur split given
    by field numbers, and with the direct preset.
    """
    mesh = TG.annulus_mesh(n_azimuthal=64, dr=0.25)
    rho = TG.annulus_source(mesh)
    psi_new, solver_new = TG.solve_annulus(mesh, rho, 5, "iterative")
    psi_old, solver_old = TG.solve_annulus(
        mesh, rho, 5, TG.field_number_parameters("iterative", solver_new._multiplier_offset,
                                                 solver_new.n_multipliers))
    psi_direct, solver_direct = TG.solve_annulus(mesh, rho, 5, "direct")
    out["split_psi_error"] = TG.relative_l2_error(psi_new, psi_old, mesh)
    out["split_coefficient_error"] = TG.max_coefficient_difference(
        solver_old.coefficients(), solver_new.coefficients())
    out["direct_psi_error"] = TG.relative_l2_error(psi_new, psi_direct, mesh)
    out["direct_coefficient_error"] = TG.max_coefficient_difference(
        solver_direct.coefficients(), solver_new.coefficients())


def gravity_matfree(out):
    """Repeated applications of the matrix-free Jacobian on two ranks.

    `test_parallel_repeated_matfree_application_is_stable`: on several
    ranks, a mixed operator with `Real` blocks assembled matrix-free could be
    right on its first application and gain a copy of a volume contribution
    in each `Real` row afterwards. Each application is compared with the
    nested assembled operator on one seeded random vector.
    """
    mesh = TG.annulus_mesh(n_azimuthal=32, dr=0.5)
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    solver = GravitySolver(psi, TG.annulus_source(mesh),
                           bcs={"top": {"dtn": CylindricalDtN(M=3)},
                                "bottom": {"dtn": CylindricalDtN(M=3)}})
    jacobian = fd.derivative(solver.F, solver.mixed_solution,
                             fd.TrialFunction(solver.mixed_space))
    matfree = fd.assemble(jacobian, mat_type="matfree").petscmat
    reference = fd.assemble(jacobian, mat_type="nest").petscmat
    x = matfree.createVecRight()
    # The same global vector on every rank count, from one seeded sequence.
    low, high = x.getOwnershipRange()
    x.setValues(range(low, high),
                np.random.default_rng(42).standard_normal(x.getSize())[low:high])
    x.assemble()
    expected = reference.createVecLeft()
    reference.mult(x, expected)
    for application in range(GRAVITY_MATFREE_APPLICATIONS):
        y = matfree.createVecLeft()
        matfree.mult(x, y)
        y.axpy(-1.0, expected)
        out[f"matfree_error_{application}"] = y.norm() / expected.norm()


def run_gravity():
    """`GravitySolver` alone on two ranks: the rows of `gravity_*`.

    Returns:
      `ranks` and the values of `gravity_parity`, `gravity_monopole` (for
      each representation of `GRAVITY_REPRESENTATIONS`), `gravity_split` and
      `gravity_matfree`.
    """
    out = {"ranks": fd.COMM_WORLD.size}
    gravity_parity(out)
    for representation in GRAVITY_REPRESENTATIONS:
        gravity_monopole(out, representation)
    gravity_split(out)
    gravity_matfree(out)
    return out


RUNS = {
    "dtn-guard-np2": run_dtn_guard,
    "checkpoint-round-trip-np2": run_checkpoint_round_trip,
    "checkpoint-write": lambda: run_checkpoint_write("checkpoint-write"),
    "checkpoint-write-np2": lambda: run_checkpoint_write("checkpoint-write-np2"),
    "checkpoint-read": run_checkpoint_read_one_rank,
    "checkpoint-read-np2": run_checkpoint_read_two_ranks,
    "projection-np2": run_projection,
    "rotation-radius-np2": run_rotation_radius,
    "gravity-np2": run_gravity,
}
assert set(RUNS) == set(PARALLEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=sorted(PARALLEL))
    name = parser.parse_args().entry
    print(f"gadopt from {gadopt.__file__}", flush=True)
    values = RUNS[name]()
    M.write_values(CASE_DIR / parallel_values(name), values)
