"""The new self-gravitating GIA solver: steps, adjoint, state and interface.

Gate 1 (`test_gia_solver_gate.py`) compares the equations with the old
solver. These tests run the new solver itself: Taylor tests with `Real`
controls in the row scalings, the in-memory state round trip and the
refusals of `restore`, and the interface rules (the order of the couplings,
the refusals, the nullspace check, the presets, `approximation.mu`, the
closure constant).

The forward steps against the stored states of the earlier solver, the
conversion of its checkpoint files, the restart from a `CheckpointFile`,
the Taylor and replay checks of a taped march (also through a restore and
under disk checkpointing) and the capped derivative solve are entries of
the `forward` and `adjoint` cases in `tests/gia` (`reference-quadrature`,
`convert-quadrature-bdf2`, `restart`, `sea-direct-viscosity`,
`sea-direct-viscosity-disk` and `guards`).
"""

import gc
import warnings
import weakref

import firedrake as fd
import numpy as np
import pytest
from firedrake import dmhooks
from firedrake.petsc import PETSc
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from pyadjoint.tape import annotate_tape
from ufl.algorithms import expand_derivatives, extract_coefficients

import gadopt
import gia_cases as C
from gadopt import gia_couplings as cp
from gadopt.gia_solver import SelfGravitatingGIASolver as NewSolver
from gadopt.gia_state import (CONFIGURATION_KEYS, configuration, has_state, read_state,
                              state_from_checkpoint, stored_configuration, write_state)
from gadopt.viscoelastic_checkpointing import written_rank_count
from test_gravity_adjoint import assert_taylor_with_guards

Case = C.Case
STEPS = (.08, .13, .07)


@pytest.fixture(scope="module")
def meshes():
    return C.annulus_meshes()


@pytest.fixture(autouse=True)
def clean_tape():
    get_working_tape().clear_tape()
    yield
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


def surface_ice(sea, k):
    """A small ice load that grows with the step `k`, over a grounding line."""
    X = fd.SpatialCoordinate(sea.I.function_space().mesh())
    theta = fd.atan2(X[1], X[0])
    sea.I.interpolate(.002 * (1 + .2 * k) * (1 - fd.cos(theta)) / 2)


#: The configuration of the stored states of the earlier solver. The
#: recording script `tests/unit/data/record_gia_references/record_all.py`
#: reads it; the `reference-quadrature` entry of the `forward` case in
#: `tests/gia` compares with those states.
FORWARD = [
    Case(family="Quadrature", scheme="bdf2", representation="lowrank",
         model="maxwell2", core="fluid", centre=True),
]


# -- Taylor and replay -----------------------------------------------------------

TAPED = [
    pytest.param(Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                      model="maxwell2", core="fluid", centre=True),
                 id="quadrature-bdf2-fluid_core-centre_of_mass"),
]


def march(meshes, case, control, steps=STEPS[:2], *, checkpoint_dir=None,
          restore_after=None, **kwargs):
    """A taped (or not) march; the functional reads `u`, `psi` and the history.

    Args:
      checkpoint_dir: march under `SingleDiskStorageSchedule` with disk
        checkpointing in this directory.
      restore_after: after this many steps, copy the state, build a second
        solver and `restore` the copy into it, and continue with that one;
        the copy and the restore are on the tape.
      kwargs: further arguments of the solver.
    """
    sub = meshes[1]

    def build():
        with stop_annotating():
            solver, u, psi = C.build_new(
                meshes, case, control=control, dt=dt, sea=sea,
                nullspace="rigid_rotation" if case.core == "fluid" else None,
                **kwargs)
        return solver, u, psi

    with stop_annotating():
        sea = C.SeaFields(sub)
        dt = C.dt_function(sub)
    solver, u, psi = build()
    if checkpoint_dir is not None:
        from checkpoint_schedules import SingleDiskStorageSchedule
        from firedrake.adjoint import enable_disk_checkpointing
        if sub.comm.rank == 0:
            checkpoint_dir.mkdir()
        sub.comm.barrier()
        enable_disk_checkpointing(checkpoint_comm=sub.comm,
                                  checkpoint_dir=str(checkpoint_dir))
        get_working_tape().enable_checkpointing(
            SingleDiskStorageSchedule(), gc_timestep_frequency=1)
    iterator = enumerate(steps)
    if checkpoint_dir is not None and annotate_tape():
        iterator = get_working_tape().timestepper(iter(iterator))
    for k, h in iterator:
        if restore_after is not None and k == restore_after:
            copies = {}
            for name, f in solver.state().items():
                copies[name] = (f if name == "previous_dt" else
                                fd.Function(f.function_space().collapse()).assign(f))
            solver, u, psi = build()
            solver.restore(copies)
        dt.assign(h)
        if case.load:
            surface_ice(sea, k)
        solver.solve()
    parent = meshes[0]
    J = C.mantle_functional(solver, u, sub)
    J += fd.assemble(psi * psi * fd.dx(domain=parent))
    sea_level = [c for c in solver.couplings if isinstance(c, cp.SeaLevel)]
    if sea_level:
        # The objective of the inversions: a sea-level expression of the
        # coupling, which keeps its dependency on the tape.
        J += fd.assemble(sea_level[0].sea_level() ** 2 * sea_level[0].measure())
    return J


# -- State ------------------------------------------------------------------------
STATE_CASES = [
    Case(family="Quadrature", scheme="bdf2", representation="lowrank",
         model="maxwell2", core="fluid", centre=True),
]
FOUR = (.08, .13, .07, .12)


def run(solver, sea, steps, first=0):
    for k, h in enumerate(steps, start=first):
        solver.dt.assign(h)
        if sea is not None:
            surface_ice(sea, k)
        solver.solve()


def values(solver):
    """Copies of every coefficient array of the state, for a bitwise compare.

    Reading `data_ro` can start a halo exchange, which every rank must join,
    so each test takes the copies once per solver, on every rank, before it
    compares anything.
    """
    return {name: np.array(f.dat.data_ro, copy=True)
            for name, f in solver.state().items()}


#: Bound on the relative difference of a continued run from the
#: uninterrupted one on more than one rank. The direct presets factorise
#: block 0 with parallel MUMPS, whose solutions of one matrix differ between
#: two factorisations in one process by about 1e-14 relative (the ranks'
#: contributions are summed in an order set by the message timing), and four
#: steps carry that difference forward.
PARALLEL_RTOL = 1e-10


def states_differ(first, second, solver, rtol):
    """The names of the state entries of two `values` copies that differ.

    Each comparison is reduced over the ranks before it is used, so every
    rank gets the same list and takes the same branch afterwards. A rank
    that left a loop at a difference only it had seen would leave the other
    ranks waiting in the next collective call.

    Args:
      first, second: copies from `values`, with the same keys.
      solver: the solver of `second`, for the communicator and to tell the
        `Real` fields.
      rtol: 0 for a bitwise comparison. Otherwise the bound on the
        difference relative to the norm of the entry in `second`; a `Real`
        field is compared against the largest `Real` value of the state
        instead, because a `Real` field whose exact value is zero (the core
        pressure under a load with no degree-0 part) holds round-off only.
    """
    from mpi4py import MPI
    comm = solver.mesh.comm
    if rtol == 0:
        return [key for key in first
                if not comm.allreduce(bool(np.array_equal(first[key], second[key])),
                                      op=MPI.LAND)]
    reals = {name for name, f in solver.state().items()
             if f.function_space().ufl_element().family() == "Real"}
    real_scale = max([comm.allreduce(float(np.max(np.abs(second[k]), initial=0.0)),
                                     op=MPI.MAX) for k in reals] or [0.0])
    out = []
    for key in first:
        difference = np.sqrt(comm.allreduce(float(np.sum((first[key] - second[key]) ** 2))))
        size = np.sqrt(comm.allreduce(float(np.sum(second[key] ** 2))))
        scale = real_scale if key in reals else size
        if difference > rtol * scale:
            out.append(key)
    return out


def load_meshes(path, name, comm=fd.COMM_WORLD):
    """The parent from a checkpoint file and its curved mantle submesh.

    The file stores the parent with its P2 coordinates. `Submesh` does not
    inherit them, so the submesh is curved again.
    """
    with fd.CheckpointFile(str(path), "r", comm=comm) as chk:
        parent = chk.load_mesh(name)
    parent.cartesian = False
    sub = C.curve_mesh(fd.Submesh(parent, 2, C.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def relative(a, b):
    """`|a - b| / |b|` of the coefficient arrays, for Functions on two meshes
    loaded from one file, which have the same numbering."""
    x, y = np.ravel(a.dat.data_ro), np.ravel(b.dat.data_ro)
    comm = a.function_space().mesh().comm
    num = comm.allreduce(float(np.sum((x - y) ** 2)))
    den = comm.allreduce(float(np.sum(y ** 2)))
    return np.sqrt(num / den)


@pytest.mark.longtest
def test_state_round_trip_in_memory_and_reset(meshes):
    """`restore` of copied Functions, then `reset` back to rest.

    The continued runs agree bit for bit in serial and to `PARALLEL_RTOL` on
    more than one rank (`SelfGravitatingGIASolver.restore`).
    """
    case = STATE_CASES[0]
    sea = None
    a, u, psi = C.build_new(meshes, case, dt=C.dt_function(meshes[1]),
                            nullspace="rigid_rotation")
    run(a, sea, FOUR[:2])
    copies = {k: (f.copy(deepcopy=True) if k != "previous_dt" else f)
              for k, f in a.state().items()}
    b, ub, psib = C.build_new(meshes, case, dt=C.dt_function(meshes[1]),
                              nullspace="rigid_rotation")
    b.restore(copies)
    comm = b.mesh.comm

    def on_every_rank(flag):
        # One answer for all ranks, so that no rank leaves the test early.
        from mpi4py import MPI
        return comm.allreduce(bool(flag), op=MPI.LAND)

    assert on_every_rank(np.array_equal(ub.dat.data_ro, u.dat.data_ro))
    assert on_every_rank(np.array_equal(psib.dat.data_ro, psi.dat.data_ro))
    run(a, sea, FOUR[2:])
    run(b, sea, FOUR[2:])
    rtol = 0 if b.mesh.comm.size == 1 else PARALLEL_RTOL
    assert not states_differ(values(a), values(b), b, rtol=rtol)
    b.reset()
    assert b.time_integrator.previous_dt is None
    for key, f in b.state().items():
        if key != "previous_dt":
            assert on_every_rank(not np.any(f.dat.data_ro)), key
    assert on_every_rank(not np.any(ub.dat.data_ro))
    with pytest.raises(ValueError, match="Unknown state entries"):
        b.restore({**copies, "something": copies["displacement"]})


#: Checkpoint files that the earlier solver (`gadopt.gia_gravity` at commit
#: 2f4b1397) wrote after two steps on the mesh of `gia_cases.MESH_FILE`,
#: in the layouts of the drivers that ran it. Written once, in serial, by
#: `tests/unit/data/record_gia_references/record_checkpoints.py` from a
#: checkout of that commit, with the field lists of the drivers of that
#: time (the helpers it imports exist only there), then copied with gzip
#: compression in the HDF5 1.10 file format, which HDF5 reads transparently.
#: The mesh in every file is the curved parent, under its default name.
OLD_CHECKPOINTS = C.REFERENCE_DIR
OLD_MESH_NAME = "firedrake_default"

#: The cases of the stored files: the file name and the configuration. Each
#: file has the `selfgrav_common` layout: every sub-function of the mixed
#: solution as `solution_{i}` in the broken encoding, the BDF2 levels as
#: `time_*`, the quadrature history also as `internal_variables_0`.
OLD_CHECKPOINT_CASES = {
    "Quadrature-bdf2-selfgrav": (
        "gia_old_checkpoint_quadrature_bdf2.h5",
        Case(family="Quadrature", scheme="bdf2", representation="lowrank",
             model="maxwell2", core="fluid", centre=True)),
}


def stored_names(solver):
    """The names of the fields a driver of the earlier solver stored for `solver`.

    The earlier solver's mixed space had the same sub-spaces in the same
    order as this solver's, so the `selfgrav_common` layout has one
    `solution_{i}` per sub-function of this solver's mixed solution.
    """
    levels = (["time_current_0", "time_previous_0"]
              if solver.time_scheme == "bdf2" else [])
    names = [f"solution_{i}" for i in range(len(solver.solution.subfunctions))]
    if solver.family == "Quadrature":
        names.append("internal_variables_0")
    return names + levels


def _old_checkpoint_converts(label):
    """Restore this solver from a stored file of the earlier solver and continue.

    The file holds the earlier solver's state after two steps. A solver on
    the mesh loaded from the file restores the converted state. A second
    solver on the same loaded mesh marches the same two steps from rest; the
    restored state agrees with it, and after both take a third step they
    agree again. The bound 1e-11 covers the difference of the two solvers
    (round-off, measured below 1e-14 per field when they ran side by side),
    the round-off of the broken encoding of the `selfgrav_common` layout,
    and the different numbering of the degrees of freedom of the mesh on
    which the earlier solver ran and of the loaded mesh.

    Returns:
      `(solver, loaded, metadata)` of the restored run, for further checks.
    """
    filename, case = OLD_CHECKPOINT_CASES[label]
    path = str(OLD_CHECKPOINTS / filename)
    loaded_meshes = load_meshes(path, OLD_MESH_NAME)
    parent, sub = loaded_meshes
    nullspace = "rigid_rotation" if case.core == "fluid" else None
    new, u, psi = C.build_new(loaded_meshes, case, dt=C.dt_function(sub),
                              nullspace=nullspace)
    names = stored_names(new)
    with fd.CheckpointFile(path, "r") as chk:
        loaded = {n: chk.load_function(parent, n, idx=0) for n in names}
        history = chk.get_timestepping_history(parent, names[0])
        writer_ranks = written_rank_count(chk, OLD_MESH_NAME)
    metadata = {k: float(v[-1]) for k, v in history.items() if k != "index"}
    state = state_from_checkpoint(new, loaded, metadata, writer_ranks=writer_ranks)
    new.restore(state)
    if case.scheme == "bdf2":
        assert new.time_integrator.previous_dt == FOUR[1]

    reference, u_ref, psi_ref = C.build_new(loaded_meshes, case, dt=C.dt_function(sub),
                                            nullspace=nullspace)
    run(reference, None, FOUR[:2])

    def assert_agree(when):
        for name, a, b in [("displacement", u, u_ref), ("potential", psi, psi_ref),
                           ("history", new.internal_variables,
                            reference.internal_variables)]:
            assert relative(a, b) <= 1e-11, (when, name, relative(a, b))

    assert_agree("restored")
    run(new, None, FOUR[2:3], first=2)
    run(reference, None, FOUR[2:3], first=2)
    assert_agree("continued")
    return new, loaded, metadata


@pytest.mark.longtest
def test_restore_needs_previous_dt_on_bdf2_and_the_file_its_configuration(
        meshes, tmp_path):
    """A BDF2 state without `previous_dt`, and a file of another scheme, are refused."""
    from pathlib import Path
    bdf2 = Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                model="maxwell2", core="fluid", centre=True)
    a, _, _ = C.build_new(meshes, bdf2, nullspace="rigid_rotation")
    run(a, None, FOUR[:2])
    state = {k: f.copy(deepcopy=True) for k, f in a.state().items()}
    del state["previous_dt"]
    b, _, _ = C.build_new(meshes, bdf2, nullspace="rigid_rotation")
    with pytest.raises(ValueError, match="previous_dt"):
        b.restore(state)
    # A backward-Euler solver takes a state without it.
    be = bdf2.but(scheme="backward_euler")
    c, _, _ = C.build_new(meshes, be, nullspace="rigid_rotation")
    d, _, _ = C.build_new(meshes, be, nullspace="rigid_rotation")
    d.restore({k: f for k, f in c.state().items() if k != "previous_dt"})
    # A file written by the BDF2 solver is refused by a backward-Euler one of
    # the same fields otherwise.
    directory = Path(meshes[0].comm.bcast(str(tmp_path), root=0))
    with fd.CheckpointFile(str(directory / "s.h5"), "w") as chk:
        chk.save_mesh(meshes[0])
        write_state(chk, a, idx=0)
    other = bdf2.but(scheme="backward_euler")
    e, _, _ = C.build_new(meshes, other, nullspace="rigid_rotation")
    with fd.CheckpointFile(str(directory / "s.h5"), "r") as chk:
        with pytest.raises(ValueError, match="time_scheme"):
            read_state(chk, e, idx=0)


def test_a_file_tells_whether_it_holds_a_state_before_a_solver_reads_it(meshes, tmp_path):
    """`has_state` and `stored_configuration` on a path and on an open file.

    No time step: the state of a solver at construction is written. A file
    with the mesh only holds no state; a file of `write_state` holds one
    under its prefix and none under another, its stored configuration is
    the one of the writing solver (also when the record group holds an
    attribute that is not a configuration key), and `read_state` refuses a
    solver of another time scheme.
    """
    from pathlib import Path
    case = Case(family="Quadrature", core="fluid")
    a, _, _ = C.build_new(meshes, case)
    directory = Path(meshes[0].comm.bcast(str(tmp_path), root=0))
    empty, written = str(directory / "mesh.h5"), str(directory / "state.h5")
    with fd.CheckpointFile(empty, "w") as chk:
        chk.save_mesh(meshes[0])
    with fd.CheckpointFile(written, "w") as chk:
        chk.save_mesh(meshes[0])
        write_state(chk, a, idx=0)
        # An attribute of the record group that is not a configuration key:
        # `stored_configuration` leaves it out.
        chk.set_attr("/gadopt_gia_state", "written_by", "a later writer")
    assert set(configuration(a)) == set(CONFIGURATION_KEYS)
    assert not has_state(empty)
    assert stored_configuration(empty) is None
    assert has_state(written) and has_state(Path(written))
    assert not has_state(written, prefix="other")
    assert stored_configuration(written) == configuration(a)
    with fd.CheckpointFile(written, "r") as chk:
        assert has_state(chk)
        assert stored_configuration(chk) == configuration(a)
    # The refusal comes from the stored configuration, before any field is
    # loaded; a read that loads the fields needs a solver on the mesh loaded
    # from the file (the `restart` entry of the `forward` case).
    b, _, _ = C.build_new(meshes, case.but(scheme="bdf2"))
    with fd.CheckpointFile(written, "r") as chk:
        with pytest.raises(ValueError, match="time_scheme"):
            read_state(chk, b, idx=0)


# -- History quadrature degree -----------------------------------------------------

#: A Quadrature configuration without a displacement condition (the family
#: refuses one), for the construction-only tests of `history_quad_degree`.
HISTORY_DEGREE_CASE = Case(family="Quadrature", core="fluid")


@pytest.mark.parametrize("family, kwargs, message", [
    ("DG", dict(history_quad_degree=2), "Quadrature history only"),
    ("Quadrature", dict(history_quad_degree=6), "above quad_degree=4"),
    ("Quadrature", dict(history_quad_degree=0), "integer of at least 2"),
    ("Quadrature", dict(history_quad_degree=1), "one point per cell"),
    ("Quadrature", dict(history_quad_degree=2.0), "integer of at least 2"),
    ("Quadrature", dict(history_quad_degree=2, internal_variable_degree=4),
     "history_quad_degree"),
    ("Quadrature", dict(internal_variable_degree=2), "history_quad_degree"),
], ids=["dg", "above", "zero", "one", "float", "degree-mismatch", "degree-without-option"])
def test_a_history_quad_degree_that_does_not_apply_is_refused(meshes, family, kwargs,
                                                              message):
    """`history_quad_degree` is refused on the DG family, above `quad_degree`, below 2.

    `internal_variable_degree` on the Quadrature family must equal the
    history degree, and the message names `history_quad_degree`. No solve.
    """
    case = HISTORY_DEGREE_CASE.but(family=family)
    with pytest.raises(ValueError, match=message):
        C.build_new(meshes, case, **kwargs)


def test_a_lower_history_degree_splits_the_relaxed_stress(meshes):
    """The history degree sets the history space and the rule of the relaxed stress.

    No solve. By default (and with `history_quad_degree` equal to
    `quad_degree`) the momentum row is one equation at `quad_degree` and
    the history has 6 points per triangle at degree 4. With degree 2 the
    history and the fluidity fields of the local solve have 3 points per
    triangle, the relaxed stress is a second equation at degree 2, and the
    momentum equation keeps degree 4. `internal_variable_degree` equal to
    the history degree is accepted.
    """
    sub = meshes[1]
    cells = sub.num_cells()
    tensors = C.approximation(HISTORY_DEGREE_CASE).n_history_tensors
    for kwargs, degree, n_equations in [({}, 4, 1),
                                        (dict(history_quad_degree=4), 4, 1),
                                        (dict(history_quad_degree=2), 2, 2),
                                        (dict(history_quad_degree=2,
                                              internal_variable_degree=2), 2, 2)]:
        solver, _, _ = C.build_new(meshes, HISTORY_DEGREE_CASE, **kwargs)
        points = {4: 6, 2: 3}[degree]
        assert solver.quad_degree == 4
        assert solver.history_quad_degree == degree == solver.history.degree
        history = solver.internal_variables
        # A symmetric 2x2 tensor stores 3 numbers per point.
        assert history.function_space().dim() == cells * points * tensors * 3, kwargs
        assert solver.history.material.q.function_space().dim() == cells * points
        assert len(solver.equations) == n_equations, kwargs
        degrees = [eq.dx.metadata()["quadrature_degree"] for eq in solver.equations]
        assert degrees == [4, 2][:n_equations], kwargs


@pytest.mark.parametrize("model", ["maxwell2", "backstress"])
def test_the_elastic_and_the_relaxed_stress_sum_to_the_stress(meshes, model):
    """`elastic_stress(u) + relaxed_stress(m)` is `stress(u, m)` to round-off.

    The two parts are what the solver integrates with two rules when the
    history degree is not `quad_degree`; `stress` is what it integrates
    with one rule. Evaluated at the points of a degree-4 rule for a smooth
    displacement and smooth history tensors.
    """
    sub = meshes[1]
    material = C.approximation(Case(model=model))
    X = fd.SpatialCoordinate(sub)
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)).interpolate(
        fd.as_vector([fd.sin(2 * X[0]) * X[1], fd.cos(X[1]) + X[0] ** 2]))
    tensors = [fd.as_matrix([[X[0] * (k + 1), X[1] - k], [X[1] - k, fd.sin(X[0] + k)]])
               for k in range(material.n_history_tensors)]
    space = fd.TensorFunctionSpace(sub, "Quadrature", 4)
    whole = fd.Function(space).interpolate(material.stress(u, internal_variables=tensors))
    parts = fd.Function(space).interpolate(
        material.elastic_stress(u) + material.relaxed_stress(tensors))
    scale = np.max(np.abs(whole.dat.data_ro))
    assert np.max(np.abs(whole.dat.data_ro - parts.dat.data_ro)) <= 1e-14 * scale


def test_a_state_of_another_history_degree_is_refused(meshes, tmp_path):
    """A file written with the degree-2 history records it, and a degree-4 solver refuses it.

    No time step: the state of a solver at construction is written. The
    record holds `history_degree` 2; `read_state` with the default degree
    refuses the file before it loads a field, with a message that names
    `history_quad_degree=2`. The read of a degree-2 file into a degree-2
    solver is the `restart-hq2` entry of the `forward` case.
    """
    from pathlib import Path
    writer, _, _ = C.build_new(meshes, HISTORY_DEGREE_CASE, history_quad_degree=2)
    directory = Path(meshes[0].comm.bcast(str(tmp_path), root=0))
    path = str(directory / "state.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(meshes[0])
        write_state(chk, writer, idx=0)
    assert stored_configuration(path)["history_degree"] == 2
    assert stored_configuration(path) == configuration(writer)
    reader, _, _ = C.build_new(meshes, HISTORY_DEGREE_CASE)
    with fd.CheckpointFile(path, "r") as chk:
        with pytest.raises(ValueError, match="history_quad_degree=2"):
            read_state(chk, reader, idx=0)


# -- Interface ----------------------------------------------------------------------

@pytest.mark.longtest
def test_the_order_of_the_couplings_does_not_matter(meshes):
    """Reversed `couplings`: the same field order and the same residual, bit for bit."""
    parent, sub = meshes
    case = Case(core="fluid", rotation=True, centre=True, load="sea_moving")
    sea = C.SeaFields(sub, ice=True)

    def build(order):
        couplings = [cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.),
                     cp.Rotation(C=C.polar_moment(meshes), Omega_sq=1.566e-3), cp.CentreOfMass(),
                     cp.SeaLevel(**sea.kwargs(case))]
        return NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                         fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                         C.approximation(case), dt=C.dt_function(sub),
                         gravity_bcs=C.gravity_bcs(parent),
                         couplings=couplings[::order],
                         dtn_representation=case.representation,
                         solver_parameters="direct",
                         internal_variable_family="Quadrature")

    a, b = build(1), build(-1)
    assert b.field_names == a.field_names
    # One state for both, defined by the coordinates, so that every row is
    # nonzero.
    x = C.coordinate_mixed_function(a, "order", 1e-3)
    C.fill(a.solution, x)
    C.fill(b.solution, x)
    with fd.assemble(a.F).dat.vec_ro as ra, fd.assemble(b.F).dat.vec_ro as rb:
        same = np.array_equal(ra.array_r, rb.array_r)
    from mpi4py import MPI
    assert a.mesh.comm.allreduce(bool(same), op=MPI.LAND)


def test_the_approximation_is_not_written(meshes):
    case = Case(rotation=True)
    material = C.approximation(case)
    parent, sub = meshes
    NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
              fd.Function(fd.FunctionSpace(parent, "CG", 2)), material,
              dt=C.dt_function(sub), gravity_bcs=C.gravity_bcs(parent),
              bcs={C.CURVE_RC: {"un": 0.0}}, solver_parameters="direct")
    assert not hasattr(material, "mu")


@pytest.mark.parametrize("couplings, bcs, message", [
    (lambda s: [cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2),
                cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2)], {}, "More than one"),
    (lambda s: [cp.CentreOfMass()], {C.CURVE_RC: {"un": 0.0}}, "requires a FluidCore"),
    (lambda s: [cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2),
                cp.SeaLevel(**C.SeaFields(s).kwargs(Case(load="sea_fixed")))],
     {}, "requires CentreOfMass"),
    (lambda s: [cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2),
                cp.SurfaceLoad(boundary=C.CURVE_RE, sigma=C.load_expression(s),
                               g_surface=1.0)],
     {C.CURVE_RE: {"normal_stress": 1.0}}, "normal_stress"),
    (lambda s: [cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2)],
     {C.CURVE_RC: {"un": 0.0}}, "FluidCore and a 'u' or 'un'"),
    (lambda s: [cp.SurfaceLoad(boundary=99, sigma=C.load_expression(s),
                               g_surface=1.0)],
     {C.CURVE_RC: {"un": 0.0}}, "empty measure"),
])
def test_invalid_combinations_are_refused(meshes, couplings, bcs, message):
    parent, sub = meshes
    with pytest.raises(ValueError, match=message):
        NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                  fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                  C.approximation(Case()), dt=C.dt_function(sub),
                  gravity_bcs=C.gravity_bcs(parent), bcs=bcs,
                  couplings=couplings(sub), solver_parameters="direct")


def test_a_sheet_in_the_gravity_conditions_is_refused(meshes):
    parent, sub = meshes
    bcs = C.gravity_bcs(parent)
    bcs[C.CURVE_RE] = {"interior_sigma": C.load_expression(parent)}
    with pytest.raises(ValueError, match="SurfaceLoad"):
        NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                  fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                  C.approximation(Case()), dt=C.dt_function(sub),
                  gravity_bcs=bcs, bcs={C.CURVE_RC: {"un": 0.0}})


def test_the_quadrature_family_refuses_a_weak_displacement_condition(meshes):
    with pytest.raises(NotImplementedError, match="natural displacement"):
        C.build_new(meshes, Case(family="Quadrature"))


def _density(sub, kind):
    """A reference density of the kind the nullspace check is tested on."""
    X = fd.SpatialCoordinate(sub)
    r = fd.sqrt(fd.dot(X, X))
    profile = 1.0 + 0.2 * (2.2037 - r)
    if kind == "DG0":
        return fd.Function(fd.FunctionSpace(sub, "DG", 0)).interpolate(profile)
    if kind == "CG1":
        return fd.Function(fd.FunctionSpace(sub, "CG", 1)).interpolate(profile)
    if kind == "jump":
        # A 10 percent jump at a radius that is not on a cell face.
        return fd.Function(fd.FunctionSpace(sub, "DG", 1)).interpolate(
            fd.conditional(r < 1.65, 1.1, 1.0))
    return 1.0 + float(kind) * X[0]


def _solver_with_density(meshes, kind, K, **kwargs):
    """A fluid-core solver with the density `kind` and bulk modulus `K`.

    Args:
      meshes: the parent annulus and its mantle submesh.
      kind: the density of `_density`.
      K: the bulk modulus.
      kwargs: further constructor arguments, for example `check_nullspace`.
    """
    parent, sub = meshes
    material = gadopt.CompressibleInternalVariableApproximation(
        bulk_modulus=K, density=_density(sub, kind), g=1.0, B_mu=C.B_MU,
        self_gravity_number=C.LAMBDA, shear_modulus=1.0, viscosity=1.0)
    return NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                     fd.Function(fd.FunctionSpace(parent, "CG", 2)), material,
                     dt=C.dt_function(sub), gravity_bcs=C.gravity_bcs(parent),
                     couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.)],
                     nullspace="rigid_rotation", internal_variable_family="Quadrature", solver_parameters="direct", **kwargs)


@pytest.mark.parametrize("K", [1.0, 1000.0])
@pytest.mark.parametrize("kind", ["DG0", "CG1", "jump"])
def test_a_radial_density_passes_the_nullspace_check(meshes, kind, K):
    """A density that is 1-D by intent, in any representation, at any bulk modulus."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver = _solver_with_density(meshes, kind, K)
    lateral, ratio, ratio_t = solver.kernel_residuals[0]
    assert lateral < 1e-3


@pytest.mark.parametrize("K", [1.0, 1000.0])
def test_a_lateral_density_is_refused_or_warned(meshes, K):
    """`1 + a x`: refused at `a = 0.01`, warned at `a = 0.001`, at any bulk modulus."""
    with pytest.raises(ValueError, match="moves density"):
        _solver_with_density(meshes, "0.01", K)
    with pytest.warns(UserWarning, match="moves density"):
        solver = _solver_with_density(meshes, "0.001", K)
    assert abs(solver.kernel_residuals[0][0] - 1.41e-3) < 1e-4


@pytest.mark.parametrize("degree", [3, 4])
def test_a_lateral_density_of_degree_three_and_four_is_refused(meshes, degree):
    """N1: a variation with no degree-1 or degree-2 content is seen too."""
    parent, sub = meshes
    X = fd.SpatialCoordinate(sub)
    theta = fd.atan2(X[1], X[0])
    material = gadopt.CompressibleInternalVariableApproximation(
        bulk_modulus=1000.0, density=1.0 + 0.3 * fd.cos(degree * theta), g=1.0,
        B_mu=C.B_MU, self_gravity_number=C.LAMBDA, shear_modulus=1.0, viscosity=1.0)
    with pytest.raises(ValueError, match="moves density"):
        NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                  fd.Function(fd.FunctionSpace(parent, "CG", 2)), material,
                  dt=C.dt_function(sub), gravity_bcs=C.gravity_bcs(parent),
                  couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.)],
                  nullspace="rigid_rotation", internal_variable_family="Quadrature", solver_parameters="direct")


def test_a_mode_that_violates_a_strong_condition_is_refused(meshes):
    """N2: `u = 0` on the CMB and the rotation given as a basis."""
    parent, sub = meshes
    V = fd.VectorFunctionSpace(sub, "CG", 2)
    X = fd.SpatialCoordinate(sub)
    basis = fd.VectorSpaceBasis([fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))])
    basis.orthonormalize()
    with pytest.raises(ValueError, match="strong displacement condition"):
        NewSolver(fd.Function(V), fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                  C.approximation(Case()), dt=C.dt_function(sub),
                  gravity_bcs=C.gravity_bcs(parent), bcs={C.CURVE_RC: {"u": 0.0}},
                  nullspace=basis, solver_parameters="direct")


@pytest.mark.longtest
def test_a_symmetric_configuration_passes_the_nullspace_check(meshes):
    solver, _, _ = C.build_new(meshes, Case(family="Quadrature", core="fluid", load=None),
                               nullspace="rigid_rotation")
    lateral, ratio, ratio_t = solver.kernel_residuals[0]
    # Round-off of the degree-4 moments: about 1e-12.
    assert lateral < 1e-10 and max(ratio, ratio_t) < 1e-3


def test_a_zero_time_step_at_construction_gives_a_finite_probe(meshes):
    """A driver may assign the first step after construction.

    The DG history equation divides by the time step, so at `dt = 0` the
    history rows of `J e` are not finite, and a NaN ratio would pass every
    bound without testing anything. The probe runs at a unit step instead,
    gives the numbers of a positive step, and leaves the caller's zero.
    """
    case = Case(family="Quadrature", core="fluid", load=None)
    dt = C.dt_function(meshes[1], 0.0)
    solver, _, _ = C.build_new(meshes, case, dt=dt, nullspace="rigid_rotation")
    reference, _, _ = C.build_new(meshes, case, nullspace="rigid_rotation")
    assert float(dt) == 0.0
    assert all(np.isfinite(solver.kernel_residuals[0]))
    np.testing.assert_allclose(solver.kernel_residuals[0],
                               reference.kernel_residuals[0], rtol=1e-8)


def test_the_nullspace_check_can_be_skipped(meshes):
    """`check_nullspace=False` skips the density and Jacobian measures.

    A lateral density that the check refuses is accepted, the measures are
    not taken, and a mode that violates a strong condition is still refused.
    """
    with pytest.raises(ValueError, match="moves density"):
        _solver_with_density(meshes, "0.01", 1.0)
    solver = _solver_with_density(meshes, "0.01", 1.0, check_nullspace=False)
    assert solver.kernel_residuals == []
    parent, sub = meshes
    V = fd.VectorFunctionSpace(sub, "CG", 2)
    X = fd.SpatialCoordinate(sub)
    basis = fd.VectorSpaceBasis([fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))])
    basis.orthonormalize()
    with pytest.raises(ValueError, match="strong displacement condition"):
        NewSolver(fd.Function(V), fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                  C.approximation(Case()), dt=C.dt_function(sub),
                  gravity_bcs=C.gravity_bcs(parent), bcs={C.CURVE_RC: {"u": 0.0}},
                  nullspace=basis, solver_parameters="direct", check_nullspace=False)


def test_the_residual_loses_only_its_part_along_the_declared_rotation(meshes):
    """`complete_residual` removes `<F, e> e` from the residual and nothing else.

    No time step is taken. The residual of the step is assembled once at a
    state with a rigid rotation and a shear in the displacement, through
    the SNES of the step with the removal switched off, so it is the
    residual that Firedrake assembles. A known multiple of the declared
    mode `e` is then added to it, so the part to remove is large and the
    test does not rest on the small `<F, e>` of the assembled residual
    alone. The checks:

    - `residual_kernel` is the transpose nullspace that Firedrake attached
      to the Jacobian of the step (`monolithic_kernel` reads private
      Firedrake attributes to get that object), so the residual and the
      right-hand side of the linear solve lose the same part;
    - the completed residual has no part along `e`, and it differs from the
      input only by `<F, e> e`, so every other row is unchanged;
    - the removed fraction is published by `_report_removed_residual`, it
      is the measured `|<F, e>| / |F|` here, and the warning is given above
      `REMOVED_RESIDUAL_WARN_FRACTION` and not below.
    """
    solver, _, _ = C.build_new(meshes, Case(family="Quadrature", core="fluid"),
                               nullspace="rigid_rotation")
    kernel = solver.residual_kernel
    snes = solver.solver.snes
    jacobian = snes.getJacobian()[0]
    assert kernel.handle == jacobian.getTransposeNullSpace().handle
    (e,) = kernel.getVecs()

    # A state with a known rotation part and a non-rigid part of about the
    # same size in the displacement block; the other blocks stay zero.
    X = fd.SpatialCoordinate(meshes[1])
    solver.solution.subfunctions[0].interpolate(
        0.3 * fd.as_vector([-X[1], X[0]])
        + 0.1 * fd.as_vector([X[0] * X[1], X[0] ** 2]))
    with solver.solution.dat.vec_ro as x:
        state = x.copy()

    # The residual as Firedrake assembles it: `complete_residual` removes
    # nothing while `residual_kernel` is `None` (multiplier path, so no
    # low-rank term is added either). Firedrake's residual callback finds
    # its context through the DM hooks, which `solve` installs; outside a
    # solve the test installs them as `solve` does.
    raw = state.duplicate()
    solver.residual_kernel = None
    try:
        with dmhooks.add_hooks(snes.getDM(), solver.solver,
                               appctx=solver.solver._ctx, save=False):
            snes.computeFunction(state, raw)
    finally:
        solver.residual_kernel = kernel
    assert raw.norm() > 0

    def complete(F):
        """Return a completed copy of `F` and the removed coefficient."""
        completed = F.copy()
        solver.complete_residual(state, completed)
        return completed, F.dot(e)

    # The assembled residual alone: its part along `e` is small, and the
    # published fraction is that part over the residual norm (the residual
    # scale of a fresh solver is the norm of its first residual; `|J x|`
    # enters only above the warning threshold).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        completed, along = complete(raw)
        solver._report_removed_residual()
    assert abs(completed.dot(e)) <= 1e-14 * raw.norm()
    np.testing.assert_allclose(solver.removed_residual_fraction,
                               abs(along) / raw.norm(), rtol=1e-10)
    assert solver.removed_residual_fraction < solver.REMOVED_RESIDUAL_WARN_FRACTION

    # A residual with a large known part along `e`.
    F = raw.copy()
    F.axpy(10.0 * raw.norm(), e)
    # Above the threshold the scale also has the lower bound `|J x|` at this
    # state; it is below `|F|` here, so the fraction stays `|<F, e>| / |F|`.
    solver._removed_largest = 0.0
    with pytest.warns(UserWarning, match="declared nullspace"):
        completed, along = complete(F)
        solver._report_removed_residual()
    assert abs(completed.dot(e)) <= 1e-14 * F.norm()
    difference = F.copy()
    difference.axpy(-1.0, completed)
    difference.axpy(-along, e)
    assert difference.norm() <= 1e-14 * F.norm()
    np.testing.assert_allclose(solver.removed_residual_fraction,
                               abs(along) / F.norm(), rtol=1e-10)


def test_a_solver_restored_to_a_converged_state_does_not_warn(meshes):
    """A new solver restored to a converged state gives no removed-part warning.

    The elastic sea-level Earth of the `repeat` row of the `limits` case,
    multiplier path, declared rigid rotation. A first solver solves the
    step twice (the second solve takes the state from the Newton tolerance
    to round-off, as in the `repeat` row); a second solver is built,
    restored to that converged state and solves the same step again. Its
    residuals are at round-off and mostly along the
    declared rotation, so against the residual norms of this solver alone
    the removed part would be of order one of the scale. The scale of
    `_report_removed_residual` has the lower bound `|J x|`, the size of the
    forcing, so the solve gives no warning (warnings as errors) and the
    published fraction is at round-off (measured 1.1e-9). The first check
    shows that the restored solve is the round-off case: its largest
    residual norm is below 1e-6 of the forcing.
    `test_the_residual_loses_only_its_part_along_the_declared_rotation`
    keeps the warning for a large removed part.
    """
    from gia_models import sea_level as S
    _, sub = meshes
    fields = S.surface_fields(sub, "caps", d_ice=1e-2)

    def build():
        solver, _, _ = S.build(meshes, fields, representation="multiplier",
                               **S.solve_settings("multiplier"),
                               internal_variable_family="Quadrature", quad_degree=C.QUAD,
                               earth=S.ELASTIC_EARTH, nullspace="rigid_rotation")
        return solver

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = build()
        first.solve()
        first.solve()
        forcing = first._residual_scale
        restored = build()
        restored.restore({k: (f.copy(deepcopy=True) if k != "previous_dt" else f)
                          for k, f in first.state().items()})
        restored.solve()
    assert restored._residual_scale < 1e-6 * forcing
    assert restored.removed_residual_fraction < 1e-6


def test_backstress_refuses_a_displacement_condition(meshes):
    """The backstress history equation has no boundary part."""
    with pytest.raises(NotImplementedError, match="natural displacement"):
        C.build_new(meshes, Case(family="Quadrature", model="backstress", core="rigid"))


def test_a_kept_dtn_operator_raises_once_its_solver_is_collected(meshes):
    """The low-rank DtN operator reads `theta_psi` through a weak reference.

    A caller that keeps `dtn_operator` (or an application context that holds
    it) after the solver is gone gets the `ReferenceError` of
    `gadopt.utility.weak_callback`, which names the cause, as from the
    solver's PETSc callbacks. No time step is taken.
    """
    solver, _, _ = C.build_new(
        meshes, Case(family="Quadrature", representation="lowrank", core="fluid"))
    operator = solver.dtn_operator
    assert operator.theta_value == solver.theta_psi_value
    alive = weakref.ref(solver)
    del solver
    gc.collect()
    PETSc.garbage_cleanup(meshes[1].comm)
    gc.collect()
    assert alive() is None
    with pytest.raises(ReferenceError, match="garbage-collected"):
        operator.theta_value


def test_a_traction_without_a_surface_load_warns(meshes):
    """A `normal_stress` loads the mechanics only, with a warning."""
    parent, sub = meshes
    with pytest.warns(UserWarning, match="normal_stress"):
        NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                  fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                  C.approximation(Case()), dt=C.dt_function(sub),
                  gravity_bcs=C.gravity_bcs(parent),
                  bcs={C.CURVE_RC: {"un": 0.0},
                       C.CURVE_RE: {"normal_stress": C.load_expression(sub)}},
                  solver_parameters="direct")


@pytest.mark.longtest
def test_the_coefficients_are_real_functions_on_the_multiplier_path(meshes):
    """No Python float, so a functional keeps its dependency."""
    solver, _, _ = C.build_new(meshes, Case(rotation=True))
    solver.solve()
    values = [v for boundary in solver.coefficients().values() for v in boundary.values()]
    assert values and all(isinstance(v, fd.Function) for v in values)


def test_ksponly_is_refused_on_a_power_law(meshes):
    with pytest.raises(ValueError, match="ksponly"):
        C.build_new(meshes, Case(model="power"), extra={"snes_type": "ksponly"})
    # A Newtonian rheology with a fixed coastline accepts it.
    C.build_new(meshes, Case(), extra={"snes_type": "ksponly"})


def test_the_preset_options_and_the_derivative_cap(meshes):
    parent, sub = meshes
    case = Case(family="Quadrature", representation="lowrank", core="fluid", centre=True)
    solver, _, _ = C.build_new(
        meshes, case, solver_parameters="iterative",
        preset_options={"outer_rtol": 1e-9, "outer_max_it": 30},
        derivative_max_it=500, default_options=False)
    assert solver.solver_parameters["ksp_rtol"] == 1e-9
    assert solver.solver_parameters["ksp_max_it"] == 30
    derivative = solver.adjoint_solver_parameters()
    assert derivative["ksp_max_it"] == 500
    assert derivative["snes_type"] == "ksponly"
    with pytest.raises(ValueError, match="supplies"):
        C.build_new(meshes, case, solver_parameters="iterative",
                    preset_options={"n_real": 3})
    with pytest.raises(ValueError, match="iterative preset"):
        C.build_new(meshes, case, preset_options={"outer_rtol": 1e-9},
                    default_options=False)


@pytest.mark.parametrize("model,load", [("maxwell2", "sea_moving"),
                                        ("power", "sea_fixed")])
def test_the_lag_options_are_accepted_on_the_quadrature_lowrank_path(meshes, model, load):
    """The production lag options pass the rebuild rule of the cached pieces.

    The GIAMIP job scripts pass `gia_rebuild_every` and
    `gia_rebuild_max_outer` and no `gia_rebuild`. The preconditioners
    refuse the two lag options under any rule other than `lagged`
    (`gadopt.preconditioners._rebuild_lag_options`), so a solver that set
    another rule on this path would stop at its first solve. Both
    configurations have a Jacobian that depends on the state: a moving
    coastline, and a power law with a fixed coastline.

    The check needs no solve. With the options of the variational solver
    inserted, `_RebuildRule.from_options` reads the rule from a PC under
    the prefix of the condensed block, as `CondensedBlockPC` does at set-up.
    """
    from gadopt.preconditioners import _RebuildRule
    case = Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                model=model, core="fluid", centre=True, load=load)
    solver, _, _ = C.build_new(
        meshes, case, solver_parameters="iterative", nullspace="rigid_rotation",
        default_options=False,
        extra={"gia_rebuild_every": 20, "gia_rebuild_max_outer": 6.0})
    # The configuration must be the one the lag is for.
    assert solver.jacobian_depends_on_solution
    # The solver sets no rule of its own.
    assert "gia_rebuild" not in solver.solver_parameters
    variational = solver.solver
    with variational.inserted_options():
        pc = PETSc.PC().create(comm=solver.mesh.comm)
        try:
            pc.setOptionsPrefix(variational.options_prefix + "dtn_fieldsplit_0_condensed_")
            rule = _RebuildRule.from_options(pc)
        finally:
            pc.destroy()
    assert (rule.rule, rule._every, rule._max_outer) == ("lagged", 20, 6.0)


@pytest.mark.longtest
def test_an_options_change_rebuilds_the_quadrature_solver_only(meshes):
    """F16: the backend and its private coefficients are built once."""
    case = Case(family="Quadrature", representation="lowrank", core="fluid")
    a, u, _ = C.build_new(meshes, case, nullspace="rigid_rotation")
    backend, inputs = a._step.backend, a._step.backend.private_inputs
    a.add_to_solver_config({"ksp_rtol": 1e-12}, reinit=True)
    assert a._step.backend is backend and backend.private_inputs is inputs
    assert a.solver.parameters["ksp_rtol"] == 1e-12
    b, ub, _ = C.build_new(meshes, case, nullspace="rigid_rotation",
                           extra={"ksp_rtol": 1e-12})
    for solver in (a, b):
        solver.solve()
    assert fd.norm(u - ub) <= 1e-12 * fd.norm(ub)


def test_the_closure_constant_follows_lambda():
    """F14: with `k_s`, `C - A = Q k_s` is UFL in `Lambda`, not a number."""
    mesh = fd.UnitIcosahedralSphereMesh(1)
    Lambda = fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(1.1)

    class Context:
        pass
    ctx = Context()
    ctx.mesh, ctx.Lambda = mesh, Lambda
    rotation = cp.Rotation(C=1.0, k_s=0.9, surface_radius=1.0, Omega_sq=1e-3)
    rotation._ctx = ctx
    rotation.check([rotation])
    constant = rotation.closure_constants[0]
    assert any(c is Lambda for c in extract_coefficients(constant))
    Q = 4 * np.pi * 1e-3 / (3 * 1.1)
    assert abs(cp.scalar_value(fd.assemble(constant * fd.dx(domain=mesh))
                               / fd.assemble(1 * fd.dx(domain=mesh))) - Q * .9) < 1e-12
    with pytest.raises(ValueError, match="disagree"):
        bad = cp.Rotation(C=1.0, C_minus_A=2 * Q * .9, k_s=0.9, surface_radius=1.0,
                          Omega_sq=1e-3)
        bad._ctx = ctx
        bad.check([bad])


@pytest.mark.longtest
@pytest.mark.parametrize("control_name", ["Lambda", "B_mu"])
def test_taylor_with_a_real_control_in_the_row_scalings(meshes, control_name):
    """`Lambda` or `B_mu` as a `Real` control on the parent mesh.

    Both enter the row scalings of every coupling facet term (fluid core,
    sea level, rotation and centre-of-mass sheets) on facets that are
    interior facets of the parent, where they are restricted with the rest
    of the integrand (`gadopt.utility.interior_facet_integrand`). A gradient
    with respect to an unrestricted one would raise.
    """
    parent, sub = meshes
    case = Case(representation="lowrank", model="maxwell2", core="fluid",
                rotation=True, centre=True, load="sea_moving")
    R = fd.FunctionSpace(parent, "R", 0)
    with stop_annotating():
        values = {"Lambda": C.LAMBDA, "B_mu": C.B_MU}
        control = fd.Function(R).assign(values[control_name])
        direction = fd.Function(R).assign(0.1 * values[control_name])

    def run_march(m):
        with stop_annotating():
            sea = C.SeaFields(sub)
            dt = C.dt_function(sub)
            material = gadopt.CompressibleInternalVariableApproximation(
                bulk_modulus=1.0, density=1.0, g=1.0,
                B_mu=m if control_name == "B_mu" else C.B_MU,
                self_gravity_number=m if control_name == "Lambda" else C.LAMBDA,
                shear_modulus=[.4, .6], viscosity=[.12, .6])
            u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2))
            psi = fd.Function(fd.FunctionSpace(parent, "CG", 2))
            sea_level = cp.SeaLevel(**sea.kwargs(case))
            solver = NewSolver(
                u, psi, material, dt=dt, gravity_bcs=C.gravity_bcs(parent),
                couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.),
                           cp.Rotation(C=C.polar_moment(meshes), Omega_sq=1.566e-3),
                           cp.CentreOfMass(), sea_level],
                nullspace="rigid_rotation", internal_variable_family="Quadrature", solver_parameters="direct",
                solver_parameters_extra=dict(snes_rtol=1e-11, snes_atol=1e-13,
                                             ksp_rtol=1e-11))
        for k, h in enumerate(STEPS[:2]):
            dt.assign(h)
            surface_ice(sea, k)
            solver.solve()
        return (fd.assemble(fd.inner(u, u) * fd.dx(domain=sub))
                + fd.assemble(psi * psi * fd.dx(domain=parent))
                + fd.assemble(sea_level.sea_level() ** 2 * sea_level.measure()))

    continue_annotation()
    J = run_march(control)
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    assert abs(float(rf(control)) - float(J)) <= 1e-10 * abs(float(J))
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)


#: The coupling parameters that can be given as a `Real` Function, with the
#: load of the case that carries them and their value in the test geometry.
REAL_PARAMETERS = {
    "SeaLevel.g_surface": ("sea_fixed", 1.0),
    "SurfaceLoad.g_surface": ("surface", 1.0),
    "Rotation.Omega_sq": ("surface", 1.566e-3),
    "FluidCore.rho_core": ("surface", 1.2),
    "FluidCore.g": ("surface", 1.0),
    # Not a coupling parameter, but every coupling facet term carries it, and
    # the traction of a `SurfaceLoad` is where a `Real` `B_mu` of the parent
    # needs the coupling facet measure.
    "approximation.B_mu": ("surface", C.B_MU),
}


#: The parameters of `REAL_PARAMETERS` whose case is a long test. The
#: sea-level case builds the sea-level coupling as well, which costs several
#: times the surface-load case, so PR CI runs the other parameters only.
LONG_REAL_PARAMETERS = {"SeaLevel.g_surface"}


@pytest.mark.parametrize("on, parameter", [
    pytest.param(on, parameter, marks=(pytest.mark.longtest,)
                 if parameter in LONG_REAL_PARAMETERS else ())
    for parameter in REAL_PARAMETERS
    for on in ("mechanics", "parent")
    # `B_mu` is a row scaling of the parent rows, so it lives on the parent.
    if not (parameter == "approximation.B_mu" and on == "mechanics")])
def test_a_coupling_parameter_can_be_a_real_function_on_either_mesh(
        meshes, parameter, on):
    """A coupling parameter as a `Real` Function of either mesh: residual and derivatives.

    The coupling facets are exterior facets of the mechanics mesh and
    interior facets of the parent. UFL requires every quantity of the parent
    to carry a restriction there and refuses one on a quantity of the
    mechanics mesh, so each parameter must be restricted by the mesh it
    lives on. Checked: the residual assembles and equals the residual with
    the plain number; the tangent action `dF/dm . dm` and the adjoint action
    `(dF/dm)^T w`, which pyadjoint assembles for a control, assemble too, and
    the two agree (`w . (dF/dm dm) = ((dF/dm)^T w) dm`).
    """
    parent, sub = meshes
    load, number = REAL_PARAMETERS[parameter]
    case = Case(family="Quadrature", core="fluid", rotation=True, centre=True, load=load)
    mesh = sub if on == "mechanics" else parent
    R = fd.FunctionSpace(mesh, "R", 0)
    real = fd.Function(R).assign(number)

    def residual_at_fixed_state(value):
        # One fixed coordinate-defined state, so that every term is nonzero.
        solver, _, _ = C.build_new(meshes, case, sea=C.SeaFields(sub, ice=True),
                                   parameters={parameter: value})
        C.write_state(solver, C.coordinate_state(solver), previous_dt=None)
        return solver

    reference = residual_at_fixed_state(number)
    solver = residual_at_fixed_state(real)
    F = solver.F
    with fd.assemble(F).dat.vec_ro as a, fd.assemble(reference.F).dat.vec_ro as b:
        assert a.norm() > 0
        difference = a.copy()
        difference.axpy(-1.0, b)
        assert difference.norm() <= 1e-14 * b.norm()
    dF = fd.derivative(F, real)
    dm = fd.Function(R).assign(1.0)
    w = C.coordinate_mixed_function(solver, "adjoint", 1.0)
    tangent = fd.assemble(fd.action(dF, dm))
    adjoint = fd.assemble(fd.action(fd.adjoint(dF), w))
    with tangent.dat.vec_ro as t, w.dat.vec_ro as wv:
        forward_pairing = t.dot(wv)
    with adjoint.dat.vec_ro as a:
        adjoint_pairing = a.sum()
    # The parameter enters the residual: the derivative is not zero.
    assert abs(forward_pairing) > 0
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)


def _adjoint_and_tangent(solver, parameter):
    """The tangent and adjoint pairings of `dF/dm` for the Function `m`.

    At a fixed coordinate-defined state, `w . (dF/dm dm)` and
    `((dF/dm)^T w) . dm` with `dm = 1` in every degree of freedom of `m`'s
    space (a `Real` or a field) and a coordinate-defined `w`. The two agree
    when the derivative is assembled right.
    """
    C.write_state(solver, C.coordinate_state(solver), previous_dt=None)
    dF = fd.derivative(solver.F, parameter)
    dm = fd.Function(parameter.function_space()).assign(1.0)
    w = C.coordinate_mixed_function(solver, "adjoint", 1.0)
    tangent = fd.assemble(fd.action(dF, dm))
    adjoint = fd.assemble(fd.action(fd.adjoint(dF), w))
    with tangent.dat.vec_ro as t, w.dat.vec_ro as wv:
        forward_pairing = t.dot(wv)
    with adjoint.dat.vec_ro as a, dm.dat.vec_ro as d:
        adjoint_pairing = a.dot(d)
    return forward_pairing, adjoint_pairing


def _normal_stress_solver(meshes, factor):
    """A fluid-core solver whose surface load is `factor` times a fixed shape.

    The load stands in a `normal_stress` of `bcs`, a boundary term of the
    momentum equation on the surface, which is an interior facet of the
    parent annulus.
    """
    parent, sub = meshes
    X = fd.SpatialCoordinate(sub)
    shape = 1e-3 * (1 + fd.cos(fd.atan2(X[1], X[0])))
    with pytest.warns(UserWarning, match="normal_stress"):
        solver = NewSolver(
            fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
            fd.Function(fd.FunctionSpace(parent, "CG", 2)),
            C.approximation(Case(core="fluid")), dt=C.dt_function(sub),
            gravity_bcs=C.gravity_bcs(parent),
            bcs={C.CURVE_RE: {"normal_stress": factor * shape}},
            couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.)],
            dtn_representation="multiplier", internal_variable_family="Quadrature", solver_parameters="direct")
    return solver


def test_a_parent_real_in_a_momentum_boundary_term_is_refused(meshes):
    """A `Real` Function of the parent in a `normal_stress` in `bcs` is refused.

    The momentum equation pairs its boundary terms with the parent's cells,
    where Firedrake assembles the adjoint of a parent `Real` twice (an
    adjoint pairing of 2.000 times the tangent pairing). The solver refuses
    such a coefficient at construction and names it. The same factor as a
    `Real` Function of the mechanics mesh is accepted, and its tangent and
    adjoint pairings agree.
    """
    parent, sub = meshes
    on_parent = fd.Function(fd.FunctionSpace(parent, "R", 0), name="load_factor")
    on_parent.assign(1.0)
    with pytest.raises(ValueError, match="'load_factor'.*mechanics mesh"):
        _normal_stress_solver(meshes, on_parent)
    on_mechanics = fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(1.0)
    forward_pairing, adjoint_pairing = _adjoint_and_tangent(
        _normal_stress_solver(meshes, on_mechanics), on_mechanics)
    assert abs(forward_pairing) > 0
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)


@pytest.mark.parametrize("family,degree", [("DG", 0), ("CG", 1)])
def test_a_parent_field_in_a_momentum_boundary_term_is_refused(meshes, family, degree):
    """A polynomial field of the parent in a `normal_stress` in `bcs` is refused.

    On a boundary that is an interior facet of the parent, Firedrake's
    adjoint of a parent DG0 field is assembled on both parent cells of the
    facet (a pairing near 2 times the tangent's), and that of a CG1 field
    changes from one assembly to the next. The solver refuses both and
    names the field. A DG0 field of the mechanics mesh is accepted, and
    its tangent and adjoint pairings agree.
    """
    parent, sub = meshes
    on_parent = fd.Function(fd.FunctionSpace(parent, family, degree), name="load_field")
    on_parent.assign(1.0)
    with pytest.raises(ValueError, match="Function 'load_field'.*Function of the mechanics mesh"):
        _normal_stress_solver(meshes, on_parent)
    if family != "DG":
        return
    on_mechanics = fd.Function(fd.FunctionSpace(sub, family, degree)).assign(1.0)
    forward_pairing, adjoint_pairing = _adjoint_and_tangent(
        _normal_stress_solver(meshes, on_mechanics), on_mechanics)
    assert abs(forward_pairing) > 0
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)


@pytest.mark.parametrize("quantity", ["SpatialCoordinate", "FacetNormal",
                                      "CellVolume", "JacobianDeterminant"])
def test_a_parent_geometric_quantity_in_a_momentum_boundary_term_is_refused(
        meshes, quantity):
    """A geometric quantity of the parent in a `normal_stress` is refused.

    The momentum equation pairs its boundary terms with the parent's cells,
    where Firedrake does not evaluate a geometric quantity of the parent at
    the points of the integrated facet: at the core-mantle boundary of this
    annulus, `CellVolume(parent)` and `JacobianDeterminant(parent)`
    integrate to 0.79 of the same quantity of the mechanics mesh, so the
    residual itself is wrong. Which quantity is wrong on which facet
    depends on the mesh, so the solver refuses every such quantity on every
    boundary at construction, and names it and the replacement. The
    `normal_stress` of `_normal_stress_solver` stands on the surface.
    """
    parent, sub = meshes
    factors = {
        "SpatialCoordinate": lambda: fd.SpatialCoordinate(parent)[0],
        "FacetNormal": lambda: fd.FacetNormal(parent)[0],
        "CellVolume": lambda: fd.CellVolume(parent),
        "JacobianDeterminant": lambda: abs(fd.JacobianDeterminant(parent)),
    }
    with pytest.raises(ValueError, match=f"{quantity}\\(potential mesh\\).*{quantity}\\(u_mesh\\)"):
        _normal_stress_solver(meshes, factors[quantity]())


@pytest.mark.longtest
def test_the_B_mu_floor_is_decided_once_at_construction(meshes):
    """The row scale of `B_mu` follows the decision taken at construction.

    A solver built with `B_mu = 0` scales its rows with the constant
    `NULL_COUPLING_ROW_SCALE`; the low-rank prefactor keeps that constant
    when `B_mu` later becomes nonzero, as the forms do. A solver built with
    a nonzero `B_mu` refuses a solve once `B_mu` is zero, because its forms
    would then delete the potential rows.
    """
    from gadopt.gia_presets import NULL_COUPLING_ROW_SCALE
    parent = meshes[0]
    case = Case(family="Quadrature", representation="lowrank", core="fluid", centre=True)

    B_mu = fd.Function(fd.FunctionSpace(parent, "R", 0)).assign(0.0)
    solver, _, _ = C.build_new(meshes, case, parameters={"approximation.B_mu": B_mu},
                               nullspace="rigid_rotation")
    assert solver.B_mu_floored
    B_mu.assign(2.0)
    expected = cp.scalar_value(solver.scaling_factor) * NULL_COUPLING_ROW_SCALE / C.LAMBDA
    assert solver.theta_psi_value == pytest.approx(expected, rel=1e-15)

    B_mu = fd.Function(fd.FunctionSpace(parent, "R", 0)).assign(C.B_MU)
    solver, _, _ = C.build_new(meshes, case, parameters={"approximation.B_mu": B_mu},
                               nullspace="rigid_rotation")
    assert not solver.B_mu_floored
    B_mu.assign(0.0)
    with pytest.raises(ValueError, match="B_mu is zero, but it was nonzero"):
        solver.solve()


def test_a_real_scaling_factor_lives_on_the_parent_without_weak_conditions(meshes):
    """A `Real` `scaling_factor`: on the parent only, and not with a weak condition.

    `scaling_factor` scales the potential rows over every cell of the parent
    as well as the momentum rows, so a Function of the mechanics mesh is
    refused with a message that names it. On the parent it is accepted with
    a fluid core, whose rows use the facet-to-facet measure, and its tangent
    and adjoint pairings agree. With a rigid core it stands in the weak `un`
    condition of the momentum equation, where its adjoint would be assembled
    twice, and it is refused. A parent `Real` time step with the same rigid
    core does not enter a boundary term of the momentum equation and keeps
    working.
    """
    parent, sub = meshes
    with pytest.raises(ValueError, match="scaling_factor lives on a mesh other"):
        C.build_new(meshes, Case(core="fluid", load="surface"),
                    scaling_factor=fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(1.0))

    c = fd.Function(fd.FunctionSpace(parent, "R", 0), name="c").assign(1.0)
    solver, _, _ = C.build_new(meshes, Case(core="fluid", load="surface"), scaling_factor=c)
    forward_pairing, adjoint_pairing = _adjoint_and_tangent(solver, c)
    assert abs(forward_pairing) > 0
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)

    with pytest.raises(ValueError, match="'c'.*a Real scaling_factor cannot be used"):
        C.build_new(meshes, Case(core="rigid", load="surface"), scaling_factor=c)

    dt = fd.Function(fd.FunctionSpace(parent, "R", 0)).assign(C.DT0)
    solver, _, _ = C.build_new(meshes, Case(core="rigid", load="surface"), dt=dt)
    forward_pairing, adjoint_pairing = _adjoint_and_tangent(solver, dt)
    assert abs(forward_pairing) > 0
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)


def test_a_discontinuous_parent_field_is_refused_as_a_coupling_parameter(meshes):
    """A DG field of the parent as a coupling parameter is refused.

    On a coupling facet, an interior facet of the parent, `avg` of a
    discontinuous field is the mean of the mantle side and the outer side,
    not the mantle's value. A continuous (CG) field of the parent is
    accepted, and so is the DG field on the mechanics mesh.
    """
    parent, sub = meshes
    case = Case(family="Quadrature", core="fluid", load="surface")
    dg = fd.Function(fd.FunctionSpace(parent, "DG", 0)).assign(1.2)
    with pytest.raises(ValueError, match="discontinuous element"):
        C.build_new(meshes, case, parameters={"FluidCore.rho_core": dg})
    cg = fd.Function(fd.FunctionSpace(parent, "CG", 1)).assign(1.2)
    C.build_new(meshes, case, parameters={"FluidCore.rho_core": cg})
    dg_mechanics = fd.Function(fd.FunctionSpace(sub, "DG", 0)).assign(1.2)
    C.build_new(meshes, case, parameters={"FluidCore.rho_core": dg_mechanics})


#: The configurations of `test_no_parent_unknown_in_a_momentum_boundary_term`:
#: `(core, load, normal_stress boundary or None)`. Each has at least one
#: boundary term of the momentum equation: the weak `un` condition of a rigid
#: core, a `normal_stress` in `bcs`, or both. A `normal_stress` stands on a
#: boundary without a load, because `SurfaceLoad` and `SeaLevel` refuse one
#: on their own boundary. `SeaLevel` needs `CentreOfMass`, which needs a fluid
#: core, so the sea-level case has no weak `un`, and its `normal_stress` is at
#: the CMB, next to the fluid core.
_MOMENTUM_BOUNDARY_CASES = {
    "rigid-un": ("rigid", "surface", None),
    "rigid-un-normal-stress": ("rigid", None, C.CURVE_RE),
    "fluid-core-normal-stress": ("fluid", None, C.CURVE_RE),
    "fluid-core-sea-level-normal-stress": ("fluid", "sea_moving", C.CURVE_RC),
}


@pytest.mark.parametrize("representation", ["multiplier", "lowrank"])
@pytest.mark.parametrize("configuration", list(_MOMENTUM_BOUNDARY_CASES))
def test_no_parent_unknown_in_a_momentum_boundary_term(meshes, representation,
                                                       configuration):
    """No boundary term of the momentum equation depends on a parent unknown.

    The refusal of parent coefficients in momentum boundary terms reads the
    coefficients of those terms. A parent unknown (the potential, a DtN
    multiplier, a rotation or centre-of-mass unknown) is a component of the
    mixed solution, whose mesh is not the parent, so the refusal cannot see
    it. Its adjoint on a boundary that is an interior facet of the parent
    would meet the same Firedrake defect. This test takes the exterior-facet
    integrals of the mechanics residual, differentiates them with respect to
    the solution, and asserts that no column of the result belongs to a
    field of the parent.

    Every configuration has rotation, which adds `Real` unknowns (and the
    DtN multipliers on that path). The boundary terms are the weak `un`
    condition of a rigid core and a `normal_stress` in `bcs`, the only two
    kinds of momentum boundary term. A fluid core adds the centre-of-mass
    rows, which a rigid core excludes. A `SeaLevel` load writes a sheet that
    reads the potential into the coupling rows, next to the boundary terms.

    The weak `un` term reads the displacement, so on a rigid-core case the
    Jacobian must have columns; none means the derivative was lost. A
    prescribed `normal_stress` is a source and has no column. On a
    fluid-core case the Jacobian then holds only the boundary terms that the
    volume terms of the momentum equation bring with them, and the test
    asserts that none of those reads a parent unknown.
    """
    from firedrake.formmanipulation import split_form
    core, load, stress_boundary = _MOMENTUM_BOUNDARY_CASES[configuration]
    case = Case(core=core, rotation=True, centre=core == "fluid", load=load,
                representation=representation)
    parent, sub = meshes
    extra_bcs = None
    if stress_boundary is not None:
        extra_bcs = {stress_boundary: {"normal_stress": C.load_expression(sub)}}
    # A `normal_stress` in `bcs` warns that it loads the mechanics only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*normal_stress",
                                category=UserWarning)
        solver, _, _ = C.build_new(meshes, case, extra_bcs=extra_bcs)
    facet = fd.Form(solver._mechanics_form.integrals_by_type("exterior_facet"))
    assert facet.integrals(), "the case has no momentum boundary term"
    jacobian = expand_derivatives(fd.derivative(facet, solver.solution))
    # `split_form` returns every block, the empty ones too.
    columns = {block.indices[1] for block in split_form(jacobian)
               if not expand_derivatives(block.form).empty()}
    parent_fields = {solver.potential_index, *solver.real_indices}
    assert solver.real_indices, "the case has no Real unknown"
    if core == "rigid":
        assert columns, "the weak un term depends on no unknown"
    assert not columns & parent_fields, (
        f"a momentum boundary term depends on the parent fields "
        f"{sorted(columns & parent_fields)}")


def test_a_discontinuous_displacement_is_refused(meshes):
    """The solver refuses a displacement outside H1 and names the element.

    A discontinuous displacement adds interior-facet terms to the momentum
    equation, which this solver does not handle.
    """
    parent, sub = meshes
    with pytest.raises(ValueError, match="u must be in a continuous space"):
        NewSolver(
            fd.Function(fd.VectorFunctionSpace(sub, "DG", 2)),
            fd.Function(fd.FunctionSpace(parent, "CG", 2)),
            C.approximation(Case()), dt=C.dt_function(sub),
            gravity_bcs=C.gravity_bcs(parent), bcs={C.CURVE_RC: {"un": 0.0}},
            internal_variable_family="Quadrature")


#: One case of each step kind: Firedrake's solve block with the multiplier
#: DtN and with the low-rank DtN (DG family, in-place solve), and the
#: Quadrature backend, which solves on a copy of its own.
FAILED_STEP_CASES = [Case(), Case(representation="lowrank"),
                     Case(family="Quadrature", core="fluid")]


@pytest.mark.parametrize("case", FAILED_STEP_CASES, ids=[c.label for c in FAILED_STEP_CASES])
def test_a_failed_step_leaves_its_start_in_the_mixed_solution(meshes, case):
    """A step solve that raises leaves the start of the step, so a retry starts there too.

    The Newton solve is stopped after one iteration with zero tolerances,
    so it raises `ConvergenceError` after it changed the iterate. The start
    is a nonzero `u` and `psi`, copied in by the step, and the `Real` fields
    as they are. After the failure every field of the mixed solution holds
    that start bit for bit, and `u` and `psi` are not written. On the DG
    family Firedrake solves for the mixed solution in place, so without the
    restore of the step the mixed solution would hold the failed iterate.
    """
    stop_after_one = {"snes_max_it": 1, "snes_rtol": 0.0, "snes_atol": 0.0,
                      "snes_stol": 0.0}
    solver, u, psi = C.build_new(meshes, case, extra=stop_after_one)
    # A smooth nonzero start in both fields, so the copy in runs and the
    # start differs from the iterate after one Newton step.
    X = fd.SpatialCoordinate(u.function_space().mesh())
    u.interpolate(fd.as_vector([1e-3 * X[0] * X[1], 2e-3 * X[0] ** 2]))
    Y = fd.SpatialCoordinate(psi.function_space().mesh())
    psi.interpolate(1e-3 * Y[0] * Y[1])
    expected = solver.solution.copy(deepcopy=True)
    expected.subfunctions[0].assign(u)
    expected.subfunctions[solver.potential_index].assign(psi)
    u_before, psi_before = u.copy(deepcopy=True), psi.copy(deepcopy=True)
    with pytest.raises(fd.ConvergenceError):
        solver.solve()
    # The solve did take its Newton step before it raised.
    assert solver.solver.snes.getIterationNumber() == 1
    for got, want in zip(solver.solution.subfunctions, expected.subfunctions):
        assert np.array_equal(got.dat.data_ro, want.dat.data_ro), got.name()
    assert np.array_equal(u.dat.data_ro, u_before.dat.data_ro)
    assert np.array_equal(psi.dat.data_ro, psi_before.dat.data_ro)
