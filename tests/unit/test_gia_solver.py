"""The new self-gravitating GIA solver: steps, adjoint, state and interface.

Gate 1 (`test_gia_solver_gate.py`) compares the equations with the old
solver. These tests run the new solver itself:

- three forward steps agree to round-off with the stored states of the
  earlier solver (`gadopt.gia_gravity`), on the three step variants (stock,
  low-rank, quadrature);
- Taylor tests (rate 2) and replay checks of a taped march, on the low-rank
  DG path with sea level, on the quadrature path with BDF2, a fluid core and
  the centre of mass, and on a power law;
- `state`, `CheckpointFile`, `restore`: a run interrupted and continued is
  bit for bit the uninterrupted run in serial, and equal to round-off on
  more ranks; checkpoint files the old solver wrote convert into a state of
  the new one;
- the interface rules: the order of the couplings, the refusals, the
  nullspace check, the presets, `approximation.mu`, the closure constant.
"""

import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from pyadjoint.tape import annotate_tape
from ufl.algorithms import extract_coefficients

import gadopt
import gia_solver_cases as C
from gadopt import gia_couplings as cp
from gadopt.gia_solver import SelfGravitatingGIASolver as NewSolver
from gadopt.gia_state import read_state, state_from_checkpoint, write_state
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


# -- Forward steps against the stored values of the earlier solver ---------------

FORWARD = [
    Case(rotation=True),
    Case(scheme="bdf2", representation="lowrank", model="maxwell2",
         core="fluid", rotation=True, centre=True, load="sea_moving"),
    Case(family="Quadrature", scheme="bdf2", representation="lowrank",
         model="maxwell2", core="fluid", centre=True),
]

#: Fingerprints of the earlier solver's (`gadopt.gia_gravity`) state after
#: each of the three steps, by case label: per step the displacement, the
#: potential and the accepted history as `[norm, probe, probe norm]`
#: (`gia_solver_cases.probe_pairing`), and the value of every `Real` field.
FORWARD_REFERENCE = C.reference_values("forward")

#: Bound on the relative difference of a solved field from the stored one.
#: The earlier solver and this one agreed to about 1e-15 when they were run
#: side by side on one machine. On another machine, another BLAS, another
#: MUMPS ordering or another rank count moves a solved state by round-off
#: times the condition number of the step, which can exceed 1e-12; a defect
#: in a term moves it by far more than 1e-10.
SOLVED_RTOL = 1e-10


@pytest.mark.parametrize("case", FORWARD, ids=[c.label for c in FORWARD])
def test_three_steps_agree_with_the_reference(meshes, case):
    """Direct presets: after each step the state agrees with the stored one.

    The displacement, the potential and the accepted history are compared
    through their coordinate-defined fingerprints, and every `Real` field by
    value, at `SOLVED_RTOL`.
    """
    sub = meshes[1]
    dt = C.dt_function(sub)
    sea = C.SeaFields(sub)
    new, u, psi = C.build_new(meshes, case, dt=dt, sea=sea,
                              nullspace="rigid_rotation")
    for k, (h, reference) in enumerate(zip(STEPS, FORWARD_REFERENCE[case.label])):
        dt.assign(h)
        if case.load:
            surface_ice(sea, k)
        new.solve()
        fields = {"displacement": C.probe_pairing(u, "displacement"),
                  "potential": C.probe_pairing(psi, "potential"),
                  "history": C.probe_pairing(new.internal_variables, "history")}
        C.assert_fingerprints_agree(
            fields, {name: reference[name] for name in fields}, rtol=SOLVED_RTOL)
        # A `Real` field whose exact value is zero (the core pressure under a
        # load with no degree-0 part) is round-off, so every `Real` field is
        # compared against the largest `Real` value of the step. In a case
        # where every `Real` value is round-off (the Quadrature case: no
        # rotation, a load with no degree-0 part), the check is that they
        # stay round-off against the size of the displacement vector.
        reals = reference["reals"]
        assert set(reals) == set(new.field_index) - {"displacement", "history", "potential"}
        displacement_size = reference["displacement"][0]
        scale = max(abs(value) for value in reals.values())
        bound = (SOLVED_RTOL * scale if scale > 1e-12 * displacement_size
                 else 1e-12 * displacement_size)
        for name, value in reals.items():
            difference = abs(float(new.real_field(name)) - value)
            assert difference <= bound, (k, name, difference)


# -- Taylor and replay -----------------------------------------------------------

TAPED = [
    pytest.param(Case(representation="lowrank", model="maxwell2", core="fluid",
                      rotation=True, centre=True, load="sea_moving"),
                 id="lowrank-DG-sea_level"),
    pytest.param(Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                      model="maxwell2", core="fluid", centre=True),
                 id="quadrature-bdf2-fluid_core-centre_of_mass"),
    pytest.param(Case(scheme="bdf2", model="power", rotation=True),
                 id="power_law"),
    # The stock step (multiplier DtN, DG) with a declared nullspace, so that
    # the recorded projection follows a stock solve block.
    pytest.param(Case(scheme="bdf2", model="maxwell2", core="fluid", centre=True),
                 id="stock_step-nullspace"),
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
    dx_m = fd.Measure("dx", domain=sub, metadata={"quadrature_degree": 4})
    history = solver.internal_variables
    J = fd.assemble((fd.inner(u, u) + .1 * fd.inner(history, history)) * dx_m)
    J += fd.assemble(psi * psi * fd.dx(domain=parent))
    sea_level = [c for c in solver.couplings if isinstance(c, cp.SeaLevel)]
    if sea_level:
        # The objective of the inversions: a sea-level expression of the
        # coupling, which keeps its dependency on the tape.
        J += fd.assemble(sea_level[0].sea_level() ** 2 * sea_level[0].measure())
    return J


@pytest.mark.parametrize("case", TAPED)
def test_taylor_and_replay(meshes, case):
    sub = meshes[1]
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, "CG", 1)).assign(.1)
        direction = fd.Function(control.function_space()).interpolate(
            .1 + .03 * fd.SpatialCoordinate(sub)[0])
    continue_annotation()
    J = march(meshes, case, control)
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    # Replay at the recorded control returns the recorded value.
    assert abs(float(rf(control)) - float(J)) <= 1e-10 * abs(float(J))
    # Replay at another control returns a new forward run there.
    with stop_annotating():
        other = control.copy(deepcopy=True).assign(control + .05 * direction)
        fresh = float(march(meshes, case, other))
    assert abs(float(rf(other)) - fresh) <= 1e-8 * abs(fresh)
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)


def taylor_case(meshes, case, **march_kwargs):
    """A Taylor test with replay at the recorded control of one taped march."""
    sub = meshes[1]
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, "CG", 1)).assign(.1)
        direction = fd.Function(control.function_space()).interpolate(
            .1 + .03 * fd.SpatialCoordinate(sub)[0])
    continue_annotation()
    J = march(meshes, case, control, **march_kwargs)
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    assert abs(float(rf(control)) - float(J)) <= 1e-10 * abs(float(J))
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)
    return rf, control


@pytest.mark.parametrize("case", [
    pytest.param(Case(representation="lowrank", model="maxwell2", core="fluid",
                      rotation=True, centre=True, load="sea_moving"),
                 id="lowrank-DG-sea_level"),
    pytest.param(Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                      model="maxwell2", core="fluid", centre=True),
                 id="quadrature-bdf2")])
def test_taylor_with_disk_checkpointing(meshes, case, tmp_path):
    """Three steps under `SingleDiskStorageSchedule`: replay and rate 2."""
    from pathlib import Path
    directory = Path(meshes[1].comm.bcast(str(tmp_path / "checkpoints"), root=0))
    taylor_case(meshes, case, steps=STEPS, checkpoint_dir=directory)


def test_taylor_through_a_restore_on_the_tape(meshes):
    """A state copied and restored into a second solver mid-march, on the tape."""
    case = Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                model="maxwell2", core="fluid", centre=True)
    taylor_case(meshes, case, steps=STEPS, restore_after=2)


@pytest.mark.parametrize("case", [
    Case(representation="lowrank", core="fluid", centre=True),
    Case(representation="multiplier", core="fluid", centre=True),
    Case(family="Quadrature", representation="lowrank", core="fluid", centre=True)],
    ids=["lowrank-DG", "multiplier-DG", "quadrature"])
def test_a_capped_derivative_solve_raises(meshes, case):
    """`derivative_max_it` caps the adjoint solve: 1 raises, 400 is enough."""
    sub = meshes[1]
    for cap, raises in ((1, True), (400, False)):
        get_working_tape().clear_tape()
        with stop_annotating():
            control = fd.Function(fd.FunctionSpace(sub, "CG", 1)).assign(.1)
        continue_annotation()
        J = march(meshes, case, control, steps=STEPS[:1],
                  solver_parameters="iterative", derivative_max_it=cap,
                  default_options=False)
        rf = ReducedFunctional(J, Control(control))
        pause_annotation()
        if raises:
            with pytest.raises(fd.ConvergenceError):
                rf.derivative()
        else:
            rf.derivative()


# -- State ------------------------------------------------------------------------

STATE_CASES = [
    # Backward Euler, DG, multiplier DtN, rigid core: no previous level.
    Case(rotation=True),
    Case(scheme="bdf2", representation="lowrank", model="maxwell2", core="fluid",
         rotation=True, centre=True, load="sea_moving"),
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


@pytest.mark.parametrize("case", STATE_CASES, ids=[c.label for c in STATE_CASES])
def test_state_round_trip_through_a_checkpoint_file(meshes, case, tmp_path):
    """Interrupted, checkpointed, restored on a loaded mesh, continued.

    The restored state equals the saved one bit for bit on any number of
    ranks. The continued run equals the uninterrupted run bit for bit in
    serial; on more than one rank it agrees to `PARALLEL_RTOL`, because two
    parallel MUMPS factorisations of one matrix do not give the same bits
    (`SelfGravitatingGIASolver.restore`).
    """
    from pathlib import Path
    directory = Path(meshes[0].comm.bcast(str(tmp_path), root=0))
    name = meshes[0].name

    def load(path):
        return load_meshes(path, name)

    def solver_on(m):
        sea = C.SeaFields(m[1]) if case.load in ("sea_fixed", "sea_moving") else None
        solver, _, _ = C.build_new(m, case, dt=C.dt_function(m[1]), sea=sea,
                                   nullspace="rigid_rotation")
        return solver, sea

    # A mesh written and loaded once, so that every run below uses one
    # numbering of the degrees of freedom.
    with fd.CheckpointFile(str(directory / "mesh.h5"), "w") as chk:
        chk.save_mesh(meshes[0])
    first = load(directory / "mesh.h5")
    interrupted, sea = solver_on(first)
    run(interrupted, sea, FOUR[:2])
    with fd.CheckpointFile(str(directory / "state.h5"), "w") as chk:
        chk.save_mesh(first[0])
        write_state(chk, interrupted, idx=0)
    saved = values(interrupted)

    second = load(directory / "state.h5")
    restarted, sea2 = solver_on(second)
    with fd.CheckpointFile(str(directory / "state.h5"), "r") as chk:
        restarted.restore(read_state(chk, restarted, idx=0))
    assert not states_differ(values(restarted), saved, restarted, rtol=0)
    run(restarted, sea2, FOUR[2:], first=2)
    continued = values(restarted)

    reference, sea3 = solver_on(first)
    run(reference, sea3, FOUR)
    rtol = 0 if reference.mesh.comm.size == 1 else PARALLEL_RTOL
    assert not states_differ(values(reference), continued, restarted, rtol=rtol)


def test_state_round_trip_in_memory_and_reset(meshes):
    """`restore` of copied Functions, then `reset` back to rest.

    The continued runs agree bit for bit in serial and to `PARALLEL_RTOL` on
    more than one rank (`SelfGravitatingGIASolver.restore`).
    """
    case = STATE_CASES[2]
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
#: 2f4b1397) wrote after two steps on the mesh of `gia_solver_cases.MESH_FILE`,
#: in the layouts of the drivers that ran it. Written once, in serial, by a
#: script that ran the earlier solver and the drivers' field lists
#: (`time_history_fields`, `time_history_metadata`), then copied with gzip
#: compression in the HDF5 1.10 file format, which HDF5 reads transparently.
#: The mesh in every file is the curved parent, under its default name.
OLD_CHECKPOINTS = C.REFERENCE_DIR
OLD_MESH_NAME = "firedrake_default"

#: The cases of the stored files: the file name, the configuration and
#: whether the file has the Exp01 layout (own names, no broken encoding,
#: `real_{i}` rows) instead of the `selfgrav_common` layout (every
#: sub-function of the mixed solution as `solution_{i}` in the broken
#: encoding, the BDF2 levels as `time_*`, the quadrature history also as
#: `internal_variables_0`).
OLD_CHECKPOINT_CASES = {
    "Quadrature-bdf2-selfgrav": (
        "gia_old_checkpoint_quadrature_bdf2.h5",
        Case(family="Quadrature", scheme="bdf2", representation="lowrank",
             model="maxwell2", core="fluid", centre=True), False),
    "DG-be-multiplier-selfgrav": (
        "gia_old_checkpoint_dg_be.h5", Case(rotation=True), False),
    "DG-bdf2-lowrank-selfgrav": (
        "gia_old_checkpoint_dg_bdf2.h5",
        Case(scheme="bdf2", representation="lowrank", core="fluid",
             rotation=True, centre=True), False),
    "DG-bdf2-lowrank-exp01": (
        "gia_old_checkpoint_exp01.h5",
        Case(scheme="bdf2", representation="lowrank", core="fluid",
             rotation=True, centre=True), True),
}


def stored_names(solver, exp01):
    """The names of the fields a driver of the earlier solver stored for `solver`.

    The earlier solver's mixed space had the same sub-spaces in the same
    order as this solver's, so the `selfgrav_common` layout has one
    `solution_{i}` per sub-function of this solver's mixed solution.
    """
    levels = (["time_current_0", "time_previous_0"]
              if solver.time_scheme == "bdf2" else [])
    if exp01:
        return (["checkpoint_displacement", "checkpoint_internal_variables"]
                + levels + ["checkpoint_potential"])
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
    filename, case, exp01 = OLD_CHECKPOINT_CASES[label]
    path = str(OLD_CHECKPOINTS / filename)
    loaded_meshes = load_meshes(path, OLD_MESH_NAME)
    parent, sub = loaded_meshes
    nullspace = "rigid_rotation" if case.core == "fluid" else None
    new, u, psi = C.build_new(loaded_meshes, case, dt=C.dt_function(sub),
                              nullspace=nullspace)
    names = stored_names(new, exp01)
    with fd.CheckpointFile(path, "r") as chk:
        loaded = {n: chk.load_function(parent, n, idx=0) for n in names}
        history = chk.get_timestepping_history(parent, names[0])
        writer_ranks = written_rank_count(chk, OLD_MESH_NAME)
    metadata = {k: float(v[-1]) for k, v in history.items() if k != "index"}
    aliases = ({"displacement": "checkpoint_displacement",
                "potential": "checkpoint_potential"} if exp01 else None)
    state = state_from_checkpoint(new, loaded, metadata, aliases=aliases,
                                  writer_ranks=writer_ranks)
    if exp01 and "time_current_0" not in loaded:
        state["history"] = loaded["checkpoint_internal_variables"]
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


def test_a_checkpoint_of_the_old_solver_converts():
    """A file of the earlier solver in the `selfgrav_common` layout, Quadrature, BDF2.

    The file is the earlier solver's own output (`OLD_CHECKPOINTS`), so this
    checks that a file the earlier code wrote restarts this solver
    (`_old_checkpoint_converts`). A file with the wrong time order is
    refused, and so is a conversion without the number of ranks of the
    writer.

    The file was written in serial and holds no quadrature point
    coordinates, so its history is stored in the node order of the serial
    partition. On more than one rank the conversion is refused, because the
    values would come back permuted inside each cell.
    """
    if fd.COMM_WORLD.size > 1:
        with pytest.raises(ValueError, match=r"written on 1 rank\(s\)"):
            _old_checkpoint_converts("Quadrature-bdf2-selfgrav")
        return
    new, loaded, metadata = _old_checkpoint_converts("Quadrature-bdf2-selfgrav")
    with pytest.raises(ValueError, match="time order"):
        state_from_checkpoint(new, loaded, {**metadata, "time_scheme_order": 1.0},
                              writer_ranks=1)
    with pytest.raises(ValueError, match="writer_ranks"):
        state_from_checkpoint(new, loaded, metadata)


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
    be = Case(rotation=True)
    c, _, _ = C.build_new(meshes, be)
    d, _, _ = C.build_new(meshes, be)
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


@pytest.mark.parametrize("label", ["DG-be-multiplier-selfgrav",
                                   "DG-bdf2-lowrank-selfgrav"])
def test_a_dg_checkpoint_of_the_old_solver_converts(label):
    """A file of the earlier solver in the DG `selfgrav_common` layout.

    The history is `solution_1`. Backward Euler stores no levels and no
    metadata; BDF2 stores the levels and the time metadata.
    """
    _old_checkpoint_converts(label)


def test_an_exp01_checkpoint_of_the_old_solver_converts():
    """A file of the earlier solver in the Exp01 layout (own names, `real_{i}`)."""
    _old_checkpoint_converts("DG-bdf2-lowrank-exp01")


# -- Interface ----------------------------------------------------------------------

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
                         solver_parameters="direct")

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
                     nullspace="rigid_rotation", solver_parameters="direct", **kwargs)


@pytest.mark.parametrize("K", [1.0, 1000.0])
@pytest.mark.parametrize("kind", ["DG0", "CG1", "jump"])
def test_a_radial_density_passes_the_nullspace_check(meshes, kind, K):
    """A density that is 1-D by intent, in any representation, at any bulk modulus."""
    import warnings
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
                  nullspace="rigid_rotation", solver_parameters="direct")


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


def test_a_symmetric_configuration_passes_the_nullspace_check(meshes):
    solver, _, _ = C.build_new(meshes, Case(core="fluid", load=None),
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
    case = Case(core="fluid", load=None)
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


def test_backstress_refuses_a_displacement_condition(meshes):
    """The backstress history equation has no boundary part."""
    with pytest.raises(NotImplementedError, match="natural displacement"):
        C.build_new(meshes, Case(model="backstress", core="rigid"))


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
    case = Case(representation="lowrank", core="fluid", centre=True)
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
                nullspace="rigid_rotation", solver_parameters="direct",
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


@pytest.mark.parametrize("on, parameter", [
    (on, parameter) for parameter in REAL_PARAMETERS
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
    case = Case(core="fluid", rotation=True, centre=True, load=load)
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


def _adjoint_and_tangent(solver, real):
    """The tangent and adjoint pairings of `dF/dm` for the `Real` Function `m`.

    At a fixed coordinate-defined state, `w . (dF/dm dm)` and
    `((dF/dm)^T w) dm` with `dm = 1` and a coordinate-defined `w`. The two
    agree when the derivative is assembled right.
    """
    C.write_state(solver, C.coordinate_state(solver), previous_dt=None)
    dF = fd.derivative(solver.F, real)
    dm = fd.Function(real.function_space()).assign(1.0)
    w = C.coordinate_mixed_function(solver, "adjoint", 1.0)
    tangent = fd.assemble(fd.action(dF, dm))
    adjoint = fd.assemble(fd.action(fd.adjoint(dF), w))
    with tangent.dat.vec_ro as t, w.dat.vec_ro as wv:
        forward_pairing = t.dot(wv)
    with adjoint.dat.vec_ro as a:
        adjoint_pairing = a.sum()
    return forward_pairing, adjoint_pairing


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
    X = fd.SpatialCoordinate(sub)
    shape = 1e-3 * (1 + fd.cos(fd.atan2(X[1], X[0])))

    def build(factor):
        with pytest.warns(UserWarning, match="normal_stress"):
            solver = NewSolver(
                fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                fd.Function(fd.FunctionSpace(parent, "CG", 2)),
                C.approximation(Case(core="fluid")), dt=C.dt_function(sub),
                gravity_bcs=C.gravity_bcs(parent),
                bcs={C.CURVE_RE: {"normal_stress": factor * shape}},
                couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.)],
                dtn_representation="multiplier", solver_parameters="direct")
        return solver

    on_parent = fd.Function(fd.FunctionSpace(parent, "R", 0), name="load_factor")
    on_parent.assign(1.0)
    with pytest.raises(ValueError, match="'load_factor'.*mechanics mesh"):
        build(on_parent)
    on_mechanics = fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(1.0)
    forward_pairing, adjoint_pairing = _adjoint_and_tangent(build(on_mechanics), on_mechanics)
    assert abs(forward_pairing) > 0
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)


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
    assert abs(forward_pairing - adjoint_pairing) <= 1e-12 * abs(forward_pairing)


def test_a_discontinuous_parent_field_is_refused_as_a_coupling_parameter(meshes):
    """A DG field of the parent as a coupling parameter is refused.

    On a coupling facet, an interior facet of the parent, `avg` of a
    discontinuous field is the mean of the mantle side and the outer side,
    not the mantle's value. A continuous (CG) field of the parent is
    accepted, and so is the DG field on the mechanics mesh.
    """
    parent, sub = meshes
    case = Case(core="fluid", load="surface")
    dg = fd.Function(fd.FunctionSpace(parent, "DG", 0)).assign(1.2)
    with pytest.raises(ValueError, match="discontinuous element"):
        C.build_new(meshes, case, parameters={"FluidCore.rho_core": dg})
    cg = fd.Function(fd.FunctionSpace(parent, "CG", 1)).assign(1.2)
    C.build_new(meshes, case, parameters={"FluidCore.rho_core": cg})
    dg_mechanics = fd.Function(fd.FunctionSpace(sub, "DG", 0)).assign(1.2)
    C.build_new(meshes, case, parameters={"FluidCore.rho_core": dg_mechanics})
