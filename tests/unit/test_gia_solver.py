"""The new self-gravitating GIA solver: steps, adjoint, state and interface.

Gate 1 (`test_gia_solver_gate.py`) compares the equations with the old
solver. These tests run the new solver itself:

- three forward steps agree with the old solver to round-off, on the three
  step variants (stock, low-rank, quadrature);
- Taylor tests (rate 2) and replay checks of a taped march, on the low-rank
  DG path with sea level, on the quadrature path with BDF2, a fluid core and
  the centre of mass, and on a power law;
- `state`, `CheckpointFile`, `restore`: a run interrupted and continued is
  bit for bit the uninterrupted run; a checkpoint of the old solver converts
  into a state of the new one;
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


# -- Forward steps against the old solver --------------------------------------

FORWARD = [
    Case(rotation=True),
    Case(scheme="bdf2", representation="lowrank", model="maxwell2",
         core="fluid", rotation=True, centre=True, load="sea_moving"),
    Case(family="Quadrature", scheme="bdf2", representation="lowrank",
         model="maxwell2", core="fluid", centre=True),
]


@pytest.mark.parametrize("case", FORWARD, ids=[c.label for c in FORWARD])
def test_three_steps_agree_with_the_old_solver(meshes, case):
    """Direct presets: the two solvers agree to round-off after each step."""
    sub = meshes[1]
    dt_new, dt_old = C.dt_function(sub), C.dt_function(sub)
    sea = C.SeaFields(sub)
    new, u, psi = C.build_new(meshes, case, dt=dt_new, sea=sea,
                              nullspace="rigid_rotation")
    old, _, layout = C.build_old(meshes, case, dt=dt_old, sea=sea,
                                 declare_nullspace=True)
    names = C.old_field_names(layout)
    reals = [n for n in names if n not in ("displacement", "history", "potential")]
    for k, h in enumerate(STEPS):
        dt_new.assign(h)
        dt_old.assign(h)
        if case.load:
            surface_ice(sea, k)
        new.solve()
        old.solve()
        z = old.solution.subfunctions
        assert fd.norm(u - z[0]) <= 1e-12 * fd.norm(u)
        assert fd.norm(psi - z[layout.potential]) <= 1e-12 * fd.norm(psi)
        old_history = (old.time_integrator.current[0] if old.time_integrator
                       else z[1])
        assert (fd.norm(new.internal_variables - old_history)
                <= 1e-12 * fd.norm(new.internal_variables))
        # A `Real` field whose exact value is zero (the core pressure under a
        # load with no degree-0 part) is round-off on both sides, so the
        # bound has an absolute floor far below every nonzero field here.
        scale = max(abs(float(z[names.index(n)])) for n in reals)
        for n in reals:
            difference = abs(float(new.real_field(n)) - float(z[names.index(n)]))
            assert difference <= 1e-12 * scale + 1e-16, n


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
    J = fd.assemble((fd.inner(u, u) + .1 * fd.inner(solver.internal_variables,
                                                     solver.internal_variables)) * dx_m)
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
    """Copies of every coefficient array of the state, for a bitwise compare."""
    return {name: np.array(f.dat.data_ro, copy=True)
            for name, f in solver.state().items()}


def load_meshes(path, name):
    """The parent from a checkpoint file and its curved mantle submesh."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos" / "gravity"))
    from validate_selfgrav_annulus import curve_mesh
    with fd.CheckpointFile(str(path), "r") as chk:
        parent = chk.load_mesh(name)
    parent.cartesian = False
    sub = curve_mesh(fd.Submesh(parent, 2, C.CELL_MANTLE))
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
    """Interrupted, checkpointed, restored on a loaded mesh, continued: bit for bit."""
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
    for key, array in values(restarted).items():
        assert np.array_equal(array, saved[key]), key
    run(restarted, sea2, FOUR[2:], first=2)

    reference, sea3 = solver_on(first)
    run(reference, sea3, FOUR)
    for key, array in values(reference).items():
        assert np.array_equal(array, values(restarted)[key]), key


def test_state_round_trip_in_memory_and_reset(meshes):
    """`restore` of copied Functions, then `reset` back to rest."""
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
    assert np.array_equal(ub.dat.data_ro, u.dat.data_ro)
    assert np.array_equal(psib.dat.data_ro, psi.dat.data_ro)
    run(a, sea, FOUR[2:])
    run(b, sea, FOUR[2:])
    for key, array in values(a).items():
        assert np.array_equal(array, values(b)[key]), key
    b.reset()
    assert b.time_integrator.previous_dt is None
    for key, f in b.state().items():
        if key != "previous_dt":
            assert not np.any(f.dat.data_ro), key
    assert not np.any(ub.dat.data_ro)
    with pytest.raises(ValueError, match="Unknown state entries"):
        b.restore({**copies, "something": copies["displacement"]})


def test_a_checkpoint_of_the_old_solver_converts(meshes):
    """The `selfgrav_common` layout of the old solver restarts the new one.

    The old solver marches two steps; its fields are encoded on the parent
    as the drivers store them (`solution_{i}`, `time_*`, broken elements) with
    its metadata. The new solver restores the converted state and both take
    a third step; they agree to the old/new round-off.
    """
    from gadopt.viscoelastic_checkpointing import CheckpointField
    from gadopt.viscoelastic_timestepping import (time_history_fields,
                                                  time_history_metadata)
    case = Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                model="maxwell2", core="fluid", centre=True)
    parent, sub = meshes
    old, z, layout = C.build_old(meshes, case, dt=C.dt_function(sub),
                                 declare_nullspace=True)
    for h in FOUR[:2]:
        old.dt.assign(h)
        old.solve()
    fields = {f"solution_{i}": f for i, f in enumerate(z.subfunctions)}
    fields.update({f"internal_variables_{j}": f for j, f in enumerate([old.internal_variables])})
    fields.update(dict(time_history_fields(old)))
    stored = {name: CheckpointField(name, f, parent).encode().copy(deepcopy=True)
              for name, f in fields.items()}
    metadata = time_history_metadata(old)
    new, u, psi = C.build_new(meshes, case, dt=C.dt_function(sub),
                              nullspace="rigid_rotation")
    new.restore(state_from_checkpoint(new, stored, metadata))
    assert new.time_integrator.previous_dt == FOUR[1]
    for h in FOUR[2:3]:
        old.dt.assign(h)
        new.dt.assign(h)
        old.solve()
        new.solve()
    zs = z.subfunctions
    assert fd.norm(u - zs[0]) <= 1e-12 * fd.norm(u)
    assert fd.norm(psi - zs[layout.potential]) <= 1e-12 * fd.norm(psi)
    assert (fd.norm(new.internal_variables - old.time_integrator.current[0])
            <= 1e-12 * fd.norm(new.internal_variables))
    with pytest.raises(ValueError, match="time order"):
        state_from_checkpoint(new, stored, {**metadata, "time_scheme_order": 1.0})


def test_restore_needs_previous_dt_on_bdf2_and_the_file_its_configuration(
        meshes, tmp_path):
    """R4 and R5: a BDF2 state without `previous_dt`, a file of another scheme."""
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


def _old_checkpoint_round_trip(meshes, case, tmp_path, *, exp01):
    """Write a checkpoint of the old solver in a driver's format, convert, continue.

    `exp01=False`: the `selfgrav_common` layout, every sub-function of the
    old mixed solution as `solution_{i}` in the broken encoding, the BDF2
    levels as `time_*`, the metadata of `time_history_metadata` as the
    timestepping values. `exp01=True`: the Exp01 driver's layout, the
    displacement, the history and the potential under its own names copied
    without the broken encoding, and the `Real` rows as `real_{i}` in the
    timestepping values.
    """
    from pathlib import Path
    from gadopt.viscoelastic_checkpointing import CheckpointField
    from gadopt.viscoelastic_timestepping import (time_history_fields,
                                                  time_history_metadata)
    directory = Path(meshes[0].comm.bcast(str(tmp_path), root=0))
    # `load_function` takes only a mesh loaded from the file, so the old run
    # is made on a loaded mesh and the new one on a second load of the file
    # it writes; both loads number the degrees of freedom alike.
    with fd.CheckpointFile(str(directory / "mesh.h5"), "w") as chk:
        chk.save_mesh(meshes[0])
    name = meshes[0].name
    first = load_meshes(directory / "mesh.h5", name)
    parent, sub = first
    old, z, layout = C.build_old(first, case, dt=C.dt_function(sub),
                                 declare_nullspace=case.core == "fluid")
    for h in FOUR[:2]:
        old.dt.assign(h)
        old.solve()
    zs = z.subfunctions
    info = dict(time_history_metadata(old)) if old.time_integrator else {}
    if exp01:
        names = {"displacement": "checkpoint_displacement",
                 "history": "checkpoint_internal_variables"}
        fields = [(names["displacement"], zs[layout.displacement]),
                  (names["history"], zs[layout.internal_variable_field])]
        fields += time_history_fields(old)
        encoded = [CheckpointField(n, f, parent, broken=False) for n, f in fields]
        potential = fd.Function(fd.FunctionSpace(parent, zs[layout.potential].ufl_element()),
                                name="checkpoint_potential").assign(zs[layout.potential])
        for i, f in enumerate(zs):
            if f.ufl_element().family() == "Real":
                info[f"real_{i}"] = float(f.dat.data_ro_with_halos[0])
    else:
        fields = [(f"solution_{i}", f) for i, f in enumerate(zs)]
        fields += time_history_fields(old)
        encoded = [CheckpointField(n, f, parent) for n, f in fields]
    info = {k: float(v) for k, v in info.items()} | {"step": 2.0}
    path = str(directory / "old.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(parent)
        for field in encoded:
            chk.save_function(field.encode(), idx=0, timestepping_info=info)
        if exp01:
            chk.save_function(potential, idx=0, timestepping_info=info)
    # Read back on the same parent: the file was written from it in this
    # process, so `load_function` accepts it.
    stored_names = [f.stored.name() for f in encoded] + (["checkpoint_potential"] if exp01 else [])
    second = load_meshes(path, name)
    with fd.CheckpointFile(path, "r") as chk:
        loaded = {n: chk.load_function(second[0], n, idx=0) for n in stored_names}
        history = chk.get_timestepping_history(second[0], stored_names[0])
    metadata = {k: float(v[-1]) for k, v in history.items() if k != "index"}
    new, u, psi = C.build_new(second, case, dt=C.dt_function(second[1]),
                              nullspace="rigid_rotation" if case.core == "fluid" else None)
    aliases = ({"displacement": "checkpoint_displacement",
                "potential": "checkpoint_potential"} if exp01 else None)
    state = state_from_checkpoint(new, loaded, metadata, aliases=aliases)
    if exp01 and "time_current_0" not in loaded:
        state["history"] = loaded["checkpoint_internal_variables"]
    new.restore(state)
    for h in FOUR[2:3]:
        old.dt.assign(h)
        new.dt.assign(h)
        old.solve()
        new.solve()
    assert relative(u, zs[0]) <= 1e-12
    assert relative(psi, zs[layout.potential]) <= 1e-12
    old_history = old.time_integrator.current[0] if old.time_integrator else zs[1]
    assert relative(new.internal_variables, old_history) <= 1e-12


@pytest.mark.parametrize("case", [Case(rotation=True),
                                  Case(scheme="bdf2", representation="lowrank",
                                       core="fluid", rotation=True, centre=True)],
                         ids=["DG-be-multiplier", "DG-bdf2-lowrank"])
def test_a_dg_checkpoint_of_the_old_solver_converts(meshes, case, tmp_path):
    """The old DG layout (history as `solution_1`), through a CheckpointFile."""
    _old_checkpoint_round_trip(meshes, case, tmp_path, exp01=False)


def test_an_exp01_checkpoint_of_the_old_solver_converts(meshes, tmp_path):
    """The Exp01 layout (own names, `real_{i}`), through a CheckpointFile."""
    case = Case(scheme="bdf2", representation="lowrank", core="fluid",
                rotation=True, centre=True)
    _old_checkpoint_round_trip(meshes, case, tmp_path, exp01=True)


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
    x = C.random_function(a.solution_space, np.random.default_rng(3), 1e-3)
    C.fill(a.solution, x)
    C.fill(b.solution, x)
    with fd.assemble(a.F).dat.vec_ro as ra, fd.assemble(b.F).dat.vec_ro as rb:
        assert np.array_equal(ra.array_r, rb.array_r)


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


def _solver_with_density(meshes, kind, K):
    parent, sub = meshes
    material = gadopt.CompressibleInternalVariableApproximation(
        bulk_modulus=K, density=_density(sub, kind), g=1.0, B_mu=C.B_MU,
        self_gravity_number=C.LAMBDA, shear_modulus=1.0, viscosity=1.0)
    return NewSolver(fd.Function(fd.VectorFunctionSpace(sub, "CG", 2)),
                     fd.Function(fd.FunctionSpace(parent, "CG", 2)), material,
                     dt=C.dt_function(sub), gravity_bcs=C.gravity_bcs(parent),
                     couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.)],
                     nullspace="rigid_rotation", solver_parameters="direct")


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


def test_backstress_refuses_a_displacement_condition(meshes):
    """R2: the backstress history equation has no boundary part."""
    with pytest.raises(NotImplementedError, match="natural displacement"):
        C.build_new(meshes, Case(model="backstress", core="rigid"))


def test_a_traction_without_a_surface_load_warns(meshes):
    """R10: a `normal_stress` loads the mechanics only."""
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
    """R9: no Python float, so a functional keeps its dependency."""
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
