"""Direct elimination in the complete two-mesh GIA/sea-level mixed system.

The Quadrature family of `SelfGravitatingGIASolver` eliminates the history
at the quadrature points by a local Newton solve and condenses the Jacobian.
These tests compare it with an unreduced reference that keeps the history
as an unknown on the same quadrature points and solves the full mixed system
by Newton with a Jacobian assembled block by block
(`tests/gia/gia_reference_mixed.py`).
"""
import gadopt  # noqa: F401  (gadopt is imported before firedrake)
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               pause_annotation, get_working_tape, stop_annotating)
from pyadjoint.tape import annotate_tape
from test_gravity_adjoint import assert_taylor_with_guards
import gia_cases
# The unreduced reference and the Quadrature-family case it is compared with.
from gia_reference_mixed import build

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest


@pytest.fixture(scope='module')
def meshes():
    """The small triangle annulus, read from `tests/unit/data` (no gmsh needed)."""
    return gia_cases.annulus_meshes()


STEPS = (.08, .13, .07, .12)


@pytest.fixture(autouse=True)
def clean_tape():
    get_working_tape().clear_tape()
    yield
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


@pytest.mark.parametrize('exponent', [1, 3, 'spatial'])
@pytest.mark.parametrize('model', ['maxwell', 'backstress'])
def test_direct_matches_coupled_quadrature(meshes, exponent, model):
    if model == 'backstress' and exponent == 'spatial':
        pytest.skip('Backstress exponent is a fixed scalar in this model')
    _, sub = meshes
    c = fd.Function(fd.FunctionSpace(sub, 'CG', 1))
    X = fd.SpatialCoordinate(sub)
    c.interpolate(.1*X[0])
    direct, dt, ice = build(meshes, c, direct=True, exponent=exponent, model=model)
    coupled, cdt, cice = build(meshes, c, direct=False, exponent=exponent, model=model)
    for k, h in enumerate(STEPS):
        dt.assign(h)
        cdt.assign(h)
        X = fd.SpatialCoordinate(ice.function_space().mesh())
        ice.interpolate(.001*(1+.2*k)*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))
        cice.assign(ice)
        direct.solve()
        # The reference Newton is not a taped solve.
        with stop_annotating():
            coupled.solve()
        for index in ('displacement', 'potential'):
            a = direct.solution.subfunctions[direct.field_index[index]]
            b = coupled.solution.subfunctions[coupled.field_index[index]]
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=5e-7, atol=2e-9)
        np.testing.assert_allclose(
            direct.internal_variables.dat.data_ro,
            coupled.solution.subfunctions[coupled.field_index['history']].dat.data_ro,
            rtol=5e-7, atol=2e-9)
        # The `Real` fields, by name; both solvers hold the same ones.
        names = [direct.field_names[i] for i in direct.real_indices]
        assert names == [coupled.field_names[j] for j in coupled.real_indices]
        for name in names:
            np.testing.assert_allclose(
                direct.real_field(name).dat.data_ro,
                coupled.real_field(name).dat.data_ro, rtol=5e-7, atol=2e-9)


def march(meshes, control, *, direct, model='maxwell', checkpoint_dir=None):
    with stop_annotating():
        solver, dt, ice = build(meshes, control, direct=direct, model=model)
    if checkpoint_dir is not None:
        from firedrake.adjoint import enable_disk_checkpointing
        from checkpoint_schedules import SingleDiskStorageSchedule
        if meshes[1].comm.rank == 0:
            checkpoint_dir.mkdir()
        meshes[1].comm.barrier()
        enable_disk_checkpointing(checkpoint_comm=meshes[1].comm, checkpoint_dir=str(checkpoint_dir))
        get_working_tape().enable_checkpointing(SingleDiskStorageSchedule(), gc_timestep_frequency=1)
    iterator = enumerate(STEPS)
    if annotate_tape():
        iterator = get_working_tape().timestepper(iter(iterator))
    for k, h in iterator:
        dt.assign(h)
        X = fd.SpatialCoordinate(ice.function_space().mesh())
        ice.interpolate(.001*(1+.2*k)*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))
        solver.solve()
    sub = meshes[1]
    u = solver.field('displacement')
    if direct:
        histories = solver.internal_variables
    else:
        histories = solver.field('history')
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree': 4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    return fd.assemble((fd.inner(u, u)+.1*fd.inner(histories, histories))*measure)


@pytest.mark.parametrize('model', ['maxwell', 'backstress'])
@pytest.mark.parametrize('checkpoint', ['memory', 'disk'])
def test_full_mixed_direct_adjoint(meshes, model, checkpoint, tmp_path):
    sub = meshes[1]
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
        direction = fd.Function(control.function_space()).interpolate(.1+.03*fd.SpatialCoordinate(sub)[0])
    continue_annotation()
    c = Control(control)
    directory = None
    if checkpoint == 'disk':
        from pathlib import Path
        directory = Path(sub.comm.bcast(str(tmp_path/'checkpoints'), root=0))
    J = march(meshes, control, direct=True, model=model, checkpoint_dir=directory)
    rf = ReducedFunctional(J, c)
    pause_annotation()
    replay = float(rf(control))
    assert abs(replay-float(J)) < 1e-8*abs(float(J))
    gradient = rf.derivative()
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)
    directional = direction._ad_dot(gradient)
    if sub.comm.size > 1:
        return  # The independently assembled monolithic reference is serial.
    eps = 1e-4
    with stop_annotating():
        plus = control.copy(deepcopy=True).assign(control + eps*direction)
        minus = control.copy(deepcopy=True).assign(control - eps*direction)
        difference = (float(march(meshes, plus, direct=False, model=model))
                      - float(march(meshes, minus, direct=False, model=model)))/(2*eps)
    assert abs(difference-directional) < 2e-5*abs(directional)


def test_direct_iterative_matches_lu_block_reference(meshes):
    _, sub = meshes
    c = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
    reference, dt, ice = build(meshes, c, direct=True)
    iterative, idt, iice = build(meshes, c, direct=True, iterative=True)
    X = fd.SpatialCoordinate(sub)
    for h in STEPS[:2]:
        dt.assign(h)
        idt.assign(h)
        ice.interpolate(.001*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))
        iice.assign(ice)
        reference.solve()
        iterative.solve()
        for a, b in zip(reference.solution.subfunctions, iterative.solution.subfunctions):
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=1e-6, atol=1e-9)
