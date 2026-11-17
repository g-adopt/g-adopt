"""Material and exterior-potential elimination in the same mixed GIA solve."""
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                              pause_annotation, get_working_tape, stop_annotating)
import test_gia_direct_bdf2 as direct
from test_gravity_adjoint import assert_taylor_with_guards

meshes = direct.meshes
clean_tape = direct.clean_tape


def load(ice, k):
    X = fd.SpatialCoordinate(ice.function_space().mesh())
    ice.interpolate(.001*(1+.2*k)*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))


@pytest.mark.parametrize('model,exponent,method', [
    ('maxwell', 1, 'pointwise'), ('maxwell', 1, 'direct'),
    ('maxwell', 3, 'direct'), ('backstress', 3, 'direct')])
def test_lowrank_matches_multipliers(meshes, model, exponent, method, moving=False):
    _, sub = meshes
    c = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
    low, dt, ice = direct.build(meshes, c, direct=True, exponent=exponent, model=model,
                               representation='lowrank', iterative=True, constitutive_method=method,
                               moving_coastline=moving)
    ref, rdt, rice = direct.build(meshes, c, direct=True, exponent=exponent, model=model,
                                 moving_coastline=moving)
    for k, h in enumerate(direct.STEPS):
        dt.assign(h)
        rdt.assign(h)
        load(ice, k)
        rice.assign(ice)
        low.solve()
        ref.solve()
        for name in ('displacement', 'potential', 'core_pressure', 'sea_level'):
            a = low.solution.subfunctions[getattr(low.layout, name)]
            b = ref.solution.subfunctions[getattr(ref.layout, name)]
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=2e-6, atol=2e-9)
        for i, j in zip(low.layout.real_fields, ref.layout.real_fields[len(ref.layout.multipliers):]):
            np.testing.assert_allclose(low.solution.subfunctions[i].dat.data_ro,
                                       ref.solution.subfunctions[j].dat.data_ro, rtol=2e-6, atol=2e-9)
        np.testing.assert_allclose(low.internal_variables.dat.data_ro,
                                   ref.internal_variables.dat.data_ro, rtol=2e-6, atol=2e-9)
    assert low.dtn_operator.jacobian_applications > 0
    block = low.solver.snes.getKSP().getPC().getPythonContext()
    reduced = block.pc.getFieldSplitSchurGetSubKSP()[0].getPC().getPythonContext()
    assert reduced.projected_operator is not None
    # Check the actual retained residual, independently of the KSP recurrence.
    # A declared curved-mesh kernel must be projected on both RHS and action.
    operator, _ = reduced.condensed_ksp.getOperators()
    rhs = reduced._rhs.copy()
    left = operator.getTransposeNullSpace()
    if left and left.handle:
        left.remove(rhs)
    residual = rhs.duplicate()
    operator.mult(reduced._solution_vec, residual)
    residual.axpy(-1., rhs)
    assert residual.norm() <= 2e-4*rhs.norm()+1e-12
    rhs.destroy()
    residual.destroy()


@pytest.mark.parametrize('model', ['maxwell', 'backstress'])
@pytest.mark.parametrize('checkpoint', ['memory', 'disk'])
def test_lowrank_adjoint(meshes, model, checkpoint, tmp_path):
    check_lowrank_adjoint(meshes, model, checkpoint, tmp_path)


def check_lowrank_adjoint(meshes, model, checkpoint, tmp_path, moving=False):
    from pathlib import Path
    _, sub = meshes
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
        direction = control.copy(deepcopy=True).interpolate(.1+.03*fd.SpatialCoordinate(sub)[0])
        solver, dt, ice = direct.build(meshes, control, direct=True, model=model,
                                      representation='lowrank', iterative=True, moving_coastline=moving)
    if moving:
        assert solver.solver_parameters['gia_rebuild'] == 'always'
    continue_annotation()
    c = Control(control)
    if checkpoint == 'disk':
        from firedrake.adjoint import enable_disk_checkpointing
        from checkpoint_schedules import SingleDiskStorageSchedule
        directory = Path(sub.comm.bcast(str(tmp_path/'checkpoints'), root=0))
        if sub.comm.rank == 0:
            directory.mkdir()
        sub.comm.barrier()
        enable_disk_checkpointing(checkpoint_comm=sub.comm, checkpoint_dir=str(directory))
        get_working_tape().enable_checkpointing(SingleDiskStorageSchedule(), gc_timestep_frequency=1)
    iterator = get_working_tape().timestepper(iter(enumerate(direct.STEPS)))
    for k, h in iterator:
        dt.assign(h)
        load(ice, k)
        solver.solve()
    u = fd.split(solver.solution)[solver.layout.displacement]
    psi = fd.split(solver.solution)[solver.layout.potential]
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree':4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    J = fd.assemble((fd.inner(u,u)+.1*fd.inner(solver.internal_variables,solver.internal_variables))*measure)
    J += fd.assemble(.01*psi**2*fd.dx(domain=meshes[0]))
    rf = ReducedFunctional(J, c)
    pause_annotation()
    assert float(rf(control)) == pytest.approx(float(J), rel=1e-8)
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)
    assert 'gia_time_alpha' not in solver.appctx


def test_lowrank_gravity_control_adjoint(meshes):
    _, sub = meshes
    with stop_annotating():
        material_control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
        gravity = fd.Function(fd.FunctionSpace(meshes[0], 'R', 0)).assign(.03)
        direction = gravity.copy(deepcopy=True).assign(.005)
        solver, dt, ice = direct.build(meshes, material_control, direct=True,
            representation='lowrank', iterative=True, gravity_control=gravity)
    continue_annotation()
    c = Control(gravity)
    for k, h in enumerate(direct.STEPS):
        dt.assign(h)
        load(ice, k)
        solver.solve()
    u = fd.split(solver.solution)[solver.layout.displacement]
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree':4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    J = fd.assemble(fd.inner(u,u)*measure)
    rf = ReducedFunctional(J, c)
    pause_annotation()
    assert_taylor_with_guards(rf, gravity, direction, J, min_rate=1.9)


def test_eliminated_lowrank_without_real_fields(meshes):
    import test_gia_gravity as base
    low, state, layout = base.build(meshes, rotation=False, condensed=True,
        dtn_representation='lowrank', solver_parameters='iterative', declare_nullspace=True)
    ref, expected, rlayout = base.build(meshes, rotation=False, condensed=True,
        dtn_representation='multiplier', solver_parameters='direct', declare_nullspace=True)
    low.solve()
    ref.solve()
    assert not layout.real_fields
    assert low.solver_parameters['pc_python_type'] == 'gadopt.ReducedGIABlockPC'
    for name in ('displacement', 'potential'):
        a, b = state.subfunctions[getattr(layout, name)], expected.subfunctions[getattr(rlayout, name)]
        np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=2e-6, atol=2e-9)


def test_lowrank_moving_coastline_matches_multipliers(meshes):
    test_lowrank_matches_multipliers(meshes, 'maxwell', 3, 'direct', moving=True)


def test_lowrank_moving_coastline_adjoint(meshes, tmp_path):
    check_lowrank_adjoint(meshes, 'maxwell', 'memory', tmp_path, moving=True)


def test_lowrank_constant_step_replay(meshes, tmp_path, monkeypatch):
    monkeypatch.setattr(direct, 'STEPS', (.1, .1, .1, .1))
    check_lowrank_adjoint(meshes, 'maxwell', 'memory', tmp_path)
