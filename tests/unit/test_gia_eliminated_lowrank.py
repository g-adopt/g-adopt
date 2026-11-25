"""Material and exterior-potential elimination in the same mixed GIA solve.

The Quadrature family of `SelfGravitatingGIASolver` eliminates the material
history at the quadrature points, and the low-rank DtN representation
replaces the exterior-potential multipliers by a factored boundary operator.
These tests check that both eliminations together reproduce the multiplier
representation of the same Quadrature-family case (`test_gia_direct_bdf2.build`)
step by step, that the retained `(u, psi)` system is solved to the stated
residual, and that the adjoint of the eliminated low-rank solve passes Taylor
tests with in-memory and on-disk checkpointing.
"""
import gadopt  # noqa: F401  (imported before firedrake, as G-ADOPT requires)
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               pause_annotation, get_working_tape, stop_annotating)
from gadopt.gia_couplings import SurfaceLoad
from gadopt.gia_solver import SelfGravitatingGIASolver
import test_gia_direct_bdf2 as direct
from test_gravity_adjoint import assert_taylor_with_guards

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest

meshes = direct.meshes
clean_tape = direct.clean_tape


def load(ice, k):
    """Write the ice load of step `k` into `ice`.

    The load is `0.001 (1 + 0.2 k) (1 + 0.5 cos 2 phi)`: a uniform part,
    which loads the degree-0 sea level, and a degree-2 part, which drives the
    deformation. It grows by 20 percent per step, so every step of the
    variable-step BDF2 history sees a different right-hand side.

    Args:
      ice: the ice-thickness Function of the `SeaLevel` coupling.
      k: the step index, from 0.
    """
    X = fd.SpatialCoordinate(ice.function_space().mesh())
    ice.interpolate(.001*(1+.2*k)*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))


@pytest.mark.parametrize('model,exponent', [
    ('maxwell', 1), ('maxwell', 3), ('backstress', 3)])
def test_lowrank_matches_multipliers(meshes, model, exponent, moving=False):
    """The low-rank iterative solve reproduces the multiplier direct solve.

    Both solvers are the Quadrature family of the same fluid-core, rotation,
    centre-of-mass and sea-level case. Every field that both hold (the
    displacement, the potential and every `Real` field outside the DtN
    multipliers, which only the multiplier solver has) and the eliminated
    history must agree after each of the four variable steps to `2e-6`
    relative and `2e-9` absolute.
    """
    _, sub = meshes
    c = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
    low, dt, ice = direct.build(meshes, c, direct=True, exponent=exponent, model=model,
                                representation='lowrank', iterative=True,
                                moving_coastline=moving)
    ref, rdt, rice = direct.build(meshes, c, direct=True, exponent=exponent, model=model,
                                  moving_coastline=moving)
    # The low-rank solver holds no `dtn_multiplier_k` field, so its fields
    # are a subset of the multiplier solver's; they are compared by name.
    assert set(low.field_index) <= set(ref.field_index)
    assert not low.multiplier_indices and ref.multiplier_indices
    for k, h in enumerate(direct.STEPS):
        dt.assign(h)
        rdt.assign(h)
        load(ice, k)
        rice.assign(ice)
        low.solve()
        ref.solve()
        for name, i in low.field_index.items():
            a = low.solution.subfunctions[i]
            b = ref.solution.subfunctions[ref.field_index[name]]
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=2e-6, atol=2e-9,
                                       err_msg=name)
        np.testing.assert_allclose(low.internal_variables.dat.data_ro,
                                   ref.internal_variables.dat.data_ro, rtol=2e-6, atol=2e-9)
    assert low.dtn_operator.jacobian_applications > 0
    # The outer preconditioner is the two-block Schur split over the `Real`
    # block; its block 0 is the eliminated `(u, psi)` system.
    block = low.solver.snes.getKSP().getPC().getPythonContext()
    reduced = block.pc.getFieldSplitSchurGetSubKSP()[0].getPC().getPythonContext()
    assert reduced.projected_operator is not None
    # The translation deflation is on by default and the declared rotations
    # leave both translations to it.
    assert reduced.deflation is not None and len(reduced.deflation.Z) == 2
    # Check the actual retained residual, independently of the KSP recurrence.
    # A declared curved-mesh kernel must be projected on both RHS and action.
    # The operator is the retained one and not the KSP's: the KSP solves the
    # deflated system `P A x~ = P b`, and the solution it leaves in
    # `_solution_vec` is the assembled `x = Q b + (I - Q A) x~` of `A x = b`.
    operator = reduced._retained_operator()
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
    """Taylor test of the eliminated low-rank solve, for both checkpoint stores."""
    check_lowrank_adjoint(meshes, model, checkpoint, tmp_path)


def check_lowrank_adjoint(meshes, model, checkpoint, tmp_path, moving=False):
    """Run four taped steps and Taylor-test a functional of the final state.

    The control scales the viscosity by `exp(control)`. The functional is
    `int |u|^2 + 0.1 |M|^2 dx` over the mantle (with `M` the eliminated
    history) plus `0.01 int psi^2 dx` over the parent, so the displacement,
    the history and the potential all carry gradient. The replay of the tape
    at the recorded control must give the recorded value to `1e-8`, and the
    Taylor rate must reach 1.9.

    Args:
      meshes: the module fixture.
      model: "maxwell" or "backstress".
      checkpoint: "memory" keeps the tape in memory; "disk" stores each step
        with `SingleDiskStorageSchedule`.
      tmp_path: the pytest directory for the disk checkpoints.
      moving: the live coastline. The solver then sets no rebuild rule, so
        the cached preconditioner pieces follow the default `lagged` rule
        in the taped march, in every replay and in the adjoint.
    """
    from pathlib import Path
    _, sub = meshes
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
        direction = control.copy(deepcopy=True).interpolate(.1+.03*fd.SpatialCoordinate(sub)[0])
        solver, dt, ice = direct.build(meshes, control, direct=True, model=model,
                                       representation='lowrank', iterative=True, moving_coastline=moving)
    if moving:
        # The default rule applies: the solver forces no rule on this path.
        assert 'gia_rebuild' not in solver.solver_parameters
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
    u = solver.field('displacement')
    psi = solver.field('potential')
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree': 4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    J = fd.assemble((fd.inner(u, u)+.1*fd.inner(solver.internal_variables, solver.internal_variables))*measure)
    J += fd.assemble(.01*psi**2*fd.dx(domain=meshes[0]))
    rf = ReducedFunctional(J, c)
    pause_annotation()
    assert float(rf(control)) == pytest.approx(float(J), rel=1e-8)
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)
    assert 'gia_time_alpha' not in solver.appctx


def test_lowrank_gravity_control_adjoint(meshes):
    """Taylor test with `B_mu`, a `Real` Function, as the control.

    `B_mu` enters the momentum residual, the row scaling of every coupling
    and the low-rank boundary operator, so this checks that a scalar control
    reaches all of them on the tape of the eliminated low-rank solve.
    """
    _, sub = meshes
    with stop_annotating():
        material_control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
        gravity = fd.Function(fd.FunctionSpace(meshes[0], 'R', 0)).assign(.03)
        direction = gravity.copy(deepcopy=True).assign(.005)
        solver, dt, ice = direct.build(meshes, material_control, direct=True,
                                       representation='lowrank', iterative=True,
                                       gravity_control=gravity)
    continue_annotation()
    c = Control(gravity)
    for k, h in enumerate(direct.STEPS):
        dt.assign(h)
        load(ice, k)
        solver.solve()
    u = solver.field('displacement')
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree': 4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    J = fd.assemble(fd.inner(u, u)*measure)
    rf = ReducedFunctional(J, c)
    pause_annotation()
    assert_taylor_with_guards(rf, gravity, direction, J, min_rate=1.9)


def build_without_real_fields(meshes, representation, preset):
    """The Quadrature-family annulus with no `Real` field in the mixed space.

    The `cos 2 phi` load of `test_gia_gravity` on Re is the only coupling.
    The Quadrature family accepts natural displacement boundaries only, and a
    `FluidCore` would add the `core_pressure` field, so the core-mantle
    boundary is traction free. The rigid rotation is then a kernel of the
    mechanics, and it is declared. The rigid translations are not declared:
    the degree-2 load has no degree-1 part, so the solves do not excite them.

    Args:
      meshes: the module fixture.
      representation: "lowrank" or "multiplier".
      preset: "iterative" or "direct".

    Returns:
      The solver. Nothing is solved.
    """
    from gia_models import annulus as base
    parent, sub = meshes
    u, psi = base.new_fields(meshes)
    return SelfGravitatingGIASolver(
        u, psi, base.approximation(), dt=1.0,
        gravity_bcs=base.gravity_bcs(parent, sheet=False), bcs={},
        couplings=[SurfaceLoad(boundary=base.CURVE_RE, sigma=base.load_density(sub),
                               g_surface=1.0)],
        internal_variable_family='Quadrature', dtn_representation=representation,
        solver_parameters=preset, nullspace='rigid_rotation')


def test_eliminated_lowrank_without_real_fields(meshes):
    """With no `Real` block, `ReducedGIABlockPC` is the outer preconditioner.

    The low-rank iterative solve and the multiplier direct solve of the same
    case must agree on the displacement and the potential to `2e-6` relative
    and `2e-9` absolute.
    """
    low = build_without_real_fields(meshes, 'lowrank', 'iterative')
    ref = build_without_real_fields(meshes, 'multiplier', 'direct')
    low.solve()
    ref.solve()
    assert not low.real_indices
    assert low.solver_parameters['pc_python_type'] == 'gadopt.ReducedGIABlockPC'
    for name in ('displacement', 'potential'):
        a = low.solution.subfunctions[low.field_index[name]]
        b = ref.solution.subfunctions[ref.field_index[name]]
        np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=2e-6, atol=2e-9)


def test_lowrank_moving_coastline_matches_multipliers(meshes):
    """The comparison of `test_lowrank_matches_multipliers` with the live coastline."""
    test_lowrank_matches_multipliers(meshes, 'maxwell', 3, moving=True)


def test_lowrank_moving_coastline_adjoint(meshes, tmp_path):
    """The Taylor test of `check_lowrank_adjoint` with the live coastline."""
    check_lowrank_adjoint(meshes, 'maxwell', 'memory', tmp_path, moving=True)


def test_lowrank_constant_step_replay(meshes, tmp_path, monkeypatch):
    """The Taylor test with four equal steps, where BDF2 has constant coefficients."""
    monkeypatch.setattr(direct, 'STEPS', (.1, .1, .1, .1))
    check_lowrank_adjoint(meshes, 'maxwell', 'memory', tmp_path)
