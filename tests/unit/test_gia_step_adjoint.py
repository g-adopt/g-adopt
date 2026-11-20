"""The recorded time step of `SelfGravitatingGIASolver` on the tape.

Every step kind with a block of G-ADOPT's own (the low-rank DtN path with the
DG history, and the Quadrature history with either DtN representation)
records the same structure: a sweep marker, the step block, a sweep marker
and the kernel projection (`gadopt.gia_step_adjoint`). The tests here check
that structure, the tangent sweep of the Quadrature step block against its
adjoint and against a finite difference, the refusal of its second-order
adjoint, and that a finished run leaves no solver alive once the tape is
gone, with either preset.

The gradients themselves are checked by the Taylor tests of
`test_gia_solver.py` (`test_taylor_and_replay` and the `Real`-control
tests), which run on both step kinds.
"""
import gc
import weakref

import firedrake as fd
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from petsc4py import PETSc
from pyadjoint.tape import annotate_tape

import gadopt
import gia_cases as C
from gadopt import gia_couplings as cp
from test_gia_solver import STEPS, NewSolver, march

Case = C.Case

#: The step kinds with a step block of G-ADOPT's own.
STEP_KINDS = [
    pytest.param(Case(representation="lowrank", model="maxwell2", core="fluid",
                      centre=True), id="lowrank-DG"),
    pytest.param(Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                      model="maxwell2", core="fluid", centre=True),
                 id="quadrature-lowrank"),
    pytest.param(Case(family="Quadrature", representation="multiplier",
                      core="fluid", centre=True), id="quadrature-multiplier"),
]


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


def _field_control(sub):
    """The viscosity control of `test_gia_solver.march` and a direction."""
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, "CG", 1)).assign(.1)
        direction = fd.Function(control.function_space()).interpolate(
            .1 + .03 * fd.SpatialCoordinate(sub)[0])
    return control, direction


def _tangent_and_gradient(rf, control, direction):
    """The tangent sweep `dJ/dm h` and the adjoint pairing `<dJ/dm, h>`.

    Both at the taped control: the replay first sets the saved values of
    the tape to it.
    """
    rf(control)
    gradient = rf.derivative()
    with gradient.dat.vec_ro as g, direction.dat.vec_ro as h:
        pairing = g.dot(h)
    tape = get_working_tape()
    rf(control)
    tape.reset_tlm_values()
    rf.controls[0].block_variable.tlm_value = direction.copy(deepcopy=True)
    tape.evaluate_tlm()
    return float(rf.functional.block_variable.tlm_value), pairing


@pytest.mark.parametrize("case", STEP_KINDS)
def test_every_step_kind_records_the_same_structure(meshes, case):
    """Marker, step block, marker, projection: once per solve, in this order."""
    from gadopt.gia_step_adjoint import GIAStepBlockMixin, TapeSweepMarkerBlock
    from gadopt.nullspaces import KernelProjectionBlock

    control, _ = _field_control(meshes[1])
    continue_annotation()
    march(meshes, case, control, steps=STEPS[:2])
    pause_annotation()
    blocks = get_working_tape().get_blocks()
    steps = [i for i, b in enumerate(blocks) if isinstance(b, GIAStepBlockMixin)]
    assert len(steps) == 2, [type(b).__name__ for b in blocks]
    for index, i in enumerate(steps, start=1):
        before, step, after, projection = blocks[i - 1:i + 3]
        assert isinstance(before, TapeSweepMarkerBlock) and before.before
        assert isinstance(after, TapeSweepMarkerBlock) and not after.before
        assert isinstance(projection, KernelProjectionBlock)
        # The step block carries the index of its forward solve; the
        # markers around it the same index.
        assert step.gia_solve_index == before.solve_index == after.solve_index
        # The projection is the one of the step solution: its input is the
        # output of the step block.
        assert any(dep is step.get_outputs()[0]
                   for dep in projection.get_dependencies())
    # One projection per step: no step block projects inside its own map.
    assert sum(isinstance(b, KernelProjectionBlock) for b in blocks) == 2


def _central_difference(rf, control, direction, eps=1e-4):
    """`(J(m + eps h) - J(m - eps h)) / (2 eps)` by two replays of the tape.

    Independent of the adjoint and of the tangent sweep: it uses only the
    replay of the step blocks.
    """
    with stop_annotating():
        plus = fd.Function(control.function_space()).assign(control + eps * direction)
        minus = fd.Function(control.function_space()).assign(control - eps * direction)
    return (float(rf(plus)) - float(rf(minus))) / (2 * eps)


@pytest.mark.parametrize("control_kind", ["field", "Lambda", "dt"])
def test_the_quadrature_step_has_a_tangent(meshes, control_kind):
    """The tangent sweep through the Quadrature step block is the derivative.

    Two checks on the same tape: the tangent equals the adjoint pairing,
    and it equals a central finite difference of the replayed functional,
    which uses neither the adjoint nor the tangent code.

    The `Lambda` control is a factor of the low-rank prefactor `theta_psi`,
    so it also checks the `theta_psi` column of the tangent right-hand side.
    The `dt` control scales the time step of both steps, so it checks the
    time coefficients of the step and, over two steps, the history inputs
    of the second one.
    """
    parent, sub = meshes
    case = Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                model="maxwell2", core="fluid", centre=True)
    if control_kind == "field":
        control, direction = _field_control(sub)
        continue_annotation()
        J = march(meshes, case, control)
    elif control_kind == "dt":
        # A Real factor on the time step of every step: dt = h * control.
        with stop_annotating():
            control = fd.Function(fd.FunctionSpace(sub, "R", 0)).assign(1.0)
            direction = fd.Function(control.function_space()).assign(0.3)
            dt = C.dt_function(sub)
            solver, u, psi = C.build_new(meshes, case, dt=dt,
                                         nullspace="rigid_rotation")
        continue_annotation()
        for h in STEPS[:2]:
            dt.assign(h * control)
            solver.solve()
        J = (fd.assemble(fd.inner(u, u) * fd.dx(domain=sub))
             + fd.assemble(psi * psi * fd.dx(domain=parent)))
    else:
        with stop_annotating():
            control = fd.Function(fd.FunctionSpace(parent, "R", 0)).assign(C.LAMBDA)
            direction = fd.Function(control.function_space()).assign(.1 * C.LAMBDA)
            dt = C.dt_function(sub)
            u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2))
            psi = fd.Function(fd.FunctionSpace(parent, "CG", 2))
            material = gadopt.CompressibleInternalVariableApproximation(
                bulk_modulus=1.0, density=1.0, g=1.0, B_mu=C.B_MU,
                self_gravity_number=control, shear_modulus=[.4, .6],
                viscosity=[.12, .6])
            solver = NewSolver(
                u, psi, material, dt=dt, time_scheme=case.scheme,
                internal_variable_family="Quadrature", quad_degree=C.QUAD,
                gravity_bcs=C.gravity_bcs(parent),
                couplings=[cp.FluidCore(boundary=C.CURVE_RC, rho_core=1.2, g=1.),
                           cp.CentreOfMass(),
                           cp.SurfaceLoad(boundary=C.CURVE_RE,
                                          sigma=C.load_expression(sub), g_surface=1.0)],
                nullspace="rigid_rotation", solver_parameters="direct",
                solver_parameters_extra=dict(snes_rtol=1e-11, snes_atol=1e-13,
                                             ksp_rtol=1e-11))
        continue_annotation()
        for h in STEPS[:2]:
            dt.assign(h)
            solver.solve()
        J = (fd.assemble(fd.inner(u, u) * fd.dx(domain=sub))
             + fd.assemble(psi * psi * fd.dx(domain=parent)))
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    tangent, pairing = _tangent_and_gradient(rf, control, direction)
    assert abs(pairing) >= 1e-6 * abs(float(J))
    assert abs(tangent - pairing) <= 1e-8 * abs(pairing), (tangent, pairing)
    # The error of a central difference is O(eps^2). On this case it is
    # 1.0e-10 (field) and 4.7e-11 (dt) relative at eps = 1e-4, so the bound
    # leaves four orders of margin.
    central = _central_difference(rf, control, direction)
    assert abs(central - tangent) <= 1e-6 * abs(tangent), (central, tangent)


def test_the_quadrature_step_refuses_a_hessian(meshes):
    """The second-order adjoint of the Quadrature step is not written: it raises."""
    case = Case(family="Quadrature", representation="multiplier", core="fluid",
                centre=True)
    control, direction = _field_control(meshes[1])
    continue_annotation()
    J = march(meshes, case, control, steps=STEPS[:1])
    rf = ReducedFunctional(J, Control(control))
    pause_annotation()
    rf.derivative()
    with pytest.raises(NotImplementedError, match="Quadrature history step"):
        rf.hessian(direction)


#: The configurations of `test_a_finished_run_is_collected`: the step kind,
#: the preset and whether the run is taped. The `"direct"` preset puts
#: `firedrake.AssembledPC` (which holds the solver's `_SNESContext` and so its
#: callbacks) in the PETSc tree; the iterative preset puts the Python
#: preconditioners that hold the application context there.
_COLLECTED_CASES = [
    pytest.param(STEP_KINDS[0].values[0], "iterative", True, id="lowrank-DG-iterative"),
    pytest.param(STEP_KINDS[0].values[0], "direct", True, id="lowrank-DG-direct"),
    pytest.param(STEP_KINDS[0].values[0], "direct", False, id="lowrank-DG-direct-untaped"),
    pytest.param(STEP_KINDS[1].values[0], "iterative", True, id="quadrature-lowrank-iterative"),
    pytest.param(STEP_KINDS[1].values[0], "direct", True, id="quadrature-lowrank-direct"),
    pytest.param(STEP_KINDS[1].values[0], "direct", False,
                 id="quadrature-lowrank-direct-untaped"),
    pytest.param(STEP_KINDS[2].values[0], "direct", True, id="quadrature-multiplier-direct"),
]


@pytest.mark.parametrize("case,preset,taped", _COLLECTED_CASES)
def test_a_finished_run_is_collected(meshes, case, preset, taped):
    """No solver of a finished run stays alive once the tape is gone.

    PETSc holds the Python preconditioner contexts from C. Two routes lead
    from such a context back to the solver: the application context, whose
    low-rank DtN operator reads `theta_psi` from the solver (iterative
    preset), and the `_SNESContext` that `firedrake.AssembledPC` holds,
    whose callbacks are methods of the solver and of the Quadrature backend
    (direct preset). If either route held the solver strongly, the solver,
    its taping solver, the cached replay and derivative solvers and all
    their preconditioner trees and factorisations would form a cycle
    through C that Python's collector cannot see, and stay alive to the end
    of the process. Both routes hold the solver weakly
    (`gadopt.utility.weak_callback`).
    """
    control, _ = _field_control(meshes[1])
    with stop_annotating():
        dt = C.dt_function(meshes[1])
        solver, u, psi = C.build_new(meshes, case, control=control, dt=dt,
                                     nullspace="rigid_rotation",
                                     solver_parameters=preset)
    if taped:
        continue_annotation()
    for h in STEPS[:2]:
        dt.assign(h)
        solver.solve()
    J = fd.assemble(fd.inner(u, u) * fd.dx(domain=meshes[1]))
    if taped:
        rf = ReducedFunctional(J, Control(control))
        pause_annotation()
        rf.derivative()
        del rf
    alive = weakref.ref(solver)
    del solver, u, psi, J
    get_working_tape().clear_tape()
    gc.collect()
    PETSc.garbage_cleanup(meshes[1].comm)
    gc.collect()
    assert alive() is None
