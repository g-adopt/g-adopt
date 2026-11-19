"""The recorded time step of `SelfGravitatingGIASolver` on the tape.

Every step kind with a block of G-ADOPT's own (the low-rank DtN path with the
DG history, and the Quadrature history with either DtN representation)
records the same structure: a sweep marker, the step block, a sweep marker
and the kernel projection (`gadopt.gia_step_adjoint`). The tests here check
that structure, the tangent sweep of the Quadrature step block against its
adjoint, and the refusal of its second-order adjoint.

The gradients themselves are checked by the Taylor tests of
`test_gia_solver.py` (`test_taylor_and_replay` and the `Real`-control
tests), which run on both step kinds.
"""
import firedrake as fd
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation,
                               stop_annotating)
from pyadjoint.tape import annotate_tape

import gadopt
import gia_solver_cases as C
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


@pytest.mark.parametrize("control_kind", ["field", "Lambda"])
def test_the_quadrature_step_has_a_tangent(meshes, control_kind):
    """The tangent sweep through the Quadrature step block equals the adjoint pairing.

    The `Lambda` control is a factor of the low-rank prefactor `theta_psi`,
    so it also checks the `theta_psi` column of the tangent right-hand side.
    """
    parent, sub = meshes
    case = Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                model="maxwell2", core="fluid", centre=True)
    if control_kind == "field":
        control, direction = _field_control(sub)
        continue_annotation()
        J = march(meshes, case, control)
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
