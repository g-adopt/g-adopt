"""Full mixed GIA BDF2 against independent BE-rescaled history equations."""
import gadopt
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                              pause_annotation, get_working_tape, stop_annotating)
from pyadjoint.tape import annotate_tape
import test_gia_gravity as base
from test_gravity_adjoint import assert_taylor_with_guards

meshes = base.meshes
STEPS = (.08, .13, .07, .12)


@pytest.fixture(autouse=True)
def clean_tape():
    get_working_tape().clear_tape()
    yield
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


def build(meshes, *, scheme='bdf2', condensed=False, exponent=1, control=None,
          iterative=False):
    parent, sub = meshes
    Z, layout = gadopt.self_gravitating_gia_space(
        sub, parent, gravity_bcs=base.gravity_bcs(parent), rotation=True,
        n_internal_variables=2, condense_internal_variables=condensed,
        self_gravity_number=base.LAMBDA, dtn_representation='multiplier')
    z = fd.Function(Z)
    R = fd.FunctionSpace(sub, 'R', 0)
    dt = fd.Function(R).assign(STEPS[0])
    material = base.approximation(shear_modulus=[.4, .6],
        viscosity=[.12, .6*(control if control is not None else 1.)],
        exponent=exponent, transition_stress=.003)
    X = fd.SpatialCoordinate(sub)
    measure = fd.Measure('dx', domain=sub, intersect_measures=(fd.dx(domain=parent),))
    options = dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11,
                   ksp_error_if_not_converged=True)
    if iterative:
        options.update(ksp_type='fgmres', pc_type='lu', pc_factor_mat_solver_type='mumps')
    solver = gadopt.SelfGravitatingGIASolver(z, material, layout=layout, dt=dt,
        bcs=base.mechanics_bcs(sub), time_scheme=scheme,
        rotation_moments={'C': fd.assemble(fd.dot(X, X)*measure)},
        nullspace=gadopt.rigid_rotation_nullspace(Z, layout),
        solver_parameters='direct', solver_parameters_extra=options)
    return solver, dt


@pytest.mark.parametrize('exponent', [1, 3])
def test_coupled_against_independent_rescaled_be(meshes, exponent):
    """BDF2 equals a BE history solve with dt=h/alpha and old=H/alpha."""
    solver, dt = build(meshes, exponent=exponent)
    reference, ref_dt = build(meshes, scheme='backward_euler', exponent=exponent)
    i = reference.layout.internal_variable_field
    old = reference.solution_old.subfunctions[i].copy(deepcopy=True)
    previous = old.copy(deepcopy=True)
    for k, h in enumerate(STEPS):
        dt.assign(h)
        solver.solve()
        if k:
            r = h/STEPS[k-1]
            alpha, beta, gamma = (1+2*r)/(1+r), 1+r, -r*r/(1+r)
        else:
            alpha, beta, gamma = 1., 1., 0.
        reference.solution_old.subfunctions[i].interpolate((beta*old+gamma*previous)/alpha)
        ref_dt.assign(h/alpha)
        reference.solve()
        previous.assign(old)
        old.assign(reference.solution.subfunctions[i])
        for actual, expected in zip(solver.solution.subfunctions, reference.solution.subfunctions):
            np.testing.assert_allclose(actual.dat.data_ro, expected.dat.data_ro, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize('condensed', [False, True])
def test_variable_step_adjoint(meshes, condensed):
    _, sub = meshes
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, 'R', 0)).assign(1.)
        solver, dt = build(meshes, condensed=condensed, control=control)
    continue_annotation()
    c = Control(control)
    for h in STEPS:
        dt.assign(h)
        solver.solve()
    u = fd.split(solver.solution)[solver.layout.displacement]
    J = fd.assemble(fd.inner(u, u)*fd.dx(domain=sub))
    rf = ReducedFunctional(J, c)
    pause_annotation()
    direction = control.copy(deepcopy=True).assign(.2)
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)


def test_default_and_bdf_startup_match(meshes):
    be, _ = build(meshes, scheme='backward_euler')
    bdf, _ = build(meshes)
    be.solve()
    bdf.solve()
    for a, b in zip(be.solution.subfunctions, bdf.solution.subfunctions):
        np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=1e-8, atol=1e-10)
