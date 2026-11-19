"""Full mixed GIA BDF2 against independent BE-rescaled history equations."""
import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                              pause_annotation, get_working_tape, stop_annotating)
from pyadjoint.tape import annotate_tape
from gadopt.gia_couplings import (CentreOfMass, FluidCore, Rotation, SeaLevel,
                                  SurfaceLoad)
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.gia_solver import SelfGravitatingGIASolver
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


def build(meshes, *, scheme='bdf2', exponent=1, control=None):
    """The DG-history solver of the BDF2 tests, with its `Real` time step.

    Rigid core (`un = 0` at the CMB), the rotation row, the prescribed
    `SIGMA_HAT cos 2 phi` load at the surface, multiplier DtN, two Maxwell
    elements and the direct preset at tight tolerances.

    Returns:
      `(solver, dt)`.
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, 'CG', 2))
    psi = fd.Function(fd.FunctionSpace(parent, 'CG', 2))
    R = fd.FunctionSpace(sub, 'R', 0)
    dt = fd.Function(R).assign(STEPS[0])
    material = base.approximation(shear_modulus=[.4, .6],
        viscosity=[.12, .6*(control if control is not None else 1.)],
        exponent=exponent, transition_stress=.003)
    X = fd.SpatialCoordinate(sub)
    measure = fd.Measure('dx', domain=sub, intersect_measures=(fd.dx(domain=parent),))
    # The load enters the traction, the potential and the rotation row once.
    load = SurfaceLoad(boundary=base.CURVE_RE,
                       sigma=base.SIGMA_HAT*fd.cos(2*fd.atan2(X[1], X[0])),
                       g_surface=1.0)
    options = dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11,
                   ksp_error_if_not_converged=True)
    solver = SelfGravitatingGIASolver(
        u, psi, material, dt=dt, time_scheme=scheme,
        gravity_bcs=base.gravity_bcs(parent, sheet=False),
        bcs={base.CURVE_RC: {'un': 0.0}},
        couplings=[Rotation(C=fd.assemble(fd.dot(X, X)*measure),
                            Omega_sq=OMEGA_SQ_EARTH), load],
        dtn_representation='multiplier', nullspace='rigid_rotation',
        solver_parameters='direct', solver_parameters_extra=options)
    return solver, dt


@pytest.mark.parametrize('exponent', [1, 3])
def test_coupled_against_independent_rescaled_be(meshes, exponent):
    """BDF2 equals a BE history solve with dt=h/alpha and old=H/alpha."""
    solver, dt = build(meshes, exponent=exponent)
    reference, ref_dt = build(meshes, scheme='backward_euler', exponent=exponent)
    i = reference.field_index['history']
    old = reference.internal_variables.copy(deepcopy=True)
    previous = old.copy(deepcopy=True)
    for k, h in enumerate(STEPS):
        dt.assign(h)
        solver.solve()
        if k:
            r = h/STEPS[k-1]
            alpha, beta, gamma = (1+2*r)/(1+r), 1+r, -r*r/(1+r)
        else:
            alpha, beta, gamma = 1., 1., 0.
        # The backward-Euler step reads its old history from the accepted
        # level, `internal_variables`, so the rescaled history goes there.
        # The same value also goes into the history field of `solution`, the
        # initial guess of the Newton solve, which changes the iterate path
        # but not the converged answer.
        reference.internal_variables.interpolate((beta*old+gamma*previous)/alpha)
        reference.solution.subfunctions[i].assign(reference.internal_variables)
        ref_dt.assign(h/alpha)
        reference.solve()
        previous.assign(old)
        old.assign(reference.internal_variables)
        for actual, expected in zip(solver.solution.subfunctions, reference.solution.subfunctions):
            np.testing.assert_allclose(actual.dat.data_ro, expected.dat.data_ro, rtol=2e-7, atol=2e-9)


def test_variable_step_adjoint(meshes):
    _, sub = meshes
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, 'R', 0)).assign(1.)
        solver, dt = build(meshes, control=control)
    continue_annotation()
    c = Control(control)
    for h in STEPS:
        dt.assign(h)
        solver.solve()
    u = solver.field('displacement')
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


def build_gate(meshes, SL_init, I_init, I, dt, *, time_scheme):
    """The coupled solver of the power-law gate case, with the BDF2 option.

    The configuration of `test_gia_transient_power_law_gate.build`: three
    Maxwell elements on the DG history, the DG0 power-law exponent and
    transition stress, low-rank DtN with the iterative preset of production
    (`newtonls`), the moving coastline, rotation, the fluid core, the
    centre-of-mass rows and the rigid-rotation nullspace. Only the data
    helpers and constants of the gate module are used here.

    Returns:
      The solver.
    """
    import test_gia_transient_power_law_gate as gate
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, 'CG', 2))
    psi = fd.Function(fd.FunctionSpace(parent, 'CG', 2))
    # One DG0 field per element; the viscosity of element j is mu_j tau_j.
    Q = fd.FunctionSpace(sub, 'DG', 0)
    shear = [fd.Function(Q).assign(mu) for mu in gate.SHEAR_MODULI]
    viscosity = [fd.Function(Q).assign(mu*tau)
                 for mu, tau in zip(gate.SHEAR_MODULI, gate.MAXWELL_TIMES)]
    Xm = fd.SpatialCoordinate(sub)
    rm = fd.sqrt(fd.dot(Xm, Xm))
    approximation = gadopt.CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=shear,
        viscosity=viscosity, bulk_shear_ratio=1,
        g=gate.FRAME.gravity_of_r(rm), B_mu=base.B_MU,
        self_gravity_number=base.LAMBDA,
        exponent=gate.exponent_field(sub),
        transition_stress=fd.Function(Q).assign(gate.transition_stress_value()),
        background_stress=0.0)
    dx_m = fd.Measure('dx', domain=sub,
                      intersect_measures=(fd.Measure('dx', domain=parent),))
    couplings = [
        FluidCore(boundary=base.CURVE_RC, rho_core=gate.FRAME.RHO_CORE,
                  g=fd.Constant(gate.FRAME.gravity_of_r(base.RC))),
        Rotation(C=fd.assemble(fd.dot(Xm, Xm)*dx_m), Omega_sq=OMEGA_SQ_EARTH),
        CentreOfMass(),
        SeaLevel(boundary=base.CURVE_RE, rho_w=gate.RHO_W, rho_i=gate.RHO_I,
                 g_surface=gate.G_RE, SL_init=SL_init, I=I, I_init=I_init,
                 N_init=fd.Constant(0.0), ur_init=fd.Constant(0.0),
                 softplus_width=gate.SOFTPLUS_WIDTH, fixed_ocean=False)]
    solver = SelfGravitatingGIASolver(
        u, psi, approximation, dt=dt, time_scheme=time_scheme,
        gravity_bcs=base.gravity_bcs(parent, sheet=False), bcs={},
        couplings=couplings, dtn_representation='lowrank',
        nullspace='rigid_rotation', solver_parameters='iterative',
        preset_options=dict(outer_rtol=gate.OUTER_RTOL, block0_rtol=1e-4,
                            snes_rtol=gate.SNES_RTOL, snes_type='newtonls'),
        solver_parameters_extra={'snes_atol': gate.SNES_ATOL})
    assert solver.solver_parameters['snes_type'] == 'newtonls'
    return solver, couplings[-1]


def test_bdf2_lowrank_moving_coastline_adjoint(meshes, monkeypatch):
    """Existing production-style gate: rotation, fluid core, COM, live ocean."""
    import test_gia_transient_power_law_gate as gate
    import gadopt.preconditioners as pcs
    events = []
    original_due = pcs._RebuildRule.due
    def due(rule, pc):
        ctx = pcs._solver_appctx(pc)
        events.append((ctx.get('gia_solve_index'), pcs._published_dt(pc)))
        return original_due(rule, pc)
    monkeypatch.setattr(pcs._RebuildRule, 'due', due)
    _, sub = meshes
    with stop_annotating():
        initial_sea, initial_ice, ice_steps = gate.shoreline_state(sub)
        ice = initial_ice.copy(deepcopy=True)
        dt = fd.Function(fd.FunctionSpace(sub, 'R', 0)).assign(STEPS[0])
        solver, sea = build_gate(meshes, initial_sea, initial_ice, ice, dt,
                                 time_scheme='bdf2')
    continue_annotation()
    control = Control(initial_sea)
    for k, h in enumerate(STEPS[:3], 1):
        dt.assign(h)
        ice.assign(ice_steps[k])
        solver.solve()
    with stop_annotating():
        target = initial_sea.copy(deepcopy=True)
        target.interpolate(initial_sea+.1*fd.sin(gate.polar_angle(sub)-gate.PHASE))
    J = fd.assemble((sea.sea_level()-target)**2*sea.measure())
    rf = ReducedFunctional(J, control)
    pause_annotation()
    assert_taylor_with_guards(rf, initial_sea, gate.control_direction(sub), J, min_rate=1.9)

    expected = [STEPS[0]]
    for k in (1, 2):
        r = STEPS[k]/STEPS[k-1]
        expected.append(STEPS[k]*(1+r)/(1+2*r))
    assert events
    for index, effective_dt in events:
        assert effective_dt == pytest.approx(expected[index-1])
    assert 'gia_time_alpha' not in solver.appctx
