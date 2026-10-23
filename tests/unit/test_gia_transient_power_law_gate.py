"""Gate test of the rheology of GIAMIP Exp09, Exp10 and Exp12 on the 2-D annulus.

The three experiments run a combination that no other test runs together:

- `N = 3` Maxwell elements in parallel (the Extended Burgers fit of Exp09 and
  Exp12), held in one internal-variable field of shape `(3, 2, 2)` on DG1;
- a power law whose exponent is a DG0 `Function` that is exactly `1.0` in some
  cells and exactly `3.5` in others (the depth rule of Exp10 and Exp12), with a
  DG0 transition stress that is positive in every cell and no background
  stress;
- the low-rank DtN representation on the uncondensed layout, with the
  iterative preset of production (`selfgrav_dtn_iterative_solver_parameters`)
  and Newton (`newtonls`);
- the sea-level equation with a moving coastline (`fixed_ocean=False`),
  rotation, the fluid core, the centre-of-mass rows and the rigid-rotation
  nullspace declared on both sides, as `exp01_common.build_solver` does;
- the time step `dt` as a `Function` on the Real space of the mantle mesh,
  assigned on the tape (decision S28 of the design handout).

Three differences to the production solver of `exp01_common.build_solver`
remain, and this test does not cover them:

- rotation: the annulus has one rotation row (the change of rotation rate),
  and the test passes the polar moment `C` only. The polar motion rows and
  the `k_s` route of production (`C - A = Q k_s`) are not exercised here;
  `test_sea_level.py::TestRotation` and the Spada benchmark cover them;
- masks: production passes the frozen surface slope (`slope`, `grad_floor`)
  and `alpha_mask = 1`; this test uses `alpha_mask = 0.5` and no slope;
- stopping rule: production solves the outer FGMRES to 1e-6 and counts a
  capped solve as converged (`outer_accept_capped=True`); this test solves
  to 1e-11 and lets a capped solve fail, as the replay and Taylor
  tolerances need.

The test tapes a march of `N_STEPS` steps of a melting ice sheet and checks
three things (design handout `NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-
2026-10-20.md`, section 12.0):

1. every forward step converges with `newtonls`, three elements reach the
   solver, and the power law is live: the factor `f` of
   `power_law_factor`, which multiplies every Maxwell time (`f <= 1`; the
   viscosity drops by `1 / f`), is exactly 1 in the cells with `n = 1` and
   clearly below 1 in the cells with `n = 3.5`, and the solution differs
   clearly from the one of the Newtonian twin;
2. a replay of the tape at a new control equals a fresh build and solve at
   that control;
3. a guarded Taylor test of a sea-level misfit with respect to the initial
   sea level `SL_init` (the control of the GIAMIP bed inversions) has rate 2.

The ice and the sea level are the `"shoreline"` state of `test_sea_level.py`:
a smooth sea level with land, and an ice sheet on the land that melts by
`MELT_PER_STEP` of its initial thickness at every step. The masks of the
moving coastline are fractional over a wide arc, so the coastline terms of the
Jacobian are live.
"""

import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from pyadjoint.tape import annotate_tape

from gadopt import (
    CompressibleInternalVariableApproximation,
    SelfGravitatingGIASolver,
    rigid_rotation_nullspace,
    self_gravitating_gia_space,
)
from gadopt.gia_gravity import (
    FluidCore,
    SeaLevel,
    selfgrav_dtn_iterative_solver_parameters,
)
from gadopt.internal_variable_equation import history_slices
import test_gia_gravity
from test_gia_gravity import (
    B_MU,
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    RC,
    RE,
    gravity_bcs,
)
from test_gravity_adjoint import (
    assert_taylor_with_guards,
    taylor_first_order_ladder,
)

#: The module fixture of `test_gia_gravity` (parent annulus, mantle
#: submesh), bound to a name of this module so that pytest finds it.
meshes = test_gia_gravity.meshes

# ---------------------------------------------------------------------------
# Constants of the case
# ---------------------------------------------------------------------------

#: The reference state of `TestCentreOfMassFrame` (core density 2 inside Rc, a
#: unit-density mantle) and its surface gravity, as in `test_sea_level.py`.
#: Reached through the module so that pytest does not collect the class here.
FRAME = test_gia_gravity.TestCentreOfMassFrame()
G_RE = FRAME.gravity_of_r(RE)

#: Water and ice density in units of the mantle density (`test_sea_level.py`).
RHO_W = 1000.0 / 4978.0
RHO_I = 931.0 / 4978.0
#: The mask steepness factor `k = alpha p / h` of the sea-level masks.
ALPHA_MASK = 0.5
#: Rotation of the shoreline state, so that it has no mirror symmetry.
PHASE = 0.5

#: The three Maxwell elements: shear moduli (their sum is the instant modulus,
#: 1 in units of mu_bar) and Maxwell times in units of t_bar. The viscosity of
#: each element is `mu_j tau_j`. The times span two decades from twice the
#: time step: `dt / tau_j` is at most 0.5 before the power law and about 2.9
#: after it (the power law lowers the viscosity by up to a factor 5.7 in the
#: most stressed cell, `1 / f` with f = 0.175). That is the regime of Exp10
#: (one element, tau_M about 140 yr against a 10 yr step, a viscosity drop
#: of up to about 17).
SHEAR_MODULI = (0.5, 0.3, 0.2)
MAXWELL_TIMES = (2.0, 20.0, 200.0)
#: The stiff variant of `test_newton_stalls_on_a_stiff_power_law`: the same
#: elements ten times faster, so `dt / tau_1 = 5` before the power law and
#: several hundred after it (a viscosity drop of up to about 100 in the
#: ocean cells of the second layer at the last Newton iterate). Exp12 reaches this regime: an EB element of 1e-2
#: tau_M under the Lloyd lower bound of 1e20 Pa s is about 0.5 yr against a
#: 10 yr step.
STIFF_MAXWELL_TIMES = (0.2, 2.0, 20.0)

#: The power law. Exponent 3.5 in the cells whose centroid is deeper than
#: `LID_DEPTH` and shallower than `LOWER_DEPTH` below Re, exactly 1.0 in every
#: other cell: the 90-670 km rule of Exp10 and Exp12 (decision S21) on the
#: annulus, whose mantle is 1.0 thick in five layers of 0.2.
N_POWER_LAW = 3.5
LID_DEPTH = 0.2
LOWER_DEPTH = 0.8

#: The time step (units of t_bar) and the number of steps of the march.
DT = 1.0
N_STEPS = 3
#: The fraction of the initial ice that melts at each step.
MELT_PER_STEP = 0.1

#: Tolerances. The outer FGMRES and Newton are tight, so that the replay and
#: the Taylor remainder are not limited by the solve.
OUTER_RTOL = 1e-11
SNES_RTOL = 1e-10
SNES_ATOL = 1e-14


@pytest.fixture(autouse=True)
def clean_tape():
    """A fresh tape for every test, and annotation off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


def polar_angle(mesh):
    """`phi = atan2(y, x)` as UFL on `mesh`."""
    X = fd.SpatialCoordinate(mesh)
    return fd.atan2(X[1], X[0])


def exponent_field(sub):
    """The DG0 exponent: exactly 3.5 in the middle layers, exactly 1.0 elsewhere.

    The depth is tested at the cell centroid (DG0 interpolation evaluates at
    the centroid), as the prepare scripts of the experiments do. The values
    are written with `dat.data[:] =` from a boolean mask, so they are the two
    floats exactly, and the test `ufl.eq(n - 1, 0)` of `power_law_factor`
    holds exactly in the `n = 1` cells.
    """
    Q = fd.FunctionSpace(sub, "DG", 0)
    X = fd.SpatialCoordinate(sub)
    depth = fd.Function(Q).interpolate(RE - fd.sqrt(fd.dot(X, X)))
    exponent = fd.Function(Q, name="exponent")
    with stop_annotating():
        d = depth.dat.data_ro
        exponent.dat.data[:] = np.where(
            (d > LID_DEPTH) & (d < LOWER_DEPTH), N_POWER_LAW, 1.0)
    return exponent


def shoreline_state(sub):
    """`SL_init`, the initial ice, and the ice of every step of the march.

    The `"shoreline"` state of `test_sea_level.py`: ocean where
    `cos(phi - PHASE) > -0.2`, land around `phi = PHASE + pi`, and an ice sheet
    on the land that is 0.36 thick at its centre. Step `k` carries
    `(1 - k MELT_PER_STEP)` of the initial ice. Each step's ice is its own
    Function, so a replay reads the ice of the right step.

    Returns:
      `(SL_init, I_init, ice)`, with `ice[k]` the ice of step `k`, `k = 0 ..
      N_STEPS`.
    """
    V = fd.FunctionSpace(sub, "CG", 2)
    phi = polar_angle(sub)
    SL_init = fd.Function(V, name="SL_init").interpolate(
        fd.cos(phi - PHASE) + 0.2)
    I_init = fd.Function(V, name="I_init").interpolate(
        0.6 * fd.max_value(0.0, -fd.cos(phi - PHASE) - 0.4))
    ice = [fd.Function(V, name=f"ice_{k}").interpolate(
        (1.0 - k * MELT_PER_STEP) * I_init) for k in range(N_STEPS + 1)]
    return SL_init, I_init, ice


def transition_stress_value():
    """The transition stress of the power law, in units of mu_bar.

    Chosen so that the power law of this case is as strong as the one the
    design expects of Exp10 and Exp12 under the large ice sheets (handout
    section 7.3: a viscosity drop 1 + (sigma / sigma_T)^2.5 from about 2 to
    17). On the Newtonian twin of this case, sqrt(J2) in the power-law cells
    after `N_STEPS` steps is 0.009 at the median and 0.023 at the largest,
    so sigma_T = 0.01 gives a drop of about 2 at the median and about 8 in
    the most stressed cell. With the power law on, the largest drop is 5.7.
    """
    return 1e-2


def build(meshes, SL_init, I_init, I, dt, *, power_law=True,
          maxwell_times=MAXWELL_TIMES):
    """The coupled solver of the gate case.

    Args:
      meshes: the module fixture (parent annulus, mantle submesh).
      SL_init: the initial sea level, a CG2 Function on the mantle (the
        control of the Taylor test).
      I_init, I: the initial ice and the ice of the current step.
      dt: the time step, a Function on the Real space of the mantle.
      power_law: False builds the Newtonian twin (exponent 1 everywhere).
      maxwell_times: the Maxwell times of the elements, units of t_bar.

    Returns:
      `(solver, layout, approximation)`.
    """
    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent, sheet=False),
        n_internal_variables=len(SHEAR_MODULI), rotation=True,
        fluid_core=True, centre_of_mass=True, sea_level=True,
        self_gravity_number=LAMBDA, condense_internal_variables=False,
        dtn_representation="lowrank")
    z = fd.Function(Z)

    # One DG0 field per element, as the prepare scripts write them; the
    # viscosity of element j is mu_j tau_j.
    Q = fd.FunctionSpace(sub, "DG", 0)
    shear = [fd.Function(Q).assign(mu) for mu in SHEAR_MODULI]
    viscosity = [fd.Function(Q).assign(mu * tau)
                 for mu, tau in zip(SHEAR_MODULI, maxwell_times)]
    power = {}
    if power_law:
        power = dict(exponent=exponent_field(sub),
                     transition_stress=fd.Function(Q).assign(
                         transition_stress_value()),
                     background_stress=0.0)
    Xm = fd.SpatialCoordinate(sub)
    rm = fd.sqrt(fd.dot(Xm, Xm))
    approximation = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=shear,
        viscosity=viscosity, bulk_shear_ratio=1,
        g=FRAME.gravity_of_r(rm), B_mu=B_MU, self_gravity_number=LAMBDA,
        **power)

    sea_level = SeaLevel(
        boundary=CURVE_RE, rho_w=RHO_W, rho_i=RHO_I, g_surface=G_RE,
        SL_init=SL_init, I=I, I_init=I_init,
        N_init=fd.Constant(0.0), ur_init=fd.Constant(0.0),
        alpha_mask=ALPHA_MASK, fixed_ocean=False)

    parameters = selfgrav_dtn_iterative_solver_parameters(
        condensed=False, outer_rtol=OUTER_RTOL, block0_rtol=1e-4,
        snes_rtol=SNES_RTOL, snes_type="newtonls",
        dtn_representation="lowrank", n_real=len(layout.real_fields))
    parameters["snes_atol"] = SNES_ATOL
    # The configuration the gate is about, stated here so a change of a
    # default cannot move the test off it.
    assert not layout.condensed
    assert parameters["snes_type"] == "newtonls"

    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    nullspace = rigid_rotation_nullspace(Z, layout)
    solver = SelfGravitatingGIASolver(
        z, approximation, layout=layout, dt=dt, bcs={},
        fluid_core=FluidCore(boundary=CURVE_RC, rho_core=FRAME.RHO_CORE,
                             g=fd.Constant(FRAME.gravity_of_r(RC))),
        sea_level=sea_level, dtn_representation="lowrank",
        rotation_moments={"C": fd.assemble(fd.dot(Xm, Xm) * dx_m)},
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=parameters)
    assert solver.dtn_representation == "lowrank"
    return solver, layout, approximation


def march(meshes, SL_init, *, power_law=True, steps=N_STEPS, record=None,
          maxwell_times=MAXWELL_TIMES):
    """Build the solver and run the march; return the misfit J.

    Everything but the per-step assignments, the solves and the misfit runs
    with annotation paused, as in `invert_exp01.py`. The ice of each step is
    assigned to the solver's ice field (on the tape when annotating), and so
    is `dt` at the first step.

    Args:
      meshes: the module fixture.
      SL_init: the initial sea level (the control).
      power_law: False runs the Newtonian twin.
      steps: the number of steps.
      record: a list that receives one dict per step (Newton iterations,
        convergence reason), or None.
      maxwell_times: the Maxwell times of the elements, units of t_bar.

    Returns:
      `(J, solver, layout, approximation)`.
    """
    parent, sub = meshes
    with stop_annotating():
        _, I_init, ice = shoreline_state(sub)
        I = fd.Function(I_init.function_space(), name="ice").assign(ice[0])
        R = fd.FunctionSpace(sub, "R", 0)
        dt = fd.Function(R, name="dt")
        solver, layout, approximation = build(meshes, SL_init, I_init, I, dt,
                                              power_law=power_law,
                                              maxwell_times=maxwell_times)
    # The time step on the tape (decision S28): a replay then reads the
    # value of this assignment.
    dt.assign(DT)
    for k in range(1, steps + 1):
        I.assign(ice[k])
        solver.solve()
        if record is not None:
            snes = solver.solver.snes
            record.append({"step": k,
                           "newton": int(snes.getIterationNumber()),
                           "reason": int(snes.getConvergedReason()),
                           "outer": int(snes.getLinearSolveIterations())})
    with stop_annotating():
        target = misfit_target(sub)
    SL = solver.sea_level()
    J = fd.assemble((SL - target) ** 2 * solver.sea_level_measure()(CURVE_RE))
    return J, solver, layout, approximation


def misfit_target(sub):
    """The "observed" sea level of the misfit: the reference `SL_init` plus 0.1 sin(phi - PHASE).

    The misfit is the one of the GIAMIP bed inversions: the predicted sea
    level at the end of the march against an observed sea level,
    `J = int (SL - target)^2 ds`. With this target the residual `SL - target`
    is of order 0.1 and has a large component along the Taylor direction
    (`control_direction`), so the first-order term of the Taylor expansion
    dominates the second-order one at the step sizes of the test and the
    negative control (gradient set to zero) has rate 1. A target equal to
    the reference `SL_init` leaves a residual of the size of the sea-level
    change of the march (about 1e-2), and the negative control then has
    rate 2.
    """
    reference, _, _ = shoreline_state(sub)
    return fd.Function(reference.function_space()).interpolate(
        reference + 0.1 * fd.sin(polar_angle(sub) - PHASE))


def control_direction(sub):
    """The Taylor direction: a shift of the initial sea level of up to 0.05.

    `sin(phi - PHASE)` is odd about the centre of the land, so it moves the
    coastline on one flank seaward and on the other landward, and the
    derivative of the ocean mask enters the gradient.
    """
    V = fd.FunctionSpace(sub, "CG", 2)
    return fd.Function(V).interpolate(0.05 * fd.sin(polar_angle(sub) - PHASE))


def taped_reduced_functional(meshes):
    """Tape the march with `SL_init` as control; return `(Jhat, m, J, solver)`."""
    _, sub = meshes
    with stop_annotating():
        SL_init, _, _ = shoreline_state(sub)
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        control = Control(SL_init)
        J, solver, _, _ = march(meshes, SL_init)
        Jhat = ReducedFunctional(J, control)
    finally:
        pause_annotation()
    return Jhat, SL_init, J, solver


def test_forward_steps_converge_and_power_law_is_live(meshes):
    """Every step converges; three elements; f is 1 where n = 1 and below 1 where n = 3.5."""
    _, sub = meshes
    with stop_annotating():
        SL_init, _, _ = shoreline_state(sub)
        rows = []
        _, solver, layout, approximation = march(meshes, SL_init, record=rows)
        # The Newtonian twin: the same march with exponent 1 everywhere.
        _, twin, _, _ = march(meshes, SL_init.copy(deepcopy=True),
                              power_law=False)
    # Three Maxwell elements reach the solver, in one field of shape (3, 2, 2).
    assert len(approximation.maxwell_times) == len(SHEAR_MODULI) == 3
    m_field = solver.solution.subfunctions[layout.internal_variable_field]
    assert m_field.ufl_shape == (3, 2, 2)
    for row in rows:
        print(f"    [gate] step {row['step']} newton {row['newton']} outer "
              f"{row['outer']} reason {row['reason']}")
        assert row["reason"] > 0, f"step {row['step']} did not converge: {row}"
    # Newton must have done real work: a moving coastline and a power law
    # make the residual nonlinear, so at least two Newton iterations per step.
    assert all(row["newton"] >= 2 for row in rows)

    # The power-law factor at the converged state, per cell (DG0 average of
    # the factor, which is evaluated at quadrature points).
    with stop_annotating():
        u = solver.solution.subfunctions[layout.displacement]
        m = solver.solution.subfunctions[layout.internal_variable_field]
        dev = approximation.deviatoric_stress(u, history_slices(m))
        Q = fd.FunctionSpace(sub, "DG", 0)
        f = fd.Function(Q).project(approximation.power_law_factor(dev))
        sigma = fd.Function(Q).project(
            approximation.second_stress_invariant(dev))
        n = approximation.exponent.dat.data_ro
        f_data = f.dat.data_ro
        print(f"    [gate] sqrt(J2): max {sigma.dat.data_ro.max():.3e}, "
              f"median {np.median(sigma.dat.data_ro):.3e}; f in n=3.5 cells "
              f"{f_data[n == N_POWER_LAW].min():.3f}..."
              f"{f_data[n == N_POWER_LAW].max():.3f}; f in n=1 cells "
              f"{f_data[n == 1.0].min():.15f}...{f_data[n == 1.0].max():.15f}")
    # Both kinds of cell exist.
    assert np.any(n == N_POWER_LAW) and np.any(n == 1.0)
    # n = 1: the factor is exactly 1 (the conditional of `power_law_factor`).
    assert np.allclose(f_data[n == 1.0], 1.0, rtol=0, atol=1e-13)
    # n = 3.5: the factor is clearly below 1 in the most stressed cell
    # (measured 0.175, a viscosity drop of 5.7).
    assert f_data[n == N_POWER_LAW].min() < 0.9
    # The solver uses the factor: the displacement differs from the one of
    # the Newtonian twin by about 10 % of its largest value (the factor
    # computed above from the converged state would be below 1 even if the
    # solver ignored it).
    with stop_annotating():
        u_twin = twin.solution.subfunctions[layout.displacement]
        difference = np.abs(u.dat.data_ro - u_twin.dat.data_ro).max()
        size = np.abs(u_twin.dat.data_ro).max()
    print(f"    [gate] displacement against the Newtonian twin: "
          f"{difference / size:.3e} of its largest value")
    assert difference > 0.05 * size


def test_replay_matches_fresh_solve(meshes):
    """The replay of the taped march at a new `SL_init` equals a fresh march there."""
    _, sub = meshes
    Jhat, SL_init, J, solver = taped_reduced_functional(meshes)
    solver_sea_level = solver.sea_level()
    dss = solver.sea_level_measure()(CURVE_RE)
    with stop_annotating():
        shifted = SL_init.copy(deepcopy=True)
        shifted += control_direction(sub)
        J_fresh, _, _, _ = march(meshes, shifted.copy(deepcopy=True))
    # The misfit a replay would give if it did not solve again: the taped
    # solution with the new `SL_init` (`SL` depends on `SL_init` directly).
    # Measured 2.1e-2 away from the fresh march (relative), against 2e-16
    # for the replay, so the 1e-8 below can only be met by a replay that
    # solves the march again.
    with stop_annotating():
        h = control_direction(sub)
        stale = float(fd.assemble(
            (solver_sea_level + h - misfit_target(sub)) ** 2 * dss))
    J_replay = float(Jhat(shifted))
    relative = abs(J_replay - float(J_fresh)) / abs(float(J_fresh))
    stale_relative = abs(stale - float(J_fresh)) / abs(float(J_fresh))
    print(f"    [gate] replay {J_replay:.12e} fresh {float(J_fresh):.12e} "
          f"rel {relative:.3e}; without a re-solve rel {stale_relative:.3e}")
    # The two solves differ only by the iterative solve to OUTER_RTOL and the
    # Newton tolerance; 1e-8 leaves several orders above that.
    assert relative <= 1e-8
    assert stale_relative > 1e-3


def test_taylor_rate_with_initial_sea_level_control(meshes):
    """The guarded Taylor test of the misfit with respect to `SL_init` has rate 2."""
    _, sub = meshes
    Jhat, SL_init, J, solver = taped_reduced_functional(meshes)
    h = control_direction(sub)
    # The direct part of the derivative, `2 int (SL - target) h ds`: `SL`
    # contains `SL_init` itself, so this part involves no solve. It is about
    # 98 % of the derivative here; the rest goes through the adjoint of the
    # march. Computed at the taped state, before any replay.
    with stop_annotating():
        direct = float(fd.assemble(
            2 * (solver.sea_level() - misfit_target(sub)) * h
            * solver.sea_level_measure()(CURVE_RE)))
    ladder = taylor_first_order_ladder(Jhat, SL_init, h)
    print(f"    [gate] R0 {[f'{r:.3e}' for r in ladder['R0']]} "
          f"R1 {[f'{r:.3e}' for r in ladder['R1']]} "
          f"R1 rate {[f'{r:.3f}' for r in ladder['R1_rate']]}; dJ.h "
          f"{ladder['djdm']:.6e}, of which through the solve "
          f"{ladder['djdm'] - direct:.3e}")
    assert_taylor_with_guards(Jhat, SL_init, h, J, min_rate=1.90)
    # The Taylor test sees the adjoint of the march: with the direct part
    # alone in place of the gradient, the remainder is first order. The
    # functional is evaluated again at the steps of the ladder above.
    eps = ladder["eps"]
    with stop_annotating():
        J0 = float(Jhat(SL_init))
        values = [float(Jhat(SL_init._ad_add(h._ad_mul(e)))) for e in eps]
        Jhat(SL_init)
    R1_direct = [abs(v - J0 - e * direct) for v, e in zip(values, eps)]
    rates = [np.log2(R1_direct[i] / R1_direct[i + 1])
             for i in range(len(R1_direct) - 1)]
    print(f"    [gate] R1 rates with the direct part only "
          f"{[f'{r:.3f}' for r in rates]}")
    assert max(rates) < 1.2


@pytest.mark.xfail(raises=fd.ConvergenceError, strict=True,
                   reason="step 3 of the stiff elements ends in "
                   "DIVERGED_LINEAR_SOLVE (outer FGMRES at its cap of 200) "
                   "after the l2 line search lets the residual grow to "
                   "about 2; dt / tau_1 = 5 "
                   "before the power law")
def test_newton_stalls_on_a_stiff_power_law(meshes):
    """A record of the limit of Newton: the stiff elements fail at step 3.

    With `STIFF_MAXWELL_TIMES` the fastest element has `dt / tau = 5`, and the
    power law lowers its Maxwell time by a further factor of up to about 100
    in the ocean cells of the second layer. Newton converges at steps 1 and
    2 (4 iterations each) and fails at step 3.

    With the `l2` line search of the preset (this test) the residual norm
    goes 7.1e-2, 4.9e-3, 4.1e-3, then grows to about 2, and at Newton
    iteration 7 the outer FGMRES reaches its cap of 200 iterations
    (`DIVERGED_LINEAR_SOLVE`). With `snes_linesearch_type bt` and
    `gia_rebuild always` (a new preconditioner at every linear solve) every
    outer solve converges to its tolerance in 18-47 iterations, and the
    residual norm stops at 4.1853e-3 after 29 iterations
    (`DIVERGED_LINE_SEARCH`). At that state the line search reports an
    initial slope of -1.7517e-5, which is -|F|^2 to four digits, so the
    linear solve is accurate. The step has norm 2.1e6 against a residual of
    4.2e-3, so the Jacobian is numerically singular along the step (singular
    value at most 2e-9), and |F| grows faster than quadratically along it.
    The remaining residual is in the internal-variable rows. The limit is a
    turning point of the backward Euler step of the stiff element at this
    dt, not the preconditioner: the step may have no nearby solution.
    Each of these converges: the fixed coastline, half the time step,
    Maxwell times ten times longer (`MAXWELL_TIMES`).

    In production the outer solve counts a capped solve as converged
    (`outer_accept_capped=True`), so there this regime shows as Newton
    reaching its iteration limit or diverging, not as the exception of this
    test. The production runs fix their steps before the run (decision
    S14), so a failure of this kind stops the run.
    """
    _, sub = meshes
    with stop_annotating():
        SL_init, _, _ = shoreline_state(sub)
        march(meshes, SL_init, maxwell_times=STIFF_MAXWELL_TIMES)
