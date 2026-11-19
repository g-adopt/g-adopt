"""First-order (Taylor) adjoint test for the coupled `SelfGravitatingGIASolver`.

`GravitySolver` (gravity-alone) has a verified first-order adjoint on both its
representations -- `tests/unit/test_gravity_adjoint.py`. The **coupled**
self-gravitating solver has none, and until this file no test referenced its
adjoint at all. This is road map §9.6 / handover S1.6 step 1: the one capability
that is a genuine *absence* rather than an optimisation, and the measurement that
decides whether the coupled tape is a free asset (worth protecting) or a broken
one (worth fixing) before any low-rank work is costed.

Laptop-safe by construction: 2-D coarse annulus (dr 0.2, nazim 32), a direct LU
solve, one solve. No 3-D, no cluster.

Design, mirroring `test_gravity_adjoint.py`:

  * **The control is a Real-space `Function`, never a `Constant` or a float.**
    `ensure_constant` (`gadopt/utility.py`) wraps a float as a `Constant`, and a
    `Constant` control is silently non-differentiable in this Firedrake (see the
    gravity-alone findings); a `Function` is passed through untouched and threads
    into the residual as a differentiable coefficient.
  * **The functional reads the solution through UFL `split` of the mixed
    solution**, i.e. a reference the tape owns -- never `float(...)` or
    `.coefficients()`, which detach (S1.6).
  * **Only `solve()` and the objective assemble are taped.** The solver is
    constructed under `stop_annotating()` so the geometry assembles in
    `__init__` never clutter the tape.
  * **Matfree-safe: first-order ladder only, no Hessian leg.**

`nullspace=None` is deliberate for the primary test: the kernel projection
inside `solve` is then a no-op and the tape holds the solve alone. With
`nullspace="rigid_rotation"`, `solve` records the projection as a
`KernelProjectionBlock` after the solve (`gadopt.nullspaces.project_out_kernel`).
The nullspace-on case has its own test here, and the projection block itself
is tested in `test_gia_nullspace_projection_adjoint.py`.

The solver is `gadopt.gia_solver.SelfGravitatingGIASolver`. The prescribed
surface load is one `SurfaceLoad` coupling: it gives the traction
`B_mu g sigma` to the momentum equation and the mass sheet `sigma` to the
potential row.
"""

import sys
from pathlib import Path

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

from gadopt.gia_couplings import CentreOfMass, FluidCore, SurfaceLoad
from gadopt.gia_solver import SelfGravitatingGIASolver

# Siblings on the test path (pytest puts tests/unit on sys.path): the guarded
# Taylor driver and matfree-safe ladder, and the shared annulus constants,
# boundary conditions of the potential and the approximation factory.
from test_gravity_adjoint import (
    assert_taylor_with_guards,
    taylor_first_order_ladder,
)
from test_gia_gravity import (
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    RC,
    RE,
    SIGMA_HAT,
    approximation,
    gravity_bcs,
)


# ---------------------------------------------------------------------------
# Fixtures: coarse 2-D annulus (annotation-free, cached) and tape hygiene
# ---------------------------------------------------------------------------
DR_MANTLE = 0.2
N_AZIMUTHAL = 32
CELL_MANTLE = 101


@pytest.fixture(autouse=True)
def clean_tape():
    """Fresh tape per test; annotation guaranteed off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


@pytest.fixture(scope="module")
def meshes():
    """The parent annulus and its mantle submesh, both P2-curved.

    Duplicated from `test_gia_gravity.meshes` (a fixture cannot be shared across
    modules) but reads the same cached `.msh`: same dr/nazim, so the file is
    generated at most once across the whole suite.
    """
    pytest.importorskip("gmsh")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "demos" / "gravity"))
    import generate_selfgrav_annulus as gen
    from validate_selfgrav_annulus import curve_mesh

    path = Path("/tmp") / f"gadopt_selfgrav_gia_{DR_MANTLE}_{N_AZIMUTHAL}.msh"
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        gen.generate(str(path), dr_mantle=DR_MANTLE, n_azimuthal=N_AZIMUTHAL)
    fd.COMM_WORLD.barrier()

    parent = curve_mesh(fd.Mesh(str(path)))
    parent.cartesian = False
    sub = curve_mesh(fd.Submesh(parent, 2, CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


# ---------------------------------------------------------------------------
# The coupled solvers
# ---------------------------------------------------------------------------
def build(meshes, *, declare_nullspace, approximation_kwargs=None,
          solver_parameters="direct", solver_parameters_extra=None):
    """The annulus solver of `test_gia_gravity`, without rotation.

    A rigid core (`un = 0` at the CMB), the degree-2 load
    `sigma = SIGMA_HAT cos(2 theta)` on the surface weighed with `g = 1`, the
    low-rank DtN representation at truncation 3, DG history and `dt = 1`.

    Args:
      meshes: the `(parent, sub)` pair of the annulus fixture.
      declare_nullspace: whether to declare `nullspace="rigid_rotation"`.
      approximation_kwargs: approximation settings on top of the defaults of
        `test_gia_gravity.approximation`; the control goes in here.
      solver_parameters: the preset name or a complete dictionary.
      solver_parameters_extra: extra PETSc options, added last.

    Returns:
      `(solver, u, psi)`.
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    Xm = fd.SpatialCoordinate(sub)
    load = SurfaceLoad(
        boundary=CURVE_RE, g_surface=1.0,
        sigma=SIGMA_HAT * fd.cos(2 * fd.atan2(Xm[1], Xm[0])))
    solver = SelfGravitatingGIASolver(
        u, psi, approximation(**(approximation_kwargs or {})),
        dt=fd.Constant(1.0), gravity_bcs=gravity_bcs(parent, sheet=False),
        bcs={CURVE_RC: {"un": 0.0}}, couplings=[load],
        dtn_representation="lowrank",
        nullspace="rigid_rotation" if declare_nullspace else None,
        solver_parameters=solver_parameters,
        solver_parameters_extra=solver_parameters_extra)
    return solver, u, psi


#: The core density of the centre-of-mass configuration.
RHO_CORE = 2.0


def gravity_of_r(r):
    """`g(r)` for a uniform core of `RHO_CORE` and a unit-density mantle.

    The 2-D Gauss law of the test's own density, `g(r) = Lambda M(<r) / (2
    pi r)`. With it a rigid translation costs no energy, which is the
    configuration the centre-of-mass frame exists for.
    """
    mass = RHO_CORE * np.pi * RC ** 2 + np.pi * (r ** 2 - RC ** 2)
    return LAMBDA * mass / (2 * np.pi * r)


def build_frame(meshes, *, approximation_kwargs=None, **solver_kwargs):
    """The centre-of-mass configuration of `test_gia_gravity`, degree-1 load.

    A fluid core, the degree-1 load `SIGMA_HAT cos(theta)` weighed with the
    surface gravity of the reference state (so that the load exerts no net
    force), the centre-of-mass rows and the multiplier DtN representation.

    Args:
      meshes: the `(parent, sub)` pair of the annulus fixture.
      approximation_kwargs: approximation settings on top of the reference
        gravity of the frame.
      solver_kwargs: passed to `SelfGravitatingGIASolver`.

    Returns:
      `(solver, u, psi)`.
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    Xm = fd.SpatialCoordinate(sub)
    rm = fd.sqrt(fd.dot(Xm, Xm))
    couplings = [
        FluidCore(boundary=CURVE_RC, rho_core=RHO_CORE,
                  g=fd.Constant(gravity_of_r(RC))),
        CentreOfMass(),
        SurfaceLoad(boundary=CURVE_RE, g_surface=gravity_of_r(RE),
                    sigma=SIGMA_HAT * fd.cos(fd.atan2(Xm[1], Xm[0])))]
    solver = SelfGravitatingGIASolver(
        u, psi, approximation(g=gravity_of_r(rm), **(approximation_kwargs or {})),
        dt=fd.Constant(1.0), gravity_bcs=gravity_bcs(parent, sheet=False),
        couplings=couplings, dtn_representation="multiplier", **solver_kwargs)
    return solver, u, psi


# ---------------------------------------------------------------------------
# Forward map and the Taylor driver
# ---------------------------------------------------------------------------
def _forward(control, meshes, *, declare_nullspace, approximation_kwargs=None,
             solver_parameters_extra=None):
    """Coupled solve with `shear_modulus = control`; J = ||u||^2 over the mantle.

    `shear_modulus` is the strongest single control on the coupled operator
    (measured 0.11 relative change in `check_control_independence.py`), so the
    directional derivative is large and the Taylor signal clean -- which is what
    a first *existence* test wants. It also governs the elastic response
    directly, so J moves strongly with it.

    Args:
      control: the Real-space `Function` that becomes `shear_modulus`.
      meshes: the `(parent, sub)` pair of the annulus fixture.
      declare_nullspace: whether to declare `nullspace="rigid_rotation"`.
      approximation_kwargs: extra approximation settings, merged on top of
        `shear_modulus`. A power-law test passes `exponent` and
        `transition_stress` here.
      solver_parameters_extra: extra PETSc options, forwarded to the solver. A
        nonlinear forward map needs its Newton tolerances tightened here, so
        that the solve is converged far below the Taylor remainder at the
        smallest step and the finite differences measure the model instead of
        the solver.
    """
    _, sub = meshes
    with stop_annotating():
        solver, _, _ = build(
            meshes,
            declare_nullspace=declare_nullspace,
            approximation_kwargs={"shear_modulus": control,
                                  **(approximation_kwargs or {})},
            solver_parameters="direct",
            solver_parameters_extra=solver_parameters_extra,
        )
    solver.solve()  # the taped variational solve
    u = solver.field("displacement")
    return fd.assemble(fd.inner(u, u) * fd.dx(domain=sub))


def _assert_replay_matches_fresh_solve(Jhat, meshes, *, declare_nullspace,
                                       approximation_kwargs=None,
                                       solver_parameters_extra=None,
                                       factor=1.05):
    """Tape replay at a shifted control equals a fresh build-and-solve there.

    The one false-green class the four Taylor guards do NOT exclude: a control
    value baked into a coefficient at `__init__` time. That would make the tape
    self-consistent -- rate 2, nonzero dJdm, negative control would all still
    pass -- while the gradient was wrong for the *true* forward map, because the
    tape would carry a frozen control the real solve does not. Replaying the
    tape at a control 5% away and comparing against a fresh, untaped
    build-and-solve at the same value catches exactly that: if anything were
    baked, the two would diverge. Mirrors `test_gravity_adjoint.assert_replay_b2`.
    """
    _, sub = meshes
    R = fd.FunctionSpace(sub, "R", 0)
    m2 = fd.Function(R).assign(factor)
    with stop_annotating():
        J_direct = float(_forward(
            m2, meshes, declare_nullspace=declare_nullspace,
            approximation_kwargs=approximation_kwargs,
            solver_parameters_extra=solver_parameters_extra))
    J_replay = float(Jhat(m2))
    assert abs(J_replay - J_direct) <= 1e-9 * abs(J_direct), (
        f"tape replay {J_replay:.12e} != fresh solve {J_direct:.12e} at "
        f"control {factor} -- a control value is baked into the tape")
    print(f"    [replay-b2] fresh={J_direct:.10e}  replay={J_replay:.10e}"
          f"  rel={abs(J_replay - J_direct) / abs(J_direct):.2e}")
    return J_direct, J_replay


def _reduced_functional(meshes, *, declare_nullspace, approximation_kwargs=None,
                        solver_parameters_extra=None):
    """Tape the forward once and wrap it, with the Real-space control and h.

    Args:
      meshes: the `(parent, sub)` pair of the annulus fixture.
      declare_nullspace: whether to declare `nullspace="rigid_rotation"`.
      approximation_kwargs: extra approximation settings for `_forward`.
      solver_parameters_extra: extra PETSc options for `_forward`.

    Returns:
      `(Jhat, control, h, J)`: the reduced functional, the control, the
      direction and the taped value.
    """
    _, sub = meshes
    R = fd.FunctionSpace(sub, "R", 0)
    control = fd.Function(R).assign(1.0)  # created annotation-free
    h = fd.Function(R).assign(1.0)        # Real space: one dof, direction is +1

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        m = Control(control)
        J = _forward(control, meshes, declare_nullspace=declare_nullspace,
                     approximation_kwargs=approximation_kwargs,
                     solver_parameters_extra=solver_parameters_extra)
        Jhat = ReducedFunctional(J, m)
    finally:
        pause_annotation()
    return Jhat, control, h, J


def test_coupled_adjoint_is_second_order(meshes):
    """The coupled adjoint exists: Taylor rate 2 with the gradient, 1 without.

    This is the deliverable. A passing rate here means pyadjoint tapes the whole
    coupled solve -- displacement, internal variable, potential, DtN multipliers
    -- and its gradient is exact, so the coupled adjoint is a free asset and the
    S1.6 sequence proceeds (mode-row Schur PC next, low-rank operator only if it
    ever buys more than the free adjoint costs).
    """
    Jhat, control, h, J = _reduced_functional(meshes, declare_nullspace=False)

    ladder = taylor_first_order_ladder(Jhat, control, h)
    print(f"\n    [ladder] R0_rate={[f'{r:.3f}' for r in ladder['R0_rate']]}"
          f"  R1_rate={[f'{r:.3f}' for r in ladder['R1_rate']]}")

    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)

    # Close the one false-green class the guards above cannot: a control baked
    # into a coefficient at construction time (the tape would stay
    # self-consistent while the gradient was wrong for the true forward map).
    _assert_replay_matches_fresh_solve(Jhat, meshes, declare_nullspace=False)


def test_coupled_adjoint_with_nullspace_declared(meshes):
    """The declared-nullspace production path also tapes to second order.

    Declaring `nullspace="rigid_rotation"` makes `solve()` project the
    rigid rotation out of the solution and record the projection as a
    `KernelProjectionBlock` after the taped solve.

    The rigid rotation this configuration projects out is a genuine kernel, so
    the solve leaves only about 1e-13 of it in `u`, and this functional cannot
    tell a recorded projection from a missing one. What this test pins is that
    the production path with a nullspace declared is differentiable to the
    same tolerance as without one. The projection is tested where it removes a
    large component, in `test_gia_nullspace_projection_adjoint.py`.
    """
    Jhat, control, h, J = _reduced_functional(meshes, declare_nullspace=True)
    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)


def test_coupled_adjoint_is_second_order_with_a_power_law(meshes):
    """The coupled tape carries the power-law tangent, not just the Newtonian one.

    What a passing rate 2 proves here. With `exponent = 3` the Maxwell times of
    the history equation carry `power_law_factor`, a nonlinear function of the
    deviatoric stress and hence of the internal variables, and the forward solve
    is Newton rather than one linear solve. pyadjoint differentiates the
    residual symbolically, so it tapes that factor's tangent like any other
    coefficient, and the adjoint solve uses the Jacobian at the **converged**
    state, which is the linearisation the gradient needs. Neither fact is
    obvious from the Newtonian test above: that one has a linear residual, where
    the Jacobian at any state is the Jacobian at the solution.

    This answers the second open question of the parent plan (`PLAN.md`
    section 3): a power law on the uncondensed layout keeps the coupled adjoint,
    so the gradient work costed against the Newtonian solver carries over.

    On tolerances. The string `"direct"` preset inherits `snes_atol 1e-10` from
    `newton_stokes_solver_parameters`, and the residual norm of this
    configuration is about 2.3e-3, so `2.3e-3 * snes_rtol` at 1e-10 is below
    `snes_atol` and `atol` is the tolerance that stops Newton. Tightening
    `snes_rtol` alone therefore changes nothing; both are set. Measured on
    2026-09-14 by the plan reviewer: with `snes_rtol 1e-10` alone, Newton takes
    3 iterations to a final norm of 1.35e-11 and the R1 rates are 1.993, 1.996,
    1.998 with the replay identical to a fresh solve to the bit; adding
    `snes_atol 1e-14` gives 4 iterations, a final norm of 1e-17 and the same
    rates. That ladder is the expected result of this test.

    `declare_nullspace=False`, as in the primary Newtonian test: the
    projection after the solve has its own tests (the nullspace-on test above
    and `test_gia_nullspace_projection_adjoint.py`).
    """
    approximation_kwargs = {"exponent": 3.0, "transition_stress": 1e-3}
    solver_parameters_extra = {"snes_rtol": 1e-10, "snes_atol": 1e-14}
    Jhat, control, h, J = _reduced_functional(
        meshes, declare_nullspace=False,
        approximation_kwargs=approximation_kwargs,
        solver_parameters_extra=solver_parameters_extra)

    ladder = taylor_first_order_ladder(Jhat, control, h)
    print(f"\n    [ladder, power law] "
          f"R0_rate={[f'{r:.3f}' for r in ladder['R0_rate']]}"
          f"  R1_rate={[f'{r:.3f}' for r in ladder['R1_rate']]}")

    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)

    # The same false-green class the Newtonian test closes: a control value
    # baked into a coefficient at construction time leaves the tape
    # self-consistent while the gradient is wrong for the true forward map.
    _assert_replay_matches_fresh_solve(
        Jhat, meshes, declare_nullspace=False,
        approximation_kwargs=approximation_kwargs,
        solver_parameters_extra=solver_parameters_extra)


def test_coupled_adjoint_through_the_centre_of_mass_rows(meshes):
    """Taylor rate 2 through the centre-of-mass multipliers, and a clean replay.

    Degree-1 load, fluid core, the reference gravity of the test's own density,
    and a functional that reads the degree-1 surface displacement and the
    multiplier itself, so that both the constraint row and its column carry
    the sensitivity. The rows are UFL on the one taped solve, so no
    hand-written adjoint is involved; this pins that it stays that way.
    """
    parent, sub = meshes
    extra = {"dtn_fieldsplit_1_pc_type": "python",
             "dtn_fieldsplit_1_pc_python_type":
                 "gadopt.DtNMultiplierDenseSchurPC",
             "ksp_rtol": 1e-12}

    def forward(control):
        with stop_annotating():
            solver, _, _ = build_frame(
                meshes, approximation_kwargs={"shear_modulus": control},
                solver_parameters="direct", solver_parameters_extra=extra)
        solver.solve()
        u = solver.field("displacement")
        lam = solver.field("centre_of_mass_0")
        X = fd.SpatialCoordinate(sub)
        rhat = X / fd.sqrt(fd.dot(X, X))
        ds_re = fd.Measure("ds", domain=sub)(CURVE_RE)
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        return fd.assemble(
            1e3 * fd.dot(u, rhat) * fd.cos(fd.atan2(X[1], X[0])) * ds_re
            + 1e6 * fd.inner(u, u) * dx_m
            + 1e8 * lam * lam * dx_m)

    R = fd.FunctionSpace(sub, "R", 0)
    control = fd.Function(R).assign(1.0)
    h = fd.Function(R).assign(1.0)
    continue_annotation()
    try:
        m = Control(control)
        J = forward(control)
        Jhat = ReducedFunctional(J, m)
    finally:
        pause_annotation()
    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)

    shifted = fd.Function(R).assign(1.05)
    with stop_annotating():
        J_direct = float(forward(shifted))
    J_replay = float(Jhat(shifted))
    assert abs(J_replay - J_direct) <= 1e-9 * abs(J_direct)
