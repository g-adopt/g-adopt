"""Adjointability tests for gadopt.gravity_solver under Firedrake/pyadjoint.

These are discrete-consistency tests of the taped forward map built by
``GravitySolver``: Taylor tests on whole derivatives, component-level probes
of the Real-space multiplier coupling and the ``subfunctions``/``assign``
extraction chain, TLM-adjoint dot-product identities, replay correctness, and
two physics-pinned gradient cross-checks. Mesh accuracy is irrelevant for the
consistency tests, so every configuration is a brutally coarse utility mesh
(the same annulus/cubed-sphere/Submesh constructions as
``test_gravity_solver``); only the two closed-form gradient checks (T16/T17)
use unit-test resolution, and T17 lives in the long-test lane.

Annotation state is global and must be managed explicitly. Nothing at module
scope flips it on: ``import firedrake.adjoint`` does not annotate, and
``gadopt.inverse`` (which annotates at import) is deliberately never imported
here. The ``clean_tape`` autouse fixture guarantees a fresh tape per test and
annotation off afterwards. Every solver is constructed under
``stop_annotating()`` so the geometry assembles in ``__init__`` never clutter
the tape; only ``solve()`` and the objective assemble are taped.
"""

from collections import Counter
from contextlib import nullcontext

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
    taylor_test,
)
from pyadjoint import AdjFloat
from pyadjoint.tape import annotate_tape

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.spherical_harmonics import real_spherical_harmonic
from test_gravity_solver import RMAX, annulus_mesh, shell_mesh_3d


# ---------------------------------------------------------------------------
# Fixtures: meshes (annotation-free, cached) and per-test tape hygiene
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_tape():
    """Fresh tape per test; annotation guaranteed off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


@pytest.fixture(scope="session")
def mesh_2d_lu():
    return annulus_mesh(n_azimuthal=64, dr=0.25)


@pytest.fixture(scope="session")
def mesh_2d_mm():
    return annulus_mesh(n_azimuthal=96, dr=0.2)


@pytest.fixture(scope="session")
def mesh_3d_mm():
    return shell_mesh_3d(refinement_level=1, dr=0.5)


# ---------------------------------------------------------------------------
# Control / perturbation builders (created by the caller under stop_annotating)
# ---------------------------------------------------------------------------
def _polar(mesh):
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    phi = fd.atan2(X[1], X[0])
    return X, r, phi


def rho_control_2d(mesh):
    """cos(2 phi) * bump(r): azimuthally pure => zero net mass to machine eps."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    _, r, phi = _polar(mesh)
    bump = fd.exp(-(((r - 1.7) / 0.15) ** 2))
    return fd.Function(Q).interpolate(fd.cos(2 * phi) * bump)


def rho_perturb_2d(mesh):
    """Non-parallel to the control, contains the active mode (cos 2 phi)."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    _, r, phi = _polar(mesh)
    bump2 = fd.exp(-(((r - 1.6) / 0.2) ** 2))
    return fd.Function(Q).interpolate(
        (0.7 * fd.cos(2 * phi) + 0.3 * fd.cos(3 * phi)) * bump2)


def rho_control_3d(mesh):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    X, r, _ = _polar(mesh)
    bump = fd.exp(-(((r - 1.7) / 0.15) ** 2))
    Y11 = real_spherical_harmonic(1, 1, X)
    return fd.Function(Q).interpolate(Y11 * bump)


def rho_perturb_3d(mesh):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    X, r, _ = _polar(mesh)
    bump2 = fd.exp(-(((r - 1.6) / 0.2) ** 2))
    Y11 = real_spherical_harmonic(1, 1, X)
    Y10 = real_spherical_harmonic(1, 0, X)
    return fd.Function(Q).interpolate((0.7 * Y11 + 0.3 * Y10) * bump2)


# ---------------------------------------------------------------------------
# Forward maps (leg a of the triad). Each builds a FRESH solver under
# stop_annotating(), then solves and returns (J, solver) in the ambient
# annotation state chosen by the caller.
# ---------------------------------------------------------------------------
def forward_2d_lu(m_rho, bc_variant):
    """No-multiplier 2-D path: Dirichlet Poisson, or a single M=0 Robin term."""
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        if bc_variant == "dirichlet2d":
            bcs = {"top": {"psi": 0.0}}
        elif bc_variant == "lu2d":
            bcs = {"top": {"dtn": CylindricalDtN(M=0)}}
        else:
            raise ValueError(bc_variant)
        solver = GravitySolver(psi, m_rho, bcs=bcs)
    solver.solve()
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


def solve_2d_mm(m_rho, gravitational_constant=1.0):
    """Build a FRESH CFG-2D-MM solver under stop_annotating() and solve it in
    the ambient annotation state; return the solver so the caller can form an
    arbitrary objective (J-psi, trace-coefficient, ...)."""
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}},
            gravitational_constant=gravitational_constant)
    solver.solve()
    return solver


def forward_2d_mm(m_rho, gravitational_constant=1.0):
    """CFG-2D-MM: top+bottom CylindricalDtN(M=2), 9 multipliers, fieldsplit."""
    solver = solve_2d_mm(m_rho, gravitational_constant)
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


def trace_coeff_integral(solver, bc_id, m_mode):
    """TAPED-INTEGRAL trace coefficient (design 1.2 form b):
    c = int(psi_m * cos(m phi) ds) / (pi R), reading psi_m off the mixed
    subfunction so the boundary quadrature is the solver's own."""
    _, R = solver.boundary_geometry[bc_id]
    phi = fd.atan2(solver.X[1], solver.X[0])
    e_k = fd.cos(m_mode * phi)
    psi_m = solver.mixed_solution.subfunctions[0]
    return fd.assemble(psi_m * e_k * solver.ds(bc_id)) / (np.pi * R)


def trace_coeff_rdirect(solver, bc_id, key):
    """R-DIRECT trace coefficient (design 1.2 form c): read the multiplier
    subfunction c_R and integrate, c = int(c_R dx) / |Omega| (c_R is a global
    Real constant, so the integral is exactly c |Omega|)."""
    i = solver._multiplier_offset + solver._multiplier_keys.index((bc_id, key))
    c_R = solver.mixed_solution.subfunctions[i]
    with stop_annotating():
        vol = float(fd.assemble(1.0 * solver.dx))
    return fd.assemble(c_R * solver.dx) / vol


def forward_3d_mm(m_rho, gravitational_constant=1.0):
    """CFG-3D-MM: top+bottom SphericalDtN(L=1), 8 multipliers, fieldsplit."""
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": SphericalDtN(L=1)},
                 "bottom": {"dtn": SphericalDtN(L=1)}},
            gravitational_constant=gravitational_constant)
    solver.solve()
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


def forward_G(G_const, geometry, rho_fixed):
    """Forward with the gravitational constant as the control (rho fixed).
    G is passed as a Constant so it stays the same object ensure_constant sees,
    keeping Control(G) reachable on the tape."""
    if geometry == "2d":
        return forward_2d_mm(rho_fixed, G_const)
    return forward_3d_mm(rho_fixed, G_const)


def rho_control2_2d(mesh):
    """A second, distinct zero-mean control for direct-solve replay checks."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    _, r, phi = _polar(mesh)
    bump = fd.exp(-(((r - 1.7) / 0.15) ** 2))
    bump2 = fd.exp(-(((r - 1.6) / 0.2) ** 2))
    return fd.Function(Q).interpolate(
        fd.cos(2 * phi) * bump + 0.5 * fd.cos(3 * phi) * bump2)


def rho_control2_3d(mesh):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    X, r, _ = _polar(mesh)
    bump = fd.exp(-(((r - 1.7) / 0.15) ** 2))
    bump2 = fd.exp(-(((r - 1.6) / 0.2) ** 2))
    Y11 = real_spherical_harmonic(1, 1, X)
    Y10 = real_spherical_harmonic(1, 0, X)
    return fd.Function(Q).interpolate(Y11 * bump + 0.5 * Y10 * bump2)


# ---------------------------------------------------------------------------
# Shared Taylor harness enforcing the design's §4 false-green guards
# ---------------------------------------------------------------------------
def assert_taylor_with_guards(rf, m, h, J_value, min_rate=1.90, eps0=0.01):
    """Run a guarded Taylor test and its embedded dJdm=0 negative control.

    Enforces every §4 anti-false-green guard:
      1. nonzero directional derivative |h.g| >= 1e-6 |J|,
      2. functional demonstrably responds: |J(m+eps0 h) - J(m)| >= 1e-8 |J|,
      3. Taylor rate >= min_rate WITH the gradient,
      4. negative control (dJdm=0) rate in [0.85, 1.15] - proving the same
         J,m,h would only reach first order without the gradient.
    All inner products go through h._ad_dot (Riesz consistency, §1.4/P5).

    Returns a dict of the observed numbers for logging.
    """
    absJ = abs(float(J_value))

    # Re-establish the replay state at the control point so the adjoint sweep
    # below is taken at m (prior b2 replays may have left it at m2).
    rf(m)

    # Guard 1: nonzero directional derivative through the Riesz map.
    g = rf.derivative()
    dJdm = h._ad_dot(g)
    assert abs(dJdm) >= 1e-6 * absJ, (
        f"directional derivative {dJdm:.3e} below floor {1e-6 * absJ:.3e}")

    # Guard 2: the functional genuinely moves along h at the largest epsilon.
    with stop_annotating():
        m_pert = m._ad_add(h._ad_mul(eps0))
    J0 = float(rf(m))
    Jp = float(rf(m_pert))
    rf(m)  # restore replay state to the taping point
    assert abs(Jp - J0) >= 1e-8 * absJ, (
        f"functional response {abs(Jp - J0):.3e} below floor {1e-8 * absJ:.3e}")

    # Guard 3: Taylor with the gradient.
    rate = taylor_test(rf, m, h)
    assert rate >= min_rate, f"Taylor rate {rate:.4f} < {min_rate}"

    # Guard 4: negative control - drop the gradient, expect first order only.
    rate0 = taylor_test(rf, m, h, dJdm=0.0)
    assert 0.85 <= rate0 <= 1.15, (
        f"negative-control rate {rate0:.4f} not in [0.85, 1.15]")

    print(f"    [guards] |dJdm|={abs(dJdm):.4e}  |dJdm|/|J|={abs(dJdm)/absJ:.3e}"
          f"  response={abs(Jp - J0):.4e}  rate={rate:.4f}  rate0={rate0:.4f}")
    return dict(dJdm=dJdm, rate=rate, rate0=rate0, J0=J0, Jp=Jp)


def taylor_first_order_ladder(rf, m, h):
    """R0/R1 Taylor ladders WITHOUT the Hessian (matfree-safe).

    Mirrors pyadjoint.taylor_test's residual definitions for orders 0 and 1
    only: R0 = |J(m+eps h) - J(m)| (expected rate 1) and R1 =
    |J(m+eps h) - J(m) - eps <g,h>| (expected rate 2). It never triggers the
    second-order adjoint, because the matfree Schur fieldsplit has no
    assembled diagonal for the Hessian action - taylor_to_dict's R2 leg
    crashes there (design gap G2). Same eps ladder as taylor_test.
    """
    with stop_annotating():
        Jm = float(rf(m))
        g = rf.derivative()
        djdm = h._ad_dot(g)
        epsilons = [0.01 / 2 ** i for i in range(4)]
        R0, R1 = [], []
        for eps in epsilons:
            mp = m._ad_add(h._ad_mul(eps))
            Jp = float(rf(mp))
            R0.append(abs(Jp - Jm))
            R1.append(abs(Jp - Jm - eps * djdm))
        rf(m)

    def rates(res):
        return [np.log(res[i] / res[i + 1]) / np.log(2)
                for i in range(len(res) - 1)]

    return dict(eps=epsilons, R0=R0, R1=R1,
                R0_rate=rates(R0), R1_rate=rates(R1), djdm=djdm)


def tape_forward(forward, m, *args):
    """Leg (b): tape a fresh forward run at control m; return (J, solver, rf)."""
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        J, solver = forward(m, *args)
    finally:
        pause_annotation()
    rf = ReducedFunctional(J, Control(m))
    return J, solver, rf


def assert_replay_b1(rf, m, J):
    """Leg (b1): re-evaluation at the taping point reproduces J."""
    val = float(rf(m))
    assert abs(val - float(J)) <= 1e-9 * abs(float(J)), (val, float(J))


def assert_replay_b2(rf, forward, m2, *args):
    """Leg (b2): replay at a new control equals a fresh direct solve there."""
    with stop_annotating():
        J_direct, _ = forward(m2, *args)
    val = float(rf(m2))
    assert abs(val - float(J_direct)) <= 1e-9 * abs(float(J_direct)), (
        val, float(J_direct))
    return float(J_direct)


# ---------------------------------------------------------------------------
# Tape-inspection helpers
# ---------------------------------------------------------------------------
def _block_counts(tape):
    return Counter(type(b).__name__ for b in tape.get_blocks())


def _reaches(tape, source_outputs, target_bv):
    """Forward reachability on the tape DAG: does data from ``source_outputs``
    (a block's output block_variables) transitively flow into ``target_bv``?

    This is the gradient-relevant notion of "dangling": the design's §1.4
    literal wording ("output is not a dependency of any later block") is
    empirically false for the check_net_mass assembles - their output IS
    consumed by the downstream abs()/comparison AdjFloat arithmetic - but that
    arithmetic subgraph is terminal and never reaches the objective J, so it
    cannot influence dJ/dm. Reachability-to-J is the property that actually
    matters and is strictly stronger than "feeds nothing at all".
    """
    reachable = {id(o) for o in source_outputs}
    blocks = tape.get_blocks()
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if any(id(d) in reachable for d in b.get_dependencies()):
                for o in b.get_outputs():
                    if id(o) not in reachable:
                        reachable.add(id(o))
                        changed = True
    return id(target_bv) in reachable


def _reaches_objective(tape, block, objective):
    return _reaches(tape, block.get_outputs(), objective.block_variable)


# ---------------------------------------------------------------------------
# T01 - Tape anatomy of one annotated solve  [P0, TAPE-INSPECT]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["clean", "dirty"])
def test_tape_structure(mesh_2d_mm, variant):
    """One annotated solve on CFG-2D-MM is exactly one NLVS solve block plus
    the assign/subfunctions extraction, preceded by two dangling AssembleBlocks
    from check_net_mass; the objective assemble that follows is not dangling.

    The clean variant constructs under stop_annotating(); the dirty variant
    constructs under annotation and must still yield one solve block with all
    the extra construction blocks dangling.
    """
    tape = get_working_tape()

    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)

    if variant == "dirty":
        continue_annotation()

    build_ctx = nullcontext() if variant == "dirty" else stop_annotating()
    with build_ctx:
        psi = fd.Function(fd.FunctionSpace(mesh_2d_mm, "CG", 1))
        solver = GravitySolver(
            psi, m,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}})

    if variant == "clean":
        continue_annotation()

    n_before_solve = len(tape.get_blocks())
    solver.solve()
    n_after_solve = len(tape.get_blocks())
    solve_blocks = tape.get_blocks()[n_before_solve:n_after_solve]

    # Objective added AFTER the solve: must reach the gradient chain.
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    pause_annotation()

    solve_counts = Counter(type(b).__name__ for b in solve_blocks)

    # Exactly one mixed solve block, whatever the multiplier count.
    assert solve_counts["NonlinearVariationalSolveBlock"] == 1

    # check_net_mass assembles rho mass and |rho| scale: two AssembleBlocks.
    # Neither reaches J (they feed only the terminal net-mass arithmetic), so
    # neither can perturb dJ/dm - the "dangling but harmless" property.
    solve_assembles = [
        b for b in solve_blocks if type(b).__name__ == "AssembleBlock"]
    assert len(solve_assembles) == 2
    for b in solve_assembles:
        assert not _reaches_objective(tape, b, J)

    # The solve block itself DOES reach J (control -> psi -> J is live).
    solve_block = next(
        b for b in solve_blocks
        if type(b).__name__ == "NonlinearVariationalSolveBlock")
    assert _reaches_objective(tape, solve_block, J)

    # solution.assign(subfunctions[0]) taped as SubfunctionBlock(s) + one
    # FunctionAssignBlock.
    assert solve_counts["SubfunctionBlock"] >= 1
    assert solve_counts["FunctionAssignBlock"] == 1

    # The objective assemble is on the live chain, not dangling.
    j_assembles = [
        b for b in tape.get_blocks()[n_after_solve:]
        if type(b).__name__ == "AssembleBlock"]
    assert len(j_assembles) == 1
    assert _reaches_objective(tape, j_assembles[0], J)
    # J genuinely depends on the solve (sanity: value is finite and positive).
    assert float(J) > 0.0

    if variant == "dirty":
        # Construction under annotation: none of the geometry/quadrature
        # assembles emitted before the solve reach J - they are pure clutter.
        pre_solve_blocks = tape.get_blocks()[:n_before_solve]
        pre_assembles = [
            b for b in pre_solve_blocks if type(b).__name__ == "AssembleBlock"]
        assert len(pre_assembles) > 0
        for b in pre_assembles:
            assert not _reaches_objective(tape, b, J)


# ---------------------------------------------------------------------------
# T02 - Taylor on rho, no-multiplier paths  [P0, TAYLOR]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bc_variant", ["dirichlet2d", "lu2d"])
def test_taylor_rho_no_multiplier(mesh_2d_lu, bc_variant):
    """rho -> psi -> J is fully taped on the two no-multiplier 2-D paths:
    a plain Dirichlet Poisson (source term alone, no DtN in the form) and a
    single M=0 Robin term (still aij + LU). Whole derivative dJ/drho."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_lu)
        h = rho_perturb_2d(mesh_2d_lu)
        m2 = rho_control2_2d(mesh_2d_lu)

    J, _, rf = tape_forward(forward_2d_lu, m, bc_variant)
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_2d_lu, m2, bc_variant)
    assert_taylor_with_guards(rf, m, h, J)


# ---------------------------------------------------------------------------
# T03 - Taylor on rho, multi-mode fieldsplit path  [P0, TAYLOR]
# ---------------------------------------------------------------------------
def test_taylor_rho_fieldsplit_2d(mesh_2d_mm):
    """CFG-2D-MM: 9 R multipliers (incl. interior mean), matfree Schur
    fieldsplit, adjoint LVS built from adjoint(dFdu). The design specified
    taylor_to_dict here for the R0/R1 ladders, but its R2/Hessian leg crashes
    on the matfree Schur fieldsplit (PETSc error 101, the getDiagonal wall;
    finding, gap G2), so this uses taylor_first_order_ladder instead - the same
    eps ladder, both R0 (~1, no gradient) and R1 (>=1.9, with gradient) on
    record, never touching the second-order adjoint. Plus the guarded dJdm=0
    control, and an assertion that the gradient is nonzero where the control
    bump is supported (guards a silently zero matfree transpose)."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)
        m2 = rho_control2_2d(mesh_2d_mm)

    J, _, rf = tape_forward(forward_2d_mm, m)
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_2d_mm, m2)

    # First-order R0/R1 ladders on record (matfree-safe; no Hessian leg).
    res = taylor_first_order_ladder(rf, m, h)
    print(f"    [T03] R0 rates={res['R0_rate']}  R1 rates={res['R1_rate']}")
    assert min(res["R0_rate"]) >= 0.9  # first order without the gradient
    assert min(res["R1_rate"]) >= 1.90  # with the gradient
    assert min(res["R1"]) >= 1e-15  # no machine-precision floor

    # Guarded Taylor + negative control on the same tape.
    assert_taylor_with_guards(rf, m, h, J)

    # Gradient supported where the control is: nonzero somewhere.
    g = rf.derivative()
    assert np.max(np.abs(g.dat.data_ro)) > 0.0


# ---------------------------------------------------------------------------
# T04 - CENTREPIECE: float() severs the tape; taped assemble restores it
# ---------------------------------------------------------------------------
def test_coefficient_float_severs_tape(mesh_2d_mm):
    """coefficients() reads float(subfunction), which is NOT annotated
    (firedrake Function.__float__ reads dat.data_ro directly). An objective
    built from that float is severed from the control: its adjoint gradient is
    EXACTLY zero and replay at any perturbed control returns the same constant.
    This is the suite's canonical negative control - it must FAIL a Taylor
    test, and does so with an exact-zero signature."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    solver = solve_2d_mm(m)
    # Build the broken objective while annotation is ON: the severing is caused
    # by float() inside coefficients(), not by pausing.
    coeffs = solver.coefficients()
    J_b = AdjFloat(coeffs["top"]["cos2"])
    pause_annotation()

    rf_b = ReducedFunctional(J_b, Control(m))

    # Exact-zero gradient: the tape is severed, so dJ/dm is identically 0.
    g = rf_b.derivative()
    assert np.all(g.dat.data_ro == 0.0), (
        f"severed objective should have EXACTLY zero gradient, got max "
        f"{np.max(np.abs(g.dat.data_ro)):.3e}")

    # Replay-constant: perturbing the control cannot move the severed output.
    J0 = float(rf_b(m))
    for eps in [0.01 / 2 ** i for i in range(4)]:
        with stop_annotating():
            m_pert = m._ad_add(h._ad_mul(eps))
        assert float(rf_b(m_pert)) == J0, (
            "severed replay must be exactly constant in the control")
    print(f"    [T04 broken] grad_norm={np.max(np.abs(g.dat.data_ro)):.1e} "
          f"(exact 0); replay constant at J={J0:.6e}")


def test_coefficient_taped_integral_taylor(mesh_2d_mm):
    """The SAME trace coefficient rebuilt as a taped boundary integral
    (form b, c = int(psi e_k ds)/norm) restores the tape: J = c^2 passes
    Taylor at rate 2, and its gradient agrees to 1e-8 with the R-DIRECT form
    (c, read off the multiplier subfunction) - the two objectives coincide on
    the solution manifold by the constraint row, so their gradients must too."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)
        m2 = rho_control2_2d(mesh_2d_mm)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    solver = solve_2d_mm(m)
    c_int = trace_coeff_integral(solver, "top", 2)
    c_dir = trace_coeff_rdirect(solver, "top", "cos2")
    J_f = c_int ** 2
    J_c = c_dir ** 2
    pause_annotation()

    rf_f = ReducedFunctional(J_f, Control(m))
    rf_c = ReducedFunctional(J_c, Control(m))

    # b1 replay for the fixed objective.
    assert_replay_b1(rf_f, m, J_f)

    # b2: replay at m2 equals a fresh direct solve at m2 (rebuild coefficient).
    with stop_annotating():
        solver2 = solve_2d_mm(m2)
        c2 = float(trace_coeff_integral(solver2, "top", 2))
        J_direct2 = c2 ** 2
    assert abs(float(rf_f(m2)) - J_direct2) <= 1e-9 * abs(J_direct2)
    rf_f(m)  # restore

    # The two forms carry the same value on the solution manifold.
    assert abs(float(J_f) - float(J_c)) <= 1e-8 * abs(float(J_f)), (
        float(J_f), float(J_c))

    # Guarded Taylor on the taped-integral objective.
    assert_taylor_with_guards(rf_f, m, h, J_f)

    # Cross-check: gradients of the two equivalent objectives agree.
    g_f = rf_f.derivative()
    g_c = rf_c.derivative()
    rel = np.max(np.abs(g_f.dat.data_ro - g_c.dat.data_ro)) / \
        np.max(np.abs(g_f.dat.data_ro))
    print(f"    [T04 fixed] J_f={float(J_f):.6e} J_c={float(J_c):.6e} "
          f"grad rel diff={rel:.3e}")
    assert rel <= 1e-8, f"integral vs R-direct gradient rel diff {rel:.3e}"


# ---------------------------------------------------------------------------
# T05 - Sensitivity through an R multiplier  [P0, TAYLOR]
# ---------------------------------------------------------------------------
def test_taylor_through_multiplier(mesh_2d_mm):
    """An objective read directly off a Real-space multiplier subfunction
    (J = c^2, form c) differentiates correctly: the SubfunctionBlock on an R
    field and the adjoint of the mixed solve with a cofunction RHS living only
    in the R components both work. Includes the modal-orthogonality hazard
    exhibit (a cos3 perturbation against the cos2 coefficient is blind)."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)
        # Pure off-mode perturbation for the hazard exhibit (zero net mass).
        Q = fd.FunctionSpace(mesh_2d_mm, "CG", 1)
        _, r, phi = _polar(mesh_2d_mm)
        bump = fd.exp(-(((r - 1.7) / 0.15) ** 2))
        h_orth = fd.Function(Q).interpolate(fd.cos(3 * phi) * bump)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    solver = solve_2d_mm(m)
    c_dir = trace_coeff_rdirect(solver, "top", "cos2")
    c_int = trace_coeff_integral(solver, "top", 2)
    J_c = c_dir ** 2
    J_i = c_int ** 2
    pause_annotation()

    rf = ReducedFunctional(J_c, Control(m))
    rf_i = ReducedFunctional(J_i, Control(m))

    assert_replay_b1(rf, m, J_c)

    # Guarded Taylor: sensitivity flows through the R multiplier.
    assert_taylor_with_guards(rf, m, h, J_c)

    # Agreement with the T04(b) taped-integral gradient (same manifold).
    g = rf.derivative()
    g_i = rf_i.derivative()
    rel = np.max(np.abs(g.dat.data_ro - g_i.dat.data_ro)) / \
        np.max(np.abs(g.dat.data_ro))
    assert rel <= 1e-8, f"R-direct vs integral gradient rel diff {rel:.3e}"

    # Hazard exhibit (NOT a pass criterion for the gradient): a pure off-mode
    # cos3 perturbation is blind to the cos2 coefficient by modal
    # orthogonality of the linear map, so <g, h_orth> ~ 0. This is WHY h must
    # contain the active mode; a Taylor test along h_orth would "pass" on 0~0.
    dJdm_active = h._ad_dot(g)
    dJdm_orth = h_orth._ad_dot(g)
    print(f"    [T05] rel(g vs integral)={rel:.3e}  <g,h>={dJdm_active:.4e}  "
          f"<g,h_orth>={dJdm_orth:.4e}  ratio={abs(dJdm_orth/dJdm_active):.3e}")
    assert abs(dJdm_orth) <= 1e-9 * abs(dJdm_active), (
        "off-mode perturbation should be near-blind to the cos2 coefficient")


# ---------------------------------------------------------------------------
# T06 - subfunctions[0] + assign preserve the tape  [P1]
# ---------------------------------------------------------------------------
def test_solution_assign_preserves_tape(mesh_2d_mm):
    """J on solver.solution (through SubfunctionBlock -> FunctionAssignBlock)
    yields a gradient identical to J on mixed_solution.subfunctions[0]
    directly, and passes Taylor. Negative control: a raw dat.data[:] copy of
    the transfer is un-annotated, severs the tape (zero gradient), and the
    equality assertion then fails - proving the test detects the broken
    transfer (same class of bug as T04's float())."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)
        V = fd.FunctionSpace(mesh_2d_mm, "CG", 1)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    solver = solve_2d_mm(m)
    J_sol = fd.assemble(solver.solution ** 2 * fd.dx)
    J_mix = fd.assemble(solver.mixed_solution.subfunctions[0] ** 2 * fd.dx)
    # Negative control: an un-annotated raw numpy copy of the same transfer.
    with stop_annotating():
        sol_raw = fd.Function(V)
        sol_raw.dat.data[:] = solver.mixed_solution.subfunctions[0].dat.data_ro
    J_raw = fd.assemble(sol_raw ** 2 * fd.dx)
    pause_annotation()

    rf_sol = ReducedFunctional(J_sol, Control(m))
    rf_mix = ReducedFunctional(J_mix, Control(m))
    rf_raw = ReducedFunctional(J_raw, Control(m))

    assert_replay_b1(rf_sol, m, J_sol)
    assert_replay_b1(rf_mix, m, J_mix)

    g_sol = rf_sol.derivative()
    g_mix = rf_mix.derivative()
    g_raw = rf_raw.derivative()

    # The two annotated routes give the same value and the same gradient.
    assert abs(float(J_sol) - float(J_mix)) <= 1e-14 * abs(float(J_mix))
    denom = np.max(np.abs(g_mix.dat.data_ro))
    rel = np.max(np.abs(g_sol.dat.data_ro - g_mix.dat.data_ro)) / denom
    print(f"    [T06] grad rel(sol vs mix)={rel:.3e}  "
          f"raw_grad_norm={np.max(np.abs(g_raw.dat.data_ro)):.1e}")
    assert rel <= 1e-14, f"assign vs subfunctions gradient rel diff {rel:.3e}"

    # Negative control: the raw-copy transfer is severed -> exact-zero gradient
    # (even though J_raw == J_mix in value), so the equality assertion fails.
    assert np.all(g_raw.dat.data_ro == 0.0)
    rel_raw = np.max(np.abs(g_raw.dat.data_ro - g_mix.dat.data_ro)) / denom
    assert rel_raw > 1e-3, "raw-copy negative control must NOT match"

    # Taylor on the solver.solution route.
    assert_taylor_with_guards(rf_sol, m, h, J_sol)


# ---------------------------------------------------------------------------
# T09 - TLM-adjoint dot-product identity  [P1, DOT]
# ---------------------------------------------------------------------------
def test_tlm_adjoint_dot_product(mesh_2d_lu, mesh_2d_mm):
    """Transpose consistency: the adjoint is the exact transpose of the
    tangent, <J^T g_hat, h> == <g_hat, J.h>, isolated from any finite-diff
    truncation.

    FINDING (recorded here, parallel to the Hessian-on-matfree gap): pyadjoint's
    native tangent sweep is broken on the DtN multiplier config. The
    NonlinearVariationalSolveBlock overrides _forward_solve / _adjoint_solve to
    use the cached fieldsplit solvers, but the tangent path
    (_assemble_and_solve_tlm_eq -> _assembled_solve) issues a bare
    firedrake.solve() with NO solver parameters, which cannot solve the R-space
    mixed system. So evaluate_tlm() raises on CFG-2D-MM.

    The identity is therefore verified two ways:
      (1) LU path (no multipliers, single field): native tape.evaluate_tlm()
          works and reproduces the adjoint dot product - proving the native TLM
          machinery itself is sound where solver-parameter inheritance is not an
          issue;
      (2) fieldsplit path (CFG-2D-MM, the DtN transpose we actually care about):
          the forward map rho->psi is exactly linear, so the tangent along h is
          psi(density=h) computed by an INDEPENDENT real solve; the adjoint dot
          product must equal 2 int(psi_m psi_h) dx exactly (both at rtol 1e-11).
    """
    # (1) Native TLM on the no-multiplier LU path.
    with stop_annotating():
        m_lu = rho_control_2d(mesh_2d_lu)
        h_lu = rho_perturb_2d(mesh_2d_lu)
    J_lu, _, rf_lu = tape_forward(forward_2d_lu, m_lu, "lu2d")
    dJdm_adj_lu = h_lu._ad_dot(rf_lu.derivative())
    m_lu.block_variable.tlm_value = h_lu
    get_working_tape().evaluate_tlm()
    dJdm_tlm_lu = float(J_lu.block_variable.tlm_value)
    rel_lu = abs(dJdm_tlm_lu - dJdm_adj_lu) / abs(dJdm_adj_lu)
    print(f"    [T09 LU native TLM] adj={dJdm_adj_lu:.10e} tlm={dJdm_tlm_lu:.10e}"
          f" rel={rel_lu:.3e}")
    assert abs(dJdm_adj_lu) >= 1e-12 and abs(dJdm_tlm_lu) >= 1e-12
    assert rel_lu <= 1e-8

    # (2) Exact independent tangent on the fieldsplit DtN path.
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)
    _, _, rf = tape_forward(forward_2d_mm, m)
    dJdm_adj = h._ad_dot(rf.derivative())
    with stop_annotating():
        psi_m = solve_2d_mm(m).solution.copy(deepcopy=True)
        psi_h = solve_2d_mm(h).solution.copy(deepcopy=True)  # linear tangent
        dJdm_tan = 2.0 * fd.assemble(psi_m * psi_h * fd.dx)
    rel = abs(dJdm_tan - dJdm_adj) / abs(dJdm_adj)
    print(f"    [T09 MM exact tangent] adj={dJdm_adj:.10e} tan={dJdm_tan:.10e}"
          f" rel={rel:.3e}")
    assert abs(dJdm_adj) >= 1e-12 and abs(dJdm_tan) >= 1e-12
    assert rel <= 1e-8, f"adjoint vs exact tangent rel diff {rel:.3e}"


def real_scalar(mesh, value):
    """A scalar Real-space (R0) Function - the adjoint-differentiable scalar
    control type (see the FINDING in test_gradient_G_constant)."""
    return fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(value)


# ---------------------------------------------------------------------------
# T08 - Taylor/FD/analytic on the scalar G control  [P1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geometry", ["2d", "3d"])
def test_gradient_G_constant(geometry, mesh_2d_mm, mesh_3d_mm):
    """G is a taped scalar control. Since psi is exactly linear in G,
    J = int psi^2 dx is exactly quadratic in G, so:
      ANALYTIC dJ/dG = 2 J / G  (exact),
      FD central difference is exact for a quadratic (rel <= 1e-9),
      TAYLOR rate >= 1.90 with the guarded harness + dJdm=0 control,
      sabotage (1.01 x dJdG) drops the rate <= 1.15.

    FINDING (important for adjoint use of the solver): a firedrake ``Constant``
    is NOT a differentiable adjoint control in this firedrake/pyadjoint
    version - its adjoint gradient comes out EXACTLY zero (verified with a
    minimal gravity-free assemble tape) and its replay does not update from a
    fresh object. The differentiable scalar control is a Real-space Function,
    FunctionSpace(mesh, "R", 0). The solver's ``ensure_constant`` wraps a float
    ``gravitational_constant`` into a Constant, so a float/Constant G is NOT
    adjointable; callers must pass a Real-space Function to control G. This
    test therefore uses an R-space Function; with it, gradient, replay, and
    taylor_test all work correctly."""
    mesh = mesh_2d_mm if geometry == "2d" else mesh_3d_mm
    with stop_annotating():
        rho_fixed = (rho_control_2d(mesh) if geometry == "2d"
                     else rho_control_3d(mesh))
        G = real_scalar(mesh, 1.3)
        h = real_scalar(mesh, 1.0)

    J, _, rf = tape_forward(forward_G, G, geometry, rho_fixed)
    J0, G0 = float(J), float(G)

    # Triad legs b1 and b2 (fresh direct solve at G2 = 1.7).
    assert_replay_b1(rf, G, J)
    assert_replay_b2(rf, forward_G, real_scalar(mesh, 1.7), geometry, rho_fixed)

    # Adjoint directional derivative at the taping point.
    rf(G)
    dJdG = h._ad_dot(rf.derivative())

    # ANALYTIC identity dJ/dG = 2 J / G (exact; psi linear in G).
    analytic = 2.0 * J0 / G0
    rel_ana = abs(dJdG - analytic) / abs(dJdG)
    assert rel_ana <= 1e-9, f"dJ/dG={dJdG:.10e} vs 2J/G={analytic:.10e}"

    # FD central difference (exact for a quadratic).
    eps = 1e-3
    Jp = float(rf(real_scalar(mesh, G0 + eps)))
    Jm = float(rf(real_scalar(mesh, G0 - eps)))
    rf(G)  # restore
    fd_val = (Jp - Jm) / (2 * eps)
    rel_fd = abs(fd_val - dJdG) / abs(dJdG)
    assert rel_fd <= 1e-9, f"FD={fd_val:.10e} vs adjoint={dJdG:.10e}"

    # Guarded Taylor + dJdm=0 negative control.
    stats = assert_taylor_with_guards(rf, G, h, J)

    # Sabotage: a 1% wrong gradient must collapse the Taylor rate.
    rate_sab = taylor_test(rf, G, h, dJdm=1.01 * dJdG)
    print(f"    [T08 {geometry}] J={J0:.6e} dJdG={dJdG:.6e} 2J/G={analytic:.6e} "
          f"rel_ana={rel_ana:.2e} FD_rel={rel_fd:.2e} "
          f"rate={stats['rate']:.4f} sabotage={rate_sab:.4f}")
    assert rate_sab <= 1.15, f"1%-wrong gradient not detected (rate {rate_sab})"


def test_firedrake_constant_is_not_differentiable_canary():
    """Regression canary for the T08 finding (gravity-free, fast).

    A firedrake Constant is currently NOT a differentiable adjoint control in
    this firedrake/pyadjoint version: its adjoint gradient is EXACTLY zero and
    its replay does not update from a fresh object. This test PINS that broken
    behavior on a minimal assemble tape. If it ever FAILS, firedrake has fixed
    Constant differentiability - at which point the solver's ensure_constant
    guidance and the T08 docstring (which steer scalar controls to a Real-space
    Function) should be revisited. A loud failure is deliberately preferred to
    a silent xfail that would flip to xpass unnoticed."""
    mesh = fd.UnitSquareMesh(4, 4)
    V = fd.FunctionSpace(mesh, "CG", 1)
    with stop_annotating():
        u = fd.Function(V).interpolate(fd.Constant(2.0))
        G = fd.Constant(1.3)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    J = fd.assemble((G * u * u) ** 2 * fd.dx)
    pause_annotation()

    rf = ReducedFunctional(J, Control(G))
    g = rf.derivative()

    # (a) The adjoint gradient is exactly zero (should be 2 J / G = 41.6).
    assert float(g) == 0.0, (
        f"firedrake Constant is now differentiable (grad={float(g)}); revisit "
        "ensure_constant guidance and the T08 R-space-Function workaround")
    # (b) Replay does not update from a fresh Constant object.
    assert float(rf(fd.Constant(1.7))) == float(J), (
        "firedrake Constant control now updates on replay; revisit T08")


# ---------------------------------------------------------------------------
# T07 - Taylor on the sigma surface-sheet control  [P1, TAYLOR]
# ---------------------------------------------------------------------------
def forward_sheet(sigma_fn, geometry):
    """rho == 0, so the ONLY source is the sheet term -4 pi G sigma v ds; this
    whole-derivative dJ/dsigma also isolates that term as a component."""
    mesh = sigma_fn.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        if geometry == "2d":
            bcs = {"top": {"dtn": CylindricalDtN(M=2), "sigma": sigma_fn},
                   "bottom": {"dtn": CylindricalDtN(M=2)}}
        else:
            bcs = {"top": {"dtn": SphericalDtN(L=1), "sigma": sigma_fn},
                   "bottom": {"dtn": SphericalDtN(L=1)}}
        solver = GravitySolver(psi, 0.0, bcs=bcs)
    solver.solve()
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


@pytest.mark.parametrize("geometry", ["2d", "3d"])
def test_taylor_sigma(geometry, mesh_2d_mm, mesh_3d_mm):
    """The sheet term -4 pi G sigma v ds is a taped dependency: with sigma a
    Function control (rho == 0), Taylor rate >= 1.90. 2-D sheet controls are
    zero-mean on the boundary (net-mass guard); 3-D has no such constraint."""
    mesh = mesh_2d_mm if geometry == "2d" else mesh_3d_mm
    with stop_annotating():
        V = fd.FunctionSpace(mesh, "CG", 1)
        if geometry == "2d":
            _, _, phi = _polar(mesh)
            sigma = fd.Function(V).interpolate(fd.cos(2 * phi))
            h = fd.Function(V).interpolate(
                0.8 * fd.cos(2 * phi) + 0.2 * fd.cos(phi))
            sigma2 = fd.Function(V).interpolate(
                fd.cos(2 * phi) + 0.5 * fd.cos(3 * phi))
        else:
            X, _, _ = _polar(mesh)
            Y11 = real_spherical_harmonic(1, 1, X)
            Y10 = real_spherical_harmonic(1, 0, X)
            sigma = fd.Function(V).interpolate(Y11)
            h = fd.Function(V).interpolate(0.8 * Y11 + 0.2 * Y10)
            sigma2 = fd.Function(V).interpolate(Y11 + 0.5 * Y10)

    J, _, rf = tape_forward(forward_sheet, sigma, geometry)
    assert_replay_b1(rf, sigma, J)
    assert_replay_b2(rf, forward_sheet, sigma2, geometry)
    print(f"    [T07 {geometry}] J={float(J):.6e}")
    assert_taylor_with_guards(rf, sigma, h, J)


# ---------------------------------------------------------------------------
# T10 - Replay correctness of ReducedFunctional  [P1]
# ---------------------------------------------------------------------------
def test_reduced_functional_replay(mesh_2d_mm):
    """rf(m2) (pure tape replay at a new control) equals a fresh forward solve
    at m2 through a newly constructed GravitySolver, and rf(m) returns the
    original J. Documents that replay bypasses the Python-level check_net_mass:
    a net-mass control replays without raising, though a direct solve raises."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        m2 = rho_control2_2d(mesh_2d_mm)

    J, _, rf = tape_forward(forward_2d_mm, m)

    # Replay at the taping point returns the original J.
    assert abs(float(rf(m)) - float(J)) <= 1e-9 * abs(float(J))

    # Replay at m2 equals a fresh direct solve at m2 (two independent solves).
    with stop_annotating():
        J_direct2, _ = forward_2d_mm(m2)
    val2 = float(rf(m2))
    rel = abs(val2 - float(J_direct2)) / abs(float(J_direct2))
    print(f"    [T10] rf(m2)={val2:.8e} direct={float(J_direct2):.8e} rel={rel:.2e}")
    assert rel <= 1e-9

    # Documented gap G3 / pitfall P2: replay skips check_net_mass entirely.
    # A net-mass control iterate replays fine, but a direct solve would raise.
    with stop_annotating():
        Q = fd.FunctionSpace(mesh_2d_mm, "CG", 1)
        _, r, _ = _polar(mesh_2d_mm)
        m_mass = fd.Function(Q).interpolate(fd.exp(-(((r - 1.7) / 0.15) ** 2)))
    val_mass = float(rf(m_mass))  # must NOT raise
    assert np.isfinite(val_mass)
    with pytest.raises(NotImplementedError, match="Net mass"):
        with stop_annotating():
            forward_2d_mm(m_mass)


# ---------------------------------------------------------------------------
# T11 - check_net_mass hygiene: dangling but harmless  [P1]
# ---------------------------------------------------------------------------
class _QuietNetMassSolver(GravitySolver):
    """CFG-2D-MM solver whose check_net_mass runs under stop_annotating(), so
    its two 0-form assembles never touch the tape (variant B)."""
    def check_net_mass(self):
        with stop_annotating():
            super().check_net_mass()


def _forward_2d_mm_quiet(m_rho):
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = _QuietNetMassSolver(
            psi, m_rho,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}})
    solver.solve()
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


def test_net_mass_blocks_dangling_harmless(mesh_2d_mm):
    """The two check_net_mass 0-form assembles land on the tape (variant A) but
    do not alter J or the gradient; wrapping check_net_mass in stop_annotating
    (variant B) removes exactly those two AssembleBlocks and changes neither J
    nor the gradient."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)

    # Variant A (stock): construction under stop_annotating, so the only
    # AssembleBlocks are the 2 net-mass ones plus the 1 objective.
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    J_a, _ = forward_2d_mm(m)
    pause_annotation()
    n_assemble_a = _block_counts(tape)["AssembleBlock"]
    g_a = ReducedFunctional(J_a, Control(m)).derivative()

    # Variant B (quiet): check_net_mass under stop_annotating.
    tape.clear_tape()
    continue_annotation()
    J_b, _ = _forward_2d_mm_quiet(m)
    pause_annotation()
    n_assemble_b = _block_counts(tape)["AssembleBlock"]
    g_b = ReducedFunctional(J_b, Control(m)).derivative()

    print(f"    [T11] AssembleBlocks A={n_assemble_a} B={n_assemble_b} "
          f"J_A={float(J_a):.8e} J_B={float(J_b):.8e}")
    assert n_assemble_a - n_assemble_b == 2
    assert n_assemble_b == 1  # only the objective assemble remains
    assert abs(float(J_a) - float(J_b)) <= 1e-15 * abs(float(J_a))
    denom = np.max(np.abs(g_a.dat.data_ro))
    rel = np.max(np.abs(g_a.dat.data_ro - g_b.dat.data_ro)) / denom
    assert rel <= 1e-14, f"net-mass hygiene changed the gradient (rel {rel:.2e})"


# ---------------------------------------------------------------------------
# T12 - Sabotage: the suite detects wrong gradients  [P1, meta]
# ---------------------------------------------------------------------------
def _residual_ladder(rf, m, h, dJdm):
    """First-order Taylor residuals |J(m+eps h) - J(m) - eps dJdm| for a given
    (possibly corrupted) dJdm, and the min convergence rate."""
    with stop_annotating():
        Jm = float(rf(m))
        eps = [0.01 / 2 ** i for i in range(4)]
        R1 = []
        for e in eps:
            Jp = float(rf(m._ad_add(h._ad_mul(e))))
            R1.append(abs(Jp - Jm - e * dJdm))
        rf(m)
    rates = [np.log(R1[i] / R1[i + 1]) / np.log(2) for i in range(len(R1) - 1)]
    return R1, min(rates)


def test_taylor_detects_sabotaged_gradient(mesh_2d_mm):
    """The >=1.90 Taylor criterion rejects a gradient wrong by 1%: both a
    multiplicative (1.01 x dJdm) and an orthogonal-noise corruption collapse
    the rate to <=1.15, with first-order residuals >=10x the clean run's. This
    calibrates the suite's detection floor empirically (not by estimate)."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d_mm)
        h = rho_perturb_2d(mesh_2d_mm)
        g_noise = rho_control2_2d(mesh_2d_mm)

    _, _, rf = tape_forward(forward_2d_mm, m)
    g = rf.derivative()
    dJdm_true = h._ad_dot(g)

    R1_clean, rate_clean = _residual_ladder(rf, m, h, dJdm_true)
    assert rate_clean >= 1.90  # sanity: the true gradient converges at 2

    # Corruption 1: multiplicative 1% error.
    R1_mul, rate_mul = _residual_ladder(rf, m, h, 1.01 * dJdm_true)

    # Corruption 2: add 1% of the gradient's l2 norm along a noise direction
    # that is NOT blind to h (<g_noise, h>_L2 != 0), then re-derive dJdm from
    # the corrupted gradient via the Riesz-consistent h._ad_dot. This is a
    # genuinely different corruption from the multiplicative one.
    overlap = fd.assemble(g_noise * h * fd.dx)
    assert abs(overlap) > 1e-8  # noise must be visible along h
    gnorm = np.linalg.norm(g.dat.data_ro)
    nnorm = np.linalg.norm(g_noise.dat.data_ro)
    g_corr = rf.derivative()  # fresh copy of the gradient
    g_corr.dat.data[:] = (g.dat.data_ro
                          + 0.01 * gnorm / nnorm * g_noise.dat.data_ro)
    dJdm_noise = h._ad_dot(g_corr)
    R1_noise, rate_noise = _residual_ladder(rf, m, h, dJdm_noise)

    print(f"    [T12] rate_clean={rate_clean:.4f} rate_mul={rate_mul:.4f} "
          f"rate_noise={rate_noise:.4f}  R1_clean[-1]={R1_clean[-1]:.3e} "
          f"R1_mul[-1]={R1_mul[-1]:.3e} R1_noise[-1]={R1_noise[-1]:.3e}")

    assert rate_mul <= 1.15, f"multiplicative sabotage not detected ({rate_mul})"
    assert rate_noise <= 1.15, f"noise sabotage not detected ({rate_noise})"
    # Corrupted first-order residuals dominate the clean run at the smallest eps.
    assert R1_mul[-1] >= 10 * R1_clean[-1]
    assert R1_noise[-1] >= 10 * R1_clean[-1]


# ---------------------------------------------------------------------------
# T13 - Taylor on rho, 3-D spherical multi-mode  [P2, TAYLOR]
# ---------------------------------------------------------------------------
def test_taylor_rho_fieldsplit_3d(mesh_3d_mm):
    """The 3-D SphericalDtN machinery (Y_lm UFL boundary forms, 8 multipliers,
    interior l=0 mean mode, fieldsplit + adjoint LVS) is adjointable: whole
    derivative dJ/drho passes Taylor. No net-mass guard in 3-D."""
    with stop_annotating():
        m = rho_control_3d(mesh_3d_mm)
        h = rho_perturb_3d(mesh_3d_mm)
        m2 = rho_control2_3d(mesh_3d_mm)

    J, _, rf = tape_forward(forward_3d_mm, m)
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_3d_mm, m2)
    print(f"    [T13] J={float(J):.6e}")
    assert_taylor_with_guards(rf, m, h, J)


# ---------------------------------------------------------------------------
# T14 - L=0 exterior corner (zero feedback row)  [P2, TAYLOR]
# ---------------------------------------------------------------------------
def forward_3d_l0(m_rho):
    """CFG-3D-L0: 3-D exterior-only SphericalDtN(L=0), the smallest Schur
    (1x1). alpha=1 makes the feedback coefficient (lam - alpha/R) vanish for
    l=0, so the multiplier is a pure diagnostic riding a nonzero constraint
    row. J = c00^2 read via the R-DIRECT multiplier subfunction: sensitivity
    flows through the constraint row ONLY."""
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho, bcs={"top": {"dtn": SphericalDtN(L=0)}})
    solver.solve()
    c = trace_coeff_rdirect(solver, "top", "Y0,0")
    return c ** 2, solver


def _rho_radial_3d(mesh, r0, w):
    """Purely radial (l=0) density - moves the monopole trace c00."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    _, r, _ = _polar(mesh)
    return fd.Function(Q).interpolate(fd.exp(-(((r - r0) / w) ** 2)))


def test_taylor_monopole_constraint_row_3d(mesh_3d_mm):
    """The adjoint holds in the degenerate L=0 exterior corner where the flux
    feedback row is identically zero: sensitivity of J = c00^2 flows purely
    through the constraint row. Taylor rate >= 1.90."""
    with stop_annotating():
        m = _rho_radial_3d(mesh_3d_mm, 1.7, 0.15)
        h = _rho_radial_3d(mesh_3d_mm, 1.6, 0.2)

    J, _, rf = tape_forward(forward_3d_l0, m)
    assert_replay_b1(rf, m, J)
    print(f"    [T14] J=c00^2={float(J):.6e}")
    assert_taylor_with_guards(rf, m, h, J)


# ---------------------------------------------------------------------------
# T15 - Cross-mesh (Submesh rho) adjointability  [P2, TAYLOR, tiered]
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def submesh_pair():
    """UnitDiskMesh parent + Submesh(r<0.55), as in test_cross_mesh_consistency.
    The Submesh is the object under test - it has no utility-mesh substitute
    and ExtrudedMesh does not support it (gap G8: 2-D only)."""
    with stop_annotating():
        mesh = fd.UnitDiskMesh(refinement_level=2)
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        DG0 = fd.FunctionSpace(mesh, "DG", 0)
        indicator = fd.Function(DG0).interpolate(
            fd.conditional(r < 0.55, 1.0, 0.0))
        marked = fd.RelabeledMesh(mesh, [indicator], [99])
        submesh = fd.Submesh(marked, marked.topological_dimension, 99)
    return marked, submesh


def _rho_sub(submesh, coeffs):
    """Zero-mean DG0 density on the Submesh: sum of cos(k phi_sub)."""
    Xs = fd.SpatialCoordinate(submesh)
    phis = fd.atan2(Xs[1], Xs[0])
    expr = sum(a * fd.cos(k * phis) for k, a in coeffs)
    return fd.Function(fd.FunctionSpace(submesh, "DG", 0)).interpolate(expr)


def forward_submesh(m_rho_sub, marked):
    """CFG-SUB: density on a Submesh, psi on the parent; the intersect_measures
    + dummy-DG0-field cross-mesh coupling is set up automatically."""
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(marked, "CG", 1))
        solver = GravitySolver(
            psi, m_rho_sub, bcs={1: {"dtn": CylindricalDtN(M=2)}})
    assert solver.cross_mesh
    solver.solve()
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


def test_taylor_rho_submesh(submesh_pair):
    """The cross-mesh (Submesh density) path is adjointable end-to-end. Tiered:
    (i) annotated construction+solve completes; (ii) rf.derivative() returns
    without raising; (iii) guarded Taylor rate >= 1.90. Exterior 2-D DtN on the
    disk boundary activates the net-mass guard, so m and h are azimuthally
    zero-mean. If the cross-mesh residual defeats the adjoint upstream this
    becomes a documented xfail (gap G4), not a deleted test."""
    marked, submesh = submesh_pair
    with stop_annotating():
        m = _rho_sub(submesh, [(2, 1.0)])
        h = _rho_sub(submesh, [(2, 0.8), (1, 0.2)])
        m2 = _rho_sub(submesh, [(2, 1.0), (3, 0.5)])

    # Tier (i): annotated construction + solve completes.
    J, _, rf = tape_forward(forward_submesh, m, marked)
    assert np.isfinite(float(J)) and float(J) > 0.0
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_submesh, m2, marked)

    # Tier (ii): the adjoint derivative returns without raising.
    g = rf.derivative()
    assert np.isfinite(np.max(np.abs(g.dat.data_ro)))
    print(f"    [T15] J={float(J):.6e} grad_max={np.max(np.abs(g.dat.data_ro)):.3e}")

    # Tier (iii): guarded Taylor.
    assert_taylor_with_guards(rf, m, h, J)


# ---------------------------------------------------------------------------
# T16 / T17 - Physics-pinned closed-form sheet gradients  [P2, ANALYTIC]
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mesh_2d_ana():
    return annulus_mesh(n_azimuthal=192, dr=0.1)


def solve_sheet_ana_2d(s, mesh, m_mode):
    """CFG-2D-ANA: sheet sigma = s cos(m phi) on the outer DtN boundary, rho=0,
    CG2, CylindricalDtN(M=4) both boundaries. G = 1."""
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        _, _, phi = _polar(mesh)
        sigma = s * fd.cos(m_mode * phi)
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"dtn": CylindricalDtN(M=4), "sigma": sigma},
                 "bottom": {"dtn": CylindricalDtN(M=4)}})
    solver.solve()
    return solver


def test_gradient_matches_closed_form_sheet_2d(mesh_2d_ana):
    """Physics-pinned: for sigma = s cos(m phi) on the outer boundary a=RMAX,
    the trace coefficient c_cos m is linear in s with the domain-side sheet
    factor dc/ds = 2 pi G a / m (G=1). The pyadjoint gradient of J=c reproduces
    it to 1e-3, excluding the nearest wrong closed form at O(1). A companion
    quadratic objective J=c^2 gives a guarded Taylor rate 2 on the same tape."""
    m_mode, a = 2, RMAX
    with stop_annotating():
        s = real_scalar(mesh_2d_ana, 1.0)
        hs = real_scalar(mesh_2d_ana, 1.0)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    solver = solve_sheet_ana_2d(s, mesh_2d_ana, m_mode)
    c = trace_coeff_integral(solver, "top", m_mode)  # J_lin = c (linear in s)
    J_quad = c ** 2
    pause_annotation()

    rf_lin = ReducedFunctional(c, Control(s))
    rf_quad = ReducedFunctional(J_quad, Control(s))

    # ANALYTIC: dc/ds = 2 pi a / m (domain-side sheet coefficient).
    dcds = hs._ad_dot(rf_lin.derivative())
    analytic = 2 * np.pi * a / m_mode
    rel = abs(dcds - analytic) / abs(analytic)
    print(f"    [T16] dc/ds={dcds:.6f} analytic=2 pi a/m={analytic:.6f} rel={rel:.3e}")
    assert rel <= 1e-3, f"dc/ds={dcds} vs 2 pi a/m={analytic} (rel {rel:.2e})"

    # Negative control: the nearest wrong closed form (a different mode's
    # factor, 2 pi a / (m+1)) is O(1) away and must be excluded by the window.
    wrong = 2 * np.pi * a / (m_mode + 1)
    assert abs(dcds - wrong) / abs(analytic) > 0.1

    # Companion quadratic Taylor (J = c^2) with the guarded harness.
    assert_taylor_with_guards(rf_quad, s, hs, J_quad)


@pytest.mark.longtest
def test_gradient_matches_closed_form_sheet_3d():
    """3-D analogue of T16 (slow lane): sigma = s Y_lm on the outer boundary,
    dc/ds = 4 pi G a / (2l+1). Unit-test resolution (the one 3-D test needing
    accuracy). L=3, (l,m)=(2,1), CG2, G=1. rel error <= 5e-3."""
    l_mode, m_order, a = 2, 1, RMAX
    mesh = shell_mesh_3d(refinement_level=2, dr=0.15)
    with stop_annotating():
        s = real_scalar(mesh, 1.0)
        hs = real_scalar(mesh, 1.0)
        X = fd.SpatialCoordinate(mesh)
        Y = real_spherical_harmonic(l_mode, m_order, X)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"dtn": SphericalDtN(L=3), "sigma": s * Y},
                 "bottom": {"dtn": SphericalDtN(L=3)}})
    solver.solve()
    _, R = solver.boundary_geometry["top"]
    Yc = real_spherical_harmonic(l_mode, m_order, solver.X)
    psi_m = solver.mixed_solution.subfunctions[0]
    c = fd.assemble(psi_m * Yc * solver.ds("top")) / (R ** 2)
    J_quad = c ** 2
    pause_annotation()

    rf_lin = ReducedFunctional(c, Control(s))
    rf_quad = ReducedFunctional(J_quad, Control(s))

    dcds = hs._ad_dot(rf_lin.derivative())
    analytic = 4 * np.pi * a / (2 * l_mode + 1)
    rel = abs(dcds - analytic) / abs(analytic)
    print(f"    [T17] dc/ds={dcds:.6f} analytic=4 pi a/(2l+1)={analytic:.6f} "
          f"rel={rel:.3e}")
    assert rel <= 5e-3, f"dc/ds={dcds} vs 4 pi a/(2l+1)={analytic} (rel {rel:.2e})"

    assert_taylor_with_guards(rf_quad, s, hs, J_quad)


# ---------------------------------------------------------------------------
# T18 - Taylor past the 128-field wall (slow lane)  [TAYLOR]
# ---------------------------------------------------------------------------
def forward_2d_past_wall(m_rho, M=35):
    """CFG-2D-WALL: top+bottom CylindricalDtN(M), 2M + (2M+1) multipliers.

    At M=35 the mixed space carries 142 sub-fields, past the 128 PETSc will
    enumerate, so this configuration exists only because the two Schur blocks
    are described by index sets. The mesh is azimuthally fine enough to
    resolve mode 35 on the boundary (the constructor's quadrature check would
    warn otherwise); it is coarse radially, since none of this tests accuracy.
    """
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": CylindricalDtN(M=M)},
                 "bottom": {"dtn": CylindricalDtN(M=M)}},
            solver_parameters="iterative")
    solver.solve()
    J = fd.assemble(solver.solution ** 2 * fd.dx)
    return J, solver


def test_taylor_rho_past_field_enumeration_wall():
    """The gradient exists at a truncation the old solver could not solve.

    This is the capability the index-set split was built for: a high-truncation
    coupled solve that is still differentiable. The preconditioner is invisible
    to the tape - it changes how each solve is executed, not what is recorded -
    and the adjoint solve reconstructs its index sets from the function-space
    layout, which is identical for the forward and transposed operators. The
    check is that this holds where the field-number description cannot run at
    all, so there is no cross-check against the old path available here; the
    Taylor rate and its negative control carry the whole claim.

    Expensive, and the cost is symbolic form processing rather than solving: a
    141-term boundary mode sum takes ~40s to turn into kernels, while a repeat
    solve of the same operator costs 0.5s. It recurs once per process no matter
    how warm the on-disk cache is, so there is no caching fix. Do not try to
    make it cheaper by coarsening the mesh - measured, that buys ~6% - and do
    not lower the truncation, which is the only real lever and would take the
    field count back below the wall, leaving the test proving nothing.

    Deliberately the same form as
    `test_gravity_solver.TestIndexSetSplit.test_solves_past_the_field_enumeration_wall`
    (same mesh, element, truncation and hence quadrature degree), so whichever
    runs second in a pytest process inherits the compiled kernels and skips
    that ~40s. Changing the truncation or the quadrature degree of either test
    alone silently costs the other one ~40s.
    """
    mesh = annulus_mesh(n_azimuthal=96, dr=0.5)
    with stop_annotating():
        m = rho_control_2d(mesh)
        h = rho_perturb_2d(mesh)

    J, solver, rf = tape_forward(forward_2d_past_wall, m)

    # Measured off the space, not computed from the truncation.
    n_fields = len(solver.mixed_space)
    assert n_fields == 1 + solver.n_multipliers
    assert n_fields > 128, f"only {n_fields} fields: not past the wall"

    assert_replay_b1(rf, m, J)
    res = assert_taylor_with_guards(rf, m, h, J)
    print(f"    [T18] fields={n_fields} multipliers={solver.n_multipliers} "
          f"rate={res['rate']:.4f} rate0={res['rate0']:.4f}")

    # Belt and braces: the substantive protection against a silently zero
    # matfree transpose is the directional-derivative floor inside
    # assert_taylor_with_guards (|dJdm| >= 1e-6 |J|); this only catches a
    # gradient that is identically zero rather than merely tiny.
    g = rf.derivative()
    assert np.max(np.abs(g.dat.data_ro)) > 0.0
