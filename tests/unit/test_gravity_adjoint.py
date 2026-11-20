"""Adjointability tests for gadopt.gravity_solver under Firedrake/pyadjoint.

The 2-D tests of the taped forward map built by ``GravitySolver`` (Taylor
tests, the Real-space multiplier coupling, the ``subfunctions``/``assign``
extraction chain, TLM-adjoint dot products, replay, the closed-form sheet
gradient, the monopole gradient against fresh solves, the sabotage
controls and the truncation past the field enumeration wall) are rows of
the ``adjoint-2d-multiplier`` entry of the ``gravity`` case in
``tests/gia`` (driver ``gia_gravity.py``). That driver imports the
controls, forward maps and tape probes of this module, and other unit
modules import ``assert_taylor_with_guards``.

The tests left here are the 3-D ones (``adjoint-3d-multiplier`` runs on CI
only) and the 2-D/3-D parametrisations whose 3-D half has no local run.
Mesh accuracy is irrelevant for these consistency tests, so every
configuration is a coarse utility mesh (the same annulus/cubed-sphere
constructions as ``test_gravity_solver``).

Annotation state is global and must be managed explicitly. Nothing at module
scope flips it on: ``import firedrake.adjoint`` does not annotate, and
``gadopt.inverse`` (which annotates at import) is deliberately never imported
here. The ``clean_tape`` autouse fixture guarantees a fresh tape per test and
annotation off afterwards. Every solver is constructed under
``stop_annotating()`` so the geometry assembles in ``__init__`` never clutter
the tape; only ``solve()`` and the objective assemble are taped.
"""


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
from pyadjoint.tape import annotate_tape

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.dtn_coupled_adjoint import SOLVE_BLOCK_TYPE
from gadopt.spherical_harmonics import real_spherical_harmonic
from test_gravity_solver import RMAX, annulus_mesh, shell_mesh_3d

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest


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
def _solves_on(blocks, predicate):
    """Solve blocks whose function space satisfies `predicate`.

    Deliberately not keyed on the block class. Firedrake emits a
    `SolveVarFormBlock` for `solve(a == L, u)` and a
    `NonlinearVariationalSolveBlock` for a `LinearVariationalSolver` built by
    hand, and which of those the enclosed-mass system uses is an
    implementation detail - it is a *Real-space solve* either way. The distinction that matters, and the one "a gravity
    solve is one solve block" always meant, is the space being solved on.
    """
    out = []
    for b in blocks:
        space = getattr(b, "function_space", None)
        if space is None and isinstance(b, SOLVE_BLOCK_TYPE):
            # A `CachedSolverBlock` (Firedrake PR 4638) keeps no space of its
            # own; the space of its one output is the space solved on.
            space = b.get_outputs()[0].output.function_space()
        if space is not None and predicate(space):
            out.append(b)
    return out


def _real_space_solves(blocks):
    """The enclosed-mass systems of the 2-D monopole datum."""
    return _solves_on(
        blocks,
        lambda V: len(V) == 1 and V.ufl_element().family() == "Real")


def _mixed_space_solves(blocks):
    """The gravity solve itself, on the mixed potential/multiplier space."""
    return _solves_on(blocks, lambda V: len(V) > 1)


def _reaches(tape, source_outputs, target_bv):
    """Forward reachability on the tape DAG: does data from ``source_outputs``
    (a block's output block_variables) transitively flow into ``target_bv``?

    This is the gradient-relevant notion of "dangling": the design's §1.4
    literal wording ("output is not a dependency of any later block") is
    empirically false for assembles whose output is consumed by terminal
    AdjFloat arithmetic (an abs() or a comparison) - but such a subgraph never
    reaches the objective J, so it cannot influence dJ/dm. Reachability-to-J is the property that actually
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
# ---------------------------------------------------------------------------
# T02 - Taylor on rho, no-multiplier paths  [P0, TAYLOR]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T03 - Taylor on rho, multi-mode fieldsplit path  [P0, TAYLOR]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T04 - CENTREPIECE: float() severs the tape; taped assemble restores it
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T05 - Sensitivity through an R multiplier  [P0, TAYLOR]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T06 - subfunctions[0] + assign preserve the tape  [P1]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T09 - TLM-adjoint dot-product identity  [P1, DOT]
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------
# T10b - the monopole channel, against a reference the tape cannot influence
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T11 - which per-solve 0-form assembles are load-bearing  [P1]
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# T16 / T17 - Physics-pinned closed-form sheet gradients  [P2, ANALYTIC]
# ---------------------------------------------------------------------------
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
