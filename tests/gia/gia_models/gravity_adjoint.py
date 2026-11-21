"""Helpers of `tests/unit/test_gravity_adjoint.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gravity_adjoint.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
import numpy as np
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.dtn_coupled_adjoint import SOLVE_BLOCK_TYPE
from gadopt.spherical_harmonics import real_spherical_harmonic


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

    This is the gradient-relevant notion of "dangling". The literal rule
    "output is not a dependency of any later block" is false for assembles
    whose output is consumed by terminal AdjFloat arithmetic (an abs() or a
    comparison), but such a subgraph never reaches the objective J, so it
    cannot influence dJ/dm. Reachability to J is the property that matters,
    and it is stronger than "feeds nothing at all".
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
