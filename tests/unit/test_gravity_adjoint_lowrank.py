"""The adjoint of the low-rank DtN path.

The multiplier path's derivative comes free from pyadjoint taping a variational
solve. The low-rank path drives a PETSc KSP directly, so the tape sees nothing
and the derivative is hand-written in `gadopt.dtn_adjoint`. These are the tests
that say whether that hand-written derivative is right.

**The acceptance criterion is the portable subset of `test_gravity_adjoint.py`,
with no tolerance weakened**, and it is reused rather than reimplemented: the
guarded Taylor harness, the perturbation builders and the meshes are imported
from that module, so a gradient that passes here passes the same four
anti-false-green guards the multiplier path is held to - a nonzero directional
derivative, a functional that demonstrably moves, rate >= 1.90 with the
gradient, and rate ~1 without it.

Structurally exempt, because they are assertions *about the multiplier tape*
rather than about the derivative: `test_tape_structure` (counts
NonlinearVariationalSolveBlocks), `test_taylor_through_multiplier` (reads an R
subfunction that does not exist here), `test_coefficient_float_severs_tape`
(about `float()` on a multiplier), and `test_net_mass_blocks_dangling_harmless`
(counts AssembleBlocks emitted by the multiplier path).

Every test builds its own control objects. Sharing one control `Function`
between two ReducedFunctionals silently cross-contaminates them - pyadjoint's
`taylor_test` leaves the control at its last perturbed value - and that
produced a replay discrepancy during development that looked like a defect in
the block.
"""

import numpy as np
import pytest
import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, get_working_tape,
    pause_annotation, stop_annotating, taylor_test)

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.spherical_harmonics import real_spherical_harmonic
from test_gravity_adjoint import (  # noqa: F401  (clean_tape is an autouse fixture)
    assert_replay_b1, assert_replay_b2, assert_taylor_with_guards, clean_tape,
    real_scalar, rho_control_2d, rho_control2_2d, rho_control_3d,
    rho_control2_3d, rho_perturb_2d, rho_perturb_3d, tape_forward,
    taylor_first_order_ladder)
from test_gravity_solver import RMAX, annulus_mesh, shell_mesh_3d


@pytest.fixture(scope="module")
def mesh_2d():
    return annulus_mesh(n_azimuthal=96, dr=0.2)


@pytest.fixture(scope="module")
def mesh_3d():
    return shell_mesh_3d(refinement_level=1, dr=0.5)


LOW = dict(dtn_representation="lowrank")


# ---------------------------------------------------------------------------
# Forward maps, mirroring test_gravity_adjoint's but on the fast path
# ---------------------------------------------------------------------------
def forward_rho_2d(m_rho, gravitational_constant=1.0):
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}},
            gravitational_constant=gravitational_constant, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_rho_3d(m_rho, gravitational_constant=1.0):
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, m_rho,
            bcs={"top": {"dtn": SphericalDtN(L=1)},
                 "bottom": {"dtn": SphericalDtN(L=1)}},
            gravitational_constant=gravitational_constant, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_no_multiplier(m_rho, bc_variant):
    """The two configurations with no DtN modes at all.

    `dirichlet2d` has no DtN boundary, so the low-rank path degenerates to a
    plain assembled Poisson solve with an empty mode set - worth covering
    because the `C` machinery must survive having nothing to build.
    """
    mesh = m_rho.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        bcs = ({"top": {"psi": 0.0}} if bc_variant == "dirichlet2d"
               else {"top": {"dtn": CylindricalDtN(M=0)}})
        solver = GravitySolver(psi, m_rho, bcs=bcs, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_sheet(sigma_fn, geometry):
    mesh = sigma_fn.function_space().mesh()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        if geometry == "2d":
            bcs = {"top": {"dtn": CylindricalDtN(M=2), "sigma": sigma_fn},
                   "bottom": {"dtn": CylindricalDtN(M=2)}}
        else:
            bcs = {"top": {"dtn": SphericalDtN(L=1), "sigma": sigma_fn},
                   "bottom": {"dtn": SphericalDtN(L=1)}}
        solver = GravitySolver(psi, 0.0, bcs=bcs, **LOW)
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def forward_G(G_const, geometry, rho_fixed):
    if geometry == "2d":
        return forward_rho_2d(rho_fixed, G_const)
    return forward_rho_3d(rho_fixed, G_const)


# ---------------------------------------------------------------------------
# Taylor on rho
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bc_variant", ["dirichlet2d", "lu2d"])
def test_taylor_rho_no_multiplier(mesh_2d, bc_variant):
    """No DtN modes: the empty-`C` corner, and a plain Dirichlet Poisson."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d)
        h = rho_perturb_2d(mesh_2d)
        m2 = rho_control2_2d(mesh_2d)

    J, _, rf = tape_forward(forward_no_multiplier, m, bc_variant)
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_no_multiplier, m2, bc_variant)
    assert_taylor_with_guards(rf, m, h, J)


def test_taylor_rho_2d(mesh_2d):
    """The 2-D DtN configuration, with both boundaries and the interior mean
    mode, i.e. the negative low-rank weight."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d)
        h = rho_perturb_2d(mesh_2d)
        m2 = rho_control2_2d(mesh_2d)

    J, _, rf = tape_forward(forward_rho_2d, m)
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_rho_2d, m2)

    residuals = taylor_first_order_ladder(rf, m, h)
    print(f"    [2d rho] R0 {residuals['R0_rate']}  R1 {residuals['R1_rate']}")
    assert min(residuals["R0_rate"]) >= 0.9
    assert min(residuals["R1_rate"]) >= 1.90
    assert min(residuals["R1"]) >= 1e-15

    assert_taylor_with_guards(rf, m, h, J)
    assert np.max(np.abs(rf.derivative().dat.data_ro)) > 0.0


def test_taylor_rho_3d(mesh_3d):
    with stop_annotating():
        m = rho_control_3d(mesh_3d)
        h = rho_perturb_3d(mesh_3d)
        m2 = rho_control2_3d(mesh_3d)

    J, _, rf = tape_forward(forward_rho_3d, m)
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_rho_3d, m2)
    assert_taylor_with_guards(rf, m, h, J)


# ---------------------------------------------------------------------------
# Taylor on sigma and on G
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geometry", ["2d", "3d"])
def test_taylor_sigma(geometry, mesh_2d, mesh_3d):
    mesh = mesh_2d if geometry == "2d" else mesh_3d
    with stop_annotating():
        V = fd.FunctionSpace(mesh, "CG", 1)
        X = fd.SpatialCoordinate(mesh)
        if geometry == "2d":
            phi = fd.atan2(X[1], X[0])
            sigma = fd.Function(V).interpolate(fd.cos(2 * phi))
            h = fd.Function(V).interpolate(
                0.8 * fd.cos(2 * phi) + 0.2 * fd.cos(phi))
            sigma2 = fd.Function(V).interpolate(
                fd.cos(2 * phi) + 0.5 * fd.cos(3 * phi))
        else:
            Y11 = real_spherical_harmonic(1, 1, X)
            Y10 = real_spherical_harmonic(1, 0, X)
            sigma = fd.Function(V).interpolate(Y11)
            h = fd.Function(V).interpolate(0.8 * Y11 + 0.2 * Y10)
            sigma2 = fd.Function(V).interpolate(Y11 + 0.5 * Y10)

    J, _, rf = tape_forward(forward_sheet, sigma, geometry)
    assert_replay_b1(rf, sigma, J)
    assert_replay_b2(rf, forward_sheet, sigma2, geometry)
    assert_taylor_with_guards(rf, sigma, h, J)


@pytest.mark.parametrize("geometry", ["2d", "3d"])
def test_gradient_G_constant(geometry, mesh_2d, mesh_3d):
    """G as a Real-space control, with the exact identity dJ/dG = 2J/G.

    psi is exactly linear in G, so J is exactly quadratic and the analytic
    derivative is available - which makes this the one Taylor test here with an
    independent closed form behind it rather than only a convergence rate.
    """
    mesh = mesh_2d if geometry == "2d" else mesh_3d
    with stop_annotating():
        rho_fixed = (rho_control_2d(mesh) if geometry == "2d"
                     else rho_control_3d(mesh))
        G = real_scalar(mesh, 1.3)
        h = real_scalar(mesh, 1.0)

    J, _, rf = tape_forward(forward_G, G, geometry, rho_fixed)
    J0, G0 = float(J), float(G)
    assert_replay_b1(rf, G, J)
    assert_replay_b2(rf, forward_G, real_scalar(mesh, 1.7), geometry, rho_fixed)

    rf(G)
    dJdG = h._ad_dot(rf.derivative())
    analytic = 2.0 * J0 / G0
    relative = abs(dJdG - analytic) / abs(dJdG)
    print(f"    [G {geometry}] dJ/dG {dJdG:.6e}  2J/G {analytic:.6e}  "
          f"rel {relative:.2e}")
    assert relative <= 1e-9

    epsilon = 1e-3
    plus = float(rf(real_scalar(mesh, G0 + epsilon)))
    minus = float(rf(real_scalar(mesh, G0 - epsilon)))
    rf(G)
    difference = (plus - minus) / (2 * epsilon)
    assert abs(difference - dJdG) / abs(dJdG) <= 1e-9

    stats = assert_taylor_with_guards(rf, G, h, J)
    sabotage = taylor_test(rf, G, h, dJdm=1.01 * dJdG)
    print(f"    [G {geometry}] rate {stats['rate']:.4f} sabotage {sabotage:.4f}")
    assert sabotage <= 1.15


# ---------------------------------------------------------------------------
# Transpose consistency and sabotage detection
# ---------------------------------------------------------------------------
def test_tlm_adjoint_dot_product(mesh_2d):
    """`<J^T g, h> == <g, J h>`, isolated from finite-difference truncation.

    Both routes are exercised: pyadjoint's native tangent sweep, which the
    multiplier path cannot run at all (its TLM issues a bare `solve()` with no
    solver parameters and the R-space mixed system defeats it), and the exact
    independent tangent available because rho -> psi is linear. The fast path
    supports the native sweep, so this is a capability the multiplier path does
    not have.
    """
    with stop_annotating():
        m = rho_control_2d(mesh_2d)
        h = rho_perturb_2d(mesh_2d)

    J, _, rf = tape_forward(forward_rho_2d, m)
    adjoint_value = h._ad_dot(rf.derivative())

    m.block_variable.tlm_value = h
    get_working_tape().evaluate_tlm()
    tlm_value = float(J.block_variable.tlm_value)
    relative = abs(tlm_value - adjoint_value) / abs(adjoint_value)
    print(f"    [tlm native] adj {adjoint_value:.10e} tlm {tlm_value:.10e} "
          f"rel {relative:.3e}")
    assert abs(adjoint_value) >= 1e-12
    assert relative <= 1e-8

    with stop_annotating():
        psi_m = forward_rho_2d(m)[1].solution.copy(deepcopy=True)
        psi_h = forward_rho_2d(h)[1].solution.copy(deepcopy=True)
        exact = 2.0 * fd.assemble(psi_m * psi_h * fd.dx)
    relative = abs(exact - adjoint_value) / abs(adjoint_value)
    print(f"    [tlm exact] adj {adjoint_value:.10e} tan {exact:.10e} "
          f"rel {relative:.3e}")
    assert relative <= 1e-8


def test_taylor_detects_sabotaged_gradient(mesh_2d):
    """The suite's detection floor, measured on this path rather than assumed."""
    with stop_annotating():
        m = rho_control_2d(mesh_2d)
        h = rho_perturb_2d(mesh_2d)
        noise = rho_control2_2d(mesh_2d)

    _, _, rf = tape_forward(forward_rho_2d, m)
    g = rf.derivative()
    true_value = h._ad_dot(g)

    def ladder(value):
        with stop_annotating():
            Jm = float(rf(m))
            epsilons = [0.01 / 2 ** i for i in range(4)]
            residuals = []
            for epsilon in epsilons:
                Jp = float(rf(m._ad_add(h._ad_mul(epsilon))))
                residuals.append(abs(Jp - Jm - epsilon * value))
            rf(m)
        rates = [np.log(residuals[i] / residuals[i + 1]) / np.log(2)
                 for i in range(len(residuals) - 1)]
        return residuals, min(rates)

    clean, clean_rate = ladder(true_value)
    assert clean_rate >= 1.90

    _, multiplicative = ladder(1.01 * true_value)
    corrupted = rf.derivative()
    scale = (0.01 * np.linalg.norm(g.dat.data_ro)
             / np.linalg.norm(noise.dat.data_ro))
    corrupted.dat.data[:] = g.dat.data_ro + scale * noise.dat.data_ro
    noisy, noise_rate = ladder(h._ad_dot(corrupted))

    print(f"    [sabotage] clean {clean_rate:.4f} mult {multiplicative:.4f} "
          f"noise {noise_rate:.4f}")
    assert multiplicative <= 1.15
    assert noise_rate <= 1.15
    assert noisy[-1] >= 10 * clean[-1]


# ---------------------------------------------------------------------------
# Cross-mesh
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def submesh_pair():
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


def _rho_sub(submesh, coefficients):
    Xs = fd.SpatialCoordinate(submesh)
    phis = fd.atan2(Xs[1], Xs[0])
    expression = sum(a * fd.cos(k * phis) for k, a in coefficients)
    return fd.Function(fd.FunctionSpace(submesh, "DG", 0)).interpolate(expression)


def forward_submesh(m_rho_sub, marked):
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(marked, "CG", 1))
        solver = GravitySolver(psi, m_rho_sub,
                               bcs={1: {"dtn": CylindricalDtN(M=2)}}, **LOW)
    assert solver.cross_mesh
    solver.solve()
    return fd.assemble(solver.solution ** 2 * fd.dx), solver


def test_taylor_rho_submesh(submesh_pair):
    """The cross-mesh path, differentiated end to end, with no mixed space."""
    marked, submesh = submesh_pair
    with stop_annotating():
        m = _rho_sub(submesh, [(2, 1.0)])
        h = _rho_sub(submesh, [(2, 0.8), (1, 0.2)])
        m2 = _rho_sub(submesh, [(2, 1.0), (3, 0.5)])

    J, solver, rf = tape_forward(forward_submesh, m, marked)
    assert solver.mixed_space is solver.solution_space
    assert np.isfinite(float(J)) and float(J) > 0.0
    assert_replay_b1(rf, m, J)
    assert_replay_b2(rf, forward_submesh, m2, marked)
    assert np.isfinite(np.max(np.abs(rf.derivative().dat.data_ro)))
    assert_taylor_with_guards(rf, m, h, J)


# ---------------------------------------------------------------------------
# The rest of the portable subset
# ---------------------------------------------------------------------------
def test_reduced_functional_replay(mesh_2d):
    """Replay at a new control equals a fresh solve there, and returns.

    Also documents the same gap the multiplier path has: replay bypasses the
    Python-level `check_net_mass`, so a net-mass control replays without
    raising although a direct solve raises.
    """
    with stop_annotating():
        m = rho_control_2d(mesh_2d)
        m2 = rho_control2_2d(mesh_2d)

    J, _, rf = tape_forward(forward_rho_2d, m)
    assert abs(float(rf(m)) - float(J)) <= 1e-9 * abs(float(J))

    with stop_annotating():
        direct, _ = forward_rho_2d(m2)
    value = float(rf(m2))
    relative = abs(value - float(direct)) / abs(float(direct))
    print(f"    [replay] rf(m2) {value:.8e} direct {float(direct):.8e} "
          f"rel {relative:.2e}")
    assert relative <= 1e-9

    with stop_annotating():
        Q = fd.FunctionSpace(mesh_2d, "CG", 1)
        X = fd.SpatialCoordinate(mesh_2d)
        r = fd.sqrt(fd.dot(X, X))
        with_mass = fd.Function(Q).interpolate(fd.exp(-(((r - 1.7) / 0.15) ** 2)))
    assert np.isfinite(float(rf(with_mass)))
    with pytest.raises(NotImplementedError, match="Net mass"):
        with stop_annotating():
            forward_rho_2d(with_mass)


def test_solution_assign_preserves_tape(mesh_2d):
    """`solver.solution` and `mixed_solution` give the same gradient.

    On the fast path `mixed_solution` is a plain Function on the potential
    space rather than a mixed one, so `subfunctions[0]` is the object itself
    and the extraction chain is shorter - which is exactly why it is worth
    checking that the chain still carries the derivative.
    """
    with stop_annotating():
        m = rho_control_2d(mesh_2d)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh_2d, "CG", 1))
        solver = GravitySolver(
            psi, m,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}}, **LOW)
    solver.solve()
    J_solution = fd.assemble(solver.solution ** 2 * fd.dx)
    J_mixed = fd.assemble(solver.mixed_solution.subfunctions[0] ** 2 * fd.dx)
    pause_annotation()

    g_solution = ReducedFunctional(J_solution, Control(m)).derivative()
    g_mixed = ReducedFunctional(J_mixed, Control(m)).derivative()
    assert abs(float(J_solution) - float(J_mixed)) <= 1e-14 * abs(float(J_mixed))
    denominator = np.max(np.abs(g_mixed.dat.data_ro))
    relative = np.max(np.abs(g_solution.dat.data_ro - g_mixed.dat.data_ro)) / denominator
    print(f"    [assign] grad rel {relative:.3e}")
    assert relative <= 1e-14


def test_taylor_monopole_constraint_row_3d(mesh_3d):
    """The degenerate `L = 0` exterior corner, which is a better test here.

    With `alpha = 1` and `lam_0 = 1/R`, the weight `(lam_0 - alpha/R)` is
    **exactly zero**, so on this path the monopole contributes nothing to `B`
    at all and `c00` is a pure linear read-off of the potential. The test is
    therefore that a gradient flows correctly through a trace coefficient whose
    mode has zero weight in the operator - a different and slightly sharper
    question than the multiplier path's version of it, which is about a
    constraint row with a zero feedback row.
    """
    with stop_annotating():
        Q = fd.FunctionSpace(mesh_3d, "CG", 1)
        X = fd.SpatialCoordinate(mesh_3d)
        r = fd.sqrt(fd.dot(X, X))
        m = fd.Function(Q).interpolate(fd.exp(-(((r - 1.7) / 0.15) ** 2)))
        h = fd.Function(Q).interpolate(fd.exp(-(((r - 1.6) / 0.2) ** 2)))

    def forward(m_rho):
        mesh = m_rho.function_space().mesh()
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(psi, m_rho,
                                   bcs={"top": {"dtn": SphericalDtN(L=0)}}, **LOW)
        solver.solve()
        # The weight really is zero, which is the premise of this test.
        assert abs(float(solver.mode_rows[0].weights[0])) < 1e-14
        c = fd.assemble(solver.solution * fd.Constant(1.0) * fd.dx)
        return c ** 2, solver

    J, _, rf = tape_forward(forward, m)
    assert_replay_b1(rf, m, J)
    print(f"    [L=0 corner] J {float(J):.6e}")
    assert_taylor_with_guards(rf, m, h, J)


def test_gradient_matches_closed_form_sheet_2d():
    """Physics-pinned: `dc/ds = 2 pi G a / m` for a sheet at radius `a`.

    The only test here with an independent closed form behind the *value* of a
    derivative rather than only its convergence rate, and it uses the trace
    coefficient rather than an energy, so it pins the recovery expression too.
    """
    m_mode, a = 2, RMAX
    mesh = annulus_mesh(n_azimuthal=192, dr=0.1)
    with stop_annotating():
        s = real_scalar(mesh, 1.0)
        hs = real_scalar(mesh, 1.0)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        X = fd.SpatialCoordinate(mesh)
        phi = fd.atan2(X[1], X[0])
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"dtn": CylindricalDtN(M=4), "sigma": s * fd.cos(m_mode * phi)},
                 "bottom": {"dtn": CylindricalDtN(M=4)}}, **LOW)
    solver.solve()
    _, R = solver.boundary_geometry["top"]
    c = fd.assemble(solver.solution * fd.cos(m_mode * phi) * solver.ds("top")) / (np.pi * R)
    J = c ** 2
    pause_annotation()

    dcds = hs._ad_dot(ReducedFunctional(c, Control(s)).derivative())
    analytic = 2 * np.pi * a / m_mode
    relative = abs(dcds - analytic) / abs(analytic)
    print(f"    [closed form] dc/ds {dcds:.6f} vs 2 pi a/m {analytic:.6f} "
          f"rel {relative:.3e}")
    assert relative <= 1e-3
    # The nearest wrong closed form is O(1) away and must be excluded.
    assert abs(dcds - 2 * np.pi * a / (m_mode + 1)) / abs(analytic) > 0.1

    assert_taylor_with_guards(ReducedFunctional(J, Control(s)), s, hs, J)


def test_taylor_at_a_truncation_the_multiplier_path_cannot_reach():
    """The replacement for the field-enumeration-wall test.

    The multiplier path's version of this exists because PETSc refuses to
    enumerate more than 128 sub-fields, and it demonstrates that the gradient
    survives past that wall. The wall does not exist here - there is one field
    - so the index-set reasoning does not port. What ports is the *purpose*:
    that the derivative exists at a truncation the shipped default solver
    cannot practically reach, which is the capability this whole path is for.

    `M = 35` gives 141 modes. Its own docstring records that the multiplier
    version costs ~40 s of per-process symbolic form processing that no cache
    can remove; on this path that term is gone, so this test is cheap and
    there is no cost argument for omitting it.
    """
    mesh = annulus_mesh(n_azimuthal=96, dr=0.5)
    with stop_annotating():
        m = rho_control_2d(mesh)
        h = rho_perturb_2d(mesh)

    def forward(m_rho):
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(
                psi, m_rho,
                bcs={"top": {"dtn": CylindricalDtN(M=35)},
                     "bottom": {"dtn": CylindricalDtN(M=35)}},
                solver_parameters="iterative", **LOW)
        solver.solve()
        return fd.assemble(solver.solution ** 2 * fd.dx), solver

    J, solver, rf = tape_forward(forward, m)
    # Counted off the objects, never from a formula for the truncation.
    n_modes = len(solver._multiplier_keys)
    assert n_modes == 141
    assert solver.n_multipliers == 0  # no fields to enumerate at all
    assert_replay_b1(rf, m, J)
    stats = assert_taylor_with_guards(rf, m, h, J)
    print(f"    [past the wall] {n_modes} modes, rate {stats['rate']:.4f}, "
          f"control {stats['rate0']:.4f}")
    assert np.max(np.abs(rf.derivative().dat.data_ro)) > 0.0


def test_coefficient_taped_integral_taylor(mesh_2d):
    """`coefficients()` is differentiable, and agrees with the integral form.

    On the multiplier path the analogous test exists because reading a
    coefficient through `float()` severs the tape, and the repair is to rebuild
    it as a taped boundary integral. Here `coefficients()` is taped directly -
    `c_k = u_k . psi / (scale_k * A_h)` is a linear functional of psi with a
    constant vector - so both forms are available and must agree.

    They are genuinely different routes: the taped `coefficients()` contracts
    against the assembled row vector the low-rank operator was built from,
    while the integral form re-assembles `psi e_k ds` symbolically. Agreement
    of their gradients is therefore a statement about the recovery expression,
    not about one implementation of it.

    This test could not exist while `coefficients()` returned raw floats, and
    its absence was consistent with that - nothing was ever red. Worth
    remembering: a missing capability and its missing test hide each other.
    """
    with stop_annotating():
        m = rho_control_2d(mesh_2d)
        h = rho_perturb_2d(mesh_2d)

    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    with stop_annotating():
        psi = fd.Function(fd.FunctionSpace(mesh_2d, "CG", 1))
        solver = GravitySolver(
            psi, m,
            bcs={"top": {"dtn": CylindricalDtN(M=2)},
                 "bottom": {"dtn": CylindricalDtN(M=2)}}, **LOW)
    solver.solve()

    c_recovered = solver.coefficients()["top"]["cos2"]
    X = fd.SpatialCoordinate(mesh_2d)
    phi = fd.atan2(X[1], X[0])
    _, R = solver.boundary_geometry["top"]
    c_integral = fd.assemble(
        solver.mixed_solution * fd.cos(2 * phi) * solver.ds("top")) / (np.pi * R)
    J_recovered = c_recovered ** 2
    J_integral = c_integral ** 2
    pause_annotation()

    # Same value on the solution manifold, by construction.
    assert abs(float(c_recovered) - float(c_integral)) <= 1e-8 * abs(float(c_integral))

    rf_recovered = ReducedFunctional(J_recovered, Control(m))
    rf_integral = ReducedFunctional(J_integral, Control(m))

    g_recovered = rf_recovered.derivative()
    g_integral = rf_integral.derivative()
    recovered_norm = np.max(np.abs(g_recovered.dat.data_ro))
    denominator = np.max(np.abs(g_integral.dat.data_ro))
    print(f"    [taped coefficients] c {float(c_recovered):.6e}  "
          f"|grad via coefficients()| {recovered_norm:.6e}  "
          f"|grad via integral| {denominator:.6e}")

    # THE DISCRIMINATOR, and it is a mechanism check rather than a magnitude
    # one. If coefficients() goes back to reading .dat.data_ro and returning
    # float(), pyadjoint sees no dependency at all and this derivative is
    # EXACTLY zero - not merely inaccurate.
    #
    # Verified by reverting the capability and measuring, rather than asserted:
    # with the untaped implementation `c` comes back as a plain `float` and the
    # gradient is 0.0 exactly, against 3.695499e-02 for the taped one. (The
    # untaped version also trips the value assertion above, because a plain
    # float cannot carry a tape at all - that is additional evidence, not a
    # substitute for this line, since a version taped in name only would reach
    # here.)
    assert recovered_norm > 0.0, (
        "the gradient through coefficients() is identically zero: the tape is "
        "severed, so coefficients() is not differentiable")
    assert denominator > 0.0, "the integral-form gradient is identically zero"

    # And the pair checks itself: one number comes from the capability, the
    # other from a route that re-assembles the same quantity symbolically and
    # therefore cannot use it. Agreement is possible only if the capability is
    # real, so neither assertion is load-bearing on its own.
    relative = np.max(
        np.abs(g_recovered.dat.data_ro - g_integral.dat.data_ro)) / denominator
    print(f"    [taped coefficients] grad rel {relative:.3e}")
    assert relative <= 1e-8

    assert_replay_b1(rf_recovered, m, J_recovered)
    assert_taylor_with_guards(rf_recovered, m, h, J_recovered)


@pytest.mark.longtest
def test_gradient_matches_closed_form_sheet_3d():
    """3-D analogue of the closed-form sheet gradient: dc/ds = 4 pi G a/(2l+1).

    Unit-test resolution rather than the coarse consistency meshes, because
    this one checks a derivative against physics rather than against another
    derivative, so the discretisation error is part of the budget.
    """
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
                 "bottom": {"dtn": SphericalDtN(L=3)}},
            solver_parameters="iterative", **LOW)
    solver.solve()
    c = solver.coefficients()["top"][f"Y{l_mode},{m_order}"]
    J = c ** 2
    pause_annotation()

    dcds = hs._ad_dot(ReducedFunctional(c, Control(s)).derivative())
    analytic = 4 * np.pi * a / (2 * l_mode + 1)
    relative = abs(dcds - analytic) / abs(analytic)
    print(f"    [closed form 3d] dc/ds {dcds:.6f} vs 4 pi a/(2l+1) "
          f"{analytic:.6f} rel {relative:.3e}")
    assert relative <= 5e-3

    assert_taylor_with_guards(ReducedFunctional(J, Control(s)), s, hs, J)
