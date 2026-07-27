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
from test_gravity_solver import annulus_mesh, shell_mesh_3d


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
