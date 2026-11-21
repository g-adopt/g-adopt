"""The adjoint of the low-rank DtN path.

The multiplier path's derivative comes free from pyadjoint taping a variational
solve. The low-rank path drives a PETSc KSP directly, so the tape sees nothing
and the derivative is hand-written in `gadopt.dtn_adjoint`. These are the tests
that say whether that hand-written derivative is right.

The 2-D tests (Taylor tests for density and sheet, replay, the
`subfunctions`/`assign` chain, TLM-adjoint dot product, the closed-form sheet
gradient, the sabotage control, the taped coefficient integral and the
truncation that the multiplier path cannot reach) are rows of the
`adjoint-2d-lowrank` entry of the `gravity` case in `tests/gia` (driver
`gia_gravity.py`). The forward maps are in
`tests/gia/gia_models/gravity_adjoint_lowrank.py`, which that driver and
this module share. The tests
left here are the 3-D ones (`adjoint-3d-lowrank` runs on CI only) and the
2-D/3-D parametrisations whose 3-D half has no local run. They use the
guarded Taylor harness, the perturbation builders and the meshes of
`test_gravity_adjoint.py`, so a gradient that passes here passes the same
four guards as the multiplier path: a nonzero directional derivative, a
functional that moves, rate >= 1.90 with the gradient, and rate ~1 without
it.

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

from gadopt import GravitySolver, SphericalDtN
from gadopt.spherical_harmonics import real_spherical_harmonic
from test_gravity_adjoint import (  # noqa: F401  (clean_tape is an autouse fixture)
    assert_replay_b1,
    assert_replay_b2,
    assert_taylor_with_guards,
    clean_tape,
)
from gia_models.gravity_adjoint import (  # noqa: F401  (clean_tape is an autouse fixture)
    real_scalar,
    rho_control_2d,
    rho_control_3d,
    rho_control2_3d,
    rho_perturb_3d,
    tape_forward,
)
from gia_models.gravity import RMAX, annulus_mesh, shell_mesh_3d
from gia_models.gravity_adjoint_lowrank import (
    LOW,
    forward_G,
    forward_rho_3d,
    forward_sheet,
)


@pytest.fixture(scope="module")
def mesh_2d():
    return annulus_mesh(n_azimuthal=96, dr=0.2)


@pytest.fixture(scope="module")
def mesh_3d():
    return shell_mesh_3d(refinement_level=1, dr=0.5)


# ---------------------------------------------------------------------------
# Taylor on rho
# ---------------------------------------------------------------------------
@pytest.mark.longtest
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
@pytest.mark.longtest
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


@pytest.mark.longtest
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
# The rest of the portable subset
# ---------------------------------------------------------------------------
@pytest.mark.longtest
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
