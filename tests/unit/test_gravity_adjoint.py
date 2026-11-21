"""Adjointability tests for gadopt.gravity_solver under Firedrake/pyadjoint.

The 2-D tests of the taped forward map built by ``GravitySolver`` (Taylor
tests, the Real-space multiplier coupling, the ``subfunctions``/``assign``
extraction chain, TLM-adjoint dot products, replay, the closed-form sheet
gradient, the monopole gradient against fresh solves, the sabotage
controls and the truncation past the field enumeration wall) are rows of
the ``adjoint-2d-multiplier`` entry of the ``gravity`` case in
``tests/gia`` (driver ``gia_gravity.py``). The controls, forward maps and
tape probes are in ``tests/gia/gia_models/gravity_adjoint.py``, which that
driver and this module share; other unit modules import
``assert_taylor_with_guards`` from this module.

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

from gadopt import GravitySolver, SphericalDtN
from gadopt.spherical_harmonics import real_spherical_harmonic
from gia_models.gravity import RMAX, annulus_mesh, shell_mesh_3d
from gia_models.gravity_adjoint import (
    _polar,
    _rho_radial_3d,
    forward_3d_l0,
    forward_3d_mm,
    forward_G,
    forward_sheet,
    real_scalar,
    rho_control2_3d,
    rho_control_2d,
    rho_control_3d,
    rho_perturb_3d,
    tape_forward,
)

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
