"""Correctness tests for the vertically-lumped preset family.

Exercises the two iterative presets that are specific to extruded meshes:

    "vlumping"     -- 2-level MG, vertical collapse on coarse level,
                      MUMPS LU at the bottom. Inexact Newton baked in.
    "vlumping_hmg" -- as above but the coarse solve descends a geometric
                      MG on the 2D base MeshHierarchy; fine-level smoother
                      is `ASMLinesmoothPC` (exact per-column solves).

All cases are intentionally tiny (8x8 base, 4 layers, 2 steps) so the
file is serial-laptop friendly.
"""
import numpy as np
import pytest
from firedrake import (
    Constant,
    ExtrudedMesh,
    ExtrudedMeshHierarchy,
    Function,
    FunctionSpace,
    MeshHierarchy,
    RectangleMesh,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    grad,
    inner,
)

from firedrake.petsc import PETSc

from gadopt.richards_solver import (
    vlumping_hmg_richards_solver_parameters,
    vlumping_richards_solver_parameters,
)

from gadopt import (
    BackwardEuler,
    ExponentialCurve,
    RichardsSolver,
    VerticallyLumpedHMGPC,
    get_boundary_ids,
)


# ---------------------------------------------------------------------------
# Problem setup helpers
# ---------------------------------------------------------------------------

# Mild exponential soil (Tracy-ish regime on a small domain).
ALPHA = 0.25
THETA_R = 0.15
THETA_S = 0.45
KS = 1.0e-05
SS = 0.0

NX = 8
NLAYERS = 4
LX = 4.0
LZ = 2.0
DEGREE = 1

DT = 5.0e4
NSTEPS = 2


def _make_soil_curve():
    return ExponentialCurve(
        theta_r=THETA_R, theta_s=THETA_S, Ks=KS, Ss=SS, alpha=ALPHA,
    )


def _make_flat_extruded_mesh():
    """Plain ExtrudedMesh with no hierarchy underneath."""
    base = RectangleMesh(NX, NX, LX, LX, quadrilateral=True)
    mesh = ExtrudedMesh(base, NLAYERS, layer_height=LZ / NLAYERS)
    mesh.cartesian = True
    return mesh


def _make_hierarchy_extruded_mesh(base_levels=1):
    """Finest level of an ExtrudedMeshHierarchy built over a 2D
    MeshHierarchy -- the shape `vlumping_hmg` requires."""
    nx_coarse = NX // (2 ** base_levels)
    base_coarse = RectangleMesh(nx_coarse, nx_coarse, LX, LX, quadrilateral=True)
    mh2d = MeshHierarchy(base_coarse, base_levels)
    mh3d = ExtrudedMeshHierarchy(
        mh2d, LZ, base_layer=NLAYERS,
        refinement_ratio=1,
        extrusion_type="uniform",
    )
    mesh = mh3d[-1]
    mesh.cartesian = True
    return mesh


def _run_short(mesh, preset):
    """Run a few Backward-Euler steps and return (head, solver)."""
    soil_curve = _make_soil_curve()
    V = FunctionSpace(mesh, "DQ", DEGREE)
    hr = -LZ

    h = Function(V, name="PressureHead")
    h.interpolate(Constant(hr))

    boundary_ids = get_boundary_ids(mesh)
    richards_bcs = {
        "top": {"h": Constant(-0.1)},
        "bottom": {"h": Constant(hr)},
        boundary_ids.left: {"flux": 0.0},
        boundary_ids.right: {"flux": 0.0},
        boundary_ids.front: {"flux": 0.0},
        boundary_ids.back: {"flux": 0.0},
    }

    solver = RichardsSolver(
        h, soil_curve, delta_t=Constant(DT),
        timestepper=BackwardEuler,
        bcs=richards_bcs,
        solver_parameters=preset,
        quad_degree=3, interior_penalty=0.5,
    )
    for _ in range(NSTEPS):
        solver.solve()
    return h, solver


# Meshes are immutable and safe to share read-only; building the
# ExtrudedMeshHierarchy is the one non-trivial cost in this file, so hoist
# both meshes to module scope. Function spaces and solvers are still built
# per test, so no mutable solver state leaks between tests.
@pytest.fixture(scope="module")
def flat_mesh():
    return _make_flat_extruded_mesh()


@pytest.fixture(scope="module")
def hierarchy_mesh():
    return _make_hierarchy_extruded_mesh(base_levels=1)


# ---------------------------------------------------------------------------
# "vlumping" and "vlumping_hmg": a short solve converges and the solutions
# agree with one another up to a few multiples of the linear tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["vlumping", "vlumping_hmg"])
def test_preset_loads_and_runs(preset, flat_mesh, hierarchy_mesh):
    """Preset resolves, solver converges, output is finite and nonzero."""
    mesh = hierarchy_mesh if preset == "vlumping_hmg" else flat_mesh
    h, _ = _run_short(mesh, preset)
    arr = h.dat.data_ro
    assert np.all(np.isfinite(arr))
    assert np.linalg.norm(arr) > 0.0


def test_vlumping_hmg_matches_vlumping(hierarchy_mesh):
    """Both presets solve the same linear system to a 1e-4 relative
    tolerance. The solutions should agree up to a small multiple of that."""
    mesh = hierarchy_mesh
    h_ref, _ = _run_short(mesh, "vlumping")
    h_new, _ = _run_short(mesh, "vlumping_hmg")

    diff = assemble((h_ref - h_new) ** 2 * dx) ** 0.5
    ref_norm = assemble(h_ref ** 2 * dx) ** 0.5
    rel = diff / ref_norm
    assert rel < 1e-2, (
        f"vlumping_hmg vs vlumping relative L2 difference {rel:.3e} too large"
    )


# ---------------------------------------------------------------------------
# Structural invariants of the VerticallyLumpedHMGPC class
# ---------------------------------------------------------------------------


def test_prol_preserves_constants(hierarchy_mesh):
    """Prolongation maps coarse-space ones to fine-space ones.

    A columnwise injection is the whole point of the construction; this
    is the strongest sanity check you can make on the Prol Mat.
    """
    from firedrake.assemble import assemble as fd_assemble
    from petsc4py import PETSc

    mesh = hierarchy_mesh
    V = FunctionSpace(mesh, "DQ", DEGREE)
    u, v = TrialFunction(V), TestFunction(V)
    A = fd_assemble(inner(u, v) * dx + inner(grad(u), grad(v)) * dx).petscmat

    pc = PETSc.PC().create(comm=mesh.comm)
    pc.setOperators(A, A)
    pc.setType("python")
    pc.setPythonContext(VerticallyLumpedHMGPC())
    pc.setDM(V.dm)
    pc.setUp()

    py_ctx = pc.getPythonContext()
    ones_c = Function(py_ctx.V_base_2d).assign(1.0)
    with ones_c.dat.vec_ro as xc:
        yf = py_ctx.Prol.createVecLeft()
        py_ctx.Prol.mult(xc, yf)
        arr = yf.getArray()

    assert np.allclose(arr, 1.0, atol=1e-10), (
        f"Prol @ ones(V_base_2d) deviates from ones(V): "
        f"min={arr.min():.3e}, max={arr.max():.3e}"
    )


def test_missing_hierarchy_raises_clearly(flat_mesh):
    """The vlumping_hmg PC must reject a flat (no-hierarchy) extruded mesh
    with a message that names the missing MeshHierarchy, rather than failing
    cryptically.

    We invoke the context's ``initialize`` directly instead of going through
    ``pc.setUp()``. ``initialize`` is exactly the code ``setUp`` runs, and the
    guard raises a plain Python ``RuntimeError`` -- so we assert on its message
    without routing it through ``PCSetUp_Python``. That C path prints PETSc's
    "Unhandled Python Exception" banner to the raw stderr fd as the exception
    unwinds, which segfaults under pytest's fd-capture on a debug PETSc build,
    and also buries the real message behind petsc4py's bare "error code 101".
    """
    from firedrake.assemble import assemble as fd_assemble
    from petsc4py import PETSc

    V = FunctionSpace(flat_mesh, "DQ", DEGREE)
    u, v = TrialFunction(V), TestFunction(V)
    A = fd_assemble(inner(u, v) * dx + inner(grad(u), grad(v)) * dx).petscmat

    pc = PETSc.PC().create(comm=flat_mesh.comm)
    pc.setOperators(A, A)
    pc.setType("python")
    pc.setPythonContext(VerticallyLumpedHMGPC())
    pc.setDM(V.dm)

    ctx = pc.getPythonContext()
    with pytest.raises(RuntimeError, match="(?i)hierarchy"):
        ctx.initialize(pc)


def test_mg_descent_on_coarse_side(hierarchy_mesh):
    """After setup, the coarse KSP's PC is a PCMG with > 1 level.
    Guards against a silent-failure mode where the hierarchy doesn't
    propagate through setDM."""
    mesh = hierarchy_mesh
    _, solver = _run_short(mesh, "vlumping_hmg")

    outer_pc = solver.solver.snes.getKSP().getPC()
    py_ctx = outer_pc.getPythonContext()
    inner_mg = py_ctx.pc
    coarse_pc = inner_mg.getMGCoarseSolve().getPC()

    try:
        nlevels = coarse_pc.getMGLevels()
    except Exception as e:
        pytest.fail(f"coarse PC is not MG: {e!r}")
    assert nlevels > 1, (
        f"coarse MG has only {nlevels} level(s); setDM did not "
        f"propagate the MeshHierarchy"
    )


def test_fine_smoother_is_asm_not_lu(hierarchy_mesh):
    """Regression guard for the `sub_sub_` prefix bug.

    The vlumping_hmg preset uses `ASMLinesmoothPC` at the fine level.
    Its inner ASM sub-PC lives at prefix `..._sub_sub_`; a single
    `_sub_` would hit ASM's own `pc_type` field and silently
    downgrade the line smoother to a full-rank LU (instant OOM on
    large meshes).
    """
    mesh = hierarchy_mesh
    _, solver = _run_short(mesh, "vlumping_hmg")

    outer_pc = solver.solver.snes.getKSP().getPC()
    inner_mg = outer_pc.getPythonContext().pc

    # Fine level is the last level of a 2-level PCMG.
    fine_ksp = inner_mg.getMGSmoother(inner_mg.getMGLevels() - 1)
    fine_pc = fine_ksp.getPC()
    # Firedrake's ASMLinesmoothPC wraps a PETSc PCASM; drill through.
    linesmooth_ctx = fine_pc.getPythonContext()
    asm_pc = linesmooth_ctx.asmpc
    assert asm_pc.getType() == "asm", (
        f"fine-level smoother resolved to PC type '{asm_pc.getType()}'; "
        f"expected 'asm' -- a downgrade to LU indicates the "
        f"sub_sub_ prefix got truncated to sub_"
    )


def test_update_is_not_noop_for_hmg(hierarchy_mesh):
    """Across two successive solves the inner operator state advances.
    Confirms `update()` reassembles the Galerkin coarse operator."""
    mesh = hierarchy_mesh
    soil_curve = _make_soil_curve()

    V = FunctionSpace(mesh, "DQ", DEGREE)
    h = Function(V, name="PressureHead").interpolate(Constant(-LZ))
    boundary_ids = get_boundary_ids(mesh)
    richards_bcs = {
        "top": {"h": Constant(-0.1)},
        "bottom": {"h": Constant(-LZ)},
        boundary_ids.left: {"flux": 0.0},
        boundary_ids.right: {"flux": 0.0},
        boundary_ids.front: {"flux": 0.0},
        boundary_ids.back: {"flux": 0.0},
    }
    solver = RichardsSolver(
        h, soil_curve, delta_t=Constant(DT),
        timestepper=BackwardEuler,
        bcs=richards_bcs,
        solver_parameters="vlumping_hmg",
        quad_degree=3, interior_penalty=0.5,
    )

    solver.solve()
    inner_mg = solver.solver.snes.getKSP().getPC().getPythonContext().pc
    A0, _ = inner_mg.getOperators()
    state_before = A0.getInfo().get("assemblies", None)
    norm_before = A0.norm()

    # Second solve: h has moved so the Jacobian genuinely changes.
    solver.solve()
    A1, _ = inner_mg.getOperators()
    state_after = A1.getInfo().get("assemblies", None)
    norm_after = A1.norm()

    advanced = (
        (state_before is not None and state_after is not None
         and state_after > state_before)
        or not np.isclose(norm_before, norm_after, rtol=1e-14, atol=0.0)
    )
    assert advanced, (
        f"inner PCMG operator did not change between solves "
        f"(norm_before={norm_before}, norm_after={norm_after}); "
        f"update() appears to be a no-op"
    )


# ---------------------------------------------------------------------------
# Lagged setup: the inner PCMG runs against a private snapshot of the
# Jacobian that only refreshes every `vlumping_lag` Newton steps.
# ---------------------------------------------------------------------------


def _lagged_preset(lag):
    params = dict(vlumping_richards_solver_parameters)
    params["vlumping_lag"] = lag
    return params


def test_lag_defaults_to_no_snapshot(flat_mesh):
    """Without the option the PC holds no snapshot and nothing changes."""
    _, solver = _run_short(flat_mesh, "vlumping")
    ctx = solver.solver.snes.getKSP().getPC().getPythonContext()
    assert ctx.lag == 1
    assert ctx._Alag is None


def test_lag_binds_inner_mg_to_the_snapshot(flat_mesh):
    """With a lag the inner PCMG sees the snapshot, not the live Jacobian."""
    _, solver = _run_short(flat_mesh, _lagged_preset(3))
    outer_pc = solver.solver.snes.getKSP().getPC()
    ctx = outer_pc.getPythonContext()
    assert ctx._Alag is not None

    inner_amat, inner_pmat = ctx.pc.getOperators()
    # Both operators must be the snapshot. KSPSetUp_Chebyshev compares the
    # state of Amat and of Pmat, so a live Amat defeats the lag.
    assert inner_amat.handle == ctx._Alag.handle
    assert inner_pmat.handle == ctx._Alag.handle

    live_amat, _ = outer_pc.getOperators()
    assert live_amat.handle != ctx._Alag.handle


def test_lag_refreshes_the_snapshot_on_the_lag_th_update(flat_mesh):
    """The snapshot follows the live Jacobian on every third update only."""
    lag = 3
    _, solver = _run_short(flat_mesh, _lagged_preset(lag))
    pc = solver.solver.snes.getKSP().getPC()
    ctx = pc.getPythonContext()
    _, live = pc.getOperators()

    # Move the live Jacobian before each update, so a refresh is visible as a
    # change in the norm of the snapshot. This corrupts the operator, so the
    # solver is not used again after this loop.
    start = ctx._nupdate
    refreshed = []
    for _ in range(2 * lag):
        live.scale(1.5)
        before = ctx._Alag.norm()
        ctx.update(pc)
        refreshed.append(
            not np.isclose(before, ctx._Alag.norm(), rtol=1e-12, atol=0.0)
        )

    expected = [(start + i + 1) % lag == 0 for i in range(2 * lag)]
    assert refreshed == expected, (
        f"snapshot refresh cadence {refreshed} does not match the expected "
        f"{expected} for vlumping_lag={lag}"
    )


def test_lag_does_not_change_the_solution(flat_mesh):
    """Lagging the preconditioner leaves the Newton solution unchanged."""
    h_ref, _ = _run_short(flat_mesh, "vlumping")
    h_lag, _ = _run_short(flat_mesh, _lagged_preset(3))

    diff = assemble((h_ref - h_lag) ** 2 * dx) ** 0.5
    ref_norm = assemble(h_ref ** 2 * dx) ** 0.5
    rel = diff / ref_norm
    assert rel < 1e-3, (
        f"lagged and unlagged solutions differ by {rel:.3e}; the outer "
        f"Krylov method should keep the true Jacobian for its residual"
    )


# ---------------------------------------------------------------------------
# Automatic Richardson damping: the damping factor follows the measured
# spectrum, and two conditions decide when to measure it again.
# ---------------------------------------------------------------------------


def _auto_preset(base=None, **extra):
    params = dict(base or vlumping_richards_solver_parameters)
    params["vlumping_omega_auto"] = True
    params.update(extra)
    return params


def test_omega_auto_defaults_off(flat_mesh):
    """Without the option the preconditioner derives nothing."""
    _, solver = _run_short(flat_mesh, "vlumping")
    ctx = solver.solver.snes.getKSP().getPC().getPythonContext()
    assert ctx.omega_auto is False
    assert ctx.omega is None


def test_omega_auto_sets_a_richardson_smoother(flat_mesh):
    """The fine smoother becomes Richardson with the derived damping."""
    _, solver = _run_short(flat_mesh, _auto_preset())
    ctx = solver.solver.snes.getKSP().getPC().getPythonContext()

    assert ctx.omega is not None
    assert ctx.omega > 0.0
    assert ctx._omega_estimates >= 1

    smoother = ctx.pc.getMGSmoother(1)
    assert smoother.getType() == "richardson"
    # The preset asked for Chebyshev. The derived value must override it.
    assert vlumping_richards_solver_parameters[
        "lumped_mg_levels_ksp_type"] == "chebyshev"


def test_omega_auto_does_not_change_the_solution(flat_mesh):
    """A different smoother must not move the Newton solution."""
    h_ref, _ = _run_short(flat_mesh, "vlumping")
    h_auto, _ = _run_short(flat_mesh, _auto_preset())

    diff = assemble((h_ref - h_auto) ** 2 * dx) ** 0.5
    ref_norm = assemble(h_ref ** 2 * dx) ** 0.5
    assert diff / ref_norm < 1e-3


def test_omega_auto_remeasures_when_the_operator_balance_moves(flat_mesh):
    """A shift changes the diagonal alone, so the balance metric moves."""
    _, solver = _run_short(flat_mesh, _auto_preset())
    pc = solver.solver.snes.getKSP().getPC()
    ctx = pc.getPythonContext()
    before = ctx._omega_estimates

    # A uniform rescaling leaves the metric alone by construction, so shift
    # the diagonal instead. This corrupts the operator; the solver is not
    # used again after this call.
    A, _ = ctx.pc.getMGSmoother(1).getOperators()
    A.shift(10.0 * A.norm(PETSc.NormType.FROBENIUS))
    ctx.update(pc)

    assert ctx._omega_estimates == before + 1


def test_omega_auto_remeasures_when_iterations_rise(flat_mesh):
    """A Newton step that costs far more iterations triggers a measurement."""
    _, solver = _run_short(flat_mesh, _auto_preset())
    pc = solver.solver.snes.getKSP().getPC()
    ctx = pc.getPythonContext()

    # Establish a reference of one iteration, then present a step that took
    # far more. The balance is untouched, so only the second condition can
    # fire.
    ctx._omega_iter_reference = 1
    ctx._omega_applies = ctx._omega_applies_seen + 20
    before = ctx._omega_estimates
    ctx.update(pc)

    assert ctx._omega_estimates == before + 1


def test_omega_auto_holds_still_when_nothing_moves(flat_mesh):
    """Neither condition fires on a repeated update with a fixed operator."""
    _, solver = _run_short(flat_mesh, _auto_preset())
    pc = solver.solver.snes.getKSP().getPC()
    ctx = pc.getPythonContext()
    before = ctx._omega_estimates

    for _ in range(4):
        ctx.update(pc)

    assert ctx._omega_estimates == before


def test_omega_auto_works_with_the_lag(flat_mesh):
    """The derived damping and the operator snapshot compose."""
    _, solver = _run_short(flat_mesh, _auto_preset(vlumping_lag=3))
    ctx = solver.solver.snes.getKSP().getPC().getPythonContext()
    assert ctx.omega is not None
    assert ctx._Alag is not None
    # Under a lag the smoother reads the snapshot, so the measurement and the
    # balance metric must both come from the snapshot.
    assert ctx.pc.getMGSmoother(1).getOperators()[0].handle == ctx._Alag.handle


def test_omega_auto_works_for_hmg(hierarchy_mesh):
    """The HMG variant derives a damping factor for its fine level too."""
    _, solver = _run_short(
        hierarchy_mesh,
        _auto_preset(vlumping_hmg_richards_solver_parameters),
    )
    ctx = solver.solver.snes.getKSP().getPC().getPythonContext()
    assert ctx.omega is not None
    assert ctx.pc.getMGSmoother(1).getType() == "richardson"
