"""Correctness tests for the vertically-lumped preset family.

Exercises the two iterative presets that are specific to extruded meshes:

    "vlumping"     -- 2-level MG, vertical collapse on coarse level,
                      MUMPS LU at the bottom. Inexact Newton baked in.
    "vlumping_hmg" -- as above but the coarse solve descends a geometric
                      MG on the 2D base MeshHierarchy; fine-level smoother
                      is ``ASMLinesmoothPC`` (exact per-column solves).

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
    MeshHierarchy -- the shape ``vlumping_hmg`` requires."""
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


# ---------------------------------------------------------------------------
# "vlumping" and "vlumping_hmg": a short solve converges and the solutions
# agree with one another up to a few multiples of the linear tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["vlumping", "vlumping_hmg"])
def test_preset_loads_and_runs(preset):
    """Preset resolves, solver converges, output is finite and nonzero."""
    mesh = (
        _make_hierarchy_extruded_mesh(base_levels=1)
        if preset == "vlumping_hmg"
        else _make_flat_extruded_mesh()
    )
    h, _ = _run_short(mesh, preset)
    arr = h.dat.data_ro
    assert np.all(np.isfinite(arr))
    assert np.linalg.norm(arr) > 0.0


def test_vlumping_hmg_matches_vlumping():
    """Both presets solve the same linear system to a 1e-4 relative
    tolerance. The solutions should agree up to a small multiple of that."""
    mesh = _make_hierarchy_extruded_mesh(base_levels=1)
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


def test_prol_preserves_constants():
    """Prolongation maps coarse-space ones to fine-space ones.

    A columnwise injection is the whole point of the construction; this
    is the strongest sanity check you can make on the Prol Mat.
    """
    from firedrake.assemble import assemble as fd_assemble
    from petsc4py import PETSc

    mesh = _make_hierarchy_extruded_mesh(base_levels=1)
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
    ones_c = Function(py_ctx.V_coarse).assign(1.0)
    with ones_c.dat.vec_ro as xc:
        yf = py_ctx.Prol.createVecLeft()
        py_ctx.Prol.mult(xc, yf)
        arr = yf.getArray()

    assert np.allclose(arr, 1.0, atol=1e-10), (
        f"Prol @ ones(V_coarse) deviates from ones(V): "
        f"min={arr.min():.3e}, max={arr.max():.3e}"
    )


def test_missing_hierarchy_raises_clearly():
    """Constructing VerticallyLumpedHMGPC on a flat extruded mesh must
    fail at setup with a message that mentions the missing hierarchy --
    not a cryptic "error code 101"."""
    from firedrake.assemble import assemble as fd_assemble
    from petsc4py import PETSc

    mesh = _make_flat_extruded_mesh()
    V = FunctionSpace(mesh, "DQ", DEGREE)
    u, v = TrialFunction(V), TestFunction(V)
    A = fd_assemble(inner(u, v) * dx + inner(grad(u), grad(v)) * dx).petscmat

    pc = PETSc.PC().create(comm=mesh.comm)
    pc.setOperators(A, A)
    pc.setType("python")
    pc.setPythonContext(VerticallyLumpedHMGPC())
    pc.setDM(V.dm)

    with pytest.raises(Exception) as excinfo:
        pc.setUp()

    # petsc4py wraps RuntimeErrors from PCSetUp_Python as PETSc.Error
    # with __str__ == "error code 101". The useful message survives on
    # __cause__ / __context__, so walk the chain.
    exc = excinfo.value
    messages = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        messages.append(str(exc).lower())
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)

    combined = " | ".join(messages)
    assert "hierarchy" in combined or "meshhierarchy" in combined, (
        f"Expected error mentioning the missing MeshHierarchy anywhere "
        f"in the exception chain, got: {messages!r}"
    )


def test_mg_descent_on_coarse_side():
    """After setup, the coarse KSP's PC is a PCMG with > 1 level.
    Guards against a silent-failure mode where the hierarchy doesn't
    propagate through setDM."""
    mesh = _make_hierarchy_extruded_mesh(base_levels=1)
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


def test_fine_smoother_is_asm_not_lu():
    """Regression guard for the ``sub_sub_`` prefix bug.

    The vlumping_hmg preset uses ``ASMLinesmoothPC`` at the fine level.
    Its inner ASM sub-PC lives at prefix ``..._sub_sub_``; a single
    ``_sub_`` would hit ASM's own ``pc_type`` field and silently
    downgrade the line smoother to a full-rank LU (instant OOM on
    large meshes).
    """
    mesh = _make_hierarchy_extruded_mesh(base_levels=1)
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


def test_update_is_not_noop_for_hmg():
    """Across two successive solves the inner operator state advances.
    Confirms ``update()`` reassembles the Galerkin coarse operator."""
    mesh = _make_hierarchy_extruded_mesh(base_levels=1)
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
