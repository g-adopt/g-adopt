"""Gate for the build-once dense Schur block-1 preconditioner.

`gadopt.DtNMultiplierDenseSchurPC` forms the multiplier Schur complement `S`
once at setup, factors it, and applies the factors from then on. This file is
the shipped-code gate for it, on the same 2-D coupled annulus the adjoint suite
uses (`test_gia_gravity_adjoint_lowrank._build`): parent annulus + mantle
submesh, `SelfGravitatingGIASolver`, `TRUNCATION = 3`, `N_AZIMUTHAL = 32`.

**A 2-D annulus establishes that the construction works, never that it helps.**
Performance is a separate Gadi step; nothing here measures wall clock or sweeps.

The binding correctness test is T1: the `S` the PC assembles must equal an
INDEPENDENT reference `S_ref` built entirely from the assembled coupled Jacobian
and nothing from the code under test. The reference follows the
`gate_lowrank_operator.py` method -- assemble the coupled Jacobian, read its
dense blocks, and eliminate -- but forms the multiplier Schur complement
directly as `S_ref = D - C @ inv(A00) @ B`, where block 0 is every non-`Real`
field and block 1 is the `Real` multipliers.

Every test states below how it can fail and its numerical floor.

The whole file is gated on `gmsh` (the annulus generator) exactly as the adjoint
suite is; without it the module skips.
"""

import sys
from pathlib import Path

import firedrake as fd
import numpy as np
import pytest
from firedrake import derivative, TrialFunction
from firedrake.petsc import PETSc

from gadopt import (
    CompressibleInternalVariableApproximation,
    CylindricalDtN,
    DtNMultiplierDenseSchurPC,
    SelfGravitatingGIASolver,
    self_gravitating_gia_space,
)
from gadopt.gia_gravity import (
    FluidCore,
    OMEGA_SQ_EARTH,
    selfgrav_dtn_schur_solver_parameters,
)
from gadopt.stokes_integrators import newton_stokes_solver_parameters

# ---------------------------------------------------------------------------
# Geometry and constants: the coarse 2-D annulus of the adjoint suite
# ---------------------------------------------------------------------------
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3
G0 = 1.0
CELL_MANTLE = 101
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5
DR_MANTLE = 0.2
N_AZIMUTHAL = 32
TRUNCATION = 3

#: Round-off floor for a 13x13 dense solve at cond ~2e1: the LU solve is exact
#: to a few ulp times the condition number, so ~1e-13 relative. Every parity and
#: solve test sits five orders under a 1e-8 gate.
PARITY_FLOOR = 1e-10


@pytest.fixture(scope="module")
def meshes():
    """Parent annulus and mantle submesh, both P2-curved.

    Same dr/nazim and the same cached `.msh` as the other GIA unit tests, so the
    file is generated at most once across the suite.
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


def _dense_schur_params():
    """The 2-D `direct` Schur preset, with block 1 on the dense-Schur PC.

    Block 0 is `preonly` + assembled LU, so its inverse is exact and the Schur
    complement the PC builds is exact and linear -- which is what lets T1 assert
    parity to roundoff rather than to a preconditioner tolerance.
    """
    params = dict(newton_stokes_solver_parameters)
    params.update(selfgrav_dtn_schur_solver_parameters)
    params["dtn_fieldsplit_1_pc_type"] = "python"
    params["dtn_fieldsplit_1_pc_python_type"] = (
        "gadopt.DtNMultiplierDenseSchurPC")
    # The true-residual monitor is pure noise in a test log; drop it.
    params.pop("ksp_monitor_true_residual", None)
    return params


def _build_solver(meshes, block1_pc="gadopt.DtNMultiplierDenseSchurPC",
                  dt=None, fluid_core=False):
    """The coupled solver on the annulus, block 1 preconditioned by `block1_pc`.

    Mirrors `test_gia_gravity_adjoint_lowrank._build` but drives the iterative
    Schur path (`DtNTwoBlockSchurPC` with a full Schur factorisation) rather than
    the monolithic direct preset, so that the block-1 python PC is handed the
    `MATSCHURCOMPLEMENT` as its `Amat`.
    """
    parent, sub = meshes
    Xp = fd.SpatialCoordinate(parent)
    Xm = fd.SpatialCoordinate(sub)
    gravity_bcs = {
        CURVE_OUTER: {"dtn": CylindricalDtN(TRUNCATION)},
        CURVE_INNER: {"dtn": CylindricalDtN(TRUNCATION)},
        CURVE_RE: {"interior_sigma":
                   SIGMA_HAT * fd.cos(2 * fd.atan2(Xp[1], Xp[0]))},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=False,
        fluid_core=fluid_core,
        n_internal_variables=1, condense_internal_variables=True,
        self_gravity_number=LAMBDA)
    z = fd.Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        bulk_shear_ratio=100.0, g=G0, B_mu=B_MU, self_gravity_number=LAMBDA)

    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    mechanics_bcs = {
        CURVE_RE: {"normal_stress":
                   fd.Constant(B_MU) * SIGMA_HAT
                   * fd.cos(2 * fd.atan2(Xm[1], Xm[0]))},
    }
    core = None
    if fluid_core:
        core = FluidCore(boundary=CURVE_RC, rho_core=2.0)
    else:
        mechanics_bcs[CURVE_RC] = {"un": 0.0}

    params = _dense_schur_params()
    params["dtn_fieldsplit_1_pc_python_type"] = block1_pc

    dt = 1.0 if dt is None else dt
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=dt, bcs=mechanics_bcs,
        fluid_core=core,
        rotation_moments={"C": fd.assemble(fd.dot(Xm, Xm) * dx_m)},
        Omega_sq=OMEGA_SQ_EARTH, solver_parameters=params)
    return solver, z, layout, Z


def _dense_ctx(solver):
    """The `DtNMultiplierDenseSchurPC` instance PETSc built inside the solve.

    Outer python PC -> its inner Schur fieldsplit -> the block-1 sub-KSP's PC ->
    its python context. Reaching it this way (rather than constructing one) is
    what makes the test check the object the real solve used.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    _, ksp_real = outer.pc.getFieldSplitSchurGetSubKSP()
    return ksp_real, ksp_real.getPC().getPythonContext()


def _reference_S(solver, z, Z):
    """`S_ref = D - C @ inv(A00) @ B` from the assembled coupled Jacobian.

    Independent of the code under test: it assembles the coupled Jacobian as a
    PETSc `nest` (the one matrix type that tolerates the `Real` blocks), reads
    every sub-block into a dense monolithic array with the same strong boundary
    conditions the solve applies, and eliminates block 0 exactly.
    """
    J = derivative(solver.F, z, TrialFunction(Z))
    Jn = fd.assemble(J, bcs=solver.strong_bcs, mat_type="nest").petscmat

    nfields = len(Z)
    reals = [i for i, V in enumerate(Z)
             if V.ufl_element().family() == "Real"]
    block0 = [i for i in range(nfields) if i not in reals]

    row_is, _ = Jn.getNestISs()
    sizes = [iset.getSize() for iset in row_is]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    N = int(starts[-1])
    full = np.zeros((N, N))
    for i in range(nfields):
        for j in range(nfields):
            block = Jn.getNestSubMatrix(i, j)
            if block is None:
                continue
            block.assemble()
            full[starts[i]:starts[i + 1], starts[j]:starts[j + 1]] = \
                block.convert("dense").getDenseArray()

    def idx(fields):
        return np.concatenate(
            [np.arange(starts[f], starts[f + 1]) for f in fields])

    i0, i1 = idx(block0), idx(reals)
    A00 = full[np.ix_(i0, i0)]
    B = full[np.ix_(i0, i1)]
    C = full[np.ix_(i1, i0)]
    D = full[np.ix_(i1, i1)]
    return D - C @ np.linalg.solve(A00, B)


@pytest.fixture(scope="module")
def built(meshes):
    """Solve once with the dense-Schur PC; return the PC's `S` and the reference.

    Module-scoped: the solve and the reference assembly are the expensive part,
    so T1-T4 share this one build.
    """
    solver, z, layout, Z = _build_solver(meshes)
    solver.solve()
    ksp_real, ctx = _dense_ctx(solver)
    S_ref = _reference_S(solver, z, Z)
    return {
        "solver": solver, "z": z, "layout": layout, "Z": Z,
        "ksp_real": ksp_real, "ctx": ctx,
        "S_pc": ctx._S.copy(), "S_ref": S_ref,
    }


def _apply_pc(petsc_pc, ctx, rhs, transpose=False):
    """Run the PC's `apply`/`applyTranspose` through real PETSc vectors."""
    A = petsc_pc.getOperators()[0]
    x, y = A.createVecRight(), A.createVecLeft()
    lo, hi = x.owner_range
    x.array_w[:] = np.asarray(rhs)[lo:hi]
    x.assemble()
    if transpose:
        ctx.applyTranspose(petsc_pc, x, y)
    else:
        ctx.apply(petsc_pc, x, y)
    out = ctx._gather(petsc_pc.comm.tompi4py(), y, ctx._n)
    x.destroy()
    y.destroy()
    return out


# ---------------------------------------------------------------------------
# T1  parity: the PC's S equals the independent reference S_ref
# ---------------------------------------------------------------------------
def test_T1_parity_against_independent_reference(built):
    """Binding correctness test.

    FAILS IF the PC forms the wrong complement -- a wrong block partition, a
    stale block-0 inverse, a missing MatMult column. FLOOR: both sides do an
    exact 13x13 elimination, so the relative difference is round-off times the
    condition number, ~1e-13; the gate is 1e-10.
    """
    S_pc, S_ref = built["S_pc"], built["S_ref"]
    rel = np.linalg.norm(S_pc - S_ref) / np.linalg.norm(S_ref)
    assert rel < PARITY_FLOOR, f"S vs S_ref relative error {rel:.3e}"


# ---------------------------------------------------------------------------
# T2  forward solve: apply solves S x = rhs
# ---------------------------------------------------------------------------
def test_T2_forward_solve(built):
    """`apply` must solve `S x = rhs`, checked against the independent `S_ref`.

    FAILS IF the stored factors invert the wrong matrix or the gather/scatter
    drops entries. FLOOR: 13x13 LU solve, ~1e-13 relative; gate 1e-10.
    """
    ctx, ksp_real, S_ref = built["ctx"], built["ksp_real"], built["S_ref"]
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal(ctx._n)
    x = _apply_pc(ksp_real.getPC(), ctx, rhs)
    residual = np.linalg.norm(S_ref @ x - rhs) / np.linalg.norm(rhs)
    assert residual < PARITY_FLOOR, f"||S x - rhs|| / ||rhs|| = {residual:.3e}"
    # And the solution itself matches the reference inverse.
    ref = np.linalg.solve(S_ref, rhs)
    assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < PARITY_FLOOR


# ---------------------------------------------------------------------------
# T3  transpose discriminator: applyTranspose solves S^T x = rhs, not S x = rhs
# ---------------------------------------------------------------------------
def test_T3_transpose_solves_S_transpose(built):
    """`applyTranspose` must solve `S^T x = rhs`.

    Because `S` is asymmetric by construction, solving `S^T` gives a DIFFERENT
    answer from solving `S` on a generic rhs. This test therefore does two
    things: it checks `applyTranspose` against `inv(S_ref.T)` to roundoff, and
    it asserts the transpose answer differs from the forward answer by a WIDE
    margin -- which is exactly what the old `applyTranspose = apply` base-class
    behaviour got wrong, so this test FAILS against that behaviour.

    FLOOR: the match to `inv(S_ref.T)` is ~1e-13 (gate 1e-10). The
    forward-vs-transpose margin is set by the operator's asymmetry, not by
    round-off, and the assertion floor is 1e-3 -- ten orders above round-off.
    """
    ctx, ksp_real, S_ref = built["ctx"], built["ksp_real"], built["S_ref"]
    rng = np.random.default_rng(11)
    rhs = rng.standard_normal(ctx._n)

    xt = _apply_pc(ksp_real.getPC(), ctx, rhs, transpose=True)
    ref_t = np.linalg.solve(S_ref.T, rhs)
    match = np.linalg.norm(xt - ref_t) / np.linalg.norm(ref_t)
    assert match < PARITY_FLOOR, \
        f"applyTranspose vs inv(S_ref.T) rhs = {match:.3e}"

    # The discriminator. If applyTranspose secretly solved S (the old
    # behaviour), it would equal the forward answer and this margin would be 0.
    xf = _apply_pc(ksp_real.getPC(), ctx, rhs, transpose=False)
    margin = np.linalg.norm(xt - xf) / np.linalg.norm(xf)
    asym = np.abs(ctx._S - ctx._S.T).max() / np.abs(ctx._S).max()
    assert margin > 1e-3, (
        f"S^T and S solves differ by only {margin:.3e} (asymmetry {asym:.3e}); "
        "the transpose is not being taken")


# ---------------------------------------------------------------------------
# T4  build-once: initialize runs once, no rebuild on the second solve
# ---------------------------------------------------------------------------
def test_T4_build_once(built):
    """The complement is built in `initialize` and never rebuilt.

    `PCBase.setUp` dispatches to `initialize` the first time and to `update` (a
    no-op) thereafter, so a second solve must not replace the cached factors.

    FAILS IF `update` rebuilds, or the build moved into a `setUp` override that
    PETSc re-invokes every solve. CHECK: object identity of the cached array and
    the cached LU factors is unchanged across a second solve.
    """
    solver, ctx = built["solver"], built["ctx"]
    S_id = id(ctx._S)
    factor_id = id(ctx._lu if hasattr(ctx, "_lu") else ctx._inv)

    solver.solve()  # a second solve at the same, constant operator

    ksp_real, ctx2 = _dense_ctx(solver)
    assert ctx2 is ctx, "the PC instance was rebuilt across solves"
    assert id(ctx2._S) == S_id, "S was rebuilt on the second solve"
    assert id(ctx2._lu if hasattr(ctx2, "_lu") else ctx2._inv) == factor_id, \
        "the factors were rebuilt on the second solve"


def test_T4b_dt_change_rebuilds_and_matches_reference(meshes):
    """A live `dt.assign(...)` must rebuild the cached dense complement.

    The solver object and state stay live. The new factor must match an
    independent elimination of the Jacobian at the new time step.
    """
    dt = fd.Constant(1.0)
    solver, z, layout, Z = _build_solver(meshes, dt=dt)
    solver.solve()
    _, ctx = _dense_ctx(solver)
    old_S_id = id(ctx._S)
    old_S = ctx._S.copy()

    dt.assign(2.0)
    solver.solve()
    _, ctx2 = _dense_ctx(solver)
    S_ref = _reference_S(solver, z, Z)

    assert ctx2 is ctx, "the PC instance changed after dt.assign"
    assert id(ctx2._S) != old_S_id, "the dense complement did not rebuild"
    assert not np.allclose(ctx2._S, old_S), "the dt change left S unchanged"
    rel = np.linalg.norm(ctx2._S - S_ref) / np.linalg.norm(S_ref)
    assert rel < PARITY_FLOOR, f"rebuilt S vs S_ref error {rel:.3e}"


def test_T4c_fluid_core_pressure_joins_the_dense_complement(meshes):
    """The physical zero diagonal gains its inverse through the Schur term."""
    solver, z, layout, Z = _build_solver(meshes, fluid_core=True)
    solver.solve()
    _, ctx = _dense_ctx(solver)
    reference = _reference_S(solver, z, Z)
    relative = np.linalg.norm(ctx._S - reference) / np.linalg.norm(reference)

    assert layout.core_pressure is not None
    assert ctx._n == len(layout.real_fields)
    assert relative < PARITY_FLOOR
    assert np.linalg.matrix_rank(ctx._S) == ctx._n

    dss = solver.fluid_core_measure()(CURVE_RC)
    flux = fd.assemble(fd.dot(
        solver.displacement, fd.FacetNormal(solver.mesh)) * dss)
    assert abs(flux) < 1e-12


# ---------------------------------------------------------------------------
# T5  no regression: the diagonal PC still solves on the same fixture
# ---------------------------------------------------------------------------
def test_T5_diagonal_pc_transpose_unchanged():
    """`DtNMultiplierDiagPC` is symmetric, so its transpose must equal its solve.

    The base class now routes `applyTranspose` through `_solve_transpose`, whose
    default forwards to `_solve`. For the diagonal PC -- whose block is diagonal,
    hence symmetric -- `apply` and `applyTranspose` must therefore give the
    IDENTICAL result. This is the backward-compatibility guarantee of the
    base-class change, checked directly rather than through an outer solve.

    (The broad regression for the diagonal PC is `test_dtn_multiplier_pc.py`,
    run alongside this file; here we pin only the one behaviour the base change
    could have altered.)

    FAILS IF the transpose hook made the diagonal PC's transpose differ from its
    forward solve. FLOOR: exact, bit-for-bit equality -- both are `rhs / d`.
    """
    from gadopt import DtNMultiplierDiagPC

    rng = np.random.default_rng(3)
    n = 9
    d = rng.standard_normal(n) + 2.0  # strictly nonzero diagonal
    A = PETSc.Mat().createDense((n, n), array=np.diag(d),
                                comm=PETSc.COMM_SELF)
    A.assemble()
    pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
    pc.setType("python")
    ctx = DtNMultiplierDiagPC()
    pc.setPythonContext(ctx)
    pc.setOperators(A, A)
    ctx.get_appctx = staticmethod(lambda _pc: {"dtn_block1_diagonal": d})
    ctx.initialize(pc)

    rhs = rng.standard_normal(n)
    forward = _apply_pc(pc, ctx, rhs, transpose=False)
    transpose = _apply_pc(pc, ctx, rhs, transpose=True)
    assert np.array_equal(forward, transpose), \
        "diagonal PC transpose diverged from its forward solve"
    assert np.allclose(forward, rhs / d, rtol=1e-14)


# ---------------------------------------------------------------------------
# T6  loud failure: singular or mis-sized S raises a NAMED error via _loud
# ---------------------------------------------------------------------------
def _pc_on_dense(array):
    """A python PC carrying a `DtNMultiplierDenseSchurPC`, over a dense Amat."""
    array = np.asarray(array, dtype=float)
    m, n = array.shape
    A = PETSc.Mat().createDense((m, n), array=array, comm=PETSc.COMM_SELF)
    A.assemble()
    pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
    pc.setType("python")
    ctx = DtNMultiplierDenseSchurPC()
    # This standalone PETSc PC has no Firedrake DM or application context.
    ctx._current_time_step = lambda _pc: None
    pc.setPythonContext(ctx)
    pc.setOperators(A, A)
    return pc, ctx


def test_T6_singular_raises_named(capsys):
    """A singular complement raises a NAMED `ValueError`, never a bare 101.

    FAILS IF the singular matrix reaches a solve as a silent NaN, or the error
    text never leaves the python PC. CHECK: the raised type is `ValueError`, its
    message names the complement as singular, and `_loud` has already written
    that message to stderr (so it sits above any eventual PETSc 101).
    """
    pc, ctx = _pc_on_dense(np.ones((4, 4)))  # rank 1, hence singular
    with pytest.raises(ValueError, match="singular"):
        ctx.initialize(pc)
    assert "singular" in capsys.readouterr().err


def test_T6_missized_raises_named(capsys):
    """A non-square Amat raises a NAMED `ValueError` before any MatMult.

    FAILS IF the size mismatch surfaces as a bare 101 naming nothing. CHECK: the
    message names the operator as non-square and reaches stderr via `_loud`.
    """
    pc, ctx = _pc_on_dense(np.ones((4, 3)))
    with pytest.raises(ValueError, match="square"):
        ctx.initialize(pc)
    assert "square" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# T7  parallel: the build keys on the GLOBAL size, on every rank
#
# This is the test the serial suite structurally cannot be. Every Real dof sits
# on one rank, so on a 2-rank run the block-1 LOCAL column count is the whole
# block on the owning rank and 0 on the other. The PC builds S redundantly on
# every rank (`_gather` Allreduces a global-length buffer; `_solve` runs the
# dense factor everywhere), so it must key the build on the GLOBAL size.
#
# Keyed on the local size instead, the non-owning rank forms a 0x0 S, skips the
# COLLECTIVE MatMult loop, and raises "zero-size array to reduction" -- while the
# owning rank blocks inside that same collective. That is the exact failure a
# 104-rank Gadi run produced (Unhandled Python Exception on every rank but 0),
# and it is invisible to every test above because a serial run has local ==
# global. Module-level function, not a method: mpi-pytest cannot relaunch a test
# that lives inside a class.
# ---------------------------------------------------------------------------
@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("fluid_core", [False, True])
def test_T7_parallel_build_keys_on_global_size(meshes, fluid_core):
    """The dense complement builds on every rank at the GLOBAL multiplier count.

    FAILS (crashes/hangs) IF the build keys on the local size: the rank owning no
    Real dofs raises inside a collective. CHECK, after a real 2-rank solve: the
    build did not raise, `ctx._n` is identical and non-zero on both ranks, `S` is
    square at that size, and the two ranks hold the SAME `S` to round-off (the
    redundant build agrees). FLOOR: the cross-rank `S` difference is round-off,
    ~1e-13; gate 1e-10.
    """
    comm = fd.COMM_WORLD
    assert comm.size == 2, "T7 must run on exactly 2 ranks"

    solver, z, layout, Z = _build_solver(meshes, fluid_core=fluid_core)
    # If the build keyed on the local size, this solve raises on the rank that
    # owns no Real dofs (and the other blocks in the collective MatMult).
    solver.solve()
    _, ctx = _dense_ctx(solver)

    # ctx._n is the global size, so it must agree across ranks and be > 0 on both
    # -- the local-keyed bug gives 13 on one rank and 0 on the other.
    ns = comm.allgather(ctx._n)
    assert ns[0] == ns[1] > 0, f"per-rank block-1 size disagrees: {ns}"
    assert ctx._S.shape == (ctx._n, ctx._n)

    # The redundant build must produce the SAME S on both ranks.
    norms = comm.allgather(float(np.linalg.norm(ctx._S)))
    assert norms[0] == pytest.approx(norms[1], rel=0.0, abs=0.0), \
        f"the two ranks built different S (norms {norms})"
    other = comm.bcast(ctx._S.copy(), root=1)
    rel = np.linalg.norm(ctx._S - other) / np.linalg.norm(ctx._S)
    assert rel < PARITY_FLOOR, f"S differs across ranks by {rel:.3e}"
