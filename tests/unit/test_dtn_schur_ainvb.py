r"""Gate for the two-block apply that caches `Z = A00^{-1} A01`.

Design: `NOTES/team/rotation-pc/11-DESIGN-AINVB.md`. `DtNTwoBlockSchurPC` gains
one boolean option, `dtn_schur_ainvb`. With the option on the class owns the
apply instead of delegating to its inner `PCFIELDSPLIT`: at every operator
change it runs `n` block-0 solves, one per `Real` row, and keeps from those
same solves both the columns `Z = A00^{-1} A01` and the exact `n x n`
complement `S = A11 - A10 Z`. Each forward apply then costs ONE block-0 solve
instead of the two that `pc_fieldsplit_schur_fact_type full` pays.

## Why almost every test here counts something

**A preconditioner that is wrong still converges to the right answer.** It
changes no residual, so a test that compares states passes whether or not the
cached path works, and the defect this project pays for is a cost with no error
message: `tests/unit/test_gia_lowrank_block0.py` exists because exactly that
cost 8 outer iterations where 3 were needed and printed nothing. So the tests
below assert on exact algebraic identities (on the fixture whose block 0 is a
direct LU, where the preconditioned operator must be the identity) and on
counts of block-0 solves (on the fixture that runs the production route).

Each test says in its docstring which wrong implementation it catches.

## The two fixtures, and why each section uses the one it does

- **`direct`**: the 2-D annulus under `selfgrav_dtn_schur_solver_parameters`,
  whose block 0 is `preonly` + assembled LU. `M0 = A00^{-1}` is then exact and
  linear, so `S` is exact, `P^{-1} A = I` holds to round-off, and the outer
  FGMRES must converge in one iteration. This is where the algebra is pinned.
  `n = 13` `Real` rows (the DtN multipliers of the multiplier representation).
- **`iterative`**: the same annulus under
  `selfgrav_dtn_iterative_solver_parameters(condensed=False,
  block0="condensed", dtn_representation="lowrank")`, whose block 0 is
  `gadopt.CondensedBlockPC`. That class counts its own eliminations, one per
  block-0 application, which makes it the independent witness for every count
  identity. `n = 2` `Real` rows, the same structure as production's 4.

**A 2-D annulus, and the 24-cell sphere at the end, establish that the
construction works, never that it helps.** Nothing here measures wall clock,
sweeps or speed; every performance number comes from Gadi.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gadopt  # noqa: F401  - before firedrake, as the drivers do
import firedrake as fd
from firedrake.petsc import PETSc

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_dtn_multiplier_dense_schur as td  # noqa: E402
import test_gia_gravity_adjoint_lowrank as tl  # noqa: E402
import test_gia_lowrank_block0 as tb  # noqa: E402
from gadopt import CompressibleInternalVariableApproximation  # noqa: E402
from gadopt import CylindricalDtN, SphericalDtN  # noqa: E402
from gadopt.gia_gravity import (  # noqa: E402
    FluidCore, OMEGA_SQ_EARTH, SelfGravitatingGIASolver,
    self_gravitating_gia_space, selfgrav_dtn_iterative_solver_parameters,
    selfgrav_dtn_schur_solver_parameters)
from gadopt.stokes_integrators import newton_stokes_solver_parameters  # noqa: E402


# ---------------------------------------------------------------------------
# Numerical floors. Every one of them is measured on the equivalent EXACT arm
# that exists today -- `fact_type full` with `gadopt.DtNMultiplierDenseSchurPC`
# on a `preonly` block 1, which is algebraically the same preconditioner the
# cached path must reproduce -- and then loosened by two orders.
# ---------------------------------------------------------------------------

#: Relative floor for `P^{-1} A x = x` and `P^{-T} A^T x = x` on the `direct`
#: fixture. Dimensionless (a ratio of vector 2-norms on the mixed state).
#: MEASURED on the exact arm: 1.5e-9 forward, 3.7e-9 transpose. The floor is
#: NOT round-off of the 3 277-dof system: it is the accuracy of the MUMPS LU of
#: block 0 against an operator whose rows span several orders of magnitude
#: (the `Real` constraint rows carry `theta_psi`, the rotation rows `Omega_sq =
#: 1.566e-3`). 1e-7 sits two orders above what is measured and nine orders
#: below the 2.5 that today's delegating path gives on the same vector.
IDENTITY_FLOOR = 1e-7

#: Relative floor for two dense linear-algebra objects that must agree to
#: round-off: the cached `S` against an independent elimination of the
#: assembled Jacobian, and the adjoint identity `<P^{-1}a, b> = <a, P^{-T}b>`.
#: MEASURED: the adjoint identity holds to 4e-15 on the exact arm, and
#: `test_dtn_multiplier_dense_schur.PARITY_FLOOR` uses 1e-10 for the same
#: elimination. Kept at 1e-10 for both, five orders above what is measured.
PARITY_FLOOR = 1e-10

#: How far apart the forward and the transpose apply must sit on one vector.
#: This is the vacuity guard for the transpose test: if the operator were
#: symmetric, a transpose apply that secretly ran the forward path would pass.
#: MEASURED on the exact arm: 8.6e-4 relative. The gate is 1e-5, two orders
#: below what is measured and ten orders above the 4e-15 round-off floor.
TRANSPOSE_MARGIN = 1e-5


# ---------------------------------------------------------------------------
# Fixtures and accessors
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def meshes():
    """The annulus pair every GIA unit test uses, built once for this module.

    The `.msh` file is cached in `/tmp` under the same name the other suites
    use, so it is generated at most once across the whole test run.
    """
    return tl.meshes.__wrapped__()


def ainvb_cache(pc):
    """The `Z`/`S` cache the option installs on a `DtNTwoBlockSchurPC`.

    Reached through the PETSc object the solve actually ran, never constructed
    by hand, so that a test checks the object PETSc built.

    Args:
      pc: the outer PETSc `PC` (python type `gadopt.DtNTwoBlockSchurPC`).

    Returns:
      The cache object. The design (section 7.1) names it `ainvb` on the
      python context and requires it to expose `n`, `S`, `Z`, `build_count`,
      `apply_count` and `apply_transpose_count`.
    """
    context = pc.getPythonContext()
    cache = getattr(context, "ainvb", None)
    assert cache is not None, (
        "the outer DtNTwoBlockSchurPC carries no `ainvb` cache, so the option "
        "dtn_schur_ainvb did not select the cached apply path")
    return cache


def outer_pc(solver):
    """The outer python PC of a `SelfGravitatingGIASolver`."""
    return solver.solver.snes.ksp.pc


def outer_iterations(solver):
    """Outer FGMRES iterations of the last solve."""
    return solver.solver.snes.ksp.getIterationNumber()


# ---------------------------------------------------------------------------
# The `direct` fixture: exact block 0, so the algebra can be pinned exactly
# ---------------------------------------------------------------------------
def direct_params(*, ainvb=True, **overrides):
    """The 2-D `direct` Schur preset, with block 1 left to the cached factors.

    Block 0 is `preonly` + `firedrake.AssembledPC` + MUMPS LU, so `M0` is an
    exact linear operator; block 1 is `preonly` + `none`, because under the
    option nothing solves it -- the cache applies the factors of the exact
    complement itself. The option is written as the raw PETSc key rather than
    through the preset, so that a change to the preset's signature cannot stop
    this fixture from building.

    Args:
      ainvb: whether to write the `dtn_schur_ainvb` key.
      overrides: extra solver-parameter entries, for the refusal tests.

    Returns:
      The solver-parameter dictionary.
    """
    params = dict(newton_stokes_solver_parameters)
    params.update(selfgrav_dtn_schur_solver_parameters)
    # The true-residual monitor is pure noise in a test log; drop it.
    params.pop("ksp_monitor_true_residual", None)
    # Block 1 is never solved on the cached path, and the design refuses any
    # other setting there (section 5.2), so the fixture writes what it asks
    # for and the refusal tests write what it refuses.
    params["dtn_fieldsplit_1_ksp_type"] = "preonly"
    params["dtn_fieldsplit_1_pc_type"] = "none"
    params.pop("dtn_fieldsplit_1_ksp_rtol", None)
    params.pop("dtn_fieldsplit_1_ksp_max_it", None)
    if ainvb:
        params["dtn_schur_ainvb"] = True
    params.update(overrides)
    return params


def build_direct(meshes, params, dt=1.0):
    """The coupled solver on the annulus under a `direct`-style dictionary.

    Mirrors `test_dtn_multiplier_dense_schur._build_solver` -- same geometry,
    same constants, same space -- and differs only in taking the solver
    parameters as an argument, because this file drives several arms of the
    same fixture.

    Args:
      meshes: the `(parent, sub)` pair.
      params: the solver-parameter dictionary.
      dt: the time step, a float or a live `Constant`.

    Returns:
      `(solver, z, layout, Z)`.
    """
    parent, sub = meshes
    Xp = fd.SpatialCoordinate(parent)
    Xm = fd.SpatialCoordinate(sub)
    gravity_bcs = {
        td.CURVE_OUTER: {"dtn": CylindricalDtN(td.TRUNCATION)},
        td.CURVE_INNER: {"dtn": CylindricalDtN(td.TRUNCATION)},
        td.CURVE_RE: {"interior_sigma":
                      td.SIGMA_HAT * fd.cos(2 * fd.atan2(Xp[1], Xp[0]))},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=False,
        fluid_core=False, n_internal_variables=1,
        condense_internal_variables=True, self_gravity_number=td.LAMBDA)
    z = fd.Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        bulk_shear_ratio=100.0, g=td.G0, B_mu=td.B_MU,
        self_gravity_number=td.LAMBDA)

    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    mechanics_bcs = {
        td.CURVE_RE: {"normal_stress":
                      fd.Constant(td.B_MU) * td.SIGMA_HAT
                      * fd.cos(2 * fd.atan2(Xm[1], Xm[0]))},
        td.CURVE_RC: {"un": 0.0},
    }
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=dt, bcs=mechanics_bcs,
        rotation_moments={"C": fd.assemble(fd.dot(Xm, Xm) * dx_m)},
        Omega_sq=OMEGA_SQ_EARTH, solver_parameters=params)
    return solver, z, layout, Z


@pytest.fixture(scope="module")
def direct(meshes):
    """One solve of the exact arm, shared by every test of section A.

    The solve and the reference elimination are the expensive part of this
    section, so the seven tests below share one build.
    """
    solver, z, layout, Z = build_direct(meshes, direct_params())
    solver.solve()
    return {"solver": solver, "z": z, "layout": layout, "Z": Z,
            "pc": outer_pc(solver),
            "S_ref": td._reference_S(solver, z, Z)}


def monolithic_field_indices(Z):
    """The block-0 and block-1 row indices of the mixed space, in field order.

    Read from `Z.dof_dset.field_ises`, which is Firedrake's own authority on
    where each sub-field lives in the monolithic row space and the very object
    `DtNTwoBlockSchurPC.initialize` merges into its two index sets. Reading it
    here rather than from the preconditioner is what makes the column test
    below independent of the code under test.

    Args:
      Z: the mixed function space.

    Returns:
      `(i0, i1)`, the non-`Real` and `Real` global row indices.
    """
    reals = [i for i, V in enumerate(Z)
             if V.ufl_element().family() == "Real"]
    ises = Z.dof_dset.field_ises
    block0 = [i for i in range(len(Z)) if i not in reals]
    i0 = np.concatenate([ises[i].getIndices() for i in block0])
    i1 = np.concatenate([ises[i].getIndices() for i in reals])
    return i0, i1


def identity_residual(pc, seed):
    """`||P^{-1} A x - x|| / ||x||` for one pseudo-random `x`.

    On a fixture whose block 0 is an exact LU this is the whole of the forward
    apply in one number: with an exact `M0` and an exact `S` the `full`
    factorisation is the operator's inverse, so the preconditioned operator is
    the identity and this quantity is round-off.

    Args:
      pc: the outer PETSc `PC`.
      seed: the seed of the random vector, so that two callers can use
        different vectors and a failure is reproducible.

    Returns:
      The relative 2-norm above. Dimensionless.
    """
    A, _ = pc.getOperators()
    rng = np.random.default_rng(seed)
    x = A.createVecRight()
    lo, hi = x.owner_range
    x.array_w[:] = rng.standard_normal(hi - lo)
    x.assemble()
    b = A.createVecLeft()
    A.mult(x, b)
    y = A.createVecRight()
    pc.apply(b, y)
    y.axpy(-1.0, x)
    relative = y.norm() / x.norm()
    x.destroy()
    b.destroy()
    y.destroy()
    return relative


# ===========================================================================
# A. The exact arm: the algebra, pinned to round-off
# ===========================================================================
def test_the_cached_complement_matches_an_independent_elimination(direct):
    """`S` is `A11 - A10 A00^{-1} A01` and nothing else.

    The reference is `test_dtn_multiplier_dense_schur._reference_S`, which
    assembles the coupled Jacobian as a PETSc `nest`, reads its dense blocks
    and eliminates block 0 exactly. Nothing of the code under test enters it.

    CATCHES: a wrong block partition; a dropped or sign-flipped `A11 e_k` term
    (the design forms `S[:, k] = A11 e_k - A10 z_k` by hand instead of letting
    `MatMult_SchurComplement` do it, so that term is newly hand-written and can
    be lost); columns built against the wrong right-hand side.
    FLOOR: `PARITY_FLOOR`, see its definition.
    """
    cache = ainvb_cache(direct["pc"])
    S = np.asarray(cache.S)
    S_ref = direct["S_ref"]
    assert S.shape == S_ref.shape, (
        f"the cache holds a {S.shape} complement where the Real block is "
        f"{S_ref.shape}")
    relative = np.linalg.norm(S - S_ref) / np.linalg.norm(S_ref)
    assert relative < PARITY_FLOOR, f"S vs S_ref relative error {relative:.3e}"


def test_the_cached_columns_solve_the_block_zero_system(direct):
    """Column `k` of `Z` satisfies `A00 z_k = A01 e_k`, for every `k`.

    Written as an identity on the MONOLITHIC operator so that it assumes
    nothing about how block-0 degrees of freedom are ordered inside the
    fieldsplit's index set. Embed `z_k` in the mixed vector with a zero `Real`
    part: the block-0 part of `A` applied to it is `A00 z_k`. Embed the unit
    `Real` vector `e_k`: the block-0 part of `A e_k` is `A01 e_k`. The two must
    agree.

    CATCHES: columns solved against the wrong right-hand side (for instance
    `A10^T e_k` instead of `A01 e_k`); columns stored in the wrong order, which
    would pair `z_k` with the wrong entry of the block-1 solution in step 7 of
    the forward apply and is invisible to the `S` test above because `S` is
    built from the same loop; a `Z` left at the values of an earlier operator.
    FLOOR: `IDENTITY_FLOOR`, because one block-0 LU solve sits behind each
    column and that is what the exact arm measures.
    """
    if fd.COMM_WORLD.size > 1:
        pytest.skip("the index arithmetic below is written for one rank")
    cache = ainvb_cache(direct["pc"])
    A, _ = direct["pc"].getOperators()
    i0, i1 = monolithic_field_indices(direct["Z"])
    assert len(cache.Z) == len(i1), (
        f"the cache holds {len(cache.Z)} columns for {len(i1)} Real rows")

    work = A.createVecRight()
    image = A.createVecLeft()
    unit = A.createVecRight()
    unit_image = A.createVecLeft()
    scale = 0.0
    errors = []
    for k, column in enumerate(cache.Z):
        # A00 z_k, read off the monolithic action on (z_k, 0).
        work.set(0.0)
        work.setValues(i0, column.array_r)
        work.assemble()
        A.mult(work, image)
        # A01 e_k, read off the monolithic action on (0, e_k).
        unit.set(0.0)
        unit.setValues([i1[k]], [1.0])
        unit.assemble()
        A.mult(unit, unit_image)
        reference = unit_image.array_r[i0]
        scale = max(scale, float(np.linalg.norm(reference)))
        errors.append(float(np.linalg.norm(image.array_r[i0] - reference)))

    # Vacuity guard: if every A01 column were zero, every z_k would trivially
    # be zero and the loop above would assert nothing.
    assert scale > 0.0, "A01 is identically zero, so this test is vacuous"
    assert max(errors) / scale < IDENTITY_FLOOR, (
        f"||A00 z_k - A01 e_k|| / max_k ||A01 e_k|| = {max(errors)/scale:.3e}")


def test_the_forward_apply_inverts_the_operator(direct):
    """`P^{-1} A x = x`: with an exact `M0` and an exact `S`, `full` is exact.

    This is the whole of the forward apply (design section 4.2) in one
    assertion, on the fixture where every approximation has been removed.

    CATCHES any step of the forward apply: a missing lower-triangular
    correction `x1 - A10 y0`, a `+Z s1` where `-Z s1` is meant, a `VecMAXPY`
    over the wrong columns, a solve of `S` where `S^{-1}` was already applied,
    a scatter that leaves part of `y` unwritten (PETSc does not guarantee `y`
    is zero on entry).
    FLOOR: `IDENTITY_FLOOR`. MEASURED on the equivalent exact arm that exists
    today (`full` + `gadopt.DtNMultiplierDenseSchurPC` + `preonly` block 1):
    1.5e-9.
    """
    relative = identity_residual(direct["pc"], seed=0)
    assert relative < IDENTITY_FLOOR, (
        f"||P^-1 A x - x|| / ||x|| = {relative:.3e}")


def test_a_rebuilt_cache_still_inverts_the_operator(meshes):
    """The SECOND build is as good as the first: `P^{-1} A = I` after `dt`.

    Three objects have to move together at a rebuild -- the `n` vectors of
    `Z`, the array `S` and the LU factors of `S`. An implementation that
    refreshes one and not another increments `build_count`, converges to the
    right answer, and costs outer iterations for the rest of a march in which
    every build after the first is a rebuild. Every other test of the
    arithmetic in this section runs on a cache that was built exactly once, so
    this is the only one that can see it.

    The `dt` change is what makes the second build a different operator: the
    effective bulk/shear ratio is `bulk_shear_ratio * (1 + dt/tau)`, so a new
    time step is a new mechanics block, and the stored columns of the old one
    describe an operator that no longer exists.

    CATCHES: a `Z` list appended to instead of replaced; LU factors kept from
    the previous array; a new `S` written into a factorisation that holds its
    own copy; a rebuild that refreshes `S` and leaves `Z` alone, or the
    reverse. All of them at once, because with an exact block-0 LU the
    identity holds only if all three describe the same operator.

    The `build_count` assertion is the vacuity guard. Without it a cache that
    never rebuilt at all would be indistinguishable from one that rebuilt
    wrongly, and a `dt` change that somehow left the operator alone would make
    the test pass for no reason.

    FLOOR: `IDENTITY_FLOOR`, the same number and the same reason as the test
    above. `test_T4b_dt_change_rebuilds_and_matches_reference` in
    `test_dtn_multiplier_dense_schur.py` is the precedent: the dense PC
    carries this test for this reason.
    """
    dt = fd.Constant(1.0)
    solver, _, _, _ = build_direct(meshes, direct_params(), dt=dt)
    solver.solve()
    # A live re-assignment, with the solver and its state untouched, which is
    # what a march does between steps.
    dt.assign(2.0)
    solver.solve()

    pc = outer_pc(solver)
    cache = ainvb_cache(pc)
    assert cache.build_count == 2, (
        f"the time step changed and the cache built {cache.build_count} "
        "time(s); the identity below would then say nothing about a rebuild")
    relative = identity_residual(pc, seed=4)
    assert relative < IDENTITY_FLOOR, (
        f"after the rebuild, ||P^-1 A x - x|| / ||x|| = {relative:.3e}")


def test_a_capped_block_zero_solve_leaves_the_next_step_solvable(meshes):
    """A block-0 solve that stops at its iteration cap must not fail the PC.

    Block 0 is an inner solve of the preconditioner, so the preset gives it a
    budget (`block0_max_it`, 200 in production) and expects it to be spent: the
    campaign record measures the cap reached on 45 percent of the
    mechanics-residual solves at dt = 100 yr. The outer FGMRES is flexible, so
    an inexact block-0 solve costs outer iterations and nothing else.

    CATCHES: a `_check_block0` that mirrors `KSPCheckSolve` without its
    `KSP_DIVERGED_ITS` exclusion. PETSc's own check reads `reason < 0 && reason
    != KSP_DIVERGED_ITS` (`src/ksp/ksp/interface/iterativ.c`); a check that
    drops the second half calls `pc.setFailedReason(SUBPC_ERROR)` on a capped
    solve, and the flag is STICKY, because `PCSetUp` never clears
    `pc->failedreason` and `KSPSetUp` reads it
    (`src/ksp/ksp/interface/itfunc.c`). The failure is therefore one step late
    and unreadable: the step that hit the cap converges, and the NEXT step
    raises `ConvergenceError` after zero nonlinear iterations with a message
    that names nothing about block 0. Against the code before the exclusion
    this test raises at step 1.

    Block 0 is forced to its cap rather than allowed to reach it: one FGMRES
    iteration against `rtol` 1e-30 caps every single block-0 solve, which is
    the same converged reason (-3) the production route reaches at 200. The
    `dt.assign` between the solves is what a march does, and it is also what
    rebuilds the cache, so the build's own block-0 solves are capped too and
    the check in `build` is exercised on the same path.

    This test asserts on survival and on the flag, not on an identity: with
    block 0 solved to one FGMRES iteration the preconditioner is poor, and the
    only claim is that a poor preconditioner is not reported as a broken one.
    """
    dt = fd.Constant(1.0)
    params = direct_params(**{"dtn_fieldsplit_0_ksp_type": "fgmres",
                              "dtn_fieldsplit_0_ksp_max_it": 1,
                              "dtn_fieldsplit_0_ksp_rtol": 1e-30})
    solver, _, _, _ = build_direct(meshes, params, dt=dt)

    for step, value in enumerate((1.0, 2.0, 3.0)):
        dt.assign(value)
        # A `ConvergenceError` here is the defect: Firedrake raises it when the
        # outer KSP exits `DIVERGED_PC_FAILED`, which is what `KSPSetUp` does
        # when it reads a failed reason left behind by an earlier step.
        solver.solve()
        pc = outer_pc(solver)
        assert pc.getFailedReason() == PETSc.PC.FailedReason.NOERROR, (
            f"step {step}: a capped block-0 solve left the outer PC flagged "
            f"as failed ({pc.getFailedReason()}); the next step would die at "
            "KSPSetUp with zero nonlinear iterations")

    # The vacuity guard. Without it a block 0 that converged inside its one
    # iteration would make every assertion above pass for the wrong reason.
    cache = ainvb_cache(outer_pc(solver))
    assert cache.ksp0.getConvergedReason() \
        == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT, (
        "block 0 was meant to stop at its cap on every solve, and its last "
        f"converged reason is {cache.ksp0.getConvergedReason()}")


def test_the_transpose_apply_inverts_the_transposed_operator(direct):
    """`P^{-T} A^T x = x`, the transpose of the same statement.

    The transpose apply is a separate arithmetic (design section 4.3): the
    columns of `Z` enter through `VecMDot` instead of `VecMAXPY`, the dense
    factors are used with `trans=1`, `A10` through `multTranspose`, and block 0
    through `solveTranspose`. On this fixture block 0 is `preonly` + LU, which
    has a transpose solve, so the path is reachable.

    CATCHES: `VecMAXPY` where `VecMDot` is meant; the forward LU solve where
    the transposed one is meant; `A10.mult` where `A10.multTranspose` is meant;
    `ksp0.solve` where `ksp0.solveTranspose` is meant; the two transpose
    solves of PETSc's own path collapsed onto the wrong vector.
    FLOOR: `IDENTITY_FLOOR`. MEASURED on the exact arm: 3.7e-9.
    """
    pc = direct["pc"]
    A, _ = pc.getOperators()
    rng = np.random.default_rng(1)
    x = A.createVecRight()
    lo, hi = x.owner_range
    x.array_w[:] = rng.standard_normal(hi - lo)
    x.assemble()
    b = A.createVecLeft()
    A.multTranspose(x, b)

    y = A.createVecRight()
    pc.applyTranspose(b, y)
    y.axpy(-1.0, x)
    relative = y.norm() / x.norm()
    assert relative < IDENTITY_FLOOR, (
        f"||P^-T A^T x - x|| / ||x|| = {relative:.3e}")


def test_the_transpose_apply_is_not_the_forward_apply(direct):
    """The vacuity guard for the test above: `P^{-T}` differs from `P^{-1}`.

    If the preconditioner happened to be symmetric, an `applyTranspose` that
    simply ran the forward path would satisfy the identity above and the test
    would prove nothing. It is not symmetric -- `A01` and `A10` are different
    blocks and `S` carries the structural asymmetry the dense Schur PC's
    docstring records -- so the two answers must differ by a wide margin.

    CATCHES: `applyTranspose` delegating to `apply`, which is the single most
    likely wrong implementation and is silent, because a wrong transpose
    changes no residual either.
    FLOOR: `TRANSPOSE_MARGIN`. MEASURED on the exact arm: 8.6e-4.
    """
    # The claim is about the CACHED applies. Without this line the test passes
    # against the delegating path, whose transpose is PETSc's own and equally
    # asymmetric, and it would then gate nothing.
    ainvb_cache(direct["pc"])
    pc = direct["pc"]
    A, _ = pc.getOperators()
    rng = np.random.default_rng(2)
    b = A.createVecLeft()
    lo, hi = b.owner_range
    b.array_w[:] = rng.standard_normal(hi - lo)
    b.assemble()

    forward = A.createVecRight()
    pc.apply(b, forward)
    transpose = A.createVecRight()
    pc.applyTranspose(b, transpose)
    difference = forward.copy()
    difference.axpy(-1.0, transpose)
    margin = difference.norm() / forward.norm()
    assert margin > TRANSPOSE_MARGIN, (
        f"the forward and the transpose apply differ by only {margin:.3e}; "
        "the transpose is not being taken")


def test_the_forward_and_transpose_applies_are_adjoint(direct):
    """`<P^{-1} a, b> = <a, P^{-T} b>` for random `a` and `b`.

    Independent of the two identity tests: it compares the two applies against
    each other rather than each against the operator, and it holds to
    round-off rather than to the accuracy of the block-0 LU, because the same
    LU appears on both sides. So it is the sharpest statement about the
    transpose in this file, five orders sharper than `IDENTITY_FLOOR`.

    CATCHES: a transpose apply that is self-consistent but built from a
    different operator than the forward one -- for instance one that transposes
    `S` but not `A10`, or that uses a `Z` from a stale build.
    FLOOR: `PARITY_FLOOR`. MEASURED on the exact arm: 4.0e-15.
    """
    # As above: the identity holds for the delegating path too, so the test
    # must first say which apply it is about.
    ainvb_cache(direct["pc"])
    pc = direct["pc"]
    A, _ = pc.getOperators()
    rng = np.random.default_rng(3)
    lo, hi = A.createVecLeft().owner_range
    a = A.createVecLeft()
    a.array_w[:] = rng.standard_normal(hi - lo)
    a.assemble()
    b = A.createVecLeft()
    b.array_w[:] = rng.standard_normal(hi - lo)
    b.assemble()

    forward = A.createVecRight()
    pc.apply(a, forward)
    transpose = A.createVecRight()
    pc.applyTranspose(b, transpose)
    left = forward.dot(b)
    right = a.dot(transpose)
    assert abs(left - right) / abs(left) < PARITY_FLOOR, (
        f"<P^-1 a, b> = {left:.12e} but <a, P^-T b> = {right:.12e}")


def test_the_outer_krylov_converges_in_one_iteration(direct):
    """On the exact arm the preconditioned operator is the identity.

    This is the count that separates `full` from `lower`: `lower` leaves a
    rank-`n` nilpotent correction `I + N` and needs two iterations, `full`
    needs one. It also separates an exact block-1 solve from an
    unpreconditioned one, which on this fixture takes three.

    CATCHES: an apply that implements `lower`, `upper` or `diag` while
    claiming `full`; a block-1 solve that uses something other than the exact
    factors of `S`.
    MEASURED on the equivalent exact arm that exists today: 1, `CONVERGED_RTOL`.
    """
    assert outer_iterations(direct["solver"]) == 1


# ===========================================================================
# B. The production route in 2-D: counts of block-0 solves
# ===========================================================================
def iterative_params(*, ainvb, representation="lowrank", **kwargs):
    """The two-block iterative preset, in one of two arms.

    - `ainvb=True`: block 1 is `preonly` + `none` and the cached path owns the
      apply. This is the arm under test.
    - `ainvb=False`: block 1 is `preonly` + `gadopt.DtNMultiplierDenseSchurPC`,
      which is the SAME preconditioner algebraically -- `full` with the exact
      complement on block 1 -- reached by the route that exists today. It is
      the reference for the iteration-count and block-0-solve comparisons.

    `dtn_fieldsplit_1_ksp_type` is forced to `preonly` in both arms, because a
    Krylov method there would itself spend one block-0 solve per iteration and
    the count model would no longer be readable. The converged-reason keys are
    dropped: they make the log unreadable and nothing here parses it.

    Args:
      ainvb: which arm to build.
      representation: the DtN representation, `"lowrank"` by default because
        that is the production route and it gives a small `Real` block.
      kwargs: passed through to the preset.

    Returns:
      The solver-parameter dictionary.
    """
    params = tb.preset(
        representation,
        multiplier_pc=("none" if ainvb
                       else "gadopt.DtNMultiplierDenseSchurPC"),
        **kwargs)
    for key in list(params):
        if key.endswith("converged_reason"):
            params.pop(key)
    params["dtn_fieldsplit_1_ksp_type"] = "preonly"
    params.pop("dtn_fieldsplit_1_ksp_rtol", None)
    params.pop("dtn_fieldsplit_1_ksp_max_it", None)
    # Both arms are `full`, named here rather than taken from the preset, whose
    # default is `lower` off `ainvb`. The reference arm has to be `full`: it is
    # the comparison of `test_the_outer_count_matches_the_full_factorisation_arm`
    # and `lower` spends more outer iterations, so passing `ainvb` through to
    # the preset is not the way to reach this. The cached arm has to be `full`
    # as well, because `dtn_schur_ainvb` is set by hand below and
    # `DtNTwoBlockSchurPC` refuses the cached apply under any other
    # factorisation type.
    params["dtn_pc_fieldsplit_schur_fact_type"] = "full"
    if ainvb:
        params["dtn_schur_ainvb"] = True
    return params


def block_zero_solves(solver):
    """Block-0 applications so far, counted by `gadopt.CondensedBlockPC`.

    `CondensedBlockPC.elimination_count` increments once per `apply`, and
    block 0's KSP is `preonly`, so one block-0 solve is one increment. The
    counter belongs to a different class from the one under test, which is
    what makes it a witness rather than a restatement.
    """
    return tb.block0_context(solver).elimination_count


@pytest.fixture(scope="module")
def two_arms(meshes):
    """One solve of each arm on the iterative fixture, with their counts."""
    out = {}
    for name, ainvb in (("cached", True), ("reference", False)):
        solver, z, layout = tb.build(
            meshes, "lowrank", solver_parameters=iterative_params(ainvb=ainvb))
        solver.solve()
        out[name] = {
            "solver": solver, "layout": layout,
            "iterations": outer_iterations(solver),
            "block0": block_zero_solves(solver),
            "n": len(layout.real_fields),
        }
    return out


def test_the_apply_costs_one_block_zero_solve(two_arms):
    """`block-0 solves = n + outer iterations`, and not `n + 2 x outer`.

    This is the claim the whole design exists for. Both sides are read from
    objects outside the class under test: `n` from the space's layout, the
    iteration count from the outer KSP, the block-0 count from
    `CondensedBlockPC`. So a counter the new class keeps about itself cannot
    make this pass.

    The first assertion is the vacuity guard. It states the cost model of the
    arm that exists today -- one build of `n` columns plus TWO block-0 solves
    per outer iteration -- and it passes today. If it ever fails, the model
    behind the second assertion has moved and the second assertion is
    measuring something else.

    CATCHES: the obvious implementation that keeps `S` but still runs the
    second block-0 solve of `full`; a build that solves twice per column
    (once for `S` through `MatMult_SchurComplement` and once for `Z`), which
    the design's section 3.2 exists to avoid and which no state comparison can
    see; an apply that rebuilds the cache.
    MEASURED on the reference arm: `2 + 2*3 = 8` block-0 solves, for `n = 2`
    and three outer iterations.
    """
    reference, cached = two_arms["reference"], two_arms["cached"]
    n = reference["n"]
    assert reference["block0"] == n + 2 * reference["iterations"], (
        "the reference arm no longer costs one build plus two block-0 solves "
        "per outer iteration, so the comparison below has lost its model")
    assert cached["block0"] == cached["n"] + cached["iterations"], (
        f"{cached['block0']} block-0 solves for {cached['iterations']} outer "
        f"iterations and n = {cached['n']}")


class TestTheSolverReachesTheChoiceTheWidthRuleMakes:
    """The solver's own call, which is where a wrong rule would cost solves.

    `selfgrav_dtn_iterative_solver_parameters` never sees the width unless a
    caller passes it, and `SelfGravitatingGIASolver` is the caller that does:
    it passes `n_real=len(layout.real_fields)`. So the dictionary tests of
    `test_dtn_multiplier_pc.py` cannot see what the solver chooses, and these
    two can. Both run one solve and read the objects PETSc built.
    """

    def test_a_wide_real_block_keeps_the_delegating_path(self, meshes):
        """About 76 `Real` rows in production, 35 here, so no complement.

        A build costs `n` block-0 solves. At this width nobody has measured
        that it amortises, so the preset must not choose either way of forming
        the complement, and the cached apply must not be installed on the
        outer PC at all.
        """
        solver, _, layout = tb.build(
            meshes, "multiplier", truncation=8, solver_parameters="iterative")
        # The vacuity guard: this truncation must actually be above the limit
        # the preset applies, or the test says nothing.
        assert len(layout.real_fields) > 16, (
            f"{len(layout.real_fields)} Real rows is not a wide block, so "
            "this test no longer exercises the rule it is about")
        assert "dtn_schur_ainvb" not in solver.solver_parameters
        assert solver.solver_parameters[
            "dtn_pc_fieldsplit_schur_fact_type"] == "lower"
        solver.solve()
        assert getattr(outer_pc(solver).getPythonContext(), "ainvb",
                       None) is None

    def test_a_narrow_real_block_reaches_the_cache_through_the_solver(
            self, meshes):
        """The production path: the solver's own call installs the cache.

        The count identity is the same one `test_the_apply_costs_one_block_
        zero_solve` states, read from objects outside the class under test:
        `n` from the layout, the iterations from the outer KSP, the block-0
        count from `gadopt.CondensedBlockPC`. One build, because this is one
        solve at one operator.
        """
        solver, _, layout = tb.build(
            meshes, "lowrank", solver_parameters="iterative")
        assert solver.solver_parameters["dtn_schur_ainvb"] is True
        solver.solve()
        cache = ainvb_cache(outer_pc(solver))
        n = len(layout.real_fields)
        assert cache.n == n
        assert cache.build_count == 1
        assert block_zero_solves(solver) == n + outer_iterations(solver), (
            f"{block_zero_solves(solver)} block-0 solves for "
            f"{outer_iterations(solver)} outer iterations and n = {n}")


def test_the_tolerance_ladder_of_the_phase_3_spike_is_untouched():
    """`demos/gravity/spikes/spike_phase3d.py:394`, call for call.

    That driver sweeps `block0_rtol` down to 1e-2 and names
    `multiplier_pc="none"` so that the preset refuses nothing and every rung
    differs in the tolerance alone. A preset that chose the cached apply there
    would form the exact complement from block-0 solves at 1e-2 without
    raising and without a line in the log, and the ladder would stop measuring
    the tolerance. Naming a block-1 preconditioner is what keeps the caller on
    the path they asked for.
    """
    for condensed in (True, False):
        for block0_rtol in (1e-2, 1e-3, 1e-4):
            p = selfgrav_dtn_iterative_solver_parameters(
                condensed=condensed, block0_rtol=block0_rtol,
                outer_rtol=1e-6, snes_rtol=1e-4, multiplier_pc="none")
            assert "dtn_schur_ainvb" not in p, (condensed, block0_rtol)
            assert p["dtn_pc_fieldsplit_schur_fact_type"] == "lower", (
                condensed, block0_rtol)
            assert p["dtn_fieldsplit_1_pc_type"] == "none", (
                condensed, block0_rtol)


def test_the_outer_count_matches_the_full_factorisation_arm(two_arms):
    """The cached path keeps `full`'s convergence, not `lower`'s.

    Saving a solve per iteration is worth nothing if the iteration count moves,
    and the two arms are algebraically the same preconditioner, so their counts
    must agree. One iteration of slack is allowed because the block-0 solve is
    an FGMRES to a tolerance and therefore not exactly a linear operator, so
    the two arms' `S` and `Z` differ slightly.

    CATCHES: a cached `Z` that is stale relative to `S`, or an apply that drops
    the `-Z s1` correction, both of which degrade the preconditioner to
    something near `lower` and cost iterations while converging to the same
    answer.
    MEASURED on the reference arm: 3 outer iterations.
    """
    assert two_arms["cached"]["iterations"] <= (
        two_arms["reference"]["iterations"] + 1)


@pytest.fixture(scope="module")
def marched(meshes):
    """Three solves at one `dt`, then one more after `dt.assign`.

    The three rebuild tests below share this one march, because each solve is
    the expensive part and the three claims are about one sequence.
    """
    solver, z, layout = tb.build(
        meshes, "lowrank", solver_parameters=iterative_params(ainvb=True),
        dt=1.0)

    def builds():
        """The cache's build count, or `None` when there is no cache.

        The fixture must not assert: an assertion here turns all three tests
        below into fixture errors, and a test that errors before its own
        assertion has verified nothing.
        """
        cache = getattr(outer_pc(solver).getPythonContext(), "ainvb", None)
        return None if cache is None else int(cache.build_count)

    for _ in range(3):
        solver.solve()
    fixed_step_builds = builds()
    block0_assemblies_before = tb.block0_context(solver).assembly_count

    solver.dt.assign(2.0)
    solver.solve()
    return {
        "solver": solver,
        "fixed_step_builds": fixed_step_builds,
        "builds_after_dt_change": builds(),
        "block0_assemblies_before": block0_assemblies_before,
        "block0_assemblies_after": tb.block0_context(solver).assembly_count,
    }


def test_the_cache_is_built_once_at_a_fixed_operator(marched):
    """Three solves at one `dt` leave the build counter at one.

    A rebuild is `n` block-0 solves, so a cache that rebuilds when nothing
    moved spends the saving the design exists to collect, and converges to the
    same answer while doing it.

    CATCHES: a rebuild keyed on the Jacobian's object state (which moves at
    every assembly) instead of on the two staleness markers; a build moved into
    `apply`.
    """
    assert marched["fixed_step_builds"] == 1, (
        "build count after three solves at one dt: "
        f"{marched['fixed_step_builds']} (None means the option "
        "installed no cache)")


def test_a_time_step_change_rebuilds_the_cache(marched):
    """`dt.assign` moves `operator_version`, so the cache must be rebuilt.

    The effective bulk/shear ratio is `bulk_shear_ratio * (1 + dt/tau)`, so a
    new time step is a different mechanics block and the columns of the old one
    describe an operator that no longer exists.

    CATCHES: a cache built once at `initialize` and never revisited, which is
    a valid preconditioner and therefore silent.
    """
    assert marched["builds_after_dt_change"] == 2, (
        "build count after the dt change: "
        f"{marched['builds_after_dt_change']} (None means the "
        "option installed no cache)")


def test_the_inner_fieldsplit_is_set_up_from_update(marched):
    """`update` must call `self.pc.setUp()` before anything else.

    THE TRAP OF THIS DESIGN (section 3.4). On the delegating path
    `self.pc.apply` calls `PCApply` on the inner fieldsplit, which calls
    `PCSetUp` on it, which re-extracts the sub-matrices with
    `MAT_REUSE_MATRIX`; Firedrake's `createSubMatrix` assembles into the reused
    target, which moves every sub-block's object state, which is what makes
    `CondensedBlockPC.update` run and reassemble its four blocks. Under the
    option nothing calls `PCApply` on the inner fieldsplit, so unless `update`
    sets it up by hand the block-0 sub-operator's state never moves,
    `CondensedBlockPC` keeps the blocks of the FIRST Jacobian for the whole
    march, and every answer stays right.

    The witness is `CondensedBlockPC.assembly_count`, which belongs to another
    class and counts reassemblies of the four block-0 blocks.

    CATCHES exactly one omission: a missing `self.pc.setUp()` at the top of
    `update`. Nothing else in this file can see it -- the states agree, the
    complement is rebuilt (the first assertion below), and only the iteration
    count suffers.
    """
    # Vacuity guard: this says the cached path did notice the dt change. If it
    # had not, the assertion below would be measuring the wrong omission.
    assert marched["builds_after_dt_change"] == 2, (
        "the cache did not rebuild at all, so this test cannot distinguish a "
        "missing inner setUp from a missing rebuild rule")
    assert (marched["block0_assemblies_after"]
            > marched["block0_assemblies_before"]), (
        "CondensedBlockPC never reassembled after the time step changed: the "
        "inner fieldsplit was never set up, so block 0 is still preconditioned "
        "with the blocks of the first Jacobian")


def test_one_build_per_nonlinear_solve_on_a_power_law(meshes):
    """With `exponent = 3` the cache is rebuilt once per solve, never per step.

    A power law publishes `operator_version = None`, because no version number
    can describe a Jacobian that moves with the state, and the rule falls back
    to `gia_solve_index`. One build per Newton iteration would roughly triple
    the block-0 work of a three-iteration step; a complement frozen at the
    state each solve starts from costs outer iterations and changes no number.

    CATCHES: a rule that rebuilds at every Newton iteration (build counter
    tracks `CondensedBlockPC.assembly_count`); a rule that never rebuilds on a
    power law, because `operator_version` is `None` and the integer branch was
    the only one written.
    """
    solver, _, _ = tb.build(
        meshes, "lowrank",
        solver_parameters=iterative_params(ainvb=True, snes_type="newtonls"),
        approximation_kwargs={"exponent": 3.0, "transition_stress": 1e-3})
    solver.solve()
    # Vacuity guard: with one Newton iteration, per-iteration and per-solve
    # rebuilding are the same rule and the test would prove nothing.
    assert solver.solver.snes.getIterationNumber() >= 2, (
        "the test is vacuous unless Newton took more than one iteration")
    cache = ainvb_cache(outer_pc(solver))
    assert cache.build_count == 1
    # The block-0 blocks ARE reassembled at every Newton iteration, which is
    # what makes "1" above a statement about the rule and not about a solve in
    # which nothing moved.
    assert tb.block0_context(solver).assembly_count >= 2
    solver.solve()
    assert cache.build_count == 2


# ===========================================================================
# C. What the configuration refuses, and how loudly
# ===========================================================================
# **Every refusal below is checked in a CHILD PROCESS, and that is not a style
# choice.** Three ingredients together kill the interpreter with a
# segmentation violation, printing nothing at all -- no traceback, no test
# report, no failure line:
#
#   1. `PETSc.Sys.popErrorHandler()`, which `gadopt/__init__.py:75` calls at
#      import time, so every test module in this repository has it;
#   2. pytest's output capture, which is on by default;
#   3. a Python exception raised inside a Firedrake python-PC callback.
#
# MEASURED in this worktree on 2026-09-19, by bisection on a three-line
# `firedrake.PCBase` that raises `ValueError` from `initialize`, driven by
# `firedrake.solve` on a `UnitSquareMesh` Poisson problem. With the
# `popErrorHandler` line: exit 139, no output. Without that one line, and
# with nothing else changed: `1 passed`, the exception arriving in Python as
# `petsc4py.PETSc.Error`. With the line and `pytest -s`: `1 passed`. The
# reproducer is transcribed in `NOTES/team/rotation-pc/12-TESTS-AINVB.md`.
# Ingredient 1 is why a reproducer that imports only firedrake does not show
# this, and why every test in this repository does have the condition.
#
# Two things follow. A refusal driven through `solver.solve()` inside pytest
# would take the whole file down with no message, which is worse than no test.
# And calling `initialize` a second time by hand is not a way round it: on a
# PC that PETSc has already set up, `pc.getDM().createSubDM` raises inside its
# own hook (MEASURED: `Unhandled Python Exception` at `DMCreateSubDM`), so the
# second call fails for a reason that has nothing to do with the refusal.
#
# So each refusal runs as a script in a child process, where the exception is
# an ordinary Python exception and the process survives it. The parent reads
# the child's exit status and its stderr.
#
# What the parent asserts on is the stderr TEXT, not the exception type.
# PETSc flattens a Python exception raised inside a python PC into
# `PETSc.Error: error code 101` with no text, so `_loud` must
# print the named cause before the exception is raised; that print is the only
# thing a log reader can find.

#: The child script. `check` selects which refusal to provoke. Written as a
#: script rather than as a pytest test for the reason the section comment
#: gives.
REFUSAL_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
import gadopt  # noqa
import firedrake as fd
import test_gia_gravity_adjoint_lowrank as tl
import test_gia_lowrank_block0 as tb
import test_dtn_schur_ainvb as suite

check = sys.argv[1]
meshes = tl.meshes.__wrapped__()

if check == "fact_type":
    # `lower` asked for, `full` delivered: the arm would believe it measured
    # a factorisation it did not run.
    solver, _, _, _ = suite.build_direct(meshes, suite.direct_params(
        **{{"dtn_pc_fieldsplit_schur_fact_type": "lower"}}))
    call = solver.solve
elif check == "block1_ksp":
    # A Krylov method configured on a block that is never solved.
    solver, _, _, _ = suite.build_direct(meshes, suite.direct_params(
        **{{"dtn_fieldsplit_1_ksp_type": "gmres",
            "dtn_fieldsplit_1_ksp_rtol": 1e-4,
            "dtn_fieldsplit_1_ksp_max_it": 200}}))
    call = solver.solve
elif check == "transpose":
    # The production route: block 0 is gadopt.CondensedBlockPC, which has no
    # transpose application, so the transpose apply cannot run.
    solver, _, _ = tb.build(
        meshes, "lowrank", solver_parameters=suite.iterative_params(ainvb=True))
    solver.solve()
    pc = suite.outer_pc(solver)
    A, _ = pc.getOperators()
    x, y = A.createVecLeft(), A.createVecRight()
    x.setRandom()
    context = pc.getPythonContext()

    def call():
        context.applyTranspose(pc, x, y)
else:
    raise SystemExit("unknown check " + check)

try:
    call()
except BaseException as exc:
    print("EXCEPTION-TYPE", type(exc).__name__, file=sys.stderr, flush=True)
    sys.exit(3)
print("NO-EXCEPTION", file=sys.stderr, flush=True)
sys.exit(0)
"""


def refusal_stderr(check):
    """Provoke one refusal in a child process and return what it printed.

    Args:
      check: `"fact_type"`, `"block1_ksp"` or `"transpose"`.

    Returns:
      The child's stderr.

    Raises:
      AssertionError: if the child did not raise at all, or died from a
        signal, or timed out.
    """
    script = REFUSAL_SCRIPT.format(tests=str(Path(__file__).resolve().parent))
    finished = subprocess.run(
        [sys.executable, "-c", script, check], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]), timeout=900)
    assert finished.returncode == 3, (
        f"the {check} configuration did not raise in the child process "
        f"(exit status {finished.returncode}; a negative status is a signal)"
        f"\n--- child stderr ---\n{finished.stderr[-3000:]}")
    return finished.stderr


def test_a_partial_factorisation_is_refused():
    """`fact_type lower` together with the option is refused, by name.

    The cached apply IS the `full` factorisation with the second block-0 solve
    replaced by the stored columns. An arm that sets `lower` and the option
    together would believe it measured `lower` and would in fact measure
    `full`, and both converge to the same answer, so nothing else would ever
    say otherwise.

    CATCHES: an implementation that reads the option and ignores the
    factorisation type.
    """
    message = refusal_stderr("fact_type")
    # `_loud` writes `[gadopt.preconditioners] <Type>: <message>`. Asserting
    # that prefix as well as the option name is what stops PETSc's own
    # `Option left: name:-...dtn_schur_ainvb` line at finalisation -- which
    # appears whenever nobody reads the option -- from satisfying this test on
    # an implementation that refused for some unrelated reason.
    assert "gadopt.preconditioners" in message and (
        "dtn_schur_ainvb" in message), (
        "the refusal did not name the option on stderr through `_loud`, so a "
        "real run would show PETSc error code 101 and nothing else"
        f"\n{message[-3000:]}")


def test_a_configured_block_one_krylov_is_refused():
    """A block-1 KSP that will never be solved is refused, by name.

    Under the option block 1 is solved by the dense factors of the exact
    complement and `dtn_fieldsplit_1_` is never entered. A `gmres` left on
    that prefix means something specific in every other arm -- its
    converged-reason lines are half of the B2 cost model -- and a
    configuration that is written and silently unused is the class of defect
    this project pays most for.

    CATCHES: an implementation that leaves the idle block-1 KSP unchecked.
    """
    message = refusal_stderr("block1_ksp")
    assert "gadopt.preconditioners" in message and (
        "dtn_fieldsplit_1_ksp_type" in message), (
        "the refusal did not name, through `_loud`, the option that fixes it"
        f"\n{message[-3000:]}")


def test_the_transpose_is_refused_when_block_zero_has_no_transpose():
    """On the production route the transpose apply is refused, by name.

    `gadopt.CondensedBlockPC` has no transpose application by design, so
    `ksp0.solveTranspose` on the production block 0 cannot run. An adjoint
    solve does not come this way -- pyadjoint solves `adjoint(J)` with the
    forward options and therefore reaches the forward `apply` -- so the
    refusal costs nothing and names the one thing a caller needs to know:
    which preconditioner lacks the transpose.

    CATCHES: a transpose apply that lets the bare PETSc error out of two
    levels down, where the log carries `KSPSolveTranspose` and
    `PCApplyTranspose_FieldSplit_Schur` in a C traceback and nothing that
    names a class the reader can act on.
    """
    message = refusal_stderr("transpose")
    assert "gadopt.preconditioners" in message and (
        "CondensedBlockPC" in message), (
        "the refusal did not name, through `_loud`, the block-0 "
        f"preconditioner that lacks a transpose solve\n{message[-3000:]}")


def test_the_preset_refuses_a_block_one_preconditioner_with_the_cache():
    """`ainvb=True` and `multiplier_pc` together are refused by the preset.

    The cached path solves block 1 with its own factors of the exact
    complement, so a preconditioner named for block 1 would be built,
    configured and never applied.
    """
    with pytest.raises(ValueError):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=False, block0="condensed",
            multiplier_pc="gadopt.DtNMultiplierDenseSchurPC", ainvb=True)


def test_the_preset_writes_the_cached_path_keys():
    """`ainvb=True` writes the option and silences the block-1 Krylov solve.

    Design section 2.2: the option, `preonly` and `none` on block 1, and NONE
    of the three block-1 Krylov keys, because no block-1 Krylov solve runs and
    a tolerance written for a solve that never happens is a lie in the log.
    """
    params = selfgrav_dtn_iterative_solver_parameters(
        condensed=False, block0="condensed", ainvb=True)
    assert params["dtn_schur_ainvb"] is True
    assert params["dtn_pc_fieldsplit_schur_fact_type"] == "full"
    assert params["dtn_fieldsplit_1_ksp_type"] == "preonly"
    assert params["dtn_fieldsplit_1_pc_type"] == "none"
    for dropped in ("dtn_fieldsplit_1_ksp_rtol",
                    "dtn_fieldsplit_1_ksp_max_it",
                    "dtn_fieldsplit_1_ksp_converged_reason"):
        assert dropped not in params, (
            f"{dropped} is written for a block-1 Krylov solve that never runs")


def test_the_preset_default_is_the_dictionary_ainvb_true_writes():
    """The preset chooses the cached apply where it forms the complement.

    Two things are pinned. `ainvb=True` named by hand gives exactly the
    dictionary the preset writes on its own, which is what makes jobs
    179511971 and 179511972 - the two full Spada runs, which passed `--ainvb`
    on the command line - a measurement of the default. And `ainvb=False`
    names the delegating path, so the argument still has three meanings and
    not two: choose, on, off.

    The preset chooses it on the low-rank representation, where the `Real`
    block is 4 rows and a build is 4 block-0 solves. On the multiplier
    representation a build is about 76 and no arm measures it, so the width
    rule keeps the preset off that path; `test_the_multiplier_representation_
    is_kept_off_the_cached_apply` in `test_dtn_multiplier_pc.py` pins that.
    """
    for kwargs in ({"condensed": False, "block0": "condensed"},
                   {"condensed": False, "dtn_representation": "lowrank"},
                   {"condensed": False, "n_real": 4}):
        default = selfgrav_dtn_iterative_solver_parameters(**kwargs)
        named_on = selfgrav_dtn_iterative_solver_parameters(
            ainvb=True, **kwargs)
        assert default == named_on, kwargs
        assert default["dtn_schur_ainvb"] is True, kwargs
        named_off = selfgrav_dtn_iterative_solver_parameters(
            ainvb=False, **kwargs)
        assert "dtn_schur_ainvb" not in named_off, kwargs
        assert named_off["dtn_pc_fieldsplit_schur_fact_type"] == "lower", kwargs


# ===========================================================================
# D. Two ranks
# ===========================================================================
# Module-level function, not a method: mpi-pytest cannot relaunch a test that
# lives inside a class.
@pytest.mark.parallel(nprocs=2)
def test_two_ranks_build_the_same_complement(meshes):
    """The build is collective and `S` is redundant, so both ranks agree.

    `S` is gathered with an `Allreduce` of length `n` and factored on every
    rank, and the block-1 apply writes only the owned slice of a redundantly
    computed solution. Every `Real` degree of freedom happens to sit on one
    rank today, so the rank that owns none must still enter every collective:
    the `n` block-0 solves of the build, the `Allreduce` per column and the
    `VecMDot` of the transpose apply.

    CATCHES: a build keyed on the LOCAL column count, which makes the
    non-owning rank form a 0x0 complement and skip the collective block-0
    solves while the owning rank blocks inside them -- the failure a 104-rank
    Gadi run produced for `DtNMultiplierDenseSchurPC` and which no serial test
    can see; a rebuild decision taken from rank-local data.
    FLOOR: `PARITY_FLOOR` for the cross-rank difference of `S`.
    """
    comm = fd.COMM_WORLD
    assert comm.size == 2, "this test must run on exactly two ranks"
    solver, _, layout = tb.build(
        meshes, "lowrank", solver_parameters=iterative_params(ainvb=True))
    solver.solve()
    assert solver.solver.snes.ksp.getConvergedReason() > 0

    cache = ainvb_cache(outer_pc(solver))
    sizes = comm.allgather(int(cache.n))
    assert sizes[0] == sizes[1] == len(layout.real_fields) > 0, (
        f"the per-rank Real block size disagrees: {sizes}")
    builds = comm.allgather(int(cache.build_count))
    assert builds[0] == builds[1] == 1, f"build counts disagree: {builds}"

    S = np.asarray(cache.S)
    other = comm.bcast(S.copy(), root=1)
    relative = np.linalg.norm(S - other) / np.linalg.norm(S)
    assert relative < PARITY_FLOOR, f"S differs across ranks by {relative:.3e}"


# ===========================================================================
# E. Three dimensions, 24 cells
# ===========================================================================
class TestThreeDimensions:
    """The 24-cell extruded cubed sphere, cached path against the reference arm.

    The sphere has a fluid core and the rotational closure, so the `Real` block
    carries the three rotation rows and the core pressure -- `n = 4`, the
    production shape -- on extruded hexahedra. Both arms share one mesh,
    because `fd.norm` on a difference of functions from two mesh objects
    refuses with "Multiple domains found".

    Nothing here is about speed. Every 3-D performance number comes from Gadi.
    """

    @pytest.fixture(scope="class")
    def solved(self):
        """Both arms solved once on one 24-cell sphere."""
        base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                               extrusion_type="radial")
        mesh.cartesian = False
        X = fd.SpatialCoordinate(mesh)
        C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
        out = {}
        for name, ainvb in (("cached", True), ("reference", False)):
            Z, layout = self_gravitating_gia_space(
                mesh, mesh,
                gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                             "bottom": {"dtn": SphericalDtN(1)}},
                rotation=True, fluid_core=True,
                condense_internal_variables=False, n_internal_variables=1,
                self_gravity_number=tl.LAMBDA,
                dtn_representation="lowrank")
            z = fd.Function(Z)
            approx = CompressibleInternalVariableApproximation(
                bulk_modulus=1.0, density=fd.Constant(1.0),
                shear_modulus=fd.Constant(1.0), viscosity=fd.Constant(1.0),
                g=fd.Constant(tl.G0), B_mu=fd.Constant(tl.B_MU),
                self_gravity_number=fd.Constant(tl.LAMBDA))
            solver = SelfGravitatingGIASolver(
                z, approx, layout=layout, dt=fd.Constant(1.0),
                bcs={"top": {"normal_stress": tl.SIGMA_HAT * X[2]}},
                rotation_moments={"C": C, "C_minus_A": 0.1 * C},
                fluid_core=FluidCore(boundary="bottom", rho_core=2.0),
                solver_parameters=iterative_params(ainvb=ainvb),
                dtn_representation="lowrank")
            solver.solve()
            out[name] = {"solver": solver, "z": z, "layout": layout,
                         "iterations": outer_iterations(solver),
                         "block0": block_zero_solves(solver),
                         "n": len(layout.real_fields)}
        return out

    def test_the_construction_works_on_the_sphere(self, solved):
        """The cached path reproduces the reference arm's state in 3-D.

        The two arms are the same preconditioner reached two ways, so the two
        states must agree to the outer tolerance. This test cannot see a wrong
        preconditioner -- that is what the count test below is for -- but it
        can see a build that fails on extruded hexahedra with a fluid core and
        the rotational closure, which is the construction this section exists
        to exercise.

        FLOOR: 1e-8 relative, the bar the 3-D tests of the low-rank suite use;
        the two arms differ only through the block-0 FGMRES tolerance.
        """
        reference, cached = solved["reference"], solved["cached"]
        for arm in (reference, cached):
            assert arm["solver"].solver.snes.getConvergedReason() > 0
            assert arm["solver"].solver.snes.ksp.getConvergedReason() > 0
        # The Real block is the four rows the design targets: three rotation
        # rows and the core pressure.
        assert cached["n"] == 4
        # Vacuity guard, and the part of this test that can fail. Two arms of
        # the same preconditioner must agree whether or not the cached path
        # exists, so the state comparison below says nothing until the cache
        # is known to have been built on this geometry.
        cache = ainvb_cache(outer_pc(cached["solver"]))
        assert cache.n == 4 and cache.build_count == 1
        for field in ("displacement", "potential"):
            a = cached["z"].subfunctions[getattr(cached["layout"], field)]
            b = reference["z"].subfunctions[
                getattr(reference["layout"], field)]
            scale = fd.norm(b)
            assert scale > 0.0
            assert fd.norm(a - b) < 1e-8 * scale

    def test_the_apply_costs_one_block_zero_solve(self, solved):
        """`n + outer iterations` block-0 solves on the sphere, as on the annulus.

        The same identity as the 2-D test, on the geometry and the `Real`-block
        shape the production run has. CATCHES the same wrong implementations,
        and in addition one that only works when every `Real` row is a DtN
        multiplier.
        """
        reference, cached = solved["reference"], solved["cached"]
        assert reference["block0"] == (
            reference["n"] + 2 * reference["iterations"]), (
            "the reference arm's cost model has moved")
        assert cached["block0"] == cached["n"] + cached["iterations"]


# ===========================================================================
# F. The adjoint
# ===========================================================================
def test_the_gradient_is_right_through_the_cached_path(meshes):
    """The cache builds on `adjoint(J)` and the gradient is unchanged.

    pyadjoint assembles `adjoint(dFdu)` and solves it with the FORWARD solver
    options, so an adjoint solve of this system reaches `apply` over the
    transposed operator, not `applyTranspose`. The class introspects that
    operator like any other: the `Real` fields are still last, block 0 is
    `A00^T`, and the whole of the build and the forward apply runs on it. On
    the low-rank path the live application context is carried through
    `LowRankVariationalSolver`, with both staleness markers frozen for the
    sweep, so the adjoint's cache is built once and kept.

    The reference is the same gradient with the option off. A preconditioner
    cannot change a converged gradient, so a difference means the adjoint solve
    did not converge to the same place.

    CATCHES: a cache that reads something pyadjoint strips from the adjoint
    solve's kwargs and raises there; a rebuild rule that thrashes or stalls on
    the frozen markers; a build that fails on the transposed operator.
    FLOOR: 1e-6 relative, the bar the adjoint suite uses. The two solves reach
    the same linear system to the same outer tolerance, so they agree far
    closer than that.
    """
    import test_gia_gravity_adjoint_lowrank as suite
    from pyadjoint import get_working_tape

    value = suite.CONTROL_VALUES["shear_modulus"]
    original = suite.SelfGravitatingGIASolver
    gradients = {}
    cache_builds = None
    for name, ainvb in (("cached", True), ("reference", False)):
        params = iterative_params(ainvb=ainvb, outer_rtol=1e-11)

        class Iterative(original):
            """The suite builds with `solver_parameters="direct"`; swap ours in."""

            def __init__(self, *args, _params=params, **kwargs):
                if kwargs.get("solver_parameters") == "direct":
                    kwargs["solver_parameters"] = _params
                super().__init__(*args, **kwargs)

        suite.SelfGravitatingGIASolver = Iterative
        try:
            functional, _, _, _, solver = suite.reduced_functional(
                meshes, "shear_modulus", value, "lowrank", fluid_core=True)
            gradients[name] = float(functional.derivative().dat.data_ro[0])
            if ainvb:
                adjoint_solver = solver.adjoint_block._ad_solvers["adjoint_lvs"]
                cache_builds = ainvb_cache(
                    adjoint_solver.snes.ksp.pc).build_count
            get_working_tape().clear_tape()
        finally:
            suite.SelfGravitatingGIASolver = original

    # Vacuity guard: the gradient comparison alone would pass whether or not
    # the adjoint solve ever reached the cached path.
    assert cache_builds == 1, (
        "the adjoint solve did not build the cache exactly once; the gradient "
        "below would then say nothing about the cached path")
    relative = abs(gradients["cached"] - gradients["reference"]) / abs(
        gradients["reference"])
    assert relative < 1e-6, (
        f"the cached path moved the gradient by {relative:.3e} relative")
