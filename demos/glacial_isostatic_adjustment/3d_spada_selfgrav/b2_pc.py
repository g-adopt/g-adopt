"""Preconditioners for the B2 solver probe.

Separate from `b2_probe` so that `pc_python_type` can name it without
re-importing the driver as a second module. Everything here lives in the probe
directory and touches no shipped code.
"""
import os

import numpy as np
from firedrake.dmhooks import get_function_space
from firedrake.petsc import PETSc
from firedrake.preconditioners.assembled import AssembledPC

from gadopt.nullspaces import rigid_body_modes


class RigidBodyAssembledPC(AssembledPC):
    """`AssembledPC` for the displacement block of the coupled GIA operator.

    Firedrake composes a declared `near_nullspace` onto the *outer* mixed
    space's field index sets, and `PCFIELDSPLIT` reads it back from there
    (`src/ksp/pc/impls/fieldsplit/fieldsplit.c:721`,
    `PetscObjectQuery(ilink->is, "nearnullspace")`). Inside a nested split whose
    index sets are built afresh from a sub-DM, that query finds nothing and the
    near-nullspace is silently lost. Building it here from the block's own
    function space is immune to how the block was reached. Six modes: the three
    rotations and the three translations, which is what GAMG wants for
    elasticity.

    Do **not** attach the three rotations as a true nullspace. They are a kernel
    of the continuum operator but are annihilated only to ~2e-06 discretely, so
    declaring them exact makes CG report `DIVERGED_INDEFINITE_PC` (measured, at
    256 iterations). `GADOPT_PROBE_KERNEL=1` switches it on to reproduce that.
    """

    def _spaces(self, pc):
        V = get_function_space(pc.getDM()).collapse()
        near = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])
        kernel = rigid_body_modes(V, rotational=True)
        return near, kernel

    def set_nullspaces(self, pc):
        super().set_nullspaces(pc)
        near, kernel = self._spaces(pc)
        Pmat = self.P.petscmat
        Pmat.setNearNullSpace(near.nullspace())
        if os.environ.get("GADOPT_PROBE_KERNEL") == "1":
            Pmat.setNullSpace(kernel.nullspace())


# ---------------------------------------------------------------------------
# The multiplier block
# ---------------------------------------------------------------------------
#
# Four options-file routes to preconditioning this block are dead, each for a
# different reason, and every reason is a property of the *route* rather than
# of the object:
#
#   jacobi        `MatGetDiagonal` on a matfree block goes through TSFC's
#                 `diagonal=True` path, which cannot unpack argument
#                 multiindices for `Real` arguments.
#   AssembledPC   needs a sub-DM the shipped `DtNTwoBlockSchurPC` threads onto
#                 the potential KSP only, and then hits `Monolithic matrix
#                 assembly not supported for systems with R-space blocks`.
#   -pc_fieldsplit_schur_precondition full
#                 PETSc would build the explicit complement itself
#                 (`fieldsplit.c:1120-1154` allocates the work matrix,
#                 `:1262-1298` fills it on the first apply, and thereafter the
#                 upper solve degenerates to a dense `MatMultAdd`). Two
#                 blockers: `DtNTwoBlockSchurPC` never calls
#                 `PCSetUpOnBlocks`, so the allocation never happens; and even
#                 with that fixed, `MatSchurComplementComputeExplicitOperator`
#                 finishes with `MatAXPY(S, -1, D)` where `D` is the matfree
#                 (c,c) block, and `MatAXPY_Basic` reaches it through
#                 `MatGetRow` (`src/mat/utils/axpy.c:233-265`), which no shell
#                 provides.
#   selfp         builds `C diag(A00)^{-1} B`; `MatSchurComplementGetPmat`
#                 needs the diagonal of the *block-0* matfree operator, so it
#                 fails at the same TSFC path as `jacobi`.
#
# Both classes below evade all four by never asking anyone to assemble or
# differentiate anything on the `Real` space.


class _RealBlockPCBase:
    """Shared plumbing for the two multiplier preconditioners.

    The multiplier block is tiny - 72 `Real` fields at L = 5, 75 with rotation -
    and Firedrake puts every `Real` degree of freedom on one rank, so the local
    size is `n` there and 0 elsewhere. Both preconditioners therefore do their
    linear algebra in numpy on every rank redundantly, which for a 75x75 problem
    is free and avoids needing a parallel dense solver at all.
    """

    def _gather(self, comm, vec, n_global):
        buf = np.zeros(n_global)
        lo, hi = vec.owner_range
        buf[lo:hi] = vec.array_r
        out = np.zeros(n_global)
        comm.Allreduce(buf, out)
        return out

    def apply(self, pc, x, y):
        comm = pc.comm.tompi4py()
        rhs = self._gather(comm, x, self._n)
        sol = self._solve(rhs)
        lo, hi = y.owner_range
        y.array_w[:] = sol[lo:hi]

    def applyTranspose(self, pc, x, y):
        self.apply(pc, x, y)

    def _solve(self, rhs):
        raise NotImplementedError


class DtNMultiplierDiagPC(_RealBlockPCBase):
    """Rung A: invert the (c,c) block exactly. It is diagonal, and it is known.

    `DtNGravityForm.boundary_bilinear` writes the constraint row as
    `(psi*e_k - scale_k*c_k) * mu_k * ds`, and `mu_k` is a `Real` test function,
    i.e. identically one, so the whole `c_k` column of that row integrates to
    `-scale_k * A_h` with `A_h` the *discrete* boundary measure the form already
    stores in `DtNGravityForm.boundary_area`. The coupled solver then multiplies
    the entire row by `theta_psi`. So

        A_(c_k, c_k) = -theta_psi * scale_k * A_h(bc_id_k)

    with no assembly and no `MatGetDiagonal`. What it misses is the Schur
    correction `C A00^{-1} B`, the DtN feedback through the potential, which is
    small for a boundary stood off to 2 Re and smaller for higher modes.
    """

    def setUp(self, pc):
        diag = _DIAGONAL.get("value")
        if diag is None:
            raise ValueError(
                "DtNMultiplierDiagPC needs the (c,c) diagonal; call "
                "b2_pc.set_dtn_diagonal(...) before the solve.")
        A, _ = pc.getOperators()
        self._n = A.getSizes()[0][1]
        if len(diag) != self._n:
            raise ValueError(
                f"multiplier block is {self._n} wide but the diagonal has "
                f"{len(diag)} entries; rotation rows are probably missing.")
        self._d = np.asarray(diag, dtype=float)

    def _solve(self, rhs):
        return rhs / self._d


class DtNMultiplierDenseSchurPC(_RealBlockPCBase):
    """Rung B: form the whole Schur complement once at setup and factor it.

    The `Amat` this preconditioner is handed is PETSc's `MATSCHURCOMPLEMENT`
    (`fieldsplit.c:1006`, `KSPSetOperators(jac->kspschur, jac->schur, ...)`), so
    **one `MatMult` of it is one application of S**. The block is 72-75 columns
    wide, so that many `MatMult`s build it entirely, each costing one block-0
    solve. Contract into a dense array, factor, and apply the factors from then
    on. No Firedrake assembly is involved anywhere, which is why this works
    where all four options-file routes above do not.

    **Correctness conditions, and the second is not what the design assumed.**

    - The coupled Jacobian must be constant across the run. It is, by the same
      argument that makes `DtNTwoBlockSchurPC.update` a no-op, so the complement
      formed once stays the right object.
    - S must be *linear*, and it is only as linear as the block-0 solve inside
      it. With `dtn_fieldsplit_0_ksp_type: preonly` that solve is a fixed linear
      operator and the complement is exact to roundoff. With an iterative
      block-0 KSP - the configuration that actually runs at production - FGMRES
      to a relative tolerance is **not** a linear operator, so each column is
      built with a slightly different effective inverse and S is approximate.
      Harmless for a preconditioner, fatal for "apply it exactly forever": the
      honest description is a very good preconditioner built once. The measured
      asymmetry printed at setup is the instrument for how far off it is, since
      the exact complement inherits the operator's own symmetry structure.
    """

    def setUp(self, pc):
        A, _ = pc.getOperators()
        n = A.getSizes()[0][1]
        comm = pc.comm.tompi4py()
        e = A.createVecRight()
        col = A.createVecLeft()
        S = np.zeros((n, n))
        for k in range(n):
            e.set(0.0)
            lo, hi = e.owner_range
            if lo <= k < hi:
                e.setValue(k, 1.0)
            e.assemble()
            A.mult(e, col)
            S[:, k] = self._gather(comm, col, n)
        e.destroy()
        col.destroy()
        self._n = n
        self._S = S
        scale = max(np.abs(S).max(), 1e-300)
        try:
            self._inv = np.linalg.inv(S)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"the {n}x{n} multiplier Schur complement is singular: {exc}"
            ) from None
        PETSc.Sys.Print(
            f"    [dense Schur] {n}x{n} built in {n} block-0 applications; "
            f"cond {np.linalg.cond(S):.3e}  "
            f"relative asymmetry {np.abs(S - S.T).max() / scale:.3e}")

    def _solve(self, rhs):
        return self._inv @ rhs


_DIAGONAL = {}


def set_dtn_diagonal(values):
    """Hand `DtNMultiplierDiagPC` the (c,c) diagonal, computed by the driver."""
    _DIAGONAL["value"] = np.asarray(values, dtype=float)


def dtn_diagonal_from_solver(solver, layout):
    """`-theta_psi * scale_k * A_h` for every multiplier, in sub-field order.

    Read off the form rather than assembled, which is the whole point. Rotation
    rows, if present, are appended with their own scaling from the solver.
    """
    form = layout.gravity_form
    theta = float(solver.theta_psi)
    scales = {}
    for bc_id, dtn in form.dtn_boundaries:
        side, R = form.boundary_geometry[bc_id]
        for mode in dtn.modes(side, R, form.X):
            scales[(bc_id, mode.key)] = float(mode.scale)
    diag = []
    for bc_id, key in form.multiplier_keys:
        diag.append(-theta * scales[(bc_id, key)] * float(form.boundary_area[bc_id]))
    if layout.rotation:
        # theta_rot,i * K_i with K = (C-A, C-A, C); road map 5.3. Not needed by
        # B1 (rotation off) and not guessed at here: a wrong constant would
        # silently mis-scale three rows of the preconditioner.
        raise NotImplementedError(
            "the rotation rows' diagonal is theta_rot,i * K_i and is not "
            "derived here; extend dtn_diagonal_from_solver before running "
            "DtNMultiplierDiagPC with rotation on.")
    return np.array(diag)
