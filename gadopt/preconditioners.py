r"""This module contains classes that augment default Firedrake preconditioners.

"""

import sys

import firedrake as fd
import numpy as np
from ufl.indexed import Indexed
from firedrake.petsc import PETSc
from .utility import InteriorBC


class FreeSurfaceMassInvPC(fd.MassInvPC):
    """Version of MassInvPC that includes free surface variables."""

    def form(
        self,
        pc: fd.PETSc.PC,
        tests: list[fd.Argument | Indexed],
        trials: list[fd.Argument | Indexed | fd.Function],
    ) -> tuple[fd.Form, list[fd.DirichletBC]]:
        """Sets the form.

        Args:
          pc:
            PETSc preconditioner
          tests:
            List of Firedrake test functions
          trials:
            List of Firedrake trial functions
        """
        appctx = self.get_appctx(pc)

        # N.B. trials[0] is pressure
        mu = appctx.get("mu", 1.0)
        a = fd.inner(1 / mu * trials[0], tests[0]) * fd.dx

        ds = appctx["ds"]
        bcs = []
        for bc_id, (eta_ind, _) in appctx["free_surface"].items():
            a += 1 / mu * fd.inner(trials[eta_ind - 1], tests[eta_ind - 1]) * ds(bc_id)

            bcs.append(InteriorBC(trials.function_space()[eta_ind - 1], 0, bc_id))

        return a, bcs


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in a fieldsplit_0 block (the Stokes velocity block, or the gravity
    potential block) in combination with gamg. Setting PETSc MatOption MAT_SPD
    (for Symmetric Positive Definite matrices) switches the Krylov method used
    for eigenvalue estimates to CG - both in the Chebyshev smoothers and in
    GAMG's smoothed-aggregation setup - and propagates the SPD flag to the
    coarse-grid operators. All of these are benign for a genuinely SPD block.

    Users can provide this class as a `pc_python_type`
    entry to a PETSc solver option dictionary.

    """
    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


class DtNTwoBlockSchurPC(fd.PCBase):
    """Schur fieldsplit whose two blocks are described by index sets.

    The gravitational Poisson solver gives every treated angular mode its own
    scalar Real-space sub-field, so its mixed space carries 1 + n fields (plus
    a cross-mesh dummy where one is used). PETSc enumerates the sub-fields of
    the DM before grouping them and refuses more than 128 of them
    (`PCFieldSplitSetDefaults`: "Cannot currently support N > 128 fields"),
    which caps the coupled DtN solve at a truncation of L = 6 on a two-boundary
    shell. That enumeration is only reached when no split has been registered
    yet, so registering the two blocks directly as index sets - the potential
    (and dummy) degrees of freedom in one, every multiplier degree of freedom
    in the other - bypasses it, leaving the Schur factorisation, the potential
    block solver and the multiplier solve exactly as they are below the cap.
    Only the description of which degrees of freedom belong to which block
    changes; the weak form, the mixed space and the taped variational problem
    are untouched, so the adjoint is unaffected.

    The two blocks are found by introspecting the mixed space of the operator
    for its Real sub-fields, never from the application context, which pyadjoint
    drops from the kwargs of the adjoint solve.

    Options for the inner fieldsplit are read under a `dtn_` prefix, e.g.
    `Gravity_dtn_fieldsplit_0_ksp_type`. Never supply `pc_fieldsplit_%d_fields`
    options there: the splits are already defined and the field lists would be
    silently ignored.

    Users can provide this class as a `pc_python_type` entry to a PETSc solver
    option dictionary; the preconditioning operator must be matrix-free.

    `update` is a no-op, which is correct for the gravitational Poisson solver
    because its Jacobian is constant by construction (see `update`). The class
    is therefore not intended for problems with a state-dependent Jacobian: on
    those, the assembled potential block would go stale silently rather than
    fail, and the inner preconditioner would need rebuilding here.
    """

    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        A, P = pc.getOperators()
        ctx = P.getPythonContext()
        W = ctx.a.arguments()[0].function_space()

        real = [i for i, V in enumerate(W)
                if V.ufl_element().family() == "Real"]
        if not real:
            raise ValueError(
                f"{type(self).__name__} needs Real sub-fields to split off, "
                "but the mixed space has none; use a plain fieldsplit.")
        i_R, n = real[0], len(real)
        # Anything but a contiguous trailing run of Real sub-fields would
        # leave sub-fields out of both blocks - a silently wrong split.
        if real != list(range(i_R, len(W))):
            raise ValueError(
                f"{type(self).__name__} requires the Real sub-fields to be "
                f"contiguous and last, but sub-fields {real} of {len(W)} are "
                "Real.")
        if i_R == 0:
            raise ValueError(
                f"{type(self).__name__} requires at least one non-Real "
                "sub-field to form the first block, but sub-field 0 is Real.")

        # field_ises is Firedrake's own authority on where each sub-field lives
        # in the monolithic row space, and the merged sets are exact in-order
        # concatenations of them, which is what lets the matrix-free submatrix
        # extraction recognise them as whole fields.
        field_ises = W.dof_dset.field_ises

        def merge(ises):
            indices = np.concatenate([iset.getIndices() for iset in ises])
            return PETSc.IS().createGeneral(
                indices.astype(PETSc.IntType), comm=pc.comm)

        inner = PETSc.PC().create(comm=pc.comm)
        inner.incrementTabLevel(1, parent=pc)
        inner.setOptionsPrefix((pc.getOptionsPrefix() or "") + "dtn_")
        inner.setOperators(A, P)
        inner.setType(PETSc.PC.Type.FIELDSPLIT)
        inner.setFieldSplitIS(("0", merge(field_ises[:i_R])),
                              ("1", merge(field_ises[i_R:i_R + n])))
        inner.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        inner.setFromOptions()
        inner.setUp()

        # Pre-registering the index sets skips the branch of
        # PCFieldSplitSetDefaults that would otherwise hand each sub-KSP a
        # sub-DM, and AssembledPC (hence SPDAssembledPC) resolves its function
        # space and its split solver context through that DM. Thread it on by
        # hand: createSubDM invokes Firedrake's own hook, which pushes the
        # split context onto the new DM as a side effect. The DM is left
        # inactive so that KSPSetUp does not try to build operators from it.
        ksp_potential, ksp_real = inner.getFieldSplitSchurGetSubKSP()
        _, subdm = pc.getDM().createSubDM(list(range(i_R)))
        ksp_potential.setDM(subdm)
        ksp_potential.setDMActive(PETSc.KSP.DMActive.ALL, False)
        # **And the same for the multiplier KSP**, which had none. Without it
        # anything on block 1 that resolves a DM - any `PCBase` subclass, since
        # `get_appctx` goes through `pc.getDM()` - has no context to resolve.
        # The recorded symptom is `AttributeError: 'NoneType' object has no
        # attribute 'appctx'`, which names neither this preconditioner nor the
        # block it came from. **Measured here it is worse than that: reverting
        # these three lines and running
        # `tests/unit/test_dtn_multiplier_pc.py::TestTheSolveAgrees` gives a
        # SEGMENTATION FAULT (exit 139), not a Python exception** - so there is
        # not even a traceback to read. That is why the four options-file
        # routes to this block all dead-end, and why `DtNMultiplierDiagPC`
        # below could not exist as shipped code until now. Three lines, and
        # they unlock the whole family.
        _, subdm_real = pc.getDM().createSubDM(list(range(i_R, len(W))))
        ksp_real.setDM(subdm_real)
        ksp_real.setDMActive(PETSc.KSP.DMActive.ALL, False)

        self.pc = inner

    def update(self, pc: PETSc.PC):
        """Updates the preconditioner state; nothing to do here.

        The gravitational Poisson Jacobian is constant by construction: the
        density and the gravitational constant enter the residual only through
        terms linear in the test function, so they vanish under
        differentiation, and every remaining coefficient (the Robin shift, the
        DtN eigenvalues and the constraint-row scalings) is fixed when the form
        is built. A repeated setup would therefore only rebuild the index sets
        the preconditioner already holds.

        Args:
          pc: PETSc preconditioner.
        """
        pass

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        """Applies the inner fieldsplit.

        Args:
          pc: PETSc preconditioner.
          x: Vector the preconditioner is applied to.
          y: Vector receiving the result.
        """
        self.pc.apply(x, y)

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        """Applies the transpose of the inner fieldsplit.

        Args:
          pc: PETSc preconditioner.
          x: Vector the preconditioner is applied to.
          y: Vector receiving the result.
        """
        self.pc.applyTranspose(x, y)

    def view(self, pc: PETSc.PC, viewer=None):
        """Prints a description of the preconditioner.

        Args:
          pc: PETSc preconditioner.
          viewer: PETSc viewer.
        """
        super().view(pc, viewer)
        # The base class quietly returns on a missing or non-ASCII viewer, so
        # repeat its test before writing anything of our own.
        if viewer is None or viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        if hasattr(self, "pc"):
            viewer.printfASCII("Two-block Schur fieldsplit defined by index sets\n")
            self.pc.view(viewer)


class _RealBlockPCBase(fd.preconditioners.base.PCBase):
    """Shared plumbing for the two multiplier preconditioners.

    The multiplier block is tiny -- 72 `Real` fields at L = 5, 75 with rotation
    -- so the linear algebra is done in numpy redundantly on every rank, which
    for a 75x75 problem is free and avoids needing a parallel dense solver.

    **The arithmetic here is layout-agnostic and nothing in it assumes where
    the `Real` dofs live.** `_gather` zero-fills, writes only its own
    `[lo:hi]` slice and `Allreduce`s; `apply` writes only the local owned slice
    of the redundantly-computed solution. Both are correct under *any*
    distribution, including a future Firedrake that scatters the `Real` block
    across ranks. That today every `Real` dof happens to sit on rank 0
    (measured: at 48 ranks the block-1 index set is 75 contiguous entries all
    owned by rank 0) is **descriptive performance context, not a load-bearing
    invariant** -- recorded that way so nobody "fixes" correct code.

    The one genuine ordering dependency is global-dof order against
    `multiplier_keys` order, which the merged index set supplies and which
    `GIASpaceLayout.real_fields` pins.

    Neither ever asks anyone to assemble or differentiate anything on the
    `Real` space, which is what lets them work where `jacobi`, `selfp`,
    `AssembledPC` and `-pc_fieldsplit_schur_precondition full` all fail. Those
    four are dead for reasons that are properties of the *route*, not of the
    object; see `DtNTwoBlockSchurPC` and the record in the B2 probe.
    """

    def initialize(self, pc):
        raise NotImplementedError

    def update(self, pc):
        """Nothing to rebuild: the coupled Jacobian is constant by construction.

        Same argument as `DtNTwoBlockSchurPC.update`.
        """

    @staticmethod
    def _loud(exc):
        """Write the message to stderr, then hand the exception back to raise.

        **PETSc flattens a Python exception raised inside a python PC into
        `PETSc.Error: error code 101`** -- measured, on exactly the misuse this
        class most expects. The exception object is then useless to whoever
        reads the log, because the text never reaches them. Printing it first
        puts the named cause directly above the 101 in the output, which is the
        only place it can still be read.

        Returns the exception rather than raising it, so the call site keeps
        `raise` and static analysis still sees a raise.
        """
        print(f"\n[{__name__}] {type(exc).__name__}: {exc}\n",
              file=sys.stderr, flush=True)
        return exc

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

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is not None:
            viewer.printfASCII(f"{type(self).__name__} on {self._n} Real rows\n")


class DtNMultiplierDiagPC(_RealBlockPCBase):
    r"""Invert the multiplier block's diagonal exactly. It is diagonal, and known.

    **Opt-in. This is not any preset's default and must not be made one** --
    both shipped presets run block 1 at `pc_type: none`, and flipping that would
    silently move every number the current campaign is producing. Select it by
    name::

        "dtn_fieldsplit_1_pc_type": "python",
        "dtn_fieldsplit_1_pc_python_type": "gadopt.DtNMultiplierDiagPC",

    The `(c, c)` entries are exact and readable straight off the form as
    `-theta_psi * scale_k * A_h` (`DtNGravityForm.multiplier_diagonal`, which
    derives the sign and the discrete area there), with no assembly and no
    `MatGetDiagonal`. Setup cost is therefore *nothing* -- no block-0 solves, no
    factorisation -- which is what makes this the one to reach for first.

    What it misses is the Schur correction `C A00^{-1} B`, the DtN feedback
    through the potential, which is small for a boundary stood off to 2 Re and
    smaller for higher modes. So it is an approximate inverse of S, not an
    exact one, and it is only ever used as a preconditioner.

    ## Scoped to `SelfGravitatingGIASolver`. Not for `GravitySolver`.

    `DtNTwoBlockSchurPC` serves two solvers and **their block-1 rows carry
    different scalings**: the coupled solver multiplies every constraint row by
    `theta_psi`, `GravitySolver` leaves them unscaled. So one diagonal cannot
    serve both, and if this ever becomes a default it must be wired **per
    solver**, not on the shared preconditioner.

    Two independent reasons to leave the gravity-alone path at `pc_type: none`,
    both measured rather than argued:

    1. **The diagonal differs.** `gadopt/gravity_solver.py` applies no row
       scaling, so its `(c,c)` entries are `-scale_k * A_h` with no `theta_psi`.
       Using the coupled diagonal there would be wrong by that factor.
    2. **That path owns the shipped, verified adjoint**
       (`tests/unit/test_gravity_adjoint.py`), and it is *not* the deferred
       one -- while this class takes its diagonal from the **appctx**, which
       `DtNTwoBlockSchurPC`'s own docstring records pyadjoint **dropping from
       the kwargs of the adjoint solve**. That is precisely why that class
       introspects the operator instead of using appctx. An appctx-carried
       diagonal is therefore the wrong mechanism on the one path with a live
       adjoint, and no guard in this class can see the difference: the count is
       right, the forward solve is right, and the failure appears only on a
       taped replay.

    `GravitySolver` supplies no `dtn_block1_diagonal` at all, so selecting this
    preconditioner there fails rather than degrading -- but **measured, it
    fails as a bare `PETSc.Error: error code 101`**, naming neither this class
    nor the missing key, because the `ValueError` below is raised inside a
    python PC and PETSc flattens it. That is the same unhelpful failure mode
    recorded elsewhere in this project, and it is another reason the wiring
    must be per solver rather than shared. Anyone wiring it for that solver must supply the **unscaled**
    diagonal and must first establish what happens to appctx under `pyadjoint`.

    For the coupled solver the same appctx caveat applies to *its* adjoint,
    which is deferred by decision. The failure mode is a loud `ValueError`
    naming the trap, never a silent zero.

    The diagonal is taken from the appctx under `dtn_block1_diagonal`, which
    the coupled solver supplies; reaching it requires the sub-DM this module
    now threads onto the multiplier KSP.

    Measured on the 3-D coarse coupled system: block-0 applications 111 -> 48,
    wall time 1.8x. Those figures come from the B2 probe and are **not** yet
    reproduced by a shipped-code verification run.
    """

    _prefix = "dtn_multiplier_diag_"

    def initialize(self, pc):
        appctx = self.get_appctx(pc)
        diag = appctx.get("dtn_block1_diagonal")
        if diag is None:
            raise self._loud(ValueError(
                "DtNMultiplierDiagPC needs the block-1 diagonal in the "
                "appctx under 'dtn_block1_diagonal'. SelfGravitatingGIASolver "
                "supplies it; a caller assembling this system by hand must "
                "pass appctx={'dtn_block1_diagonal': ...} with one entry per "
                "Real sub-field, multipliers first then any rotation rows.\n"
                "Two common causes, and the second is not a mistake in your "
                "options:\n"
                "  * this is GravitySolver, which supplies no diagonal and "
                "whose rows are UNSCALED - see the class docstring; that path "
                "is deliberately left at pc_type: none.\n"
                "  * this is an ADJOINT solve. pyadjoint drops appctx from the "
                "kwargs of the adjoint solve (see DtNTwoBlockSchurPC), so a "
                "preconditioner that reads appctx cannot be used on a taped "
                "replay. This is a loud failure by design; it must never "
                "become a silent zero diagonal."))
        A, _ = pc.getOperators()
        self._n = A.getSizes()[0][1]
        diag = np.asarray(diag, dtype=float)
        if diag.size != self._n:
            raise self._loud(ValueError(
                f"the multiplier block is {self._n} wide but the supplied "
                f"diagonal has {diag.size} entries. If rotation is on, the "
                "three closure rows must be included; if it is off, they must "
                "not be."))
        if np.any(diag == 0.0):
            raise self._loud(ValueError(
                "the block-1 diagonal contains a zero entry, so it cannot be "
                "inverted; a mode with zero scale or a boundary with zero "
                "discrete area would do that, and both are bugs upstream."))
        self._d = diag

    def _solve(self, rhs):
        return rhs / self._d
