r"""This module contains classes that augment default Firedrake preconditioners.

"""

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
        ksp_potential, _ = inner.getFieldSplitSchurGetSubKSP()
        _, subdm = pc.getDM().createSubDM(list(range(i_R)))
        ksp_potential.setDM(subdm)
        ksp_potential.setDMActive(PETSc.KSP.DMActive.ALL, False)

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
