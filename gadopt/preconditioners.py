r"""This module contains classes that augment default Firedrake preconditioners."""

import firedrake as fd

from .utility import InteriorBC


class FreeSurfaceMassInvPC(fd.MassInvPC):
    """Version of MassInvPC that includes free surface variables."""

    def form(
        self,
        pc: fd.PETSc.PC,
        tests: list[fd.Argument | fd.ufl.indexed.Indexed],
        trials: list[fd.Argument | fd.ufl.indexed.Indexed | fd.Function],
    ) -> tuple[fd.Form, list[fd.DirichletBC]]:
        """Sets the form.

        Args:
          pc:
            PETSc preconditioner.
          tests:
            List of Firedrake test functions.
          trials:
            List of Firedrake trial functions.
        """
        appctx = self.get_appctx(pc)

        # N.B. trials[0] is pressure
        mu = appctx.get("mu", 1.0)
        a = fd.inner(1 / mu * trials[0], tests[0]) * fd.dx

        ds = appctx["ds"]
        bcs = []
        for bc_id, eta_ind in appctx["free_surface"].items():
            a += 1 / mu * fd.inner(trials[1 + eta_ind], tests[1 + eta_ind]) * ds(bc_id)

            bcs.append(InteriorBC(trials.function_space()[1 + eta_ind], 0, bc_id))

        return a, bcs


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in the velocity fieldsplit_0 block in combination with gamg.

    Setting PETSc MatOption MAT_SPD (for Symmetric Positive Definite matrices) at the
    moment only changes the Krylov method for the eigenvalue estimate in the Chebyshev
    smoothers to CG.

    Users can provide this class as a `pc_python_type` entry to a PETSc solver option
    dictionary.

    """

    def initialize(self, pc: fd.PETSc.PC) -> None:
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)
