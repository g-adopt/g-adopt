r"""This module contains classes that augment default Firedrake preconditioners.

"""

import firedrake as fd
from firedrake.petsc import PETSc
from .utility import InteriorBC


class FreeSurfaceMassInvPC(fd.MassInvPC):

    """Version of MassInvPC that includes free surface variables.

    """
    def form(self, pc, test, trial):
        appctx = self.get_appctx(pc)
        mu = appctx.get("mu", 1.0)
        a = fd.inner(1/mu * trial[0], test[0])*fd.dx
        bcs = []
        ds = appctx['ds']
        c = 0  # Counter for free surfaces, N.b the first trial[0] is pressure
        for id in appctx["free_surface_id_list"]:
            a += fd.inner(1/mu * trial[1+c], test[1+c])*ds(id)
            bcs.append(InteriorBC(trial.function_space().sub(1+c), 0, id))
            c += 1
        return a, bcs


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in the velocity fieldsplit_0 block in combination with gamg.
    Setting PETSc MatOption MAT_SPD (for Symmetric Positive Definite matrices)
    at the moment only changes the Krylov method for the eigenvalue
    estimate in the Chebyshev smoothers to CG.

    Users can provide this class as a `pc_python_type`
    entry to a PETSc solver option dictionary.

    """
    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc:
            PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)
