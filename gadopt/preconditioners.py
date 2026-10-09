r"""This module contains classes that augment default Firedrake preconditioners.

"""

import firedrake as fd
from firedrake.petsc import PETSc
from ufl.indexed import Indexed

from .utility import InteriorBC


class FreeSurfaceMassInvPC(fd.MassInvPC):
    """Version of MassInvPC that includes free surface variables.

    The free-surface deflection's diagonal block in the Schur complement of the coupled
    system carries two contributions: a mass term, scaled by the free-surface buoyancy
    and the time step, `buoyancy * theta / dt`, which dominates for time steps well
    below the free-surface relaxation time; and the velocity-mediated coupling, scaled
    by `1 / mu`, which dominates otherwise. Including both terms, with the buoyancy
    prefactor of the free-surface equations, makes the preconditioner consistent across
    regimes and material parameterisations.
    """

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
        theta = appctx.get("theta")
        dt = appctx.get("dt")
        bcs = []
        for bc_id, (eta_ind, buoyancy) in appctx["free_surface"].items():
            # Without a time step, only the velocity-mediated coupling contributes,
            # matching the scaling of the pressure mass
            scale = 1 / mu
            if dt is not None and theta is not None:
                scale = scale + buoyancy * theta / dt
            a += scale * fd.inner(trials[eta_ind - 1], tests[eta_ind - 1]) * ds(bc_id)

            bcs.append(InteriorBC(trials.function_space()[eta_ind - 1], 0, bc_id))

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
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)
