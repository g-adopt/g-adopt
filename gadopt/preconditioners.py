r"""This module contains classes that augment default Firedrake preconditioners.

"""

import numpy as np

import firedrake as fd
from firedrake.petsc import PETSc
from ufl.indexed import Indexed

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

    For use in the velocity fieldsplit_0 block in combination with gamg.
    Setting PETSc MatOption MAT_SPD (for Symmetric Positive Definite matrices)
    at the moment only changes the Krylov method for the eigenvalue
    estimate in the Chebyshev smoothers to CG.

    Users can provide this class as a `pc_python_type`
    entry to a PETSc solver option dictionary.

    """
    def initialize(self, pc: PETSc.PC) -> None:
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


class BalancedConformalPC(SPDAssembledPC):
    r"""Add a balanced four-mode correction to rigid-mode GAMG.

    This preconditioner expects the complete ten-dimensional conformal Killing
    near-nullspace on a three-dimensional velocity block, ordered as the
    :class:`~gadopt.nullspaces.ConformalKillingNearNullspace` builder produces
    it: three rotations, three translations, one dilation, and three special
    conformal modes.  Only the first six rigid modes are attached to the
    internally assembled matrix and therefore passed to the configured inner
    preconditioner (normally GAMG).

    The remaining four modes form the columns of ``W`` in the balanced coarse
    correction

    .. math::

       Q &= W (W^T A W)^{-1} W^T, \\
       M_B^{-1} &= Q + (I - QA) M^{-1} (I - AQ),

    where ``A`` is the assembled velocity preconditioning operator and
    ``M^{-1}`` is the configured ``assembled_`` preconditioner.  For symmetric
    ``A`` and ``M^{-1}``, this construction is symmetric and can be used with
    conjugate gradients.  It retains the complete conformal coarse space
    without requiring every first-level GAMG aggregate to represent ten local
    candidates.

    The coarse matrix is only four by four and is replicated on every rank.
    It is rebuilt whenever Firedrake updates the assembled operator.
    """

    _rigid_mode_count = 6
    _conformal_mode_count = 4

    def initialize(self, pc: PETSc.PC) -> None:
        """Initialise the assembled preconditioner and balanced correction."""
        super().initialize(pc)

        matrix = self.P.petscmat
        complete_near_nullspace = matrix.getNearNullSpace()
        if complete_near_nullspace.handle == 0:
            raise ValueError(
                "BalancedConformalPC requires the complete conformal "
                "near-nullspace."
            )

        complete_modes = tuple(complete_near_nullspace.getVecs())
        expected_mode_count = self._rigid_mode_count + self._conformal_mode_count
        if len(complete_modes) != expected_mode_count:
            raise ValueError(
                "BalancedConformalPC requires exactly ten near-null modes "
                f"but received {len(complete_modes)}."
            )

        # Retain the complete PETSc object so that its borrowed vectors remain
        # valid after replacing the near-nullspace attached to the matrix.
        self._complete_near_nullspace = complete_near_nullspace
        self._rigid_modes = complete_modes[:self._rigid_mode_count]
        self._conformal_modes = complete_modes[self._rigid_mode_count:]
        self._rigid_near_nullspace = PETSc.NullSpace().create(
            vectors=self._rigid_modes,
            comm=pc.comm,
        )
        matrix.setNearNullSpace(self._rigid_near_nullspace)

        self._coarse_projection = matrix.createVecRight()
        self._balanced_residual = matrix.createVecRight()
        self._inner_correction = matrix.createVecRight()
        self._operator_action = matrix.createVecLeft()
        self._projected_operator_action = matrix.createVecRight()
        self._mode_operator_action = matrix.createVecLeft()
        self._build_coarse_matrix()

    def _build_coarse_matrix(self) -> None:
        """Assemble and factor the four-dimensional Galerkin operator."""
        matrix = self.P.petscmat
        mode_count = self._conformal_mode_count
        coarse_matrix = np.empty(
            (mode_count, mode_count),
            dtype=PETSc.ScalarType,
        )

        for column, mode in enumerate(self._conformal_modes):
            matrix.mult(mode, self._mode_operator_action)
            coarse_matrix[:, column] = self._mode_operator_action.mDot(
                self._conformal_modes
            )

        # Roundoff and parallel reductions can introduce a minute skew part.
        coarse_matrix = (coarse_matrix + coarse_matrix.T.conj()) / 2
        try:
            coarse_factor = np.linalg.cholesky(coarse_matrix)
        except np.linalg.LinAlgError as error:
            eigenvalues = np.linalg.eigvalsh(coarse_matrix)
            raise ValueError(
                "The conformal coarse operator is not positive definite; "
                f"eigenvalues are {eigenvalues}."
            ) from error

        self.coarse_matrix = coarse_matrix
        self.coarse_factor = coarse_factor
        self.coarse_condition_number = float(np.linalg.cond(coarse_matrix))

    def _apply_coarse_projection(self, source: PETSc.Vec, result: PETSc.Vec) -> None:
        """Apply ``W (W^T A W)^-1 W^T`` to ``source``."""
        coarse_rhs = np.asarray(
            source.mDot(self._conformal_modes),
            dtype=PETSc.ScalarType,
        )
        intermediate = np.linalg.solve(self.coarse_factor, coarse_rhs)
        coarse_solution = np.linalg.solve(
            self.coarse_factor.T.conj(),
            intermediate,
        )

        result.set(0)
        result.maxpy(coarse_solution, self._conformal_modes)

    def _apply_balanced(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec, *, transpose: bool) -> None:
        """Apply the balanced coarse correction and configured inner PC."""
        matrix = self.P.petscmat

        # Right projection: r = (I - A Q) x.
        self._apply_coarse_projection(x, self._coarse_projection)
        matrix.mult(self._coarse_projection, self._balanced_residual)
        self._balanced_residual.scale(-1)
        self._balanced_residual.axpy(1, x)

        if transpose:
            super().applyTranspose(
                pc,
                self._balanced_residual,
                self._inner_correction,
            )
        else:
            super().apply(
                pc,
                self._balanced_residual,
                self._inner_correction,
            )

        # Left projection and exact coarse correction:
        # y = (I - Q A) z + Q x.
        matrix.mult(self._inner_correction, self._operator_action)
        self._apply_coarse_projection(
            self._operator_action,
            self._projected_operator_action,
        )
        self._inner_correction.copy(y)
        y.axpy(-1, self._projected_operator_action)
        y.axpy(1, self._coarse_projection)

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Apply the balanced conformal preconditioner."""
        self._apply_balanced(pc, x, y, transpose=False)

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Apply the transpose balanced conformal preconditioner."""
        self._apply_balanced(pc, x, y, transpose=True)

    def update(self, pc: PETSc.PC) -> None:
        """Update the assembled operator and four-dimensional correction."""
        super().update(pc)
        self.P.petscmat.setNearNullSpace(self._rigid_near_nullspace)
        self._build_coarse_matrix()

    def view(self, pc: PETSc.PC, viewer=None) -> None:
        """Display the inner preconditioner and coarse-space diagnostics."""
        super().view(pc, viewer)
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT(pc.comm)
        viewer.printfASCII(
            "Balanced conformal correction with 6 GAMG modes and 4 global "
            f"modes (coarse condition number {self.coarse_condition_number:g})\n"
        )
