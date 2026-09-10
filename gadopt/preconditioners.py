r"""This module contains classes that augment default Firedrake preconditioners.

"""

import functools

import firedrake as fd
from ufl.indexed import Indexed
from firedrake.petsc import PETSc
from firedrake.assemble import get_assembler
from firedrake.dmhooks import get_function_space
from firedrake.slate import AssembledVector
from firedrake.slate.static_condensation.la_utils import LAContext
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
    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


class SubstitutedDisplacementPC(fd.AuxiliaryOperatorPC):
    r"""Precondition the displacement Schur complement of a coupled GIA solve.

    `CoupledInternalVariableSolver` keeps the internal variables `m_i` as
    unknowns next to the displacement `u`. Eliminating them exactly from the
    coupled Jacobian gives the displacement operator of the substituted solver
    (`InternalVariableSolver`), because the backward-Euler update

    $$ m_i^{new} = \frac{m_i^{old} + (dt/\tau_i)\,d(u)}{1 + dt/\tau_i} $$

    is linear in the deviatoric strain `d(u)` and the internal-variable block
    is a cell-local DG mass matrix. The eliminated operator is therefore the
    elastic displacement block minus the viscous part of the shear response:

    $$ S = A_{uu} - \sum_i 2\mu_i \frac{dt}{\tau_i + dt}
           \int d(v) : d(u) \, dx $$

    which has shear coefficient `eta_eff = sum_i mu_i tau_i / (tau_i + dt)`.

    This class builds that operator as the preconditioning matrix for the
    displacement split of a Schur fieldsplit that eliminates the internal
    variables first. It reuses the elastic block `A_uu` handed to it by the
    fieldsplit, which already carries the Nitsche boundary terms and the
    free-surface prestress term, and subtracts the volume correction above.
    It differs from the exact Schur complement in two places. The exact
    complement carries the L2 projection of `d(u)` onto the DQ1
    internal-variable space, which for a Q2 displacement on hexahedra is not
    the identity, while the correction here uses `d(u)` itself; that is a
    volume difference. And the boundary terms of `A_uu` keep the elastic
    coefficient `mu0`, so on a boundary with a weak normal condition the two
    differ by a boundary term as well. GAMG then runs on a sparse matrix that
    is spectrally close to the substituted operator.

    The application context must carry:

    - `approximation`: the `InternalVariableApproximation` of the solve, for
      `shear_modulus`, `maxwell_times` and `deviatoric_strain`.
    - `dt`: the time step.
    - `scaling_factor` (optional, default 1): the factor the coupled solver
      applies to the displacement residual.
    - `displacement_near_nullspace` (optional): a callable that takes the
      displacement space and returns a `VectorSpaceBasis`. It is used when the
      fieldsplit hands over no near-nullspace, which is what GAMG needs for
      the rigid-body modes.

    Options for the inner solver use the `aux_` prefix, for example
    `fieldsplit_1_aux_pc_type: gamg`. The matrix is flagged SPD.
    """

    def form(self, pc: PETSc.PC, test, trial):
        _, P = pc.getOperators()
        if P.getType() != "python":
            raise ValueError(
                f"{type(self).__name__} needs a matrix-free operator "
                "(mat_type matfree) so that it can read the displacement "
                "block's bilinear form."
            )
        context = P.getPythonContext()
        if not context.on_diag:
            raise ValueError("Only makes sense to precondition a diagonal block")
        elastic_block = context.a
        bcs = context.row_bcs

        appctx = self.get_appctx(pc)
        missing = [key for key in ("approximation", "dt") if key not in appctx]
        if missing:
            raise KeyError(
                f"{type(self).__name__} needs {missing} in the application "
                "context; CoupledInternalVariableSolver.set_solver_options sets them."
            )
        approximation = appctx["approximation"]
        dt = appctx["dt"]
        scaling_factor = appctx.get("scaling_factor", 1)

        # Use the arguments of the extracted block so that the correction lives
        # on the same (sub)space as the elastic block it is added to.
        block_test, block_trial = elastic_block.arguments()
        # Reuse the quadrature degree of the elastic block's cell integrals so
        # that the two parts of the form are integrated consistently.
        degrees = {
            integral.metadata().get("quadrature_degree")
            for integral in elastic_block.integrals()
            if integral.integral_type() == "cell"
        }
        degrees.discard(None)
        dx = fd.dx(degree=max(degrees)) if degrees else fd.dx

        # Viscous part of the shear response that the eliminated internal
        # variables remove from the elastic block. Each Maxwell element with
        # shear modulus mu_i and Maxwell time tau_i contributes the fraction
        # dt/(tau_i + dt) of its elastic stiffness after one implicit step.
        # These are the unscaled Maxwell times: for a power-law rheology the
        # residual scales them by the power-law factor, so this operator is
        # right for a Newtonian rheology only.
        strain = approximation.deviatoric_strain(block_trial)
        correction = 0
        for mu_i, tau_i in zip(
            approximation.shear_modulus, approximation.maxwell_times
        ):
            correction += 2 * mu_i * dt / (tau_i + dt)
        viscous_part = (
            scaling_factor
            * fd.inner(fd.nabla_grad(block_test), correction * strain)
            * dx
        )
        return elastic_block - viscous_part, bcs

    def set_nullspaces(self, pc: PETSc.PC):
        """Copy the parent nullspaces, and build the near-nullspace if absent."""
        super().set_nullspaces(pc)
        Pmat = self.P.petscmat
        if Pmat.getNearNullSpace().handle != 0:
            return
        builder = self.get_appctx(pc).get("displacement_near_nullspace")
        if builder is None:
            return
        V = get_function_space(pc.getDM()).collapse()
        Pmat.setNearNullSpace(builder(V).nullspace())

    def initialize(self, pc: PETSc.PC):
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


class InternalVariableSCPC(fd.SCPC):
    r"""Static condensation of the DG internal variables of a coupled GIA solve.

    The coupled internal-variable Jacobian has one displacement block and one
    cell-local block for each internal variable (a DG mass matrix scaled by
    `1/dt + 1/tau`). Slate eliminates those blocks cell by cell and assembles
    the condensed displacement operator

    $$ S = A_{uu} - \sum_i A_{u m_i} A_{m_i m_i}^{-1} A_{m_i u} $$

    as a sparse matrix on the displacement space, which is the exact Schur
    complement of the coupled system. The condensed system is solved with the
    options under the `condensed_field_` prefix, and the internal variables
    are recovered locally afterwards.

    Firedrake's `SCPC` accepts at most three fields, so this class handles one
    or two internal variables. The Schur fieldsplit layout in
    ``tests/3d_sphere_burgers/coupled_solver_variants.py`` has no such limit.

    The elimination treats the internal-variable blocks as mutually uncoupled.
    For Newtonian rheology that is exact. For power-law rheology the Maxwell
    times depend on the deviatoric stress, which couples every internal
    variable to every other one through the Jacobian; those cross blocks are
    dropped here, so the linear solve is approximate and Newton pays for it
    with extra iterations. The nonlinear residual is untouched.

    Firedrake's `SCPC` eliminates every field at once through its
    `SchurComplementBuilder`. The internal-variable blocks do not couple to
    each other, so this class inverts each cell-local block on its own
    instead, one Slate inverse per field, and back-substitutes per field.

    Firedrake's `SCPC` reads `condensed_field_nullspace` from the application
    context and sets it on the condensed operator only. GAMG on the elastic
    displacement operator also needs the rigid-body modes as a near-nullspace,
    and on the preconditioning matrix when one is assembled, so this class
    sets both from two callables in the application context, each taking the
    displacement space and returning a `VectorSpaceBasis`:
    `condensed_field_nullspace` and `condensed_field_near_nullspace`.

    This class overrides `condensed_system` and `local_solver_calls` and reads
    the private attributes `cxt`, `weight`, `S` and `S_pc` of `SCPC`. It was
    written against Firedrake main of 2026-08-06 (450aa7905); a Firedrake
    update can change any of these.

    Select with `pc_type: python`, `pc_python_type: gadopt.InternalVariableSCPC`
    and `pc_sc_eliminate_fields: "1,2"` (every internal-variable field index).
    The operator must be matrix-free (`mat_type: matfree`).
    """

    def condensed_system(self, A, rhs, elim_fields, prefix, pc):
        blocks = A.blocks
        vectors = AssembledVector(rhs).blocks
        kept = [index for index in range(len(rhs.subfunctions)) if index not in elim_fields]
        if kept != [0]:
            raise ValueError(
                "InternalVariableSCPC keeps the displacement (field 0) and "
                f"eliminates every other field; got eliminated fields {elim_fields}."
            )
        # Per-field inverse of the cell-local internal-variable block.
        inverse_blocks = tuple(blocks[i, i].inv for i in elim_fields)
        condensed_operator = blocks[0, 0]
        condensed_rhs = vectors[0]
        for field, inverse in zip(elim_fields, inverse_blocks):
            condensed_operator -= blocks[0, field] * inverse * blocks[field, 0]
            condensed_rhs -= blocks[0, field] * inverse * vectors[field]
        return LAContext(condensed_operator, condensed_rhs, (0,)), inverse_blocks

    def local_solver_calls(self, A, rhs, solution, elim_fields, inverse_blocks):
        calls = []
        displacement = AssembledVector(solution.subfunctions[0])
        for field, inverse in zip(elim_fields, inverse_blocks):
            field_rhs = AssembledVector(rhs.subfunctions[field])
            # Back-substitute: m_i = A_ii^-1 (b_i - A_iu u), cell by cell.
            local_solution = inverse * (field_rhs - A.blocks[field, 0] * displacement)
            calls.append(
                functools.partial(
                    get_assembler(
                        local_solution,
                        form_compiler_parameters=self.cxt.fc_params,
                    ).assemble,
                    tensor=solution.subfunctions[field],
                )
            )
        return calls

    def initialize(self, pc):
        super().initialize(pc)
        appctx = self.cxt.appctx
        condensed_space = self.weight.function_space()
        matrices = [self.S.petscmat]
        if hasattr(self, "S_pc"):
            matrices.append(self.S_pc.petscmat)
        nullspace = appctx.get("condensed_field_nullspace")
        if nullspace is not None:
            basis = nullspace(condensed_space).nullspace()
            for matrix in matrices:
                matrix.setNullSpace(basis)
        near_nullspace = appctx.get("condensed_field_near_nullspace")
        if near_nullspace is not None:
            basis = near_nullspace(condensed_space).nullspace()
            for matrix in matrices:
                matrix.setNearNullSpace(basis)
