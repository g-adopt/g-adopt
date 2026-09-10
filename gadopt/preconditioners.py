r"""This module contains classes that augment default Firedrake preconditioners.

"""

import functools

import firedrake as fd
from ufl.indexed import Indexed
from firedrake.petsc import PETSc
from firedrake.assemble import get_assembler
from firedrake import dmhooks
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

    The coupled internal-variable Jacobian has a displacement block and one
    cell-local block for the internal-variable field (a DG mass matrix scaled
    by `1/dt + 1/tau_i` per Maxwell element, plus the cross coupling between
    elements that a power-law rheology introduces). Slate eliminates that
    block cell by cell and assembles the condensed displacement operator

    $$ S = A_{uu} - A_{uM} A_{MM}^{-1} A_{Mu} $$

    as a sparse matrix on the displacement space, which is the exact Schur
    complement of the coupled system. The condensed system is solved with the
    options under the `condensed_field_` prefix, and the internal variables
    are recovered locally afterwards. Every Maxwell element lives in the one
    field, so the elimination is always of field 1 and the number of elements
    never appears here. For a power-law rheology the cross coupling between
    elements sits inside the `M` block, so its Slate inverse is the exact
    tangent elimination.

    `SCPC`'s own Schur builder assumes the eliminated fields come before the
    kept one (its default keeps the last field), and fails on the `(u, M)`
    layout where the displacement comes first. This class therefore writes
    the condensed system and the back-substitution itself, in
    `condensed_system` and `local_solver_calls`, as one Slate inverse of the
    `M` block. On top of that it adds two things to `SCPC`.

    **Nullspaces.** `SCPC` reads `condensed_field_nullspace` from the
    application context and sets it on the condensed operator. GAMG on the
    displacement operator also needs the rigid-body modes as a near-nullspace,
    and CG on a singular operator needs the transpose nullspace to make the
    right-hand side consistent, so this class also reads
    `condensed_field_near_nullspace` and `condensed_field_transpose_nullspace`.
    Each is a callable taking the condensed space and returning a
    `VectorSpaceBasis`; `CoupledInternalVariableSolver` sets them from its
    `nullspace`, `transpose_nullspace` and `near_nullspace` arguments.

    **Operator reuse.** `SCPC.update` reassembles the condensed matrix on every
    preconditioner setup, which PETSc requests on every linear solve. A
    Newtonian march with a fixed time step has a constant Jacobian, so the
    condensed matrix and the GAMG hierarchy built on it can serve the whole
    march. This class reads `operator_version` from the application context
    and reassembles only when the version differs from the one the current
    matrix was built with. `None` means "always rebuild", which the solver
    publishes when the Jacobian depends on the solution (a power-law
    rheology, where the operator changes within one Newton solve). Without
    the key the class behaves as `SCPC` does. `assembly_count` records how
    many times the condensed operator has been assembled.

    This class reads the attributes `cxt`, `weight`, `S` and `S_pc` of `SCPC`,
    and `CoupledInternalVariableSolver` reads `_bases` of Firedrake's
    `MixedVectorSpaceBasis` to hand over the displacement nullspaces. Both
    were written against Firedrake main of 2026-08-06 (450aa7905); a
    Firedrake update can change any of these.

    Select with `pc_type: python`, `pc_python_type: gadopt.InternalVariableSCPC`
    and `pc_sc_eliminate_fields: "1"`. The operator must be matrix-free
    (`mat_type: matfree`).
    """

    def condensed_system(self, A, rhs, elim_fields, prefix, pc):
        blocks = A.blocks
        vectors = AssembledVector(rhs).blocks
        if list(elim_fields) != [1] or len(rhs.subfunctions) != 2:
            raise ValueError(
                "InternalVariableSCPC keeps the displacement (field 0) and "
                "eliminates the internal-variable field (field 1) of a two-field "
                f"space; got eliminated fields {list(elim_fields)} of "
                f"{len(rhs.subfunctions)}."
            )
        # One Slate inverse of the cell-local internal-variable block, which
        # for a power-law rheology also holds the coupling between elements.
        inverse = blocks[1, 1].inv
        condensed_operator = blocks[0, 0] - blocks[0, 1] * inverse * blocks[1, 0]
        condensed_rhs = vectors[0] - blocks[0, 1] * inverse * vectors[1]
        return LAContext(condensed_operator, condensed_rhs, (0,)), inverse

    def local_solver_calls(self, A, rhs, solution, elim_fields, inverse):
        displacement = AssembledVector(solution.subfunctions[0])
        field_rhs = AssembledVector(rhs.subfunctions[1])
        # Back-substitute: M = A_MM^-1 (b_M - A_Mu u), cell by cell.
        local_solution = inverse * (field_rhs - A.blocks[1, 0] * displacement)
        return [
            functools.partial(
                get_assembler(
                    local_solution, form_compiler_parameters=self.cxt.fc_params
                ).assemble,
                tensor=solution.subfunctions[1],
            )
        ]

    @PETSc.Log.EventDecorator("SCPCInit")
    def initialize(self, pc):
        """Build the condensed system, its solver and the local recovery.

        This is `SCPC.initialize` of Firedrake main (2026-08-06) with one
        change: a strong boundary condition on a *component* of the
        displacement (`ux`, `uy`, `uz` in the G-ADOPT boundary dictionary)
        is carried over to the same component of the condensed space, where
        `SCPC` refuses it because the component subspace has no field index
        of its own. `SCPC`'s guard against more than three fields is dropped
        on purpose: `condensed_system` enforces the two-field layout. Everything
        else is unchanged, so the attributes the rest of the class reads
        (`cxt`, `weight`, `S`, `S_pc`, `condensed_ksp`) are the ones `SCPC`
        defines.
        """
        from firedrake.bcs import DirichletBC
        from firedrake.function import Function
        from firedrake.cofunction import Cofunction
        from firedrake.functionspace import FunctionSpace
        from firedrake.matrix_free.operators import ImplicitMatrixContext
        from firedrake.parloops import par_loop, INC
        from firedrake.slate.slate import Tensor
        from ufl import dx
        import numpy as np

        prefix = (pc.getOptionsPrefix() or "") + "condensed_field_"
        A, P = pc.getOperators()
        self.cxt = A.getPythonContext()
        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        self.bilinear_form = self.cxt.a

        # Retrieve the mixed function space
        W = self.bilinear_form.arguments()[0].function_space()

        elim_option = (pc.getOptionsPrefix() or "") + "pc_sc_eliminate_fields"
        # By default, we condense down to the last field in the mixed space.
        elim_fields = PETSc.Options().getIntArray(elim_option, range(len(W) - 1))
        elim_fields = list(map(int, elim_fields))

        condensed_fields = list(set(range(len(W))) - set(elim_fields))
        if len(condensed_fields) != 1:
            raise NotImplementedError("Cannot condense to more than one field")

        c_field, = condensed_fields

        # Need to duplicate a space which is NOT
        # associated with a subspace of a mixed space.
        Vc = FunctionSpace(W.mesh()[c_field], W[c_field].ufl_element())
        bcs = []
        for bc in self.cxt.row_bcs:
            space = bc.function_space()
            # A component subspace (`Z.sub(0).sub(1)`) has no index of its
            # own; its field is the index of its parent and its component
            # selects the same component of the condensed space.
            component = space.component
            field = space.parent.index if component is not None else space.index
            if field != c_field:
                raise NotImplementedError("Strong BC set on unsupported space")
            target = Vc if component is None else Vc.sub(component)
            bcs.append(DirichletBC(target, 0, bc.sub_domain))

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        self.c_field = c_field
        self.condensed_rhs = Cofunction(Vc.dual())
        self.residual = Cofunction(W.dual())
        self.solution = Function(W)

        shapes = (Vc.finat_element.space_dimension(), np.prod(Vc.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        self.weight = Function(Vc)
        par_loop((domain, instructions), dx, {"w": (self.weight, INC)})
        with self.weight.dat.vec as wc:
            wc.reciprocal()

        # Get expressions for the condensed linear system
        A_tensor = Tensor(self.bilinear_form)
        reduced_sys, schur_builder = self.condensed_system(
            A_tensor, self.residual, elim_fields, prefix, pc
        )
        S_expr = reduced_sys.lhs
        r_expr = reduced_sys.rhs

        # Construct the condensed right-hand side
        self._assemble_Srhs = get_assembler(
            r_expr, bcs=bcs, form_compiler_parameters=self.cxt.fc_params
        ).assemble

        # Allocate and set the condensed operator
        form_assembler = get_assembler(
            S_expr,
            bcs=bcs,
            form_compiler_parameters=self.cxt.fc_params,
            mat_type=mat_type,
            options_prefix=prefix,
            appctx=self.get_appctx(pc),
        )
        self.S = form_assembler.allocate()
        self._assemble_S = form_assembler.assemble

        self._assemble_S(tensor=self.S)
        Smat = self.S.petscmat

        # If a different matrix is used for preconditioning,
        # assemble this as well
        if A != P:
            self.cxt_pc = P.getPythonContext()
            P_tensor = Tensor(self.cxt_pc.a)
            P_reduced_sys, _ = self.condensed_system(
                P_tensor, self.residual, elim_fields, prefix, pc
            )
            S_pc_expr = P_reduced_sys.lhs
            self.S_pc_expr = S_pc_expr

            # Allocate and set the condensed operator
            form_assembler = get_assembler(
                S_pc_expr,
                bcs=bcs,
                form_compiler_parameters=self.cxt.fc_params,
                mat_type=mat_type,
                options_prefix=prefix,
                appctx=self.get_appctx(pc),
            )
            self.S_pc = form_assembler.allocate()
            self._assemble_S_pc = form_assembler.assemble

            self._assemble_S_pc(tensor=self.S_pc)
            Smat_pc = self.S_pc.petscmat

        else:
            self.S_pc_expr = S_expr
            Smat_pc = Smat

        # Get nullspace for the condensed operator (if any).
        # This is provided as a user-specified callback which
        # returns the basis for the nullspace.
        nullspace = self.cxt.appctx.get("condensed_field_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(Vc)
            Smat.setNullSpace(nsp.nullspace())

        # Create a SNESContext for the DM associated with the trace problem
        self._ctx_ref = self.new_snes_ctx(
            pc, S_expr, bcs, mat_type, self.cxt.fc_params, options_prefix=prefix
        )

        # Push new context onto the dm associated with the condensed problem
        c_dm = Vc.dm

        # Set up ksp for the condensed problem
        c_ksp = PETSc.KSP().create(comm=pc.comm)
        c_ksp.incrementTabLevel(1, parent=pc)

        # Set the dm for the condensed solver
        c_ksp.setDM(c_dm)
        c_ksp.setDMActive(PETSc.KSP.DMActive.ALL, False)
        c_ksp.setOptionsPrefix(prefix)
        c_ksp.setOperators(A=Smat, P=Smat_pc)
        self.condensed_ksp = c_ksp

        with dmhooks.add_hooks(c_dm, self, appctx=self._ctx_ref, save=False):
            c_ksp.setFromOptions()

        # Set up local solvers for backwards substitution
        self.local_solvers = self.local_solver_calls(
            A_tensor, self.residual, self.solution, elim_fields, schur_builder
        )

        # Additions to `SCPC.initialize` start here.
        self.assembly_count = 1
        self._operator_version = self._published_version()

        appctx = self.cxt.appctx
        matrices = [Smat]
        if hasattr(self, "S_pc"):
            matrices.append(Smat_pc)
        transpose_nullspace = appctx.get("condensed_field_transpose_nullspace")
        if transpose_nullspace is not None:
            basis = transpose_nullspace(Vc).nullspace()
            for matrix in matrices:
                matrix.setTransposeNullSpace(basis)
        near_nullspace = appctx.get("condensed_field_near_nullspace")
        if near_nullspace is not None:
            basis = near_nullspace(Vc).nullspace()
            for matrix in matrices:
                matrix.setNearNullSpace(basis)

    def _published_version(self):
        """The operator version the solver publishes, or a sentinel.

        The sentinel is a fresh object, so a context without the key never
        compares equal to a stored version and the operator is rebuilt on
        every update, which is what `SCPC` does.
        """
        return self.cxt.appctx.get("operator_version", object())

    def update(self, pc):
        version = self._published_version()
        if version is not None and version == self._operator_version:
            return
        self._operator_version = version
        self.assembly_count += 1
        super().update(pc)
