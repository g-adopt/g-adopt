r"""This module contains classes that augment default Firedrake preconditioners.

"""

from math import isfinite

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
    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


class _WeightedDivergenceGradient:
    """Matrix-free action of ``divergence * mass^-1 * gradient``."""

    def __init__(self, divergence, inverse_velocity_mass, gradient):
        self.divergence = divergence
        self.inverse_velocity_mass = inverse_velocity_mass
        self.gradient = gradient
        self.velocity_work = gradient.createVecLeft()

    def mult(self, mat, x, y):
        """Apply the weighted pressure Laplacian."""
        self.gradient.mult(x, self.velocity_work)
        self.velocity_work.pointwiseMult(
            self.inverse_velocity_mass, self.velocity_work
        )
        self.divergence.mult(self.velocity_work, y)

    def multTranspose(self, mat, x, y):
        """Apply the transpose of the weighted pressure Laplacian."""
        self.divergence.multTranspose(x, self.velocity_work)
        self.velocity_work.pointwiseMult(
            self.inverse_velocity_mass, self.velocity_work
        )
        self.gradient.multTranspose(self.velocity_work, y)

    def destroy(self, mat):
        """Destroy PETSc work vectors owned by this context."""
        self.velocity_work.destroy()


class DensityAwareBFBTPC(fd.PCBase):
    r"""Density-aware weighted BFBT approximation of a Schur inverse.

    For a saddle-point Jacobian with velocity block :math:`A`, pressure
    gradient :math:`G`, and density-weighted continuity block
    :math:`D_\rho`, this class applies

    .. math::

       (D_\rho C^{-1}G)^{-1}
       D_\rho C^{-1} A C^{-1}G
       (D_\rho C^{-1}G)^{-1}.

    Here :math:`C` is a diagonal approximation to the velocity mass matrix
    weighted by :math:`\sqrt{\mu}`. This is the weighted BFBT construction of
    Rudi, Stadler, and Ghattas (2017), generalised to retain G-ADOPT's actual
    left and right off-diagonal blocks. In particular, TALA/ALA continuity
    uses :math:`D_\rho` rather than replacing it by :math:`G^T`.

    Each inverse weighted pressure Laplacian is computed by an inner KSP with
    options prefix ``bfbt_``. Its exact operator is the matrix-free product
    :math:`D_\rho C^{-1}G`; an assembled
    :math:`(\rho/\sqrt{\mu})`-weighted pressure Laplacian is supplied as the
    inner preconditioning matrix. For ALA, the auxiliary operator also
    includes the pressure-buoyancy contribution to :math:`G`, retaining its
    non-constant right nullspace. The defaults are FGMRES with GAMG and can be
    overridden, for example, using ``bfbt_ksp_type`` and ``bfbt_pc_type``.
    A failed inner solve raises by default rather than silently returning a
    corrupted preconditioner application. Set
    ``bfbt_raise_on_inner_failure false`` only for controlled diagnostics.

    Transpose application is deliberately unsupported. A
    convergence-controlled Krylov inverse is not a fixed linear map, and even
    a ``preonly`` configuration would require an explicit contract that the
    selected inner preconditioner implements its numerical transpose. Failing
    clearly is safer than silently producing an incorrect adjoint.

    ``bfbt_mass_lumping`` may be ``diagonal`` (the default) or ``rowsum``.
    Row-sum lumping is the construction used in the original paper, but can
    have zero entries for higher-order nodal elements on simplices. The
    positive mass diagonal is therefore the robust default for G-ADOPT's
    extruded spherical meshes.

    The square-root viscosity is interpolated into a discontinuous scalar
    coefficient before assembling the auxiliary forms. This keeps a complex
    nonlinear rheology out of TSFC's quadrature-degree estimation and avoids
    differentiating that rheology a second time in the preconditioner. The
    default coefficient is cellwise constant; ``bfbt_weight_degree`` selects
    a higher discontinuous degree if additional within-cell resolution is
    beneficial.

    ``bfbt_nullspace_policy`` controls pressure-gauge treatment. The default,
    ``schur``, uses the same quotient space supplied to the outer Schur solve.
    This is needed for G-ADOPT's ALA gauge, which is not in general an exact
    null mode of the discrete gradient. The implementation tests and reports
    that discrepancy rather than claiming exactness. The alternative
    ``verified`` policy attaches only modes that the weighted pressure
    operators annihilate to ``bfbt_nullspace_test_tolerance``. It is useful
    diagnostically but may leave the near-singular ALA inner solve expensive.

    Reference:
      Rudi, J., Stadler, G., and Ghattas, O. (2017), *Weighted BFBT
      Preconditioner for Stokes Flow Problems with Highly Heterogeneous
      Viscosity*, SIAM Journal on Scientific Computing, 39(5), S272-S297,
      https://doi.org/10.1137/16M108450X.
    """

    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC) -> None:
        """Build the weighted mass, pressure Laplacian, and inner KSP."""
        from firedrake.assemble import get_assembler

        if pc.getType() != "python":
            raise ValueError("DensityAwareBFBTPC expects a Python PC")

        A, P = pc.getOperators()
        if A.getType() != "schurcomplement":
            raise ValueError(
                "DensityAwareBFBTPC must precondition a PETSc Schur complement"
            )
        if P.getType() != "python":
            raise ValueError(
                "DensityAwareBFBTPC requires the matrix-free pressure block"
            )

        self._set_blocks(A)

        pressure_context = P.getPythonContext()
        pressure_test, pressure_trial = pressure_context.a.arguments()
        if pressure_test.function_space() != pressure_trial.function_space():
            raise ValueError("Pressure test and trial spaces differ")
        pressure_space = pressure_test.function_space()

        if self.velocity.getType() != "python":
            raise ValueError(
                "DensityAwareBFBTPC requires the matrix-free velocity block"
            )
        velocity_context = self.velocity.getPythonContext()
        velocity_test, velocity_trial = velocity_context.a.arguments()
        if velocity_test.function_space() != velocity_trial.function_space():
            raise ValueError("Velocity test and trial spaces differ")
        velocity_space = velocity_test.function_space()

        appctx = self.get_appctx(pc)
        appctx_fcp = appctx.get("form_compiler_parameters") or {}
        velocity_fcp = dict(velocity_context.fc_params or {})
        velocity_fcp.update(appctx_fcp)
        pressure_fcp = dict(pressure_context.fc_params or {})
        pressure_fcp.update(appctx_fcp)
        rho = appctx.get("rho_continuity", 1)
        viscosity = appctx.get("viscosity")
        if viscosity is None:
            # Backward-compatible reconstruction from the application context
            # historically supplied to Firedrake's pressure MassInvPC.
            viscosity = appctx.get("mu", 1) * rho
        weight_expression = appctx.get("bfbt_weight", fd.sqrt(viscosity))

        prefix = (pc.getOptionsPrefix() or "") + "bfbt_"
        opts = PETSc.Options()
        weight_degree = opts.getInt(prefix + "weight_degree", 0)
        if weight_degree < 0:
            raise ValueError("bfbt_weight_degree must be non-negative")
        weight_space = fd.FunctionSpace(
            velocity_space.mesh(), "DG", weight_degree
        )
        self.weight = fd.Function(weight_space, name="BFBTWeight")
        self.weight_expression = weight_expression
        self._update_weight()

        self.mass_lumping = opts.getString(
            prefix + "mass_lumping", "diagonal"
        ).lower()
        if self.mass_lumping not in {"diagonal", "rowsum"}:
            raise ValueError(
                "bfbt_mass_lumping must be either 'diagonal' or 'rowsum'"
            )

        velocity_trial = fd.TrialFunction(velocity_space)
        velocity_test = fd.TestFunction(velocity_space)
        mass_form = fd.inner(
            self.weight * velocity_trial, velocity_test
        ) * fd.dx
        if self.mass_lumping == "diagonal":
            mass_assembler = get_assembler(
                mass_form,
                form_compiler_parameters=velocity_fcp,
                diagonal=True,
            )
        else:
            ones = fd.Function(velocity_space)
            ones.assign(1)
            mass_assembler = get_assembler(
                fd.action(mass_form, ones),
                form_compiler_parameters=velocity_fcp,
            )
        self.weighted_velocity_mass = mass_assembler.allocate()
        self._assemble_weighted_velocity_mass = mass_assembler.assemble
        self.inverse_velocity_mass = self.gradient.createVecLeft()
        self._update_inverse_velocity_mass()

        pressure = fd.TrialFunction(pressure_space)
        pressure_test = fd.TestFunction(pressure_space)
        pressure_gradient = fd.grad(pressure)
        pressure_buoyancy = appctx.get("pressure_buoyancy")
        if pressure_buoyancy is not None:
            pressure_gradient -= pressure_buoyancy * pressure
        weighted_pressure_gradient = (rho / self.weight) * pressure_gradient
        pressure_laplacian = (
            fd.inner(
                weighted_pressure_gradient,
                fd.grad(pressure_test),
            )
            * fd.dx
        )
        lp_mat_type = opts.getString(prefix + "mat_type", "aij")
        lp_assembler = get_assembler(
            pressure_laplacian,
            form_compiler_parameters=pressure_fcp,
            mat_type=lp_mat_type,
            options_prefix=prefix,
        )
        self.pressure_laplacian = lp_assembler.allocate()
        self._assemble_pressure_laplacian = lp_assembler.assemble
        self._assemble_pressure_laplacian(tensor=self.pressure_laplacian)

        laplacian_sizes = (
            self.divergence.getSizes()[0], self.gradient.getSizes()[1]
        )
        laplacian_context = _WeightedDivergenceGradient(
            self.divergence, self.inverse_velocity_mass, self.gradient
        )
        self.exact_pressure_laplacian = PETSc.Mat().createPython(
            laplacian_sizes, context=laplacian_context, comm=pc.comm
        )
        self.exact_pressure_laplacian.setUp()
        self.nullspace_test_tolerance = opts.getReal(
            prefix + "nullspace_test_tolerance", 1e-10
        )
        self.nullspace_policy = opts.getString(
            prefix + "nullspace_policy", "schur"
        ).lower()
        if self.nullspace_policy not in {"schur", "verified"}:
            raise ValueError(
                "bfbt_nullspace_policy must be either 'schur' or 'verified'"
            )
        self._set_pressure_nullspaces(A)

        self.ksp = PETSc.KSP().create(comm=pc.comm)
        self.ksp.incrementTabLevel(1, parent=pc)
        self.ksp.setOptionsPrefix(prefix)
        self.ksp.setType(PETSc.KSP.Type.FGMRES)
        self.ksp.setTolerances(rtol=1e-2, max_it=200)
        self.ksp.getPC().setType(PETSc.PC.Type.GAMG)
        self.ksp.setOperators(
            self.exact_pressure_laplacian,
            self.pressure_laplacian.petscmat,
        )
        self.ksp.setFromOptions()
        self.raise_on_inner_failure = opts.getBool(
            prefix + "raise_on_inner_failure", True
        )
        self.inner_iterations_total = 0
        self.inner_solves_total = 0
        self.inner_failures_total = 0
        self.last_inner_iterations = ()
        self.last_inner_reasons = ()

        pressure_0 = self.gradient.createVecRight()
        pressure_1 = self.divergence.createVecLeft()
        velocity_0 = self.gradient.createVecLeft()
        velocity_1 = self.velocity.createVecLeft()
        self.workspace = (pressure_0, pressure_1, velocity_0, velocity_1)

    def _set_blocks(self, schur_complement: PETSc.Mat) -> None:
        """Store the current Jacobian blocks underlying a Schur matrix."""
        velocity, _, gradient, divergence, _ = (
            schur_complement.getSchurComplementSubMatrices()
        )
        self.velocity = velocity
        self.gradient = gradient
        self.divergence = divergence

    def _set_pressure_nullspaces(self, schur_complement: PETSc.Mat) -> None:
        """Transfer pressure quotient data according to the selected policy.

        G-ADOPT's analytical ALA pressure gauge is generally only an
        approximate null mode of the discrete gradient. The default ``schur``
        policy deliberately preserves the outer Schur quotient on both inner
        operators; ``verified`` attaches a mode as exact only after the
        absolute residual test controlled by
        ``bfbt_nullspace_test_tolerance``. The latter is a diagnostic whose
        tolerance must be scaled for the operator and units being tested.
        """
        empty_nullspace = PETSc.NullSpace()
        self.exact_pressure_laplacian.setNullSpace(empty_nullspace)
        self.exact_pressure_laplacian.setTransposeNullSpace(empty_nullspace)
        self.pressure_laplacian.petscmat.setNullSpace(empty_nullspace)
        self.pressure_laplacian.petscmat.setTransposeNullSpace(empty_nullspace)
        self.pressure_laplacian.petscmat.setNearNullSpace(empty_nullspace)

        nullspace = schur_complement.getNullSpace()
        self.right_nullspace_is_exact = None
        self.auxiliary_right_nullspace_is_exact = None
        if nullspace.handle != 0:
            exact_is_null = self._is_exact_nullspace(
                self.exact_pressure_laplacian,
                nullspace,
            )
            auxiliary_is_null = self._is_exact_nullspace(
                self.pressure_laplacian.petscmat,
                nullspace,
            )
            self.right_nullspace_is_exact = exact_is_null
            self.auxiliary_right_nullspace_is_exact = auxiliary_is_null
            if exact_is_null or self.nullspace_policy == "schur":
                self.exact_pressure_laplacian.setNullSpace(nullspace)
            if auxiliary_is_null or self.nullspace_policy == "schur":
                self.pressure_laplacian.petscmat.setNullSpace(nullspace)
            else:
                self.pressure_laplacian.petscmat.setNearNullSpace(nullspace)

        transpose_nullspace = schur_complement.getTransposeNullSpace()
        self.left_nullspace_is_exact = None
        self.auxiliary_left_nullspace_is_exact = None
        if transpose_nullspace.handle != 0:
            exact_is_left_null = self._is_exact_nullspace(
                self.exact_pressure_laplacian,
                transpose_nullspace,
                transpose=True,
            )
            auxiliary_is_left_null = self._is_exact_nullspace(
                self.pressure_laplacian.petscmat,
                transpose_nullspace,
                transpose=True,
            )
            self.left_nullspace_is_exact = exact_is_left_null
            self.auxiliary_left_nullspace_is_exact = auxiliary_is_left_null
            if exact_is_left_null or self.nullspace_policy == "schur":
                self.exact_pressure_laplacian.setTransposeNullSpace(
                    transpose_nullspace
                )
            if auxiliary_is_left_null or self.nullspace_policy == "schur":
                self.pressure_laplacian.petscmat.setTransposeNullSpace(
                    transpose_nullspace
                )

        near_nullspace = schur_complement.getNearNullSpace()
        if (
            near_nullspace.handle != 0
            and self.pressure_laplacian.petscmat.getNearNullSpace().handle == 0
        ):
            self.pressure_laplacian.petscmat.setNearNullSpace(near_nullspace)

    def _is_exact_nullspace(
        self,
        operator: PETSc.Mat,
        nullspace: PETSc.NullSpace,
        *,
        transpose: bool = False,
    ) -> bool:
        """Return whether every supplied null vector is discretely annihilated."""
        has_constant = nullspace.hasConstant()
        candidates = list(nullspace.getVecs())
        constant = None
        if has_constant:
            constant = (
                operator.createVecLeft()
                if transpose
                else operator.createVecRight()
            )
            constant.set(1)
            constant.normalize()
            candidates.append(constant)

        residual = (
            operator.createVecRight()
            if transpose
            else operator.createVecLeft()
        )
        try:
            for candidate in candidates:
                if transpose:
                    operator.multTranspose(candidate, residual)
                else:
                    operator.mult(candidate, residual)
                if residual.norm() > self.nullspace_test_tolerance:
                    return False
        finally:
            residual.destroy()
            if constant is not None:
                constant.destroy()
        return True

    def _update_inverse_velocity_mass(self) -> None:
        """Reassemble and invert the selected weighted mass diagonal."""
        self._assemble_weighted_velocity_mass(
            tensor=self.weighted_velocity_mass
        )
        with self.weighted_velocity_mass.dat.vec_ro as mass:
            index, minimum = mass.min()
            if not isfinite(minimum) or minimum <= 0:
                raise ValueError(
                    f"BFBT {self.mass_lumping} velocity-mass entry {index} "
                    f"is non-positive ({minimum}). Use "
                    "-bfbt_mass_lumping diagonal for higher-order simplex "
                    "elements and check that viscosity is positive."
                )
            mass.copy(self.inverse_velocity_mass)
        self.inverse_velocity_mass.reciprocal()

    def _update_weight(self) -> None:
        """Interpolate the current square-root viscosity for cheap assembly."""
        self.weight.interpolate(self.weight_expression)

    def update(self, pc: PETSc.PC) -> None:
        """Update state-dependent weights and auxiliary operators."""
        A, _ = pc.getOperators()
        self._set_blocks(A)
        laplacian_context = self.exact_pressure_laplacian.getPythonContext()
        laplacian_context.velocity = self.velocity
        laplacian_context.gradient = self.gradient
        laplacian_context.divergence = self.divergence

        self._update_weight()
        self._update_inverse_velocity_mass()
        self.exact_pressure_laplacian.assemble()
        self._assemble_pressure_laplacian(tensor=self.pressure_laplacian)
        self._set_pressure_nullspaces(A)
        self.ksp.setOperators(
            self.exact_pressure_laplacian,
            self.pressure_laplacian.petscmat,
        )

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Apply the density-aware weighted BFBT Schur inverse."""
        pressure_0, pressure_1, velocity_0, velocity_1 = self.workspace
        iterations = []
        reasons = []
        self._solve_inner(x, pressure_0, "left", iterations, reasons)
        self.gradient.mult(pressure_0, velocity_0)
        velocity_0.pointwiseMult(self.inverse_velocity_mass, velocity_0)
        self.velocity.mult(velocity_0, velocity_1)
        velocity_1.pointwiseMult(self.inverse_velocity_mass, velocity_1)
        self.divergence.mult(velocity_1, pressure_1)
        self._solve_inner(pressure_1, y, "right", iterations, reasons)
        self.last_inner_iterations = tuple(iterations)
        self.last_inner_reasons = tuple(reasons)

    def _solve_inner(
        self,
        rhs: PETSc.Vec,
        solution: PETSc.Vec,
        side: str,
        iterations: list[int],
        reasons: list[int],
        *,
        transpose: bool = False,
    ) -> None:
        """Apply one weighted pressure inverse and record its outcome."""
        if transpose:
            self.ksp.solveTranspose(rhs, solution)
        else:
            self.ksp.solve(rhs, solution)
        reason = self.ksp.getConvergedReason()
        iteration_count = self.ksp.getIterationNumber()
        self.inner_solves_total += 1
        self.inner_iterations_total += iteration_count
        iterations.append(iteration_count)
        reasons.append(reason)
        if reason <= 0:
            self.inner_failures_total += 1
            if self.raise_on_inner_failure:
                raise RuntimeError(
                    f"BFBT {side} inner pressure solve failed after "
                    f"{iteration_count} iterations with PETSc reason {reason}."
                )

    def applyTranspose(
        self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec
    ) -> None:
        """Reject transpose use until the inner-PC contract is established."""
        raise NotImplementedError(
            "DensityAwareBFBTPC is currently forward-only; a tested "
            "transpose contract for every selectable inner PC is not yet "
            "available."
        )

    def view(self, pc: PETSc.PC, viewer=None) -> None:
        """Display the BFBT construction and its inner solver."""
        super().view(pc, viewer)
        if viewer is None or viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(
            "Density-aware weighted BFBT Schur inverse\n"
            f"Velocity mass approximation: {self.mass_lumping}\n"
            f"Pressure nullspace policy: {self.nullspace_policy}\n"
            f"Right null mode verified exact: {self.right_nullspace_is_exact}\n"
            f"Left null mode verified exact: {self.left_nullspace_is_exact}\n"
            "Inner weighted pressure-Laplacian KSP:\n"
        )
        self.ksp.view(viewer)

    def destroy(self, pc: PETSc.PC) -> None:
        """Destroy PETSc objects owned by this preconditioner."""
        if hasattr(self, "workspace"):
            for vector in self.workspace:
                vector.destroy()
        if hasattr(self, "inverse_velocity_mass"):
            self.inverse_velocity_mass.destroy()
        if hasattr(self, "ksp"):
            self.ksp.destroy()
        if hasattr(self, "exact_pressure_laplacian"):
            self.exact_pressure_laplacian.destroy()
        if hasattr(self, "pressure_laplacian"):
            self.pressure_laplacian.petscmat.destroy()
