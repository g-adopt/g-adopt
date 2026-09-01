r"""This module contains classes that augment default Firedrake preconditioners.

"""

from math import isfinite

import firedrake as fd
from firedrake.petsc import PETSc
from ufl.indexed import Indexed

from .utility import InteriorBC


_BFBT_APPLY_EVENT = PETSc.Log.Event("GAdoptBFBTApply")
_BFBT_RIGHT_SOLVE_EVENT = PETSc.Log.Event("GAdoptBFBTRightSolve")
_BFBT_MIDDLE_EVENT = PETSc.Log.Event("GAdoptBFBTMiddle")
_BFBT_LEFT_SOLVE_EVENT = PETSc.Log.Event("GAdoptBFBTLeftSolve")


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


class _BFBTPressureInverse:
    """Objects owned by one algebraic side of weighted BFBT."""

    def __init__(
        self,
        *,
        side,
        weight_expression,
        weight_space,
        velocity_space,
        pressure_space,
        rho,
        pressure_buoyancy,
        mass_lumping,
        velocity_fcp,
        pressure_fcp,
        mat_type,
        options_prefix,
        divergence,
        gradient,
        pc,
    ):
        from firedrake.assemble import get_assembler

        self.side = side
        self.weight_expression = weight_expression
        self.weight = fd.Function(
            weight_space,
            name=f"BFBT{side.capitalize()}Weight",
        )
        self.weight.interpolate(weight_expression)

        velocity_trial = fd.TrialFunction(velocity_space)
        velocity_test = fd.TestFunction(velocity_space)
        mass_form = fd.inner(
            self.weight * velocity_trial,
            velocity_test,
        ) * fd.dx
        self.mass_ones = None
        if mass_lumping == "diagonal":
            mass_assembler = get_assembler(
                mass_form,
                form_compiler_parameters=velocity_fcp,
                diagonal=True,
            )
        else:
            self.mass_ones = fd.Function(velocity_space)
            self.mass_ones.assign(1)
            mass_assembler = get_assembler(
                fd.action(mass_form, self.mass_ones),
                form_compiler_parameters=velocity_fcp,
            )
        self.weighted_velocity_mass = mass_assembler.allocate()
        self.assemble_weighted_velocity_mass = mass_assembler.assemble
        self.inverse_velocity_mass = gradient.createVecLeft()
        self.update_inverse_velocity_mass(mass_lumping)

        pressure = fd.TrialFunction(pressure_space)
        pressure_test = fd.TestFunction(pressure_space)
        pressure_gradient = fd.grad(pressure)
        if pressure_buoyancy is not None:
            pressure_gradient -= pressure_buoyancy * pressure
        pressure_laplacian = (
            fd.inner(
                (rho / self.weight) * pressure_gradient,
                fd.grad(pressure_test),
            )
            * fd.dx
        )
        laplacian_assembler = get_assembler(
            pressure_laplacian,
            form_compiler_parameters=pressure_fcp,
            mat_type=mat_type,
            options_prefix=options_prefix,
        )
        self.pressure_laplacian = laplacian_assembler.allocate()
        self.assemble_pressure_laplacian = laplacian_assembler.assemble
        self.assemble_pressure_laplacian(tensor=self.pressure_laplacian)

        sizes = (divergence.getSizes()[0], gradient.getSizes()[1])
        context = _WeightedDivergenceGradient(
            divergence,
            self.inverse_velocity_mass,
            gradient,
        )
        self.exact_pressure_laplacian = PETSc.Mat().createPython(
            sizes,
            context=context,
            comm=pc.comm,
        )
        self.exact_pressure_laplacian.setUp()
        self.ksp = PETSc.KSP().create(comm=pc.comm)
        self.ksp.incrementTabLevel(1, parent=pc)
        self.ksp.setOptionsPrefix(options_prefix)
        self.ksp.setType(PETSc.KSP.Type.FGMRES)
        self.ksp.setTolerances(rtol=1e-2, max_it=200)
        self.ksp.getPC().setType(PETSc.PC.Type.GAMG)
        self.set_ksp_operators()
        self.ksp.setFromOptions()
        self.inner_initial_guess_was_overridden = (
            self.ksp.getInitialGuessNonzero()
        )
        self.ksp.setInitialGuessNonzero(False)

    def update_inverse_velocity_mass(self, mass_lumping):
        """Reassemble and invert this side's weighted mass diagonal."""
        self.assemble_weighted_velocity_mass(
            tensor=self.weighted_velocity_mass
        )
        with self.weighted_velocity_mass.dat.vec_ro as mass:
            index, minimum = mass.min()
            if not isfinite(minimum) or minimum <= 0:
                raise ValueError(
                    f"BFBT {self.side} {mass_lumping} velocity-mass entry "
                    f"{index} is non-positive ({minimum}). Use "
                    "-bfbt_mass_lumping diagonal for higher-order simplex "
                    "elements and check that viscosity is positive."
                )
            mass.copy(self.inverse_velocity_mass)
        self.inverse_velocity_mass.reciprocal()

    def set_ksp_operators(self):
        """Associate the exact action and assembled preconditioning matrix."""
        self.ksp.setOperators(
            self.exact_pressure_laplacian,
            self.pressure_laplacian.petscmat,
        )

    def update(self, gradient, divergence, mass_lumping):
        """Refresh coefficients and block references after a Jacobian update."""
        context = self.exact_pressure_laplacian.getPythonContext()
        context.gradient = gradient
        context.divergence = divergence
        self.weight.interpolate(self.weight_expression)
        self.update_inverse_velocity_mass(mass_lumping)
        self.exact_pressure_laplacian.assemble()
        self.assemble_pressure_laplacian(tensor=self.pressure_laplacian)
        self.pressure_laplacian.petscmat.assemble()

    def destroy(self):
        """Destroy PETSc resources owned by this side."""
        self.inverse_velocity_mass.destroy()
        self.ksp.destroy()
        self.exact_pressure_laplacian.destroy()
        self.pressure_laplacian.petscmat.destroy()


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
    includes the pressure-buoyancy contribution to :math:`G`; this makes that
    experimental preconditioning matrix nonsymmetric, so its GAMG behaviour
    must be validated at production scale. The defaults are FGMRES with GAMG
    and can be overridden, for example, using ``bfbt_ksp_type`` and
    ``bfbt_pc_type``.
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
    ``verified``, attaches a mode as an exact PETSc nullspace only when the
    weighted pressure operator annihilates it under an absolute test augmented
    by a scale proxy from the assembled auxiliary operator. If the outer Schur
    matrix omits transpose-nullspace
    metadata, a verified right mode is tested independently as a left
    candidate. It is attached to each inner matrix only when that matrix's
    transpose action also passes. The alternative ``schur`` policy uses the
    same quotient space supplied to the outer Schur solve even when the mode
    is not an exact discrete null mode. It exists only for controlled ALA
    experiments and must be selected explicitly.

    The convergence-controlled inner FGMRES makes this a variable
    preconditioner. The enclosing pressure solver must therefore use a
    flexible Krylov method such as FGMRES unless the inner application is a
    fixed linear operation, such as ``preonly`` or a prescribed number of
    norm-free Richardson iterations.

    Reference:
      Rudi, J., Stadler, G., and Ghattas, O. (2017), *Weighted BFBT
      Preconditioner for Stokes Flow Problems with Highly Heterogeneous
      Viscosity*, SIAM Journal on Scientific Computing, 39(5), S272-S297,
      https://doi.org/10.1137/16M108450X.
    """

    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC) -> None:
        """Build the weighted mass, pressure Laplacian, and inner KSP."""
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
        self._validate_pressure_space(pressure_space)

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

        self.mass_lumping = opts.getString(
            prefix + "mass_lumping", "diagonal"
        ).lower()
        if self.mass_lumping not in {"diagonal", "rowsum"}:
            raise ValueError(
                "bfbt_mass_lumping must be either 'diagonal' or 'rowsum'"
            )
        pressure_buoyancy = appctx.get("pressure_buoyancy")
        lp_mat_type = opts.getString(prefix + "mat_type", "aij")
        self.nullspace_test_tolerance = opts.getReal(
            prefix + "nullspace_test_tolerance", 1e-12
        )
        self.nullspace_test_relative_tolerance = opts.getReal(
            prefix + "nullspace_test_relative_tolerance", 1e-12
        )
        self._validate_nullspace_test_tolerances(
            self.nullspace_test_tolerance,
            self.nullspace_test_relative_tolerance,
        )
        self.nullspace_policy = opts.getString(
            prefix + "nullspace_policy", "verified"
        ).lower()
        if self.nullspace_policy not in {"schur", "verified"}:
            raise ValueError(
                "bfbt_nullspace_policy must be either 'schur' or 'verified'"
            )

        def build_side(side, weight_expression):
            return _BFBTPressureInverse(
                side=side,
                weight_expression=weight_expression,
                weight_space=weight_space,
                velocity_space=velocity_space,
                pressure_space=pressure_space,
                rho=rho,
                pressure_buoyancy=pressure_buoyancy,
                mass_lumping=self.mass_lumping,
                velocity_fcp=velocity_fcp,
                pressure_fcp=pressure_fcp,
                mat_type=lp_mat_type,
                options_prefix=prefix,
                divergence=self.divergence,
                gradient=self.gradient,
                pc=pc,
            )

        shared_side = build_side("shared", weight_expression)
        self.sides = {"right": shared_side, "left": shared_side}

        default_inner_rtol = shared_side.ksp.getTolerances()[0]
        self.right_inner_rtol = opts.getReal(
            prefix + "right_ksp_rtol", default_inner_rtol
        )
        self.left_inner_rtol = opts.getReal(
            prefix + "left_ksp_rtol", default_inner_rtol
        )
        for side, tolerance in (
            ("right", self.right_inner_rtol),
            ("left", self.left_inner_rtol),
        ):
            if not isfinite(tolerance) or not 0 < tolerance < 1:
                raise ValueError(
                    f"bfbt_{side}_ksp_rtol must be finite and between zero "
                    "and one"
                )
        self._set_pressure_nullspaces(A)
        self._set_legacy_side_aliases()
        self.raise_on_inner_failure = opts.getBool(
            prefix + "raise_on_inner_failure", True
        )
        self.inner_iterations_total = 0
        self.inner_solves_total = 0
        self.inner_failures_total = 0
        self.update_count = 0
        self.last_inner_iterations = ()
        self.last_inner_reasons = ()
        self.last_inner_tolerances = ()
        self.inner_iterations_by_side = {"left": 0, "right": 0}
        self.inner_solves_by_side = {"left": 0, "right": 0}
        self.inner_failures_by_side = {"left": 0, "right": 0}
        self.inner_initial_guess_was_overridden = any(
            bundle.inner_initial_guess_was_overridden
            for bundle in self._unique_sides()
        )

        pressure_0 = self.gradient.createVecRight()
        pressure_1 = self.divergence.createVecLeft()
        velocity_0 = self.gradient.createVecLeft()
        velocity_1 = self.velocity.createVecLeft()
        self.workspace = (pressure_0, pressure_1, velocity_0, velocity_1)

    @staticmethod
    def _validate_nullspace_test_tolerances(
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> None:
        """Reject tolerances that could silently accept every null mode."""
        if not isfinite(absolute_tolerance) or absolute_tolerance < 0:
            raise ValueError(
                "bfbt_nullspace_test_tolerance must be finite and non-negative"
            )
        if not isfinite(relative_tolerance) or relative_tolerance < 0:
            raise ValueError(
                "bfbt_nullspace_test_relative_tolerance must be finite and "
                "non-negative"
            )

    @staticmethod
    def _validate_pressure_space(
        pressure_space: fd.functionspaceimpl.WithGeometry,
    ) -> None:
        """Reject coupled free-surface Schur blocks not represented by BFBT."""
        if len(pressure_space) != 1 or pressure_space.value_shape != ():
            raise ValueError(
                "DensityAwareBFBTPC supports a scalar pressure Schur block "
                "only; coupled pressure/free-surface blocks require "
                "FreeSurfaceMassInvPC or a dedicated surface-aware BFBT."
            )

    def _set_blocks(self, schur_complement: PETSc.Mat) -> None:
        """Store the current Jacobian blocks underlying a Schur matrix."""
        velocity, _, gradient, divergence, _ = (
            schur_complement.getSchurComplementSubMatrices()
        )
        self.velocity = velocity
        self.gradient = gradient
        self.divergence = divergence

    def _unique_sides(self):
        """Yield each independently owned side bundle exactly once."""
        seen = set()
        for side in ("right", "left"):
            bundle = self.sides[side]
            identity = id(bundle)
            if identity not in seen:
                seen.add(identity)
                yield bundle

    def _set_legacy_side_aliases(self) -> None:
        """Retain the original single-weight diagnostic API on the right side."""
        bundle = self.sides["right"]
        self.weight = bundle.weight
        self.weight_expression = bundle.weight_expression
        self.weighted_velocity_mass = bundle.weighted_velocity_mass
        self.inverse_velocity_mass = bundle.inverse_velocity_mass
        self.pressure_laplacian = bundle.pressure_laplacian
        self.exact_pressure_laplacian = bundle.exact_pressure_laplacian
        self.ksp = bundle.ksp
        diagnostic_names = (
            "exact_right_nullspace_attached",
            "auxiliary_right_nullspace_attached",
            "exact_left_nullspace_attached",
            "auxiliary_left_nullspace_attached",
            "nullspace_test_auxiliary_operator_scale",
            "nullspace_test_threshold",
            "right_nullspace_source",
            "right_nullspace_is_exact",
            "auxiliary_right_nullspace_is_exact",
            "right_nullspace_residual",
            "auxiliary_right_nullspace_residual",
            "left_nullspace_source",
            "left_nullspace_fallback_used",
            "left_nullspace_is_exact",
            "auxiliary_left_nullspace_is_exact",
            "left_nullspace_residual",
            "auxiliary_left_nullspace_residual",
        )
        for name in diagnostic_names:
            setattr(self, name, getattr(bundle, name))

    def _set_pressure_nullspaces(self, schur_complement: PETSc.Mat) -> None:
        """Transfer pressure quotient data according to the selected policy.

        G-ADOPT's analytical ALA pressure gauge is generally only an
        approximate null mode of the discrete gradient. The default
        ``verified`` policy attaches a mode as exact only after an absolute
        residual test augmented by an assembled-auxiliary-operator scale
        proxy. The explicitly selected experimental
        ``schur`` policy instead preserves an actual outer-Schur quotient on
        both inner operators. A right mode used to repair missing transpose
        metadata is never force-attached by the ``schur`` policy.
        """
        right_nullspace = schur_complement.getNullSpace()
        if right_nullspace.handle != 0:
            self.schur_right_nullspace = right_nullspace
            right_source = "schur"
        else:
            right_nullspace = getattr(
                self,
                "schur_right_nullspace",
                right_nullspace,
            )
            right_source = (
                "cached_schur" if right_nullspace.handle != 0 else "none"
            )
        transpose_nullspace = schur_complement.getTransposeNullSpace()
        if transpose_nullspace.handle != 0:
            self.schur_left_nullspace = transpose_nullspace
            left_source = "schur"
        else:
            transpose_nullspace = getattr(
                self,
                "schur_left_nullspace",
                transpose_nullspace,
            )
            left_source = (
                "cached_schur" if transpose_nullspace.handle != 0 else "none"
            )
        near_nullspace = schur_complement.getNearNullSpace()
        if near_nullspace.handle != 0:
            self.schur_near_nullspace = near_nullspace
        else:
            near_nullspace = getattr(
                self,
                "schur_near_nullspace",
                near_nullspace,
            )
        for bundle in self._unique_sides():
            exact = bundle.exact_pressure_laplacian
            auxiliary = bundle.pressure_laplacian.petscmat
            empty_nullspace = PETSc.NullSpace()
            exact.setNullSpace(empty_nullspace)
            exact.setTransposeNullSpace(empty_nullspace)
            auxiliary.setNullSpace(empty_nullspace)
            auxiliary.setTransposeNullSpace(empty_nullspace)
            exact.setNearNullSpace(empty_nullspace)
            auxiliary.setNearNullSpace(empty_nullspace)
            bundle.exact_right_nullspace_attached = False
            bundle.auxiliary_right_nullspace_attached = False
            bundle.exact_left_nullspace_attached = False
            bundle.auxiliary_left_nullspace_attached = False

            bundle.nullspace_test_auxiliary_operator_scale = auxiliary.norm(
                PETSc.NormType.INFINITY
            )
            if not isfinite(bundle.nullspace_test_auxiliary_operator_scale):
                raise ValueError(
                    f"BFBT {bundle.side} auxiliary pressure-operator scale "
                    "is not finite"
                )
            bundle.nullspace_test_threshold = (
                self.nullspace_test_tolerance
                + self.nullspace_test_relative_tolerance
                * bundle.nullspace_test_auxiliary_operator_scale
            )
            if not isfinite(bundle.nullspace_test_threshold):
                raise ValueError(
                    f"BFBT {bundle.side} nullspace acceptance threshold is "
                    "not finite"
                )

            bundle.right_nullspace_source = right_source
            bundle.right_nullspace_is_exact = None
            bundle.auxiliary_right_nullspace_is_exact = None
            bundle.right_nullspace_residual = None
            bundle.auxiliary_right_nullspace_residual = None
            if right_nullspace.handle != 0:
                (
                    bundle.right_nullspace_is_exact,
                    bundle.right_nullspace_residual,
                ) = self._test_nullspace(
                    exact,
                    right_nullspace,
                    bundle.nullspace_test_threshold,
                )
                (
                    bundle.auxiliary_right_nullspace_is_exact,
                    bundle.auxiliary_right_nullspace_residual,
                ) = self._test_nullspace(
                    auxiliary,
                    right_nullspace,
                    bundle.nullspace_test_threshold,
                )

            candidate_left_nullspace = transpose_nullspace
            bundle_left_source = left_source
            if (
                candidate_left_nullspace.handle == 0
                and right_nullspace.handle != 0
                and bundle.right_nullspace_is_exact is True
            ):
                candidate_left_nullspace = right_nullspace
                bundle_left_source = "verified_right_fallback"
            bundle.left_nullspace_source = bundle_left_source
            bundle.left_nullspace_fallback_used = (
                bundle_left_source == "verified_right_fallback"
            )
            bundle.left_nullspace_is_exact = None
            bundle.auxiliary_left_nullspace_is_exact = None
            bundle.left_nullspace_residual = None
            bundle.auxiliary_left_nullspace_residual = None
            if candidate_left_nullspace.handle != 0:
                (
                    bundle.left_nullspace_is_exact,
                    bundle.left_nullspace_residual,
                ) = self._test_nullspace(
                    exact,
                    candidate_left_nullspace,
                    bundle.nullspace_test_threshold,
                    transpose=True,
                )
                (
                    bundle.auxiliary_left_nullspace_is_exact,
                    bundle.auxiliary_left_nullspace_residual,
                ) = self._test_nullspace(
                    auxiliary,
                    candidate_left_nullspace,
                    bundle.nullspace_test_threshold,
                    transpose=True,
                )

            if (
                self.nullspace_policy == "verified"
                and bundle.right_nullspace_is_exact is True
                and bundle.left_nullspace_is_exact is not True
            ):
                self._set_legacy_side_aliases()
                raise ValueError(
                    f"DensityAwareBFBTPC {bundle.side} operator verified an "
                    "exact right pressure nullspace but no compatible "
                    "transpose nullspace for D_rho C^-1 G. Check whether "
                    "rho_continuity is discontinuous or supply consistent "
                    "Schur transpose-nullspace metadata. Measured transpose "
                    f"residual {bundle.left_nullspace_residual}; acceptance "
                    f"threshold {bundle.nullspace_test_threshold}."
                )

            force_right_schur = self.nullspace_policy == "schur"
            if right_nullspace.handle != 0:
                if bundle.right_nullspace_is_exact or force_right_schur:
                    exact.setNullSpace(right_nullspace)
                    bundle.exact_right_nullspace_attached = True
                else:
                    exact.setNearNullSpace(right_nullspace)
                if (
                    bundle.auxiliary_right_nullspace_is_exact
                    or force_right_schur
                ):
                    auxiliary.setNullSpace(right_nullspace)
                    bundle.auxiliary_right_nullspace_attached = True
                else:
                    auxiliary.setNearNullSpace(right_nullspace)

            force_actual_schur_left = (
                self.nullspace_policy == "schur"
                and bundle_left_source in {"schur", "cached_schur"}
            )
            if candidate_left_nullspace.handle != 0:
                if (
                    bundle.left_nullspace_is_exact
                    or force_actual_schur_left
                ):
                    exact.setTransposeNullSpace(candidate_left_nullspace)
                    bundle.exact_left_nullspace_attached = True
                if (
                    bundle.auxiliary_left_nullspace_is_exact
                    or force_actual_schur_left
                ):
                    auxiliary.setTransposeNullSpace(candidate_left_nullspace)
                    bundle.auxiliary_left_nullspace_attached = True

            if (
                near_nullspace.handle != 0
                and auxiliary.getNearNullSpace().handle == 0
            ):
                auxiliary.setNearNullSpace(near_nullspace)

        self._set_legacy_side_aliases()

    def _test_nullspace(
        self,
        operator: PETSc.Mat,
        nullspace: PETSc.NullSpace,
        threshold: float,
        *,
        transpose: bool = False,
    ) -> tuple[bool, float]:
        """Return verification status and maximum null-vector residual."""
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
        maximum_residual = 0.0
        try:
            for candidate in candidates:
                if transpose:
                    operator.multTranspose(candidate, residual)
                else:
                    operator.mult(candidate, residual)
                maximum_residual = max(maximum_residual, residual.norm())
        finally:
            residual.destroy()
            if constant is not None:
                constant.destroy()
        return maximum_residual <= threshold, maximum_residual

    def update(self, pc: PETSc.PC) -> None:
        """Update state-dependent weights and auxiliary operators."""
        self.update_count += 1
        A, _ = pc.getOperators()
        self._set_blocks(A)
        for bundle in self._unique_sides():
            bundle.update(
                self.gradient,
                self.divergence,
                self.mass_lumping,
            )
        self._set_pressure_nullspaces(A)
        for bundle in self._unique_sides():
            bundle.set_ksp_operators()
        self._set_legacy_side_aliases()

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Apply the density-aware weighted BFBT Schur inverse."""
        with _BFBT_APPLY_EVENT:
            right_side = self.sides["right"]
            left_side = self.sides["left"]
            pressure_0, pressure_1, velocity_0, velocity_1 = self.workspace
            iterations = []
            reasons = []
            tolerances = []
            self._solve_inner(
                x,
                pressure_0,
                "right",
                iterations,
                reasons,
                tolerances,
            )
            with _BFBT_MIDDLE_EVENT:
                self.gradient.mult(pressure_0, velocity_0)
                velocity_0.pointwiseMult(
                    right_side.inverse_velocity_mass,
                    velocity_0,
                )
                self.velocity.mult(velocity_0, velocity_1)
                velocity_1.pointwiseMult(
                    left_side.inverse_velocity_mass,
                    velocity_1,
                )
                self.divergence.mult(velocity_1, pressure_1)
            self._solve_inner(
                pressure_1,
                y,
                "left",
                iterations,
                reasons,
                tolerances,
            )
            self.last_inner_iterations = tuple(iterations)
            self.last_inner_reasons = tuple(reasons)
            self.last_inner_tolerances = tuple(tolerances)

    def _solve_inner(
        self,
        rhs: PETSc.Vec,
        solution: PETSc.Vec,
        side: str,
        iterations: list[int],
        reasons: list[int],
        tolerances: list[float],
        *,
        transpose: bool = False,
    ) -> None:
        """Apply one weighted pressure inverse and record its outcome."""
        bundle = self.sides[side]
        ksp = bundle.ksp
        tolerance = (
            self.right_inner_rtol if side == "right" else self.left_inner_rtol
        )
        ksp.setTolerances(rtol=tolerance)
        tolerances.append(tolerance)
        event = (
            _BFBT_RIGHT_SOLVE_EVENT
            if side == "right"
            else _BFBT_LEFT_SOLVE_EVENT
        )
        with event:
            if transpose:
                ksp.solveTranspose(rhs, solution)
            else:
                ksp.solve(rhs, solution)
        reason = ksp.getConvergedReason()
        iteration_count = ksp.getIterationNumber()
        self.inner_solves_total += 1
        self.inner_iterations_total += iteration_count
        self.inner_solves_by_side[side] += 1
        self.inner_iterations_by_side[side] += iteration_count
        iterations.append(iteration_count)
        reasons.append(reason)
        if reason <= 0:
            self.inner_failures_total += 1
            self.inner_failures_by_side[side] += 1
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
            f"Rightmost inner relative tolerance: {self.right_inner_rtol}\n"
            f"Leftmost inner relative tolerance: {self.left_inner_rtol}\n"
            f"Pressure nullspace policy: {self.nullspace_policy}\n"
            f"Right null mode verified exact: {self.right_nullspace_is_exact}\n"
            f"Left null mode verified exact: {self.left_nullspace_is_exact}\n"
            f"Left null mode source: {self.left_nullspace_source}\n"
            "Nonzero inner initial guess overridden: "
            f"{self.inner_initial_guess_was_overridden}\n"
            f"Right null residual: {self.right_nullspace_residual}\n"
            f"Left null residual: {self.left_nullspace_residual}\n"
            "Auxiliary pressure-operator scale proxy: "
            f"{self.nullspace_test_auxiliary_operator_scale}\n"
            f"Nullspace acceptance threshold: {self.nullspace_test_threshold}\n"
            "Right weighted pressure-Laplacian KSP:\n"
        )
        self.sides["right"].ksp.view(viewer)

    def destroy(self, pc: PETSc.PC) -> None:
        """Destroy PETSc objects owned by this preconditioner."""
        if hasattr(self, "workspace"):
            for vector in self.workspace:
                vector.destroy()
        if hasattr(self, "sides"):
            for bundle in self._unique_sides():
                bundle.destroy()
