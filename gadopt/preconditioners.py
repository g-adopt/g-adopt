r"""This module contains classes that augment default Firedrake preconditioners.

Includes preconditioners for:
- Stokes: FreeSurfaceMassInvPC, SPDAssembledPC
- Richards: VerticallyLumpedPC, VerticallyLumpedHMGPC

"""

import re

import firedrake as fd
from firedrake import (
    FiniteElement,
    Function,
    FunctionSpace,
    PCBase,
    TensorProductElement,
    TrialFunction,
)
from firedrake.assemble import assemble
from firedrake.dmhooks import get_function_space
from firedrake.interpolation import interpolate
from firedrake.mg.utils import get_level
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


# =========================================================================
# Richards equation preconditioners
# =========================================================================


class _AutoRichardsonMixin:
    """Derive the Richardson damping factor from the measured spectrum.

    A Chebyshev smoother makes PETSc estimate the extreme eigenvalues of
    ``B^-1 A`` on every Newton step, because Firedrake reassembles the
    Jacobian in place and the operator state changes each time. Each estimate
    costs ten preconditioned GMRES iterations and ten global reductions, and
    on an extruded basin problem that is the largest single item in the setup
    budget.

    A Richardson smoother needs one number instead, the damping factor
    ``omega``. This mixin measures it the same way PETSc does, with a short
    GMRES run over the level smoother's own preconditioner, but it does the
    measurement rarely instead of every Newton step.

    Set ``vlumping_omega_auto`` to enable it. The fine-level smoother then
    becomes Richardson, whatever the preset asked for, and ``omega`` follows
    the measured spectrum.

    The damping factor
    -----------------

    Two quantities bound it. The first is the classical optimum of a
    stationary iteration over the measured interval, ``2/(lmin + lmax)``. The
    second is the stability bound: Richardson contracts while
    ``|1 - omega*lambda| < 1``, which for one eigenvalue means
    ``omega < 2 Re(lambda)/|lambda|^2``. The damping factor is the smaller of
    the two, times ``vlumping_omega_margin`` for headroom, times
    ``vlumping_omega_safety``.

    Which of the two binds depends on the shape of the spectrum, and the
    difference is not academic. On a clustered spectrum, which is what a
    serial run on a near-isotropic mesh produces, the classical optimum
    binds. On a spread spectrum, which is what a real decomposition produces
    once block-Jacobi ILU drops the coupling at the rank boundaries, the
    stability bound binds and the rule reduces to ``margin * 2/lmax``. Every
    measurement in the 832-rank basin runs took the second branch.

    The margin matters because the bound is not conservative. At the default
    of 0.9 the damping factor is ``1.8/lmax`` and the spectrum can grow by
    11% before the smoother amplifies its top mode. PETSc's Chebyshev carries
    a comparable margin in its transform, ``emax = 1.1 lmax``, and re-measures
    on every Newton step, so it recovers from drift that this scheme has to
    detect first.

    When to measure again
    ---------------------

    Three conditions, checked once per Newton step. All three are cheap,
    because the check itself runs far more often than the measurement.

    The first is a change in the diagonal of the operator. The Jacobian is
    about ``(C(h)/dt) M + K(h)``, so the mass term contributes to the
    diagonal in inverse proportion to the timestep. An adaptive timestep that
    ramps by a factor of 720 therefore moves the diagonal, and with it the
    balance between the two terms that sets the spectrum. This condition
    fires before the iteration count degrades, which the second cannot.

    The second is a rise in the linear iteration count. The reference is the
    median over the first few Newton steps after a measurement, not a single
    step, because the first Newton step of a timestep is systematically
    cheaper than the rest and a single sample makes the threshold trip on
    ordinary variation. Under a lag this condition is checked only on the
    steps where the snapshot refreshes: between refreshes a rise in the
    iteration count comes from the staleness of the snapshot, and
    re-measuring an operator that has not changed cannot help.

    The third is a plain interval. It exists because the first two conditions
    detect a change in one quantity each, and neither is guaranteed to notice
    a slow drift in something else. It is a backstop, so its default is long.

    Options, all with the outer preconditioner's prefix:

    ``vlumping_omega_auto``
        Boolean. Off by default.
    ``vlumping_omega_safety``
        Scale factor on the derived damping factor. Default 1.0.
    ``vlumping_omega_margin``
        Fraction of the stability bound to keep. Default 0.9.
    ``vlumping_omega_esteig_steps``
        GMRES iterations for one measurement. Default 20.
    ``vlumping_omega_diagonal_ratio``
        Re-measure when the norm of the diagonal moves by this factor in
        either direction. Default 2.0.
    ``vlumping_omega_iter_growth``
        Re-measure when the linear iteration count of one Newton step exceeds
        the reference count by this factor. Default 1.5.
    ``vlumping_omega_iter_window``
        Number of Newton steps that establish the reference. Default 4.
    ``vlumping_omega_interval``
        Re-measure unconditionally after this many Newton steps. Default 200.
        Set 0 to disable the backstop.
    """

    def _omega_initialize(self, pc):
        """Read the options and take the first measurement."""
        opts = PETSc.Options()
        prefix = pc.getOptionsPrefix() or ""
        self.omega_auto = opts.getBool(prefix + "vlumping_omega_auto", False)
        self.omega = None
        self._omega_applies = 0
        self._omega_applies_seen = 0
        self._omega_iter_samples = []
        self._omega_iter_reference = None
        self._omega_diagonal = None
        self._omega_estimates = 0
        self._omega_since_estimate = 0
        if not self.omega_auto:
            return
        self.omega_safety = opts.getReal(prefix + "vlumping_omega_safety", 1.0)
        self.omega_margin = opts.getReal(prefix + "vlumping_omega_margin", 0.9)
        self.omega_esteig_steps = opts.getInt(
            prefix + "vlumping_omega_esteig_steps", 20
        )
        self.omega_diagonal_ratio = opts.getReal(
            prefix + "vlumping_omega_diagonal_ratio", 2.0
        )
        self.omega_iter_growth = opts.getReal(
            prefix + "vlumping_omega_iter_growth", 1.5
        )
        self.omega_iter_window = opts.getInt(
            prefix + "vlumping_omega_iter_window", 4
        )
        self.omega_interval = opts.getInt(prefix + "vlumping_omega_interval", 200)
        self._omega_estimate(pc, "startup")
        if self.omega is None:
            raise RuntimeError(
                "vlumping_omega_auto is set, but the first measurement of the "
                "spectrum gave no usable damping factor. The smoother would "
                "silently keep the preset's own type. Check that the level "
                "preconditioner is non-singular, or unset the option."
            )

    def _omega_smoother(self):
        """The fine-level smoother of the two-level cycle."""
        return self.pc.getMGSmoother(1)

    @staticmethod
    def _omega_diagonal_norm(A):
        """Norm of the diagonal, which the mass term scales as 1/dt."""
        diag = A.createVecLeft()
        A.getDiagonal(diag)
        value = diag.norm()
        diag.destroy()
        return value

    def _omega_estimate(self, pc, reason):
        """Measure the spectrum once and set the damping factor from it."""
        smoother = self._omega_smoother()
        A, _ = smoother.getOperators()

        # Mirror KSPSetUp_Chebyshev: a short GMRES run over the smoother's own
        # preconditioner, with a random right-hand side. The estimator borrows
        # the preconditioner and never owns it, so it must not set operators
        # of its own. A KSP reads its operators from its PC.
        est = PETSc.KSP().create(comm=pc.comm)
        est.setType(PETSc.KSP.Type.GMRES)
        est.setComputeEigenvalues(True)
        est.setTolerances(rtol=1e-12, max_it=self.omega_esteig_steps)
        est.setPC(smoother.getPC())
        b = A.createVecLeft()
        x = A.createVecLeft()
        b.setRandom()
        est.solve(b, x)
        eigenvalues = est.computeEigenvalues()
        est.destroy()
        b.destroy()
        x.destroy()

        self._omega_estimates += 1
        self._omega_since_estimate = 0
        self._omega_diagonal = self._omega_diagonal_norm(A)
        self._omega_iter_samples = []
        self._omega_iter_reference = None

        if eigenvalues.size == 0:
            PETSc.Sys.Print(
                "VLumping omega: the estimator returned no eigenvalues, "
                f"keeping omega = {self.omega}"
            )
            return

        real = eigenvalues.real
        lmax = float(real.max())
        if lmax <= 0.0:
            PETSc.Sys.Print(
                "VLumping omega: the measured spectrum has no positive real "
                f"part, keeping omega = {self.omega}"
            )
            return

        # Classical optimum over the measured interval. A short Arnoldi run
        # resolves the top of the spectrum well and the bottom badly, so fall
        # back to PETSc's own assumption, lmin = 0.1 lmax, when the measured
        # lower end is not positive.
        lmin = float(real.min())
        if lmin <= 0.0:
            lmin = 0.1 * lmax
        optimum = 2.0 / (lmin + lmax)

        # Stability bound. Richardson contracts inside a disk, not an
        # interval, which matters because the gravity-advection term makes
        # the Jacobian non-symmetric and the spectrum complex.
        bounds = [
            2.0 * e.real / abs(e) ** 2
            for e in eigenvalues
            if e.real > 0.0 and abs(e) > 0.0
        ]
        limit = self.omega_margin * min(bounds) if bounds else optimum
        omega = self.omega_safety * min(optimum, limit)
        binding = "stability" if limit < optimum else "optimum"

        self.omega = omega
        self._omega_apply(smoother, omega)

        PETSc.Sys.Print(
            f"VLumping omega: measurement {self._omega_estimates} ({reason}), "
            f"spectrum = [{lmin:.4f}, {lmax:.4f}], omega = {omega:.4f} "
            f"({binding} bound), diagonal = {self._omega_diagonal:.6e}"
        )

    @staticmethod
    def _omega_apply(smoother, omega):
        """Make the smoother a Richardson iteration with this damping factor.

        The options database is the only route, because petsc4py has no
        binding for KSPRichardsonSetScale, and the type must go through it as
        well, because the preset holds its own entry for the smoother's
        ksp_type that setFromOptions would otherwise restore.

        The database is global and Firedrake reuses an options prefix across
        solver instances, so every key this method writes is removed again
        before it returns. KSPSetFromOptions has already copied the values
        into the KSP by then.
        """
        options = PETSc.Options()
        prefix = smoother.getOptionsPrefix() or ""

        # PCMG gives a level its own prefix, "mg_levels_1_", while a preset
        # usually writes the option against the generic "mg_levels_" prefix.
        # Both must be hidden. Use getAll() rather than hasName(), because
        # hasName() follows PETSc's alias and answers yes for a
        # level-specific key that does not exist; restoring such a key would
        # create it.
        candidates = {prefix, re.sub(r"\d+_$", "", prefix)}
        present = options.getAll()
        saved = {}
        for candidate in candidates:
            for suffix in ("pc_python_type", "ksp_type"):
                key = candidate + suffix
                if key in present:
                    saved[key] = present[key]
                    options.delValue(key)

        written = (prefix + "ksp_type", prefix + "ksp_richardson_scale")
        options[written[0]] = "richardson"
        options[written[1]] = omega
        try:
            smoother.setFromOptions()
        finally:
            for key in written:
                options.delValue(key)
            for key, value in saved.items():
                options[key] = value

    def _omega_update(self, pc):
        """Re-measure when any of the three conditions is met."""
        if not self.omega_auto:
            return

        iterations = self._omega_applies - self._omega_applies_seen
        self._omega_applies_seen = self._omega_applies
        self._omega_since_estimate += 1

        # Under a lag the smoother reads a snapshot that changes only on a
        # refresh step. Between refreshes the operator is identical, so a
        # measurement would return what it returned last time. Check only on
        # the steps where the snapshot moves.
        lagged = getattr(self, "_Alag", None) is not None
        refreshed = (not lagged) or self._nupdate % self.lag == 0

        reason = None
        if refreshed:
            diagonal = self._omega_diagonal_norm(
                self._omega_smoother().getOperators()[0]
            )
            if self._omega_diagonal:
                ratio = diagonal / self._omega_diagonal
                if (ratio > self.omega_diagonal_ratio
                        or ratio < 1.0 / self.omega_diagonal_ratio):
                    reason = f"the diagonal moved by {ratio:.2f}"

            if reason is None and iterations > 0:
                if len(self._omega_iter_samples) < self.omega_iter_window:
                    # Establish the reference over several Newton steps. The
                    # first step of a timestep is systematically cheaper than
                    # the rest, so one sample makes the threshold trip on
                    # ordinary variation.
                    self._omega_iter_samples.append(iterations)
                    ordered = sorted(self._omega_iter_samples)
                    self._omega_iter_reference = ordered[len(ordered) // 2]
                elif iterations > self.omega_iter_growth * self._omega_iter_reference:
                    reason = (
                        f"linear iterations rose from a reference of "
                        f"{self._omega_iter_reference} to {iterations}"
                    )

        if (reason is None and self.omega_interval > 0
                and self._omega_since_estimate >= self.omega_interval):
            reason = f"{self._omega_since_estimate} Newton steps since the last measurement"

        if reason is not None:
            self._omega_estimate(pc, reason)

    def _omega_count_apply(self):
        self._omega_applies += 1


class _LaggedOperatorMixin:
    """Hold a private snapshot of the operator for the inner multigrid.

    Firedrake reassembles the Jacobian in place on every Newton step, so the
    PETSc object state of the operator changes every time. The inner PCMG
    reacts to that change with a full setup: the Galerkin product P^T A P, the
    coarse factorisation, and the eigenvalue estimate of a Chebyshev smoother.
    For an extruded 3-D problem that setup cost dominates the solve.

    This mixin gives the inner PCMG a private copy of the operator and
    refreshes the copy every ``lag`` Newton steps. Between two refreshes the
    object state does not change, so PETSc skips the complete setup. The outer
    Krylov method keeps the true Jacobian for its residual, therefore the
    accuracy of the Newton step does not change. Only the preconditioner is
    stale.

    The snapshot is necessary. A lag that leaves the live Jacobian visible to
    the smoother makes the smoother evaluate B_old^-1 A_live. A polynomial or
    stationary smoother contracts only on a bounded region, and modes that
    leave that region during a sharp wetting front make the multigrid cycle
    expansive. With the snapshot the smoother evaluates B_old^-1 A_old, which
    is inside the region by construction, and the drift reaches the outer
    Krylov method as a benign near-identity perturbation.

    Set the lag with the integer option ``<prefix>vlumping_lag``. The default
    is 1, which refreshes on every Newton step and reproduces the unlagged
    behaviour exactly. The snapshot costs one extra copy of the Jacobian.

    The boolean option ``<prefix>vlumping_lag_smoother`` decides whether the
    fine smoother reads the snapshot as well. The default is on, which is the
    coherent setting described above. Turn it off and the smoother keeps the
    live Jacobian, so the multigrid cycle approximates the inverse of the
    current operator with a stale smoother. On the cases measured so far that
    is faster, about 17% against 9%, and it costs no extra iterations. It is
    not the default because it evaluates ``B_old^-1 A_live``, which has no
    guarantee of contraction once a front moves the spectrum.
    """

    def _lag_initialize(self, pc, P):
        """Read the lag option and bind the inner PC to the snapshot."""
        prefix = pc.getOptionsPrefix() or ""
        self.lag = PETSc.Options().getInt(prefix + "vlumping_lag", 1)
        self.lag_smoother = PETSc.Options().getBool(
            prefix + "vlumping_lag_smoother", True
        )
        self._nupdate = 0
        if self.lag <= 1:
            self._Alag = None
            return

        # Copy the values now, because the first solve uses this matrix.
        self._Alag = P.duplicate(copy=True)
        # Both arguments must be the snapshot. KSPSetUp_Chebyshev compares the
        # state of Amat and of Pmat, so a live Amat re-triggers the estimate
        # and defeats the lag.
        self.pc.setOperators(self._Alag, self._Alag)

        # The fine smoother needs the snapshot as well, and it needs it
        # explicitly. PCSetUp_MG binds the operators of the finest smoother
        # only while they still match the operators of the PC
        # (src/ksp/pc/impls/mg/mg.c:888-892). The first setup bound them to
        # the live Jacobian, so the guard is now false and the smoother would
        # keep the live operator for the whole run. It would then evaluate
        # B_old^-1 A_live, which is the combination that loses the guarantee
        # of contraction during a sharp front.
        #
        # The multigrid residual callback cannot be corrected the same way.
        # PCMGSetResidual takes a permanent reference on the first setup
        # (mgfunc.c:151-156) and petsc4py has no binding for it, so the
        # mid-cycle residual keeps the live operator.
        if self.lag_smoother:
            self.pc.getMGSmoother(1).setOperators(self._Alag, self._Alag)

    def _lag_update(self, pc):
        """Refresh the snapshot when the lag interval is complete."""
        if self._Alag is None:
            return
        self._nupdate += 1
        if self._nupdate % self.lag == 0:
            _, P = pc.getOperators()
            # MatCopy increases the object state, so the inner PCMG runs a
            # complete setup on this step and skips it on the others.
            P.copy(self._Alag, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)

    def _lag_destroy(self):
        if getattr(self, "_Alag", None) is not None:
            self._Alag.destroy()
            self._Alag = None


class VerticallyLumpedPC(_AutoRichardsonMixin, _LaggedOperatorMixin, PCBase):
    """Two-level MG that collapses the vertical dimension on the coarse level.

    This preconditioner targets extruded meshes of high aspect ratio. It
    follows Kramer et al. (2010, doi:10.1016/j.ocemod.2010.08.001) and the
    ocean modelling community. It uses two levels:

        Fine level   : V = V_horiz x V_vert  (full 3D tensor-product space)
        Coarse level : V_lumped = V_horiz x R  (vertically constant)

    The prolongation P is the natural injection from V_lumped into V, which
    replicates a coarse DOF along its column. Galerkin projection then forms
    the coarse operator A_c = P^T A P. That projection collapses the whole
    vertical dimension onto a 2D-like problem, which MUMPS or GAMG solves
    cheaply.

    Solver options for the coarse level use the prefix ``lumped_mg_coarse_``.
    The fine-level smoother uses ``lumped_mg_levels_``.

    The integer option ``vlumping_lag`` rebuilds the preconditioner every N
    Newton steps against a private snapshot of the Jacobian. The default is 1,
    which rebuilds on every step. See ``_LaggedOperatorMixin``.

    The fine space must be a scalar tensor-product space on an extruded mesh.
    """

    def initialize(self, pc):
        options_prefix = pc.getOptionsPrefix()
        A, P = pc.getOperators()
        V = get_function_space(pc.getDM())

        # A mixed space has no factor_elements, so the lumped space below cannot
        # be built for one. Reject it here with a clear message.
        if len(V) != 1:
            raise RuntimeError(
                "VerticallyLumpedPC needs a scalar function space. It received "
                f"a mixed space with {len(V)} components. Apply the "
                "preconditioner to a single field, through a fieldsplit."
            )

        mesh = V.mesh()

        # An extruded mesh has a tensor-product cell, so its ufl_cell has
        # sub_cells and its element has factor_elements. A non-extruded mesh has
        # neither, and both lookups raise AttributeError.
        try:
            _, vcell = mesh.ufl_cell().sub_cells
            hele, _ = V.ufl_element().factor_elements
        except AttributeError as exc:
            raise RuntimeError(
                "VerticallyLumpedPC needs an extruded mesh and a "
                "tensor-product element. It received a space on "
                f"{mesh.ufl_cell()}. Build the mesh with ExtrudedMesh, or use "
                "a different preconditioner."
            ) from exc

        # Vertically constant space V_lumped = V_horiz x R.
        vele = FiniteElement("R", vcell, 0)
        ele = TensorProductElement(hele, vele)
        V_lumped = FunctionSpace(mesh, ele)

        # Prolongation: interpolation from V_lumped to V.
        trial = TrialFunction(V_lumped)
        Prol = assemble(interpolate(trial, V)).petscmat

        # Configure the 2-level multigrid.
        self.pc = PETSc.PC().create(comm=pc.comm)
        self.pc.setOptionsPrefix(options_prefix + "lumped_")
        self.pc.incrementTabLevel(1, parent=pc)
        # Propagate the outer DM so that level smoothers, for example
        # ASMLinesmoothPC, can resolve the fine function space.
        self.pc.setDM(pc.getDM())

        # Enable the Galerkin coarse operator A_c = P^T A P. petsc4py has no
        # setMGGalerkin() binding, so we use the options database instead.
        options = PETSc.Options()
        options[options_prefix + "lumped_pc_mg_galerkin"] = "both"

        self.pc.setOperators(A, P)
        self.pc.setType("mg")
        self.pc.setMGLevels(2)
        self.pc.setMGInterpolation(1, Prol)
        self.pc.setFromOptions()
        self.pc.setUp()
        # Bind the inner PCMG to a lagged snapshot when the user asks for one.
        # This runs after the first setUp, so the multigrid residual callback
        # keeps the live operator. See _LaggedOperatorMixin.
        self._lag_initialize(pc, P)
        # Derive the Richardson damping factor when the user asks for it. The
        # first measurement needs a preconditioner that is already set up.
        self._omega_initialize(pc)
        self.update(pc)

    def update(self, pc):
        # Firedrake reassembles the Jacobian in place on every Newton iteration.
        # The inner PCMG holds a reference to the same Mat. It still needs
        # setUp() to recompute the Galerkin coarse operator A_c = P^T A P from
        # the updated state.
        self._lag_update(pc)
        self._omega_update(pc)
        self.pc.setUp()

    def apply(self, pc, X, Y):
        self._omega_count_apply()
        self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        self.pc.applyTranspose(X, Y)

    def destroy(self, pc):
        self._lag_destroy()
        if hasattr(self, "pc"):
            self.pc.destroy()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.pushASCIITab()
        viewer.printfASCII(
            "Vertically lumped MG: collapses the vertical dimension on the "
            "coarse level (see Kramer et al. 2010)\n"
        )
        self.pc.view(viewer)


class VerticallyLumpedHMGPC(_AutoRichardsonMixin, _LaggedOperatorMixin, PCBase):
    """Two-level MG with the coarse level on the fine mesh's 2D base.

    This preconditioner uses the same two levels as ``VerticallyLumpedPC``.
    It builds the coarse level on the 2D base of the fine mesh, through
    ``mesh._base_mesh``, instead of on the extruded mesh with a vertical
    R-element. The coarse KSP then descends a geometric multigrid on the 2D
    base ``MeshHierarchy``, which gives optimal complexity for large
    horizontal problems.

    Note that "coarse" here means coarse relative to the 3D fine space. The
    base mesh ``V_base_2d`` sits at the *finest* level of the base hierarchy.

    Fine space : V          = hele x vele      (full 3D tensor-product)
    Coarse     : V_base_2d  = hele on base_2d  (same horizontal element)

    The integer option ``vlumping_lag`` applies here as well. See
    ``_LaggedOperatorMixin``.

    The prolongation from ``V_base_2d`` into ``V`` is the same-mesh
    interpolation from the vertically constant 3D space ``V_R = hele x R``
    into ``V``. That matrix needs no relabelling. A ``hele x Real`` space and
    the horizontal ``hele`` space on the base share one PETSc DOF layout, so
    the matrix already has the right column space. The note in ``initialize``
    gives the reason.

    The base mesh of the fine mesh must live in a ``MeshHierarchy``. If it
    does not, ``initialize`` raises ``RuntimeError``.
    """

    def initialize(self, pc):
        options_prefix = pc.getOptionsPrefix() or ""
        A, P = pc.getOperators()
        V = get_function_space(pc.getDM())

        # A mixed space has no factor_elements, so the spaces below cannot be
        # built for one. Reject it here with a clear message.
        if len(V) != 1:
            raise RuntimeError(
                "VerticallyLumpedHMGPC needs a scalar function space. It "
                f"received a mixed space with {len(V)} components. Apply the "
                "preconditioner to a single field, through a fieldsplit."
            )

        mesh = V.mesh()

        base_mesh = getattr(mesh, "_base_mesh", None)
        if base_mesh is None:
            raise RuntimeError(
                "VerticallyLumpedHMGPC needs an extruded fine mesh. This mesh "
                "has no _base_mesh. Use VerticallyLumpedPC for a non-extruded "
                "mesh."
            )

        hierarchy, level = get_level(base_mesh)
        if hierarchy is None or level is None:
            raise RuntimeError(
                "VerticallyLumpedHMGPC needs the 2D base mesh of the fine mesh "
                "to be part of a MeshHierarchy. The coarse PCMG descends that "
                "hierarchy. Build the mesh with "
                "ExtrudedMeshHierarchy(MeshHierarchy(base, L), ...) and L >= 1, "
                "or use VerticallyLumpedPC instead."
            )

        hele, _ = V.ufl_element().factor_elements
        # V_base_2d is coarse relative to the 3D fine space V. Within the base
        # hierarchy it is the finest level, at index `level`.
        V_base_2d = FunctionSpace(base_mesh, hele)
        self.V_base_2d = V_base_2d

        # Prolongation from V_base_2d (hele on the 2D base) into V (full 3D).
        #
        # We build it as the same-mesh interpolation from V_R into V, where
        # V_R = hele x Real is the vertically constant 3D space. That matrix is
        # already the prolongation we want. V_R and V_base_2d share one PETSc
        # layout: the same owned size, the same ownership range, and the same
        # lgmap including ghost order. No relabelling is therefore needed.
        #
        # This equality is structural, not coincidental. An ExtrudedMeshTopology
        # reuses the base mesh's topology_dm, _dm_renumbering and
        # _entity_classes. For a hele x Real element, the section and node-class
        # computations then collapse the vertical extent away. create_section
        # uses on_base=True, node_classes uses real_tensorproduct=True, and the
        # dof offset is zero. As a result, V_R and V_base_2d take their global
        # numbering from the same computation over the same DM.
        #
        # A base-to-extruded relabelling is not an alternative. Cross-mesh
        # interpolation needs a matching geometric dimension, and the base is 2D
        # while the extruded mesh is 3D.
        vcell = mesh.ufl_cell().sub_cells[1]
        vele_R = FiniteElement("R", vcell, 0)
        V_R = FunctionSpace(mesh, TensorProductElement(hele, vele_R))
        Prol = assemble(interpolate(TrialFunction(V_R), V)).petscmat
        self.Prol = Prol

        # Guard the layout assumption, because a later failure is worse. A pure
        # size mismatch surfaces as a cryptic PETSc MatMatMult error in the
        # Galerkin setup. A same-size reordering is worse still, because it
        # gives a silently wrong coarse operator. Raise here instead.
        with Function(V_base_2d).dat.vec_ro as cvec:
            base_owned = cvec.getLocalSize()
        prol_cols_owned = Prol.getLocalSize()[1]
        if base_owned != prol_cols_owned:
            raise RuntimeError(
                "VerticallyLumpedHMGPC: the prolongation column layout "
                f"({prol_cols_owned} owned) does not match V_base_2d "
                f"({base_owned} owned). The hele x Real space no longer "
                "collapses onto the base layout."
            )

        # Configure the 2-level MG.
        self.pc = PETSc.PC().create(comm=pc.comm)
        self.pc.setOptionsPrefix(options_prefix + "lumped_")
        self.pc.incrementTabLevel(1, parent=pc)
        self.pc.setDM(pc.getDM())

        # Galerkin coarse operator: A_c = P^T A P.
        options = PETSc.Options()
        options[options_prefix + "lumped_pc_mg_galerkin"] = "both"

        self.pc.setOperators(A, P)
        self.pc.setType("mg")
        self.pc.setMGLevels(2)
        self.pc.setMGInterpolation(1, Prol)

        # Build the chain of horizontal interpolations along the base
        # MeshHierarchy. Level 0 is the coarsest and V_base_2d is at `level`.
        base_interp_mats = self._build_base_hierarchy_interpolations(
            hierarchy, level, hele
        )
        self._base_interp_mats = base_interp_mats

        # Explicit base-mesh interpolations plus Galerkin coarsening avoid the
        # DM-based coarsen hook. That hook needs a coarsened UFL appctx, which
        # is not available here.
        if base_interp_mats:
            coarse_ksp = self.pc.getMGCoarseSolve()
            coarse_pc = coarse_ksp.getPC()
            coarse_pc.setType("mg")
            nlevels_coarse = len(base_interp_mats) + 1
            coarse_pc.setMGLevels(nlevels_coarse)
            for i, Pi in enumerate(base_interp_mats):
                coarse_pc.setMGInterpolation(i + 1, Pi)
            coarse_pc.setOptionsPrefix(
                options_prefix + "lumped_mg_coarse_"
            )
            options = PETSc.Options()
            options[options_prefix + "lumped_mg_coarse_pc_mg_galerkin"] = "both"

        self.pc.setFromOptions()
        self.pc.setUp()
        self._lag_initialize(pc, P)
        self._omega_initialize(pc)
        self.update(pc)

    @staticmethod
    def _build_base_hierarchy_interpolations(hierarchy, fine_level, hele):
        """Assemble horizontal interpolation matrices for the base MG."""
        mats = []
        V_c = FunctionSpace(hierarchy[0], hele)
        for k in range(1, fine_level + 1):
            V_f = FunctionSpace(hierarchy[k], hele)
            trial = TrialFunction(V_c)
            P_k = assemble(interpolate(trial, V_f)).petscmat
            mats.append(P_k)
            V_c = V_f  # the fine space becomes the coarse space next round
        return mats

    def update(self, pc):
        self._lag_update(pc)
        self._omega_update(pc)
        self.pc.setUp()

    def apply(self, pc, X, Y):
        self._omega_count_apply()
        self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        self.pc.applyTranspose(X, Y)

    def destroy(self, pc):
        self._lag_destroy()
        if hasattr(self, "pc"):
            self.pc.destroy()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.pushASCIITab()
        viewer.printfASCII(
            "Vertically lumped MG (HMG variant): coarse level built on "
            "the 2D base of an ExtrudedMeshHierarchy, coarse solve is "
            "geometric MG on the base hierarchy.\n"
        )
        self.pc.view(viewer)
