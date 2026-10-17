r"""Route 1.5b: the hand-written adjoint for the coupled low-rank DtN solve.

The self-gravitating GIA solver adds the low-rank DtN feedback `B` to the
coupled Jacobian through PETSc-level callbacks (`gadopt.dtn_coupled`), so the
term is invisible to pyadjoint's tape: the stock solve block sees only the
stock operator `A` and none of `B`. Its replay, adjoint and tangent are then
all wrong by the whole `theta_psi B0 psi` contribution.

## Why "adopt the block" rather than build one

The multiplier path is a `NonlinearVariationalSolver` on a plain UFL residual,
so pyadjoint tapes it and the adjoint is exact for free. The low-rank path
carries `B` at the PETSc level, so the derivative is written by hand - exactly
as `gadopt.dtn_adjoint` does for the gravity-alone solver. The difference is the
install mechanism: `dtn_adjoint` creates a fresh block; route 1.5b lets the
stock annotated solve run, then takes the stock block off the tape and
re-classes it (`block.__class__ = ...`), so that everything public in pyadjoint
keeps working and only four internals are overridden.

## The five overrides

  1. `_forward_solve`  - replay through `A + B`. The stock forward solver
     already carries the augmentation callbacks, so only `theta_psi` has to be
     bound to this block's control values before the solve.
  2. `_adjoint_solve`  - solve `(A + B)^T lambda = seed` through Firedrake's
     own adjoint variational solver for the block, with `theta_psi` bound.
     That solver is built from the forward solver's constructor keywords, so
     it carries BOTH augmentation callbacks; `B` is symmetric, so its
     transpose action is the same `apply_local`. The keywords it is built from
     are the ones `LowRankVariationalSolver.solve` supplies as `adj_kwargs`:
     the forward application context (`dtn_operator`, the nullspace
     providers, `operator_version`), the forward nullspaces swapped, and
     `snes_type: ksponly`.
  3. `_assemble_and_solve_tlm_eq` - the tangent operator is `A + B`, solved
     as a variational problem on the tangent form with the same keywords, and
     the tangent right-hand side carries the `(d theta_psi/dm) B0 psi` term.
  4 & 5. The `(d theta_psi/dm) B0 psi` term on `evaluate_adj_component` and on
     the tangent right-hand side. `theta_psi = scaling_factor B_mu / Lambda`,
     all live coefficients, so its control derivative is a scalar that
     multiplies `B0 psi` - the low-rank action at prefactor one.

## Both augmentations, and `ksponly`, on every derivative solve

The adjoint and the tangent systems are linear whatever the rheology, and the
augmentation of the Jacobian and of the residual must come together. A
derivative solve built with the Jacobian callback alone under `newtonls`
(inherited from the forward dictionary, whose direct preset names no
`snes_type`) takes one correct step to `(A + B)^-1 dJdu`, finds the residual
`-B x_1` there because the residual carries no `B`, and walks to `A^-1 dJdu`:
a gradient wrong by `||B x|| / ||A x||`, a few percent, with every solver
reporting convergence (measured: Taylor rate 1.08, gradient 1.9e-2 out, on
the direct preset). So `derivative_solver_kwargs` supplies both callbacks
and forces `ksponly`, and `test_gia_lowrank_block0.py` pins one Newton
iteration under `newtonls` as the check that the two operators agree.

Three overrides alone are self-consistent and 93.65% wrong: the adjoint and the
tangent agree with each other because both use `A + B`, and both miss the
`(d theta_psi/dm) B0 psi` column of `dF/dm`. The switch
`solver._include_theta_derivative` turns the fifth override off, which is what
the L11b instrument-sensitivity test flips.

## theta_psi is bound per solve, per block

The augmentation callbacks read `theta_psi` off the live solver, whose
coefficients hold the CURRENT control values and not the tape's saved ones. At
replay or adjoint at a perturbed control the live value is stale. Worse, two
blocks of a timestep loop share one forward and one adjoint solver, so binding
`theta_psi` once at install leaves the shared solvers carrying whichever block
installed last. So the prefactor is computed from THIS block's saved dependency
values and set on the shared operator immediately before each solve, and
restored after.

## Two Firedrakes: the stock solve block and the cached solver block

Firedrake PR 4638 ("NLVS: cache more adjoint solvers/forms and fix Hessian")
puts a `CachedSolverBlock` on the tape for every annotated variational solve.
Its forward, adjoint and tangent solvers are built once by the forward solver,
from the forward solver's constructor keywords, and shared by every block that
solver adds. The module supports both Firedrakes and finds out at import which
one it runs on (`CACHED_SOLVER_BLOCKS`):

  - stock: the adjoint keywords go to `solve` (`LowRankVariationalSolver`),
    and the stock block becomes a `CoupledLowRankDtNSolveBlock` with the five
    overrides above.
  - cached: the adjoint and tangent keywords go to the constructor
    (`derivative_constructor_kwargs`), the callbacks reach every cached solver
    through the constructor keywords, and the cached block becomes a
    `CachedCoupledLowRankDtNSolveBlock`, which binds `theta_psi` around each
    solve and adds the theta derivative.

On the cached Firedrake the constructor keywords need the fix that takes
`adj_kwargs` and `tlm_kwargs` out of the keywords before the solver is
initialised; without it the constructor raises `TypeError`.
"""

from contextlib import contextmanager

import numpy as np
from firedrake import (Cofunction, LinearVariationalProblem,
                       LinearVariationalSolver, NonlinearVariationalSolver,
                       TrialFunction, assemble, derivative)
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock
from pyadjoint.tape import annotate_tape

# Firedrake PR 4638 ("NLVS: cache more adjoint solvers/forms and fix Hessian")
# replaces the stock solve block of an annotated variational solve with a
# `CachedSolverBlock`. That block replays and differentiates through solvers
# that the forward solver builds once and shares with every block it puts on
# the tape, and it takes the keywords of the adjoint and tangent solvers from
# the constructor of the forward solver, not from `solve`. Both Firedrakes are
# supported here until the PR is merged: the module finds out once which one
# it runs on.
try:
    from firedrake.adjoint_utils.blocks import CachedSolverBlock
except ImportError:
    CachedSolverBlock = None

#: True on a Firedrake whose annotated variational solves put a
#: `CachedSolverBlock` on the tape (Firedrake PR 4638).
CACHED_SOLVER_BLOCKS = CachedSolverBlock is not None

#: The class of the block an annotated variational solve puts on the tape on
#: this Firedrake. Tests and the adopt function look for this class.
SOLVE_BLOCK_TYPE = (CachedSolverBlock if CACHED_SOLVER_BLOCKS
                    else NonlinearVariationalSolveBlock)

__all__ = ["CACHED_SOLVER_BLOCKS", "CachedCoupledLowRankDtNSolveBlock",
           "CoupledLowRankDtNSolveBlock",
           "LowRankVariationalSolver", "SOLVE_BLOCK_TYPE",
           "adopt_coupled_lowrank_block", "derivative_constructor_kwargs",
           "derivative_solver_kwargs", "require_controls_reach_block"]


def derivative_solver_kwargs(gia_solver, suffix: str, *, transpose: bool) -> dict:
    """Constructor keywords for one LINEAR derivative solve of the low-rank solver.

    The forward dictionary with `snes_type` forced to `ksponly`, the forward
    application context, an options prefix of its own, both augmentation
    callbacks and the nullspaces. For the adjoint (`transpose=True`) the
    nullspace and the transpose nullspace change places, because the adjoint
    operator is the transpose of the forward one; the near-nullspace is the
    same set of modes on either side.

    Args:
      gia_solver: the `SelfGravitatingGIASolver` whose callbacks and context
        the solve carries.
      suffix: appended to the solver name as the options prefix, so a
        command-line option can reach one derivative solve and not the other.
      transpose: `True` for the adjoint system, `False` for the tangent.

    Returns:
      Keyword arguments for `LinearVariationalSolver`.
    """
    parameters = dict(gia_solver.solver_parameters)
    # A linear system: one Krylov solve from a zero guess is the answer, and
    # `newtonls` would spend one more residual evaluation to confirm it.
    parameters["snes_type"] = "ksponly"
    nullspace, transpose_nullspace = gia_solver.nullspace, gia_solver.transpose_nullspace
    if transpose:
        nullspace, transpose_nullspace = transpose_nullspace, nullspace
    return dict(
        solver_parameters=parameters,
        appctx=gia_solver.appctx,
        options_prefix=gia_solver.name + suffix,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        near_nullspace=gia_solver.near_nullspace,
        post_function_callback=gia_solver.augment_residual,
        post_jacobian_callback=gia_solver.augment_jacobian,
    )


class LowRankVariationalSolver(NonlinearVariationalSolver):
    """The forward solver of the low-rank path, naming its own adjoint solver.

    Firedrake's tape builds one adjoint `LinearVariationalSolver` per forward
    solver, from the keywords the annotated `solve` receives as `adj_kwargs`
    (`firedrake/adjoint_utils/variational_solver.py`,
    `adjoint_utils/blocks/solving.py: solve_init_params`). Without them it
    copies the forward keywords and drops the application context, so the
    block-0 preconditioner of the iterative preset would run the adjoint with
    no `dtn_operator`, no nullspace providers and no reuse marker. This
    subclass supplies `adj_kwargs` on every annotated solve. The keywords are
    read at solve time and not stored at construction, because the tape clones
    the forward solver from its constructor keywords for the replay
    (`type(self)(problem, **self._ad_kwargs)`), and the clone must find them
    the same way.
    """

    #: Set by `SelfGravitatingGIASolver.set_solver`; the owner of the
    #: callbacks, the context and the nullspaces.
    gia_solver = None

    def solve(self, **kwargs):
        # Only an annotated solve takes `adj_kwargs`: Firedrake's wrapper pops
        # the tape keywords when it is recording and passes everything else
        # to the plain solve, which refuses them. `annotate_tape` on a COPY,
        # because it removes the `annotate` key from the mapping it is given
        # and the wrapper needs to see that key itself. With cached solver
        # blocks the keywords go to the constructor instead
        # (`derivative_constructor_kwargs`), and `solve` refuses them.
        if (not CACHED_SOLVER_BLOCKS and self.gia_solver is not None
                and annotate_tape(dict(kwargs))):
            kwargs.setdefault("adj_kwargs", derivative_solver_kwargs(
                self.gia_solver, "_lowrank_adj", transpose=True))
        return super().solve(**kwargs)


def derivative_constructor_kwargs(gia_solver) -> dict:
    """The adjoint and tangent keywords for the forward solver's constructor.

    With cached solver blocks (Firedrake PR 4638) the forward solver builds
    its adjoint and tangent linear solvers from `adj_kwargs` and `tlm_kwargs`
    given to its constructor. Without them both would inherit the forward
    keywords: the forward `snes_type` (Newton line search on a linear system,
    one residual evaluation more per solve) and the forward nullspaces in the
    forward slots. With the stock solve block the adjoint keywords go to
    `solve` (`LowRankVariationalSolver.solve`) and the tangent solver is built
    by `CoupledLowRankDtNSolveBlock`, so this returns nothing.

    The keywords are read once, here. `derivative_solver_kwargs` copies the
    solver parameters, so a later change of the forward dictionary does not
    reach the derivative solvers. The application context is the forward one;
    Firedrake clones its UFL entries for the cached solvers.

    Args:
      gia_solver: the `SelfGravitatingGIASolver` whose callbacks, context and
        nullspaces the derivative solves carry. Its solver parameters,
        nullspaces and context must be set before this is called.

    Returns:
      `{"adj_kwargs": ..., "tlm_kwargs": ...}` with cached solver blocks, else
      an empty `dict`.
    """
    if not CACHED_SOLVER_BLOCKS:
        return {}
    return {
        "adj_kwargs": derivative_solver_kwargs(
            gia_solver, "_lowrank_adj", transpose=True),
        "tlm_kwargs": derivative_solver_kwargs(
            gia_solver, "_lowrank_tlm", transpose=False),
    }


def require_controls_reach_block(block, controls):
    """D1: raise if any control is absent from the block's dependencies.

    A control that never enters the residual is not a dependency, so the stock
    gradient is a silent `0.0` with only a `WARNING:root:Adjoint value is None`
    on stderr - invisible in a batch log (REVIEW-ADJOINT S6.5 D1). This turns
    that into an exception.

    The solver cannot discover its controls on its own: pyadjoint does not mark
    a `Control` on the tape, so the caller that knows the controls passes them.
    """
    deps = {id(dep.output) for dep in block.get_dependencies()}
    for control in controls:
        if id(control) not in deps:
            raise ValueError(
                "The control is not among the low-rank solve block's "
                "dependencies, so it never reaches the residual. The stock "
                "gradient would be a silent 0.0 with only a "
                "'WARNING:root:Adjoint value is None' on stderr "
                "(REVIEW-ADJOINT S6.5 D1). Raised here instead.")


def _scalar(value) -> float:
    """`float()` of one scalar carrier, `Constant` or `Real` `Function` alike.

    The twin of `gadopt.gia_gravity.scalar_value`, kept local so this module
    does not import `gia_gravity` and close a cycle.
    """
    try:
        return float(value)
    except (TypeError, NotImplementedError):
        return float(np.asarray(value.dat.data_ro, dtype=float).reshape(-1)[0])


class _ThetaPsiTerms:
    """What both coupled low-rank blocks add to a stock solve block.

    The low-rank feedback enters the residual and the Jacobian as
    `theta_psi B0 psi` through PETSc callbacks, with
    `theta_psi = scaling_factor B_mu / Lambda`. The callbacks read the
    prefactor off the shared `dtn_operator`, which holds the LIVE value. Each
    block binds the prefactor to its own saved values around its solves, and
    adds the `(d theta_psi / dm) B0 psi` term, which no form carries, to the
    adjoint gradient and to the tangent right-hand side. Around the same solves
    it also sets `gia_solve_index` to the index of its forward solve, so that
    the rebuild rule of the preconditioners sees the steps move during replay
    and adjoint (`_bound_to_block`).

    Mixed into `CoupledLowRankDtNSolveBlock` (stock Firedrake) and
    `CachedCoupledLowRankDtNSolveBlock` (Firedrake PR 4638). Neither is ever
    constructed: `adopt_coupled_lowrank_block` changes the `__class__` of the
    block the annotated solve produced.
    """

    #: The owning `SelfGravitatingGIASolver`, set by the adopt function. Carries
    #: the shared `dtn_operator`, the solver parameters and the
    #: `_include_theta_derivative` switch.
    gia_solver = None
    #: The `gia_solve_index` of the forward solve that put this block on the
    #: tape, set by the adopt function. `_bound_to_block` writes it into the
    #: application contexts around each replay, adjoint and tangent solve.
    gia_solve_index = None

    # -- theta_psi, bound from THIS block's saved control values -------------
    def _saved_scalar(self, obj) -> float:
        """`obj`'s value at the tape point: its saved output if it is a
        dependency, otherwise its live value (a control-independent constant)."""
        for dep in self.get_dependencies():
            if dep.output is obj:
                return _scalar(dep.saved_output)
        return _scalar(obj)

    def _theta_objects(self):
        s = self.gia_solver
        return float(s.scaling_factor), s.approximation.B_mu, s.Lambda

    def _bound_theta(self) -> float:
        """`scaling_factor * B_mu / Lambda` at this block's saved values."""
        sf, b_mu, lam = self._theta_objects()
        b = self._saved_scalar(b_mu)
        if b == 0.0:
            b = 1.0  # NULL_COUPLING_ROW_SCALE: a deleted row, not a decoupled one
        return sf * b / self._saved_scalar(lam)

    def _dtheta(self, control) -> float:
        """`d theta_psi / d control`, zero unless the control is B_mu or Lambda."""
        sf, b_mu, lam = self._theta_objects()
        if control is b_mu:
            return sf / self._saved_scalar(lam)
        if control is lam:
            b = self._saved_scalar(b_mu)
            if b == 0.0:
                b = 1.0
            lam_v = self._saved_scalar(lam)
            return -sf * b / (lam_v * lam_v)
        return 0.0

    # -- the solve index the preconditioners read ----------------------------
    def _solve_index_contexts(self):
        """The application contexts the preconditioners of this block read.

        On stock Firedrake the replay solver is built from the forward
        solver's keywords and the derivative solvers from
        `derivative_solver_kwargs`, so all of them carry the forward
        solver's own `appctx` dict. `CachedCoupledLowRankDtNSolveBlock`
        overrides this with the cloned dicts of the cached solvers.

        Returns:
          A list of the `appctx` dicts to write `gia_solve_index` into.
        """
        return [self.gia_solver.appctx]

    @contextmanager
    def _bound_to_block(self):
        """Bind what the shared solvers read to this block for one solve.

        Two values are bound, and both restored on exit:

        - `theta_psi` on the shared `dtn_operator`, from this block's saved
          control values. The augmentation callbacks read it at every
          application.
        - `gia_solve_index` in the application context, set to the index of
          the forward solve that put this block on the tape. The rebuild rule
          of the preconditioners (`gadopt.preconditioners._RebuildRule`)
          rebuilds when this index moves. Nothing else moves it during a
          replay or an adjoint sweep, because those solves do not go through
          `SelfGravitatingGIASolver.solve`. Without this binding the replay
          and the adjoint keep the preconditioner of their first solve for
          every later step of every sweep. With it, a replay rebuilds at the
          same steps as the forward run, and the adjoint sweep rebuilds as it
          walks back through the steps.

        Yields:
          The shared `dtn_operator`.
        """
        op = self.gia_solver.dtn_operator
        saved = op.theta
        op.theta = self._bound_theta()
        # A block adopted without an index (none recorded) leaves the
        # contexts alone. Each context is restored to its own previous value,
        # so the forward solver's count is unchanged after a replay on stock
        # Firedrake, where the replay shares the forward dict.
        contexts = ([] if self.gia_solve_index is None
                    else self._solve_index_contexts())
        missing = object()
        saved_index = [ctx.get("gia_solve_index", missing) for ctx in contexts]
        for ctx in contexts:
            ctx["gia_solve_index"] = self.gia_solve_index
        try:
            yield op
        finally:
            op.theta = saved
            for ctx, value in zip(contexts, saved_index):
                if value is missing:
                    ctx.pop("gia_solve_index", None)
                else:
                    ctx["gia_solve_index"] = value

    # -- the low-rank action, as vectors ------------------------------------
    def _B0_psi(self) -> Cofunction:
        """`B0 psi` (prefactor one) at the forward solution, as a cofunction."""
        output = self.get_outputs()[0]
        psi = output.saved_output
        out = Cofunction(output.output.function_space().dual())
        op = self.gia_solver.dtn_operator
        with psi.dat.vec_ro as xv, out.dat.vec as yv:
            yv.set(0.0)
            op.apply_local(xv.array_r, yv.array_w, theta=1.0)
        return out

    def _lambda_dot_B0_psi(self, adj_sol) -> float:
        """`lambda . (B0 psi)`, the scalar the theta derivative multiplies."""
        B0_psi = self._B0_psi()
        with adj_sol.dat.vec_ro as lv, B0_psi.dat.vec_ro as bv:
            return float(lv.dot(bv))

    # -- the theta derivative ------------------------------------------------
    def _add_theta_adjoint_term(self, out, block_variable, prepared, adj_input):
        """Add `-(d theta_psi / dm) lambda . (B0 psi)` to the gradient `out`.

        The stock gradient is `-lambda^T dF/dm` from the forms, which do not
        carry `B`. The missing part is a scalar times `d theta_psi / dm`, so it
        is nonzero only for the controls `B_mu` and `Lambda`, which are `Real`
        functions or constants: the term is added to every entry of `out`.
        """
        if not getattr(self.gia_solver, "_include_theta_derivative", True):
            return out
        if prepared is None or adj_input is None:
            return out
        control = block_variable.output
        dth = self._dtheta(control)
        if dth == 0.0:
            return out
        extra = -dth * self._lambda_dot_B0_psi(prepared["adj_sol"])
        if out is None:
            out = Cofunction(control.function_space().dual())
        out.dat.data_wo[:] = np.asarray(out.dat.data_ro) + extra
        return out

    def _theta_tangent_coefficient(self) -> float:
        """`(d theta_psi / dm) mdot`, summed over the dependencies that have a
        tangent value. Zero when the theta derivative is switched off."""
        if not getattr(self.gia_solver, "_include_theta_derivative", True):
            return 0.0
        coeff = 0.0
        for dep in self.get_dependencies():
            tlm = dep.tlm_value
            if tlm is None:
                continue
            dth = self._dtheta(dep.output)
            if dth != 0.0:
                coeff += dth * _scalar(tlm)
        return coeff


class CoupledLowRankDtNSolveBlock(_ThetaPsiTerms, NonlinearVariationalSolveBlock):
    """The stock coupled solve block, re-classed to carry `A + B`.

    Used on a Firedrake without cached solver blocks. Never constructed
    directly: `adopt_coupled_lowrank_block` changes the `__class__` of the
    stock block the annotated solve produced, so the dependencies, the outputs
    and the `_ad_solvers` dict are the stock ones.
    """

    def __str__(self):
        return "CoupledLowRankDtNSolveBlock"

    # -- 1. forward replay through A + B ------------------------------------
    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        with self._bound_to_block():
            return super()._forward_solve(lhs, rhs, func, bcs, **kwargs)

    # -- 2. adjoint solve (A + B)^T lambda = seed ---------------------------
    def _adjoint_solve(self, dJdu, compute_bdy):
        """The stock adjoint solve, with `theta_psi` bound to this block.

        The stock route assigns this block's saved coefficient values into the
        adjoint form (`_ad_solver_replace_forms`), sets the seed as the
        right-hand side and solves with the block's adjoint variational
        solver. That solver carries both augmentation callbacks and the
        forward context through `LowRankVariationalSolver.solve`, so `A + B`
        is on the Jacobian and on the residual alike, and its `ksponly` takes
        one Krylov solve from zero. Only the prefactor the callbacks read has
        to be this block's value for the duration of the solve.
        """
        with self._bound_to_block():
            return super()._adjoint_solve(dJdu, compute_bdy)

    # -- 4. the theta derivative on the adjoint gradient --------------------
    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        out = super().evaluate_adj_component(inputs, adj_inputs, block_variable,
                                             idx, prepared)
        return self._add_theta_adjoint_term(out, block_variable, prepared,
                                            adj_inputs[0])

    # -- 3 & 5. the tangent operator and its theta right-hand side ----------
    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        coeff = self._theta_tangent_coefficient()
        if coeff != 0.0:
            # tangent rhs is `-(dF/dm) mdot`; add the `-(d theta/dm) B0 psi`
            # column that the stock form is missing. Through the monolithic
            # vec, never `dat.data`: `dFdm` is a cofunction on the MIXED
            # space, whose `dat.data` is a tuple of per-field arrays.
            B0_psi = self._B0_psi()
            if not isinstance(dFdm, Cofunction):
                dFdm = assemble(dFdm)
            dFdm = dFdm.copy()
            with dFdm.dat.vec as dv, B0_psi.dat.vec_ro as bv:
                dv.axpy(-coeff, bv)
        # The tangent operator as a FORM, not the pre-assembled matrix pyadjoint
        # hands in: a variational solve carries the two callbacks and the
        # forward context, and its residual is a plain form that a fieldsplit
        # can split. The assembled matrix's residual is `Action(MatrixBase, u)`,
        # which UFL cannot split, so the iterative preset could not run on it.
        # A solver per tangent solve is the cost of a test-only path.
        u = self.get_outputs()[0]
        dFdu_form = derivative(self._create_F_form(), u.saved_output,
                               TrialFunction(u.output.function_space()))
        problem = LinearVariationalProblem(dFdu_form, dFdm, dudm, bcs=bcs)
        tangent = LinearVariationalSolver(problem, **derivative_solver_kwargs(
            self.gia_solver, "_lowrank_tlm", transpose=False))
        with self._bound_to_block():
            tangent.solve()
        return dudm


if CACHED_SOLVER_BLOCKS:
    class CachedCoupledLowRankDtNSolveBlock(_ThetaPsiTerms, CachedSolverBlock):
        """The cached solve block of Firedrake PR 4638, re-classed for `A + B`.

        The forward, adjoint and tangent solvers of a `CachedSolverBlock` are
        built once by the forward solver and shared by every block it puts on
        the tape. They are built from the forward solver's constructor
        keywords, so they carry both augmentation callbacks, and the adjoint
        and tangent ones the keywords of `derivative_constructor_kwargs`:
        `A + B` is on every operator with no override. What the forms cannot
        carry is added here: `theta_psi` bound to this block's saved values
        around each solve (the shared solvers make this necessary for every
        block, since they read one prefactor), `gia_solve_index` bound to the
        index of this block's forward solve in the cached solvers' contexts,
        and the `(d theta_psi / dm) B0 psi` term on the gradient and the
        tangent.

        The second-order leg is refused: the Hessian of `theta_psi B0 psi` is
        not written.
        """

        def __str__(self):
            return "CachedCoupledLowRankDtNSolveBlock"

        def _solve_index_contexts(self):
            """The `appctx` dicts of the cached replay, tangent and adjoint solvers.

            Firedrake builds these solvers once per forward solver and gives
            each a copy of the forward `appctx` made at that time
            (`_ad_clone_kwargs`), in which `gia_solve_index` is the integer of
            the first taped solve. The preconditioners read the dict the solver
            context holds (`solver._ctx.appctx`), so that is the dict to write.
            One solver can appear twice if Firedrake shares it; the list holds
            each dict once.
            """
            contexts = []
            for cache in (self.forward_cache, self.tangent_cache,
                          self.adjoint_cache):
                ctx = cache.solver._ctx.appctx
                if not any(ctx is c for c in contexts):
                    contexts.append(ctx)
            return contexts

        # -- forward replay through A + B -----------------------------------
        def recompute_component(self, inputs, block_variable, idx, prepared):
            with self._bound_to_block():
                return super().recompute_component(inputs, block_variable,
                                                   idx, prepared)

        # -- adjoint solve (A + B)^T lambda = seed ---------------------------
        def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
            with self._bound_to_block():
                return super().prepare_evaluate_adj(inputs, adj_inputs,
                                                    relevant_dependencies)

        # -- the theta derivative on the adjoint gradient --------------------
        def evaluate_adj_component(self, inputs, adj_inputs, block_variable,
                                   idx, prepared=None):
            out = super().evaluate_adj_component(inputs, adj_inputs,
                                                 block_variable, idx, prepared)
            return self._add_theta_adjoint_term(out, block_variable, prepared,
                                                adj_inputs[0])

        # -- the tangent, with the theta column on its right-hand side -------
        def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable,
                                   idx, prepared=None):
            """The tangent solve of `CachedSolverBlock`, plus the theta column.

            The same steps as `CachedSolverBlock.evaluate_tlm_component`: set
            the dependencies and their tangent values on the cached forms,
            assemble the right-hand side `-(dF/dm) mdot` term by term, and
            solve with the cached tangent solver. Between the assembly and the
            solve the `-(d theta/dm) mdot B0 psi` column is added, and the
            solve runs with `theta_psi` bound to this block.
            """
            self.update_dependencies(use_output=True)
            self.update_tlm_dependencies()

            tlm_rhs = self.tangent_cache.rhs
            tlm_rhs.zero()
            for dFdm, dep in zip(self.tangent_cache.dFdm_forms,
                                 self.get_dependencies()):
                if dep.tlm_value is None:
                    continue
                if dep.output is self.forward_cache.func:
                    continue
                tlm_rhs += assemble(dFdm)

            coeff = self._theta_tangent_coefficient()
            if coeff != 0.0:
                B0_psi = self._B0_psi()
                with tlm_rhs.dat.vec as dv, B0_psi.dat.vec_ro as bv:
                    dv.axpy(-coeff, bv)

            solver = self.tangent_cache.solver
            solver._problem.u.zero()
            with self._bound_to_block():
                solver.solve()
            return solver._problem.u.copy(deepcopy=True)

        def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                     relevant_dependencies):
            raise NotImplementedError(
                "The second-order adjoint of the coupled low-rank DtN solve is "
                "not written: the Hessian of theta_psi B0 psi is missing.")
else:
    CachedCoupledLowRankDtNSolveBlock = None


def adopt_coupled_lowrank_block(gia_solver, tape, n0, controls=None):
    """Re-class the stock solve block this solve added, and record it.

    Identity, never tape position: `update_total_mass` and
    `project_out_nullspace` add blocks around the solve, so `get_blocks()[-1]`
    is unsound. The block is the one solve block whose outputs include the
    solver's own solution.

    Args:
      gia_solver: the `SelfGravitatingGIASolver`; `gia_solver.solution` is the
        output identity and `gia_solver.adjoint_block` is set to the result.
      tape: the working tape.
      n0: `len(tape.get_blocks())` recorded BEFORE the annotated solve.
      controls: optional iterable of control `Function`s. When given, D1 is
        checked here: a control absent from the block's dependencies raises,
        rather than producing a silent `0.0` gradient later.

    The block class depends on the Firedrake: a `NonlinearVariationalSolveBlock`
    becomes a `CoupledLowRankDtNSolveBlock`, and with Firedrake PR 4638 a
    `CachedSolverBlock` becomes a `CachedCoupledLowRankDtNSolveBlock`
    (`SOLVE_BLOCK_TYPE`).

    Returns:
      The adopted block, or `None` if the solve added no annotated solve block.
    """
    added = tape.get_blocks()[n0:]
    solution = gia_solver.solution
    ours = [b for b in added
            if isinstance(b, SOLVE_BLOCK_TYPE)
            and any(o.output is solution for o in b.get_outputs())]
    if not ours:
        return None
    if len(ours) != 1:
        raise RuntimeError(
            f"{len(ours)} solve blocks output the coupled solution since the "
            "solve started; route 1.5b needs exactly one to adopt. Identify by "
            "output identity, never by tape position (REVIEW-ADJOINT L26).")
    block = ours[0]
    block.__class__ = (CachedCoupledLowRankDtNSolveBlock if CACHED_SOLVER_BLOCKS
                       else CoupledLowRankDtNSolveBlock)
    block.gia_solver = gia_solver
    # `SelfGravitatingGIASolver.solve` raised the index before this solve, so
    # this is the index the forward solve's preconditioners saw.
    block.gia_solve_index = gia_solver.appctx.get("gia_solve_index")

    if controls is not None:
        require_controls_reach_block(block, controls)

    gia_solver.adjoint_block = block
    return block
