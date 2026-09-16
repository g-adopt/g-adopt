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
"""

from contextlib import contextmanager

import numpy as np
from firedrake import (Cofunction, LinearVariationalProblem,
                       LinearVariationalSolver, NonlinearVariationalSolver,
                       TrialFunction, assemble, derivative)
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock
from pyadjoint.tape import annotate_tape

__all__ = ["CoupledLowRankDtNSolveBlock", "LowRankVariationalSolver",
           "adopt_coupled_lowrank_block", "derivative_solver_kwargs",
           "require_controls_reach_block"]


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
        # and the wrapper needs to see that key itself.
        if self.gia_solver is not None and annotate_tape(dict(kwargs)):
            kwargs.setdefault("adj_kwargs", derivative_solver_kwargs(
                self.gia_solver, "_lowrank_adj", transpose=True))
        return super().solve(**kwargs)


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


class CoupledLowRankDtNSolveBlock(NonlinearVariationalSolveBlock):
    """The stock coupled solve block, re-classed to carry `A + B`.

    Never constructed directly: `adopt_coupled_lowrank_block` changes the
    `__class__` of the stock block the annotated solve produced, so the
    dependencies, the outputs and the `_ad_solvers` dict are the stock ones.
    """

    #: The owning `SelfGravitatingGIASolver`, set by the adopt function. Carries
    #: the shared `dtn_operator`, the solver parameters and the
    #: `_include_theta_derivative` switch.
    gia_solver = None

    def __str__(self):
        return "CoupledLowRankDtNSolveBlock"

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

    @contextmanager
    def _theta_bound(self):
        """Set the shared operator's prefactor to this block's value for the
        duration of one solve, then restore the live callable."""
        op = self.gia_solver.dtn_operator
        saved = op.theta
        op.theta = self._bound_theta()
        try:
            yield op
        finally:
            op.theta = saved

    # -- the low-rank action, as vectors ------------------------------------
    def _B0_psi(self) -> Cofunction:
        """`B0 psi` (prefactor one) at the forward solution, as a cofunction."""
        Z = self.function_space
        psi = self.get_outputs()[0].saved_output
        out = Cofunction(Z.dual())
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

    # -- 1. forward replay through A + B ------------------------------------
    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        with self._theta_bound():
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
        with self._theta_bound():
            return super()._adjoint_solve(dJdu, compute_bdy)

    # -- 4. the theta derivative on the adjoint gradient --------------------
    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        out = super().evaluate_adj_component(inputs, adj_inputs, block_variable,
                                             idx, prepared)
        if not getattr(self.gia_solver, "_include_theta_derivative", True):
            return out
        if prepared is None or adj_inputs[0] is None:
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

    # -- 3 & 5. the tangent operator and its theta right-hand side ----------
    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        if getattr(self.gia_solver, "_include_theta_derivative", True):
            coeff = 0.0
            for dep in self.get_dependencies():
                tlm = dep.tlm_value
                if tlm is None:
                    continue
                dth = self._dtheta(dep.output)
                if dth != 0.0:
                    coeff += dth * _scalar(tlm)
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
        with self._theta_bound():
            tangent.solve()
        return dudm


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

    Returns:
      The adopted block, or `None` if the solve added no annotated solve block.
    """
    added = tape.get_blocks()[n0:]
    solution = gia_solver.solution
    ours = [b for b in added
            if isinstance(b, NonlinearVariationalSolveBlock)
            and any(o.output is solution for o in b.get_outputs())]
    if not ours:
        return None
    if len(ours) != 1:
        raise RuntimeError(
            f"{len(ours)} solve blocks output the coupled solution since the "
            "solve started; route 1.5b needs exactly one to adopt. Identify by "
            "output identity, never by tape position (REVIEW-ADJOINT L26).")
    block = ours[0]
    block.__class__ = CoupledLowRankDtNSolveBlock
    block.gia_solver = gia_solver

    if controls is not None:
        require_controls_reach_block(block, controls)

    gia_solver.adjoint_block = block
    return block
