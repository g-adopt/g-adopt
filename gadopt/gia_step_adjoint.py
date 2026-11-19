r"""What every recorded time step of `SelfGravitatingGIASolver` puts on the tape.

One call of `SelfGravitatingGIASolver.solve` under annotation records, in this
order:

    TapeSweepMarkerBlock(before)    no dependencies, no outputs
    the step block                  the coupled solve of the step
    TapeSweepMarkerBlock(after)     no dependencies, no outputs
    KernelProjectionBlock           the rigid-rotation projection, if declared

followed by the blocks of the history accept and of the copy into `u` and
`psi`. The step block is one of two kinds:

  - The low-rank DtN path with the DG history: the solve block of Firedrake's
    annotated variational solve, adopted by
    `gadopt.dtn_coupled_adjoint.adopt_coupled_lowrank_block`
    (`CachedCoupledLowRankDtNSolveBlock`, or `CoupledLowRankDtNSolveBlock` on
    a Firedrake without cached solver blocks).
  - The Quadrature history, with either DtN representation:
    `gadopt.constitutive_integrators.ConstitutiveStepBlock`, which records the
    coupled solve and the local constitutive solve at the quadrature points
    as one map.

Both kinds inherit `GIAStepBlockMixin`. It holds what the two have in common
and what no UFL form can carry:

  1. The binding of the shared solver state to the block for each of its
     solves (`_bound_to_block`): the prefactor `theta_psi` of the low-rank
     DtN term, and the markers the rebuild rule of the preconditioners reads
     (`gia_solve_index`, `gia_dt`, `gia_time_alpha`, `gia_tape_sweep`). A
     replay, an adjoint sweep and a tangent sweep do not run
     `SelfGravitatingGIASolver.solve`, so without this binding their solves
     read the values of the last forward step.
  2. The derivative of `theta_psi` with respect to its factors. The low-rank
     term `theta_psi B0 psi` enters the residual through PETSc callbacks, so
     the forms miss the column `(d theta_psi / dm) B0 psi` of `dF/dm`. The
     mixin adds it to the adjoint gradient and to the tangent right-hand side.

The kernel projection is not part of any step block: both kinds leave the
gauge of their solution free, and the `KernelProjectionBlock` that follows
the step removes it on the forward, the replay, the adjoint and the tangent
sweep alike (`gadopt.nullspaces`).

The multiplier DtN path with the DG history (`_StockStep` of the GIA solver)
puts Firedrake's solve block on the tape unchanged; its residual is a plain
UFL form and it has no low-rank term, so it needs no adoption.
"""

from contextlib import contextmanager

import numpy as np
from firedrake import Cofunction
from pyadjoint import Block

from .utility import scalar_value

__all__ = ["GIAStepBlockMixin", "TapeSweepMarkerBlock"]

#: The value of a missing key, distinct from every value a context can hold.
_MISSING = object()


class TapeSweepMarkerBlock(Block):
    """A tape block that raises `gia_tape_sweep` when a sweep of the tape starts.

    The rebuild rule of the GIA preconditioners (`gadopt.preconditioners.
    _RebuildRule`) and the condensed operator of `InternalVariableSCPC` key
    on markers that only `SelfGravitatingGIASolver.solve` moves. A replay at
    a new control value, or the adjoint sweep that follows it, does not run
    that method, so without a further marker the replay and derivative
    solvers keep preconditioner pieces built for the old control value. With
    a capped outer solve that is a wrong answer and not only a slow one. This
    block gives the rule that marker.

    `SelfGravitatingGIASolver.solve` puts two of these on the tape around each
    annotated step block: one before it (`before=True`), which a replay and a
    tangent sweep reach before the step block, and one after it, which an
    adjoint and a Hessian sweep reach before the step block, because they run
    the tape backwards. A block counts a new sweep when the direction of the
    sweep changes, or when the solve index it belongs to does not move on in
    the direction of the sweep (a replay that starts again at the first step,
    or a tape of one step). Within one sweep the counter stays, so the lagged
    rebuild of an adjoint sweep keeps its lag.

    The block has no dependencies and no outputs, so it changes no value on
    the tape. It writes into the solver's application context, which the
    forward, replay and derivative solvers of every step kind share on this
    Firedrake; `GIAStepBlockMixin._bound_to_block` also copies it into the
    contexts of the cached solvers of an adopted block.

    Args:
      gia_solver: The `SelfGravitatingGIASolver` whose application context
        holds the counter.
      solve_index: The `gia_solve_index` of the forward solve this block
        belongs to.
      before: `True` for the block in front of the step block, `False` for
        the one behind it.
    """

    def __init__(self, gia_solver, solve_index, before):
        super().__init__()
        self.gia_solver = gia_solver
        self.solve_index = solve_index
        self.before = before

    def __str__(self):
        return f"TapeSweepMarkerBlock({'before' if self.before else 'after'})"

    def _visit(self, direction):
        """Count a new sweep if this visit does not continue the last one.

        Args:
          direction: `+1` for a sweep in tape order (replay, tangent), `-1`
            for a sweep in reverse order (adjoint, Hessian).
        """
        solver = self.gia_solver
        last = getattr(solver, "_tape_sweep_visit", None)
        if (last is None or last[0] != direction
                or (self.solve_index - last[1]) * direction <= 0):
            solver.appctx["gia_tape_sweep"] = solver.appctx.get("gia_tape_sweep", 0) + 1
        solver._tape_sweep_visit = (direction, self.solve_index)

    # pyadjoint calls these four for every block of a sweep. The stock
    # versions return early for a block without outputs, so they are
    # replaced, not extended.
    def recompute(self, markings=False):
        if self.before:
            self._visit(+1)

    def evaluate_tlm(self, markings=False):
        if self.before:
            self._visit(+1)

    def evaluate_adj(self, markings=False):
        if not self.before:
            self._visit(-1)

    def evaluate_hessian(self, markings=False):
        if not self.before:
            self._visit(-1)


class GIAStepBlockMixin:
    r"""What every step block of `SelfGravitatingGIASolver` adds to its solves.

    Mixed into a `pyadjoint.Block` subclass whose dependencies include the
    coefficients of the step residual. The block sets `gia_solver` and
    `gia_solve_index` when it is put on the tape, and wraps each of its
    replay, adjoint and tangent solves in `_bound_to_block`.

    **Values from the tape.** Every value this class binds or differentiates
    is read from the block's saved dependency values (`_saved_scalar`), not
    from the live objects. The live objects hold the values of the last
    forward assignment; the saved values are those of this block's point on
    the tape, also during a replay at another control value.

    **The low-rank prefactor.** With the low-rank DtN representation the
    residual carries `theta_psi B0 psi` through PETSc callbacks, with

        theta_psi = c B_mu / Lambda

    (`c` the scaling factor, `B_mu` floored to `NULL_COUPLING_ROW_SCALE` at
    zero). The callbacks read `theta_psi` off the shared `dtn_operator`. The
    forms do not carry this term, so the derivative

        d theta_psi / dm = (B_mu / Lambda) dc/dm + (c / Lambda) dB_mu/dm
                           - (c B_mu / Lambda^2) dLambda/dm

    times `B0 psi` is added here: `-(d theta_psi / dm) lambda . B0 psi` to
    the gradient of each dependency `m` (`_add_theta_adjoint_term`), and
    `-(d theta_psi / dm) mdot B0 psi` to the tangent right-hand side
    (`_theta_tangent_coefficient`, `_B0_action`). The switch
    `gia_solver._include_theta_derivative` turns this term off, for the test
    that shows the gradient is wrong without it.
    """

    #: The owning `SelfGravitatingGIASolver`. It carries the shared
    #: `dtn_operator`, the application context and the `theta_psi` factors.
    gia_solver = None
    #: The `gia_solve_index` of the forward solve that put this block on the
    #: tape. `_bound_to_block` writes it into the application contexts around
    #: each replay, adjoint and tangent solve.
    gia_solve_index = None

    # -- Values at this block's point on the tape -------------------------------

    def _saved_scalar(self, obj) -> float:
        """The value of `obj` at this block's point on the tape, as a float.

        Its saved output if `obj` is a dependency of the block; otherwise its
        live value, which then cannot change on the tape (a float or a
        `Constant`).
        """
        for dep in self.get_dependencies():
            if dep.output is obj:
                return scalar_value(dep.saved_output)
        return scalar_value(obj)

    def _saved_dt(self):
        """The time step of this block's forward solve, as a float, or None.

        A time step that is a `Function` on the Real space is a coefficient
        of the residual and so a dependency of the step block, and its saved
        output is its value at this point of the tape: the value of the
        forward solve in an adjoint sweep, and the replayed value in a
        replay. A float or a `Constant` (a UFL constant value, not a
        coefficient, so not a dependency) gives None, and the rebuild rule
        then reads the live value.
        """
        dt = getattr(self.gia_solver, "dt", None)
        return None if dt is None else self._saved_dependency(dt)

    def _saved_alpha(self):
        """The leading BDF coefficient of this block's step, or None.

        `time_integrator.alpha` is a Real Function that the history
        integrator assigns before every step (1 for backward Euler, the
        variable-step BDF2 value otherwise), so it is a dependency of the
        step block whenever the residual reads it.
        """
        integrator = getattr(self.gia_solver, "time_integrator", None)
        return None if integrator is None else self._saved_dependency(integrator.alpha)

    def _saved_dependency(self, obj):
        """The saved value of the dependency `obj` as a float, or None if absent."""
        for dep in self.get_dependencies():
            if dep.output is obj:
                return scalar_value(dep.saved_output)
        return None

    # -- The low-rank prefactor and its derivative -------------------------------

    @property
    def _lowrank(self) -> bool:
        """Whether the step residual carries the low-rank DtN term."""
        return self.gia_solver.dtn_representation == "lowrank"

    def _theta_factors(self):
        """The saved values `(c, B_mu, Lambda)` of `theta_psi`, and their objects.

        Returns:
          `(objects, values)`: the objects `(scaling_factor, B_mu, Lambda)`,
          and their values at this point of the tape, with `B_mu` floored to
          `NULL_COUPLING_ROW_SCALE` at zero. The floor is a constant, so at a
          zero `B_mu` the prefactor does not depend on `B_mu`, and its object
          is given as None.
        """
        from .gia_presets import NULL_COUPLING_ROW_SCALE
        solver = self.gia_solver
        objects = [solver.scaling_factor, solver.approximation.B_mu, solver.Lambda]
        values = [self._saved_scalar(obj) for obj in objects]
        if values[1] == 0.0:
            values[1] = NULL_COUPLING_ROW_SCALE
            objects[1] = None
        return objects, values

    def _bound_theta(self) -> float:
        """`theta_psi = c B_mu / Lambda` at this block's saved values."""
        _, (c, b_mu, lam) = self._theta_factors()
        return c * b_mu / lam

    def _dtheta(self, control) -> float:
        """`d theta_psi / d control`, zero unless the control is one of its factors."""
        objects, (c, b_mu, lam) = self._theta_factors()
        derivatives = (b_mu / lam, c / lam, -c * b_mu / (lam * lam))
        return sum(value for obj, value in zip(objects, derivatives)
                   if obj is not None and control is obj)

    def _theta_derivative_on(self) -> bool:
        """Whether the `theta_psi` derivative is added (low-rank, switch on)."""
        return self._lowrank and getattr(
            self.gia_solver, "_include_theta_derivative", True)

    def _B0_action(self, state) -> Cofunction:
        """`B0 x`, the low-rank action at prefactor one, at the mixed state `x`.

        Args:
          state: A Function on the mixed space of the step.

        Returns:
          A Cofunction on the dual of that space.
        """
        out = Cofunction(state.function_space().dual())
        op = self.gia_solver.dtn_operator
        with state.dat.vec_ro as xv, out.dat.vec_rw as yv:
            yv.set(0.0)
            op.apply_local(xv.array_r, yv.array_w, theta=1.0)
        return out

    def _add_theta_adjoint_term(self, out, control, state, adjoint):
        """Add `-(d theta_psi / d control) lambda . B0 x` to the gradient `out`.

        The gradient from the forms is `-lambda^T dF/dm` without `B`. The
        missing part is a scalar times `d theta_psi / dm`, so it is nonzero
        only for the factors of `theta_psi`, which are scalar `Real`
        Functions: the scalar is added to every entry of `out`.

        Args:
          out: The gradient from the forms, a Cofunction, or None.
          control: The dependency the gradient belongs to.
          state: The forward solution `x` of this block.
          adjoint: The adjoint solution `lambda` of this block.

        Returns:
          `out` with the term added; a new Cofunction if `out` was None and
          the term is not zero.
        """
        if not self._theta_derivative_on() or adjoint is None:
            return out
        dth = self._dtheta(control)
        if dth == 0.0:
            return out
        if control.ufl_shape or control.function_space().ufl_element().family() != "Real":
            raise ValueError("The factors of theta_psi must be scalar Real Functions "
                             "to be differentiated on the low-rank DtN path.")
        B0_x = self._B0_action(state)
        with adjoint.dat.vec_ro as lv, B0_x.dat.vec_ro as bv:
            pairing = float(lv.dot(bv))
        extra = -dth * pairing
        if out is None:
            out = Cofunction(control.function_space().dual())
        out.dat.data_wo[:] = np.asarray(out.dat.data_ro) + extra
        return out

    def _theta_tangent_coefficient(self) -> float:
        """`(d theta_psi / dm) mdot`, summed over the dependencies with a tangent.

        Zero when the term is off. The tangent right-hand side then carries
        `-coefficient B0 x` (`_B0_action`).
        """
        if not self._theta_derivative_on():
            return 0.0
        coefficient = 0.0
        for dep in self.get_dependencies():
            if dep.tlm_value is None:
                continue
            dth = self._dtheta(dep.output)
            if dth != 0.0:
                coefficient += dth * scalar_value(dep.tlm_value)
        return coefficient

    # -- The shared solver state, bound to this block ---------------------------

    def _solve_index_contexts(self):
        """The application contexts the preconditioners of this block read.

        The solvers of the step, its derivative solvers and, on stock
        Firedrake, the replay solver all carry the solver's own `appctx`
        dict. `CachedCoupledLowRankDtNSolveBlock` overrides this with the
        dicts of its cached solvers, which Firedrake may have copied.

        Returns:
          A list of the `appctx` dicts to bind, each dict once.
        """
        return [self.gia_solver.appctx]

    @contextmanager
    def _bound_to_block(self):
        """Bind what the shared solvers read to this block, for one solve.

        Five values are bound; all but the sweep counter are restored on exit.

        - `theta_psi` on the shared `dtn_operator` (low-rank path only), from
          this block's saved values. The augmentation callbacks read it at
          every application.
        - `gia_solve_index` in the application contexts, set to the index of
          the forward solve that put this block on the tape. The rebuild rule
          of the preconditioners (`gadopt.preconditioners._RebuildRule`)
          rebuilds when this index moves. Nothing else moves it during a
          replay or a derivative sweep, because those solves do not go
          through `SelfGravitatingGIASolver.solve`. With the binding, a replay
          rebuilds at the same steps as the forward run, and an adjoint sweep
          rebuilds as it walks back through the steps.
        - `gia_dt` and `gia_time_alpha`, the time step and the leading BDF
          coefficient of this block's forward solve. The rebuild rule also
          rebuilds when the effective step `dt / alpha` differs from the one
          of the last build (`gadopt.preconditioners._published_dt`). The
          live objects hold the values of the last forward assignment.
        - `gia_tape_sweep`, the counter of `TapeSweepMarkerBlock`, copied from
          the solver's own context into every context the preconditioners
          read. Where a cached solver got a copy of the dict, the copy would
          otherwise keep the counter of the time it was made. The counter
          only grows and is not restored.

        A block put on the tape without a solve index leaves the index
        alone; a dependency that is absent (a float time step) is left alone.
        Each context is restored to its own previous value, so the forward
        solver's state is unchanged after a replay that shares its dict.

        Yields:
          None.
        """
        contexts = self._solve_index_contexts()
        bound = {}
        if self.gia_solve_index is not None:
            bound["gia_solve_index"] = self.gia_solve_index
        dt = self._saved_dt()
        if dt is not None:
            bound["gia_dt"] = dt
        alpha = self._saved_alpha()
        if alpha is not None:
            bound["gia_time_alpha"] = alpha
        saved = [{key: ctx.get(key, _MISSING) for key in bound} for ctx in contexts]
        sweep = self.gia_solver.appctx.get("gia_tape_sweep")
        for ctx in contexts:
            ctx.update(bound)
            if sweep is not None:
                ctx["gia_tape_sweep"] = sweep
        op = self.gia_solver.dtn_operator if self._lowrank else None
        if op is not None:
            saved_theta = op.theta
            op.theta = self._bound_theta()
        try:
            yield
        finally:
            if op is not None:
                op.theta = saved_theta
            for ctx, previous in zip(contexts, saved):
                for key, value in previous.items():
                    if value is _MISSING:
                        ctx.pop(key, None)
                    else:
                        ctx[key] = value
