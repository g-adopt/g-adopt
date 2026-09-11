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
  2. `_adjoint_solve`  - solve `(A + B)^T lambda = seed`. `A + B` is NOT
     symmetric (the matrix-free Dirichlet rows are zeroed but not lifted), so
     the genuine transpose form `adjoint(dFdu)` is assembled; `B` is symmetric,
     so its transpose action is the same `apply_local`.
  3. `_assemble_and_solve_tlm_eq` - the tangent operator is `A + B`, and the
     tangent right-hand side carries the `(d theta_psi/dm) B0 psi` term.
  4 & 5. The `(d theta_psi/dm) B0 psi` term on `evaluate_adj_component` and on
     the tangent right-hand side. `theta_psi = scaling_factor B_mu / Lambda`,
     all live coefficients, so its control derivative is a scalar that
     multiplies `B0 psi` - the low-rank action at prefactor one.

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
from firedrake import (Cofunction, Function, TrialFunction, adjoint, assemble,
                       derivative, solve)
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock

from .dtn_coupled import install_augmented_context

__all__ = ["CoupledLowRankDtNSolveBlock", "adopt_coupled_lowrank_block",
           "require_controls_reach_block"]


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
        u = self.get_outputs()[0].output
        F_form = self._create_F_form()
        dFdu = derivative(F_form, self.get_outputs()[0].saved_output,
                          TrialFunction(u.function_space()))
        adj_form = adjoint(dFdu)
        bcs = self._homogenize_bcs()
        A = assemble(adj_form, bcs=bcs, mat_type="matfree")

        rhs = dJdu.copy()
        for bc in self.bcs:
            bc.zero(rhs)

        adj_sol = Function(u.function_space())
        op = self.gia_solver.dtn_operator
        with self._theta_bound():
            # `A + B` symmetric part of B: `apply_local` is its own transpose.
            install_augmented_context(A.petscmat, op)
            solve(A, adj_sol, rhs,
                  solver_parameters=self.gia_solver.solver_parameters,
                  options_prefix=self.gia_solver.name + "_lowrank_adj")

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = self._compute_adj_bdy(adj_sol, None, adj_form,
                                                dJdu.copy())
        return adj_sol, adj_sol_bdy

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
        op = self.gia_solver.dtn_operator
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
        with self._theta_bound():
            install_augmented_context(dFdu.petscmat, op)
            solve(dFdu, dudm, dFdm,
                  solver_parameters=self.gia_solver.solver_parameters,
                  options_prefix=self.gia_solver.name + "_lowrank_tlm")
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
