r"""pyadjoint blocks for the low-rank DtN gravitational Poisson path.

The multiplier path is a `NonlinearVariationalSolver` on a plain UFL residual,
so pyadjoint tapes it and the adjoint comes out for free. The low-rank path
drives a PETSc `KSP` directly - it has to, because `NonlinearVariationalSolver`
offers no hook to hand PETSc a custom `Amat` while taking `Pmat` from a form -
and a PETSc-level operator is invisible to a tape. So the derivative is written
by hand here.

## The mathematics, which is easier than the bookkeeping

The solve is

    K psi = b(rho, sigma, G),      K = A + B

with **`K` symmetric and entirely control-independent**. Density, surface
density and the gravitational constant enter only terms linear in the test
function, so they vanish under differentiation of the operator; every remaining
coefficient in `K` is geometry - the Robin shift, the DtN eigenvalues, the
constraint scalings, the boundary radius.

That is measured, not argued from inspection, because it is the premise the
whole derivation rests on and it would stop being true the day someone adds a
density-dependent Robin coefficient. Applying the built operator to a fixed
random vector at two densities and two gravitational constants:

    || K(rho1)x - K(rho2)x ||_inf / ||Kx||          = 0.000e+00
    || K(G=1)x - K(G=6.674e-11)x ||_inf / ||Kx||    = 0.000e+00

Bit-identical, not merely close. Therefore

    tangent:  delta_psi = K^-1 (db/dm) delta_m
    adjoint:  m_bar     = (db/dm)^T K^-T psi_bar = (db/dm)^T K^-1 psi_bar

**The same operator, the same preconditioner, only a different right-hand
side.** There is no second factorisation, no transposed nullspace subtlety, and
`K^T = K` is not assumed - it is measured, to 1e-12, by
`test_operator_is_symmetric`.

`db/dm` is a plain UFL form, so its transposed action is an assembly:
`assemble(action(adjoint(derivative(rhs_form, m)), lambda))`. The sharp edge is
the primal/dual bookkeeping, not the calculus.

The second derivative of `b` vanishes for any single control - `b` is linear in
`rho`, in `sigma` and in `G` separately - but **not** for a mixed
`(G, rho)` perturbation, since `b` contains `4 pi G rho v dx`. The Hessian term
below is written with the general UFL second derivative rather than dropped, so
a two-control Hessian is right rather than silently zero.

## Trace coefficients are taped, deliberately

`c_k = u_k . psi / (scale_k A_h)` is a linear functional of `psi` with constant
`u_k`, so it is trivially differentiable, and the trace spectrum *is* the geoid
- a primary inversion observable in this field rather than a diagnostic. Making
it taped costs one small block and keeps a capability the multiplier path has
for free.
"""

import numpy as np
import ufl
from firedrake import Cofunction, Function, action, adjoint, assemble, derivative
from pyadjoint import AdjFloat, Block
from pyadjoint.tape import annotate_tape, get_working_tape, no_annotations

__all__ = ["LowRankGravitySolveBlock", "ModeRow", "TraceCoefficientBlock",
           "annotate_lowrank_solve", "taped_mode_coefficients",
           "taped_trace_coefficients"]


class LowRankGravitySolveBlock(Block):
    """One low-rank DtN solve, differentiated by hand.

    Dependencies are the coefficients of the right-hand side form that pyadjoint
    can see; the operator contributes none, because it is control-independent.
    """

    def __init__(self, solver, solution, dependencies):
        super().__init__()
        self._form = None
        self._substitution = {}
        self.solver = solver
        self.rhs_form = solver.rhs_form
        self.function_space = solution.function_space()
        self._dependencies_order = list(dependencies)
        for coefficient in self._dependencies_order:
            self.add_dependency(coefficient)
        # A FRESH block variable for the output. Re-using the existing one
        # makes the solution's post-solve state share identity with its
        # pre-solve state, and the tape's DAG is then malformed in a way that
        # replays correctly and differentiates to nonsense.
        self.add_output(solution.create_block_variable())

    def __str__(self):
        return "LowRankGravitySolveBlock"

    # -- forward -----------------------------------------------------------
    @no_annotations
    def recompute_component(self, inputs, block_variable, idx, prepared):
        """Replay: substitute the saved inputs into the form and re-solve.

        The values are put in by `ufl.replace`, never by assigning into the
        coefficient objects the form refers to. Assigning would overwrite the
        user's own control `Function` - the same object `Control(m)` holds - so
        every replay at a perturbed control would permanently move the control.
        Measured, that presents as a Taylor test whose *gradient-free* R0 ladder
        has negative rates, because `J(m)` drifts between evaluations while each
        individual replay agrees exactly with a fresh solve. Both facts are true
        at once and only the second one is reassuring.
        """
        self.solver.mixed_solution.assign(0)
        self.solver._solve_lowrank(self._replaced(inputs))
        return self.solver.mixed_solution.copy(deepcopy=True)

    def _replaced(self, inputs):
        """The right-hand side form with the saved dependency values in it.

        Also records the substitution, because everything downstream has to
        differentiate with respect to the coefficient that is now *in* the
        form. Differentiating the replaced form with respect to the original
        object gives an empty form, and `action` on a form with no arguments
        raises `IndexError: tuple index out of range` from deep inside UFL -
        an error message that says nothing at all about the actual mistake.
        """
        mapping = {coefficient: value
                   for coefficient, value in zip(self._dependencies_order, inputs)
                   if value is not None and value is not coefficient}
        self._substitution = mapping
        return ufl.replace(self.rhs_form, mapping) if mapping else self.rhs_form

    def _in_form(self, coefficient):
        """The object standing for `coefficient` in the current form."""
        return getattr(self, "_substitution", {}).get(coefficient, coefficient)

    # -- adjoint -----------------------------------------------------------
    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        """Solve `K lambda = psi_bar` once, then reuse it for every dependency.

        `K` is symmetric, so the adjoint solve is the forward solve with a
        different right-hand side - the same matrix, the same GAMG hierarchy,
        no second setup.
        """
        self._form = self._replaced(inputs)
        return self._solve_with(adj_inputs[0])

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if prepared is None or adj_inputs[0] is None:
            return None
        coefficient = self._dependencies_order[idx]
        return self._rhs_derivative_transpose(coefficient, prepared)

    # -- tangent -----------------------------------------------------------
    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_dependencies):
        self._form = self._replaced(inputs)
        return None

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        """`delta_psi = K^-1 (db/dm) delta_m`, summed over perturbed controls."""
        rhs = None
        for coefficient, perturbation in zip(self._dependencies_order, tlm_inputs):
            if perturbation is None:
                continue
            contribution = assemble(action(
                derivative(self._form, self._in_form(coefficient)), perturbation))
            rhs = _accumulate(rhs, contribution)
        if rhs is None:
            return None
        return self._solve_with(rhs)

    # -- hessian -----------------------------------------------------------
    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        self._form = self._replaced(inputs)
        first = (self._solve_with(adj_inputs[0])
                 if adj_inputs and adj_inputs[0] is not None else None)
        second = (self._solve_with(hessian_inputs[0])
                  if hessian_inputs and hessian_inputs[0] is not None else None)
        return first, second

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        """Second-order adjoint.

        Two terms. The first is the adjoint applied to the second-order input,
        which is all there is for a single control since `b` is linear in each
        of `rho`, `sigma` and `G` separately. The second is the mixed
        derivative of `b` contracted with the first-order adjoint, which is
        nonzero only when two controls are perturbed together - `4 pi G rho v`
        being bilinear in `(G, rho)`.
        """
        first, second = prepared
        coefficient = self._dependencies_order[idx]
        out = None
        if second is not None:
            out = self._rhs_derivative_transpose(coefficient, second)
        if first is None:
            return out
        for other, perturbation in zip(self._dependencies_order,
                                       (bv.tlm_value for bv in self.get_dependencies())):
            if perturbation is None:
                continue
            mixed = derivative(
                derivative(self._form, self._in_form(coefficient)),
                self._in_form(other))
            try:
                contribution = assemble(
                    action(adjoint(action(mixed, perturbation)), first))
            except Exception:
                # An identically zero mixed derivative: UFL declines to form
                # it rather than returning a zero form.
                continue
            out = _accumulate(out, contribution)
        return out

    # -- helpers -----------------------------------------------------------
    def _solve_with(self, rhs):
        """Solve `K x = rhs` with the existing KSP; return a primal Function.

        The right-hand side is copied rather than used in place: it belongs to
        pyadjoint, and the strong-condition zeroing below would otherwise
        mutate the tape's own adjoint value.
        """
        solution = Function(self.function_space)
        rhs_function = _as_cofunction(rhs, self.function_space)
        if self.solver.strong_bcs:
            # Homogeneous conditions on the adjoint/tangent: the constrained
            # values are data, so their sensitivity is carried by the control
            # they came from, not by this solve.
            owned = self.function_space.nodes.owned.local_size
            # A partial write, so `data_rw`: a write-only access in pyop3
            # drops any pending halo reduction of the whole vector.
            for bc in self.solver.strong_bcs:
                nodes = bc.nodes[bc.nodes < owned]
                rhs_function.dat.data_rw[nodes] = 0.0
        with rhs_function.dat.vec_ro as b, solution.dat.vec_rw as x:
            self.solver.ksp.solve(b, x)
        return solution

    def _rhs_derivative_transpose(self, coefficient, multiplier):
        """`(db/dm)^T multiplier`, as a value in the dual of `m`'s space."""
        form = derivative(getattr(self, "_form", None) or self.rhs_form,
                          self._in_form(coefficient))
        return assemble(action(adjoint(form), multiplier))


class TraceCoefficientBlock(Block):
    """One trace coefficient `c_k = u_k . psi / (scale_k A_h)` on the tape.

    A linear functional of the potential with a constant vector, so the
    derivative is that same vector: the adjoint of a scalar seed `c_bar` is
    `c_bar * u_k / (scale_k A_h)` as a cofunction on the solution's space,
    and the Hessian is the same map applied to the second-order seed.

    The one block of both low-rank DtN solvers. The solution is the
    potential itself for `GravitySolver` and the mixed solution of
    `SelfGravitatingGIASolver`; the row (`ModeRow`) holds the indices of the
    potential degrees of freedom in that solution's local vector, so the
    block reads and writes the solution only through `dat.vec_ro` and
    `dat.vec_rw`, which give one local array for a plain and a mixed space
    alike (a mixed `dat.data_ro` is a tuple of arrays).

    **Why it exists.** Reading the trace off `dat.data_ro` and returning
    `float()` severs the tape by construction. The trace spectrum is the
    geoid, the observable an inversion of either solver most often targets,
    so on the low-rank path the coefficients would have correct values with
    a gradient of exactly zero.

    Args:
      solution: The Function the coefficient is read from.
      row: The `ModeRow` of the coefficient.
      value: The `AdjFloat` that holds the coefficient's value.
    """

    def __init__(self, solution, row, value):
        super().__init__()
        self.function_space = solution.function_space()
        self.row = row
        self.add_dependency(solution)
        self.add_output(value.create_block_variable())

    def __str__(self):
        return "TraceCoefficientBlock"

    @staticmethod
    def _local(function):
        """The local (owned) entries of `function`, one array for any space."""
        with function.dat.vec_ro as vec:
            return np.array(vec.array_r, dtype=float)

    def _seeded_cofunction(self, seed):
        """`seed * u_k / (scale_k A_h)` as a cofunction on the solution's space."""
        out = Cofunction(self.function_space.dual())
        with out.dat.vec_rw as vec:
            self.row.scatter_into(vec.array_w, float(seed))
        return out

    @no_annotations
    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.row.comm.allreduce(self.row.local_dot(self._local(inputs[0])))

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        seed = adj_inputs[0]
        return None if seed is None else self._seeded_cofunction(seed)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        perturbation = tlm_inputs[0]
        if perturbation is None:
            return None
        return self.row.comm.allreduce(self.row.local_dot(self._local(perturbation)))

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear in psi, so the second derivative is the first applied to the
        # second-order seed.
        seed = hessian_inputs[0]
        return None if seed is None else self._seeded_cofunction(seed)


class ModeRow:
    """One mode's boundary functional, scaled for coefficient recovery.

    **The recovery weight comes from `rows.recovery`, the same array the
    forward value uses**, so the `1 / (scale_k A_h)` denominator exists in
    exactly one place and cannot drift between the value and its derivative.

    Args:
      rows: The `BoundaryModeRows` of one DtN boundary.
      index: The mode's position in `rows`.
      dofs: The local indices of the boundary degrees of freedom in the
        local vector of the solution: `rows.dofs` when the solution is the
        potential itself, the shifted indices of the potential field in a
        mixed solution otherwise (`gadopt.dtn_coupled.monolithic_rows`).
      comm: The communicator of the sum over ranks.
    """

    def __init__(self, rows, index, dofs, comm):
        self.values = rows.rows[index] * rows.recovery[index]
        self.dofs = dofs
        self.comm = comm

    def local_dot(self, x_local):
        """This rank's part of `u_k . x / (scale_k A_h)`."""
        if not self.values.size:
            return 0.0
        return float(self.values @ np.asarray(x_local)[self.dofs])

    def scatter_into(self, target, scale):
        """Add `scale * u_k / (scale_k A_h)` into the local array `target`."""
        if self.values.size:
            target[self.dofs] += scale * self.values


def taped_mode_coefficients(solution, boundaries, mode_rows, dofs, comm):
    """The trace coefficients of `solution`, `{bc_id: {key: AdjFloat}}`.

    Each value is recorded as a `TraceCoefficientBlock` when annotation is
    on, so it keeps its dependency on the solution. Off the tape it is the
    same number with no block added, so a forward-only run pays nothing.

    Args:
      solution: The Function the coefficients are read from.
      boundaries: The `(bc_id, ...)` pairs of the DtN boundaries.
      mode_rows: One `BoundaryModeRows` per boundary.
      dofs: One array per boundary, the local indices of its degrees of
        freedom in the local vector of `solution` (see `ModeRow`).
      comm: The communicator of the sums.

    Raises:
      RuntimeError: If the three sequences differ in length: they are paired
        by position and a `zip` would hide the mismatch.
    """
    boundaries, mode_rows, dofs = list(boundaries), list(mode_rows), list(dofs)
    if not len(boundaries) == len(mode_rows) == len(dofs):
        raise RuntimeError(
            f"{len(boundaries)} DtN boundaries against {len(mode_rows)} "
            f"mode-row sets and {len(dofs)} row-index sets; these are paired "
            "by position and a zip would hide the mismatch.")
    local = TraceCoefficientBlock._local(solution)
    tape = get_working_tape() if annotate_tape() else None
    out = {}
    for (bc_id, _), rows, indices in zip(boundaries, mode_rows, dofs):
        out[bc_id] = {}
        for index, key in enumerate(rows.keys):
            row = ModeRow(rows, index, indices, comm)
            value = AdjFloat(comm.allreduce(row.local_dot(local)))
            if tape is not None:
                tape.add_block(TraceCoefficientBlock(solution, row, value))
            out[bc_id][key] = value
    return out


def _accumulate(total, contribution):
    """`total + contribution`, kept a `Cofunction` rather than a lazy sum.

    Adding two `Cofunction`s in Firedrake yields a `FormSum`, which defers the
    addition instead of performing it, and a `FormSum` is neither something
    `_as_cofunction` can read nor something pyadjoint will accept as a
    derivative value. `assemble` collapses it.

    This matters only when **two or more dependencies are perturbed at once**,
    which is why it went unnoticed: every earlier test perturbed one control.
    The 2-D monopole datum makes it ordinary, since a perturbation of `rho`
    reaches the right-hand side twice - directly, and through the enclosed
    mass - so the tangent now has two contributions for a single control.
    """
    if total is None:
        return contribution
    return assemble(total + contribution)


def _as_cofunction(value, function_space):
    """A dual-space object holding `value`, whatever pyadjoint handed us."""
    out = Cofunction(function_space.dual())
    if isinstance(value, Cofunction):
        out.dat.data_wo[:] = value.dat.data_ro
        return out
    if isinstance(value, Function):
        out.dat.data_wo[:] = value.dat.data_ro
    else:
        out.dat.data_wo[:] = np.asarray(value, dtype=float)
    return out


def annotate_lowrank_solve(solver):
    """Tape one low-rank solve, if annotation is on."""
    if not annotate_tape():
        return
    dependencies = _rhs_dependencies(solver)
    solution = solver.mixed_solution
    block = LowRankGravitySolveBlock(solver, solution, dependencies)
    get_working_tape().add_block(block)
    block.get_outputs()[0].save_output()


def taped_trace_coefficients(solver):
    """`{bc_id: {key: AdjFloat}}` of a low-rank `GravitySolver`, differentiable.

    The potential is the whole solution there, so each row indexes the
    potential space directly (`taped_mode_coefficients`).
    """
    return taped_mode_coefficients(
        solver.mixed_solution, solver.dtn_boundaries, solver.mode_rows,
        [rows.dofs for rows in solver.mode_rows], solver.mesh.comm)


def _rhs_dependencies(solver):
    """The right-hand side's differentiable coefficients, in a stable order."""
    from ufl.algorithms.analysis import extract_coefficients

    seen, out = set(), []
    for coefficient in extract_coefficients(solver.rhs_form):
        if id(coefficient) in seen:
            continue
        seen.add(id(coefficient))
        if hasattr(coefficient, "block_variable"):
            out.append(coefficient)
    return out
