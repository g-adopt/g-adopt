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

__all__ = ["LowRankGravitySolveBlock", "TraceCoefficientBlock",
           "annotate_lowrank_solve", "taped_trace_coefficients"]


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
            owned = self.function_space.dof_dset.size
            for bc in self.solver.strong_bcs:
                nodes = bc.nodes[bc.nodes < owned]
                rhs_function.dat.data_wo[nodes] = 0.0
        with rhs_function.dat.vec_ro as b, solution.dat.vec as x:
            self.solver.ksp.solve(b, x)
        return solution

    def _rhs_derivative_transpose(self, coefficient, multiplier):
        """`(db/dm)^T multiplier`, as a value in the dual of `m`'s space."""
        form = derivative(getattr(self, "_form", None) or self.rhs_form,
                          self._in_form(coefficient))
        return assemble(action(adjoint(form), multiplier))


class TraceCoefficientBlock(Block):
    """One trace coefficient `c_k = u_k . psi / (scale_k A_h)`.

    A linear functional of the potential with a constant vector, so the
    derivative is that same vector: the adjoint of a scalar seed `c_bar` is
    `c_bar * u_k / (scale_k A_h)` as a cofunction on the potential space.
    """

    def __init__(self, solution, row, value):
        super().__init__()
        self.function_space = solution.function_space()
        self.row = row
        self.add_dependency(solution)
        self.add_output(value.create_block_variable())

    def __str__(self):
        return "TraceCoefficientBlock"

    @no_annotations
    def recompute_component(self, inputs, block_variable, idx, prepared):
        psi = np.asarray(inputs[0].dat.data_ro, dtype=float)
        local = float(self.row.local_dot(psi))
        return self.row.comm.allreduce(local)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        seed = adj_inputs[0]
        if seed is None:
            return None
        out = Cofunction(self.function_space.dual())
        self.row.scatter_into(out.dat.data_wo, float(seed))
        return out

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        perturbation = tlm_inputs[0]
        if perturbation is None:
            return None
        psi = np.asarray(perturbation.dat.data_ro, dtype=float)
        return self.row.comm.allreduce(float(self.row.local_dot(psi)))

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear in psi, so the second derivative is the first applied to the
        # second-order seed.
        seed = hessian_inputs[0]
        if seed is None:
            return None
        out = Cofunction(self.function_space.dual())
        self.row.scatter_into(out.dat.data_wo, float(seed))
        return out


class _SingleModeRow:
    """One mode's boundary functional, scaled for coefficient recovery."""

    def __init__(self, rows, index, comm):
        self.values = rows.rows[index] * rows.recovery[index]
        self.dofs = rows.dofs
        self.comm = comm

    def local_dot(self, psi_local):
        if not self.values.size:
            return 0.0
        return self.values @ psi_local[self.dofs]

    def scatter_into(self, target, scale):
        if self.values.size:
            target[self.dofs] += scale * self.values


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
    """`{bc_id: {key: AdjFloat}}`, differentiable when annotation is on.

    Reading the coefficients off `psi.dat.data_ro` and returning `float()`
    severs the tape by construction - it is the same sever that
    `test_coefficient_float_severs_tape` documents as a defect on the
    multiplier path. The trace spectrum *is* the geoid, which is the observable
    an inversion of this solver most often targets, so on a path justified by
    inversions it has to be differentiable.

    The recovery weight comes from `rows.recovery`, the same array the forward
    value uses, so the `1/(scale_k * A_h)` denominator exists in exactly one
    place and cannot drift between the value and its derivative.
    """
    out = {}
    comm = solver.mesh.comm
    psi = solver.mixed_solution
    psi_local = np.asarray(psi.dat.data_ro, dtype=float)
    tape = get_working_tape() if annotate_tape() else None
    for (bc_id, _), rows in zip(solver.dtn_boundaries, solver.mode_rows):
        out[bc_id] = {}
        for index, key in enumerate(rows.keys):
            row = _SingleModeRow(rows, index, comm)
            value = AdjFloat(comm.allreduce(float(row.local_dot(psi_local))))
            if tape is not None:
                tape.add_block(TraceCoefficientBlock(psi, row, value))
            out[bc_id][key] = value
    return out


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
