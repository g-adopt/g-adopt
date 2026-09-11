r"""The DtN low-rank update inside a COUPLED mixed space.

`gadopt.dtn_lowrank` builds `B = sum_k w_k u_k u_k^T` for the scalar gravity
solver, where the potential is the whole unknown and the rows index the
potential space directly. The self-gravitating GIA solver puts the potential in
a mixed space beside the displacement, the internal variables and the `Real`
fields, so the same `B` has to be applied to a monolithic mixed vector and
carries the potential row's scaling.

This module holds only what that costs: the index shift onto the mixed vector,
and the operator that uses it. The mode rows themselves, the elimination and
the Dirichlet masking are unchanged and come from `gadopt.dtn_lowrank`.

## The two things that are different, and both have been paid for

**The index shift.** `build_boundary_mode_rows` returns owned local indices
into the potential space. The coupled residual and Jacobian act on the mixed
vector, so every index moves by the owned sizes of the fields in front of the
potential. Those sizes come from the field index sets and never from
`subfunctions[i].dat`; see `psi_local_offset`.

**The prefactor.** `SelfGravitatingGIASolver` multiplies the whole potential
row - the DtN constraint and feedback rows included - by
`theta_psi = scaling_factor * B_mu / Lambda`, so it multiplies `B`. It is held
as a callable and read at every application, never folded into the weights,
because all three of its factors can be adjoint controls and a frozen product
cannot be corrected at replay time.
"""

import numpy as np
from firedrake import Cofunction
from firedrake.matrix_free.operators import ImplicitMatrixContext
from pyadjoint import Block, AdjFloat, annotate_tape, get_working_tape, no_annotations

__all__ = ["AugmentedImplicitMatrixContext", "CoupledLowRankDtN",
           "CoupledTraceCoefficientBlock", "install_augmented_context",
           "monolithic_rows", "psi_local_offset",
           "taped_coupled_trace_coefficients"]


def psi_local_offset(mixed_space, psi_field: int) -> int:
    """Local offset of field `psi_field` inside `Function(Z).dat.vec`.

    The owned sizes of the preceding fields, taken from the field index sets.

    **Do NOT sum `Z.subfunctions[i].dat` sizes.** A `Real` space reports `dat`
    size 1 on every rank but owns 0 entries of the monolithic vector on every
    rank above 0, so an offset built that way drifts by one per preceding
    `Real` field. Measured on 2, 3 and 4 ranks and cross-mesh
    (`NOTES/fastdtn/REVIEW-FORWARD.md` section 3.2): with the layout
    `(u, psi, R, R, R)` on 4 ranks the true offsets on rank 1 are
    `[0, 24, 36, 36, 36]`, the field index sets agree, and the `dat` recipe
    returns `[0, 24, 36, 37, 38]`.

    With the `Real` fields **after** psi the wrong recipe returns the right psi
    offset by accident, so a test of psi alone cannot see the defect. It is
    already wrong today for the rotation `Real` fields.

    A cross-mesh configuration produces a rank on which the mechanics `Submesh`
    owns no displacement dofs at all, so the offset is 0 there. That is legal
    and this function returns 0, rather than treating 0 as "not found".
    """
    ises = mixed_space.dof_dset.field_ises
    return int(sum(ises[i].getLocalSize() for i in range(psi_field)))


def monolithic_rows(mixed_space, psi_field: int, dofs) -> np.ndarray:
    """`dofs` from `build_boundary_mode_rows` as rows of the monolithic vector.

    `build_boundary_mode_rows` returns owned local indices into the potential
    space, with no ghosts. The coupled residual and Jacobian act on the mixed
    vector, so every index must be shifted by the owned sizes of the fields in
    front of psi.
    """
    return psi_local_offset(mixed_space, psi_field) + np.asarray(
        dofs, dtype=np.int64)


class CoupledLowRankDtN:
    r"""`B = theta_psi * B0` on the psi rows of a monolithic mixed vector.

    `B0 = sum_k w_k u_k u_k^T` is the DtN feedback the multiplier
    representation obtains through its Schur complement, eliminated by hand.
    One application is, per boundary,

        y[psi] += theta * C^T (W (C x[psi]))

    two small dense products against the boundary entries with one `Allreduce`
    of the `k`-vector in between, exactly as `LowRankDtNOperator` does it for
    the gravity-alone solver. The only differences here are the index shift
    onto the mixed vector and the `theta_psi` prefactor.

    **`theta_psi` and `B0` stay separately addressable, deliberately.** The
    prefactor is `scaling_factor * B_mu / Lambda`, all three of which can be
    adjoint controls, and it is read through a callable at every application
    rather than folded into `weights` once. A frozen product cannot be
    corrected at replay time, which is the failure the adjoint design exists to
    avoid; and `apply_local(..., theta=1.0)` gives `B0` alone, which is what a
    derivative with respect to one of those three needs.

    Attributes:
      mode_rows: One `BoundaryModeRows` per DtN boundary, positionally aligned
        with `DtNGravityForm.dtn_boundaries`.
      rows_mono: The same rows' `dofs`, shifted onto the monolithic vector.
      applications: Counts applications, so a test can confirm the operator was
        exercised rather than bypassed.
    """

    def __init__(self, mixed_space, psi_field, mode_rows, theta, comm):
        self.mixed_space = mixed_space
        self.psi_field = psi_field
        self.mode_rows = list(mode_rows)
        self.psi_offset = psi_local_offset(mixed_space, psi_field)
        self.rows_mono = [monolithic_rows(mixed_space, psi_field, rows.dofs)
                          for rows in self.mode_rows]
        #: Read at every application, never at construction. `theta_psi` is a
        #: UFL expression over `Constant`s and any of them can be a control.
        self.theta = theta
        self.comm = comm
        self.applications = 0
        #: D3: applications made through the augmented Jacobian
        #: action specifically. `applications` counts residual and
        #: Jacobian alike, so it cannot distinguish an
        #: augmentation that was installed from one that ran.
        self.jacobian_applications = 0

    @property
    def theta_value(self) -> float:
        """The prefactor now. A callable is called; anything else is cast."""
        theta = self.theta
        return float(theta() if callable(theta) else theta)

    @property
    def n_modes(self) -> int:
        return sum(len(rows.keys) for rows in self.mode_rows)

    def coefficients(self, x_local):
        """Trace coefficients `c_k` per boundary, summed across ranks.

        `theta_psi` does not appear: it multiplies the constraint row and the
        feedback row alike and cancels out of `c_k`.
        """
        out = []
        for rows, mono in zip(self.mode_rows, self.rows_mono):
            if rows.rows.size:
                local = rows.rows @ np.asarray(x_local)[mono] * rows.recovery
            else:
                local = np.zeros(len(rows.keys))
            total = np.empty_like(local)
            self.comm.Allreduce(local, total)
            out.append(total)
        return out

    def apply_local(self, x_local, y_local, theta=None) -> None:
        """`y[psi] += theta * B0 x[psi]`, in place, on local arrays.

        Args:
          x_local: The owned entries of the monolithic vector, read only.
          y_local: The owned entries of the target, written in place.
          theta: Override the prefactor. Pass `1.0` for `B0` alone.
        """
        scale = self.theta_value if theta is None else float(theta)
        for rows, mono in zip(self.mode_rows, self.rows_mono):
            if rows.rows.size:
                local = rows.rows @ np.asarray(x_local)[mono]
            else:
                local = np.zeros(len(rows.keys))
            total = np.empty_like(local)
            # The modes are global and the boundary dofs are partitioned, so
            # the `k`-vector must be summed before it is scattered back. This
            # is the whole of the parallel communication.
            self.comm.Allreduce(local, total)
            if rows.rows.size:
                y_local[mono] += scale * (rows.rows.T @ (rows.weights * total))
        self.applications += 1

    def mult(self, x_vec, y_vec, theta=None) -> None:
        """`y = B x` on PETSc vectors. `y` is zeroed first.

        `array_r` and `array_w` rather than the context manager, to match
        `LowRankDtNOperator._add_low_rank`: `array_w` hands back the same local
        memory and marks it modified, so the `+=` below accumulates onto the
        zeros just written rather than onto whatever PETSc last left there.
        """
        y_vec.set(0.0)
        self.apply_local(x_vec.array_r, y_vec.array_w, theta=theta)

    def add_mult(self, x_vec, y_vec, theta=None) -> None:
        """`y += B x` on PETSc vectors, leaving whatever `y` already held."""
        self.apply_local(x_vec.array_r, y_vec.array_w, theta=theta)


class AugmentedImplicitMatrixContext(ImplicitMatrixContext):
    """`ImplicitMatrixContext` whose action carries the low-rank DtN update.

    Mechanism M1 of `NOTES/fastdtn/REVIEW-FORWARD.md` section 2.2: subclass the
    matrix-free context and install it with `Mat.setPythonContext` from a
    `post_jacobian_callback`. Two properties make this the right mechanism.

    **`createSubMatrix` is left alone, and that is what keeps `B` out of
    `Pmat`.** `firedrake/matrix_free/operators.py:435` builds sub-blocks as
    `ImplicitMatrixContext(asub, ...)` and not `type(self)(asub, ...)`, so a
    subclass never propagates into a fieldsplit block and GAMG never sees the
    dense update. Nobody chose that; it is guard G1's whole subject.

    **The bcs are already handled by the parent.** `super().mult` zeroes the
    constrained entries of the input and writes the input back on the
    constrained rows of the output. `B`'s rows and columns at those degrees of
    freedom were zeroed by `apply_dirichlet_to_rows`, so adding it afterwards
    cannot disturb them.

    `B` is symmetric, so the transpose action is the same action. Stated rather
    than inherited, because PETSc uses it for a transposed solve and getting it
    wrong would surface only in the adjoint.
    """

    #: Set by the installer. Not a constructor argument, because
    #: `createSubMatrix` builds the parent class and would not pass it on.
    dtn_operator = None

    def mult(self, mat, X, Y):
        super().mult(mat, X, Y)
        if self.dtn_operator is not None:
            self.dtn_operator.apply_local(X.array_r, Y.array_w)
            # D3. Counted on the OPERATOR, which is shared, rather than on this
            # context, which is one per matrix. "Was it installed" and "did it
            # run" are different questions and only the second one matters: a
            # guard that fails to install leaves this at 0 while every reason
            # code still reads converged.
            self.dtn_operator.jacobian_applications += 1

    def multTranspose(self, mat, X, Y):
        super().multTranspose(mat, X, Y)
        if self.dtn_operator is not None:
            self.dtn_operator.apply_local(X.array_r, Y.array_w)
            self.dtn_operator.jacobian_applications += 1


def install_augmented_context(Jmat, dtn_operator):
    """Swap `Jmat`'s python context for one that adds `B`. Idempotent.

    **The guard is the context's TYPE, not the matrix's identity, and the
    difference is a silent wrong answer.** The obvious key is `Jmat.handle`
    against a set of handles already done. It is unsound: PETSc reuses freed
    `Mat` addresses immediately, and forty build/destroy cycles were measured
    producing **one distinct handle across all forty matrices** - 39 repeats of
    an earlier address. A genuinely new Jacobian whose handle reuses a freed one
    is then skipped, so the augmentation sits in the residual and is absent from
    the Jacobian, which under `ksponly` converges with a clean reason and the
    wrong answer.

    That defect hides from the test anyone would write first. In a tape /
    replay / adjoint sweep the matrices coexist, so their addresses really are
    distinct and the handle key looks correct. It fires where matrices are
    destroyed between installs - a solver rebuilt across timesteps, which is the
    GIA production pattern.

    A boolean flag is worse again: pyadjoint builds a different solver object
    for the replay from the same kwargs, so one closure is shared across several
    matrices and a flag set while taping makes every later one skip
    (`NOTES/fastdtn/REVIEW-FORWARD.md` section 7.1, `DIVERGED_DTOL`).

    Asking the object what it is carries no bookkeeping at all: there is no set
    to go stale and no key to get wrong.

    Args:
      Jmat: the PETSc `Mat` handed to `post_jacobian_callback`.
      dtn_operator: the `CoupledLowRankDtN` to add.

    Returns:
      Whether an install happened on this call.
    """
    ctx = Jmat.getPythonContext()
    if isinstance(ctx, AugmentedImplicitMatrixContext):
        return False
    new = AugmentedImplicitMatrixContext(
        ctx.a, row_bcs=ctx.row_bcs, col_bcs=ctx.col_bcs,
        fc_params=ctx.fc_params, appctx=ctx.appctx)
    new.on_diag = ctx.on_diag
    new.dtn_operator = dtn_operator
    Jmat.setPythonContext(new)
    return True


class _CoupledModeRow:
    """One mode's boundary functional, indexed into the MONOLITHIC mixed vector.

    The scalar solver's `_SingleModeRow` indexes the potential space, which is
    the whole unknown there. Here the potential is one field of a mixed space,
    so the same values sit at `monolithic_rows` instead.

    **The recovery weight comes from `rows.recovery`, the same array the
    forward value uses**, so the `1 / (scale_k * A_h)` denominator exists in
    exactly one place and cannot drift between the value and its derivative.
    That is the detail `gadopt.dtn_adjoint` insists on and it is inherited
    here rather than re-derived.
    """

    def __init__(self, rows, index, mono, comm):
        self.values = rows.rows[index] * rows.recovery[index]
        self.dofs = mono
        self.comm = comm

    def local_dot(self, x_local):
        if not self.values.size:
            return 0.0
        return float(self.values @ np.asarray(x_local)[self.dofs])

    def scatter_into(self, target, scale):
        if self.values.size:
            target[self.dofs] += scale * self.values


class CoupledTraceCoefficientBlock(Block):
    """One trace coefficient `c_k = u_k . psi / (scale_k A_h)`, on the tape.

    The coupled twin of `gadopt.dtn_adjoint.TraceCoefficientBlock`. The maths is
    that module's and is not re-derived: `c_k` is a linear functional of the
    potential with a constant vector, so the adjoint of a scalar seed `c_bar` is
    `c_bar * u_k / (scale_k A_h)` as a cofunction, and the Hessian is the same
    map applied to the second-order seed because the functional is linear.

    **Why this exists at all.** Reading the trace off `dat.data_ro` and
    returning `float()` severs the tape by construction - the same sever
    `test_coefficient_float_severs_tape` documents as a defect on the
    multiplier path. `geoid()` depends on `coefficients()` and B5 reports
    `N(0)` and `N(180)` through it, so on the low-rank path the method would
    return correct values with a gradient of exactly zero: a different silent
    failure on the same method, one layer down from the truncating `zip`.

    The only difference from the scalar version is the indexing. The dependency
    is the **mixed** solution, the cofunction lives in the mixed dual space, and
    the rows are monolithic, so `dat.data_ro` cannot be used - a mixed `Dat`
    returns a tuple of arrays. The monolithic local vector is taken through
    `dat.vec_ro`.
    """

    def __init__(self, solution, row, value):
        super().__init__()
        self.function_space = solution.function_space()
        self.row = row
        self.add_dependency(solution)
        self.add_output(value.create_block_variable())

    def __str__(self):
        return "CoupledTraceCoefficientBlock"

    @staticmethod
    def _local(function):
        with function.dat.vec_ro as vec:
            return np.array(vec.array_r, dtype=float)

    @no_annotations
    def recompute_component(self, inputs, block_variable, idx, prepared):
        local = self.row.local_dot(self._local(inputs[0]))
        return self.row.comm.allreduce(local)

    def _seeded_cofunction(self, seed):
        out = Cofunction(self.function_space.dual())
        with out.dat.vec as vec:
            self.row.scatter_into(vec.array_w, float(seed))
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        seed = adj_inputs[0]
        return None if seed is None else self._seeded_cofunction(seed)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        perturbation = tlm_inputs[0]
        if perturbation is None:
            return None
        return self.row.comm.allreduce(
            self.row.local_dot(self._local(perturbation)))

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        seed = hessian_inputs[0]
        return None if seed is None else self._seeded_cofunction(seed)


def taped_coupled_trace_coefficients(solver):
    """`{bc_id: {key: AdjFloat}}` for the coupled low-rank path, differentiable.

    The port of `gadopt.dtn_adjoint.taped_trace_coefficients`. Off the tape -
    annotation disabled - it returns the same numbers with no blocks added, so
    a forward-only run pays nothing.
    """
    operator = solver.dtn_operator
    local = CoupledTraceCoefficientBlock._local(solver.solution)
    tape = get_working_tape() if annotate_tape() else None
    boundaries = solver.form.dtn_boundaries
    if not (len(boundaries) == len(operator.mode_rows)
            == len(operator.rows_mono)):
        raise RuntimeError(
            f"{len(boundaries)} DtN boundaries against "
            f"{len(operator.mode_rows)} mode-row sets and "
            f"{len(operator.rows_mono)} row-index sets; these are paired by "
            "position and a zip would hide the mismatch.")
    out = {}
    for (bc_id, _), rows, mono in zip(boundaries, operator.mode_rows,
                                      operator.rows_mono):
        out[bc_id] = {}
        for index, key in enumerate(rows.keys):
            row = _CoupledModeRow(rows, index, mono, operator.comm)
            value = AdjFloat(operator.comm.allreduce(row.local_dot(local)))
            if tape is not None:
                tape.add_block(
                    CoupledTraceCoefficientBlock(solver.solution, row, value))
            out[bc_id][key] = value
    return out
