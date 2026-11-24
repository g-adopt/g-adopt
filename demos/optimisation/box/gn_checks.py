"""Correctness checks of the Gauss-Newton product, shared by the box and the cylinder.

Each function here runs one check of ``NOTES/team/gauss-newton/PLAN.md`` section 2.2
on a problem that the caller has built, appends its measured values to a results
dictionary, and returns True when the check passes. The thresholds are arguments with
the plan's values as defaults, so that a script that changes one has to say so at the
call site.

The threshold rule of the plan: where a solver is on the path (a projection solves
with CG at ``ksp_rtol 1e-8``, and the box ``T0`` projection is on the path of both
regularisation terms), a check passes below 1e-7. Pure arithmetic and bitwise checks
keep their own thresholds.

``<w, y>`` of a primal Function ``w`` and a dual Cofunction ``y`` is ``y(w)``,
assembled, so it is correct in parallel.

The callers are ``prior_campaign/check_gauss_newton.py`` (box) and
``cylindrical/checks.py`` (``gn`` and ``gn-curvature``).
"""

import resource
import sys
import time

import numpy as np
from mpi4py import MPI
from firedrake import Cofunction, Function, TestFunction, assemble
from petsc4py import PETSc
from pyadjoint import MinimizationProblem, stop_annotating

from gauss_newton import GaussNewtonHessianMat, GaussNewtonOperator


# ---------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------

def say(text, comm=MPI.COMM_WORLD):
    """Print on rank 0 and flush, so that a job log shows progress at once."""
    if comm.rank == 0:
        print(text, flush=True)


def verdict(results, check_id, name, value, threshold, mode="below"):
    """Record one measured value against its threshold, print it, and return the pass.

    Args:
        results: the results dictionary; the entry goes into ``results["tests"]``.
        check_id: the id of the plan, for example ``"G5"``.
        name: what was measured.
        value: the measured number (or a bool for ``mode="true"``).
        threshold: the bound.
        mode: ``"below"`` passes for ``value <= threshold``, ``"above"`` for
            ``value > threshold``, ``"equal"`` for ``value == threshold`` and
            ``"true"`` for a true ``value``.

    Returns:
        bool
    """
    if mode == "below":
        ok = bool(value <= threshold)
    elif mode == "above":
        ok = bool(value > threshold)
    elif mode == "equal":
        ok = bool(value == threshold)
    elif mode == "true":
        ok = bool(value)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    entry = {"id": check_id, "name": name, "value": value, "threshold": threshold,
             "mode": mode, "pass": ok}
    results.setdefault("tests", []).append(entry)
    shown = f"{value:.3e}" if isinstance(value, float) else str(value)
    say(f"  {'PASS' if ok else 'FAIL'}  {check_id} {name}: {shown} "
        f"({mode} {threshold})")
    return ok


def pairing(dual, primal):
    """``<primal, dual>``: the action of a Cofunction on a Function, a number."""
    return float(assemble(dual(primal)))


def global_max_abs(array, comm=MPI.COMM_WORLD):
    """``max |a|`` over the owned entries of every rank."""
    local = float(np.max(np.abs(array))) if array.size else 0.0
    return comm.allreduce(local, op=MPI.MAX)


def max_relative(a, b, comm=MPI.COMM_WORLD):
    """``max |a - b| / max |b|`` of two Functions or Cofunctions of one space."""
    da = np.asarray(a.dat.data_ro, dtype=float)
    db = np.asarray(b.dat.data_ro, dtype=float)
    scale = global_max_abs(db, comm)
    diff = global_max_abs(da - db, comm)
    return diff / scale if scale > 0 else diff


def bitwise_equal(a, b, comm=MPI.COMM_WORLD):
    """True when two arrays of owned entries are equal on every rank."""
    local = bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return comm.allreduce(local, op=MPI.LAND)


def peak_rss_gb():
    """Peak resident memory of this process in GB (``ru_maxrss`` is bytes on macOS)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss * (1e-9 if sys.platform == "darwin" else 1e-6)


def cofunction_copy(c):
    """An independent copy of a Cofunction."""
    out = Cofunction(c.function_space())
    out.assign(c)
    return out


def full_hessian(rf, v):
    """pyadjoint's Hessian action ``H v`` at the tape point of the last gradient."""
    return cofunction_copy(rf.hessian(v, apply_riesz=False))


# ---------------------------------------------------------------------------------
# G1: term discovery and weights
# ---------------------------------------------------------------------------------

def check_terms(operator, expected, results, check_id="G1", tolerance=1e-12):
    """The terms found on the tape against independently computed weights.

    Args:
        operator: the ``GaussNewtonOperator``.
        expected: list of ``(description, weight)`` in tape order, one per term,
            computed by the caller from the data and the weights of the problem and
            not from the tape.
        results: the results dictionary.
        check_id: the id under which the values are stored.
        tolerance: relative threshold of each weight; pure arithmetic, so 1e-12.

    Returns:
        bool
    """
    terms = operator.terms
    results[f"{check_id}_terms"] = [{"label": t.label, "weight": t.weight}
                                    for t in terms]
    ok = verdict(results, check_id, "number of terms", len(terms), len(expected),
                 mode="equal")
    if not ok:
        return False
    worst = 0.0
    for term, (description, weight) in zip(terms, expected):
        error = abs(term.weight - weight) / abs(weight)
        say(f"    {term.label}: weight {term.weight:.15e}, expected {weight:.15e} "
            f"({description}), relative difference {error:.2e}")
        worst = max(worst, error)
    ok &= verdict(results, check_id, "largest relative weight difference", worst,
                  tolerance)
    ok &= verdict(results, check_id, "linear combination defect |sum w f - J| / |J|",
                  operator.linear_combination_defect(), tolerance)
    return ok


# ---------------------------------------------------------------------------------
# G2: first-variation seeds against the gradient
# ---------------------------------------------------------------------------------

def check_first_variation(operator, inner_product, results, tolerance=1e-7):
    """First-variation seeds plus the Gauss-Newton adjoint sweep against ``rf.derivative()``.

    Returns:
        bool
    """
    rf = operator.rf
    gradient = cofunction_copy(rf.derivative(apply_riesz=False))
    seeded = operator.first_variation_gradient()
    rel_max = max_relative(seeded, gradient)
    difference = cofunction_copy(seeded)
    difference.assign(seeded - gradient)
    rel_dual = float(inner_product.dual_norm(difference) / inner_product.dual_norm(gradient))
    ok = verdict(results, "G2", "max |a - b| / max |b| (seeded sweep against gradient)",
                 rel_max, tolerance)
    ok &= verdict(results, "G2", "dual norm of a - b over dual norm of b", rel_dual,
                  tolerance)
    return ok


# ---------------------------------------------------------------------------------
# G4 to G7 at the original data
# ---------------------------------------------------------------------------------

def check_products(operator, metric_context, v, w, results, has_regularisation,
                   regularisation_terms=None, tolerance=1e-7, nonempty=1e-4,
                   fixed_nodes=None):
    """G4 (the comparison is not empty), G5 (symmetry), G6 (positivity) and G7.

    The tape must hold a replay and a gradient at the point of the check, at the
    original data, so that the residuals are not zero.

    Args:
        operator: the ``GaussNewtonOperator``.
        metric_context: the ``riesz_hessian.InnerProductMat`` of the problem, whose
            ``mult`` applies the metric ``A`` (primal to dual, in PETSc vectors).
        v, w: the two directions, Functions in the control space.
        results: the results dictionary.
        has_regularisation: whether the objective holds the regularisation terms.
            Then ``H_GN >= A`` must hold, else ``H_GN >= 0``.
        regularisation_terms: the terms whose Gauss-Newton product must equal
            ``A v`` (G6), or None.
        tolerance: the threshold where a solver is on the path.
        nonempty: G4 passes above this value.
        fixed_nodes: local indices of control nodes that BNK holds in its fixed set
            (the pinned wall nodes of the box), or None. The comparison of
            ``H_GN,reg v`` with ``A v`` leaves them out. On the box the
            regularisation acts on ``T0``, the projection of the control with the
            wall values imposed, so its Hessian is ``P^T A P`` with ``P`` the
            derivative of that projection. ``P^T`` gives the wall rows the coupling
            ``M_cf M_ff^{-1}`` of the mass matrix, while the Dirichlet metric has
            zero wall rows. The free rows agree, and only the free rows enter the
            Krylov solve of a run with pinned walls. Both values are recorded.

    Returns:
        bool
    """
    rf = operator.rf
    ok = True
    # pyadjoint's full product needs the adjoint state of a gradient at this point.
    rf.derivative(apply_riesz=False)
    Hv = full_hessian(rf, v)
    Gv = operator.action(v)
    Gw = operator.action(w)

    # G4: the full product and the Gauss-Newton product differ at the original data,
    # so the zero-residual comparison of G3 is not passed by an operator that returns
    # H v.
    ok &= verdict(results, "G4", "max |H v - H_GN v| / max |H v| at the original data",
                  max_relative(Gv, Hv), nonempty, mode="above")

    # G5: symmetry.
    wGv, vGw = pairing(Gv, w), pairing(Gw, v)
    results["G5_values"] = {"<w,H_GN v>": wGv, "<v,H_GN w>": vGw}
    ok &= verdict(results, "G5", "|<w,H_GN v> - <v,H_GN w>| / |<w,H_GN v>|",
                  abs(wGv - vGw) / abs(wGv), tolerance)

    # G6: positivity, against A when the regularisation is in the objective.
    vGv = pairing(Gv, v)
    Av = metric_action(metric_context, v)
    vAv = pairing(Av, v)
    vHv = pairing(Hv, v)
    results["G6_values"] = {"<v,H_GN v>": vGv, "<v,A v>": vAv, "<v,H v>": vHv}
    if has_regularisation:
        ok &= verdict(results, "G6", "(<v,H_GN v> - <v,A v>) / <v,A v>, must be >= -tol",
                      -(vGv - vAv) / vAv, tolerance)
    else:
        ok &= verdict(results, "G6", "<v,H_GN v> / |<v,H_GN v>|, must be >= -tol",
                      -vGv / abs(vGv), tolerance)
    if regularisation_terms:
        Rv = operator.action(v, terms=regularisation_terms)
        every = max_relative(Rv, Av)
        results["G6_reg_against_A_all_nodes"] = every
        if fixed_nodes is None or len(fixed_nodes) == 0:
            ok &= verdict(results, "G6", "max |H_GN,reg v - A v| / max |A v|", every,
                          tolerance)
        else:
            free = np.ones(len(Av.dat.data_ro), dtype=bool)
            free[np.asarray(fixed_nodes, dtype=int)] = False
            a = np.asarray(Rv.dat.data_ro, dtype=float)[free]
            b = np.asarray(Av.dat.data_ro, dtype=float)[free]
            value = global_max_abs(a - b) / global_max_abs(b)
            say(f"    info  H_GN,reg v against A v on every node (wall rows "
                f"included): {every:.3e}")
            ok &= verdict(results, "G6", "max |H_GN,reg v - A v| / max |A v| on the "
                          "free nodes", value, tolerance)

    # G7: the adjoint half against two tangent-linear sweeps.
    ds_v = [ds.copy(deepcopy=True) for ds in operator.tlm_states(v)]
    ds_w = [ds.copy(deepcopy=True) for ds in operator.tlm_states(w)]
    operator.tape.reset_tlm_values()
    curvature = float(sum(operator.term_curvatures(ds_v, ds_w)))
    results["G7_values"] = {"<w,H_GN v>": wGv, "sum_k w_k f_k''[ds_k(v), ds_k(w)]": curvature}
    ok &= verdict(results, "G7", "|<w,H_GN v> - sum_k w_k f_k''[ds_k(v),ds_k(w)]| / |..|",
                  abs(wGv - curvature) / abs(curvature), tolerance)
    return ok


def metric_action(metric_context, v):
    """``A v`` as a Cofunction, through ``InnerProductMat.mult``."""
    interface = metric_context.vec_interface
    x = interface.new_petsc()
    y = interface.new_petsc()
    interface.to_petsc(x, v)
    metric_context.mult(None, x, y)
    out = Cofunction(v.function_space().dual())
    interface.from_petsc(y, out)
    return out


# ---------------------------------------------------------------------------------
# G3: zero residual
# ---------------------------------------------------------------------------------

def check_zero_residual(operator, data_outputs, v, w, results, tolerance=1e-7,
                        residual_tolerance=1e-28):
    """Full product against Gauss-Newton product at data that the model fits exactly.

    The caller has already set every data leaf to the model value at the first guess
    and replayed (``zero_residual_box`` or ``zero_residual_cylinder``), and built the
    operator again after the replay, because the weights follow the data.

    The residual test bounds the contribution of each data term to J, ``w_k f_k``,
    in the units of J. The raw output ``f_k`` of a velocity term of the cylinder is
    an integral of a squared velocity, and the non-dimensional surface velocities are
    of order 1e3 (its weight ``norm_obs / norm_u_surface`` is about 1e-7), so a raw
    output at round-off level is about 1e-27 there, and an absolute bound on it
    would test the velocity scale instead of the zero residual. Both values are
    recorded.

    Args:
        operator: the ``GaussNewtonOperator`` built after the replay.
        data_outputs: list of ``(label, value)`` of every data term after the replay.
        v, w: the directions.
        results: the results dictionary.

    Returns:
        bool
    """
    rf = operator.rf
    weights = {term.label: term.weight for term in operator.terms}
    contributions = [weights[label] * value for label, value in data_outputs]
    results["G3_data_terms"] = [{"label": label, "value": value,
                                 "weight": weights[label],
                                 "contribution": weights[label] * value}
                                for label, value in data_outputs]
    results["G3_largest_raw_data_term"] = max(abs(value) for _, value in data_outputs)
    say(f"    info  largest raw data term output {results['G3_largest_raw_data_term']:.3e}")
    ok = verdict(results, "G3", "largest data term contribution w_k f_k after the change",
                 max(abs(c) for c in contributions), residual_tolerance)
    for name, direction in (("v", v), ("w", w)):
        # A Gauss-Newton product clears the adjoint state, so each full product
        # needs a new gradient.
        rf.derivative(apply_riesz=False)
        Hd = full_hessian(rf, direction)
        Gd = operator.action(direction)
        ok &= verdict(results, "G3", f"max |H {name} - H_GN {name}| / max |H {name}|",
                      max_relative(Gd, Hd), tolerance)
    return ok


def zero_residual_box(problem):
    """Set the box observation to the model value at the current tape point, replay.

    The temperature misfit reads ``Tobs``, a data Function with no producer block.
    Its block variable holds a checkpoint, which a replay reads instead of the
    Function. Both are set to ``T_F(m*)``, the final temperature of the last replay.
    The normalisations ``int Tobs^2`` and ``int |grad Tobs|^2`` are assemblies on
    the tape too, so the replay changes ``c_d`` and ``c_s``; the operator must be
    built again afterwards.

    Args:
        problem: the box ``InverseProblem``, replayed at the point of the check.

    Returns:
        list of ``(label, value)``: the misfit after the replay, with the label of
        its term in ``find_quadratic_terms``.
    """
    rf = problem.reduced_functional
    with stop_annotating():
        final = problem.T.block_variable.saved_output.copy(deepcopy=True)
        _set_leaf(problem.Tobs.block_variable, problem.Tobs, final)
        rf(problem.control.tape_value())
    misfit, label = _term_output_reading(rf, problem.Tobs)
    return [(label, misfit)]


def zero_residual_cylinder(problem, operator):
    """Set every cylindrical observation to the model value at the tape point, replay.

    The data terms of the cylinder and their data leaves:

    - the temperature misfit reads the observed temperature, a data Function in the
      state space with no producer block;
    - the velocity misfit of step ``n`` reads ``uobs``, which an assignment block
      wrote from a velocity Function loaded from the reference state. That loaded
      Function is the leaf; its new value is the velocity part of the mixed state
      ``z_n`` that the term reads.

    The replay recomputes every block, also the normalisations of the objective
    (``int Tobs^2``, ``int_top |uobs_{N-1}|^2`` and the two regularisation norms),
    so the weights follow the new data, and the operator must be built again.

    Args:
        problem: the ``CylindricalProblem``, replayed at the point of the check.
        operator: a ``GaussNewtonOperator`` of that tape, for its term list.

    Returns:
        ``(data_labels, list of (label, value))``: the labels of the data terms, and
        their outputs after the replay.
    """
    rf = problem.reduced_functional
    producer = {out: block for block in rf.tape.get_blocks() for out in block.get_outputs()}
    changes = []
    labels = []
    for term in operator.terms:
        state = term.state
        space = state.output.function_space()
        deps = [dep for dep in term.block.get_dependencies()
                if dep is not state and isinstance(dep.output, Function)]
        if len(space) > 1:
            # A velocity term: the mixed state of one step.
            uobs = [dep for dep in deps if dep.output.name() == "uobs"]
            if len(uobs) != 1:
                raise RuntimeError(f"velocity term {term.label} reads no 'uobs'")
            assign = producer[uobs[0]]
            leaves = [dep for dep in assign.get_dependencies()
                      if dep not in producer and isinstance(dep.output, Function)]
            if len(leaves) != 1:
                raise RuntimeError(f"the assignment of uobs for {term.label} has "
                                   f"{len(leaves)} data leaves")
            value = state.saved_output.subfunctions[0].copy(deepcopy=True)
            changes.append((leaves[0], value))
            labels.append(term.label)
        else:
            # The temperature misfit reads data in its own state space; the
            # regularisation terms read the layer average, which is not.
            data = [dep for dep in deps if dep not in producer
                    and dep.output.function_space() == space]
            if data:
                if len(data) != 1:
                    raise RuntimeError(f"term {term.label} reads {len(data)} data "
                                       f"Functions of its state space")
                changes.append((data[0], state.saved_output.copy(deepcopy=True)))
                labels.append(term.label)
    with stop_annotating():
        for bv, value in changes:
            _set_leaf(bv, bv.output, value)
        rf(problem.control.tape_value())
    outputs = [(term.label, float(term.output.saved_output)) for term in operator.terms
               if term.label in labels]
    return labels, outputs


def _set_leaf(bv, function, value):
    """Give a data leaf a new value: its checkpoint, and the Function itself."""
    function.assign(value)
    if bv.checkpoint is not None:
        bv.checkpoint = value.copy(deepcopy=True)


def _term_output_reading(rf, data_function):
    """The output of the assembly block that reads ``data_function``, after a replay."""
    from firedrake.adjoint_utils.blocks.assembly import AssembleBlock
    target = data_function.block_variable
    values = []
    for index, block in enumerate(rf.tape.get_blocks()):
        if isinstance(block, AssembleBlock) and target in block.get_dependencies():
            # Normalisations read the data alone; the term reads the state as well.
            if len(block.get_dependencies()) > 2:
                values.append((float(block.get_outputs()[0].saved_output), index))
    if len(values) != 1:
        raise RuntimeError(f"expected one term block that reads {data_function.name()}, "
                           f"found {len(values)}")
    value, index = values[0]
    # The label of find_quadratic_terms: tape index and state name. The state of the
    # box misfit is the temperature.
    return value, f"tape{index}:Temperature"


# ---------------------------------------------------------------------------------
# G8: the TAO path
# ---------------------------------------------------------------------------------

class StubTao:
    """The three methods of a TAO object that ``GaussNewtonHessianMat.update`` reads.

    A real TAO object cannot be told which iteration it is at or how its last Krylov
    solve ended, so the switch and the fall-back are tested with this stub.
    """

    class _KSP:
        def __init__(self, reason):
            self.reason = reason

        def getConvergedReason(self):
            return self.reason

    def __init__(self, iteration, gnorm=1.0, reason=0):
        self.iteration = iteration
        self.gnorm = gnorm
        self.ksp = self._KSP(reason)

    def getIterationNumber(self):
        return self.iteration

    def getSolutionStatus(self):
        return (self.iteration, 0.0, self.gnorm, 0.0, 0.0, 0)

    def getKSP(self):
        return self.ksp


def build_tao(problem, variant="tao-bnls-riesz"):
    """A TAO object built as ``run_tao`` builds it, with the zero shift.

    Returns:
        ``(solver, tao, pmat, pmat_context)``; the caller keeps them alive.
    """
    import optimisers
    from pyadjoint.optimization.tao_solver import TAOSolver
    from riesz_hessian import build_preconditioner_matrix

    params = optimisers.tao_parameters(variant, 1, 1.0,
                                       {"tao_bnk_imax": 0.0, "tao_bnk_ksp_max_it": 5})
    pmat, pmat_context = build_preconditioner_matrix(problem)
    solver = TAOSolver(MinimizationProblem(problem.reduced_functional,
                                           bounds=problem.bounds), params, Pmat=pmat)
    return solver, solver.tao, pmat, pmat_context


def check_tao_path(problem, operator, v, results):
    """G8: the matrix routes to the operator, and the switch and fall-back work.

    Returns:
        bool
    """
    solver, tao, pmat, pmat_context = build_tao(problem)
    hessian_mat, P = tao.getHessian()[:2]
    interface = pmat_context.vec_interface
    x = interface.new_petsc()
    interface.to_petsc(x, problem.control.tape_value())
    xv = interface.new_petsc()
    interface.to_petsc(xv, v)
    ok = True

    # Mode gn: mult on a PETSc vector equals the operator's action, bitwise.
    gn = GaussNewtonHessianMat(hessian_mat, operator, choice="gn")
    mat = gn.create_mat()
    gn.update(StubTao(0), x, mat, P)
    y = interface.new_petsc()
    mat.mult(xv, y)
    reference = operator.action(v)
    yr = interface.new_petsc()
    interface.to_petsc(yr, reference)
    ok &= verdict(results, "G8", "mode gn: Mat.mult equals operator.action bitwise",
                  bitwise_equal(y.getArray(readonly=True), yr.getArray(readonly=True)),
                  True, mode="true")

    # gn-then-full with switch_it 1: gn at iteration 0, full at iteration 1.
    switch = GaussNewtonHessianMat(hessian_mat, operator, choice="gn-then-full",
                                   switch_it=1)
    smat = switch.create_mat()
    switch.update(StubTao(0), x, smat, P)
    ok &= verdict(results, "G8", "gn-then-full: mode after the update of iteration 0",
                  switch.mode, "gn", mode="equal")
    switch.update(StubTao(1), x, smat, P)
    ok &= verdict(results, "G8", "gn-then-full: mode after the update of iteration 1",
                  switch.mode, "full", mode="equal")
    # The full product of the switch matrix and pyadjoint's matrix after the update
    # that the switch matrix ran on it: the same tape state, so bitwise equal.
    y_full = interface.new_petsc()
    smat.mult(xv, y_full)
    y_py = interface.new_petsc()
    hessian_mat.mult(xv, y_py)
    ok &= verdict(results, "G8", "mode full: Mat.mult equals pyadjoint's matrix bitwise",
                  bitwise_equal(y_full.getArray(readonly=True), y_py.getArray(readonly=True)),
                  True, mode="true")

    # Fall-back: a full solve that ended on negative curvature sends the next
    # iteration to gn, with a switch entry; the update after that goes back to full.
    negative = PETSc.KSP.ConvergedReason.CONVERGED_NEG_CURVE
    switch.update(StubTao(2, reason=negative), x, smat, P)
    ok &= verdict(results, "G8", "fall-back: mode after a CONVERGED_NEG_CURVE solve",
                  switch.mode, "gn", mode="equal")
    ok &= verdict(results, "G8", "fall-back: switch entry recorded",
                  len(switch.switches) == 2 and "fall-back" in switch.switches[-1]["reason"],
                  True, mode="true")
    switch.update(StubTao(3, reason=PETSc.KSP.ConvergedReason.CONVERGED_RTOL), x, smat, P)
    ok &= verdict(results, "G8", "fall-back: mode at the update after the fall-back",
                  switch.mode, "full", mode="equal")
    results["G8_switches"] = switch.switches
    del solver, tao, pmat, mat, smat
    return ok


# ---------------------------------------------------------------------------------
# G9: the stale adjoint guard
# ---------------------------------------------------------------------------------

def check_stale_guard(problem, operator, v, results):
    """G9: after a Gauss-Newton product no full product can use the stale state.

    Returns:
        bool
    """
    rf = operator.rf
    solver, tao, pmat, pmat_context = build_tao(problem)
    hessian_mat, P = tao.getHessian()[:2]
    interface = pmat_context.vec_interface
    ok = True
    rf(problem.control.tape_value())
    rf.derivative(apply_riesz=False)
    reference = full_hessian(rf, v)

    operator.action(v)
    # The release step cleared the adjoint state of every block.
    left = [type(b).__name__ for b in operator.tape.get_blocks()
            if getattr(b, "adj_state", None) is not None]
    ok &= verdict(results, "G9", "blocks with an adjoint state after a GN product",
                  len(left), 0, mode="equal")

    # (a) The matrix in mode full refuses.
    guard = GaussNewtonHessianMat(hessian_mat, operator, choice="full")
    gmat = guard.create_mat()
    xv = interface.new_petsc()
    interface.to_petsc(xv, v)
    y = interface.new_petsc()
    raised = None
    try:
        guard.mult(gmat, xv, y)
    except RuntimeError as error:
        raised = str(error)
    results["G9_matrix_error"] = raised
    ok &= verdict(results, "G9", "(a) GaussNewtonHessianMat.mult in mode full raises "
                  "RuntimeError", raised is not None, True, mode="true")

    # (b) pyadjoint's own product raises too.
    raised = None
    try:
        rf.hessian(v, apply_riesz=False)
    except Exception as error:  # noqa: BLE001 -- any refusal counts, the type is recorded
        raised = f"{type(error).__name__}: {error}"
    results["G9_pyadjoint_error"] = raised
    ok &= verdict(results, "G9", "(b) rf.hessian after a GN product raises",
                  raised is not None, True, mode="true")

    # After a new gradient the full product is the reference again, bitwise. No
    # replay in between, so the forward states are those of the reference.
    rf.derivative(apply_riesz=False)
    again = full_hessian(rf, v)
    ok &= verdict(results, "G9", "rf.hessian after rf.derivative() equals the reference "
                  "bitwise", bitwise_equal(again.dat.data_ro, reference.dat.data_ro),
                  True, mode="true")
    # With pyadjoint's update (a replay at the same point and a gradient) the
    # forward solves start from other initial guesses, so the product can differ at
    # round-off. Reported, not a gate.
    x = interface.new_petsc()
    interface.to_petsc(x, problem.control.tape_value())
    type(hessian_mat.getPythonContext()).update(tao, x, hessian_mat, P)
    after_update = full_hessian(rf, v)
    results["G9_after_update_max_relative"] = max_relative(after_update, reference)
    say(f"    info  after pyadjoint's update: max relative difference "
        f"{results['G9_after_update_max_relative']:.3e}")
    del solver, tao, pmat, gmat
    return ok


# ---------------------------------------------------------------------------------
# G10: counts and memory
# ---------------------------------------------------------------------------------

def check_counts(problem, v, results, products=10, expected_count=None):
    """G10: ten Gauss-Newton products count as ten and touch no other count.

    Returns:
        bool
    """
    import optimisers
    rf = problem.reduced_functional
    recorder = optimisers.Recorder(problem)
    recorder.attach()
    rf(problem.control.tape_value())
    before = {"forward": recorder.forward, "adjoint": recorder.adjoint,
              "hessian": recorder.hessian, "gradients": len(recorder.gradients)}
    operator = GaussNewtonOperator(rf, expected_count=expected_count, recorder=recorder)
    rss_before = peak_rss_gb()
    times = []
    for _ in range(products):
        t0 = time.perf_counter()
        operator.action(v)
        times.append(time.perf_counter() - t0)
    rss_after = peak_rss_gb()
    after = {"forward": recorder.forward, "adjoint": recorder.adjoint,
             "hessian": recorder.hessian, "gradients": len(recorder.gradients)}
    results["G10_values"] = {"before": before, "after": after,
                             "gn_hessian": recorder.gn_hessian,
                             "gn_hessian_times": recorder.gn_hessian_times,
                             "peak_rss_before_gb": rss_before,
                             "peak_rss_after_gb": rss_after, "wall_per_product": times}
    say(f"    info  peak RSS {rss_before:.3f} GB before, {rss_after:.3f} GB after "
        f"{products} products; product times {np.round(times, 3).tolist()} s")
    ok = verdict(results, "G10", "recorder.gn_hessian after the products",
                 recorder.gn_hessian, products, mode="equal")
    ok &= verdict(results, "G10", "len(gn_hessian_times)", len(recorder.gn_hessian_times),
                  products, mode="equal")
    ok &= verdict(results, "G10", "forward, adjoint, hessian and gradient rows unchanged",
                  before == after, True, mode="true")
    # The recorder stays attached to the reduced functional. It only counts, so a
    # later check sees the same numbers, and the caller runs this check last.
    return ok


# ---------------------------------------------------------------------------------
# G11: curvature at the first guess
# ---------------------------------------------------------------------------------

def first_direction_curvature(problem, operator):
    """``pHp/pAp`` and ``pH_GN p/pAp`` along ``p = A^{-1} g``, the first CG direction.

    The tape must hold a replay at the first guess. ``g`` is the gradient in the
    dual space, ``p`` its Riesz representative in the metric of the control (with
    the Dirichlet operator ``p`` is zero on the wall nodes), ``pAp`` the metric norm
    squared of ``p``.

    Returns:
        dict with ``J``, ``pAp``, ``pHp``, ``pHGNp``, the two ratios and the dropped
        term ``(pHp - pH_GN p)/pAp``, and the wall time of each call.
    """
    rf = operator.rf
    ip = problem.inner_product
    out = {}
    t0 = time.perf_counter()
    out["J"] = float(rf(problem.control.tape_value()))
    out["time_forward"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = cofunction_copy(rf.derivative(apply_riesz=False))
    out["time_gradient"] = time.perf_counter() - t0
    p = ip.riesz_map(g)
    pAp = float(assemble(ip.form(p, p)))
    t0 = time.perf_counter()
    Hp = full_hessian(rf, p)
    out["time_full_product"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Gp = operator.action(p)
    out["time_gn_product"] = time.perf_counter() - t0
    pHp, pGp = pairing(Hp, p), pairing(Gp, p)
    out.update(pAp=pAp, pHp=pHp, pHGNp=pGp, pHp_over_pAp=pHp / pAp,
               pHGNp_over_pAp=pGp / pAp, dropped_over_pAp=(pHp - pGp) / pAp,
               g_dot_p=pairing(g, p))
    return out


# ---------------------------------------------------------------------------------
# G13: the predicted-fall bookkeeping
# ---------------------------------------------------------------------------------

def _explicit_product_recorder():
    """The Krylov recorder class plus one explicit product ``d^T Q d`` after each solve.

    ``Q`` is the operator of the solve (the Hessian on the inactive set, with the
    zero shift), applied to the solution ``d`` once more. That costs one product
    per solve, so the class exists for the check only. It is built here, inside a
    function, so that importing this module does not import ``recording``.
    """
    import recording

    class ExplicitProductRecorder(recording.KrylovRecorder):
        def _post_solve(self, ksp, rhs, x):
            super()._post_solve(ksp, rhs, x)
            operator = ksp.getOperators()[0]
            y = x.duplicate()
            operator.mult(x, y)
            self.solves[-1]["dHd_explicit"] = float(x.dot(y))
            y.destroy()

    return ExplicitProductRecorder


# The three exits of G13: a Gauss-Newton solve that converges inside a large radius,
# one that a small radius cuts, and one full-Hessian solve, whatever its exit.
PREDICTED_FALL_CASES = (
    ("gn, inside the radius", "gn", 1e4, 50, "CONVERGED_RTOL"),
    ("gn, on the radius", "gn", 1e-3, 50, "CONVERGED_STEP_LENGTH"),
    ("full", "full", 1.0, 20, None),
)


def check_predicted_fall(build, results, cases=PREDICTED_FALL_CASES):
    """G13: ``2 (cg_objective + b . d)`` against an explicit ``d^T H d`` for each exit.

    STCG updates its model value in separate places for a converged solve, a step to
    the radius and a negative-curvature step, so each exit that a run can take is
    tested against one explicit product of the solve operator.

    Args:
        build: a callable that returns ``(problem, expected term count)`` with a
            fresh tape at the first guess.
        results: the results dictionary.
        cases: ``(label, hessian, radius, Krylov cap, expected reason or None)``,
            with an optional sixth entry, a dict of further TAO options (the cylinder
            loosens the tolerance of its converged case to keep the check short).

    Returns:
        bool
    """
    import optimisers
    original = optimisers.KrylovRecorder
    optimisers.KrylovRecorder = _explicit_product_recorder()
    ok = True
    try:
        for case in cases:
            label, hessian, radius, cap, reason = case[:5]
            options = {"tao_bnk_ksp_max_it": cap, "tao_bnk_imax": 0}
            if len(case) > 5:
                options.update(case[5])
            problem, count = build()
            with stop_annotating():
                recorder = optimisers.run_tao(
                    problem, "tao-bnls-riesz", max_it=1, initial_radius=radius,
                    extra_options=options, hessian=hessian, gn_expected_terms=count)
            solves = [s for row in recorder.iterations
                      for s in row.get("krylov_solves", [])]
            solves += recorder.final_diagnostics.get("krylov_solves_unassigned", [])
            solve = solves[0]
            from_model = 2.0 * (solve["cg_objective"] + solve["rhs_dot_sol"])
            explicit = solve["dHd_explicit"]
            results.setdefault("G13_solves", []).append(
                {"case": label, "reason": solve["reason_name"],
                 "iterations": solve["iterations"], "cg_objective": solve["cg_objective"],
                 "rhs_dot_sol": solve["rhs_dot_sol"], "dHd_from_model": from_model,
                 "dHd_explicit": explicit, "trust_radius": solve["trust_radius"],
                 "step_norm": solve["step_norm"], "rtol": solve["rtol"]})
            if reason is not None:
                ok &= verdict(results, "G13", f"{label}: stop reason", solve["reason_name"],
                              reason, mode="equal")
            ok &= verdict(results, "G13", f"{label}: |2 (q + b.d) - d^T H d| / |d^T H d|",
                          abs(from_model - explicit) / abs(explicit), 1e-7)
    finally:
        optimisers.KrylovRecorder = original
    return ok
