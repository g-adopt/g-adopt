"""Short checks of the cylindrical comparison driver. Each one can fail.

Usage:
    python checks.py rol-params
    python checks.py objective --rundir RUN --timesteps 5 --delta-t 1.2e-5 [--make-reference]
    python checks.py hvp       --rundir RUN --timesteps 5 --delta-t 1.2e-5
    python checks.py spd       --rundir RUN --timesteps 5 --delta-t 1.2e-5
    python checks.py timing    --rundir RUN --timesteps 125 --delta-t 1.2e-5
    python checks.py gn        --rundir RUN --timesteps 2 --delta-t 1.2e-5 [--make-reference]
    python checks.py gn-curvature --rundir RUN --timesteps 125 --delta-t 1.2e-5

Every check prints its numbers, a PASS or FAIL line against a threshold that is fixed
in this file, and writes a JSON file of its results with ``--json``. The process
exits with status 1 on a failure.

``rol-params``. The ROL parameter list of ``compare.py rol-lbfgs`` against the one of
``inverse.inverse()`` (g-adopt's ``minimisation_parameters`` with an initial radius
of 1e-3). Passes when the two differ only in the iteration limit, which the work
budget replaces. No PDE solve.

``objective``. The driver reproduces the ROL objective and gradient of
``inverse.py`` at the first iterate. The production path builds
``gadopt.inverse.ROLSolver`` on the reduced functional of
``generate_inverse_problem``, as ``LinMoreOptimiser`` does, and asks its ROL
objective for the value and the gradient at the first guess. The driver path does
the same through ``cylindrical_problem.build_problem`` and the ROL objects of
``optimisers.run_rol``, first with the L2 inner product of g-adopt, then with the
regularisation metric. Compared: J, the dual gradient (independent of the inner
product), ROL's gradient vector under L2, and ``A g_A = g`` for the metric gradient
``g_A``. The two tapes are recorded one after the other in the same process.

``hvp``. A Hessian-vector product through the driver's TAO matrices against
``tests/hessian/cylindrical_hvp_taylor.py``, both on the temperature-misfit
objective of the harness (case Tobs), in the rank-independent smooth direction
(``smooth-stable``, seed 1). The driver side builds ``TAOSolver`` with the
preconditioning matrix of ``riesz_hessian.py``, as ``run_tao`` does, and applies
pyadjoint's Hessian matrix and the metric-shifted matrix at zero shift. With
``--fd`` the driver also compares its product on the full production objective
against a central difference of the gradient.

``spd``. The preconditioning matrix and the Riesz operator on a free set of the
size BNK sees. Symmetry of ``Z^T A Z`` and of ``Z^T A^{-1} Z`` on random vectors;
the smallest eigenvalue of each, which must be positive; ``A^{-1} A = I`` through
the Riesz map; and ``A`` against the Hessian of the taped regularisation terms.

``gn``. The checks G1 to G7, G9, G11 (values only) and G13 of the Gauss-Newton
product (NOTES/team/gauss-newton/PLAN.md section 2.2) on the production objective, at
the given step count: the term list and the weights against the normalisations
computed from the data, the seeded gradient, the zero-residual comparison, symmetry,
positivity against the metric, the tangent-linear half, the stale-adjoint guard, the
curvature along the first CG direction, and the predicted-fall bookkeeping of three
one-iteration TAO runs. Serial. The routines are those of ``box/gn_checks.py``.

``gn-curvature``. The curvature of the production objective at the first guess along
``p = A^{-1} g``, the first CG direction: one gradient, one full product and one
Gauss-Newton product, with ``pHp/pAp``, ``pH_GN p/pAp``, the dropped term, the
per-call times and the peak memory per rank. Step 2 of the curvature job of section
5.2 of the plan, on any number of ranks. Fails only if a number is not finite.
"""

import argparse
import copy
import json
import sys
import time

import numpy as np
from mpi4py import MPI

import cylindrical_problem as cp  # sets the import paths; must come first
from cylindrical_problem import build_problem, prepare_reference

from firedrake import (Function, TestFunction, TrialFunction, assemble,
                       dot, dx, grad)
from petsc4py import PETSc
from pyadjoint import MinimizationProblem, stop_annotating

import optimisers  # noqa: E402 -- demos/optimisation/box
from gadopt.inverse import ROLSolver, minimisation_parameters
from riesz_hessian import build_preconditioner_matrix, build_shifted_hessian
import compare

COMM = MPI.COMM_WORLD


def say(text):
    """Print on rank 0 and flush, so that a job log shows progress at once."""
    if COMM.rank == 0:
        print(text, flush=True)


def relative(a, b):
    """``||a - b|| / ||b||`` of two local arrays, global over the communicator."""
    diff = COMM.allreduce(float(np.dot(a - b, a - b)), op=MPI.SUM)
    ref = COMM.allreduce(float(np.dot(b, b)), op=MPI.SUM)
    return float(np.sqrt(diff / ref)) if ref > 0 else float(np.sqrt(diff))


def verdict(name, value, threshold, results):
    """Record one comparison against its threshold and print it."""
    ok = bool(value <= threshold)
    results.setdefault("tests", []).append(
        {"name": name, "value": value, "threshold": threshold, "pass": ok})
    say(f"  {'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (threshold {threshold:.0e})")
    return ok


# ---------------------------------------------------------------------------------
# rol-params
# ---------------------------------------------------------------------------------

def flatten(tree, prefix=()):
    """``{path tuple: value}`` of a nested dictionary."""
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flatten(value, prefix + (key,)))
        else:
            out[prefix + (key,)] = value
    return out


def check_rol_params(args, results):
    """The driver's ROL parameter list against the one of ``inverse.inverse()``."""
    reference = copy.deepcopy(minimisation_parameters)
    # The two changes that inverse.inverse() makes.
    reference["Status Test"]["Iteration Limit"] = 5
    reference["Step"]["Trust Region"]["Initial Radius"] = 1e-3

    parsed = compare.parse_args(["--optimiser", "rol-lbfgs", "--rundir", "unused"])
    # optimiser_settings needs the number of control dofs only for TAO.
    kwargs, _ = compare.optimiser_settings(parsed, problem=_FakeProblem())
    rol_kwargs = {k: kwargs[k] for k in ("initial_radius", "krylov_max_it",
                                         "secant_storage", "krylov_rtol", "krylov_atol")}
    driver = optimisers.rol_parameters("rol-lbfgs", kwargs["max_it"], **rol_kwargs)

    a, b = flatten(reference), flatten(driver)
    differ = sorted({k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")})
    for key in differ:
        say(f"  differs: {'/'.join(key)}: inverse.py {a.get(key, '<absent>')!r}, "
            f"driver {b.get(key, '<absent>')!r}")
    # The iteration limit is replaced by the work budget. "Use as Preconditioner" is
    # absent from g-adopt's list and False in the driver's; ROL reads it with the
    # default false (ROL_TypeB_LinMoreAlgorithm_Def.hpp:116,
    # ``get("Use as Preconditioner", false)``), so the two lists configure the same
    # algorithm. Any other difference fails the check.
    allowed = {("Status Test", "Iteration Limit")}
    if b.get(("General", "Secant", "Use as Preconditioner")) is False and \
            ("General", "Secant", "Use as Preconditioner") not in a:
        allowed.add(("General", "Secant", "Use as Preconditioner"))
    unexpected = [k for k in differ if k not in allowed]
    results["differences"] = {"/".join(k): [a.get(k), b.get(k)] for k in differ}
    return verdict("entries that differ beyond the iteration limit and a default",
                   float(len(unexpected)), 0.0, results)


class _FakeProblem:
    """The one attribute of a problem that ``optimiser_settings`` reads for ROL."""

    class _Space:
        def dim(self):
            return 66048

    class _Tic:
        def function_space(self):
            return _FakeProblem._Space()

    Tic = _Tic()


# ---------------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------------

def rol_value_gradient(solver):
    """ROL's objective value and gradient vector at the first iterate of a solver.

    ``update`` with ``UpdateType.Initial`` is what ROL calls first: it replays the
    tape at the vector. ``gradient`` then runs the adjoint and applies the Riesz map
    of the control.

    Returns:
        ``(J, local array of the gradient vector)``
    """
    import ROL
    x = solver.rolvector
    objective = solver.rolobjective
    objective.update(x, ROL.UpdateType.Initial, 0)
    value = float(objective.value(x, 0.0))
    g = x.clone()
    objective.gradient(g, x, 0.0)
    return value, np.array(g.dat[0].dat.data_ro, dtype=float)


def check_objective(args, results):
    """The driver reproduces the ROL objective and gradient of inverse.py."""
    weights = dict(alpha_T=args.alpha_T, alpha_u=args.alpha_u, alpha_d=args.alpha_d,
                   alpha_s=args.alpha_s)
    params = copy.deepcopy(minimisation_parameters)
    params["Step"]["Trust Region"]["Initial Radius"] = 1e-3

    # ---- production: generate_inverse_problem + gadopt.inverse.ROLSolver -------------
    prepare_reference(args.rundir, args.timesteps, args.delta_t, make=args.make_reference)
    import inverse as production
    t0 = time.perf_counter()
    produced = production.generate_inverse_problem(checkpointing_schedule=None,
                                                   uimposed=False, **weights)
    t_tape = time.perf_counter() - t0
    rf = produced["reduced_functional"]
    Tic = produced["control"]
    lower = Function(Tic.function_space()).assign(0.0)
    upper = Function(Tic.function_space()).assign(1.0)
    with stop_annotating():
        solver = ROLSolver(MinimizationProblem(rf, bounds=(lower, upper)), params,
                           inner_product="L2")
        t0 = time.perf_counter()
        J_prod, g_prod = rol_value_gradient(solver)
        t_fg = time.perf_counter() - t0
        dual_prod = np.array(rf.derivative(apply_riesz=False).dat.data_ro, dtype=float)
    taped = float(produced["objective"])
    m0 = np.array(Tic.dat.data_ro, dtype=float)
    outside = COMM.allreduce(int(np.sum((m0 < 0.0) | (m0 > 1.0))), op=MPI.SUM)
    say(f"production: J taped {taped:.15e}, J replay {J_prod:.15e}; "
        f"taping {t_tape:.1f} s, replay + gradient {t_fg:.1f} s; "
        f"{outside} first-guess nodes outside [0, 1]")
    results.update(J_taped=taped, J_production=J_prod, first_guess_outside=outside,
                   time_taping=t_tape, time_replay_gradient=t_fg)

    # ---- driver, L2 inner product: the ROL objects of run_rol ---------------------
    out = {}
    for riesz in ("L2", "regularisation"):
        problem = build_problem(args.rundir, timesteps=args.timesteps,
                                delta_t=args.delta_t, inner_product=riesz, **weights)
        with stop_annotating():
            solver = ROLSolver(MinimizationProblem(problem.reduced_functional,
                                                   bounds=problem.bounds),
                               params, inner_product=problem.inner_product.name,
                               vector_class=optimisers.FormROLVector,
                               vector_args=[problem.inner_product])
            J_drv, g_drv = rol_value_gradient(solver)
            dual = problem.reduced_functional.derivative(apply_riesz=False)
            dual_drv = np.array(dual.dat.data_ro, dtype=float)
            # A g_A: the metric applied to ROL's gradient vector. It must give back
            # the dual gradient, because g_A = A^{-1} g.
            gA = Function(problem.Tic.function_space())
            gA.dat.data[:] = g_drv
            back = assemble(problem.inner_product.form(
                gA, TestFunction(problem.Tic.function_space())))
            back = np.array(back.dat.data_ro, dtype=float)
        out[riesz] = dict(J=J_drv, g=g_drv, dual=dual_drv, back=back,
                          weights=problem.weights, taped=problem.taped_objective)
        say(f"driver {riesz}: J replay {J_drv:.15e}, J taped {problem.taped_objective:.15e}")

    results["weights"] = out["regularisation"]["weights"]
    ok = True
    ok &= verdict("J taped, driver against production",
                  abs(out["L2"]["taped"] - taped) / abs(taped), 1e-12, results)
    ok &= verdict("J at the first iterate, driver L2 against production",
                  abs(out["L2"]["J"] - J_prod) / abs(J_prod), 1e-12, results)
    ok &= verdict("J at the first iterate, driver metric against production",
                  abs(out["regularisation"]["J"] - J_prod) / abs(J_prod), 1e-12, results)
    ok &= verdict("dual gradient, driver L2 against production",
                  relative(out["L2"]["dual"], dual_prod), 1e-10, results)
    ok &= verdict("dual gradient, driver metric against production",
                  relative(out["regularisation"]["dual"], dual_prod), 1e-10, results)
    ok &= verdict("ROL gradient vector under L2, driver against production",
                  relative(out["L2"]["g"], g_prod), 1e-8, results)
    ok &= verdict("A g_A against the dual gradient (metric Riesz map)",
                  relative(out["regularisation"]["back"], dual_prod), 1e-8, results)
    # The L2 representative must not coincide with the metric one: if it did, the
    # metric would not be reaching the gradient at all.
    gap = relative(out["regularisation"]["g"], out["L2"]["g"])
    say(f"  info  metric gradient vs L2 gradient vector: {gap:.3e} (must be O(1))")
    results["metric_vs_L2_gradient"] = gap
    ok &= verdict("metric and L2 gradient vectors differ (1 / gap)", 1.0 / max(gap, 1e-300),
                  1e3, results)
    return ok


# ---------------------------------------------------------------------------------
# hvp
# ---------------------------------------------------------------------------------

def tao_hessian_actions(problem, m, h):
    """Apply the driver's TAO Hessian matrices to ``h`` at the control value ``m``.

    The TAO object is built as ``run_tao`` builds it for ``tao-bnls-riesz``: the
    preconditioning matrix of ``riesz_hessian.py`` and ``TAOSolver``. The Hessian
    update callback replays the tape at ``m`` and runs the adjoint, which is what
    BNK triggers before its first Krylov solve.

    Returns:
        dict with the local arrays of pyadjoint's product and of the metric-shifted
        matrix at zero shift, and the same with a shift ``pert`` together with the
        reference ``H h + pert A h``.
    """
    from pyadjoint.optimization.tao_solver import PETScVecInterface, TAOSolver

    params = optimisers.tao_parameters("tao-bnls-riesz", 1, 1.0,
                                       {"tao_bnk_ksp_max_it": 1})
    pmat, pctx = build_preconditioner_matrix(problem)
    solver = TAOSolver(MinimizationProblem(problem.reduced_functional,
                                           bounds=problem.bounds), params, Pmat=pmat)
    tao = solver.tao
    H_py, P = tao.getHessian()[:2]
    interface = PETScVecInterface((problem.Tic,))
    x = interface.new_petsc()
    interface.to_petsc(x, (m,))
    hv = interface.new_petsc()
    interface.to_petsc(hv, (h,))
    y = hv.duplicate()

    out = {}
    t0 = time.perf_counter()
    type(H_py.getPythonContext()).update(tao, x, H_py, P)
    out["time_update"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    H_py.mult(hv, y)
    out["time_hvp"] = time.perf_counter() - t0
    out["pyadjoint"] = np.array(y.getArray(readonly=True), dtype=float)

    shifted, sctx = build_shifted_hessian(tao, pctx)
    sctx.update(tao, x, shifted, P)
    shifted.mult(hv, y)
    out["shifted_zero"] = np.array(y.getArray(readonly=True), dtype=float)
    # With a shift: H h + pert A h, against the two products computed apart.
    pert = 0.37
    sctx.shift(shifted, pert)
    shifted.mult(hv, y)
    out["shifted_pert"] = np.array(y.getArray(readonly=True), dtype=float)
    Ah = hv.duplicate()
    pctx.mult(None, hv, Ah)
    out["reference_pert"] = out["pyadjoint"] + pert * np.array(Ah.getArray(readonly=True))
    del solver, tao, pmat, shifted
    return out


def check_hvp(args, results):
    """The driver's Hessian-vector product against the harness."""
    from taylor_diagnostics import make_perturbation
    import cylindrical_hvp_taylor as harness

    prepare_reference(args.rundir, args.timesteps, args.delta_t, make=args.make_reference)

    # ---- harness: tests/hessian/cylindrical_hvp_taylor.py ------------------------
    rf_h, m_h = harness.build_reduced_functional("Tobs", args.timesteps, args.delta_t)
    h = make_perturbation(m_h.function_space(), "smooth-stable", 1)
    with stop_annotating():
        J_h = float(rf_h(m_h))
        rf_h.derivative()
        t0 = time.perf_counter()
        Hh_h = rf_h.hessian(h, apply_riesz=False)
        t_h = time.perf_counter() - t0
    Hh_h = np.array(Hh_h.dat.data_ro, dtype=float)
    h_arr = np.array(h.dat.data_ro, dtype=float)
    hHh_h = COMM.allreduce(float(np.dot(h_arr, Hh_h)), op=MPI.SUM)
    say(f"harness: J {J_h:.15e}, <h,Hh> {hHh_h:.15e}, product {t_h:.1f} s")

    # ---- driver, the same objective (Tobs alone) ------------------------------------
    problem = build_problem(args.rundir, timesteps=args.timesteps, delta_t=args.delta_t,
                            alpha_T=1.0, alpha_u=-1, alpha_d=-1, alpha_s=-1)
    h_d = make_perturbation(problem.Tic.function_space(), "smooth-stable", 1)
    with stop_annotating():
        acts = tao_hessian_actions(problem, problem.Tic, h_d)
    hHh_d = COMM.allreduce(float(np.dot(h_arr, acts["pyadjoint"])), op=MPI.SUM)
    say(f"driver: <h,Hh> {hHh_d:.15e}; update {acts['time_update']:.1f} s, "
        f"product {acts['time_hvp']:.1f} s")
    results.update(J_harness=J_h, hHh_harness=hHh_h, hHh_driver=hHh_d,
                   time_hvp_harness=t_h, time_hvp_driver=acts["time_hvp"])

    ok = True
    ok &= verdict("direction, driver against harness",
                  relative(np.array(h_d.dat.data_ro), h_arr), 1e-14, results)
    ok &= verdict("H h, driver (pyadjoint matrix) against harness",
                  relative(acts["pyadjoint"], Hh_h), 1e-10, results)
    ok &= verdict("H h, driver (metric-shifted matrix, zero shift) against harness",
                  relative(acts["shifted_zero"], Hh_h), 1e-10, results)
    ok &= verdict("<h,Hh>, driver against harness",
                  abs(hHh_d - hHh_h) / abs(hHh_h), 1e-10, results)
    ok &= verdict("(H + pert A) h, metric-shifted matrix against H h + pert A h",
                  relative(acts["shifted_pert"], acts["reference_pert"]), 1e-12, results)

    if args.fd:
        ok &= check_fd(args, results)
    return ok


def check_fd(args, results):
    """The driver's product on the full objective against a central difference.

    ``(g(m + eps h) - g(m - eps h)) / (2 eps)`` with ``g`` the dual gradient, at two
    epsilons. A decay of the relative difference by about 100 per decade of eps is
    truncation; the best value bounds the error of ``H h``.
    """
    from taylor_diagnostics import make_perturbation
    problem = build_problem(args.rundir, timesteps=args.timesteps, delta_t=args.delta_t,
                            alpha_T=args.alpha_T, alpha_u=args.alpha_u,
                            alpha_d=args.alpha_d, alpha_s=args.alpha_s)
    rf = problem.reduced_functional
    Q1 = problem.Tic.function_space()
    h = make_perturbation(Q1, "smooth-stable", 1)
    m = Function(Q1).assign(problem.Tic)
    with stop_annotating():
        acts = tao_hessian_actions(problem, m, h)
        Hh = acts["pyadjoint"]
        rows = []
        for eps in args.fd_eps:
            plus = Function(Q1).assign(m + eps * h)
            minus = Function(Q1).assign(m - eps * h)
            rf(plus)
            gp = np.array(rf.derivative(apply_riesz=False).dat.data_ro, dtype=float)
            rf(minus)
            gm = np.array(rf.derivative(apply_riesz=False).dat.data_ro, dtype=float)
            fd = (gp - gm) / (2 * eps)
            rows.append({"eps": eps, "relative": relative(Hh, fd)})
            say(f"  fd eps {eps:.0e}: ||Hh - fd|| / ||fd|| = {rows[-1]['relative']:.3e}")
    results["fd_full_objective"] = rows
    best = min(r["relative"] for r in rows)
    return verdict("full objective, H h against central difference (best eps)",
                   best, 1e-3, results)


# ---------------------------------------------------------------------------------
# spd
# ---------------------------------------------------------------------------------

def check_spd(args, results):
    """Symmetry and definiteness of the preconditioner and the Riesz operator."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if COMM.size != 1:
        raise SystemExit("spd runs in serial: it assembles the operator with scipy")
    problem = build_problem(args.rundir, timesteps=args.timesteps, delta_t=args.delta_t,
                            alpha_T=args.alpha_T, alpha_u=args.alpha_u,
                            alpha_d=args.alpha_d, alpha_s=args.alpha_s)
    Q1 = problem.Tic.function_space()
    ip = problem.inner_product
    n = Q1.dim()
    rng = np.random.default_rng(7)

    # A free set of the kind BNK sees: every node of the first guess within 1e-3 of
    # a bound is active (the two walls and the saturated cold and hot regions), plus
    # a random tenth of the rest, so that the set is not only the walls.
    m0 = np.array(problem.Tic.dat.data_ro, dtype=float)
    active = (m0 <= 1e-3) | (m0 >= 1 - 1e-3)
    active |= rng.random(n) < 0.1
    free = np.flatnonzero(~active).astype(PETSc.IntType)
    say(f"free set: {free.size} of {n} nodes ({n - free.size} active)")
    results.update(control_dofs=n, free=int(free.size))

    pmat, pctx = build_preconditioner_matrix(problem)
    iset = PETSc.IS().createGeneral(free, comm=PETSc.COMM_SELF)
    sub = pmat.createSubMatrix(iset, iset)
    sctx = sub.getPythonContext()

    def apply(kind, v):
        x = PETSc.Vec().createWithArray(np.ascontiguousarray(v))
        y = x.duplicate()
        if kind == "A":
            sub.mult(x, y)
        else:
            sctx.apply_riesz(x, y)
        return np.array(y.getArray(readonly=True), dtype=float)

    ok = True
    for kind, label in (("A", "Z^T A Z (preconditioning matrix)"),
                        ("Ainv", "Z^T A^-1 Z (preconditioner)")):
        worst = 0.0
        for _ in range(5):
            x, y = rng.standard_normal(free.size), rng.standard_normal(free.size)
            Px, Py = apply(kind, x), apply(kind, y)
            # |y^T P x - x^T P y| against sqrt(x^T P x * y^T P y), the scale of a
            # bilinear form on these two vectors.
            asym = abs(np.dot(y, Px) - np.dot(x, Py)) / np.sqrt(
                abs(np.dot(x, Px) * np.dot(y, Py)))
            worst = max(worst, asym)
        ok &= verdict(f"symmetry of {label}", worst, 1e-12, results)

    # The assembled metric, for the eigenvalues. Its free block is the operator that
    # the masked matrix applies; the comparison below proves it.
    A = assemble(ip.form(TrialFunction(Q1), TestFunction(Q1))).petscmat
    indptr, indices, values = A.getValuesCSR()
    A_csr = sp.csr_matrix((values, indices, indptr), shape=(n, n))
    A_ff = A_csr[free][:, free].tocsc()
    x = rng.standard_normal(free.size)
    ok &= verdict("masked matrix against the assembled free block",
                  relative(apply("A", x), A_ff @ x), 1e-12, results)

    # The preconditioner against the principal free block of A^-1, computed with an
    # independent factorisation: solve A u = Z x with scipy and keep u on the free set.
    t0 = time.perf_counter()
    lu = spla.splu(A_csr.tocsc())
    x = rng.standard_normal(free.size)
    full = np.zeros(n)
    full[free] = x
    ok &= verdict("preconditioner against the free block of A^-1 (scipy LU)",
                  relative(apply("Ainv", x), lu.solve(full)[free]), 1e-10, results)

    # Eigenvalues. A_ff directly, by shift-invert Lanczos for the smallest. The
    # preconditioner (A^-1)_ff is a principal block of A^-1, so by Cauchy interlacing
    # its eigenvalues lie in [1 / lambda_max(A), 1 / lambda_min(A)]; with the identity
    # just checked, the extremes of the full A bound it, and it is positive definite
    # exactly when A is.
    def extremes(matrix):
        low = float(spla.eigsh(matrix, k=1, sigma=0.0, which="LM",
                               return_eigenvectors=False)[0])
        high = float(spla.eigsh(matrix, k=1, which="LA", return_eigenvectors=False)[0])
        return low, high

    lam_min, lam_max = extremes(A_ff)
    full_min, full_max = extremes(A_csr.tocsc())
    say(f"  A_ff eigenvalues in [{lam_min:.4e}, {lam_max:.4e}]; full A in "
        f"[{full_min:.4e}, {full_max:.4e}], so (A^-1)_ff in "
        f"[{1 / full_max:.4e}, {1 / full_min:.4e}]; {time.perf_counter() - t0:.1f} s")
    results.update(A_ff_min=lam_min, A_ff_max=lam_max, A_min=full_min, A_max=full_max,
                   Ainv_ff_bounds=[1 / full_max, 1 / full_min])
    # Positive means larger than roundoff on the scale of the largest eigenvalue.
    ok &= verdict("A_ff positive definite (-lambda_min / lambda_max)",
                  -lam_min / lam_max, -1e-12, results)
    ok &= verdict("A positive definite, hence (A^-1)_ff (-lambda_min / lambda_max)",
                  -full_min / full_max, -1e-12, results)
    # A direct look at the preconditioner as well: its Rayleigh quotients on random
    # vectors, which must all be positive.
    rq = min(float(np.dot(v, apply("Ainv", v)) / np.dot(v, v))
             for v in (rng.standard_normal(free.size) for _ in range(5)))
    results["Ainv_ff_min_rayleigh"] = rq
    ok &= verdict("(A^-1)_ff Rayleigh quotients positive (-min / upper bound)",
                  -rq * full_min, 0.0, results)

    # The Riesz map inverts the full metric: A^-1 (A x) = x.
    xf = Function(Q1)
    xf.dat.data[:] = rng.standard_normal(n)
    Ax = assemble(ip.form(xf, TestFunction(Q1)))
    back = ip.riesz_map(Ax)
    ok &= verdict("Riesz map: A^-1 A x = x", relative(np.array(back.dat.data_ro),
                                                      np.array(xf.dat.data_ro)),
                  1e-10, results)

    # The metric against the Hessian of the taped regularisation terms. With no
    # misfit in the objective, generate_inverse_problem runs no time step, so this
    # tape is cheap. A smooth direction, as the Hessian harness uses.
    from taylor_diagnostics import make_perturbation
    import inverse as production
    produced = production.generate_inverse_problem(
        alpha_T=-1, alpha_u=-1, alpha_d=args.alpha_d, alpha_s=args.alpha_s,
        checkpointing_schedule=None, uimposed=False)
    rf_reg = produced["reduced_functional"]
    # generate_inverse_problem loads the mesh again, so this tape lives on a mesh
    # object of its own. The direction is made on each mesh from the same node
    # coordinates (smooth-stable), and the two meshes have the same numbering,
    # which the comparison of the two directions below confirms.
    Q1_reg = produced["control"].function_space()
    h_reg = make_perturbation(Q1_reg, "smooth-stable", 1)
    h = make_perturbation(Q1, "smooth-stable", 1)
    ok &= verdict("direction on the two meshes",
                  relative(np.array(h_reg.dat.data_ro), np.array(h.dat.data_ro)),
                  1e-14, results)
    with stop_annotating():
        rf_reg(produced["control"])
        rf_reg.derivative()
        H_reg = np.array(rf_reg.hessian(h_reg, apply_riesz=False).dat.data_ro,
                         dtype=float)
        Ah = np.array(assemble(ip.form(h, TestFunction(Q1))).dat.data_ro, dtype=float)
    h_arr = np.array(h.dat.data_ro, dtype=float)
    results.update(hAh=float(np.dot(h_arr, Ah)), hHreg_h=float(np.dot(h_arr, H_reg)))
    say(f"  <h, A h> {np.dot(h_arr, Ah):.12e}, <h, H_reg h> {np.dot(h_arr, H_reg):.12e}")
    ok &= verdict("A h against the Hessian of the taped regularisation",
                  relative(Ah, H_reg), 1e-6, results)
    return ok


# ---------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------

def peak_rss_gb():
    """Peak resident memory of this process in GB, and the maximum over the ranks.

    ``ru_maxrss`` is in bytes on macOS and in kilobytes on Linux.
    """
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1e-9 if sys.platform == "darwin" else 1e-6
    local = rss * scale
    return local, COMM.allreduce(local, op=MPI.MAX), COMM.allreduce(local, op=MPI.SUM)


def check_timing(args, results):
    """Cost of each call and memory per rank on the production objective.

    Not a correctness check: it measures what the run plan needs. Two calls each of
    the functional, the gradient and the Hessian-vector product, the second of each
    being the steady cost (the first builds solvers and kernels). It also prints
    ``<h, H h>`` of the full objective at the first guess in the smooth direction
    of the harness, with ``<h, A h>`` beside it: a negative value shows negative
    curvature that survives the regularisation. Fails only if a number is not finite.
    """
    from taylor_diagnostics import make_perturbation
    t0 = time.perf_counter()
    problem = build_problem(args.rundir, timesteps=args.timesteps, delta_t=args.delta_t,
                            alpha_T=args.alpha_T, alpha_u=args.alpha_u,
                            alpha_d=args.alpha_d, alpha_s=args.alpha_s,
                            make_reference=args.make_reference)
    results["time_build"] = time.perf_counter() - t0
    rss_after_tape = peak_rss_gb()
    rf = problem.reduced_functional
    Q1 = problem.Tic.function_space()
    m = Function(Q1).assign(problem.Tic)
    h = make_perturbation(Q1, "smooth-stable", 1)
    times = {"functional": [], "gradient": [], "hessian": []}
    with stop_annotating():
        for _ in range(2):
            t = time.perf_counter()
            J = float(rf(m))
            times["functional"].append(time.perf_counter() - t)
            t = time.perf_counter()
            rf.derivative(apply_riesz=False)
            times["gradient"].append(time.perf_counter() - t)
        for _ in range(2):
            t = time.perf_counter()
            Hh = rf.hessian(h, apply_riesz=False)
            times["hessian"].append(time.perf_counter() - t)
        Ah = assemble(problem.inner_product.form(h, TestFunction(Q1)))
    hHh = float(assemble(Hh(h)))
    hAh = float(assemble(Ah(h)))
    rss = peak_rss_gb()
    steady = {k: v[-1] for k, v in times.items()}
    results.update(J=J, hHh_full=hHh, hAh=hAh, times=times,
                   hessian_per_forward=steady["hessian"] / steady["functional"],
                   gradient_per_forward=steady["gradient"] / steady["functional"],
                   peak_rss_after_tape_gb={"rank0": rss_after_tape[0],
                                           "max": rss_after_tape[1],
                                           "sum": rss_after_tape[2]},
                   peak_rss_gb={"rank0": rss[0], "max": rss[1], "sum": rss[2]},
                   weights=problem.weights,
                   initial_misfit=problem.initial_misfit(),
                   initial_misfit_floor=problem.initial_misfit_floor)
    say(f"  J {J:.10e}; <h,Hh> full objective {hHh:.6e}, <h,Ah> {hAh:.6e}")
    say(f"  functional {times['functional']}, gradient {times['gradient']}, "
        f"hessian {times['hessian']} s; steady H/F {results['hessian_per_forward']:.2f}, "
        f"G/F {results['gradient_per_forward']:.2f}")
    say(f"  peak RSS per rank: max {rss[1]:.2f} GB, sum {rss[2]:.2f} GB "
        f"(after taping: max {rss_after_tape[1]:.2f} GB)")
    say(f"  initial misfit {results['initial_misfit']:.6e}, floor "
        f"{problem.initial_misfit_floor:.6e}; c_d {problem.weights['c_d']:.6e}, "
        f"c_s {problem.weights['c_s']:.6e}")
    finite = all(np.isfinite(v) for v in (J, hHh, hAh))
    return verdict("all numbers finite (0 = yes)", 0.0 if finite else 1.0, 0.0, results)


# ---------------------------------------------------------------------------------
# gn and gn-curvature
# ---------------------------------------------------------------------------------

def gn_directions(Q1):
    """``v``: the smooth-stable field with seed 1; ``w``: a random field with seed 2.

    ``w`` is scaled to a maximum of 0.1. It is drawn per rank, so it is the same
    field only on the same number of ranks; ``gn`` runs in serial.
    """
    from taylor_diagnostics import make_perturbation
    v = make_perturbation(Q1, "smooth-stable", 1)
    w = Function(Q1, name="Random_Direction")
    rng = np.random.default_rng(2)
    w.dat.data[:] = rng.standard_normal(w.dat.data.shape)
    scale = COMM.allreduce(float(np.max(np.abs(w.dat.data_ro))), op=MPI.MAX)
    w.dat.data[:] *= 0.1 / scale
    return v, w


def cylinder_weights(problem, timesteps, Tobs=None, u_last=None):
    """The outer weights of the cylindrical terms, computed from the data alone.

    In ``generate_inverse_problem`` the alpha weights are Real coefficients inside the
    forms, and the outer factors are ``norm_obs / norm_u_surface`` (velocity, one per
    step), 1 (temperature), ``norm_obs / norm_damping`` and ``norm_obs /
    norm_smoothing``, with

        norm_obs       = int Tobs^2 dx,
        norm_u_surface = int_top |uobs_{N-1}|^2 ds,
        norm_damping   = int (Tobs - Taverage)^2 dx,
        norm_smoothing = int |grad(Tobs - Taverage)|^2 dx.

    Args:
        problem: the ``CylindricalProblem``; the process is in its run directory.
        timesteps: the number of steps of the tape.
        Tobs: the observed temperature to use, default the reference state.
        u_last: the observed surface velocity of the last step, default the
            reference state.

    Returns:
        list of ``(description, weight)`` in tape order: the velocity terms of the
        time loop, then the temperature, damping and smoothing terms.
    """
    from firedrake import CheckpointFile
    from gadopt import ds_t
    weights = problem.weights
    with stop_annotating():
        if Tobs is None:
            Tobs = problem.Tobs
        if u_last is None:
            with CheckpointFile("Checkpoint_State.h5", "r") as f:
                u_last = f.load_function(problem.mesh, "Velocity", idx=timesteps - 1)
        Tav = problem.Taverage
        norm_obs = float(assemble(Tobs ** 2 * dx))
        norm_u = float(assemble(dot(u_last, u_last) * ds_t))
        norm_damping = float(assemble((Tobs - Tav) ** 2 * dx))
        norm_smoothing = float(assemble(dot(grad(Tobs - Tav), grad(Tobs - Tav)) * dx))
    out = []
    if weights["alpha_u"] > 0:
        out += [(f"velocity step {n}, norm_obs / norm_u_surface", norm_obs / norm_u)
                for n in range(timesteps)]
    if weights["alpha_T"] > 0:
        out.append(("temperature, 1", 1.0))
    if weights["alpha_d"] > 0:
        out.append(("damping, norm_obs / norm_damping", norm_obs / norm_damping))
    if weights["alpha_s"] > 0:
        out.append(("smoothing, norm_obs / norm_smoothing", norm_obs / norm_smoothing))
    return out


def build_gn_problem(args):
    """The production problem and its expected number of quadratic terms."""
    from gauss_newton import expected_term_count_cylinder
    problem = build_problem(args.rundir, timesteps=args.timesteps, delta_t=args.delta_t,
                            alpha_T=args.alpha_T, alpha_u=args.alpha_u,
                            alpha_d=args.alpha_d, alpha_s=args.alpha_s,
                            make_reference=args.make_reference)
    return problem, expected_term_count_cylinder(problem.weights, args.timesteps)


def check_gn(args, results):
    """G1 to G7, G9, G11 (values) and G13 of the Gauss-Newton product."""
    import gn_checks
    from gauss_newton import GaussNewtonOperator

    if COMM.size != 1:
        raise SystemExit("gn runs in serial")
    t0 = time.perf_counter()
    problem, count = build_gn_problem(args)
    results["time_taping"] = time.perf_counter() - t0
    # The reference state exists now; the second tape of G3 reads it.
    args.make_reference = False
    rf = problem.reduced_functional
    Q1 = problem.Tic.function_space()
    v, w = gn_directions(Q1)
    m0 = Function(Q1).assign(problem.Tic)
    parts = set(args.gn_parts)
    ok = True
    if parts == {"G3"}:
        # The first tape is the tape of G3.
        return check_gn_zero_residual(args, results, problem, count)
    if "G1-G11" not in parts:
        if "G13" in parts:
            ok &= check_gn_predicted_fall(results, problem, count, m0)
        if "G3" in parts:
            problem, count = build_gn_problem(args)
            ok &= check_gn_zero_residual(args, results, problem, count)
        return ok
    with stop_annotating():
        rf(m0)
        operator = GaussNewtonOperator(rf, expected_count=count)
        say("## G1")
        ok &= gn_checks.check_terms(operator, cylinder_weights(problem, args.timesteps),
                                    results)
        say("## G2")
        ok &= gn_checks.check_first_variation(operator, problem.inner_product, results)
        say("## G4 to G7")
        _, metric = build_preconditioner_matrix(problem)
        regularisation = [t for t in operator.terms
                          if t.state.output.function_space()
                          == problem.Tic_true.function_space()
                          and t.state.output.name() == "T_0"]
        ok &= gn_checks.check_products(operator, metric, v, w, results,
                                       has_regularisation=True,
                                       regularisation_terms=regularisation)
        say("## G9")
        ok &= gn_checks.check_stale_guard(problem, operator, v, results)
        say("## G11 (values only)")
        rf(m0)
        values = gn_checks.first_direction_curvature(problem, operator)
        results["G11"] = values
        say(f"    J {values['J']:.10e}, pAp {values['pAp']:.6e}, pHp/pAp "
            f"{values['pHp_over_pAp']:.6g}, pH_GN p/pAp {values['pHGNp_over_pAp']:.6g}, "
            f"dropped {values['dropped_over_pAp']:.6g}")
        ok &= verdict("G11 numbers finite (0 = yes)",
                      0.0 if all(np.isfinite(list(values.values()))) else 1.0, 0.0, results)

    if "G13" in parts:
        ok &= check_gn_predicted_fall(results, problem, count, m0)
    if "G3" in parts:
        problem, count = build_gn_problem(args)
        ok &= check_gn_zero_residual(args, results, problem, count)
    return ok


def check_gn_predicted_fall(results, problem, count, m0):
    """G13 on the cylinder: three one-iteration TAO runs from the first guess."""
    import gn_checks
    rf = problem.reduced_functional
    say("## G13")

    def build_again():
        # One tape for the three runs: each starts at the first guess again.
        with stop_annotating():
            rf(m0)
        return problem, count
    # The cylinder cases: the converged Gauss-Newton solve with a tolerance of 0.5
    # (a product costs about 1.6 forward runs here, and the box already tests the
    # tolerance 1e-2), the solve cut at a small radius, and a full solve of at most
    # three products.
    cases = (("gn, inside the radius", "gn", 1e4, 20, "CONVERGED_RTOL",
              {"tao_bnk_ksp_rtol": 0.5}),
             ("gn, on the radius", "gn", 1e-3, 20, "CONVERGED_STEP_LENGTH"),
             ("full", "full", 0.79, 3, None))
    return gn_checks.check_predicted_fall(build_again, results, cases=cases)


def check_gn_zero_residual(args, results, problem, count):
    """G3 on the cylinder, on a tape of its own, because the data change for good."""
    import gn_checks
    from gauss_newton import GaussNewtonOperator
    say("## G3 (on a new tape, the data change for good)")
    ok = True
    rf = problem.reduced_functional
    Q1 = problem.Tic.function_space()
    v, w = gn_directions(Q1)
    with stop_annotating():
        rf(Function(Q1).assign(problem.Tic))
        first = GaussNewtonOperator(rf, expected_count=count)
        labels, data_terms = gn_checks.zero_residual_cylinder(problem, first)
        operator = GaussNewtonOperator(rf, expected_count=count)
        # The weights after the change, against the data after the change: the new
        # observed temperature is the final temperature, and the new surface
        # velocity of the last step is that of the last mixed state.
        velocity_terms = [t for t in operator.terms if len(t.state.output.function_space()) > 1]
        T_final = problem.T.block_variable.saved_output
        u_final = (velocity_terms[-1].state.saved_output.subfunctions[0]
                   if velocity_terms else None)
        changed = cylinder_weights(problem, args.timesteps, Tobs=T_final, u_last=u_final)
        sub = {}
        ok &= gn_checks.check_terms(operator, changed, sub, check_id="G3-weights")
        results["G3_weights"] = sub
        results.setdefault("tests", []).extend(sub.get("tests", []))
        ok &= gn_checks.check_zero_residual(operator, data_terms, v, w, results)
    return ok


def check_gn_curvature(args, results):
    """Curvature along the first CG direction at the first guess, on any rank count."""
    import gn_checks
    from gauss_newton import GaussNewtonOperator

    t0 = time.perf_counter()
    problem, count = build_gn_problem(args)
    results["time_taping"] = time.perf_counter() - t0
    rss_tape = peak_rss_gb()
    rf = problem.reduced_functional
    with stop_annotating():
        rf(Function(problem.Tic.function_space()).assign(problem.Tic))
        t0 = time.perf_counter()
        operator = GaussNewtonOperator(rf, expected_count=count)
        results["time_operator"] = time.perf_counter() - t0
        values = gn_checks.first_direction_curvature(problem, operator)
    rss = peak_rss_gb()
    results.update(values)
    results.update(gn_terms=operator.describe(), weights=problem.weights,
                   linear_defect=operator.linear_defects,
                   peak_rss_after_tape_gb={"rank0": rss_tape[0], "max": rss_tape[1],
                                           "sum": rss_tape[2]},
                   peak_rss_gb={"rank0": rss[0], "max": rss[1], "sum": rss[2]},
                   control_dofs=problem.Tic.function_space().dim())
    say(f"  J {values['J']:.10e}; pAp {values['pAp']:.6e}; pHp/pAp "
        f"{values['pHp_over_pAp']:.6g}; pH_GN p/pAp {values['pHGNp_over_pAp']:.6g}; "
        f"dropped (pHp - pH_GN p)/pAp {values['dropped_over_pAp']:.6g}")
    say(f"  times: forward {values['time_forward']:.1f} s, gradient "
        f"{values['time_gradient']:.1f} s, full product {values['time_full_product']:.1f} s, "
        f"GN product {values['time_gn_product']:.1f} s; peak RSS per rank max "
        f"{rss[1]:.2f} GB, sum {rss[2]:.2f} GB")
    finite = all(np.isfinite([values[k] for k in ("J", "pAp", "pHp", "pHGNp")]))
    return verdict("all numbers finite (0 = yes)", 0.0 if finite else 1.0, 0.0, results)


# ---------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=("rol-params", "objective", "hvp", "spd", "timing",
                                          "gn", "gn-curvature"))
    parser.add_argument("--rundir", default=None)
    parser.add_argument("--timesteps", type=int, default=5)
    parser.add_argument("--delta-t", type=float, default=1.2e-5)
    parser.add_argument("--make-reference", action="store_true")
    parser.add_argument("--alpha-T", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_T"])
    parser.add_argument("--alpha-u", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_u"])
    parser.add_argument("--alpha-d", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_d"])
    parser.add_argument("--alpha-s", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_s"])
    parser.add_argument("--fd", action="store_true",
                        help="hvp: also check the full objective against a central "
                             "difference of the gradient")
    parser.add_argument("--fd-eps", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument("--json", default=None, help="write the results here")
    parser.add_argument("--gn-parts", nargs="+", default=["G1-G11", "G13", "G3"],
                        choices=("G1-G11", "G13", "G3"),
                        help="gn: the parts to run (default all). G1-G11 is G1, G2, G4 "
                             "to G7, G9 and G11 on one tape; G13 the three TAO runs on "
                             "the same tape; G3 a tape of its own")
    args = parser.parse_args(argv)
    if args.check != "rol-params" and args.rundir is None:
        parser.error("--rundir is needed for this check")
    from pathlib import Path
    json_path = Path(args.json).resolve() if args.json else None

    results = {"check": args.check, "argv": sys.argv, "ranks": COMM.size,
               "timesteps": args.timesteps, "delta_t": args.delta_t}
    t0 = time.perf_counter()
    say(f"### check {args.check}")
    ok = {"rol-params": check_rol_params, "objective": check_objective,
          "hvp": check_hvp, "spd": check_spd, "timing": check_timing,
          "gn": check_gn, "gn-curvature": check_gn_curvature}[args.check](args, results)
    results["pass"] = bool(ok)
    results["wall_time"] = time.perf_counter() - t0
    say(f"### {args.check}: {'PASS' if ok else 'FAIL'} in {results['wall_time']:.1f} s")
    if json_path is not None and COMM.rank == 0:
        json_path.write_text(json.dumps(results, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
