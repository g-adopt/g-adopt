"""Checks of the Gauss-Newton product on the box (NOTES/team/gauss-newton/PLAN.md 2.2).

Usage (from a directory outside every firedrake source tree):

    # G1 to G10, G12, G13 and the switch smoke on the tiny case
    python check_gauss_newton.py TINY_STATE.h5 --json box-gn.json

    # G11 and one Gauss-Newton solve on the 64-cell state (window 80)
    python check_gauss_newton.py --g11 REFERENCE_STATE.h5 --json box-g11.json

    # The prices of the box rule: taping, gradient, 5 full and 5 GN products
    python check_gauss_newton.py --timing REFERENCE_STATE.h5 --json timing.json

The tiny state is ``forward.py --steps 10 --resolution 16``. The tiny checks run with
two objectives:

- ``alpha0``: the misfit alone, the metric of the gate-1 command (``--alpha-d 0
  --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3``), free walls. One term.
- ``tutorial``: the tutorial regularisation in the objective (``--alpha-d 1e-3
  --alpha-s 1e-3``) with the settings of the box runs, pinned walls and the
  Dirichlet operator. Three terms, and the metric equals the Hessian of the two
  regularisation terms.

Directions: ``v`` is the smooth field of ``check_box_hessian.smooth_direction`` and
``w`` a random field with seed 1, both zero on the wall nodes and of maximum 0.1.

The process exits with status 1 when any check fails. Every measured value goes to
the JSON file.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOX = HERE.parent
sys.path.insert(0, str(BOX))
sys.path.insert(0, str(HERE))

from firedrake import Function, assemble, dot, dx, grad  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from pyadjoint import stop_annotating  # noqa: E402

import gn_checks as gc_  # noqa: E402
import optimisers  # noqa: E402
from check_box_hessian import smooth_direction  # noqa: E402
from gauss_newton import GaussNewtonOperator, expected_term_count_box  # noqa: E402
from gn_checks import say, verdict  # noqa: E402
from problem import build_problem  # noqa: E402

# The objectives of the tiny checks. ``window`` 5 of the 10 steps, as gate 1.
TINY = {
    "alpha0": dict(window=5, alpha_d=0.0, alpha_s=0.0, inner_product="regularisation",
                   metric_alpha_d=1e-3, metric_alpha_s=1e-3, initial_guess="observed"),
    "tutorial": dict(window=5, alpha_d=1e-3, alpha_s=1e-3, inner_product="regularisation",
                     initial_guess="observed", wall_treatment="pinned",
                     prior_wall_bc="dirichlet"),
}
# The 64-cell objective of G11 and of the box runs.
BOX64 = dict(window=80, alpha_d=1e-3, alpha_s=1e-3, inner_product="regularisation",
             initial_guess="observed", wall_treatment="pinned", prior_wall_bc="dirichlet")
# The targets of G11 (REVIEW-PLAN.md, measurement M1): pH_GN p / pAp and pHp / pAp.
G11_TARGETS = {"pHGNp_over_pAp": 273.5, "pHp_over_pAp": 91.4}

# The gate-1 command of the box campaign (gates_v2.slurm), which G0, G12 and the
# switch smoke start from.
GATE1 = ["--optimiser", "tao-bnls-riesz", "--riesz", "regularisation", "--window", "5",
         "--alpha-d", "0", "--alpha-s", "0", "--metric-alpha-d", "1e-3",
         "--metric-alpha-s", "1e-3", "--initial-guess", "observed",
         "--initial-radius", "1.0"]


def directions(problem):
    """``v`` (smooth) and ``w`` (random, seed 1), zero on the walls, maximum 0.1."""
    V = problem.Tic.function_space()
    v = smooth_direction(V)
    w = Function(V, name="Random_Direction")
    rng = np.random.default_rng(1)
    w.dat.data[:] = rng.standard_normal(w.dat.data.shape)
    for nodes in problem.wall_nodes.values():
        w.dat.data[nodes] = 0.0
    w.dat.data[:] *= 0.1 / np.max(np.abs(w.dat.data_ro))
    return v, w


def expected_weights(problem, config):
    """The weights of the box terms from the data, computed apart from the tape.

    ``c_d = alpha_d int Tobs^2 / int Taverage^2`` and ``c_s = alpha_s int Tobs^2 /
    int |grad Tobs|^2``, with the degree-6 quadrature of ``problem.py``. The misfit
    has weight 1.

    Returns:
        list of ``(description, weight)`` in tape order.
    """
    measure = dx(degree=6)
    with stop_annotating():
        norm_obs = float(assemble(problem.Tobs ** 2 * measure))
        norm_damping = float(assemble(problem.Taverage ** 2 * measure))
        norm_smoothing = float(assemble(dot(grad(problem.Tobs), grad(problem.Tobs))
                                        * measure))
    out = [("temperature misfit", 1.0)]
    if config["alpha_d"] > 0:
        out.append(("damping, alpha_d norm_obs / norm_damping",
                    config["alpha_d"] * norm_obs / norm_damping))
    if config["alpha_s"] > 0:
        out.append(("smoothing, alpha_s norm_obs / norm_smoothing",
                    config["alpha_s"] * norm_obs / norm_smoothing))
    return out


def run_objective_checks(state, name, config, results):
    """G1 to G10 and G3 for one objective of the tiny case."""
    from riesz_hessian import build_preconditioner_matrix
    say(f"## objective {name}")
    problem = build_problem(state, **config)
    rf = problem.reduced_functional
    count = expected_term_count_box(problem.weights)
    v, w = directions(problem)
    ok = True
    with stop_annotating():
        rf(problem.control.tape_value())
        operator = GaussNewtonOperator(rf, expected_count=count)
        ok &= gc_.check_terms(operator, expected_weights(problem, config), results)
        ok &= gc_.check_first_variation(operator, problem.inner_product, results)
        _, metric = build_preconditioner_matrix(problem)
        regularisation = operator.terms[1:] if len(operator.terms) > 1 else None
        fixed = None
        if config.get("wall_treatment") == "pinned":
            fixed = np.concatenate([np.asarray(n, dtype=int)
                                    for n in problem.wall_nodes.values()])
        ok &= gc_.check_products(operator, metric, v, w, results,
                                 has_regularisation=config["alpha_d"] > 0,
                                 regularisation_terms=regularisation,
                                 fixed_nodes=fixed)
        ok &= gc_.check_tao_path(problem, operator, v, results)
        ok &= gc_.check_stale_guard(problem, operator, v, results)
        ok &= gc_.check_counts(problem, v, results, expected_count=count)

    # G3 on a new tape, because the data change for good.
    problem = build_problem(state, **config)
    rf = problem.reduced_functional
    v, w = directions(problem)
    with stop_annotating():
        rf(problem.control.tape_value())
        data_terms = gc_.zero_residual_box(problem)
        operator = GaussNewtonOperator(rf, expected_count=count)
        results["G3_weights_after_change"] = [t.weight for t in operator.terms]
        ok &= gc_.check_zero_residual(operator, data_terms, v, w, results)
    return ok


# ---------------------------------------------------------------------------------
# G12 and the switch smoke: whole runs of inverse.py, in a subprocess
# ---------------------------------------------------------------------------------

def run_inverse(state, extra, output, log):
    """Run ``inverse.py`` on the tiny state with the gate-1 command and ``extra``.

    A subprocess, so that PETSc's report of unused options at the end of the process
    is in the log, and so that the runs start from a fresh tape.

    Returns:
        ``(record, log text, return code)``
    """
    command = [sys.executable, str(BOX / "inverse.py"), "--checkpoint", str(state),
               *GATE1, *extra, "--output", str(output)]
    with open(log, "w") as f:
        code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                              env=os.environ.copy()).returncode
    text = Path(log).read_text()
    record = json.loads(Path(output).read_text()) if Path(output).exists() else None
    return record, text, code


def all_solves(record):
    """Every Krylov solve of a record, in order."""
    out = []
    for row in record["iterations"]:
        out.extend(row.get("krylov_solves", []) or [])
    out.extend(record.get("krylov_solves_unassigned", []) or [])
    return out


def unused_petsc_options(text):
    """The options of PETSc's ``Option left`` report that are not driver arguments."""
    names = []
    for line in text.splitlines():
        if line.startswith("Option left: name:"):
            name = line.split("name:", 1)[1].split()[0]
            if not name.startswith("--"):
                names.append(name)
    return names


def ew_expected(solves, rtol0=0.5, rtolmax=0.5, alpha=0.5 * (1.0 + 5.0 ** 0.5),
                gamma=1.0, threshold=0.1):
    """The tolerance of each solve by PETSc's Eisenstat-Walker version 2.

    ``KSPPreSolve_SNESEW`` (snes.c) with the TAO pre-solve (taosolver.c), which sets
    the iteration count to the TAO iteration and the norm to ``|b|_2``:

    - TAO iteration 0: ``rtol0``;
    - else ``gamma (|b_k| / |b_{k-1}|)^alpha``, raised to
      ``gamma rtol_{k-1}^alpha`` when that bound is above ``threshold``;
    - always capped at ``rtolmax``.

    ``|b_{k-1}|`` and ``rtol_{k-1}`` are those of the previous solve (the post-solve
    stores them), zero before the first solve.

    Returns:
        list of float.
    """
    out = []
    previous_norm, previous_rtol = 0.0, 0.0
    for solve in solves:
        if solve["tao_iteration"] == 0:
            rtol = rtol0
        else:
            rtol = (float("inf") if previous_norm == 0.0
                    else gamma * (solve["rhs_norm"] / previous_norm) ** alpha)
            bound = gamma * previous_rtol ** alpha
            if bound > threshold:
                rtol = max(rtol, bound)
        out.append(min(rtol, rtolmax))
        previous_norm, previous_rtol = solve["rhs_norm"], solve["rtol"]
    return out


def check_ew(state, workdir, results):
    """G12: the Krylov solves of a ``--krylov-ew`` run use the Eisenstat-Walker tolerance."""
    say("## G12: Eisenstat-Walker tolerance")
    record, text, code = run_inverse(
        state, ["--hessian", "gn", "--krylov-ew", "--tao-opt", "tao_bnk_imax=0",
                "--max-it", "3"], workdir / "g12-ew.json", workdir / "g12-ew.log")
    ok = verdict(results, "G12", "inverse.py --krylov-ew exit status", code, 0, mode="equal")
    if record is None:
        return False
    solves = all_solves(record)
    rtols = [s["rtol"] for s in solves]
    expected = ew_expected(solves)
    results["G12_solves"] = [{k: s.get(k) for k in ("tao_iteration", "rtol", "rhs_norm",
                                                     "iterations", "reason_name",
                                                     "hessian_product")}
                             for s in solves]
    results["G12_expected_rtol"] = expected
    ok &= verdict(results, "G12", "number of solves", len(solves), 2, mode="above")
    ok &= verdict(results, "G12", "rtol of solve 0", rtols[0], 0.5, mode="equal")
    worst = max(abs(a - b) / b for a, b in zip(rtols, expected))
    ok &= verdict(results, "G12", "largest relative difference of rtol from version 2",
                  worst, 1e-12)
    ok &= verdict(results, "G12", "solves with rtol 1e-2",
                  sum(1 for r in rtols if r == 1e-2), 0, mode="equal")
    unused = unused_petsc_options(text)
    results["G12_unused_options"] = unused
    ok &= verdict(results, "G12", "PETSc options left unused", len(unused), 0,
                  mode="equal")
    # The refusal of an option set that turns the forcing term off again.
    params = optimisers.tao_parameters("tao-bnls-riesz", 3, 1.0,
                                       {"tao_bnk_imax": 0, "tao_ksp_ew": "false"},
                                       krylov_ew=True)
    raised = None
    try:
        optimisers.check_newton_options(params, "tao-bnls-riesz", hessian="gn",
                                        krylov_ew=True)
    except ValueError as error:
        raised = str(error)
    results["G12_refusal"] = raised
    ok &= verdict(results, "G12", "run_tao refuses krylov_ew with tao_ksp_ew false",
                  raised is not None, True, mode="true")
    params = optimisers.tao_parameters("tao-bnls-riesz", 3, 1.0,
                                       {"tao_bnk_imax": 0, "tao_bnk_ksp_ew_rtol0": 0.3},
                                       krylov_ew=True)
    raised = None
    try:
        optimisers.check_newton_options(params, "tao-bnls-riesz", hessian="gn",
                                        krylov_ew=True)
    except ValueError as error:
        raised = str(error)
    ok &= verdict(results, "G12", "run_tao refuses krylov_ew with rtol0 changed",
                  raised is not None, True, mode="true")
    return ok


def check_switch_smoke(state, workdir, results):
    """The switch smoke: gate 1 with ``gn-then-full`` and switch iteration 1."""
    say("## switch smoke: gn-then-full, switch iteration 1, 3 iterations")
    record, text, code = run_inverse(
        state, ["--hessian", "gn-then-full", "--gn-switch-it", "1", "--tao-opt",
                "tao_bnk_imax=0", "--max-it", "3"],
        workdir / "smoke-switch.json", workdir / "smoke-switch.log")
    ok = verdict(results, "smoke", "inverse.py gn-then-full exit status", code, 0,
                 mode="equal")
    if record is None:
        return False
    solves = all_solves(record)
    modes = [s.get("hessian_product") for s in solves]
    reasons = [s.get("reason_name") for s in solves]
    results["smoke_modes"] = modes
    results["smoke_reasons"] = reasons
    results["smoke_switches"] = record.get("hessian_switches")
    # The expected modes: gn first, then full, except one gn after a full solve that
    # stopped on negative curvature (the fall-back).
    expected = []
    for k in range(len(solves)):
        if k == 0:
            expected.append("gn")
        elif modes[k - 1] == "full" and reasons[k - 1] == "CONVERGED_NEG_CURVE":
            expected.append("gn")
        else:
            expected.append("full")
    ok &= verdict(results, "smoke", "solve modes", str(modes), str(expected), mode="equal")
    ok &= verdict(results, "smoke", "number of solves", len(solves), 3, mode="equal")
    ok &= verdict(results, "smoke", "switch entries in hessian_switches",
                  len(record.get("hessian_switches") or []), 0, mode="above")
    ok &= verdict(results, "smoke", "recording_errors", len(record["recording_errors"]), 0,
                  mode="equal")
    gn_its = sum(s["iterations"] for s in solves if s.get("hessian_product") == "gn")
    full_its = sum(s["iterations"] for s in solves if s.get("hessian_product") == "full")
    results["smoke_counts"] = {"gn_hessian": record["gn_hessian"],
                               "hessian": record["hessian"], "gn_solve_iterations": gn_its,
                               "full_solve_iterations": full_its}
    ok &= verdict(results, "smoke", "gn_hessian equals the iterations of the gn solves",
                  record["gn_hessian"], gn_its, mode="equal")
    ok &= verdict(results, "smoke", "hessian equals the iterations of the full solves",
                  record["hessian"], full_its, mode="equal")
    return ok


# ---------------------------------------------------------------------------------
# G13: the predicted-fall bookkeeping
# ---------------------------------------------------------------------------------

def check_predicted_fall(state, results):
    """G13 on the tiny case with the tutorial objective (``gn_checks.check_predicted_fall``)."""
    def build():
        problem = build_problem(state, **TINY["tutorial"])
        return problem, expected_term_count_box(problem.weights)
    return gc_.check_predicted_fall(build, results)


# ---------------------------------------------------------------------------------
# G11 and the timing of the box rule
# ---------------------------------------------------------------------------------

def check_g11(state, workdir, results):
    """G11 on the 64-cell state, then one Gauss-Newton solve at radius 1e4, cap 8."""
    say("## G11: curvature along p = A^-1 g at the first guess, 64 cells, window 80")
    t0 = time.perf_counter()
    problem = build_problem(state, **BOX64)
    results["time_taping"] = time.perf_counter() - t0
    rf = problem.reduced_functional
    with stop_annotating():
        rf(problem.control.tape_value())
        operator = GaussNewtonOperator(rf, expected_count=expected_term_count_box(
            problem.weights))
        values = gc_.first_direction_curvature(problem, operator)
    results["G11"] = values
    say(f"    J {values['J']:.10e}, pAp {values['pAp']:.6e}, pHp/pAp "
        f"{values['pHp_over_pAp']:.6g}, pH_GN p/pAp {values['pHGNp_over_pAp']:.6g}, "
        f"dropped {values['dropped_over_pAp']:.6g}")
    ok = True
    for key, target in G11_TARGETS.items():
        ok &= verdict(results, "G11", f"{key} against {target} (3 digits)",
                      abs(values[key] - target) / target, 5e-3)

    say("## G11: one Gauss-Newton solve, radius 1e4, Krylov cap 8")
    with stop_annotating():
        recorder = optimisers.run_tao(
            problem, "tao-bnls-riesz", max_it=1, initial_radius=1e4,
            extra_options={"tao_bnk_ksp_max_it": 8, "tao_bnk_imax": 0},
            hessian="gn", gn_expected_terms=expected_term_count_box(problem.weights),
            output=workdir / "g11-gn-solve.json")
    rows = recorder.iterations
    solve = rows[-1]["krylov_solves"][0] if rows[-1].get("krylov_solves") else None
    summary = {"J_before": rows[0]["J"], "J_after": rows[-1]["J"],
               "step_length": rows[-1].get("step_length"),
               "gn_hessian": recorder.gn_hessian, "forward": recorder.forward,
               "adjoint": recorder.adjoint, "recording_errors":
               recorder.final_diagnostics.get("recording_errors")}
    if solve is not None:
        summary.update({k: solve.get(k) for k in ("iterations", "reason_name",
                                                   "ritz_values", "rtol", "step_norm",
                                                   "rhs_dot_sol", "cg_objective")})
    results["G11_solve"] = summary
    say(f"    {json.dumps(summary, default=str)}")
    ok &= verdict(results, "G11", "the GN solve ran without recording errors",
                  not summary["recording_errors"], True, mode="true")
    return ok


def run_timing(state, results, products=5, config=None):
    """The prices of the box rule, measured in one process on one node.

    Taping, two forward calls, two gradients, ``products`` full products and
    ``products`` Gauss-Newton products at the first guess, in that order, in the
    smooth direction ``v``. The times are those of the ``Recorder`` of the runs,
    taken by the same callbacks, so that the prices measure what a run's work
    counter measures (the garbage collection after a product is outside both
    times). The first call of each kind builds solvers and kernels, so the medians
    over the calls after the first are the prices:

    - ``full_per_forward = median(full product) / median(forward)``, the price of a
      full product in the work of the rule;
    - ``gn_over_full = median(GN product) / median(full product)``, ``r_GN``.

    Returns:
        bool (True when every number is finite)
    """
    say("## timing")
    config = BOX64 if config is None else config
    t0 = time.perf_counter()
    problem = build_problem(state, **config)
    results["time_taping"] = time.perf_counter() - t0
    rf = problem.reduced_functional
    v, _ = directions(problem)
    recorder = optimisers.Recorder(problem)
    recorder.attach()
    m = problem.control.tape_value()
    with stop_annotating():
        for _ in range(2):
            rf(m)
            rf.derivative(apply_riesz=False)
        for _ in range(products):
            rf.hessian(v, apply_riesz=False)
        operator = GaussNewtonOperator(rf, expected_count=expected_term_count_box(
            problem.weights), recorder=recorder)
        for _ in range(products):
            operator.action(v)
    times = {"forward": recorder.forward_times, "gradient": recorder.gradient_times,
             "full": recorder.hessian_times, "gn": recorder.gn_hessian_times}
    steady = {k: float(np.median(vals[1:])) for k, vals in times.items()}
    results.update(times=times, steady_median=steady,
                   forward_time=steady["forward"],
                   gradient_per_forward=steady["gradient"] / steady["forward"],
                   full_per_forward=steady["full"] / steady["forward"],
                   gn_per_forward=steady["gn"] / steady["forward"],
                   gn_over_full=steady["gn"] / steady["full"],
                   peak_rss_gb=gc_.peak_rss_gb(), config={k: v for k, v in config.items()})
    say(f"    steady medians {steady}; full_per_forward {results['full_per_forward']:.4f}, "
        f"gn_over_full {results['gn_over_full']:.4f}, peak RSS {results['peak_rss_gb']:.2f} GB")
    return all(np.isfinite([results["full_per_forward"], results["gn_over_full"]]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("state", nargs="?", default=None, help="tiny reference state")
    parser.add_argument("--g11", default=None, help="64-cell reference state for G11")
    parser.add_argument("--timing", default=None, help="64-cell reference state for timing")
    parser.add_argument("--timing-tiny", action="store_true",
                        help="--timing on the tiny objective (a smoke of the code path)")
    parser.add_argument("--json", default=None, help="write the results here")
    parser.add_argument("--only", nargs="*", default=None,
                        help="tiny case: run only these parts (alpha0, tutorial, G12, "
                             "G13, smoke)")
    args = parser.parse_args(argv)
    workdir = Path.cwd()
    results = {"argv": sys.argv, "petsc": ".".join(map(str, PETSc.Sys.getVersion()))}
    t0 = time.perf_counter()
    ok = True
    if args.g11:
        ok &= check_g11(str(Path(args.g11).resolve()), workdir, results)
    elif args.timing:
        config = None
        if args.timing_tiny:
            config = TINY["tutorial"]
        ok &= run_timing(str(Path(args.timing).resolve()), results, config=config)
    else:
        if args.state is None:
            parser.error("give the tiny state, --g11 or --timing")
        state = str(Path(args.state).resolve())
        parts = args.only or ["alpha0", "tutorial", "G12", "G13", "smoke"]
        for name in ("alpha0", "tutorial"):
            if name in parts:
                sub = {}
                ok &= run_objective_checks(state, name, TINY[name], sub)
                results[name] = sub
        if "G12" in parts:
            ok &= check_ew(state, workdir, results.setdefault("G12", {}))
        if "G13" in parts:
            ok &= check_predicted_fall(state, results.setdefault("G13", {}))
        if "smoke" in parts:
            ok &= check_switch_smoke(state, workdir, results.setdefault("smoke", {}))
    results["pass"] = bool(ok)
    results["wall_time"] = time.perf_counter() - t0
    say(f"### check_gauss_newton: {'PASS' if ok else 'FAIL'} in {results['wall_time']:.1f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
