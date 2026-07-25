r"""Summarize the gravitational Poisson DtN scaling study.

Scans a results directory for the JSON sidecars written by
``gravity_cubed_sphere.py`` (coupled solve + correctness anchor) and
``capacitance_gravity.py`` (the wall-free L-cost diagnostic) and prints the
tables to read across. The sidecars are the stable interface; the raw PETSc
logs remain the source of truth. Enforces the per-degree excitation floor here
rather than in the model, so an expensive run is never aborted after producing
its data.

    <python> summarize_gravity_scaling.py --results-dir results/
"""

import argparse
import json
from pathlib import Path

# Pure-Python (no firedrake): safe to import for pytest collection and to run on
# a login node.


def _load(results_dir, pattern):
    out = []
    for path in sorted(Path(results_dir).rglob(pattern)):
        try:
            with open(path) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _grid(rows, row_key, col_key, cell):
    """Render a level x L grid from a list of dict rows."""
    rowvals = sorted({r[row_key] for r in rows})
    colvals = sorted({r[col_key] for r in rows})
    by = {(r[row_key], r[col_key]): r for r in rows}
    width = 10
    head = f"{row_key:>6} |" + "".join(f"{c:>{width}}" for c in colvals)
    lines = [head, "-" * len(head)]
    for rv in rowvals:
        cells = []
        for cv in colvals:
            r = by.get((rv, cv))
            cells.append(f"{cell(r):>{width}}" if r else f"{'-':>{width}}")
        lines.append(f"{rv:>6} |" + "".join(cells))
    return "\n".join(lines)


def summarize_coupled(rows):
    """Iteration counts, wall times and memory, read off the warm phase.

    The warm phase is the right one for the solve-cost tables: it is the same
    arithmetic as the cold phase but without the compilation, so it is what a
    production run inside a time loop or an inversion actually pays per solve.
    The cold phase is read separately, in `summarize_kernel_cost`.
    """
    if not rows:
        return "No coupled-solve sidecars found."
    out = ["=== Coupled solve, warm phase (weak-scaling axis) ===", ""]
    out.append("fieldsplit_0 iterations per invocation "
               "(flat across level = the multigrid scales):")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['fieldsplit_0_iterations_per_invocation']:.2f}"))
    out.append("")
    out.append("fieldsplit_1 iterations (mean) vs 2(L+1):")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['fieldsplit_1_iterations_mean']:.0f}"))
    out.append("")
    out.append("outer FGMRES iterations:")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['outer_iterations']:.0f}"))
    out.append("")
    out.append("steady solve wall time, max over ranks (s) "
               "(flat across level = weak scaling holds):")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['steady_solve_time']:.1f}"))
    out.append("")
    out.append("peak resident memory, max over ranks (GB):")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['peak_memory_gb']:.2f}"))
    return "\n".join(out)


def summarize_accuracy(rows):
    """Error against the closed form, and its convergence with refinement.

    Two things are being read here at once. Down a column, the error should fall
    by about four per level, which is CG1 converging and is the evidence that
    the solve is right - the check that matters in a code area where a parallel
    solve has been shown to converge cleanly to a wrong answer. Along a row, the
    error stops falling once the mesh can no longer resolve the modes being
    asked for, which is the answer to how high a truncation a given resolution
    can carry.
    """
    if not rows:
        return "No coupled-solve sidecars found."
    out = ["=== Accuracy against the closed-form potential ===", ""]
    out.append("relative L2 error:")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['analytic_relative_l2']:.2e}"))
    out.append("")
    out.append("nodal maximum error:")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{r['analytic_nodal_max']:.2e}"))
    out.append("")

    # Convergence ratio between consecutive levels at fixed L. Four is the CG1
    # expectation; substantially less means the angular discretisation, not the
    # radial one, has become the limit at that truncation.
    by = {(r["level"], r["lmax"]): r for r in rows}
    levels = sorted({r["level"] for r in rows})
    truncations = sorted({r["lmax"] for r in rows})
    out.append("error ratio between consecutive levels at fixed L "
               "(4.0 = second order; << 4 means the truncation is "
               "under-resolved):")
    head = f"{'levels':>8} |" + "".join(f"{f'L={t}':>10}" for t in truncations)
    out.append(head)
    out.append("-" * len(head))
    for coarse, fine in zip(levels, levels[1:]):
        cells = []
        for t in truncations:
            a, b = by.get((coarse, t)), by.get((fine, t))
            if a and b and b["analytic_relative_l2"] > 0:
                ratio = a["analytic_relative_l2"] / b["analytic_relative_l2"]
                cells.append(f"{ratio:>10.2f}")
            else:
                cells.append(f"{'-':>10}")
        out.append(f"{f'{coarse}->{fine}':>8} |" + "".join(cells))
    return "\n".join(out)


def summarize_kernel_cost(cold, warm):
    """Decompose the first-solve cost into what caching removes and what it cannot.

    Three regimes, and the differences between them are the measurement:

      cold first solve   TSFC form compilation + C compilation + per-process
                         symbolic work + the solve itself
      warm first solve   per-process symbolic work + the solve itself
      steady solve       the solve itself

    So `cold - warm` is the part an on-disk cache removes, and `warm - steady`
    is the part it cannot: work redone in every fresh interpreter no matter how
    warm the cache is. That second number is the one that matters for a
    production run at high truncation, because it is paid on every process
    start and it grows with the mode count.
    """
    if not cold and not warm:
        return "No sidecars found."
    out = ["=== Kernel generation and per-process startup ===", ""]
    if warm:
        out.append("warm first solve minus steady solve (s) - the per-process "
                   "cost no cache removes:")
        out.append(_grid(warm, "level", "lmax",
                         lambda r: f"{r['first_solve_overhead']:.1f}"))
        out.append("")
    if cold and warm:
        by_warm = {(r["level"], r["lmax"]): r for r in warm}

        def cacheable(r):
            other = by_warm.get((r["level"], r["lmax"]))
            if other is None:
                return "-"
            return f"{r['warmup_time'] - other['warmup_time']:.1f}"

        out.append("cold first solve minus warm first solve (s) - compilation, "
                   "which a warm cache removes:")
        out.append(_grid(cold, "level", "lmax", cacheable))
        out.append("")
    if cold:
        out.append("cold-phase disk cache misses (kernels and forms actually "
                   "generated):")
        out.append(_grid(cold, "level", "lmax",
                         lambda r: f"{r.get('disk_cache_misses', '-')}"))
        out.append("")
        out.append("cold total process time (s), max over ranks:")
        out.append(_grid(cold, "level", "lmax",
                         lambda r: f"{r['total_process_time']:.0f}"))
        out.append("")
    if warm:
        out.append("warm-phase disk cache misses (should be near zero; a large "
                   "number means the cache did not carry between the two "
                   "invocations and the phases are not what they claim):")
        out.append(_grid(warm, "level", "lmax",
                         lambda r: f"{r.get('disk_cache_misses', '-')}"))
    return "\n".join(out)


def summarize_marginal_mode_cost(rows):
    """What one more angular mode costs, in solve time and in startup.

    Reported as the increment between adjacent truncations divided by the
    increment in multiplier count, which is the honest reading of "the cost of
    adding another mode": the boundary assembly is O(L^4) per application, so
    the marginal cost is not constant and a single per-mode figure would
    mislead.
    """
    if not rows:
        return "No coupled-solve sidecars found."
    by = {(r["level"], r["lmax"]): r for r in rows}
    levels = sorted({r["level"] for r in rows})
    truncations = sorted({r["lmax"] for r in rows})
    out = ["=== Marginal cost of an angular mode ===", "",
           "d(steady solve) / d(multiplier count), in milliseconds per mode:"]
    head = f"{'level':>6} |" + "".join(
        f"{f'{a}->{b}':>10}" for a, b in zip(truncations, truncations[1:]))
    out.append(head)
    out.append("-" * len(head))
    for lev in levels:
        cells = []
        for a, b in zip(truncations, truncations[1:]):
            lo, hi = by.get((lev, a)), by.get((lev, b))
            if lo and hi:
                dn = hi["n_multipliers"] - lo["n_multipliers"]
                dt = hi["steady_solve_time"] - lo["steady_solve_time"]
                cells.append(f"{1000 * dt / dn:>10.1f}")
            else:
                cells.append(f"{'-':>10}")
        out.append(f"{lev:>6} |" + "".join(cells))
    out.append("")
    out.append("steady solve time per multiplier (ms), as a level check:")
    out.append(_grid(rows, "level", "lmax",
                     lambda r: f"{1000 * r['steady_solve_time'] / r['n_multipliers']:.1f}"))
    return "\n".join(out)


def summarize_capacitance(rows):
    if not rows:
        return "No capacitance sidecars found."
    out = ["=== Capacitance diagnostic (L-cost axis, all L, wall-free) ===", ""]
    out.append("offline GMRES on S, flat rhs (vs 2(L+1); << n means mild L-cost):")
    out.append(_grid(rows, "level", "lmax", lambda r: f"{r['gmres_flat']}"))
    out.append("")
    out.append("cond(S) (flat in L = high truncation nearly free):")
    out.append(_grid(rows, "level", "lmax", lambda r: f"{r['cond_S']:.1f}"))
    out.append("")
    out.append("2x2 block off-diagonal weight ||S - blockdiag||/||S||:")
    out.append(_grid(rows, "level", "lmax", lambda r: f"{r['block_offdiag_weight']:.1e}"))
    out.append("")
    # Non-convergence or symmetry problems are worth surfacing loudly.
    bad = [r for r in rows if not r.get("gmres_converged", True)]
    if bad:
        out.append("WARNING: offline GMRES did not converge for: "
                   + ", ".join(f"(lvl {r['level']}, L {r['lmax']})" for r in bad))
    return "\n".join(out)


def check_excitation(rows):
    """Enforce the per-degree excitation floor across all coupled runs."""
    problems = []
    for r in rows:
        for bc_id, flags in r.get("boundary_excitation_ok", {}).items():
            starved = [l for l, ok in flags.items() if not ok]
            if starved:
                problems.append(
                    f"(level {r['level']}, L {r['lmax']}, boundary {bc_id}): "
                    f"starved degrees {starved}")
    if not problems:
        return "Excitation floor: OK (every degree excited on every boundary)."
    return "Excitation floor VIOLATIONS:\n  " + "\n  ".join(problems)


def summarize_anchor(rows, tol=1e-6):
    if not rows:
        return "No correctness-anchor sidecars found."
    out = ["=== Correctness anchor (direct vs iterative) ==="]
    for r in sorted(rows, key=lambda r: (r["level"], r["lmax"])):
        rel = r["relative_difference"]
        verdict = "OK" if rel < tol else "FAIL"
        out.append(f"  level {r['level']}, L {r['lmax']}: "
                   f"rel diff {rel:.3e}  [{verdict}]")
    return "\n".join(out)


def main(results_dir):
    all_coupled = _load(results_dir, "summary_*.json")
    failed = [r for r in all_coupled if "failed" in r]
    all_coupled = [r for r in all_coupled
                   if "fieldsplit_0_iterations_per_invocation" in r]
    # The main grids read the iterative variant; selfp sidecars carry the same
    # (level, lmax) keys and would silently overwrite iterative rows in the grid.
    coupled = [r for r in all_coupled if r.get("variant") == "iterative"]
    # Each case is run twice against different cache states. Splitting them is
    # not cosmetic: both phases carry the same (level, lmax) key, so a single
    # grid would silently show whichever sidecar sorted last, and every cost
    # table would be a coin toss between two numbers that differ by the entire
    # compilation cost.
    warm = [r for r in coupled if r.get("cache_phase") in (None, "warm")]
    cold = [r for r in coupled if r.get("cache_phase") == "cold"]
    capacitance = _load(results_dir, "capacitance_*.json")
    anchor = _load(results_dir, "anchor_*.json")

    print(summarize_coupled(warm))
    print()
    print(summarize_accuracy(warm))
    print()
    print(summarize_kernel_cost(cold, warm))
    print()
    print(summarize_marginal_mode_cost(warm))
    print()
    if capacitance:
        print(summarize_capacitance(capacitance))
        print()
    excitation = check_excitation(warm)
    print(excitation)
    print()
    print(summarize_anchor(anchor))
    if failed:
        print()
        print("CASES THAT FAILED (recorded rather than lost):")
        for r in sorted(failed, key=lambda r: (r["level"], r["lmax"])):
            print(f"  level {r['level']}, L {r['lmax']}, phase "
                  f"{r.get('cache_phase')}: {r['failed'][:160]}")
    # Nonzero exit on a floor violation so a CI wrapper can fail on it.
    return 1 if "VIOLATION" in excitation else 0


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results",
                        help="directory to scan for sidecar JSON files")
    args = parser.parse_args()
    sys.exit(main(args.results_dir))
