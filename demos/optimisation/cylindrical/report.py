"""The per-run tables and the verdicts of the cylindrical comparison, as Markdown.

Usage:
    python report.py --reference rol-lbfgs-L2.json --newton tao-zero.json \
        [--rol-metric rol-lbfgs-A.json] [--levels 100 200 300 400] > tables.md

The verdict comes from ``summarise.py``, the decision rule fixed before the runs. This
script prints, beside the rule and without changing it:

- J, the misfit term and the regularisation term of every run at each level of work
  and at the end of the run (work at the pooled prices of the rule);
- the reverse reading: the work at which ROL reaches the J that Newton has at ``W*``;
- the work to reach 50, 90 and 99 percent of ``J0 - J_min``, with ``J_min`` the lowest
  J that any run reached;
- the fall of J over the last 50 work units of each run;
- the Krylov record of the Newton run: iterations per solve, solves at the cap, solves
  that stop on the trust radius, negative-curvature stops with ``pHp/pAp``, shifts;
- ``initial_misfit`` at the end, as support only (the minimiser of P is not the truth);
- the final state of each run (sweep counts, stop reason).

It only reads the records.
"""

import argparse
import io
from collections import Counter
from contextlib import redirect_stdout

import summarise


def solves(record):
    """Every Krylov solve of a TAO record, in order, with its iteration number."""
    out = []
    for row in record["iterations"]:
        for solve in row.get("krylov_solves", []) or []:
            out.append(dict(solve, iteration=row["iteration"]))
    out.extend(record.get("krylov_solves_unassigned", []) or [])
    return out


def terms_table(rows, levels):
    """J, misfit term and regularisation term at each level and at the end."""
    lines = ["| run | at work | iteration | work | J | misfit term | regularisation term |",
             "|---|---|---|---|---|---|---|"]
    for name, rr in rows.items():
        for level in list(levels) + ["end"]:
            if level == "end":
                row = rr[-1]
            else:
                eligible = [r for r in rr if r["work_priced"] <= level]
                if not eligible or rr[-1]["work_priced"] < level:
                    # The run never reached this level; the last row below it would be
                    # its end, which the "end" line already gives.
                    lines.append(f"| {name} | {level:g} | not reached | | | | |")
                    continue
                row = eligible[-1]
            lines.append(f"| {name} | {level if level == 'end' else f'{level:g}'} | "
                         f"{row['iteration']} | {row['work_priced']:.1f} | {row['J']:.6e} | "
                         f"{row.get('misfit_term', float('nan')):.6e} | "
                         f"{row.get('prior_term', float('nan')):.6e} |")
    return "\n".join(lines)


def reverse_reading(rows, w_star):
    """Work at which the ROL run reaches the J that the Newton run has at W*."""
    j_newton, newton_row = summarise.value_at(rows["P-Newton"], w_star)
    crossing = summarise.work_to_reach(rows["P-ROL-L2"], j_newton)
    head = (f"Newton's J at W* = {w_star:g}: {j_newton:.6e} (iteration "
            f"{newton_row['iteration']}, work {newton_row['work_priced']:.1f}). ")
    if crossing is None:
        last = rows["P-ROL-L2"][-1]
        return head + (f"P-ROL-L2 never reaches it; its last J is {last['J']:.6e} at "
                       f"work {last['work_priced']:.1f}.")
    return head + f"P-ROL-L2 reaches it between work {crossing[0]:.1f} and {crossing[1]:.1f}."


def fraction_table(rows):
    """Work to reach 50, 90 and 99 percent of J0 - J_min, per run, as intervals."""
    j0 = rows["P-ROL-L2"][0]["J"]
    j_min = min(r["J"] for rr in rows.values() for r in rr)
    lines = [f"J0 = {j0:.6e}, lowest J of any run = {j_min:.6e}.", "",
             "| run | 50 % | 90 % | 99 % |", "|---|---|---|---|"]
    for name, rr in rows.items():
        cells = []
        for fraction in (0.5, 0.9, 0.99):
            target = j0 - fraction * (j0 - j_min)
            crossing = summarise.work_to_reach(rr, target)
            cells.append("not reached" if crossing is None
                         else f"{crossing[0]:.1f} to {crossing[1]:.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def last_fall(rows, window=50.0):
    """The fall of J over the last ``window`` work units of each run."""
    lines = ["| run | work at end | J at end | J at end - 50 | relative fall |",
             "|---|---|---|---|---|"]
    for name, rr in rows.items():
        end = rr[-1]
        j_before, before = summarise.value_at(rr, end["work_priced"] - window)
        if before is None:
            lines.append(f"| {name} | {end['work_priced']:.1f} | {end['J']:.6e} | "
                         "run shorter than 50 | |")
            continue
        lines.append(f"| {name} | {end['work_priced']:.1f} | {end['J']:.6e} | "
                     f"{j_before:.6e} (work {before['work_priced']:.1f}) | "
                     f"{(j_before - end['J']) / j_before:.3e} |")
    return "\n".join(lines)


def krylov_summary(record):
    """Krylov iterations per solve, caps, radius stops, shifts and negative curvature."""
    all_solves = solves(record)
    if not all_solves:
        return "No Krylov solves recorded."
    reasons = Counter(s.get("reason_name") for s in all_solves)
    caps = sum(1 for s in all_solves if s.get("at_cap"))
    radius = reasons.get("CONVERGED_STEP_LENGTH", 0)
    shifts = sorted({s.get("hessian_shift", 0.0) for s in all_solves})
    its = [s.get("iterations", 0) for s in all_solves]
    lines = [f"{len(all_solves)} Krylov solves, {sum(its)} Krylov iterations "
             f"(each one Hessian-vector product). Reasons: {dict(reasons)}. At the cap: "
             f"{caps}. Stopped on the trust radius: {radius}. Shifts seen: {shifts}.", "",
             "| iteration | Krylov its | reason | at cap | step norm | trust radius | "
             "ritz min | ritz max | active |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in all_solves:
        def fmt(value, spec):
            return format(value, spec) if isinstance(value, (int, float)) else str(value)
        lines.append(f"| {s.get('iteration')} | {s.get('iterations')} | "
                     f"{s.get('reason_name')} | {s.get('at_cap')} | "
                     f"{fmt(s.get('step_norm'), '.4g')} | {fmt(s.get('trust_radius'), '.4g')} | "
                     f"{fmt(s.get('ritz_min'), '.4g')} | {fmt(s.get('ritz_max'), '.4g')} | "
                     f"{s.get('active')} |")
    negative = [s for s in all_solves if s.get("negative_curvature")]
    lines += ["", f"Negative-curvature stops: {len(negative)}."]
    if negative:
        lines += ["", "| iteration | Krylov its | pHp | pAp | pHp/pAp | shift |",
                  "|---|---|---|---|---|---|"]
        for s in negative:
            nc = s["negative_curvature"]
            lines.append(f"| {s['iteration']} | {s.get('iterations')} | {nc['pHp']:.4e} | "
                         f"{nc['pAp']:.4e} | {nc['pHp_over_pAp']:.4g} | {nc['pert']:.3g} |")
    errors = record.get("recording_errors") or []
    lines.append(f"\nRecording errors: {errors if errors else 'none'}.")
    return "\n".join(lines)


def final_table(records, rows):
    """Final state of each run."""
    lines = ["| run | iterations | J | initial_misfit (support only) | work (own) | "
             "work (pooled) | fwd / adj / HVP | wall (h) | stop |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, record in records.items():
        r = record["iterations"][-1]
        lines.append(f"| {name} | {r['iteration']} | {r['J']:.6e} | "
                     f"{r['initial_misfit']:.4e} | {r['work']:.1f} | "
                     f"{rows[name][-1]['work_priced']:.1f} | {record['forward']} / "
                     f"{record['adjoint']} / {record['hessian']} | "
                     f"{record.get('wall_time', 0.0) / 3600:.1f} | "
                     f"{record.get('stop_reason', '')} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--newton", required=True)
    parser.add_argument("--rol-metric", default=None)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=list(summarise.LEVELS))
    args = parser.parse_args(argv)

    paths = {"P-ROL-L2": args.reference, "P-Newton": args.newton}
    if args.rol_metric:
        paths["P-ROL-A"] = args.rol_metric
    records = {name: summarise.load(path) for name, path in paths.items()}
    gp, hp, medians = summarise.pooled_prices(list(records.values()))
    rows = {name: summarise.priced_rows(r, gp, hp) for name, r in records.items()}
    reached = min(rr[-1]["work_priced"] for rr in rows.values())
    candidates = [level for level in sorted(args.levels) if level <= reached]
    w_star = candidates[-1] if candidates else None

    print("## Verdicts (summarise.py, the rule)\n")
    summary_argv = ["--reference", args.reference, "--newton", args.newton,
                    "--levels", *map(str, args.levels)]
    if args.rol_metric:
        summary_argv += ["--rol-metric", args.rol_metric]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summarise.main(summary_argv)
    print("```\n" + buffer.getvalue().rstrip() + "\n```")

    print(f"\n## Beside the rule\n\nPooled medians: forward {medians['forward']:.1f} s, "
          f"gradient {medians['gradient']:.1f} s, product "
          f"{medians['hessian'] or float('nan'):.1f} s; prices: gradient {gp:.3f}, "
          f"product {(hp or float('nan')):.3f} forward-equivalents. All work below is at "
          "these prices.\n")
    print("### Final state\n")
    print(final_table(records, rows))
    print("\n### J and its two parts at each level of work\n")
    print(terms_table(rows, args.levels))
    if w_star is not None:
        print("\n### Reverse reading\n")
        print(reverse_reading(rows, w_star))
    print("\n### Work to 50, 90 and 99 percent of J0 - J_min\n")
    print(fraction_table(rows))
    print("\n### Fall of J over the last 50 work units\n")
    print(last_fall(rows))
    print("\n### Krylov record of P-Newton\n")
    print(krylov_summary(records["P-Newton"]))


if __name__ == "__main__":
    main()
