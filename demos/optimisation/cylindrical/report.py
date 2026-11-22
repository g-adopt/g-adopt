"""The per-run table and the verdicts of the cylindrical comparison, as Markdown.

Usage:
    python report.py --reference rol-lbfgs-L2.json --newton tao-zero.json \
        --rol-metric rol-lbfgs-A.json [--levels 50 100 150 200] > tables.md

The verdicts come from ``summarise.py``, the decision rule fixed before the runs. This
script adds the per-run table: J and its four terms, ``initial_misfit``, the work
(the run's own count and the pooled re-priced count), the sweep counts, the stop reason,
and for the Newton run the Krylov reasons, the negative-curvature solves with their
``pHp/pAp``, the shifts and the caps reached. It only reads the records.
"""

import argparse
import io
from collections import Counter
from contextlib import redirect_stdout

import summarise


def last_row(record):
    """The last iteration row of a record."""
    return record["iterations"][-1]


def solves(record):
    """Every Krylov solve of a TAO record, in order, with its iteration number."""
    out = []
    for row in record["iterations"]:
        for solve in row.get("krylov_solves", []) or []:
            out.append(dict(solve, iteration=row["iteration"]))
    out.extend(record.get("krylov_solves_unassigned", []) or [])
    return out


def run_table(records, rows):
    """Markdown table of the final state of each run."""
    lines = ["| run | iterations | J | temperature | velocity | damping | smoothing | "
             "initial_misfit | work (own) | work (pooled) | fwd / adj / HVP | stop |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, record in records.items():
        r = last_row(record)
        p = rows[name][-1]
        lines.append(
            f"| {name} | {r['iteration']} | {r['J']:.6e} | "
            f"{r.get('temperature_term', float('nan')):.4e} | "
            f"{r.get('velocity_term', float('nan')):.4e} | "
            f"{r.get('damping_term', float('nan')):.4e} | "
            f"{r.get('smoothing_term', float('nan')):.4e} | "
            f"{r['initial_misfit']:.4e} | {r['work']:.1f} | {p['work_priced']:.1f} | "
            f"{record['forward']} / {record['adjoint']} / {record['hessian']} | "
            f"{record.get('stop_reason', '')} |")
    return "\n".join(lines)


def krylov_summary(name, record):
    """Krylov reasons, caps, shifts and negative-curvature stops of a TAO run."""
    all_solves = solves(record)
    if not all_solves:
        return f"{name}: no Krylov solves recorded."
    reasons = Counter(s.get("reason_name") for s in all_solves)
    caps = sum(1 for s in all_solves if s.get("at_cap"))
    shifts = sorted({s.get("hessian_shift", 0.0) for s in all_solves})
    lines = [f"{name}: {len(all_solves)} Krylov solves, "
             f"{sum(s.get('iterations', 0) for s in all_solves)} Krylov iterations; "
             f"reasons {dict(reasons)}; at the cap {caps}; shifts seen {shifts}."]
    negative = [s for s in all_solves if s.get("negative_curvature")]
    lines.append(f"Negative-curvature solves: {len(negative)}.")
    if negative:
        lines.append("")
        lines.append("| iteration | Krylov its | pHp | pAp | pHp/pAp | shift | step norm |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in negative:
            nc = s["negative_curvature"]
            lines.append(f"| {s['iteration']} | {s.get('iterations')} | {nc['pHp']:.4e} | "
                         f"{nc['pAp']:.4e} | {nc['pHp_over_pAp']:.4g} | {nc['pert']:.3g} | "
                         f"{s.get('step_norm')} |")
    errors = record.get("recording_errors") or []
    lines.append(f"Recording errors: {errors if errors else 'none'}.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--newton", required=True)
    parser.add_argument("--rol-metric", required=True)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=list(summarise.LEVELS))
    args = parser.parse_args(argv)

    paths = {"P-ROL-L2": args.reference, "P-Newton": args.newton,
             "P-ROL-A": args.rol_metric}
    records = {name: summarise.load(path) for name, path in paths.items()}
    gp, hp, medians = summarise.pooled_prices(list(records.values()))
    rows = {name: summarise.priced_rows(r, gp, hp) for name, r in records.items()}

    print("## Per-run table\n")
    print(f"Pooled medians: forward {medians['forward']:.1f} s, gradient "
          f"{medians['gradient']:.1f} s, product {medians['hessian'] or float('nan'):.1f} s; "
          f"prices: gradient {gp:.3f}, product {(hp or float('nan')):.3f} "
          "forward-equivalents.\n")
    print(run_table(records, rows))
    print("\n## Krylov record of P-Newton\n")
    print(krylov_summary("P-Newton", records["P-Newton"]))
    print("\n## Verdicts (summarise.py)\n")
    buffer = io.StringIO()
    summary_argv = ["--reference", args.reference, "--newton", args.newton,
                    "--rol-metric", args.rol_metric, "--levels", *map(str, args.levels)]
    with redirect_stdout(buffer):
        summarise.main(summary_argv)
    print("```")
    print(buffer.getvalue().rstrip())
    print("```")


if __name__ == "__main__":
    main()
