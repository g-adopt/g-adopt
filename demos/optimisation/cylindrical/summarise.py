"""The decision rule of the cylindrical comparison, applied to the run records.

Usage:
    python summarise.py --reference rol-lbfgs-L2.json --newton tao-zero.json \
        [--rol-metric rol-lbfgs-A.json] [--extra tao-zero-r0.87.json ...]

The rule was fixed before any run (NOTES/team/cylindrical-newton/PLAN.md section 5).
This script is that rule in code, so that the verdict is computed and not read off a
plot.

Configuration P has the regularisation in the objective. Its minimiser is not the
truth (J at the truth is about 0.17 against J0 = 0.189), so the error against the
truth is not the measure. The measure is the work to reach a value of J:

1. Work is re-priced with one set of prices for every run: the median forward,
   gradient and Hessian-product times pooled over all records given. A gradient
   costs ``median(gradient) / median(forward)`` forward-equivalents and a product
   ``median(hessian) / median(forward)``. Shared nodes change the times of a single
   run; the pooled medians remove that from the comparison.
2. ``W*`` is the largest of 100, 200, 300 and 400 that every run reached (revision 3
   of the rule, fixed on 2026-11-22 before any result of the budget-400 runs; the
   levels were 50 to 200 for the budget-200 run set, which was cancelled unread).
3. ``J_ref`` is the J of the reference run (P-ROL-L2, what g-adopt runs today) at the
   last iteration with work at or below ``W*``.
4. For a challenger, ``W_N`` is the work at which its J first falls to ``J_ref`` or
   below. The crossing lies between the work of the iteration before and the work of
   that iteration, ``[W_lo, W_hi]``; no interpolation is done (the box convention).
5. ``S = W* / W_N`` as the interval ``[W* / W_hi, W* / W_lo]``. The verdict uses the
   lower end ``S_lo = W* / W_hi``, the speed-up that the record proves:
   ``S_lo >= 3``: the challenger wins; ``1.5 <= S_lo < 3``: undecided (for Newton,
   the run at radius 0.87 decides); ``S_lo < 1.5``: no win shown. A challenger that
   never reaches ``J_ref`` has ``S < 1``.

The same rule reads P-ROL-A against P-ROL-L2, the effect of the inner product alone.
``initial_misfit`` is printed beside it as supporting information only.

A second reading follows the rule, as a sensitivity check and not as the rule: every
run priced with its own per-call ratios. The pooled product price divides a median
taken from the Newton record alone by a forward median pooled over three nodes, so load
on the Newton node moves it; a run's own ratios take both times from one node. The
script says so when the two readings give different verdicts.
"""

import argparse
import json
from statistics import median

LEVELS = (100.0, 200.0, 300.0, 400.0)


def load(path):
    """The JSON record of one run."""
    with open(path) as f:
        return json.load(f)


def pooled_prices(records):
    """Forward-equivalent prices of a gradient and a product, from pooled medians.

    Returns:
        ``(gradient_price, hessian_price, medians)``. The Hessian price is None when
        no record holds a product.
    """
    forward = [t for r in records for t in r.get("forward_times", [])]
    gradient = [t for r in records for t in r.get("gradient_times", [])]
    hessian = [t for r in records for t in r.get("hessian_times", [])]
    f, g = median(forward), median(gradient)
    h = median(hessian) if hessian else None
    return g / f, (h / f if h else None), {"forward": f, "gradient": g, "hessian": h}


def priced_rows(record, gradient_price, hessian_price):
    """The iteration rows of a record with the re-priced work under ``work_priced``."""
    rows = []
    for row in record["iterations"]:
        work = row["forward"] + gradient_price * row["adjoint"]
        if row.get("hessian"):
            work += hessian_price * row["hessian"]
        rows.append(dict(row, work_priced=work))
    return rows


def value_at(rows, level, key="J"):
    """``key`` of the last row with work at or below ``level``, and that row."""
    eligible = [r for r in rows if r["work_priced"] <= level]
    return (eligible[-1][key], eligible[-1]) if eligible else (None, None)


def work_to_reach(rows, target):
    """``(W_lo, W_hi)`` around the first row with J at or below ``target``, or None."""
    previous = 0.0
    for row in rows:
        if row["J"] <= target:
            return previous, row["work_priced"]
        previous = row["work_priced"]
    return None


def verdict(s_lo, newton=True):
    """The class of the rule for the lower end of S."""
    if s_lo >= 3.0:
        return "wins"
    if s_lo >= 1.5:
        return "undecided" + (" (the radius-0.87 run decides)" if newton else "")
    return "no win shown"


def compare(name, rows, reference_rows, w_star, newton):
    """Apply the rule to one challenger and print the result."""
    j_ref, ref_row = value_at(reference_rows, w_star)
    crossing = work_to_reach(rows, j_ref)
    print(f"\n{name}: J_ref = {j_ref:.6e} (reference at work "
          f"{ref_row['work_priced']:.1f}, iteration {ref_row['iteration']})")
    if crossing is None:
        last = rows[-1]
        print(f"  never reaches J_ref; last J {last['J']:.6e} at work "
              f"{last['work_priced']:.1f}: S < 1, no win shown")
        return {"J_ref": j_ref, "S": None, "verdict": "no win shown"}
    w_lo, w_hi = crossing
    s_lo = w_star / w_hi
    s_hi = w_star / w_lo if w_lo > 0 else float("inf")
    result = verdict(s_lo, newton)
    print(f"  W_N in [{w_lo:.1f}, {w_hi:.1f}]; S in [{s_lo:.2f}, {s_hi:.2f}]: {result}")
    im, _ = value_at(rows, w_star, "initial_misfit")
    im_ref, _ = value_at(reference_rows, w_star, "initial_misfit")
    print(f"  supporting only: initial_misfit at W* {im:.4e} against {im_ref:.4e} "
          "(P's minimiser is not the truth)")
    return {"J_ref": j_ref, "W_N": [w_lo, w_hi], "S": [s_lo, s_hi], "verdict": result}


def apply_rule(records, rows, levels):
    """W*, and the rule for P-Newton and P-ROL-A (and any extra run) on priced rows.

    Args:
        records: the records by name.
        rows: the priced iteration rows by name.
        levels: the candidate levels of W*.

    Returns:
        dict with ``W_star`` and one result per challenger.
    """
    reached = min(r[-1]["work_priced"] for r in rows.values())
    candidates = [level for level in sorted(levels) if level <= reached]
    if not candidates:
        raise SystemExit(f"no run reached work {min(levels):g} "
                         f"(smallest final work {reached:.1f})")
    w_star = candidates[-1]
    print(f"W* = {w_star:.0f} (smallest final work over the runs {reached:.1f})")
    out = {"W_star": w_star}
    out["P-Newton"] = compare("P-Newton (zero shift) against P-ROL-L2", rows["P-Newton"],
                              rows["P-ROL-L2"], w_star, newton=True)
    if "P-ROL-A" in rows:
        out["P-ROL-A"] = compare("P-ROL-A against P-ROL-L2 (inner product alone)",
                                 rows["P-ROL-A"], rows["P-ROL-L2"], w_star, newton=False)
    for name in records:
        if name.startswith("extra:"):
            out[name] = compare(f"{name} against P-ROL-L2", rows[name], rows["P-ROL-L2"],
                                w_star, newton=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, help="P-ROL-L2 record")
    parser.add_argument("--newton", required=True, help="P-Newton zero-shift record")
    parser.add_argument("--rol-metric", default=None,
                        help="P-ROL-A record; deferred in the budget-400 run set")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="conditional runs, read against the reference as Newton")
    parser.add_argument("--json", default=None, help="write the verdicts here")
    parser.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                        help="candidate levels of W*; the rule uses 50 100 150 200, "
                             "other values are for testing the script on short runs")
    args = parser.parse_args(argv)

    paths = {"P-ROL-L2": args.reference, "P-Newton": args.newton}
    if args.rol_metric:
        paths["P-ROL-A"] = args.rol_metric
    paths.update({f"extra:{p}": p for p in args.extra})
    records = {name: load(path) for name, path in paths.items()}
    gp, hp, medians = pooled_prices(list(records.values()))
    print(f"pooled medians: forward {medians['forward']:.1f} s, gradient "
          f"{medians['gradient']:.1f} s, product {medians['hessian'] or float('nan'):.1f} s; "
          f"prices: gradient {gp:.3f}, product {hp or float('nan'):.3f}")
    out = {"prices": {"gradient": gp, "hessian": hp}, "medians": medians}
    # The rule: one set of pooled prices for every run.
    rows = {name: priced_rows(r, gp, hp) for name, r in records.items()}
    out["pooled"] = apply_rule(records, rows, args.levels)
    # A sensitivity reading, not the rule: each run priced with its own per-call
    # ratios (median gradient / median forward and median product / median forward
    # of that run). The pooled product price comes from the Newton record alone, so
    # load on that node moves it; this reading shows whether the verdict depends on
    # that. A run without products takes the pooled product price, which it never
    # uses.
    print("\n--- sensitivity: each run priced with its own per-call ratios ---")
    own_rows = {}
    for name, record in records.items():
        own_gp, own_hp, _ = pooled_prices([record])
        own_rows[name] = priced_rows(record, own_gp, own_hp if own_hp else hp)
        print(f"{name}: gradient {own_gp:.3f}, product "
              f"{(own_hp if own_hp else float('nan')):.3f}")
    out["own"] = apply_rule(records, own_rows, args.levels)
    for key in [k for k in ("P-Newton", "P-ROL-A") if k in out["pooled"]]:
        if out["pooled"][key]["verdict"] != out["own"][key]["verdict"]:
            print(f"\nNOTE: the verdict for {key} differs between the pooled prices "
                  f"({out['pooled'][key]['verdict']}) and the own per-call ratios "
                  f"({out['own'][key]['verdict']}).")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
