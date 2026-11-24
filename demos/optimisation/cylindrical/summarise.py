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

With ``--gn`` the challenger is P-GN, the Gauss-Newton run (NOTES/team/gauss-newton/
PLAN.md section 5.4): the same rule with P-GN in place of P-Newton, prices pooled
over P-ROL-L2 and P-GN, and a Gauss-Newton product priced at the pooled median of
the Gauss-Newton product times over the pooled forward median. Work of a row:
``forward + p_G adjoint + p_full hessian + p_gn gn_hessian``. Beside the rule, in
this mode: each run priced with its own per-call ratios; the prices of
``--prices-json`` (the prices of cylindrical-newton/RESULT.md) for the forward,
gradient and full-product costs, with the Gauss-Newton price measured in P-GN; and,
with ``--newton``, the same rule with P-Newton as the reference (the Gauss-Newton
effect alone). When the initial radius of P-GN differs from that of P-Newton, that
last reading changes the product and the radius together, and the output says so.

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
        no record holds a product. ``medians`` also holds the pooled median of the
        Gauss-Newton product times under ``gn_hessian`` (None when no record holds
        one); ``gn_price`` gives its price.
    """
    forward = [t for r in records for t in r.get("forward_times", [])]
    gradient = [t for r in records for t in r.get("gradient_times", [])]
    hessian = [t for r in records for t in r.get("hessian_times", [])]
    gn = [t for r in records for t in r.get("gn_hessian_times", []) or []]
    f, g = median(forward), median(gradient)
    h = median(hessian) if hessian else None
    return g / f, (h / f if h else None), {"forward": f, "gradient": g, "hessian": h,
                                           "gn_hessian": median(gn) if gn else None}


def gn_price(medians):
    """The forward-equivalent price of a Gauss-Newton product from pooled medians, or None."""
    return medians["gn_hessian"] / medians["forward"] if medians.get("gn_hessian") else None


def priced_rows(record, gradient_price, hessian_price, gn_hessian_price=None):
    """The iteration rows of a record with the re-priced work under ``work_priced``.

    ``forward + p_G adjoint + p_full hessian + p_gn gn_hessian``; a count that is
    zero or absent adds nothing, so a record without Gauss-Newton products is priced
    ``forward + p_G adjoint + p_full hessian`` exactly.
    """
    rows = []
    for row in record["iterations"]:
        work = row["forward"] + gradient_price * row["adjoint"]
        if row.get("hessian"):
            work += hessian_price * row["hessian"]
        if row.get("gn_hessian"):
            work += gn_hessian_price * row["gn_hessian"]
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


def gn_rule(records, rows, levels, reference, label):
    """W* over the two runs and the rule for P-GN against one reference.

    Args:
        records: the records by name; must hold ``P-GN`` and ``reference``.
        rows: the priced rows by name.
        levels: the candidate levels of W*.
        reference: the name of the reference run.
        label: the heading of the reading.

    Returns:
        dict with ``W_star`` and the result of ``compare``.
    """
    pair = {name: rows[name] for name in ("P-GN", reference)}
    reached = min(r[-1]["work_priced"] for r in pair.values())
    candidates = [level for level in sorted(levels) if level <= reached]
    if not candidates:
        raise SystemExit(f"no run reached work {min(levels):g} "
                         f"(smallest final work {reached:.1f})")
    w_star = candidates[-1]
    print(f"W* = {w_star:.0f} (smallest final work of {reference} and P-GN {reached:.1f})")
    result = compare(label, rows["P-GN"], rows[reference], w_star, newton=False)
    if result["verdict"].startswith("undecided"):
        result["verdict"] = ("undecided (a second P-GN run at the other radius, 0.79 or "
                             "50.6, decides)")
        print(f"  -> {result['verdict']}")
    return {"W_star": w_star, "P-GN": result}


def price_pair(records, names):
    """Pooled prices over the named records: gradient, full product, GN product."""
    gp, hp, medians = pooled_prices([records[n] for n in names])
    return gp, hp, gn_price(medians), medians


def check_gn_records(records):
    """Refuse records that are not the runs they are given as, in the ``--gn`` mode.

    P-ROL-L2: ``rol-lbfgs`` with the L2 inner product. P-GN: ``tao-bnls-riesz`` with a
    ``hessian_choice`` other than ``full`` and at least one Gauss-Newton product.
    P-Newton: ``tao-bnls-riesz`` with no Gauss-Newton product; its record was made
    before ``hessian_choice`` existed, so the key is checked only when present. All
    records share the trajectory and the objective weights.

    Raises:
        SystemExit: with the reason.
    """
    def need(name, key, ok, what):
        if not ok(records[name].get(key)):
            raise SystemExit(f"summarise.py: {name} has {key} = "
                             f"{records[name].get(key)!r}; the run needs {what}")

    need("P-ROL-L2", "optimiser", lambda v: v == "rol-lbfgs", "rol-lbfgs")
    need("P-ROL-L2", "riesz", lambda v: v == "L2", "L2")
    need("P-GN", "optimiser", lambda v: v == "tao-bnls-riesz", "tao-bnls-riesz")
    need("P-GN", "hessian_choice", lambda v: v in ("gn", "gn-then-full"),
         "gn or gn-then-full")
    need("P-GN", "gn_hessian", lambda v: bool(v) and v > 0, "at least one GN product")
    if "P-Newton" in records:
        need("P-Newton", "optimiser", lambda v: v == "tao-bnls-riesz", "tao-bnls-riesz")
        need("P-Newton", "gn_hessian", lambda v: not v, "no GN product")
        if "hessian_choice" in records["P-Newton"]:
            need("P-Newton", "hessian_choice", lambda v: v == "full", "full")
    first = records["P-ROL-L2"]
    for name, record in records.items():
        for key in ("timesteps", "delta_t"):
            if record.get(key) != first.get(key):
                raise SystemExit(f"summarise.py: {name} has {key} = {record.get(key)!r}, "
                                 f"P-ROL-L2 has {first.get(key)!r}")
        for key in ("alpha_T", "alpha_u", "alpha_d", "alpha_s"):
            mine = (record.get("weights") or {}).get(key)
            theirs = (first.get("weights") or {}).get(key)
            if mine != theirs:
                raise SystemExit(f"summarise.py: {name} has weight {key} = {mine!r}, "
                                 f"P-ROL-L2 has {theirs!r}")


def main_gn(args):
    """The rule with P-GN as the challenger, and its readings."""
    paths = {"P-ROL-L2": args.reference, "P-GN": args.gn}
    if args.newton:
        paths["P-Newton"] = args.newton
    records = {name: load(path) for name, path in paths.items()}
    check_gn_records(records)
    out = {}

    # The rule: prices pooled over the two records compared.
    gp, hp, gnp, medians = price_pair(records, ("P-ROL-L2", "P-GN"))
    print(f"pooled medians over P-ROL-L2 and P-GN: forward {medians['forward']:.1f} s, "
          f"gradient {medians['gradient']:.1f} s, GN product "
          f"{medians['gn_hessian'] or float('nan'):.1f} s; prices: gradient {gp:.3f}, "
          f"GN product {gnp or float('nan'):.3f}")
    rows = {n: priced_rows(r, gp, hp or 0.0, gnp) for n, r in records.items()}
    out["prices"] = {"gradient": gp, "hessian": hp, "gn_hessian": gnp}
    out["pooled"] = gn_rule(records, rows, args.levels, "P-ROL-L2",
                            "P-GN (Gauss-Newton) against P-ROL-L2")

    # Reading: each run with its own per-call ratios.
    print("\n--- sensitivity: each run priced with its own per-call ratios ---")
    own_rows = {}
    for name, record in records.items():
        own_gp, own_hp, own_medians = pooled_prices([record])
        own_gnp = gn_price(own_medians)
        own_rows[name] = priced_rows(record, own_gp, own_hp if own_hp else (hp or 0.0),
                                     own_gnp if own_gnp else gnp)
        print(f"{name}: gradient {own_gp:.3f}, product "
              f"{(own_hp if own_hp else float('nan')):.3f}, GN product "
              f"{(own_gnp if own_gnp else float('nan')):.3f}")
    out["own"] = gn_rule(records, own_rows, args.levels, "P-ROL-L2",
                         "P-GN against P-ROL-L2, own ratios")

    # Reading: the prices of cylindrical-newton/RESULT.md, with the GN price of P-GN.
    if args.prices_json:
        fixed = load(args.prices_json)
        _, _, gn_medians = pooled_prices([records["P-GN"]])
        p_gn = gn_price(gn_medians)
        print(f"\n--- prices of {args.prices_json}: gradient {fixed['gradient']:.3f}, "
              f"product {fixed.get('hessian') or float('nan'):.3f}; GN product measured "
              f"in P-GN {p_gn or float('nan'):.3f} ---")
        fixed_rows = {n: priced_rows(r, fixed["gradient"], fixed.get("hessian") or 0.0,
                                     p_gn) for n, r in records.items()}
        out["fixed_prices"] = gn_rule(records, fixed_rows, args.levels, "P-ROL-L2",
                                      "P-GN against P-ROL-L2, RESULT.md prices")

    # Reading: P-Newton as the reference, the Gauss-Newton effect alone.
    if "P-Newton" in records:
        gp_n, hp_n, gnp_n, _ = price_pair(records, ("P-Newton", "P-GN"))
        newton_rows = {n: priced_rows(r, gp_n, hp_n or 0.0, gnp_n)
                       for n, r in records.items()}
        print("\n--- P-GN against P-Newton (prices pooled over the two) ---")
        r_gn = records["P-GN"].get("initial_radius")
        r_n = records["P-Newton"].get("initial_radius")
        if r_gn is None or r_n is None or abs(r_gn - r_n) > 1e-12 * max(abs(r_n or 1), 1):
            print(f"NOTE: P-GN starts at radius {r_gn} and P-Newton at {r_n}: this "
                  "reading changes the product and the radius together, so it is not "
                  "the Gauss-Newton effect alone.")
        out["against_newton"] = gn_rule(records, newton_rows, args.levels, "P-Newton",
                                        "P-GN against P-Newton")

    for key in ("own", "fixed_prices", "against_newton"):
        if key in out and out[key]["P-GN"]["verdict"] != out["pooled"]["P-GN"]["verdict"]:
            print(f"\nNOTE: the verdict for P-GN differs between the rule "
                  f"({out['pooled']['P-GN']['verdict']}) and the reading {key} "
                  f"({out[key]['P-GN']['verdict']}).")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, help="P-ROL-L2 record")
    parser.add_argument("--newton", default=None,
                        help="P-Newton zero-shift record; required without --gn")
    parser.add_argument("--gn", default=None,
                        help="P-GN record: the Gauss-Newton run as the challenger")
    parser.add_argument("--prices-json", default=None,
                        help="with --gn: fixed prices {\"gradient\": .., \"hessian\": ..} "
                             "(those of cylindrical-newton/RESULT.md) for a reading "
                             "beside the rule")
    parser.add_argument("--rol-metric", default=None,
                        help="P-ROL-A record; deferred in the budget-400 run set")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="conditional runs, read against the reference as Newton")
    parser.add_argument("--json", default=None, help="write the verdicts here")
    parser.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                        help="candidate levels of W*; the rule uses 50 100 150 200, "
                             "other values are for testing the script on short runs")
    args = parser.parse_args(argv)
    if args.gn:
        return main_gn(args)
    if args.newton is None:
        parser.error("--newton is required without --gn")

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
