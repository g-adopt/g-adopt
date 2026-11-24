"""The decision rule of the Gauss-Newton box runs, applied to the run records.

Usage:
    python summarise_gn.py --rol T-ROL/record.json --newton-small T-N/record.json \\
        --newton-large T-N-R/record.json --gn T-GN-R/record.json [more GN arms] \\
        --timing-json timing.json [--side-ratio 2.4822] [--context old.json ...] \\
        [--json verdict.json]

The rule was fixed before any run (NOTES/team/gauss-newton/PLAN.md section 4.5,
revision 3). This script is that rule in code, so that the verdict is computed and
not read off a plot. Only the standard library is used.

The rule
--------

All runs have the tutorial regularisation in the objective, so the minimiser is not
the truth and J has a floor. The measure is the work to reach a value of J.

- Work of an iteration row: ``forward + adjoint + p_full hessian + p_full r_GN
  gn_hessian``, with ``p_full = full_per_forward`` and ``r_GN = gn_over_full`` from
  ``timing.json``, measured in one process on the machine of the runs (gate (g) of
  section 4.3). A gradient costs one forward-equivalent, as in the work counter of
  the runs.
- ``J_X(W)``: the J of the last iteration row of run X with work at or below W.
- ``W_X(j)``: the interval ``[W_lo, W_hi]`` between the work of the iteration before
  run X first has ``J <= j`` and the work of that iteration. No interpolation.
- ``S(X, Y, W*) = W* / W_hi(X, J_Y(W*))``, the lower end of the speed-up. A run that
  never reaches the value has ``S = 0``.
- ``N*``: the better full-Hessian run at each level, ``J_N*(W) = min(J_T-N(W),
  J_T-N-R(W))``. A Gauss-Newton arm is compared with the best Newton result, so a
  gain from the radius alone is not counted as a gain from the product.

Tests for a Gauss-Newton arm X:

- A, against ROL: ``S(X, T-ROL, 200) >= 1`` and ``S(X, T-ROL, 400) >= 1.5``;
- B, the Gauss-Newton effect: ``S(X, N*, 200) >= 1.5`` and ``S(X, N*, 400) >= 1``;
- C, no harm: ``S(X, N*, 600) >= 0.67``.

Floor guard: ``J_min`` is the lowest J of any run. A level W* is valid for a
reference Y when ``J_Y(W*) - J_min >= 0.05 (J_0 - J_min)``, with ``J_0`` the first J
of Y. If 400 or 600 is not valid, the largest valid level below it from 150, 200,
300, 400 replaces it, and the output says so. If no level is valid, that part of the
test cannot be evaluated; it counts as failed, and the output says that the verdict
needs a decision by a person.

Verdicts: (A, B, C) = (pass, pass, pass) V1; (pass, pass, fail) V1b; (pass, fail,
pass) V2; (fail, pass, pass) V3; (fail, fail, pass) V4; any other case with C failed
V5. With several arms, the verdict comes from the arm with the largest
``S(X, T-ROL, 400)``. Test A is also printed for T-N and T-N-R.

Readings beside the rule, which never change the verdict: the rule with each run's
own price ratios, and the rule with the common ratio ``--side-ratio`` (2.4822, the
Setonix ratio of the earlier unregularised records) in place of ``full_per_forward``.
The output states every case where such a reading gives another verdict. Then J,
the misfit term and the prior term at fixed levels of work, the Krylov record of each
run, the line-search step lengths, ``initial_misfit`` with the wall split, and
``gn_linear_defect_max``.

Errors: the script exits with an error when no ``timing.json`` is given, and when a
record made with ``--krylov-ew`` has every Krylov solve at ``rtol`` 1e-2 (the
Eisenstat-Walker rule did not act).
"""

import argparse
import json
import sys
from collections import Counter
from statistics import median

# The levels of the tests, the threshold of each part, and the levels of the guard.
TESTS = {
    "A": (("rol", 200.0, 1.0), ("rol", 400.0, 1.5)),
    "B": (("nstar", 200.0, 1.5), ("nstar", 400.0, 1.0)),
    "C": (("nstar", 600.0, 0.67),),
}
GUARDED_LEVELS = (400.0, 600.0)
REPLACEMENT_LEVELS = (150.0, 200.0, 300.0, 400.0)
FLOOR_FRACTION = 0.05
REPORT_LEVELS = (50.0, 100.0, 200.0, 300.0, 400.0, 600.0, 800.0)
VERDICTS = {
    (True, True, True): ("V1", "Gauss-Newton confirmed on the box",
                         "P-GN with the settings of this arm, if the curvature job passes"),
    (True, True, False): ("V1b", "early gain and late loss",
                          "no P-GN (test C fails); report to Sia. A P-GN with the "
                          "fall-back switch needs a new decision of Sia"),
    (True, False, True): ("V2", "ROL caught without a Gauss-Newton effect",
                          "test A also for T-N and T-N-R (printed below); P-GN with the "
                          "settings of this arm, if the curvature job passes; report "
                          "whether the Newton baseline alone catches ROL"),
    (False, True, True): ("V3", "Gauss-Newton helps and does not catch ROL",
                          "P-GN with the settings of this arm, if the curvature job passes"),
    (False, False, True): ("V4", "no gain and no harm",
                           "the box shows no gain and does not have the curvature of the "
                           "cylinder; P-GN with the settings of this arm, if the "
                           "curvature job passes"),
}
V5 = ("V5", "harm", "no P-GN; stop the Gauss-Newton line and report")


class RuleError(Exception):
    """An input that the rule must refuse."""


def load(path):
    """The JSON record of one run."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------
# Prices and work
# ---------------------------------------------------------------------------------

def prices_from_timing(timing):
    """``(p_full, r_GN)`` from the ``timing.json`` of gate (g).

    Raises:
        RuleError: when a price is missing or not positive.
    """
    try:
        p_full, r_gn = float(timing["full_per_forward"]), float(timing["gn_over_full"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuleError(f"timing.json lacks full_per_forward or gn_over_full: {error}")
    if not (p_full > 0 and r_gn > 0):
        raise RuleError(f"the prices must be positive, got p_full={p_full}, r_GN={r_gn}")
    return p_full, r_gn


def own_prices(record, fallback_full, fallback_gn):
    """The price of a full and of a Gauss-Newton product from one run's own times.

    Medians over the run's per-call times: ``median(hessian) / median(forward)`` and
    ``median(gn_hessian) / median(forward)``. A run without products of one kind
    takes the fallback for that kind, which it never uses.

    Returns:
        ``(full price, gn price)`` in forward-equivalents.
    """
    forward = record.get("forward_times") or []
    full = record.get("hessian_times") or []
    gn = record.get("gn_hessian_times") or []
    if not forward:
        return fallback_full, fallback_gn
    f = median(forward)
    return (median(full) / f if full else fallback_full,
            median(gn) / f if gn else fallback_gn)


def priced_rows(record, p_full, p_gn):
    """The iteration rows with the work of the rule under ``work_rule``.

    Args:
        record: a run record.
        p_full: forward-equivalents per full product.
        p_gn: forward-equivalents per Gauss-Newton product (``p_full r_GN`` in the
            rule).
    """
    rows = []
    for row in record["iterations"]:
        work = (row["forward"] + row["adjoint"] + p_full * row.get("hessian", 0)
                + p_gn * row.get("gn_hessian", 0))
        rows.append(dict(row, work_rule=work))
    return rows


def value_at(rows, level, key="J"):
    """``key`` of the last row with work at or below ``level``, and that row."""
    eligible = [r for r in rows if r["work_rule"] <= level]
    return (eligible[-1][key], eligible[-1]) if eligible else (None, None)


def work_to_reach(rows, target):
    """``(W_lo, W_hi)`` around the first row with J at or below ``target``, or None.

    A row whose J equals the target exactly counts as reaching it, so a value met at
    an iteration boundary gives the work of that iteration as ``W_hi``.
    """
    previous = 0.0
    for row in rows:
        if row["J"] <= target:
            return previous, row["work_rule"]
        previous = row["work_rule"]
    return None


def speedup(rows, target, level):
    """``S = level / W_hi`` for reaching ``target``, and the interval; 0 if never reached."""
    crossing = work_to_reach(rows, target)
    if crossing is None:
        return 0.0, None
    return level / crossing[1], crossing


# ---------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------

class Reference:
    """A reference of the rule: one run (T-ROL) or the pointwise better of two (N*).

    Args:
        name: for the output.
        runs: list of priced row lists. ``J(W)`` is the minimum over them of the J
            of the last row at or below ``W``.
    """

    def __init__(self, name, runs):
        self.name = name
        self.runs = runs

    def J(self, level):
        """``J_Y(W)``, or None when no run has a row at or below the level."""
        values = [value_at(rows, level)[0] for rows in self.runs]
        values = [v for v in values if v is not None]
        return min(values) if values else None

    def J0(self):
        """The first J of the reference."""
        return min(rows[0]["J"] for rows in self.runs)


def effective_level(reference, level, j_min):
    """The level that the floor guard allows in place of ``level``.

    Returns:
        ``(level or None, note or None)``.
    """
    def valid(w):
        j = reference.J(w)
        return j is not None and j - j_min >= FLOOR_FRACTION * (reference.J0() - j_min)

    if level not in GUARDED_LEVELS or valid(level):
        return level, None
    candidates = [w for w in REPLACEMENT_LEVELS if w < level and valid(w)]
    if not candidates:
        return None, (f"floor guard: {level:g} is not valid for {reference.name} and no "
                      f"level of {REPLACEMENT_LEVELS} below it is; this part cannot be "
                      f"evaluated and counts as failed. The verdict needs a decision by a "
                      f"person.")
    return candidates[-1], (f"floor guard: {level:g} is not valid for {reference.name} "
                            f"(J within {FLOOR_FRACTION:g} of the floor); {candidates[-1]:g} "
                            f"replaces it")


def evaluate_arm(rows, references, j_min):
    """Tests A, B and C for one arm.

    Args:
        rows: the priced rows of the arm.
        references: ``{"rol": Reference, "nstar": Reference}``.
        j_min: the lowest J of any run.

    Returns:
        dict with one entry per test (``pass``, the parts) and the verdict.
    """
    out = {"notes": []}
    passes = {}
    for test, parts in TESTS.items():
        details = []
        ok = True
        for ref_key, level, threshold in parts:
            reference = references[ref_key]
            used, note = effective_level(reference, level, j_min)
            if note:
                out["notes"].append(f"test {test}: {note}")
            if used is None:
                details.append({"reference": reference.name, "level": level,
                                "level_used": None, "threshold": threshold,
                                "S": None, "pass": False})
                ok = False
                continue
            target = reference.J(used)
            s, crossing = speedup(rows, target, used)
            part_ok = s >= threshold
            details.append({"reference": reference.name, "level": level,
                            "level_used": used, "J_ref": target, "threshold": threshold,
                            "S": s, "W_interval": crossing, "pass": part_ok})
            ok &= part_ok
        out[test] = {"pass": ok, "parts": details}
        passes[test] = ok
    key = (passes["A"], passes["B"], passes["C"])
    # (pass, pass, fail) is V1b; every other combination with C failed is V5.
    code, label, consequence = VERDICTS.get(key, V5)
    out["verdict"] = {"code": code, "label": label, "consequence": consequence}
    return out


def apply_rule(records, p_full, r_gn, own=False):
    """The whole rule for a set of records.

    Args:
        records: ``{"T-ROL": rec, "T-N": rec, "T-N-R": rec, "<gn arm>": rec, ...}``;
            every name other than the three references is a Gauss-Newton arm.
        p_full, r_gn: the prices of the rule.
        own: price each run with its own per-call ratios instead.

    Returns:
        dict with the arms, test A of T-N and T-N-R, ``J_min`` and the chosen arm.
    """
    rows = {}
    for name, record in records.items():
        full, gn = ((p_full, p_full * r_gn) if not own
                    else own_prices(record, p_full, p_full * r_gn))
        rows[name] = priced_rows(record, full, gn)
    j_min = min(r["J"] for rr in rows.values() for r in rr)
    references = {"rol": Reference("T-ROL", [rows["T-ROL"]]),
                  "nstar": Reference("N* = min(T-N, T-N-R)", [rows["T-N"], rows["T-N-R"]])}
    arms = [name for name in records if name not in ("T-ROL", "T-N", "T-N-R")]
    if not arms:
        raise RuleError("no Gauss-Newton arm given")
    out = {"J_min": j_min, "arms": {}, "newton_test_A": {}}
    for name in arms:
        out["arms"][name] = evaluate_arm(rows[name], references, j_min)
    for name in ("T-N", "T-N-R"):
        result = evaluate_arm(rows[name], references, j_min)
        out["newton_test_A"][name] = result["A"]
    # The arm with the largest S(X, T-ROL, 400), the second part of test A.
    def s400(name):
        part = out["arms"][name]["A"]["parts"][1]
        return part["S"] if part["S"] is not None else -1.0
    chosen = max(arms, key=s400)
    out["chosen"] = chosen
    out["verdict"] = out["arms"][chosen]["verdict"]
    out["rows"] = rows
    return out


# ---------------------------------------------------------------------------------
# Input checks and the readings beside the rule
# ---------------------------------------------------------------------------------

def solves(record):
    """Every Krylov solve of a record, in order, with its iteration number."""
    out = []
    for row in record["iterations"]:
        for solve in row.get("krylov_solves", []) or []:
            out.append(dict(solve, iteration=row["iteration"]))
    out.extend(record.get("krylov_solves_unassigned", []) or [])
    return out


def check_records(records):
    """Refuse a ``--krylov-ew`` record whose solves all have ``rtol`` 1e-2.

    Raises:
        RuleError
    """
    for name, record in records.items():
        if record.get("krylov_ew"):
            all_solves = solves(record)
            if all_solves and all(s.get("rtol") == 1e-2 for s in all_solves):
                raise RuleError(f"{name} was run with --krylov-ew and every Krylov solve "
                                f"has rtol 1e-2: the Eisenstat-Walker rule did not act")


def format_result(name, result):
    """Lines of text for one arm."""
    lines = [f"{name}: verdict {result['verdict']['code']} "
             f"({result['verdict']['label']}); consequence: {result['verdict']['consequence']}"]
    for test in ("A", "B", "C"):
        parts = []
        for p in result[test]["parts"]:
            if p["S"] is None:
                parts.append(f"S({name}, {p['reference']}, {p['level']:g}) not evaluable")
                continue
            at = (f"{p['level_used']:g}" if p["level_used"] == p["level"]
                  else f"{p['level_used']:g} for {p['level']:g}")
            interval = ("never reached" if p["W_interval"] is None
                        else f"W in [{p['W_interval'][0]:.1f}, {p['W_interval'][1]:.1f}]")
            parts.append(f"S({name}, {p['reference']}, {at}) = {p['S']:.3f} "
                         f"(J_ref {p['J_ref']:.6e}, {interval}, need >= {p['threshold']:g})")
        lines.append(f"  test {test}: {'pass' if result[test]['pass'] else 'fail'}")
        lines += [f"    {text}" for text in parts]
    lines += [f"  NOTE {note}" for note in result["notes"]]
    return lines


def side_tables(records, rows):
    """J and its two parts at the report levels, the Krylov record and the rest."""
    lines = ["", "--- beside the rule (work at the prices of the rule) ---", "",
             "| run | at work | iteration | work | J | misfit term | prior term |",
             "|---|---|---|---|---|---|---|"]
    for name, rr in rows.items():
        for level in REPORT_LEVELS:
            if rr[-1]["work_rule"] < level:
                lines.append(f"| {name} | {level:g} | not reached | | | | |")
                continue
            _, row = value_at(rr, level)
            lines.append(f"| {name} | {level:g} | {row['iteration']} | "
                         f"{row['work_rule']:.1f} | {row['J']:.6e} | "
                         f"{row.get('misfit_term', float('nan')):.6e} | "
                         f"{row.get('prior_term', float('nan')):.6e} |")
    for name, record in records.items():
        all_solves = solves(record)
        lines += ["", f"Krylov record of {name}: {len(all_solves)} solves."]
        if all_solves:
            reasons = Counter(s.get("reason_name") for s in all_solves)
            modes = Counter(s.get("hessian_product") for s in all_solves)
            caps = sum(1 for s in all_solves if s.get("at_cap"))
            lines.append(f"  reasons {dict(reasons)}; products {dict(modes)}; at the cap "
                         f"{caps}")
            lines.append("  iterations per solve: "
                         + " ".join(str(s.get("iterations")) for s in all_solves))
            lines.append("  rtol per solve: "
                         + " ".join(f"{s.get('rtol'):.3g}" if s.get("rtol") is not None
                                    else "?" for s in all_solves))
            negative = [s for s in all_solves if s.get("negative_curvature")]
            for s in negative:
                nc = s["negative_curvature"]
                lines.append(f"  negative curvature at iteration {s['iteration']} "
                             f"({s.get('hessian_product')}): pHp/pAp "
                             f"{nc.get('pHp_over_pAp')}")
        steps = [r.get("step_length") for r in record["iterations"][1:]]
        lines.append("  line-search step lengths: "
                     + " ".join(f"{s:.3g}" if isinstance(s, (int, float)) else "?"
                                for s in steps))
        last = record["iterations"][-1]
        lines.append(f"  end: initial_misfit {last.get('initial_misfit')}, bottom "
                     f"{last.get('initial_misfit_bottom')}, top "
                     f"{last.get('initial_misfit_top')}, interior "
                     f"{last.get('initial_misfit_interior')}; gn_linear_defect_max "
                     f"{record.get('gn_linear_defect_max')}")
    return lines


def compare_readings(rule, reading, label):
    """Lines that state where a side reading gives another verdict than the rule."""
    lines = []
    for name, result in rule["arms"].items():
        other = reading["arms"][name]["verdict"]["code"]
        if other != result["verdict"]["code"]:
            lines.append(f"NOTE: {label}: the verdict of {name} is {other}, the rule gives "
                         f"{result['verdict']['code']}.")
    if reading["chosen"] != rule["chosen"]:
        lines.append(f"NOTE: {label}: the chosen arm is {reading['chosen']}, the rule "
                     f"chooses {rule['chosen']}.")
    return lines


def summarise(records, timing, side_ratio=2.4822, context=None):
    """The rule, the two side readings and the side tables, as text and a dict.

    Args:
        records: the records by name (``T-ROL``, ``T-N``, ``T-N-R`` and the arms).
        timing: the parsed ``timing.json``.
        side_ratio: the common ratio of the second side reading.
        context: ``{name: record}`` of finished records with another objective, or
            None.

    Returns:
        ``(lines, result dict)``.

    Raises:
        RuleError: for the inputs that the rule refuses.
    """
    if timing is None:
        raise RuleError("no timing.json: the rule prices the products with "
                        "full_per_forward and gn_over_full from gate (g)")
    check_records(records)
    p_full, r_gn = prices_from_timing(timing)
    rule = apply_rule(records, p_full, r_gn)
    lines = [f"prices of the rule (timing.json): p_full = full_per_forward = {p_full:.4f}, "
             f"r_GN = gn_over_full = {r_gn:.4f}, so a GN product costs "
             f"{p_full * r_gn:.4f} forward-equivalents",
             f"J_min (lowest J of any run) = {rule['J_min']:.6e}", ""]
    for name in rule["arms"]:
        lines += format_result(name, rule["arms"][name])
    lines.append("")
    for name, test in rule["newton_test_A"].items():
        parts = ", ".join(f"S at {p['level_used']}: {p['S']:.3f}" if p["S"] is not None
                          else "not evaluable" for p in test["parts"])
        lines.append(f"test A for {name} against T-ROL: {'pass' if test['pass'] else 'fail'} "
                     f"({parts})")
    lines += ["", f"VERDICT (rule): {rule['verdict']['code']} from arm {rule['chosen']}: "
              f"{rule['verdict']['label']}. Consequence: {rule['verdict']['consequence']}."]

    own = apply_rule(records, p_full, r_gn, own=True)
    side = apply_rule(records, side_ratio, r_gn)
    lines += ["", "--- side reading 1: each run priced with its own per-call ratios ---"]
    for name, record in records.items():
        full, gn = own_prices(record, p_full, p_full * r_gn)
        lines.append(f"{name}: full product {full:.4f}, GN product {gn:.4f}")
    lines += [f"verdict: {own['verdict']['code']} from arm {own['chosen']}"]
    lines += compare_readings(rule, own, "own per-call ratios")
    lines += ["", f"--- side reading 2: common ratio {side_ratio} in place of "
              f"full_per_forward ---", f"verdict: {side['verdict']['code']} from arm "
              f"{side['chosen']}"]
    lines += compare_readings(rule, side, f"common ratio {side_ratio}")
    lines += side_tables(records, rule["rows"])
    if context:
        lines += ["", "--- context: finished records with ANOTHER objective (no "
                  "regularisation in J); not comparable with the rule ---"]
        for name, record in context.items():
            rr = priced_rows(record, p_full, p_full * r_gn)
            cells = []
            for level in (100.0, 200.0, 400.0, 600.0):
                j, _ = value_at(rr, level)
                cells.append(f"J({level:g}) = {j:.3e}" if j is not None else
                             f"J({level:g}) = -")
            lines.append(f"{name}: " + ", ".join(cells))
    result = {"prices": {"p_full": p_full, "r_GN": r_gn},
              "rule": {k: v for k, v in rule.items() if k != "rows"},
              "own": {k: v for k, v in own.items() if k != "rows"},
              "side_ratio": {k: v for k, v in side.items() if k != "rows"}}
    return lines, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rol", required=True, help="T-ROL record")
    parser.add_argument("--newton-small", required=True, help="T-N record (radius 0.2)")
    parser.add_argument("--newton-large", required=True, help="T-N-R record (radius 12.8)")
    parser.add_argument("--gn", nargs="+", required=True,
                        help="Gauss-Newton arms: T-GN-R and each optional arm that ran")
    parser.add_argument("--timing-json", default=None,
                        help="timing.json of gate (g): full_per_forward and gn_over_full")
    parser.add_argument("--side-ratio", type=float, default=2.4822,
                        help="common ratio of the second side reading")
    parser.add_argument("--context", nargs="*", default=[],
                        help="finished records with another objective, for context")
    parser.add_argument("--json", default=None, help="write the verdicts here")
    args = parser.parse_args(argv)
    records = {"T-ROL": load(args.rol), "T-N": load(args.newton_small),
               "T-N-R": load(args.newton_large)}
    for path in args.gn:
        name = path
        record = load(path)
        records[name] = record
    timing = load(args.timing_json) if args.timing_json else None
    context = {path: load(path) for path in args.context}
    try:
        lines, result = summarise(records, timing, args.side_ratio, context)
    except RuleError as error:
        print(f"summarise_gn.py: error: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
