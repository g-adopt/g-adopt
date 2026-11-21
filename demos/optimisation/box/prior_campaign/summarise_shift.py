"""Work to reach a value of J, and the Hessian shift of BNK, for the box campaign runs.

Step S2 of the prior-covariance campaign compared runs by ``initial_misfit`` at fixed
work. That measure moved by a factor of 124 when the initial trust radius changed by
10 percent, because some runs stall after one Krylov solve that stops on negative
curvature. This script prints the measures of step S2b instead:

1. For each threshold of ``J`` (default 1e-3, 1e-4, 1e-5 and 1e-6), the work at the
   first iteration whose ``J`` is at or below it, and ``initial_misfit`` at that
   iteration. ``J`` is the objective that the optimiser minimises, so the work to
   reach a ``J`` compares the optimisers on their own objective, and
   ``initial_misfit`` at that point shows how far the control is from the truth at
   equal fit to the data. The minimiser is not unique (the objective carries no
   regularisation), so two runs at equal ``J`` can hold different controls.
2. The same work divided by the work of a reference run, by default the run called
   ``a``.
3. The number of Krylov solves that stop with ``CONVERGED_NEG_CURVE``.
4. Per Krylov solve: the converged reason, the iteration count, ``ritz_min`` of the
   preconditioned (shifted) Hessian, the step norm, and BNK's Hessian shift.

The shift. BNK adds ``pert * I`` to the reduced Hessian before a Krylov solve. A
record with the field ``hessian_shift`` in its Krylov solves holds the exact value
per solve (see ``recording.py`` and ``riesz_hessian.py``). The records of steps S0
and S2 do not hold it. For every record the script also replays the update rules of PETSc
v3.25.0 (``TaoBNKSafeguardStep`` and ``TaoBNKPerformLineSearch`` in ``bnk.c``) from
the recorded converged reasons and metric gradient norms, with the parameters of the
run (the defaults of ``TaoCreate_BNK`` merged with the ``tao_options`` of the record):

- after a solve that stops on negative curvature (or another indefinite reason):
  ``pert = min(imax, max(imin, imfac * gnorm))`` when ``pert`` was zero
  ("initialise"), else ``pert = min(pmax, max(pgfac * pert, pmgfac * gnorm))``
  ("increase");
- after any other solve: ``pert = min(psfac * pert, pmsfac * gnorm)``, and zero when
  that is below ``pmin`` ("decrease");
- a Newton step that is not a descent direction, or a failed line search on a Newton
  step, applies "initialise" or "increase" as well. The record does not hold either
  event, so the script names it only when the recorded shift matches that branch
  and no other ("not descent or line search failed").

``gnorm`` is ``bnk->gnorm``: the norm of the projected gradient in the metric of the
control (``TaoGradientNorm``), at the iterate where the solve was computed. The
record holds it as ``gradient_norm_metric`` of the iteration row before the row of
the solve. Where a recorded shift exists, the replay is checked against it and then
continues from the recorded value. Where none exists, the replayed value is an
estimate, and the table says so.

Work. Every run is re-priced with one Hessian-to-forward ratio,
``work = forward + adjoint + ratio * hessian``, so that the speed of the node does
not move the work axis. ``--common-ratio 2.4822`` is the pooled ratio of step S2.

Usage:
    python summarise_shift.py a=PATH d=PATH ... --common-ratio 2.4822 \\
        [--thresholds 1e-3 1e-4 1e-5 1e-6] [--reference a] [--plot FILE]

Only the standard library is used, except for ``--plot``, which needs matplotlib.
"""

import argparse
import json
import math
from collections import Counter

# The BNK perturbation parameters and their defaults in ``TaoCreate_BNK``, PETSc
# v3.25.0 (``bnk.c:1265-1275``). The Setonix container prints the same defaults
# under ``-help``.
BNK_SHIFT_DEFAULTS = {
    "sval": 0.0,     # starting perturbation
    "imin": 1.0e-4,  # lower clamp of the initial perturbation
    "imax": 1.0e+2,  # upper clamp of the initial perturbation
    "imfac": 1.0e-1,  # initial perturbation per unit of gnorm
    "pmin": 1.0e-12,  # a decreased perturbation below this becomes zero
    "pmax": 1.0e+2,  # upper clamp of an increased perturbation
    "pgfac": 1.0e+1,  # growth factor of an increase
    "psfac": 4.0e-1,  # shrink factor of a decrease
    "pmgfac": 1.0e-1,  # increased perturbation per unit of gnorm, lower clamp
    "pmsfac": 1.0e-1,  # decreased perturbation per unit of gnorm, upper clamp
}

# The KSP reasons on which ``TaoBNKSafeguardStep`` treats the matrix or the
# preconditioner as indefinite and increases the perturbation.
INDEFINITE_REASONS = {"CONVERGED_NEG_CURVE", "DIVERGED_INDEFINITE_MAT",
                      "DIVERGED_INDEFINITE_PC", "DIVERGED_BREAKDOWN",
                      "DIVERGED_NANORINF"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", metavar="NAME=PATH",
                        help="a short name and the path of one record")
    parser.add_argument("--common-ratio", default=None,
                        help="re-price every run with one Hessian-to-forward ratio; "
                             "without it, the work of each record is used as recorded")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[1e-3, 1e-4, 1e-5, 1e-6], help="values of J")
    parser.add_argument("--reference", default="a",
                        help="run whose work-to-J is the denominator of table 2")
    parser.add_argument("--group", action="append", default=[], metavar="NAME=RUN,RUN,...",
                        help="a group of runs whose work to reach J is summarised by the "
                             "median of the lower and of the upper ends of the intervals")
    parser.add_argument("--levels", nargs="+", type=float, default=[200, 400, 600, 800],
                        help="common work levels of the initial_misfit table")
    parser.add_argument("--plot", default=None,
                        help="write J, ritz_min and the shift against work to this file")
    return parser.parse_args(argv)


def shift_parameters(record):
    """The perturbation parameters of one run: defaults, then the run's options.

    Args:
        record: a JSON record of ``inverse.py``.

    Returns:
        dict with the keys of ``BNK_SHIFT_DEFAULTS``.
    """
    params = dict(BNK_SHIFT_DEFAULTS)
    for key, value in (record.get("tao_options") or {}).items():
        name = key[len("tao_bnk_"):] if key.startswith("tao_bnk_") else None
        if name in params and value is not None:
            params[name] = float(value)
    return params


def _initialise(p, gnorm, params):
    """BNK's first perturbation: ``min(imax, max(imin, imfac * gnorm))``."""
    return min(params["imax"], max(params["imin"], params["imfac"] * gnorm))


def _increase(p, gnorm, params):
    """BNK's indefinite-case update: initialise from zero, otherwise grow."""
    if p <= 0.0:
        return _initialise(p, gnorm, params)
    return min(params["pmax"], max(params["pgfac"] * p, params["pmgfac"] * gnorm))


def _decrease(p, gnorm, params):
    """BNK's update after a good solve: shrink, and drop to zero below ``pmin``."""
    p = min(params["psfac"] * p, params["pmsfac"] * gnorm)
    return 0.0 if p < params["pmin"] else p


def _same(a, b):
    """Equal to roundoff. The replay repeats BNK's arithmetic, so 1e-12 is ample."""
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)


def replay_shift(record):
    """Per-solve shift: recorded, replayed, and the rule that set it.

    The replay walks the solves in order. The value of solve ``j`` comes from the
    value of solve ``j - 1``, its converged reason, and ``gnorm`` at the iterate
    where solve ``j - 1`` was computed. The candidate branches of BNK are:

    - ``decrease``: the solve was not indefinite, its step was a descent direction
      and the line search accepted it;
    - ``initialise`` or ``increase``: the solve was indefinite (negative
      curvature), the step then a descent direction and accepted;
    - the same two after an indefinite solve, followed by a second ``increase``
      because the line search on the Newton step failed;
    - ``decrease`` followed by ``initialise`` or ``increase``: a good solve whose
      line search failed;
    - ``initialise`` or ``increase`` alone after a good solve: the Newton step was
      not a descent direction.

    Without a recorded value the first branch that the reason allows is taken.

    Args:
        record: a JSON record of ``inverse.py``.

    Returns:
        list of dicts per solve: ``iteration``, ``solve``, ``reason``,
        ``recorded`` (None when the record holds no shift), ``replayed`` (None when
        the record has no metric gradient norm), ``rule`` (how the value came
        about), ``match`` (recorded equals replayed, or None).
    """
    params = shift_parameters(record)
    rows = record["iterations"]
    gnorm_by_iteration = {row["iteration"]: row.get("gradient_norm_metric")
                          for row in rows}
    out = []
    pert = params["sval"]
    previous = None  # (reason, gnorm at the iterate of the previous solve)
    for row in rows:
        for solve in row.get("krylov_solves", []):
            recorded = solve.get("hessian_shift")
            replayed, rule = None, None
            if previous is None:
                replayed, rule = pert, "start (sval)"
            elif previous[1] is not None:
                reason, g = previous
                if reason in INDEFINITE_REASONS:
                    first = _increase(pert, g, params)
                    branch = "initialise" if pert <= 0.0 else "increase"
                    candidates = [
                        (first, f"{branch} after {reason}"),
                        (_increase(first, g, params),
                         f"{branch} after {reason}, then increase (line search failed)"),
                    ]
                else:
                    branch = "initialise" if pert <= 0.0 else "increase"
                    down = _decrease(pert, g, params)
                    candidates = [
                        (down, "decrease after a good solve"),
                        (_increase(down, g, params),
                         "decrease, then increase (line search failed)"),
                        (_increase(pert, g, params),
                         f"{branch} (Newton step not descent)"),
                    ]
                replayed, rule = candidates[0]
                if recorded is not None:
                    for value, name in candidates:
                        if _same(value, recorded):
                            replayed, rule = value, name
                            break
                    else:
                        rule = "no BNK branch matches"
                if pert <= 0.0 and replayed == 0.0:
                    # A decrease from zero stays zero: BNK made no shift at all.
                    rule = "none (pert stays 0)"
            match = None if recorded is None or replayed is None else _same(recorded, replayed)
            out.append({"iteration": row["iteration"], "solve": solve.get("solve"),
                        "reason": solve.get("reason_name"), "recorded": recorded,
                        "replayed": replayed, "rule": rule, "match": match})
            # The next solve starts from the recorded value when there is one, so
            # one unexplained change does not spoil the rest of the replay.
            if recorded is not None:
                pert = recorded
            elif replayed is not None:
                pert = replayed
            # gnorm at the iterate of this solve: the row before the row that holds
            # the solve (the monitor of that row ran at the iterate the solve used).
            previous = (solve.get("reason_name"),
                        gnorm_by_iteration.get(row["iteration"] - 1))
    return out


def reprice(record, ratio):
    """Replace ``work`` of every iteration row by the common-ratio work, in place."""
    for row in record["iterations"]:
        row["work_recorded"] = row["work"]
        row["work"] = row["forward"] + row["adjoint"] + ratio * row["hessian"]


def first_below(rows, threshold):
    """The first iteration row with ``J <= threshold`` and the row before it.

    No interpolation is done: ``J`` crossed the threshold somewhere inside the
    iteration that ends at the returned row, and one iteration can cost up to about
    250 forward-equivalents (a Krylov solve at its cap of 100). So the work to
    reach the threshold is known only to lie in ``(work before, work at]``, and
    both ends are reported.

    Returns:
        ``(before, at)``: ``at`` is the first row with ``J <= threshold`` or None;
        ``before`` is the row before it, or None when ``at`` is the first row.
    """
    before = None
    for row in rows:
        if row["J"] <= threshold:
            return before, row
        before = row
    return None, None


def work_bracket(entry):
    """``(lower, upper)`` bounds of the work to reach a threshold, from ``first_below``."""
    before, at = entry
    if at is None:
        return None
    return (0.0 if before is None else before["work"], at["work"])


def at_work(rows, level):
    """The last iteration row whose work does not pass ``level``, or None."""
    best = None
    for row in rows:
        if row["work"] <= level:
            best = row
    return best


def stalls(record, run_length=3, fraction=0.1, reasons=None):
    """Runs of consecutive Krylov solves whose step is short against the trust radius.

    A stall shows as consecutive Krylov solves that each give a step far below the
    trust radius, so the iterate barely moves. It has two known forms, and the test
    must catch both:

    - BNK's identity shift at 64 cells: the solves converge on their relative
      tolerance (``CONVERGED_RTOL``), because the shifted model is minimised well
      inside the trust region by a short scaled-gradient step;
    - D150 at 150 cells: the solves stop at the Krylov cap (``DIVERGED_MAX_IT``) with
      steps of 0.0013 to 0.002 of the radius, because the large shift also slows
      the Krylov method down.

    The converged reason is therefore not part of the default test. A solve counts
    as short if ``step_norm < fraction * trust_radius``, whatever its reason. Both
    are nodal Euclidean norms, as STCG measures them.

    Args:
        record: a JSON record of ``inverse.py`` with Krylov solve records.
        run_length: the smallest number of consecutive short solves that counts.
        fraction: a step below this fraction of the trust radius is short.
        reasons: ``None`` to accept every converged reason, or a collection of
            reason names (for example ``{"CONVERGED_RTOL"}``) to restrict the
            test to those reasons.

    Returns:
        list of ``(first iteration, last iteration, reasons)`` of each such run,
        with ``reasons`` a sorted list of the converged reasons in that run.
    """
    out, current = [], []

    def close():
        # Keep the current run if it is long enough, then start a new one.
        if len(current) >= run_length:
            out.append((current[0][0], current[-1][0], sorted({r for _, r in current})))
        current.clear()

    for row in record["iterations"]:
        for s in row.get("krylov_solves", []):
            reason = s.get("reason_name")
            short = (s.get("step_norm") is not None and s.get("trust_radius")
                     and s["step_norm"] < fraction * s["trust_radius"]
                     and (reasons is None or reason in reasons))
            if short:
                current.append((row["iteration"], reason))
                continue
            close()
    close()
    return out


def short_solves(record, max_its=3, fraction=0.02, long_its=20):
    """Short RTOL solves with a tiny step, and whether a long solve came just before.

    After a long Krylov solve the active set often changes, and the next solve can
    stop on its relative tolerance after 1 to 3 iterations with a step far inside
    the trust radius, because the residual sits in the top of the spectrum. Such a
    solve costs a whole outer iteration (a forward and an adjoint sweep, plus its
    Hessian products) for almost no step.

    Args:
        record: a JSON record of ``inverse.py``, with work already re-priced.
        max_its: largest iteration count of a short solve.
        fraction: a short solve has ``step_norm < fraction * trust_radius``.
        long_its: a solve with at least this many iterations counts as long.

    Returns:
        list of dicts: ``iteration``, ``its``, ``step_norm``, ``trust_radius``,
        ``ritz_max``, ``after_long`` (bool), ``active_change`` (active count
        against the solve before), ``work`` (work of that outer iteration).
    """
    rows = record["iterations"]
    work_of = {r["iteration"]: r["work"] for r in rows}
    previous = None
    out = []
    for row in rows:
        for s in row.get("krylov_solves", []):
            if (s.get("reason_name") == "CONVERGED_RTOL" and s["iterations"] <= max_its
                    and s.get("step_norm") is not None and s.get("trust_radius")
                    and s["step_norm"] < fraction * s["trust_radius"]):
                out.append({
                    "iteration": row["iteration"], "its": s["iterations"],
                    "step_norm": s["step_norm"], "trust_radius": s["trust_radius"],
                    "ritz_max": s.get("ritz_max"),
                    "after_long": previous is not None and previous["iterations"] >= long_its,
                    "active_change": (None if previous is None
                                      else s["active"] - previous["active"]),
                    "work": work_of[row["iteration"]] - work_of.get(row["iteration"] - 1, 0.0),
                })
            previous = s
    return out


def _median(values):
    """Median of a non-empty list of numbers."""
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _fmt(value, spec):
    """Format a number, or a dash for None."""
    return "-" if value is None else format(value, spec)


def plot(records, shifts, path, title):
    """J, ritz_min and the shift against work, one line per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for name, record in records.items():
        rows = record["iterations"]
        line, = axes[0].semilogy([r["work"] for r in rows], [r["J"] for r in rows],
                                 marker=".", label=name)
        work_of = {r["iteration"]: r["work"] for r in rows}
        solves = [(row["iteration"], s) for row in rows
                  for s in row.get("krylov_solves", [])]
        ritz = [(work_of[i], s["ritz_min"]) for i, s in solves if s.get("ritz_min")]
        if ritz:
            axes[1].semilogy(*zip(*ritz), marker=".", color=line.get_color())
        values = [(work_of[e["iteration"]],
                   e["recorded"] if e["recorded"] is not None else e["replayed"],
                   e["recorded"] is not None) for e in shifts[name]]
        values = [(w, v, exact) for w, v, exact in values if v]
        if values:
            exact = all(v[2] for v in values)
            axes[2].semilogy([v[0] for v in values], [v[1] for v in values],
                             marker="." if exact else "x",
                             linestyle="-" if exact else ":", color=line.get_color())
        for i, s in solves:
            if s.get("reason_name") == "CONVERGED_NEG_CURVE":
                axes[0].axvline(work_of[i], color=line.get_color(), alpha=0.3, lw=1)
    axes[0].set_title("J")
    axes[1].set_title("ritz_min of the preconditioned shifted Hessian")
    axes[2].set_title("BNK shift (solid: recorded, dotted: replayed)")
    for ax in axes:
        ax.set_xlabel("work (forward-equivalents)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title + "; vertical lines: negative-curvature solves")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    args = parse_args(argv)
    records = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        records[name] = json.load(open(path))
    title = "work as recorded (each run's own measured ratio)"
    if args.common_ratio is not None:
        ratio = float(args.common_ratio)
        for record in records.values():
            reprice(record, ratio)
        title = f"work re-priced with one ratio {ratio:.4f} (Hessian-vector product / forward)"
    print(f"== {title}")

    print("\n== shift parameters of each run (BNK defaults merged with tao_options)")
    for name, record in records.items():
        params = shift_parameters(record)
        changed = {k: v for k, v in params.items() if v != BNK_SHIFT_DEFAULTS[k]}
        print(f"{name:>8s} {changed if changed else 'defaults'}")

    print("\n== where each run stopped: its own Hessian-to-forward ratio, which priced its "
          "budget, and the work of its last iteration as recorded and at the table ratio")
    print(f"{'run':>8s} {'own ratio':>9s} {'budget':>7s} {'last it':>7s} "
          f"{'work recorded':>13s} {'work here':>9s}  stop")
    for name, record in records.items():
        last = record["iterations"][-1]
        own = (record.get("call_cost") or {}).get("hessian_per_forward")
        print(f"{name:>8s} {_fmt(own, '9.3f')} {record.get('work_budget_requested', 0):7.0f} "
              f"{last['iteration']:7d} {last.get('work_recorded', last['work']):13.1f} "
              f"{last['work']:9.1f}  {record.get('stop_reason')}")

    print("\n== work to reach J <= threshold: the first iteration at or below it, the work "
          "of the iteration before and of that iteration, and initial_misfit there")
    head = " ".join(f"{'J<=' + format(t, '.0e'):>32s}" for t in args.thresholds)
    print(f"{'run':>8s} {head}   {'final J':>10s} {'final work':>10s}")
    reached = {}
    for name, record in records.items():
        cells = []
        for t in args.thresholds:
            before, row = first_below(record["iterations"], t)
            reached[(name, t)] = (before, row)
            if row is None:
                cells.append(f"{'not reached':>32s}")
                continue
            low = 0.0 if before is None else before["work"]
            cells.append(f"{row['iteration']:3d} {low:6.1f}-{row['work']:6.1f} "
                         f"{row['initial_misfit']:.3e}".rjust(32))
        last = record["iterations"][-1]
        print(f"{name:>8s} {' '.join(cells)}   {last['J']:10.3e} {last['work']:10.1f}")
    print("   (each cell: iteration, work before-work at, initial_misfit at that iteration)")

    if args.reference in records:
        # The crossing lies inside one iteration in both runs, so the ratio is only
        # known to lie between (lower bound of this run) / (upper bound of the
        # reference) and (upper bound of this run) / (lower bound of the reference).
        print(f"\n== work to reach J, divided by that of run {args.reference}: the range "
              "that the iteration steps of both runs allow")
        print(f"{'run':>8s} " + " ".join(f"{'J<=' + format(t, '.0e'):>14s}" for t in args.thresholds))
        for name in records:
            cells = []
            for t in args.thresholds:
                mine = work_bracket(reached[(name, t)])
                ref = work_bracket(reached[(args.reference, t)])
                if mine is None or ref is None:
                    cells.append(f"{'-':>14s}")
                    continue
                if name == args.reference:
                    cells.append(f"{'reference':>14s}")
                    continue
                low = mine[0] / ref[1]
                high = math.inf if ref[0] == 0.0 else mine[1] / ref[0]
                cells.append(f"{low:6.2f}-{high:<7.2f}".rjust(14))
            print(f"{name:>8s} " + " ".join(cells))

    print("\n== initial_misfit at the last iteration with common work <= level")
    print(f"{'run':>8s} " + " ".join(f"{format(level, '.0f'):>18s}" for level in args.levels))
    for name, record in records.items():
        cells = []
        for level in args.levels:
            row = at_work(record["iterations"], level)
            last = record["iterations"][-1]
            if row is None:
                cells.append(f"{'-':>18s}")
            elif row is last and last["work"] < level and "budget" in str(record.get("stop_reason")):
                # The run stopped before this level, so the value is its last one.
                cells.append(f"({row['iteration']:2d}) {row['initial_misfit']:.3e} stop".rjust(18))
            else:
                cells.append(f"({row['iteration']:2d}) {row['initial_misfit']:.3e}".rjust(18))
        print(f"{name:>8s} " + " ".join(cells))
    print("   (iteration in brackets; 'stop' marks a run that stopped below that level)")

    print("\n== initial_misfit at common work, split: total / wall rows (bottom + top) / interior")
    for name, record in records.items():
        cells = []
        for level in args.levels:
            row = at_work(record["iterations"], level)
            if row is None or row.get("initial_misfit_interior") is None:
                cells.append(f"{'-':>32s}")
                continue
            walls = row["initial_misfit_bottom"] + row["initial_misfit_top"]
            cells.append(f"{row['initial_misfit']:.2e}/{walls:.2e}/"
                         f"{row['initial_misfit_interior']:.2e}".rjust(32))
        print(f"{name:>8s} " + " ".join(cells))
    print("   levels: " + " ".join(f"{level:.0f}" for level in args.levels))

    for item in args.group:
        group, _, members = item.partition("=")
        members = [m for m in members.split(",") if m in records]
        print(f"\n== group {group} ({', '.join(members)}): median work to reach J")
        for t in args.thresholds:
            brackets = [work_bracket(reached[(m, t)]) for m in members]
            shown = ", ".join("-" if b is None else f"{b[0]:.1f}-{b[1]:.1f}" for b in brackets)
            if any(b is None for b in brackets):
                print(f"   J<={t:.0e}: not reached by every member ({shown})")
                continue
            print(f"   J<={t:.0e}: median lower {_median([b[0] for b in brackets]):7.1f}, "
                  f"median upper {_median([b[1] for b in brackets]):7.1f}   ({shown})")

    print("\n== short RTOL solves (<= 3 iterations, step < 0.02 trust radius)")
    for name, record in records.items():
        found = short_solves(record)
        after = [f for f in found if f["after_long"]]
        print(f"{name:>8s} {len(found)} short, {len(after)} after a long solve (>= 20 its), "
              f"work of those iterations {sum(f['work'] for f in found):.1f}")
        for f in found:
            print(f"           it {f['iteration']:3d} its {f['its']} step {f['step_norm']:.2e} "
                  f"radius {f['trust_radius']:.3g} ritz_max {_fmt(f['ritz_max'], '.2e')} "
                  f"active change {f['active_change']} work {f['work']:.1f} "
                  f"{'after long' if f['after_long'] else ''}")

    # A run of short steps also appears near the end of a converging run, where the
    # Newton step shrinks with the gradient while the trust radius stays large. J at
    # the first and last iteration of each run tells the two cases apart: a stall
    # holds J flat at a high value, the end phase holds J far below every threshold.
    print("\n== stalls: runs of >= 3 consecutive Krylov solves with step norm "
          "< 0.1 trust radius, any converged reason (J at first and last iteration)")
    for name, record in records.items():
        found = stalls(record)
        J = {row["iteration"]: row["J"] for row in record["iterations"]}
        print(f"{name:>8s} " + (", ".join(
            f"iterations {a}-{b} ({'/'.join(r)}; J {J[a]:.2e} to {J[b]:.2e})"
            for a, b, r in found) if found else "none"))

    print("\n== negative-curvature stops: iteration, work, shift used in that solve, shift "
          "set for the next solve, curvature in the metric")
    for name, record in records.items():
        solves = [(row["iteration"], row["work"], s) for row in record["iterations"]
                  for s in row.get("krylov_solves", [])]
        for k, (iteration, work, s) in enumerate(solves):
            if s.get("reason_name") != "CONVERGED_NEG_CURVE":
                continue
            nc = s.get("negative_curvature") or {}
            following = solves[k + 1][2].get("hessian_shift") if k + 1 < len(solves) else None
            print(f"{name:>8s} it {iteration:3d} work {work:7.1f} shift {_fmt(s.get('hessian_shift'), '.3e')} "
                  f"next {_fmt(following, '.3e')} mode {nc.get('shift_mode', '-')} "
                  f"pHp {_fmt(nc.get('pHp'), '.3e')} pAp {_fmt(nc.get('pAp'), '.3e')} "
                  f"pHp/pAp {_fmt(nc.get('pHp_over_pAp'), '.3e')} kappa {_fmt(nc.get('kappa'), '.3e')}")

    shifts = {name: replay_shift(record) for name, record in records.items()}
    print("\n== Krylov solves: count, negative-curvature solves, and the shift record")
    for name, record in records.items():
        solves = [s for row in record["iterations"] for s in row.get("krylov_solves", [])]
        reasons = Counter(s["reason_name"] for s in solves)
        exact = [e for e in shifts[name] if e["recorded"] is not None]
        status = ("recorded" if exact and len(exact) == len(shifts[name])
                  else "replayed (estimate)" if shifts[name] else "none")
        mismatches = sum(1 for e in exact if e["match"] is False)
        print(f"{name:>8s} solves {len(solves):3d}  neg-curve "
              f"{reasons.get('CONVERGED_NEG_CURVE', 0)}  shift {status}"
              + (f", replay mismatches {mismatches}" if exact else ""))

    print("\n== per solve: iteration, work, reason, its, ritz_min, step norm, trust radius, "
          "shift (recorded / replayed), rule")
    for name, record in records.items():
        print(f"-- {name}")
        work_of = {r["iteration"]: r["work"] for r in record["iterations"]}
        by_solve = {(e["iteration"], e["solve"]): e for e in shifts[name]}
        for row in record["iterations"]:
            for s in row.get("krylov_solves", []):
                e = by_solve.get((row["iteration"], s.get("solve")), {})
                print(f"   {row['iteration']:3d} {work_of[row['iteration']]:7.1f} "
                      f"{s['reason_name']:>22s} {s['iterations']:3d} "
                      f"{_fmt(s.get('ritz_min'), '11.4e')} {_fmt(s.get('step_norm'), '10.3e')} "
                      f"{_fmt(s.get('trust_radius'), '9.3g')} "
                      f"{_fmt(e.get('recorded'), '11.4e')} / {_fmt(e.get('replayed'), '11.4e')}"
                      f"  {e.get('rule') or ''}")

    if args.plot:
        plot(records, shifts, args.plot, title)


if __name__ == "__main__":
    main()
