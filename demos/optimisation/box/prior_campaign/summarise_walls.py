"""Summarise the wall-treatment runs of step S2 of the prior-covariance campaign.

Reads one JSON record per run, as written by ``inverse.py`` with the recording of
``recording.py``, and prints four tables:

1. ``initial_misfit`` against work: the value at the last iteration whose work does
   not pass each of a list of work levels, total and split into the bottom row, the
   top row and the remainder.
2. The final iteration of each run: the same split, the work, and the Krylov solves
   (count, iterations, how many reached the cap, the converged reasons).
3. The fixed and active set sizes of the Krylov solves over the run.
4. The wall nodes that entered and left the active set between consecutive solves.

Work is the forward-equivalent work of the record, priced from measured wall time, so
two runs compare at equal work, and the iteration that crosses a level differs from
run to run. Each record prices a Hessian-vector product by its own measured ratio to
a forward call, so the timing of the node moves the work axis of each run. With
``--common-ratio`` every run is re-priced with one ratio from its recorded counts,
``work = forward + adjoint + ratio * hessian``. ``--common-ratio pooled`` uses the
ratio of all the records together: the mean time of a Hessian-vector product over
the mean time of a forward call, both pooled over every call of every record. A run is read at its last iteration below each level, never by
interpolation, because ``initial_misfit`` need not change smoothly between iterations.

Usage:
    python summarise_walls.py a=PATH b=PATH ... [--levels 100 200 300 400 500 600]

Only the standard library is used.
"""

import argparse
import json
from collections import Counter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", metavar="NAME=PATH",
                        help="a short name and the path of one record")
    parser.add_argument("--common-ratio", default=None,
                        help="re-price every run with one Hessian-to-forward ratio: "
                             "a number, or 'pooled' for the pooled ratio of all "
                             "records")
    parser.add_argument("--plot", default=None,
                        help="write initial_misfit against work to this image file")
    parser.add_argument("--levels", nargs="+", type=float,
                        default=[50, 100, 200, 300, 400, 500, 600],
                        help="work levels for the first table")
    return parser.parse_args(argv)


def at_work(rows, level):
    """The last iteration row whose work does not pass ``level``, or None."""
    best = None
    for row in rows:
        if row["work"] <= level:
            best = row
    return best


def solves_of(record):
    """Every Krylov solve of a record, with the iteration it belongs to."""
    out = []
    for row in record["iterations"]:
        for solve in row.get("krylov_solves", []):
            out.append((row["iteration"], solve))
    return out


def wall_transitions(solves):
    """Count wall nodes that enter and leave the active set between solves.

    Args:
        solves: list of ``(iteration, solve record)``.

    Returns:
        list of ``(iteration, entered, left, size)`` per solve after the first, with
        the counts summed over both walls; ``size`` is the number of active wall
        nodes in that solve.
    """
    out = []
    previous = None
    for iteration, solve in solves:
        current = set(solve.get("active_wall_bottom") or []) | set(
            solve.get("active_wall_top") or [])
        if previous is not None:
            out.append((iteration, len(current - previous), len(previous - current),
                        len(current)))
        previous = current
    return out


def pooled_ratio(records):
    """Mean Hessian-vector product time over mean forward time, over every record."""
    forward = [t for r in records.values() for t in r["forward_times"]]
    hessian = [t for r in records.values() for t in r["hessian_times"]]
    return (sum(hessian) / len(hessian)) / (sum(forward) / len(forward))


def reprice(records, ratio):
    """Replace ``work`` of every iteration row by the common-ratio work, in place."""
    for record in records.values():
        for row in record["iterations"]:
            row["work_recorded"] = row["work"]
            row["work"] = row["forward"] + row["adjoint"] + ratio * row["hessian"]


def plot(records, path, title):
    """Draw initial_misfit against work, total and wall rows, one line per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    panels = (("initial_misfit", "total"), ("initial_misfit_interior", "remainder"),
              ("walls", "bottom + top rows"))
    for name, record in records.items():
        rows = record["iterations"]
        work = [r["work"] for r in rows]
        for ax, (field, label) in zip(axes, panels):
            if field == "walls":
                values = [r.get("initial_misfit_bottom", float("nan"))
                          + r.get("initial_misfit_top", float("nan")) for r in rows]
            else:
                values = [r.get(field, float("nan")) for r in rows]
            ax.semilogy(work, values, marker=".", label=name)
            ax.set_title(f"initial_misfit, {label}")
            ax.set_xlabel("work (forward-equivalents)")
            ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
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
        ratio = (pooled_ratio(records) if args.common_ratio == "pooled"
                 else float(args.common_ratio))
        reprice(records, ratio)
        title = f"work re-priced with one ratio {ratio:.4f} (Hessian-vector product / forward)"
    print(f"== {title}")
    for name, record in records.items():
        cost = record.get("call_cost", {})
        print(f"   {name}: own ratio {cost.get('hessian_per_forward')}, "
              f"mean forward {cost.get('mean_forward_time')}")
    if args.plot:
        plot(records, args.plot, title)

    fields = ("initial_misfit", "initial_misfit_bottom", "initial_misfit_top",
              "initial_misfit_interior")
    print("== initial_misfit at the last iteration with work <= level")
    print(f"{'run':>4s} {'level':>6s} {'it':>4s} {'work':>8s} " +
          " ".join(f"{f.replace('initial_misfit', 'im'):>14s}" for f in fields))
    for name, record in records.items():
        for level in args.levels:
            row = at_work(record["iterations"], level)
            if row is None:
                continue
            print(f"{name:>4s} {level:6.0f} {row['iteration']:4d} {row['work']:8.1f} " +
                  " ".join(f"{row.get(f, float('nan')):14.6e}" for f in fields))

    print("\n== final iteration")
    print(f"{'run':>4s} {'it':>4s} {'work':>8s} " +
          " ".join(f"{f.replace('initial_misfit', 'im'):>14s}" for f in fields) +
          f" {'solves':>6s} {'ksp its':>7s} {'at cap':>6s} {'max its':>7s}  reasons")
    for name, record in records.items():
        row = record["iterations"][-1]
        solves = [s for _, s in solves_of(record)]
        reasons = Counter(s["reason_name"] for s in solves)
        print(f"{name:>4s} {row['iteration']:4d} {row['work']:8.1f} " +
              " ".join(f"{row.get(f, float('nan')):14.6e}" for f in fields) +
              f" {len(solves):6d} {sum(s['iterations'] for s in solves):7d} "
              f"{sum(s['at_cap'] for s in solves):6d} "
              f"{max((s['iterations'] for s in solves), default=0):7d}  "
              + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
        print(f"     stop: {record.get('stop_reason')}; wall deviation at the end: "
              f"bottom {row.get('wall_max_dev_bottom')}, top {row.get('wall_max_dev_top')}; "
              f"recording errors: {record.get('recording_errors')}")

    print("\n== active set per Krylov solve: fixed / lower / upper / active wall nodes")
    for name, record in records.items():
        cells = []
        for iteration, s in solves_of(record):
            walls = len(s.get("active_wall_bottom") or []) + len(s.get("active_wall_top") or [])
            cells.append(f"{iteration}:{s['active_fixed']}/{s['active_lower']}/"
                         f"{s['active_upper']}/{walls}")
        print(f"{name:>4s} " + " ".join(cells))

    print("\n== wall nodes entering / leaving the active set between solves")
    for name, record in records.items():
        transitions = wall_transitions(solves_of(record))
        entered = sum(t[1] for t in transitions)
        left = sum(t[2] for t in transitions)
        detail = " ".join(f"{it}:+{e}/-{l}" for it, e, l, _ in transitions if e or l)
        print(f"{name:>4s} entered {entered:4d} left {left:4d}   {detail}")


if __name__ == "__main__":
    main()
