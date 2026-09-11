"""Plot a campaign of box-benchmark runs on the axes of the paper.

Figure 5 of Ghelichkhan et al. (2024, GMD) has three panels against the cost, which
is the number of forward plus adjoint calculations: the objective functional, the
final misfit ``int (T(t_F) - Tobs)^2 dx`` and the initial misfit
``int (Tic - Tic_true)^2 dx``. This script draws the same three panels from the JSON
files that ``inverse.py`` writes, with one line per optimiser, so that a run of this
benchmark can be compared with that figure and the optimisers can be compared with
each other at equal cost.

A Hessian-vector product is not counted in the cost by the paper, because the paper
never computes one. It is drawn here as a separate panel, because for the Newton
methods it is where most of the work goes.

Usage:
    python plot_campaign.py [--runs DIR] [--riesz L2|regularisation|both]
                            [--output campaign.png] [--x cost|hessian|time]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per optimiser family, one line style per member of the family, so that
# the ROL and TAO versions of the same idea sit next to each other in the legend.
STYLE = {
    "rol-lbfgs": ("black", "-"),
    "rol-newton": ("tab:red", "-"),
    "rol-newton-noprec": ("tab:red", "--"),
    "tao-blmvm": ("tab:grey", "-"),
    "tao-bqnktr": ("tab:green", "-"),
    "tao-bqnkls": ("tab:green", "--"),
    "tao-bntr": ("tab:blue", "-"),
    "tao-bntr-noprec": ("tab:blue", "--"),
    "tao-bntr-riesz": ("tab:blue", ":"),
    "tao-bnls": ("tab:purple", "-"),
    "tao-bnls-riesz": ("tab:purple", ":"),
}

PANELS = [
    ("J", "Objective functional"),
    ("final_misfit", r"Final misfit  $\int (T_{t_F} - T_{obs})^2\,dx$"),
    ("initial_misfit", r"Initial misfit  $\int (T_{IC} - T_{IC}^{true})^2\,dx$"),
    ("control_error", r"Relative error  $\|T_{IC} - T_{IC}^{true}\| / \|T_{IC}^{true}\|$"),
]

X_LABEL = {
    "cost": "Cost  #(FWD+GRAD)",
    "work": "Forward-equivalent work  #FWD + #GRAD + r x #HVP",
    "hessian": "Hessian-vector products",
    "time": "Wall time (s)",
}

# Used when a run holds no measured ratio of its own. Measured on the box case at
# 150x150, window 10: 1.96 in the PR #4638 environment and 3.94 in firedrake main.
# A Hessian-vector product is a tangent-linear sweep plus a second-order adjoint
# sweep. PR #4638 caches the solvers and forms of both, which halves the cost and
# brings it to the two forward sweeps that theory gives.
DEFAULT_HESSIAN_RATIO = 1.96


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", default=".", help="directory holding the run JSON files")
    parser.add_argument("--riesz", default="both", choices=("L2", "regularisation", "both"))
    parser.add_argument("--x", default="work", choices=tuple(X_LABEL),
                        help="horizontal axis: sweep count, Hessian-vector products or time")
    parser.add_argument("--output", default="campaign.png")
    return parser.parse_args(argv)


def load_runs(directory):
    """Read every run JSON below ``directory``.

    Args:
        directory: a directory holding ``<optimiser>_<riesz>.json`` files, either
            directly or one level down, which is the layout ``run_campaign.sh`` uses.

    Returns:
        list of dict, sorted by optimiser name.
    """
    directory = Path(directory)
    files = sorted(set(directory.glob("*.json")) | set(directory.glob("*/*.json")))
    runs = []
    for path in files:
        with open(path) as f:
            record = json.load(f)
        if "iterations" not in record:
            continue  # not a run file
        record["path"] = str(path)
        runs.append(record)
    return sorted(runs, key=lambda r: r["optimiser"])


def hessian_ratio(record):
    """Cost of one Hessian-vector product in forward calls, for one run.

    The run's own measurement is used when it has one, because it was made in the
    same environment and at the same size as the run itself.

    Args:
        record: one run dictionary.

    Returns:
        float
    """
    measured = (record.get("call_cost") or {}).get("hessian_per_forward")
    return float(measured) if measured else DEFAULT_HESSIAN_RATIO


def series(record, key, x_key):
    """Pull one curve out of a run record.

    The iteration rows are used, because they are the accepted iterates. Rows that
    lack the key, which happens for the first row of a run that has not evaluated a
    gradient yet, are skipped.

    Args:
        record: one run dictionary.
        key: the measure to plot.
        x_key: ``cost``, ``hessian`` or ``time``.

    Returns:
        ``(xs, ys)``, two lists.
    """
    ratio = hessian_ratio(record)
    xs, ys = [], []
    for row in record["iterations"]:
        if key not in row:
            continue
        if x_key == "work":
            # Everything an optimiser spends, priced in forward calls. This is the
            # only axis on which a Newton method and a quasi-Newton method can be
            # compared, because the paper's axis leaves Hessian products out.
            x = row["forward"] + row["adjoint"] + ratio * row["hessian"]
        elif x_key in row:
            x = row[x_key]
        else:
            continue
        xs.append(x)
        ys.append(row[key])
    return xs, ys


# Levels of the initial misfit at which the work is reported. A threshold crossing is
# a measure that does not depend on where a run happened to stop, which the lowest
# value on the path does: a method that stops early can show a lower path minimum than
# a method that keeps going and starts fitting what the data cannot see.
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5)


def work_to_threshold(record, threshold):
    """Forward-equivalent work at which the initial misfit first falls below a level.

    Args:
        record: one run dictionary.
        threshold: the level of the initial misfit.

    Returns:
        float, or None when the run never reached the level.
    """
    ratio = hessian_ratio(record)
    for row in record["iterations"]:
        if row.get("initial_misfit") is None:
            continue
        if row["initial_misfit"] <= threshold:
            return row.get("work") or (row["forward"] + row["adjoint"]
                                       + ratio * row["hessian"])
    return None


def threshold_table(runs):
    """Print the work each run needed to reach each level of the initial misfit."""
    header = "  ".join(f"{t:>10.0e}" for t in THRESHOLDS)
    print()
    print("Forward-equivalent work to reach an initial misfit of:")
    print(f"{'optimiser':18s} {'metric':15s} {header}   {'lowest':>10s} {'at work':>9s}")
    print("-" * (35 + len(header) + 22))
    for record in runs:
        cells = []
        for threshold in THRESHOLDS:
            work = work_to_threshold(record, threshold)
            cells.append(f"{'never':>10s}" if work is None else f"{work:10.0f}")
        values = [(row["initial_misfit"], row.get("work", 0))
                  for row in record["iterations"] if row.get("initial_misfit") is not None]
        low, at = min(values) if values else (float("nan"), float("nan"))
        print(f"{record['optimiser']:18s} {record['riesz']:15s} {'  '.join(cells)}   "
              f"{low:10.3e} {at:9.0f}")


def draw(runs, riesz, x_key, output):
    """Draw the panels, one column per Riesz map.

    Args:
        runs: the loaded run records.
        riesz: ``L2``, ``regularisation`` or ``both``.
        x_key: the horizontal axis key.
        output: file to write.
    """
    metrics = [riesz] if riesz != "both" else ["L2", "regularisation"]
    metrics = [m for m in metrics if any(r["riesz"] == m for r in runs)]
    if not metrics:
        raise SystemExit("no runs found for the requested inner product")

    fig, axes = plt.subplots(len(PANELS), len(metrics),
                             figsize=(6 * len(metrics), 3.2 * len(PANELS)),
                             squeeze=False, sharex="col")

    for col, metric in enumerate(metrics):
        for row, (key, title) in enumerate(PANELS):
            ax = axes[row][col]
            for record in runs:
                if record["riesz"] != metric:
                    continue
                xs, ys = series(record, key, x_key)
                if not xs:
                    continue
                colour, style = STYLE.get(record["optimiser"], ("tab:orange", "-"))
                ax.plot(xs, ys, style, color=colour, marker="o", markersize=2.5,
                        label=record["optimiser"])
            ax.set_yscale("log")
            ax.set_title(f"{title}\n{metric} inner product", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            if row == len(PANELS) - 1:
                ax.set_xlabel(X_LABEL[x_key])
    # One legend for the whole figure, taken from the first panel that has lines.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"written {output}")


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run JSON files in {args.runs}")
    print(f"{len(runs)} runs:")
    for record in runs:
        last = record["iterations"][-1] if record["iterations"] else {}
        # ``final_control_error`` exists only in the final dump. A run stopped by the
        # wall-clock limit has every iteration row but no final dump, so the key is
        # read with a default and the row is marked as cut short.
        cut = "" if "final_control_error" in record else "  (stopped early)"
        print(f"  {record['optimiser']:18s} {record['riesz']:15s} "
              f"cost {last.get('cost', 0):5d}  hvp {record.get('hessian', 0):5d}  "
              f"IC {last.get('initial_misfit', float('nan')):.4e}  "
              f"FIN {last.get('final_misfit', float('nan')):.4e}  "
              f"err {record.get('final_control_error', last.get('control_error', float('nan'))):.4e}"
              f"  work {last.get('forward', 0) + last.get('adjoint', 0) + hessian_ratio(record) * last.get('hessian', 0):7.0f}"
              f"{cut}")
    threshold_table(runs)
    draw(runs, args.riesz, args.x, args.output)


if __name__ == "__main__":
    main()
