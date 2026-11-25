r"""Plot the gravitational Poisson DtN scaling study.

Reads the JSON sidecars `gravity_cubed_sphere.py` writes and produces the four
figures the study is read through, one per question it asks:

  `weak_scaling.png`     steady solve time and potential-block iteration count
                         against level, one line per truncation. Flat is the
                         result being tested for.
  `accuracy.png`         error against the closed-form potential against
                         truncation, one line per level. Where a line flattens,
                         that mesh has stopped resolving the modes asked of it,
                         which is the resolution ceiling on `L`.
  `kernel_cost.png`      cold and warm first-solve cost and the steady solve
                         beneath them, against truncation. The gap between cold
                         and warm is what a cache removes; the gap between warm
                         and steady is what it cannot.
  `marginal_mode.png`    incremental cost per added multiplier between adjacent
                         truncations.

Matplotlib only, no firedrake, so it runs on a login node.

    <python> plot_gravity_scaling.py --results-dir results/ --out-dir figures/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load(results_dir):
    """Sidecars grouped by cache phase, keyed on (level, lmax)."""
    phases = defaultdict(dict)
    for path in sorted(Path(results_dir).rglob("summary_*.json")):
        try:
            with open(path) as f:
                row = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "fieldsplit_0_iterations_per_invocation" not in row:
            continue                       # a failure record, or another variant
        if row.get("variant") != "iterative":
            continue
        phases[row.get("cache_phase") or "warm"][(row["level"], row["lmax"])] = row
    return phases


def _series(rows, xkey, ykey, groupkey):
    """Group rows into {group: (xs, ys)} sorted by x, skipping missing values."""
    grouped = defaultdict(list)
    for row in rows.values():
        if row.get(ykey) is None:
            continue
        grouped[row[groupkey]].append((row[xkey], row[ykey]))
    return {g: tuple(zip(*sorted(pairs))) for g, pairs in sorted(grouped.items())}


def plot_weak_scaling(warm, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in (
        (axes[0], "steady_solve_time", "steady solve, max over ranks (s)"),
        (axes[1], "fieldsplit_0_iterations_per_invocation",
         "potential-block iterations per invocation"),
    ):
        for marker, (lmax, (xs, ys)) in zip(
                MARKERS, _series(warm, "level", key, "lmax").items()):
            ax.plot(xs, ys, marker=marker, label=f"L = {lmax}")
        ax.set_xlabel("refinement level (ranks $\\times$ 8 per level)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Weak scaling: flat lines mean the cost per core is holding")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "weak_scaling.png", dpi=150)
    plt.close(fig)


def plot_accuracy(warm, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for marker, (level, (xs, ys)) in zip(
            MARKERS, _series(warm, "lmax", "analytic_relative_l2", "level").items()):
        ax.plot(xs, ys, marker=marker, label=f"level {level}")
    ax.set_xlabel("DtN truncation $L$")
    ax.set_ylabel("relative $L^2$ error against the closed form")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Where a line stops falling, the mesh has run out of resolution")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "accuracy.png", dpi=150)
    plt.close(fig)


def plot_kernel_cost(cold, warm, out_dir):
    levels = sorted({lev for lev, _ in warm})
    fig, axes = plt.subplots(1, len(levels), figsize=(3.4 * len(levels), 4.0),
                             squeeze=False, sharey=True)
    for ax, level in zip(axes[0], levels):
        for rows, key, style, label in (
            (cold, "warmup_time", "-o", "cold first solve"),
            (warm, "warmup_time", "-s", "warm first solve"),
            (warm, "steady_solve_time", "--^", "steady solve"),
        ):
            pairs = sorted((lm, r[key]) for (lev, lm), r in rows.items()
                           if lev == level and r.get(key) is not None)
            if pairs:
                ax.plot(*zip(*pairs), style, label=label)
        ax.set_title(f"level {level}")
        ax.set_xlabel("DtN truncation $L$")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("wall time, max over ranks (s)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Kernel generation: cold to warm is cacheable, "
                 "warm to steady is not")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "kernel_cost.png", dpi=150)
    plt.close(fig)


def plot_marginal_mode(warm, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    by_level = defaultdict(list)
    for (level, lmax), row in warm.items():
        if row.get("steady_solve_time") is not None:
            by_level[level].append((lmax, row["n_multipliers"],
                                    row["steady_solve_time"]))
    for marker, (level, entries) in zip(MARKERS, sorted(by_level.items())):
        entries.sort()
        xs, ys = [], []
        for (lo_l, lo_n, lo_t), (hi_l, hi_n, hi_t) in zip(entries, entries[1:]):
            if hi_n == lo_n:
                continue
            xs.append(hi_l)
            ys.append(1000.0 * (hi_t - lo_t) / (hi_n - lo_n))
        if xs:
            ax.plot(xs, ys, marker=marker, label=f"level {level}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("DtN truncation $L$ (increment from the previous truncation)")
    ax.set_ylabel("marginal solve cost per multiplier (ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("What one more angular mode costs per solve")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "marginal_mode.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    phases = load(args.results_dir)
    warm, cold = phases.get("warm", {}), phases.get("cold", {})
    if not warm and not cold:
        raise SystemExit(f"no usable sidecars under {args.results_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if warm:
        plot_weak_scaling(warm, out_dir)
        plot_accuracy(warm, out_dir)
        plot_marginal_mode(warm, out_dir)
    if cold or warm:
        plot_kernel_cost(cold, warm, out_dir)
    print(f"wrote figures to {out_dir}/ from {len(warm)} warm and "
          f"{len(cold)} cold cases")


if __name__ == "__main__":
    main()
