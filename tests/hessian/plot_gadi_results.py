"""Plot the Gadi Hessian Taylor campaign from the JSON files the harness writes.

Reads every ``*.json`` under ``gadi-results/`` (the rsync target of
``/scratch/xd2/sg8812/hessian-runs/``) and produces four figures plus a Markdown
summary table:

- ``taylor_remainders.png``: R1 and R2 against epsilon, one panel per run, with
  reference slopes of 2 and 3. This is the standard picture.
- ``r2_ratios.png``: the ratio R2(eps) / R2(eps/2) against epsilon for every run. A
  correct Hessian gives 8. A Hessian defect D eps^2 drives the ratio towards 4 if D
  has the sign of the cubic, and above 8 and through a sign change if it has the
  opposite sign. This panel is where the smooth floor and the hard clip separate.
- ``hessian_defect.png``: the fitted defect |D| relative to <h,Hh> against the
  trajectory length, smooth floor versus hard clip.
- ``timings.png``: wall time per call of functional, derivative and Hessian-vector
  product, first and steady calls, against the trajectory length.

Runs whose sweep starts at eps 0.05 with five levels (the first 60-step pair) are
plotted but marked, because that window is pre-asymptotic for 60 steps and the
fitted D is not identified there; see NOTES/ideas/01-hessian-accuracy/REVIEW_60STEP.md.

Usage:
    python plot_gadi_results.py [--results DIR] [--out DIR]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_runs(results_dir):
    """Read every JSON result and attach the metadata parsed from its file name.

    Args:
        results_dir (Path): directory tree holding the ``*.json`` files.

    Returns:
        list of dict: one entry per run with keys "case", "steps", "visc", "tag",
            "eps0", "levels", "path" and "data" (the JSON payload).
    """
    runs = []
    pattern = re.compile(
        r"(?P<case>\w+)_t(?P<steps>\d+)_(?P<visc>production|hard)_(?P<tag>[\w.-]+?)"
        r"(?:_e(?P<eps0>[\d.]+)_L(?P<levels>\d+))?\.json$")
    for path in sorted(results_dir.rglob("*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        d = m.groupdict()
        with open(path) as f:
            data = json.load(f)
        eps = data["result"]["eps"]
        runs.append({
            "case": d["case"],
            "steps": int(d["steps"]),
            "visc": "smooth floor" if d["visc"] == "production" else "hard clip",
            "tag": d["tag"],
            "eps0": float(d["eps0"]) if d["eps0"] else eps[0],
            "levels": int(d["levels"]) if d["levels"] else len(eps),
            "path": path,
            "data": data,
        })
    return runs


def run_label(run, with_sweep=True):
    """A short label for legends and titles."""
    s = f"{run['case']}, {run['steps']} steps, {run['visc']}"
    if with_sweep:
        s += f", eps0 {run['eps0']:g}, {run['levels']} levels"
    return s


def ratios(values):
    """Consecutive ratios R(eps) / R(eps/2), NaN where a value is zero."""
    v = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return v[:-1] / v[1:]


def plot_remainders(runs, out):
    """R1 and R2 against epsilon, one panel per run, with reference slopes."""
    # The two quadratic terms sit at roundoff and would only show noise.
    runs = sorted((r for r in runs if r["case"] in ("Tobs", "uobs")),
                  key=lambda r: (r["steps"], r["case"], r["visc"], r["levels"]))
    n = len(runs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax, run in zip(axes.flat, runs):
        res = run["data"]["result"]
        eps = np.asarray(res["eps"])
        R1 = np.asarray(res["R1"])
        R2 = np.asarray(res["R2"])
        ax.loglog(eps, R1, "o-", label="R1 (first order remainder)")
        ax.loglog(eps, R2, "s-", label="R2 (second order remainder)")
        # Reference slopes anchored at the smallest epsilon.
        e = np.array([eps[-1], eps[0]])
        ax.loglog(e, R1[-1] * (e / eps[-1]) ** 2, "k:", lw=1, label="slope 2")
        ax.loglog(e, R2[-1] * (e / eps[-1]) ** 3, "k--", lw=1, label="slope 3")
        ax.set_title(run_label(run), fontsize=9)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("remainder")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Second-order Taylor test, cylindrical adjoint case, Gadi, 16 ranks, "
                 "firedrake PR #4638", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "taylor_remainders.png", dpi=150)
    plt.close(fig)


def plot_ratios(runs, out):
    """R2(eps)/R2(eps/2) against epsilon for the misfit terms; 8 means a pure cubic."""
    runs = sorted((r for r in runs if r["case"] in ("Tobs", "uobs")),
                  key=lambda r: (r["steps"], r["case"], r["visc"], r["levels"]))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = {20: "tab:blue", 60: "tab:orange", 125: "tab:green"}
    for run in runs:
        res = run["data"]["result"]
        eps = np.asarray(res["eps"])
        rat = ratios(res["R2"])
        # Plot the ratio at the smaller of the two epsilons it joins.
        x = eps[1:]
        style = "-" if run["visc"] == "smooth floor" else "--"
        marker = "o" if run["case"] == "Tobs" else "^"
        # Only the 60-step sweeps from eps 0.05 are pre-asymptotic; at 20 steps the
        # same window is fine, as the ratios of 8.3 to 8.4 show.
        pre = run["steps"] >= 60 and run["levels"] == 5
        alpha = 0.35 if pre else 1.0
        label = run_label(run, with_sweep=False)
        if pre:
            label += " (pre-asymptotic window)"
        ax.semilogx(x, rat, style, marker=marker, color=colours[run["steps"]],
                    alpha=alpha, label=label)
    ax.axhline(8, color="k", lw=1)
    ax.axhline(4, color="k", lw=0.5, ls=":")
    ax.text(ax.get_xlim()[0] * 1.1, 8.15, "8: pure cubic, Hessian exact", fontsize=8)
    ax.text(ax.get_xlim()[0] * 1.1, 4.15, "4: eps^2 defect dominates", fontsize=8)
    ax.set_ylim(3, 12)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("R2(eps) / R2(eps / 2)")
    ax.set_title("Consecutive ratio of the second-order remainder", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "r2_ratios.png", dpi=150)
    plt.close(fig)


def plot_defect(runs, out):
    """Fitted Hessian defect |D| / <h,Hh> against trajectory length, Tobs only."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for visc, marker, colour in (("smooth floor", "o", "tab:green"), ("hard clip", "s", "tab:red")):
        pts = []
        for run in runs:
            if run["case"] != "Tobs" or run["visc"] != visc:
                continue
            # Prefer the sweep that started inside the series range. The five-level
            # sweeps from 0.05 at 60 steps do not identify D.
            if run["steps"] >= 60 and run["levels"] == 5:
                continue
            d = run["data"]
            fit = d.get("fit3", {}).get("DCE") or {"D": d["fit"]["fits"][0][1]}
            D_rel = abs(fit["D"]) / abs(d["result"]["hHh"])
            pts.append((run["steps"], D_rel))
        pts.sort()
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], marker=marker, color=colour,
                    ls="-", label=visc)
    ax.axhline(3e-6, color="k", ls=":", lw=1)
    ax.text(22, 3.6e-6, "resolution of the test", fontsize=8)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("|D| / <h, H h>   (fitted eps^2 defect)")
    ax.set_title("Hessian defect from the (D, C, E) fit, Tobs term", fontsize=11)
    ax.set_xticks([20, 60, 125])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hessian_defect.png", dpi=150)
    plt.close(fig)


def plot_timings(runs, out):
    """Wall time per call against trajectory length, misfit terms, smooth floor."""
    runs = [r for r in runs if r["case"] in ("Tobs", "uobs") and r["visc"] == "smooth floor"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for key, colour in (("functional", "tab:blue"), ("derivative", "tab:orange"), ("hessian", "tab:green")):
        first, steady = {}, {}
        for run in runs:
            t = run["data"]["result"]["timings"][key]
            first.setdefault(run["steps"], []).append(t[0])
            steady.setdefault(run["steps"], []).append(t[-1])
        steps = sorted(first)
        ax.plot(steps, [np.mean(steady[s]) for s in steps], "o-", color=colour, label=f"{key}, steady")
        ax.plot(steps, [np.mean(first[s]) for s in steps], "o--", color=colour, alpha=0.5, label=f"{key}, first call")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("wall time per call (s), 16 ranks")
    ax.set_title("Cost per call", fontsize=11)
    ax.set_xticks([20, 60, 125])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for key, colour in (("derivative", "tab:orange"), ("hessian", "tab:green")):
        rows = {}
        for run in runs:
            t = run["data"]["result"]["timings"]
            rows.setdefault(run["steps"], []).append(t[key][-1] / t["functional"][-1])
        steps = sorted(rows)
        ax.plot(steps, [np.mean(rows[s]) for s in steps], "o-", color=colour, label=f"{key} / functional, steady")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("ratio to one functional evaluation")
    ax.set_title("Relative cost, steady state", fontsize=11)
    ax.set_xticks([20, 60, 125])
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Cylindrical adjoint case, Gadi, 16 ranks, PR #4638, 2D direct LU. "
                 "Linear sweeps run as newtonls, so these are upper bounds.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "timings.png", dpi=150)
    plt.close(fig)


def write_table(runs, out):
    """Markdown summary table, one row per run."""
    lines = ["| steps | case | rheology | sweep | R1 rate (last) | R2 rate (last) | R2 ratio (last) | D / <h,Hh> | Hessian / functional |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run in sorted(runs, key=lambda r: (r["steps"], r["case"], r["visc"], r["levels"])):
        d = run["data"]
        res = d["result"]
        r1 = d["rates"]["R1"][-1]
        r2 = d["rates"]["R2"][-1]
        rat = ratios(res["R2"])[-1]
        hHh = res["hHh"]
        Jm = res["Jm"]
        if run["case"] in ("damping", "smoothing"):
            r2_text, rat_text, D_text = "roundoff", "", f"R2/J < {max(res['R2']) / abs(Jm):.0e}"
        else:
            r2_text, rat_text = f"{r2:.2f}", f"{rat:.2f}"
            fit = d.get("fit3", {}).get("DCE")
            D = fit["D"] if fit else d["fit"]["fits"][0][1]
            D_text = f"{abs(D) / abs(hHh):.1e}"
            if run["steps"] >= 60 and run["levels"] == 5:
                D_text += " (not identified)"
        t = res["timings"]
        hf = t["hessian"][-1] / t["functional"][-1]
        lines.append(f"| {run['steps']} | {run['case']} | {run['visc']} | eps0 {run['eps0']:g}, {run['levels']} levels "
                     f"| {r1:.3f} | {r2_text} | {rat_text} | {D_text} | {hf:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--results", default=here / "gadi-results")
    parser.add_argument("--out", default=here / "gadi-results" / "figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.results))
    print(f"{len(runs)} runs")
    plot_remainders(runs, out)
    plot_ratios(runs, out)
    plot_defect(runs, out)
    plot_timings(runs, out)
    write_table(runs, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
