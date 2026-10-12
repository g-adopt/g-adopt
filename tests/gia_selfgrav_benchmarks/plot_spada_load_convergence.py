"""Compare available direct cap runs with uncorrected TABOO degrees 2–256.

Run with NumPy and Matplotlib, without Firedrake:
    python plot_spada_load_convergence.py

Automatically includes output subdirectories containing summaries and profiles.
The legacy degree-10 baseline uses its complete projected spectrum when direct
profiles are absent; it is explicitly labelled as a reconstructed result.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legder, legval

from plot_spada_profiles import taboo_coefficients


HERE = Path(__file__).resolve().parent
QUANTITIES = ("U", "V", "N")
NAMES = ("Radial displacement U", "Southward displacement V", "Geoid N")


def evaluate(coefficients, theta_deg, quantity):
    theta = np.deg2rad(theta_deg)
    if quantity == "V":
        values = -np.sin(theta) * legval(np.cos(theta), legder(coefficients))
        values[np.isclose(theta_deg, 0) | np.isclose(theta_deg, 180)] = 0
        return values
    return legval(np.cos(theta), coefficients)


def load_runs(root):
    runs = []
    for path in sorted(root.glob("*/summary_cap.json")):
        summary = json.loads(path.read_text())
        if summary["case"] != "cap" or summary["smoke"]:
            continue
        representation = summary.get("load_representation", "spectral")
        cutoff = summary.get("load_nmax", summary.get("nmax", 10))
        label = "G-ADOPT spatial (subtract degs. 0-1)" if representation == "spatial" else f"G-ADOPT load 2–{cutoff}"
        if all((path.parent / f"profiles_cap_{e['t_kyr']:g}kyr.npz").exists()
               for e in summary["epochs"]):
            runs.append((cutoff or float("inf"), label, summary, path.parent))
    baseline = HERE / "summary_cap.json"
    if baseline.exists() and not any(r[0] == 10 for r in runs):
        summary = json.loads(baseline.read_text())
        if summary["case"] == "cap" and not summary["smoke"] and summary.get("nmax") == 10:
            runs.append((10, "G-ADOPT load 2–10 (reconstructed)", summary, None))
    return sorted(runs, key=lambda run: run[0])


def model_profile(run, epoch, theta):
    _, _, _, directory = run
    if directory is not None:
        with np.load(directory / f"profiles_cap_{epoch['t_kyr']:g}kyr.npz") as data:
            if not np.array_equal(data["colatitude_deg"], theta):
                raise ValueError(f"Different sampling grid in {directory}")
            if not np.isclose(float(data["t_kyr"]), epoch["t_kyr"]):
                raise ValueError("Profile epoch mismatch")
            if not np.all(data["found"]):
                raise ValueError(f"Missing profile points in {directory}")
            return {q: np.where(data["found"], data[f"{q}_m"], np.nan)
                    for q in QUANTITIES}
    result = {}
    degrees = np.asarray(epoch["degrees"], dtype=int)
    for q in QUANTITIES:
        coefficients = np.zeros(int(degrees.max()) + 1)
        coefficients[degrees] = epoch[f"{q}_n"]
        result[q] = evaluate(coefficients, theta, q)[None, :]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=HERE / "outputs")
    parser.add_argument("--max-colatitude", type=float, default=25)
    args = parser.parse_args()
    if not 0 < args.max_colatitude <= 180:
        parser.error("--max-colatitude must be in (0, 180]")
    runs = load_runs(args.outputs)
    if not runs:
        parser.error("No complete direct-profile runs found")
    theta = np.linspace(0, 180, 1801)
    selected = theta <= args.max_colatitude
    reference = np.load(HERE / "reference.npz", allow_pickle=False)
    epochs = sorted(set.intersection(*[{e["t_kyr"] for e in r[2]["epochs"]} for r in runs]))
    if not epochs:
        parser.error("Runs have no common epochs")
    showcase = [t for t in (0, 5, 20) if t in epochs]
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(runs)))
    profiles, errors = [plt.subplots(len(showcase), 3, figsize=(14, 3.1 * len(showcase)),
                                    squeeze=False, layout="constrained", sharex=True)
                        for _ in range(2)]
    # Degree 128 is the current practical baseline; higher cutoff is not
    # uniformly more accurate on this mesh. Fall back to the largest cutoff.
    best = next((r for r in runs if r[0] == 128), runs[-1])
    best_figure, best_axes = plt.subplots(len(epochs), 3, figsize=(14, 2.55 * len(epochs)),
                                        squeeze=False, layout="constrained", sharex=True)
    records = []
    for epoch_index, t in enumerate(epochs):
        coefficients = taboo_coefficients(reference, t)
        refs = {q: evaluate(coefficients[q], theta, q) for q in QUANTITIES}
        for run_index, run in enumerate(runs):
            epoch = next(e for e in run[2]["epochs"] if e["t_kyr"] == t)
            values = model_profile(run, epoch, theta)
            for col, q in enumerate(QUANTITIES):
                residual = values[q] - refs[q][None, :]
                for region, mask in [("near_cap", selected), ("global", np.ones(theta.size, bool))]:
                    amplitude = np.max(np.abs(refs[q][mask]))
                    rms = np.sqrt(np.mean(residual[:, mask] ** 2))
                    maximum = np.max(np.abs(residual[:, mask]))
                    records.append(dict(load=run[1], epoch_kyr=t, quantity=q, region=region,
                                        rms_error_m=rms, max_error_m=maximum,
                                        rms_pct_reference_peak=100*rms/amplitude,
                                        max_pct_reference_peak=100*maximum/amplitude,
                                        max_meridian_spread_m=np.max(np.ptp(values[q][:, mask], axis=0))))
                if t in showcase:
                    row = showcase.index(t)
                    for axes, fields in [(profiles[1], values[q]), (errors[1], residual)]:
                        ax = axes[row, col]
                        ax.plot(theta[selected], fields[0, selected], color=colors[run_index],
                                linewidth=1.5, label=run[1])
                if run is best:
                    ax = best_axes[epoch_index, col]
                    with np.load(run[3] / f"profiles_cap_{t:g}kyr.npz") as data:
                        longitudes = data["longitude_deg"]
                    for longitude, field in zip(longitudes, values[q]):
                        ax.plot(theta[selected], field[selected], linewidth=1.1,
                                label=f"{best[1]}, {longitude:g}° meridian")
                    ax.plot(theta[selected], refs[q][selected], "k--", linewidth=1.7,
                            label="TABOO 2–256")
                    ax.set_title(f"{NAMES[col]} — {t:g} kyr")
    for row, t in enumerate(showcase):
        coefficients = taboo_coefficients(reference, t)
        for col, q in enumerate(QUANTITIES):
            profiles[1][row, col].plot(theta[selected], evaluate(coefficients[q], theta, q)[selected],
                                       "k--", linewidth=1.8, label="TABOO 2–256")
            errors[1][row, col].axhline(0, color="black", linestyle="--", linewidth=1)
            for axes in (profiles[1], errors[1]):
                axes[row, col].set_title(f"{NAMES[col]} — {t:g} kyr")
    for figure, axes, filename, title in [
        (*profiles, "load_convergence.png", "Load-degree comparison against TABOO"),
        (*errors, "load_convergence_errors.png", "G-ADOPT minus TABOO (metres)"),
        (best_figure, best_axes, "best_vs_taboo.png", f"{best[1]} against TABOO, all epochs")]:
        for ax in axes.flat:
            ax.axvline(10, color="grey", linestyle=":", linewidth=.8)
            ax.set_ylabel("m")
            ax.grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel("Colatitude (degrees)")
        axes[0, 0].legend(fontsize=7)
        detail = ("All three direct meridians, sampled ≈6.4 m below surface"
                  if filename == "best_vs_taboo.png" else
                  "0° direct meridian; samples ≈6.4 m below surface")
        figure.suptitle(title + "\n" + detail, fontsize=12)
        figure.savefig(args.outputs / filename, dpi=180)
        figure.savefig(args.outputs / filename.replace(".png", ".pdf"))
        plt.close(figure)
        print(f"Saved {args.outputs / filename}")
    with (args.outputs / "load_convergence_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for col, q in enumerate(QUANTITIES):
        for t in showcase:
            rms = [next(e["rms_error_m"] for e in records
                        if e["load"] == run[1] and e["epoch_kyr"] == t
                        and e["quantity"] == q and e["region"] == "near_cap") for run in runs]
            axes[col].plot(range(len(runs)), rms, "o-", label=f"{t:g} kyr")
        axes[col].set_xticks(range(len(runs)), ["spatial\n(subtract degs. 0-1)" if np.isinf(r[0]) else str(r[0]) for r in runs])
        axes[col].set_xlabel("Load cutoff (spatial shown separately)")
        axes[col].set_ylabel("RMS difference from TABOO (m)")
        axes[col].set_yscale("log")
        axes[col].set_title(NAMES[col])
        axes[col].grid(alpha=.2)
    axes[0].legend()
    figure.suptitle(f"Load convergence: 0–{args.max_colatitude:g}° RMS error, all sampled meridians\n"
                   "TABOO 2–256; degree-10 baseline reconstructed from its spectrum")
    for extension in ("png", "pdf"):
        figure.savefig(args.outputs / f"load_convergence_rms.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
