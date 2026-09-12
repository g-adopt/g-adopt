"""Analyse the saved four-case paired/refinement study without rerunning solves."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASES = [(10000, 0.25), (10000, 0.5), (10000, 1.0), (100000, 0.5)]


def analyse(results: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    references = json.loads(Path(__file__).with_name("king_reference.json").read_text())

    def load(ra, di, n, formulation="full"):
        return json.loads((results / f"ra{ra}_di{di}_n{n}_{formulation}.json").read_text())

    summary = []
    for ra, di in CASES:
        fine, coarse = load(ra, di, 64), load(ra, di, 32)
        ref = [r for r in references if r["ra"] == ra and r["di"] == di]
        vt = next(r for r in ref if r["code"] == "VT")
        full = np.load(results / f"ra{ra}_di{di}_n64_full.npz")
        pert = np.load(results / f"ra{ra}_di{di}_n64_perturbation.npz")
        np.testing.assert_allclose(full["xy"], pert["xy"], rtol=0, atol=1e-14)
        summary.append({
            "ra": ra, "di": di, "nu": fine["nu_top_reaction"], "vrms": fine["vrms"],
            "mean_temperature": fine["mean_surface_relative_temperature"],
            "nu_vt": vt["nu"], "vrms_vt": vt["vrms"],
            "nu_vs_vt_percent": 100*(fine["nu_top_reaction"]/vt["nu"]-1),
            "vrms_vs_vt_percent": 100*(fine["vrms"]/vt["vrms"]-1),
            "nu_intercode_range": [min(r["nu"] for r in ref), max(r["nu"] for r in ref)],
            "vrms_intercode_range": [min(r["vrms"] for r in ref), max(r["vrms"] for r in ref)],
            "nu_mesh_change_percent": 100*(fine["nu_top_reaction"]/coarse["nu_top_reaction"]-1),
            "vrms_mesh_change_percent": 100*(fine["vrms"]/coarse["vrms"]-1),
            "paired_temperature_max_nodal": float(np.max(np.abs(full["temperature"]-pert["temperature"]))),
            "paired_velocity_relative_nodal_l2": float(
                np.linalg.norm(full["velocity"]-pert["velocity"])
                / np.linalg.norm(full["velocity"])),
            "viscous_heating": fine["viscous_heating"],
            "adiabatic_work": fine["adiabatic_work_full"],
            "heating_work_imbalance_percent": (
                100*(fine["viscous_heating"]-fine["adiabatic_work_full"])
                / fine["viscous_heating"]),
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, (ra, di) in zip(axes.flat, CASES):
        data = np.load(results / f"ra{ra}_di{di}_n64_full.npz")
        xy, velocity = data["xy"], data["velocity"]
        count = int(np.sqrt(len(xy)))
        assert count**2 == len(xy)
        X, Y = xy[:, 0].reshape(count, count).T, xy[:, 1].reshape(count, count).T
        im = ax.pcolormesh(X, Y, data["temperature"].reshape(count, count).T-0.091,
                           vmin=0, vmax=1, cmap="inferno", shading="auto", rasterized=True)
        ax.streamplot(X[0], Y[:, 0], velocity[:, 0].reshape(count, count).T,
                      velocity[:, 1].reshape(count, count).T, color="cyan", density=0.8,
                      linewidth=0.55, arrowsize=0.75)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y (upward)",
               title=f"Ra = {ra:g}, Di = {di:g}", aspect="equal")
    fig.colorbar(im, ax=list(axes.flat), label="Surface-relative temperature", shrink=0.8)
    fig.suptitle("Full-temperature TALA: converged single-roll solutions (64 × 64 cells)")
    fig.savefig(output / "temperature_fields.png", dpi=180)
    fig.savefig(output / "temperature_fields.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (ra, di) in zip(axes.flat, CASES):
        sizes = [16, 32, 64]
        if (results / f"ra{ra}_di{di}_n128_full.json").exists():
            sizes.append(128)
        rows = [load(ra, di, n) for n in sizes]
        finest = rows[-1]["nu_top_reaction"]
        for key, label, marker in [
            ("nu_top_gradient", "Boundary gradient", "x"),
            ("nu_top_reaction", "Weak-form reaction", "o"),
        ]:
            ax.plot(sizes, [100*(r[key]/finest-1) for r in rows], marker=marker, label=label)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels=sizes)
        ax.set(xlabel="Cells per side", ylabel="Difference from finest reaction Nu (%)",
               title=f"Ra = {ra:g}, Di = {di:g}")
        ax.legend(fontsize=8)
    fig.suptitle("Boundary-gradient heat flux converges more slowly")
    fig.savefig(output / "heat_flux_convergence.png", dpi=180)
    plt.close(fig)

    lines = ["# Cartesian full-temperature TALA: numerical results", "",
             "Four King cases, paired temperature formulations on 16/32/64-cell meshes.", "",
             "| Ra | Di | Nu (64 cells) | Vrms | Mean surface-relative T | Nu vs VT (%) | Vrms vs VT (%) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary:
        lines.append(f"| {r['ra']:g} | {r['di']:g} | {r['nu']:.6f} | {r['vrms']:.6f} "
                     f"| {r['mean_temperature']:.6f} | {r['nu_vs_vt_percent']:+.4f} "
                     f"| {r['vrms_vs_vt_percent']:+.4f} |")
    lines.extend(["", "Nu here is the integrated weak-form boundary reaction, normalised by unit conductive flux.",
                  "VT is one published discretisation, not an exact reference solution.", "",
                  "| Ra | Di | 32-to-64 Nu change (%) | Vrms change (%) | Max paired nodal T difference | Phi minus W (% of Phi) |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for r in summary:
        lines.append(f"| {r['ra']:g} | {r['di']:g} | {r['nu_mesh_change_percent']:+.6f} "
                     f"| {r['vrms_mesh_change_percent']:+.6f} | {r['paired_temperature_max_nodal']:.3e} "
                     f"| {r['heating_work_imbalance_percent']:.4f} |")
    (output / "TABLES.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.results, args.output)
