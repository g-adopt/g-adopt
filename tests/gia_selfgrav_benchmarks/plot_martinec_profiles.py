r"""Plot saved Martinec B/C/D runs against VEGA without Firedrake.

Usage:
    python3 plot_martinec_profiles.py outputs/B_ladder/summary_B.json --labels Ladder
    python3 plot_martinec_profiles.py outputs/B_ladder/summary_B.json \
        outputs/B_uniform/summary_B.json --labels Ladder "Uniform 10 yr" --output outputs/B_plots

Requires numpy, scipy, matplotlib and giamip. Run on a login node or a local
machine with reference data cached; giamip may download references on first use.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline

import giamip
from giamip.benchmarks import metrics
from giamip.benchmarks.results import BenchmarkResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Match the existing benchmark test's historical exclusion width. This is
# a scoring choice, not the current mesh's 78 km spacing or a paper prescription.
DEFAULT_BAND_DEG = 1.5 * np.degrees(500.0 / 6371.0)
QUANTITIES = {"U": "Vertical displacement U", "N": "Geoid displacement N",
              "S": "Sea-surface variation S"}


@dataclass
class Run:
    path: Path
    label: str
    summary: dict
    series: dict
    profiles: dict
    result: BenchmarkResult


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_run(path, label):
    """Read and validate complete saved outputs without changing their arrays."""
    summary = json.loads(path.read_text())
    letter = summary["case"]
    if letter not in ("B", "C", "D"):
        raise ValueError(f"{path}: spatial reference profiles are supported for B, C and D")
    case = giamip.case(f"martinec2018-{letter}")
    if summary["smoke"]:
        raise ValueError(f"{path}: expected a complete benchmark, not a smoke run")
    series = read_npz(path.parent / summary["timeseries_file"])
    t = series["t_kyr"]
    if (t.ndim != 1 or t.size == 0 or not np.isfinite(t).all()
            or not np.all(np.diff(t) > 0) or t[0] < 0
            or not np.isclose(t[-1], case.terminal_time_kyr)):
        raise ValueError(f"{path}: time series must ascend to {case.terminal_time_kyr:g} kyr")
    if summary["steps"] != t.size:
        raise ValueError(f"{path}: summary step count differs from time series")
    for key in ("h_UF_m", "dt_yr"):
        if series[key].shape != t.shape or not np.isfinite(series[key]).all():
            raise ValueError(f"{path}: invalid time series {key}")
    if np.any(series["dt_yr"][t > 0] <= 0):
        raise ValueError(f"{path}: marched timestep lengths must be positive")
    profiles = {}
    for filename in summary["profile_files"]:
        data = read_npz(path.parent / filename)
        epoch = float(data["t_kyr"])
        if not np.isfinite(epoch) or epoch in profiles:
            raise ValueError(f"{filename}: invalid or duplicate profile epoch")
        if not np.any(np.isclose(t, epoch, rtol=0, atol=1e-9)):
            raise ValueError(f"{filename}: epoch has no solved time-series sample")
        for spec in case.profiles:
            x = data[f"{spec.name}_colatitude_deg"]
            found = data[f"{spec.name}_found"]
            if (x.ndim != 1 or x.size < 2 or not np.isfinite(x).all()
                    or not np.all(np.diff(x) > 0) or x[0] < 0 or x[-1] > 180):
                raise ValueError(f"{filename}: invalid {spec.name} colatitudes")
            if found.shape != x.shape or not found.all():
                raise ValueError(f"{filename}: unlocated {spec.name} profile points")
            if not np.isclose(float(data[f"{spec.name}_longitude_deg"]), spec.longitude_deg):
                raise ValueError(f"{filename}: incorrect {spec.name} longitude")
            for q in spec.quantities:
                y = data[f"{spec.name}_{q}_m"]
                if y.shape != x.shape or not np.isfinite(y).all():
                    raise ValueError(f"{filename}: invalid {spec.name}/{q} values")
        profiles[epoch] = data
    terminal = [data for epoch, data in profiles.items()
                if np.isclose(epoch, case.terminal_time_kyr)]
    if len(terminal) != 1:
        raise ValueError(f"{path}: missing unique terminal profile at {case.terminal_time_kyr:g} kyr")
    result = BenchmarkResult(letter, label)
    for spec in case.profiles:
        for q in spec.quantities:
            result.add_profile(spec.name, q, terminal[0][f"{spec.name}_colatitude_deg"],
                               terminal[0][f"{spec.name}_{q}_m"])
    result.add_time_series("h_UF", t, series["h_UF_m"])
    return Run(path.resolve(), label, summary, series, dict(sorted(profiles.items())), result)


def save_figure(fig, directory, name):
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=180)
        print(f"Saved {path}")
        paths.append(path.name)
    plt.close(fig)
    return paths


def mark_features(ax, case, spec, band=None):
    for feature in case.feature_colatitudes(spec.name):
        if band is not None and band > 0:
            ax.axvspan(feature - band, feature + band, color="0.5", alpha=.12, linewidth=0)
        ax.axvline(feature, color="tab:orange", ls=":", lw=1)
    ax.set_xlim(spec.colatitude_interval_deg)
    ax.grid(alpha=.2)


def load_contributors(letter):
    """Read published members, excluding VEGA (ZM) already shown as reference."""
    path = BenchmarkResult.reference_path(letter)
    return [BenchmarkResult.read(path, member, case=letter)
            for member in BenchmarkResult.members(path) if not member.startswith("ZM")]


def plot_terminal(runs, case, reference, spec, band, contributors=()):
    fig, axes = plt.subplots(2, len(spec.quantities), squeeze=False, sharex="col",
                             figsize=(4.6 * len(spec.quantities), 6.6), layout="constrained")
    for j, q in enumerate(spec.quantities):
        ref = reference.profile(spec.name, q)
        interpolate_reference = CubicSpline(ref.axis, ref.values)
        lo, hi = spec.colatitude_interval_deg
        xref = np.linspace(lo, hi, 1001)
        for i, contributor in enumerate(contributors):
            # Retain each archive curve's native grid. In the residual panel,
            # use the published differences, as the scoring envelopes do.
            for ax, curves in zip(axes[:, j], (contributor.profiles, contributor.differences)):
                curve = curves.get((spec.name, q))
                if curve is None:
                    continue  # For example, WW did not supply basin S in B.
                keep = (curve.axis >= lo) & (curve.axis <= hi)
                ax.plot(curve.axis[keep], curve.values[keep], color=plt.get_cmap("Dark2")(i % 8),
                        ls=("-", "--", "-.", ":")[i % 4], lw=.9, alpha=.8,
                        label=contributor.member, zorder=2)
        axes[0, j].plot(xref, interpolate_reference(xref), "k--", lw=1.7,
                        label="VEGA reference", zorder=3)
        for i, run in enumerate(runs):
            curve = run.result.profile(spec.name, q)
            keep = (curve.axis >= lo) & (curve.axis <= hi)
            x, y = curve.axis[keep], curve.values[keep]
            axes[0, j].plot(x, y, color=f"C{i % 10}", lw=1.8, label=run.label, zorder=4)
            axes[1, j].plot(x, y - interpolate_reference(x), color=f"C{i % 10}",
                            lw=1.8, label=run.label, zorder=4)
        axes[0, j].set_title(QUANTITIES[q])
        axes[0, j].set_ylabel("m, positive upward")
        axes[1, j].axhline(0, color="black", lw=.7)
        axes[1, j].set_ylabel("Difference from VEGA (m)" if contributors else "Run − VEGA (m)")
        axes[1, j].set_xlabel("Colatitude (degrees)")
        for ax in axes[:, j]:
            mark_features(ax, case, spec, band)
    if contributors:
        handles = {}
        for ax in axes.flat:
            lines, labels = ax.get_legend_handles_labels()
            handles.update(zip(labels, lines))
        fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncols=5, fontsize=9)
    else:
        axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Martinec {case.letter}: {spec.name} meridian, longitude {spec.longitude_deg:g}°, "
                 f"{case.terminal_time_kyr:g} kyr\n"
                 f"Orange: cap margins / coastline; shaded ±{band:.3g}° bands excluded from scores")
    return fig


def plot_evolution(runs, case, spec):
    fig, axes = plt.subplots(len(runs), len(spec.quantities), squeeze=False, sharex=True,
                             figsize=(4.6 * len(spec.quantities), 3.1 * len(runs)), layout="constrained")
    epochs = sorted({epoch for run in runs for epoch in run.profiles})
    colors = {epoch: plt.get_cmap("viridis")(i / max(1, len(epochs) - 1)) for i, epoch in enumerate(epochs)}
    lo, hi = spec.colatitude_interval_deg
    for row, run in zip(axes, runs):
        for ax, q in zip(row, spec.quantities):
            for epoch, data in run.profiles.items():
                x = data[f"{spec.name}_colatitude_deg"]
                keep = (x >= lo) & (x <= hi)
                ax.plot(x[keep], data[f"{spec.name}_{q}_m"][keep], color=colors[epoch], label=f"{epoch:g} kyr")
            ax.set_title(f"{run.label}: {QUANTITIES[q]}")
            ax.set_ylabel("m, positive upward")
            ax.set_xlabel("Colatitude (degrees)")
            mark_features(ax, case, spec)
        row[0].legend(fontsize=8)
    fig.suptitle(f"Martinec {case.letter}: {spec.name} profile evolution — saved G-ADOPT states")
    return fig


def plot_history(runs, case, reference):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, layout="constrained")
    for i, run in enumerate(runs):
        t = run.series["t_kyr"]
        axes[0].plot(t, run.series["h_UF_m"], color=f"C{i % 10}", label=run.label)
        positive = t > 0  # The tiny elastic diagnostic is labelled t=0, not an elapsed interval.
        steps = run.series["dt_yr"][positive]
        axes[1].step(np.r_[0., t[positive]], np.r_[steps[0], steps], where="pre",
                     color=f"C{i % 10}", label=run.label)
    if "h_UF" in reference.time_series:
        curve = reference.series("h_UF")
        axes[0].plot(curve.axis, curve.values, "k--", label="VEGA reference")
    else:
        # Without a reference history (as in B), S-N gives only the terminal value.
        value, _ = metrics.uniform_layer(case.letter)
        axes[0].plot([case.terminal_time_kyr], [value], "k*", ms=11,
                     label=f"VEGA at {case.terminal_time_kyr:g} kyr")
    axes[0].set_ylabel(r"Uniform water layer $h_{UF}$ (m)")
    axes[1].set_ylabel("Timestep (yr)")
    axes[1].set_xlabel("Time (kyr)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlim(0, case.terminal_time_kyr)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle(f"Martinec {case.letter}: water-layer history and marched timesteps")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path, help="one or more summary_B/C/D.json files for the same case")
    parser.add_argument("--labels", nargs="+", help="one label per summary; defaults to parent folder names")
    parser.add_argument("--output", type=Path, help="figure directory; default: plots/ beside the first summary")
    parser.add_argument("--contributors", action="store_true",
                        help="overlay other published codes and their archived differences from VEGA at terminal time")
    parser.add_argument("--band", type=float, default=DEFAULT_BAND_DEG,
                        help="exclusion-band half width in degrees; default matches test_benchmarks.py (6.744912°)")
    args = parser.parse_args(argv)
    if not np.isfinite(args.band) or args.band < 0:
        parser.error("--band must be finite and nonnegative")
    labels = args.labels if args.labels is not None else [path.parent.name for path in args.summaries]
    if len(labels) != len(args.summaries) or len(set(labels)) != len(labels):
        parser.error("provide one unique label per summary with --labels")
    runs = [load_run(path, label) for path, label in zip(args.summaries, labels)]
    if len({run.result.case for run in runs}) != 1:
        parser.error("all summaries must describe the same case")
    letter = runs[0].result.case
    case = giamip.case(f"martinec2018-{letter}")
    reference = BenchmarkResult.reference(letter)
    contributors = load_contributors(letter) if args.contributors else []
    # Score before rendering; missing reference curves or uncovered intervals
    # should fail before leaving behind a partly generated figure collection.
    scores = [{"run": run.label, **row} for run in runs
              for row in metrics.envelope_check(run.result, band_half_width_deg=args.band)]
    directory = args.output or args.summaries[0].parent / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"font.size": 10}):
        for spec in case.profiles:
            paths += save_figure(plot_terminal(runs, case, reference, spec, args.band, contributors), directory,
                                 f"martinec_{letter}_{spec.name}_terminal")
            paths += save_figure(plot_evolution(runs, case, spec), directory,
                                 f"martinec_{letter}_{spec.name}_evolution")
        paths += save_figure(plot_history(runs, case, reference), directory, f"martinec_{letter}_history")
    with (directory / "profile_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    manifest = {"case": letter, "giamip_version": giamip.__version__, "band_half_width_deg": float(args.band),
                "contributors": [result.member for result in contributors],
                "runs": [{"label": run.label, "summary": str(run.path),
                          "mesh_md5": run.summary.get("mesh", {}).get("md5")} for run in runs],
                "figures": paths}
    (directory / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Profile scores and plot manifest written to {directory.resolve()}")


if __name__ == "__main__":
    main()
