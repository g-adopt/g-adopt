"""Plot nonrotating cap/disc baseline profiles against Spada/TABOO.

Run: python3 plot_spada_baselines.py outputs/spada-study
Saves a three-row, two-column figure as PDF/PNG, plus input provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_spada_profiles import reference_profiles  # noqa: E402

# Okabe–Ito palette, with yellow omitted for contrast on a white background.
TIME_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
                "#D55E00", "#56B4E9", "#000000")


def read_profile(directory, summary, t, longitude, maximum, hashes):
    """Select an actual saved meridian, without averaging spatial differences."""
    if not any(np.isclose(epoch["t_kyr"], t) for epoch in summary["epochs"]):
        raise ValueError(f"No saved epoch at {t:g} kyr in {directory}")
    path = directory / f"profiles_{summary['case']}_{t:g}kyr.npz"
    with np.load(path, allow_pickle=False) as archive:
        profile = dict(archive)
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if str(profile["profile_kind"]) != "meridian":
        raise ValueError(f"Expected north-pole meridians in {path}")
    if not np.isclose(float(profile["t_kyr"]), t):
        raise ValueError(f"Wrong profile epoch in {path}")
    difference = (profile["longitude_deg"] - longitude + 180) % 360 - 180
    matches = np.flatnonzero(np.isclose(difference, 0, atol=1e-10, rtol=0))
    if matches.size != 1:
        raise ValueError(f"Expected one saved meridian at longitude {longitude:g}° in {path}")
    index = int(matches[0])
    coordinate = profile["coordinate_deg"]
    if maximum > coordinate[-1] + 1e-9:
        raise ValueError(f"Requested extent exceeds saved profile in {path}")
    keep = coordinate <= maximum + 1e-9
    if not profile["found"][index, keep].all():
        raise ValueError(f"Missing sample points on the selected meridian in {path}")
    for key in ("U_m", "along_m", "N_m"):
        if not np.isfinite(profile[key][index, keep]).all():
            raise ValueError(f"Nonfinite {key} on the selected meridian in {path}")
    return profile, index, keep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("outputs/spada-study"))
    parser.add_argument("--times-kyr", nargs="+", type=float, default=[0., 1., 5., 20.])
    parser.add_argument("--longitude", type=float, default=0., help="saved meridian in degrees")
    parser.add_argument("--max-colatitude", type=float, default=25.)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/spada-study-analysis/spada_nonrotating_baselines"),
                        help="output stem; writes .pdf, .png and .json")
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_colatitude) or not 0 < args.max_colatitude <= 180:
        parser.error("--max-colatitude must be in (0, 180]")
    if not np.isfinite(args.longitude):
        parser.error("--longitude must be finite")
    if (not all(np.isfinite(t) and t >= 0 for t in args.times_kyr)
            or len(set(args.times_kyr)) != len(args.times_kyr)):
        parser.error("--times-kyr must contain distinct finite nonnegative times")

    colours = [TIME_COLOURS[i % len(TIME_COLOURS)] for i in range(len(args.times_kyr))]
    quantities = (("U", "Radial displacement (m)"),
                  ("along", "Tangential displacement (m)"),
                  ("N", "Geoid height (m)"))
    hashes = {str(args.reference): hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 13, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.2), sharex=True, sharey="row")
        with np.load(args.reference, allow_pickle=False) as reference:
            nmax = int(reference["degrees"][-1])
            for column, shape in enumerate(("cap", "disc")):
                directory = args.root / f"{shape}-off" / "baseline"
                summary_path = directory / "summary_cap.json"
                summary = json.loads(summary_path.read_text())
                hashes[str(summary_path)] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
                if (summary["rotation"] or summary["smoke"] or summary["case"] != "cap"
                        or summary["load_shape"] != shape
                        or summary["load_representation"] != "spatial"
                        or not np.isclose(summary["load_centre_deg"][0], 0.)):
                    raise ValueError(f"Expected a nonrotating, north-pole spatial {shape} run: {summary_path}")
                axes[0, column].set_title(f"{shape.capitalize()} load", pad=12)
                for t, colour in zip(args.times_kyr, colours):
                    profile, index, keep = read_profile(
                        directory, summary, t, args.longitude, args.max_colatitude, hashes)
                    x = profile["coordinate_deg"][keep]
                    curves = reference_profiles(reference, t, profile, summary)
                    for row, (quantity, _) in enumerate(quantities):
                        ax = axes[row, column]
                        ax.plot(x, profile[quantity + "_m"][index, keep],
                                color=colour, linewidth=1.9, zorder=2)
                        ax.plot(x, curves[quantity][index, keep], color=colour,
                                linestyle="none", marker="x", markersize=4,
                                markeredgewidth=.9, markevery=.045, zorder=3)
                for row, (_, label) in enumerate(quantities):
                    ax = axes[row, column]
                    ax.set_xlim(0, args.max_colatitude)
                    ax.axvline(10, color="0.65", linestyle=":", linewidth=.9, zorder=0)
                    ax.axhline(0, color="0.75", linewidth=.6, zorder=0)
                    ax.grid(alpha=.18, linewidth=.6)
                    ax.text(.96, .96,
                            f"({chr(ord('a') + 2*row + column)})", transform=ax.transAxes,
                            ha="right", va="top")
                    if column == 0:
                        ax.set_ylabel(label)
                axes[-1, column].set_xlabel("Angular distance from load centre (°)")

        lower, upper = axes[0, 0].get_ylim()
        axes[0, 0].set_ylim(lower, upper + .07 * (upper - lower))

        times = [Line2D([], [], color=colour, linewidth=2,
                        label="Elastic (0 kyr)" if t == 0 else f"{t:g} kyr")
                 for t, colour in zip(args.times_kyr, colours)]
        methods = [Line2D([], [], color="0.2", linewidth=1.9, label="G-ADOPT baseline"),
                   Line2D([], [], color="0.2", linestyle="none", marker="x",
                          markersize=4, markeredgewidth=.9, label="TABOO")]
        axes[0, 0].legend(handles=[*times, *methods], loc="lower right",
                          fontsize=8, handlelength=2.8, borderaxespad=1,
                          frameon=True, facecolor="white", edgecolor="black",
                          framealpha=1, fancybox=False)
        fig.subplots_adjust(left=.12, right=.98, bottom=.07, top=.95, hspace=.14, wspace=.09)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = args.output.with_suffix("." + extension)
            fig.savefig(path, dpi=220)
            print(f"Saved {path}")
        plt.close(fig)

    provenance = {"times_kyr": args.times_kyr, "longitude_deg": args.longitude % 360,
                  "colour_palette": "Okabe–Ito (yellow omitted)", "time_colours_hex": colours,
                  "max_colatitude_deg": args.max_colatitude, "reference_degrees": [2, nmax],
                  "tangential_positive": "away from the north-pole load centre",
                  "inputs_sha256": hashes,
                  "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__}
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
