"""Plot nonrotating cap/disc profiles against TABOO and individual variants.

Run: python3 plot_spada_baselines.py outputs/spada-study
Variants: python3 plot_spada_baselines.py outputs/spada-study --all-variants
Rates: python3 plot_spada_baselines.py outputs/spada-study --rates
Fine mesh: python3 plot_spada_baselines.py outputs/spada-study --baseline-run mesh125
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
VARIANTS = {
    "dt2": ("G-ADOPT: timestep ×2", {"timestep_scale": 2.}),
    "mesh500": ("G-ADOPT: 500 km mesh", {"mesh_h_km": 500.}),
    "litho1": ("G-ADOPT: 1 lithosphere layer", {"litho_layers": 1}),
    "bulk100": ("G-ADOPT: K/μ = 100", {"bulk_shear_ratio": 100.}),
    "dtn2": ("G-ADOPT: DtN degree 2", {"dtn_degree": 2}),
    "mesh125": ("G-ADOPT: 125 km mesh", {"mesh_h_km": 125.}),
    "offaxis": ("G-ADOPT: off-axis load", {"load_centre_deg": [25., 75.]}),
}
CONTROLS = ("case", "rotation", "load_shape", "load_centre_deg", "load_representation",
            "load_nmax", "diagnostic_nmax", "timestep_scale", "mesh_h_km", "litho_layers",
            "bulk_shear_ratio", "dtn_degree", "outer_rtol", "block0_rtol")


def summary_path(root, shape, variant=None):
    scenario = f"{shape}-offaxis-off" if variant == "offaxis" else f"{shape}-off"
    directory = "baseline" if variant in (None, "offaxis") else variant
    return root / scenario / directory / "summary_cap.json"


def read_summary(path, shape, hashes, *, offaxis=False):
    summary = json.loads(path.read_text())
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    centre = [25., 75.] if offaxis else [0., 0.]
    if (summary["rotation"] or summary["smoke"] or summary["case"] != "cap"
            or summary["load_shape"] != shape or summary["load_representation"] != "spatial"
            or not np.allclose(summary["load_centre_deg"], centre, rtol=0, atol=1e-12)):
        raise ValueError(f"Expected a nonrotating spatial {shape} run at {centre}: {path}")
    return summary


def validate_comparison(baseline, changed, variant):
    """Do not label a run by a parameter unless its saved settings match."""
    overrides = VARIANTS[variant][1]
    for key in CONTROLS:
        expected = overrides.get(key, baseline.get(key))
        if changed.get(key) != expected:
            raise ValueError(f"{variant}: expected {key}={expected}, found {changed.get(key)}")
    if all(baseline.get(key) == changed.get(key) for key in overrides):
        raise ValueError(f"{variant}: advertised setting is identical to baseline")


def validate_sampling(base, changed, base_index, changed_index, base_keep, changed_keep, *, offaxis=False):
    keys = ["coordinate_deg", "depth_fraction"]
    if not offaxis:
        keys += ["unit_vectors", "tangent_vectors", "right_vectors"]
    for key in keys:
        a, b = base[key], changed[key]
        if key == "coordinate_deg":
            a, b = a[base_keep], b[changed_keep]
        elif key != "depth_fraction":
            a, b = a[base_index, base_keep], b[changed_index, changed_keep]
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-10):
            raise ValueError(f"Variant profile {key} differs; comparisons require matching samples")


def rate_interval(profile, t):
    """Validate the preceding solved-step interval represented by a saved rate."""
    keys = ("rate_valid", "rate_interval_start_kyr", "rate_interval_end_kyr", "rate_dt_yr")
    if any(key not in profile or np.asarray(profile[key]).shape != () for key in keys):
        raise ValueError("Rate profiles require scalar interval metadata")
    if t <= 0 or not bool(profile["rate_valid"]):
        raise ValueError("Saved rates are undefined at the elastic instant or marked invalid")
    start, end, dt = (float(profile[key]) for key in keys[1:])
    if (not np.isfinite([start, end, dt]).all() or not 0 <= start < end or dt <= 0
            or not np.isclose(end, t, rtol=0, atol=1e-10)
            or not np.isclose((end-start)*1000, dt, rtol=1e-10, atol=1e-9)):
        raise ValueError(f"Invalid rate interval at {t:g} kyr")
    return {"t_kyr": t, "start_kyr": start, "end_kyr": end, "dt_yr": dt}


def panel_reference(reference, t, profile, summary, *, rates=False):
    """Reference values in metres, or baseline-interval average rates in m/yr."""
    if not rates:
        return reference_profiles(reference, t, profile, summary)
    interval = rate_interval(profile, t)
    first = reference_profiles(reference, interval["start_kyr"], profile, summary)
    last = reference_profiles(reference, interval["end_kyr"], profile, summary)
    return {key: (last[key] - first[key]) / interval["dt_yr"] for key in last}


def read_profile(directory, summary, t, longitude, maximum, hashes, *, bearing=None, rates=False):
    """Select an actual saved meridian, without averaging spatial differences."""
    if not any(np.isclose(epoch["t_kyr"], t) for epoch in summary["epochs"]):
        raise ValueError(f"No saved epoch at {t:g} kyr in {directory}")
    path = directory / f"profiles_{summary['case']}_{t:g}kyr.npz"
    with np.load(path, allow_pickle=False) as archive:
        profile = dict(archive)
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    kind = "meridian" if bearing is None else "load-centred"
    if str(profile["profile_kind"]) != kind:
        raise ValueError(f"Expected {kind} profiles in {path}")
    if not np.isclose(float(profile["t_kyr"]), t):
        raise ValueError(f"Wrong profile epoch in {path}")
    angles = profile["longitude_deg"] if bearing is None else profile["bearing_deg"]
    requested = longitude if bearing is None else bearing
    difference = (angles - requested + 180) % 360 - 180
    matches = np.flatnonzero(np.isclose(difference, 0, atol=1e-10, rtol=0))
    if matches.size != 1:
        raise ValueError(f"Expected one saved profile at {requested:g}° in {path}")
    if bearing is not None and not np.allclose(profile["centre_deg"], summary["load_centre_deg"]):
        raise ValueError(f"Off-axis profile is not centred on the load in {path}")
    index = int(matches[0])
    coordinate = profile["coordinate_deg"]
    if maximum > coordinate[-1] + 1e-9:
        raise ValueError(f"Requested extent exceeds saved profile in {path}")
    keep = (coordinate >= -1e-9) & (coordinate <= maximum + 1e-9)
    if not profile["found"][index, keep].all():
        raise ValueError(f"Missing sample points on the selected meridian in {path}")
    if rates:
        rate_interval(profile, t)
    suffix = "_m_per_yr" if rates else "_m"
    for key in (quantity + suffix for quantity in ("U", "along", "N")):
        if key not in profile or not np.isfinite(profile[key][index, keep]).all():
            raise ValueError(f"Nonfinite {key} on the selected meridian in {path}")
    return profile, index, keep


def render(args, variant, output):
    colours = [TIME_COLOURS[i % len(TIME_COLOURS)] for i in range(len(args.times_kyr))]
    quantities = (("U", "Radial displacement rate (mm/yr)" if args.rates else "Radial displacement (m)"),
                  ("along", "Tangential displacement rate (mm/yr)" if args.rates else "Tangential displacement (m)"),
                  ("N", "Geoid rate (mm/yr)" if args.rates else "Geoid height (m)"))
    suffix, scale = ("_m_per_yr", 1000.) if args.rates else ("_m", 1.)
    hashes = {str(args.reference): hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    for name in ("plot_spada_profiles.py", "spada_loads.py"):
        path = Path(__file__).with_name(name)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    availability = {}
    intervals = {}
    mesh_sizes = {}
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 13, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.2), sharex=True, sharey="row")
        with np.load(args.reference, allow_pickle=False) as reference:
            nmax = int(reference["degrees"][-1])
            for column, shape in enumerate(("cap", "disc")):
                path = summary_path(args.root, shape, args.baseline_run)
                directory = path.parent
                summary = read_summary(path, shape, hashes)
                if args.baseline_run != "baseline":
                    standard = read_summary(summary_path(args.root, shape), shape, hashes)
                    validate_comparison(standard, summary, args.baseline_run)
                mesh_sizes[shape] = summary["mesh_h_km"]
                changed_path = summary_path(args.root, shape, variant) if variant else None
                changed_summary = None
                if changed_path is not None and changed_path.exists():
                    changed_summary = read_summary(changed_path, shape, hashes, offaxis=variant == "offaxis")
                    validate_comparison(summary, changed_summary, variant)
                availability[shape] = changed_summary is not None if variant else True
                intervals[shape] = {"baseline": [], "variant": []}
                missing = variant is not None and changed_summary is None
                axes[0, column].set_title(f"{shape.capitalize()} load" + (" (no result)" if missing else ""), pad=12)
                for t, colour in zip(args.times_kyr, colours):
                    profile, index, keep = read_profile(
                        directory, summary, t, args.longitude, args.max_colatitude, hashes, rates=args.rates)
                    x = profile["coordinate_deg"][keep]
                    curves = panel_reference(reference, t, profile, summary, rates=args.rates)
                    if args.rates:
                        intervals[shape]["baseline"].append(rate_interval(profile, t))
                    if changed_summary is not None:
                        changed, changed_index, changed_keep = read_profile(
                            changed_path.parent, changed_summary, t, args.longitude,
                            args.max_colatitude, hashes, bearing=args.bearing if variant == "offaxis" else None,
                            rates=args.rates)
                        validate_sampling(profile, changed, index, changed_index, keep, changed_keep,
                                          offaxis=variant == "offaxis")
                        if args.rates:
                            intervals[shape]["variant"].append(rate_interval(changed, t))
                    for row, (quantity, _) in enumerate(quantities):
                        ax = axes[row, column]
                        ax.plot(x, scale * profile[quantity + suffix][index, keep],
                                color=colour, linewidth=1.9, zorder=2)
                        if changed_summary is not None:
                            ax.plot(x, scale * changed[quantity + suffix][changed_index, changed_keep],
                                    color=colour, linewidth=1.65, linestyle=(0, (4, 3)), zorder=2.5)
                        ax.plot(x, scale * curves[quantity][index, keep], color=colour,
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
        if args.rates:
            lower, upper = axes[2, 0].get_ylim()
            axes[2, 0].set_ylim(lower, upper + .10 * (upper - lower))

        times = [Line2D([], [], color=colour, linewidth=2,
                        label="Elastic (0 kyr)" if t == 0 else f"{t:g} kyr")
                 for t, colour in zip(args.times_kyr, colours)]
        label = "G-ADOPT baseline" if variant else "G-ADOPT"
        methods = [Line2D([], [], color="0.2", linewidth=1.9, label=label)]
        if variant:
            methods.append(Line2D([], [], color="0.2", linewidth=1.65, linestyle=(0, (4, 3)),
                                  label=VARIANTS[variant][0]))
        methods.append(Line2D([], [], color="0.2", linestyle="none", marker="x",
                              markersize=4, markeredgewidth=.9, label="TABOO"))
        axes[0, 0].legend(handles=[*times, *methods], loc="lower right",
                          fontsize=8, handlelength=2.8, borderaxespad=1,
                          frameon=True, facecolor="white", edgecolor="black",
                          framealpha=1, fancybox=False)
        fig.subplots_adjust(left=.12, right=.98, bottom=.07, top=.95, hspace=.14, wspace=.09)
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = output.with_suffix("." + extension)
            fig.savefig(path, dpi=220)
            print(f"Saved {path}")
        plt.close(fig)

    provenance = {"times_kyr": args.times_kyr, "longitude_deg": args.longitude % 360,
                  "variant": variant, "variant_available": availability,
                  "baseline_run": args.baseline_run, "baseline_mesh_h_km": mesh_sizes,
                  "quantity_kind": "rate" if args.rates else "value", "units": "mm/yr" if args.rates else "m",
                  "rate_intervals": intervals if args.rates else None,
                  "reference_rate_convention": "average over baseline saved rate interval" if args.rates else None,
                  "offaxis_bearing_deg": args.bearing if variant == "offaxis" else None,
                  "colour_palette": "Okabe–Ito (yellow omitted)", "time_colours_hex": colours,
                  "max_colatitude_deg": args.max_colatitude, "reference_degrees": [2, nmax],
                  "tangential_positive": "away from each load centre (positive profile-distance branch)",
                  "inputs_sha256": hashes,
                  "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__}
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return availability


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("outputs/spada-study"))
    parser.add_argument("--baseline-run", choices=("baseline", "mesh125"), default="baseline",
                        help="run used for solid cap/disc curves; mesh125 makes a standalone fine-mesh figure")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--variant", choices=VARIANTS, help="one dashed comparison against baseline")
    selection.add_argument("--all-variants", action="store_true", help="one figure per available nonrotating variant")
    parser.add_argument("--rates", action="store_true", help="adjacent-step rates in mm/yr; TABOO averaged over baseline intervals")
    parser.add_argument("--times-kyr", nargs="+", type=float,
                        help="epochs: default 0, 1, 5, 20; with --rates: 0.1, 1, 5, 20 kyr")
    parser.add_argument("--longitude", type=float, default=0., help="saved north-pole meridian in degrees")
    parser.add_argument("--bearing", type=float, default=0., help="saved off-axis profile bearing; positive branch only")
    parser.add_argument("--max-colatitude", type=float, default=25.)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.npz"))
    parser.add_argument("--output", type=Path, help="output stem for one figure; writes .pdf, .png and .json")
    parser.add_argument("--output-dir", type=Path,
                        help="batch directory; defaults to nonrotating_variants or nonrotating_rate_variants under outputs/spada-study-analysis")
    args = parser.parse_args(argv)
    if args.baseline_run != "baseline" and (args.variant or args.all_variants):
        parser.error("--baseline-run mesh125 makes a standalone figure; omit --variant and --all-variants")
    if args.times_kyr is None:
        args.times_kyr = [0.1 if args.rates else 0., 1., 5., 20.]
    if args.output_dir is None:
        args.output_dir = Path("outputs/spada-study-analysis") / (
            "nonrotating_rate_variants" if args.rates else "nonrotating_variants")
    if not np.isfinite(args.max_colatitude) or not 0 < args.max_colatitude <= 180:
        parser.error("--max-colatitude must be in (0, 180]")
    if not np.isfinite(args.longitude) or not np.isfinite(args.bearing):
        parser.error("--longitude and --bearing must be finite")
    if (not all(np.isfinite(t) and t >= 0 for t in args.times_kyr)
            or len(set(args.times_kyr)) != len(args.times_kyr)):
        parser.error("--times-kyr must contain distinct finite nonnegative times")
    if args.rates and any(t == 0 for t in args.times_kyr):
        parser.error("--rates requires positive epochs; saved rates at the elastic instant are undefined")
    if args.all_variants and args.output is not None:
        parser.error("Use --output-dir with --all-variants")
    variants = list(VARIANTS) if args.all_variants else [args.variant]
    suffix = "_rates" if args.rates else ""
    index = ["# Nonrotating Spada variant" + (" rate" if args.rates else "") + " panels", "",
             "Solid: G-ADOPT baseline. Dashed: one changed run. Crosses: TABOO (degrees 2–256).",
             "Times: " + ", ".join(f"{t:g} kyr" for t in args.times_kyr) + ".", "",
             "| Variant | Cap | Disc | PDF | PNG |", "|---|---|---|---|---|"]
    if args.rates:
        index[2:2] = ["Rates are in mm/yr. TABOO crosses are averaged over each baseline's saved interval.",
                      "Dashed rates retain the changed run's interval; timestep variants therefore include averaging differences.", ""]
    produced = 0
    for variant in variants:
        if variant and not any(summary_path(args.root, shape, variant).exists() for shape in ("cap", "disc")):
            if not args.all_variants:
                parser.error(f"No completed nonrotating summaries for {variant}")
            print(f"Skipping {variant}: no completed nonrotating summaries")
            index.append(f"| {variant} | unavailable | unavailable | — | — |")
            continue
        if args.all_variants:
            output = args.output_dir / f"spada_nonrotating_{variant}{suffix}"
        else:
            name = variant or "baselines" + (f"_{args.baseline_run}" if args.baseline_run != "baseline" else "")
            output = args.output or Path(f"outputs/spada-study-analysis/spada_nonrotating_{name}{suffix}")
        available = render(args, variant, output)
        produced += 1
        statuses = ["available" if available[shape] else "no completed run" for shape in ("cap", "disc")]
        index.append(f"| {variant} | {statuses[0]} | {statuses[1]} | "
                     f"[PDF]({output.name}.pdf) | [PNG]({output.name}.png) |")
    if args.all_variants:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "index.md").write_text("\n".join(index) + "\n")
        print(f"Saved {produced} separate figures; index: {args.output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
