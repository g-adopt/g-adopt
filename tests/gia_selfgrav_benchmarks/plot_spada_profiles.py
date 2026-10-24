"""Plot Spada profiles, with TABOO overlays for non-rotating meridians.

Usage: python plot_spada_profiles.py [summary_cap.json] [--max-colatitude 25]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legder, legval
from spada_loads import load_coefficients


def taboo_coefficients(reference, t_kyr, load_shape="cap"):
    """Reconstruct all available degrees for the selected Table 4 load."""
    degrees = reference["degrees"]
    selected = degrees >= 2
    n = degrees[selected]
    nmax = int(n.max())
    if not np.array_equal(n, np.arange(2, nmax + 1)):
        raise ValueError("TABOO reference degrees must be consecutive from degree 2")
    sigma = load_coefficients(nmax, load_shape)[n]
    rates = reference["spectrum_s"][selected]
    relaxation = -np.expm1(rates * t_kyr) / rates
    result = {}
    for quantity, symbol in [("U", "h"), ("V", "l"), ("N", "k")]:
        love = (reference[f"{symbol}_elastic"][selected]
                + (1.0 if symbol == "k" else 0.0)
                - (reference[f"{symbol}_residues"][selected] * relaxation).sum(axis=1))
        coefficients = np.zeros(nmax + 1)
        coefficients[n] = 3.0 / 5511.68 * sigma / (2 * n + 1) * love
        result[quantity] = coefficients
    return result


def plot_direct_profiles(summary, directory, maximum, output):
    """Plot sampled vector fields without imposing an axisymmetric reference."""
    epochs = summary["epochs"]
    fig, axes = plt.subplots(len(epochs), 4, figsize=(15, 2.5*len(epochs)),
                             squeeze=False, sharex=True, layout="constrained")
    first_geometry = None
    for row, epoch in zip(axes, epochs):
        path = directory / f"profiles_{summary['case']}_{epoch['t_kyr']:g}kyr.npz"
        with np.load(path, allow_pickle=False) as data:
            if not np.isclose(float(data["t_kyr"]), epoch["t_kyr"]):
                raise ValueError(f"Profile epoch mismatch in {path}")
            kind = str(data["profile_kind"])
            coordinate = data["coordinate_deg"]
            labels = data["profile_labels"]
            geometry = (kind, coordinate.copy(), labels.copy(), data["unit_vectors"].copy())
            if first_geometry is None:
                first_geometry = geometry
            elif (geometry[0] != first_geometry[0]
                  or any(not np.array_equal(a, b) for a, b in zip(geometry[1:], first_geometry[1:]))):
                raise ValueError(f"Profile sampling changed between epochs in {path}")
            fields = [("U_m", "Radial displacement U"),
                      ("along_m", "Along-profile displacement"),
                      ("across_m", "Rightward cross-profile displacement"),
                      ("N_m", "Geoid height N")]
            selected = np.abs(coordinate) <= maximum
            if not np.all(data["found"]):
                print(f"Warning: missing sample points at {epoch['t_kyr']:g} kyr")
            for ax, (key, title) in zip(row, fields):
                for label, values, found in zip(labels, data[key], data["found"]):
                    ax.plot(coordinate[selected], np.where(found, values, np.nan)[selected],
                            label=str(label))
                ax.set_title(f"{title} — {epoch['t_kyr']:g} kyr")
                ax.set_ylabel("m")
                ax.grid(alpha=.25)
                if kind == "load-centred":
                    ax.axvline(0, color="grey", linestyle=":", linewidth=.7)
    xlabel = ("Signed distance from profile centre (degrees)" if first_geometry[0] == "load-centred"
              else "Colatitude (degrees)")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize="small")
    fig.suptitle(f"Spada {summary.get('load_shape', 'cap')}, {summary['case']}: direct profiles"
                 + (" (smoke run)" if summary["smoke"] else ""))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path,
                        default=Path(__file__).with_name("summary_cap.json"))
    parser.add_argument("--max-colatitude", "--max-distance", type=float, default=180.0,
                        help="maximum colatitude, or absolute signed distance for centred profiles")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("reference.npz"))
    parser.add_argument("--profiles-dir", type=Path, default=None,
                        help="overlay direct profiles written by --write_profiles")
    args = parser.parse_args()
    if not 0 < args.max_colatitude <= 180:
        parser.error("--max-colatitude must be in (0, 180]")
    summary = json.loads(args.summary.read_text())
    load_shape = summary.get("load_shape", "cap")
    directory = args.profiles_dir or args.summary.parent
    direct_only = summary["case"] == "polar-motion"
    sampling = summary.get("profile_sampling")
    direct_only |= sampling is not None and sampling["kind"] == "load-centred"
    if args.profiles_dir is not None and summary["epochs"]:
        first_path = directory / f"profiles_{summary['case']}_{summary['epochs'][0]['t_kyr']:g}kyr.npz"
        with np.load(first_path, allow_pickle=False) as data:
            direct_only |= str(data.get("profile_kind", "meridian")) != "meridian"
    if direct_only:
        output = args.output or args.summary.with_name(f"spada_{load_shape}_{summary['case']}_profiles.png")
        plot_direct_profiles(summary, directory, args.max_colatitude, output)
        return

    theta_deg = np.linspace(0, args.max_colatitude, 1001)
    theta = np.deg2rad(theta_deg)
    epochs = summary["epochs"]
    reference = np.load(args.reference, allow_pickle=False)
    fig, axes = plt.subplots(len(epochs), 3, figsize=(12, 2.4 * len(epochs)),
                             squeeze=False, sharex=True, layout="constrained")
    names = {"U": "Radial displacement", "V": "Southward displacement",
             "N": "Geoid height"}
    for row, epoch in zip(axes, epochs):
        degrees = np.asarray(epoch["degrees"], dtype=int)
        full_reference = taboo_coefficients(reference, epoch["t_kyr"], load_shape)
        direct = None
        if args.profiles_dir is not None:
            direct = np.load(args.profiles_dir /
                             f"profiles_cap_{epoch['t_kyr']:g}kyr.npz")
            if not np.all(direct["found"]):
                print(f"Warning: missing sample points at {epoch['t_kyr']:g} kyr")
        for ax, quantity in zip(row, names):
            curves = []
            diagnostic_range = f"{degrees.min()}–{degrees.max()}"
            for suffix, label, style in [
                    ("", f"G-ADOPT diagnostics {diagnostic_range}", "-"),
                    ("_ref", f"TABOO diagnostics {diagnostic_range}", "--")]:
                coefficients = np.zeros(int(degrees.max()) + 1)
                coefficients[degrees] = epoch[f"{quantity}_n{suffix}"]
                curves.append((coefficients, label, style))
            nmax = len(full_reference[quantity]) - 1
            load_representation = summary.get("load_representation", "spectral")
            load_nmax = summary.get("load_nmax", summary.get("nmax", int(degrees.max())))
            if load_representation == "spectral" and load_nmax > degrees.max():
                matched_reference = full_reference[quantity].copy()
                matched_reference[load_nmax + 1:] = 0.0
                label = (f"TABOO load 2–{load_nmax}" if load_nmax <= nmax else
                         f"TABOO available 2–{nmax} (load 2–{load_nmax})")
                curves.append((matched_reference, label, "-."))
            reference_label = (f"TABOO spatial-{load_shape} reference 2–{nmax}"
                               if load_representation == "spatial" else
                               f"TABOO full-{load_shape} reference 2–{nmax}")
            curves.append((full_reference[quantity], reference_label, ":"))
            for coefficients, label, style in curves:
                if quantity == "V":
                    profile = -np.sin(theta) * legval(
                        np.cos(theta), legder(coefficients))
                    profile[0] = 0.0
                    if args.max_colatitude == 180:
                        profile[-1] = 0.0
                else:
                    profile = legval(np.cos(theta), coefficients)
                ax.plot(theta_deg, profile, style, label=label)
            if direct is not None:
                selected = direct["colatitude_deg"] <= args.max_colatitude
                for longitude, values, found in zip(direct["longitude_deg"],
                                                    direct[f"{quantity}_m"],
                                                    direct["found"]):
                    ax.plot(direct["colatitude_deg"][selected],
                            np.where(found, values, np.nan)[selected],
                            linewidth=0.9, alpha=0.75,
                            label=f"Direct {longitude:g}°")
            ax.set_title(f"{quantity}: {names[quantity]} — {epoch['t_kyr']:g} kyr")
            ax.set_ylabel("m")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize="small")
    for ax in axes[-1]:
        ax.set_xlabel("Colatitude (degrees)")
    fig.suptitle(f"Spada {load_shape}: projected degrees {degrees.min()}–{degrees.max()}"
                 + (" (smoke run)" if summary["smoke"] else ""))
    output = args.output or args.summary.with_name(f"spada_{load_shape}_profiles.png")
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
