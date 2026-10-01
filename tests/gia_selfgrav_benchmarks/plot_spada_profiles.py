"""Plot the degree-2–10 surface profiles stored in a Spada cap summary.

Usage: python plot_spada_profiles.py [summary_cap.json] [--max-colatitude 25]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legder, legval


def taboo_coefficients(reference, t_kyr):
    """Reconstruct all available degrees using the driver's cap and TABOO formulas."""
    degrees = reference["degrees"]
    selected = degrees >= 2
    n = degrees[selected]
    nmax = int(n.max())
    if not np.array_equal(n, np.arange(2, nmax + 1)):
        raise ValueError("TABOO reference degrees must be consecutive from degree 2")
    alpha = np.deg2rad(10.0)
    sigma = -931.0 * 1500.0 / (4 * (1 - np.cos(alpha))) * (
        (np.cos((n + 1) * alpha) - np.cos((n + 2) * alpha)) / (n + 1.5)
        - (np.cos((n - 1) * alpha) - np.cos(n * alpha)) / (n - 0.5))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path,
                        default=Path(__file__).with_name("summary_cap.json"))
    parser.add_argument("--max-colatitude", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("reference.npz"))
    parser.add_argument("--profiles-dir", type=Path, default=None,
                        help="overlay direct profiles written by --write_profiles")
    args = parser.parse_args()
    if not 0 < args.max_colatitude <= 180:
        parser.error("--max-colatitude must be in (0, 180]")
    summary = json.loads(args.summary.read_text())
    if summary["case"] != "cap":
        parser.error("the summary must be for the cap case")

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
        full_reference = taboo_coefficients(reference, epoch["t_kyr"])
        direct = None
        if args.profiles_dir is not None:
            direct = np.load(args.profiles_dir /
                             f"profiles_cap_{epoch['t_kyr']:g}kyr.npz")
            if not np.all(direct["found"]):
                print(f"Warning: missing sample points at {epoch['t_kyr']:g} kyr")
        for ax, quantity in zip(row, names):
            curves = []
            for suffix, label, style in [("", "G-ADOPT 2–10", "-"),
                                          ("_ref", "TABOO 2–10", "--")]:
                coefficients = np.zeros(int(degrees.max()) + 1)
                coefficients[degrees] = epoch[f"{quantity}_n{suffix}"]
                curves.append((coefficients, label, style))
            nmax = len(full_reference[quantity]) - 1
            curves.append((full_reference[quantity], f"TABOO 2–{nmax}", ":"))
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
    fig.suptitle(f"Spada cap: projected degrees {degrees.min()}–{degrees.max()}"
                 + (" (smoke run)" if summary["smoke"] else ""))
    output = args.output or args.summary.with_name("spada_cap_profiles.png")
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
