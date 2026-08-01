"""Extract the Spada et al. (2011) benchmark reference data into a single .npz.

The source is the COST ES0701 submission tree (``WSCOTT-2023/`` at the worktree
root): 85 MB, untracked, and mostly GMT scratch and PostScript.  What this pulls
out of it is the *spectral* representation -- the h, l and k loading Love numbers
as (elastic, fluid, residues) and the relaxation spectrum s_j(n) -- plus the
T02-03 polar-motion series.  Everything spatial is then synthesised at run time
by ``taboo_synthesis.py``, which is what makes the truncation degree, the
colatitude grid, the load shape and the epoch all free parameters.

Two traps in the archive, both silent, both handled here:

* The ``.ela``/``.flu``/``.res`` triples carry *absolute values* of the residues
  (they exist to make Fig. 4's log plot).  They reconstruct h and silently mangle
  l and k.  This reads the ``.dat`` files, which are signed.
* The residue lists have **twelve** entries while the spectrum has **nine**.
  Positions 6, 9 and 12 (1-indexed) are Maxwell modes of vanishing strength and
  are dropped here by *fixed index*.  Dropping them by magnitude is correct at
  n = 2 and silently drops four modes at n = 128, where a genuine mode is also
  tiny.

Usage::

    python3 extract_reference.py [--archive WSCOTT-2023] [--output reference.npz]
"""

import argparse
import json
import pathlib

import numpy as np

# Positions 6, 9, 12 (1-indexed) of the twelve-entry residue list are the
# Maxwell modes.  Fixed index, never magnitude -- see the module docstring.
MAXWELL_COLUMNS = (5, 8, 11)

PROVENANCE = {
    "source": "COST ES0701 GIA benchmark submission tree (Spada et al. 2011)",
    "code": "TABOO, Task#1",
    "created": "2009.10.17",
    "model": "M03-L70-V01",
    "love_numbers": "T01-02/{h,l,k}_L_T01-02_GS.dat, signed, n = 2..256",
    "spectrum": "T01-01/spec_T01_01_GS.dat, s_j in kyr^-1, Maxwell modes removed",
    "polar_motion": "T02-03/, Barletta, viscoelastic normal modes",
    "contributors": "Giorgio Spada, Florence Colleoni, Valentina R. Barletta",
    "maxwell_columns_dropped": list(MAXWELL_COLUMNS),
    "note": (
        "Residues are stored both raw (twelve columns, as in the archive) and "
        "cleaned (nine columns, aligned with the spectrum). Use the cleaned ones."
    ),
}


def read_love(archive, symbol):
    """Read one signed Love-number table: n, elastic, fluid, twelve residues."""
    path = archive / "T01-02" / f"{symbol}_L_T01-02_GS.dat"
    data = np.loadtxt(path, comments="#")
    assert data.shape[1] == 15, f"{path}: expected 15 columns, got {data.shape[1]}"
    return data[:, 0].astype(int), data[:, 1], data[:, 2], data[:, 3:]


def read_spectrum(archive):
    """Read the relaxation spectrum. Units are kyr^-1 -- the ``.mod`` file is
    tau_j in *years*, and mixing the two is a factor of 1000."""
    data = np.loadtxt(archive / "T01-01" / "spec_T01_01_GS.dat", comments="#")
    assert data.shape[1] == 10, f"expected degree + 9 modes, got {data.shape[1]}"
    return data[:, 0].astype(int), data[:, 1:]


def read_polar_motion_vector(archive, name):
    """Read one of the dense mx(t), my(t), |m|(t) grids."""
    data = np.loadtxt(archive / "T02-03" / name, comments="#")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def read_polar_motion_modulus(archive, load, chandler):
    """Read one Barletta |m|(t), |mdot|(t) table and its geometrical factor."""
    path = archive / "T02-03" / f"M3-L70-V01_t02-03_{load}_{chandler}.txt"
    text = path.read_text().splitlines()
    factor = []
    for line in text:
        if line.strip().startswith("# Re:") or line.strip().startswith("# Im:"):
            factor.append(float(line.split(":")[1]))
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1], data[:, 2], np.array(factor)


def main():
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=pathlib.Path,
                        default=here.parents[2] / "WSCOTT-2023")
    parser.add_argument("--output", type=pathlib.Path, default=here / "reference.npz")
    args = parser.parse_args()

    archive = args.archive
    if not archive.is_dir():
        raise SystemExit(f"archive not found: {archive}")

    out = {}

    degrees_spec, spectrum = read_spectrum(archive)
    keep = [j for j in range(12) if j not in MAXWELL_COLUMNS]

    degrees = None
    for symbol in ("h", "l", "k"):
        n, elastic, fluid, residues_raw = read_love(archive, symbol)
        if degrees is None:
            degrees = n
        assert (n == degrees).all(), "Love-number tables disagree on degree"
        residues = residues_raw[:, keep]

        # The dropped columns must be the vanishing ones. If this ever trips,
        # the archive's mode ordering has changed and the fixed indices are wrong.
        dropped = np.abs(residues_raw[:, list(MAXWELL_COLUMNS)]).max()
        assert dropped < 1e-18, (
            f"{symbol}: columns {MAXWELL_COLUMNS} are not Maxwell modes "
            f"(max |residue| = {dropped:.3e})"
        )

        out[f"{symbol}_elastic"] = elastic
        out[f"{symbol}_fluid"] = fluid
        out[f"{symbol}_residues"] = residues
        out[f"{symbol}_residues_raw"] = residues_raw

    # The spectrum covers the same degrees as the Love numbers, and the fluid
    # limit must reconstruct as elastic - sum_j residue_j / s_j. This is the one
    # check that ties the residue columns to the spectrum columns.
    assert (degrees_spec == degrees).all(), "spectrum and Love numbers disagree on degree"
    for symbol in ("h", "l", "k"):
        recon = out[f"{symbol}_elastic"] - (out[f"{symbol}_residues"] / spectrum).sum(axis=1)
        err = np.abs(recon - out[f"{symbol}_fluid"]).max()
        assert err < 1e-4, f"{symbol}: fluid limit does not reconstruct ({err:.3e})"
        print(f"  {symbol}: fluid limit reconstructs to {err:.3e}")

    out["degrees"] = degrees
    out["spectrum_s"] = spectrum

    t, mx, my, absm = read_polar_motion_vector(archive, "Rot_cwoff_GRID_cap_m.txt")
    out["pm_cap_t_kyr"] = t
    out["pm_cap_mx_deg"] = mx
    out["pm_cap_my_deg"] = my
    out["pm_cap_absm_deg"] = absm

    t, mdx, mdy, absmd = read_polar_motion_vector(archive, "Rot_cwoff_GRID_cap_mdot.txt")
    out["pm_cap_mdot_t_kyr"] = t
    out["pm_cap_mdotx_deg_per_myr"] = mdx
    out["pm_cap_mdoty_deg_per_myr"] = mdy
    out["pm_cap_absmdot_deg_per_myr"] = absmd

    for load in ("cap", "disc", "point"):
        for chandler in ("cwoff", "cwon"):
            t, m, md, factor = read_polar_motion_modulus(archive, load, chandler)
            tag = f"pm_{load}_{chandler}"
            out[f"{tag}_t_kyr"] = t
            out[f"{tag}_absm_deg"] = m
            out[f"{tag}_absmdot_deg_per_myr"] = md
            out[f"{tag}_geometrical_factor"] = factor

    out["provenance"] = np.array(json.dumps(PROVENANCE, indent=2))

    np.savez_compressed(args.output, **out)
    size = args.output.stat().st_size
    print(f"\nwrote {args.output} ({size / 1024:.1f} kB, {len(out)} arrays)")
    print(f"degrees n = {degrees[0]}..{degrees[-1]}, {spectrum.shape[1]} relaxation modes")


if __name__ == "__main__":
    main()
