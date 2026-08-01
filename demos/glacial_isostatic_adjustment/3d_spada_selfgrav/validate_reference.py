"""Acceptance gates for the extracted reference data and the synthesiser.

Each gate states the number it expects before it measures it, in the manner of
``demos/gravity/spikes/gate_*.py``.  Needs the archive, because the whole point
is to check the synthesiser against the archive's own spatial files; after this
has passed once, nothing downstream needs the archive again.

    python3 validate_reference.py [--archive WSCOTT-2023]

Gates:

  V1  load mass       cap coefficients integrate to TABOO's header value
  V2  six epochs      U, V, N against T02-01/disp_t*_cap.dat, < 1e-3 m
  V2b disc            the same for the disc load
  V2c point           reported only: the archive file is Lmax = 512, tables are 256
  V2d rates           the analytic time derivative against T02-02
  V3  fluid vs t1000  the two numbers that differ by 7 m and must not be conflated
  V4  TABOO vs VILMA  the spread between two independent solutions of the problem
"""

import argparse
import pathlib

import numpy as np

from taboo_synthesis import (A_EARTH, RHO_ICE, TabooReference, cap_load,
                             disc_load, load_mass, point_load)

EPOCHS = [("0", 0.0), ("1", 1.0), ("2", 2.0), ("5", 5.0), ("10", 10.0), ("1000", 1000.0)]
TABOO_HEADER_MASS = 3.607171340900778e18
NMAX = 128            # the archive's spatial files are Lmin, Lmax = 2, 128
TOLERANCE = 1e-3      # metres


def read_taboo_spatial(path):
    """colat, u_rad, u_col, u_lon, geoid on the file's own colatitude grid.

    T02-01 is 721 rows at 0.25 deg; T02 (the point load) is 181 rows at 1 deg.
    Every one of these files repeats its final theta = 180 deg row, and the
    duplicate is dropped here after checking that it is one.
    """
    data = np.loadtxt(path, comments="#")
    if np.array_equal(data[-1], data[-2]):
        data = data[:-1]
    assert data.shape[1] == 5, f"{path}: expected 5 columns"
    return data


def read_vilma(path):
    """Klemann's spectral-FE answer: colatitude, then one column per epoch
    (0, 1, 2, 5, 10 kyr).  Poles omitted."""
    return np.loadtxt(path, comments="#")


def gate_load_mass():
    print("V1  load mass")
    print(f"    expect cap and disc both {TABOO_HEADER_MASS:.15e} kg")
    ok = True
    for name, sigma in (("cap", cap_load(0)), ("disc", disc_load(0))):
        mass = load_mass(sigma)
        rel = abs(mass - TABOO_HEADER_MASS) / TABOO_HEADER_MASS
        print(f"    {name:5s} {mass:.15e}  rel {rel:.2e}")
        ok &= rel < 1e-7
    # Independent of the coefficient formula: the analytic integrals.
    a = np.deg2rad(10.0)
    analytic_cap = 4 / 3 * np.pi * A_EARTH**2 * RHO_ICE * 1500.0 * (1 - np.cos(a))
    analytic_disc = 2 * np.pi * A_EARTH**2 * RHO_ICE * 1000.0 * (1 - np.cos(a))
    print(f"    analytic cap  {analytic_cap:.15e}")
    print(f"    analytic disc {analytic_disc:.15e}")
    ok &= abs(analytic_cap - TABOO_HEADER_MASS) / TABOO_HEADER_MASS < 1e-7
    ok &= abs(analytic_disc - TABOO_HEADER_MASS) / TABOO_HEADER_MASS < 1e-7
    print(f"    {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate_six_epochs(ref, archive):
    print("V2  six cap epochs against the archive, 721-point 0.25 deg grid")
    print(f"    expect max |delta| < {TOLERANCE:g} m in all three fields;")
    print("    previously measured worst 2.29e-04 m in U at t = 1000 kyr")
    sigma = cap_load(NMAX)
    print(f"    {'epoch':>8} {'max|dU|':>11} {'max|dV|':>11} {'max|dN|':>11}")
    ok = True
    for tag, t in EPOCHS:
        obs = read_taboo_spatial(archive / "T02-01" / f"disp_t{tag}_cap.dat")
        U, V, N = ref.synthesise(t, np.deg2rad(obs[:, 0]), sigma, nmax=NMAX)
        d = [np.abs(U - obs[:, 1]).max(), np.abs(V - obs[:, 2]).max(),
             np.abs(N - obs[:, 4]).max()]
        ok &= max(d) < TOLERANCE
        print(f"    {tag:>8} {d[0]:11.3e} {d[1]:11.3e} {d[2]:11.3e}")
    print(f"    {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate_disc(ref, archive):
    """The cap gate exercises ``cap_load``; nothing else exercises ``disc_load``
    beyond its total mass, and the disc epochs are free to check."""
    print("V2b five disc epochs against the archive")
    print(f"    expect max |delta| < {TOLERANCE:g} m in all three fields")
    sigma = disc_load(NMAX)
    print(f"    {'epoch':>8} {'max|dU|':>11} {'max|dV|':>11} {'max|dN|':>11}")
    ok = True
    for tag, t in [e for e in EPOCHS if e[0] != "1000"] + [("1000", 1000.0)]:
        path = archive / "T02-01" / f"disp_t{tag}_disc.dat"
        if not path.exists():
            continue
        obs = read_taboo_spatial(path)
        U, V, N = ref.synthesise(t, np.deg2rad(obs[:, 0]), sigma, nmax=NMAX)
        d = [np.abs(U - obs[:, 1]).max(), np.abs(V - obs[:, 2]).max(),
             np.abs(N - obs[:, 4]).max()]
        ok &= max(d) < TOLERANCE
        print(f"    {tag:>8} {d[0]:11.3e} {d[1]:11.3e} {d[2]:11.3e}")
    print(f"    {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate_point(ref, archive):
    """The point load of ``T02/``, mass 1e18 kg -- note the T02-03 polar-motion
    point load is a *different* load at 10e18 kg.

    **Reported, not gated, and it cannot be gated from this .npz.** Those files
    carry ``Lmin, Lmax = 2, 512`` while the Love-number tables stop at n = 256,
    and a point load's series does not converge: sigma_n/(2n+1) is constant in
    n, so U(0 deg) grows without bound as the truncation rises. The residual
    below is dominated by the missing n = 257..512 tail and says nothing about
    the synthesiser. It shrinks with n_max, which is the evidence for that.
    """
    print("V2c point load (T02, Lmax = 512) -- REPORTED, NOT GATED")
    print("    the tables stop at n = 256, so the missing tail dominates;")
    print("    expect the residual to shrink as n_max rises, and no more")
    for nmax in (128, 256):
        sigma = point_load(nmax, mass=1e18)
        obs = read_taboo_spatial(archive / "T02" / "disp_t0_point.dat")
        U, V, N = ref.synthesise(0.0, np.deg2rad(obs[:, 0]), sigma, nmax=nmax)
        print(f"    n_max = {nmax:3d}  t = 0  max|dU| = {np.abs(U - obs[:, 1]).max():9.3f} m"
              f"   U(0 deg) = {U[0]:9.3f} vs {obs[0, 1]:9.3f}")
    print()
    return True


def gate_rates(ref, archive):
    """Rates come from differentiating (19) analytically. T02-02 tabulates them,
    so the derivative is checkable rather than merely plausible.

    Units: the T02-02 *column headings* say mm/yr, while the README-level
    description and ``output_sec2parab/README`` say m/yr. mm/yr is numerically
    identical to m/kyr, which is what the synthesiser produces, so no conversion
    is applied -- and the fact that this gate then passes at 1e-4 is what
    settles which of the two labels is right.
    """
    print("V2d cap rates against T02-02, analytic time derivative of eq. (19)")
    print(f"    expect max |delta| < {TOLERANCE:g} m/kyr (= mm/yr)")
    sigma = cap_load(NMAX)
    print(f"    {'epoch':>8} {'max|dU.|':>11} {'max|dV.|':>11} {'max|dN.|':>11}")
    ok = True
    for tag, t in EPOCHS:
        path = archive / "T02-02" / f"rate_t{tag}_cap.dat"
        if not path.exists():
            continue
        obs = read_taboo_spatial(path)
        U, V, N = ref.synthesise(t, np.deg2rad(obs[:, 0]), sigma, nmax=NMAX,
                                 rate=True)
        d = [np.abs(U - obs[:, 1]).max(), np.abs(V - obs[:, 2]).max(),
             np.abs(N - obs[:, 4]).max()]
        ok &= max(d) < TOLERANCE
        print(f"    {tag:>8} {d[0]:11.3e} {d[1]:11.3e} {d[2]:11.3e}")
    print(f"    {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate_fluid_versus_t1000(ref, archive):
    print("V3  the fluid limit is not t = 1000 kyr")
    print("    expect fluid U(0 deg) = -395.01 m, t1000 U(0 deg) = -388.11 m")
    sigma = cap_load(NMAX)
    pole = np.array([0.0])
    U_fluid = ref.synthesise(0.0, pole, sigma, nmax=NMAX, fluid=True)[0][0]
    U_t1000 = ref.synthesise(1000.0, pole, sigma, nmax=NMAX)[0][0]
    U_file = read_taboo_spatial(archive / "T02-01" / "disp_t1000_cap.dat")[0, 1]
    print(f"    fluid limit      {U_fluid:.5f} m")
    print(f"    synthesised t1000 {U_t1000:.5f} m")
    print(f"    archive t1000     {U_file:.5f} m")
    print(f"    difference        {U_fluid - U_file:.3f} m "
          f"({abs(U_fluid - U_file) / abs(U_file) * 100:.2f}%)")
    ok = (abs(U_fluid + 395.01) < 0.01 and abs(U_file + 388.11) < 0.01
          and abs(U_t1000 - U_file) < TOLERANCE)
    print(f"    {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate_taboo_versus_vilma(ref, archive):
    """The spread between two independent numerical solutions of the same
    problem.  This is the floor on any accuracy claim, and it is established
    here, before we have an answer of our own to be tempted by."""
    print("V4  TABOO versus VILMA (Klemann, spectral FE)")
    print("    expect the published solutions to disagree by 1-2 m near the")
    print("    load edge at theta = 10 deg, and much less in the far field")
    vilma_dir = archive / "T02-01" / "output_sec2parab"
    epochs = [0.0, 1.0, 2.0, 5.0, 10.0]
    fields = [("u", "U", 1), ("v", "V", 2), ("e", "N", 4)]
    theta_full = np.deg2rad(np.arange(721) * 0.25)
    sigma = cap_load(NMAX)
    taboo = {t: ref.synthesise(t, theta_full, sigma, nmax=NMAX) for t in epochs}

    for grid, label in (("cost", "fine, 0.25-15 deg"), ("costg", "global, 1-179 deg")):
        print(f"\n    {label}")
        print(f"    {'field':>5} {'epoch':>6} {'max|d|':>10} {'at theta':>9} "
              f"{'rms':>10} {'max|d|/max|f|':>14} {'near<=15d':>10} {'far>30d':>9}")
        for key, name, col in fields:
            data = read_vilma(vilma_dir / f"{key}2{grid}.dat")
            th = data[:, 0]
            # The VILMA colatitudes are an exact subset of the TABOO 0.25 deg
            # grid, so this is index selection, not interpolation.
            idx = np.rint(th / 0.25).astype(int)
            assert np.abs(idx * 0.25 - th).max() < 1e-9, "VILMA grid is not a subset"
            near, far = th <= 15.0, th > 30.0
            for j, t in enumerate(epochs):
                ours = taboo[t][["U", "V", "N"].index(name)][idx]
                theirs = data[:, 1 + j]
                d = ours - theirs
                i = np.argmax(np.abs(d))
                nm = np.abs(d[near]).max() if near.any() else np.nan
                fm = np.abs(d[far]).max() if far.any() else np.nan
                print(f"    {name:>5} {t:6.0f} {np.abs(d).max():10.3f} "
                      f"{th[i]:8.2f}d {np.sqrt((d**2).mean()):10.3f} "
                      f"{np.abs(d).max() / np.abs(ours).max():14.4f} "
                      f"{nm:10.3f} {fm:9.3f}")
    print("\n    (reported, not gated: this is a measurement of the reference "
          "spread)\n")
    return True


def main():
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=pathlib.Path,
                        default=here.parents[2] / "WSCOTT-2023")
    parser.add_argument("--npz", type=pathlib.Path, default=here / "reference.npz")
    args = parser.parse_args()

    ref = TabooReference(args.npz)
    results = [
        gate_load_mass(),
        gate_six_epochs(ref, args.archive),
        gate_disc(ref, args.archive),
        gate_point(ref, args.archive),
        gate_rates(ref, args.archive),
        gate_fluid_versus_t1000(ref, args.archive),
        gate_taboo_versus_vilma(ref, args.archive),
    ]
    print("ALL GATES PASS" if all(results) else "FAILURES ABOVE")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
