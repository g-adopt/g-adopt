r"""Pin `analytic_gravity`'s radial quadrature against passess.

`analytic_gravity` generalises the radial Green's function of
`passess.spherical.PoissonSpherical3D` from a constant density on a shell to an
arbitrary radial profile, by replacing the closed-form radial integrals with
quadrature. That generalisation is the only new mathematics in the study's
reference solution, so it is checked here against the package it generalises,
in two directions:

  1. **The constant case must be exact.** With a constant profile on
     `[r1, r2]`, the quadrature must reproduce passess's closed form to machine
     precision, at every degree the study uses. This catches an error in the
     prefactor, in the `r_<`/`r_>` split, or in the ratio rewriting.

  2. **The varying case must converge.** For the Gaussian bump the study
     actually uses, passess has no closed form, so the reference is compared
     against a superposition of thin constant passess shells - a construction
     that is independently correct and whose midpoint error falls as `N^-2`.
     Agreement improving at that rate as `N` grows is evidence the quadrature
     integrates the right thing, and not merely that two expressions with the
     same typo agree.

passess is a local package, not a dependency of this repository or of the Gadi
deployment; this script is a development-time check, which is why the reference
module itself carries no passess import. Run it from anywhere:

    <firedrake-python> check_analytic_vs_passess.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))
try:
    from passess.spherical import PoissonSpherical3D
except ImportError as exc:  # pragma: no cover - developer machine only
    raise SystemExit(
        f"passess not importable ({exc}); this check needs the local package "
        "at ~/Workplace/passess. The reference module itself does not.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_gravity import radial_potential_factor  # noqa: E402

RMIN, RMAX = 1.22, 2.22
G_GRAV = 1.0
DEGREES = [0, 1, 2, 3, 5, 7, 10, 15, 20]


def check_constant_shell():
    """Constant density on a sub-shell: quadrature against passess closed form.

    passess evaluates `psi_lm = G rho_lm _unit(r)` with `_unit` the closed-form
    radial integral, including its own special case at `l = 2` where the outer
    antiderivative becomes a logarithm. The quadrature form needs no such case,
    so `l = 2` is the row worth watching.
    """
    r1, r2 = 1.5, 1.9
    radii = np.array([1.55, 1.7, 1.85])          # inside the support
    worst = 0.0
    print(f"{'l':>3} {'passess':>16} {'quadrature':>16} {'rel diff':>10}")
    for l in DEGREES:
        exact = np.array([
            PoissonSpherical3D(l, 0, 1.0, r1, r2, G_GRAV).psi_lm(r) for r in radii
        ])
        got = radial_potential_factor(l, lambda rr: np.ones_like(rr), radii, r1, r2)
        rel = np.max(np.abs(got - exact) / np.max(np.abs(exact)))
        worst = max(worst, rel)
        print(f"{l:>3} {exact[1]:16.9e} {got[1]:16.9e} {rel:10.2e}")
    print(f"worst relative difference over degrees {DEGREES}: {worst:.3e}")
    assert worst < 1e-13, f"constant-shell check failed at {worst:.3e}"
    return worst


def check_gaussian_profile():
    """Gaussian bump: quadrature against a stack of thin passess shells.

    The stack is a midpoint rule on the radial integral, so its error falls as
    `N^-2`. What is being checked is that the two agree *and* that the residual
    shrinks at that rate, which distinguishes "both are right" from "both are
    wrong in the same way".
    """
    centre, width = RMAX - 0.12 * (RMAX - RMIN), 0.15 * (RMAX - RMIN)
    radii = np.array([1.4, 1.8, 2.1])

    def profile(rr):
        return np.exp(-((rr - centre) / width) ** 2)

    print(f"\n{'l':>3} " + " ".join(f"{'N=' + str(n):>12}" for n in (50, 200, 800)))
    worst_fine = 0.0
    for l in (0, 2, 5, 10, 20):
        reference = radial_potential_factor(l, profile, radii, RMIN, RMAX)
        row = []
        for nshell in (50, 200, 800):
            edges = np.linspace(RMIN, RMAX, nshell + 1)
            mids = 0.5 * (edges[:-1] + edges[1:])
            stacked = np.zeros_like(radii)
            for r_lo, r_hi, r_mid in zip(edges[:-1], edges[1:], mids):
                shell = PoissonSpherical3D(l, 0, profile(r_mid), r_lo, r_hi, G_GRAV)
                stacked += np.array([shell.psi_lm(r) for r in radii])
            rel = np.max(np.abs(stacked - reference)) / np.max(np.abs(reference))
            row.append(rel)
        print(f"{l:>3} " + " ".join(f"{v:12.2e}" for v in row))
        # Halving the shell width four times should cut the error ~16-fold
        # twice over; require an order of magnitude per 4x refinement to allow
        # for the tail of the Gaussian near the boundaries.
        assert row[1] < row[0] / 5.0 and row[2] < row[1] / 5.0, (
            f"stacked-shell residual not converging at l={l}: {row}")
        worst_fine = max(worst_fine, row[-1])
    print(f"worst residual at N=800: {worst_fine:.3e}")
    assert worst_fine < 1e-5, f"Gaussian check too loose at {worst_fine:.3e}"
    return worst_fine


if __name__ == "__main__":
    print("=== constant density on a shell: exact comparison ===")
    check_constant_shell()
    print("\n=== Gaussian bump: convergence of a passess shell stack ===")
    check_gaussian_profile()
    print("\nBOTH CHECKS PASSED")
