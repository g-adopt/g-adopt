"""Discrete fluid limit of the actual B5 discretisation, in one direct solve.

The mantle shear modulus is scaled by `--eps` while the bulk modulus is held at
`ratio * mu_original` (via a DG0 `bulk_shear_ratio` field), so eps -> 0 gives
the relaxed Maxwell state: zero deviatoric stress in the mantle, elastic
lithosphere, full self-gravity, DtN, fluid core. Block 0 is solved with MUMPS
so the conditioning of K/mu' = ratio/eps does not matter.

The per-degree U_n, V_n, N_n are compared with the TABOO *fluid* Love numbers
(h_f, l_f, 1+k_f), and with the elastic ones when eps = 1.

    python probe_fluid_limit.py --mesh b2_coarse_ar7.msh --eps 1e-6
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gadopt  # noqa: F401,E402  (before firedrake, as the drivers do)
import numpy as np  # noqa: E402
from firedrake import (COMM_WORLD, Constant, Function, FunctionSpace,  # noqa: E402
                       SpatialCoordinate, conditional, dot, sqrt)

import b1_elastic as b1  # noqa: E402
import generate_selfgrav_sphere as gen  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", default=os.path.join(HERE, "b2_coarse_ar7.msh"))
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--s", type=float, default=None,
                   help="Laplace variable in kyr^-1: scale each mantle layer's mu by "
                        "s/(s + 1/tau_j), tau_j = eta_j/mu_j (correspondence principle). "
                        "Overrides --eps.")
    p.add_argument("--ratio", type=float, default=100.0)
    p.add_argument("--nmax", type=int, default=10)
    p.add_argument("--nproj", type=int, default=20)
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--quad-degree", type=int, default=60)
    p.add_argument("--outer-rtol", type=float, default=1e-10)
    p.add_argument("--no-condense", action="store_true")
    p.add_argument("--no-curve", action="store_true", help="straight-sided (affine) cells")
    p.add_argument("--rigid-core", action="store_true", help="un = 0 at the CMB instead of the fluid core")
    p.add_argument("--udeg", type=int, default=2, help="displacement polynomial degree")
    p.add_argument("--lith-fluid", action="store_true",
                   help="scale the lithosphere mu by the same factor as the upper mantle (no plate)")
    args = p.parse_args()

    say = lambda *a: print(*a, flush=True) if COMM_WORLD.rank == 0 else None  # noqa: E731

    # scale the mantle shear modulus (rows 1..3 of LAYERS; row 0 is the lithosphere)
    L = [list(r) for r in b1.LAYERS]
    KYR = 3.15576e10
    MU_BAR, ETA_BAR = 1.0e11, 1.0e21   # LAYERS columns 3 and 4 are mu/1e11 Pa and eta/1e21 Pa s
    facs = []
    for r in L[1:]:
        if args.s is None:
            f = args.eps
        else:
            tau_kyr = (r[4] * ETA_BAR) / (r[3] * MU_BAR) / KYR
            f = args.s / (args.s + 1.0 / tau_kyr)
        facs.append(f)
        r[3] = r[3] * f
    if args.lith_fluid:
        L[0][3] = L[0][3] * facs[0]
    b1.LAYERS = [tuple(r) for r in L]
    say(f"  mantle mu factors (UM, TZ, LM): {facs}")
    say(f"probe_fluid_limit: eps={args.eps} ratio={args.ratio} mesh={args.mesh}")
    say("  LAYERS (r_out, r_in, rho, mu, eta):")
    for r in b1.LAYERS:
        say(f"    {r}")

    t0 = time.perf_counter()
    parent, sub, _, _ = b1.build_meshes("coarse", path=args.mesh, curve_enabled=not args.no_curve)
    say(f"  meshes built in {time.perf_counter() - t0:.1f}s")

    # K = ratio * mu_original everywhere: ratio/eps in the mantle, ratio in the lithosphere
    X = SpatialCoordinate(sub)
    r = sqrt(dot(X, X))
    expr = Constant(args.ratio / facs[2])                       # lower mantle
    expr = conditional(r > Constant(b1.LAYERS[2][1]), Constant(args.ratio / facs[1]), expr)
    expr = conditional(r > Constant(b1.LAYERS[1][1]), Constant(args.ratio / facs[0]), expr)
    expr = conditional(r > Constant(b1.LAYERS[0][1]),
                       Constant(args.ratio / (facs[0] if args.lith_fluid else 1.0)), expr)
    ratio_field = Function(FunctionSpace(sub, "DG", 0), name="bulk_shear_ratio").interpolate(expr)
    vals = ratio_field.dat.data_ro
    say(f"  bulk_shear_ratio field: min {vals.min():.4e} max {vals.max():.4e}")

    if args.udeg != 2:
        import functools
        _orig_space = b1.self_gravitating_gia_space
        b1.self_gravitating_gia_space = functools.partial(_orig_space, displacement_degree=args.udeg)
        say(f"  displacement degree {args.udeg}")
    t0 = time.perf_counter()
    solver, z, layout, sigma_n, sigma_parent, sigma_sub = b1.build_solver(
        parent, sub, args.nmax, dtn_degree=args.dtn_degree,
        block0="lu", outer_rtol=args.outer_rtol,
        near_nullspace=False, condense=not args.no_condense,
        bulk_shear_ratio=ratio_field, snes_type="ksponly", rigid_core=args.rigid_core)
    say(f"  solver built in {time.perf_counter() - t0:.1f}s; "
        f"{z.function_space().dim()} dofs")

    t0 = time.perf_counter()
    solver.solve()
    say(f"  solved in {time.perf_counter() - t0:.1f}s")

    U_n, V_n, N_n = b1.surface_spectra(solver, parent, sub, args.nproj,
                                       quad_degree=args.quad_degree)

    ref = taboo.TabooReference()
    d = ref.data
    deg = list(d["degrees"])
    sig_dim = taboo.cap_load(args.nmax)
    nn = np.arange(2, args.nmax + 1)
    cc = 3.0 / taboo.RHO_BAR * sig_dim[2:args.nmax + 1] / (2 * nn + 1)
    say(f"\n  per-degree, model vs TABOO fluid (h_f, l_f, 1+k_f) and elastic (h_e, l_e, 1+k_e)")
    say(f"  {'n':>3} {'U_n':>11} {'U/Uf':>8} {'U/Ue':>8} | {'V_n':>11} {'V/Vf':>8} {'V/Ve':>8} | "
        f"{'N_n':>11} {'N/Nf':>8} {'N/Ne':>8}")
    for j, n in enumerate(nn):
        i = deg.index(n)
        hf, lf, kf = d["h_fluid"][i], d["l_fluid"][i], 1 + d["k_fluid"][i]
        he, le, ke = d["h_elastic"][i], d["l_elastic"][i], 1 + d["k_elastic"][i]
        say(f"  {n:>3} {U_n[n]:>11.6f} {U_n[n]/(cc[j]*hf):>8.4f} {U_n[n]/(cc[j]*he):>8.4f} | "
            f"{V_n[n]:>11.6f} {V_n[n]/(cc[j]*lf):>8.4f} {V_n[n]/(cc[j]*le):>8.4f} | "
            f"{N_n[n]:>11.6f} {N_n[n]/(cc[j]*kf):>8.4f} {N_n[n]/(cc[j]*ke):>8.4f}")
    say(f"  degree 0/1: U0 {U_n[0]:.3e} U1 {U_n[1]:.3e} N0 {N_n[0]:.3e} N1 {N_n[1]:.3e}")
    say("RESULT probe_fluid_limit done")


if __name__ == "__main__":
    main()
