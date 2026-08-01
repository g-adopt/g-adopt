r"""A3: the reference state, and the 3-D DtN truncation measurement.

Handoff §2.3.  Every gate states the number it expects before it measures it.

    G-1   |g_0| at the five interfaces against the paper's gravity column,
          10.457 / 10.024 / 9.978 / 9.854 / 9.815 m s^-2.  Target 1e-3
          relative at `--coarse`.
    G-2   g_0 is radial: max |g_0 - (g_0.rhat)rhat| / max|g_0|, globally and
          per shell.
    E-L   the boundary-treatment residual of an *untreated* degree l, against
          road map §1.2's eps_l = (Re/2Re)^(2l+1) (l+1-alpha)/(l+alpha).
          The 3-D counterpart of `validate_selfgrav_annulus.py`'s stage 2,
          which did not exist and which is what L = 5 rests on.

Run::

    python3 gate_refstate.py --configuration coarse [--stage g|e|all]

A failure in G-1 is a statement about one of four things -- the flux
condition, the DtN, the density field, or the non-dimensionalisation -- so the
gate isolates rather than merely reporting.  `--isolate` runs the four
discriminating checks described in `isolation_report`.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
import numpy as np
from firedrake import (COMM_WORLD, Constant, Function, FunctionSpace, Mesh,
                       SpatialCoordinate, assemble, avg, conditional, dS, dx,
                       grad, inner, sqrt)
from gadopt import GravitySolver, SphericalDtN
from gadopt.spherical_harmonics import real_spherical_harmonic

import generate_selfgrav_sphere as gen
import reference_state as rs
from validate_selfgrav_sphere import curve_mesh, provenance, tangle_census

MESH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "refstate.msh")


def say(*a):
    if COMM_WORLD.rank == 0:
        print(*a, flush=True)


def build_mesh(**kwargs):
    if COMM_WORLD.rank == 0:
        gen.generate(MESH_FILE, **kwargs)
    COMM_WORLD.barrier()
    mesh = curve_mesh(Mesh(MESH_FILE))
    # Every driver that builds a mesh says how tangled it is, in the arm that
    # actually ran.  This one never untangles, so a nonzero count here is the
    # unrepaired baseline the A/B needs on the other side.
    tangle_census(mesh, "gate_refstate parent, untangle=False")
    return mesh


# ---------------------------------------------------------------------------
# G-1 and G-2
# ---------------------------------------------------------------------------
def gate_g1(psi, quad_degree=6):
    say("\nG-1  |g_0| at the five interfaces against the paper's column")
    say("     expect 10.457 / 10.024 / 9.978 / 9.854 / 9.815 m s^-2,")
    say("     which §2.1 derives from the layered rho_0 as")
    say("     10.4571 / 10.0250 / 9.9783 / 9.8546 / 9.8155; target 1e-3 rel")
    say(f"     {'r (km)':>8} {'tag':>4} {'|g| computed':>13} {'analytic':>10} "
        f"{'paper':>8} {'rel':>10} {'area rel':>10}")
    ok = True
    rows = []
    for r_km in sorted(rs.GRAVITY_COLUMN, reverse=True):
        tag = rs.SURFACE_TAGS[r_km]
        mean_r, rms_t, area = rs.surface_gravity(psi, tag, interior=True)
        g_phys = abs(mean_r) * rs.G_BAR
        want = rs.analytic_gravity(r_km)
        rel = abs(g_phys - want) / want
        r_nd = r_km / (rs.D_SCALE / 1e3)
        area_rel = abs(area - 4 * np.pi * r_nd**2) / (4 * np.pi * r_nd**2)
        flag = "ok " if rel <= 1e-3 else "FAIL"
        ok &= rel <= 1e-3
        say(f"[{flag}] {r_km:8.0f} {tag:4d} {g_phys:13.5f} {want:10.5f} "
            f"{rs.GRAVITY_COLUMN[r_km]:8.3f} {rel:10.2e} {area_rel:10.2e}")
        rows.append((r_km, g_phys, want, rel, rms_t))
    say(f"     {'G-1 PASS' if ok else 'G-1 FAIL'}")

    # G-1b.  The surface average above is the functional §2.3 asks for, and it
    # is also the worst-conditioned one available: it evaluates a discontinuous
    # grad(psi) exactly on the surface where its cell-to-cell jump is largest.
    # The momentum equation never forms it -- the buoyancy term integrates
    # rho_0 grad(psi) over volume -- so the volume norm is reported alongside,
    # and the two together separate "the field is inaccurate" from "the
    # estimator is".
    say("\nG-1b relative L2 error of g_0 over each region, against the closed")
    say("     form (this, not the surface average, is what the momentum")
    say("     equation integrates)")
    for tag, name in [(gen.CELL_MANTLE, "mantle"), (gen.CELL_INNER, "inner"),
                      (gen.CELL_BUFFER, "buffer")]:
        l2, l2r = rs.volume_gravity_error(psi, tag)
        say(f"     {name:<7} |g - g_exact|_L2 / |g_exact|_L2 = {l2:.3e}   "
            f"(radial component {l2r:.3e})")
    return ok, rows


def gate_g2(psi, rows):
    """G-2: is g_0 radial?

    Reported as a *robust* statistic, because the literal max-norm §2.3 asks
    for is unusable here and the reason is worth stating.  At `--coarse`
    exactly **one cell of 113 653** carries |g_0| = 1.16e+05 against a median
    of 1.0028, and a second carries 5.0; both sit on the interfaces of the
    shells that are thinner than one lateral cell, where A2's subdivision
    mechanism leaves a few sliver tetrahedra.  `max|g_t| / max|g|` divides one
    outlier by another and returned 7.1e-08 -- which looks like a triumph and
    means nothing.  So the denominator is the shell median and the 99.9th
    percentile is printed beside the max, which makes the outlier visible
    instead of letting it set the scale.
    """
    say("\nG-2  g_0 is radial")
    say("     expect a few 1e-3, worst where the cells are coarsest;")
    say("     the max-norm is reported but is set by single sliver cells")
    mesh = psi.function_space().mesh()
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    rhat = X / r
    g = grad(psi)
    g_t = g - inner(g, rhat) * rhat
    V = FunctionSpace(mesh, "DG", 0)
    rc = Function(V).interpolate(r)
    tang = Function(V).interpolate(sqrt(inner(g_t, g_t)))
    tot = Function(V).interpolate(sqrt(inner(g, g)))
    owned = mesh.cell_set.size
    rr = rc.dat.data_ro[:owned]
    tt = tang.dat.data_ro[:owned]
    gg = tot.dat.data_ro[:owned]
    say(f"     {'r_in':>9} {'r_out':>9} {'tag':>4} {'cells':>7} {'med|g|':>8} "
        f"{'max|g|':>10} {'max|gt|/med':>12} {'p99.9|gt|/med':>14}")
    worst = 0.0
    for r_in, r_out, tag in gen.shells():
        sel = (rr > r_in) & (rr < r_out)
        if not sel.any():
            continue
        G, T = gg[sel], tt[sel]
        med = float(np.median(G))
        ratio = float(np.percentile(T, 99.9)) / med
        worst = max(worst, ratio)
        say(f"     {r_in:9.6f} {r_out:9.6f} {tag:4d} {int(sel.sum()):7d} "
            f"{med:8.4f} {G.max():10.2e} {T.max() / med:12.3e} {ratio:14.3e}")
    n_bad = int((gg > 10 * np.median(gg)).sum())
    say(f"\n     sliver cells (|g| > 10x the global median): {n_bad} of {owned}")
    say(f"     G-2, robust (worst p99.9 |g_t| / median |g| over shells): "
        f"{worst:.3e}")
    say("\n     tangential g on the five interfaces (rms/|g|, from G-1):")
    for r_km, g_phys, _, _, rms_t in rows:
        say(f"     r = {r_km:6.0f} km   {rms_t * rs.G_BAR / g_phys:.3e}")
    return worst


def isolation_report(mesh, psi, rho_0):
    """The four things a G-1 failure can be, each measured separately.

    §2.3 says a failure here is a statement about the flux condition, the DtN,
    the density field or the non-dimensionalisation.  Each of the four leaves a
    different fingerprint, so they are separated rather than guessed at:

    * **the density field** -- discrete `int rho_0 dV` against the analytic
      total mass.  A non-conforming shell or a wrong layer radius shows here
      and nowhere else, and it is independent of the solve entirely.
    * **the non-dimensionalisation** -- a factor error in `Lambda / 4 pi`
      multiplies every row of G-1 by the same constant, so the *ratios*
      between rows stay right while the values do not.  Reported as the
      spread of the five ratios.
    * **the flux condition** -- it carries only the unmeshed core, so dropping
      it changes g at Rc by the core's share and leaves the outer rows nearly
      alone.  Measured as the predicted share.
    * **the DtN** -- Gauss's law on each sphere, which the discrete solution
      satisfies only if the boundary treatment is consistent.
    """
    say("\nIsolation (what a G-1 failure would be):")
    # The discrete integral covers only the *meshed* domain, so the unmeshed
    # core has to come off the analytic total before they are comparable.
    # Comparing against the whole-Earth mass instead reads as a 4% density
    # error that is not there -- which is exactly the mistake this line exists
    # to stop the next reader making.
    m_disc = assemble(rho_0 * dx(domain=mesh))
    m_total = rs.total_mass() / (rs.RHO_BAR * rs.D_SCALE**3)
    core = (rs.LAYERS_KM[-1][2] / rs.RHO_BAR) * 4 / 3 * np.pi * gen.R_INNER**3
    m_want = m_total - core
    say(f"  density   int rho_0 dV = {m_disc:.9f}, analytic (meshed only) "
        f"{m_want:.9f}, rel {abs(m_disc - m_want) / m_want:.2e}")
    say(f"  flux      unmeshed core mass {core:.9f} = "
        f"{core / m_total * 100:.2f}% of the total {m_total:.9f}; dropping "
        f"the flux would move g(Rc) by that much")
    say(f"  scaling   Lambda = {rs.LAMBDA:.6f} (§2.2: 1.361325), "
        f"G_solver = Lambda/4pi = {rs.LAMBDA / (4 * np.pi):.9f}")
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    say(f"  DtN       Gauss's law on each sphere, "
        f"-(1/4pi r^2) oint dpsi/dn dS vs G M(<r)/r^2:")
    for r_km in sorted(rs.GRAVITY_COLUMN, reverse=True):
        tag = rs.SURFACE_TAGS[r_km]
        r_nd = r_km / (rs.D_SCALE / 1e3)
        # oint dpsi/dn dS = -4 pi G M(<r), so |g| = -(1/4 pi r^2) oint.
        flux = assemble(avg(inner(grad(psi), X / r)) * dS(tag, domain=mesh))
        gval = -flux / (4 * np.pi * r_nd**2) * rs.G_BAR
        want = rs.analytic_gravity(r_km)
        say(f"            r = {r_km:6.0f} km  |g|_Gauss {gval:9.5f}  "
            f"analytic {want:9.5f}  rel {abs(gval - want) / want:.2e}")


# ---------------------------------------------------------------------------
# E-L : the 3-D boundary-treatment residual
# ---------------------------------------------------------------------------
def shell_harmonic(X, r, r_in, r_out, l, m=0):
    """A single real Y_lm supported on a mesh-conforming shell."""
    return conditional(r >= r_in,
                       conditional(r <= r_out,
                                   real_spherical_harmonic(l, m, X), 0.0), 0.0)



def epsilon_stage(mesh, degrees, L_untreated=5, source_shell=1, degree=2,
                  representation="lowrank"):
    r"""A/B the same single-degree source with the mode untreated and treated.

    The instrument is the *difference* of two solves on the same mesh, because
    they share their discretisation error almost exactly -- only the boundary
    rows differ -- so it cancels, and what survives is the spurious growing
    branch the untreated boundary injects.  That is what eps_l is.  Absolute
    errors against a closed form cannot see it: at `--coarse` the CG2
    discretisation floor is orders above eps_l for every l of interest, which
    is exactly the finding the 2-D stage 2 recorded.
    """
    say(f"\nE-L  boundary-treatment residual, exterior DtN at 2 Re")
    say(f"     predicted eps_l(Re) = (Re/2Re)^(2l+1) (l+1-alpha)/(l+alpha), "
        f"alpha = 1")
    say(f"     road map §1.2 quotes 1.0e-04 at l = 6, and the claim that")
    say(f"     L = 5 suffices rests on this being small and falling with l.")
    r_in, r_out, _ = gen.shells()[source_shell]
    say(f"     source: a single Y_l0 on the mesh-conforming shell "
        f"[{r_in:.6f}, {r_out:.6f}], untreated truncation L = {L_untreated}")

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    say(f"     {'l':>3} {'predicted':>11} {'A/B trace':>11} {'ratio':>8} "
        f"{'A/B volume':>11} {'ratio':>8}")
    rows = []
    for l in degrees:
        rho = shell_harmonic(X, r, r_in, r_out, l)
        psis = {}
        for label, L in (("untreated", L_untreated), ("treated", l)):
            p = Function(FunctionSpace(mesh, "CG", degree))
            GravitySolver(
                p, rho,
                bcs={gen.SURF_OUTER: {"dtn": SphericalDtN(L=L)},
                     gen.SURF_INNER: {"dtn": SphericalDtN(L=L)}},
                gravitational_constant=rs.LAMBDA / (4 * np.pi),
                source_quad_degree=2 * l + 2 * degree + 4,
                dtn_representation=representation).solve()
            psis[label] = p
        d = psis["untreated"] - psis["treated"]
        dq = 2 * l + 2 * degree + 4
        d_Re = dS(gen.SURF_RE, domain=mesh, degree=dq)
        d_man = dx(gen.CELL_MANTLE, domain=mesh, degree=dq)
        ab_trace = np.sqrt(assemble(avg(d**2) * d_Re)
                           / assemble(avg(psis["treated"]**2) * d_Re))
        ab_vol = np.sqrt(assemble(d**2 * d_man)
                         / assemble(psis["treated"]**2 * d_man))
        pred = (gen.RE / gen.R_OUTER)**(2 * l + 1) * l / (l + 1)
        say(f"     {l:3d} {pred:11.3e} {ab_trace:11.3e} "
            f"{ab_trace / pred:8.3f} {ab_vol:11.3e} {ab_vol / pred:8.3f}")
        rows.append((l, pred, ab_trace, ab_vol))

    # Null control: the same A/B between two truncations that BOTH treat the
    # mode.  Whatever this returns is the floor of the measurement.
    l = degrees[0]
    rho = shell_harmonic(X, r, r_in, r_out, l)
    ctrl = []
    for L in (l, l + 2):
        p = Function(FunctionSpace(mesh, "CG", degree))
        GravitySolver(
            p, rho,
            bcs={gen.SURF_OUTER: {"dtn": SphericalDtN(L=L)},
                 gen.SURF_INNER: {"dtn": SphericalDtN(L=L)}},
            gravitational_constant=rs.LAMBDA / (4 * np.pi),
            source_quad_degree=2 * l + 2 * degree + 4,
            dtn_representation=representation).solve()
        ctrl.append(p)
    dq = 2 * l + 2 * degree + 4
    d_Re = dS(gen.SURF_RE, domain=mesh, degree=dq)
    floor = np.sqrt(assemble(avg((ctrl[0] - ctrl[1])**2) * d_Re)
                    / assemble(avg(ctrl[1]**2) * d_Re))
    say(f"\n     null control, l = {l} treated by both L = {l} and L = {l + 2}: "
        f"{floor:.3e}")
    say("     (the measurement floor; the table is meaningful above it)")
    seq = [row[2] for row in rows]
    say(f"     monotonically decreasing in l: "
        f"{all(b < a for a, b in zip(seq, seq[1:]))}")
    return rows, floor


def gate_stability(mesh, g_surface, degrees=(1, 2, 3), dtn_degree=5, degree=2,
                   representation="lowrank"):
    r"""S-1: does *this* Phi_0 make long integrations well posed?

    Road map §2.5 asks the question and, less obviously, already answers it in
    closed form.  Its fluid-limit mode-n stiffness is `(g_s - g_Lambda/n)` in
    2-D, where `g_Lambda` comes from the sheet constant `psi_n(a) =
    Lambda sigma a / (2n)`.  In 3-D the sheet constant is
    `psi_n(a) = Lambda sigma a / (2n+1)`, so for a free surface carrying a
    deflection `zeta = zeta_n Y_n` and therefore a mass sheet
    `sigma = rho_s zeta`, the restoring stress is

        rho_s zeta_n [ g_s - Lambda rho_s Re / (2n+1) ]

    and the relaxation is stable in mode n exactly when the bracket is
    positive.  The margin

        k_n = 1 - Lambda rho_s Re / ((2n+1) g_s)

    is worst at the lowest degree present and is a pure property of the
    reference state, so it can be evaluated *before* any time stepping -- which
    is the point, since §9's alternative is discovering it at t = 1000 kyr.

    Measured rather than only predicted: a unit degree-n sheet is applied at Re
    through `interior_sigma` -- the same mechanism the ice load will use, so
    this rehearses handoff trap 1 as well -- and `psi_n` is read back off the
    trace.  `g_s` comes from G-1's own measurement, not from the formula, so
    the margin is computed from the discrete reference state throughout.
    """
    say("\nS-1  fluid-limit stability margin of this reference state")
    say("     road map §2.5: mode-n stiffness ~ (g_s - Lambda rho_s Re/(2n+1)),")
    say("     unstable when it turns negative, worst at the lowest degree.")
    rho_s = rs.LAYERS_KM[0][2] / rs.RHO_BAR
    lam_crit = 3.0 * (g_surface / rs.G_BAR) / (rho_s * gen.RE)
    say(f"     rho_s = {rho_s:.6f}, Re = {gen.RE:.6f}, "
        f"g_s = {g_surface / rs.G_BAR:.6f} (non-dim, from G-1)")
    say(f"     expect psi_n(unit sheet) = Lambda Re/(2n+1) = "
        f"{rs.LAMBDA * gen.RE / 3:.5f} / {rs.LAMBDA * gen.RE / 5:.5f} / "
        f"{rs.LAMBDA * gen.RE / 7:.5f} at n = 1/2/3")
    X = SpatialCoordinate(mesh)
    say(f"     {'n':>3} {'psi_n pred':>11} {'psi_n meas':>11} {'k_n pred':>10} "
        f"{'k_n meas':>10}")
    ok = True
    for n in degrees:
        Y = real_spherical_harmonic(n, 0, X)
        psi = Function(FunctionSpace(mesh, "CG", degree))
        L = max(dtn_degree, n)
        GravitySolver(
            psi, Constant(0.0),
            bcs={gen.SURF_OUTER: {"dtn": SphericalDtN(L=L)},
                 gen.SURF_INNER: {"dtn": SphericalDtN(L=L)},
                 gen.SURF_RE: {"interior_sigma": Y}},
            gravitational_constant=rs.LAMBDA / (4 * np.pi),
            dtn_representation=representation).solve()
        dq = 2 * n + 2 * degree + 4
        d_Re = dS(gen.SURF_RE, domain=mesh, degree=dq)
        meas = (assemble(avg(psi * Y) * d_Re) / assemble(avg(Y * Y) * d_Re))
        pred = rs.LAMBDA * gen.RE / (2 * n + 1)
        k_pred = 1.0 - pred * rho_s / (g_surface / rs.G_BAR)
        k_meas = 1.0 - meas * rho_s / (g_surface / rs.G_BAR)
        ok &= k_meas > 0
        say(f"     {n:3d} {pred:11.5f} {meas:11.5f} {k_pred:10.4f} "
            f"{k_meas:10.4f}")
    say(f"\n     critical Lambda (k_1 = 0) = 3 g_s/(rho_s Re) = "
        f"{lam_crit:.5f}; this model runs at Lambda = {rs.LAMBDA:.5f},")
    say(f"     a margin of {lam_crit / rs.LAMBDA:.3f}x.  "
        f"{'STABLE' if ok else 'UNSTABLE — degree one will grow'}")
    return ok


def main():
    global MESH_FILE
    provenance(os.path.basename(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", default="coarse",
                    choices=list(gen.CONFIGURATIONS))
    ap.add_argument("--stage", default="all", choices=["g", "e", "s", "all"])
    ap.add_argument("--dtn-degree", type=int, default=5)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--eps-degrees", type=int, nargs="+",
                    default=[6, 7, 8, 9, 10])
    ap.add_argument("--representation", default="lowrank",
                    choices=["multiplier", "lowrank"])
    ap.add_argument("--h", type=float, default=None,
                    help="override the lateral spacing (non-dimensional)")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--mesh-file", default=MESH_FILE)
    args = ap.parse_args()
    MESH_FILE = args.mesh_file

    say(f"Configuration {args.configuration}, DtN L = {args.dtn_degree}, "
        f"CG{args.degree}, {args.representation}")
    mesh = build_mesh(configuration=args.configuration, h=args.h)
    say(f"  {mesh.comm.allreduce(mesh.cell_set.size)} cells, "
        f"{COMM_WORLD.size} rank(s)")

    ok = True
    g_surface = rs.analytic_gravity(6371.0)
    if args.stage in ("g", "all"):
        extra = {"snes_monitor": None, "ksp_monitor": None} if args.monitor \
            else None
        psi, rho_0 = rs.solve_reference_potential(
            mesh, dtn_degree=args.dtn_degree, degree=args.degree,
            dtn_representation=args.representation,
            solver_parameters_extra=extra)
        ok, rows = gate_g1(psi)
        gate_g2(psi, rows)
        isolation_report(mesh, psi, rho_0)
        g_surface = rows[0][1]
    if args.stage in ("s", "all"):
        gate_stability(mesh, g_surface, dtn_degree=args.dtn_degree,
                       degree=args.degree, representation=args.representation)
    if args.stage in ("e", "all"):
        epsilon_stage(mesh, args.eps_degrees, L_untreated=args.dtn_degree,
                      degree=args.degree, representation=args.representation)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
