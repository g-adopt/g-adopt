"""B5: the viscoelastic Spada benchmark, self-gravitating, with fields written out.

WHAT THIS IS FOR
    B1 solves one *elastic* snapshot and prints a table. This marches the same
    problem through time against the benchmark's relaxation, and writes the
    fields to disk so the comparison can be redone and looked at without paying
    for the solve again.

    The load is the Spada Task-1 axisymmetric cap, applied as a HEAVISIDE STEP at
    t = 0 and held. That is the convention `taboo_synthesis.love_time` encodes,
    `(1 - exp(s t)) / s`, so the reference and the model see the same history.

WHAT IT WRITES
    <output>/b5-<label>.h5    a Firedrake `CheckpointFile`: the meshes and the
                              physical state at every requested epoch. This is
                              the primary record. Fields are Firedrake
                              `Function`s and belong in a checkpoint, not in a
                              flattened numpy array -- the post-processing script
                              reopens this and can project, interpolate or
                              re-mesh at will.
    <output>/b5-<label>.pvd   VTK for Paraview, same epochs.
    stdout                    the per-epoch comparison table against TABOO.

    Deliberately NOT written: a .npz of coefficients. Anything spectral is
    derived, and deriving it from the checkpoint later costs nothing and can be
    redone when the definition changes -- which it has, twice.

TIME-STEP CHANGES

    The transient solver receives one Firedrake `Constant` for `dt`. The driver
    changes this value with `dt.assign(...)`. The UFL forms then use the new
    value when Firedrake assembles the next Jacobian. The solver and all state
    fields remain live for the complete march.

    The dense Schur preconditioner caches a numerical complement. It reads the
    same live `dt` from the application context. It rebuilds the complement only
    when the value changes.

THE TRUNCATION IS NOT A PHYSICAL ICE CAP, AND THAT IS FINE
    At `--nmax 10` the cap's 10 degree kink is nowhere near resolved and the
    series rings. The comparison is still exact, because
    `TabooReference.synthesise` truncates the reference at the same `nmax`, so
    both sides see the identical band-limited load. Expect ringing in Paraview;
    it is the load, not the solver.

RUN
    python3 b5_viscoelastic.py --mesh b2_coarse_ar7.msh --nmax 10 \
        --dtn-degree 5 --bulk-shear-ratio 100 --epochs 0 1 2 5 10 20
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gadopt  # noqa: E402,F401  (import gadopt before firedrake)
import numpy as np  # noqa: E402
from firedrake import CheckpointFile, COMM_WORLD, Constant, VTKFile  # noqa: E402

import b1_elastic as b1  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402
from b4_polar_motion import DT_LADDER_YR, T_BAR_YR, time_ladder  # noqa: E402


def say(msg):
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


def compare_epoch(t_kyr, U_n, V_n, N_n, ref, sigma_dim, nmax, theta_fine):
    """One row of the benchmark table: model against TABOO at this epoch."""
    U_ref, V_ref, N_ref = ref.synthesise(t_kyr, theta_fine, sigma_dim, nmax=nmax)

    u0_over_u2 = abs(U_n[0]) / max(abs(U_n[2]), 1.0e-300)
    n0_over_n2 = abs(N_n[0]) / max(abs(N_n[2]), 1.0e-300)
    say(f"DEGREE_ZERO t_kyr={t_kyr:g} "
        f"U_0={U_n[0]:.16e} U0_over_U2={u0_over_u2:.16e} "
        f"N_0={N_n[0]:.16e} N0_over_N2={n0_over_n2:.16e}")

    # Spatial, in the driver's own currency: SIGNED argmax for V, because the
    # reference peaks positive and the model may not. Taking max|V| instead
    # hides a sign flip completely -- that is exactly how the tangential
    # discrepancy was mis-read as a 20x amplitude loss for months.
    # **`series_from(nmin=2)`, NOT `series_at`.** `series_at` sums from n = 0 and
    # `dseries_at` from n = 1, while the load is built over `range(2, nmax+1)`
    # and the reference over `np.arange(2, nmax+1)`. Any degree-0 or degree-1
    # content in the model therefore lands in U(0), U(180) and max V with
    # NOTHING on the reference side to match it. `series_from`'s own docstring
    # says so; this file used `series_at` on 2026-08-12 and the result was a
    # reported U(0) ratio of 1.0523 where the honest value over n = 2..10 is
    # 1.0195, and a U(180) ratio of 0.7218 where it is 1.0137 -- the model
    # carried -0.515 m of degree-0/1 content, which swamps a -1.76 m antipodal
    # signal. The per-degree ratios were right all along.
    Um = b1.series_from(U_n, theta_fine, nmin=2, kind="P")
    Nm = b1.series_from(N_n, theta_fine, nmin=2, kind="P")
    Vm = b1.series_from(V_n, theta_fine, nmin=2, kind="dP")
    jm, jr = int(np.argmax(Vm)), int(np.argmax(V_ref))

    say(f"\n  t = {t_kyr:g} kyr")
    say(f"    {'quantity':<12}{'model':>13}{'TABOO':>13}{'ratio':>9}")
    for name, mod, rf in (("U(0)", Um[0], U_ref[0]),
                          ("N(0)", Nm[0], N_ref[0]),
                          ("U(180)", Um[-1], U_ref[-1]),
                          ("N(180)", Nm[-1], N_ref[-1])):
        r = mod / rf if abs(rf) > 1e-30 else float("nan")
        say(f"    {name:<12}{mod:>13.5f}{rf:>13.5f}{r:>9.4f}")
    say(f"    {'max V':<12}{Vm[jm]:>13.5f}{V_ref[jr]:>13.5f}"
        f"{Vm[jm] / V_ref[jr]:>9.4f}"
        f"   at {np.rad2deg(theta_fine[jm]):.2f} vs "
        f"{np.rad2deg(theta_fine[jr]):.2f} deg")
    # PER-DEGREE, which is what the Love-number propagator predicts directly and
    # what separates a broadband error (mesh) from one concentrated at low degree
    # (nullspace, monopole datum). The two spatial numbers above cannot: U(0)
    # sums P_n(1) = 1 with no cancellation while U(180) sums (-1)^n with maximal
    # cancellation, so the same per-degree error shows up very differently in
    # them. Reference coefficients are rebuilt exactly as `synthesise` does.
    hb, lb, kb = ref.love_time(t_kyr, nmax)
    nn = np.arange(ref.nmin_available, nmax + 1)
    cc = 3.0 / taboo.RHO_BAR * sigma_dim[ref.nmin_available:nmax + 1] / (2 * nn + 1)
    say(f"    per-degree   {'n':>3} {'U_n model':>12} {'U_n TABOO':>12} "
        f"{'ratio':>8} {'V_n ratio':>10} {'N_n ratio':>10}")
    for j, n in enumerate(nn):
        um, ur = U_n[n], cc[j] * hb[j]
        vm, vr = V_n[n], cc[j] * lb[j]
        # `love_time` ALREADY adds the direct term for k (`direct = 1.0` there),
        # so `kb` is `1 + k'` and multiplying by `(1 + kb)` counts it twice.
        # Measured before this was fixed: N_n ratios of 0.43-0.48, i.e. the
        # factor of ~2, read as a geoid error when it was arithmetic here.
        nm_, nr = N_n[n], cc[j] * kb[j]
        say(f"    {'':<12} {n:>3} {um:>12.6f} {ur:>12.6f} "
            f"{um / ur if abs(ur) > 1e-30 else float('nan'):>8.4f} "
            f"{vm / vr if abs(vr) > 1e-30 else float('nan'):>10.4f} "
            f"{nm_ / nr if abs(nr) > 1e-30 else float('nan'):>10.4f}")

    return dict(t_kyr=t_kyr, U0=Um[0], U0_ref=U_ref[0], N0=Nm[0],
                N0_ref=N_ref[0], Vmax=Vm[jm], Vmax_ref=V_ref[jr],
                Vth=np.rad2deg(theta_fine[jm]),
                Vth_ref=np.rad2deg(theta_fine[jr]))


def save_state(chk, solver, z, layout, idx, t_kyr):
    """Checkpoint the mixed state COMPONENT BY COMPONENT.

    `chk.save_function(z)` on the mixed function raises

        NonUniqueMeshSequenceError: Found multiple meshes ... where a single
        mesh is expected

    because the coupled space genuinely spans two meshes: `u` and the internal
    variables live on the mechanics submesh, while `psi` and every `Real` field
    live on the parent. `CheckpointFile.save_function` wants one mesh per call,
    so each subfunction is saved on its own.

    The `Real` fields are the DtN multipliers, core pressure, and rotation
    scalars. They are instantaneous solve variables and are not checkpoint
    history. The next solve reconstructs them from the saved physical state.
    """
    idxs = [layout.displacement, layout.potential]
    names = ["displacement", "potential"]
    for i, name in zip(idxs, names):
        f = z.subfunctions[i]
        f.rename(name)
        chk.save_function(f, idx=idx)
    if layout.condensed:
        internal_variables = solver.internal_variables
    else:
        internal_variables = [z.subfunctions[i]
                              for i in layout.internal_variables]
    for k, f in enumerate(internal_variables):
        f.rename(f"internal_variable_{k}")
        chk.save_function(f, idx=idx)
    return len(idxs) + len(internal_variables)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=None, help="explicit .msh (a low-anisotropy one)")
    p.add_argument("--configuration", default="coarse")
    p.add_argument("--h", type=float, default=None,
                   help="explicit lateral spacing in km, generating a mesh "
                        "instead of reading one. For local smoke tests only: "
                        "the production runs pass --mesh.")
    p.add_argument("--nmax", type=int, default=10)
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--bulk-shear-ratio", type=float, default=100.0)
    p.add_argument("--displacement-degree", type=int, default=2)
    p.add_argument("--internal-variable-degree", type=int, default=1)
    p.add_argument("--epochs", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0], help="kyr")
    p.add_argument("--dt-yr", type=float, default=None,
                   help="override the graded ladder with one uniform segment")
    p.add_argument("--outer-rtol", type=float, default=1e-6,
                   help="Outer FGMRES tolerance. The library preset's value. "
                        "B1 uses 1e-8 deliberately, so that solver tolerance is "
                        "excluded as an explanation for its ONE elastic "
                        "snapshot; inheriting that for a 138-step march is not "
                        "the same trade. Measured at 1e-8 on the AR-7 mesh at "
                        "LMAX=10: two outer iterations in 18 minutes, i.e. one "
                        "solve is a large fraction of an hour and the march is "
                        "not viable.")
    p.add_argument("--quad-degree", type=int, default=40)
    p.add_argument("--u-pc", default="gadopt.NearlyIncompressibleAssembledPC")
    p.add_argument("--multiplier-pc", default="none",
                   help="block-1 PC. gadopt.DtNMultiplierDenseSchurPC is the "
                        "build-once dense Schur complement: 19 block-0 calls "
                        "and 69 s/step against none's 354 and 1036 s (Gadi "
                        "176103130). REQUIRES --block0-rtol 1e-4; at 1e-2 it "
                        "stagnates, because S is only as linear as the block-0 "
                        "solve that builds it.")
    p.add_argument("--block0-rtol", type=float, default=1e-2,
                   help="block-0 tolerance. 1e-4 when using the dense block-1 PC.")
    p.add_argument("--snes-type", default="ksponly", choices=["ksponly", "newtonls"])
    p.add_argument("--nproj", type=int, default=None)
    p.add_argument("--label", default="b5")
    p.add_argument("--output", default=None, help="directory for h5/pvd")
    p.add_argument("--no-vtk", action="store_true")
    args = p.parse_args()

    out = args.output or HERE
    epochs = sorted(set(args.epochs))
    nmax = args.nmax
    nproj = args.nproj or nmax
    theta_fine = np.linspace(0.0, np.pi, 4001)

    say("=" * 78)
    say("B5  viscoelastic Spada benchmark, self-gravitating")
    say("=" * 78)
    say(f"  mesh {args.mesh or args.configuration}   nmax {nmax}   "
        f"DtN L {args.dtn_degree}   K/mu {args.bulk_shear_ratio:g}")
    say(f"  displacement degree {args.displacement_degree}   "
        f"internal-variable degree {args.internal_variable_degree}")
    say(f"  epochs (kyr) {epochs}   u_pc {args.u_pc}   snes {args.snes_type}")
    say("  load: WSCOTT parabolic cap, 1500 m ice, 10 deg radius, "
        "931 kg/m^3, Heaviside step")
    say(f"  comparison degrees: 2..{nmax}; DtN exact through degree "
        f"{args.dtn_degree} on the two buffer boundaries")
    say(f"  tolerances: outer {args.outer_rtol:g}; block 0 "
        f"{args.block0_rtol:g}; block 1 {args.multiplier_pc}")

    parent, sub, _, _ = b1.build_meshes(args.configuration, h=args.h,
                                       path=args.mesh)

    ref = taboo.TabooReference(os.path.join(HERE, "reference.npz"))
    sigma_dim = taboo.cap_load(nmax)          # dimensional, for the reference

    # **`--epochs` must truncate the march, and `time_ladder` will not do it.**
    # It spans whatever ladder it is given -- `DT_LADDER_YR` runs to 20 kyr --
    # and uses the epoch list only to mark WHERE OUTPUT IS WRITTEN. Passing the
    # bare ladder therefore marches to 20 kyr whatever `--epochs` says, writing
    # nothing after the last requested epoch. Measured: `--epochs 0 1 2` gave
    # "5 segments, 138 solves" where 38 were wanted, and a probe meant to stop
    # at 2 kyr was on course to grind to 20. Truncate the ladder here instead.
    t_end = max(epochs) * 1000.0
    if args.dt_yr is None:
        lad = []
        for upto, dt in DT_LADDER_YR:
            if upto >= t_end:
                lad.append((t_end, dt))
                break
            lad.append((upto, dt))
        ladder = tuple(lad)
    else:
        ladder = ((t_end, args.dt_yr),)
    segments = time_ladder(epochs, ladder)
    say(f"  ladder {ladder}")
    say(f"  {len(segments)} output segments, one transient solver, "
        f"{sum(s[3] for s in segments)} solves total")

    h5 = os.path.join(out, f"b5-{args.label}.h5")
    # ONE VTK FILE PER MESH. `VTKFile.write(*z.subfunctions)` raises
    # `ValueError: All functions must be on same mesh` for exactly the reason
    # `save_state` exists: `u` lives on the mechanics submesh and `psi` on the
    # parent. Measured -- job b5march died here on 2026-08-12 AFTER writing the
    # t = 0 checkpoint, so the checkpoint fix alone was not enough and the march
    # never started.
    vtk_u = vtk_psi = None
    if not args.no_vtk:
        vtk_u = VTKFile(os.path.join(out, f"b5-{args.label}-mechanics.pvd"))
        vtk_psi = VTKFile(os.path.join(out, f"b5-{args.label}-potential.pvd"))
    rows = []

    with CheckpointFile(h5, "w") as chk:
        # BOTH meshes: the mechanics submesh carries `u` and the internal
        # variables, the parent carries `psi`. See `save_state`.
        chk.save_mesh(sub)
        chk.save_mesh(parent)

        solver, z, layout = None, None, None

        # t = 0 must be solved SEPARATELY, and `time_ladder` says so by
        # dropping it: it builds its segment list from `e > 0` only. The load is
        # a Heaviside step, so the t = 0 state is the instantaneous ELASTIC
        # response, which is the dt -> 0 limit rather than anything the march
        # produces -- the first marched step at dt = 10 yr already carries some
        # relaxation. This is B1's own snapshot at `DT_ELASTIC`, and it is the
        # anchor: it must reproduce the already-validated elastic answer, so a
        # wrong mesh or load shows up here before any service unit is spent on
        # the march.
        if any(abs(e) < 1e-12 for e in epochs):
            say("\n  t = 0: the elastic snapshot, solved at DT_ELASTIC "
                f"({b1.DT_ELASTIC:g} Maxwell times) rather than marched")
            solver, z, layout, _, _, _ = b1.build_solver(
                parent, sub, nmax, dtn_degree=args.dtn_degree,
                udeg=args.displacement_degree,
                ivdeg=args.internal_variable_degree,
                condense=True, outer_rtol=args.outer_rtol,
                bulk_shear_ratio=args.bulk_shear_ratio,
                u_pc=args.u_pc, snes_type=args.snes_type,
                block0_rtol=args.block0_rtol, multiplier_pc=args.multiplier_pc,
                dt=b1.DT_ELASTIC)
            tic = time.time()
            solver.solve()
            say(f"      elastic solve {time.time() - tic:.1f} s")
            U_n, V_n, N_n = b1.surface_spectra(
                solver, parent, sub, nproj, quad_degree=args.quad_degree)
            rows.append(compare_epoch(0.0, U_n, V_n, N_n, ref,
                                      sigma_dim, nmax, theta_fine))
            save_state(chk, solver, z, layout, 0, 0.0)
            if vtk_u is not None:
                vtk_u.write(z.subfunctions[layout.displacement], time=0.0)
                vtk_psi.write(z.subfunctions[layout.potential], time=0.0)
            say("      written: checkpoint idx 0, t = 0 kyr")
            # The march starts from REST, not from this state. With the load
            # held from t = 0, marching from zero reproduces the elastic
            # response inside the first step; seeding it would count it twice.

        if segments:
            dt = Constant(segments[0][2] / T_BAR_YR)
            solver, z, layout, sigma_n, _, _ = b1.build_solver(
                parent, sub, nmax, dtn_degree=args.dtn_degree,
                udeg=args.displacement_degree,
                ivdeg=args.internal_variable_degree,
                condense=True, outer_rtol=args.outer_rtol,
                bulk_shear_ratio=args.bulk_shear_ratio,
                u_pc=args.u_pc, snes_type=args.snes_type,
                block0_rtol=args.block0_rtol, multiplier_pc=args.multiplier_pc,
                dt=dt)

        previous_dt_yr = None
        for t0, t1, dt_yr, nsteps, is_epoch in segments:
            if previous_dt_yr is None or dt_yr != previous_dt_yr:
                dt.assign(dt_yr / T_BAR_YR)
                say(f"\n      dt assigned: {dt_yr:g} yr "
                    f"({float(dt):.8g} Maxwell times)")
                previous_dt_yr = dt_yr

            say(f"\n  {t0 / 1000:7.3f} -> {t1 / 1000:7.3f} kyr   "
                f"dt {dt_yr:7.2f} yr   {nsteps:4d} steps")
            for k in range(nsteps):
                tic = time.time()
                solver.solve()
                elapsed = time.time() - tic
                t_step_kyr = (t0 + (k + 1) * dt_yr) / 1000.0
                say(f"TIMESTEP t_kyr={t_step_kyr:.9g} "
                    f"dt_yr={dt_yr:.9g} segment_step={k + 1} "
                    f"segment_steps={nsteps} wall_s={elapsed:.6f}")
                if k == 0:
                    say(f"      first step {elapsed:6.1f} s")
                elif k == 1:
                    say(f"      per step   {elapsed:6.1f} s")

            if is_epoch:
                t_kyr = t1 / 1000.0
                U_n, V_n, N_n = b1.surface_spectra(
                    solver, parent, sub, nproj,
                    quad_degree=args.quad_degree)
                row = compare_epoch(t_kyr, U_n, V_n, N_n, ref,
                                    sigma_dim, nmax, theta_fine)
                rows.append(row)
                ratio = abs(row["U0"] / row["U0_ref"])
                if not (0.1 < ratio < 10.0):
                    raise RuntimeError(
                        f"U(0) ratio {ratio:.4g} at t = {t_kyr} kyr is "
                        "outside [0.1, 10]. The marched state is broken.")
                save_state(chk, solver, z, layout, len(rows) - 1, t_kyr)
                if vtk_u is not None:
                    vtk_u.write(z.subfunctions[layout.displacement],
                                time=t_kyr)
                    vtk_psi.write(z.subfunctions[layout.potential],
                                  time=t_kyr)
                say(f"      written: checkpoint idx {len(rows) - 1}, "
                    f"t = {t_kyr} kyr")

    say("\n" + "=" * 78)
    say("SUMMARY  model / TABOO")
    say("=" * 78)
    say(f"  {'t (kyr)':>8}{'U(0)':>10}{'N(0)':>10}{'max V':>10}"
        f"{'V peak deg':>12}{'ref deg':>10}")
    for r in rows:
        say(f"  {r['t_kyr']:>8g}{r['U0'] / r['U0_ref']:>10.4f}"
            f"{r['N0'] / r['N0_ref']:>10.4f}{r['Vmax'] / r['Vmax_ref']:>10.4f}"
            f"{r['Vth']:>12.2f}{r['Vth_ref']:>10.2f}")
    say(f"\n  checkpoint: {h5}")
    if vtk_u is not None:
        say(f"  vtk       : b5-{args.label}-{{mechanics,potential}}.pvd")
    say(f"RESULT label={args.label} completed_epochs="
        f"{','.join(f'{e:g}' for e in epochs)} checkpoint={h5}")
    say("\n  At K/mu = 100 the ELASTIC (t = 0) U_n ratio is predicted 1.0229 at")
    say("  n = 2, not 1.0: nu = 0.495 is not incompressible. See")
    say("  NOTES/RESULTS-2026-08-11.md and SUBMIT-INCOMPRESSIBLE.sh.")


if __name__ == "__main__":
    main()
