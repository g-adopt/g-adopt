"""The baseline cost of the MULTIPLIER DtN path in the coupled solver.

WHAT THIS IS FOR
    The low-rank DtN path will be judged against the multiplier path. This
    measures the multiplier path, so the yardstick exists before the thing it
    measures. It is deliberately written against the incumbent alone, and it
    takes `--representation` so that the same instrument, unchanged, measures
    the low-rank path when that path exists.

EVERY COUNT IS A MEDIAN OVER REPEATED SOLVES, AND THE SPREAD IS PRINTED
    Repeating the same solve does not repeat the same counts. Measured on the
    annulus at L = 3, three identical warm solves gave 27, 25 and 25 block-0
    applications and 474, 467 and 464 inner iterations. `--nsolve` sets how many
    solves are run; the first is the cold one and the rest give the median and
    the spread. A difference between two paths that is smaller than that spread
    is noise, and the table prints the spread next to every count so that this
    cannot be overlooked.

WHAT IT REPORTS, PER SOLVE
    n_modes           the DtN multiplier count, read from the form rather than
                      from the formula 2 (L + 1)^2. In 3-D with two spherical
                      boundaries the two agree; in 2-D they do not, and a
                      formula that is right for one geometry and silently wrong
                      for the other is not an instrument.
    outer             outer FGMRES iterations.
    block0_applies    applications of block 0 of `DtNTwoBlockSchurPC`. THIS IS
                      THE COST, not the outer count and not the number of times
                      the Python `PC.apply` was entered. Read from the
                      `dtn_fieldsplit_0_` converged-reason lines and from
                      nowhere else, which is what
                      `selfgrav_dtn_iterative_solver_parameters` says its
                      per-split lines are for.
    block0_its        total inner FGMRES iterations summed over those
                      applications.
    mg_sweeps         multigrid sweeps, per inner split (`u`, `psi`, and `m`
                      when it is not condensed). One sweep is one V-cycle: the
                      inner splits run `ksp_type preonly`, so each reported line
                      is exactly one application of GAMG.
    block1_applies    applications of the `Real` block, for completeness.
    t_cold, t_warm    wall clock. The first solve carries form compilation; the
                      later ones do not. Both are reported because an iteration
                      count on its own has misled this project before.

                      **t_cold IS ONLY A COLD TIME IN THE FIRST CONFIGURATION
                      OF A PROCESS.** The kernel cache is per process, and two
                      configurations share most of their kernels, so a `--sweep`
                      gives a genuine cold time for its first value and a
                      partly-warm one for the rest. Measured on the annulus:
                      L = 2 took 40.4 s, then L = 3 took 7.0 s and L = 5 took
                      23.1 s in the same process. To measure the compile cost
                      against `t_cold = 106 + 3.844 n + 0.00343 n^2`, run one
                      configuration per process, and clear the cache between
                      them.

HOW THE COUNTS ARE TAKEN
    PETSc writes converged reasons through its own viewer onto file descriptor
    1, not through `sys.stdout`. `contextlib.redirect_stdout` therefore captures
    nothing and every count comes back zero, which reads as a solver that did no
    work rather than as a broken instrument. The capture here is at the file
    descriptor level, as `NOTES/poisson/bench/amg_applications.py` does it.

    `getIterationNumber()` is not used. It returns the LAST solve of a repeated
    inner solve, and nothing says the others agree.

TWO CASES, ONE INSTRUMENT
    --case annulus2d  A small 2-D annulus. Runs on a laptop in about a minute.
                      It establishes that the instrument works and that its
                      numbers move. It establishes NOTHING about whether
                      anything helps: see rule 2 of NOTES/fastdtn/HISTORICAL-PLAN.md.
    --case spada3d    The B1/B5 configuration, through `b1_elastic.build_solver`,
                      so the harness measures the driver's own solver rather
                      than a second one built to look like it. Every 3-D number
                      comes from Gadi.

                      **There is no smaller 3-D case than `--configuration
                      coarse`.** `--h` is offered by `b1.build_meshes` but
                      `generate_selfgrav_sphere` fails in gmsh at h = 800, 1000,
                      1500 and 2000 km, all with `PLC Error: A segment and a
                      facet intersect at point`. The layer radii are fixed, so a
                      lateral spacing above about 500 km cannot resolve them.
                      Use `--case annulus2d` for a laptop-scale check.

RUN
    # local, the instrument check
    PYTHONPATH=<worktree> python3 bench_dtn_baseline.py --case annulus2d

    # the sensitivity sweep that rule 7 of the plan requires
    PYTHONPATH=<worktree> python3 bench_dtn_baseline.py --case annulus2d \
        --sweep dtn-degree 3 5 8
    PYTHONPATH=<worktree> python3 bench_dtn_baseline.py --case annulus2d \
        --sweep block0-rtol 1e-1 1e-2 1e-4

    # on Gadi, inside a job
    mpiexec -np 96 python3 bench_dtn_baseline.py --case spada3d \
        --mesh b2_coarse_ar7.msh --dtn-degree 5 --json baseline-L5.json
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# PETSc reads `sys.argv` into its own options database at import, so
# `--dtn-degree 5` lands there as an unrecognised option and is echoed as a
# spelling mistake at exit. Hide the command line from it and give it back to
# argparse afterwards.
_ARGV, sys.argv = list(sys.argv), sys.argv[:1]

import gadopt  # noqa: E402,F401  (import gadopt before firedrake)
import numpy as np  # noqa: E402
from firedrake import COMM_WORLD  # noqa: E402

# The reason-line format PETSc writes. Both outcomes are matched: a solve that
# hit its cap reports DIVERGED_ITS and still did the work, and dropping those
# lines would undercount block 0 exactly in the runs where block 0 is dearest.
_REASON = re.compile(
    r"Linear\s+(\S*?)\s*solve (?:converged|did not converge) due to "
    r"(\w+) iterations (\d+)")
_SNES = re.compile(
    r"Nonlinear\s+(\S*?)\s*solve (?:converged|did not converge) due to "
    r"(\w+) iterations (\d+)")


def say(msg):
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# The capture
# --------------------------------------------------------------------------

def capture_solve(solver):
    """One solve, returning (seconds, the text PETSc wrote to fd 1).

    The redirect is on every rank. PETSc's default viewer prints on rank 0 of
    the solve's communicator only, so the other ranks come back empty, and that
    is checked rather than assumed: see `parse_counts`, which returns the rank-0
    text and the number of ranks that saw any reason line at all.
    """
    import tempfile
    with tempfile.TemporaryFile(mode="w+") as sink:
        saved = os.dup(1)
        sys.stdout.flush()
        os.dup2(sink.fileno(), 1)
        try:
            tic = time.perf_counter()
            solver.solve()
            dt = time.perf_counter() - tic
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        sink.seek(0)
        text = sink.read()
    return dt, text


def parse_counts(text):
    """Classify every converged-reason line by its options prefix.

    The prefixes nest, so the order of the tests is load bearing: an inner
    sweep's prefix CONTAINS `dtn_fieldsplit_0_`, and testing for block 0 first
    would count every sweep as a block-0 application. That mistake inflates the
    headline cost by roughly the sweep count and looks entirely plausible.
    """
    lines = [(m.group(1), m.group(2), int(m.group(3)))
             for m in _REASON.finditer(text)]
    out = {
        "outer": 0, "outer_reason": None,
        "block0_applies": 0, "block0_its": 0, "block0_diverged": 0,
        "block1_applies": 0, "block1_its": 0,
        "mg_sweeps": {}, "unclassified": [],
    }
    for prefix, reason, its in lines:
        inner = re.search(r"dtn_fieldsplit_0_fieldsplit_(\d+)_$", prefix)
        if inner is not None:
            k = f"split_{inner.group(1)}"
            out["mg_sweeps"][k] = out["mg_sweeps"].get(k, 0) + 1
        elif prefix.endswith("dtn_fieldsplit_0_"):
            out["block0_applies"] += 1
            out["block0_its"] += its
            if reason.startswith("DIVERGED"):
                out["block0_diverged"] += 1
        elif prefix.endswith("dtn_fieldsplit_1_"):
            out["block1_applies"] += 1
            out["block1_its"] += its
        elif "fieldsplit" not in prefix:
            # The outer FGMRES. There is exactly one of these per linear solve;
            # with `snes_type newtonls` there is one per Newton step and the
            # last is the one that matters, so take the last and record how
            # many there were.
            out["outer"] = its
            out["outer_reason"] = reason
            out["outer_solves"] = out.get("outer_solves", 0) + 1
        else:
            out["unclassified"].append(prefix)
    snes = [(m.group(1), m.group(2), int(m.group(3))) for m in _SNES.finditer(text)]
    out["newton_steps"] = snes[-1][2] if snes else None
    out["reason_lines"] = len(lines)
    return out


# --------------------------------------------------------------------------
# The two cases
# --------------------------------------------------------------------------

def build_annulus(args):
    """A small 2-D self-gravitating GIA annulus on the production preset.

    The geometry and the material are the S7 smoke test's; the SOLVER is
    `selfgrav_dtn_iterative_solver_parameters`, i.e. the 3-D preset, because the
    quantity being measured is that preset's block structure. The 2-D default
    preset does a direct solve on block 0 and would report one application of
    everything, which is a true statement about a configuration nobody runs.
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(HERE)), "gravity"))
    import generate_selfgrav_annulus as gen
    from validate_selfgrav_annulus import curve_mesh

    from firedrake import (Function, Measure, Mesh,  # noqa: E402
                           SpatialCoordinate, Submesh, assemble, atan2, cos,
                           dot)
    from gadopt import (CompressibleInternalVariableApproximation,  # noqa: E402
                        CylindricalDtN, SelfGravitatingGIASolver,
                        selfgrav_dtn_iterative_solver_parameters,
                        self_gravitating_gia_space)
    from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402

    mesh_path = os.path.join(args.workdir, f"bench_annulus_{args.dr:g}.msh")
    if COMM_WORLD.rank == 0 and not os.path.exists(mesh_path):
        gen.generate(mesh_path, dr_mantle=args.dr, n_azimuthal=args.nazim)
    COMM_WORLD.barrier()

    parent = curve_mesh(Mesh(mesh_path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False

    B_mu, Lambda = 1.2769, 1.1116
    X = SpatialCoordinate(parent)
    sigma = 1.0e-3 * cos(2 * atan2(X[1], X[0]))
    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(args.dtn_degree)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(args.dtn_degree)},
        gen.CURVE_RE: {"interior_sigma": sigma},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=args.rotation,
        self_gravity_number=Lambda,
        condense_internal_variables=args.condense)
    z = Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        bulk_shear_ratio=args.bulk_shear_ratio,
        g=1.0, B_mu=B_mu, self_gravity_number=Lambda)

    Xm = SpatialCoordinate(sub)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    C = assemble(1.0 * dot(Xm, Xm) * dx_m)
    bcs = {
        gen.CURVE_RC: {"un": 0.0},
        gen.CURVE_RE: {"normal_stress":
                       B_mu * 1.0e-3 * cos(2 * atan2(Xm[1], Xm[0]))},
    }
    params = selfgrav_dtn_iterative_solver_parameters(
        condensed=args.condense, block0_rtol=args.block0_rtol,
        outer_rtol=args.outer_rtol, block0_max_it=args.block0_max_it,
        snes_type=args.snes_type, u_pc=args.u_pc)
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=args.dt, bcs=bcs,
        rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH,
        solver_parameters=params)
    meta = dict(cells_parent=parent.num_cells(), cells_mantle=sub.num_cells(),
                dofs=Z.dim())
    return solver, z, layout, meta


def build_spada(args):
    """The B1/B5 3-D configuration, through the driver's own builder.

    `b1.build_solver` takes no `block0_rtol`, so the dictionary is built here
    with `b1.condensed_solver_parameters` and handed in. That is the driver's
    OWN builder, not a copy of it, so a change to B1's preset reaches this
    harness rather than leaving the baseline measuring a configuration that no
    longer exists.
    """
    import b1_elastic as b1
    parent, sub, _, _ = b1.build_meshes(args.configuration, h=args.h,
                                        path=args.mesh)
    params = b1.condensed_solver_parameters(
        outer_rtol=args.outer_rtol, block0_rtol=args.block0_rtol,
        block0_max_it=args.block0_max_it, u_pc=args.u_pc,
        snes_type=args.snes_type)
    solver, z, layout, _, _, _ = b1.build_solver(
        parent, sub, args.nmax, dtn_degree=args.dtn_degree,
        rotation=args.rotation, condense=args.condense,
        bulk_shear_ratio=args.bulk_shear_ratio,
        solver_parameters=params,
        dt=b1.DT_ELASTIC if args.dt is None else args.dt)
    meta = dict(cells_parent=parent.num_cells(), cells_mantle=sub.num_cells(),
                dofs=z.function_space().dim())
    return solver, z, layout, meta


CASES = {"annulus2d": build_annulus, "spada3d": build_spada}


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

def measure(args):
    """Build one configuration and solve it `--nsolve` times."""
    if args.representation != "multiplier":
        raise SystemExit(
            f"--representation {args.representation!r}: the coupled solver "
            "takes no such keyword yet. This harness is the BASELINE and "
            "measures the multiplier path. When `dtn_representation` reaches "
            "`SelfGravitatingGIASolver`, pass it through in the two builders "
            "above and nothing else here changes.")

    t0 = time.perf_counter()
    solver, z, layout, meta = CASES[args.case](args)
    t_build = time.perf_counter() - t0

    n_modes = layout.gravity_form.n_multipliers
    rec = dict(case=args.case, representation=args.representation,
               dtn_degree=args.dtn_degree, n_modes=n_modes,
               n_multiplier_fields=len(layout.multipliers),
               n_rotation_fields=len(layout.rotation),
               formula_2Lp1sq=2 * (args.dtn_degree + 1) ** 2,
               block0_rtol=args.block0_rtol, outer_rtol=args.outer_rtol,
               block0_max_it=args.block0_max_it, condense=args.condense,
               ranks=COMM_WORLD.size, t_build=t_build, **meta)

    solves = []
    for k in range(args.nsolve):
        z.assign(0.0)
        if hasattr(solver, "solution_old"):
            solver.solution_old.assign(0.0)
        dt, text = capture_solve(solver)
        counts = parse_counts(text)
        # Whether ANY rank saw reason lines, and how many did. On the default
        # PETSc viewer this must be exactly 1. If it is 0 the instrument is
        # blind; if it is more than 1 the counts below are a single rank's and
        # the others are being thrown away silently.
        saw = COMM_WORLD.allreduce(1 if counts["reason_lines"] else 0)
        counts["ranks_reporting"] = saw
        counts["seconds"] = dt
        counts["phase"] = "cold" if k == 0 else "warm"
        solves.append(counts)
        if args.echo and COMM_WORLD.rank == 0:
            sys.stdout.write(text)

    warm = solves[1:] or solves
    rec["t_cold"] = solves[0]["seconds"]
    rec["t_warm"] = float(np.median([s["seconds"] for s in warm]))
    rec["t_warm_spread"] = (
        float(max(s["seconds"] for s in warm) - min(s["seconds"] for s in warm))
        if len(warm) > 1 else None)
    rec["solves"] = solves

    # **REPEATING THE SAME SOLVE DOES NOT REPEAT THE SAME COUNTS**, and this is
    # measured rather than assumed. On the annulus at L = 3 three identical warm
    # solves gave 27, 25 and 25 block-0 applications and 474, 467 and 464 inner
    # iterations. So a single solve carries a spread of a few per cent, and any
    # comparison of two paths that turns on less than that is reading noise.
    # The median over the warm solves is the number; the spread is reported
    # beside it and is the floor.
    for key in ("outer", "block0_applies", "block0_its", "block0_diverged",
                "block1_applies"):
        vals = [s[key] for s in warm]
        rec[key] = int(np.median(vals))
        rec[key + "_spread"] = int(max(vals) - min(vals))
    sweeps = [sum(s["mg_sweeps"].values()) for s in warm]
    rec["mg_sweeps_total"] = int(np.median(sweeps))
    rec["mg_sweeps_total_spread"] = int(max(sweeps) - min(sweeps))
    last = warm[-1]
    rec["mg_sweeps"] = last["mg_sweeps"]
    for key in ("ranks_reporting", "outer_reason", "newton_steps"):
        rec[key] = last[key]
    rec["unclassified"] = sorted(set(last["unclassified"]))
    return rec


def check_floors(rec):
    """The floors, checked rather than described.

    Rule 7 of `NOTES/fastdtn/HISTORICAL-PLAN.md`: a measurement must be able to come back
    negative. Each of these is a way for the instrument to be silently blind,
    and each has a value that a blind instrument would produce.
    """
    problems = []
    if rec["ranks_reporting"] == 0:
        problems.append(
            "no rank saw a converged-reason line: the fd-1 capture is blind, "
            "or `ksp_converged_reason` is not set on this configuration. "
            "Every count below is a floor, not a measurement.")
    elif rec["ranks_reporting"] > 1:
        problems.append(
            f"{rec['ranks_reporting']} ranks reported converged-reason lines; "
            "the counts are one rank's and the rest are discarded.")
    if rec["outer"] == 0:
        problems.append("outer FGMRES reported 0 iterations: the solve was a "
                        "no-op, or the outer prefix was not matched.")
    if rec["block0_applies"] == 0:
        problems.append(
            "0 block-0 applications: the preconditioner is not "
            "`DtNTwoBlockSchurPC`, or its sub-KSPs carry no "
            "`ksp_converged_reason`. This is the value a blind instrument "
            "returns and it must never be read as 'block 0 is free'.")
    elif rec["block0_applies"] < rec["outer"]:
        problems.append(
            f"block-0 applications ({rec['block0_applies']}) is below the "
            f"outer count ({rec['outer']}): a flexible outer Krylov applies "
            "the preconditioner at least once per iteration, so this is "
            "under-counting.")
    if rec["mg_sweeps_total"] and rec["mg_sweeps_total"] < rec["block0_applies"]:
        problems.append(
            f"{rec['mg_sweeps_total']} multigrid sweeps against "
            f"{rec['block0_applies']} block-0 applications: every application "
            "runs the multiplicative sweep at least once.")
    if not rec["mg_sweeps"]:
        problems.append("no inner-split lines: the multigrid sweeps are not "
                        "being counted at all.")
    if rec["unclassified"]:
        problems.append("prefixes matched no rule, so their lines are "
                        f"uncounted: {rec['unclassified']}")
    return problems


def report(recs, args):
    say("")
    say("=" * 100)
    say(f"BASELINE  multiplier DtN path   case={args.case}   "
        f"ranks={COMM_WORLD.size}")
    say("=" * 100)
    say("  Median over the warm solves, with the observed spread in brackets. "
        "The spread is the floor: a difference smaller than it is noise.")
    head = (f"{'L':>3} {'modes':>6} {'Rfields':>8} {'outer':>9} "
            f"{'blk0':>10} {'blk0 its':>12} {'div':>4} {'blk1':>7} "
            f"{'sweeps':>12} {'t_cold':>9} {'t_warm':>9} {'+-':>8}")
    say(head)
    say("-" * len(head))
    for r in recs:
        tw = "-" if r["t_warm"] is None else f"{r['t_warm']:.3f}"
        sp = "-" if r["t_warm_spread"] is None else f"{r['t_warm_spread']:.3f}"

        def pm(key):
            return f"{r[key]}[{r.get(key + '_spread', 0)}]"

        say(f"{r['dtn_degree']:>3} {r['n_modes']:>6} "
            f"{r['n_multiplier_fields'] + r['n_rotation_fields']:>8} "
            f"{pm('outer'):>9} {pm('block0_applies'):>10} "
            f"{pm('block0_its'):>12} "
            f"{r['block0_diverged']:>4} {r['block1_applies']:>7} "
            f"{pm('mg_sweeps_total'):>12} {r['t_cold']:>9.3f} {tw:>9} {sp:>8}")
    say("")
    for r in recs:
        say(f"  L={r['dtn_degree']} block0_rtol={r['block0_rtol']:g}  "
            f"sweeps by split, last solve only {r['mg_sweeps']}  "
            f"outer_reason={r['outer_reason']}  "
            f"newton_steps={r['newton_steps']}  dofs={r['dofs']}")
    say("")
    bad = False
    for r in recs:
        for p in check_floors(r):
            say(f"  INSTRUMENT WARNING (L={r['dtn_degree']}, "
                f"block0_rtol={r['block0_rtol']:g}): {p}")
            bad = True
    if not bad:
        say("  instrument: every count above its floor, exactly one rank "
            "reporting, no unclassified prefixes.")
    say("")
    say("  Reminder: this is a 2-D annulus unless --case spada3d. It shows "
        "that the instrument works. It shows nothing about whether anything "
        "helps." if args.case == "annulus2d" else
        "  3-D. Quote this only if it came from Gadi.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="annulus2d", choices=sorted(CASES))
    p.add_argument("--representation", default="multiplier",
                   help="'multiplier' is the only value the coupled solver "
                        "accepts today; see `measure`.")
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--block0-rtol", type=float, default=1e-2)
    p.add_argument("--outer-rtol", type=float, default=1e-6)
    p.add_argument("--block0-max-it", type=int, default=200,
                   help="B1's and B5's value, not the library preset's 60. "
                        "A2's anisotropic lithosphere puts the condensed "
                        "[u, psi] sweep in the 174-388 band, so a cap of 60 "
                        "binds on every application and the baseline would "
                        "measure the cap rather than the solver.")
    p.add_argument("--snes-type", default="ksponly",
                   choices=["ksponly", "newtonls"])
    p.add_argument("--u-pc", default="gadopt.RigidBodyAssembledPC")
    p.add_argument("--condense", action="store_true", default=True)
    p.add_argument("--no-condense", dest="condense", action="store_false")
    p.add_argument("--rotation", action="store_true", default=False)
    p.add_argument("--bulk-shear-ratio", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=None,
                   help="timestep in Maxwell times. The default is the elastic "
                        "snapshot. NOTE the effective bulk/shear ratio is "
                        "nominal x (1 + dt/tau), so dt chooses the operating "
                        "point as much as --bulk-shear-ratio does.")
    p.add_argument("--nsolve", type=int, default=3,
                   help="solves per configuration. The first is cold, the rest "
                        "are warm; the warm spread is the timing noise floor.")
    # annulus only
    p.add_argument("--dr", type=float, default=0.15)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--workdir", default=None)
    # spada only
    p.add_argument("--mesh", default=None)
    p.add_argument("--configuration", default="coarse")
    p.add_argument("--h", type=float, default=None)
    p.add_argument("--nmax", type=int, default=10)
    # sweeps and output
    p.add_argument("--sweep", nargs="+", default=None,
                   metavar=("VARIABLE", "VALUE"),
                   help="repeat the measurement over one variable, e.g. "
                        "`--sweep dtn-degree 3 5 8` or "
                        "`--sweep block0-rtol 1e-1 1e-2 1e-4`.")
    p.add_argument("--echo", action="store_true",
                   help="replay the captured PETSc output, to check the "
                        "parser against what PETSc actually wrote.")
    p.add_argument("--json", default=None)
    args = p.parse_args(_ARGV[1:])

    if args.dt is None and args.case == "annulus2d":
        args.dt = 1.0
    args.workdir = args.workdir or HERE

    if args.sweep:
        name, values = args.sweep[0].replace("-", "_"), args.sweep[1:]
        if not hasattr(args, name):
            raise SystemExit(f"--sweep {name}: no such argument")
        caster = type(getattr(args, name))
        recs = []
        for v in values:
            setattr(args, name, caster(v))
            say(f"\n### {name} = {getattr(args, name)}")
            recs.append(measure(args))
            recs[-1]["swept"] = name
    else:
        recs = [measure(args)]

    report(recs, args)
    if args.json and COMM_WORLD.rank == 0:
        with open(args.json, "w") as f:
            json.dump(recs, f, indent=2)
        say(f"  json: {args.json}")


if __name__ == "__main__":
    main()
