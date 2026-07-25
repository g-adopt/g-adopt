r"""Weak/stress-scaling model for the gravitational Poisson DtN solver.

One `(level, L)` case of the scaling study described in `SCALING-ANALYSIS.md`.
It builds an extruded cubed-sphere shell, drives it with a two-bump
spherical-harmonic checkerboard density (so every treated DtN mode on both
boundaries is genuinely excited), solves with the iterative `GravitySolver`
preset, and records the iteration counts, the per-degree boundary power
spectrum, and the wall times into a JSON sidecar next to the log.

The solver bundle here fixes the two things the study needs that the shipped
preset does not do on its own: the `fieldsplit_1` GMRES is run non-restarted
(`ksp_gmres_restart = n`, since PETSc's default restart of 30 would censor every
count above 30) and capped at `~1.2 n` (the guaranteed GMRES termination point
plus slack, so a pathological corner fails fast and diagnosably rather than on
walltime).

Run locally against this worktree with, e.g.

    PYTHONPATH=<worktree> <firedrake-python> gravity_cubed_sphere.py 4 --lmax 5
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# PETSc (initialised when gadopt imports petsc4py) scans sys.argv and warns
# about our argparse flags as unknown options. Blank argv across the import so
# PETSc sees only the script name; argparse re-reads the real argv below.
_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from gadopt import *  # noqa: E402
from gadopt.spherical_harmonics import real_spherical_harmonic_numpy  # noqa: E402
sys.argv = _argv

# Same shell as the Stokes scaling model, so mesh resolution and aspect ratio
# are directly comparable.
RMIN, RMAX = 1.22, 2.22

# Isotropic refinement: the extrusion layer count doubles with the level in
# lockstep with the horizontal refinement, holding cell shape fixed.
LAYERS = {4: 8, 5: 16, 6: 32, 7: 64}

# PCFIELDSPLIT used to refuse more than 128 fields, and the mixed space carries
# 1 + 2(L+1)^2 of them, which capped this model at L = 6. The shipped preset now
# describes the two Schur blocks by index set (gadopt.DtNTwoBlockSchurPC), so the
# field enumeration never runs and the coupled solve reaches any truncation. The
# cap that remains is a cost decision, not an architectural one: each matrix-free
# application re-assembles the boundary mode forms at O(L^4), so the high-L cost
# question stays with the wall-free capacitance routine (chunk 2).
MAX_COUPLED_L = 10


def build_checkerboard_source(mesh, V, lmax):
    r"""Two-bump spherical-harmonic checkerboard density, on the space `V`.

    Density is `g_out(r) sum_lm a_l Y_lm + g_in(r) sum_lm b_l Y_lm`: a thin
    radial bump just inside the outer boundary carrying weights `a_l` that
    compensate the exterior `(r_out/RMAX)^(l+1)` falloff to the top boundary,
    plus a matching bump just outside the inner boundary with weights `b_l`
    compensating the interior `(r_in/RMIN)^l` falloff to the bottom boundary.
    Each bump only meaningfully reaches its own boundary, so the two weight sets
    flatten the two boundary spectra independently.

    Built by nodal numpy evaluation (the codebase idiom for mode fields) rather
    than a symbolic `(lmax+1)^2`-term UFL sum, which would blow up the volume
    quadrature estimate and the compiled kernel. Returns the density `Function`.
    """
    H = RMAX - RMIN
    r_out = RMAX - 0.12 * H          # bump centre just inside the outer boundary
    r_in = RMIN + 0.12 * H           # bump centre just outside the inner boundary
    width = 0.15 * H                 # a bit over one level-4 layer, resolvable

    # Node coordinates of V (CG1: dofs at the vertices), same local ordering as
    # the scalar dofs we write into.
    Vc = VectorFunctionSpace(mesh, "CG", 1)
    xyz = Function(Vc).interpolate(SpatialCoordinate(mesh)).dat.data_ro
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))   # colatitude in [0, pi]
    phi = np.arctan2(y, x)                          # longitude

    g_out = np.exp(-((r - r_out) / width) ** 2)
    g_in = np.exp(-((r - r_in) / width) ** 2)

    ang_out = np.zeros_like(r)
    ang_in = np.zeros_like(r)
    for l in range(lmax + 1):
        a_l = (RMAX / r_out) ** (l + 1)
        b_l = (r_in / RMIN) ** l
        for m in range(-l, l + 1):
            Y = real_spherical_harmonic_numpy(l, m, theta, phi)
            ang_out += a_l * Y
            ang_in += b_l * Y

    rho = Function(V, name="density")
    rho.dat.data[:] = g_out * ang_out + g_in * ang_in
    return rho


def solver_bundle(n, full_view=True):
    """PETSc options bundle: full solver view plus the fieldsplit_1 fixes.

    The sub-solver options sit under `dtn`, the options prefix of the
    `gadopt.DtNTwoBlockSchurPC` that now describes the two Schur blocks; at the
    top level they would no longer reach PETSc (and the solver says so).
    """
    bundle = {
        "dtn": {
            "fieldsplit_1": {
                # Non-restarted GMRES: PETSc's default restart of 30 is < n for
                # L >= 5 and would censor the very count the L axis measures.
                # This is the study instrumenting itself; the shipped preset
                # deliberately leaves the restart alone.
                "ksp_gmres_restart": n,
                # ~1.2 n: full GMRES terminates by n, this is that bound plus slack.
                "ksp_max_it": n + n // 5,
                "ksp_converged_reason": None,
            },
        },
    }
    if full_view:
        bundle.update({
            "ksp_view": None,                   # full PC tree once post-solve;
                                                # recurses into fieldsplit_0 GAMG
            "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
        })
        bundle["dtn"]["fieldsplit_0"] = {
            # one converged-reason line per A00 invocation (the AMG count).
            # No inner ksp_view: it would dump the whole GAMG hierarchy after
            # every one of the ~50 invocations; the outer ksp_view already
            # prints that hierarchy once.
            "ksp_converged_reason": None,
        }
    return bundle


def selfp_bundle(n):
    """Second fieldsplit_1 configuration: approximate-Schur selfp + Jacobi.

    May not compose with the matfree potential block (selfp needs the assembled
    A00 diagonal); run behind a flag and treated as best-effort.
    """
    return {
        "dtn": {
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_1": {
                "ksp_gmres_restart": n,
                "ksp_max_it": n + n // 5,
                "pc_type": "jacobi",
                "ksp_converged_reason": None,
            },
        },
    }


def attach_monitors(solver):
    """Per-invocation iteration collection off the fieldsplit sub-KSPs.

    Returns (outer_ksp, collectors) where collectors is a dict of lists holding
    the raw iteration-number stream of each fieldsplit sub-KSP; final
    per-invocation counts are recovered by detecting the reset to iteration 0.

    The fieldsplit is one level down from the outer PC: the preset selects
    `gadopt.DtNTwoBlockSchurPC`, whose Python context holds the PC that actually
    carries the two index-set splits. Asking the outer PC for its sub-KSPs
    instead raises, so reach through the context.

    This raises rather than degrading to empty streams. Every iteration count
    the study reports comes from here, and a harness that returns nothing while
    exiting successfully would publish a scaling verdict measured from no data.
    """
    streams = {"fieldsplit_0": [], "fieldsplit_1": []}
    outer = solver.solver.snes.getKSP()
    pc = outer.getPC()
    if pc.getType() == "python":
        pc = pc.getPythonContext().pc
    sub = pc.getFieldSplitSubKSP()

    def make(stream):
        def monitor(ksp, it, rnorm):
            stream.append((it, float(rnorm)))
        return monitor

    # Schur fieldsplit exposes (A00 KSP, Schur KSP).
    if len(sub) >= 1 and sub[0] is not None:
        sub[0].setMonitor(make(streams["fieldsplit_0"]))
    if len(sub) >= 2 and sub[1] is not None:
        sub[1].setMonitor(make(streams["fieldsplit_1"]))
    return outer, streams


def split_invocations(stream):
    """Split a monitor stream of (it, rnorm) pairs into per-invocation lists.

    A new invocation starts each time the iteration number resets to 0.
    """
    invocations, current = [], []
    for it, rnorm in stream:
        if it == 0 and current:
            invocations.append(current)
            current = []
        current.append((it, rnorm))
    if current:
        invocations.append(current)
    return invocations


def per_invocation_finals(stream):
    """Final iteration count of each solve invocation from a (it, rnorm) stream.

    The value just before each reset to 0 is that invocation's converged count.
    """
    return [inv[-1][0] for inv in split_invocations(stream)]


def degree_power(boundary_coeffs):
    """Per-degree boundary power from a {mode_key: coeff} mapping.

    Mode keys are "Yl,m"; power at degree l is the sum of squared coefficients
    over its orders.
    """
    power = defaultdict(float)
    for key, val in boundary_coeffs.items():
        l = int(key[1:].split(",")[0])
        power[l] += float(val) ** 2
    return {l: power[l] for l in sorted(power)}


def excitation_flags(power, floor_decades=2.0):
    """Flag degrees whose power falls more than `floor_decades` below the median."""
    vals = np.array([v for v in power.values() if v > 0.0])
    if vals.size == 0:
        return {l: False for l in power}
    threshold = np.median(vals) * 10.0 ** (-floor_decades)
    return {l: bool(power[l] >= threshold) for l in power}


def build_shell(ref_level):
    """Extruded cubed-sphere shell with the study's radii, plus its CG1 space."""
    nlayers = LAYERS[ref_level]
    mesh2d = CubedSphereMesh(RMIN, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial",
                        layer_height=(RMAX - RMIN) / nlayers)
    mesh.cartesian = False
    V = FunctionSpace(mesh, "CG", 1)
    return mesh, V, get_boundary_ids(mesh)


def correctness_anchor(ref_level, lmax, out_dir=None):
    """Run direct and iterative presets on the same case, compare the potential.

    The scaling numbers are only meaningful if the iterative preset solves the
    same system the direct one does. Solves both, returns the relative L2
    difference of the potential (which earlier unit tests confirm is machine-
    precision small), and writes it to a sidecar. Only runs at coupled-reachable
    L (both presets go through the fieldsplit).
    """
    if lmax > MAX_COUPLED_L:
        raise ValueError(f"anchor needs a coupled-reachable L<={MAX_COUPLED_L}")
    mesh, V, boundary = build_shell(ref_level)
    rho = build_checkerboard_source(mesh, V, lmax)
    dtn = SphericalDtN(lmax)
    bcs = {boundary.top: {"dtn": dtn}, boundary.bottom: {"dtn": dtn}}
    n = 2 * (lmax + 1) ** 2

    psis, coeffs = {}, {}
    for variant in ("direct", "iterative"):
        psi = Function(V, name=f"potential_{variant}")
        # Pass the restart bundle so the iterative Schur GMRES is non-restarted
        # (default restart 30 < n); harmless on the direct preset.
        solver = GravitySolver(
            psi, rho, bcs=bcs, solver_parameters=variant,
            solver_parameters_extra=solver_bundle(n, full_view=False))
        solver.solve()
        psis[variant] = psi
        coeffs[variant] = solver.coefficients()

    # Verdict on the potential (L2). All norms are equivalent on this fixed
    # space, so the choice cannot change the machine-precision verdict.
    diff = norm(psis["direct"] - psis["iterative"])
    scale = norm(psis["direct"])
    rel = float(diff / scale) if scale > 0 else float(diff)
    # The DtN-relevant part: the two presets differ in path only through the
    # fieldsplit_1 solve (iterative GMRES rtol 1e-6 vs MUMPS-exact), so the trace
    # coefficients are where a multiplier-side regression would first show.
    coeff_diff = max(
        (abs(coeffs["direct"][bc][k] - coeffs["iterative"][bc][k])
         for bc in coeffs["direct"] for k in coeffs["direct"][bc]),
        default=0.0)
    log(f"correctness anchor level={ref_level} L={lmax}: "
        f"||direct - iterative|| / ||direct|| = {rel:.3e}, "
        f"max coeff diff = {coeff_diff:.3e}")

    out_dir = Path(out_dir) if out_dir else Path.cwd()
    if mesh.comm.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = out_dir / f"anchor_level{ref_level}_lmax{lmax}.json"
        with open(sidecar, "w") as f:
            json.dump({"level": ref_level, "lmax": lmax,
                       "relative_difference": rel,
                       "max_coefficient_difference": coeff_diff}, f, indent=2)
        log(f"wrote {sidecar}")
    return rel


def model(ref_level, lmax, variant="iterative", selfp=False, probe=None,
          out_dir=None, full_view=True, timed=None):
    """Run one (level, L) case and write its JSON sidecar."""
    if lmax > MAX_COUPLED_L:
        raise ValueError(
            f"L={lmax} exceeds the coupled-solve limit L<={MAX_COUPLED_L}: the "
            f"mixed space would have {1 + 2 * (lmax + 1) ** 2} fields and "
            "PCFIELDSPLIT caps at 128. Use the capacitance routine for the "
            "high-L cost question instead of this coupled-solve model.")
    nlayers = LAYERS[ref_level]
    # Level 7 does a single timed solve (a 512-rank solve is expensive at any L);
    # the probe (a second full solve) defaults off there for the same reason.
    n_timed = timed if timed is not None else (1 if ref_level >= 7 else 3)
    if probe is None:
        probe = ref_level < 7

    setup_stage = PETSc.Log.Stage("gravity_setup")
    solve_stage = PETSc.Log.Stage("gravity_solve")

    mesh2d = CubedSphereMesh(RMIN, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial",
                        layer_height=(RMAX - RMIN) / nlayers)
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    psi = Function(V, name="potential")
    n = 2 * (lmax + 1) ** 2

    log(f"level {ref_level}, L {lmax}: potential DOFs {V.dim()}, "
        f"multipliers {n} (2 x (L+1)^2)")

    rho = build_checkerboard_source(mesh, V, lmax)

    extra = selfp_bundle(n) if selfp else solver_bundle(n, full_view=full_view)
    dtn = SphericalDtN(lmax)
    bcs = {boundary.top: {"dtn": dtn}, boundary.bottom: {"dtn": dtn}}

    out_dir = Path(out_dir) if out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "selfp" if selfp else variant

    def write_sidecar(summary):
        sidecar = out_dir / f"summary_level{ref_level}_lmax{lmax}_{tag}.json"
        if mesh.comm.rank == 0:
            with open(sidecar, "w") as f:
                json.dump(summary, f, indent=2)
            log(f"wrote {sidecar}")

    t0 = time.perf_counter()
    with setup_stage:
        solver = GravitySolver(
            psi, rho, bcs=bcs,
            solver_parameters=variant if not selfp else None,
            solver_parameters_extra=extra,
        )
    construct_time = time.perf_counter() - t0

    # Warm-up solve: triggers kernel compile and PC setup so the sub-KSPs exist.
    # The selfp variant may not compose with the matfree potential block; record
    # that as data rather than crashing before the sidecar is written.
    t0 = time.perf_counter()
    try:
        solver.solve()
    except Exception as exc:
        log(f"solve failed for variant {tag}: {exc}")
        write_sidecar({
            "level": ref_level, "lmax": lmax, "variant": tag,
            "n_multipliers": n, "failed": str(exc),
        })
        return None
    warmup_time = time.perf_counter() - t0
    outer, streams = attach_monitors(solver)

    outer_counts, wall_times = [], []
    f0_finals, f1_finals = [], []
    for _ in range(n_timed):
        solver.mixed_solution.assign(0)     # zero the initial guess
        for s in streams.values():
            s.clear()
        t0 = time.perf_counter()
        with solve_stage:
            solver.solve()
        wall_times.append(time.perf_counter() - t0)
        outer_counts.append(outer.getIterationNumber())
        f0_finals.extend(per_invocation_finals(streams["fieldsplit_0"]))
        f1_finals.extend(per_invocation_finals(streams["fieldsplit_1"]))

    # Boundary spectrum readback: the solved trace coefficients per mode.
    coeffs = solver.coefficients()
    spectrum, flags = {}, {}
    for bc_id, bc_coeffs in coeffs.items():
        power = degree_power(bc_coeffs)
        spectrum[str(bc_id)] = power
        flags[str(bc_id)] = excitation_flags(power)
        starved = [l for l, ok in flags[str(bc_id)].items() if not ok]
        log(f"boundary {bc_id}: per-degree power "
            + ", ".join(f"l{l}={p:.3e}" for l, p in power.items()))
        if starved:
            log(f"boundary {bc_id}: STARVED degrees (below floor): {starved}")

    f0_mean = float(np.mean(f0_finals)) if f0_finals else None
    f1_mean = float(np.mean(f1_finals)) if f1_finals else None
    f1_max = int(np.max(f1_finals)) if f1_finals else None
    outer_mean = float(np.mean(outer_counts)) if outer_counts else None

    log(f"outer FGMRES {outer_mean}, fieldsplit_0 mean/invocation {f0_mean}, "
        f"fieldsplit_1 mean {f1_mean} (max {f1_max}), "
        f"wall {np.mean(wall_times):.3f}s")

    # Optional probe solve: dump the fieldsplit_0 residual histories from one
    # solve, split by invocation, to a separate file. The python monitor already
    # captures (it, rnorm) cheaply, so this needs no PETSc-options replay (which
    # would be a no-op on an already-set-up sub-KSP).
    if probe:
        for s in streams.values():
            s.clear()
        solver.mixed_solution.assign(0)
        solver.solve()
        histories = split_invocations(streams["fieldsplit_0"])
        if mesh.comm.rank == 0:
            probe_file = out_dir / f"probe_level{ref_level}_lmax{lmax}_{tag}.json"
            with open(probe_file, "w") as f:
                json.dump({"fieldsplit_0_residual_histories": histories}, f)
            log(f"wrote probe {probe_file} ({len(histories)} invocations)")

    write_sidecar({
        "level": ref_level,
        "lmax": lmax,
        "variant": tag,
        "layers": nlayers,
        "potential_dofs": V.dim(),
        "n_multipliers": n,
        "n_timed": n_timed,
        "outer_iterations": outer_mean,
        "fieldsplit_0_iterations_per_invocation": f0_mean,
        "fieldsplit_1_iterations_mean": f1_mean,
        "fieldsplit_1_iterations_max": f1_max,
        "construct_time": construct_time,
        "warmup_time": warmup_time,
        "wall_times": wall_times,
        "boundary_degree_power": spectrum,
        "boundary_excitation_ok": flags,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", type=int, help="cubed-sphere refinement level")
    parser.add_argument("--lmax", type=int, default=5, help="DtN truncation L")
    parser.add_argument("--variant", choices=["iterative", "direct"],
                        default="iterative")
    parser.add_argument("--selfp", action="store_true",
                        help="use the selfp + Jacobi fieldsplit_1 variant")
    parser.add_argument("--probe", dest="probe", action="store_true", default=None)
    parser.add_argument("--no-probe", dest="probe", action="store_false")
    parser.add_argument("--no-full-view", dest="full_view", action="store_false",
                        help="skip the full ksp_view/monitor bundle (faster local runs)")
    parser.add_argument("--timed", type=int, default=None,
                        help="override the number of timed re-solves")
    parser.add_argument("--anchor", action="store_true",
                        help="run the direct-vs-iterative correctness anchor instead")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.anchor:
        correctness_anchor(args.level, args.lmax, out_dir=args.out_dir)
    else:
        model(args.level, args.lmax, variant=args.variant, selfp=args.selfp,
              probe=args.probe, out_dir=args.out_dir, full_view=args.full_view,
              timed=args.timed)
