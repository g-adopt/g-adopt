r"""Weak/stress-scaling model for the gravitational Poisson DtN solver.

One `(level, L)` case of the scaling study described in `SCALING-ANALYSIS.md`.
It builds an extruded cubed-sphere shell, drives it with a two-bump
spherical-harmonic checkerboard density (so every treated DtN mode on both
boundaries is genuinely excited), solves with the iterative `GravitySolver`
preset, and records into a JSON sidecar next to the log: the iteration counts,
the error against the closed-form potential, the per-degree boundary power
spectrum, the wall times split into their construction, first-solve and
steady-state parts, and the peak memory.

The error against the closed form is the load-bearing correctness number, not a
nicety. The source is band-limited to degree `L` and the DtN maps are exact for
every mode they treat, so `analytic_gravity` gives the exact solution of the
continuous problem this discretises, and the difference is discretisation error
alone. That makes one quantity serve as both the correctness check - which
matters because this solver has demonstrated that it can converge cleanly to a
wrong answer in parallel - and the measure of how high a truncation a given
mesh can actually carry.

The solver bundle here fixes the two things the study needs that the shipped
preset does not do on its own: the `fieldsplit_1` GMRES is run non-restarted
(`ksp_gmres_restart = n`, since PETSc's default restart of 30 would censor every
count above 30) and capped at `~1.2 n` (the guaranteed GMRES termination point
plus slack, so a pathological corner fails fast and diagnosably rather than on
walltime).

Kernel-generation cost is measured by running the same case twice against a
cache that is empty on the first invocation, labelled with `--cache-phase`.
Three regimes come out of it, and they are genuinely different costs: the cold
first solve pays C compilation and TSFC form compilation; the warm first solve
in a fresh process still pays the per-process symbolic work that no on-disk
cache can save; and a repeat solve in the same process pays neither.

Run locally against this worktree with, e.g.

    PYTHONPATH=<worktree> <firedrake-python> gravity_cubed_sphere.py 4 --lmax 5
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py import MPI

# PETSc (initialised when gadopt imports petsc4py) scans sys.argv and warns
# about our argparse flags as unknown options. Blank argv across the import so
# PETSc sees only the script name; argparse re-reads the real argv below.
_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from gadopt import *  # noqa: E402
sys.argv = _argv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_gravity import (  # noqa: E402
    analytic_potential, checkerboard_mode_sums,
)

# Same shell as the Stokes scaling model, so mesh resolution and aspect ratio
# are directly comparable.
RMIN, RMAX = 1.22, 2.22

# Isotropic refinement: the extrusion layer count doubles with the level in
# lockstep with the horizontal refinement, holding cell shape fixed.
LAYERS = {4: 8, 5: 16, 6: 32, 7: 64}

# PCFIELDSPLIT used to refuse more than 128 fields, and the mixed space carries
# 1 + 2(L+1)^2 of them, which capped this model at L = 6. The shipped preset now
# describes the two Schur blocks by index set (gadopt.DtNTwoBlockSchurPC), so the
# field enumeration never runs and the coupled solve reaches any truncation. What
# remains is cost, not architecture: each matrix-free application re-assembles
# the boundary mode forms at O(L^4) and the per-process symbolic form processing
# grows with the mode count too, which is what this study measures. The cap is
# kept only so that a mistyped truncation fails immediately instead of after an
# hour of form processing.
MAX_COUPLED_L = 20


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
    quadrature estimate and the compiled kernel.

    Every order of a degree carries the same weight, so the angular part is
    built from the per-degree sums `S_l = sum_m Y_lm`, which are returned
    alongside the density and reused verbatim by the closed-form reference. A
    reference that re-derived its own angular fields could disagree with the
    source through a normalisation or sign convention; sharing them makes that
    class of error impossible rather than unlikely.

    Returns:
      `(rho, params)` where `params` carries everything
      `analytic_gravity.analytic_potential` needs to evaluate the exact
      potential of this same density.
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

    weights_out = [(RMAX / r_out) ** (l + 1) for l in range(lmax + 1)]
    weights_in = [(r_in / RMIN) ** l for l in range(lmax + 1)]
    mode_sums = checkerboard_mode_sums(lmax, theta, phi)

    ang_out = np.zeros_like(r)
    ang_in = np.zeros_like(r)
    for l in range(lmax + 1):
        ang_out += weights_out[l] * mode_sums[l]
        ang_in += weights_in[l] * mode_sums[l]

    rho = Function(V, name="density")
    rho.dat.data[:] = g_out * ang_out + g_in * ang_in

    params = {
        "lmax": lmax, "rmin": RMIN, "rmax": RMAX, "r_out": r_out, "r_in": r_in,
        "width": width, "weights_out": weights_out, "weights_in": weights_in,
        "xyz": xyz, "mode_sums": mode_sums,
    }
    return rho, params


def analytic_error(V, psi, params):
    """Relative L2 and nodal errors of `psi` against the closed-form potential.

    The mean-removed error is reported beside the plain one so that a constant
    offset - the one thing a DtN nullspace could introduce - shows up as itself
    instead of inflating the error and inviting the wrong diagnosis.

    Args:
      V: The CG1 space `psi` lives on.
      psi: Computed potential.
      params: Source parameters from `build_checkerboard_source`.

    Returns:
      Dict of error measures.
    """
    exact = Function(V, name="potential_exact")
    exact.dat.data[:] = analytic_potential(
        params["xyz"], params["lmax"], params["rmin"], params["rmax"],
        params["r_out"], params["r_in"], params["width"],
        params["weights_out"], params["weights_in"],
        mode_sums=params["mode_sums"])

    scale = norm(exact)
    rel_l2 = float(norm(psi - exact) / scale) if scale > 0 else float(norm(psi - exact))

    # Nodal maximum, reduced across ranks: an L2 norm can hide a defect confined
    # to a few degrees of freedom, which is exactly what a boundary-condition or
    # index-set error would look like.
    local_max = float(np.max(np.abs(psi.dat.data_ro - exact.dat.data_ro))
                      if psi.dat.data_ro.size else 0.0)
    nodal_max = V.mesh().comm.allreduce(local_max, op=MPI.MAX)

    shifted = Function(V).assign(psi - exact)
    volume = assemble(Function(V).assign(1.0) * dx)
    offset = float(assemble(shifted * dx) / volume)
    shifted.dat.data[:] -= offset
    rel_l2_demeaned = float(norm(shifted) / scale) if scale > 0 else float(norm(shifted))

    return {
        "analytic_relative_l2": rel_l2,
        "analytic_relative_l2_demeaned": rel_l2_demeaned,
        "analytic_nodal_max": nodal_max,
        "analytic_mean_offset": offset,
        "analytic_norm": float(scale),
    }


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


def peak_memory_gb(comm):
    """Peak resident set size in GB, maximised over ranks.

    `ru_maxrss` is in kibibytes on Linux and in bytes on macOS, which is a
    silent factor of 1024 between the machine this is developed on and the one
    it runs on.
    """
    local = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    local_gb = local / 1024**2 if platform.system() == "Linux" else local / 1024**3
    return float(comm.allreduce(local_gb, op=MPI.MAX))


def cache_statistics():
    """Hit and miss counts of the PyOP2 and TSFC caches, per cache.

    Only populated when `PYOP2_CACHE_INFO=1` is set in the environment, which
    is what makes PyOP2 wrap its dict-like caches in the counting subclass. The
    counts are what distinguish a genuinely cold run from one that quietly
    found a populated cache directory, which is the whole basis of the
    cold-versus-warm cost split: without them, "cold" is an assertion about the
    filesystem rather than a measurement of what the process did.

    Returns:
      Dict keyed by `module.function [cache class]`, each a dict of hit, miss,
      size, and where the cache lives. Empty if instrumentation is off.
    """
    try:
        from pyop2.caching import cache_filter
    except ImportError:
        return {}

    stats = {}
    for entry in cache_filter():
        hit, miss, size, _ = entry.get_stats()
        if hit < 0 and miss < 0:
            continue                       # uninstrumented cache: no counters
        key = (f"{entry.func_module}.{entry.func_name} "
               f"[{entry.cache_name}]")
        record = stats.setdefault(
            key, {"hit": 0, "miss": 0, "size": 0, "location": str(entry.cache_loc)})
        record["hit"] += max(hit, 0)
        record["miss"] += max(miss, 0)
        record["size"] += max(size, 0)
    return stats


def cache_file_counts(comm):
    """Files sitting in the kernel cache directories, maximised over ranks.

    A second, independent reading of the same fact the hit/miss counters
    report, arrived at without going through PyOP2 at all: a genuinely cold
    phase starts with empty directories and a warm one does not. Worth the six
    lines, because every instrument in this study that was trusted without a
    second path to the same number has at some point turned out to be measuring
    nothing.
    """
    counts = {}
    for label, var in (("pyop2", "PYOP2_CACHE_DIR"),
                       ("tsfc", "FIREDRAKE_TSFC_KERNEL_CACHE_DIR"),
                       ("xdg", "XDG_CACHE_HOME")):
        path = os.environ.get(var)
        local = (sum(1 for p in Path(path).rglob("*") if p.is_file())
                 if path and Path(path).is_dir() else 0)
        counts[label] = int(comm.allreduce(local, op=MPI.MAX))
    return counts


def cache_totals(stats):
    """Aggregate hit and miss counts over the disk-backed caches only.

    Memory caches hit on the second and later lookups within a process
    regardless of what is on disk, so including them would blur exactly the
    distinction being measured. A disk-cache miss is a kernel that had to be
    generated.
    """
    hits = misses = 0
    for key, record in stats.items():
        if record["location"] == "Memory":
            continue
        hits += record["hit"]
        misses += record["miss"]
    return {"disk_cache_hits": hits, "disk_cache_misses": misses}


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
    rho, source_params = build_checkerboard_source(mesh, V, lmax)
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

    # Both presets against the closed form as well. Preset parity says the two
    # paths agree; only this says they agree on the right answer.
    against_exact = {
        variant: analytic_error(V, psis[variant], source_params)
        for variant in ("direct", "iterative")
    }
    for variant, err in against_exact.items():
        log(f"  {variant} vs closed form: relative L2 "
            f"{err['analytic_relative_l2']:.3e}")

    out_dir = Path(out_dir) if out_dir else Path.cwd()
    if mesh.comm.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = out_dir / f"anchor_level{ref_level}_lmax{lmax}.json"
        with open(sidecar, "w") as f:
            json.dump({"level": ref_level, "lmax": lmax,
                       "n_ranks": mesh.comm.size,
                       "relative_difference": rel,
                       "max_coefficient_difference": coeff_diff,
                       "against_closed_form": against_exact}, f, indent=2)
        log(f"wrote {sidecar}")
    return rel


def model(ref_level, lmax, variant="iterative", selfp=False, probe=None,
          out_dir=None, full_view=True, timed=None, cache_phase=None):
    """Run one (level, L) case and write its JSON sidecar.

    Args:
      ref_level: Cubed-sphere refinement level.
      lmax: DtN truncation.
      variant: `iterative` or `direct` preset.
      selfp: Use the approximate-Schur `selfp` fieldsplit_1 variant.
      probe: Dump per-invocation fieldsplit_0 residual histories.
      out_dir: Where the sidecar goes.
      full_view: Include the full `ksp_view` / monitor bundle.
      timed: Override the number of timed re-solves.
      cache_phase: `cold` or `warm`, recorded in the sidecar and its filename so
        the two invocations of one job do not overwrite each other.
    """
    if lmax > MAX_COUPLED_L:
        raise ValueError(
            f"L={lmax} exceeds this model's ceiling of L<={MAX_COUPLED_L}. The "
            "128-field PCFIELDSPLIT wall that used to cap it at 6 is gone "
            "(gadopt.DtNTwoBlockSchurPC describes the two Schur blocks by index "
            f"set), so {1 + 2 * (lmax + 1) ** 2} fields would set up fine; the "
            "ceiling is a cost guard, since both the O(L^4) boundary assembly "
            "and the per-process symbolic form processing grow with the mode "
            "count. Raise MAX_COUPLED_L deliberately if that is what you want.")
    nlayers = LAYERS[ref_level]
    # Two timed solves at level 7: the steady-state cost is what the
    # marginal-cost-per-mode reading rests on, so it wants more than one sample
    # even at 512 ranks. The probe (a further full solve) stays off there.
    n_timed = timed if timed is not None else (2 if ref_level >= 7 else 3)
    if probe is None:
        probe = ref_level < 7

    process_start = time.perf_counter()
    # Stages, so -log_view attributes cost to the phase that incurred it. The
    # first solve is its own stage because it is the only one that pays kernel
    # generation, and lumping it in with the timed solves is what would hide
    # the number this study exists to measure.
    setup_stage = PETSc.Log.Stage("gravity_setup")
    first_solve_stage = PETSc.Log.Stage("gravity_first_solve")
    solve_stage = PETSc.Log.Stage("gravity_solve")

    mesh2d = CubedSphereMesh(RMIN, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial",
                        layer_height=(RMAX - RMIN) / nlayers)
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)
    comm = mesh.comm

    V = FunctionSpace(mesh, "CG", 1)
    psi = Function(V, name="potential")
    n = 2 * (lmax + 1) ** 2

    log(f"level {ref_level}, L {lmax}: potential DOFs {V.dim()}, "
        f"multipliers {n} (2 x (L+1)^2), cache phase {cache_phase or 'unset'}")

    t0 = time.perf_counter()
    rho, source_params = build_checkerboard_source(mesh, V, lmax)
    source_time = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)

    extra = selfp_bundle(n) if selfp else solver_bundle(n, full_view=full_view)
    dtn = SphericalDtN(lmax)
    bcs = {boundary.top: {"dtn": dtn}, boundary.bottom: {"dtn": dtn}}

    out_dir = Path(out_dir) if out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "selfp" if selfp else variant
    phase_suffix = f"_{cache_phase}" if cache_phase else ""

    def write_sidecar(summary):
        sidecar = (out_dir
                   / f"summary_level{ref_level}_lmax{lmax}_{tag}{phase_suffix}.json")
        if comm.rank == 0:
            with open(sidecar, "w") as f:
                json.dump(summary, f, indent=2)
            log(f"wrote {sidecar}")

    def failure_record(exc):
        return {
            "level": ref_level, "lmax": lmax, "variant": tag,
            "cache_phase": cache_phase, "n_multipliers": n,
            "potential_dofs": V.dim(), "failed": str(exc),
            "peak_memory_gb": peak_memory_gb(comm),
        }

    t0 = time.perf_counter()
    with setup_stage:
        solver = GravitySolver(
            psi, rho, bcs=bcs,
            solver_parameters=variant if not selfp else None,
            solver_parameters_extra=extra,
        )
    # Max over ranks throughout: the redundant symbolic work is per-process, so
    # a mean would understate what the job actually waits for.
    construct_time = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)

    # First solve: the only one that pays kernel generation. On a cold cache
    # that is TSFC form compilation plus C compilation plus the per-process
    # symbolic work; on a warm cache the first two are cache hits and the
    # residue against a repeat solve is the part no cache can save.
    # The selfp variant may not compose with the matfree potential block; record
    # that as data rather than crashing before the sidecar is written.
    t0 = time.perf_counter()
    try:
        with first_solve_stage:
            solver.solve()
    except Exception as exc:
        log(f"solve failed for variant {tag}: {exc}")
        write_sidecar(failure_record(exc))
        return None
    warmup_time = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
    cache_after_first = cache_statistics()
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
        wall_times.append(comm.allreduce(time.perf_counter() - t0, op=MPI.MAX))
        outer_counts.append(outer.getIterationNumber())
        f0_finals.extend(per_invocation_finals(streams["fieldsplit_0"]))
        f1_finals.extend(per_invocation_finals(streams["fieldsplit_1"]))

    # Against the closed form. This is the correctness verdict for the case, and
    # since the source is band-limited to L and the DtN maps are exact for every
    # mode they treat, it is discretisation error with no truncation component -
    # so it also reads as how much of this truncation the mesh can resolve.
    errors = analytic_error(V, psi, source_params)
    log(f"analytic error level={ref_level} L={lmax}: relative L2 "
        f"{errors['analytic_relative_l2']:.3e}, nodal max "
        f"{errors['analytic_nodal_max']:.3e}, mean offset "
        f"{errors['analytic_mean_offset']:.3e}")

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
            probe_file = (out_dir
                          / f"probe_level{ref_level}_lmax{lmax}_{tag}{phase_suffix}.json")
            with open(probe_file, "w") as f:
                json.dump({"fieldsplit_0_residual_histories": histories}, f)
            log(f"wrote probe {probe_file} ({len(histories)} invocations)")

    steady_time = float(np.mean(wall_times)) if wall_times else None
    cache_stats = cache_statistics()
    summary = {
        "level": ref_level,
        "lmax": lmax,
        "variant": tag,
        "cache_phase": cache_phase,
        "layers": nlayers,
        "potential_dofs": V.dim(),
        "n_multipliers": n,
        "n_fields": 1 + n,
        "n_ranks": comm.size,
        "n_timed": n_timed,
        "outer_iterations": outer_mean,
        "fieldsplit_0_iterations_per_invocation": f0_mean,
        "fieldsplit_1_iterations_mean": f1_mean,
        "fieldsplit_1_iterations_max": f1_max,
        # Every time below is a max over ranks, in seconds.
        "source_time": source_time,
        "construct_time": construct_time,
        "warmup_time": warmup_time,
        "wall_times": wall_times,
        "steady_solve_time": steady_time,
        # The generation cost that no on-disk cache removes: what the first
        # solve pays over and above a repeat solve of the same operator.
        "first_solve_overhead": (warmup_time - steady_time
                                 if steady_time is not None else None),
        "total_process_time": comm.allreduce(
            time.perf_counter() - process_start, op=MPI.MAX),
        "peak_memory_gb": peak_memory_gb(comm),
        "boundary_degree_power": spectrum,
        "boundary_excitation_ok": flags,
        "cache_stats": cache_stats,
        "cache_stats_after_first_solve": cache_after_first,
        "cache_file_counts": cache_file_counts(comm),
    }
    summary.update(errors)
    summary.update(cache_totals(cache_stats))
    write_sidecar(summary)
    return summary


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
    parser.add_argument("--cache-phase", choices=["cold", "warm"], default=None,
                        help="label this invocation's kernel-cache state; the "
                             "study runs each case twice, cold then warm")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.anchor:
        correctness_anchor(args.level, args.lmax, out_dir=args.out_dir)
    else:
        model(args.level, args.lmax, variant=args.variant, selfp=args.selfp,
              probe=args.probe, out_dir=args.out_dir, full_view=args.full_view,
              timed=args.timed, cache_phase=args.cache_phase)
