"""Case definitions and output parser for the Richards parallel-scaling
long-test suite.

Three weak-scaling case families exercise the iterative presets of
RichardsSolver:

    cockett         - synthetic 3D heterogeneous infiltration, extruded
                      quads on a box. Isotropic cells. No external
                      dependencies.
    murr_vertical   - basin floodplain, fixed horizontal resolution
                      (1775 m), vertical layers scale with node count.
                      The aspect ratio climbs from 500:1 to 4000:1, so
                      this is the case that tests whether the lumped
                      presets stay independent of vertical resolution.
    murr_seasonal   - same basin, fixed 300 vertical layers, horizontal
                      resolution halves with node count, integrated with
                      three-month time steps on a near-saturated column.

Each level keys a node count on Gadi's Sapphire Rapids ``normalsr``
queue (104 CPUs per node). ``CASE_SOLVERS`` records which presets each
case exercises, and why.

Both Murrumbidgee cases need ``omega`` for the mesh, and the terrain
bundle that doit's ``fetch_data`` task downloads. omega is needed only
to run these cases, so it is not a dependency of g-adopt itself.

``get_data`` merges three sources per run:

    stdout                 mean linear iterations per Newton step.
    ``-log_view`` profile  PCSetUp, PCApply, KSPSolve and SNESSolve wall
                           times with their load-imbalance ratios, plus
                           the run total.
    ``params_<tag>.log``   total water content at the end of the run,
                           with the step and failed-step counts that say
                           whether two runs followed the same dt path.

The parser is deliberately forgiving: missing files raise
``FileNotFoundError`` (skip the assertion in the consumer) and missing
metrics produce ``nan``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Hardware assumptions (Gadi normalsr)
# -----------------------------------------------------------------------------
# Sapphire Rapids nodes on the normalsr queue expose 104 usable cores each.
CPUS_PER_NODE = 104


# -----------------------------------------------------------------------------
# Case families
# -----------------------------------------------------------------------------
# Keys are node counts on the target system. Values are the per-case
# parameters consumed by the benchmark driver.

# Cockett et al. (2018) 3D heterogeneous infiltration, weak-scaled at
# ~18M DOFs per node with DQ1 on an extruded quad mesh.
COCKETT_CASES: dict[int, dict[str, Any]] = {
    # Horizontal cell counts are picked divisible by 8 so vlumping_hmg's
    # coarse 2D mesh (nx / 2**hmg_levels) is a whole number of cells for
    # hmg_levels up to 3. Keeps ~18M DOFs per node at DQ1.
    1: {"nx": 120, "nz": 156, "steps": 30},
    2: {"nx": 152, "nz": 196, "steps": 30},
    4: {"nx": 192, "nz": 247, "steps": 30},
    8: {"nx": 240, "nz": 312, "steps": 30},
}

# Lower Murrumbidgee, vertical weak scaling: horizontal resolution fixed
# at 1775 m (paper resolution, ~22k triangles); vertical layers scale
# with node count (150 per node). Aspect ratio grows with scale.
MURR_VERTICAL_CASES: dict[int, dict[str, Any]] = {
    1: {"horiz_res": 1775, "layers": 150},
    2: {"horiz_res": 1775, "layers": 300},
    4: {"horiz_res": 1775, "layers": 600},
    8: {"horiz_res": 1775, "layers": 1200},
}

# Lower Murrumbidgee, seasonal regime. Same horizontal weak-scaling ladder as
# the ordinary basin case (fixed 300 layers, resolution halves with node
# count), but integrated with three-month time steps on a near-saturated
# basin.
#
# This is the regime that separates the solvers. The discriminant is the
# column-integrated diffusion number
#
#     D_col = dt * T / (S_col * L^2),   T = int K dz,  S_col = int (Ss S + C) dz
#
# which is the conditioning of the vertically collapsed 2-D operator that a
# single-level preconditioner has no coarse correction for. It is NOT the mesh
# aspect ratio: the ordinary-regime runs already carry extreme aspect ratio and
# block-Jacobi is fastest there. Raising dt to three months and flattening the
# retention curve drives D_col up until block-Jacobi's admissible time step
# collapses as 1/L^2, while the direct-coarse lumped presets keep taking full
# steps at a low, flat iteration count.
#
# The soil levers (see the driver's --watertable-offset / --retention-flatten)
# are applied per level by meta.py, not stored here.
MURR_SEASONAL_CASES: dict[int, dict[str, Any]] = {
    1: {"horiz_res": 1775, "layers": 300},
    2: {"horiz_res": 1250, "layers": 300},
    4: {"horiz_res": 880, "layers": 300},
    8: {"horiz_res": 620, "layers": 300},
}

CASES: dict[str, dict[int, dict[str, Any]]] = {
    "cockett": COCKETT_CASES,
    "murr_vertical": MURR_VERTICAL_CASES,
    "murr_seasonal": MURR_SEASONAL_CASES,
}

# -----------------------------------------------------------------------------
# Seasonal regime parameters
# -----------------------------------------------------------------------------
# dt ceiling of three months (3 x 31 days) with a ramp from 60 s, a raised
# water table, and a retention curve flattened by 3. Ss stays 0: the
# Ss*S*Dt(h) mass term is not needed for the mechanism and BackwardEuler stays
# well posed without it, because the SIPG diffusion and the Robin side
# condition anchor the saturated cells.
SEASONAL_PARAMETERS: dict[str, float] = {
    "dt_init": 60.0,
    "dt_max": 8_035_200.0,
    "dt_growth": 1.5,
    "dt_shrink": 0.5,
    "t_final": 40_000_000.0,
    "watertable_offset": 5.0,
    "retention_flatten": 3.0,
    "ss": 0.0,
}


# -----------------------------------------------------------------------------
# Solver scope
# -----------------------------------------------------------------------------
# Every iterative preset exercised anywhere in the suite. The ``direct``
# preset is excluded because it does not weak-scale. ``bjacobi`` is not a
# RichardsSolver preset: it is the single-level baseline defined inside
# cockett_3d.py, kept here so the matrix records what was measured.
SOLVERS: tuple[str, ...] = (
    "iterative", "bjacobi", "vlumping", "vlumping_linesmooth", "vlumping_hmg",
)

# Per-case solver lists.
#
# Cockett takes everything: it is the isotropic case, the only one where all
# five complete, and therefore the only place a baseline can carry a numeric
# assertion.
#
# The Murrumbidgee families drop the two single-level solvers for opposite
# reasons. BoomerAMG (``iterative``) diverges within a few time steps on the
# basin operator, because Hypre's coarsening cannot cope with the
# horizontal/vertical anisotropy. ``bjacobi`` survives the ordinary regime but
# cannot take a step in the seasonal one. Neither produces an iteration count
# to regress against, so benchmarking them there would only record a failure.
#
# ``vlumping_hmg`` is excluded from the seasonal case specifically: its
# iterative coarse solve thrashes at long time steps, where it needs several
# times the step count of the direct-coarse presets to reach t_final. That is
# a real property worth knowing, but it is not a stable regression baseline.
# Cockett and murr_vertical cover the preset instead.
_MURR_VERTICAL_SOLVERS: tuple[str, ...] = (
    "vlumping", "vlumping_linesmooth", "vlumping_hmg",
)
_MURR_SEASONAL_SOLVERS: tuple[str, ...] = ("vlumping", "vlumping_linesmooth")
CASE_SOLVERS: dict[str, tuple[str, ...]] = {
    "cockett": SOLVERS,
    "murr_vertical": _MURR_VERTICAL_SOLVERS,
    "murr_seasonal": _MURR_SEASONAL_SOLVERS,
}
assert set(CASE_SOLVERS) == set(CASES), \
    "CASE_SOLVERS must cover every case in CASES"


def all_triples() -> Iterator[tuple[str, int, str]]:
    """Yield ``(case, level, solver)`` tuples honouring CASE_SOLVERS.

    Ordering matches meta.py's step enumeration (case, then level, then solver);
    keep the loops in that order so pytest parametrisation IDs stay stable.
    """
    for case, levels in CASES.items():
        for level in levels:
            for solver in CASE_SOLVERS[case]:
                yield case, level, solver


# -----------------------------------------------------------------------------
# Output parsing
# -----------------------------------------------------------------------------
# Richards emits anonymous KSP solves (no named prefix), so the token
# between "Linear" and "solve" is optional. This also matches prefixed
# logs if someone wires RichardsSolver into a named outer solve later.
_LINEAR_ITERATIONS_RE = re.compile(
    r"\s+Linear(?:\s+\S+)? solve converged due to \S+ iterations (\d+)"
)


def _parse_linear_iterations(path: Path) -> float:
    """Return the mean linear-iteration count across all time steps.

    Richards solves one nonlinear system per step; each Newton iteration
    runs one KSP solve. We average over every Krylov convergence line
    PETSc emits with ``ksp_converged_reason`` enabled.
    """
    iters: list[int] = []
    with path.open() as f:
        for line in f:
            # search (not match) so we're robust to minor PETSc version
            # differences in leading whitespace.
            if m := _LINEAR_ITERATIONS_RE.search(line):
                iters.append(int(m.group(1)))
    return float(np.mean(iters)) if iters else float("nan")


#: PETSc ``-log_view`` events recorded for every run, mapped to the column
#: name used in the reference CSV. PCSetUp and PCApply split preconditioner
#: cost into the part paid once per Jacobian and the part paid per Krylov
#: iteration, which is the trade the lumped presets are making; KSPSolve and
#: SNESSolve bracket them so a change can be read as a fraction of the whole.
_PROFILE_EVENTS: dict[str, str] = {
    "PCSetUp": "pc_setup",
    "PCApply": "pc_apply",
    "KSPSolve": "ksp_solve",
    "SNESSolve": "snes_solve",
}


def _parse_log_view_profile(path: Path) -> dict[str, float]:
    """Extract per-event and total wall times from a PETSc ``-log_view``.

    The profile is written by the ``run.template`` wrapper to
    ``profile_<case>_<solver>_<level>.txt``. PETSc's log_view uses the
    column layout::

        Event        Count Count-ratio  Time-max  Time-ratio  ...
        PCSetUp        N    1.0        <pc-time>  1.0        ...

    so the wall time of interest sits at index 3 after ``line.split()``
    and the load-imbalance ratio, max over min across ranks, at index 4.
    The aggregate run time comes from the summary block at the top::

        Time (sec):  <max>  <ratio>  <avg>  ...

    where the mean is index 4.

    Event names are matched with a trailing space, so ``PCSetUp`` does not
    also swallow the ``PCSetUpOnBlocks`` and ``PCSetUp_GAMG+`` entries that
    share its prefix. Only the first occurrence of each event is taken:
    log_view repeats the table per stage, and the main stage comes first.

    Returns:
        A dict carrying one entry per name in ``_PROFILE_EVENTS``, the
        matching ``<name>_ratio`` load imbalance, and ``solve_time``. Any
        value is ``nan`` if log_view did not record it.
    """
    data: dict[str, float] = {"solve_time": float("nan")}
    for column in _PROFILE_EVENTS.values():
        data[column] = float("nan")
        data[f"{column}_ratio"] = float("nan")

    with path.open() as f:
        for line in f:
            for event, column in _PROFILE_EVENTS.items():
                if not np.isnan(data[column]) or not line.startswith(event + " "):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    data[column] = float(parts[3])
                    data[f"{column}_ratio"] = float(parts[4])
            if np.isnan(data["solve_time"]) and line.startswith("Time (sec):"):
                parts = line.split()
                if len(parts) >= 5:
                    data["solve_time"] = float(parts[4])
    return data


def _parse_params_log(path: Path) -> dict[str, float]:
    """Extract the physical diagnostics from a driver's ``ParameterLog``.

    The drivers write one whitespace-separated row per accepted timestep,
    under a header naming the columns. ``theta_total``, the total water
    content in m^3, is the suite's only check that a solve reached the
    right answer rather than merely converging: iteration counts and
    timings are both blind to a preconditioner that converges to the
    wrong solution.

    The final row is the one that matters, since it carries the end state
    the whole trajectory produced. ``steps`` and ``failed`` come back
    alongside it because the basin cases adapt their timestep: a run that
    failed a different number of steps followed a different dt trajectory
    and so did not integrate the same problem, which makes its
    ``theta_total`` and its timings incomparable rather than merely
    different.

    Returns:
        ``{"theta_total": ..., "steps": ..., "failed": ...}``, each
        ``nan`` when the log recorded no completed step.
    """
    data = {"theta_total": float("nan"),
            "steps": float("nan"),
            "failed": float("nan")}

    with path.open() as f:
        lines = [ln.split() for ln in f if ln.strip()]
    if len(lines) < 2:
        return data

    header, last = lines[0], lines[-1]
    row = dict(zip(header, last))
    data["theta_total"] = float(row.get("theta_total", "nan"))
    data["steps"] = float(row.get("step", "nan"))
    # Only the basin driver adapts dt, so only it records failed steps.
    data["failed"] = float(row.get("failed", 0.0))
    return data


def get_data(
    case: str,
    solver: str,
    level: int,
    base_path: Path | None = None,
) -> dict[str, float]:
    """Return the iteration/timing metrics for a (case, solver, level) run.

    Args:
        case: A key of ``CASES``.
        solver: A preset name listed in ``CASE_SOLVERS[case]``.
        level: Node count key from the matching case dict.
        base_path: Directory containing ``<case>_<solver>_<level>.out`` and
            ``profile_<case>_<solver>_<level>.txt`` (matches the filenames
            emitted by ``run.template``). Defaults to the caller's CWD.

    Returns:
        A dict of every metric named in the module docstring. Any metric
        whose source file is absent, or which that file did not record,
        is ``nan``.

    Raises:
        FileNotFoundError: None of the three per-run outputs can be
            located, which means the case did not run at all.
    """
    base_path = Path(base_path) if base_path is not None else Path()
    tag = f"{case}_{solver}_{level}"
    out_path = base_path / f"{tag}.out"
    profile_path = base_path / f"profile_{tag}.txt"
    params_path = base_path / f"params_{tag}.log"

    if not any(p.exists() for p in (out_path, profile_path, params_path)):
        raise FileNotFoundError(
            f"no outputs for case={case}, solver={solver}, level={level} "
            f"under {base_path}"
        )

    data: dict[str, float] = {
        "linear_iterations": float("nan"),
        "solve_time": float("nan"),
        "theta_total": float("nan"),
        "steps": float("nan"),
        "failed": float("nan"),
    }
    for column in _PROFILE_EVENTS.values():
        data[column] = float("nan")
        data[f"{column}_ratio"] = float("nan")

    if out_path.exists():
        data["linear_iterations"] = _parse_linear_iterations(out_path)
    if profile_path.exists():
        data.update(_parse_log_view_profile(profile_path))
    if params_path.exists():
        data.update(_parse_params_log(params_path))
    return data
