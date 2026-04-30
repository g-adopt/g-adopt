"""Case definitions and output parser for the Richards parallel-scaling
long-test suite.

Three weak-scaling case families exercise the g-adopt default iterative
presets for RichardsSolver:

    cockett           - synthetic 3D heterogeneous infiltration, extruded
                        quads on a box. No external dependencies.
    murr_vertical     - Lower Murrumbidgee floodplain, fixed horizontal
                        resolution (1775 m), vertical layers scale with
                        node count. Requires ``omega`` and the CSV data
                        bundle under ``murrumbidgee_data/``.
    murr_horizontal   - Lower Murrumbidgee floodplain, fixed 300 vertical
                        layers, horizontal resolution halves with node
                        count (paper's headline scaling figure).

Each level keys a node count on Gadi's Sapphire Rapids ``normalsr`` queue
(104 CPUs per node). The three iterative presets exercised at every level
are ``"iterative"`` (Hypre BoomerAMG), ``"vlumping"`` (vertically lumped
2-level MG) and ``"vlumping_hmg"`` (vlumping + geometric MG on the 2D
base hierarchy).

The ``get_data`` parser extracts mean linear-iteration count from the
driver's stdout and ``PCSetUp`` / total-wall-time from a PETSc
``-log_view`` profile written alongside it. The parser is deliberately
forgiving: missing files raise ``FileNotFoundError`` (skip the assertion
in the consumer) and missing metrics produce ``nan``.
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

# Lower Murrumbidgee, horizontal weak scaling: fixed 300 layers; horizontal
# resolution halves with node count, matching Morrow et al. (2026).
MURR_HORIZONTAL_CASES: dict[int, dict[str, Any]] = {
    1: {"horiz_res": 1775, "layers": 300},
    2: {"horiz_res": 1250, "layers": 300},
    4: {"horiz_res": 880, "layers": 300},
    8: {"horiz_res": 620, "layers": 300},
}

CASES: dict[str, dict[int, dict[str, Any]]] = {
    "cockett": COCKETT_CASES,
    "murr_vertical": MURR_VERTICAL_CASES,
    "murr_horizontal": MURR_HORIZONTAL_CASES,
}


# -----------------------------------------------------------------------------
# Solver scope
# -----------------------------------------------------------------------------
# All iterative string presets exposed by RichardsSolver. The direct preset
# is excluded because it does not weak-scale.
SOLVERS: tuple[str, ...] = ("iterative", "vlumping", "vlumping_hmg")

# Per-case solver lists. BoomerAMG (``iterative``) is excluded from the two
# Murrumbidgee families: the operator's strong horizontal/vertical anisotropy
# makes Hypre's coarsening diverge within a few time steps (see Morrow et al.
# 2026), so benchmarking it there would just record a failure. The point of
# the ``vlumping`` presets is precisely that they stay robust in that regime.
_MURR_SOLVERS: tuple[str, ...] = ("vlumping", "vlumping_hmg")
CASE_SOLVERS: dict[str, tuple[str, ...]] = {
    "cockett": SOLVERS,
    "murr_vertical": _MURR_SOLVERS,
    "murr_horizontal": _MURR_SOLVERS,
}
assert set(CASE_SOLVERS) == set(CASES), \
    "CASE_SOLVERS must cover every case in CASES"


def all_triples() -> Iterator[tuple[str, int, str]]:
    """Yield ``(case, level, solver)`` tuples honouring CASE_SOLVERS.

    Ordering matches meta.py's step enumeration (case → level → solver);
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


def _parse_log_view_profile(path: Path) -> dict[str, float]:
    """Extract PCSetUp time and total wall time from a PETSc ``-log_view``.

    The profile is written by the ``run.template`` wrapper to
    ``profile_<case>_<solver>_<level>.txt``. PETSc's log_view uses the
    column layout::

        Event        Count Count-ratio  Time-max  Time-ratio  ...
        PCSetUp        N    1.0        <pc-time>  1.0        ...

    so the wall-time of interest sits at index 3 after ``line.split()``.
    The aggregate run time comes from the summary block at the top::

        Time (sec):  <max>  <ratio>  <avg>  ...

    where the mean is index 4.

    Returns ``{"pc_setup": ..., "solve_time": ...}``; either value may
    be ``nan`` if log_view did not record the event.
    """
    data: dict[str, float] = {"pc_setup": float("nan"), "solve_time": float("nan")}
    with path.open() as f:
        for line in f:
            # Match "PCSetUp " (note the trailing space) to avoid the
            # PCSetUpOnBlocks / PCSetUp_GAMG+ follow-up entries that start
            # with the same prefix.
            if np.isnan(data["pc_setup"]) and line.startswith("PCSetUp "):
                parts = line.split()
                if len(parts) >= 4:
                    data["pc_setup"] = float(parts[3])
            if np.isnan(data["solve_time"]) and line.startswith("Time (sec):"):
                parts = line.split()
                if len(parts) >= 5:
                    data["solve_time"] = float(parts[4])
    return data


def get_data(
    case: str,
    solver: str,
    level: int,
    base_path: Path | None = None,
) -> dict[str, float]:
    """Return the iteration/timing metrics for a (case, solver, level) run.

    Args:
        case: ``"cockett"``, ``"murr_vertical"`` or ``"murr_horizontal"``.
        solver: ``"iterative"``, ``"vlumping"`` or ``"vlumping_hmg"``.
        level: Node count key from the matching case dict.
        base_path: Directory containing ``<case>_<solver>_<level>.out`` and
            ``profile_<case>_<solver>_<level>.txt`` (matches the filenames
            emitted by ``run.template``). Defaults to the caller's CWD.

    Returns:
        ``{"linear_iterations": ..., "pc_setup": ..., "solve_time": ...}``.
        Any metric absent from the output files is ``nan``.

    Raises:
        FileNotFoundError: Neither the stdout nor the log_view profile
            can be located for this run.
    """
    base_path = Path(base_path) if base_path is not None else Path()
    out_path = base_path / f"{case}_{solver}_{level}.out"
    profile_path = base_path / f"profile_{case}_{solver}_{level}.txt"

    if not out_path.exists() and not profile_path.exists():
        raise FileNotFoundError(
            f"no outputs for case={case}, solver={solver}, level={level} "
            f"under {base_path}"
        )

    data: dict[str, float] = {
        "linear_iterations": float("nan"),
        "pc_setup": float("nan"),
        "solve_time": float("nan"),
    }
    if out_path.exists():
        data["linear_iterations"] = _parse_linear_iterations(out_path)
    if profile_path.exists():
        data.update(_parse_log_view_profile(profile_path))
    return data
