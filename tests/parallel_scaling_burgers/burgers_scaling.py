"""Weak scaling of the Burgers internal-variable solvers on Gadi.

One job per (level, configuration). Each job runs the substituted reference
solver and one coupled configuration on the same mesh, so that every ratio in
the analysis is measured inside one process on one partition.

Levels follow ``tests/parallel_scaling``: the cubed-sphere refinement level is
the level, the radial layer count roughly doubles per level to keep the cell
aspect ratio fixed, and the core count grows by 8 per level so that the
degrees of freedom per core stay near constant. The coupled system carries
two DQ1 tensor internal variables (144 dofs per cell) next to the Q2
displacement (about 24 dofs per cell), so the core counts are four times
those of the Stokes scaling test.

Levels 1 to 3 exist for local smoke tests only.

Usage::

    python burgers_scaling.py LEVEL CONFIG [-n STEPS]

``CONFIG`` is one of ``solver_configs.COUPLED_CONFIGURATIONS``; the
substituted solver always runs as the reference.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3d_sphere_burgers"))

cases = {
    # Radial cells per shell: lithosphere (70 km), upper mantle (350 km),
    # transition zone (250 km), lower mantle (2221 km). Chosen so that the
    # radial spacing is close to the lateral spacing of the cubed sphere at
    # that level (313, 156 and 78 km at levels 5, 6 and 7).
    1: {"cores": 1, "layers": [1, 1, 1, 1]},
    2: {"cores": 4, "layers": [1, 1, 1, 2]},
    3: {"cores": 8, "layers": [1, 1, 1, 4]},
    5: {"cores": 104, "layers": [1, 1, 1, 7]},
    6: {"cores": 832, "layers": [1, 2, 2, 14]},
    7: {"cores": 6656, "layers": [1, 4, 3, 28]},
}

DT_YEARS = 1000.0


def _stage_events(profile_path):
    """Per-stage event table of a flat PETSc text log.

    Returns ``{stage name: {event name: (calls, max time over ranks)}}``.
    The stage sections of ``-log_view`` start with ``--- Event Stage N: name``.
    Events nest inside each other and each time is the maximum over ranks,
    so the numbers serve as regression values, not as a cost breakdown (use
    the nested XML log and ``plot_breakdown.py`` for that).
    """
    stages, current = {}, None
    with open(profile_path) as f:
        for line in f:
            if m := re.match(r"--- Event Stage \d+: (\S+)", line):
                current = stages.setdefault(m.group(1), {})
                continue
            parts = line.split()
            if current is not None and len(parts) > 4 and parts[1].isdigit():
                try:
                    current[parts[0]] = (int(parts[1]), float(parts[3]))
                except ValueError:
                    pass
    return stages


def get_data(level, config, base_path=None, tag=""):
    """Timing and iteration metrics of one (level, config) job.

    Reads ``level_L_CONFIG[_TAG]_full.out`` and ``profile_L_CONFIG[_TAG].txt``
    under ``base_path``. Needs no Firedrake. Returns a dict with, for ``name``
    in ``substituted`` and ``config``:

    - ``{name}_displacement_iterations``: mean GAMG V-cycles per step, and
      ``..._per_step`` the list. For ``multiplicative`` the KSP runs once per
      outer sweep, so the per-step value is the sum over the sweeps of that
      step and the outer sweep counts are in ``{name}_outer_iterations``.
    - ``{name}_solve``: wall time of the solver's PETSc stage.
    - ``{name}_pc_setup``: ``PCSetUp`` time inside that stage.
    - ``summary``: the JSON record the driver prints.
    """
    from solver_configs import displacement_ksp_prefix

    base_path = base_path or Path()
    suffix = f"_{tag}" if tag else ""
    output_path = base_path / f"level_{level}_{config}{suffix}_full.out"
    profile_path = base_path / f"profile_{level}_{config}{suffix}.txt"
    if not (output_path.exists() and profile_path.exists()):
        raise FileNotFoundError(f"outputs for level {level} {config} not found")

    prefixes = {
        "substituted": displacement_ksp_prefix("substituted"),
        config: displacement_ksp_prefix(config),
    }
    # Iterations accumulate until the driver's "step N name:" line closes the
    # step, which sums the inner solves of every outer sweep.
    pending = {name: 0 for name in prefixes}
    per_step = {name: [] for name in prefixes}
    outer = {name: [] for name in prefixes}
    data = {}
    with open(output_path) as f:
        for line in f:
            if m := re.match(
                r"\s+Linear (\S+) solve (?:converged|did not converge) due to (\S+) iterations (\d+)",
                line,
            ):
                for name, prefix in prefixes.items():
                    if m.group(1) == prefix:
                        pending[name] += int(m.group(3))
                    elif m.group(1) == prefix.split("_fieldsplit")[0].split("_condensed")[0] + "_":
                        outer[name].append(int(m.group(3)))
            elif m := re.match(r"^step (\d+) (\S+): ", line):
                name = m.group(2)
                per_step[name].append(pending[name])
                pending[name] = 0
            elif line.startswith("GADOPT_BURGERS_SCALING "):
                data["summary"] = json.loads(line.split(" ", 1)[1])

    for name in prefixes:
        counts = np.array(per_step[name])
        data[f"{name}_displacement_iterations"] = counts.mean() if counts.size else np.nan
        data[f"{name}_displacement_iterations_per_step"] = counts.tolist()
        data[f"{name}_outer_iterations"] = outer[name]

    stages = _stage_events(profile_path)
    for name in prefixes:
        events = stages.get(f"burgers_{name}_solve", {})
        data[f"{name}_solve"] = events.get("SNESSolve", (0, np.nan))[1]
        data[f"{name}_pc_setup"] = events.get("PCSetUp", (0, np.nan))[1]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="burgers_scaling", description="Run one Burgers weak-scaling job"
    )
    parser.add_argument("level", type=int)
    parser.add_argument("config")
    parser.add_argument("-n", "--steps", type=int, default=4, help="number of timesteps")
    args = parser.parse_args()

    # Import firedrake only when actually running, so that test collection
    # can import this module cheaply.
    from burgers_sphere import model

    model(args.level, args.config, cases[args.level]["layers"], DT_YEARS, args.steps)
