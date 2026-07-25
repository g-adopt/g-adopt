r"""Layer-two integration: case grid and sidecar parser for the longtest harness.

Mirrors ``tests/parallel_scaling/scaling.py`` but reads the JSON sidecars the
model and capacitance routine write, rather than grepping PETSc logs. Importable
without firedrake (pytest collection and login-node use); firedrake is imported
only when a case is actually run from ``__main__``.
"""

import argparse
import json
from pathlib import Path

# Coupled weak-scaling ladder (the 128-field wall caps L at 6).
cases = {
    4: {"cores": 1, "layers": 8},
    5: {"cores": 8, "layers": 16},
    6: {"cores": 64, "layers": 32},
    7: {"cores": 512, "layers": 64},
}

COUPLED_L = [2, 5, 6]
CAPACITANCE_L = [2, 5, 10, 20, 30]


def _pick(base_path, pattern, level, lmax):
    """Newest sidecar matching the pattern; deterministic under duplicates."""
    matches = sorted(Path(base_path).rglob(pattern),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"no sidecar {pattern} for level {level}, L {lmax} under {base_path}")
    return matches[-1]


def get_coupled(level, lmax, base_path=None):
    """Metrics for one coupled (level, L) case from its sidecar."""
    base_path = Path(base_path) if base_path else Path()
    with open(_pick(base_path, f"summary_level{level}_lmax{lmax}_iterative.json",
                    level, lmax)) as f:
        d = json.load(f)
    return {
        "fieldsplit_0_iterations": d["fieldsplit_0_iterations_per_invocation"],
        "fieldsplit_1_iterations": d["fieldsplit_1_iterations_mean"],
        "outer_iterations": d["outer_iterations"],
    }


def get_capacitance(level, lmax, base_path=None):
    """Metrics for one capacitance (level, L) case from its sidecar."""
    base_path = Path(base_path) if base_path else Path()
    with open(_pick(base_path, f"capacitance_level{level}_lmax{lmax}.json",
                    level, lmax)) as f:
        d = json.load(f)
    return {
        "gmres_flat": d["gmres_flat"],
        "cond_S": d["cond_S"],
        "block_offdiag_weight": d["block_offdiag_weight"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scaling")
    parser.add_argument("level", type=int)
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--kind", choices=["coupled", "capacitance"],
                        default="coupled")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.kind == "coupled":
        from gravity_cubed_sphere import model
        model(args.level, args.lmax, out_dir=args.out_dir)
    else:
        from capacitance_gravity import run
        run(args.level, args.lmax, out_dir=args.out_dir)
