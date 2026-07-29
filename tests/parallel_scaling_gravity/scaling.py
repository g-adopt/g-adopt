r"""Layer-two integration: case grid and sidecar parser for the longtest harness.

Mirrors ``tests/parallel_scaling/scaling.py`` but reads the JSON sidecars the
model and capacitance routine write, rather than grepping PETSc logs. Importable
without firedrake (pytest collection and login-node use); firedrake is imported
only when a case is actually run from ``__main__``.
"""

import argparse
import json
from pathlib import Path

# Coupled weak-scaling ladder: one factor of eight per level, in both the mesh
# and the rank count.
cases = {
    4: {"cores": 1, "layers": 8},
    5: {"cores": 8, "layers": 16},
    6: {"cores": 64, "layers": 32},
    7: {"cores": 512, "layers": 64},
}

# The truncations must match `submit_gravity_scaling.COUPLED_L`. They are stated
# twice because this layer is importable without firedrake and the submit script
# is the bare-run generator, and they have already drifted apart once: this list
# sat at [2, 5, 6] with a comment attributing the ceiling to PETSc's 128-field
# limit long after that limit stopped binding, so the doit layer was quietly
# testing a third of the grid the campaign ran.
COUPLED_L = [2, 3, 5, 6, 7, 10]
REPRESENTATIONS = ["multiplier", "lowrank"]
CAPACITANCE_L = [2, 5, 10, 20, 30]


def _pick(base_path, pattern, level, lmax):
    """Newest sidecar matching the pattern; deterministic under duplicates."""
    matches = sorted(Path(base_path).rglob(pattern),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"no sidecar {pattern} for level {level}, L {lmax} under {base_path}")
    return matches[-1]


def get_coupled(level, lmax, base_path=None, representation="multiplier"):
    """Metrics for one coupled (level, L, representation) case from its sidecar.

    The glob is deliberately loose at the tail: the scaling study runs each case
    twice and suffixes the sidecar with its cache phase, while this layer's own
    runs pass no phase and produce the bare name. Matching both means a study
    results tree can be read by this layer instead of looking empty, and
    `_pick`'s newest-wins rule then selects the warm phase, which is the one
    whose iteration counts belong in a regression reference.

    The representation is matched explicitly rather than globbed over. Both paths
    write a sidecar for the same `(level, L)`, so a pattern that admitted either
    would hand back whichever one sorted last - and since the two differ by more
    than an order of magnitude in `amg_applications_per_solve`, a reference
    checked against the wrong one would fail in a way that looks like a
    regression in the solver.
    """
    base_path = Path(base_path) if base_path else Path()
    pattern = f"summary_level{level}_lmax{lmax}_iterative_{representation}*.json"
    with open(_pick(base_path, pattern, level, lmax)) as f:
        d = json.load(f)
    return {
        "amg_applications_per_solve": d.get("amg_applications_per_solve"),
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
    parser.add_argument("--representation", choices=REPRESENTATIONS,
                        default="multiplier")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.kind == "coupled":
        from gravity_cubed_sphere import model
        model(args.level, args.lmax, out_dir=args.out_dir,
              representation=args.representation)
    else:
        from capacitance_gravity import run
        run(args.level, args.lmax, out_dir=args.out_dir)
