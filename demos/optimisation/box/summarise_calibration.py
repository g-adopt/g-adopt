"""Read the radius calibration and name the best initial radius per optimiser.

The sweep in ``calibrate_radius.sh`` runs every optimiser at several initial
trust-region radii, for both inner products, on a coarse mesh. This script reads the
result and names the radius to use, and can write it as a shell file that
``run_campaign.sh`` reads.

The measure is the initial misfit reached at a shared amount of forward-equivalent
work, so a radius that reaches a lower misfit by spending many more Hessian-vector
products does not win. Work is forward replays plus adjoint sweeps plus the measured
cost of a Hessian-vector product times their number, and the cost comes from the run
itself.

The radius of the TAO trust-region types does not transfer between meshes. Their
truncated conjugate-gradient solve measures the step in the plain Euclidean norm of
the nodal values, which is ``1/h`` times the L2 norm of the same field, so the same
physical step has twice the norm on a mesh with twice the cells per side. Those types
are scaled by ``--scale-tao``, which must be the ratio of the campaign resolution to
the calibration resolution. ROL and the TAO quasi-Newton types measure the step in the
control inner product, which does not depend on the mesh, and are left alone.

Usage:
    python summarise_calibration.py [--runs ~/runs/radius-calibration]
                                    [--scale-tao 2] [--emit radii.sh]
"""

import argparse
import json
from pathlib import Path

# Types whose trust-region radius is measured in the plain Euclidean norm of the nodal
# values, through the truncated conjugate-gradient solve of PETSc (``stcg.c``, with
# ``bnk.c`` taking the radius from it). The line-search types use the radius only as a
# bound on the first step, so they are scaled as well; the quasi-Newton types that
# never build a Krylov space on the true Hessian are not.
MESH_DEPENDENT_RADIUS = {
    "tao-bntr", "tao-bntr-noprec", "tao-bntr-riesz", "tao-bqnktr",
    "tao-bnls", "tao-bnls-riesz", "tao-bqnkls",
}

# The same fallback as ``plot_campaign.DEFAULT_HESSIAN_RATIO``.
DEFAULT_HESSIAN_RATIO = 1.96


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", default=str(Path.home() / "runs/radius-calibration"))
    parser.add_argument("--budget", type=float, default=None,
                        help="compare the misfit at this forward-equivalent work; the "
                             "default is the smallest final work over the radii of one "
                             "optimiser, so every radius is judged at a work that all "
                             "of them reached")
    parser.add_argument("--scale-tao", type=float, default=1.0,
                        help="multiply the radius of the mesh-dependent TAO types by "
                             "this factor, which is the ratio of the campaign "
                             "resolution to the calibration resolution")
    parser.add_argument("--emit", default=None,
                        help="write the chosen radii as a shell file for run_campaign.sh")
    return parser.parse_args(argv)


def work_series(record):
    """Forward-equivalent work and initial misfit at every iteration of one run.

    Args:
        record: one run dictionary.

    Returns:
        list of ``(work, initial_misfit)``.
    """
    ratio = (record.get("call_cost") or {}).get("hessian_per_forward") or DEFAULT_HESSIAN_RATIO
    out = []
    for row in record["iterations"]:
        if "initial_misfit" not in row:
            continue
        work = row.get("work") or (row["forward"] + row["adjoint"] + ratio * row["hessian"])
        out.append((work, row["initial_misfit"]))
    return out


def misfit_at(series, budget):
    """Initial misfit at the last iteration whose work is within the budget."""
    value = None
    for work, misfit in series:
        if work <= budget:
            value = misfit
    return value


def shell_name(optimiser, metric):
    """Variable name for one ``(optimiser, inner product)`` pair."""
    return "RADIUS_" + optimiser.replace("-", "_") + "_" + metric


def load(root):
    """Read every calibration run below ``root``.

    Directory names are ``<optimiser>_<metric>_r<radius>``.

    Args:
        root: the calibration directory.

    Returns:
        dict keyed by ``(optimiser, metric)``, each a dict of radius to record.
    """
    runs = {}
    for path in sorted(root.glob("*/run.json")):
        with open(path) as f:
            record = json.load(f)
        if not record.get("iterations"):
            continue
        head, _, radius = path.parent.name.rpartition("_r")
        optimiser, _, metric = head.rpartition("_")
        runs.setdefault((optimiser, metric), {})[float(radius)] = record
    return runs


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.runs).expanduser()
    runs = load(root)
    if not runs:
        raise SystemExit(f"no calibration runs in {root}")

    print(f"{'optimiser':18s} {'metric':15s} {'radius':>8s} {'work':>8s} "
          f"{'IC at budget':>14s} {'IC final':>12s}")
    print("-" * 82)
    best = {}
    for (optimiser, metric), by_radius in sorted(runs.items()):
        series = {r: work_series(rec) for r, rec in by_radius.items()}
        series = {r: s for r, s in series.items() if s}
        if not series:
            continue
        budget = args.budget or min(s[-1][0] for s in series.values())
        scored = []
        for radius in sorted(series):
            value = misfit_at(series[radius], budget)
            if value is not None:
                scored.append((value, radius))
        winner = min(scored)[1] if scored else None
        best[(optimiser, metric)] = winner
        for radius in sorted(series):
            value = misfit_at(series[radius], budget)
            final_work, final_misfit = series[radius][-1]
            mark = "  <-- best" if radius == winner else ""
            shown = "n/a" if value is None else f"{value:.4e}"
            print(f"{optimiser:18s} {metric:15s} {radius:8.3g} {final_work:8.0f} "
                  f"{shown:>14s} {final_misfit:12.4e}{mark}")
        print(f"{'':18s} {'':15s} {'budget':>8s} {budget:8.0f}")

    print()
    print("Chosen radii" + (f", TAO trust-region types scaled by {args.scale_tao}"
                            if args.scale_tao != 1.0 else ""))
    lines = []
    for (optimiser, metric), radius in sorted(best.items()):
        if radius is None:
            continue
        scaled = radius * args.scale_tao if optimiser in MESH_DEPENDENT_RADIUS else radius
        note = "  (scaled)" if scaled != radius else ""
        print(f"  {optimiser:18s} {metric:15s} {scaled:g}{note}")
        lines.append(f"{shell_name(optimiser, metric)}={scaled:g}")

    if args.emit:
        header = (
            "# Initial trust-region radius per optimiser and inner product.\n"
            "# Written by summarise_calibration.py from the sweep in "
            f"{root}.\n"
            f"# The TAO types whose radius is measured in nodal coordinates are "
            f"scaled by {args.scale_tao}, because that norm is 1/h times the L2 norm.\n")
        Path(args.emit).write_text(header + "\n".join(lines) + "\n")
        print(f"\nwritten {args.emit}")


if __name__ == "__main__":
    main()
