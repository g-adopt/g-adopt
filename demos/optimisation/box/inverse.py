"""Driver of the box optimisation benchmark.

Builds the inverse problem from a reference state, runs one optimiser, and writes a
JSON record of the run. The record holds the twin-experiment measures at every
gradient evaluation and at every optimiser iteration, together with the tape sweep
counts, so that runs with different optimisers can be compared by forward-solve
budget and not by iteration count.

Usage:
    python inverse.py --optimiser rol-lbfgs [--checkpoint reference-state.h5]
                      [--window 10] [--max-it 20] [--riesz L2|regularisation]
                      [--alpha-d 1e-3] [--alpha-s 1e-3] [--initial-radius 1e-2]
                      [--tao-opt key=value ...] [--output run.json]

Optimiser names are listed in ``optimisers.VARIANTS``. The default output name is
built from the optimiser and the Riesz map, so a sweep over both writes distinct files.

Taping happens in ``problem.build_problem`` and nowhere else. This file never touches
the tape.
"""

import argparse
import sys
from pathlib import Path

from mpi4py import MPI

from gadopt import *
from optimisers import VARIANTS, run
from problem import build_problem


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--optimiser", default="rol-lbfgs", choices=VARIANTS)
    parser.add_argument("--checkpoint", default="reference-state.h5",
                        help="reference state from forward.py")
    parser.add_argument("--window", type=int, default=10,
                        help="timesteps inverted, counted from the end of the trajectory")
    parser.add_argument("--max-it", type=int, default=20, help="optimiser iteration limit")
    parser.add_argument("--riesz", default="L2", choices=("L2", "regularisation"),
                        help="inner product on the control space")
    parser.add_argument("--alpha-d", type=float, default=1e-3, help="damping weight")
    parser.add_argument("--alpha-s", type=float, default=1e-3, help="smoothing weight")
    parser.add_argument("--initial-radius", type=float, default=1e-2,
                        help="initial trust-region radius, ROL and TAO")
    parser.add_argument("--initial-guess", default="average",
                        choices=("average", "observed", "true"),
                        help="'observed' is the first guess of the paper")
    parser.add_argument("--metric-alpha-d", type=float, default=None,
                        help="mass weight of the regularisation inner product; "
                             "defaults to --alpha-d. Set it when the objective has "
                             "no regularisation, where the default would be zero "
                             "and the Riesz solve singular")
    parser.add_argument("--metric-alpha-s", type=float, default=None,
                        help="stiffness weight of the regularisation inner product; "
                             "defaults to --alpha-s")
    parser.add_argument("--work-budget", type=float, default=0.0,
                        help="stop when forward replays plus adjoint sweeps plus the "
                             "measured cost of the Hessian-vector products passes "
                             "this number; 0 removes the limit. Every method gets "
                             "the same budget, so that methods with very different "
                             "costs per iteration are read at the same total cost")
    parser.add_argument("--gradient-rtol", type=float, default=0.0,
                        help="stop when the convergence measure falls to this "
                             "fraction of its starting value; 0 runs to --max-it")
    parser.add_argument("--krylov-max-it", type=int, default=20,
                        help="inner Krylov iteration limit (ROL truncated CG, TAO BNK KSP)")
    parser.add_argument("--no-riesz-j0", action="store_true",
                        help="TAO bqnk types: leave the LMVM initial Hessian as the identity")
    parser.add_argument("--tao-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="extra PETSc option for TAO, repeatable; KEY alone sets a flag")
    parser.add_argument("--rol-shift", type=float, default=0.0, metavar="FACTOR",
                        help="ROL only: shift the Hessian action by FACTOR times the "
                             "gradient norm, as PETSc BNK does "
                             "(pert = min(1e2, max(1e-4, 0.1|g|))). 0 applies no "
                             "shift. Use 0.1 to match BNK's initial rule. The misfit "
                             "Hessian with no regularisation is indefinite, and ROL "
                             "has no way to regularise it from inside the library")
    parser.add_argument("--rol-opt", action="append", default=[], metavar="PATH=VALUE",
                        help="override one entry of the ROL parameter list, repeatable. "
                             "PATH is the slash-separated path through the nested "
                             "parameter dictionary, for example "
                             "'Step/Trust Region/Radius Growing Rate=2.5'. The value is "
                             "read as a Python literal, so 2.5 is a float, 5 is an "
                             "integer and true is the string; write True for the boolean")
    parser.add_argument("--output", default=None, help="JSON output file")
    parser.add_argument("--save-control", action="store_true",
                        help="write the final control to <output stem>.h5")
    return parser.parse_args(argv)


def provenance():
    """Record which code produced a run.

    ``demos/optimisation/`` is untracked at the moment, so the commit alone does not
    identify the demo code. The dictionary therefore holds the commit of the worktree
    and the hash of every Python file of this demo, so that a result can always be
    matched to the code that made it.

    Returns:
        dict with ``commit`` and ``demo_files``. Missing information is recorded as
        None instead of raising, because a run must never fail over its own metadata.
    """
    import hashlib
    import subprocess

    def git_head(directory):
        """HEAD of the repository that holds ``directory``, or None."""
        try:
            return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None

    here = Path(__file__).resolve().parent
    demo_files = {}
    for path in sorted(here.glob("*.py")):
        try:
            demo_files[path.name] = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        except OSError:
            demo_files[path.name] = None

    # Which firedrake produced the run. The two environments on this machine give
    # different costs per call, so a result is meaningless without it.
    import firedrake
    import pyadjoint
    firedrake_path = Path(firedrake.__file__).resolve().parent

    # The library versions. The Setonix container and the reunion environment run
    # different PETSc releases, and a comparison across machines must name both.
    from importlib import metadata
    from petsc4py import PETSc

    def package_version(name):
        """Installed version of a distribution, or None."""
        try:
            return metadata.version(name)
        except Exception:
            return None

    return {
        "commit": git_head(here),
        "demo_files": demo_files,
        "firedrake": str(firedrake_path),
        "firedrake_commit": git_head(firedrake_path),
        "pyadjoint": str(Path(pyadjoint.__file__).resolve().parent),
        "python": sys.executable,
        "petsc_version": ".".join(str(v) for v in PETSc.Sys.getVersion()),
        "petsc4py_version": package_version("petsc4py"),
        "firedrake_version": package_version("firedrake"),
        "pyadjoint_version": package_version("pyadjoint-ad"),
    }


def prior_record(problem, riesz):
    """The ``prior`` block of the run record.

    The prior-covariance campaign states the regularisation, the Riesz map and the
    Krylov preconditioner as one operator, a prior covariance ``B`` with a standard
    deviation ``sigma``, a correlation length ``L``, an order, a wall treatment and a
    mean ``m0``. This code does not define that operator yet. The inner product it
    uses is the legacy metric ``a(u, v) = 2 c_d (u, v) + 2 c_s (grad u, grad v)``,
    so the block holds the coefficients of that metric, and None for every quantity
    that does not exist in the code.

    Args:
        problem: the ``InverseProblem``.
        riesz: the ``--riesz`` choice, ``"L2"`` or ``"regularisation"``.

    Returns:
        dict
    """
    weights = problem.weights
    block = {
        "kind": "legacy-metric" if riesz == "regularisation" else "L2",
        # ``metric_c_d`` and ``metric_c_s`` are the coefficients of the records;
        # the form that the Riesz map inverts carries twice these values
        # (``problem.py``, ``make_inner_product(..., c_d=2 * m_c_d, c_s=2 * m_c_s)``).
        "metric_c_d": weights.get("metric_c_d"),
        "metric_c_s": weights.get("metric_c_s"),
        "form_c_d": None,
        "form_c_s": None,
        "sigma": None,
        "L": None,
        "order": None,
        "wall_treatment": None,
        "m0": None,
        "sigma_obs": None,
        "pointwise_std_measured": None,
    }
    if riesz == "regularisation":
        block["form_c_d"] = 2 * weights["metric_c_d"]
        block["form_c_s"] = 2 * weights["metric_c_s"]
    return block


def parse_tao_options(items):
    """Turn ``["a=1", "b"]`` into ``{"a": "1", "b": None}``."""
    options = {}
    for item in items:
        key, sep, value = item.partition("=")
        options[key] = value if sep else None
    return options


def parse_rol_options(items):
    """Turn ``["A/B=2.5"]`` into ``{("A", "B"): 2.5}``.

    The ROL parameter list is a nested dictionary, so an override needs the whole
    path to the entry and not only its name. The path is split on "/" and the value
    is read with ``ast.literal_eval``, which gives a float for 2.5, an integer for 5
    and a boolean for True. A value that is not a Python literal is kept as the
    string it was written as, which is what ROL wants for entries such as
    ``Subproblem Model``.

    Args:
        items: the raw ``PATH=VALUE`` strings from the command line.

    Returns:
        dict mapping a tuple of keys to the value to write there.
    """
    import ast
    options = {}
    for item in items:
        path, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--rol-opt needs PATH=VALUE, and got {item!r}")
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        options[tuple(path.split("/"))] = parsed
    return options


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output or f"{args.optimiser}_{args.riesz}.json")

    problem = build_problem(
        args.checkpoint, window=args.window,
        alpha_d=args.alpha_d, alpha_s=args.alpha_s,
        inner_product=args.riesz, initial_guess=args.initial_guess,
        metric_alpha_d=args.metric_alpha_d, metric_alpha_s=args.metric_alpha_s,
    )
    log(f"Problem built: window {problem.window} steps, control dofs "
        f"{problem.Tic.function_space().dim()}, inner product {problem.inner_product.name}, "
        f"initial control error {problem.control_error():.4e}")

    # The settings are assembled before the run, so that the recorder can write a
    # complete record after every iteration. A run stopped by a wall-clock limit
    # then still leaves everything up to its last iteration on disk.
    record = {
        "optimiser": args.optimiser,
        "riesz": args.riesz,
        "window": problem.window,
        "weights": problem.weights,
        "control_dofs": problem.Tic.function_space().dim(),
        "ranks": MPI.COMM_WORLD.size,
        "max_it": args.max_it,
        "initial_radius": args.initial_radius,
        "norm_obs": problem.norm_obs,
        "norm_ic": problem.norm_ic,
        "initial_guess": args.initial_guess,
        "gradient_rtol": args.gradient_rtol,
        "work_budget_requested": args.work_budget,
        "checkpoint": str(args.checkpoint),
        "provenance": provenance(),
        # The timestep and the physical length of the window; ``window`` is a count.
        "delta_t": problem.delta_t,
        "window_time": problem.window_time,
        "krylov_max_it": args.krylov_max_it,
        "prior": prior_record(problem, args.riesz),
    }

    kwargs = {"max_it": args.max_it, "initial_radius": args.initial_radius,
              "output": output, "metadata": record,
              "gradient_rtol": args.gradient_rtol,
              "work_budget": args.work_budget}
    if args.optimiser.startswith("rol"):
        kwargs["krylov_max_it"] = args.krylov_max_it
        kwargs["overrides"] = parse_rol_options(args.rol_opt)
        kwargs["hessian_shift"] = args.rol_shift
    else:
        extra = {"tao_bnk_ksp_max_it": args.krylov_max_it}
        extra.update(parse_tao_options(args.tao_opt))
        kwargs["extra_options"] = extra
        kwargs["riesz_initial_hessian"] = not args.no_riesz_j0

    recorder = run(problem, args.optimiser, **kwargs)

    # Final dump: the same record plus the measures that only exist at the end.
    record["final_control_error"] = float(problem.control_error())
    record["tao_options"] = kwargs.get("extra_options")
    # The ROL overrides are recorded with their paths joined back into one string, so
    # that the JSON keeps a form that can be pasted straight onto a command line.
    record["rol_shift"] = args.rol_shift
    record["rol_options"] = {"/".join(path): value
                             for path, value in (kwargs.get("overrides") or {}).items()}
    record["final_initial_misfit"] = problem.initial_misfit()
    recorder.metadata = record
    recorder.write()
    record.update(recorder.summary())
    cost = recorder.summary()["call_cost"]
    log(f"Run finished: {recorder.forward} forward, {recorder.adjoint} adjoint, "
        f"{recorder.hessian} Hessian-vector products; final control error "
        f"{record['final_control_error']:.4e}; written to {output}")
    log(f"Call cost: forward {cost['mean_forward_time'] or float('nan'):.1f} s, "
        f"gradient x{cost['gradient_per_forward'] or float('nan'):.2f}, "
        f"Hessian-vector product x{cost['hessian_per_forward'] or float('nan'):.2f}; "
        f"stop: {recorder.stop_reason}")

    if args.save_control:
        with CheckpointFile(str(output.with_suffix(".h5")), "w") as f:
            f.save_mesh(problem.mesh)
            final = Function(problem.Tic.function_space(), name="Control").assign(
                problem.control.tape_value())
            f.save_function(final, name="Control")


if __name__ == "__main__":
    main()
