"""Driver of the cylindrical optimiser comparison: one optimiser per run, one record.

The comparison puts the method that the box campaign selected, ``tao-bnls-riesz``
(PETSc TAO bounded Newton line search, STCG preconditioned by the control metric,
BNK's Hessian shift at zero), against ``rol-lbfgs`` (ROL Lin-More with an L-BFGS
secant), which is what g-adopt runs today, on the inverse problem of
``tests/adjoint_2d_cylindrical/inverse.py``. Both optimisers see the same objective,
the same control, the same bounds [0, 1] and the same inner product.

Usage:
    # once per trajectory: write the reference state
    python compare.py --make-reference-only --rundir RUN --timesteps 125 --delta-t 1.2e-5

    # one run
    python compare.py --optimiser tao-bnls-riesz --bnk-shift zero --rundir RUN \
        --timesteps 125 --delta-t 1.2e-5 --work-budget 200 --output tao-zero.json
    python compare.py --optimiser rol-lbfgs --rundir RUN --timesteps 125 \
        --delta-t 1.2e-5 --work-budget 200 --output rol.json

The optimiser back-ends are the ones of the box benchmark, imported unchanged from
``demos/optimisation/box/``: ``optimisers.run_rol`` and ``optimisers.run_tao``, the
preconditioner and the metric-shifted Hessian of ``riesz_hessian.py``, and the
per-solve Krylov record of ``recording.py``. The problem comes from
``cylindrical_problem.py``, which tapes the production model through
``generate_inverse_problem``.

How each optimiser is configured.

``rol-lbfgs`` takes g-adopt's ``minimisation_parameters`` with the two changes that
``inverse.inverse()`` makes for this case, an initial radius of 1e-3 and an iteration
limit, and with ``optimisers.rol_parameters`` given g-adopt's own Krylov limit of 10
(the box default is 20). ``checks.py rol-params`` proves that the resulting parameter
list equals g-adopt's in every other entry. The differences from
``gadopt.inverse.LinMoreOptimiser`` are three and do not change the iterates: the
status test also records each iteration and stops on the work budget, the vectors
carry the inner product of this run (``FormROLVector``), and ROL's own checkpointing
to disk is off. With ``--riesz L2`` the inner product is the one of g-adopt today.

``tao-bnls-riesz`` takes ``optimisers.tao_parameters`` with a Krylov cap (default
20), the initial trust radius of ``default_tao_radius`` and the shift of
``--bnk-shift``:

- ``zero``: ``-tao_bnk_imax 0``. Every first shift BNK sets is then
  ``min(imax, ...) = 0``, and a shift that is zero is never increased, so STCG's trust
  radius alone bounds the step after negative curvature. The working default of the
  box campaign (DECISIONS.md item 10).
- ``metric``: BNK's default rules, with the shift applied as ``pert * A``
  (``riesz_hessian.ShiftedHessianMat``).
- ``identity``: PETSc's own ``pert * I`` in nodal coordinates, which stalled the box
  runs. Not part of the plan; kept for a control run.

Work. The record counts forward replays, adjoint sweeps and Hessian-vector products
through the reduced-functional callbacks, and prices a product at its measured cost
in forward calls. ``--work-budget`` stops every optimiser at the same work.
"""

import argparse
import hashlib
import math
import subprocess
import sys
from pathlib import Path

from mpi4py import MPI

import cylindrical_problem as cp  # sets the import paths; must come first
from cylindrical_problem import BOX_DIR, CYLINDRICAL_DIR, HERE, build_problem, prepare_reference

from gadopt import log  # noqa: E402
import optimisers  # noqa: E402 -- demos/optimisation/box
from optimisers import Recorder, VARIANTS, run  # noqa: E402

# The nodal trust radius of the box runs whose evidence selected the zero shift: Z-d
# at 0.22 and Z-e at 0.2 on the 64-cell box, whose CG1 control has 65^2 = 4225 nodes.
BOX_RADIUS = 0.2
BOX_NODES = 4225

# g-adopt's settings for this case (tests/adjoint_2d_cylindrical/inverse.py and
# gadopt/inverse.py minimisation_parameters).
GADOPT_ROL_RADIUS = 1e-3
GADOPT_ROL_KRYLOV_MAX_IT = 10


def default_tao_radius(control_dofs):
    """The initial trust radius of BNK for this mesh, from the box runs.

    STCG measures the step in the Euclidean norm of the nodal values
    (``-ksp_cg_dtype unpreconditioned``, the default), so a radius means a root mean
    square nodal change of ``radius / sqrt(n)``. The box zero-shift runs used 0.2
    on 4225 nodes, which is an rms nodal temperature change of 3.1e-3 per step.
    The same rms change on this mesh needs ``0.2 * sqrt(n / 4225)``: 0.79 for the
    66048 nodes of the annulus. BNLS changes the radius from the step lengths after
    each iteration, so this sets the size of the first steps only.

    Args:
        control_dofs: global number of control nodes.

    Returns:
        float
    """
    return BOX_RADIUS * math.sqrt(control_dofs / BOX_NODES)


class CylindricalRecorder(Recorder):
    """The box recorder with the terms of the cylindrical objective.

    The box recorder writes ``misfit_term = final_misfit``, which is right for the
    box, whose misfit is the temperature term alone. Here the misfit is the
    temperature term plus the velocity term, so this class rewrites ``misfit_term``
    as ``J - prior_term`` and adds the parts:

    - ``temperature_term = alpha_T * final_misfit``;
    - ``velocity_term = misfit_term - temperature_term``, the surface velocity
      misfit. It is the remainder of J, because the accumulated sum over the
      timesteps is not a separate object on the tape;
    - ``damping_term`` and ``smoothing_term`` from ``CylindricalProblem.measures``.

    ``mark_iteration`` copies these keys from the most recent gradient row into the
    iteration row, beside the keys that the box recorder copies. Nothing here writes
    into a vector that an optimiser uses.
    """

    EXTRA_KEYS = ("temperature_term", "velocity_term", "damping_term",
                  "smoothing_term")

    def _derivative_cb(self, J, dJ, values):
        out = super()._derivative_cb(J, dJ, values)
        row = self.gradients[-1]
        alpha_T = self.problem.weights["alpha_T"]
        row["temperature_term"] = (alpha_T if alpha_T > 0 else 0.0) * row["final_misfit"]
        row["misfit_term"] = row["J"] - row["prior_term"]
        row["velocity_term"] = row["misfit_term"] - row["temperature_term"]
        return out

    def mark_iteration(self, k, **extra):
        if self.gradients:
            for key in self.EXTRA_KEYS:
                if key in self.gradients[-1] and key not in extra:
                    extra[key] = self.gradients[-1][key]
        super().mark_iteration(k, **extra)


def install_recorder():
    """Make ``run_rol`` and ``run_tao`` build ``CylindricalRecorder``.

    Both runners create their recorder as ``Recorder(problem, ...)`` from the module
    namespace of ``optimisers``, so replacing that name is enough. The box files stay
    unchanged; a ``recorder_class`` argument on the two runners would make this
    unnecessary.
    """
    optimisers.Recorder = CylindricalRecorder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--optimiser", default="tao-bnls-riesz", choices=VARIANTS,
                        help="rol-lbfgs or tao-bnls-riesz for the comparison; the "
                             "other box variants run as well but are not part of it")
    parser.add_argument("--rundir", required=True,
                        help="directory of the reference state; created if needed")
    parser.add_argument("--timesteps", type=int, default=cp.PRODUCTION_TIMESTEPS)
    parser.add_argument("--delta-t", type=float, default=cp.PRODUCTION_DELTA_T,
                        help="timestep; production 6e-6, the negative curvature of "
                             "NOTES/CYLINDRICAL-HESSIAN-2026-11.md is at 1.2e-5")
    parser.add_argument("--make-reference", action="store_true",
                        help="write the reference state, then run")
    parser.add_argument("--make-reference-only", action="store_true",
                        help="write the reference state and stop")
    parser.add_argument("--riesz", default="regularisation",
                        choices=("regularisation", "L2"),
                        help="inner product of the control for every optimiser")
    parser.add_argument("--alpha-T", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_T"])
    parser.add_argument("--alpha-u", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_u"])
    parser.add_argument("--alpha-d", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_d"])
    parser.add_argument("--alpha-s", type=float, default=cp.PRODUCTION_WEIGHTS["alpha_s"])
    parser.add_argument("--work-budget", type=float, default=0.0,
                        help="stop at this forward-equivalent work; 0 removes the "
                             "limit. Give every optimiser the same number")
    parser.add_argument("--max-it", type=int, default=1000,
                        help="iteration limit; the budget normally stops the run first")
    parser.add_argument("--initial-radius", type=float, default=None,
                        help="initial trust radius. Default: 1e-3 for ROL (g-adopt's "
                             "value for this case), default_tao_radius() for TAO")
    parser.add_argument("--krylov-max-it", type=int, default=None,
                        help="Krylov cap. Default: 10 for ROL (g-adopt), 20 for TAO")
    parser.add_argument("--bnk-shift", default="zero",
                        choices=("zero", "metric", "identity"),
                        help="BNK's Hessian shift: zero (-tao_bnk_imax 0), metric "
                             "(pert * A) or identity (PETSc's pert * I)")
    parser.add_argument("--gradient-rtol", type=float, default=0.0,
                        help="stop when the convergence measure falls to this "
                             "fraction of its first value; 0 leaves it to the budget")
    parser.add_argument("--tao-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="extra PETSc option for TAO, repeatable")
    parser.add_argument("--output", default=None, help="JSON record")
    parser.add_argument("--save-control", action="store_true",
                        help="write the final control to <output stem>.h5")
    return parser.parse_args(argv)


def provenance():
    """The code that made a run: commit, file hashes, library versions.

    The demo and the box files it imports are hashed, so that a record names the
    exact code even when the working tree differs from the commit. A missing entry is
    None, because a run must never fail over its own metadata.

    Returns:
        dict
    """
    def git_head(directory):
        try:
            return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None

    files = {}
    for directory in (HERE, BOX_DIR, CYLINDRICAL_DIR):
        for path in sorted(directory.glob("*.py")):
            try:
                files[f"{directory.name}/{path.name}"] = hashlib.sha1(
                    path.read_bytes()).hexdigest()[:12]
            except OSError:
                files[f"{directory.name}/{path.name}"] = None

    import firedrake
    import pyadjoint
    from importlib import metadata
    from petsc4py import PETSc

    def version(name):
        try:
            return metadata.version(name)
        except Exception:
            return None

    firedrake_path = Path(firedrake.__file__).resolve().parent
    return {
        "commit": git_head(HERE),
        "files": files,
        "firedrake": str(firedrake_path),
        "firedrake_commit": git_head(firedrake_path),
        "pyadjoint": str(Path(pyadjoint.__file__).resolve().parent),
        "python": sys.executable,
        "petsc_version": ".".join(str(v) for v in PETSc.Sys.getVersion()),
        "firedrake_version": version("firedrake"),
        "pyadjoint_version": version("pyadjoint-ad"),
    }


def parse_tao_options(items):
    """Turn ``["a=1", "b"]`` into ``{"a": "1", "b": None}``."""
    options = {}
    for item in items:
        key, sep, value = item.partition("=")
        options[key] = value if sep else None
    return options


def optimiser_settings(args, problem):
    """The keyword arguments of ``optimisers.run`` for this run, and their record.

    Args:
        args: the parsed command line.
        problem: the ``CylindricalProblem``.

    Returns:
        ``(kwargs, settings)``: the arguments of the runner, and a dictionary of the
        settings for the record.
    """
    dofs = problem.Tic.function_space().dim()
    kwargs = {"max_it": args.max_it, "gradient_rtol": args.gradient_rtol,
              "work_budget": args.work_budget}
    settings = {}
    if args.optimiser.startswith("rol"):
        radius = GADOPT_ROL_RADIUS if args.initial_radius is None else args.initial_radius
        krylov = (GADOPT_ROL_KRYLOV_MAX_IT if args.krylov_max_it is None
                  else args.krylov_max_it)
        # g-adopt's minimisation_parameters: secant storage 10, Krylov relative and
        # absolute tolerances 1e-2 and 1e-4. Passed explicitly, so that a change of
        # the box defaults cannot change this run.
        kwargs.update(initial_radius=radius, krylov_max_it=krylov, secant_storage=10,
                      krylov_rtol=1e-2, krylov_atol=1e-4)
        settings.update(initial_radius=radius, krylov_max_it=krylov,
                        rol_reference="gadopt.inverse.minimisation_parameters")
    else:
        radius = (default_tao_radius(dofs) if args.initial_radius is None
                  else args.initial_radius)
        krylov = 20 if args.krylov_max_it is None else args.krylov_max_it
        extra = {"tao_bnk_ksp_max_it": krylov}
        if args.bnk_shift == "zero":
            # The first shift is min(imax, max(imin, imfac * gnorm)) = 0, and an
            # increase applies only to a positive shift, so the shift stays zero.
            extra["tao_bnk_imax"] = 0.0
        extra.update(parse_tao_options(args.tao_opt))
        kwargs.update(initial_radius=radius, extra_options=extra,
                      bnk_shift="metric" if args.bnk_shift == "metric" else "identity")
        settings.update(initial_radius=radius,
                        initial_radius_rule=(f"{BOX_RADIUS} * sqrt({dofs} / {BOX_NODES})"
                                             if args.initial_radius is None else "given"),
                        krylov_max_it=krylov, tao_options=extra,
                        bnk_shift=args.bnk_shift)
    return kwargs, settings


def main(argv=None):
    args = parse_args(argv)
    # Paths given on the command line are relative to where the command was typed;
    # the problem builder moves the process into the run directory.
    output = Path(args.output or f"{args.optimiser}_{args.riesz}.json").resolve()

    if args.make_reference_only:
        prepare_reference(args.rundir, args.timesteps, args.delta_t, make=True)
        log(f"reference state written in {Path(args.rundir).resolve()}")
        return

    install_recorder()
    problem = build_problem(
        args.rundir, timesteps=args.timesteps, delta_t=args.delta_t,
        alpha_T=args.alpha_T, alpha_u=args.alpha_u, alpha_d=args.alpha_d,
        alpha_s=args.alpha_s, inner_product=args.riesz,
        make_reference=args.make_reference)
    log(f"problem: {problem.timesteps} steps of {problem.delta_t}, "
        f"{problem.Tic.function_space().dim()} control dofs, inner product "
        f"{problem.inner_product.name}, J at taping {problem.taped_objective:.10e}, "
        f"initial control error {problem.control_error():.4e}")

    kwargs, settings = optimiser_settings(args, problem)
    record = {
        "optimiser": args.optimiser,
        "riesz": args.riesz,
        "ranks": MPI.COMM_WORLD.size,
        "max_it": args.max_it,
        "work_budget_requested": args.work_budget,
        "gradient_rtol": args.gradient_rtol,
        "argv": sys.argv,
        "provenance": provenance(),
        **settings,
        **problem.record(),
    }
    kwargs.update(output=output, metadata=record)

    recorder = run(problem, args.optimiser, **kwargs)

    record["final_control_error"] = float(problem.control_error())
    record["final_initial_misfit"] = problem.initial_misfit()
    recorder.metadata = record
    recorder.write()
    cost = recorder.summary()["call_cost"]
    log(f"run finished: {recorder.forward} forward, {recorder.adjoint} adjoint, "
        f"{recorder.hessian} Hessian-vector products, work {recorder.work():.1f}; "
        f"initial misfit {record['final_initial_misfit']:.4e}; stop: "
        f"{recorder.stop_reason}; written to {output}")
    log(f"call cost: forward {cost['mean_forward_time'] or float('nan'):.1f} s, "
        f"gradient x{cost['gradient_per_forward'] or float('nan'):.2f}, "
        f"Hessian-vector product x{cost['hessian_per_forward'] or float('nan'):.2f}")

    if args.save_control:
        from firedrake import CheckpointFile, Function
        with CheckpointFile(str(output.with_suffix(".h5")), "w") as f:
            f.save_mesh(problem.mesh)
            final = Function(problem.Tic.function_space(), name="Control").assign(
                problem.control.tape_value())
            f.save_function(final, name="Control")


if __name__ == "__main__":
    main()
