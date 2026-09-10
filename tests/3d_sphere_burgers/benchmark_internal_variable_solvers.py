"""Benchmark the substituted and coupled Burgers internal-variable solvers.

This is intentionally separate from ``3d_sphere_burgers.py``: it uses the same
geometry, material profiles, loading, and boundary conditions, but omits output
and diagnostics that would contaminate timings. The problem and the solver
configurations live in ``coupled_solver_variants.py`` and are shared with the
Gadi weak-scaling driver in ``tests/parallel_scaling_burgers``.

Run, for example, with::

    python benchmark_internal_variable_solvers.py --reflevel 2 --dg0-layers 1 --steps 3

The substituted solver always runs as the reference. Select the coupled
configurations to compare against it with ``--configs``, for example::

    python benchmark_internal_variable_solvers.py --configs schur-a11 static-condensation

The first step includes any lazy compilation and assembly costs. Use
``--steps 3`` or more and compare the first-step and later-step timings.
"""

import argparse
import json
import pprint
from time import perf_counter

from mpi4py import MPI

from gadopt import PETSc, errornorm, norm
from coupled_solver_variants import burgers_problem, construct_solver
from solver_configs import COUPLED_CONFIGURATIONS, displacement_ksp_prefix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflevel", type=int, default=2)
    parser.add_argument(
        "--dg0-layers", type=int, nargs="+", default=[1],
        help="Radial cells per rheological shell: one integer or four.",
    )
    parser.add_argument("--dt-years", type=float, default=1000)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--outer-rtol", type=float, default=1e-5)
    parser.add_argument("--internal-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--internal-solver", choices=("cg-sor", "local-lu"), default="local-lu",
        help="Internal fields: CG/SOR or assembled rank-local block Jacobi/LU.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=["schur-a11"],
        choices=COUPLED_CONFIGURATIONS,
        help="Coupled configurations to run next to the substituted reference.",
    )
    parser.add_argument("--schur-ksp", choices=("cg", "gmres", "fgmres"), default="gmres")
    parser.add_argument("--schur-outer", choices=("fgmres", "preonly"), default="preonly")
    parser.add_argument("--constant-jacobian", action="store_true")
    parser.add_argument("--symmetric-tensor", action="store_true")
    parser.add_argument("--device-type", choices=("HOST", "CUDA", "HIP"), default="HOST")
    parser.add_argument("--telescope-factor", type=int, default=1)
    parser.add_argument("--show-solver-parameters", action="store_true")
    parser.add_argument("--show-ksp-reasons", action="store_true")
    args = parser.parse_args()
    if len(args.dg0_layers) == 1:
        args.dg0_layers = args.dg0_layers[0]
    elif len(args.dg0_layers) != 4:
        parser.error("--dg0-layers takes one integer or four")
    return args


def timed(comm, callback):
    comm.Barrier()
    start = perf_counter()
    result = callback()
    comm.Barrier()
    return comm.allreduce(perf_counter() - start, op=MPI.MAX), result


def gpu_parameters(args):
    result = {"device_type": args.device_type}
    if args.telescope_factor != 1:
        result["telescope_factor"] = args.telescope_factor
    return result


def main():
    args = parse_args()
    problem = burgers_problem(
        args.reflevel, args.dg0_layers, args.dt_years, args.symmetric_tensor
    )
    mesh, V, S, *_ = problem
    configs = ["substituted", *args.configs]

    solvers, displacements, results = {}, {}, {}
    for config in configs:
        setup_time, (solver, displacement) = timed(
            mesh.comm,
            lambda config=config: construct_solver(
                config,
                problem,
                outer_rtol=args.outer_rtol,
                internal_rtol=args.internal_rtol,
                internal_solver=args.internal_solver,
                show_ksp_reasons=args.show_ksp_reasons,
                constant_jacobian=args.constant_jacobian,
                gpu_extra_parameters=gpu_parameters(args),
                schur_ksp=args.schur_ksp,
                schur_outer=args.schur_outer,
            ),
        )
        solvers[config] = solver
        displacements[config] = displacement
        results[config] = {
            "setup_seconds": setup_time,
            "displacement_ksp_prefix": displacement_ksp_prefix(config),
            "steps": [],
        }
        if args.show_solver_parameters and mesh.comm.rank == 0:
            print(
                f"GADOPT_BURGERS_SOLVER_PARAMETERS {config}\n"
                + pprint.pformat(solver.solver_parameters, sort_dicts=True)
            )

    stages = {config: PETSc.Log.Stage(f"burgers_{config}_solve") for config in configs}
    failed = set()
    for step in range(1, args.steps + 1):
        for config in configs:
            if config in failed:
                continue

            def solve(config=config):
                with stages[config]:
                    solvers[config].solve()

            try:
                solve_time, _ = timed(mesh.comm, solve)
            except Exception as error:  # noqa: BLE001 - record and carry on
                failed.add(config)
                results[config]["steps"].append({"step": step, "failed": repr(error)})
                continue
            snes = solvers[config].solver.snes
            results[config]["steps"].append(
                {
                    "step": step,
                    "solve_seconds": solve_time,
                    "snes_iterations": snes.getIterationNumber(),
                    "top_level_ksp_iterations": snes.getLinearSolveIterations(),
                }
            )
            if config != "substituted":
                difference = errornorm(displacements["substituted"], displacements[config])
                reference = norm(displacements["substituted"])
                results[config]["steps"][-1]["relative_l2_difference"] = (
                    difference / max(float(reference), 1e-300)
                )

    results["configuration"] = vars(args) | {
        "displacement_dofs": V.dim(),
        "internal_variable_dofs_each": S.dim(),
    }
    if mesh.comm.rank == 0:
        print("GADOPT_BURGERS_BENCHMARK " + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
