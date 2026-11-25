"""Run the substituted reference and one coupled configuration on one mesh."""

import json
import sys
from pathlib import Path
from time import perf_counter

from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3d_sphere_burgers"))

from gadopt import PETSc, errornorm, norm  # noqa: E402
from coupled_solver_variants import burgers_problem, construct_solver  # noqa: E402


def _timed(comm, callback):
    comm.Barrier()
    start = perf_counter()
    callback()
    comm.Barrier()
    return comm.allreduce(perf_counter() - start, op=MPI.MAX)


def model(level, config, layers, dt_years, steps):
    problem = burgers_problem(level, layers, dt_years)
    mesh, V, S, *_ = problem
    configs = ["substituted", config]

    solvers, displacements, stages, results = {}, {}, {}, {}
    for name in configs:
        solver, displacement = construct_solver(name, problem, show_ksp_reasons=True)
        solvers[name] = solver
        displacements[name] = displacement
        stages[name] = PETSc.Log.Stage(f"burgers_{name}_solve")
        results[name] = []
    PETSc.Sys.Print(
        f"level {level} config {config}: displacement dofs {V.dim()}, "
        f"internal variable dofs {2 * S.dim()}, ranks {mesh.comm.size}"
    )

    failed = set()
    for step in range(1, steps + 1):
        for name in configs:
            if name in failed:
                continue

            def solve(name=name):
                with stages[name]:
                    solvers[name].solve()

            try:
                seconds = _timed(mesh.comm, solve)
            except Exception as error:  # noqa: BLE001 - record the failure, keep going
                failed.add(name)
                results[name].append({"step": step, "failed": repr(error)})
                PETSc.Sys.Print(f"{name} failed at step {step}: {error!r}")
                continue
            snes = solvers[name].solver.snes
            record = {
                "step": step,
                "solve_seconds": seconds,
                "snes_iterations": snes.getIterationNumber(),
                "top_level_ksp_iterations": snes.getLinearSolveIterations(),
            }
            if name != "substituted" and "substituted" not in failed:
                difference = errornorm(displacements["substituted"], displacements[name])
                record["relative_l2_difference"] = difference / max(
                    float(norm(displacements["substituted"])), 1e-300
                )
            results[name].append(record)
            PETSc.Sys.Print(f"step {step} {name}: {seconds:.3f} s")

    summary = {
        "level": level,
        "config": config,
        "ranks": mesh.comm.size,
        "displacement_dofs": V.dim(),
        "internal_variable_dofs": 2 * S.dim(),
        "results": results,
    }
    PETSc.Sys.Print("GADOPT_BURGERS_SCALING " + json.dumps(summary, sort_keys=True))
