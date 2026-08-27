"""Reproducible pressure-preconditioner benchmark for G-ADOPT Stokes solves.

Run one configuration per process, for example::

    python benchmark.py --case linear --contrast 1e6 --pc mass
    python benchmark.py --case tala --contrast 1e6 --pc bfbt
    python benchmark.py --case viscoplastic --pc bfbt

The first solve includes JIT compilation and cold PETSc setup. Warm timings
are repeated from the same initial state. All timings are communicator
barrier-to-barrier maximum-rank wall times, and only rank zero writes JSON.
Correctness is checked independently using the assembled equation residual.
"""

import argparse
import json
import sys
from importlib import import_module
from statistics import median
from time import perf_counter


def argument_parser():
    """Return command-line options while leaving PETSc options untouched."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("linear", "tala", "ala", "viscoplastic"),
        required=True,
    )
    parser.add_argument("--pc", choices=("mass", "bfbt"), required=True)
    parser.add_argument("--contrast", type=float, default=1e6)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--velocity-pc", choices=("lu", "gamg"), default="lu")
    parser.add_argument(
        "--bfbt-inner-ksp", choices=("preonly", "fgmres"), default="fgmres"
    )
    parser.add_argument("--bfbt-inner-rtol", type=float, default=1e-2)
    parser.add_argument(
        "--bfbt-mass-lumping",
        choices=("diagonal", "rowsum"),
        default="diagonal",
    )
    parser.add_argument("--bfbt-weight-degree", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=3)
    return parser


if __name__ == "__main__":
    _arguments, _petsc_arguments = argument_parser().parse_known_args()
    sys.argv = [sys.argv[0], *_petsc_arguments]
else:
    _arguments = None

fd = import_module("firedrake")
gadopt = import_module("gadopt")
(
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
    StokesSolver,
    TruncatedAnelasticLiquidApproximation,
    create_stokes_nullspace,
) = (
    gadopt.AnelasticLiquidApproximation,
    gadopt.BoussinesqApproximation,
    gadopt.StokesSolver,
    gadopt.TruncatedAnelasticLiquidApproximation,
    gadopt.create_stokes_nullspace,
)


def solver_parameters(
    preconditioner,
    velocity_pc,
    bfbt_inner_ksp,
    bfbt_inner_rtol,
    bfbt_mass_lumping,
    bfbt_weight_degree,
    nonlinear,
):
    """Build comparable full-Schur solver parameters."""
    if velocity_pc == "lu":
        velocity_parameters = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "lu",
        }
    else:
        velocity_parameters = {
            "ksp_type": "cg",
            "ksp_rtol": 1e-8,
            "pc_type": "python",
            "pc_python_type": "gadopt.SPDAssembledPC",
            "assembled_pc_type": "gamg",
            "assembled_mg_levels_pc_type": "sor",
            "assembled_pc_gamg_threshold": 0.01,
        }

    pressure_parameters = {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 300,
        "pc_type": "python",
    }
    if preconditioner == "mass":
        pressure_parameters.update(
            {
                "pc_python_type": "firedrake.MassInvPC",
                "Mp_pc_type": "ksp",
                "Mp_ksp_ksp_type": "cg",
                "Mp_ksp_ksp_rtol": 1e-8,
                "Mp_ksp_pc_type": "sor",
            }
        )
    else:
        pressure_parameters.update(
            {
                "pc_python_type": "gadopt.DensityAwareBFBTPC",
                "bfbt_ksp_type": bfbt_inner_ksp,
                "bfbt_ksp_rtol": bfbt_inner_rtol,
                "bfbt_ksp_max_it": 100,
                "bfbt_pc_type": "gamg",
                "bfbt_mass_lumping": bfbt_mass_lumping,
                "bfbt_weight_degree": bfbt_weight_degree,
            }
        )

    parameters = {
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0": velocity_parameters,
        "fieldsplit_1": pressure_parameters,
    }
    if nonlinear:
        parameters.update(
            {
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "snes_rtol": 1e-8,
                "snes_atol": 1e-10,
                "snes_max_it": 40,
            }
        )
    else:
        parameters["snes_type"] = "ksponly"
    return parameters


def build_case(args):
    """Build linear, TALA/ALA, or demo-equivalent viscoplastic Stokes case."""
    mesh = fd.UnitSquareMesh(args.n, args.n, quadrilateral=True)
    mesh.cartesian = True
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    u, _ = fd.split(solution)
    temperature = fd.Function(temperature_space)
    x, y = fd.SpatialCoordinate(mesh)
    temperature.interpolate(
        1 - y + 0.05 * fd.cos(fd.pi * x) * fd.sin(fd.pi * y)
    )

    if args.case == "viscoplastic":
        gamma_temperature = fd.Constant(fd.ln(1e5))
        gamma_depth = fd.Constant(fd.ln(10))
        mu_star = fd.Constant(0.001)
        yield_stress = fd.Constant(1)
        strain_rate = fd.sym(fd.grad(u))
        strain_rate_invariant = fd.sqrt(
            fd.inner(strain_rate, strain_rate) + 1e-10
        )
        linear_viscosity = fd.exp(
            -gamma_temperature * temperature + gamma_depth * (1 - y)
        )
        plastic_viscosity = mu_star + yield_stress / strain_rate_invariant
        viscosity = (
            2 * linear_viscosity * plastic_viscosity
            / (linear_viscosity + plastic_viscosity)
        )
        approximation = BoussinesqApproximation(100, mu=viscosity)
    else:
        viscosity = fd.exp(fd.ln(fd.Constant(args.contrast)) * x)
        if args.case == "linear":
            approximation = BoussinesqApproximation(1, mu=viscosity)
        elif args.case == "tala":
            density = fd.Function(temperature_space).interpolate(
                fd.exp(0.5 * (1 - y))
            )
            approximation = TruncatedAnelasticLiquidApproximation(
                1, 0.5, rho=density, mu=viscosity
            )
        else:
            density = fd.Function(temperature_space).interpolate(
                fd.exp(0.5 * (1 - y))
            )
            approximation = AnelasticLiquidApproximation(
                1, 0.5, rho=density, mu=viscosity
            )

    bcs = {
        1: {"ux": 0},
        2: {"ux": 0},
        3: {"uy": 0},
        4: {"uy": 0},
    }
    nullspace_parameters = {}
    if args.case == "ala":
        nullspace_parameters = {
            "ala_approximation": approximation,
            "top_subdomain_id": 4,
        }
    nullspace = create_stokes_nullspace(
        mixed_space, closed=True, **nullspace_parameters
    )
    transpose_nullspace = create_stokes_nullspace(mixed_space, closed=True)
    parameters = solver_parameters(
        args.pc,
        args.velocity_pc,
        args.bfbt_inner_ksp,
        args.bfbt_inner_rtol,
        args.bfbt_mass_lumping,
        args.bfbt_weight_degree,
        args.case == "viscoplastic",
    )
    if args.case == "ala" and args.pc == "bfbt":
        parameters["fieldsplit_1"]["bfbt_nullspace_policy"] = "schur"
    solver = StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=bcs,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        solver_parameters=parameters,
    )
    return solution, solver


def timed_solve(solution, solver, warm_repeats):
    """Run cold and repeated warm solves with MPI-safe diagnostics."""
    comm = solution.function_space().mesh().comm
    mpi = import_module("mpi4py.MPI")

    comm.barrier()
    start = perf_counter()
    solver.solve()
    comm.barrier()
    cold_seconds = comm.allreduce(perf_counter() - start, op=mpi.MAX)

    snes = solver.solver.snes
    velocity_ksp, pressure_ksp = snes.ksp.pc.getFieldSplitSubKSP()

    velocity_counter = {"solves": 0, "iterations": 0, "failures": 0}
    pressure_counter = {"solves": 0, "iterations": 0, "failures": 0}

    def count_velocity(ksp, rhs, result):
        velocity_counter["solves"] += 1
        velocity_counter["iterations"] += ksp.getIterationNumber()
        velocity_counter["failures"] += int(ksp.getConvergedReason() <= 0)

    def count_pressure(ksp, rhs, result):
        pressure_counter["solves"] += 1
        pressure_counter["iterations"] += ksp.getIterationNumber()
        pressure_counter["failures"] += int(ksp.getConvergedReason() <= 0)

    velocity_ksp.setPostSolve(count_velocity)
    pressure_ksp.setPostSolve(count_pressure)

    warm_samples = []
    warm_work = []
    pressure_pc = pressure_ksp.pc.getPythonContext()
    uses_bfbt = pressure_pc.__class__.__name__ == "DensityAwareBFBTPC"
    for _ in range(warm_repeats):
        solution.assign(0)
        for counter in (velocity_counter, pressure_counter):
            counter.update(solves=0, iterations=0, failures=0)
        if uses_bfbt:
            bfbt_iterations_before = pressure_pc.inner_iterations_total
            bfbt_solves_before = pressure_pc.inner_solves_total
            bfbt_failures_before = pressure_pc.inner_failures_total

        comm.barrier()
        start = perf_counter()
        solver.solve()
        comm.barrier()
        elapsed = comm.allreduce(perf_counter() - start, op=mpi.MAX)
        warm_samples.append(elapsed)

        sample = {
            "velocity_solves": velocity_counter["solves"],
            "velocity_iterations": velocity_counter["iterations"],
            "velocity_failures": velocity_counter["failures"],
            "pressure_solves": pressure_counter["solves"],
            "pressure_iterations": pressure_counter["iterations"],
            "pressure_failures": pressure_counter["failures"],
            "snes_iterations": snes.getIterationNumber(),
            "snes_linear_iterations": snes.getLinearSolveIterations(),
        }
        if uses_bfbt:
            sample.update(
                bfbt_inner_iterations=(
                    pressure_pc.inner_iterations_total
                    - bfbt_iterations_before
                ),
                bfbt_inner_solves=(
                    pressure_pc.inner_solves_total - bfbt_solves_before
                ),
                bfbt_inner_failures=(
                    pressure_pc.inner_failures_total
                    - bfbt_failures_before
                ),
            )
        warm_work.append(sample)

    velocity_ksp.setPostSolve(None)
    pressure_ksp.setPostSolve(None)

    result = {
        "cold_seconds": cold_seconds,
        "warm_seconds": median(warm_samples),
        "warm_seconds_samples": warm_samples,
        "warm_work_samples": warm_work,
        "mpi_size": comm.size,
        "snes_iterations": snes.getIterationNumber(),
        "snes_linear_iterations": snes.getLinearSolveIterations(),
        "velocity_iterations_last": velocity_ksp.getIterationNumber(),
        "pressure_iterations_last": pressure_ksp.getIterationNumber(),
    }
    if uses_bfbt:
        result["bfbt_inner_iterations_last"] = pressure_pc.ksp.getIterationNumber()
        result["bfbt_inner_reasons_last_apply"] = pressure_pc.last_inner_reasons

    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    result["equation_residual"] = residual.dat.norm
    result["momentum_residual"] = residual.subfunctions[0].dat.norm
    result["continuity_residual"] = residual.subfunctions[1].dat.norm
    return result


def main(args):
    """Run one cold and one warm solve for the selected configuration."""
    solution, solver = build_case(args)
    result = vars(args) | timed_solve(solution, solver, args.warm_repeats)
    if solution.function_space().mesh().comm.rank == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(_arguments)
