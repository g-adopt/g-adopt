"""Reproducible pressure-preconditioner benchmark for G-ADOPT Stokes solves.

Run one configuration per process, for example::

    python benchmark.py --case linear --contrast 1e6 --pc mass
    python benchmark.py --case tala --contrast 1e6 --pc bfbt
    python benchmark.py --case viscoplastic --pc bfbt

The first solve includes JIT compilation and cold PETSc setup. The reported
``warm_seconds`` is a second solve from the same initial state and is the
appropriate timing for comparing preconditioners. Correctness is checked
independently using the assembled equation residual.
"""

import argparse
import json
import sys
from importlib import import_module
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
    solver = StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=bcs,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        solver_parameters=solver_parameters(
            args.pc,
            args.velocity_pc,
            args.bfbt_inner_ksp,
            args.bfbt_inner_rtol,
            args.bfbt_mass_lumping,
            args.bfbt_weight_degree,
            args.case == "viscoplastic",
        ),
    )
    return solution, solver


def timed_solve(solution, solver):
    """Run cold and warm solves and return PETSc iteration diagnostics."""
    start = perf_counter()
    solver.solve()
    cold_seconds = perf_counter() - start

    solution.assign(0)
    start = perf_counter()
    solver.solve()
    warm_seconds = perf_counter() - start

    snes = solver.solver.snes
    velocity_ksp, pressure_ksp = snes.ksp.pc.getFieldSplitSubKSP()
    result = {
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "snes_iterations": snes.getIterationNumber(),
        "snes_linear_iterations": snes.getLinearSolveIterations(),
        "velocity_iterations_last": velocity_ksp.getIterationNumber(),
        "pressure_iterations_last": pressure_ksp.getIterationNumber(),
    }
    if pressure_ksp.pc.getPythonContext().__class__.__name__ == "DensityAwareBFBTPC":
        result["bfbt_inner_iterations_last"] = (
            pressure_ksp.pc.getPythonContext().ksp.getIterationNumber()
        )

    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    result["equation_residual"] = residual.dat.norm
    result["momentum_residual"] = residual.subfunctions[0].dat.norm
    result["continuity_residual"] = residual.subfunctions[1].dat.norm
    return result


def main(args):
    """Run one cold and one warm solve for the selected configuration."""
    solution, solver = build_case(args)
    result = vars(args) | timed_solve(solution, solver)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(_arguments)
