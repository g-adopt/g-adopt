"""Compare velocity near-nullspace candidates for cubic TALA and ALA.

This benchmark is the direct 3-D Cartesian analogue of G-ADOPT's square
compressible-convection demos.  It solves one frozen-temperature Stokes
problem with fieldsplit/GAMG.  Each invocation runs one near-nullspace arm so
PETSc setup and timing remain isolated.
"""

import argparse
import json
import sys
from importlib import import_module
from statistics import median
from time import perf_counter


def argument_parser():
    """Return benchmark arguments while preserving PETSc command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--approximation", choices=("tala", "ala"), required=True)
    parser.add_argument(
        "--modes",
        choices=(
            "none",
            "rotations",
            "rigid-raw",
            "rigid-constrained",
            "conformal-balanced",
            "conformal-raw",
            "conformal-constrained",
        ),
        required=True,
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--viscosity-contrast", type=float, default=1.0)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--velocity-rtol", type=float, default=1.0e-7)
    parser.add_argument("--pressure-rtol", type=float, default=1.0e-7)
    return parser


if __name__ == "__main__":
    _arguments, _petsc_arguments = argument_parser().parse_known_args()
    sys.argv = [sys.argv[0], *_petsc_arguments]
else:
    _arguments = None

fd = import_module("firedrake")
gadopt = import_module("gadopt")
mpi = import_module("mpi4py.MPI")


def near_nullspace_for(modes, mixed_space):
    """Return one control, rigid-body, or conformal candidate space."""
    if modes == "none":
        return None
    if modes == "rigid-raw":
        return gadopt.create_stokes_nullspace(
            mixed_space,
            closed=False,
            rotational=True,
            translations=[0, 1, 2],
        )

    options = {
        "rotational": True,
        "translations": (0, 1, 2),
        "dilation": modes.startswith("conformal"),
        "special_conformal": modes.startswith("conformal"),
        "constrain_strong_bcs": modes.endswith("constrained"),
    }
    if modes == "rotations":
        options.update(
            translations=(),
            dilation=False,
            special_conformal=False,
            constrain_strong_bcs=False,
        )
    return gadopt.ConformalKillingNearNullspace(
        **options,
    )


def solver_parameters(args):
    """Return the same fieldsplit/GAMG solver for every benchmark arm."""
    velocity_pc = (
        "gadopt.BalancedConformalPC"
        if args.modes == "conformal-balanced"
        else "gadopt.SPDAssembledPC"
    )
    return {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0": {
            "ksp_type": "cg",
            "ksp_rtol": args.velocity_rtol,
            "ksp_max_it": 1000,
            "pc_type": "python",
            "pc_python_type": velocity_pc,
            "assembled_pc_type": "gamg",
            "assembled_mg_levels_pc_type": "sor",
            "assembled_pc_gamg_threshold": 0.01,
            "assembled_pc_gamg_square_graph": 100,
            "assembled_pc_gamg_coarse_eq_limit": 1000,
            "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
        },
        "fieldsplit_1": {
            "ksp_type": "fgmres",
            "ksp_rtol": args.pressure_rtol,
            "ksp_max_it": 500,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "ksp",
            "Mp_ksp_ksp_type": "cg",
            "Mp_ksp_ksp_rtol": 1.0e-7,
            "Mp_ksp_pc_type": "sor",
        },
    }


def build_case(args):
    """Build one frozen Stokes solve from the compressible demo physics."""
    mesh = fd.UnitCubeMesh(args.n, args.n, args.n, hexahedral=True)
    mesh.cartesian = True
    boundary = gadopt.get_boundary_ids(mesh)

    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)

    x = fd.SpatialCoordinate(mesh)
    rayleigh = fd.Constant(1.0e5)
    dissipation = fd.Constant(0.5)
    surface_temperature = fd.Constant(0.091)
    density = fd.Function(temperature_space).interpolate(
        fd.exp((1 - x[2]) * dissipation)
    )
    reference_temperature = fd.Function(temperature_space).interpolate(
        surface_temperature * fd.exp((1 - x[2]) * dissipation)
        - surface_temperature
    )
    viscosity = fd.Function(temperature_space).interpolate(
        fd.exp(
            fd.ln(fd.Constant(args.viscosity_contrast)) * (1 - x[2])
        )
    )
    temperature = fd.Function(temperature_space).interpolate(
        (1 - (surface_temperature * fd.exp(dissipation) - surface_temperature))
        * (
            (1 - x[2])
            + 0.05
            * fd.cos(fd.pi * x[0])
            * fd.cos(fd.pi * x[1])
            * fd.sin(fd.pi * x[2])
        )
    )

    approximation_type = (
        gadopt.TruncatedAnelasticLiquidApproximation
        if args.approximation == "tala"
        else gadopt.AnelasticLiquidApproximation
    )
    approximation = approximation_type(
        rayleigh,
        dissipation,
        rho=density,
        Tbar=reference_temperature,
        mu=viscosity,
    )

    if args.approximation == "ala":
        nullspace = gadopt.create_stokes_nullspace(
            mixed_space,
            closed=True,
            rotational=False,
            ala_approximation=approximation,
            top_subdomain_id=boundary.top,
        )
        transpose_nullspace = gadopt.create_stokes_nullspace(
            mixed_space,
            closed=True,
            rotational=False,
        )
    else:
        nullspace = gadopt.create_stokes_nullspace(
            mixed_space,
            closed=True,
            rotational=False,
        )
        transpose_nullspace = nullspace

    boundary_conditions = {
        boundary.bottom: {"uz": 0},
        boundary.top: {"uz": 0},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
        boundary.front: {"uy": 0},
        boundary.back: {"uy": 0},
    }
    near_nullspace = near_nullspace_for(args.modes, mixed_space)
    solver = gadopt.StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=boundary_conditions,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        near_nullspace=near_nullspace,
        solver_parameters=solver_parameters(args),
    )
    return solution, solver, approximation


def candidate_energies(solver, approximation):
    """Return the volume energy quotient of every supplied velocity candidate."""
    if solver.near_nullspace is None:
        return []
    modes = next(iter(solver.near_nullspace))._vecs
    energies = []
    for mode in modes:
        numerator = fd.assemble(
            fd.inner(fd.grad(mode), approximation.stress(mode)) * fd.dx
        )
        denominator = fd.assemble(fd.inner(mode, mode) * fd.dx)
        energies.append(numerator / denominator)
    return energies


def timed_runs(solution, solver, warm_repeats):
    """Run cold and warm solves and count every nested fieldsplit iteration."""
    comm = solution.function_space().mesh().comm
    comm.barrier()
    start = perf_counter()
    solver.solve()
    comm.barrier()
    cold_seconds = comm.allreduce(perf_counter() - start, op=mpi.MAX)

    velocity_ksp, pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()
    counters = {
        "velocity_solves": 0,
        "velocity_iterations": 0,
        "velocity_failures": 0,
        "pressure_solves": 0,
        "pressure_iterations": 0,
        "pressure_failures": 0,
    }

    def count_velocity(ksp, rhs, result):
        counters["velocity_solves"] += 1
        counters["velocity_iterations"] += ksp.getIterationNumber()
        counters["velocity_failures"] += int(ksp.getConvergedReason() <= 0)

    def count_pressure(ksp, rhs, result):
        counters["pressure_solves"] += 1
        counters["pressure_iterations"] += ksp.getIterationNumber()
        counters["pressure_failures"] += int(ksp.getConvergedReason() <= 0)

    velocity_ksp.setPostSolve(count_velocity)
    pressure_ksp.setPostSolve(count_pressure)
    warm_seconds = []
    warm_work = []
    for _ in range(warm_repeats):
        solution.assign(0)
        for key in counters:
            counters[key] = 0
        comm.barrier()
        start = perf_counter()
        solver.solve()
        comm.barrier()
        warm_seconds.append(
            comm.allreduce(perf_counter() - start, op=mpi.MAX)
        )
        warm_work.append(dict(counters))

    velocity_ksp.setPostSolve(None)
    pressure_ksp.setPostSolve(None)
    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    assembled_velocity = velocity_ksp.pc.getPythonContext().P.petscmat
    attached = assembled_velocity.getNearNullSpace()
    attached_count = len(attached.getVecs()) if attached.handle != 0 else 0
    velocity_pc_context = velocity_ksp.pc.getPythonContext()
    return {
        "cold_seconds": cold_seconds,
        "warm_seconds": median(warm_seconds),
        "warm_seconds_samples": warm_seconds,
        "warm_work_samples": warm_work,
        "attached_velocity_near_nullspace_count": attached_count,
        "balanced_coarse_mode_count": len(
            getattr(velocity_pc_context, "_conformal_modes", ())
        ),
        "balanced_coarse_condition_number": getattr(
            velocity_pc_context,
            "coarse_condition_number",
            None,
        ),
        "equation_residual": residual.dat.norm,
        "momentum_residual": residual.subfunctions[0].dat.norm,
        "continuity_residual": residual.subfunctions[1].dat.norm,
    }


def main(args):
    """Run one benchmark arm and print rank-zero JSON."""
    solution, solver, approximation = build_case(args)
    result = vars(args)
    result["mpi_size"] = solution.function_space().mesh().comm.size
    result["velocity_dofs"] = solver.solution_space.sub(0).dim()
    result["pressure_dofs"] = solver.solution_space.sub(1).dim()
    result["candidate_energy_quotients"] = candidate_energies(
        solver,
        approximation,
    )
    result.update(timed_runs(solution, solver, args.warm_repeats))
    if solution.function_space().mesh().comm.rank == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(_arguments)
