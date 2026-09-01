"""Local 2-D cylindrical analogue of the production GPlates TALA solve.

This benchmark keeps the annular geometry, radially varying reference density,
heterogeneous viscosity, strong surface velocity, weak free-slip base, pressure
gauge, and iterative velocity/pressure blocks.  It intentionally uses a
frozen analytic viscosity so that BFBT and PETSc hierarchy behaviour can be
understood before adding the production strain-rate-dependent rheology.

Run one pressure preconditioner per process, for example::

    python tests/bfbt/cylindrical_tala.py --pc mass
    python tests/bfbt/cylindrical_tala.py --pc bfbt
    mpiexec -n 2 python tests/bfbt/cylindrical_tala.py --pc bfbt

Unknown options are passed to PETSc.  The first solve includes JIT and setup;
reported warm solves restart from zero.  Timings are barrier-to-barrier
maximum-rank values and only rank zero writes JSON.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from statistics import median
from time import perf_counter


def argument_parser():
    """Return benchmark arguments while leaving PETSc options untouched."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc", choices=("mass", "bfbt"), required=True)
    parser.add_argument("--ncells", type=int, default=32)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--density-contrast", type=float, default=5665 / 3200)
    parser.add_argument("--density-space", choices=("dq", "cg"), default="cg")
    parser.add_argument("--radial-viscosity-contrast", type=float, default=1e2)
    parser.add_argument("--lateral-viscosity-contrast", type=float, default=1e4)
    parser.add_argument("--surface-velocity-amplitude", type=float, default=0.1)
    parser.add_argument("--azimuthal-mode", type=int, default=4)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--pressure-rtol", type=float, default=1e-6)
    parser.add_argument("--velocity-rtol", type=float, default=1e-8)
    parser.add_argument("--mass-inner-ksp", choices=("cg", "preonly"), default="cg")
    parser.add_argument("--mass-inner-pc", choices=("jacobi", "lu"), default="jacobi")
    parser.add_argument(
        "--bfbt-inner-ksp",
        choices=("fgmres", "preonly", "richardson"),
        default="fgmres",
    )
    parser.add_argument("--bfbt-inner-pc", choices=("gamg", "hypre", "lu"), default="gamg")
    parser.add_argument("--bfbt-inner-rtol", type=float, default=1e-2)
    parser.add_argument("--bfbt-right-inner-rtol", type=float)
    parser.add_argument("--bfbt-left-inner-rtol", type=float)
    parser.add_argument("--bfbt-inner-max-it", type=int, default=200)
    parser.add_argument("--bfbt-mass-lumping", choices=("diagonal", "rowsum"), default="diagonal")
    parser.add_argument("--bfbt-weight-degree", type=int, default=0)
    parser.add_argument("--bfbt-gamg-threshold", type=float)
    parser.add_argument("--bfbt-gamg-threshold-scale", type=float)
    parser.add_argument("--bfbt-gamg-coarse-eq-limit", type=int)
    parser.add_argument("--bfbt-gamg-process-eq-limit", type=int)
    parser.add_argument("--bfbt-gamg-agg-nsmooths", type=int)
    parser.add_argument("--bfbt-gamg-aggressive-coarsening", type=int)
    parser.add_argument(
        "--bfbt-gamg-aggressive-square-graph",
        choices=("on", "off"),
    )
    parser.add_argument("--json-file")
    return parser


ORIGINAL_ARGUMENTS = list(sys.argv[1:])
if __name__ == "__main__":
    _arguments, _petsc_arguments = argument_parser().parse_known_args()
    sys.argv = [sys.argv[0], *_petsc_arguments]
else:
    _arguments = None
    _petsc_arguments = []

# Executing a script inside ``tests/bfbt`` does not otherwise put the checkout
# root on sys.path.  An editable installation may point at a different G-ADOPT
# worktree, which would make every result scientifically ambiguous.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

fd = import_module("firedrake")
gadopt = import_module("gadopt")
MPI = import_module("mpi4py.MPI")
PETSc = import_module("petsc4py.PETSc")

if Path(gadopt.__file__).resolve().parents[1] != REPOSITORY_ROOT:
    raise RuntimeError(
        f"Imported G-ADOPT from {Path(gadopt.__file__).resolve()}, "
        f"not the benchmark checkout {REPOSITORY_ROOT}"
    )


RMIN = 1.208
RMAX = 2.208


def validate_arguments(args):
    """Reject configurations that do not define the intended test problem."""
    if args.ncells < 8 or args.nlayers < 2:
        raise ValueError("Use at least 8 azimuthal cells and 2 radial layers")
    if args.ncells % 2:
        raise ValueError("--ncells must be even for the parallel circle mesh")
    finite_values = {
        "density contrast": args.density_contrast,
        "radial viscosity contrast": args.radial_viscosity_contrast,
        "lateral viscosity contrast": args.lateral_viscosity_contrast,
        "surface velocity amplitude": args.surface_velocity_amplitude,
        "pressure rtol": args.pressure_rtol,
        "velocity rtol": args.velocity_rtol,
        "BFBT inner rtol": args.bfbt_inner_rtol,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if args.density_contrast < 1:
        raise ValueError("--density-contrast must be at least one")
    if args.radial_viscosity_contrast < 1:
        raise ValueError("--radial-viscosity-contrast must be at least one")
    if args.lateral_viscosity_contrast < 1:
        raise ValueError("--lateral-viscosity-contrast must be at least one")
    if args.azimuthal_mode < 1:
        raise ValueError("--azimuthal-mode must be positive")
    if args.warm_repeats < 1:
        raise ValueError("--warm-repeats must be positive")
    if not 0 < args.pressure_rtol < 1:
        raise ValueError("--pressure-rtol must be between zero and one")
    if not 0 < args.velocity_rtol < 1:
        raise ValueError("--velocity-rtol must be between zero and one")
    if not 0 < args.bfbt_inner_rtol < 1:
        raise ValueError("--bfbt-inner-rtol must be between zero and one")
    for side, tolerance in (
        ("right", args.bfbt_right_inner_rtol),
        ("left", args.bfbt_left_inner_rtol),
    ):
        if tolerance is not None and (
            not math.isfinite(tolerance) or not 0 < tolerance < 1
        ):
            raise ValueError(
                f"--bfbt-{side}-inner-rtol must be finite and between zero and one"
            )
    if args.bfbt_inner_max_it < 1:
        raise ValueError("--bfbt-inner-max-it must be positive")
    if args.bfbt_weight_degree < 0:
        raise ValueError("--bfbt-weight-degree must be non-negative")
    if args.bfbt_gamg_threshold is not None and (
        not math.isfinite(args.bfbt_gamg_threshold)
        or args.bfbt_gamg_threshold < 0
    ):
        raise ValueError("--bfbt-gamg-threshold must be finite and non-negative")
    if args.bfbt_gamg_threshold_scale is not None and (
        not math.isfinite(args.bfbt_gamg_threshold_scale)
        or args.bfbt_gamg_threshold_scale <= 0
    ):
        raise ValueError(
            "--bfbt-gamg-threshold-scale must be finite and positive"
        )
    if (
        args.bfbt_gamg_coarse_eq_limit is not None
        and args.bfbt_gamg_coarse_eq_limit < 1
    ):
        raise ValueError("--bfbt-gamg-coarse-eq-limit must be positive")
    if (
        args.bfbt_gamg_process_eq_limit is not None
        and args.bfbt_gamg_process_eq_limit < 1
    ):
        raise ValueError("--bfbt-gamg-process-eq-limit must be positive")
    if args.bfbt_gamg_agg_nsmooths is not None and args.bfbt_gamg_agg_nsmooths < 0:
        raise ValueError("--bfbt-gamg-agg-nsmooths must be non-negative")
    if (
        args.bfbt_gamg_aggressive_coarsening is not None
        and args.bfbt_gamg_aggressive_coarsening < 0
    ):
        raise ValueError("--bfbt-gamg-aggressive-coarsening must be non-negative")
    gamg_values = (
        args.bfbt_gamg_threshold,
        args.bfbt_gamg_threshold_scale,
        args.bfbt_gamg_coarse_eq_limit,
        args.bfbt_gamg_process_eq_limit,
        args.bfbt_gamg_agg_nsmooths,
        args.bfbt_gamg_aggressive_coarsening,
        args.bfbt_gamg_aggressive_square_graph,
    )
    if any(value is not None for value in gamg_values) and (
        args.pc != "bfbt" or args.bfbt_inner_pc != "gamg"
    ):
        raise ValueError("BFBT GAMG options require --pc bfbt --bfbt-inner-pc gamg")


def solver_parameters(args):
    """Build matched full-Schur configurations with iterative velocity solves."""
    pressure_parameters = {
        "ksp_type": "fgmres",
        "ksp_rtol": args.pressure_rtol,
        "ksp_max_it": 300,
        "pc_type": "python",
    }
    if args.pc == "mass":
        pressure_parameters.update(
            {
                "pc_python_type": "firedrake.MassInvPC",
                "Mp_pc_type": "ksp",
                "Mp_ksp_ksp_type": args.mass_inner_ksp,
                "Mp_ksp_ksp_rtol": 1e-8,
                "Mp_ksp_ksp_error_if_not_converged": True,
                "Mp_ksp_pc_type": args.mass_inner_pc,
            }
        )
        if args.mass_inner_pc == "lu":
            pressure_parameters["Mp_ksp_pc_factor_mat_solver_type"] = "mumps"
    else:
        pressure_parameters.update(
            {
                "pc_python_type": "gadopt.DensityAwareBFBTPC",
                "bfbt_ksp_type": args.bfbt_inner_ksp,
                "bfbt_ksp_rtol": args.bfbt_inner_rtol,
                "bfbt_ksp_max_it": args.bfbt_inner_max_it,
                "bfbt_pc_type": args.bfbt_inner_pc,
                "bfbt_mass_lumping": args.bfbt_mass_lumping,
                "bfbt_weight_degree": args.bfbt_weight_degree,
                "bfbt_nullspace_policy": "verified",
            }
        )
        optional_gamg_parameters = {
            "bfbt_pc_gamg_threshold": args.bfbt_gamg_threshold,
            "bfbt_pc_gamg_threshold_scale": args.bfbt_gamg_threshold_scale,
            "bfbt_pc_gamg_coarse_eq_limit": args.bfbt_gamg_coarse_eq_limit,
            "bfbt_pc_gamg_process_eq_limit": args.bfbt_gamg_process_eq_limit,
            "bfbt_pc_gamg_agg_nsmooths": args.bfbt_gamg_agg_nsmooths,
            "bfbt_pc_gamg_aggressive_coarsening": (
                args.bfbt_gamg_aggressive_coarsening
            ),
        }
        pressure_parameters.update(
            {key: value for key, value in optional_gamg_parameters.items() if value is not None}
        )
        if args.bfbt_right_inner_rtol is not None:
            pressure_parameters["bfbt_right_ksp_rtol"] = (
                args.bfbt_right_inner_rtol
            )
        if args.bfbt_left_inner_rtol is not None:
            pressure_parameters["bfbt_left_ksp_rtol"] = (
                args.bfbt_left_inner_rtol
            )
        if args.bfbt_gamg_aggressive_square_graph is not None:
            pressure_parameters["bfbt_pc_gamg_aggressive_square_graph"] = (
                args.bfbt_gamg_aggressive_square_graph == "on"
            )
        if args.bfbt_inner_pc == "hypre":
            pressure_parameters["bfbt_pc_hypre_type"] = "boomeramg"
        elif args.bfbt_inner_pc == "lu":
            pressure_parameters["bfbt_pc_factor_mat_solver_type"] = "mumps"
        if args.bfbt_inner_ksp == "richardson":
            # A norm-free Richardson solve applies exactly max_it residual
            # corrections. This is a fixed linear map and avoids subsidiary
            # convergence tests and Krylov orthogonalisation.
            pressure_parameters["bfbt_ksp_norm_type"] = "none"

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
            "ksp_max_it": 300,
            "pc_type": "python",
            "pc_python_type": "gadopt.SPDAssembledPC",
            "assembled_pc_type": "gamg",
            "assembled_mg_levels_ksp_type": "chebyshev",
            "assembled_mg_levels_pc_type": "jacobi",
            "assembled_pc_gamg_threshold": 0.01,
        },
        "fieldsplit_1": pressure_parameters,
    }


def build_case(args):
    """Construct the frozen-viscosity annular TALA problem."""
    circle = fd.CircleManifoldMesh(args.ncells, radius=RMIN, degree=2)
    mesh = fd.ExtrudedMesh(
        circle,
        layers=args.nlayers,
        layer_height=(RMAX - RMIN) / args.nlayers,
        extrusion_type="radial",
    )
    mesh.cartesian = False
    boundary = gadopt.get_boundary_ids(mesh)

    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "DQ", 2)
    density_space = fd.FunctionSpace(mesh, args.density_space.upper(), 2)
    viscosity_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = fd.MixedFunctionSpace([velocity_space, pressure_space])
    solution = fd.Function(mixed_space, name="Stokes")

    coordinates = fd.SpatialCoordinate(mesh)
    radius = fd.sqrt(fd.inner(coordinates, coordinates))
    depth = (RMAX - radius) / (RMAX - RMIN)
    theta = fd.atan2(coordinates[1], coordinates[0])
    radial_envelope = fd.sin(fd.pi * depth) ** 2

    density = fd.Function(density_space, name="ReferenceDensity").interpolate(
        fd.exp(fd.ln(fd.Constant(args.density_contrast)) * depth)
    )
    radial_viscosity = fd.exp(
        fd.ln(fd.Constant(args.radial_viscosity_contrast)) * depth
    )
    lateral_viscosity = fd.exp(
        fd.ln(fd.Constant(args.lateral_viscosity_contrast))
        * radial_envelope
        * (1 + fd.cos(args.azimuthal_mode * theta))
        / 2
    )
    viscosity = fd.Function(viscosity_space, name="FrozenViscosity").interpolate(
        radial_viscosity * lateral_viscosity
    )

    conductive_temperature = fd.ln(RMAX / radius) / fd.ln(RMAX / RMIN)
    temperature = fd.Function(temperature_space, name="Temperature").interpolate(
        conductive_temperature
        + 0.05
        * fd.cos(args.azimuthal_mode * theta)
        * fd.sin(fd.pi * depth)
    )
    approximation = gadopt.TruncatedAnelasticLiquidApproximation(
        1.0,
        0.9492824,
        rho=density,
        mu=viscosity,
    )

    tangential = fd.as_vector((-coordinates[1] / radius, coordinates[0] / radius))
    surface_velocity = fd.Function(velocity_space, name="SurfaceVelocity").interpolate(
        args.surface_velocity_amplitude * fd.sin(2 * theta) * tangential
    )
    boundary_conditions = {
        boundary.bottom: {"un": 0},
        boundary.top: {"u": surface_velocity},
    }
    nullspace = gadopt.create_stokes_nullspace(
        mixed_space,
        closed=True,
        rotational=False,
    )
    near_nullspace = gadopt.create_stokes_nullspace(
        mixed_space,
        closed=False,
        rotational=True,
        translations=[0, 1],
    )
    solver = gadopt.StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=boundary_conditions,
        solver_parameters=solver_parameters(args),
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        near_nullspace=near_nullspace,
    )
    fields = {
        "density": density,
        "viscosity": viscosity,
        "temperature": temperature,
        "surface_velocity": surface_velocity,
        "pressure_space": pressure_space,
    }
    return solution, solver, fields


def global_extrema(function):
    """Return extrema safely when an MPI rank owns no entries."""
    comm = function.comm
    values = function.dat.data_ro
    local_minimum = float(values.min()) if values.size else float("inf")
    local_maximum = float(values.max()) if values.size else float("-inf")
    return (
        comm.allreduce(local_minimum, op=MPI.MIN),
        comm.allreduce(local_maximum, op=MPI.MAX),
    )


def gamg_metrics(pc):
    """Return hierarchy and complexity metrics for a configured GAMG PC."""
    if pc.getType() != "gamg":
        return {"pc_type": pc.getType()}
    try:
        level_count = pc.getMGLevels()
        level_sizes = []
        level_nonzeros = []
        for level in range(level_count):
            level_ksp = pc.getMGCoarseSolve() if level == 0 else pc.getMGSmoother(level)
            _, level_matrix = level_ksp.getOperators()
            level_sizes.append(level_matrix.getSize()[0])
            level_nonzeros.append(
                level_matrix.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"]
            )
        return {
            "pc_type": "gamg",
            "levels": level_count,
            "level_sizes": level_sizes,
            "level_nonzeros": level_nonzeros,
            "grid_complexity": sum(level_sizes) / level_sizes[-1],
            "operator_complexity": sum(level_nonzeros) / level_nonzeros[-1],
        }
    except Exception as error:  # PETSc configuration diagnostics must not hide a valid solve.
        return {"pc_type": "gamg", "diagnostic_error": f"{type(error).__name__}: {error}"}


def ksp_norm_type_name(norm_type):
    """Return a stable JSON name for a PETSc KSP norm type."""
    names = {
        int(PETSc.KSP.NormType.NONE): "none",
        int(PETSc.KSP.NormType.PRECONDITIONED): "preconditioned",
        int(PETSc.KSP.NormType.UNPRECONDITIONED): "unpreconditioned",
        int(PETSc.KSP.NormType.NATURAL): "natural",
    }
    return names.get(int(norm_type), f"unknown({int(norm_type)})")


def analytic_pressure_vector(
    matrix,
    pressure_space,
    *,
    left=False,
    mode=1,
    radial_mode=1,
):
    """Interpolate a physical pressure probe independent of MPI numbering."""
    coordinates = fd.SpatialCoordinate(pressure_space.mesh())
    radius = fd.sqrt(fd.inner(coordinates, coordinates))
    depth = (RMAX - radius) / (RMAX - RMIN)
    theta = fd.atan2(coordinates[1], coordinates[0])
    probe = fd.Function(pressure_space).interpolate(
        fd.sin(mode * theta) * fd.sin(radial_mode * fd.pi * depth)
        + 0.3
        * fd.cos((mode + 2) * theta)
        * fd.sin((radial_mode + 1) * fd.pi * depth)
    )
    vector = matrix.createVecLeft() if left else matrix.createVecRight()
    with probe.dat.vec_ro as probe_vector:
        probe_vector.copy(vector)
    return vector


def bfbt_operator_metrics(context, pressure_space, side="right"):
    """Compare exact and auxiliary weighted pressure operators on fixed probes."""
    bundle = context.sides[side]
    exact = bundle.exact_pressure_laplacian
    auxiliary = bundle.pressure_laplacian.petscmat
    pressure = analytic_pressure_vector(exact, pressure_space, mode=1)
    second_pressure = analytic_pressure_vector(
        exact,
        pressure_space,
        mode=1,
        radial_mode=2,
    )
    left_pressure = analytic_pressure_vector(
        exact,
        pressure_space,
        left=True,
        mode=3,
    )
    exact_action = exact.createVecLeft()
    exact_second_action = exact.createVecLeft()
    exact_transpose_action = exact.createVecRight()
    auxiliary_action = auxiliary.createVecLeft()
    auxiliary_transpose_action = auxiliary.createVecRight()
    difference = exact.createVecLeft()
    constant = exact.createVecRight()
    left_constant = exact.createVecLeft()
    constant_action = exact.createVecLeft()
    left_constant_action = exact.createVecRight()
    auxiliary_constant_action = auxiliary.createVecLeft()
    auxiliary_left_constant_action = auxiliary.createVecRight()
    vectors = (
        pressure,
        second_pressure,
        left_pressure,
        exact_action,
        exact_second_action,
        exact_transpose_action,
        auxiliary_action,
        auxiliary_transpose_action,
        difference,
        constant,
        left_constant,
        constant_action,
        left_constant_action,
        auxiliary_constant_action,
        auxiliary_left_constant_action,
    )
    try:
        nullspace = exact.getNullSpace()
        if nullspace.handle != 0:
            nullspace.remove(pressure)
            nullspace.remove(second_pressure)
        transpose_nullspace = exact.getTransposeNullSpace()
        if transpose_nullspace.handle != 0:
            transpose_nullspace.remove(left_pressure)
        exact.mult(pressure, exact_action)
        exact.mult(second_pressure, exact_second_action)
        exact.multTranspose(left_pressure, exact_transpose_action)
        auxiliary.mult(pressure, auxiliary_action)
        auxiliary.multTranspose(left_pressure, auxiliary_transpose_action)
        exact_action.copy(difference)
        difference.axpy(-1, auxiliary_action)
        exact_norm = exact_action.norm()
        auxiliary_norm = auxiliary_action.norm()
        exact_transpose_norm = exact_transpose_action.norm()
        auxiliary_transpose_norm = auxiliary_transpose_action.norm()
        denominator = exact_norm * auxiliary_norm
        symmetry_denominator = (
            abs(pressure.dot(exact_second_action))
            + abs(second_pressure.dot(exact_action))
        )
        constant.set(1)
        constant.normalize()
        left_constant.set(1)
        left_constant.normalize()
        exact.mult(constant, constant_action)
        exact.multTranspose(left_constant, left_constant_action)
        auxiliary.mult(constant, auxiliary_constant_action)
        auxiliary.multTranspose(left_constant, auxiliary_left_constant_action)
        auxiliary_scale = exact_action.dot(auxiliary_action) / auxiliary_action.dot(
            auxiliary_action
        )
        exact_action.copy(difference)
        difference.axpy(-auxiliary_scale, auxiliary_action)
        return {
            "exact_auxiliary_best_scaled_relative_difference": difference.norm() / exact_norm,
            "exact_auxiliary_action_cosine": exact_action.dot(auxiliary_action) / denominator,
            "exact_auxiliary_action_scale": auxiliary_scale,
            "exact_symmetry_defect": (
                abs(pressure.dot(exact_second_action) - second_pressure.dot(exact_action))
                / symmetry_denominator
                if symmetry_denominator
                else 0.0
            ),
            "exact_constant_right_action": constant_action.norm(),
            "exact_constant_left_action": left_constant_action.norm(),
            "auxiliary_constant_right_action": auxiliary_constant_action.norm(),
            "auxiliary_constant_left_action": auxiliary_left_constant_action.norm(),
            "exact_constant_right_relative_action": constant_action.norm() / exact_norm,
            "exact_constant_left_relative_action": (
                left_constant_action.norm() / exact_transpose_norm
            ),
            "auxiliary_constant_right_relative_action": (
                auxiliary_constant_action.norm() / auxiliary_norm
            ),
            "auxiliary_constant_left_relative_action": (
                auxiliary_left_constant_action.norm() / auxiliary_transpose_norm
            ),
            "algebraic_side": side,
            "right_nullspace_is_exact": bundle.right_nullspace_is_exact,
            "left_nullspace_is_exact": bundle.left_nullspace_is_exact,
            "auxiliary_right_nullspace_is_exact": bundle.auxiliary_right_nullspace_is_exact,
            "auxiliary_left_nullspace_is_exact": bundle.auxiliary_left_nullspace_is_exact,
            "left_nullspace_source": bundle.left_nullspace_source,
            "left_nullspace_fallback_used": bundle.left_nullspace_fallback_used,
            "exact_left_nullspace_attached": bundle.exact_left_nullspace_attached,
            "auxiliary_left_nullspace_attached": bundle.auxiliary_left_nullspace_attached,
            "right_nullspace_residual": bundle.right_nullspace_residual,
            "left_nullspace_residual": bundle.left_nullspace_residual,
            "auxiliary_right_nullspace_residual": (
                bundle.auxiliary_right_nullspace_residual
            ),
            "auxiliary_left_nullspace_residual": (
                bundle.auxiliary_left_nullspace_residual
            ),
            "nullspace_test_threshold": bundle.nullspace_test_threshold,
            "nullspace_test_auxiliary_operator_scale": (
                bundle.nullspace_test_auxiliary_operator_scale
            ),
        }
    finally:
        for vector in vectors:
            vector.destroy()


def timed_solve(solution, solver, fields, warm_repeats):
    """Run cold and warm solves and return correctness and work diagnostics."""
    mesh = solution.function_space().mesh()
    comm = mesh.comm

    solution.assign(0)
    solver.solution_old.assign(0)
    initial_residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    initial_equation_residual = initial_residual.dat.norm
    initial_momentum_residual = initial_residual.subfunctions[0].dat.norm
    initial_continuity_residual = initial_residual.subfunctions[1].dat.norm

    comm.barrier()
    start = perf_counter()
    solver.solve()
    comm.barrier()
    cold_seconds = comm.allreduce(perf_counter() - start, op=MPI.MAX)

    snes = solver.solver.snes
    velocity_ksp, pressure_ksp = snes.ksp.pc.getFieldSplitSubKSP()
    velocity_counter = {"solves": 0, "iterations": 0, "failures": 0}
    pressure_counter = {
        "solves": 0,
        "iterations": 0,
        "failures": 0,
        "convergence_histories": [],
        "residual_norms": [],
    }

    def count_solve(counter, ksp):
        counter["solves"] += 1
        counter["iterations"] += ksp.getIterationNumber()
        counter["failures"] += int(ksp.getConvergedReason() <= 0)

    def count_pressure(ksp, rhs, result):
        count_solve(pressure_counter, ksp)
        pressure_counter["convergence_histories"].append(
            list(ksp.getConvergenceHistory())
        )
        pressure_counter["residual_norms"].append(ksp.getResidualNorm())

    velocity_ksp.setPostSolve(lambda ksp, rhs, result: count_solve(velocity_counter, ksp))
    pressure_ksp.setConvergenceHistory(length=1000, reset=True)
    pressure_ksp.setPostSolve(count_pressure)
    pressure_context = pressure_ksp.pc.getPythonContext()
    uses_bfbt = pressure_context.__class__.__name__ == "DensityAwareBFBTPC"
    nested_ksp = None
    if uses_bfbt:
        nested_ksp = pressure_context.ksp
    elif pressure_context.pc.getType() == "ksp":
        nested_ksp = pressure_context.pc.getKSP()
    nested_counter = {
        "solves": 0,
        "iterations": 0,
        "failures": 0,
        "iterations_by_solve": [],
        "reasons": [],
    }
    if nested_ksp is not None:
        nested_ksp.setConvergenceHistory(length=1000, reset=True)

        def count_nested(ksp, rhs, result):
            iterations = ksp.getIterationNumber()
            reason = int(ksp.getConvergedReason())
            nested_counter["solves"] += 1
            nested_counter["iterations"] += iterations
            nested_counter["failures"] += int(reason <= 0)
            nested_counter["iterations_by_solve"].append(iterations)
            nested_counter["reasons"].append(reason)

        nested_ksp.setPostSolve(count_nested)

    warm_seconds = []
    warm_work = []
    for _ in range(warm_repeats):
        solution.assign(0)
        solver.solution_old.assign(0)
        velocity_counter.update(solves=0, iterations=0, failures=0)
        pressure_counter.update(
            solves=0,
            iterations=0,
            failures=0,
            convergence_histories=[],
            residual_norms=[],
        )
        nested_counter.update(
            solves=0,
            iterations=0,
            failures=0,
            iterations_by_solve=[],
            reasons=[],
        )
        if uses_bfbt:
            side_iterations_before = dict(
                pressure_context.inner_iterations_by_side
            )
            side_solves_before = dict(pressure_context.inner_solves_by_side)
            inner_before = pressure_context.inner_iterations_total
            inner_solves_before = pressure_context.inner_solves_total
            inner_failures_before = pressure_context.inner_failures_total

        comm.barrier()
        start = perf_counter()
        solver.solve()
        comm.barrier()
        warm_seconds.append(comm.allreduce(perf_counter() - start, op=MPI.MAX))

        sample = {
            "velocity_solves": velocity_counter["solves"],
            "velocity_iterations": velocity_counter["iterations"],
            "velocity_failures": velocity_counter["failures"],
            "pressure_solves": pressure_counter["solves"],
            "pressure_iterations": pressure_counter["iterations"],
            "pressure_failures": pressure_counter["failures"],
            "pressure_convergence_histories": pressure_counter[
                "convergence_histories"
            ],
            "pressure_residual_norms": pressure_counter["residual_norms"],
            "snes_iterations": snes.getIterationNumber(),
            "snes_linear_iterations": snes.getLinearSolveIterations(),
        }
        if nested_ksp is not None:
            sample.update(
                pressure_pc_inner_solves=nested_counter["solves"],
                pressure_pc_inner_iterations=nested_counter["iterations"],
                pressure_pc_inner_failures=nested_counter["failures"],
                pressure_pc_inner_iterations_by_solve=nested_counter[
                    "iterations_by_solve"
                ],
                pressure_pc_inner_reasons=nested_counter["reasons"],
            )
        if uses_bfbt:
            sample.update(
                bfbt_inner_solves=pressure_context.inner_solves_total - inner_solves_before,
                bfbt_inner_iterations=pressure_context.inner_iterations_total - inner_before,
                bfbt_inner_failures=pressure_context.inner_failures_total - inner_failures_before,
                bfbt_inner_iterations_by_side={
                    side: pressure_context.inner_iterations_by_side[side]
                    - side_iterations_before[side]
                    for side in ("right", "left")
                },
                bfbt_inner_solves_by_side={
                    side: pressure_context.inner_solves_by_side[side]
                    - side_solves_before[side]
                    for side in ("right", "left")
                },
            )
        warm_work.append(sample)

    velocity_ksp.setPostSolve(None)
    pressure_ksp.setPostSolve(None)
    if nested_ksp is not None:
        nested_ksp.setPostSolve(None)

    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    density_minimum, density_maximum = global_extrema(fields["density"])
    viscosity_minimum, viscosity_maximum = global_extrema(fields["viscosity"])
    density_jump_l2 = fd.sqrt(
        fd.assemble(
            fd.jump(fields["density"]) ** 2
            * (fd.dS_h(domain=mesh) + fd.dS_v(domain=mesh))
        )
    )
    velocity, pressure = solution.subfunctions
    volume = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
    pressure_mean = fd.assemble(pressure * fd.dx) / volume
    pressure_quotient_norm = fd.sqrt(
        fd.assemble((pressure - pressure_mean) ** 2 * fd.dx)
    )
    normal = fd.FacetNormal(mesh)
    top_velocity_error = fd.sqrt(
        fd.assemble(
            fd.inner(velocity - fields["surface_velocity"], velocity - fields["surface_velocity"])
            * fd.ds_t(domain=mesh)
        )
    )
    bottom_normal_flux = fd.sqrt(
        fd.assemble(fd.inner(velocity, normal) ** 2 * fd.ds_b(domain=mesh))
    )
    strong_continuity_residual = fd.sqrt(
        fd.assemble(fd.div(fields["density"] * velocity) ** 2 * fd.dx)
    )
    result = {
        "cold_seconds": cold_seconds,
        "warm_seconds": median(warm_seconds),
        "warm_seconds_samples": warm_seconds,
        "warm_work_samples": warm_work,
        "mpi_size": comm.size,
        "velocity_dofs": solution.function_space().sub(0).dim(),
        "pressure_dofs": solution.function_space().sub(1).dim(),
        "density_minimum": density_minimum,
        "density_maximum": density_maximum,
        "density_jump_l2": density_jump_l2,
        "viscosity_minimum": viscosity_minimum,
        "viscosity_maximum": viscosity_maximum,
        "equation_residual": residual.dat.norm,
        "momentum_residual": residual.subfunctions[0].dat.norm,
        "continuity_residual": residual.subfunctions[1].dat.norm,
        "initial_equation_residual": initial_equation_residual,
        "initial_momentum_residual": initial_momentum_residual,
        "initial_continuity_residual": initial_continuity_residual,
        "relative_equation_residual": residual.dat.norm / initial_equation_residual,
        "relative_momentum_residual": (
            residual.subfunctions[0].dat.norm / initial_momentum_residual
            if initial_momentum_residual
            else None
        ),
        "continuity_residual_over_initial_equation_residual": (
            residual.subfunctions[1].dat.norm / initial_equation_residual
        ),
        "velocity_norm": velocity.dat.norm,
        "pressure_quotient_l2_norm": pressure_quotient_norm,
        "top_velocity_l2_error": top_velocity_error,
        "bottom_normal_flux_l2": bottom_normal_flux,
        "strong_density_continuity_l2": strong_continuity_residual,
        "velocity_ksp_reason": int(velocity_ksp.getConvergedReason()),
        "pressure_ksp_reason": int(pressure_ksp.getConvergedReason()),
        "velocity_ksp_type": velocity_ksp.getType(),
        "pressure_ksp_type": pressure_ksp.getType(),
        "pressure_pc_type": pressure_ksp.pc.getType(),
        "velocity_pc": gamg_metrics(velocity_ksp.pc.getPythonContext().pc),
    }
    if nested_ksp is not None:
        nested_norm_type = nested_ksp.getNormType()
        nested_rtol, nested_atol, nested_divtol, nested_max_it = (
            nested_ksp.getTolerances()
        )
        measures_nested_residual = (
            nested_norm_type != PETSc.KSP.NormType.NONE
        )
        result.update(
            {
                "pressure_pc_inner_ksp_type": nested_ksp.getType(),
                "pressure_pc_inner_ksp_norm_type": ksp_norm_type_name(
                    nested_norm_type
                ),
                "pressure_pc_inner_ksp_prefix": nested_ksp.getOptionsPrefix(),
                "pressure_pc_inner_ksp_reason": int(nested_ksp.getConvergedReason()),
                "pressure_pc_inner_ksp_iterations": nested_ksp.getIterationNumber(),
                "pressure_pc_inner_ksp_rtol": nested_rtol,
                "pressure_pc_inner_ksp_atol": nested_atol,
                "pressure_pc_inner_ksp_divtol": nested_divtol,
                "pressure_pc_inner_ksp_max_it": nested_max_it,
                "pressure_pc_inner_ksp_residual_measured": (
                    measures_nested_residual
                ),
                "pressure_pc_inner_ksp_residual": (
                    nested_ksp.getResidualNorm()
                    if measures_nested_residual
                    else None
                ),
                "pressure_pc_inner_last_convergence_history": (
                    list(nested_ksp.getConvergenceHistory())
                    if measures_nested_residual
                    else None
                ),
                "pressure_pc_inner_pc": gamg_metrics(nested_ksp.pc),
            }
        )
    if uses_bfbt:
        result["bfbt_operators"] = bfbt_operator_metrics(
            pressure_context,
            fields["pressure_space"],
        )
    return result


def main(args):
    """Run one cylindrical pressure-PC configuration."""
    validate_arguments(args)
    solution, solver, fields = build_case(args)
    result = vars(args) | timed_solve(solution, solver, fields, args.warm_repeats)
    result["geometry"] = "2d_annulus"
    result["rmin"] = RMIN
    result["rmax"] = RMAX
    result["gadopt_path"] = str(Path(gadopt.__file__).resolve())
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    try:
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_dirty"] = None
    source_paths = (
        REPOSITORY_ROOT / "gadopt/preconditioners.py",
        REPOSITORY_ROOT / "tests/unit/test_bfbt_preconditioner.py",
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "tests/bfbt/run_cylindrical_tala.sh",
    )
    result["tested_source_sha256"] = {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
    }
    result["petsc_version"] = list(PETSc.Sys.getVersion())
    result["command"] = [sys.executable, str(Path(__file__).resolve()), *ORIGINAL_ARGUMENTS]
    result["petsc_arguments"] = _petsc_arguments
    result["velocity_options_prefix"] = (
        solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[0].getOptionsPrefix()
    )
    result["pressure_options_prefix"] = (
        solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1].getOptionsPrefix()
    )
    output = json.dumps(result, sort_keys=True)
    if solution.function_space().mesh().comm.rank == 0:
        if args.json_file:
            output_path = Path(args.json_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output + "\n")
        print(output)


if __name__ == "__main__":
    main(_arguments)
