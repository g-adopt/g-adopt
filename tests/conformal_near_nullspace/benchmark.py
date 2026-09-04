"""MPI-safe benchmark for 3-D spherical-shell near-nullspace candidates.

Run one candidate set per process or PBS job. PETSc options not recognised by
the argument parser are retained, so ``-log_view`` and scoped ``-info`` can be
used for detailed GAMG setup and coarse-grid diagnostics.  The benchmark can
either mirror the GPlates strong top-velocity condition or impose free slip
weakly at both radii.  The latter retains the three rotational null modes.
"""

import argparse
import json
import sys
from importlib import import_module
from statistics import median
from time import perf_counter


def argument_parser():
    """Return benchmark options while leaving PETSc options untouched."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approximation",
        choices=("boussinesq", "tala", "ala"),
        default="tala",
    )
    parser.add_argument(
        "--velocity-boundary",
        choices=("strong-top", "free-slip"),
        default="strong-top",
    )
    parser.add_argument(
        "--rotation-nullspace",
        choices=("auto", "exact", "omit"),
        default="auto",
        help=(
            "Register shell rotations as an exact nullspace, omit them from "
            "the exact nullspace, or select exact only for free-slip boundaries."
        ),
    )
    parser.add_argument(
        "--modes",
        choices=(
            "none",
            "rotations",
            "rigid-raw",
            "rigid-constrained",
            "conformal-balanced",
            "conformal-balanced-constrained",
            "conformal-ritz",
            "conformal-ritz-constrained",
            "conformal-raw",
            "conformal-constrained",
        ),
        required=True,
    )
    parser.add_argument("--refinement-level", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--contrast", type=float, default=1e4)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--velocity-rtol", type=float, default=5e-4)
    parser.add_argument("--pressure-rtol", type=float, default=5e-3)
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
    """Build one arm of the raw/constrained and six/ten-mode comparison."""
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
    return gadopt.ConformalKillingNearNullspace(**options)


def solver_parameters(args):
    """Return a production-shaped full-Schur/GAMG configuration."""
    if args.modes.startswith("conformal-balanced"):
        velocity_pc = "gadopt.BalancedConformalPC"
    elif args.modes.startswith("conformal-ritz"):
        velocity_pc = "gadopt.RitzConformalPC"
    else:
        velocity_pc = "gadopt.SPDAssembledPC"
    velocity_parameters = {
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
    }
    return {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0": velocity_parameters,
        "fieldsplit_1": {
            "ksp_type": "fgmres",
            "ksp_rtol": args.pressure_rtol,
            "ksp_max_it": 200,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "ksp",
            "Mp_ksp_ksp_type": "cg",
            "Mp_ksp_ksp_rtol": 1e-5,
            "Mp_ksp_pc_type": "sor",
        },
    }


def approximation_for(args, density, viscosity):
    """Return the selected approximation with matched viscosity and density."""
    approximation_name = getattr(args, "approximation", "tala")
    if approximation_name == "boussinesq":
        return gadopt.BoussinesqApproximation(Ra=1, mu=viscosity)
    approximation_type = (
        gadopt.TruncatedAnelasticLiquidApproximation
        if approximation_name == "tala"
        else gadopt.AnelasticLiquidApproximation
    )
    return approximation_type(
        Ra=1,
        Di=0.5,
        rho=density,
        mu=viscosity,
    )


def nullspaces_for(args, mixed_space, approximation, boundary):
    """Build exact right and transpose nullspaces for the chosen operator."""
    free_slip = getattr(args, "velocity_boundary", "strong-top") == "free-slip"
    rotation_nullspace = getattr(args, "rotation_nullspace", "auto")
    if rotation_nullspace == "exact" and not free_slip:
        raise ValueError(
            "Exact rotations are incompatible with the strong top-velocity boundary."
        )
    rotational = rotation_nullspace == "exact" or (
        rotation_nullspace == "auto" and free_slip
    )
    if getattr(args, "approximation", "tala") == "ala":
        nullspace = gadopt.create_stokes_nullspace(
            mixed_space,
            closed=True,
            rotational=rotational,
            ala_approximation=approximation,
            top_subdomain_id=boundary.top,
        )
        transpose_nullspace = gadopt.create_stokes_nullspace(
            mixed_space,
            closed=True,
            rotational=rotational,
        )
        return nullspace, transpose_nullspace

    nullspace = gadopt.create_stokes_nullspace(
        mixed_space,
        closed=True,
        rotational=rotational,
    )
    return nullspace, nullspace


def build_case(args):
    """Build a curved, high-contrast, frozen spherical-shell Stokes case."""
    rmin, rmax = 1.22, 2.22
    base_mesh = fd.CubedSphereMesh(
        rmin,
        refinement_level=args.refinement_level,
        degree=2,
    )
    mesh = fd.ExtrudedMesh(
        base_mesh,
        layers=args.layers,
        extrusion_type="radial",
    )
    mesh.cartesian = False
    boundary = gadopt.get_boundary_ids(mesh)

    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)

    X = fd.SpatialCoordinate(mesh)
    radius = fd.sqrt(fd.dot(X, X))
    temperature = fd.Function(temperature_space).interpolate(
        (rmax - radius) / (rmax - rmin)
        + 0.01 * X[0] * X[1] / radius**2
    )
    density = fd.Function(temperature_space).interpolate(
        fd.exp(0.5 * (rmax - radius) / (rmax - rmin))
    )
    viscosity = fd.Function(temperature_space).interpolate(
        fd.exp(
            fd.ln(fd.Constant(args.contrast))
            * (radius - rmin)
            / (rmax - rmin)
        )
    )
    approximation = approximation_for(args, density, viscosity)

    if getattr(args, "velocity_boundary", "strong-top") == "free-slip":
        bcs = {
            boundary.bottom: {"un": 0},
            boundary.top: {"un": 0},
        }
    else:
        imposed_surface_velocity = fd.as_vector((-X[1], X[0], 0)) * 1e-3
        bcs = {
            boundary.bottom: {"un": 0},
            boundary.top: {"u": imposed_surface_velocity},
        }
    nullspace, transpose_nullspace = nullspaces_for(
        args,
        mixed_space,
        approximation,
        boundary,
    )
    near_nullspace = near_nullspace_for(args.modes, mixed_space)
    solver = gadopt.StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=bcs,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        near_nullspace=near_nullspace,
        solver_parameters=solver_parameters(args),
    )
    return solution, solver, boundary, approximation


def candidate_diagnostics(solver, boundary):
    """Measure candidate count, volume strain, and boundary compatibility."""
    if solver.near_nullspace is None:
        return {
            "candidate_count": 0,
            "candidate_strain_energy": [],
            "candidate_top_trace": [],
            "candidate_bottom_normal": [],
        }

    velocity_basis = next(iter(solver.near_nullspace))
    modes = velocity_basis._vecs
    identity = fd.Identity(3)
    normal = fd.FacetNormal(solver.mesh)
    result = {
        "candidate_count": len(modes),
        "candidate_strain_energy": [],
        "candidate_top_trace": [],
        "candidate_bottom_normal": [],
    }
    for mode in modes:
        strain = fd.sym(fd.grad(mode)) - identity * fd.div(mode) / 3
        result["candidate_strain_energy"].append(
            fd.assemble(fd.inner(strain, strain) * fd.dx)
        )
        result["candidate_top_trace"].append(
            fd.assemble(
                fd.inner(mode, mode) * fd.ds_t(domain=solver.mesh)
            )
        )
        result["candidate_bottom_normal"].append(
            fd.assemble(
                fd.dot(mode, normal) ** 2
                * fd.ds_b(domain=solver.mesh)
            )
        )
    return result


def rotation_operator_diagnostics(solver, boundary, approximation, matrix):
    """Measure how closely shell rotations satisfy velocity and continuity blocks."""
    velocity_space = solver.solution_space.sub(0).collapse()
    pressure_space = solver.solution_space.sub(1).collapse()
    rotations = gadopt.rigid_body_modes(
        velocity_space,
        rotational=True,
    )._vecs
    matrix_norm = matrix.norm()
    velocity_residuals = []
    velocity_relative_residuals = []
    velocity_rayleigh_quotients = []
    for rotation in rotations:
        image = matrix.createVecLeft()
        with rotation.dat.vec_ro as vector:
            matrix.mult(vector, image)
            velocity_residuals.append(image.norm() / vector.norm())
            velocity_relative_residuals.append(
                image.norm() / (matrix_norm * vector.norm())
            )
            velocity_rayleigh_quotients.append(
                abs(vector.dot(image)) / vector.dot(vector).real
            )

    all_free_slip = all(
        "un" in solver.bcs[boundary_id]
        for boundary_id in (boundary.bottom, boundary.top)
    )
    if not all_free_slip:
        return {
            "rotation_velocity_block_residuals": velocity_residuals,
            "rotation_velocity_block_relative_residuals": (
                velocity_relative_residuals
            ),
            "rotation_velocity_block_rayleigh_quotients": (
                velocity_rayleigh_quotients
            ),
            "rotation_continuity_block_residuals": None,
            "rotation_boundary_normal_energies": None,
        }

    pressure_test = fd.TestFunction(pressure_space)
    rho = approximation.rho_continuity()
    normal = fd.FacetNormal(solver.mesh)
    boundary_measure = fd.ds_b(domain=solver.mesh) + fd.ds_t(domain=solver.mesh)
    continuity_residuals = []
    normal_energies = []
    for rotation in rotations:
        mode_norm = fd.assemble(fd.inner(rotation, rotation) * fd.dx) ** 0.5
        continuity_residual = fd.assemble(
            -pressure_test * fd.div(rho * rotation) * fd.dx
            + pressure_test
            * rho
            * fd.dot(normal, rotation)
            * boundary_measure
        )
        continuity_residuals.append(continuity_residual.dat.norm / mode_norm)
        normal_energies.append(
            fd.assemble(fd.dot(normal, rotation) ** 2 * boundary_measure)
        )
    return {
        "rotation_velocity_block_residuals": velocity_residuals,
        "rotation_velocity_block_relative_residuals": (
            velocity_relative_residuals
        ),
        "rotation_velocity_block_rayleigh_quotients": (
            velocity_rayleigh_quotients
        ),
        "rotation_continuity_block_residuals": continuity_residuals,
        "rotation_boundary_normal_energies": normal_energies,
    }


def timed_runs(solution, solver, boundary, approximation, warm_repeats):
    """Run cold and warm Stokes solves with total nested-work counters."""
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

    assembled_velocity = velocity_ksp.pc.getPythonContext().P.petscmat
    attached_near_nullspace = assembled_velocity.getNearNullSpace()
    attached_count = (
        len(attached_near_nullspace.getVecs())
        if attached_near_nullspace.handle != 0
        else 0
    )
    velocity_pc_context = velocity_ksp.pc.getPythonContext()
    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    result = {
        "mpi_size": comm.size,
        "velocity_dofs": solver.solution_space.sub(0).dim(),
        "pressure_dofs": solver.solution_space.sub(1).dim(),
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
        "ritz_eigenvalues": (
            velocity_pc_context.ritz_eigenvalues.tolist()
            if hasattr(velocity_pc_context, "ritz_eigenvalues")
            else None
        ),
        "ritz_relative_gap": getattr(
            velocity_pc_context,
            "ritz_relative_gap",
            None,
        ),
        "ritz_used_rigid_fallback": getattr(
            velocity_pc_context,
            "ritz_used_fallback",
            None,
        ),
        "ritz_minimum_relative_gap": getattr(
            velocity_pc_context,
            "ritz_minimum_relative_gap",
            None,
        ),
        "ritz_basis_orthonormality_error": getattr(
            velocity_pc_context,
            "complete_mode_orthonormality_error",
            None,
        ),
        "ritz_principal_angle_change": getattr(
            velocity_pc_context,
            "ritz_principal_angle_change",
            None,
        ),
        "assembled_velocity_block_size": assembled_velocity.getBlockSize(),
        "equation_residual": residual.dat.norm,
        "momentum_residual": residual.subfunctions[0].dat.norm,
        "continuity_residual": residual.subfunctions[1].dat.norm,
    }
    result.update(
        rotation_operator_diagnostics(
            solver,
            boundary,
            approximation,
            assembled_velocity,
        )
    )
    return result


def main(args):
    """Run one near-nullspace benchmark arm and print rank-zero JSON."""
    solution, solver, boundary, approximation = build_case(args)
    result = vars(args)
    result.update(candidate_diagnostics(solver, boundary))
    result.update(
        timed_runs(
            solution,
            solver,
            boundary,
            approximation,
            args.warm_repeats,
        )
    )
    if solution.function_space().mesh().comm.rank == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(_arguments)
