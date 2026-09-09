"""Benchmark substituted and coupled Burgers internal-variable formulations.

This is intentionally separate from ``3d_sphere_burgers.py``: it uses the same
geometry, material profiles, loading, and boundary conditions, but omits output and
diagnostics that would contaminate timings.

Run, for example, with::

    python benchmark_internal_variable_solvers.py --reflevel 2 --dg0-layers 1 --steps 2

To use assembled rank-local LU solves for both internal fields in the Schur
strategy, add ``--schur-internal-variables --internal-solver local-lu``.
The default ``--internal-solver cg-sor`` retains the original iterative solves.
Local LU is exact only when internal equations have no off-rank couplings;
``--internal-rtol`` is unused for this direct strategy.

The first step includes any lazy compilation/assembly costs.  Use ``--steps 3`` or
more and compare both the first-step and subsequent-step timings.
"""

import argparse
from copy import deepcopy
import functools
import json
import pprint
from time import perf_counter

import numpy as np
from firedrake import SCPC
from mpi4py import MPI

from gadopt import (
    CompressibleInternalVariableApproximation,
    Constant,
    CoupledInternalVariableSolver,
    CubedSphereMesh,
    ExtrudedMesh,
    Function,
    FunctionSpace,
    InternalVariableSolver,
    DeleteParam,
    MixedVectorSpaceBasis,
    PETSc,
    SpatialCoordinate,
    TensorFunctionSpace,
    VectorFunctionSpace,
    atan2,
    errornorm,
    get_boundary_ids,
    norm,
    pi,
    rigid_body_modes,
    sqrt,
    tanh,
)
from gadopt.utility import extruded_layer_heights, initialise_background_field


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflevel", type=int, default=2)
    parser.add_argument("--dg0-layers", type=int, default=1)
    parser.add_argument("--dt-years", type=float, default=1000)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--outer-rtol", type=float, default=1e-5)
    parser.add_argument("--internal-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--internal-solver", choices=("cg-sor", "local-lu"), default="cg-sor",
        help="Internal fields: CG/SOR or assembled rank-local block Jacobi/LU.",
    )
    parser.add_argument("--scaling-factor", type=float, default=1)
    parser.add_argument(
        "--split-type",
        choices=("multiplicative", "symmetric_multiplicative"),
        default="symmetric_multiplicative",
    )
    parser.add_argument("--constant-jacobian", action="store_true")
    parser.add_argument("--symmetric-tensor", action="store_true")
    parser.add_argument("--device-type", choices=("HOST", "CUDA", "HIP"), default="HOST")
    parser.add_argument("--telescope-factor", type=int, default=1)
    parser.add_argument("--show-solver-parameters", action="store_true")
    parser.add_argument("--show-ksp-reasons", action="store_true")
    parser.add_argument("--group-internal-variables", action="store_true")
    parser.add_argument("--schur-internal-variables", action="store_true")
    parser.add_argument("--static-condensation", action="store_true")
    parser.add_argument(
        "--order", choices=("substituted-first", "coupled-first"), default="substituted-first"
    )
    args = parser.parse_args()
    if args.static_condensation and args.internal_solver != "cg-sor":
        parser.error("--internal-solver does not apply to --static-condensation")
    selected_block_variants = sum(
        (
            args.group_internal_variables,
            args.schur_internal_variables,
            args.static_condensation,
        )
    )
    if selected_block_variants > 1:
        parser.error(
            "--group-internal-variables, --schur-internal-variables, and "
            "--static-condensation are mutually exclusive"
        )
    if args.static_condensation and (
        args.device_type != "HOST" or args.telescope_factor != 1
    ):
        parser.error("--static-condensation currently supports CPU execution only")
    return args


def max_over_ranks(value, comm):
    return comm.allreduce(value, op=MPI.MAX)


def timed(comm, callback):
    comm.Barrier()
    start = perf_counter()
    result = callback()
    comm.Barrier()
    return max_over_ranks(perf_counter() - start, comm), result


def iteration_counts(solver):
    """Return SNES iterations and accumulated linear iterations for the last solve."""
    snes = solver.solver.snes
    ksp = snes.ksp
    context = ksp.pc.getPythonContext() if ksp.pc.getType() == "python" else None
    if isinstance(context, NullspaceAwareSCPC):
        return snes.getIterationNumber(), context.condensed_ksp.getIterationNumber()
    return snes.getIterationNumber(), snes.getLinearSolveIterations()


def has_option(options, key):
    return key in options or any(
        has_option(value, key) for value in options.values() if isinstance(value, dict)
    )


def has_option_value(options, key, expected):
    return options.get(key) == expected or any(
        has_option_value(value, key, expected)
        for value in options.values()
        if isinstance(value, dict)
    )


def validate_solver_options(substituted, coupled, args):
    """Catch benchmark configurations that silently stop being comparable."""
    seq = substituted.solver_parameters
    cpl = coupled.solver_parameters
    assert seq["pc_python_type"] == "gadopt.SPDAssembledPC"
    if args.static_condensation:
        assert cpl["pc_python_type"] == "__main__.NullspaceAwareSCPC"
        assert cpl["pc_sc_eliminate_fields"] == "1,2"
        assert cpl["condensed_field"]["ksp_type"] == "cg"
        assert cpl["condensed_field"]["pc_type"] == "gamg"
        return
    coupled_displacement = (
        cpl["fieldsplit_1"]
        if args.schur_internal_variables
        else cpl["fieldsplit_0"]
    )
    assert coupled_displacement["pc_python_type"] == "gadopt.SPDAssembledPC"
    for displacement_options in (seq, coupled_displacement):
        assert has_option_value(displacement_options, "ksp_type", "cg")
        assert has_option_value(displacement_options, "pc_type", "gamg")
    if args.schur_internal_variables:
        assert cpl["pc_type"] == "fieldsplit"
        assert cpl["pc_fieldsplit_type"] == "schur"
        assert cpl["pc_fieldsplit_0_fields"] == "1,2"
        assert cpl["pc_fieldsplit_1_fields"] == "0"
        internal_group = cpl["fieldsplit_0"]
        assert internal_group["fieldsplit_0"] == internal_group["fieldsplit_1"]
    elif args.group_internal_variables:
        assert cpl["pc_fieldsplit_0_fields"] == "0"
        assert cpl["pc_fieldsplit_1_fields"] == "1,2"
        assert (
            cpl["fieldsplit_1"]["fieldsplit_0"]
            == cpl["fieldsplit_1"]["fieldsplit_1"]
        )
    else:
        assert cpl["fieldsplit_1"] == cpl["fieldsplit_2"]
    internal = (
        cpl["fieldsplit_0"]["fieldsplit_0"] if args.schur_internal_variables
        else cpl["fieldsplit_1"]["fieldsplit_0"] if args.group_internal_variables
        else cpl["fieldsplit_1"]
    )
    if args.internal_solver == "local-lu":
        assert internal["ksp_type"] == "preonly"
        assert internal["assembled_pc_type"] == "bjacobi"
        assert internal["assembled_sub_ksp_type"] == "preonly"
        assert internal["assembled_sub_pc_type"] == "lu"
    assert not has_option(seq, "ksp_monitor")
    assert not has_option(cpl, "ksp_monitor")
    if not args.show_ksp_reasons:
        assert not has_option(seq, "ksp_converged_reason")
        assert not has_option(cpl, "ksp_converged_reason")


def material_field(space, values, radii, coordinate, name):
    field = Function(space, name=name)
    initialise_background_field(field, values, coordinate, radii)
    return field


def make_problem(args):
    radii = np.asarray([6371e3, 6301e3, 5951e3, 5701e3, 3480e3])
    depth = radii[0] - radii[-1]
    radii /= depth

    surface_mesh = CubedSphereMesh(
        radii[-1], refinement_level=args.reflevel, degree=2, name="surface_mesh"
    )
    layer_heights = extruded_layer_heights(args.dg0_layers, radii)
    mesh = ExtrudedMesh(
        surface_mesh,
        layers=len(layer_heights),
        layer_height=layer_heights,
        extrusion_type="radial",
    )
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    tensor_kwargs = {"symmetry": True} if args.symmetric_tensor else {}
    S = TensorFunctionSpace(mesh, "DQ", 1, **tensor_kwargs)
    DG0 = FunctionSpace(mesh, "DG", 0)
    DG1 = FunctionSpace(mesh, "DG", 1)
    X = SpatialCoordinate(mesh)

    density_scale = 4500
    shear_scale = 1e11
    viscosity_scale = 1e21
    maxwell_scale = viscosity_scale / shear_scale
    year = 8.64e4 * 365.25
    dt = Constant(args.dt_years * year / maxwell_scale)

    density = material_field(
        DG0, np.asarray([3037, 3438, 3871, 4978]) / density_scale,
        radii, X, "density"
    )
    shear_values = 0.5 * np.asarray([0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]) / shear_scale
    shear_1 = material_field(DG0, shear_values, radii, X, "shear modulus 1")
    shear_2 = material_field(DG0, shear_values, radii, X, "shear modulus 2")
    bulk = material_field(DG0, 2 * shear_values, radii, X, "bulk modulus")
    viscosity_values_log = np.log10(
        0.5 * np.asarray([1e40, 1e21, 1e21, 2e21]) / viscosity_scale
    )
    viscosity_1 = material_field(DG1, viscosity_values_log, radii, X, "viscosity 1")
    viscosity_2 = material_field(DG1, viscosity_values_log, radii, X, "viscosity 2")
    viscosity_1.interpolate(10**viscosity_1)
    viscosity_2.interpolate(10**viscosity_2)

    B_mu = Constant(density_scale * depth * 9.815 / shear_scale)
    distance_from_axis = sqrt(X[0] ** 2 + X[1] ** 2)
    colatitude = atan2(distance_from_axis, X[2])
    disc_halfwidth = (2 * pi / 360) * 10
    surface_dx = 200e3
    surface_resolution = 2 * pi / (2 * pi * radii[0] * depth / surface_dx)
    disc = 0.5 * (1 - tanh((abs(colatitude) - disc_halfwidth) / (2 * surface_resolution)))
    ice_load = B_mu * (931 / density_scale) * (1000 / depth) * disc
    bcs = {
        boundary.bottom: {"un": 0},
        boundary.top: {"normal_stress": ice_load, "free_surface": {}},
    }

    def approximation():
        # Solvers set approximation.mu, so each formulation needs its own instance.
        return CompressibleInternalVariableApproximation(
            bulk_modulus=bulk,
            density=density,
            shear_modulus=[shear_1, shear_2],
            viscosity=[viscosity_1, viscosity_2],
            B_mu=B_mu,
            bulk_shear_ratio=1.94,
        )

    return mesh, V, S, dt, bcs, approximation


def make_nullspaces(V, S, Z=None):
    exact_u = rigid_body_modes(V, rotational=True)
    near_u = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])
    if Z is None:
        return exact_u, near_u
    # S is a placeholder basis for each tensor field, not a pressure nullspace.
    exact = MixedVectorSpaceBasis(Z, [exact_u, Z.sub(1), Z.sub(2)])
    near = MixedVectorSpaceBasis(Z, [near_u, Z.sub(1), Z.sub(2)])
    return exact, near


def gpu_parameters(args):
    result = {"device_type": args.device_type}
    if args.telescope_factor != 1:
        result["telescope_factor"] = args.telescope_factor
    return result


class SchurCoupledInternalVariableSolver(CoupledInternalVariableSolver):
    """Move the maintained displacement split behind an internal-variable Schur block."""

    def set_solver_options(self, solver_preset, solver_extras, gpu_extras):
        super().set_solver_options(solver_preset, solver_extras, gpu_extras)
        displacement_options = deepcopy(self.solver_parameters["fieldsplit_0"])
        internal_options = deepcopy(self.solver_parameters["fieldsplit_1"])
        self.solver_parameters |= {
            "pc_fieldsplit_type": "schur",
            # Match G-ADOPT's maintained full Schur factorisation convention.
            "pc_fieldsplit_schur_type": "full",
            # Precondition the exact Schur action with its displacement block.
      #      "pc_fieldsplit_schur_precondition": "a11",
            "pc_fieldsplit_0_fields": "1,2",
            "pc_fieldsplit_1_fields": "0",
            "fieldsplit_0": internal_options,
            "fieldsplit_1": displacement_options,
        }


class NullspaceAwareSCPC(SCPC):
    """SCPC for two uncoupled eliminated fields, with a near-nullspace."""

    def condensed_system(self, A, rhs, elim_fields, prefix, pc):
        from firedrake.slate import AssembledVector
        from firedrake.slate.static_condensation.la_utils import LAContext

        if elim_fields != [1, 2]:
            raise ValueError("This SCPC variant expects to eliminate fields 1 and 2")

        blocks = A.blocks
        vectors = AssembledVector(rhs).blocks
        inverse_1 = blocks[1, 1].inv
        inverse_2 = blocks[2, 2].inv
        condensed_operator = (
            blocks[0, 0]
            - blocks[0, 1] * inverse_1 * blocks[1, 0]
            - blocks[0, 2] * inverse_2 * blocks[2, 0]
        )
        condensed_rhs = (
            vectors[0]
            - blocks[0, 1] * inverse_1 * vectors[1]
            - blocks[0, 2] * inverse_2 * vectors[2]
        )
        inverse_blocks = (inverse_1, inverse_2)
        return (
            LAContext(condensed_operator, condensed_rhs, (0,)),
            inverse_blocks,
        )

    def local_solver_calls(self, A, rhs, solution, elim_fields, inverse_blocks):
        from firedrake.assemble import get_assembler
        from firedrake.slate import AssembledVector

        calls = []
        displacement = AssembledVector(solution.subfunctions[0])
        for field, inverse in zip(elim_fields, inverse_blocks):
            field_rhs = AssembledVector(rhs.subfunctions[field])
            reduced_rhs = field_rhs - A.blocks[field, 0] * displacement
            local_solution = inverse * reduced_rhs
            calls.append(
                functools.partial(
                    get_assembler(
                        local_solution,
                        form_compiler_parameters=self.cxt.fc_params,
                    ).assemble,
                    tensor=solution.subfunctions[field],
                )
            )
        return calls

    def initialize(self, pc):
        super().initialize(pc)
        callback = self.cxt.appctx.get("condensed_field_near_nullspace")
        if callback is not None:
            condensed_space = self.weight.function_space()
            near_nullspace = callback(condensed_space)
            self.S.petscmat.setNearNullSpace(near_nullspace.nullspace())
            if hasattr(self, "S_pc"):
                self.S_pc.petscmat.setNearNullSpace(near_nullspace.nullspace())


class StaticCondensationCoupledInternalVariableSolver(CoupledInternalVariableSolver):
    """Eliminate the DG internal variables locally with Slate SCPC."""

    def set_solver_options(self, solver_preset, solver_extras, gpu_extras):
        super().set_solver_options(solver_preset, solver_extras, gpu_extras)
        displacement_options = self.solver_parameters["fieldsplit_0"]
        condensed_options = {
            "mat_type": "aij",
            "ksp_type": "cg",
            "ksp_rtol": displacement_options["ksp_rtol"],
            **deepcopy(displacement_options["assembled"]),
        }
        if "ksp_converged_reason" in displacement_options:
            condensed_options["ksp_converged_reason"] = None

        snes_options = {
            key: value
            for key, value in self.solver_parameters.items()
            if key.startswith("snes_")
        }
        self.solver_parameters = snes_options | {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "__main__.NullspaceAwareSCPC",
            "pc_sc_eliminate_fields": "1,2",
            "condensed_field": condensed_options,
        }
        self.appctx |= {
            "condensed_field_nullspace": lambda V: rigid_body_modes(
                V, rotational=True
            ),
            "condensed_field_near_nullspace": lambda V: rigid_body_modes(
                V, rotational=True, translations=[0, 1, 2]
            ),
        }


def construct_substituted(problem, args):
    mesh, V, S, dt, bcs, approximation = problem
    u = Function(V, name="substituted displacement")
    internal = [Function(S, name=f"substituted internal variable {i}") for i in (1, 2)]
    exact, near = make_nullspaces(V, S)
    solver = InternalVariableSolver(
        u,
        approximation(),
        dt=dt,
        internal_variables=internal,
        bcs=bcs,
        solver_parameters="iterative",
        solver_parameters_extra={
            "snes_monitor": DeleteParam,
            "ksp_monitor": DeleteParam,
            "ksp_converged_reason": None if args.show_ksp_reasons else DeleteParam,
            "ksp_rtol": args.outer_rtol,
        },
        gpu_extra_parameters=gpu_parameters(args),
        constant_jacobian=args.constant_jacobian,
        nullspace=exact,
        transpose_nullspace=exact,
        near_nullspace=near,
    )
    return solver, u


def construct_coupled(problem, args):
    mesh, V, S, dt, bcs, approximation = problem
    Z = V * S * S
    z = Function(Z, name="coupled Burgers solution")
    exact, near = make_nullspaces(V, S, Z)
    internal_options = {
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "sor",
        "ksp_rtol": args.internal_rtol,
        "ksp_monitor": DeleteParam,
        "ksp_converged_reason": None if args.show_ksp_reasons else DeleteParam,
    }
    if args.internal_solver == "local-lu":
        # Exact for element-local internal equations: no off-rank couplings.
        # PETSc reuses the local factors until the assembled matrix changes.
        internal_options |= {
            "ksp_type": "preonly",
            "ksp_rtol": DeleteParam,
            "assembled_pc_type": "bjacobi",
            "assembled_sub_ksp_type": "preonly",
            "assembled_sub_pc_type": "lu",
        }
    extras = {
        "snes_monitor": DeleteParam,
        "ksp_monitor": DeleteParam,
        "ksp_converged_reason": None if args.show_ksp_reasons else DeleteParam,
        "ksp_rtol": args.outer_rtol,
        "pc_fieldsplit_type": args.split_type,
        # PETSc creates one split per mixed field.  Configure both Burgers
        # internal-variable splits, rather than only fieldsplit_1 as the preset does.
        "fieldsplit_1": internal_options,
        "fieldsplit_2": internal_options,
    }
    if args.group_internal_variables or args.schur_internal_variables:
        extras |= {
            "pc_fieldsplit_0_fields": "0",
            "pc_fieldsplit_1_fields": "1,2",
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",
                "ksp_rtol": DeleteParam,
                "pc_python_type": DeleteParam,
                "assembled_pc_type": DeleteParam,
                "ksp_converged_reason": (
                    None if args.show_ksp_reasons else DeleteParam
                ),
                "fieldsplit_0": internal_options,
                "fieldsplit_1": internal_options,
            },
            "fieldsplit_2": DeleteParam,
        }
    if args.show_ksp_reasons:
        extras["fieldsplit_0"] = {"ksp_converged_reason": None}
    if args.constant_jacobian:
        # This is a fixed-dt Newtonian problem.  The coupled preset otherwise keeps
        # Newton line-search options even when the variational problem is linear.
        extras["snes_type"] = "ksponly"
    if args.static_condensation:
        solver_class = StaticCondensationCoupledInternalVariableSolver
    elif args.schur_internal_variables:
        solver_class = SchurCoupledInternalVariableSolver
    else:
        solver_class = CoupledInternalVariableSolver
    solver = solver_class(
        z,
        approximation(),
        dt=dt,
        scaling_factor=args.scaling_factor,
        bcs=bcs,
        solver_parameters="iterative",
        solver_parameters_extra=extras,
        gpu_extra_parameters=gpu_parameters(args),
        constant_jacobian=args.constant_jacobian,
        nullspace=exact,
        transpose_nullspace=exact,
        near_nullspace=near,
    )
    return solver, z.subfunctions[0]


def main():
    args = parse_args()
    problem = make_problem(args)
    mesh, V, S, *_ = problem
    constructors = {
        "substituted": construct_substituted,
        "coupled": construct_coupled,
    }
    order = (
        ("substituted", "coupled")
        if args.order == "substituted-first"
        else ("coupled", "substituted")
    )
    results = {}
    solvers = {}
    displacements = {}
    displacement_differences = []
    solve_stages = {
        name: PETSc.Log.Stage(f"burgers_{name}_solve") for name in constructors
    }

    for name in order:
        setup_time, (solver, displacement) = timed(
            mesh.comm, lambda name=name: constructors[name](problem, args)
        )
        solvers[name] = solver
        displacements[name] = displacement
        results[name] = {"setup_seconds": setup_time, "steps": []}

    validate_solver_options(solvers["substituted"], solvers["coupled"], args)
    if args.show_solver_parameters and mesh.comm.rank == 0:
        for name in ("substituted", "coupled"):
            print(
                f"GADOPT_BURGERS_SOLVER_PARAMETERS {name}\n"
                + pprint.pformat(solvers[name].solver_parameters, sort_dicts=True)
            )

    for step in range(args.steps):
        for name in order:
            def solve(name=name):
                with solve_stages[name]:
                    solvers[name].solve()

            solve_time, _ = timed(mesh.comm, solve)
            outer, inner = iteration_counts(solvers[name])
            results[name]["steps"].append(
                {"step": step + 1, "solve_seconds": solve_time,
                 "snes_iterations": outer,
                 "top_level_ksp_iterations": inner}
            )
        absolute_difference = errornorm(
            displacements["substituted"], displacements["coupled"]
        )
        reference_norm = norm(displacements["substituted"])
        displacement_differences.append(
            {
                "step": step + 1,
                "l2": absolute_difference,
                "relative_l2": absolute_difference
                / max(float(reference_norm), 1e-300),
            }
        )

    u_substituted = displacements["substituted"]
    u_coupled = displacements["coupled"]
    absolute_difference = errornorm(u_substituted, u_coupled)
    reference_norm = norm(u_substituted)
    results["comparison"] = {
        "displacement_l2_difference": absolute_difference,
        "displacement_relative_l2_difference": absolute_difference / max(float(reference_norm), 1e-300),
        "steps": displacement_differences,
        "displacement_dofs": V.dim(),
        "internal_variable_dofs_each": S.dim(),
        "substituted_total_stored_dofs": V.dim() + 2 * S.dim(),
        "coupled_system_dofs": V.dim() + 2 * S.dim(),
    }
    results["configuration"] = vars(args)
    if mesh.comm.rank == 0:
        print("GADOPT_BURGERS_BENCHMARK " + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
