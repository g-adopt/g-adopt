import firedrake as fd
import numpy as np

from gadopt import (
    BalancedConformalPC,
    ConformalKillingNearNullspace,
    RitzConformalPC,
    StokesSolver,
    TruncatedAnelasticLiquidApproximation,
    create_stokes_nullspace,
    get_boundary_ids,
)


def build_conformal_pc_solver(
    pc_python_type="gadopt.BalancedConformalPC",
    near_nullspace=None,
    ritz_min_relative_gap=None,
):
    """Build a small TALA problem using a conformal velocity PC."""
    mesh = fd.UnitCubeMesh(2, 2, 2, hexahedral=True)
    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)

    x = fd.SpatialCoordinate(mesh)
    temperature = fd.Function(temperature_space).interpolate(
        1 - x[2] + 0.05 * fd.cos(fd.pi * x[0]) * fd.sin(fd.pi * x[2])
    )
    density = fd.Function(temperature_space).interpolate(
        fd.exp(0.5 * (1 - x[2]))
    )
    viscosity = fd.Function(temperature_space).interpolate(
        fd.exp(fd.ln(fd.Constant(4.0)) * (1 - x[2]))
    )
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=1.0e5,
        Di=0.5,
        rho=density,
        mu=viscosity,
    )
    nullspace = create_stokes_nullspace(mixed_space, closed=True)
    boundary_conditions = {
        boundary.bottom: {"uz": 0},
        boundary.top: {"uz": 0},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
        boundary.front: {"uy": 0},
        boundary.back: {"uy": 0},
    }
    velocity_parameters = {
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-7,
        "ksp_max_it": 300,
        "pc_type": "python",
        "pc_python_type": pc_python_type,
        "assembled_pc_type": "gamg",
        "assembled_mg_levels_pc_type": "sor",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
        "assembled_pc_gamg_coarse_eq_limit": 1000,
        "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
    }
    if ritz_min_relative_gap is not None:
        velocity_parameters["ritz_min_relative_gap"] = (
            ritz_min_relative_gap
        )

    solver = StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=boundary_conditions,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        near_nullspace=(
            ConformalKillingNearNullspace()
            if near_nullspace is None
            else near_nullspace
        ),
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_0": velocity_parameters,
            "fieldsplit_1": {
                "ksp_type": "fgmres",
                "ksp_rtol": 1.0e-7,
                "pc_type": "python",
                "pc_python_type": "firedrake.MassInvPC",
                "Mp_pc_type": "lu",
            },
        },
    )
    return solution, solver, viscosity, x


def test_balanced_conformal_pc_uses_six_plus_four_modes():
    """The inner GAMG and balanced correction receive six and four modes."""
    solution, solver, viscosity, coordinates = build_conformal_pc_solver()
    solver.solve()

    velocity_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[0]
    context = velocity_ksp.pc.getPythonContext()
    assert isinstance(context, BalancedConformalPC)

    matrix = context.P.petscmat
    assert len(matrix.getNearNullSpace().getVecs()) == 6
    assert len(context._complete_near_nullspace.getVecs()) == 10
    assert len(context._conformal_modes) == 4
    assert np.allclose(context.coarse_matrix, context.coarse_matrix.T.conj())
    assert np.linalg.eigvalsh(context.coarse_matrix).min() > 0
    assert np.isfinite(context.coarse_condition_number)

    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    assert residual.dat.norm < 1.0e-4

    # Verify the balanced implementation and its transpose agree, including
    # the configured inner GAMG application.
    source = matrix.createVecRight()
    applied = matrix.createVecRight()
    transpose_applied = matrix.createVecRight()
    source.setRandom()
    velocity_ksp.pc.apply(source, applied)
    velocity_ksp.pc.applyTranspose(source, transpose_applied)
    transpose_applied.axpy(-1, applied)
    assert transpose_applied.norm() < 1.0e-12 * applied.norm()

    # A coefficient change must rebuild the replicated coarse operator while
    # retaining the six-mode near-nullspace on GAMG.
    old_coarse_matrix = context.coarse_matrix.copy()
    viscosity.interpolate(2 + coordinates[0] + coordinates[1] + coordinates[2])
    solution.assign(0)
    solver.solve()
    assert not np.allclose(context.coarse_matrix, old_coarse_matrix)
    assert len(matrix.getNearNullSpace().getVecs()) == 6


def test_ritz_conformal_pc_selects_and_updates_six_modes():
    """The Ritz PC selects six modes and refreshes them after reassembly."""
    solution, solver, viscosity, coordinates = build_conformal_pc_solver(
        "gadopt.RitzConformalPC"
    )
    solver.solve()

    velocity_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[0]
    context = velocity_ksp.pc.getPythonContext()
    assert isinstance(context, RitzConformalPC)

    matrix = context.P.petscmat
    assert len(matrix.getNearNullSpace().getVecs()) == 6
    assert len(context._complete_near_nullspace.getVecs()) == 10
    assert context.ritz_matrix.shape == (10, 10)
    assert context.ritz_coefficients.shape == (10, 6)
    assert np.allclose(context.ritz_matrix, context.ritz_matrix.T.conj())
    assert np.all(np.diff(context.ritz_eigenvalues) >= 0)
    assert np.isfinite(context.ritz_relative_gap)
    assert context.ritz_relative_gap >= 0
    assert not context.ritz_used_fallback
    assert context.complete_mode_orthonormality_error < 1.0e-12

    selected_gram = np.asarray(
        [mode.mDot(context._retained_modes) for mode in context._retained_modes]
    )
    assert np.allclose(selected_gram, np.eye(6))
    assert np.linalg.norm(context.ritz_coefficients[6:, :]) > 0.1

    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    assert residual.dat.norm < 1.0e-4

    source = matrix.createVecRight()
    applied = matrix.createVecRight()
    transpose_applied = matrix.createVecRight()
    source.setRandom()
    velocity_ksp.pc.apply(source, applied)
    velocity_ksp.pc.applyTranspose(source, transpose_applied)
    transpose_applied.axpy(-1, applied)
    assert transpose_applied.norm() < 1.0e-12 * applied.norm()

    old_eigenvalues = context.ritz_eigenvalues.copy()
    viscosity.interpolate(2 + coordinates[0] + coordinates[1] + coordinates[2])
    solution.assign(0)
    solver.solve()
    assert not np.allclose(context.ritz_eigenvalues, old_eigenvalues)
    assert context.ritz_principal_angle_change is not None
    assert len(matrix.getNearNullSpace().getVecs()) == 6


def test_ritz_conformal_pc_falls_back_at_an_unresolved_cutoff():
    """A user-selected minimum gap triggers the deterministic rigid space."""
    _, solver, _, _ = build_conformal_pc_solver(
        "gadopt.RitzConformalPC",
        ritz_min_relative_gap=2.0,
    )
    solver.solve()

    velocity_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[0]
    context = velocity_ksp.pc.getPythonContext()
    assert context.ritz_used_fallback
    assert np.allclose(context.ritz_coefficients, np.eye(10, 6))
