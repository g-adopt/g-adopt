import math

import firedrake as fd
import pytest

from gadopt import (
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
    DensityAwareBFBTPC,
    StokesSolver,
    TruncatedAnelasticLiquidApproximation,
    create_stokes_nullspace,
)


def bfbt_parameters():
    """Return deterministic small-problem BFBT solver parameters."""
    return {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "lu",
        },
        "fieldsplit_1": {
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 100,
            "pc_type": "python",
            "pc_python_type": "gadopt.DensityAwareBFBTPC",
            "bfbt_ksp_type": "fgmres",
            "bfbt_ksp_rtol": 1e-8,
            "bfbt_pc_type": "gamg",
        },
    }


def build_solver(
    approximation_name,
    *,
    quadrilateral=True,
    viscosity_scale=None,
    discontinuous_density=False,
    nullspace_policy=None,
    right_inner_rtol=None,
    left_inner_rtol=None,
):
    """Construct a variable-viscosity Boussinesq or variable-density TALA case."""
    mesh = fd.UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
    mesh.cartesian = True
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = velocity_space * pressure_space

    solution = fd.Function(mixed_space)
    temperature = fd.Function(temperature_space)
    x, y = fd.SpatialCoordinate(mesh)
    temperature.interpolate(
        1 - y + 0.1 * fd.sin(fd.pi * x) * fd.sin(fd.pi * y)
    )
    if viscosity_scale is None:
        viscosity_scale = fd.Constant(1)
    viscosity = viscosity_scale * fd.exp(6 * x)

    if approximation_name == "Boussinesq":
        approximation = BoussinesqApproximation(1, mu=viscosity)
    elif approximation_name in {"TALA", "ALA"}:
        if discontinuous_density:
            density_space = fd.FunctionSpace(mesh, "DQ", 0)
            density = fd.Function(density_space).interpolate(
                1 + fd.conditional(fd.gt(x, 0.47), 1, 0)
            )
        else:
            density = fd.Function(temperature_space).interpolate(
                fd.exp(0.5 * (1 - y))
            )
        approximation_class = (
            AnelasticLiquidApproximation
            if approximation_name == "ALA"
            else TruncatedAnelasticLiquidApproximation
        )
        approximation = approximation_class(1, 0.5, rho=density, mu=viscosity)
    else:
        raise ValueError(f"Unknown approximation {approximation_name}")

    zero = fd.Constant((0.0, 0.0))
    bcs = {boundary: {"u": zero} for boundary in (1, 2, 3, 4)}
    nullspace_parameters = {}
    if approximation_name == "ALA":
        nullspace_parameters = {
            "ala_approximation": approximation,
            "top_subdomain_id": 4,
        }
    nullspace = create_stokes_nullspace(
        mixed_space, closed=True, **nullspace_parameters
    )
    transpose_nullspace = create_stokes_nullspace(mixed_space, closed=True)
    parameters = bfbt_parameters()
    if approximation_name == "ALA":
        parameters["fieldsplit_1"]["bfbt_nullspace_policy"] = "schur"
    if nullspace_policy is not None:
        parameters["fieldsplit_1"]["bfbt_nullspace_policy"] = (
            nullspace_policy
        )
    if right_inner_rtol is not None:
        parameters["fieldsplit_1"]["bfbt_right_ksp_rtol"] = right_inner_rtol
    if left_inner_rtol is not None:
        parameters["fieldsplit_1"]["bfbt_left_ksp_rtol"] = left_inner_rtol
    solver = StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=bcs,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        solver_parameters=parameters,
    )
    return solver


def test_bfbt_uses_side_specific_inner_tolerances():
    """The algebraic rightmost and leftmost solves retain distinct controls."""
    solver = build_solver(
        "TALA",
        right_inner_rtol=1e-7,
        left_inner_rtol=1e-3,
    )
    solver.solve()

    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    assert bfbt.right_inner_rtol == pytest.approx(1e-7)
    assert bfbt.left_inner_rtol == pytest.approx(1e-3)
    assert bfbt.last_inner_tolerances == pytest.approx((1e-7, 1e-3))
    assert bfbt.inner_solves_by_side["right"] > 0
    assert (
        bfbt.inner_solves_by_side["right"]
        == bfbt.inner_solves_by_side["left"]
    )
    assert sum(bfbt.inner_iterations_by_side.values()) == (
        bfbt.inner_iterations_total
    )
    assert fd.assemble(solver.F, bcs=solver.strong_bcs).dat.norm < 1e-8


@pytest.mark.parametrize(
    ("approximation_name", "quadrilateral"),
    [
        ("Boussinesq", True),
        ("TALA", True),
        ("ALA", True),
        ("Boussinesq", False),
    ],
)
def test_density_aware_bfbt_solve(approximation_name, quadrilateral):
    """BFBT solves incompressible, TALA/ALA, quadrilateral, and simplex cases."""
    solver = build_solver(
        approximation_name, quadrilateral=quadrilateral
    )
    solver.solve()

    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    component_norms = [part.dat.norm for part in residual.subfunctions]
    if approximation_name == "ALA":
        # G-ADOPT's numerically constructed non-constant ALA pressure
        # nullspace is not an exact null mode of the discrete gradient block.
        # Different pressure gauges can therefore leave different momentum
        # residuals even after an exact solve. The continuity residual is the
        # gauge-independent range-space check relevant to this pressure PC.
        assert component_norms[1] < 1e-8
    else:
        assert residual.dat.norm < 1e-8

    # Inspect the actual Python PC created inside fieldsplit_1.
    fieldsplit_ksps = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()
    bfbt = fieldsplit_ksps[1].pc.getPythonContext()
    assert fieldsplit_ksps[1].getConvergedReason() > 0
    assert bfbt.mass_lumping == "diagonal"
    assert bfbt.sides["right"] is bfbt.sides["left"]
    assert bfbt.inverse_velocity_mass.min()[1] > 0
    assert bfbt.inner_solves_total > 0
    assert bfbt.inner_iterations_total > 0
    assert bfbt.inner_failures_total == 0
    assert all(reason > 0 for reason in bfbt.last_inner_reasons)
    if approximation_name == "ALA":
        assert bfbt.right_nullspace_is_exact is False
        assert bfbt.auxiliary_right_nullspace_is_exact is False
        assert bfbt.nullspace_policy == "schur"
        assert bfbt.left_nullspace_source == "none"
        assert bfbt.left_nullspace_fallback_used is False
    else:
        assert bfbt.right_nullspace_is_exact is True
        assert bfbt.left_nullspace_is_exact is True
        assert bfbt.auxiliary_left_nullspace_is_exact is True
        assert bfbt.left_nullspace_source == "verified_right_fallback"
        assert bfbt.left_nullspace_fallback_used is True
        assert bfbt.exact_left_nullspace_attached is True
        assert bfbt.auxiliary_left_nullspace_attached is True
        assert bfbt.exact_pressure_laplacian.getTransposeNullSpace().handle != 0
        assert (
            bfbt.pressure_laplacian.petscmat.getTransposeNullSpace().handle
            != 0
        )


def test_bfbt_rejects_incompatible_discontinuous_density_nullspace():
    """A discontinuous density cannot inherit a false transpose null mode."""
    solver = build_solver(
        "TALA",
        discontinuous_density=True,
        nullspace_policy="schur",
    )
    solver.solve()
    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    operator, _ = pressure_ksp.pc.getOperators()
    assert bfbt.right_nullspace_is_exact is True
    assert bfbt.left_nullspace_is_exact is False
    assert bfbt.auxiliary_left_nullspace_is_exact is True
    assert bfbt.left_nullspace_source == "verified_right_fallback"
    assert bfbt.exact_left_nullspace_attached is False
    assert bfbt.auxiliary_left_nullspace_attached is True

    bfbt.nullspace_policy = "verified"
    with pytest.raises(
        ValueError,
        match="no compatible transpose nullspace.*rho_continuity",
    ):
        bfbt._set_pressure_nullspaces(operator)
    assert bfbt.exact_right_nullspace_attached is False
    assert bfbt.auxiliary_right_nullspace_attached is False
    assert bfbt.exact_left_nullspace_attached is False
    assert bfbt.auxiliary_left_nullspace_attached is False
    assert bfbt.exact_pressure_laplacian.getNullSpace().handle == 0
    assert bfbt.exact_pressure_laplacian.getTransposeNullSpace().handle == 0
    assert bfbt.pressure_laplacian.petscmat.getNullSpace().handle == 0
    assert (
        bfbt.pressure_laplacian.petscmat.getTransposeNullSpace().handle == 0
    )


@pytest.mark.parametrize(
    ("absolute_tolerance", "relative_tolerance", "message"),
    [
        (math.inf, 0.0, "test_tolerance"),
        (math.nan, 0.0, "test_tolerance"),
        (0.0, math.inf, "relative_tolerance"),
        (0.0, math.nan, "relative_tolerance"),
        (-1.0, 0.0, "test_tolerance"),
        (0.0, -1.0, "relative_tolerance"),
    ],
)
def test_bfbt_rejects_invalid_nullspace_tolerances(
    absolute_tolerance,
    relative_tolerance,
    message,
):
    """Invalid diagnostics cannot make every pressure mode appear exact."""
    with pytest.raises(ValueError, match=message):
        DensityAwareBFBTPC._validate_nullspace_test_tolerances(
            absolute_tolerance,
            relative_tolerance,
        )


def test_weighted_pressure_laplacian_transpose():
    """The exact TALA pressure operator implements a consistent transpose."""
    solver = build_solver("TALA")
    solver.solve()
    fieldsplit_ksps = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()
    bfbt = fieldsplit_ksps[1].pc.getPythonContext()
    laplacian = bfbt.exact_pressure_laplacian

    x = laplacian.createVecRight()
    y = laplacian.createVecLeft()
    laplacian_x = laplacian.createVecLeft()
    transpose_laplacian_y = laplacian.createVecRight()
    x.setRandom()
    y.setRandom()

    nullspace = laplacian.getNullSpace()
    if nullspace.handle != 0:
        nullspace.remove(x)
    transpose_nullspace = laplacian.getTransposeNullSpace()
    if transpose_nullspace.handle != 0:
        transpose_nullspace.remove(y)

    laplacian.mult(x, laplacian_x)
    laplacian.multTranspose(y, transpose_laplacian_y)
    assert laplacian_x.dot(y) == pytest.approx(
        x.dot(transpose_laplacian_y), rel=1e-12, abs=1e-12
    )

    for vector in (x, y, laplacian_x, transpose_laplacian_y):
        vector.destroy()


def test_bfbt_update_increments_exact_operator_state():
    """Updating a reused Python matrix tells PETSc its action has changed."""
    solver = build_solver("TALA")
    solver.solve()
    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    state_before = bfbt.exact_pressure_laplacian.stateGet()

    bfbt.update(pressure_ksp.pc)

    assert bfbt.exact_pressure_laplacian.stateGet() > state_before
    assert bfbt.left_nullspace_source == "verified_right_fallback"
    assert bfbt.left_nullspace_is_exact is True
    assert bfbt.auxiliary_left_nullspace_is_exact is True
    assert bfbt.exact_left_nullspace_attached is True
    assert bfbt.auxiliary_left_nullspace_attached is True


def test_bfbt_update_tracks_changed_viscosity_and_resolves():
    """A reused nonlinear PC refreshes its state-dependent auxiliary data."""
    viscosity_scale = fd.Constant(1)
    solver = build_solver("TALA", viscosity_scale=viscosity_scale)
    solver.solve()
    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    weight_integral_before = fd.assemble(bfbt.weight * fd.dx)
    operator_state_before = bfbt.exact_pressure_laplacian.stateGet()

    viscosity_scale.assign(4)
    solver.solution.assign(0)
    solver.solve()

    weight_integral_after = fd.assemble(bfbt.weight * fd.dx)
    residual = fd.assemble(solver.F, bcs=solver.strong_bcs)
    assert weight_integral_after == pytest.approx(
        2 * weight_integral_before, rel=1e-12
    )
    assert bfbt.exact_pressure_laplacian.stateGet() > operator_state_before
    assert bfbt.inner_failures_total == 0
    assert residual.dat.norm < 1e-8
    assert bfbt.exact_left_nullspace_attached is True
    assert bfbt.auxiliary_left_nullspace_attached is True
    assert bfbt.exact_pressure_laplacian.getTransposeNullSpace().handle != 0
    assert (
        bfbt.pressure_laplacian.petscmat.getTransposeNullSpace().handle != 0
    )


@pytest.mark.parametrize(
    "form_compiler_parameters",
    [None, {"quadrature_degree": 6}],
)
def test_bfbt_accepts_application_form_compiler_parameters(
    form_compiler_parameters,
):
    """Auxiliary forms handle both unset and production quadrature context."""
    solver = build_solver("TALA")
    solver.appctx["form_compiler_parameters"] = form_compiler_parameters

    solver.solve()

    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    assert bfbt.inner_failures_total == 0


def test_bfbt_rejects_coupled_pressure_free_surface_space():
    """The volume pressure PC cannot silently flatten a surface unknown."""
    mesh = fd.UnitSquareMesh(1, 1)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    free_surface_space = fd.FunctionSpace(mesh, "CG", 1)

    with pytest.raises(ValueError, match="pressure/free-surface"):
        DensityAwareBFBTPC._validate_pressure_space(
            pressure_space * free_surface_space
        )


def test_bfbt_rejects_unsupported_transpose_application():
    """The forward-only PC cannot silently provide an incorrect adjoint."""
    solver = build_solver("TALA")
    solver.solve()
    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    pressure = bfbt.exact_pressure_laplacian.createVecRight()
    result = bfbt.exact_pressure_laplacian.createVecLeft()

    with pytest.raises(NotImplementedError, match="currently forward-only"):
        bfbt.applyTranspose(pressure_ksp.pc, pressure, result)

    pressure.destroy()
    result.destroy()


def test_bfbt_inner_failure_is_not_silent():
    """A failed inner inversion cannot return a corrupted PC application."""
    solver = build_solver("TALA")
    solver.solve()
    pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    bfbt = pressure_ksp.pc.getPythonContext()
    rhs = bfbt.exact_pressure_laplacian.createVecRight()
    result = bfbt.exact_pressure_laplacian.createVecLeft()
    rhs.setRandom()
    nullspace = bfbt.exact_pressure_laplacian.getNullSpace()
    if nullspace.handle != 0:
        nullspace.remove(rhs)
    original_tolerances = bfbt.ksp.getTolerances()
    bfbt.ksp.setTolerances(max_it=0)

    with pytest.raises(
        RuntimeError,
        match="BFBT right inner pressure solve failed",
    ):
        bfbt.apply(pressure_ksp.pc, rhs, result)

    bfbt.ksp.setTolerances(*original_tolerances)
    rhs.destroy()
    result.destroy()
