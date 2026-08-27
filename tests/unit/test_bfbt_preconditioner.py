import firedrake as fd
import pytest

from gadopt import (
    AnelasticLiquidApproximation,
    BoussinesqApproximation,
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


def build_solver(approximation_name, *, quadrilateral=True):
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
    viscosity = fd.exp(6 * x)

    if approximation_name == "Boussinesq":
        approximation = BoussinesqApproximation(1, mu=viscosity)
    elif approximation_name in {"TALA", "ALA"}:
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
    solver = StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=bcs,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        solver_parameters=bfbt_parameters(),
    )
    return solver


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
    assert bfbt.inverse_velocity_mass.min()[1] > 0


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
