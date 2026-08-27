import firedrake as fd
import pytest

from gadopt import (
    BoussinesqApproximation,
    ConformalKillingNearNullspace,
    StokesSolver,
    TruncatedAnelasticLiquidApproximation,
    create_stokes_nullspace,
    rigid_body_modes,
)


@pytest.fixture
def three_dimensional_spaces():
    mesh = fd.UnitCubeMesh(2, 2, 2)
    mesh.cartesian = True
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    return velocity_space, pressure_space, velocity_space * pressure_space


def test_conformal_modes_span_deviatoric_strain_kernel(three_dimensional_spaces):
    _, _, mixed_space = three_dimensional_spaces
    near_nullspace = ConformalKillingNearNullspace()._build(mixed_space, [])
    basis = tuple(near_nullspace)[0]

    assert isinstance(basis, fd.VectorSpaceBasis)
    assert len(basis._vecs) == 10
    assert basis.is_orthonormal()

    identity = fd.Identity(3)
    for mode in basis._vecs:
        deviatoric_strain = fd.sym(fd.grad(mode)) - identity * fd.div(mode) / 3
        squared_norm = fd.assemble(fd.inner(deviatoric_strain, deviatoric_strain) * fd.dx)
        assert squared_norm == pytest.approx(0, abs=1e-24)


def test_tala_stress_annihilates_conformal_modes(three_dimensional_spaces):
    velocity_space, _, mixed_space = three_dimensional_spaces
    x = fd.SpatialCoordinate(velocity_space.mesh())
    viscosity_space = fd.FunctionSpace(velocity_space.mesh(), "CG", 2)
    viscosity = fd.Function(viscosity_space)
    viscosity.interpolate(1 + x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=1,
        Di=1,
        mu=viscosity,
    )
    basis = tuple(ConformalKillingNearNullspace()._build(mixed_space, []))[0]

    for mode in basis._vecs:
        stress = approximation.stress(mode)
        squared_norm = fd.assemble(fd.inner(stress, stress) * fd.dx)
        assert squared_norm == pytest.approx(0, abs=1e-24)


def test_conformal_specification_uses_velocity_modes_only(
    three_dimensional_spaces,
):
    velocity_space, pressure_space, mixed_space = three_dimensional_spaces
    near_nullspace = ConformalKillingNearNullspace()._build(mixed_space, [])

    velocity_basis, pressure_placeholder = tuple(near_nullspace)
    assert isinstance(velocity_basis, fd.VectorSpaceBasis)
    assert len(velocity_basis._vecs) == 10
    assert pressure_placeholder == pressure_space
    assert velocity_basis._vecs[0].ufl_element() == velocity_space.ufl_element()


def test_conformal_modes_require_three_dimensions():
    mesh = fd.UnitSquareMesh(2, 2)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)

    with pytest.raises(ValueError, match="three-dimensional volumetric"):
        ConformalKillingNearNullspace()._build(
            velocity_space * pressure_space,
            [],
        )


def test_conformal_modes_reject_surface_embedded_in_three_dimensions():
    mesh = fd.UnitIcosahedralSphereMesh(refinement_level=0)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)

    with pytest.raises(ValueError, match="three-dimensional volumetric"):
        ConformalKillingNearNullspace()._build(
            velocity_space * pressure_space,
            [],
        )


def test_conformal_modes_are_homogeneous_on_strong_velocity_boundary(
    three_dimensional_spaces,
):
    _, _, mixed_space = three_dimensional_spaces
    boundary_condition = fd.DirichletBC(
        mixed_space.sub(0),
        fd.as_vector((3, 4, 5)),
        1,
    )
    near_nullspace = ConformalKillingNearNullspace(
        constrain_strong_bcs=True
    )._build(
        mixed_space,
        [boundary_condition],
    )

    for mode in tuple(near_nullspace)[0]._vecs:
        trace_norm = fd.assemble(
            fd.inner(mode, mode) * fd.ds(1, domain=mixed_space.mesh())
        )
        assert trace_norm == pytest.approx(0, abs=1e-24)


def test_raw_conformal_modes_retain_unconstrained_boundary_values(
    three_dimensional_spaces,
):
    _, _, mixed_space = three_dimensional_spaces
    boundary_condition = fd.DirichletBC(mixed_space.sub(0), 0, 1)
    near_nullspace = ConformalKillingNearNullspace()._build(
        mixed_space,
        [boundary_condition],
    )
    trace_norms = [
        fd.assemble(fd.inner(mode, mode) * fd.ds(1, domain=mixed_space.mesh()))
        for mode in tuple(near_nullspace)[0]._vecs
    ]

    assert max(trace_norms) > 0


def test_stokes_solver_materialises_conformal_specification(
    three_dimensional_spaces,
):
    _, _, mixed_space = three_dimensional_spaces
    solution = fd.Function(mixed_space)
    solver = StokesSolver(
        solution,
        TruncatedAnelasticLiquidApproximation(Ra=1, Di=0.5),
        bcs={1: {"u": fd.Constant((1.0, 0.0, 0.0))}},
        near_nullspace=ConformalKillingNearNullspace(),
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    velocity_basis = tuple(solver.near_nullspace)[0]
    assert isinstance(solver.near_nullspace, fd.MixedVectorSpaceBasis)
    assert len(velocity_basis._vecs) == 10


def test_conformal_specification_preserves_extra_field_placeholders(
    three_dimensional_spaces,
):
    velocity_space, pressure_space, _ = three_dimensional_spaces
    free_surface_space = fd.FunctionSpace(velocity_space.mesh(), "CG", 1)
    mixed_space = velocity_space * pressure_space * free_surface_space
    near_nullspace = ConformalKillingNearNullspace()._build(mixed_space, [])

    assert tuple(near_nullspace)[2] == mixed_space.sub(2)


def test_conformal_specification_is_rejected_as_exact_nullspace(
    three_dimensional_spaces,
):
    _, _, mixed_space = three_dimensional_spaces
    solution = fd.Function(mixed_space)

    with pytest.raises(TypeError, match="near_nullspace argument"):
        StokesSolver(
            solution,
            BoussinesqApproximation(Ra=1),
            nullspace=ConformalKillingNearNullspace(),
        )


def test_rigid_body_modes_remain_unchanged(three_dimensional_spaces):
    velocity_space, _, _ = three_dimensional_spaces
    basis = rigid_body_modes(
        velocity_space,
        rotational=True,
        translations=[0, 1, 2],
    )

    assert isinstance(basis, fd.VectorSpaceBasis)
    assert len(basis._vecs) == 6
    assert basis.is_orthonormal()


def test_invalid_translation_direction_has_clear_error(three_dimensional_spaces):
    with pytest.raises(ValueError, match="selected from 0, 1, and 2"):
        ConformalKillingNearNullspace(translations=(3,))


@pytest.mark.parametrize("translations", [(True,), (1.0,), (1, 1)])
def test_invalid_translation_types_are_rejected(translations):
    with pytest.raises(ValueError, match="Translation directions"):
        ConformalKillingNearNullspace(translations=translations)


def test_conformal_modes_on_curved_extruded_spherical_mesh():
    base_mesh = fd.CubedSphereMesh(1.22, refinement_level=1, degree=2)
    mesh = fd.ExtrudedMesh(base_mesh, layers=2, extrusion_type="radial")
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    near_nullspace = ConformalKillingNearNullspace()._build(
        velocity_space * pressure_space,
        [],
    )
    identity = fd.Identity(3)
    strain_quotients = []
    for mode in tuple(near_nullspace)[0]._vecs:
        deviatoric_strain = fd.sym(fd.grad(mode)) - identity * fd.div(mode) / 3
        strain_energy = fd.assemble(
            fd.inner(deviatoric_strain, deviatoric_strain) * fd.dx
        )
        mode_energy = fd.assemble(fd.inner(mode, mode) * fd.dx)
        strain_quotients.append(strain_energy / mode_energy)

    assert max(strain_quotients[:6]) < 1e-20
    assert max(strain_quotients) < 1e-2


def test_conformal_modes_reach_velocity_assembled_preconditioner():
    mesh = fd.UnitCubeMesh(2, 2, 2)
    mesh.cartesian = True
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    temperature_space = fd.FunctionSpace(mesh, "CG", 2)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    temperature = fd.Function(temperature_space)
    density = fd.Function(temperature_space)
    viscosity = fd.Function(temperature_space)
    x, y, z = fd.SpatialCoordinate(mesh)
    temperature.interpolate(1 - z + 0.01 * x * y)
    density.interpolate(fd.exp(0.5 * (1 - z)))
    viscosity.interpolate(1 + x + y + z)
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=1,
        Di=0.5,
        rho=density,
        mu=viscosity,
    )
    zero = fd.Constant((0.0, 0.0, 0.0))
    boundary_conditions = {
        boundary: {"u": zero} for boundary in (1, 2, 3, 4, 5, 6)
    }
    nullspace = create_stokes_nullspace(mixed_space, closed=True)
    solver = StokesSolver(
        solution,
        approximation,
        temperature,
        bcs=boundary_conditions,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        near_nullspace=ConformalKillingNearNullspace(),
        solver_parameters={
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_0": {
                "ksp_type": "cg",
                "ksp_rtol": 1e-8,
                "pc_type": "python",
                "pc_python_type": "gadopt.SPDAssembledPC",
                "assembled_pc_type": "gamg",
                "assembled_mg_levels_pc_type": "sor",
            },
            "fieldsplit_1": {
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-8,
                "pc_type": "python",
                "pc_python_type": "firedrake.MassInvPC",
                "Mp_pc_type": "lu",
            },
        },
    )
    solver.solve()

    velocity_ksp, pressure_ksp = solver.solver.snes.ksp.pc.getFieldSplitSubKSP()
    assembled_velocity_matrix = velocity_ksp.pc.getPythonContext().P.petscmat
    velocity_near_nullspace = assembled_velocity_matrix.getNearNullSpace()
    assert velocity_near_nullspace.handle != 0
    assert len(velocity_near_nullspace.getVecs()) == 10
    assert velocity_near_nullspace.hasConstant() is False
    assert assembled_velocity_matrix.getBlockSize() == 3
    for vector in velocity_near_nullspace.getVecs():
        assert vector.getSizes() == assembled_velocity_matrix.getSizes()[0]

    pressure_near_nullspace = pressure_ksp.getOperators()[1].getNearNullSpace()
    assert pressure_near_nullspace.handle == 0

    viscosity.interpolate(2 + x + y + z)
    solver.solve()
    updated_matrix = velocity_ksp.pc.getPythonContext().P.petscmat
    assert len(updated_matrix.getNearNullSpace().getVecs()) == 10
