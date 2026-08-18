import firedrake as fd
import numpy as np
import pytest

from copy import deepcopy

from gadopt.approximations import BoussinesqApproximation, MaxwellApproximation
from gadopt.stokes_integrators import (
    StokesSolver,
    InternalVariableSolver,
    direct_stokes_solver_parameters,
    iterative_outer_stokes_solver_parameters,
    iterative_fieldsplit_0_cpu_params,
    newton_stokes_solver_parameters,
    gia_outer_solver_parameters,
    gia_iterative_cpu_assembled_parameters,
)
from gadopt.solver_options_manager import DeleteParam
from gadopt.utility import initialise_background_field

test_cases = [
    "unspecified",
    "direct",
    "iterative",
    "dictionary",
    "cartesian_false",
    "add_parameter",
    "delete_parameter",
    "change_tolerance",
]

test_cases_mantle = ["linear_false"]


@pytest.mark.parametrize("test_case", test_cases + test_cases_mantle)
def test_solver_parameters_argument(test_case):
    mesh = fd.UnitSquareMesh(10, 10)

    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
    temperature = fd.Function(func_space_temp, name="Temperature")

    base_linear_params_with_log = {"snes_type": "ksponly", "snes_monitor": None}
    example_solver_params = {"mat_type": "aij", "ksp_type": "cg", "pc_type": "sor"}

    mu = 1
    mesh.cartesian = True

    # Create copies of the solver parameters
    direct_params = deepcopy(direct_stokes_solver_parameters)
    iterative_params = deepcopy(iterative_outer_stokes_solver_parameters)
    iterative_params["fieldsplit_0"] = deepcopy(iterative_fieldsplit_0_cpu_params)["fieldsplit_0"]
    newton_params = deepcopy(newton_stokes_solver_parameters)

    match test_case:
        case "unspecified":
            solver_parameters = None
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | direct_params
        case "direct":
            solver_parameters = "direct"
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | direct_params
        case "iterative":
            solver_parameters = "iterative"
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | iterative_params
            expected_value["fieldsplit_1"]["ksp_converged_reason"] = None
        case "dictionary":
            solver_parameters = example_solver_params
            solver_parameters_extra = None
            expected_value = example_solver_params
        case "cartesian_false":
            mesh.cartesian = False
            solver_parameters = None
            solver_parameters_extra = None
            expected_value = (
                base_linear_params_with_log | direct_stokes_solver_parameters
            )
        case "linear_false":
            mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))
            solver_parameters = "direct"
            solver_parameters_extra = None
            expected_value = {"snes_monitor": None} | newton_params | direct_params
        case "add_parameter":
            solver_parameters = None
            solver_parameters_extra = {"ksp_converged_reason": None}
            expected_value = (
                {"ksp_converged_reason": None} | base_linear_params_with_log | direct_params
            )
        case "delete_parameter":
            solver_parameters = None
            solver_parameters_extra = {"snes_monitor": DeleteParam}
            expected_value = base_linear_params_with_log | direct_params
            del expected_value["snes_monitor"]
        case "change_tolerance":
            solver_parameters = "iterative"
            solver_parameters_extra = {
                "fieldsplit_0": {"ksp_rtol": 1e-4},
                "fieldsplit_1": {"ksp_rtol": 1e-3},
            }
            expected_value = base_linear_params_with_log | iterative_params
            expected_value["fieldsplit_0"]["ksp_rtol"] = 1e-4
            expected_value["fieldsplit_1"]["ksp_rtol"] = 1e-3
            expected_value["fieldsplit_1"]["ksp_converged_reason"] = None

    approximation = BoussinesqApproximation(1, mu=mu)

    stokes_solver = StokesSolver(
        stokes_function,
        approximation,
        temperature,
        solver_parameters=solver_parameters,
        solver_parameters_extra=solver_parameters_extra,
    )

    assert stokes_solver.solver_parameters == expected_value

    with pytest.raises(ValueError):
        StokesSolver(stokes_function, approximation, temperature, solver_parameters="")


@pytest.mark.parametrize("test_case", test_cases)
def test_gia_solver_parameters_argument(test_case):
    mesh = fd.UnitSquareMesh(10, 10, quadrilateral=True)

    V = fd.VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space
    S = fd.TensorFunctionSpace(mesh, "DQ", 1)  # Stress tensor function space
    DQ0 = fd.FunctionSpace(mesh, "DQ", 0)  # Density/viscosity/shear modulus function space
    u = fd.Function(V, name="displacement")  # field to hold our displacement solution
    m = fd.Function(S, name="Internal variable")  # Lagged internal variable at previous timestep

    X = fd.SpatialCoordinate(mesh)

    base_linear_params_with_log = {"snes_type": "ksponly", "snes_monitor": None}
    example_solver_params = {"mat_type": "aij", "ksp_type": "cg", "pc_type": "sor"}

    mesh.cartesian = True

    # Create copies of the solver parameters
    direct_params = deepcopy(direct_stokes_solver_parameters)
    iterative_params = deepcopy(gia_outer_solver_parameters) | deepcopy(
        gia_iterative_cpu_assembled_parameters
    )

    match test_case:
        case "unspecified":
            solver_parameters = None
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | direct_params
        case "direct":
            solver_parameters = "direct"
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | direct_params
        case "iterative":
            solver_parameters = "iterative"
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | iterative_params
            expected_value["ksp_converged_reason"] = None
        case "dictionary":
            solver_parameters = example_solver_params
            solver_parameters_extra = None
            expected_value = example_solver_params
        case "cartesian_false":
            mesh.cartesian = False
            solver_parameters = None
            solver_parameters_extra = None
            expected_value = (
                base_linear_params_with_log | direct_stokes_solver_parameters
            )
        case "add_parameter":
            solver_parameters = None
            solver_parameters_extra = {"ksp_converged_reason": None}
            expected_value = (
                {"ksp_converged_reason": None} | base_linear_params_with_log | direct_params
            )
        case "delete_parameter":
            solver_parameters = None
            solver_parameters_extra = {"snes_monitor": DeleteParam}
            expected_value = base_linear_params_with_log | direct_params
            del expected_value["snes_monitor"]
        case "change_tolerance":
            solver_parameters = "iterative"
            solver_parameters_extra = {"ksp_rtol": 1e-4}
            expected_value = base_linear_params_with_log | iterative_params
            expected_value["ksp_rtol"] = 1e-4
            expected_value["ksp_converged_reason"] = None

    domain_depth = 2891e3
    density_scale = 4500
    shear_modulus_scale = 1e11
    viscosity_scale = 1e21
    gravity_scale = 9.81
    B_mu = fd.Constant(density_scale * domain_depth * gravity_scale / shear_modulus_scale)

    radius_values_nondim = np.array([6371e3, 6301e3, 5951e3, 5701e3, 3480e3])/domain_depth
    density_values_nondim = np.array([3037, 3438, 3871, 4978])/density_scale
    shear_modulus_values_nondim = np.array([0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11])/shear_modulus_scale
    viscosity_values_nondim = np.array([1e40, 1e21, 1e21, 2e21])/viscosity_scale
    bulk_shear_ratio = 2
    bulk_modulus_values_nondim = shear_modulus_values_nondim

    density = fd.Function(DQ0, name="density")
    initialise_background_field(
        density, density_values_nondim, X, radius_values_nondim,
        shift=radius_values_nondim[-1])

    shear_modulus = fd.Function(DQ0, name="shear modulus")
    initialise_background_field(
        shear_modulus, shear_modulus_values_nondim, X, radius_values_nondim,
        shift=radius_values_nondim[-1])

    bulk_modulus = fd.Function(DQ0, name="bulk modulus")
    initialise_background_field(
        bulk_modulus, bulk_modulus_values_nondim, X, radius_values_nondim,
        shift=radius_values_nondim[-1])

    viscosity = fd.Function(DQ0, name="viscosity")
    initialise_background_field(
        viscosity, viscosity_values_nondim, X, radius_values_nondim,
        shift=radius_values_nondim[-1])

    approximation = MaxwellApproximation(
        bulk_modulus=bulk_modulus,
        density=density,
        shear_modulus=shear_modulus,
        viscosity=viscosity,
        B_mu=B_mu,
        bulk_shear_ratio=bulk_shear_ratio)

    stokes_solver = InternalVariableSolver(
        u,
        approximation,
        dt=1,
        internal_variables=m,
        solver_parameters=solver_parameters,
        solver_parameters_extra=solver_parameters_extra,
    )

    assert stokes_solver.solver_parameters == expected_value

    with pytest.raises(ValueError):
        InternalVariableSolver(
            u, approximation, dt=1, internal_variables=m, solver_parameters=""
        )
