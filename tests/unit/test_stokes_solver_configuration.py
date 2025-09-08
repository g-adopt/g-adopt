import firedrake as fd
import pytest

from gadopt.approximations import BoussinesqApproximation
from gadopt.stokes_integrators import (
    StokesSolver,
    direct_stokes_solver_parameters,
    iterative_stokes_solver_parameters,
    newton_stokes_solver_parameters,
)
from gadopt.solver_options_manager import DeleteParam

test_cases = [
    "unspecified",
    "direct",
    "iterative",
    "dictionary",
    "cartesian_false",
    "linear_false",
    "add_parameter",
    "delete_parameter",
    "change_tolerance",
]


@pytest.mark.parametrize("test_case", test_cases)
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

    match test_case:
        case "unspecified":
            solver_parameters = None
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | direct_stokes_solver_parameters
        case "direct":
            solver_parameters = "direct"
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | direct_stokes_solver_parameters
        case "iterative":
            solver_parameters = "iterative"
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | iterative_stokes_solver_parameters
        case "dictionary":
            solver_parameters = example_solver_params
            solver_parameters_extra = None
            expected_value = example_solver_params
        case "cartesian_false":
            mesh.cartesian = False
            solver_parameters = None
            solver_parameters_extra = None
            expected_value = base_linear_params_with_log | iterative_stokes_solver_parameters
            expected_value["fieldsplit_1"] |= {"ksp_converged_reason": None}
        case "linear_false":
            mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))
            solver_parameters = "direct"
            solver_parameters_extra = None
            expected_value = {"snes_monitor": None} | newton_stokes_solver_parameters | direct_stokes_solver_parameters
        case "add_parameter":
            solver_parameters = None
            solver_parameters_extra = {"ksp_converged_reason": None}
            expected_value = (
                {"ksp_converged_reason": None} | base_linear_params_with_log | direct_stokes_solver_parameters
            )
        case "delete_parameter":
            solver_parameters = None
            solver_parameters_extra = {"snes_monitor": DeleteParam}
            expected_value = base_linear_params_with_log | direct_stokes_solver_parameters
            del expected_value["snes_monitor"]
        case "change_tolerance":
            solver_parameters = "iterative"
            solver_parameters_extra = {
                "fieldsplit_0": {"ksp_rtol": 1e-4},
                "fieldsplit_1": {"ksp_rtol": 1e-3},
            }
            expected_value = base_linear_params_with_log | iterative_stokes_solver_parameters
            expected_value["fieldsplit_0"]["ksp_rtol"] = 1e-4
            expected_value["fieldsplit_1"]["ksp_rtol"] = 1e-3

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
