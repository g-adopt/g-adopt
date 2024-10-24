import firedrake as fd
import pytest

from gadopt.approximations import Approximation
from gadopt.stokes_integrators import (
    StokesSolver,
    direct_stokes_solver_parameters,
    iterative_stokes_solver_parameters,
    newton_stokes_solver_parameters,
)


def test_solver_parameters_argument():
    mesh = fd.UnitSquareMesh(10, 10)

    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    base_linear_params_with_log = {"snes_type": "ksponly", "snes_monitor": None}
    example_solver_params = {"mat_type": "aij", "ksp_type": "cg", "pc_type": "sor"}

    for test_case in [
        "unspecified",
        "direct",
        "iterative",
        "dictionary",
        "cartesian_false",
        "linear_false",
    ]:
        mu = 1
        mesh.cartesian = True

        match test_case:
            case "unspecified":
                solver_parameters = None
                expected_value = (
                    base_linear_params_with_log | direct_stokes_solver_parameters
                )
            case "direct":
                solver_parameters = "direct"
                expected_value = (
                    base_linear_params_with_log | direct_stokes_solver_parameters
                )
            case "iterative":
                solver_parameters = "iterative"
                expected_value = (
                    base_linear_params_with_log | iterative_stokes_solver_parameters
                )
            case "dictionary":
                solver_parameters = example_solver_params
                expected_value = example_solver_params
            case "cartesian_false":
                mesh.cartesian = False
                solver_parameters = None
                expected_value = (
                    base_linear_params_with_log | iterative_stokes_solver_parameters
                )
                expected_value["fieldsplit_1"] |= {"ksp_converged_reason": None}
            case "linear_false":
                mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))
                solver_parameters = "direct"
                expected_value = (
                    {"snes_monitor": None}
                    | newton_stokes_solver_parameters
                    | direct_stokes_solver_parameters
                )

        approximation = Approximation("BA", dimensional=False, parameters={"mu": mu})
        stokes_solver = StokesSolver(
            stokes_function, approximation, solver_parameters=solver_parameters
        )

        assert stokes_solver.solver_parameters == expected_value

    with pytest.raises(ValueError):
        StokesSolver(stokes_function, approximation, solver_parameters="")
