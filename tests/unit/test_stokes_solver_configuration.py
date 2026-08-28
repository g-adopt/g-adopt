import firedrake as fd
import pytest
from unittest.mock import patch

from copy import deepcopy

from gadopt.approximations import (
    BoussinesqApproximation,
    MaxwellApproximation,
    CompressibleInternalVariableApproximation,
)
from gadopt.stokes_integrators import (
    StokesSolver,
    InternalVariableSolver,
    CoupledInternalVariableSolver,
    direct_stokes_solver_parameters,
    iterative_outer_stokes_solver_parameters,
    newton_stokes_solver_parameters,
    gia_outer_solver_parameters,
    coupled_gia_solver_parameters,
    iterative_cuda_ksp_workarounds_inner
)
from gadopt.solver_options_manager import DeleteParam

TEST_CASES = [
    "unspecified",
    "direct",
    "iterative",
    "dictionary",
    "cartesian_false",
    "add_parameter",
    "delete_parameter",
    "change_tolerance",
    "linear_false"
]


# Section 1 - Pre-defined expected solver settings
#
# These are constructed manually here to avoid dependence on the solver options
# construction process being tested. Any entries to this section can use existing
# solver MappingProxyObjects from stokes_integrators.py but any modifications to
# those must be performed in this section and not by any of the machinery in
# StokesSolver.set_solver_options().
BASE_LINEAR_PARAMS_WITH_LOG = {"snes_type": "ksponly", "snes_monitor": None}

ITERATIVE_FIELDSPLIT_0_CPU = {
    "fieldsplit_0": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-5,
        "ksp_max_it": 1000,
        "pc_type": "python",
        "pc_python_type": "gadopt.SPDAssembledPC",
        "assembled": {
            "pc_type": "gamg",
            "mg_levels_pc_type": "sor",
            "pc_gamg_threshold": 0.01,
            "pc_gamg_square_graph": 100,
            "pc_gamg_coarse_eq_limit": 1000,
            "pc_gamg_mis_k_minimum_degree_ordering": True,
        },
    }
}

ITERATIVE_STOKES_CPU_PARAMS = (
    dict(iterative_outer_stokes_solver_parameters)
    | {
        "fieldsplit_1": iterative_outer_stokes_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_CPU)
)

ITERATIVE_STOKES_CPU_DEBUG_PARAMS = deepcopy(ITERATIVE_STOKES_CPU_PARAMS)
ITERATIVE_STOKES_CPU_DEBUG_PARAMS["fieldsplit_0"]["ksp_converged_reason"] = None
ITERATIVE_STOKES_CPU_DEBUG_PARAMS["fieldsplit_1"]["ksp_monitor"] = None

ITERATIVE_FREE_SURFACE_CPU_PARAMS = deepcopy(ITERATIVE_STOKES_CPU_PARAMS) | {
    "pc_fieldsplit_0_fields": "0",
    "pc_fieldsplit_1_fields": "1,2",
} | BASE_LINEAR_PARAMS_WITH_LOG
ITERATIVE_FREE_SURFACE_CPU_PARAMS["fieldsplit_1"]["pc_python_type"] = "gadopt.FreeSurfaceMassInvPC"

ITERATIVE_FIELDSPLIT_0_GPU = {
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "gadopt.SPDAssembledPC",
        "assembled": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.OffloadPC",
            "offload": {
                "ksp_type": "preonly",
                "pc_type": "ksp",
                "ksp": {
                    "ksp_type": "cg",
                    "ksp_rtol": 1e-5,
                    "ksp_max_it": 1000,
                    "pc_type": "gamg",
                    "mg_levels_pc_type": "jacobi",
                    "mg_levels_ksp_max_it": 2,
                    "pc_gamg_threshold": 0.01,
                    "pc_gamg_square_graph": 100,
                    "pc_gamg_coarse_eq_limit": 1000,
                    "pc_gamg_mis_k_minimum_degree_ordering": True,
                },
            },
        },
    }
}

ITERATIVE_STOKES_HIP_PARAMS = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | iterative_outer_stokes_solver_parameters
    | {
        "fieldsplit_1": iterative_outer_stokes_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU)
)

ITERATIVE_STOKES_CUDA_PARAMS = deepcopy(ITERATIVE_STOKES_HIP_PARAMS)
ITERATIVE_STOKES_CUDA_PARAMS["fieldsplit_0"]["assembled"]["offload"]["ksp"] |= iterative_cuda_ksp_workarounds_inner

ITERATIVE_STOKES_CUDA_DEBUG_PARAMS = deepcopy(ITERATIVE_STOKES_CUDA_PARAMS)
ITERATIVE_STOKES_CUDA_DEBUG_PARAMS["fieldsplit_0"]["assembled"]["offload"]["ksp"]["ksp_converged_reason"] = None
ITERATIVE_STOKES_CUDA_DEBUG_PARAMS["fieldsplit_1"]["ksp_monitor"] = None


ITERATIVE_FIELDSPLIT_0_GPU_TELESCOPE = {
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "gadopt.SPDAssembledPC",
        "assembled": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.OffloadPC",
            "offload": {
                "ksp_type": "preonly",
                "pc_type": "telescope",
                "pc_telescope_reduction_factor": 2,
                "telescope": {
                    "ksp_type": "preonly",
                    "pc_type": "ksp",
                    "ksp": {
                        "ksp_type": "cg",
                        "ksp_rtol": 1e-5,
                        "ksp_max_it": 1000,
                        "pc_type": "gamg",
                        "mg_levels_pc_type": "jacobi",
                        "mg_levels_ksp_max_it": 2,
                        "pc_gamg_threshold": 0.01,
                        "pc_gamg_square_graph": 100,
                        "pc_gamg_coarse_eq_limit": 1000,
                        "pc_gamg_mis_k_minimum_degree_ordering": True,
                    },
                },
            },
        },
    }
}

ITERATIVE_STOKES_HIP_PARAMS_TELESCOPE = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | iterative_outer_stokes_solver_parameters
    | {
        "fieldsplit_1": iterative_outer_stokes_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU_TELESCOPE)
)
ITERATIVE_STOKES_CUDA_PARAMS_TELESCOPE = deepcopy(ITERATIVE_STOKES_HIP_PARAMS_TELESCOPE)
ITERATIVE_STOKES_CUDA_PARAMS_TELESCOPE["fieldsplit_0"]["assembled"]["offload"]["telescope"]["ksp"] |= iterative_cuda_ksp_workarounds_inner

ITERATIVE_STOKES_CUDA_DEBUG_PARAMS_TELESCOPE = deepcopy(ITERATIVE_STOKES_CUDA_PARAMS_TELESCOPE)
ITERATIVE_STOKES_CUDA_DEBUG_PARAMS_TELESCOPE["fieldsplit_0"]["assembled"]["offload"]["telescope"]["ksp"]["ksp_converged_reason"] = None
ITERATIVE_STOKES_CUDA_DEBUG_PARAMS_TELESCOPE["fieldsplit_1"]["ksp_monitor"] = None

ITERATIVE_GIA_BASE = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | gia_outer_solver_parameters
)

ITERATIVE_GIA_CPU_PARAMS = ITERATIVE_GIA_BASE | deepcopy(
    ITERATIVE_FIELDSPLIT_0_CPU["fieldsplit_0"] | {"ksp_converged_reason": None}
)

ITERATIVE_GIA_HIP_PARAMS = ITERATIVE_GIA_BASE | deepcopy(
    ITERATIVE_FIELDSPLIT_0_GPU["fieldsplit_0"]
)
ITERATIVE_GIA_HIP_PARAMS["assembled"]["offload"]["ksp"]["ksp_converged_reason"] = None

ITERATIVE_GIA_CUDA_PARAMS = deepcopy(ITERATIVE_GIA_HIP_PARAMS)
ITERATIVE_GIA_CUDA_PARAMS["assembled"]["offload"]["ksp"] |= iterative_cuda_ksp_workarounds_inner

ITERATIVE_GIA_HIP_PARAMS_TELESCOPE = (
    deepcopy(ITERATIVE_GIA_BASE)
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU_TELESCOPE["fieldsplit_0"])
)
ITERATIVE_GIA_HIP_PARAMS_TELESCOPE["assembled"]["offload"]["telescope"]["ksp"]["ksp_converged_reason"] = None

ITERATIVE_GIA_CUDA_PARAMS_TELESCOPE = deepcopy(ITERATIVE_GIA_HIP_PARAMS_TELESCOPE)
ITERATIVE_GIA_CUDA_PARAMS_TELESCOPE["assembled"]["offload"]["telescope"]["ksp"] |= iterative_cuda_ksp_workarounds_inner

ITERATIVE_GIA_COUPLED_CPU_PARAMS = (
    {"snes_monitor": None}
    | coupled_gia_solver_parameters
    | {
        "fieldsplit_1": coupled_gia_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_CPU)
    | newton_stokes_solver_parameters
)

ITERATIVE_GIA_COUPLED_HIP_PARAMS = (
    {"snes_monitor": None}
    | coupled_gia_solver_parameters
    | {
        "fieldsplit_1": coupled_gia_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU)
    | newton_stokes_solver_parameters
)

ITERATIVE_GIA_COUPLED_CUDA_PARAMS = deepcopy(ITERATIVE_GIA_COUPLED_HIP_PARAMS)
ITERATIVE_GIA_COUPLED_CUDA_PARAMS["fieldsplit_0"]["assembled"]["offload"]["ksp"] |= iterative_cuda_ksp_workarounds_inner

DIRECT_GIA_COUPLED_CPU_PARAMS = (
    newton_stokes_solver_parameters | direct_stokes_solver_parameters
)

# Section 2 - Test variants
#
# Construct tuples that will be used to build parameter sets for fixtures in the
# sections. Tuples are expected to have 5 elements and contain a test type, expected
# direct and iterative solver settings, the device type passed to the solver and any
# additional GPU options.

TEST_VARIANTS = [
    ("stokes", direct_stokes_solver_parameters, ITERATIVE_STOKES_CPU_PARAMS, "HOST", {}),
    ("stokes_hip", direct_stokes_solver_parameters, ITERATIVE_STOKES_HIP_PARAMS, "HIP", {}),
    ("stokes_cuda", direct_stokes_solver_parameters, ITERATIVE_STOKES_CUDA_PARAMS, "CUDA", {}),
    ("stokes_cuda_telescope", direct_stokes_solver_parameters, ITERATIVE_STOKES_CUDA_PARAMS_TELESCOPE, "CUDA", {"telescope_factor": 2}),
    ("gia", direct_stokes_solver_parameters, ITERATIVE_GIA_CPU_PARAMS, "HOST", {}),
    ("gia_hip", direct_stokes_solver_parameters, ITERATIVE_GIA_HIP_PARAMS, "HIP", {}),
    ("gia_cuda", direct_stokes_solver_parameters, ITERATIVE_GIA_CUDA_PARAMS, "CUDA", {}),
    ("gia_cuda_telescope", direct_stokes_solver_parameters, ITERATIVE_GIA_CUDA_PARAMS_TELESCOPE, "CUDA", {"telescope_factor": 2}),
    ("coupled_gia", DIRECT_GIA_COUPLED_CPU_PARAMS, ITERATIVE_GIA_COUPLED_CPU_PARAMS, "HOST", {}),
    ("coupled_gia_gpu", DIRECT_GIA_COUPLED_CPU_PARAMS, ITERATIVE_GIA_COUPLED_CUDA_PARAMS, "CUDA", {}),
]

TEST_VARIANTS_3D = [
    ("stokes_3d", direct_stokes_solver_parameters, ITERATIVE_STOKES_CPU_PARAMS, "HOST", {}),
]


# Section 3 - Fixtures
#
# Define fixtures for mesh and fields and the test configurations. Differences in
# mesh/function configuration are determined by the test type. Since the fixture
# is evaluated before the test, raise the ValueError for invalid test type in
# mesh_and_fields.
@pytest.fixture
def mesh_and_fields(request):

    test_type = request.param[0]

    if test_type == "stokes_3d":
        mesh = fd.UnitCubeMesh(4, 4, 4)
    else:
        mesh = fd.UnitSquareMesh(10, 10, quadrilateral="gia" in test_type)
    mesh.cartesian = True

    if test_type.startswith("stokes"):
        func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
        func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
        func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
        stokes_function = fd.Function(func_space_stokes)

        func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
        temperature = fd.Function(func_space_temp, name="Temperature")

        return mesh, stokes_function, temperature

    # left with GIA tests after here
    V = fd.VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space
    S = fd.TensorFunctionSpace(mesh, "DQ", 1)  # Stress tensor function space
    DQ0 = fd.FunctionSpace(mesh, "DQ", 0)  # Density/viscosity/shear modulus function space
    density = fd.Function(DQ0).assign(1)

    if test_type.startswith("gia"):
        u = fd.Function(V, name="displacement")  # field to hold our displacement solution
        m = fd.Function(S, name="Internal variable")  # Lagged internal variable at previous timestep

        return mesh, u, m, density
    elif test_type.startswith("coupled_gia"):
        Z = fd.MixedFunctionSpace([V, S])
        z = fd.Function(Z)

        return mesh, z, density
    else:
        raise ValueError(f"Invalid test type: {test_type}")


@pytest.fixture
def case_configurations(request):

    cfg, direct_base, iterative_base, dim = request.param

    iterative_different_tolerance = deepcopy(iterative_base)
    if iterative_base["pc_type"] == "fieldsplit":
        # Indicates GPU solve
        if iterative_base["fieldsplit_0"]["ksp_type"] == "preonly":
            # Indicates 'telescope_factor' has been specified
            if "telescope" in iterative_different_tolerance["fieldsplit_0"]["assembled"]["offload"]:
                iterative_different_tolerance["fieldsplit_0"]["assembled"]["offload"]["telescope"]["ksp"]["ksp_rtol"] = 1e-4
                tol_dict = {"fieldsplit_0": {"assembled": {"offload": {"telescope": {"ksp": {"ksp_rtol": 1e-4}}}}},
                            "fieldsplit_1": {"ksp_rtol": 1e-3}}
            else:
                iterative_different_tolerance["fieldsplit_0"]["assembled"]["offload"]["ksp"]["ksp_rtol"] = 1e-4
                tol_dict = {"fieldsplit_0": {"assembled": {"offload": {"ksp": {"ksp_rtol": 1e-4}}}},
                            "fieldsplit_1": {"ksp_rtol": 1e-3}}
        else:
            iterative_different_tolerance["fieldsplit_0"]["ksp_rtol"] = 1e-4
            tol_dict = {"fieldsplit_0": {"ksp_rtol": 1e-4}, "fieldsplit_1": {"ksp_rtol": 1e-3}}
        iterative_different_tolerance["fieldsplit_1"]["ksp_rtol"] = 1e-3
    else:
        # Indicates GPU solve
        if iterative_base["ksp_type"] == "preonly":
            # Indicates 'telescope_factor' has been specified
            if "telescope" in iterative_different_tolerance["assembled"]["offload"]:
                iterative_different_tolerance["assembled"]["offload"]["telescope"]["ksp"]["ksp_rtol"] = 1e-4
                tol_dict = {"assembled": {"offload": {"telescope": {"ksp": {"ksp_rtol": 1e-4}}}}}
            else:
                iterative_different_tolerance["assembled"]["offload"]["ksp"]["ksp_rtol"] = 1e-4
                tol_dict = {"assembled": {"offload": {"ksp": {"ksp_rtol": 1e-4}}}}
        else:
            iterative_different_tolerance["ksp_rtol"] = 1e-4
            tol_dict = {"ksp_rtol": 1e-4}
    configs = {
        "unspecified": {
            "solver_parameters": None,
            "solver_parameters_extra": None,
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | (direct_base if dim == 2 else iterative_base),
            "mu": 1,
            "cartesian": True,
        },
        "direct": {
            "solver_parameters": "direct",
            "solver_parameters_extra": None,
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | direct_base,
            "mu": 1,
            "cartesian": True,
        },
        "iterative": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": None,
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_base,
            "mu": 1,
            "cartesian": True,
        },
        "dictionary": {
            "solver_parameters": {"mat_type": "aij", "ksp_type": "cg", "pc_type": "sor"},
            "solver_parameters_extra": None,
            "expected": {"mat_type": "aij", "ksp_type": "cg", "pc_type": "sor"},
            "mu": 1,
            "cartesian": True,
        },
        "cartesian_false": {
            "solver_parameters": None,
            "solver_parameters_extra": None,
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | (direct_base if dim == 2 else iterative_base),
            "mu": 1,
            "cartesian": False,
        },
        "add_parameter": {
            "solver_parameters": None,
            "solver_parameters_extra": {"ksp_converged_reason": None},
            "expected": {"ksp_converged_reason": None}
            | BASE_LINEAR_PARAMS_WITH_LOG | (direct_base if dim == 2 else iterative_base),
            "mu": 1,
            "cartesian": True,
        },
        "delete_parameter": {
            "solver_parameters": None,
            "solver_parameters_extra": {"snes_monitor": DeleteParam},
            "expected": {"snes_type": "ksponly"} | (direct_base if dim == 2 else iterative_base),
            "mu": 1,
            "cartesian": True,
        },
        "change_tolerance": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": tol_dict,
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_different_tolerance,
            "mu": 1,
            "cartesian": True,
        },
        "linear_false": {
            "solver_parameters": "direct",
            "solver_parameters_extra": None,
            "expected": BASE_LINEAR_PARAMS_WITH_LOG
            | direct_base
            | newton_stokes_solver_parameters,
            "mu": "sym_grad",  # Marker for non-constant mu
            "cartesian": True,
        },
    }

    return configs[cfg]


# Section 4 - Test parameters
#
# Construct test parameter tuples. Note that the 2nd and 3rd elements of this tuple
# are passed as arguments to their respective fixtures (test_case_config and
# mesh_and_fields), whereas the remainder are used directly. The GIA solver is in
# control of whether to use (non)-linear parameters, hence exclude the combination
# of "gia" and "linear_false"
all_tests = []
for test_type, direct_params, iter_params, device_type, gpu_param in TEST_VARIANTS:
    for case in TEST_CASES:
        # GIA tests do not have a linear/non-linear option
        if "gia" in test_type and case == "linear_false":
            continue
        all_tests.append(
            (
                test_type,
                (case, direct_params, iter_params, 2),
                (test_type,),
                device_type,
                gpu_param,
            )
        )
for test_type, direct_params, iter_params, device_type, gpu_param in TEST_VARIANTS_3D:
    for case in TEST_CASES:
        # GIA tests do not have a linear/non-linear option
        if "gia" in test_type and case == "linear_false":
            continue
        all_tests.append(
            (
                test_type,
                (case, direct_params, iter_params, 3),
                (test_type,),
                device_type,
                gpu_param,
            )
        )

debugging_tests = [
    ("stokes", ("stokes",), ("iterative", None, BASE_LINEAR_PARAMS_WITH_LOG | ITERATIVE_STOKES_CPU_DEBUG_PARAMS), "HOST", {}),
    ("stokes", ("stokes",), ("iterative", None, BASE_LINEAR_PARAMS_WITH_LOG | ITERATIVE_STOKES_CUDA_DEBUG_PARAMS), "CUDA", {}),
    ("stokes", ("stokes",), ("iterative", None, BASE_LINEAR_PARAMS_WITH_LOG | ITERATIVE_STOKES_CUDA_DEBUG_PARAMS_TELESCOPE), "CUDA", {"telescope_factor": 2}),
    ("gia", ("gia",), ("iterative", None, ITERATIVE_GIA_CPU_PARAMS | {"ksp_monitor": None}), "HOST", {}),
    ("gia", ("gia",), ("iterative", None, ITERATIVE_GIA_CUDA_PARAMS | {"ksp_monitor": None}), "CUDA", {})
]


def idfn(fixture_value):
    if isinstance(fixture_value, str):
        return fixture_value
    elif isinstance(fixture_value, tuple):
        return fixture_value[0]
    else:
        return ""


# Section 5 - Tests
#
# Selects the relevant tests and expected results for a given solver setup.
@pytest.mark.parametrize("test_type, case_configurations, mesh_and_fields, device_type, gpu_param", all_tests, ids=idfn, indirect=["case_configurations", "mesh_and_fields"])
def test_solver_parameters(test_type, case_configurations, mesh_and_fields, device_type, gpu_param):

    approximation = None
    solver = None

    mesh_and_field_out = mesh_and_fields

    mesh = mesh_and_field_out[0]
    # Apply cartesian flag override per test case
    mesh.cartesian = case_configurations["cartesian"]
    gpu_extra_parameters = {"device_type": device_type} | gpu_param

    if test_type.startswith("stokes"):
        _, stokes_function, temperature = mesh_and_field_out
        # Handle non-standard viscosity definition
        mu = case_configurations["mu"]
        if mu == "sym_grad":
            mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))

        approximation = BoussinesqApproximation(1, mu=mu)

        solver = StokesSolver(
            stokes_function,
            approximation,
            temperature,
            solver_parameters=case_configurations["solver_parameters"],
            solver_parameters_extra=case_configurations["solver_parameters_extra"],
            gpu_extra_parameters=gpu_extra_parameters,
        )
    elif test_type.startswith("gia"):
        _, u, m, density = mesh_and_field_out

        approximation = MaxwellApproximation(
            bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
        )

        solver = InternalVariableSolver(
            u,
            approximation,
            dt=1,
            internal_variables=m,
            solver_parameters=case_configurations["solver_parameters"],
            solver_parameters_extra=case_configurations["solver_parameters_extra"],
            gpu_extra_parameters=gpu_extra_parameters,
        )
    elif test_type.startswith("coupled_gia"):
        _, z, density = mesh_and_field_out

        approximation = CompressibleInternalVariableApproximation(
            bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
        )

        solver = CoupledInternalVariableSolver(
            z,
            approximation,
            dt=0.1,
            solver_parameters=case_configurations["solver_parameters"],
            solver_parameters_extra=case_configurations["solver_parameters_extra"],
            gpu_extra_parameters=gpu_extra_parameters,
        )

    # Verify solver parameters match expected configuration
    assert solver.solver_parameters == case_configurations["expected"]


@pytest.mark.parametrize("mesh_and_fields", [("stokes",)], ids=idfn, indirect=True)
def test_bad_solver_params_input(mesh_and_fields):
    _, stokes_function, temperature = mesh_and_fields

    approximation = BoussinesqApproximation(1)

    with pytest.raises(ValueError):
        StokesSolver(
            stokes_function,
            approximation,
            temperature,
            solver_parameters="",
        )


def test_free_surface_params():
    mesh = fd.UnitCubeMesh(4, 4, 4)
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = fd.FunctionSpace(mesh, "CG", 1)  # Pressure and free surface function space (scalar)
    Q = fd.FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = fd.MixedFunctionSpace([V, W, W])  # Mixed function space for velocity, pressure and eta.

    z = fd.Function(Z)  # A field over the mixed function space Z.
    T = fd.Function(Q)

    stokes_bcs = {
        3: {"uy": 0},
        4: {"free_surface": {"RaFS": 10.0}},
        1: {"ux": 0},
        2: {"ux": 0},
    }
    approximation = BoussinesqApproximation(1)
    solver = StokesSolver(
        z,
        approximation,
        T,
        dt=fd.Constant(1e-6),
        bcs=stokes_bcs,
        gpu_extra_parameters={"device_type": "HOST"},
    )

    assert solver.solver_parameters == ITERATIVE_FREE_SURFACE_CPU_PARAMS


@pytest.mark.parametrize("test_type, mesh_and_fields, test_params, device, gpu_extra_parameters", debugging_tests, ids=idfn, indirect=["mesh_and_fields"])
def test_debugging_config(test_type, mesh_and_fields, test_params, device, gpu_extra_parameters):

    solver_parameters, solver_parameters_extra, expected = test_params
    # Patch the version of log_level imported by stokes_integrators
    with patch("gadopt.stokes_integrators.log_level", 10):
        test_solver_parameters(
            test_type,
            {
                "solver_parameters": solver_parameters,
                "solver_parameters_extra": solver_parameters_extra,
                "expected": expected,
                "mu": 1,
                "cartesian": True,
            },
            mesh_and_fields,
            device,
            gpu_extra_parameters
        )
