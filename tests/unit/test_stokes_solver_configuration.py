import firedrake as fd
import pytest

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
)
from gadopt.solver_options_manager import DeleteParam

# Section 1 - Test cases
#
# Different solver settings need different test cases, group those here
ITERATIVE_TEST_CASES = ["iterative"]

TEST_CASES = [
    "unspecified",
    "direct",
    "dictionary",
    "cartesian_false",
    "add_parameter",
    "delete_parameter",
] + ITERATIVE_TEST_CASES

TEST_CASES_STOKES = ["linear_false", "change_tolerance_fs"]
TEST_CASES_STOKES_GPU = ["linear_false", "change_tolerance_fs_gpu"]
TEST_CASES_STOKES_GPU_TELESCOPE = ["change_tolerance_fs_gpu_telescope"]
TEST_CASES_GIA = ["change_tolerance"]
TEST_CASES_GIA_GPU = ["change_tolerance_gpu"]
TEST_CASES_GIA_GPU_TELESCOPE = ["change_tolerance_gpu_telescope"]
TEST_CASES_GIA_COUPLED = ["change_tolerance_fs"]
TEST_CASES_GIA_COUPLED_GPU = ["change_tolerance_fs_gpu"]

BASE_LINEAR_PARAMS_WITH_LOG = {"snes_type": "ksponly", "snes_monitor": None}

# Section 2 - Pre-defined expected solver settings
#
# These are constructed manually here to avoid dependence on the solver options
# construction process being tested. Any entries to this section can use existing
# solver MappingProxyObjects from stokes_integrators.py but any modifications to
# those must be performed in this section and not by any of the machinery in
# StokesSolver.set_solver_options().
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
                    "pc_gamg_square_0_mat_product_algorithm_backend_cpu": True,
                    "pc_gamg_square_1_mat_product_algorithm_backend_cpu": True,
                    "pc_gamg_square_2_mat_product_algorithm_backend_cpu": True,
                },
            },
        },
    }
}

ITERATIVE_STOKES_GPU_PARAMS = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | iterative_outer_stokes_solver_parameters
    | {
        "fieldsplit_1": iterative_outer_stokes_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU)
)

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
                        "pc_gamg_square_0_mat_product_algorithm_backend_cpu": True,
                        "pc_gamg_square_1_mat_product_algorithm_backend_cpu": True,
                        "pc_gamg_square_2_mat_product_algorithm_backend_cpu": True,
                    },
                },
            },
        },
    }
}

ITERATIVE_STOKES_GPU_PARAMS_TELESCOPE = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | iterative_outer_stokes_solver_parameters
    | {
        "fieldsplit_1": iterative_outer_stokes_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU_TELESCOPE)
)

ITERATIVE_GIA_CPU_PARAMS = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | gia_outer_solver_parameters
    | {"ksp_converged_reason": None}
    | deepcopy(ITERATIVE_FIELDSPLIT_0_CPU["fieldsplit_0"])
)

ITERATIVE_GIA_GPU_PARAMS = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | gia_outer_solver_parameters
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU["fieldsplit_0"])
)
ITERATIVE_GIA_GPU_PARAMS["assembled"]["offload"]["ksp"]["ksp_converged_reason"] = None

ITERATIVE_GIA_GPU_PARAMS_TELESCOPE = (
    deepcopy(BASE_LINEAR_PARAMS_WITH_LOG)
    | gia_outer_solver_parameters
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU_TELESCOPE["fieldsplit_0"])
)
ITERATIVE_GIA_GPU_PARAMS_TELESCOPE["assembled"]["offload"]["telescope"]["ksp"]["ksp_converged_reason"] = None

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

ITERATIVE_GIA_COUPLED_GPU_PARAMS = (
    {"snes_monitor": None}
    | coupled_gia_solver_parameters
    | {
        "fieldsplit_1": coupled_gia_solver_parameters["fieldsplit_1"]
        | {"ksp_converged_reason": None}
    }
    | deepcopy(ITERATIVE_FIELDSPLIT_0_GPU)
    | newton_stokes_solver_parameters
)

DIRECT_GIA_COUPLED_CPU_PARAMS = (
    newton_stokes_solver_parameters | direct_stokes_solver_parameters
)


# Section 3 - Mesh and Fields Fixtures
#
# Define fixtures for mesh and fields for Stokes 2D, Stokes 3D, GIA and Coupled GIA.
@pytest.fixture
def stokes_mesh_and_fields():
    mesh = fd.UnitSquareMesh(10, 10)
    mesh.cartesian = True

    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
    temperature = fd.Function(func_space_temp, name="Temperature")

    return mesh, stokes_function, temperature


@pytest.fixture
def stokes_mesh_and_fields_3d():
    mesh = fd.UnitCubeMesh(2, 2, 2)
    mesh.cartesian = True

    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
    temperature = fd.Function(func_space_temp, name="Temperature")

    return mesh, stokes_function, temperature


@pytest.fixture
def gia_mesh_and_fields():
    mesh = fd.UnitSquareMesh(10, 10, quadrilateral=True)
    mesh.cartesian = True

    V = fd.VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space
    S = fd.TensorFunctionSpace(mesh, "DQ", 1)  # Stress tensor function space
    DQ0 = fd.FunctionSpace(mesh, "DQ", 0)  # Density/viscosity/shear modulus function space
    u = fd.Function(V, name="displacement")  # field to hold our displacement solution
    m = fd.Function(S, name="Internal variable")  # Lagged internal variable at previous timestep
    density = fd.Function(DQ0).assign(1)

    return mesh, u, m, density


@pytest.fixture
def coupled_gia_mesh_and_fields():
    mesh = fd.UnitSquareMesh(10, 10, quadrilateral=True)
    mesh.cartesian = True

    V = fd.VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space
    S = fd.TensorFunctionSpace(mesh, "DQ", 1)  # Stress tensor function space
    DQ0 = fd.FunctionSpace(mesh, "DQ", 0)      # Density/viscosity/shear modulus function space

    Z = fd.MixedFunctionSpace([V, S])
    z = fd.Function(Z)
    density = fd.Function(DQ0).assign(1)

    return mesh, z, density


# Section 4 - Test definitions
#
# Constructs a dictionary with the test names as keys and a dict containing
# 'solver_parameters', 'solver_parameters_extra', 'expected', 'mu' and
# 'cartesian' keys. The argument to the fixture defines the configuration to be
# tested, expected parameters and the dimensionality of the problem. The configs
# dict is constructed in-place as much as possible.
@pytest.fixture
def test_case_config(request):

    cfg, direct_base, iterative_base, dim = request.param

    iterative_different_tolerance = deepcopy(iterative_base)
    if iterative_base["pc_type"] == "fieldsplit":
        # Indicates GPU solve
        if iterative_base["fieldsplit_0"]["ksp_type"] == "preonly":
            # Indicates 'telescope_factor' has been specified
            if "telescope" in iterative_different_tolerance["fieldsplit_0"]["assembled"]["offload"]:
                iterative_different_tolerance["fieldsplit_0"]["assembled"]["offload"]["telescope"]["ksp"]["ksp_rtol"] = 1e-4
            else:
                iterative_different_tolerance["fieldsplit_0"]["assembled"]["offload"]["ksp"]["ksp_rtol"] = 1e-4
        else:
            iterative_different_tolerance["fieldsplit_0"]["ksp_rtol"] = 1e-4
        iterative_different_tolerance["fieldsplit_1"]["ksp_rtol"] = 1e-3
    else:
        # Indicates GPU solve
        if iterative_base["ksp_type"] == "preonly":
            # Indicates 'telescope_factor' has been specified
            if "telescope" in iterative_different_tolerance["assembled"]["offload"]:
                iterative_different_tolerance["assembled"]["offload"]["telescope"]["ksp"]["ksp_rtol"] = 1e-4
            else:
                iterative_different_tolerance["assembled"]["offload"]["ksp"]["ksp_rtol"] = 1e-4
        else:
            iterative_different_tolerance["ksp_rtol"] = 1e-4
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
        "change_tolerance_fs": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": {
                "fieldsplit_0": {"ksp_rtol": 1e-4},
                "fieldsplit_1": {"ksp_rtol": 1e-3},
            },
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_different_tolerance,
            "mu": 1,
            "cartesian": True,
        },
        "change_tolerance_fs_gpu": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": {
                "fieldsplit_0": {"assembled": {"offload": {"ksp": {"ksp_rtol": 1e-4}}}},
                "fieldsplit_1": {"ksp_rtol": 1e-3},
            },
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_different_tolerance,
            "mu": 1,
            "cartesian": True,
        },
        "change_tolerance_fs_gpu_telescope": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": {
                "fieldsplit_0": {
                    "assembled": {"offload": {"telescope": {"ksp": {"ksp_rtol": 1e-4}}}}
                },
                "fieldsplit_1": {"ksp_rtol": 1e-3},
            },
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_different_tolerance,
            "mu": 1,
            "cartesian": True,
        },
        "change_tolerance": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": {"ksp_rtol": 1e-4},
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_different_tolerance,
            "mu": 1,
            "cartesian": True,
        },
        "change_tolerance_gpu": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": {
                "assembled": {"offload": {"ksp": {"ksp_rtol": 1e-4}}}
            },
            "expected": BASE_LINEAR_PARAMS_WITH_LOG | iterative_different_tolerance,
            "mu": 1,
            "cartesian": True,
        },
        "change_tolerance_gpu_telescope": {
            "solver_parameters": "iterative",
            "solver_parameters_extra": {
                "assembled": {"offload": {"telescope": {"ksp": {"ksp_rtol": 1e-4}}}}
            },
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


# Section 5 - Tests
#
# Selects the relevant tests and expected results for a given solver setup. Different
# solver setups have different relevant tests.
@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_STOKES_CPU_PARAMS, 2)
        for i in TEST_CASES + TEST_CASES_STOKES
    ],
    ids=TEST_CASES + TEST_CASES_STOKES,
    indirect=True,
)
def test_stokes_solver_parameters_cpu(test_case_config, stokes_mesh_and_fields):
    mesh, stokes_function, temperature = stokes_mesh_and_fields

    # Apply cartesian flag override per test case
    mesh.cartesian = test_case_config["cartesian"]

    # Handle non-standard viscosity definition
    mu = test_case_config["mu"]
    if mu == "sym_grad":
        mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))

    approximation = BoussinesqApproximation(1, mu=mu)

    stokes_solver = StokesSolver(
        stokes_function,
        approximation,
        temperature,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "HOST"},
    )

    # Verify solver parameters match expected configuration
    assert stokes_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_STOKES_CPU_PARAMS, 3)
        for i in TEST_CASES + TEST_CASES_STOKES
    ],
    ids=TEST_CASES + TEST_CASES_STOKES,
    indirect=True,
)
def test_stokes_solver_parameters_cpu_3d(test_case_config, stokes_mesh_and_fields_3d):
    mesh, stokes_function, temperature = stokes_mesh_and_fields_3d

    # Apply cartesian flag override per test case
    mesh.cartesian = test_case_config["cartesian"]

    # Handle non-standard viscosity definition
    mu = test_case_config["mu"]
    if mu == "sym_grad":
        mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))

    approximation = BoussinesqApproximation(1, mu=mu)

    stokes_solver = StokesSolver(
        stokes_function,
        approximation,
        temperature,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "HOST"},
    )

    # Verify solver parameters match expected configuration
    assert stokes_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_STOKES_GPU_PARAMS, 2)
        for i in TEST_CASES + TEST_CASES_STOKES_GPU
    ],
    ids=TEST_CASES + TEST_CASES_STOKES_GPU,
    indirect=True,
)
def test_stokes_solver_parameters_gpu(test_case_config, stokes_mesh_and_fields):
    mesh, stokes_function, temperature = stokes_mesh_and_fields

    # Apply cartesian flag override per test case
    mesh.cartesian = test_case_config["cartesian"]

    # Handle non-standard viscosity definition
    mu = test_case_config["mu"]
    if mu == "sym_grad":
        mu = fd.sym(fd.grad(fd.split(stokes_function)[0]))

    approximation = BoussinesqApproximation(1, mu=mu)

    stokes_solver = StokesSolver(
        stokes_function,
        approximation,
        temperature,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "CUDA"},
    )

    # Verify solver parameters match expected configuration
    assert stokes_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_STOKES_GPU_PARAMS_TELESCOPE, 2)
        for i in ITERATIVE_TEST_CASES + TEST_CASES_STOKES_GPU_TELESCOPE
    ],
    ids=ITERATIVE_TEST_CASES + TEST_CASES_STOKES_GPU_TELESCOPE,
    indirect=True,
)
def test_stokes_solver_parameters_gpu_telescope(
    test_case_config, stokes_mesh_and_fields
):
    mesh, stokes_function, temperature = stokes_mesh_and_fields

    # Apply cartesian flag override per test case
    mesh.cartesian = test_case_config["cartesian"]

    approximation = BoussinesqApproximation(1)

    stokes_solver = StokesSolver(
        stokes_function,
        approximation,
        temperature,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "CUDA", "telescope_factor": 2},
    )

    # Verify solver parameters match expected configuration
    assert stokes_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_GIA_CPU_PARAMS, 2)
        for i in TEST_CASES + TEST_CASES_GIA
    ],
    ids=TEST_CASES + TEST_CASES_GIA,
    indirect=True,
)
def test_gia_solver_parameters_cpu(test_case_config, gia_mesh_and_fields):
    mesh, u, m, density = gia_mesh_and_fields

    mesh.cartesian = test_case_config["cartesian"]

    approximation = MaxwellApproximation(
        bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
    )

    gia_solver = InternalVariableSolver(
        u,
        approximation,
        dt=1,
        internal_variables=m,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "HOST"},
    )

    assert gia_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_GIA_GPU_PARAMS, 2)
        for i in TEST_CASES + TEST_CASES_GIA_GPU
    ],
    ids=TEST_CASES + TEST_CASES_GIA,
    indirect=True,
)
def test_gia_solver_parameters_gpu(test_case_config, gia_mesh_and_fields):
    mesh, u, m, density = gia_mesh_and_fields

    mesh.cartesian = test_case_config["cartesian"]

    approximation = MaxwellApproximation(
        bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
    )

    gia_solver = InternalVariableSolver(
        u,
        approximation,
        dt=1,
        internal_variables=m,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "CUDA"},
    )

    assert gia_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, direct_stokes_solver_parameters, ITERATIVE_GIA_GPU_PARAMS_TELESCOPE, 2)
        for i in ITERATIVE_TEST_CASES + TEST_CASES_GIA_GPU_TELESCOPE
    ],
    ids=ITERATIVE_TEST_CASES + TEST_CASES_GIA_GPU_TELESCOPE,
    indirect=True,
)
def test_gia_solver_parameters_gpu_telescope(test_case_config, gia_mesh_and_fields):
    mesh, u, m, density = gia_mesh_and_fields

    mesh.cartesian = test_case_config["cartesian"]

    approximation = MaxwellApproximation(
        bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
    )

    gia_solver = InternalVariableSolver(
        u,
        approximation,
        dt=1,
        internal_variables=m,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "CUDA", "telescope_factor": 2},
    )

    assert gia_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, DIRECT_GIA_COUPLED_CPU_PARAMS, ITERATIVE_GIA_COUPLED_CPU_PARAMS, 2)
        for i in TEST_CASES + TEST_CASES_GIA_COUPLED
    ],
    ids=TEST_CASES + TEST_CASES_GIA_COUPLED,
    indirect=True,
)
def test_gia_coupled_solver_parameters_cpu(
    test_case_config, coupled_gia_mesh_and_fields
):
    mesh, z, density = coupled_gia_mesh_and_fields

    mesh.cartesian = test_case_config["cartesian"]

    approximation = CompressibleInternalVariableApproximation(
        bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
    )

    coupled_solver = CoupledInternalVariableSolver(
        z,
        approximation,
        dt=0.1,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "HOST"},
    )

    assert coupled_solver.solver_parameters == test_case_config["expected"]


@pytest.mark.parametrize(
    "test_case_config",
    [
        (i, DIRECT_GIA_COUPLED_CPU_PARAMS, ITERATIVE_GIA_COUPLED_GPU_PARAMS, 2)
        for i in TEST_CASES + TEST_CASES_GIA_COUPLED_GPU
    ],
    ids=TEST_CASES + TEST_CASES_GIA_COUPLED_GPU,
    indirect=True,
)
def test_gia_coupled_solver_parameters_gpu(
    test_case_config, coupled_gia_mesh_and_fields
):
    mesh, z, density = coupled_gia_mesh_and_fields

    mesh.cartesian = test_case_config["cartesian"]

    approximation = CompressibleInternalVariableApproximation(
        bulk_modulus=1, viscosity=1, shear_modulus=1, B_mu=1.27, density=density
    )

    coupled_solver = CoupledInternalVariableSolver(
        z,
        approximation,
        dt=0.1,
        solver_parameters=test_case_config["solver_parameters"],
        solver_parameters_extra=test_case_config["solver_parameters_extra"],
        gpu_extra_parameters={"device_type": "CUDA"},
    )

    assert coupled_solver.solver_parameters == test_case_config["expected"]


def test_bad_solver_params_input(stokes_mesh_and_fields):
    _, stokes_function, temperature = stokes_mesh_and_fields

    approximation = BoussinesqApproximation(1)

    with pytest.raises(ValueError):
        StokesSolver(
            stokes_function,
            approximation,
            temperature,
            solver_parameters="",
        )
