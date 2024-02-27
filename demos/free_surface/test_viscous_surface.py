import numpy as np
from pathlib import Path
import pytest

base = Path(__file__).parent.resolve()


def run_benchmark(model):
    # Run default case run for four dt factors
    if model.name == 'explicit':
        dtf_start = 1
        iterative = False
    else:
        dtf_start = 2
        if model.name == 'implicit-cylindrical-iterative':
            iterative = False  # Cylinder cases already default to iterative so don't need to test this again!
        else:
            iterative = True

    dt_factors = dtf_start / (2**np.arange(4))

    errors = []
    errors_zeta = []

    for dtf in dt_factors:
        simulation = model(dtf)
        simulation.run_simulation()
        errors.append(simulation.final_error)

        if simulation.bottom_free_surface:
            errors_zeta.append(simulation.final_zeta_error)

    if simulation.bottom_free_surface:
        np.savetxt(f"errors-{model.name}-top-free-surface-coupling.dat", errors)
        np.savetxt(f"errors-{model.name}-bottom-free-surface-coupling.dat", errors_zeta)
    else:
        np.savetxt(f"errors-{model.name}-free-surface-coupling.dat", errors)

    if iterative:
        # Rerun with iterative solvers
        errors_iterative = []
        errors_zeta_iterative = []

        for dtf in dt_factors:
            simulation = model(dtf, iterative_2d=True)
            simulation.run_simulation()
            errors_iterative.append(simulation.final_error)

            if simulation.bottom_free_surface:
                errors_zeta_iterative.append(simulation.final_zeta_error)

        if simulation.bottom_free_surface:
            np.savetxt(f"errors-{model.name}-iterative-top-free-surface-coupling.dat", errors_iterative)
            np.savetxt(f"errors-{model.name}-iterative-bottom-free-surface-coupling.dat", errors_zeta_iterative)
        else:
            np.savetxt(f"errors-{model.name}-iterative-free-surface-coupling.dat", errors_iterative)


@pytest.fixture
def expected_errors():
    return np.load(base / "expected_errors.npz")


cases = [("explicit", [1.4, 1.1, 1.0]),
         ("implicit", [2.0, 2.0, 2.0]),
         ("implicit-iterative", [2.0, 2.0, 2.0]),
         ("implicit-both-top", [2.0, 2.0, 2.0]),
         ("implicit-both-bottom", [2.0, 2.0, 2.0]),
         ("implicit-both-iterative-top", [2.0, 2.0, 2.0]),
         ("implicit-both-iterative-bottom", [2.0, 2.0, 2.0]),
         ("implicit-buoyancy-top", [2.0, 2.0, 2.0]),
         ("implicit-buoyancy-bottom", [2.0, 2.0, 2.0]),
         ("implicit-buoyancy-iterative-top", [2.0, 2.0, 2.0]),
         ("implicit-buoyancy-iterative-bottom", [2.0, 2.0, 2.0]),
         ("implicit-cylindrical-iterative", [2.0, 2.0, 2.0]),
         ]


@pytest.mark.parametrize("coupling,expected_convergence", cases)
def test_free_surface(coupling, expected_convergence, expected_errors):

    expected_errors = expected_errors[coupling]
    errors = np.loadtxt(base / f"errors-{coupling}-free-surface-coupling.dat")

    # check that norm(q) is the same as previously run
    assert np.allclose(errors, expected_errors, rtol=1e-6, atol=1e-16)

    # use the highest resolution analytical solutions as the reference in scaling
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    print(f"{coupling} convergence:", convergence)
    print()
    assert np.allclose(convergence, expected_convergence, rtol=1e-1)
