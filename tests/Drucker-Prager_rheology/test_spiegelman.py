import itertools
import numpy as np
from pathlib import Path
import pytest
from copy import deepcopy

conf = {
    "ui-mui": list(zip((2.5e-3, 5e-3), (1e23, 1e24))),
    "nx": (64,),
    "ny": (16,),
    "picard": [50, 0, 5, 15, 25],
    "stab": [False, True],
}


def param_sets_1e23():
    test_conf = deepcopy(conf)
    ui, mui = test_conf.pop("ui-mui")[0]
    return [(ui, mui, *i) for i in itertools.product(*test_conf.values())]


def param_sets_1e24():
    test_conf = deepcopy(conf)
    ui, mui = test_conf.pop("ui-mui")[1]
    del test_conf["picard"][0]  # drop picard-only
    return [(ui, mui, *i) for i in itertools.product(*test_conf.values())]


def run_id(ui, mui, nx, ny, picard_iterations, stabilisation):
    return f"{ui}_{mui}_{nx}_{ny}_{picard_iterations}_{stabilisation}"


def residual(dirname: str):
    output_dir = Path(__file__).parent.resolve() / dirname

    with open(output_dir / "picard.txt", "r") as f:
        picard_residuals = np.array([float(line.split()[1]) for line in f])

    with open(output_dir / "newton.txt", "r") as f:
        newton_residuals = np.array([float(line.split()[4]) for line in f])

    residuals = np.concatenate((picard_residuals, newton_residuals))

    return residuals[50] / residuals[0]


@pytest.mark.parametrize(
    "params", param_sets_1e23(), ids=[run_id(*i) for i in param_sets_1e23()]
)
def test_spiegelman_1e23(params):
    # test Picard-only relative residual goes below 5e-14 relative
    # test all Newton are below 5e-14 after 50 iterations
    assert residual(f"spiegelman_{run_id(*params)}") < 5e-14


@pytest.mark.parametrize(
    "params", param_sets_1e24(), ids=[run_id(*i) for i in param_sets_1e24()]
)
def test_spiegelman_1e24(params):
    ui, mui, nx, ny, _, stab = params

    if stab:
        # test Picard-only around 1e-3 after 50 iterations
        picard_dirname = f"spiegelman_{run_id(ui, mui, nx, ny, 50, False)}"
        picard_only = residual(picard_dirname)
        assert 1e-4 < picard_only < 1e-3
        # test all stabilised Newton below Picard after 50 iterations
        tol = picard_only
    else:
        # test all unstab. Newton below 1e-14 after 50 iterations
        tol = 1e-14
    assert residual(f"spiegelman_{run_id(*params)}") < tol
