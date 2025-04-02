import itertools
from pathlib import Path

import numpy as np

conf = {
    "ui-mui": list(zip((2.5e-3, 5e-3), (1e23, 1e24))),
    "nx": (64,),
    "ny": (16,),
    "picard": [50, 0, 5, 15, 25],
    "stab": [False, True],
}


def param_sets(config):
    return itertools.product(*config.values())


def residual(ui, mui, nx, ny, picard_iterations, stabilisation):
    output_dir = (
        Path(__file__).parent.resolve()
        / f"spiegelman_{ui}_{mui}_{nx}_{ny}_{picard_iterations}_{stabilisation}"
    )

    with open(output_dir / "picard.txt") as f:
        picard_residuals = np.array([float(line.split()[1]) for line in f])

    with open(output_dir / "newton.txt") as f:
        newton_residuals = np.array([float(line.split()[4]) for line in f])

    residuals = np.concatenate((picard_residuals, newton_residuals))

    return residuals[50] / residuals[0]


def test_spiegelman_1e23():
    test_conf = conf.copy()
    ui, mui = test_conf.pop("ui-mui")[0]
    del test_conf["picard"][0]  # drop picard-only

    # test Picard-only relative residual goes below 5e-14 relative
    # after 50 iterations
    picard_only = residual(ui, mui, test_conf["nx"][0], test_conf["ny"][0], 50, False)

    assert picard_only < 5e-14

    # test all Newton are below Picard-only after 50 iterations
    newton_residuals = np.array(
        [residual(ui, mui, *params) for params in param_sets(test_conf)]
    )

    assert np.all(newton_residuals < picard_only)


def test_spiegelman_1e24():
    test_conf = conf.copy()
    ui, mui = test_conf.pop("ui-mui")[1]
    del test_conf["picard"][0]  # drop picard-only

    # test Picard-only around 1e-3 after 50 iterations
    picard_only = residual(ui, mui, test_conf["nx"][0], test_conf["ny"][0], 50, False)

    assert 1e-4 < picard_only < 1e-3

    # test all stabilised Newton below Picard after 50 iterations
    stabilised_conf = test_conf.copy()
    stabilised_conf["stab"] = (True,)

    stabilised_residuals = np.array(
        [residual(ui, mui, *params) for params in param_sets(stabilised_conf)]
    )

    assert np.all(stabilised_residuals < picard_only)

    # test all unstab. Newton below 1e-14 after 50 iterations
    unstabilised_conf = test_conf.copy()
    unstabilised_conf["stab"] = (False,)

    unstabilised_residuals = np.array(
        [residual(ui, mui, *params) for params in param_sets(unstabilised_conf)]
    )

    assert np.all(unstabilised_residuals < 1e-14)


if __name__ == "__main__":
    for c in param_sets(conf):
        c = c[0] + c[1:]
        print("_".join([str(x) for x in c]))
