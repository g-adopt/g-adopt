from pathlib import Path

import analytical
import numpy as np
import pandas as pd
import pytest

enabled_cases = {
    "smooth_cylindrical_freeslip": {
        "convergence": (4.0, 2.0, 2.0),
        "rtol": 1e-1,
        "ns_lb": 1e-1,
    },
    "smooth_cylindrical_zeroslip": {
        "convergence": (4.0, 2.0, 2.0),
        "ns_lb": 1e-1,
        "ns_ub": 7e-1,
    },
    "smooth_cylindrical_freesurface": {"convergence": (4.0, 2.0, 2.0), "rtol": 1e-1},
    "delta_cylindrical_freeslip": {"convergence": (1.5, 0.5, 2.0)},
    "delta_cylindrical_zeroslip": {"convergence": (1.5, 0.5, 2.0), "ns_lb": 5e-2},
    "delta_cylindrical_freeslip_dpc": {"convergence": (3.5, 2.0, 2.0), "rtol": 1e-1},
    "delta_cylindrical_zeroslip_dpc": {
        "convergence": (3.5, 2.0, 2.0),
        "rtol": 2e-1,
        "ns_lb": 5e-2,
    },
    "smooth_spherical_freeslip": {
        "convergence": (4.0, 2.0, 2.0),
        "rtol": 1e-1,
        "ns_lb": 1e-1,
    },
    "smooth_spherical_zeroslip": {
        "convergence": (4.0, 2.0, 2.0),
        "rtol": 1e-1,
        "ns_lb": 1e-1,
    },
}

longtest_cases = [
    "smooth_cylindrical_freesurface",
    "smooth_spherical_freeslip",
    "smooth_spherical_zeroslip",
]

params = {
    f"{l1}_{l2}_{l3}": v3
    for l1, v1 in analytical.cases.items()
    for l2, v2 in v1.items()
    for l3, v3 in v2.items()
    if f"{l1}_{l2}_{l3}" in enabled_cases.keys()
}

configs = []
for name, conf in params.items():
    # these two keys don't form a part of the parameter matrix
    conf = conf.copy()
    conf.pop("cores")
    conf.pop("levels")
    permutate = conf.pop("permutate", True)

    for combination in analytical.param_sets(conf, permutate):
        conf_tuple = (name, enabled_cases[name], dict(zip(conf.keys(), combination)))
        if name in longtest_cases:
            configs.append(pytest.param(*conf_tuple, marks=pytest.mark.longtest))
        else:
            configs.append(conf_tuple)


def idfn(val):
    if isinstance(val, dict):
        return "-".join([f"{k}{v}" for k, v in val.items()])


@pytest.mark.parametrize("name,expected,config", configs, ids=idfn)
def test_analytical(name, expected, config):
    levels = analytical.get_case(analytical.cases, name)["levels"]

    b = Path(__file__).parent.resolve()

    dats = [
        pd.read_csv(
            b / "errors-{}-levels{}-{}.dat".format(name, level, idfn(config)),
            sep=" ",
            header=None,
        )
        for level in levels
    ]

    cols_anal = ["l2anal_u", "l2anal_p"]
    cols_err = ["l2error_u", "l2error_p"]
    if name.split("_")[-1] == "freesurface":
        cols_anal.append("l2anal_eta")
        cols_err.append("l2error_eta")
    else:
        cols_anal.append("l2anal_sigma")
        cols_err.append("l2error_sigma")

    dat = pd.concat(dats)

    dat.columns = cols_err + cols_anal
    dat.insert(0, "level", levels)
    dat = dat.set_index("level")

    # use the highest resolution analytical solutions as the reference
    ref = dat.iloc[-1][cols_anal].rename(index=lambda s: s.replace("anal", "error"))
    errs = dat[cols_err] / ref
    errs = errs.reset_index(drop=True)  # drop resolution label

    convergence = np.log2(errs.shift() / errs).drop(index=0)
    expected_convergence = pd.Series(expected["convergence"], index=cols_err)

    if name.split("_")[-1] == "freesurface":
        # Make sure velocity, pressure, and free surface have the theoretical rates
        assert np.allclose(
            convergence, expected_convergence, rtol=expected.get("rtol", 1e-2)
        )
    else:
        # Make sure velocity and pressure have the theoretical rates
        assert np.allclose(
            convergence[cols_err[:-1]],
            expected_convergence[cols_err[:-1]],
            rtol=expected.get("rtol", 1e-2),
        )
        # Make sure normal stress convergence is bounded
        assert all(
            convergence[cols_err[-1]]
            > expected_convergence.iloc[-1] - expected.get("ns_lb", 1e-2)
        )
        assert all(
            convergence[cols_err[-1]]
            < expected_convergence.iloc[-1] + expected.get("ns_ub", 1e-2)
        )
