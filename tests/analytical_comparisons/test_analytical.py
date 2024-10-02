from pathlib import Path

import analytical
import numpy as np
import pandas as pd
import pytest

enabled_cases = {
    "smooth/cylindrical/free_slip": {"convergence": (4.0, 2.0), "rtol": 1e-1},
    "smooth/cylindrical/zero_slip": {"convergence": (4.0, 2.0)},
    "smooth/cylindrical/free_surface": {"convergence": (4.0, 2.0, 2.0), "rtol": 1e-1},
    "delta/cylindrical/free_slip": {"convergence": (1.5, 0.5)},
    "delta/cylindrical/zero_slip": {"convergence": (1.5, 0.5)},
    "delta/cylindrical/free_slip_dpc": {"convergence": (3.5, 2.0), "rtol": 1e-1},
    "delta/cylindrical/zero_slip_dpc": {"convergence": (3.5, 2.0), "rtol": 2e-1},
    "smooth/spherical/free_slip": {"convergence": (4.0, 2.0), "rtol": 1e-1},
    "smooth/spherical/zero_slip": {"convergence": (4.0, 2.0), "rtol": 1e-1},
}

longtest_cases = [
    "smooth/cylindrical/free_surface",
    "smooth/spherical/free_slip",
    "smooth/spherical/zero_slip",
]

params = {
    f"{l1}/{l2}/{l3}": v3
    for l1, v1 in analytical.cases.items()
    for l2, v2 in v1.items()
    for l3, v3 in v2.items()
    if f"{l1}/{l2}/{l3}" in enabled_cases.keys()
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
            b
            / "errors-{}-levels{}-{}.dat".format(
                name.replace("/", "_"), level, idfn(config)
            ),
            sep=" ",
            header=None,
        )
        for level in levels
    ]

    cols_anal = ["l2anal_u", "l2anal_p"]
    cols_err = ["l2error_u", "l2error_p"]
    if name.split("/")[-1] == "free_surface":
        cols_anal.append("l2anal_eta")
        cols_err.append("l2error_eta")

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

    assert np.allclose(
        convergence, expected_convergence, rtol=expected.get("rtol", 1e-2)
    )
