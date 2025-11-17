import operator
from pathlib import Path

import numpy as np
import pytest

base = (Path(__file__).parent / "benchmarks").resolve()

diagnostics = {
    "crameri_2012": [
        (
            lambda data: abs(
                np.subtract(data["max_topography"], data["max_topography_analytical"])
            ).max(),
            operator.le,
            4e-2,
        )
    ],
    "gerya_2003": [
        (
            lambda data: abs(
                1.0
                - np.asarray(data["block_area"])[
                    np.asarray(data["output_time"]) >= 1.0
                ].max()
            ),
            operator.le,
            5e-3,
        )
    ],
    "robey_2019": [
        (lambda data: abs(max(data["rms_velocity"]) - 284.4), operator.le, 1e-1),
        (lambda data: abs(max(data["entrainment"]) - 0.919), operator.le, 1e-3),
    ],
    "schmalholz_2011": [
        (
            lambda data: abs(
                np.asarray(data["normalised_time"])[
                    np.asarray(data["slab_necking"]) <= 0.2
                ].min()
                - 0.812
            ),
            operator.le,
            1e-3,
        )
    ],
    "schmeling_2008": [
        (
            lambda data: abs(
                np.asarray(data["output_time"])[
                    np.asarray(data["slab_tip_depth"]) >= 600.0
                ].min()
                - 47.8
            ),
            operator.le,
            1e-1,
        )
    ],
    "tosi_2015": [
        (lambda data: abs(data["avg_temperature"][-1] - 0.5275), operator.le, 8e-5),
        (lambda data: abs(data["nusselt_top"][-1] - 6.64), operator.le, 3e-2),
        (lambda data: abs(data["nusselt_bottom"][-1] - 6.65), operator.le, 4e-2),
        (lambda data: abs(data["rms_velocity"][-1] - 79.1), operator.le, 6e-2),
        (lambda data: abs(data["min_visc"][-1] - 1.92e-4), operator.le, 3e-7),
        (lambda data: abs(data["max_visc"][-1] - 1.92), operator.le, 6e-2),
    ],
    "trim_2023": [
        (lambda data: abs(data["rms_velocity"][-1] - 157.0796), operator.le, 5e0),
    ],
    "van_keken_1997_isothermal": [
        (lambda data: abs(max(data["rms_velocity"]) - 3.1e-3), operator.le, 4e-5),
        (lambda data: abs(data["rms_velocity"][-1] - 2.15e-4), operator.le, 2e-5),
        (lambda data: abs(data["entrainment"][-1] - 0.802), operator.le, 3e-3),
    ],
    "van_keken_1997_thermochemical": [
        (
            lambda data: abs(
                np.asarray(data["rms_velocity"])[
                    (np.asarray(data["output_time"]) >= 0.0214)
                    & (np.asarray(data["output_time"]) <= 0.0218)
                ].max()
                - 487.0
            ),
            operator.le,
            4e1,
        )
    ],
    "woidt_1978": [
        (lambda data: abs(max(data["rms_velocity"]) - 1.76e-11), operator.le, 1e-13),
        (lambda data: abs(max(data["entrainment"]) - 0.237), operator.le, 1e-3),
    ],
}


@pytest.mark.parametrize("bench_name,bench_diagnostics", diagnostics.items())
def test_multi_material(bench_name, bench_diagnostics):
    diag_file = np.load(base / bench_name / "output_0_reference.npz", allow_pickle=True)
    for diag_function, diag_operator, diag_threshold in bench_diagnostics:
        diag_data = diag_file["diag_fields"][()]
        assert diag_operator(diag_function(diag_data), diag_threshold)
