import operator
from pathlib import Path

import numpy as np
import pytest

base = Path(__file__).parent.resolve()

diagnostics = {
    "crameri_2012": [
        (
            lambda data: np.abs(
                np.subtract(data["max_topography"], data["max_topography_analytical"])
            ).max(),
            operator.le,
            0.15,
        )
    ],
    "gerya_2003": [
        (
            lambda data: np.abs(1 - np.asarray(data["block_area"])).max(),
            operator.le,
            0.035,
        )
    ],
    "robey_2019": [
        (lambda data: abs(max(data["rms_velocity"]) - 282.4), operator.le, 0.1),
        (lambda data: abs(max(data["entrainment"]) - 0.929), operator.le, 0.001),
    ],
    "schmalholz_2011": [
        (
            lambda data: abs(
                np.asarray(data["normalised_time"])[
                    np.asarray(data["slab_necking"]) <= 0.2
                ].min()
                - 0.83
            ),
            operator.le,
            0.002,
        ),
    ],
    "schmeling_2008": [
        (
            lambda data: abs(
                np.asarray(data["output_time"])[
                    np.asarray(data["slab_tip_depth"]) >= 600
                ].min()
                - 43
            ),
            operator.le,
            0.1,
        ),
    ],
    "trim_2023": [
        (
            lambda data: abs(
                np.asarray(data["rms_velocity"])[
                    np.asarray(data["output_time"]) <= 2.5e-3
                ].max()
                - 155
            ),
            operator.le,
            4,
        ),
    ],
}


@pytest.mark.parametrize("bench_name,bench_diagnostics", diagnostics.items())
def test_multi_material(bench_name, bench_diagnostics):
    diag_file = np.load(base / bench_name / "output.npz", allow_pickle=True)
    for diag_function, diag_operator, diag_threshold in bench_diagnostics:
        diag_data = diag_file["diag_fields"][()]
        assert diag_operator(diag_function(diag_data), diag_threshold)
