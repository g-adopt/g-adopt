import operator
from pathlib import Path

import numpy as np
import pytest

base = Path(__file__).parent.resolve()

diagnostics = {
    "gerya_2003": [
        ("block_area", lambda x: np.abs(1 - np.asarray(x)).max(), operator.le, 0.035)
    ],
    "robey_2019": [
        ("rms_velocity", lambda x: abs(max(x) - 282.4), operator.le, 0.1),
        ("entrainment", lambda x: abs(max(x) - 0.929), operator.le, 0.001),
    ],
}


@pytest.mark.parametrize("bench_name,bench_diagnostics", diagnostics.items())
def test_multi_material(bench_name, bench_diagnostics):
    diag_file = np.load(base / bench_name / "output.npz", allow_pickle=True)
    for diag_name, diag_function, diag_operator, diag_threshold in bench_diagnostics:
        diag_data = diag_file["diag_fields"][()][diag_name]
        assert diag_operator(diag_function(diag_data), diag_threshold)
