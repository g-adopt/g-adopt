import pytest
from pathlib import Path
import pandas as pd

cases = [
    "2D_cartesian/base_case",
    ("2D_cartesian/base_case", "minimal_"),
    "2D_cartesian/compressible_case",
    "2D_cartesian/viscoplastic_case",
    "2D_cylindrical",
    "3D_cartesian",
    "3D_spherical",
]


def get_convergence(base, prefix=""):
    return pd.read_csv(base / f"{prefix}params.log", sep="\\s+", header=0).iloc[-1]


@pytest.mark.parametrize("benchmark", cases)
def test_benchmark(benchmark):
    # multiple cases in one directory, give a prefix to the files
    prefix = ""
    if isinstance(benchmark, tuple):
        benchmark, prefix = benchmark

    b = Path(__file__).parent.resolve() / benchmark
    df = get_convergence(b, prefix)
    expected = pd.read_pickle(b / f"{prefix}expected.pkl")

    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected)
