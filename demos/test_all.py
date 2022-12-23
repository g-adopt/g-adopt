import pytest
from pathlib import Path
import pandas as pd
import warnings

cases = [
    "base_case",
]

def get_convergence(base):
    return pd.read_csv(base / "params.log", sep="\\s+", header=0).iloc[-1]


@pytest.mark.parametrize("benchmark", cases)
def test_benchmark(benchmark):
    b = Path(__file__).parent.resolve() / benchmark
    df = get_convergence(b)
    expected = pd.read_pickle(b / "expected.pkl")

    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected, check_names=False)

    if df.name != expected.name:
        warnings.warn(f"Convergence changed: expected {expected.name}, got {df.name}")
