import pytest
from pathlib import Path
import pandas as pd

cases = [
    "base_case",
    "2d_compressible",
    "viscoplastic_case",
    "2d_cylindrical",
]


def get_convergence(base):
    return pd.read_csv(base / "params.log", sep="\\s+", header=0).iloc[-1]


@pytest.mark.parametrize("benchmark", cases)
def test_benchmark(benchmark):
    """Test a benchmark case against the expected convergence result.

    We save the expected result (as a Pandas dataframe) pickled on
    disk. Similarly, we load the diagnostic parameters from the run to
    be tested into a dataframe. The dataframe then contains one row
    per iteration, where the columns correspond to the diagnostic
    values.

    Perhaps confusingly, the row "names" are the iteration number. For
    the first assertion, we only check the actual values, and not the
    number of iterations for convergence. The second assertion is to
    check that we take the same number of iterations to converge, with
    a little bit of leeway.

    """

    b = Path(__file__).parent.resolve() / benchmark
    df = get_convergence(b)
    expected = pd.read_pickle(b / "expected.pkl")

    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected, check_names=False)

    assert abs(df.name - expected.name) <= 2
