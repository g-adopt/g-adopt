import pandas as pd
from pathlib import Path


# Minimal G-ADOPT example test
def test_base_case():
    b = Path(__file__).parent.resolve()
    df = pd.read_csv(b / "params.log", sep="\\s+", header=0).iloc[-1]
    expected = pd.read_pickle(b / "expected.pkl")
    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected, check_names=False)
    assert abs(df.name - expected.name) <= 2
