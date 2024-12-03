import pandas as pd


# Minimal G-ADOPT example test
def test_base_case():
    df = pd.read_csv("params.log", sep="\\s+", header=0).iloc[-1]
    expected = pd.read_pickle("expected.pkl")
    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected, check_names=False)
    assert abs(df.name - expected.name) <= 2
