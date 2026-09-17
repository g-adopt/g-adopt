from pathlib import Path

import pandas as pd

base = Path(__file__).parent.resolve()


def test_field_integrals():
    df = pd.read_csv(base / "params.log", sep=r"\s+", header=0)
    expected = pd.read_pickle(base / "expected.pkl")

    # pandas >= 3 gives string column labels an Arrow-backed dtype when pyarrow
    # is installed and a plain one when it is not, so the label dtype differs
    # between a local run and the pyarrow-free CI image. Compare the values, not
    # that environment-dependent label dtype (the labels themselves are still
    # checked for equality).
    pd.testing.assert_frame_equal(df, expected, rtol=1e-3, check_column_type=False)
