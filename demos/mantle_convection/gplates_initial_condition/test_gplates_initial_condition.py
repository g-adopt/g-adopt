from pathlib import Path

import pandas as pd

base = Path(__file__).parent.resolve()


def test_initial_condition_integrals():
    df = pd.read_csv(base / "params.log", sep=r"\s+", header=0)
    expected = pd.read_pickle(base / "expected.pkl")

    pd.testing.assert_frame_equal(df, expected, rtol=1e-3)
