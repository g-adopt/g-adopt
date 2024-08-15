from pathlib import Path

import pandas as pd
import pytest

cases = {
    "base_case": {},
    "free_surface": {"extra_checks": ["eta_min", "eta_max"]},
    "2d_compressible_TALA": {},
    "2d_compressible_ALA": {},
    "viscoplastic_case": {},
    "2d_cylindrical": {"extra_checks": ["T_min", "T_max"]},
    "3d_spherical": {"extra_checks": ["t_dev_avg"]},
    "3d_cartesian": {"rtol": 1e-4},
    "gplates_global": {"extra_checks": ["u_rms_top"]},
}


def get_convergence(base):
    return pd.read_csv(base / "params.log", sep="\\s+", header=0).iloc[-1]


@pytest.mark.parametrize("benchmark", cases.keys())
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

    compare_params = cases[benchmark]
    convergence_tolerance = compare_params.pop("iterations", 2)
    extra_checks = compare_params.pop("extra_checks", [])

    pd.testing.assert_series_equal(
        df[["u_rms", "nu_top"] + extra_checks],
        expected,
        check_names=False,
        **compare_params,
    )

    assert abs(df.name - expected.name) <= convergence_tolerance
