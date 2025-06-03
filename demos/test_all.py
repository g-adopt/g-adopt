import pytest
from pathlib import Path
import pandas as pd

mc_path = "mantle_convection"
gia_path = "glacial_isostatic_adjustment"

cases = {
    f"{mc_path}/base_case": {"extra_checks": ["nu_top"]},
    f"{mc_path}/free_surface": {"extra_checks": ["nu_top", "eta_min", "eta_max"]},
    f"{mc_path}/2d_compressible_TALA": {"extra_checks": ["nu_top"]},
    f"{mc_path}/2d_compressible_ALA": {"extra_checks": ["nu_top"]},
    f"{mc_path}/viscoplastic_case": {"extra_checks": ["nu_top"]},
    f"{mc_path}/2d_cylindrical": {"extra_checks": ["nu_top", "T_min", "T_max"]},
    f"{mc_path}/3d_spherical": {"extra_checks": ["nu_top", "t_dev_avg"]},
    f"{mc_path}/3d_cartesian": {"extra_checks": ["nu_top"], "rtol": 1e-4},
    f"{mc_path}/gplates_global": {"extra_checks": ["nu_top", "u_rms_top"]},
    "../tests/2d_cylindrical_TALA_DG": {
        "extra_checks": ["nu_top", "avg_t", "FullT_min", "FullT_max"]
    },
    "../tests/viscoplastic_case_DG": {"extra_checks": ["nu_top", "avg_t"]},
    f"{gia_path}/base_case": {"extra_checks": ["disp_min", "disp_max"]},
    f"{gia_path}/2d_cylindrical": {"extra_checks": ["disp_min", "disp_max"]},
}


def construct_pytest_params():
    out = []
    for case in cases:
        if case.startswith(".."):
            out.append(case)
        else:
            out.append(pytest.param(case, marks=pytest.mark.demo))
    return out


def get_convergence(base):
    return pd.read_csv(base / "params.log", sep="\\s+", header=0).iloc[-1]


def check_series(
    actual,
    expected,
    *,
    compare_params,
    convergence_tolerance,
    extra_checks,
):
    pd.testing.assert_series_equal(
        actual[["u_rms"] + extra_checks], expected,
        check_names=False, **compare_params
    )

    assert abs(actual.name - expected.name) <= convergence_tolerance


@pytest.mark.parametrize("benchmark", construct_pytest_params())
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

    if isinstance(expected, list):
        assertion = None

        for expected_df in expected:
            try:
                check_series(
                    df,
                    expected_df,
                    compare_params=compare_params,
                    convergence_tolerance=convergence_tolerance,
                    extra_checks=extra_checks,
                )
            except AssertionError as e:
                assertion = e
            else:
                # this test was successful, so we reset the assertion and
                # break early
                assertion = None
                break

        # if none of the tests passed, re-raise the last failure
        if assertion:
            raise assertion

    else:
        check_series(
            df,
            expected,
            compare_params=compare_params,
            convergence_tolerance=convergence_tolerance,
            extra_checks=extra_checks,
        )
