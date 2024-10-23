import pytest
from pathlib import Path
from cases import cases


@pytest.mark.parametrize("case_name", cases)
@pytest.mark.skip(reason="currently untested")
def test_annulus_taylor_test(case_name):
    with open(Path(__file__).parent.resolve() / f"{case_name}.conv", "r") as f:
        minconv = float(f.read())

    assert minconv > 1.9
