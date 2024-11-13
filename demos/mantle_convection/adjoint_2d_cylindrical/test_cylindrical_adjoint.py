from pathlib import Path

import pytest
from cases import cases


@pytest.mark.parametrize("case_name", cases)
@pytest.mark.skip(reason="demo not yet working")
def test_annulus_taylor_test(case_name):
    with open(Path(__file__).parent.resolve() / f"{case_name}.conv", "r") as f:
        minconv = float(f.read())

    assert minconv > 1.9
