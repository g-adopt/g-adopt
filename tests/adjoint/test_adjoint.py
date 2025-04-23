import pytest
from pathlib import Path
from cases import cases, schedules


@pytest.mark.parametrize("case_name", cases)
@pytest.mark.parametrize("schedule_name", schedules.keys())
def test_rectangular_taylor_test(case_name, schedule_name):
    case_scheduler = f"{case_name}_{schedule_name}"
    with open(Path(__file__).parent.resolve() / f"{case_scheduler}.conv", "r") as f:
        minconv = float(f.read())

    assert minconv > 1.98
