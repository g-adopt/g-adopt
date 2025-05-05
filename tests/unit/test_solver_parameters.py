import os
from pathlib import Path
import re
import subprocess
import sys


def test_solver_params():
    unbuffered_env = os.environ.copy()
    unbuffered_env["PYTHONUNBUFFERED"] = "1"

    test_path = Path(__file__).parent.resolve() / "data/solver_params.py"

    params_output = subprocess.run(
        [sys.executable, str(test_path), "-options_monitor"],
        capture_output=True,
        text=True,
        env=unbuffered_env,
    ).stdout.splitlines()
    mark = params_output.index("BEGIN REDUCEDFUNCTIONAL CALL")

    state = 0  # searching for "setting option" line
    setting_re = r"Setting option: (.*?)_gadopt_test"
    solve_re = r"\s+Residual norms for {}"
    remove_re = r"Removing option: {}_gadopt_test"
    solve_name = None
    matches = 0

    for line in params_output[mark + 1 :]:
        if state == 0:
            if m := re.match(setting_re, line):
                solve_name = m.group(1)
                state = 1
                continue

        if state == 1:
            if re.match(solve_re.format(solve_name), line):
                state = 2
                continue

        if state == 2:
            if re.match(remove_re.format(solve_name), line):
                state = 0
                matches += 1
                solve_name = None
                continue

    assert matches > 0
