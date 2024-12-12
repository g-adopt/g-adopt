import sys
from pathlib import Path

from test_all import cases, get_convergence

if __name__ == "__main__":
    all_cases = cases

    if sys.argv[1:]:
        requested_cases = set(sys.argv[1:])
        cases = [
            case.rstrip("/") for case in all_cases.keys() if case in requested_cases
        ]

    for case in cases:
        extra_checks = all_cases[case].get("extra_checks", [])
        b = Path(__file__).parent.resolve() / case
        df = get_convergence(b)[["u_rms"] + extra_checks]

        df.to_pickle(b / "expected.pkl")
