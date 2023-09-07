import sys
from pathlib import Path

from test_all import cases, get_convergence

if __name__ == "__main__":
    if sys.argv[1:]:
        requested_cases = set(sys.argv[1:])
        cases = [
            case for case in cases.keys() if
            case in requested_cases
        ]

    for case in cases:
        b = Path(__file__).parent.resolve() / case
        df = get_convergence(b)[["u_rms", "nu_top"]]

        df.to_pickle(b / "expected.pkl")
