#!/usr/bin/env python3
import argparse
from pathlib import Path

from test_all import cases, get_convergence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Expected demo results",
        description="Generates a pickle file with the expected demo results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("cases", nargs="*", help="Cases to process")
    args = parser.parse_args()

    requested_cases = {case.rstrip("/") for case in args.cases}

    for case in requested_cases:
        try:
            case_config = cases[case]
        except KeyError:
            print(f"Skipping unknown case: {case}")
            continue

        extra_checks = case_config.get("extra_checks", [])
        primary_checks = case_config.get("primary_checks", ["u_rms"])

        b = Path(__file__).parent.resolve() / case
        df = get_convergence(b)[primary_checks + extra_checks]

        df.to_pickle(b / "expected.pkl")
