import argparse
from cases import cases, schedulers
from itertools import product
import subprocess
import sys


def submit_subcommand(args):
    case_names = list(cases.keys())
    scheduler_names = list(schedulers.keys())

    combinations = list(product(case_names, scheduler_names))

    procs = {}
    cores = 4
    for test_mode in combinations:

        command = args.template.format(
            cores=cores,
            mem=4 * cores,
            test_mode="_".join(test_mode),
        )
        # procs[test_mode] = subprocess.Popen(
        print(" ".join(
            [
                *command.split(),
                sys.executable,
                "inverse.py",
                "_".join(test_mode)
                ,
            ])
        )
        return

    failed = False

    for cmd, proc in procs.items():
        if proc.wait() != 0:
            print(f"{cmd} failed: {proc.returncode}")
            failed = True

    if failed:
        sys.exit(1)


def run_subcommand(args):
    pass


parser = argparse.ArgumentParser(
    prog="adjoint_2d_cylindrical",
    description="Run/submit adjoint 2D cylindrical test cases",
)
subparsers = parser.add_subparsers(title="subcommands")
parser_run = subparsers.add_parser(
    "run", help="Run a test case", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser_run.add_argument("case", help="Name of the test case to run")
parser_run.add_argument(
    "scheduler", help="Name of the scheduler to use"
)
parser_run.set_defaults(func=run_subcommand)

parser_submit = subparsers.add_parser(
    "submit", help="Submit a PBS job to run a specific case"
)
parser_submit.add_argument(
    "-t",
    "--template",
    default="mpiexec -np {cores}",
    help="Template command for running commands under MPI",
)
parser_submit.add_argument("case", help="Name of the test case to submit")
parser_run.add_argument(
    "scheduler", help="Name of the scheduler to use"
)
parser_submit.set_defaults(func=submit_subcommand)

args = parser.parse_args()
args.func(args)
