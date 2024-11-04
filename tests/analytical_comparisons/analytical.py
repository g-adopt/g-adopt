import argparse
import importlib
import itertools
import subprocess
import sys

cases = {
    "smooth": {
        "cylindrical": {
            "freeslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "k": [2, 8],
                "n": [1, 4],
            },
            "zeroslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "k": [2, 8],
                "n": [1, 4],
            },
            "freesurface": {
                "cores": [1, 4, 6],
                "levels": [2**i for i in [1, 2, 3]],
                "k": [2, 8],
                "n": [1, 4],
            },
        },
        "spherical": {
            "freeslip": {
                "cores": [24, 48, 96, 192],  # cascade lake
                "levels": [3, 4, 5, 6],
                "l": [2, 8],
                "m": [2, 1],  # divide l by this value to get actual m
                "k": [3, 9],
                "permutate": False,
            },
            "zeroslip": {
                "cores": [24, 48, 96, 192],
                "levels": [3, 4, 5, 6],
                "l": [2, 8],
                "m": [2, 1],
                "k": [3, 9],
                "permutate": False,
            },
        },
    },
    "delta": {
        "cylindrical": {
            "freeslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
            "freeslip_dpc": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
            "zeroslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
            "zeroslip_dpc": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
        },
    },
}


def get_case(cases, config):
    config = config.split("_", maxsplit=2)
    while config:
        cases = cases[config.pop(0)]

    return cases


def param_sets(config, permutate=False):
    if permutate:
        return itertools.product(*config.values())

    return zip(*config.values())


def run_subcommand(args):
    from mpi4py import MPI

    try:
        model = importlib.import_module(args.case).model
    except ModuleNotFoundError:
        raise ValueError(f"unknown case {args.case}")

    errors = model(*args.params)
    if MPI.COMM_WORLD.rank == 0:
        config = get_case(cases, args.case)
        config.pop("cores")
        errfile_name = "errors-{}-{}.dat".format(
            args.case.replace("/", "_"),
            "-".join([f"{k}{v}" for k, v in zip(config.keys(), args.params)]),
        )

        with open(errfile_name, "w") as f:
            f.write(" ".join(str(x) for x in errors))


def submit_subcommand(args):
    config = get_case(cases, args.case)
    procs = {}

    permutate = config.pop("permutate", True)

    for level, cores in zip(config.pop("levels"), config.pop("cores")):
        for params in param_sets(config, permutate):
            paramstr = "-".join([str(v) for v in params])
            command = args.template.format(
                cores=cores, mem=4 * cores, params=paramstr, level=level
            )

            procs[paramstr] = subprocess.Popen(
                [
                    *command.split(),
                    sys.executable,
                    sys.argv[0],
                    "run",
                    args.case,
                    str(level),
                    *[str(v) for v in params],
                ]
            )

    failed = False

    for cmd, proc in procs.items():
        if proc.wait() != 0:
            print(f"{cmd} failed: {proc.returncode}")
            failed = True

    if failed:
        sys.exit(1)


def count_subcommand(args):
    config = get_case(cases, args.case)
    permutate = config.pop("permutate", True)
    levels = zip(config.pop("levels"), config.pop("cores"))
    params = param_sets(config, permutate)

    print(len(list(levels)) * len(list(params)))


# two modes, submit and run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analytical",
        description="Run/submit analytical test cases",
    )
    subparsers = parser.add_subparsers(title="subcommands")

    parser_run = subparsers.add_parser(
        "run",
        help="run a specific configuration of a case (usually not manually invoked)",
    )
    parser_run.add_argument("case")
    parser_run.add_argument("params", type=int, nargs="*")
    parser_run.set_defaults(func=run_subcommand)
    parser_submit = subparsers.add_parser(
        "submit", help="submit a PBS job to run a specific case"
    )
    parser_submit.add_argument(
        "-t",
        "--template",
        default="mpiexec -np {cores}",
        help="template command for running commands under MPI",
    )
    parser_submit.add_argument("case")
    parser_submit.set_defaults(func=submit_subcommand)
    parser_count = subparsers.add_parser(
        "count", help="return the number of jobs to run for a specific case"
    )
    parser_count.add_argument("case")
    parser_count.set_defaults(func=count_subcommand)

    args = parser.parse_args()
    args.func(args)
