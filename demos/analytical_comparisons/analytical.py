import argparse
import itertools
import subprocess
import sys

cases = {
    "smooth": {
        "cylindrical": {
            "free_slip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "k": [2, 4, 8],
                "n": [1, 2, 4],
            },
            "zero_slip": {
                "k": [2, 4],
                "levels": [2, 3, 4, 5],
            },
        },
        "spherical": {
            "free_slip": {
                "l": [2, 4, 8],
                "k": [3, 5, 9],
                "levels": [3, 4, 5, 6],
                "layers": [8, 16, 32, 64],
            },
            "zero_slip": {
                "l": [2, 4, 8],
                "k": [3, 5, 9],
                "levels": [3, 4, 5, 6],
                "layers": [8, 16, 32, 64],
            },
        },
    },
    "delta": {
        "cylindrical": {
            "free_slip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 4, 8],
            },
            "free_slip_dpc": {
                "n": [2, 4, 8],
                "levels": [2, 3, 4, 5],
            },
            "zero_slip": {
                "n": [2, 4, 8],
                "levels": [2, 3, 4, 5],
            },
            "zero_slip_dpc": {
                "n": [2, 4, 8],
                "levels": [2, 3, 4, 5],
            },
        },
    },
}


def get_case(cases, config):
    config = config.split("/")
    while config:
        cases = cases[config.pop(0)]

    return cases


def run_subcommand(args):
    from mpi4py import MPI

    if args.case == "smooth/cylindrical/free_slip":
        from smooth_cylindrical_freeslip import model
    elif args.case == "delta/cylindrical/free_slip":
        from delta_cylindrical_freeslip import model
    else:
        raise ValueError(f"unknown case {args.case}")

    errors = model(*args.params)
    if MPI.COMM_WORLD.rank == 0:
        config = get_case(cases, args.case)
        config.pop("cores")
        errfile_name = "errors-{}-{}.dat".format(
            args.case.replace("/", "_"),
            "-".join([f"{k}{v}" for k, v in zip(config.keys(), args.params)])
        )

        with open(errfile_name, "w") as f:
            f.write(" ".join(str(x) for x in errors))


def submit_subcommand(args):
    config = get_case(cases, args.case)
    procs = {}

    for level, cores in zip(config.pop("levels"), config.pop("cores")):
        for params in itertools.product(*config.values()):
            command = args.template.format(cores=cores)
            paramstr = "-".join([str(v) for v in params])

            procs[paramstr] = subprocess.Popen(
                [
                    *command.split(), sys.executable, sys.argv[0],
                    "run", args.case, str(level), *[str(v) for v in params],
                ]
            )

    failed = False

    for cmd, proc in procs.items():
        if proc.wait() != 0:
            print(f"{cmd} failed: {proc.returncode}")
            failed = True

    if failed:
        sys.exit(1)


# two modes, submit and run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analytical",
        description="Run/submit analytical test cases",
    )
    subparsers = parser.add_subparsers(title="subcommands")

    parser_run = subparsers.add_parser("run", help="run a specific configuration of a case (usually not manually invoked)")
    parser_run.add_argument("case")
    parser_run.add_argument("params", type=int, nargs="*")
    parser_run.set_defaults(func=run_subcommand)
    parser_submit = subparsers.add_parser("submit", help="submit a PBS job to run a specific case")
    parser_submit.add_argument("-t", "--template", default="mpiexec -np {cores}", help="template command for running commands under MPI")
    parser_submit.add_argument("case")
    parser_submit.set_defaults(func=submit_subcommand)

    args = parser.parse_args()
    args.func(args)
