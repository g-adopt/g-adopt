import argparse
import asyncio
import itertools
import math
import os
import sys
import importlib
from pathlib import Path

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
                "cores": [24, 48, 104, 208],  # sapphire rapids
                "levels": [3, 4, 5, 6],
                "l": [2, 8],
                "m": [2, 1],  # divide l by this value to get actual m
                "k": [3, 9],
                "permutate": False,
            },
            "zeroslip": {
                "cores": [24, 48, 104, 208],
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


async def run_subcommand(args):
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


async def run_subproc(args, level, cores, params):
    paramstr = "-".join([str(v) for v in params])
    # HPC-specific formatting - will be ignored by the default template
    errname = args.errname.format(level=level, params=paramstr)
    outname = args.outname.format(level=level, params=paramstr)
    jobname = f"analytical_{paramstr}"
    if "procs_per_node" in args.extra_format:
        # This can't be known until we know the number of cores, but it is required for
        # job submission on some systems. asyncio.create_subprocess_exec uses execvpe,
        # therefore we can't use $(( shell maths )) either
        args.extra_format["nodes"] = math.ceil(cores / args.extra_format["procs_per_node"])

    command = args.template.format(
        cores=cores,
        mem=4 * cores,
        params=paramstr,
        level=level,
        errname=errname,
        outname=outname,
        jobname=jobname,
        **args.extra_format,
    )

    proc = await asyncio.create_subprocess_exec(
        *command.split(),
        os.path.basename(sys.executable),
        sys.argv[0],
        "run",
        args.case,
        str(level),
        *[str(v) for v in params],
    )

    await proc.wait()

    return level, paramstr, proc.returncode


async def submit_subcommand(args):
    config = get_case(cases, args.case)

    permutate = config.pop("permutate", True)

    procs = []
    for level, cores in zip(config.pop("levels"), config.pop("cores")):
        for params in param_sets(config, permutate):
            procs.append(run_subproc(args, level, cores, params))

    failed = False
    for coro in asyncio.as_completed(procs):
        lvl, cmd, rc = await coro
        if rc != 0:
            print(f"Level {lvl}, {cmd} failed: {rc}")
            failed = True
        else:
            print(f"Level {lvl}, {cmd} succeeded: {rc}")

    sys.exit(failed)


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
    parser_submit = subparsers.add_parser("submit", help="submit a PBS job to run a specific case")
    group = parser_submit.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--template",
        default="mpiexec -np {cores}",
        help="template command for running commands under MPI",
    )
    group.add_argument(
        "-H",
        "--HPC",
        help="Detect HPC system and run using known template for batch job submission for that system",
        action="store_true",
    )
    parser_submit.add_argument(
        "-e", "--errname", type=str, help="stderr file for batch job", default="batch_output/l{level}.err"
    )
    parser_submit.add_argument(
        "-o", "--outname", type=str, help="stdout file for batch job", default="batch_output/l{level}.out"
    )
    parser_submit.add_argument("case")
    parser_submit.set_defaults(func=submit_subcommand)

    args = parser.parse_args()
    if hasattr(args, "HPC") and args.HPC:
        # Find HPC utilities - only if needed.
        test_util_path = str((Path(__file__).resolve().parent.parent / "util"))
        sys.path.insert(0, test_util_path)
        from hpc import get_hpc_properties

        sys.path.pop(0)
        args.template, args.extra_format = get_hpc_properties()
    else:
        args.extra_format = {}

    asyncio.run(args.func(args))
