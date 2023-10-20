import argparse
import subprocess
import sys

cases = {
    5: {
        "cores": 24,
        "layers": 16,
        "timestep": 5e-8,
    },
    6: {
        "cores": 192,
        "layers": 32,
        "timestep": 2.5e-8,
    },
    7: {
        "cores": 1536,
        "layers": 64,
        "timestep": 1.25e-8,
    },
    8: {
        "cores": 12288,
        "layers": 128,
        "timestep": 6.25e-9,
    },
}


def run_subcommand(args):
    from stokes_cubed_sphere import model

    model(args.level, args.layers, args.timestep, steps=args.steps)


def submit_subcommand(args):
    config = cases[args.level]
    cores = config.pop("cores")
    command = args.template.format(cores=cores)

    proc = subprocess.Popen(
        [
            *command.split(), sys.executable, sys.argv[0],
            "run", "-n", "2",
            str(args.level), *[str(v) for v in config.values()],
        ]
    )
    if proc.wait() != 0:
        print(f"level {args.level} failed first timestep: {proc.returncode}")
        sys.exit(1)

    proc = subprocess.Popen(
        [
            *command.split(), sys.executable, sys.argv[0],
            "run", str(args.level), *[str(v) for v in config.values()],
        ]
    )
    if proc.wait() != 0:
        print(f"level {args.level} failed full timesteps: {proc.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="scaling",
        description="Run/submit parallel scaling test casse",
    )
    subparsers = parser.add_subparsers(title="subcommands")

    parser_run = subparsers.add_parser("run", help="run a specific configuration of a case (usually not manually invoked)")
    parser_run.add_argument("level", type=int)
    parser_run.add_argument("layers", type=int)
    parser_run.add_argument("timestep", type=float)
    parser_run.add_argument("-n", "--steps", type=int, help="number of timesteps to run")
    parser_run.set_defaults(func=run_subcommand)
    parser_submit = subparsers.add_parser("submit", help="submit a PBS job to run a specific case")
    parser_submit.add_argument("-t", "--template", default="mpiexec -np {cores}", help="template command for running commands under MPI")
    parser_submit.add_argument("level", type=int)
    parser_submit.set_defaults(func=submit_subcommand)

    args = parser.parse_args()
    args.func(args)
