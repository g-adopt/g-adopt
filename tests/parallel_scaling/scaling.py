import argparse
import numpy as np
import os
import re
import subprocess
import sys
import socket
from pathlib import Path

cases = {
    5: {
        "cores": 26,
        "layers": 16,
        "timestep": 5e-8,
    },
    6: {
        "cores": 208,
        "layers": 32,
        "timestep": 2.5e-8,
    },
    7: {
        "cores": 1664,
        "layers": 64,
        "timestep": 1.25e-8,
    },
    8: {
        "cores": 13312,
        "layers": 128,
        "timestep": 6.25e-9,
    },
}


class HPCDetails:
    batch_templates = {
        "gadi": "qsub -v GADOPT_CHECKOUT={gadopt_checkout},GADOPT_SETUP={gadopt_setup} -W block=true -N scaling_{level} -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus={cores},walltime=04:00:00,mem={mem}GB,wd,jobfs=200GB -q normalsr -P {project} -o pbs_output/l{level}.out -e pbs_output/l{level}.err -- ./run_gadi.sh {level}",
        "setonix": "sbatch --export GADOPT_CHECKOUT={gadopt_checkout},GADOPT_SETUP={gadopt_setup} --wait -J scaling_{level} --exclusive --ntasks={cores} --nodes={nodes} -t 4:00:00 -p work -A {project} -o slurm_output/l{level}.out -o slurm_output/l{level}.err -- ./run_setonix.sh {level}",
    }
    required_environment = {
        "gadi": {"gadopt_checkout", "gadopt_setup", "project"},
        "setonix": {"gadopt_checkout", "gadopt_setup", "project"},
    }

    @staticmethod
    def hpc_system_is_gadi():
        return socket.gethostname().startswith("gadi")

    @staticmethod
    def hpc_system_is_setonix():
        if os.path.isfile("/software/pawsey/motd"):
            with open("/software/pawsey/motd") as f:
                for line in f.readlines():
                    if line == "     ___  ___| |_ ___  _ __ (_)_  __           ,########(,,  ...     ,########/,\n":
                        return True
            return False
        else:
            return False


def get_hpc_template():
    system = None
    for name in dir(HPCDetails):
        if name.startswith("hpc_system_is"):
            if getattr(HPCDetails, name)():
                system = name.split("_")[-1]
    if not system:
        raise KeyError("HPC system requested but could not identify system")
    # check environment variables
    for var in HPCDetails.required_environment[system]:
        if var not in os.environ:
            raise KeyError(f"{var} is required in environment when running on {system}")
    return HPCDetails.batch_templates[system], {var: os.environ[var] for var in HPCDetails.required_environment[system]}


def get_data(level, base_path=None):
    """Return the timing and iteration metrics for a given level"""

    base_path = base_path or Path()
    output_path = base_path / f"level_{level}_full.out"
    profile_path = base_path / f"profile_{level}.txt"

    if not (output_path.exists() and profile_path.exists()):
        raise FileNotFoundError(f"outputs for level {level} not found")

    # total time (profile)
    data = {}

    iteration_component_map = {
        "ImplicitMidpoint-Equation_stage0_": "energy",
        "Stokes_fieldsplit_0_": "velocity",
        "Stokes_fieldsplit_1_": "pressure",
    }

    iterations = {
        "energy": [],
        "velocity": [],
        "pressure": [],
    }

    with open(output_path, "r") as f:
        for line in f:
            if m := re.match(r"\s+Linear (\S+) solve converged due to CONVERGED_RTOL iterations (\d+)", line):
                iterations[iteration_component_map[m.group(1)]].append(int(m.group(2)))

    for k, v in iterations.items():
        data[f"{k}_iterations"] = np.mean(np.array(v))

    with open(profile_path, "r") as f:
        for line in f:
            if "stokes_solve:" in line:
                data["stokes_solve"] = float(line.split()[2])
            if "energy_solve:" in line:
                data["energy_solve"] = float(line.split()[2])

            if "snes_function" not in data and line.startswith("SNESFunctionEval"):
                data["snes_function"] = float(line.split()[3])
            if "snes_jacobian" not in data and line.startswith("SNESJacobianEval"):
                data["snes_jacobian"] = float(line.split()[3])

            # space is important to avoid the PCSetup_GAMG+ entry
            if "pc_setup" not in data and line.startswith("PCSetUp "):
                data["pc_setup"] = float(line.split()[3])

            if line.startswith("Time"):
                data["total_time"] = float(line.split()[4])

    return data


def run_subcommand(args):
    from stokes_cubed_sphere import model

    model(args.level, args.layers, args.timestep, steps=args.steps)


def submit_subcommand(args):
    config = cases[args.level]
    cores = config.pop("cores")
    command = args.template.format(
        cores=cores, mem=4*cores, level=args.level, nodes=max(1, cores//104), **args.extra_format
    )
    proc = subprocess.Popen(
        [
            *command.split(),
            os.path.basename(sys.executable),
            sys.argv[0],
            "run",
            str(args.level),
            *[str(v) for v in config.values()],
        ],
    )
    if proc.wait() != 0:
        print(f"level {args.level} failed: {proc.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="scaling",
        description="Run/submit parallel scaling test casse",
    )
    subparsers = parser.add_subparsers(title="subcommands")
    parser_run = subparsers.add_parser(
        "run", help="run a specific configuration of a case (usually not manually invoked)"
    )
    parser_run.add_argument("level", type=int)
    parser_run.add_argument("layers", type=int)
    parser_run.add_argument("timestep", type=float)
    parser_run.add_argument("-n", "--steps", type=int, help="number of timesteps to run")
    parser_run.set_defaults(func=run_subcommand)
    parser_submit = subparsers.add_parser("submit", help="submit a PBS job to run a specific case")
    group = parser_submit.add_mutually_exclusive_group()
    group.add_argument(
        "-t", "--template", default="mpiexec -np {cores}", help="template command for running commands under MPI"
    )
    group.add_argument(
        "-H",
        "--HPC",
        help="Detect HPC system and run using known template for batch job submission for that system",
        action="store_true",
    )
    parser_submit.add_argument("level", type=int)
    parser_submit.set_defaults(func=submit_subcommand)

    args = parser.parse_args()
    if hasattr(args, "HPC") and args.HPC:
        args.template, args.extra_format = get_hpc_template()
    args.func(args)
