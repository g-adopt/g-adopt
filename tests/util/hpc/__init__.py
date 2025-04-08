import os
import socket
from pathlib import Path

from typing import Dict, Callable, Tuple, Set

batch_templates: Dict[str, str] = {
    "gadi": "qsub -v GADOPT_CHECKOUT={gadopt_checkout},GADOPT_SETUP={gadopt_setup} -W block=true -N {jobname} -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus={cores},walltime=04:00:00,mem={mem}GB,wd,jobfs=200GB -q normalsr -P {project} -o {outname} -e {errname} -- {script_path}",
    "setonix": "sbatch --export GADOPT_CHECKOUT={gadopt_checkout},GADOPT_SETUP={gadopt_setup} --wait -J {jobname} --exclusive --ntasks={cores} --nodes={nodes} -t 4:00:00 -p work -A {project} -o {outname} -e {errname} -- {script_path}",
}

required_environment: Dict[str, Set[str]] = {
    "gadi": {"gadopt_checkout", "project"},
    "setonix": {"gadopt_checkout", "project"},
}

system_identifiers: Dict[str, Callable] = {
    "gadi": lambda: socket.gethostname().startswith("gadi"),
    "setonix": lambda: socket.getfqdn().endswith("setonix.pawsey.org.au"),
}


def get_hpc_properties() -> Tuple[str, str, Dict[str, str]]:
    system = None
    for s, func in system_identifiers.items():
        if func():
            system = s
    if not system:
        raise KeyError("HPC system requested but could not identify system")
    # check environment variables
    this_dir = Path(__file__).resolve().parent
    for var in required_environment[system]:
        if var not in os.environ:
            raise KeyError(f"{var} is required in environment when running on {system}")
    format_params = {var: os.environ[var] for var in required_environment[system]}
    wd = Path().resolve()
    if (wd / f"run_{system}.sh").exists():
        script_dir=wd
    else:
        script_dir=this_dir
    format_params["script_path"] = str(script_dir / f"run_{system}.sh")
    if "gadopt_setup" not in format_params:
        format_params["gadopt_setup"] = str(this_dir / f"{system}_gadopt_setup.sh")
    return system, batch_templates[system], format_params
