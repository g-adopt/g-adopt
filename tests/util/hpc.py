import os
import socket

from typing import Dict, Callable, Tuple, Set

batch_templates: Dict[str, str] = {
    "gadi": "qsub -v GADOPT_CHECKOUT={gadopt_checkout},GADOPT_SETUP={gadopt_setup} -W block=true -N scaling_{level} -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus={cores},walltime=04:00:00,mem={mem}GB,wd,jobfs=200GB -q normalsr -P {project} -o batch_output/l{level}.out -e batch_output/l{level}.err -- ./run_gadi.sh {level}",
    "setonix": "sbatch --export GADOPT_CHECKOUT={gadopt_checkout},GADOPT_SETUP={gadopt_setup} --wait -J scaling_{level} --exclusive --ntasks={cores} --nodes={nodes} -t 4:00:00 -p work -A {project} -o batch_output/l{level}.out -o batch_output/l{level}.err -- ./run_setonix.sh {level}",
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
    for var in required_environment[system]:
        if var not in os.environ:
            raise KeyError(f"{var} is required in environment when running on {system}")
    return system, batch_templates[system], {var: os.environ[var] for var in required_environment[system]}
