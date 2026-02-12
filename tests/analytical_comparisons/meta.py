from .analytical import cases, get_case
from .test_analytical import configs, idfn

case_names = [
    "smooth_cylindrical_freeslip",
    "smooth_cylindrical_zeroslip",
    "delta_cylindrical_freeslip",
    "delta_cylindrical_zeroslip",
    "delta_cylindrical_freeslip_dpc",
    "delta_cylindrical_zeroslip_dpc",
]

hpc_case_names = [
    "smooth_cylindrical_freesurface",
    "smooth_spherical_freeslip",
    "smooth_spherical_zeroslip",
]

steps = {}
for c in case_names:
    outputs = []
    levels = get_case(cases, c)["levels"]

    for config in [x for x in configs if x[0] == c]:
        for level in levels:
            outputs.append(f"errors-{c}-levels{level}-{idfn(config[2])}.dat")

    steps[c] = {
        "entrypoint": "analytical.py",
        "cores": 1,
        "args": """submit -t "tsp -N {cores} -f mpiexec -np {cores}" """ + c,
        "use_tsp": False,
        "outputs": outputs,
    }

for c in hpc_case_names:
    outputs = []
    levels = get_case(cases, c)["levels"]
    for config in [x for x in configs if x[0] == c]:
        for level in levels:
            outputs.append(f"errors-{c}-levels{level}-{idfn(config[2])}.dat")

    steps[c] = {
        "hpc_entrypoint": "analytical.py",
        "cores": 1,
        "args": """submit -t "tsp -N {cores} -f mpiexec -np {cores}" """ + c,
        "use_tsp": False,
        "outputs": outputs,
    }

pytest = "python3 -m tests/analytical_comparisons -m 'not longtest'"
