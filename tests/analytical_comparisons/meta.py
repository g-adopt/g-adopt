# to get case information, we need a relative import to
# this file, but it's not in a package
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from analytical import cases, get_case  # noqa: E402
from test_analytical import configs, idfn  # noqa: E402

case_names = [
    "smooth_cylindrical_freeslip",
    "smooth_cylindrical_zeroslip",
    "delta_cylindrical_freeslip",
    "delta_cylindrical_zeroslip",
    "delta_cylindrical_freeslip_dpc",
    "delta_cylindrical_zeroslip_dpc",
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

pytest = "python3 -m tests/analytical_comparisons -m 'not longtest'"
