from pathlib import Path
import itertools
from .test_spiegelman import conf, run_id

steps = {}
for param_set in itertools.product(*conf.values()):
    conf_name = run_id(*(param_set[0] + param_set[1:]))
    conf_dir = Path(f"spiegelman_{conf_name}")

    steps[conf_name] = {
        "entrypoint": "spiegelman.py",
        "args": conf_name,
        "outputs": [conf_dir / "picard.txt", conf_dir / "newton.txt"],
    }

pytest = "local"
