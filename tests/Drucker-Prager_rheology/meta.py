from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from test_spiegelman import conf, param_sets  # noqa: E402

steps = {}
for param_set in param_sets(conf):
    conf_name = "_".join(str(x) for x in param_set[0] + param_set[1:])
    conf_dir = Path(f"spiegelman_{conf_name}")

    steps[conf_name] = {
        "entrypoint": "spiegelman.py",
        "args": conf_name,
        "outputs": [conf_dir / "picard.txt", conf_dir / "newton.txt"],
    }

pytest = "local"
