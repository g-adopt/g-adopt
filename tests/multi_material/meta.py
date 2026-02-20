from pathlib import Path

current_dir = Path(__file__).parent
cases = current_dir.glob("benchmarks/*")

steps = {}
for c in cases:
    case_dir = c.relative_to(current_dir)
    steps[c.name] = {
        "entrypoint": "run_benchmark.py",
        "cores": 4,
        "args": f"{case_dir} --without-plot",
        "outputs": [case_dir / "output_0_reference.npz"],
    }

pytest = "local"
