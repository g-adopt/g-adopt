from .burgers_scaling import cases

configs = ["multiplicative", "schur-a11", "schur-substituted", "static-condensation"]

steps = {
    f"level_{level}_{config}": {
        "hpc_entrypoint": "burgers_scaling.py",
        "cores": cases[level]["cores"],
        "outputs": [
            f"profile_{level}_{config}.txt",
            f"level_{level}_{config}_warmup.out",
            f"level_{level}_{config}_warmup.err",
            f"level_{level}_{config}_full.out",
            f"level_{level}_{config}_full.err",
            f"l{level}_{config}.out",
            f"l{level}_{config}.err",
        ],
        "args": f"{level} {config}",
        "launcher_args": (
            f"-v LEVEL={level},CONFIG={config} -N burgers_{level}_{config} "
            f"-o l{level}_{config}.out -e l{level}_{config}.err "
            "--template-file ./run.template"
        ),
    }
    for level in (5, 6, 7)
    for config in configs
}

pytest_hpc = "local"
