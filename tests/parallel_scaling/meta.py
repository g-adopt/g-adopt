from .scaling import cases

steps = {
    f"level_{level}": {
        "hpc_entrypoint": "scaling.py",
        "cores": cases[level]["cores"],
        "outputs": [
            f"profile_{level}.txt",
            f"level_{level}_warmup.out",
            f"level_{level}_warmup.err",
            f"level_{level}_full.out",
            f"level_{level}_full.err",
            f"l{level}.out",
            f"l{level}.err",
        ],
        "args": f"{level}",
        "launcher_args": f"-v LEVEL={level} -N scaling_{level} -o l{level}.out -e l{level}.err --template-file ./run.template",
    }
    for level in range(5, 8)
}

pytest_hpc = "local"
