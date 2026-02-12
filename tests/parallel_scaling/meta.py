from .scaling import cases

steps = {
    f"level_{level}": {
        "hpc_entrypoint": "scaling.py",
        "cores": metadata["cores"],
        "outputs": [
            f"profile{level}.txt",
            f"l{level}.out",
            f"l{level}.err",
        ],
        "args": f"{level}",
        "launcher_args": f"-v LEVEL={level} -N scaling_{level} -o l{level}.out -e l{level}.err --template-file ./run.template",
    }
    for level, metadata in cases.items()
}

pytest = "local"
