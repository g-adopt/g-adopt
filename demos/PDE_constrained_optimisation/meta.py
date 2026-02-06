def _gen_step(name):
    return {
        "entrypoint": f"PDE_constrained_{name}.py",
        "cores": 1,
        "outputs": [f"functional_{name}.txt"],
    }

steps = {name: _gen_step(name) for name in ("boundary", "field")}
pytest = "local"
