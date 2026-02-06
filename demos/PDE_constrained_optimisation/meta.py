def _gen_step(name):
    return {
        "entrypoint": f"PDE_constrained_{name}.py",
        "notebook": f"PDE_constrained_{name}.ipynb",
        "cores": 1,
        "outputs": [f"functional_{name}.txt"],
    }


steps = {name: _gen_step(name) for name in ("boundary", "field")}
pytest = "local"
