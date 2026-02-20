cases = {
    "explicit": {
        "cores": 1,
        "outputs": ["explicit"],
    },
    "implicit": {
        "cores": 1,
        "outputs": ["implicit", "implicit-iterative"],
    },
    "implicit_top_bottom": {
        "cores": 1,
        "outputs": ["implicit-both-top", "implicit-both-bottom", "implicit-both-iterative-top", "implicit-both-iterative-bottom"],
    },
    "implicit_top_bottom_buoyancy": {
        "cores": 8,
        "outputs": ["implicit-buoyancy-top", "implicit-buoyancy-bottom", "implicit-buoyancy-iterative-top", "implicit-buoyancy-iterative-bottom"],
    },
    "implicit_cylindrical": {
        "cores": 8,
        "outputs": ["implicit-cylindrical-iterative"],
    },
}

steps = {}
for c, params in cases.items():
    steps[c] = {
        "entrypoint": f"{c}_free_surface.py",
        "cores": params["cores"],
        "outputs": [f"errors-{o}-free-surface-coupling.dat" for o in params["outputs"]],
    }

pytest = "local"
