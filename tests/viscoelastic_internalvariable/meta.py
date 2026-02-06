cases = ["viscoelastic-compressible", "viscoelastic-compressible-burgers", "elastic-compressible"]

steps = {}
for c in cases:
    steps[c] = {
        "entrypoint": "internalvariable_viscoelastic_freesurface.py",
        "args": f"--case {c}",
        "outputs": [f"errors-{c}-free-surface.dat"],
    }

pytest = "local"
