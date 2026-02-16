cases = ["elastic", "viscoelastic", "viscous"]

steps = {}
for c in cases:
    steps[c] = {
        "entrypoint": "zhong_viscoelastic_free_surface.py",
        "args": f"--case {c}",
        "outputs": [f"errors-{c}-zhong-free-surface.dat"],
    }

pytest = "local"
