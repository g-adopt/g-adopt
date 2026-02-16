steps = {}
for np in [1, 2]:
    steps[f"np{np}"] = {
        "entrypoint": "helmholtz.py",
        "cores": np,
        "outputs": [f"full_optimisation_np{np}.dat"],
    }

pytest = "local"
