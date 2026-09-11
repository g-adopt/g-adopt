steps = {
    "forward": {
        "entrypoint": "forward.py",
        "notebook": "forward.ipynb",
        "cores": 2,
        "outputs": ["reference-state.h5"],
    },
    "inverse": {
        "entrypoint": "inverse.py",
        "notebook": "inverse.ipynb",
        "cores": 2,
        "deps": [
            {
                "case": "demos/optimisation/box",
                "step": "forward",
                "artifact": "reference-state.h5",
                "notebook": "forward.ipynb",
            },
        ],
        "outputs": ["rol-lbfgs_L2.json"],
    },
}
pytest = "local"
