steps = {
    "forward": {
        "entrypoint": "adjoint_forward.py",
        "notebook": "adjoint_forward.ipynb",
        "cores": 2,
        "outputs": ["adjoint-demo-checkpoint-state.h5"],
    },
    "adjoint": {
        "entrypoint": "adjoint.py",
        "notebook": "adjoint.ipynb",
        "cores": 1,
        "deps": [
            {
                "case": "demos/mantle_convection/adjoint",
                "step": "forward",
                "artifact": "adjoint-demo-checkpoint-state.h5",
                "notebook": "adjoint_forward.ipynb",
            },
        ],
        "outputs": ["functional.txt"],
    },
}
pytest = "local"
