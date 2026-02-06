steps = {
    "forward": {
        "entrypoint": "adjoint_forward.py",
        "cores": 2,
        "outputs": ["adjoint-demo-checkpoint-state.h5"],
    },
    "adjoint": {
        "entrypoint": "adjoint.py",
        "cores": 1,
        "deps": [
            {
                "case": "demos/mantle_convection/adjoint",
                "step": "forward",
                "artifact": "adjoint-demo-checkpoint-state.h5",
            },
        ],
        "outputs": ["functional.txt"],
    },
}
pytest = "local"
