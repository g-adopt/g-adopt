entrypoint = "smoothing.py"
notebook = "smoothing.ipynb"
cores = 1
deps = [
    {
        "case": "tests/adjoint_2d_cylindrical",
        "step": None,
        "artifact": "Checkpoint230.h5",
        "link_as": "smoothing-example.h5",
    },
]
outputs = ["params.log"]
pytest = "auto"
