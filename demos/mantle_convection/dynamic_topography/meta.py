entrypoint = "dynamic_topography.py"
notebook = "dynamic_topography.ipynb"
deps = [
    {
        "case": "demos/mantle_convection/base_case",
        "step": "run",
        "artifact": "Final_State.h5",
        "notebook": "base_case.ipynb",
        "link": False,
    }
]
outputs = ["params.log"]
pytest = "auto"
