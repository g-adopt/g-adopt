entrypoint = "free_surface.py"
notebook = "free_surface.ipynb"
notebook_ouptuts = ["temperature_warp.gif"]
deps = [
    {
        "case": "demos/mantle_convection/base_case",
        "notebook": "base_case.ipynb",
        "link": False,
    },
]
outputs = ["params.log"]
pytest = "auto"
