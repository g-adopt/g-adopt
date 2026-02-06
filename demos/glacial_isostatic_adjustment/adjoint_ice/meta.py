entrypoint = "adjoint_ice.py"
notebook = "adjoint_ice.ipynb"
deps = [
    {
        "case": "demos/glacial_isostatic_adjustment/2d_cylindrical_lvv",
        "step": "run",
        "artifact": "forward-2d-cylindrical-disp-vel.h5",
        "notebook": "2d_cylindrical_lvv.ipynb",
    },
]
outputs = ["functional.txt"]
pytest = "local"
