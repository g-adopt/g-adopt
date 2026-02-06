entrypoint = "3d_sphere_burgers.py"
cores = 8
args = "--reflevel 4 --DG0_layers 3 --lateral_visc --viscosity_ratio 0.1"
outputs = ["params.log"]
pytest = "auto"
