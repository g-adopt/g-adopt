resolution = 20

entrypoint = "3d_weerdesteijn_coupled.py"
cores = 8
args = "--refined_surface --bulk_shear_ratio 1.94 --dt_years 5000 --DG0_layers 2 --power_law --Tend 10e3"
outputs = ["params.log"]
pytest = "auto"
mesh = {
    "geo": "../3d_weerdesteijn/weerdesteijn_box_refined_surface_nondim.geo",
    "msh": f"weerdesteijn_box_refined_surface_{resolution}km_nondim.msh",
    "args": f"-setnumber refined_dx {resolution}",
}
