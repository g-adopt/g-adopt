steps = {
    "level_4": {
        "entrypoint": "3d_sphere_burgers.py",
        "cores": 8,
        "args": "--reflevel 4 --DG0_layers 3 --lateral_visc --viscosity_ratio 0.1 --write_output",
        "outputs": ["params.log"],
    }
}

for level in range(5, 8):
    weak_scale_factor = 2**(level - 5)
    steps[f"level_{level}"] = {
        "hpc_entrypoint": "3d_sphere_burgers.py",
        "cores": 26 * weak_scale_factor**3,
        "outputs": [
            f"profile_{level}.txt",
            f"level_{level}_warmup.out",
            f"level_{level}_warmup.err",
            f"level_{level}_full.out",
            f"level_{level}_full.err",
            f"l{level}.out",
            f"l{level}.err",
        ],
        "args": f"--reflevel {level} --DG0_layers {5*weak_scale_factor} --dt_years {250 / weak_scale_factor} --viscosity_ratio 0.1 --lateral_visc",
        "launcher_args": f"-v LEVEL={level},DT_YEARS={250 / weak_scale_factor},RUN_END={5000 / weak_scale_factor} -N scaling_{level} -o l{level}.out -e l{level}.err --template-file ./run.template",

    }

pytest = "auto"
pytest_hpc = "local"
