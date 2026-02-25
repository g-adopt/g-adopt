cases = ["damping", "smoothing", "Tobs", "uobs", "uimposed"]
schedulers = ["noscheduler", "fullmemory", "fullstorage"]

checkpoint_filename = "adjoint-demo-checkpoint-state.h5"

steps = {}
for c in cases:
    for s in schedulers:
        case_name = f"{c}_{s}"

        steps[case_name] = {
            "entrypoint": "taylor_test.py",
            "cores": 2,
            "args": case_name,
            "outputs": [f"{case_name}.conv"],
            "deps": [
                {
                    "case": "tests/adjoint",
                    "step": "forward",
                    "artifact": checkpoint_filename,
                    "link": False,
                },
            ],
        }

steps["forward"] = {
    "entrypoint": None,  # dependency-only
    "deps": [
        {
            "case": "demos/mantle_convection/adjoint",
            "step": "forward",
            "artifact": checkpoint_filename,
            "teardown": False,
        },
    ],
    "outputs": [checkpoint_filename],
}

pytest = "local"
