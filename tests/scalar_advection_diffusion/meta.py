from .test_scalar_advection_diffusion_DH27 import conf, param_sets

steps = {}
for params in param_sets:
    # formatted case name to run the case
    case_name = "_".join(str(x) for x in params[0])
    # formatted parameter string for output filename
    param_str = "_".join(f"{p[0]}{p[1]}" for p in zip(conf.keys(), params[0]))

    steps[case_name] = {
        "entrypoint": "scalar_advection_diffusion_DH27.py",
        "args": case_name,
        "outputs": [f"errors-{param_str}.dat"],
    }

steps["integrated_q"] = {
    "entrypoint": "scalar_advection_diffusion.py",
    "outputs": ["integrated_q.log"],
}
steps["integrated_q_DH219"] = {
    "entrypoint": "scalar_advection_diffusion_DH219_skew.py",
    "outputs": ["integrated_q_DH219.log"],
}

pytest = "local"
