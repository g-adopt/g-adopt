from .richards import cases

steps = {}

for module_name, bc_dict in cases.items():
    for bc_type, degree_dict in bc_dict.items():
        for degree_key, config in degree_dict.items():
            case_name = f"{module_name}_{bc_type}_{degree_key}"
            for nodes, cores in zip(config["levels"], config["cores"]):
                degree = config["degree"]
                step_key = f"{case_name}-nodes{nodes}-dq{degree}"

                steps[step_key] = {
                    "entrypoint": "richards.py",
                    "args": f"{case_name} {nodes} {degree} {bc_type}",
                    "cores": cores,
                    "outputs": [f"errors-{step_key}.dat"],
                }

pytest = "local"
