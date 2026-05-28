from .richards import cases

steps = {}

for l1, v1 in cases.items():
    for l2, v2 in v1.items():
        for l3, config in v2.items():
            case_name = f"{l1}_{l2}_{l3}"
            for nodes, cores in zip(config["levels"], config["cores"]):
                degree = config["degree"]
                bc_type = config["bc_type"]
                step_key = f"{case_name}-nodes{nodes}-dq{degree}"

                steps[step_key] = {
                    "entrypoint": "richards.py",
                    "args": f"{case_name} {nodes} {degree} {bc_type}",
                    "cores": cores,
                    "outputs": [f"errors-{step_key}.dat"],
                }

pytest = "local"
