from .analytical import cases, get_case
from .test_analytical import longtest_cases, idfn

import itertools
from collections.abc import Generator


class ParamSet:
    def __init__(self, config):
        self.idx_map = {}
        idx = 0
        to_combine = []
        for k in config:
            if k in ("cores", "levels", "permutate"):
                continue
            to_combine.append(config[k])
            self.idx_map[idx] = k
            idx += 1
        if config.get("permutate", True):
            self.combinations = itertools.product(*to_combine)
        else:
            self.combinations = zip(*to_combine)

    def __iter__(self) -> Generator[tuple[str, str | None]]:
        for combination in self.combinations:
            yield (
                " ".join(str(i) for i in combination),
                idfn({self.idx_map[i]: v for i, v in enumerate(combination)}),
            )


case_names = [
    f"{i}_{j}_{k}"
    for i in cases
    for j in cases[i]
    for k in cases[i][j]
    if f"{i}_{j}_{k}" not in longtest_cases
]

steps = {}
for c in case_names:
    case_meta = get_case(cases, c)
    ps = ParamSet(case_meta)
    for test_input_str, test_id in ps:
        for level, cores in zip(case_meta["levels"], case_meta["cores"]):
            step_key = f"{c}-levels{level}-{test_id}"

            outputs = [f"errors-{step_key}.dat"]

            steps[step_key] = {
                "entrypoint": "analytical.py",
                "args": f"{c} {level} {test_input_str}",
                "cores": cores,
                "outputs": outputs,
            }

for c in longtest_cases:
    case_meta = get_case(cases, c)
    ps = ParamSet(case_meta)
    for test_input_str, test_id in ps:
        for level, cores in zip(case_meta["levels"], case_meta["cores"]):
            step_key = f"{c}-levels{level}-{test_id}"

            outputs = [f"errors-{step_key}.dat"]

            steps[step_key] = {
                "hpc_entrypoint": "analytical.py",
                "args": f"{c} {level} {test_input_str}",
                "launcher_args": f"-o {step_key}.out -e {step_key}.err -N analytical_{step_key}",
                "cores": cores,
                "outputs": outputs,
            }

pytest = "python3 -m pytest tests/analytical_comparisons -m 'not longtest'"

pytest_hpc = "python3 -m pytest tests/analytical_comparisons -m longtest"
