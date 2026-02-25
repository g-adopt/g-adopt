from .cases import cases, schedulers
import itertools

steps = {
    "forward": {
        "hpc_entrypoint": "forward.py",
        "cores": 16,
        "launcher_args": "-N adjoint_2d_cylindrical_forward -o forward.out -e forward.err -v PYOP2_SPMD_STRICT=1",
        "outputs": [
            "Checkpoint_State.h5",
            "forward.out",
            "forward.err",
        ],
    },
}
for case, scheduler in itertools.product(cases, schedulers):
    key = f"{case}_{scheduler}"
    steps[key] = {
        "hpc_entrypoint": "inverse.py",
        "cores": 16,
        "launcher_args": f"-m 64GB -N {key} -o {key}.out -e {key}.err -v PYOP2_SPMD_STRICT=1",
        "args": key,
        "outputs": [f"{key}_functional.dat", f"{key}.out", f"{key}.err"],
        "deps": [
            {
                "case": "tests/adjoint_2d_cylindrical",
                "step": "forward",
                "artifact": "Checkpoint_State.h5",
            },
        ],
    }

pytest_hpc = "local"
