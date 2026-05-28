"""HPC step registry for the Richards parallel-scaling long-test suite.

Each step covers one (case, solver, level) triple. ``CASE_SOLVERS`` in
``scaling.py`` defines which solver presets are exercised per case;
BoomerAMG (``iterative``) is excluded from the Murrumbidgee families
because Hypre diverges on that operator within a few time steps.

The level number is a node count on Gadi's ``normalsr`` queue
(``CPUS_PER_NODE = 104``). All steps use ``run.template`` to export
``PETSC_OPTIONS="-log_view :profile_<tag>.txt"`` before launching the
driver, so the timings consumed by ``test_parallel_scaling_richards.py``
land in a predictable location.
"""

from .richards_scaling import CASES, CASE_SOLVERS, CPUS_PER_NODE


# Entry points are case-specific; both Murrumbidgee variants share one driver.
_ENTRYPOINTS = {
    "cockett": "cockett_3d.py",
    "murr_vertical": "murrumbidgee_3d.py",
    "murr_horizontal": "murrumbidgee_3d.py",
}


def _args_for(case: str, solver: str, params: dict) -> str:
    """Build the command-line tail for a driver invocation."""
    if case == "cockett":
        return (
            f"--nx {params['nx']} --nz {params['nz']} "
            f"--solver {solver} --steps {params['steps']}"
        )
    # Both Murrumbidgee variants share the same flag surface; data_dir is
    # resolved relative to the job CWD (the test directory on Gadi).
    return (
        f"--horiz-res {params['horiz_res']} --layers {params['layers']} "
        f"--solver {solver} --data-dir ./murrumbidgee_data"
    )


steps = {}
for case, case_levels in CASES.items():
    for level, params in case_levels.items():
        cores = level * CPUS_PER_NODE
        for solver in CASE_SOLVERS[case]:
            tag = f"{case}_{solver}_{level}"
            steps[tag] = {
                "hpc_entrypoint": _ENTRYPOINTS[case],
                "cores": cores,
                "outputs": [
                    f"{tag}.out",
                    f"{tag}.err",
                    f"profile_{tag}.txt",
                ],
                "args": _args_for(case, solver, params),
                "launcher_args": (
                    f"-v TAG={tag} -N richards_{tag} "
                    f"-o {tag}.out -e {tag}.err "
                    f"--template-file ./run.template"
                ),
            }


pytest_hpc = "local"
