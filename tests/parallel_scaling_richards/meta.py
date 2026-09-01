"""HPC step registry for the Richards parallel-scaling long-test suite.

Each step covers one (case, solver, level) triple. ``CASE_SOLVERS`` in
``richards_scaling.py`` defines which solver presets are exercised per
case and records why each case drops the ones it drops.

The level number is a node count on Gadi's ``normalsr`` queue
(``CPUS_PER_NODE = 104``). All steps use ``run.template`` to export
``PETSC_OPTIONS="-log_view :profile_<tag>.txt"`` before launching the
driver, so the timings consumed by ``test_parallel_scaling_richards.py``
land in a predictable location.

Neither Murrumbidgee case reads external data: the driver builds its
terrain and its spatial fields from analytic ``omega`` surfaces, so a
step needs nothing in the job directory beyond the drivers themselves.
"""

from .richards_scaling import (
    CASES,
    CASE_SOLVERS,
    CPUS_PER_NODE,
    SEASONAL_PARAMETERS,
)


# Entry points are case-specific; both Murrumbidgee variants share one driver.
_ENTRYPOINTS = {
    "cockett": "cockett_3d.py",
    "murr_vertical": "murrumbidgee_3d.py",
    "murr_seasonal": "murrumbidgee_3d.py",
}


def _args_for(case: str, solver: str, params: dict) -> str:
    """Build the command-line tail for a driver invocation."""
    if case == "cockett":
        return (
            f"--nx {params['nx']} --nz {params['nz']} "
            f"--solver {solver} --steps {params['steps']}"
        )

    args = (
        f"--horiz-res {params['horiz_res']} --layers {params['layers']} "
        f"--solver {solver}"
    )
    if case == "murr_seasonal":
        # The seasonal regime is the ordinary basin driver plus a
        # three-month dt ceiling and the two soil levers that push the
        # column-integrated diffusion number up. See SEASONAL_PARAMETERS.
        p = SEASONAL_PARAMETERS
        args += (
            f" --dt-init {p['dt_init']} --dt-max {p['dt_max']}"
            f" --dt-growth {p['dt_growth']} --dt-shrink {p['dt_shrink']}"
            f" --t-final {p['t_final']}"
            f" --watertable-offset {p['watertable_offset']}"
            f" --retention-flatten {p['retention_flatten']}"
            f" --ss {p['ss']}"
        )
    return args


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


# Read-only input bundle for the basin cases: observational terrain and
# field grids. Fetched by doit's fetch_data task before any job is
# submitted, because compute nodes have no outbound network. Declared for
# every step so a partial run (a single case) still pulls it in; the
# download itself is skipped when the file is already present.
_BASIN_DATA = [{
    "url": "https://data.gadopt.org/github-actions/murrumbidgee_data.npz",
    "file": "murrumbidgee_data.npz",
}]
for _tag, _step in steps.items():
    if not _tag.startswith("cockett_"):
        _step["data"] = _BASIN_DATA
del _tag, _step

pytest_hpc = "local"
