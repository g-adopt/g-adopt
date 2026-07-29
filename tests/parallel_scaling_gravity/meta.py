r"""Layer-two doit metadata: one HPC step per scaling case.

Mirrors ``tests/parallel_scaling/meta.py``. Coupled cases run the model at the
weak-scaling ladder; capacitance cases run the diagnostic at levels 4-5. PBS
spool files are named ``pbs_*`` so they never collide with the JSON sidecars and
profile files the case produces.
"""

from .scaling import cases, COUPLED_L, CAPACITANCE_L, REPRESENTATIONS

CAPACITANCE_LEVELS = [4, 5]


# Short PBS job name (older PBSPro caps -N at 15 chars); the long `name` is kept
# for filenames. Matches the bare generator's scheme.
_PREFIX = {"coupled": "sv", "capacitance": "cap"}
_REP_SUFFIX = {"multiplier": "m", "lowrank": "r"}


def _step(name, level, lmax, kind, sidecar, representation=None):
    suffix = _REP_SUFFIX.get(representation or "", "")
    jobname = f"g{_PREFIX[kind]}{level}L{lmax}{suffix}"   # e.g. gsv4L5m, gcap4L30
    extra = f" --representation {representation}" if representation else ""
    return {
        "hpc_entrypoint": "scaling.py",
        "cores": cases[level]["cores"],
        "args": f"{level} --lmax {lmax} --kind {kind}{extra}",
        "outputs": [
            sidecar,
            f"profile_{name}.txt",
            f"pbs_{name}.out",
            f"pbs_{name}.err",
        ],
        "launcher_args": (
            f"-v LEVEL={level},LMAX={lmax},KIND={kind} -N {jobname} "
            f"-o pbs_{name}.out -e pbs_{name}.err --template-file ./run.template"),
    }


steps = {}

for level in cases:
    for lmax in COUPLED_L:
        for representation in REPRESENTATIONS:
            name = f"coupled_l{level}_L{lmax}_{representation}"
            steps[name] = _step(
                name, level, lmax, "coupled",
                f"summary_level{level}_lmax{lmax}_iterative_"
                f"{representation}.json",
                representation=representation)

for level in CAPACITANCE_LEVELS:
    for lmax in CAPACITANCE_L:
        name = f"capacitance_l{level}_L{lmax}"
        steps[name] = _step(
            name, level, lmax, "capacitance",
            f"capacitance_level{level}_lmax{lmax}.json")

pytest_hpc = "local"
