from .burgers_scaling import cases

configs = ["multiplicative", "schur-a11", "schur-substituted", "static-condensation"]

# The jobs write PETSc's nested log (ascii_xml): one campaign then serves both
# the regression values of test_burgers_scaling.py and the cost breakdown of
# plot_breakdown.py, and get_data reads either log format.

steps = {
    f"level_{level}_{config}": {
        "hpc_entrypoint": "burgers_scaling.py",
        "cores": cases[level]["cores"],
        "outputs": [
            f"profile_{level}_{config}.xml",
            f"level_{level}_{config}_warmup.out",
            f"level_{level}_{config}_warmup.err",
            f"level_{level}_{config}_full.out",
            f"level_{level}_{config}_full.err",
            f"l{level}_{config}.out",
            f"l{level}_{config}.err",
        ],
        "args": f"{level} {config}",
        "launcher_args": (
            f"-v LEVEL={level},CONFIG={config},LOG_FORMAT=ascii_xml "
            f"-N burgers_{level}_{config} "
            f"-o l{level}_{config}.out -e l{level}_{config}.err "
            "--template-file ./run.template"
        ),
    }
    for level in (5, 6, 7)
    for config in configs
}

pytest_hpc = "local"
