"""Steps of the GIA cases. One step is one driver process for one entry.

The entries, their drivers, core counts and outputs are listed in
`gia_entries.py`, which the test files also read, so that a name exists in
one place only. Each entry becomes the step `<case>-<entry>`, and the steps
compile and run in parallel under `doit run_case`. The weekly entry has
`"hpc": True`: it becomes a step of `doit run_case_hpc`, which submits it
through `gadopt_hpcrun`.
"""
import shlex

from .gia_entries import CASE_DIR, ENTRIES, WEEKLY_TEST

steps = {}
for e in ENTRIES:
    step = {"cores": e.get("cores", 1), "outputs": e["outputs"],
            "args": f"--entry {e['entry']}"}
    # A weekly entry has "hpc": True and runs through gadopt_hpcrun.
    step["hpc_entrypoint" if e.get("hpc") else "entrypoint"] = e["driver"]
    if "deps" in e:
        step["deps"] = e["deps"]
    if "launcher_args" in e:
        step["launcher_args"] = e["launcher_args"]
    steps[f"{e['case']}-{e['entry']}"] = step

# `doit check` runs every test file of this directory after the PR steps.
pytest = "local"
# `doit check_hpc` runs this command, from the repository root, after the
# HPC steps. Only `test_gia_hpc.py` reads their outputs. "local" would also
# run the test files of the PR steps, whose outputs exist only on PR CI.
pytest_hpc = f"python3 -m pytest {shlex.quote(str(CASE_DIR / WEEKLY_TEST))}"
