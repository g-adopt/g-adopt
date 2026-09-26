r"""doit metadata: one HPC step per case of the two self-gravitating benchmarks.

`doit run_case_hpc` submits each step through `gadopt_hpcrun` with
`run.template`, and `doit check_hpc` (or `pytest -m longtest`) then runs
`test_benchmarks.py` on the outputs. The layout follows
`tests/parallel_scaling_gravity/meta.py`.

The core counts are whole `normalsr` nodes (104 cores each) at about 20 000
degrees of freedom per core. The counts come from the meshes of
`selfgrav_common.MESHES`, with gmsh 4.15.2:

    mesh       unknowns     nodes   cores   unknowns per core
    spada      30 300 043   15      1560    19 423
    martinec   37 604 539   18      1872    20 088

80 percent of the unknowns are the DG2 internal variables, which the
preconditioner eliminates cell by cell.

The walltimes are provisional. No run on these meshes exists yet. Set each
one from the first timed run of its case. The normalsr queue allows at most
24 h for a job of 1144 to 2080 cores, so 24 h is the upper limit for all
five steps.

The Martinec steps read their case from the Python package `giamip`, in the
`demos` extra of G-ADOPT. The weekly Firedrake module build installs that
extra, so this route needs a module build that has `giamip`. `run.template`
adds nothing to `PYTHONPATH`. Until the module has `giamip`, submit the
Martinec cases with `run_benchmark.pbs` and its `EXTRA_PYTHONPATH` variable
(the header of that script gives the commands). `pip install --user` does not
work on Gadi, because the module Python is a virtual environment.
"""

#: `(driver, case, short name, cores, walltime)` for every step. PBSPro caps
#: the job name at 15 characters, so each step has a short name.
_CASES = (
    ("spada", "cap", "gia_sp_cap", 1560, "24:00:00"),
    ("spada", "polar-motion", "gia_sp_pm", 1560, "24:00:00"),
    ("martinec", "B", "gia_mt_B", 1872, "24:00:00"),
    ("martinec", "C", "gia_mt_C", 1872, "24:00:00"),
    ("martinec", "D", "gia_mt_D", 1872, "24:00:00"),
)


#: The terminal time of each Martinec case, kyr: the time of the published
#: profiles, which `test_benchmarks.py` reads.
_TERMINAL_KYR = {"B": 10, "C": 15, "D": 15}


def _outputs(driver, case):
    """The files that the step writes and that `test_benchmarks.py` reads."""
    files = [f"summary_{case}.json", f"params_{case}.log",
             f"pbs_{driver}_{case}.out", f"pbs_{driver}_{case}.err"]
    if driver == "martinec":
        files += [f"martinec-{case}-timeseries.npz",
                  f"martinec-{case}-profiles_{_TERMINAL_KYR[case]}kyr.npz"]
    return files


steps = {
    f"{driver}_{case}": {
        "hpc_entrypoint": f"{driver}.py",
        "cores": cores,
        "args": f"--case {case}",
        "outputs": _outputs(driver, case),
        "launcher_args": (
            f"-N {name} -t {walltime} "
            f"-o pbs_{driver}_{case}.out -e pbs_{driver}_{case}.err "
            "--template-file ./run.template"),
    }
    for driver, case, name, cores, walltime in _CASES
}

pytest_hpc = "local"
