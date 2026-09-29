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

The walltimes are about 1.5 times the measured runs on Gadi (2026-09-26 to
2026-09-29, `firedrake/main-20260918`): Spada cap and polar motion 1 h 01
min each, Martinec C 1 h 53 min, D 9 h 57 min, B 12 h 38 min over two jobs.
The normalsr queue allows at most 24 h for a job of 1144 to 2080 cores.

Case B does not finish in one job on Gadi at present. The job crashes after
step 998 of 1000 with a segmentation fault inside HCOLL, the collectives
library of the Gadi Open MPI, in its multicast broadcast
(`vmc_bcast_multiroot`), during an `MPI_Allreduce` of the low-rank DtN
operator. It happened in three runs at the same step. A job with
`RESTART=checkpoint_B.h5` finishes the case from the checkpoint at step 950
(see the README). Until the crash is fixed, the step for case B needs that
second job.

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
    ("spada", "cap", "gia_sp_cap", 1560, "01:30:00"),
    ("spada", "polar-motion", "gia_sp_pm", 1560, "01:30:00"),
    ("martinec", "B", "gia_mt_B", 1872, "18:00:00"),
    ("martinec", "C", "gia_mt_C", 1872, "03:00:00"),
    ("martinec", "D", "gia_mt_D", 1872, "15:00:00"),
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
