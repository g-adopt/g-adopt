r"""Generate (and optionally submit) the Gadi PBS scripts for the scaling study.

Writes one self-contained PBS script per case into
``results/level_{level}/lmax_{L}/`` in the ``morrow2026`` bare-submission style.
Coupled-solve cases run at ``L <= 10`` (the 128-field wall that used to cap them
at 6 is gone; what remains is the O(L^4) cost of the boundary assembly), and the
capacitance diagnostic at all ``L`` on levels 4-5 (plus a level-6 spot check).

Dry run (default) writes the scripts for inspection; ``--submit`` calls ``qsub``.

    <python> submit_gravity_scaling.py                 # dry run, all cases
    <python> submit_gravity_scaling.py --levels 4 5    # subset
    <python> submit_gravity_scaling.py --submit        # qsub each

Warm the shared kernel cache before the expensive levels: submit level 4 first
(`--submit --levels 4`), let it finish, then submit the rest
(`--submit --levels 5 6 7`). Because kernels depend on the form, not the mesh
size, the level-4 compile populates the pinned `PYOP2_CACHE_DIR` and the larger
levels reuse it — this replaces a `qsub -W depend` chain with a simple two-step
submission and avoids first-compile write races at 512 ranks.

The environment block targets Gadi: project xd2, the fp50 Firedrake module, and
PYTHONPATH pinned to the deployment worktree (essential — the editable G-ADOPT
install otherwise points Python at a different checkout).
"""

import argparse
import subprocess
from pathlib import Path

# --- Gadi deployment configuration -----------------------------------------
PROJECT = "xd2"
STORAGE = "scratch/xd2+gdata/fp50"
MODULE_USE = "/g/data/fp50/modules"
MODULE = "firedrake/main-20260716"
WORKTREE = "/scratch/xd2/sg8812/g-adopt-worktrees/sghelichkhani/poisson"

# Weak-scaling ladder. Gadi's normal queue requires the PBS ncpus *request* to
# be a whole number of 48-core nodes for any multi-node job, but the study wants
# mpiexec to launch exactly the ladder core count. So the request (request_ncpus)
# and the launch (launch_ncpus) are decoupled: level 6 requests 2 nodes (96) and
# launches 64; level 7 requests 11 nodes (528) and launches 512. Memory follows
# the node ceiling (~190 GB usable/node) multi-node, ncpus*4 sub-node.
NODE_CORES = 48
NODE_MEM_GB = 190
LADDER = {
    4: {"launch": 1, "request": 1, "coupled_wall": "00:20:00"},
    5: {"launch": 8, "request": 8, "coupled_wall": "00:20:00"},
    6: {"launch": 64, "request": 96, "coupled_wall": "01:00:00"},
    7: {"launch": 512, "request": 528, "coupled_wall": "04:00:00"},
}


def mem_gb(request_ncpus):
    """GB to request: node ceiling for multi-node, ncpus*4 sub-node."""
    if request_ncpus > NODE_CORES:
        return (request_ncpus // NODE_CORES) * NODE_MEM_GB
    return max(4, request_ncpus * 4)


# The 128-field wall is gone (the two Schur blocks are described by index sets),
# so the coupled solve reaches high truncation too. L = 7 and 10 are carried to
# exercise past the old cap; beyond that the O(L^4) boundary assembly makes the
# coupled solve the expensive way to answer a question the capacitance
# diagnostic already answers wall-free.
COUPLED_L = [2, 5, 6, 7, 10]
CAPACITANCE_L = [2, 5, 10, 20, 30]    # wall-free
# The O(L^4) boundary assembly is worst at the coarsest level (per-rank facets
# halve per level), so the level-4/5 high-L capacitance rows get extra walltime.
CAPACITANCE_LEVELS = [4, 5]
CAPACITANCE_SPOT = (6, 10)            # single level-6 spot check at one L


def capacitance_wall(level, lmax):
    if level <= 5 and lmax >= 20:
        return "02:00:00"
    if lmax >= 10:
        return "00:40:00"
    return "00:20:00"


PBS_TEMPLATE = """#!/bin/bash
#PBS -P {project}
#PBS -q normal
#PBS -l ncpus={request_ncpus}
#PBS -l mem={mem}GB
#PBS -l walltime={walltime}
#PBS -l storage={storage}
#PBS -l wd
#PBS -N {jobname}
#PBS -o pbs_{jobname}.out
#PBS -e pbs_{jobname}.err

source /etc/profile
module use {module_use}
module load {module}

export PYTHONPATH={worktree}:$PYTHONPATH
# One BLAS thread per rank: the capacitance dense eig runs redundantly on every
# rank, and threaded BLAS on top would thrash the node.
export OMP_NUM_THREADS=1
# Pin the kernel cache to a shared filesystem so warm-up compiles are visible to
# later jobs (fp50 may otherwise redirect it to node-local jobfs).
export PYOP2_CACHE_DIR={worktree}/.cache/pyop2
export XDG_CACHE_HOME={worktree}/.cache
export PETSC_OPTIONS="-log_view :{casedir}/profile.txt -options_left"

cd {casedir}
mpiexec -np {launch_ncpus} python3 {worktree}/tests/parallel_scaling_gravity/{script} \\
    {level} --lmax {lmax} {extra} \\
    --out-dir {casedir} > {casedir}/run.out 2> {casedir}/run.err
"""


def case_dir(results_dir, kind, level, lmax):
    return Path(results_dir) / kind / f"level_{level}" / f"lmax_{lmax}"


def write_script(results_dir, kind, level, lmax, script, extra, walltime):
    request_ncpus = LADDER[level]["request"]
    launch_ncpus = LADDER[level]["launch"]
    cdir = case_dir(results_dir, kind, level, lmax)
    cdir.mkdir(parents=True, exist_ok=True)
    resolved = cdir.resolve()
    if not str(resolved).startswith("/scratch") and not str(resolved).startswith("/g/data"):
        print(f"  WARNING: case dir {resolved} is not on Gadi (/scratch or "
              "/g/data); generate ON Gadi inside the deployment worktree, or "
              "these baked-in paths will be wrong when submitted.")
    # Short job name: older PBSPro caps -N at 15 chars; keep it compact and
    # unambiguous (coupled and capacitance both start with 'c').
    prefix = {"coupled": "sv", "capacitance": "cap", "anchor": "an"}[kind]
    jobname = f"g{prefix}{level}L{lmax}"   # e.g. gsv4L5, gcap4L30, gan5L5
    text = PBS_TEMPLATE.format(
        project=PROJECT, request_ncpus=request_ncpus, launch_ncpus=launch_ncpus,
        mem=mem_gb(request_ncpus), walltime=walltime, storage=STORAGE,
        jobname=jobname, module_use=MODULE_USE, module=MODULE,
        worktree=WORKTREE, casedir=str(resolved), script=script,
        level=level, lmax=lmax, extra=extra)
    path = cdir / "run.pbs"
    path.write_text(text)
    return path


def build_cases(levels, truncations):
    """Yield (kind, level, lmax, script, extra, walltime) for every case."""
    for level in levels:
        for lmax in COUPLED_L:
            if truncations and lmax not in truncations:
                continue
            yield ("coupled", level, lmax, "gravity_cubed_sphere.py", "",
                   LADDER[level]["coupled_wall"])
    for level in CAPACITANCE_LEVELS:
        if level not in levels:
            continue
        for lmax in CAPACITANCE_L:
            if truncations and lmax not in truncations:
                continue
            yield ("capacitance", level, lmax, "capacitance_gravity.py", "",
                   capacitance_wall(level, lmax))
    spot_level, spot_lmax = CAPACITANCE_SPOT
    if spot_level in levels and (not truncations or spot_lmax in truncations):
        yield ("capacitance", spot_level, spot_lmax, "capacitance_gravity.py",
               "", capacitance_wall(spot_level, spot_lmax))
    # Correctness anchor: direct vs iterative at levels 4 and 5, a coupled L.
    for level in (4, 5):
        if level in levels and (not truncations or 5 in truncations):
            yield ("anchor", level, 5, "gravity_cubed_sphere.py", "--anchor",
                   LADDER[level]["coupled_wall"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[4, 5, 6, 7], help="levels to include")
    parser.add_argument("--truncations", type=int, nargs="+", default=None,
                        help="restrict to these L values")
    parser.add_argument("--submit", action="store_true",
                        help="qsub each script (default: dry run, write only)")
    args = parser.parse_args()

    written = []
    for kind, level, lmax, script, extra, wall in build_cases(
            args.levels, args.truncations):
        path = write_script(args.results_dir, kind, level, lmax, script, extra,
                            wall)
        written.append((path, kind, level, lmax))
        print(f"{'submit' if args.submit else 'wrote':>7}: "
              f"{kind:>11} level {level} L {lmax:>2} -> {path}")
        if args.submit:
            subprocess.run(["qsub", str(path.name)], cwd=path.parent, check=True)

    print(f"\n{len(written)} scripts "
          f"{'submitted' if args.submit else 'written (dry run; --submit to qsub)'}.")


if __name__ == "__main__":
    main()
