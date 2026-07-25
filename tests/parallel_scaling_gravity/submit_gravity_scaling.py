r"""Generate (and optionally submit) the Gadi PBS scripts for the scaling study.

Writes one self-contained PBS script per case into
``results/coupled/level_{level}/lmax_{L}/`` in the ``morrow2026`` bare-submission
style. Each script runs the same case **twice**, cold then warm, which is what
separates kernel-generation cost from solve cost; see "the two phases" below.

Dry run (default) writes the scripts for inspection; ``--submit`` calls ``qsub``.

    <python> submit_gravity_scaling.py                    # dry run, all cases
    <python> submit_gravity_scaling.py --levels 4 5       # stage one
    <python> submit_gravity_scaling.py --submit --levels 4 5

The study is deliberately staged: submit levels 4 and 5, read the measured cold
and warm costs out of the sidecars, and only then size and submit levels 6 and
7. The cold cost is dominated by per-process symbolic form processing that
scales with the mode count and is unmeasured above a few hundred modes, so
walltimes for the expensive corner are set from measurement rather than guessed.

**The queue.** Everything runs on ``normalsr`` (Sapphire Rapids, 104 cores per
node, 2 SU per core-hour). Every level requests **whole nodes** even where it
launches far fewer ranks: a sub-node request shares its node with other jobs,
and a study whose headline number is a wall time cannot afford a co-tenant
competing for memory bandwidth. The idle cores are the price of a clean
measurement, and at these node counts they cost a few hundred SU.

**The ladder.** Launch counts are 1, 8, 64, 512 for levels 4 to 7, one factor
of eight per level, holding the potential block near thirteen thousand degrees
of freedom per core. That is the ladder the recorded level-4 numbers were taken
on, so the weak-scaling axis stays comparable to what is already in
``expected.csv``.

**The two phases.** Kernel caches are pointed at ``$PBS_JOBFS``, which is
node-local and wiped when the job ends. Three things follow, all of them
wanted. Every job's first invocation is genuinely cold, because no earlier job
can have left anything behind. The second invocation is warm, because it is the
same nodes and the same directories. And nodes cannot race each other to write
the same file, because each writes its own private filesystem - the failure the
shared-cache version of this script had to avoid by warming level 4 first.
``XDG_CACHE_HOME`` is deliberately left on the shared filesystem: the wipe
covers the PyOP2 and TSFC caches, which are where kernels and compiled forms
live.
"""

import argparse
import subprocess
from pathlib import Path

# --- Gadi deployment configuration -----------------------------------------
PROJECT = "xd2"
QUEUE = "normalsr"
STORAGE = "scratch/xd2+gdata/fp50"
MODULE_USE = "/g/data/fp50/modules"
MODULE = "firedrake/main-20260716"
WORKTREE = "/scratch/xd2/sg8812/g-adopt-worktrees/sghelichkhani/poisson"

# normalsr: 104 cores and 500 GB per node; PBS wants whole multiples of 104 for
# anything above one node, and we ask for whole nodes everywhere regardless so
# that no other job shares the hardware being timed.
NODE_CORES = 104
NODE_MEM_GB = 400          # under the 500 GB ceiling, far above what we need
NODE_JOBFS_GB = 50         # kernel caches only; the study writes no big files

LADDER = {
    4: {"launch": 1, "nodes": 1},
    5: {"launch": 8, "nodes": 1},
    6: {"launch": 64, "nodes": 1},
    7: {"launch": 512, "nodes": 5},
}

# The truncations: the set the level-4 record already covers, plus the three
# this study adds. L = 15 and 20 are past anything the coupled solve has ever
# run at, in three dimensions or two.
COUPLED_L = [2, 3, 5, 6, 7, 10, 15, 20]

# Walltime per case, covering *both* invocations plus their timed solves.
# Two costs drive it and they scale differently: the O(L^4) boundary assembly
# per matrix-free application, worst at the coarsest level because per-rank
# facet counts halve as the level rises, and the per-process symbolic form
# processing, which is level-independent and paid redundantly by every rank.
# These are stage-one estimates from a single-rank level-4 calibration; the
# level 6 and 7 rows are revised from stage-one measurements before submission.
WALLTIME_BY_L = {
    2: "01:00:00",
    3: "01:00:00",
    5: "01:00:00",
    6: "01:00:00",
    7: "01:30:00",
    10: "02:00:00",
    15: "03:00:00",
    20: "05:00:00",
}


def walltime(level, lmax):
    """Walltime for one case, both phases included."""
    return WALLTIME_BY_L[lmax]


def resources(level):
    """PBS ncpus, mem and jobfs for a level, as whole exclusive nodes."""
    nodes = LADDER[level]["nodes"]
    return {
        "request_ncpus": nodes * NODE_CORES,
        "launch_ncpus": LADDER[level]["launch"],
        "mem": nodes * NODE_MEM_GB,
        "jobfs": nodes * NODE_JOBFS_GB,
        "nodes": nodes,
    }


PBS_TEMPLATE = """#!/bin/bash
#PBS -P {project}
#PBS -q {queue}
#PBS -l ncpus={request_ncpus}
#PBS -l mem={mem}GB
#PBS -l jobfs={jobfs}GB
#PBS -l walltime={walltime}
#PBS -l storage={storage}
#PBS -l wd
#PBS -N {jobname}
#PBS -o pbs_{jobname}.out
#PBS -e pbs_{jobname}.err

# One (level, L) case of the gravitational Poisson DtN scaling study, run twice:
# once against an empty kernel cache and once against the cache the first run
# left behind. See submit_gravity_scaling.py for why the caches live on jobfs
# and why the node request exceeds the rank count.

source /etc/profile
module use {module_use}
module load {module}

export PYTHONPATH={worktree}:$PYTHONPATH
# One BLAS thread per rank; threaded BLAS underneath MPI would thrash the node
# and make the wall times meaningless.
export OMP_NUM_THREADS=1
# Turn on PyOP2's cache hit/miss counters so each sidecar records what the
# process actually generated, rather than us inferring it from the filesystem.
export PYOP2_CACHE_INFO=1

# Node-local kernel caches: private per node, so nodes cannot race, and empty at
# job start, so the first invocation is genuinely cold. The module would
# otherwise point these at shared scratch, where a previous job's kernels would
# make every run warm.
export PYOP2_CACHE_DIR=$PBS_JOBFS/pyop2
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$PBS_JOBFS/tsfc
# The PBS script itself runs only on the head node, so create the directories on
# every node. One process per node, hence --map-by ppr:1:node.
mpiexec -n {nodes} --map-by ppr:1:node \\
    bash -c 'mkdir -p $PBS_JOBFS/pyop2 $PBS_JOBFS/tsfc'

cd {casedir}

for PHASE in cold warm; do
    export PETSC_OPTIONS="-log_view :{casedir}/profile_$PHASE.txt -options_left"
    echo "=== phase $PHASE: $(date) ==="
    mpiexec -np {launch_ncpus} python3 {worktree}/tests/parallel_scaling_gravity/{script} \\
        {level} --lmax {lmax} --cache-phase $PHASE {extra} \\
        --out-dir {casedir} > {casedir}/run_$PHASE.out 2> {casedir}/run_$PHASE.err
    echo "=== phase $PHASE exit $?: $(date) ==="
    # What the cold phase left on each node, as an independent check on the
    # cache counters in the sidecar.
    mpiexec -n {nodes} --map-by ppr:1:node \\
        bash -c 'echo "$(hostname) $PHASE cache files: $(find $PBS_JOBFS/pyop2 \
$PBS_JOBFS/tsfc -type f 2>/dev/null | wc -l)"' \\
        >> {casedir}/cache_files.txt
done
"""


def case_dir(results_dir, kind, level, lmax):
    return Path(results_dir) / kind / f"level_{level}" / f"lmax_{lmax}"


def write_script(results_dir, kind, level, lmax, script, extra, wall):
    res = resources(level)
    cdir = case_dir(results_dir, kind, level, lmax)
    cdir.mkdir(parents=True, exist_ok=True)
    resolved = cdir.resolve()
    if not str(resolved).startswith(("/scratch", "/g/data")):
        print(f"  WARNING: case dir {resolved} is not on Gadi (/scratch or "
              "/g/data); generate ON Gadi inside the deployment worktree, or "
              "these baked-in paths will be wrong when submitted.")
    # Short job name: older PBSPro caps -N at 15 characters.
    prefix = {"coupled": "sv", "capacitance": "cap", "anchor": "an"}[kind]
    jobname = f"g{prefix}{level}L{lmax}"
    text = PBS_TEMPLATE.format(
        project=PROJECT, queue=QUEUE, walltime=wall, storage=STORAGE,
        jobname=jobname, module_use=MODULE_USE, module=MODULE,
        worktree=WORKTREE, casedir=str(resolved), script=script,
        level=level, lmax=lmax, extra=extra, **res)
    path = cdir / "run.pbs"
    path.write_text(text)
    return path


def build_cases(levels, truncations, capacitance, anchor):
    """Yield (kind, level, lmax, script, extra, walltime) for every case."""
    for level in levels:
        for lmax in COUPLED_L:
            if truncations and lmax not in truncations:
                continue
            yield ("coupled", level, lmax, "gravity_cubed_sphere.py", "",
                   walltime(level, lmax))
    # The direct-preset anchor. The closed-form error now covers every case, so
    # this is corroboration rather than the gate it once was: it says the
    # iterative preset agrees with a direct factorisation, where the analytic
    # error says both agree with the truth. Only where MUMPS is affordable.
    if anchor:
        for level in (4, 5):
            if level in levels and (not truncations or 5 in truncations):
                yield ("anchor", level, 5, "gravity_cubed_sphere.py",
                       "--anchor", walltime(level, 5))
    # The wall-free capacitance diagnostic settled the L-cost question before
    # the 128-field wall was lifted and is not part of this study; kept behind a
    # flag so the rows can be regenerated without resurrecting the script.
    if capacitance:
        for level in (4, 5):
            if level in levels:
                for lmax in (2, 5, 10, 20, 30):
                    if truncations and lmax not in truncations:
                        continue
                    yield ("capacitance", level, lmax, "capacitance_gravity.py",
                           "", "02:00:00")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[4, 5, 6, 7], help="levels to include")
    parser.add_argument("--truncations", type=int, nargs="+", default=None,
                        help="restrict to these L values")
    parser.add_argument("--anchor", action="store_true",
                        help="also generate the direct-vs-iterative anchor cases")
    parser.add_argument("--capacitance", action="store_true",
                        help="also generate the wall-free capacitance cases")
    parser.add_argument("--submit", action="store_true",
                        help="qsub each script (default: dry run, write only)")
    args = parser.parse_args()

    written, total_su = [], 0.0
    for kind, level, lmax, script, extra, wall in build_cases(
            args.levels, args.truncations, args.capacitance, args.anchor):
        path = write_script(args.results_dir, kind, level, lmax, script, extra,
                            wall)
        res = resources(level)
        hours = sum(float(part) / 60**i
                    for i, part in enumerate(wall.split(":")))
        total_su += res["request_ncpus"] * hours * 2.0
        written.append((path, kind, level, lmax))
        print(f"{'submit' if args.submit else 'wrote':>7}: {kind:>11} "
              f"level {level} L {lmax:>2} "
              f"({res['launch_ncpus']:>3} ranks on {res['nodes']} node(s), "
              f"{wall}) -> {path}")
        if args.submit:
            subprocess.run(["qsub", str(path.name)], cwd=path.parent, check=True)

    print(f"\n{len(written)} scripts "
          f"{'submitted' if args.submit else 'written (dry run; --submit to qsub)'}."
          f"\nWorst-case charge if every job runs its full walltime: "
          f"{total_su:,.0f} SU.")


if __name__ == "__main__":
    main()
