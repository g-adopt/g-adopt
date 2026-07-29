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

Three cache directories move, not two. PyOP2 holds compiled objects and TSFC
holds compiled forms, which are the obvious ones; loopy also keeps its code
generation, scheduling and preprocessing results in sqlite files under
``XDG_CACHE_HOME``, and PyOP2 goes through loopy code generation on the assembly
path. Leaving that one shared would have let a "cold" run answer the codegen and
scheduling steps out of a warm sqlite file while reporting itself as cold.
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
NODE_MEM_GB = 480          # just under the 500 GB normalsr ceiling
NODE_JOBFS_GB = 50         # kernel caches only; the study writes no big files

# Level 7 spreads its 512 ranks over eight nodes rather than the five that would
# hold them, for two reasons that stage one measured.
#
# Memory. Peak per-rank memory tracks the multiplier count and barely moves with
# the level, because the bookkeeping for the scalar Real fields is replicated on
# every rank rather than distributed: 0.66 to 2.11 GB across L = 2 to 10 at
# level 4, and 0.82 to 2.31 GB at level 5. Extrapolated to L = 20 that is close
# to 7 GB per rank during the compile-heavy cold phase, so 102 ranks on a node
# would want around 700 GB against a 500 GB ceiling. At 64 per node it is about
# 430 GB, inside the 480 GB requested.
#
# Comparability. Level 6 puts 64 ranks on one node, so holding level 7 at the
# same 64 per node keeps memory bandwidth per rank the same between the two
# levels being compared. Five nodes would have given level 7 *more* bandwidth
# per rank than level 6 and quietly flattered the weak-scaling result.
LADDER = {
    4: {"launch": 1, "nodes": 1},
    5: {"launch": 8, "nodes": 1},
    6: {"launch": 64, "nodes": 1},
    7: {"launch": 512, "nodes": 8},
}

# The truncations for this campaign. It stops at 10 deliberately: every earlier
# number in this study was taken under a boundary quadrature default of
# `2 (max_mode + element_degree)`, which has since been replaced by a mesh-aware
# rule, and the degree that rule asks for differs from the old one by a factor of
# several at these truncations. Nothing measured before is comparable, so this
# campaign re-establishes the low-L base before spending anything on the corner.
# L = 15, 20 and 30 come back once these rows are read.
COUPLED_L = [2, 3, 5, 6, 7, 10]

# Both DtN representations, on every case. The comparison is the point of the
# campaign, so it is the grid rather than an option on it: a run that measured
# one path would have to be repeated in full to learn anything about the other,
# and the two paths must meet the same mesh, the same source and the same cache
# state for the difference between them to be a measurement.
REPRESENTATIONS = ["multiplier", "lowrank"]

# Walltime per case, covering *both* invocations plus their timed solves.
# Two costs drive it on the multiplier path and they scale differently: the
# O(L^4) boundary assembly per matrix-free application, worst at the coarsest
# level because per-rank facet counts halve as the level rises, and the
# per-process symbolic form processing, which is level-independent and paid
# redundantly by every rank.
#
# These are the old campaign's numbers, and they are kept as the multiplier
# path's allowance for one reason: they were measured under the *old* quadrature
# default, which asked for a higher degree than the mesh-aware rule now does, so
# as an upper bound they survive the change even though as an estimate they do
# not. Erring long here costs queue time in the worst case; erring short costs
# the run.
WALLTIME_BY_L = {
    2: "01:00:00",
    3: "01:00:00",
    5: "01:00:00",
    6: "01:00:00",
    7: "01:30:00",
    10: "02:00:00",
}

# The low-rank path's first solve is independent of the mode count and its steady
# solve is one multigrid solve, so its walltime is not expected to move with L at
# all. One hour everywhere is already generous by a wide margin - but it is a
# guess, not a measurement, because nothing on this path has run past four ranks,
# and the first campaign is what turns it into a measurement.
LOWRANK_WALLTIME = "01:00:00"


# Level 7 gets half again as long. Not because the arithmetic per core grows -
# the ladder holds that fixed - but because communication does. At level 6 on 64
# ranks, SFReduce and SFBcast inside the assembly parloops already account for
# roughly half the solve time, against 1.5% at level 5 and nothing at level 4,
# and the Real-field global reductions add a further term that grows with the
# truncation. Level 7 runs eight times as many ranks over eight nodes, so its
# solves are expected to be slower than the weak-scaling ideal, and a walltime
# set from the lower levels would be set from the wrong regime.
WALLTIME_MULTIPLIER = {7: 1.5}


def walltime(level, lmax, representation="multiplier"):
    """Walltime for one case, both phases included."""
    base = (LOWRANK_WALLTIME if representation == "lowrank"
            else WALLTIME_BY_L[lmax])
    factor = WALLTIME_MULTIPLIER.get(level, 1.0)
    if factor == 1.0:
        return base
    hours, minutes, seconds = (int(p) for p in base.split(":"))
    total = int((hours * 3600 + minutes * 60 + seconds) * factor)
    return "%02d:%02d:%02d" % (total // 3600, (total % 3600) // 60, total % 60)


def resources(level):
    """PBS ncpus, mem and jobfs for a level, as whole exclusive nodes."""
    nodes = LADDER[level]["nodes"]
    launch = LADDER[level]["launch"]
    # Ranks per node, pinned explicitly rather than left to the default mapping:
    # per-rank memory bandwidth depends on how many ranks share a node, so a
    # study comparing wall times across levels cannot let it vary silently.
    per_node = -(-launch // nodes)          # ceiling division
    return {
        "request_ncpus": nodes * NODE_CORES,
        "launch_ncpus": launch,
        "mem": nodes * NODE_MEM_GB,
        "jobfs": nodes * NODE_JOBFS_GB,
        "nodes": nodes,
        "per_node": per_node,
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

# Form processing at high truncation recurses to a depth that grows with the
# number of mixed sub-fields, so the model lifts Python's recursion limit. Lift
# the stack limit to match: on Python 3.11 pure-Python frames do not consume C
# stack, but the traversals pass through C dispatch, and a job that dies of stack
# exhaustion an hour into compilation is an expensive way to learn that.
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true

export PYTHONPATH={worktree}:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
# One BLAS thread per rank; threaded BLAS underneath MPI would thrash the node
# and make the wall times meaningless.
export OMP_NUM_THREADS=1
export OMPI_MCA_io=ompio
export MPLCONFIGDIR=$PBS_JOBFS/matplotlib
# Turn on PyOP2's cache hit/miss counters so each sidecar records what the
# process actually generated, rather than us inferring it from the filesystem.
export PYOP2_CACHE_INFO=1

# Node-local kernel caches: private per node, so nodes cannot race, and empty at
# job start, so the first invocation below is genuinely cold. Setting these is
# belt and braces - the fp50 wrapper already redirects the cache directories,
# XDG_CACHE_HOME included, into $PBS_JOBFS inside a job, which was verified on
# this module rather than assumed - but naming them explicitly means the phases
# do not depend on that wrapper behaviour staying as it is. PyOP2 and Firedrake
# create these directories themselves on whichever node needs them, so there is
# nothing to pre-create.
#
# Per-node rather than shared is also the safer choice, not merely the
# convenient one. Divergent cache state is only dangerous where it splits a
# communicator that a collective runs over, and PyOP2's compile communicator is
# node-local, so divergence aligned with node boundaries cannot split it. A
# shared cache on a network filesystem instead gives divergence driven by
# metadata-visibility skew, which cuts across nodes, and that is the shape that
# hangs. `PYOP2_NODE_LOCAL_COMPILATION` must therefore stay at its default of 1:
# switching it off makes every rank copy the compiled object out of rank 0's
# temporary directory, which does not exist on the other nodes.
export PYOP2_CACHE_DIR=$PBS_JOBFS/pyop2
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$PBS_JOBFS/tsfc
# The third cache, and the one it would be easy to miss: loopy keeps its
# code-generation, scheduling and preprocessing results in sqlite files under
# XDG_CACHE_HOME. Leaving that on shared storage would make the cold phase pay
# neither loopy codegen nor scheduling, understating it by however much that
# costs, while still calling itself cold. It also puts a busy-retrying sqlite
# file on a network filesystem with 512 ranks reaching for it.
export XDG_CACHE_HOME=$PBS_JOBFS/xdg-cache
# TMPDIR is deliberately not set here. Python silently ignores a TMPDIR that
# does not exist and falls back to /tmp, so pointing it at a subdirectory
# created only on the head node would send compile scratch to /tmp on every
# other node without any error. PBS already provides $PBS_JOBFS on each node.

cd {casedir}

for PHASE in {phases}; do
    export PETSC_OPTIONS="-log_view :{casedir}/profile_$PHASE.txt -options_left"
    echo "=== phase $PHASE begins $(date) ==="
    mpiexec -np {launch_ncpus} --map-by ppr:{per_node}:node \\
        python3 {worktree}/tests/parallel_scaling_gravity/{script} \\
        {level} --lmax {lmax} --cache-phase $PHASE {extra} \\
        --out-dir {casedir} > {casedir}/run_$PHASE.out 2> {casedir}/run_$PHASE.err
    echo "=== phase $PHASE exit $? at $(date) ==="
done
# The independent check on the cache counters - how many files are actually
# sitting in each cache directory - is taken inside the model, per rank and
# maximised, and lands in the sidecar as cache_file_counts. Doing it here with
# pbsdsh looked cheaper and produced an empty file with no error, which is the
# characteristic failure of this whole area: an instrument that reports nothing
# while exiting zero.
"""


def case_dir(results_dir, kind, level, lmax, representation=None):
    """Directory for one case.

    The representation is a directory level, not a filename suffix, because each
    case writes a profile file and two run logs under names that carry no
    representation of their own; putting the two paths in one directory would
    have the second overwrite the first's `-log_view` output, which is the
    source of truth for everything the sidecars summarise.
    """
    base = Path(results_dir) / kind / f"level_{level}" / f"lmax_{lmax}"
    return base / representation if representation else base


def write_script(results_dir, kind, level, lmax, script, extra, wall,
                 representation=None):
    res = resources(level)
    cdir = case_dir(results_dir, kind, level, lmax, representation)
    cdir.mkdir(parents=True, exist_ok=True)
    resolved = cdir.resolve()
    if not str(resolved).startswith(("/scratch", "/g/data")):
        print(f"  WARNING: case dir {resolved} is not on Gadi (/scratch or "
              "/g/data); generate ON Gadi inside the deployment worktree, or "
              "these baked-in paths will be wrong when submitted.")
    # Short job name: older PBSPro caps -N at 15 characters.
    prefix = {"coupled": "sv", "capacitance": "cap", "anchor": "an",
              "parity": "par"}[kind]
    suffix = {"multiplier": "m", "lowrank": "r"}.get(representation or "", "")
    jobname = f"g{prefix}{level}L{lmax}{suffix}"
    # The cold/warm split measures what kernel caching removes, which is a cost
    # question. The parity gate asks whether two representations agree, and the
    # answer cannot depend on the cache, so running it twice would buy nothing
    # and would have the second invocation overwrite the first's sidecar.
    phases = "cold" if kind == "parity" else "cold warm"
    text = PBS_TEMPLATE.format(
        project=PROJECT, queue=QUEUE, walltime=wall, storage=STORAGE,
        jobname=jobname, module_use=MODULE_USE, module=MODULE,
        worktree=WORKTREE, casedir=str(resolved), script=script,
        level=level, lmax=lmax, extra=extra, phases=phases, **res)
    path = cdir / "run.pbs"
    path.write_text(text)
    return path


def build_cases(levels, truncations, capacitance, anchor, parity=True,
                representations=None):
    """Yield (kind, level, lmax, script, extra, walltime, representation)."""
    representations = representations or REPRESENTATIONS
    # The parity gate first, and deliberately at the head of the list: it is
    # what makes every wall time below a comparison rather than two unrelated
    # numbers, and the low-rank path has never run past four ranks, so the one
    # thing capable of invalidating the whole campaign is its behaviour under
    # distribution. Cheap - one solve each way - and it fails loudly.
    if parity:
        for level in levels:
            yield ("parity", level, 5, "gravity_cubed_sphere.py", "--parity",
                   "00:30:00", None)
    for level in levels:
        for lmax in COUPLED_L:
            if truncations and lmax not in truncations:
                continue
            for representation in representations:
                yield ("coupled", level, lmax, "gravity_cubed_sphere.py",
                       f"--representation {representation}",
                       walltime(level, lmax, representation), representation)
    # The direct-preset anchor. The closed-form error now covers every case, so
    # this is corroboration rather than the gate it once was: it says the
    # iterative preset agrees with a direct factorisation, where the analytic
    # error says both agree with the truth. Only where MUMPS is affordable.
    if anchor:
        for level in (4, 5):
            if level in levels and (not truncations or 5 in truncations):
                yield ("anchor", level, 5, "gravity_cubed_sphere.py",
                       "--anchor", walltime(level, 5), None)
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
                           "", "02:00:00", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    # Stage one is levels 4 to 6, all of which fit on a single node. Level 7 is
    # eight nodes and is held back on purpose: it is the rung where communication
    # rather than arithmetic sets the time, so its walltime wants to be set from
    # the level-6 measurement rather than from a guess, and none of the numbers
    # this campaign re-establishes exist yet at any level.
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[4, 5, 6], help="levels to include")
    parser.add_argument("--truncations", type=int, nargs="+", default=None,
                        help="restrict to these L values")
    parser.add_argument("--representations", nargs="+", default=None,
                        choices=REPRESENTATIONS,
                        help="restrict to these DtN representations")
    parser.add_argument("--no-parity", dest="parity", action="store_false",
                        help="skip the multiplier-vs-lowrank parity gate")
    parser.add_argument("--anchor", action="store_true",
                        help="also generate the direct-vs-iterative anchor cases")
    parser.add_argument("--capacitance", action="store_true",
                        help="also generate the wall-free capacitance cases")
    parser.add_argument("--submit", action="store_true",
                        help="qsub each script (default: dry run, write only)")
    args = parser.parse_args()

    written, total_su = [], 0.0
    for kind, level, lmax, script, extra, wall, representation in build_cases(
            args.levels, args.truncations, args.capacitance, args.anchor,
            parity=args.parity, representations=args.representations):
        path = write_script(args.results_dir, kind, level, lmax, script, extra,
                            wall, representation)
        res = resources(level)
        hours = sum(float(part) / 60**i
                    for i, part in enumerate(wall.split(":")))
        total_su += res["request_ncpus"] * hours * 2.0
        written.append((path, kind, level, lmax))
        print(f"{'submit' if args.submit else 'wrote':>7}: {kind:>8} "
              f"{representation or '-':>10} level {level} L {lmax:>2} "
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
