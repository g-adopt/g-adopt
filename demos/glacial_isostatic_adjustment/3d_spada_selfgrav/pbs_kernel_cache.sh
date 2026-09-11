#!/bin/bash
# The kernel caches for a PBS job, with a persistent store behind them.
#
# WHAT THIS IS FOR
#     Firedrake compiles a kernel for every form it sees. On the coupled
#     self-gravitating solver that compile is the dominant cost of a short job:
#     measured 7 minutes at L = 5 and 23 minutes at L = 10
#     (NOTES/RUN-LOG-2026-08-12.md section 3). The caches used to live on
#     $PBS_JOBFS, which PBS deletes when the job ends, so every submission paid
#     that bill again. This file keeps a copy of the caches on /scratch and
#     stages it in and out.
#
# HOW TO USE IT
#     source pbs_kernel_cache.sh   # after `module load firedrake/...`
#     kernel_cache_begin
#     ... the mpiexec that does the work ...
#     kernel_cache_end
#
# THE THREE MODES
#     CACHE_MODE=warm   Default. Copy the store in before the run and merge new
#                       entries back after it. This is what removes the repeat
#                       compile.
#     CACHE_MODE=cold   Start from an empty cache, and merge back after. Use
#                       this to measure a cold compile time, which is what the
#                       pc-sweep and the compile-scaling runs measure.
#     CACHE_MODE=off    Start empty and write nothing back. This is the
#                       behaviour of the scripts before 2026-08-12, kept so an
#                       old measurement stays reproducible.
#
#     CACHE_ROOT names the store. The default is
#     /scratch/xd2/sg8812/firedrake-cache.
#
# WHY THE CACHES STILL RUN ON $PBS_JOBFS DURING THE JOB
#     The gravity scaling campaign recorded the reason and it is not a style
#     preference (NOTES/poisson/campaign2/results/**/run.pbs). PyOP2 compiles
#     on a node-local communicator, so a node-local cache cannot split a
#     collective. A cache directory on Lustre, read and written by every rank
#     while the job runs, gives cache divergence driven by metadata-visibility
#     skew, and that skew cuts across node boundaries, which is the shape that
#     hangs. loopy also keeps sqlite files under XDG_CACHE_HOME, and an sqlite
#     file on Lustre with 100 ranks retrying on it is the same problem again.
#
#     Staging keeps the in-job semantics identical to the configuration that has
#     already run, and moves the persistence outside the run. PYOP2_NODE_LOCAL_
#     COMPILATION must stay at its default of 1 for the same reason.
#
# WHAT CAN GO WRONG, AND WHAT HAPPENS THEN
#     /scratch on Gadi deletes files that nothing has read for 100 days. A
#     purged store makes the next job a cold one. That is the safe direction.
#     A stage-in or stage-out failure prints a warning and the job continues,
#     again as a cold run. Nothing here can fail the science.

: ${CACHE_ROOT:=/scratch/xd2/sg8812/firedrake-cache}
: ${CACHE_MODE:=warm}

# PyOP2's own hit/miss counters. This is the instrument that says whether the
# stage-in did anything: without it, "the run was faster" and "the cache was
# used" are separate claims and only the first is visible.
export PYOP2_CACHE_INFO=1

export PYOP2_CACHE_DIR=$PBS_JOBFS/pyop2
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$PBS_JOBFS/tsfc
export XDG_CACHE_HOME=$PBS_JOBFS/xdg-cache
export MPLCONFIGDIR=$PBS_JOBFS/matplotlib

# The three directories that carry compiled work. `matplotlib` is deliberately
# absent: it is a font list, it costs a second, and it is not worth a copy.
_CACHE_DIRS="pyop2 tsfc xdg-cache"

# Create them now, at source time, and not in `kernel_cache_begin`. Both PBS
# scripts run a `python3 -c 'import gadopt'` between the two, and matplotlib
# warns and falls back to the home directory when MPLCONFIGDIR does not exist.
for _d in $_CACHE_DIRS matplotlib; do
    mkdir -p "$PBS_JOBFS/$_d" 2>/dev/null || true
done
unset _d

_cache_nodes() {
    if [ -n "${PBS_NODEFILE:-}" ] && [ -r "${PBS_NODEFILE}" ]; then
        sort -u "$PBS_NODEFILE" | wc -l
    else
        echo 1
    fi
}

# Run one command once per node. A single-node job is the common case here
# (ncpus=104 is one normalsr node), but a two-node job must stage on both or the
# second node runs cold while the log says warm.
_cache_each_node() {
    local n
    n=$(_cache_nodes)
    if [ "$n" -le 1 ]; then
        bash -c "$1"
    else
        mpiexec -np "$n" --map-by ppr:1:node bash -c "$1" || return 1
    fi
}

_cache_count() {
    find "$1" -type f 2>/dev/null | wc -l | tr -d ' '
}

kernel_cache_begin() {
    for d in $_CACHE_DIRS matplotlib; do
        mkdir -p "$PBS_JOBFS/$d"
    done
    echo "=== kernel cache: mode=$CACHE_MODE root=$CACHE_ROOT ==="
    if [ "$CACHE_MODE" != "warm" ]; then
        echo "=== kernel cache: no stage-in; this run compiles cold ==="
        return 0
    fi
    mkdir -p "$CACHE_ROOT" || true
    for d in $_CACHE_DIRS; do
        mkdir -p "$CACHE_ROOT/$d" || true
    done
    local cmd=""
    for d in $_CACHE_DIRS; do
        cmd="$cmd rsync -a '$CACHE_ROOT/$d/' '$PBS_JOBFS/$d/';"
    done
    if _cache_each_node "$cmd"; then
        for d in $_CACHE_DIRS; do
            echo "=== kernel cache: staged in $d, $(_cache_count "$PBS_JOBFS/$d") files ==="
        done
    else
        echo "=== kernel cache: WARNING stage-in failed; this run compiles cold ==="
    fi
}

kernel_cache_end() {
    if [ "$CACHE_MODE" = "off" ]; then
        echo "=== kernel cache: mode=off, nothing written back ==="
        return 0
    fi
    for d in $_CACHE_DIRS; do
        echo "=== kernel cache: $d holds $(_cache_count "$PBS_JOBFS/$d") files after the run ==="
    done
    # `--ignore-existing` for the two content-addressed caches: two nodes that
    # generated the same kernel wrote the same bytes under the same name, so the
    # first writer wins and no reader sees a half-written file. rsync writes to a
    # temporary name and renames, so a concurrent reader never sees a partial one
    # either.
    local cmd=""
    for d in pyop2 tsfc; do
        cmd="$cmd rsync -a --ignore-existing '$PBS_JOBFS/$d/' '$CACHE_ROOT/$d/';"
    done
    _cache_each_node "$cmd" \
        || echo "=== kernel cache: WARNING stage-out of pyop2/tsfc failed ==="
    # loopy's sqlite files are NOT content addressed and must not be merged from
    # several nodes at once. Take the head node's copy only. Every node compiles
    # the same forms, so the head node's copy is representative; the cost of
    # being wrong about that is a partial cache, not a wrong answer.
    rsync -a "$PBS_JOBFS/xdg-cache/" "$CACHE_ROOT/xdg-cache/" \
        || echo "=== kernel cache: WARNING stage-out of xdg-cache failed ==="
    for d in $_CACHE_DIRS; do
        echo "=== kernel cache: store $d now holds $(_cache_count "$CACHE_ROOT/$d") files ==="
    done
}
