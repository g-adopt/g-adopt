#!/bin/bash -l
# GIAMIP Exp09: the forward run from the bed of an inversion at the fine
# steps (forward_exp09.py --schedule fine or fine_b), on Setonix (Pawsey),
# project pawsey0821, with checkpoints and resubmission at the wall limit.
#
# Schedule A (fine): 50 yr steps to year -26000, then 10 yr steps to year
# +1000: 4620 steps from 122 ka, 3780 from 80 ka. Schedule B (fine_b):
# 100 yr, then 25 yr, and 10 yr in the fast melt windows of the ice history.
# The ice is linear in time between the protocol years; the surface fields
# and the parameter log are written at the protocol years only.
#
# The job must have the nodes and ranks of the prepare job that wrote the
# input file (6 nodes, 768 ranks for the Exp09, Exp10 and Exp12 runs; 3
# nodes for the L0-3nodes files of the other experiments: pass --nodes=3 to
# sbatch). The script stops with an error otherwise.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/forward-fine
#   cd /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/forward-fine
#   sbatch --export=NONE,STATE=../invert-full/state.h5 \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_forward_fine.slurm
#
# THE RESTART. The run copies STATE into the run directory
# (forward_state.h5), loads its meshes and fields from that copy and writes
# its checkpoints into it. It writes a checkpoint every CHECKPOINT_EVERY
# steps (at the next protocol year) and, with WALL_LIMIT_HOURS, a last one
# before that wall time, then exits and leaves the file RESTART_NEEDED. This
# script then submits itself again with RESTART=1 and the same variables and
# node count; the next leg continues from the last checkpoint and appends to
# the surface file and the parameter log. WALL_LIMIT_HOURS defaults to the
# time limit of the job minus 1 h (23 h for the default 24 h). The next leg
# gets the node count and the time limit of the current one; other sbatch
# options of the first submission (partition, qos) are not carried. To
# continue a run by hand after a crash, submit with RESTART=1 in the same
# directory: it continues from the last periodic checkpoint.
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   STATE             the state.h5 of the inversion (required on the first
#                     leg; a restart reads forward_state.h5)
#   SCHEDULE          fine (default), fine_b, or windows (schedule C:
#                     the protocol steps plus 25 yr and 10 yr steps in the
#                     fast melt windows; 732 steps from 122 ka for
#                     ICE-6G_D, 762 for GLAC3c profile 1, 792 for
#                     profile 2; with a schedule-C state file the ice to
#                     year 0 comes from the file, as in the inversion)
#   WALL_LIMIT_HOURS  stop with a checkpoint before this wall time (default:
#                     the time limit of the job minus 1 h)
#   CHECKPOINT_EVERY  steps between periodic checkpoints (default of the
#                     script: 200)
#   RESTART           1 to continue from the last checkpoint
#   RESUBMIT          0 to stop after this leg even if RESTART_NEEDED exists
#   EXTRA             further arguments of forward_exp09.py
#   SURFACE_LAYOUT    the lfs setstripe options of a new surface file
#                     (default "-c 1 -p flash", see run_forward_protocol.slurm)
#
#SBATCH --job-name=exp09-fwdf
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=24:00:00
#SBATCH --output=exp09-fwdf-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp09
OUT=${OUT:-${SLURM_SUBMIT_DIR}}
SCHEDULE=${SCHEDULE:-fine}
# The time limit of this job (D-HH:MM:SS, HH:MM:SS or MM:SS), from Slurm,
# with the #SBATCH value as the fallback when the controller does not
# answer. The default wall limit of the run is this limit minus 1 h, and the
# next leg gets the same limit.
LIMIT=$(squeue -h -j ${SLURM_JOB_ID} -o %l 2>/dev/null)
LIMIT=${LIMIT:-24:00:00}
limit_hours() {
    local t=$1 d=0 h=0 m=0 sec=0
    if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
    IFS=: read -r a b c <<< "$t"
    if [ -n "$c" ]; then h=$a; m=$b; sec=$c; elif [ -n "$b" ]; then m=$a; sec=$b; else m=$a; fi
    echo "scale=2; $d * 24 + $((10#$h)) + $((10#$m)) / 60 + $((10#$sec)) / 3600" | bc
}
DEFAULT_WALL=$(echo "$(limit_hours ${LIMIT}) - 1" | bc)
WALL_LIMIT_HOURS=${WALL_LIMIT_HOURS:-${DEFAULT_WALL}}
echo "job time limit ${LIMIT}, wall limit of the run ${WALL_LIMIT_HOURS} h"
# A wall limit of 0 or less would stop every leg after its first step and
# resubmit it without end (a job of 1 h or less with the default).
if [ "$(echo "${WALL_LIMIT_HOURS} > 0" | bc)" != "1" ]; then
    echo "ERROR: the wall limit ${WALL_LIMIT_HOURS} h is not positive; set WALL_LIMIT_HOURS or a longer --time"
    exit 1
fi
RESTART=${RESTART:-0}
RESUBMIT=${RESUBMIT:-1}

ARGS="--output_path ${OUT} --schedule ${SCHEDULE}"
ARGS="${ARGS} --wall_limit_hours ${WALL_LIMIT_HOURS}"
if [ -n "${CHECKPOINT_EVERY:-}" ]; then
    ARGS="${ARGS} --checkpoint_every ${CHECKPOINT_EVERY}"
fi
SURFACE=${OUT}/forward_exp09_surface.h5
mkdir -p ${OUT}
if [ "${RESTART}" = "1" ]; then
    # The state argument is required by the script; a restart reads the
    # run's own copy (forward_state.h5) and ignores it for the meshes.
    ARGS="${ARGS} --state ${STATE:-${OUT}/forward_state.h5} --restart"
else
    STATE=${STATE:?set STATE to the state.h5 of the inversion}
    ARGS="${ARGS} --state ${STATE}"
    # A new surface file on one flash stripe (see run_forward_protocol.slurm
    # for the measurements behind this layout).
    SURFACE_LAYOUT=${SURFACE_LAYOUT:-"-c 1 -p flash"}
    rm -f ${SURFACE} ${OUT}/RESTART_NEEDED
    lfs setstripe ${SURFACE_LAYOUT} ${SURFACE} || { echo "lfs setstripe failed"; exit 1; }
fi
ARGS="${ARGS} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "arguments: ${ARGS}"
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/forward_exp09.py ${ARGS}
status=$?

srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"

# The next leg: the run stopped at the wall limit with a checkpoint.
if [ ${status} -eq 0 ] && [ -f ${OUT}/RESTART_NEEDED ] && [ "${RESUBMIT}" = "1" ]; then
    echo "RESTART_NEEDED: $(cat ${OUT}/RESTART_NEEDED); submitting the next leg"
    cd ${OUT}
    # The next leg gets the node count, the rank count and the time limit
    # of this one, and the variables of this script. The rank count must be
    # passed: a leg on 12 nodes of 64 ranks (--nodes=12 --ntasks=768) would
    # otherwise get the 128 ranks per node of the #SBATCH line, and the
    # input file loads only on the rank count it was written for. Other
    # sbatch options given on the first sbatch line (partition, qos,
    # reservation) are not carried over.
    if ! sbatch --nodes=${SLURM_JOB_NUM_NODES} --ntasks=${SLURM_NTASKS} --time=${LIMIT} \
        --export=NONE,RESTART=1,SCHEDULE=${SCHEDULE},WALL_LIMIT_HOURS=${WALL_LIMIT_HOURS},CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-},EXTRA="${EXTRA:-}",OUT=${OUT} \
        ${HERE}/run_forward_fine.slurm; then
        echo "ERROR: the submission of the next leg failed; submit it by hand with RESTART=1"
        exit 1
    fi
fi
exit ${status}
