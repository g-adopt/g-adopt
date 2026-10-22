#!/bin/bash -l
# GIAMIP Exp01 with G-ADOPT on Setonix (Pawsey), project pawsey0821.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/exp01/runs/L0-1step
#   cd /scratch/pawsey0821/$USER/exp01/runs/L0-1step
#   sbatch --nodes=1 --time=02:00:00 \
#       --export=NONE,STEPS=1 $MY_GADOPT/tests/giamip/run_exp01.slurm
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   STEPS      the number of forcing steps of this job (empty: run to the end)
#   RESTART    a checkpoint file to continue from (same number of ranks), or
#              "auto": the newest checkpoint of the run directory if there is
#              one, else a new run
#   REFINE     uniform refinement levels of the coarse mesh (default 0, L0)
#   EXTRA      further arguments of exp01.py, for example "--k_s 0.9452"
#
# A long run is a chain of jobs with RESTART=auto, each submitted with
# --dependency=afterok on the one before. The driver stops before the time
# limit of the job with a checkpoint and exit code 0. A job of the chain that
# finds the run finished exits at once.
#
# The environment file loads firedrake/2026.10.0 alone (the g-adopt module
# would put its own gadopt before the worktree) and puts the worktree, the
# giamip and giamsh installations and the GIAMIP data on the paths.
#
#SBATCH --job-name=exp01
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=02:00:00
#SBATCH --output=exp01-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
source ${PERSIST}/env-exp01.sh
export OMP_NUM_THREADS=1
# The matplotlib font cache, built once on a login node with the same
# container (env-exp01.sh sets EXP01_MPLCONFIGDIR; the driver also applies
# it). Every rank only reads it, so no rank builds a cache or waits for a lock.
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

MESH=${MESH:-${PERSIST}/meshes/n2_giamip_exp01_o1/n2_giamip_exp01_o1.msh}
OUT=${OUT:-${SLURM_SUBMIT_DIR}}

# A chained job of a finished run has nothing to do.
if grep -qs '"state": "finished"' ${OUT}/summary_exp01.json; then
    echo "the run in ${OUT} is finished; nothing to do"
    exit 0
fi

# Stripe the run directory over 32 Lustre targets: a checkpoint at L2 is
# about 25 GB, written by every rank. New files take the setting.
lfs setstripe -c 32 -S 4M ${OUT} 2>/dev/null || true

# RESTART=auto: the newest checkpoint of this run, if there is one.
if [ "${RESTART:-}" = "auto" ]; then
    if [ -e ${OUT}/checkpoint_exp01_latest.h5 ]; then
        RESTART=${OUT}/checkpoint_exp01_latest.h5
    else
        RESTART=
    fi
fi

# The time left in this job, in hours, so that the driver stops in time.
# squeue gives [days-]hours:minutes:seconds, or minutes:seconds.
LEFT=$(squeue -h -j ${SLURM_JOB_ID} -o %L)
LEFT_H=$(echo ${LEFT} | awk -F'[-:]' '{
    if (NF == 4) s = $1 * 86400 + $2 * 3600 + $3 * 60 + $4;
    else if (NF == 3) s = $1 * 3600 + $2 * 60 + $3;
    else s = $1 * 60 + $2;
    printf "%.3f", s / 3600.0 }')

ARGS="--mesh ${MESH} --output_path ${OUT} --refine ${REFINE:-0}"
ARGS="${ARGS} --wall_limit_hours ${LEFT_H}"
[ -n "${STEPS:-}" ] && ARGS="${ARGS} --steps ${STEPS}"
[ -n "${RESTART:-}" ] && ARGS="${ARGS} --restart ${RESTART}"
ARGS="${ARGS} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID}: ${SLURM_NTASKS} ranks on ${SLURM_JOB_NUM_NODES} nodes"
echo "gadopt: ${MY_GADOPT} ($(git -C ${MY_GADOPT} rev-parse --short HEAD))"
echo "arguments: ${ARGS}"
module list 2>&1
date

srun --export=ALL -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_NTASKS} -c 1 \
    python3 ${MY_GADOPT}/tests/giamip/exp01.py ${ARGS}
status=$?
# Remove the kernel caches of this job from every node (env-exp01.sh puts
# them in /tmp per job; /tmp is node memory).
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
