#!/bin/bash -l
# GIAMIP Exp01: a forward run from the bed of an inversion (forward_exp01.py),
# on Setonix (Pawsey), project pawsey0821.
#
# The job must have the nodes and ranks of the prepare job that wrote the
# input file (3 nodes, 384 ranks for L0-3nodes); the script stops with an
# error otherwise.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/forward-exactmap
#   cd /scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/forward-exactmap
#   sbatch --export=NONE,STATE=../invert-full-exactmap/state.h5 \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_forward.slurm
#
# A short test: add EXTRA="--steps 3 --plus_step always" and --time=00:20:00.
#
# The time limit is for L0: the full run took 41 min there (job 51182503),
# most of it in the first step (preconditioner build and kernel compilation)
# and in the solves. The solve time grows by about 4x per mesh level, so a
# finer level needs a larger --time at submit.
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   STATE   the state.h5 of the inversion (required)
#   EXTRA   further arguments of forward_exp01.py
#
#SBATCH --job-name=exp01-fwd
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=02:00:00
#SBATCH --output=exp01-fwd-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
# The environment of the inversion, so both jobs run the same code. The
# forward run also imports giamip (the ice of +1000 yr), as the prepare job
# does in the same environment.
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp01
OUT=${OUT:-${SLURM_SUBMIT_DIR}}
STATE=${STATE:?set STATE to the state.h5 of the inversion}

ARGS="--state ${STATE} --output_path ${OUT} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "arguments: ${ARGS}"
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/forward_exp01.py ${ARGS}
status=$?

# The kernel caches of this job, in /tmp of each node.
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
