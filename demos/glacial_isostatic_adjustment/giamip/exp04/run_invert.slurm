#!/bin/bash -l
# GIAMIP Exp04: the bed at the start of the run by adjoint (invert_exp04.py),
# on Setonix (Pawsey), project pawsey0821.
#
# The job must have the nodes and ranks of the prepare job that wrote the
# input file; the script stops with an error otherwise.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/giamip/exp04/L0-2nodes/taylor-3steps
#   cd /scratch/pawsey0821/$USER/giamip/exp04/L0-2nodes/taylor-3steps
#   sbatch --export=NONE,INPUTS=../prepare/inputs_exp04.h5,MODE=taylor,STEPS=3 \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp04/run_invert.slurm
#
# For a test, add --partition=debug to sbatch. A run that stopped at the time
# limit continues with RESTART=state.h5 (in the same run directory).
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   INPUTS          the input file of the prepare job (required)
#   MODE            taylor or invert (required)
#   STEPS           stop the march after this many steps (empty: to year 0)
#   ITERATIONS      ROL iterations (default 10)
#   RESTART         a state.h5 to continue from (empty: cold start)
#   CHECKPOINT_DIR  the tape checkpoints (default /tmp of each node, which is
#                   node memory on Setonix; use /scratch when memory is short)
#   EXTRA           further arguments of invert_exp04.py
#
#SBATCH --job-name=exp04-inv
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=24:00:00
#SBATCH --output=exp04-inv-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
# Firedrake 2026.10.0 with the PR 4638 overlay (cached replay solvers and
# the appctx fix) and the PyOP2 copy with the MixedDataSet cache-key fix
# (without it the memory grows at every replay and adjoint step).
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp04
OUT=${OUT:-${SLURM_SUBMIT_DIR}}
INPUTS=${INPUTS:?set INPUTS to the input file of the prepare job}
MODE=${MODE:?set MODE=taylor or MODE=invert}

ARGS="--inputs ${INPUTS} --output_path ${OUT} --mode ${MODE}"
ARGS="${ARGS} --iterations ${ITERATIONS:-10}"
if [ -n "${STEPS:-}" ]; then
    ARGS="${ARGS} --steps ${STEPS}"
fi
if [ -n "${RESTART:-}" ]; then
    ARGS="${ARGS} --restart_state ${RESTART}"
fi
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    ARGS="${ARGS} --checkpoint_dir ${CHECKPOINT_DIR}"
fi
ARGS="${ARGS} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "firedrake overlay ${FIREDRAKE_PR4638_OVERLAY}: $(head -1 ${FIREDRAKE_PR4638_OVERLAY}/../BRANCH_COMMIT)"
echo "pyop2 fix ${PYOP2_MIXED_DATASET_FIX}"
echo "arguments: ${ARGS}"
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/invert_exp04.py ${ARGS}
status=$?

# The tape checkpoints and the kernel caches of this job, in /tmp of each
# node, the tape checkpoints in CHECKPOINT_DIR, and the shared part of the
# disk checkpointer, which pyadjoint writes in the working directory.
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/giamip-adjoint-${SLURM_JOB_ID} /tmp/pyop2-${SLURM_JOB_ID} \
    /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    rm -rf ${CHECKPOINT_DIR}/giamip-adjoint-${SLURM_JOB_ID}
fi
rm -rf ${SLURM_SUBMIT_DIR}/firedrake_adjoint_checkpoint_* 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
