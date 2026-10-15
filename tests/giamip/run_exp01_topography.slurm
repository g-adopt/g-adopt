#!/bin/bash -l
# GIAMIP Exp01, the bed at 122 ka by adjoint (exp01_topography.py), on
# Setonix (Pawsey), project pawsey0821.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/exp01/topo-adjoint/taylor-L0-3steps
#   cd /scratch/pawsey0821/$USER/exp01/topo-adjoint/taylor-L0-3steps
#   sbatch --export=NONE,MODE=taylor,STEPS=3 \
#       $MY_GADOPT/tests/giamip/run_exp01_topography.slurm
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   MODE        taylor or invert
#   STEPS       stop the march after this many steps (empty: to year 0)
#   ITERATIONS  ROL iterations of the inversion (default 10)
#   REFINE      uniform refinement levels of the coarse mesh (default 0, L0)
#   EXTRA       further arguments of exp01_topography.py
#
# The environment is the Exp01 one with the Firedrake PR 4638 overlay first
# on PYTHONPATH (env-exp01-pr4638.sh). The tape checkpoints go to /tmp of
# each node (a RAM file system on Setonix) and are removed at the end.
#
#SBATCH --job-name=exp01-adj
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=exp01-adj-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
source ${PERSIST}/env-exp01-pr4638.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

MESH=${MESH:-${PERSIST}/meshes/n2_giamip_exp01_o1/n2_giamip_exp01_o1.msh}
OUT=${OUT:-${SLURM_SUBMIT_DIR}}
MODE=${MODE:?set MODE=taylor or MODE=invert}

ARGS="--mesh ${MESH} --refine ${REFINE:-0} --output_path ${OUT} --mode ${MODE}"
ARGS="${ARGS} --iterations ${ITERATIONS:-10}"
if [ -n "${STEPS:-}" ]; then
    ARGS="${ARGS} --steps ${STEPS}"
fi
ARGS="${ARGS} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "firedrake overlay ${FIREDRAKE_PR4638_OVERLAY}: $(head -1 ${FIREDRAKE_PR4638_OVERLAY}/../BRANCH_COMMIT)"
echo "arguments: ${ARGS}"
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 \
    ${MY_GADOPT}/tests/giamip/exp01_topography.py ${ARGS}
status=$?

# The tape checkpoints and the kernel caches of this job, in /tmp of each node.
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/exp01-adjoint-${SLURM_JOB_ID} /tmp/pyop2-${SLURM_JOB_ID} \
    /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
