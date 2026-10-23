#!/bin/bash -l
# GIAMIP Exp08: the input file of the bed inversion (prepare_exp08.py), on
# Setonix (Pawsey), project pawsey0821.
#
# The input file holds a mesh partitioned for the number of ranks of this
# job. Run the inversion on the same number of nodes and ranks.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/giamip/exp08/L0-2nodes/prepare
#   cd /scratch/pawsey0821/$USER/giamip/exp08/L0-2nodes/prepare
#   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp08/run_prepare.slurm
#
# For a test, add --partition=debug to sbatch. For another node count, add
# --nodes=<n> to sbatch, and use the same <n> for the inversion.
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   MESH    the coarse mesh (default: the shared mesh n2_giamip_all_o1)
#   REFINE  uniform refinement levels (default 0, L0)
#   EXTRA   further arguments of prepare_exp08.py
#
#SBATCH --job-name=exp08-prep
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=exp08-prep-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
# The environment of the inversion, so both jobs run the same code.
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
# g-drift (commit 9273a8c) and giamip at gia-mip 0cf5f39, which carries the
# model file prem-glad-m25.json, come first. The GLAD-M25 file is in the
# g-drift data folder, downloaded and checked on a login node.
export PYTHONPATH=/software/projects/pawsey0821/sghelichkhani/python-extra-3d:${PYTHONPATH}
export GDRIFT_DATA_DIR=/software/projects/pawsey0821/sghelichkhani/gdrift-data
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp08
MESH=${MESH:-${PERSIST}/meshes/n2_giamip_all_o1/n2_giamip_all_o1.msh}
OUT=${OUT:-${SLURM_SUBMIT_DIR}}

ARGS="--mesh ${MESH} --refine ${REFINE:-0} --output_path ${OUT} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "arguments: ${ARGS}"
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/prepare_exp08.py ${ARGS}
status=$?

# The kernel caches of this job, in /tmp of each node.
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
