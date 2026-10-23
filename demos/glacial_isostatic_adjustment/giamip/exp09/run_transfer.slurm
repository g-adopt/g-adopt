#!/bin/bash -l
# GIAMIP Exp09: move the inverted bed of one mesh level onto the input file
# of another level (transfer_bed_exp09.py), on Setonix (Pawsey), project
# pawsey0821.
#
# The job must have the nodes and ranks of the prepare job that wrote the
# target file; the script stops with an error otherwise. Set the node count
# at submit with --nodes. The target file is changed in place: the bed is
# appended as entry 0 of bed_start_parent_cg2_m. An inversion then starts
# from it with RESTART=<target> (run_invert.slurm, MODE=invert).
#
# Before the append, the job copies the target to <target>.bak in the same
# directory and removes the copy only after the script exits with 0 (which
# includes its reload check of the appended entry). If the job fails, the
# .bak file is the clean prepare output: move it back over the target before
# a new transfer. The job stops if a .bak file already exists, so a clean
# copy is never overwritten by a copy of a broken target.
#
# Submit from any directory. Exp09 starts warm from the L0 bed of Exp03
# (design handout P20, S20): same mesh, ice and topography, so START=bed.
# The Exp03 state file was written on 3 nodes; the target on 6 nodes.
# The source record says Exp03, so the transfer needs
# --allow_other_experiment; this is correct because the mesh (same MD5), the
# ice and the observed bed of Exp03 and Exp09 are the same:
#
#   cd /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/prepare
#   sbatch --export=NONE,START=bed,EXTRA=--allow_other_experiment,TARGET=$PWD/inputs_exp09.h5,SOURCE=/scratch/pawsey0821/$USER/giamip/exp03/L0-3nodes/invert-full/state.h5 \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_transfer.slurm
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   TARGET  the input file of prepare_exp09.py at the target level (required)
#   SOURCE  the state.h5 of the source inversion (required)
#   START   change (default) or bed: write the target observed bed plus
#           the change of the source inversion, or the source bed. Exp09
#           uses bed: the source is the Exp03 L0 inversion on the same mesh
#           with the same observed bed, so the source bed is the start bed
#   OUT     the directory of the log and the JSON record (default: the
#           directory of TARGET)
#   EXTRA   further arguments of transfer_bed_exp09.py (for example
#           --target_sha256 yes)
#
#SBATCH --job-name=exp09-xfer
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=exp09-xfer-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
# The environment of the inversion, so both jobs run the same code.
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp09
TARGET=${TARGET:?set TARGET to the input file of the target level}
SOURCE=${SOURCE:?set SOURCE to the state.h5 of the source inversion}

ARGS="--target ${TARGET} --source ${SOURCE} --start ${START:-change}"
if [ -n "${OUT:-}" ]; then
    ARGS="${ARGS} --output_path ${OUT}"
fi
ARGS="${ARGS} ${EXTRA:-}"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "arguments: ${ARGS}"
date

# The backup of the target. HDF5 has no journal, so a job that stops
# during the append (time limit, out of memory, a file system error) can
# leave a target that no longer loads. The copy is in the directory of the
# target, so it gets the Lustre striping of that directory (the prepare
# directory, striped before the prepare job). It costs the size of the
# target for the time of the job, about 100 GB at L2 (a few minutes for cp
# on /scratch).
BACKUP=${TARGET}.bak
if [ -e "${BACKUP}" ]; then
    echo "error: ${BACKUP} exists; it can be the clean copy of a failed" \
        "transfer. Move it back over the target or remove it first." >&2
    exit 1
fi
echo "copying the target to ${BACKUP}"
date
if ! cp -p "${TARGET}" "${BACKUP}"; then
    echo "error: the copy of the target failed; the target is unchanged" >&2
    rm -f "${BACKUP}"
    exit 1
fi
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/transfer_bed_exp09.py ${ARGS}
status=$?

# The script exits with 0 only after its reload check of the appended
# entry, so only then is the backup no longer needed. On any other exit the
# backup stays as the clean target.
if [ ${status} -eq 0 ]; then
    rm -f "${BACKUP}"
    echo "transfer done; removed ${BACKUP}"
else
    echo "transfer failed; the clean target is ${BACKUP}" >&2
fi

# The kernel caches of this job, in /tmp of each node.
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
