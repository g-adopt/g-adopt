#!/bin/bash -l
# GIAMIP Exp12: move an inverted bed onto the Exp12 input file
# (transfer_bed_exp12.py), on Setonix (Pawsey), project pawsey0821. Exp12
# starts warm from the Exp01 L0 bed with START=change (design handout of
# 2026-10-20, P20): the Exp12 observed bed (Bedmap3) plus the change that the
# Exp01 L0 inversion found.
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
# Submit from any directory, for example for the Exp12 L0 input file made on
# 6 nodes and the L0 bed of Exp01:
#
#   cd /scratch/pawsey0821/$USER/giamip/exp12/L0-6nodes/prepare
#   sbatch --nodes=6 --export=NONE,TARGET=$PWD/inputs_exp12.h5,SOURCE=/scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/invert-full-exactmap/state.h5 \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_transfer.slurm
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   TARGET  the input file of prepare_exp12.py at the target level (required)
#   SOURCE  the state.h5 of the source inversion (required)
#   START   change (default) or bed: write the target observed bed plus
#           the change of the source inversion, or the source bed. Exp12
#           uses change: its observed bed (Bedmap3) differs from the Exp01
#           observed bed, and change starts the inversion from its own
#           observed bed plus the correction of the Exp01 L0 inversion
#   OUT     the directory of the log and the JSON record (default: the
#           directory of TARGET)
#   EXTRA   further arguments of transfer_bed_exp12.py (for example
#           --target_sha256 yes). Exp12 needs --allow_other_experiment, because the source is Exp01
#
#SBATCH --job-name=exp12-xfer
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=exp12-xfer-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
# The environment of the inversion, so both jobs run the same code.
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp12
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

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/transfer_bed_exp12.py ${ARGS}
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
