#!/bin/bash -l
# GIAMIP Exp10: the forward run from the bed of an inversion at the steps of
# the GIAMIP protocol (forward_exp10.py --schedule protocol), on Setonix
# (Pawsey), project pawsey0821.
#
# The march has 462 steps from 122 ka: 500 yr steps to year -26000, then
# 100 yr steps to year +1000; year 0 is step 452. The ice of every step is
# read from giamip in the job with the transfer of prepare_exp10.py. The
# preconditioner is rebuilt every 20 solves (REBUILD_EVERY) and at the
# change of dt after year -26000.
#
# The job must have the nodes and ranks of the prepare job that wrote the
# input file (3 nodes, 384 ranks for L0-3nodes); the script stops with an
# error otherwise.
#
# Submit from the run directory, for example:
#
#   mkdir -p /scratch/pawsey0821/$USER/giamip/exp10/L0-3nodes/forward-protocol
#   cd /scratch/pawsey0821/$USER/giamip/exp10/L0-3nodes/forward-protocol
#   sbatch --export=NONE,STATE=../invert-full/state.h5 \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_forward_protocol.slurm
#
# The time limit is for L0 on 3 nodes. In the forward run at 2 kyr steps
# (Exp01 job 51182503) the solve of a step took 18-45 s, the first step
# 320 s (the preconditioner build and the kernel compilation), the write
# 2-4 s. 462
# steps then take about 3-6 h. 10 h leaves room for slower steps; a finer
# level needs about 4x the time per level, and then more nodes or a restart
# (the script has none).
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   STATE          the state.h5 of the inversion (required)
#   REBUILD_EVERY  solves between preconditioner rebuilds (default 20)
#   EXTRA          further arguments of forward_exp10.py
#   SURFACE_LAYOUT the lfs setstripe options of the surface file (default
#                  "-c 1 -p flash", see below)
#
#SBATCH --job-name=exp10-fwdp
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=10:00:00
#SBATCH --output=exp10-fwdp-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
# The environment of the inversion, so both jobs run the same code. The
# forward run also imports giamip (the ice of every protocol year), as the
# prepare job does in the same environment.
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp10
OUT=${OUT:-${SLURM_SUBMIT_DIR}}
STATE=${STATE:?set STATE to the state.h5 of the inversion}
REBUILD_EVERY=${REBUILD_EVERY:-20}

ARGS="--state ${STATE} --output_path ${OUT} --schedule protocol"
ARGS="${ARGS} --gia_rebuild_every ${REBUILD_EVERY} ${EXTRA:-}"

# The Lustre layout of the surface file. All 384 ranks append their small
# parts of every field to this one file at every step (CheckpointFile, about
# 15 MB per step in 14 datasets). The time of such a write depends on the
# response time of the OSTs that hold the file, much more than the data
# volume does. With the default layout of /scratch (the first 50 MB on one
# flash OST, the rest on 4 disk OSTs) the append took 30-640 s per step
# when the file had disk OSTs among 44-50, against 2-10 s on other disk OSTs
# (2026-10-20). The probe job 51222184 on the same 3 nodes and with the same
# code gave 92-142 s per step with the disk OSTs of the Exp04 file and 1-3 s
# with those of the Exp05 file. A single stripe on the flash pool keeps the
# whole file on flash OSTs, and the Lustre allocator picks the OST; in the
# probe job 51223061 this layout gave 0.8 s per step. The file has about
# 15 MB per step, 5.7 GB after 378 steps and 7 GB after 462 steps.
# CheckpointFile opens the file with mode "w", which truncates it and keeps
# this layout. The run writes a new surface file in any case, so removing
# an old one here loses nothing that the run would keep.
SURFACE=${OUT}/forward_exp10_surface.h5
SURFACE_LAYOUT=${SURFACE_LAYOUT:-"-c 1 -p flash"}
mkdir -p ${OUT}
rm -f ${SURFACE}
lfs setstripe ${SURFACE_LAYOUT} ${SURFACE} || { echo "lfs setstripe failed"; exit 1; }
echo "surface file layout: $(lfs getstripe -c ${SURFACE}) stripe(s), pool $(lfs getstripe -p ${SURFACE}), OST $(lfs getstripe -i ${SURFACE})"

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} ranks"
echo "worktree ${MY_GADOPT} at $(cd ${MY_GADOPT} && git rev-parse --short HEAD 2>/dev/null)"
echo "arguments: ${ARGS}"
date

srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/forward_exp10.py ${ARGS}
status=$?

# The kernel caches of this job, in /tmp of each node.
srun --ntasks-per-node=1 -N ${SLURM_JOB_NUM_NODES} rm -rf \
    /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
echo "exit code ${status}"
exit ${status}
