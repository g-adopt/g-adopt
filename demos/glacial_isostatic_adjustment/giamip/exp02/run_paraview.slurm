#!/bin/bash -l
# GIAMIP Exp02: the ParaView series of a forward run (paraview_exp02.py), on
# Setonix (Pawsey), project pawsey0821.
#
# One rank is enough at L0 and gives the fewest VTK files (a VTKFile writes
# one file per rank and step). Use a few ranks at a finer level with
# --ntasks=<n> at submit.
#
# Do not use the rank count of the forward run (384 for L0-3nodes). On that
# count CheckpointFile loads the surface submesh with its saved
# distribution, and that load is broken: the loaded mesh counts cells more
# than once (job 51182555: area 248.4 instead of 61.0 on 384 ranks; 1 and
# 16 ranks give the right mesh). Any other count partitions the mesh again
# and loads it correctly.
#
# Submit from the run directory of the forward run, for example:
#
#   cd /scratch/pawsey0821/$USER/giamip/exp02/L0-3nodes/forward-protocol
#   sbatch --ntasks=1 --export=NONE \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp02/run_paraview.slurm
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   INPUT   the surface file (default forward_exp02_surface.h5 in the submit
#           directory)
#   OUTDIR  the directory of the ParaView series (default paraview/ in the
#           submit directory)
#   EXTRA   further arguments of paraview_exp02.py (for example --degree 1)
#
# The memory request is 64G. The only measurement is the 63-step series of
# job 51185413: 3.78 GB on one rank. The protocol series has 463 steps, and
# the memory at that step count is not measured. If the memory increases
# with the step count, 32G is near the limit. The extra memory costs little,
# because the job uses one node for a few minutes.
#
#SBATCH --job-name=exp02-pv
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=exp02-pv-%j.out

set -u
PERSIST=/software/projects/pawsey0821/sghelichkhani
source ${PERSIST}/env-exp01-pr4638-pyop2fix.sh
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${EXP01_MPLCONFIGDIR:-${HOME}/.config/matplotlib}

HERE=${MY_GADOPT}/demos/glacial_isostatic_adjustment/giamip/exp02
INPUT=${INPUT:-${SLURM_SUBMIT_DIR}/forward_exp02_surface.h5}
OUTDIR=${OUTDIR:-${SLURM_SUBMIT_DIR}/paraview}

echo "job ${SLURM_JOB_ID}, ${SLURM_NTASKS} ranks"
echo "input ${INPUT}, output ${OUTDIR}, extra ${EXTRA:-}"
date
srun --export=ALL -n ${SLURM_NTASKS} -c 1 python3 ${HERE}/paraview_exp02.py \
    --input ${INPUT} --output_path ${OUTDIR} ${EXTRA:-}
status=$?
rm -rf /tmp/pyop2-${SLURM_JOB_ID} /tmp/tsfc-${SLURM_JOB_ID} 2>/dev/null || true
date
du -sh ${OUTDIR}
echo "exit code ${status}"
exit ${status}
