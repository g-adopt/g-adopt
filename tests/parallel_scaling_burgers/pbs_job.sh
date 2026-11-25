#!/bin/bash
# One weak-scaling job: the substituted reference plus one coupled
# configuration at one level. Submit from this directory with, for example:
#
#   qsub -v LEVEL=5,CONFIG=schur-a11 -l ncpus=104 -l mem=480GB -l walltime=01:00:00 pbs_job.sh
#
# Resource guide (normalsr, 104 cores per node):
#   level 5:   104 cores,  mem 480GB,  walltime 01:00:00
#   level 6:   832 cores,  mem 3840GB, walltime 02:00:00
#   level 7:  6656 cores,  mem 30TB,   walltime 05:00:00  (queue limit at that size)
# CONFIG is one of: multiplicative schur-a11 schur-substituted static-condensation
#
# The first run is a one-step warmup that pays the compilation cost. The second
# run is the measured one and writes a PETSc log to profile_${LEVEL}_${CONFIG}.txt.
# For the nested log with rank-averaged self times, used by plot_breakdown.py:
#
#   qsub -v LEVEL=5,CONFIG=schur-a11,RUN_TAG=nested,LOG_FORMAT=ascii_xml ... pbs_job.sh
#PBS -P xd2
#PBS -q normalsr
#PBS -l storage=scratch/xd2+gdata/xd2+gdata/fp50
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -W umask=0022

set -eu
: "${LEVEL:?set LEVEL}"
: "${CONFIG:?set CONFIG}"
: "${GADOPT_CHECKOUT:=$(cd ../.. && pwd)}"

# The default g-adopt module carries Firedrake 2026.4.2, which predates the
# GPU solver options that current main imports (get_device_type). Use a
# dated Firedrake main build instead; the branch's gadopt then comes from the
# worktree through PYTHONPATH.
module use /g/data/fp50/modules
module load "${FIREDRAKE_MODULE:-firedrake/main-20260902}"
export PYTHONPATH="${GADOPT_CHECKOUT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export STEPS="${STEPS:-4}"

# RUN_TAG adds a suffix to every output file name, so a rerun does not
# overwrite an earlier one. LOG_FORMAT selects the PETSc log format: empty
# gives the flat text table (one line per event, maximum over ranks, events
# nested inside each other); ascii_xml gives the nested tree with the self
# time of every node averaged over ranks, which a cost breakdown needs.
: "${RUN_TAG:=}"
: "${LOG_FORMAT:=}"
SUFFIX="${RUN_TAG:+_$RUN_TAG}"
case "$LOG_FORMAT" in
    "") EXT=txt ;;
    ascii_xml) EXT=xml ;;
    *) EXT=txt ;;
esac

echo "gadopt from: $(python -c 'import gadopt; print(gadopt.__file__)')"

mpirun -np "${PBS_NCPUS}" python burgers_scaling.py "${LEVEL}" "${CONFIG}" -n 1 \
    > "level_${LEVEL}_${CONFIG}${SUFFIX}_warmup.out" 2> "level_${LEVEL}_${CONFIG}${SUFFIX}_warmup.err"

export PETSC_OPTIONS="-log_view :profile_${LEVEL}_${CONFIG}${SUFFIX}.${EXT}${LOG_FORMAT:+:$LOG_FORMAT}"
mpirun -np "${PBS_NCPUS}" python burgers_scaling.py "${LEVEL}" "${CONFIG}" -n "${STEPS}" \
    > "level_${LEVEL}_${CONFIG}${SUFFIX}_full.out" 2> "level_${LEVEL}_${CONFIG}${SUFFIX}_full.err"
