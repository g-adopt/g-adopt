#!/bin/bash
# PBS wrapper for the cylindrical Hessian Taylor test on Gadi.
#
# One job runs one objective term at one trajectory length with one rheology. The
# reference state for that (STEPS, VISC) pair is built on the first job that needs
# it and reused afterwards, so submit the first job of a pair on its own and let it
# finish before the other cases of the same pair are submitted.
#
# Variables, passed with qsub -v:
#   STEPS   trajectory length in timesteps (default 20). CI runs 125.
#   CASE    Tobs | uobs | damping | smoothing (default Tobs)
#   VISC    production | hard (default production). production is forward.py as it
#           stands, the smooth floor. hard restores the former conditional clip, as
#           the control run that must show the eps^2 defect.
#   FD_TAG  firedrake module tag (default JHopeCollins_nlvs-hessian-fix, the PR
#           #4638 container, paired with petsc/3.25.0 by the private module file in
#           ~/modules). main-20260902 is the other tested choice.
#   LEVELS  number of epsilon halvings (default 5)
#   EPS0    largest epsilon of the sweep (default 0.05). For long trajectories the
#           series in eps has a small radius, so start lower, for example 0.0125.
#   REPEATS number of timed calls of functional, derivative and Hessian (default:
#           the harness default of 2, 2, 3). Use 1 for a pure Taylor sweep.
#
# Usage:
#   cd /scratch/xd2/sg8812/g-adopt-worktrees/sghelichkhani/hessian/tests/hessian
#   qsub -v STEPS=20,CASE=Tobs submit_gadi.sh
#   qsub -v STEPS=20,CASE=uobs submit_gadi.sh              # after the Tobs job
#   qsub -v STEPS=60,CASE=Tobs,VISC=hard submit_gadi.sh
#   qsub -v STEPS=125,CASE=Tobs -l walltime=10:00:00 submit_gadi.sh
#
# Resources match the CI longtest for this case: 16 cores and 64 GB.

#PBS -P xd2
#PBS -q normalsr
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l jobfs=50GB
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/xd2+gdata/xd2+gdata/fp50
#PBS -l wd
#PBS -j oe
#PBS -N hessian_taylor

# Strict mode is set after /etc/profile, because the system lang.sh references LANG
# without a default and trips set -u.
source /etc/profile
set -eo pipefail

STEPS="${STEPS:-20}"
CASE="${CASE:-Tobs}"
VISC="${VISC:-production}"
FD_TAG="${FD_TAG:-JHopeCollins_nlvs-hessian-fix}"
LEVELS="${LEVELS:-5}"
EPS0="${EPS0:-0.05}"
REPEATS="${REPEATS:-}"
NCPUS="${PBS_NCPUS:-16}"

module use /g/data/fp50/modules
module use "${HOME}/modules"
module load "firedrake/${FD_TAG}"

WORKTREE=/scratch/xd2/sg8812/g-adopt-worktrees/sghelichkhani/hessian
export PYTHONPATH="${WORKTREE}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMPI_MCA_io=ompio
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
# The CI longtest sets this too. It makes PyOP2 fail loudly on a collective call
# that not every rank reaches.
export PYOP2_SPMD_STRICT=1

# One run directory per (firedrake tag, steps, rheology). The reference state
# Checkpoint_State.h5 inside it is shared by every case of that triple.
RUNROOT=/scratch/xd2/sg8812/hessian-runs
RUNDIR="${RUNROOT}/${FD_TAG}/t${STEPS}_${VISC}"
mkdir -p "${RUNDIR}"

# The sweep parameters are part of the file name, so that a second sweep on the
# same reference state does not overwrite the first.
TAG="${CASE}_t${STEPS}_${VISC}_${FD_TAG}_e${EPS0}_L${LEVELS}"
JSON="${RUNDIR}/${TAG}.json"
LOG="${RUNDIR}/${TAG}.log"

echo "==== Gadi Hessian Taylor job ===="
echo "PBS_JOBID  = ${PBS_JOBID:-<interactive>}"
echo "PBS_NCPUS  = ${NCPUS}"
echo "STEPS      = ${STEPS}"
echo "CASE       = ${CASE}"
echo "VISC       = ${VISC}"
echo "FD_TAG     = ${FD_TAG}"
echo "EPS0       = ${EPS0}"
echo "LEVELS     = ${LEVELS}"
echo "REPEATS    = ${REPEATS:-default}"
echo "RUNDIR     = ${RUNDIR}"
echo "Worktree   = $(git -C "${WORKTREE}" rev-parse --short HEAD) $(git -C "${WORKTREE}" branch --show-current)"
echo "Started:    $(date -Is)"
echo

# Build the reference state on first use only. The forward run and the inverse
# problem must share the trajectory length and the rheology, so the state is keyed
# on both through the directory name.
MAKE_REF=""
if [[ ! -f "${RUNDIR}/Checkpoint_State.h5" ]]; then
    MAKE_REF="--make-reference"
    echo "No reference state in ${RUNDIR}: the forward model runs first."
fi

VISC_ARGS=""
if [[ "${VISC}" != "production" ]]; then
    VISC_ARGS="--viscosity ${VISC}"
fi
REPEAT_ARGS=""
if [[ -n "${REPEATS}" ]]; then
    REPEAT_ARGS="--repeats ${REPEATS}"
fi

cd "${WORKTREE}/tests/hessian"

# Wall time of the whole job, as a check on the per-call timings in the log.
START=$(date +%s)
mpiexec -n "${NCPUS}" python3 cylindrical_hvp_taylor.py "${CASE}" \
    --timesteps "${STEPS}" \
    --rundir "${RUNDIR}" \
    --perturbation smooth --seed 42 \
    --eps0 "${EPS0}" --levels "${LEVELS}" \
    --json "${JSON}" \
    ${MAKE_REF} ${VISC_ARGS} ${REPEAT_ARGS} 2>&1 | tee "${LOG}"
END=$(date +%s)

echo
echo "Finished:   $(date -Is)"
echo "Wall time:  $((END - START)) s on ${NCPUS} ranks"
echo "JSON:       ${JSON}"
