#!/bin/bash --login
# The smoke test of step S2b: the Hessian shift recording, before the long runs.
#
# Usage: sbatch smoke_s2b.slurm <code dir> <run dir> <reference state> <mode> \
#            <check value> [inverse.py arguments of the 64-cell case ...]
#
# Mode "regression":
#   1. the tiny box case (10 steps, resolution 16, window 5, 4 iterations), once with
#      the default options and once with a starting shift (-tao_bnk_sval 1e-3,
#      -tao_bnk_psfac 0.1); check_shift.py checks that every recorded shift follows
#      the BNK rules, and that the second run starts at 1e-3;
#   2. the 64-cell case with the given arguments (run (d) of S2 with --max-it 8, so
#      that it holds the negative-curvature solve and the first shifted solve after
#      it); compare_records.py checks that every field of the S2 record except the
#      wall-clock fields is bitwise equal over the shared rows and Krylov solves, and
#      check_shift.py checks that the first shift equals min(imax, max(imin,
#      imfac * gnorm)) from the recorded gnorm.
#   <check value> is the path of the S2 record of the same case.
# Mode "shift":
#   the 64-cell case with the given arguments (run (d) with the shift options of
#   S2b and --max-it 8); check_shift.py checks that the first positive shift equals
#   <check value>, for example 4e-5, which shows that the shift options reached BNK.
#
# The job fails when a check fails, so the long runs that depend on it (afterok)
# start only when the recording and the options work.
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --job-name=s2b-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
set -euo pipefail
# srun refuses to start when SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree
# ("cpus-per-task set by two different environment variables"). The login shell of a
# --login job carries a value of its own, so clear it and give srun the count.
unset SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=1

# Job scripts run with --export=NONE, so everything comes in as an argument.
CODE_DIR=$1; RUN_DIR=$2; CHECKPOINT=$3; MODE=$4; CHECK_VALUE=$5; shift 5
# ---- environment of this job (the same block in every job of the campaign) ----
export MY_GADOPT="$CODE_DIR"
source /software/projects/pawsey0821/sghelichkhani/env-hessian.sh
trap 'rm -rf "$PYOP2_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT
echo "### job $SLURM_JOB_ID on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE_DIR"
cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
python3 -c "import gadopt; print('gadopt    ', gadopt.__file__)"
# ---- end of the environment block ----

BOX=$CODE_DIR/demos/optimisation/box
JOBS=$BOX/prior_campaign
mkdir -p "$RUN_DIR"

if [ "$MODE" = regression ]; then
    echo "### tiny case: default options, and a starting shift"
    mkdir -p "$RUN_DIR/tiny"; cd "$RUN_DIR/tiny"
    [ -f reference-state.h5 ] || srun -n 1 -c 1 python3 "$BOX/forward.py" \
        --steps 10 --resolution 16 --output reference-state.h5
    TINY=(--optimiser tao-bnls-riesz --riesz regularisation --window 5 --max-it 4
          --alpha-d 0 --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3
          --initial-guess observed --initial-radius 0.2 --checkpoint reference-state.h5)
    srun -n 1 -c 1 python3 "$BOX/inverse.py" "${TINY[@]}" --output default.json
    srun -n 1 -c 1 python3 "$BOX/inverse.py" "${TINY[@]}" --output sval.json \
        --tao-opt tao_bnk_sval=1e-3 --tao-opt tao_bnk_psfac=0.1
    python3 "$JOBS/check_shift.py" default.json
    python3 "$JOBS/check_shift.py" sval.json --expect-first 1e-3
fi

echo "### 64-cell case: $*"
mkdir -p "$RUN_DIR/r64"; cd "$RUN_DIR/r64"
srun -n 1 -c 1 python3 "$BOX/inverse.py" --checkpoint "$CHECKPOINT" \
    --output smoke.json "$@"

echo "### checks"
if [ "$MODE" = regression ]; then
    # Bitwise equality with the S2 record in every shared field, rows and solves.
    python3 "$JOBS/compare_records.py" "$CHECK_VALUE" smoke.json \
        --rows both --all-fields --solves --quiet
    python3 "$JOBS/check_shift.py" smoke.json --expect-first-from-gnorm
else
    python3 "$JOBS/check_shift.py" smoke.json --expect-first "$CHECK_VALUE"
fi
echo "### done"
