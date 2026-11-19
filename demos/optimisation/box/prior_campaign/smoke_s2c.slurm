#!/bin/bash --login
# A smoke test of step S2c: one short 64-cell run, compared with its S2 record and
# checked by check_s2c.py. The job fails when a check fails, so the long runs that
# depend on it (afterok) start only when it passes.
#
# Usage: sbatch smoke_s2c.slurm <code dir> <run dir> <reference state> <S2 record> \
#            <rows> <mode> <need negative curvature: 0|1> [inverse.py arguments ...]
#
# <rows> "all": every shared row and solve must equal the S2 record bitwise (except
#        the wall-clock fields): the regression test of default flags.
# <rows> N:     only the first N iteration rows and their solves must be equal: for a
#        run whose Hessian shift differs from the S2 run only after a
#        negative-curvature solve, which is in row N - 1.
# <mode> identity|metric: the shift mode that check_s2c.py expects.
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --job-name=s2c-smoke
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
CODE_DIR=$1; RUN_DIR=$2; CHECKPOINT=$3; S2_RECORD=$4; ROWS=$5; MODE=$6; NEED=$7
shift 7
# ---- environment of this job (the same block in every job of the campaign) ----
export MY_GADOPT="$CODE_DIR"
source /software/projects/pawsey0821/sghelichkhani/env-hessian.sh
trap 'rm -rf "$PYOP2_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT
echo "### job $SLURM_JOB_ID on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE_DIR"
cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
python3 -c "import gadopt; print('gadopt    ', gadopt.__file__)"
# ---- end of the environment block ----

JOBS=$CODE_DIR/demos/optimisation/box/prior_campaign
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
echo "### inverse.py $*"
srun -n 1 -c 1 python3 "$CODE_DIR/demos/optimisation/box/inverse.py" \
    --checkpoint "$CHECKPOINT" --output smoke.json "$@"

echo "### checks"
if [ "$ROWS" = all ]; then
    python3 "$JOBS/compare_records.py" "$S2_RECORD" smoke.json \
        --rows both --all-fields --solves --quiet
else
    python3 "$JOBS/compare_records.py" "$S2_RECORD" smoke.json \
        --rows iterations --all-fields --solves --max-rows "$ROWS" --quiet
fi
CHECK=(--mode "$MODE")
[ "$NEED" = 1 ] && CHECK+=(--need-negative-curvature)
python3 "$JOBS/check_s2c.py" smoke.json "${CHECK[@]}"
echo "### done"
