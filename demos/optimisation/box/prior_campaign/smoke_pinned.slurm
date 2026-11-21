#!/bin/bash --login
# The smoke test of step S2: short runs of the two pinned cases on the campaign mesh,
# then check_pinned.py on both records. The job fails when a check fails, so the long
# runs that depend on it (afterok) start only when the pinning works.
#
# Usage: sbatch smoke_pinned.slurm <code dir> <run dir> <reference state> \
#            [inverse.py arguments common to both cases ...]
#
# Case b: --wall-treatment pinned. Case c: the same with --prior-wall-bc dirichlet.
#SBATCH --account=pawsey0821
#SBATCH --partition=debug
#SBATCH --job-name=s2-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
set -euo pipefail
# srun refuses to start when SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree
# ("cpus-per-task set by two different environment variables"). The login shell of a
# --login job carries a value of its own, so clear it and give srun the count.
unset SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=1

CODE_DIR=$1; RUN_DIR=$2; CHECKPOINT=$3; shift 3
# ---- environment of this job (the same block in every job of the campaign) ----
export MY_GADOPT="$CODE_DIR"
source /software/projects/pawsey0821/sghelichkhani/env-hessian-v2.sh
trap 'rm -rf "$PYOP3_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT
echo "### job $SLURM_JOB_ID on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE_DIR"
cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
python3 -c "import gadopt; print('gadopt    ', gadopt.__file__)"
# ---- end of the environment block ----

BOX=$CODE_DIR/demos/optimisation/box
for CASE in b c; do
    EXTRA=(--wall-treatment pinned)
    [ "$CASE" = c ] && EXTRA+=(--prior-wall-bc dirichlet)
    mkdir -p "$RUN_DIR/$CASE"; cd "$RUN_DIR/$CASE"
    echo "### smoke $CASE: ${EXTRA[*]}"
    srun -n 1 -c 1 python3 "$BOX/inverse.py" --checkpoint "$CHECKPOINT" \
        --output smoke.json "$@" "${EXTRA[@]}"
done
echo "### checks"
python3 "$BOX/prior_campaign/check_pinned.py" "$RUN_DIR/b/smoke.json" "$RUN_DIR/c/smoke.json"
echo "### done"
