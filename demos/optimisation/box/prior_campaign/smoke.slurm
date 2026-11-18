#!/bin/bash --login
# A smoke test of a code change in a few minutes: the tiny box case (10 steps,
# resolution 16, window 5, 4 iterations) with tao-bnls-riesz, once with code copy A
# and once with code copy B, then a comparison of the two records. Use it before a
# long run, to catch a crash or a change of path early.
#
# Usage: sbatch smoke.slurm <code dir A> <code dir B> <run dir>
#
# Code copy B must hold demos/optimisation/box/prior_campaign/compare_records.py.
#SBATCH --account=pawsey0821
#SBATCH --partition=debug
#SBATCH --job-name=box-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
set -euo pipefail
# srun refuses to start when SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree
# ("cpus-per-task set by two different environment variables"). The login shell of a
# --login job carries a value of its own, so clear it and give srun the count.
unset SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=1

CODE_A=$1; CODE_B=$2; RUN_DIR=$3
mkdir -p "$RUN_DIR/A" "$RUN_DIR/B"

ARGS=(--optimiser tao-bnls-riesz --riesz regularisation --window 5 --max-it 4
      --alpha-d 0 --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3
      --initial-guess observed --initial-radius 0.2)

# The reference state is made once, with code copy A, and both sides read it.
# Each side runs in a subshell of its own, so that the two environments never mix.
for SIDE in A B; do
    (
        if [ "$SIDE" = A ]; then CODE_DIR=$CODE_A; else CODE_DIR=$CODE_B; fi
        echo "##### side $SIDE"
        # ---- environment of this job (the same block in every job of the campaign) ----
        export MY_GADOPT="$CODE_DIR"
        source /software/projects/pawsey0821/sghelichkhani/env-hessian.sh
        trap 'rm -rf "$PYOP2_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT
        echo "### job $SLURM_JOB_ID on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "### code $CODE_DIR"
        cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
        python3 -c "import gadopt; print('gadopt    ', gadopt.__file__)"
        # ---- end of the environment block ----
        cd "$RUN_DIR"
        [ -f reference-state.h5 ] || srun -n 1 -c 1 python3 \
            "$CODE_DIR/demos/optimisation/box/forward.py" --steps 10 --resolution 16 \
            --output reference-state.h5
        cd "$SIDE"
        srun -n 1 -c 1 python3 "$CODE_DIR/demos/optimisation/box/inverse.py" \
            "${ARGS[@]}" --checkpoint ../reference-state.h5 --output smoke.json
    )
done

echo "##### comparison, A is the reference"
(
    export MY_GADOPT="$CODE_B"
    source /software/projects/pawsey0821/sghelichkhani/env-hessian.sh
    # A mismatch in any compared field gives exit status 1, which fails the job
    # through ``set -e``, so that a change of path cannot pass unnoticed.
    python3 "$CODE_B/demos/optimisation/box/prior_campaign/compare_records.py" \
        "$RUN_DIR/A/smoke.json" "$RUN_DIR/B/smoke.json" --rows both --all-fields
)
echo "### done"
