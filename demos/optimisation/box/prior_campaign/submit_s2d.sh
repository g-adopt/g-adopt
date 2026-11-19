#!/bin/bash
# Submit step S2d of the prior-covariance campaign on Setonix: the wall treatment
# again, now with the zero shift that step S2c made the working default.
#
# Usage (on a Setonix login node):
#     submit_s2d.sh <code dir> <S2c Z-d record>
#
# Step S2 compared (b) pinned walls with the Neumann operator and (c) pinned walls
# with the Dirichlet operator, but its spread came from BNK's identity shift. With
# -tao_bnk_imax 0 BNK never shifts, and STCG's trust region alone handles negative
# curvature. This step runs three radii of each treatment, so that the comparison
# has a spread of its own.
#
# Jobs:
#   smoke   S2c run Z-d to iteration 8 with this code: bitwise equal to the Z-d
#           record in every shared row and Krylov solve (the new step_norm_metric
#           field only adds), and the curvature record at iteration 7;
#   Z-b-0.18, Z-b-0.2, Z-b-0.22   pinned walls, Neumann operator;
#   Z-c-0.18, Z-c-0.22            pinned walls, Dirichlet operator.
# S2 run (c) at radius 0.2 is the third (c) member: it met no negative curvature and
# took only Newton steps (its -tao_view), so BNK never set a shift in it, and it is
# valid under every shift mode.
#
# The decision rule, fixed before the runs: for J <= 1e-4 and J <= 1e-6, take the
# median over the three radii of the work at the first iteration at or below J (the
# upper end of each interval), for (b) and for (c); the medians of the lower ends
# are reported beside them. If the medians of (b) and (c) differ by less than a
# factor of 1.5 at both thresholds, take (c), because it is the physical statement.
# Otherwise take the faster one, and write down why.
set -euo pipefail
CODE=$1
ZD_RECORD=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
REF64=$BASE/reference/r64-s80/reference-state.h5
STEP=$BASE/S2d
RECORD=tao-bnls-riesz_regularisation.json
mkdir -p "$STEP"
cd "$STEP"

# The S2 command at 64 cells, Krylov cap 100, and the zero shift.
PRIOR=(--prior-sigma 31.594026758304967 --prior-L 0.22071129807892717)
COMMON=(--window 80 --alpha-d 0 --alpha-s 0 "${PRIOR[@]}" --initial-guess observed
        --gradient-rtol 1e-6 --max-it 500 --optimiser tao-bnls-riesz
        --riesz regularisation --krylov-max-it 100 --tao-opt tao_bnk_imax=0)

# Iteration 7 of Z-d holds its first negative-curvature solve, iteration 8 the solve
# after it.
SMOKE=$(sbatch --parsable --job-name=s2d-smoke "$JOBS/smoke_s2c.slurm" "$CODE" \
    "$STEP/smoke" "$REF64" "$ZD_RECORD" all identity 1 \
    "${COMMON[@]}" --work-budget 800 --initial-radius 0.22 --max-it 8)
echo "smoke: job $SMOKE"

run() {
    # run <name> <inverse.py options ...>
    local name=$1; shift
    local job
    job=$(sbatch --parsable --dependency=afterok:$SMOKE --time=20:00:00 \
        --job-name="s2d-$name" "$JOBS/inverse.slurm" "$CODE" "$STEP/$name" "$REF64" \
        "$RECORD" "${COMMON[@]}" --work-budget 800 "$@")
    echo "run $name: job $job ($*)"
}
for R in 0.18 0.2 0.22; do
    run "Z-b-$R" --wall-treatment pinned --initial-radius "$R"
done
for R in 0.18 0.22; do
    run "Z-c-$R" --wall-treatment pinned --prior-wall-bc dirichlet --initial-radius "$R"
done
