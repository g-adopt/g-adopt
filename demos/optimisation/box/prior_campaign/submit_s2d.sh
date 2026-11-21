#!/bin/bash
# Submit step S2d of the prior-covariance campaign on Setonix: the wall treatment
# again, now with the zero shift that step S2c made the working default.
#
# Usage (on a Setonix login node):
#     submit_s2d.sh <code dir> <step dir> <S2c Z-d record | none>
#
# <step dir> receives one directory per run. With "none" in place of the Z-d record,
# the smoke job is not submitted and the runs start at once. That is the mode for a
# code copy or a container in which a bitwise match with the S2c record is not
# expected; the gates of that container must then have passed before the call
# (submit_v2.sh, step V2).
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
#   Z-c-0.18, Z-c-0.22            pinned walls, Dirichlet operator;
#   Z150    the M150 command of S2c with the zero shift in place of the metric shift,
#           budget 600, long partition.
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
STEP=$2
ZD_RECORD=$3
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
# The reference states of S2 and S2c. Every run of the step inverts the same data as
# the records it is compared with.
REF64=$BASE/reference/r64-s80/reference-state.h5
REF150=$BASE/reference/r150-s80/reference-state.h5
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
DEPENDENCY=()
if [ "$ZD_RECORD" != none ]; then
    SMOKE=$(sbatch --parsable --job-name=s2d-smoke "$JOBS/smoke_s2c.slurm" "$CODE" \
        "$STEP/smoke" "$REF64" "$ZD_RECORD" all identity 1 \
        "${COMMON[@]}" --work-budget 800 --initial-radius 0.22 --max-it 8)
    echo "smoke: job $SMOKE"
    DEPENDENCY=(--dependency=afterok:$SMOKE)
fi

run() {
    # run <name> <inverse.py options ...>
    local name=$1; shift
    local job
    job=$(sbatch --parsable "${DEPENDENCY[@]}" --time=20:00:00 \
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

# The September 150-cell command of S2c (D150, M150): legacy metric weights, radius
# 0.469 (the calibrated radius scaled by 150/32) and the default Krylov cap of 20. It
# is the M150 command with -tao_bnk_imax 0 in place of --bnk-shift metric. M150
# reached budget 600 in 42.9 h on the long partition, so the limit is 3.5 days.
SEPT150=(--window 80 --alpha-d 0 --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3
         --initial-guess observed --gradient-rtol 1e-6 --max-it 500
         --optimiser tao-bnls-riesz --riesz regularisation --initial-radius 0.469)
job=$(sbatch --parsable "${DEPENDENCY[@]}" --partition=long --mem=32G \
    --time=3-12:00:00 --job-name=s2d-Z150 "$JOBS/inverse.slurm" "$CODE" "$STEP/Z150" \
    "$REF150" "$RECORD" "${SEPT150[@]}" --work-budget 600 --tao-opt tao_bnk_imax=0)
echo "run Z150: job $job"
