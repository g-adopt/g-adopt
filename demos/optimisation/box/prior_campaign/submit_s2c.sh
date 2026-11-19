#!/bin/bash
# Submit step S2c of the prior-covariance campaign on Setonix: BNK's shift in the metric.
#
# Usage (on a Setonix login node):
#     submit_s2c.sh <code dir> <S2 run directory>
#
# Step S2b showed that BNK's shift pert * I, with pert = 0.1 * gnorm and gnorm the
# metric norm of the gradient, is 2100 to 1e5 times the nodal Hessian at 64 cells,
# and that runs stall after a negative-curvature solve until it decays. gnorm has
# the scale of a shift by pert * A, with A the metric of the control. This step runs
# BNK with that shift (--bnk-shift metric) and BNK's DEFAULT perturbation options,
# beside a zero-shift control (identity shift with -tao_bnk_imax 0, so that every
# initialisation gives pert = min(0, ...) = 0 and STCG's trust region alone handles
# negative curvature).
#
# Jobs:
#   ref150          the 150-cell reference state (80 steps), work partition;
#   smoke-a         S2 run (a) with default flags to iteration 5: bitwise equal to
#                   the S2 record in every shared row and solve;
#   smoke-d         S2 run (d) with default flags to iteration 7, which holds its
#                   negative-curvature solve: bitwise equal, and the curvature record;
#   smoke-metric    run (d) with --bnk-shift metric to iteration 8: rows 0 to 7 equal
#                   to S2 (d), and the shift after the negative-curvature solve is
#                   applied by the Hessian operator;
#   M-b, M-d, M-e   S2 runs (b), (d), (e) with --bnk-shift metric, budget 800;
#   Z-d, Z-e        S2 runs (d), (e) with -tao_bnk_imax 0, budget 800;
#   D150            the September 150-cell tao-bnls-riesz regularisation command with
#                   the recorder, budget 300, long partition;
#   M150            the same with --bnk-shift metric, budget 600, long partition.
# The long runs start only when the three smoke jobs pass; the 150-cell runs also
# wait for the reference state.
set -euo pipefail
CODE=$1
S2=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
REF64=$BASE/reference/r64-s80/reference-state.h5
REF150_DIR=$BASE/reference/r150-s80
STEP=$BASE/S2c
RECORD=tao-bnls-riesz_regularisation.json
mkdir -p "$STEP"
cd "$STEP"

# The S2 command at 64 cells: the September operator as a prior, from the doubles
# of the S0 run D2, objective weights zero, Krylov cap 100.
PRIOR=(--prior-sigma 31.594026758304967 --prior-L 0.22071129807892717)
COMMON=(--window 80 --alpha-d 0 --alpha-s 0 "${PRIOR[@]}" --initial-guess observed
        --gradient-rtol 1e-6 --max-it 500 --optimiser tao-bnls-riesz
        --riesz regularisation --krylov-max-it 100)
# The case options of the S2 runs.
CASE_A=(--initial-radius 0.2)
CASE_B=(--initial-radius 0.2 --wall-treatment pinned)
CASE_D=(--initial-radius 0.22)
CASE_E=(--initial-radius 0.2 --initial-guess-walls)

REF150=$(sbatch --parsable --job-name=s2c-ref150 "$JOBS/forward.slurm" "$CODE" \
    "$REF150_DIR" 80 150)
echo "ref150:       job $REF150"

smoke() {
    # smoke <name> <S2 record> <rows> <mode> <need negative curvature> <args ...>
    local name=$1 record=$2 rows=$3 mode=$4 need=$5; shift 5
    sbatch --parsable --job-name="s2c-smoke-$name" "$JOBS/smoke_s2c.slurm" "$CODE" \
        "$STEP/smoke-$name" "$REF64" "$record" "$rows" "$mode" "$need" \
        "${COMMON[@]}" --work-budget 600 "$@"
}
# Iteration 5 of (a) holds its first long Krylov solve (35 iterations).
SMOKE_A=$(smoke a "$S2/a/$RECORD" all identity 0 "${CASE_A[@]}" --max-it 5)
echo "smoke-a:      job $SMOKE_A"
# Iteration 7 of (d) holds its negative-curvature solve.
SMOKE_D=$(smoke d "$S2/d/$RECORD" all identity 1 "${CASE_D[@]}" --max-it 7)
echo "smoke-d:      job $SMOKE_D"
# Iteration 8 of (d) holds the first solve after it, the first with a shift.
SMOKE_M=$(smoke metric "$S2/d/$RECORD" 8 metric 1 "${CASE_D[@]}" --bnk-shift metric --max-it 8)
echo "smoke-metric: job $SMOKE_M"
SMOKES=afterok:$SMOKE_A:$SMOKE_D:$SMOKE_M

long64() {
    # long64 <name> <inverse.py options ...>
    local name=$1; shift
    local job
    job=$(sbatch --parsable --dependency=$SMOKES --time=18:00:00 --job-name="s2c-$name" \
        "$JOBS/inverse.slurm" "$CODE" "$STEP/$name" "$REF64" "$RECORD" \
        "${COMMON[@]}" --work-budget 800 "$@")
    echo "run $name: job $job ($*)"
}
long64 M-b "${CASE_B[@]}" --bnk-shift metric
long64 M-d "${CASE_D[@]}" --bnk-shift metric
long64 M-e "${CASE_E[@]}" --bnk-shift metric
long64 Z-d "${CASE_D[@]}" --tao-opt tao_bnk_imax=0
long64 Z-e "${CASE_E[@]}" --tao-opt tao_bnk_imax=0

# The September 150-cell command (~/runs/box-r150-w80/run_r150.sh): legacy metric
# weights, radius 0.469 (the calibrated radius scaled by 150/32), the default Krylov
# cap of 20. One forward call costs about 290 s on Setonix and the September run held
# 11.6 GB, so 300 forward-equivalents take about 28 h with the last iteration.
SEPT150=(--window 80 --alpha-d 0 --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3
         --initial-guess observed --gradient-rtol 1e-6 --max-it 500
         --optimiser tao-bnls-riesz --riesz regularisation --initial-radius 0.469)
long150() {
    # long150 <name> <time> <inverse.py options ...>
    local name=$1 time=$2; shift 2
    local job
    job=$(sbatch --parsable --dependency=$SMOKES:$REF150 --partition=long --mem=32G \
        --time="$time" --job-name="s2c-$name" "$JOBS/inverse.slurm" "$CODE" \
        "$STEP/$name" "$REF150_DIR/reference-state.h5" "$RECORD" "${SEPT150[@]}" "$@")
    echo "run $name: job $job ($*)"
}
long150 D150 2-00:00:00 --work-budget 300
long150 M150 3-12:00:00 --work-budget 600 --bnk-shift metric
