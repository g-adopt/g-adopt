#!/bin/bash
# Submit step S2b of the prior-covariance campaign on Setonix: the BNK Hessian shift.
#
# Usage (on a Setonix login node):
#     submit_s2b.sh <code dir> <S2 record of run d>
#
# Step S2 found runs that stall after one Krylov solve that stops on negative
# curvature. BNK then adds a shift pert * I to the Hessian, initialised at
# imfac * gnorm = 0.1 * gnorm with gnorm the METRIC norm of the gradient (about 9 in
# run d), while the Hessian and the metric A = 2 c_d M + 2 c_s K are of order 1e-4
# in the nodal units of MatShift. This step records the shift and tests it.
#
# Jobs:
#   smoke-regression  the tiny case with and without a starting shift, and run d of
#                     S2 to iteration 8, which must equal the S2 record bitwise in
#                     every shared field and record a first shift of 0.1 * gnorm;
#   smoke-shift       run d with the S2b shift options to iteration 8, which must
#                     record a first shift of imax = 4e-5;
#   d-1200            run d of S2 unchanged, with a work budget of 1200;
#   d-shift, e-shift, a-shift
#                     runs d, e and a of S2 with the S2b shift options, budget 800.
# The four long runs start only when both smoke jobs pass.
#
# The shift options, chosen from bnk.c and the spectrum of the metric at 64 cells
# (lambda_min(A) = 4.74e-7, lambda_max(A) = 3.90e-4) before any S2b result existed:
#   imax  = 4e-5  about 0.1 lambda_max(A). The preconditioned shift A^{-1} pert I
#                 then lies between 0.1 (roughest mode) and 84 (constant mode),
#                 against 2400 to 2e6 for the default first shift of run d (0.93).
#   imin  = 4e-7  about lambda_min(A); a floor that imfac * gnorm never reaches.
#   pmax  = 4e-4  about lambda_max(A), the cap of every increase. The increase rule
#                 is min(pmax, max(pgfac * pert, pmgfac * gnorm)) with pmgfac * gnorm
#                 = 0.1 * gnorm of order 0.1 to 1, so an indefinite solve while
#                 pert > 0 sets the shift to pmax directly, whatever pert was. The
#                 preconditioned shift is then at most 1 on the roughest mode.
#   psfac = 0.1   each good solve divides the shift by 10 instead of 2.5.
#   pmin  = 1e-7  so 4e-5 -> 4e-6 -> 4e-7 -> zero: three shifted solves after one
#                 negative-curvature solve, then no shift.
# imfac, pgfac, pmgfac and pmsfac stay at their defaults: with gnorm of order 1 to
# 10 the clamps above decide every value.
set -euo pipefail
CODE=$1
S2_D_RECORD=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
REFERENCE=$BASE/reference/r64-s80/reference-state.h5
STEP=$BASE/S2b
mkdir -p "$STEP"
cd "$STEP"

# The S2 command: the September operator as a prior, from the doubles of the S0 run
# D2, objective weights zero, Krylov cap 100.
PRIOR=(--prior-sigma 31.594026758304967 --prior-L 0.22071129807892717)
COMMON=(--window 80 --alpha-d 0 --alpha-s 0 "${PRIOR[@]}" --initial-guess observed
        --gradient-rtol 1e-6 --max-it 500 --optimiser tao-bnls-riesz
        --riesz regularisation --krylov-max-it 100)
SHIFT=(--tao-opt tao_bnk_imin=4e-7 --tao-opt tao_bnk_imax=4e-5
       --tao-opt tao_bnk_pmax=4e-4 --tao-opt tao_bnk_psfac=0.1
       --tao-opt tao_bnk_pmin=1e-7)

# Iteration 8 of run d holds the first solve after the negative-curvature solve of
# iteration 7, so --max-it 8 is the shortest run that shows a shift.
SMOKE_R=$(sbatch --parsable --job-name=s2b-smoke-regression "$JOBS/smoke_s2b.slurm" \
    "$CODE" "$STEP/smoke-regression" "$REFERENCE" regression "$S2_D_RECORD" \
    "${COMMON[@]}" --work-budget 600 --initial-radius 0.22 --max-it 8)
echo "smoke-regression: job $SMOKE_R"
SMOKE_S=$(sbatch --parsable --job-name=s2b-smoke-shift "$JOBS/smoke_s2b.slurm" \
    "$CODE" "$STEP/smoke-shift" "$REFERENCE" shift 4e-5 \
    "${COMMON[@]}" --work-budget 600 --initial-radius 0.22 "${SHIFT[@]}" --max-it 8)
echo "smoke-shift:      job $SMOKE_S"

submit() {
    # submit <name> <time> <inverse.py options ...>
    local name=$1 time=$2; shift 2
    local job
    job=$(sbatch --parsable --dependency=afterok:$SMOKE_R:$SMOKE_S --time="$time" \
        --job-name="s2b-$name" "$JOBS/inverse.slurm" "$CODE" "$STEP/$name" "$REFERENCE" \
        tao-bnls-riesz_regularisation.json "${COMMON[@]}" "$@")
    echo "run $name: job $job ($*)"
}
# Run d of S2 took 6.4 h for 600; 1200 needs about 13 h plus the overshoot of the
# last iteration, so the run gets the limit of the work partition.
submit d-1200 24:00:00 --initial-radius 0.22 --work-budget 1200
# Budget 800 covers the work at which run a of S2 reached its last J (779.5 at the
# common ratio). The slowest S2 run took 8.7 h for 600.
submit d-shift 18:00:00 --initial-radius 0.22 --work-budget 800 "${SHIFT[@]}"
submit e-shift 18:00:00 --initial-radius 0.2 --initial-guess-walls --work-budget 800 "${SHIFT[@]}"
submit a-shift 18:00:00 --initial-radius 0.2 --work-budget 800 "${SHIFT[@]}"
