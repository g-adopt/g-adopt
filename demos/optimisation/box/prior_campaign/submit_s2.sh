#!/bin/bash
# Submit step S2 of the prior-covariance campaign on Setonix: the wall treatment.
#
# Usage (on a Setonix login node):
#     submit_s2.sh <code dir> <S1 test 3 record>
#
# Jobs:
#   smoke  the two pinned cases for two iterations, then check_pinned.py (debug);
#   T3     the S1 test 3 command at this code with default wall flags, which must
#          reproduce the S1 record bitwise except work (test 3 of S2);
#   a to e the five 600-budget runs, which start only when the smoke job passes:
#          a  the present state;
#          b  wall nodes pinned, Neumann operator;
#          c  wall nodes pinned, Dirichlet operator;
#          d  the present state with initial radius 0.22, the noise reference;
#          e  the present state with the first guess carrying the wall values.
# All runs use the September operator stated as a prior, objective weights zero and a
# Krylov cap of 100, so that a capped count does not hide the effect of the walls.
set -euo pipefail
CODE=$1
S1_RECORD=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
REFERENCE=$BASE/reference/r64-s80/reference-state.h5
STEP=$BASE/S2
mkdir -p "$STEP"
cd "$STEP"

# The September operator as a prior, from the doubles of the S0 run D2.
PRIOR=(--prior-sigma 31.594026758304967 --prior-L 0.22071129807892717)
COMMON=(--window 80 --alpha-d 0 --alpha-s 0 "${PRIOR[@]}" --initial-guess observed
        --gradient-rtol 1e-6 --max-it 500 --optimiser tao-bnls-riesz
        --riesz regularisation)

SMOKE=$(sbatch --parsable "$JOBS/smoke_pinned.slurm" "$CODE" "$STEP/smoke" "$REFERENCE" \
    "${COMMON[@]}" --max-it 2 --krylov-max-it 100 --initial-radius 0.2)
echo "smoke:  job $SMOKE"

# Test 3 of S2: the S1 test 3 command, unchanged, at this code.
T3=$(sbatch --parsable --job-name=s2-T3 "$JOBS/inverse.slurm" "$CODE" "$STEP/T3" \
    "$REFERENCE" tao-bnls-riesz_regularisation.json "${COMMON[@]}" \
    --work-budget 60 --initial-radius 0.2)
echo "T3:     job $T3 (compare with $S1_RECORD)"

LONG=(--work-budget 600 --krylov-max-it 100)
declare -A CASE=(
    [a]="--initial-radius 0.2"
    [b]="--initial-radius 0.2 --wall-treatment pinned"
    [c]="--initial-radius 0.2 --wall-treatment pinned --prior-wall-bc dirichlet"
    [d]="--initial-radius 0.22"
    [e]="--initial-radius 0.2 --initial-guess-walls"
)
for RUN in a b c d e; do
    # The case options are single words, so word splitting of the string is safe.
    # shellcheck disable=SC2206
    OPTIONS=(${CASE[$RUN]})
    JOB=$(sbatch --parsable --dependency=afterok:$SMOKE --time=14:00:00 \
        --job-name="s2-$RUN" "$JOBS/inverse.slurm" "$CODE" "$STEP/$RUN" "$REFERENCE" \
        tao-bnls-riesz_regularisation.json "${COMMON[@]}" "${LONG[@]}" "${OPTIONS[@]}")
    echo "run $RUN:  job $JOB  (${CASE[$RUN]})"
done
