#!/bin/bash
# Submit step S0 of the prior-covariance campaign on Setonix: the reference state, a
# smoke test, and the two 60-budget baseline runs D1 and D2.
#
# Usage (on a Setonix login node):
#     submit_s0.sh <code dir D1> <code dir D2>
#
# D1 runs the code of the September baseline command at the branch tip before the
# recording change. D2 runs the same command with the recording change. D1 against
# the September record tests the platform and every code change since September. D2
# against D1 tests that the recording changes no behaviour.
#
# The reference state is made once, with the code of D1, and both runs read it. The
# two runs start when the reference job ends without an error.
set -euo pipefail
CODE_D1=$1
CODE_D2=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE_D2/demos/optimisation/box/prior_campaign
REFERENCE=$BASE/reference/r64-s80
STEP=$BASE/S0
mkdir -p "$REFERENCE" "$STEP"
cd "$STEP"

# The September baseline command (NOTES/HANDOUT-BOX-PRIOR-2026-11-18.md, section 5)
# with the budget of S0.
BASELINE=(--window 80 --alpha-d 0 --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3
          --initial-guess observed --gradient-rtol 1e-6 --max-it 500 --work-budget 60
          --optimiser tao-bnls-riesz --riesz regularisation --initial-radius 0.2)

DEPEND=()
if [ ! -f "$REFERENCE/reference-state.h5" ]; then
    FORWARD=$(sbatch --parsable "$JOBS/forward.slurm" "$CODE_D1" "$REFERENCE" 80 64)
    echo "reference state: job $FORWARD"
    DEPEND=(--dependency=afterok:$FORWARD)
fi

SMOKE=$(sbatch --parsable "$JOBS/smoke.slurm" "$CODE_D1" "$CODE_D2" "$STEP/smoke")
echo "smoke test:      job $SMOKE"

for RUN in D1 D2; do
    if [ "$RUN" = D1 ]; then CODE=$CODE_D1; else CODE=$CODE_D2; fi
    # "${DEPEND[@]+"${DEPEND[@]}"}" expands to nothing when DEPEND is empty. The
    # plain "${DEPEND[@]}" is an "unbound variable" error under set -u in bash
    # before 4.4, and the form here works in every version.
    JOB=$(sbatch --parsable ${DEPEND[@]+"${DEPEND[@]}"} --job-name="s0-$RUN" "$JOBS/inverse.slurm" \
        "$CODE" "$STEP/$RUN" "$REFERENCE/reference-state.h5" \
        tao-bnls-riesz_regularisation.json "${BASELINE[@]}")
    echo "run $RUN:          job $JOB"
done
