#!/bin/bash
# Submit acceptance test 3 of step S1 of the prior-covariance campaign on Setonix: the
# 60-budget baseline command with the legacy metric weights replaced by the prior
# flags --prior-sigma and --prior-L.
#
# Usage (on a Setonix login node):
#     submit_s1.sh <code dir> <reference record>
#
# The reference record is the S0 run D2. sigma and L are computed from the doubles of
# its metric coefficients, sigma = 1/sqrt(metric_c_d) and L = sqrt(metric_c_s /
# metric_c_d), and passed with 17 significant digits, which reproduces each double
# exactly. Rounded digits would change the operator, and the comparison with D2 would
# then measure the rounding instead of the code.
#
# Compare the result with D2 by compare_records.py. If the coefficients of the two
# records are bitwise equal, every old field except work must be bitwise equal. If
# they are not, iterations 0 to 4 must agree to 1e-12.
set -euo pipefail
CODE=$1
RECORD=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
REFERENCE=$BASE/reference/r64-s80/reference-state.h5
STEP=$BASE/S1
mkdir -p "$STEP"
cd "$STEP"

# The system python of the login node is enough for json and math, and repr() of a
# float gives the shortest string that reads back as the same double.
read -r SIGMA L < <(python3 - "$RECORD" <<'PY'
import json, math, sys
weights = json.load(open(sys.argv[1]))["weights"]
c_d, c_s = weights["metric_c_d"], weights["metric_c_s"]
print(repr(1.0 / math.sqrt(c_d)), repr(math.sqrt(c_s / c_d)))
PY
)
echo "sigma $SIGMA, L $L, from $RECORD"

# The September baseline command with the budget of S0, and the prior flags in place
# of --metric-alpha-d and --metric-alpha-s.
COMMAND=(--window 80 --alpha-d 0 --alpha-s 0 --prior-sigma "$SIGMA" --prior-L "$L"
         --initial-guess observed --gradient-rtol 1e-6 --max-it 500 --work-budget 60
         --optimiser tao-bnls-riesz --riesz regularisation --initial-radius 0.2)

JOB=$(sbatch --parsable --job-name=s1-T3 "$JOBS/inverse.slurm" \
    "$CODE" "$STEP/T3" "$REFERENCE" tao-bnls-riesz_regularisation.json "${COMMAND[@]}")
echo "test 3: job $JOB"
