#!/bin/bash
# The three steps of the Gauss-Newton curvature job (NOTES/team/gauss-newton/PLAN.md
# section 5.2). job.slurm runs this script with bash after it has set the
# environment, inside the work directory of the job.
#
# Usage (through submit.sh gn-curv): gn_curv_steps.sh <code dir> <base dir>
#
#   <base dir>  /scratch/pawsey0821/sghelichkhani/hessian/cylindrical-newton, which
#               holds the 5-step state of the gates (gates/run-t5) and the 125-step
#               reference state (reference/s125-dt1.2e-5)
#
# set -e: a failed step ends the job, and step 1 is the gate of the other two. The
# checks G3 and G13 ran before this job, in the serial job "gn-gate" at 2 steps.
set -euo pipefail
CODE_DIR=$1
BASE=$2
D=$CODE_DIR/demos/optimisation/cylindrical
T5=$BASE/gates/run-t5
REF=$BASE/reference/s125-dt1.2e-5

echo "### step 1: checks.py gn G1-G11, 5 steps, serial ($(date -u +%H:%M:%SZ))"
srun -n 1 -c 1 python3 "$D/checks.py" gn --rundir "$T5" --timesteps 5 --delta-t 1.2e-5 \
    --gn-parts G1-G11 --json "$PWD/gn-t5.json"

echo "### step 2: checks.py gn-curvature, 125 steps, $SLURM_NTASKS ranks ($(date -u +%H:%M:%SZ))"
srun -n "$SLURM_NTASKS" -c 1 python3 "$D/checks.py" gn-curvature --rundir "$REF" \
    --timesteps 125 --delta-t 1.2e-5 --json "$PWD/gn-curvature-s125.json"

echo "### step 3: one Gauss-Newton iteration, cap 20, radius 50.6 ($(date -u +%H:%M:%SZ))"
# The P-GN settings: inner product A, zero shift, Gauss-Newton product. Radius 50.6
# is the cylinder value of the box radius 12.8 (compare.default_tao_radius), so the
# radius does not cut the solve; the record holds the step norm to compare with
# 0.79. The control of the iteration and the final control are saved, so that the
# nodes clipped by the bounds can be counted against the first guess.
srun -n "$SLURM_NTASKS" -c 1 python3 "$D/compare.py" --rundir "$REF" --timesteps 125 \
    --delta-t 1.2e-5 --optimiser tao-bnls-riesz --riesz regularisation --bnk-shift zero \
    --hessian gn --initial-radius 50.6 --max-it 1 --krylov-max-it 20 --save-control \
    --output "$PWD/gn-solve-s125.json"
echo "### all three steps done ($(date -u +%H:%M:%SZ))"
