#!/bin/bash
# Submit step V2 of the prior-covariance campaign on Setonix: the move to the container
# of 2026-11-20 (env-hessian-v2.sh: firedrake a11e4be9, pyop3, PETSc 3.26.0, UFL
# overlay), in two phases.
#
# Usage (on a Setonix login node):
#     submit_v2.sh gates <code dir>
#     submit_v2.sh runs <code dir>
#
# Phase "gates": short jobs that test the new container against records of the old
# one. None of them is a repeat of a finished run.
#   gates     gates_v2.slurm on the debug partition: tiny reference state,
#             check_wrapper.py in three wall settings, the Hessian Taylor harness,
#             and gate 1 (rol-lbfgs and tao-bnls-riesz on the tiny case);
#   pinned    smoke_pinned.slurm with the S2 smoke command: (b) and (c) at 64 cells
#             for two iterations, then check_pinned.py;
#   a5        S2 run (a) to iteration 5 with the S2 flags. Iteration 5 holds the
#             first long Krylov solve of (a) (35 iterations). Compare the record with
#             the S2 (a) record with compare_records.py;
#   ref64     the 64-cell reference state made again in the new container, to
#             compare with the state that every 64-cell record inverted.
# The runs of phase "runs" must wait until these jobs end and their results are
# read. No Slurm dependency links the phases, because a match across containers is
# judged by its size, not by an exit status.
#
# Phase "runs": submit_s2d.sh with this code copy, the step directory V2 and no
# smoke job: Z-b at radii 0.18, 0.2, 0.22, Z-c at radii 0.18 and 0.22, and Z150.
# Every other member of the comparison is an existing record of the old container
# (S2 a and c, S2c Z-d and Z-e, the September rol-lbfgs records).
set -euo pipefail
PHASE=$1
CODE=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
JOBS=$CODE/demos/optimisation/box/prior_campaign
STEP=$BASE/V2
REF64=$BASE/reference/r64-s80/reference-state.h5
mkdir -p "$STEP"
cd "$STEP"

# The S2 command at 64 cells, as submit_s2.sh and submit_s2c.sh give it: the
# September operator as a prior, objective weights zero, Krylov cap 100, BNK's
# default shift options.
PRIOR=(--prior-sigma 31.594026758304967 --prior-L 0.22071129807892717)
COMMON=(--window 80 --alpha-d 0 --alpha-s 0 "${PRIOR[@]}" --initial-guess observed
        --gradient-rtol 1e-6 --max-it 500 --optimiser tao-bnls-riesz
        --riesz regularisation --krylov-max-it 100)

case $PHASE in
gates)
    job=$(sbatch --parsable "$JOBS/gates_v2.slurm" "$CODE" "$STEP/gates")
    echo "gates:  job $job"
    # The S2 smoke command (submit_s2.sh): two iterations of (b) and (c).
    job=$(sbatch --parsable --job-name=v2-pinned "$JOBS/smoke_pinned.slurm" "$CODE" \
        "$STEP/pinned" "$REF64" "${COMMON[@]}" --max-it 2 --initial-radius 0.2)
    echo "pinned: job $job"
    # S2 run (a) to iteration 5: budget 600 as in S2, radius 0.2. The S2c smoke-a
    # job (53414183) ran the same command in the old container in 1:39.
    job=$(sbatch --parsable --job-name=v2-a5 "$JOBS/inverse.slurm" "$CODE" "$STEP/a5" \
        "$REF64" tao-bnls-riesz_regularisation.json "${COMMON[@]}" --work-budget 600 \
        --initial-radius 0.2 --max-it 5)
    echo "a5:     job $job"
    job=$(sbatch --parsable --job-name=v2-ref64 "$JOBS/forward.slurm" "$CODE" \
        "$STEP/reference/r64-s80" 80 64)
    echo "ref64:  job $job"
    ;;
runs)
    "$JOBS/submit_s2d.sh" "$CODE" "$STEP" none
    ;;
*)
    echo "unknown phase $PHASE (gates or runs)" >&2
    exit 2
    ;;
esac
