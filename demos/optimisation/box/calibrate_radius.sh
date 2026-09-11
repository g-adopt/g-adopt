#!/bin/bash
# Find the initial trust-region radius for each optimiser.
#
# Why this exists: the first campaign gave every method the radius 1e-2 that
# g-adopt uses for ROL in the cylindrical script. The TAO bounded Newton types
# then spent six iterations growing the radius before they could take a useful
# step, because BNK grows the radius by about a factor of two per accepted step.
# A test at smoke size showed the initial misfit reached at iteration 4 with
# radius 0.01, at iteration 2 with 0.1 and at iteration 1 with 1.0. One value for
# every method measures the value and not the method.
#
# The sweep runs the settings of the real campaign (the first guess of the paper, the
# full 80-step window) on a coarse mesh, for both inner products.
#
# WARNING: the radius of the TAO trust-region types is not mesh independent. Their
# truncated conjugate-gradient solve measures the step in the plain Euclidean norm of
# the nodal values (`stcg.c:637`, and `bnk.c:447-452` takes the radius from it), which
# is 1/h times the L2 norm of the same field. Going from 32 cells to 64 doubles it, so
# `summarise_calibration.py --scale-tao 2` scales those types. ROL and the TAO
# quasi-Newton types measure the step in the control inner product, which does not
# depend on the mesh.
#
# Usage:
#   ./calibrate_radius.sh                       # every optimiser, four radii
#   RESOLUTION=32 MAX_IT=6 ./calibrate_radius.sh
#
# Read the result with:
#   python summarise_calibration.py --runs ~/runs/radius-calibration
set -euo pipefail

export PATH=/opt/homebrew/bin:$PATH
export OMP_NUM_THREADS=1

ENV=${ENV:-pr4638}
case "$ENV" in
    pr4638) source ~/Workplace/firedrake-pr4638/activate ;;
    main)   export PATH=/opt/homebrew/bin:$PATH
            export OMP_NUM_THREADS=1
            source ~/Workplace/firedrake/venv/bin/activate ;;
    *)      echo "ENV must be pr4638 or main, and is '$ENV'" >&2; exit 1 ;;
esac
echo "environment: $ENV, python $(command -v python)"

WORKTREE=/Users/sghelichkhani/Workplace/g-adopt-worktrees/sghelichkhani/hessian
BOX=$WORKTREE/demos/optimisation/box
export PYTHONPATH=$WORKTREE:$BOX

STEPS=${STEPS:-80}
RESOLUTION=${RESOLUTION:-32}
WINDOW=${WINDOW:-80}
MAX_IT=${MAX_IT:-6}
JOBS=${JOBS:-8}
WALL=${WALL:-30m}
RADII=${RADII:-"0.01 0.1 1.0 10.0"}
RIESZ=${RIESZ:-"L2 regularisation"}
METRIC_ALPHA_D=${METRIC_ALPHA_D:-1e-3}
METRIC_ALPHA_S=${METRIC_ALPHA_S:-1e-3}
ALPHA_D=${ALPHA_D:-0}
ALPHA_S=${ALPHA_S:-0}
OPTIMISERS=${OPTIMISERS:-"rol-lbfgs rol-newton rol-newton-noprec tao-blmvm tao-bqnktr tao-bqnkls tao-bntr tao-bntr-noprec tao-bntr-riesz tao-bnls tao-bnls-riesz"}
OUT=${OUT:-$HOME/runs/radius-calibration}

mkdir -p "$OUT"
cd "$OUT"

if [ ! -f reference-state.h5 ]; then
    echo "forward: steps=$STEPS resolution=$RESOLUTION"
    python "$BOX/forward.py" --steps "$STEPS" --resolution "$RESOLUTION" > forward.log 2>&1
fi

# The useful radius depends on the inner product as well as on the method: the step
# that carries the first guess to the truth has a regularisation norm about a tenth of
# its L2 norm on this problem, so a radius calibrated under one metric is wrong by
# that factor under the other.
for ip in $RIESZ; do
 for o in $OPTIMISERS; do
  for rad in $RADII; do
    d="$OUT/${o}_${ip}_r${rad}"
    if [ -f "$d/run.json" ] && grep -q final_control_error "$d/run.json"; then
        echo "skip $o $rad (done)"; continue
    fi
    mkdir -p "$d"; cp reference-state.h5 "$d/"
    echo "start $o $ip radius $rad"
    ( set +e ; cd "$d" ; timeout "$WALL" python "$BOX/inverse.py" \
          --optimiser "$o" --riesz "$ip" --window "$WINDOW" --max-it "$MAX_IT" \
          --alpha-d "$ALPHA_D" --alpha-s "$ALPHA_S" \
          --metric-alpha-d "$METRIC_ALPHA_D" --metric-alpha-s "$METRIC_ALPHA_S" \
          --initial-guess observed --initial-radius "$rad" \
          --output run.json > run.log 2>&1 ; status=$? ; \
      if [ "$status" -ne 0 ]; then
          echo "$o $ip $rad stopped early (exit $status)" >> "$OUT/incomplete.txt"
      fi ) &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 10; done
  done
 done
done
wait
echo "calibration finished in $OUT"
