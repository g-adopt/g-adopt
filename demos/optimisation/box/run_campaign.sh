#!/bin/bash
# Run the box optimiser comparison on reunion: every optimiser in VARIANTS, for
# each Riesz map, as independent serial jobs in the background. One directory
# per run, one log per run, one JSON per run. Set STEPS, RESOLUTION, WINDOW,
# MAX_IT and JOBS in the environment to change the size and the concurrency.
#
# Usage:
#   ./run_campaign.sh              # full size, 8 jobs at a time
#   STEPS=10 RESOLUTION=16 WINDOW=5 MAX_IT=4 ./run_campaign.sh   # smoke size
#   WALL=2h ./run_campaign.sh     # shorter wall-clock limit per run
#
# The reference forward run is made once in $OUT and shared by every inverse
# run through a copy of reference-state.h5 (a copy, not a symlink, so that
# parallel HDF5 never opens the file through a link).
set -euo pipefail

export PATH=/opt/homebrew/bin:$PATH
export OMP_NUM_THREADS=1
# Which firedrake to use. "pr4638" is the private environment for PR #4638,
# branch JHopeCollins/nlvs-hessian-fix, which caches the adjoint solvers and forms
# and fixes the Hessian of the nonlinear variational solver. "main" is the shared
# environment. The two give different per-call costs, so a campaign must not mix
# them; the environment is recorded in every run through the firedrake path.
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
RESOLUTION=${RESOLUTION:-64}
WINDOW=${WINDOW:-80}
MAX_IT=${MAX_IT:-500}
# Forward-equivalent work every method is allowed. This is the limit that matters:
# an iteration of a gradient-only method costs about 2 forward calls and an iteration
# of a Newton-Krylov method costs about 2 + 2k with k Krylov iterations, so one
# iteration limit gives the two methods budgets that differ by more than ten times.
WORK_BUDGET=${WORK_BUDGET:-600}
# Regularisation weights of the objective. Zero on both leaves the temperature
# misfit alone.
ALPHA_D=${ALPHA_D:-0}
ALPHA_S=${ALPHA_S:-0}
# Weights of the regularisation inner product. They are separate from the objective
# weights, so the metric still exists when the objective carries no regularisation.
METRIC_ALPHA_D=${METRIC_ALPHA_D:-1e-3}
METRIC_ALPHA_S=${METRIC_ALPHA_S:-1e-3}
# First guess. "observed" is the choice of Ghelichkhan et al. (2024, GMD).
GUESS=${GUESS:-observed}
# Stop when the convergence measure of the method falls to this fraction of its
# starting value.
GRAD_RTOL=${GRAD_RTOL:-1e-6}
JOBS=${JOBS:-8}
# Wall-clock limit per run. Every run writes its JSON after each iteration, so a run
# that hits the limit keeps its whole history and only loses the iterations it never
# reached. This bounds the campaign: the Newton variants spend many Hessian-vector
# products per iteration and would otherwise run for days.
WALL=${WALL:-8h}
OUT=${OUT:-$HOME/runs/box-s${STEPS}-r${RESOLUTION}-w${WINDOW}-a${ALPHA_D}}

OPTIMISERS="rol-lbfgs rol-newton rol-newton-noprec tao-blmvm tao-bqnktr tao-bqnkls tao-bntr tao-bntr-noprec tao-bntr-riesz tao-bnls tao-bnls-riesz"
RIESZ="L2 regularisation"

# The inner product reaches the TAO bounded Newton types only through the Riesz
# preconditioner of the "-riesz" variants. For the others it changes nothing, and the
# first campaign confirmed it: tao-bntr, tao-bntr-noprec and tao-bnls gave the same
# numbers to every digit under both inner products. Those three are run once.
skip_pair() {
    case "$1:$2" in
        tao-bntr:regularisation|tao-bntr-noprec:regularisation|tao-bnls:regularisation)
            return 0 ;;
        *) return 1 ;;
    esac
}

# Initial trust-region radius per optimiser AND per inner product. One value for
# every method is wrong: the first campaign used 1e-2 for all of them and the TAO
# bounded Newton types spent six iterations growing the radius before they could take
# a useful step. The radius also depends on the inner product, because the step that
# carries the first guess to the truth has a regularisation norm about a tenth of its
# L2 norm on this problem.
#
# The values come from calibrate_radius.sh and summarise_calibration.py --emit, which
# writes a file of RADIUS_<optimiser>_<metric> variables. Without that file the
# campaign stops, because falling back to one value is the mistake this replaces.
RADII_FILE=${RADII_FILE:-$OUT/radii.sh}
if [ -f "$RADII_FILE" ]; then
    # shellcheck disable=SC1090
    . "$RADII_FILE"
else
    {
        echo "no radius file at $RADII_FILE."
        echo "Run ./calibrate_radius.sh, then summarise it:"
        echo "  python $BOX/summarise_calibration.py --runs ~/runs/radius-calibration --scale-tao N --emit $RADII_FILE"
        echo "where N is $RESOLUTION divided by the calibration resolution."
        echo "Set RADII_FILE to point somewhere else, or ALLOW_DEFAULT_RADIUS=1 to run"
        echo "with 1e-2 for ROL and 1.0 for TAO, which is not a calibrated comparison."
    } >&2
    [ "${ALLOW_DEFAULT_RADIUS:-0}" = "1" ] || exit 1
fi

radius_for() {
    # $1 optimiser, $2 inner product. The name matches summarise_calibration.py.
    local name="RADIUS_${1//-/_}_$2"
    local value="${!name:-}"
    if [ -n "$value" ]; then
        echo "$value"
    else
        case "$1" in
            rol-*)  echo "${ROL_RADIUS:-1e-2}" ;;
            *)      echo "${TAO_RADIUS:-1.0}" ;;
        esac
    fi
}

mkdir -p "$OUT"
cd "$OUT"

# Reference run, once.
if [ ! -f reference-state.h5 ]; then
    echo "forward: steps=$STEPS resolution=$RESOLUTION"
    python "$BOX/forward.py" --steps "$STEPS" --resolution "$RESOLUTION" > forward.log 2>&1
fi

# One inverse run per (optimiser, riesz). Wait when JOBS runs are active.
for r in $RIESZ; do
  for o in $OPTIMISERS; do
    if skip_pair "$o" "$r"; then
        echo "skip $o $r (the inner product does not reach this method; same as its L2 run)"
        continue
    fi
    d="$OUT/${o}_${r}"
    # A JSON exists from the first iteration onwards, so completion is judged by the
    # final_control_error field, which only the final dump carries.
    # A run is finished when its JSON carries final_control_error, which only the
    # final dump writes, or when it was stopped by the wall-clock limit and left the
    # marker below. Without the marker a stopped run would start again from iteration
    # 0 at every later invocation, because no optimiser here can be warm started.
    if [ -f "$d/${o}_${r}.json" ] && grep -q final_control_error "$d/${o}_${r}.json"; then
        echo "skip $o $r (done)"; continue
    fi
    if [ -f "$d/stopped-at-wall-limit" ]; then echo "skip $o $r (stopped at wall limit)"; continue; fi
    mkdir -p "$d"; cp reference-state.h5 "$d/"
    echo "start $o $r"
    # ``set -e`` is switched off inside the subshell. With it on, the shell exits the
    # moment ``timeout`` returns non-zero, before the exit status can be read, and a
    # run cut at the wall-clock limit leaves neither its marker nor a line in
    # incomplete.txt.
    ( set +e ; cd "$d" ; timeout "$WALL" python "$BOX/inverse.py" --optimiser "$o" --riesz "$r" \
          --window "$WINDOW" --max-it "$MAX_IT" \
          --alpha-d "$ALPHA_D" --alpha-s "$ALPHA_S" \
          --metric-alpha-d "$METRIC_ALPHA_D" --metric-alpha-s "$METRIC_ALPHA_S" \
          --initial-guess "$GUESS" --gradient-rtol "$GRAD_RTOL" \
          --work-budget "$WORK_BUDGET" \
          --initial-radius "$(radius_for "$o" "$r")" > run.log 2>&1 ; status=$? ; \
      if [ "$status" -ne 0 ]; then
          echo "$o $r stopped early (exit $status)" >> "$OUT/incomplete.txt"
          # Exit 124 is the wall-clock limit of timeout. The JSON on disk then holds
          # every iteration the run reached.
          if [ "$status" -eq 124 ]; then touch "$d/stopped-at-wall-limit"; fi
      fi ) &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 30; done
  done
done
wait
echo "all runs finished in $OUT"
