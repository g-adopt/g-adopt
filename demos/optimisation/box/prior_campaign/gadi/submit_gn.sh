#!/bin/bash
# Submit the Gauss-Newton box runs on Gadi (project nm03, queue normal, serial).
#
# Usage (on a Gadi login node; from reunion through
#        ssh gadi 'source /etc/profile && bash <code>/demos/optimisation/box/prior_campaign/gadi/submit_gn.sh ...'):
#     submit_gn.sh <code dir> gates [<parent code dir>]
#     submit_gn.sh <code dir> runs
#     submit_gn.sh <code dir> optional T-GN|T-GN-R-EW
#
# The plan, the costs and the gates are in NOTES/team/gauss-newton/PLAN.md (revision
# 3) of the reunion worktree. No PBS dependency links two phases: a person reads the
# results of a phase, writes GATES.md, and only then submits the next.
#
# Phase "gates": gates_gn.pbs (the tiny-case checks, 16 GB, 1 h) and smoke_gn.pbs
#   (the 64-cell smoke, timing.json and G11, 24 GB, 4 h). GATES.md must hold r_GN,
#   full_per_forward and the forward time of timing.json before phase "runs".
# Phase "runs": the four required runs, 24 GB, 20 h each:
#   T-ROL   rol-lbfgs, L2, free walls, radius 1.0      the g-adopt baseline
#   T-N     tao-bnls-riesz, full Hessian, radius 0.2   the Newton baseline
#   T-N-R   tao-bnls-riesz, full Hessian, radius 12.8  the radius alone
#   T-GN-R  tao-bnls-riesz, Gauss-Newton, radius 12.8  Gauss-Newton where it can differ
# Phase "optional": one optional arm, each only after its own approval by Sia:
#   T-GN       Gauss-Newton, radius 0.2
#   T-GN-R-EW  Gauss-Newton, radius 12.8, Eisenstat-Walker tolerance
#
# The decision rule (PLAN.md section 4.5, fixed and committed with summarise_gn.py
# before any run; nobody reads a run record before every run of the rule has ended).
# Work of a row: forward + adjoint + p_full hessian + p_full r_GN gn_hessian, with
# p_full = full_per_forward and r_GN = gn_over_full of timing.json. J_X(W) is the J
# of the last row of run X with work at or below W; W_X(j) the work interval in
# which X first reaches J <= j; S(X, Y, W*) = W* / (upper end of W_X(J_Y(W*))), and 0
# if X never reaches it. N* is the better of T-N and T-N-R at each level. For each
# Gauss-Newton arm X:
#   A, against ROL:   S(X, T-ROL, 200) >= 1 and S(X, T-ROL, 400) >= 1.5;
#   B, the GN effect: S(X, N*, 200) >= 1.5 and S(X, N*, 400) >= 1;
#   C, no harm:       S(X, N*, 600) >= 0.67.
# Floor guard: a level is valid for a reference Y when J_Y(W*) - J_min >= 0.05
# (J_0 - J_min); an invalid 400 or 600 is replaced by the largest valid level below
# it from 150, 200, 300, 400. Verdicts: V1 (A, B, C pass), V1b (A, B pass, C fails),
# V2 (A, C pass, B fails), V3 (B, C pass, A fails), V4 (only C passes), V5 (C fails
# otherwise). P-GN on the cylinder needs test C and the curvature job to pass.
set -euo pipefail
CODE=$(cd "$1" && pwd)
PHASE=$2
GADI_DIR=$CODE/demos/optimisation/box/prior_campaign/gadi
source "$GADI_DIR/run_flags.sh"
GN_BASE=/scratch/nm03/sg8812/hessian/gauss-newton
mkdir -p "$GN_BASE"

submit_arm() {
    # $1: the run name, then the arguments of the arm. Prints the job id.
    local name=$1; shift
    local out=$GN_BASE/$name
    mkdir -p "$out"
    if [ -e "$out/record.json" ]; then
        echo "$out/record.json exists; a finished run is never repeated" >&2
        return 1
    fi
    printf '%s\n' "$@" > "$out/args"
    qsub -v CODE="$CODE",OUT="$out" -N "gn-$name" -o "$out/pbs.log" "$GADI_DIR/inverse.pbs"
}

case $PHASE in
gates)
    PARENT=${3:-$GN_BASE/code-84ade15f}
    [ -d "$PARENT/demos/optimisation/box" ] || { echo "no parent copy $PARENT" >&2; exit 2; }
    mkdir -p "$GN_BASE/gates" "$GN_BASE/smoke"
    g=$(qsub -v CODE="$CODE",PARENT="$PARENT",OUT="$GN_BASE/gates" \
        -o "$GN_BASE/gates/pbs.log" "$GADI_DIR/gates_gn.pbs")
    s=$(qsub -v CODE="$CODE",OUT="$GN_BASE/smoke" -o "$GN_BASE/smoke/pbs.log" \
        "$GADI_DIR/smoke_gn.pbs")
    echo "gates: $g (in $GN_BASE/gates), smoke: $s (in $GN_BASE/smoke)"
    ;;
runs)
    for arm in T-ROL T-N T-N-R T-GN-R; do
        case $arm in
        T-ROL)  args=("${COMMON[@]}" "${ROL[@]}" "${T_ROL[@]}") ;;
        T-N)    args=("${COMMON[@]}" "${NEWTON[@]}" "${T_N[@]}") ;;
        T-N-R)  args=("${COMMON[@]}" "${NEWTON[@]}" "${T_N_R[@]}") ;;
        T-GN-R) args=("${COMMON[@]}" "${NEWTON[@]}" "${T_GN_R[@]}") ;;
        esac
        # The id first, so that a failed qsub stops the script (set -e).
        job=$(submit_arm "$arm" "${args[@]}")
        echo "$arm: $job"
    done
    ;;
optional)
    arm=${3:?name the optional arm: T-GN or T-GN-R-EW}
    case $arm in
    T-GN)      args=("${COMMON[@]}" "${NEWTON[@]}" "${T_GN[@]}") ;;
    T-GN-R-EW) args=("${COMMON[@]}" "${NEWTON[@]}" "${T_GN_R_EW[@]}") ;;
    *) echo "unknown optional arm $arm (T-GN or T-GN-R-EW)" >&2; exit 2 ;;
    esac
    job=$(submit_arm "$arm" "${args[@]}")
    echo "$arm: $job"
    ;;
*)
    echo "unknown phase $PHASE (gates, runs or optional)" >&2
    exit 2
    ;;
esac
