#!/bin/bash
# Submit the cylindrical optimiser comparison on Setonix.
#
# Usage (on a Setonix login node):
#     submit.sh gates       <code dir>
#     submit.sh timing      <code dir>
#     submit.sh runs        <code dir>
#     submit.sh hessian20   <code dir>
#     submit.sh rolA        <code dir>   (deferred, see below)
#     submit.sh radius      <code dir>   (conditional, see below)
#     submit.sh metric      <code dir>   (conditional, see below)
#     submit.sh gn-gate     <code dir>   (gate of the curvature job, see below)
#     submit.sh gn-curv     <code dir>   (Gauss-Newton curvature job, see below)
#     submit.sh gn-smoke    <code dir>   (only if P-GN uses the switch, see below)
#     RADIUS=<r> submit.sh gn <code dir> (P-GN, see below)
#
# The plan, the costs and the decision rule are in NOTES/team/cylindrical-newton/PLAN.md
# of the reunion worktree. Every phase waits for a person to read the results of the
# phase before; no Slurm dependency links two phases.
#
# Phase "gates": the checks of checks.py in the container, serial, on the 5-step
#   trajectory of 1.2e-5 that reunion also ran; a 16-rank smoke of each of the three
#   run arms on that trajectory (2 iterations); and the scaling measurement at 20
#   steps on 8, 16 and 32 ranks.
# Phase "timing": the 125-step reference state, and the 125-step timing on 8 and on
#   16 ranks. It gives the cost of one forward call and of one Hessian-vector
#   product, the memory per rank, and <h, H h> and <h, A h> at the first guess.
# Phase "hessian20": the November Hessian verification of
#   NOTES/CYLINDRICAL-HESSIAN-2026-11.md, repeated on the code after the merge
#   a0c9a6d1 (g-adopt #540 rewrites the weak normal-velocity terms and adds the
#   viscosity-derivative terms, so a Stokes solve differs by about 2e-5 from the code
#   the November campaign ran). Serial, 20 steps of 6e-6, smooth floor, smooth
#   direction with seed 1: the Taylor sweep (eps0 0.05, 7 levels) and the central
#   difference of the gradient (eps 1e-2, 1e-3, 1e-4), as in November.
# Phase "runs": the two runs of configuration P (production objective of
#   inverse.py), on one reference state, one rank count and one work budget (400):
#     P-ROL-L2    rol-lbfgs, L2 inner product: g-adopt today, as inverse.inverse()
#     P-Newton    tao-bnls-riesz, zero shift, inner product A, Krylov cap 100. The box
#                 records show the Newton gain starting at the first long Krylov
#                 solve (16 to 42 CG iterations), which a cap of 20 cuts off.
# Conditional phases, never submitted by "runs":
#   "rolA"    P-ROL-A, rol-lbfgs with the inner product A, which separates the inner
#             product from the algorithm; deferred until Newton shows a win;
#   "radius"  P-Newton at initial radius 0.87 (cap 100), only if the rule of PLAN.md section 5
#             gives "undecided";
#   "metric"  P-Newton with the metric shift (cap 100), only if the zero-shift run stalls
#             after a negative-curvature solve.
#
# The Gauss-Newton phases (NOTES/team/gauss-newton/PLAN.md sections 5.2 and 5.3):
#   "gn-gate"  one serial job on work, 48 GB, 2.5 h: checks.py gn at 2 steps with
#              --gn-parts G13 G3 (the three one-iteration TAO runs of G13 and the
#              zero-residual comparison of G3 with the cylinder's data leaves), on a
#              2-step reference state that the job writes. It must pass before
#              "gn-curv" is submitted; "gn-curv" refuses to submit otherwise.
#   "gn-curv"  one highmem job, 16 ranks, 120 GB, 9 h, of three steps
#              (gn_curv_steps.sh, set -e, so a failed step ends the job):
#              1. checks.py gn --gn-parts G1-G11 on the 5-step trajectory of the
#                 gates, serial: G1, G2, G4 to G7, G9 and G11 in the image (G3 and
#                 G13 ran in "gn-gate");
#              2. checks.py gn-curvature at 125 steps on 16 ranks: pHp/pAp and
#                 pH_GN p/pAp along p = A^-1 g at the first guess of P;
#              3. compare.py, one Gauss-Newton BNLS iteration at 125 steps, Krylov
#                 cap 20, radius 50.6 (the cylinder value of the box radius 12.8),
#                 control saved: CG iterations, stop reason, Ritz values, step norm,
#                 b.d and the STCG model value, line-search step, J before and after.
#              Stop condition for P-GN: step 1 passes and the actual fall of J is at
#              least 0.1 of the predicted fall (section 5.2 of the plan).
#   "gn-smoke" a 16-rank smoke at 5 steps, 2 iterations, gn-then-full with switch
#              iteration 1; only when Sia asks for the switch in P-GN.
#   "gn"       P-GN: tao-bnls-riesz, inner product A, zero shift, Krylov cap 100,
#              Gauss-Newton product, budget 400, into runs-GN-b<budget>-n<ranks>/.
#              Runs only after box test C and the curvature job pass, and after
#              the P-GN settings section of the plan is written and reviewed. The
#              settings come from the environment, with no default for the radius:
#                RADIUS     initial radius (0.79, 50.6 or the value of CYL-CURV.md)
#                HESSIAN    gn (default) or gn-then-full (only if Sia asks)
#                KRYLOV_EW  1 for the Eisenstat-Walker tolerance, else 1e-2
#
# The resources of the runs are environment variables with the values of the plan,
# so that the timing phase can change them without an edit:
#   RANKS (16), PARTITION (highmem), MEM (120G), WALL (96:00:00), BUDGET (400).
# The defaults come from the 125-step timing jobs of 2026-11-22: 16 ranks take 624 s
# per forward call and 4.5 GB per rank at peak (8 ranks: 799 s, 6.9 GB). highmem
# allows at most 7.9 GB per allocated CPU and gives each task a whole core (2 CPUs),
# so Slurm allocates max(2 x ranks, mem / 7.9 GB rounded up to whole cores) CPUs and
# bills each one: 16 ranks with 120 GB bill 32 per hour, and 8 ranks with 80 GB bill
# 32 as well (10 GB per task needs two cores per task), at 1.28 times the wall time.
# highmem runs at most two jobs of one user at a time; the third run waits until one
# of the first two ends.
set -euo pipefail
PHASE=$1
CODE=$2
BASE=/scratch/pawsey0821/sghelichkhani/hessian/cylindrical-newton
JOB=$CODE/demos/optimisation/cylindrical/setonix/job.slurm
STEPS=125
DT=1.2e-5
REF=$BASE/reference/s${STEPS}-dt${DT}
RANKS=${RANKS:-16}
PARTITION=${PARTITION:-highmem}
MEM=${MEM:-120G}
WALL=${WALL:-96:00:00}
BUDGET=${BUDGET:-400}
RUNS=$BASE/runs-P-b$BUDGET-n$RANKS
mkdir -p "$BASE"
cd "$BASE"

# Shared by every run of configuration P: the trajectory, the production objective
# (the default weights of compare.py are those of inverse.inverse()), the bounds
# [0, 1], the work budget, and the control saved after every iteration.
COMMON=(--rundir "$REF" --timesteps $STEPS --delta-t $DT --work-budget "$BUDGET"
        --max-it 1000 --save-control)

submit_run() {
    # $1 job name, then compare.py arguments. Prints the job id.
    local name=$1; shift
    mkdir -p "$RUNS"
    sbatch --parsable --job-name="$name" --partition="$PARTITION" --ntasks="$RANKS" \
        --mem="$MEM" --time="$WALL" "$JOB" "$CODE" "$RUNS" compare.py "${COMMON[@]}" "$@"
}

case $PHASE in
gates)
    G=$BASE/gates
    mkdir -p "$G"
    # The checks on the 5-step trajectory. The first writes the reference state.
    # objective and hvp hold three tapes at once: 36 and 44 GB peak on reunion.
    t5=(--rundir "$G/run-t5" --timesteps 5 --delta-t 1.2e-5)
    big=(--partition=highmem --mem=64G --time=04:00:00)
    j1=$(sbatch --parsable --job-name=cyl-objective "${big[@]}" "$JOB" "$CODE" "$G" \
        checks.py objective "${t5[@]}" --make-reference --json "$G/objective-t5.json")
    j2=$(sbatch --parsable --job-name=cyl-hvp "${big[@]}" --dependency=afterok:$j1 \
        "$JOB" "$CODE" "$G" checks.py hvp "${t5[@]}" --fd --json "$G/hvp-t5.json")
    j3=$(sbatch --parsable --job-name=cyl-spd "${big[@]}" --dependency=afterok:$j1 \
        "$JOB" "$CODE" "$G" checks.py spd "${t5[@]}" --json "$G/spd-t5.json")
    j4=$(sbatch --parsable --job-name=cyl-rolparams --time=00:10:00 --mem=8G "$JOB" \
        "$CODE" "$G" checks.py rol-params --json "$G/rol-params.json")
    echo "checks: objective $j1, hvp $j2, spd $j3, rol-params $j4"
    # The three run arms on 16 ranks, 2 iterations, on the same 5-step state. The
    # driver has run on 1 and 4 ranks before, never on 16.
    smoke=(--partition=highmem --ntasks=16 --mem=64G --time=02:00:00
           --dependency=afterok:$j1)
    s=$G/smoke-n16
    a=$(sbatch --parsable --job-name=cyl-smoke-rol-l2 "${smoke[@]}" "$JOB" "$CODE" "$s" \
        compare.py "${t5[@]}" --max-it 2 --save-control --optimiser rol-lbfgs \
        --riesz L2 --output "$s/rol-lbfgs-L2.json")
    b=$(sbatch --parsable --job-name=cyl-smoke-newton "${smoke[@]}" "$JOB" "$CODE" "$s" \
        compare.py "${t5[@]}" --max-it 2 --save-control --optimiser tao-bnls-riesz \
        --bnk-shift zero --riesz regularisation --output "$s/tao-zero.json")
    c=$(sbatch --parsable --job-name=cyl-smoke-rol-a "${smoke[@]}" "$JOB" "$CODE" "$s" \
        compare.py "${t5[@]}" --max-it 2 --save-control --optimiser rol-lbfgs \
        --riesz regularisation --output "$s/rol-lbfgs-A.json")
    echo "16-rank smokes: ROL-L2 $a, Newton $b, ROL-A $c (after $j1)"
    # Scaling at 20 steps: the reference in serial, then the timing on 8, 16 and 32
    # ranks. Memory per rank is set generously, so that no rank count is refused.
    r20=$(sbatch --parsable --job-name=cyl-ref20 --mem=32G "$JOB" "$CODE" "$G" \
        compare.py --make-reference-only --rundir "$G/run-t20" --timesteps 20 \
        --delta-t 1.2e-5)
    for n in 8 16 32; do
        j=$(sbatch --parsable --job-name=cyl-scale$n --dependency=afterok:$r20 \
            --partition=highmem --ntasks=$n --mem=$((n * 6))G --time=04:00:00 \
            "$JOB" "$CODE" "$G" checks.py timing --rundir "$G/run-t20" \
            --timesteps 20 --delta-t 1.2e-5 --json "$G/timing-t20-n$n.json")
        echo "scale $n: $j (after reference $r20)"
    done
    ;;
timing)
    mkdir -p "$REF"
    r=$(sbatch --parsable --job-name=cyl-ref125 --partition=highmem --ntasks=16 \
        --mem=120G --time=06:00:00 "$JOB" "$CODE" "$REF" compare.py \
        --make-reference-only --rundir "$REF" --timesteps $STEPS --delta-t $DT)
    for n in 8 16; do
        t=$(sbatch --parsable --job-name=cyl-timing125-n$n --dependency=afterok:$r \
            --partition=highmem --ntasks=$n --mem=120G --time=16:00:00 "$JOB" "$CODE" \
            "$REF" checks.py timing --rundir "$REF" --timesteps $STEPS --delta-t $DT \
            --json "$REF/timing-n$n.json")
        echo "timing $n ranks: $t (after reference $r)"
    done
    ;;
runs)
    j=$(submit_run cyl-P-rol-l2 --optimiser rol-lbfgs --riesz L2 \
        --output "$RUNS/rol-lbfgs-L2.json")
    echo "P-ROL-L2: $j"
    j=$(submit_run cyl-P-newton --optimiser tao-bnls-riesz --bnk-shift zero \
        --riesz regularisation --krylov-max-it 100 --output "$RUNS/tao-zero.json")
    echo "P-Newton: $j"
    ;;
rolA)
    j=$(submit_run cyl-P-rol-a --optimiser rol-lbfgs --riesz regularisation \
        --output "$RUNS/rol-lbfgs-A.json")
    echo "P-ROL-A:  $j"
    ;;
hessian20)
    H=$BASE/hessian20
    mkdir -p "$H"
    harness=../../../tests/hessian/cylindrical_hvp_taylor.py
    november=(Tobs --timesteps 20 --rundir "$H/run" --viscosity smooth
              --perturbation smooth --seed 1)
    serial=(--partition=work --mem=40G --time=08:00:00)
    t=$(sbatch --parsable --job-name=cyl-h20-taylor "${serial[@]}" "$JOB" "$CODE" "$H" \
        $harness "${november[@]}" --make-reference --and-measure --eps0 0.05 \
        --levels 7 --repeats 1 --json "$H/taylor.json")
    f=$(sbatch --parsable --job-name=cyl-h20-fd "${serial[@]}" --dependency=afterok:$t \
        "$JOB" "$CODE" "$H" $harness "${november[@]}" --fd-check --fd-scheme central \
        --fd-eps 1e-2 1e-3 1e-4)
    echo "Taylor $t, central difference $f (after $t)"
    ;;
radius)
    j=$(submit_run cyl-P-newton-r087 --optimiser tao-bnls-riesz --bnk-shift zero \
        --riesz regularisation --krylov-max-it 100 --initial-radius 0.87 --output "$RUNS/tao-zero-r0.87.json")
    echo "P-Newton, radius 0.87: $j"
    ;;
metric)
    j=$(submit_run cyl-P-newton-metric --optimiser tao-bnls-riesz --bnk-shift metric \
        --riesz regularisation --krylov-max-it 100 --output "$RUNS/tao-metric.json")
    echo "P-Newton, metric shift: $j"
    ;;
gn-gate)
    G=$BASE/gn-gate
    mkdir -p "$G"
    j=$(sbatch --parsable --job-name=cyl-gn-gate --partition=work --ntasks=1 \
        --mem=48G --time=02:30:00 "$JOB" "$CODE" "$G" checks.py gn \
        --rundir "$G/run-t2" --timesteps 2 --delta-t 1.2e-5 --make-reference \
        --gn-parts G13 G3 --json "$G/gn-gate-t2.json")
    echo "Gauss-Newton gate (G13, G3 at 2 steps): $j (output in $G)"
    ;;
gn-curv)
    # The gate must have passed: its JSON holds the top-level "pass": true.
    if ! grep -q '^ "pass": true' "$BASE/gn-gate/gn-gate-t2.json" 2>/dev/null; then
        echo "gn-gate has not passed ($BASE/gn-gate/gn-gate-t2.json); submit it first" >&2
        exit 1
    fi
    C=$BASE/gn-curvature
    mkdir -p "$C"
    # The script name ends in .sh, so job.slurm runs it with bash and passes the code
    # directory first; the script runs the three steps with their own srun counts.
    j=$(sbatch --parsable --job-name=cyl-gn-curv --partition=highmem --ntasks=16 \
        --mem=120G --time=09:00:00 "$JOB" "$CODE" "$C" gn_curv_steps.sh "$BASE")
    echo "Gauss-Newton curvature job: $j (output in $C)"
    ;;
gn-smoke)
    s=$BASE/gn-smoke-n16
    t5=(--rundir "$BASE/gates/run-t5" --timesteps 5 --delta-t 1.2e-5)
    j=$(sbatch --parsable --job-name=cyl-gn-smoke --partition=highmem --ntasks=16 \
        --mem=64G --time=01:30:00 "$JOB" "$CODE" "$s" compare.py "${t5[@]}" --max-it 2 \
        --optimiser tao-bnls-riesz --bnk-shift zero --riesz regularisation \
        --hessian gn-then-full --gn-switch-it 1 --output "$s/tao-gn-then-full.json")
    echo "16-rank switch smoke: $j"
    ;;
gn)
    : "${RADIUS:?set RADIUS to the initial radius of the P-GN settings section}"
    HESSIAN=${HESSIAN:-gn}
    KRYLOV=()
    if [ "${KRYLOV_EW:-0}" = 1 ]; then KRYLOV=(--krylov-ew); fi
    RUNS=$BASE/runs-GN-b$BUDGET-n$RANKS
    j=$(submit_run cyl-P-gn --optimiser tao-bnls-riesz --bnk-shift zero \
        --riesz regularisation --krylov-max-it 100 --hessian "$HESSIAN" \
        --initial-radius "$RADIUS" "${KRYLOV[@]}" --output "$RUNS/tao-gn.json")
    echo "P-GN ($HESSIAN, radius $RADIUS, ${KRYLOV[*]:-rtol 1e-2}): $j"
    ;;
*)
    echo "unknown phase $PHASE (gates, timing, runs, rolA, radius, metric, hessian20," \
         "gn-gate, gn-curv, gn-smoke or gn)" >&2
    exit 2
    ;;
esac
