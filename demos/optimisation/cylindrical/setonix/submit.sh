#!/bin/bash
# Submit the cylindrical optimiser comparison on Setonix, in three phases.
#
# Usage (on a Setonix login node):
#     submit.sh gates  <code dir>
#     submit.sh timing <code dir>
#     submit.sh runs   <code dir>
#
# The plan, the costs and the decision rule are in NOTES/team/cylindrical-newton/PLAN.md
# of the reunion worktree. Every phase waits for a person to read the results of the
# phase before; no Slurm dependency links two phases.
#
# Phase "gates": the short checks of checks.py in the container, serial, on the
#   5-step trajectory of 1.2e-5 that reunion also ran, and the scaling measurement
#   at 20 steps on 8, 16 and 32 ranks. A few hours and a few tens of SU in all.
# Phase "timing": the 125-step reference state and the 125-step timing on the rank
#   count of the runs. It gives the cost of one forward call, the cost of one
#   Hessian-vector product, the memory per rank, and the curvature <h, H h> of the
#   full objective at the first guess.
# Phase "runs": the comparison itself, every run on the same reference state, the
#   same rank count and the same work budget.
#
# The resources of the runs are environment variables with the values of the plan,
# so that the timing phase can change them without an edit:
#   RANKS (16), PARTITION (highmem), MEM (100G), WALL (96:00:00), BUDGET (200),
#   CONFIGS ("N P").
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
MEM=${MEM:-100G}
WALL=${WALL:-96:00:00}
BUDGET=${BUDGET:-200}
mkdir -p "$BASE"
cd "$BASE"

case $PHASE in
gates)
    G=$BASE/gates
    mkdir -p "$G"
    # The three checks on the 5-step trajectory. The first writes the reference.
    common=(--rundir "$G/run-t5" --timesteps 5 --delta-t 1.2e-5)
    # objective and hvp hold three tapes at once: 36 GB peak on reunion at 5 steps.
    big=(--partition=highmem --mem=64G --time=04:00:00)
    j1=$(sbatch --parsable --job-name=cyl-objective "${big[@]}" "$JOB" "$CODE" "$G" checks.py \
        objective "${common[@]}" --make-reference --json "$G/objective-t5.json")
    j2=$(sbatch --parsable --job-name=cyl-hvp "${big[@]}" --dependency=afterok:$j1 "$JOB" "$CODE" \
        "$G" checks.py hvp "${common[@]}" --fd --json "$G/hvp-t5.json")
    j3=$(sbatch --parsable --job-name=cyl-spd "${big[@]}" --dependency=afterok:$j1 "$JOB" "$CODE" \
        "$G" checks.py spd "${common[@]}" --json "$G/spd-t5.json")
    j4=$(sbatch --parsable --job-name=cyl-rolparams --time=00:10:00 "$JOB" "$CODE" \
        "$G" checks.py rol-params --json "$G/rol-params.json")
    echo "checks: objective $j1, hvp $j2, spd $j3, rol-params $j4"
    # Scaling at 20 steps: the reference in serial, then the timing on 8, 16 and 32
    # ranks. Memory per rank is set generously, so that no rank count is refused.
    r20=$(sbatch --parsable --job-name=cyl-ref20 "$JOB" "$CODE" "$G" compare.py \
        --make-reference-only --rundir "$G/run-t20" --timesteps 20 --delta-t 1.2e-5)
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
    r=$(sbatch --parsable --job-name=cyl-ref125 --partition="$PARTITION" \
        --ntasks="$RANKS" --mem="$MEM" --time=06:00:00 "$JOB" "$CODE" "$REF" \
        compare.py --make-reference-only --rundir "$REF" --timesteps $STEPS \
        --delta-t $DT)
    t=$(sbatch --parsable --job-name=cyl-timing125 --dependency=afterok:$r \
        --partition="$PARTITION" --ntasks="$RANKS" --mem="$MEM" --time=12:00:00 \
        "$JOB" "$CODE" "$REF" checks.py timing --rundir "$REF" --timesteps $STEPS \
        --delta-t $DT --json "$REF/timing-n$RANKS.json")
    echo "reference $r, timing $t ($RANKS ranks, $PARTITION, $MEM)"
    ;;
runs)
    # CONFIGS: "N" (misfit-only objective, the metric of inverse.py as inner
    # product; negative curvature along the smooth direction), "P" (the production
    # objective of inverse.py, regularisation included), or "N P".
    for config in ${CONFIGS:-N P}; do
        case $config in
        N) objective=(--alpha-d 0 --alpha-s 0) ;;
        P) objective=() ;;
        *) echo "unknown configuration $config (N or P)" >&2; exit 2 ;;
        esac
        R=$BASE/runs-$config-b$BUDGET-n$RANKS
        mkdir -p "$R"
        # Shared by every run of a configuration: the trajectory, the objective, the
        # metric of inverse.py as the inner product (unless --riesz L2 is added
        # below), the bounds [0, 1] and the work budget.
        common=(--rundir "$REF" --timesteps $STEPS --delta-t $DT --work-budget "$BUDGET"
                --max-it 1000 --save-control "${objective[@]}")
        submit_run() {
            # $1 job name, then compare.py arguments.
            local name=$1; shift
            sbatch --parsable --job-name="$name" --partition="$PARTITION" \
                --ntasks="$RANKS" --mem="$MEM" --time="$WALL" "$JOB" "$CODE" "$R" \
                compare.py "${common[@]}" "$@"
        }
        j=$(submit_run cyl$config-rol --optimiser rol-lbfgs --riesz regularisation \
            --output "$R/rol-lbfgs.json")
        echo "$config rol-lbfgs, metric:          $j"
        j=$(submit_run cyl$config-zero --optimiser tao-bnls-riesz --bnk-shift zero \
            --riesz regularisation --output "$R/tao-zero.json")
        echo "$config tao-bnls-riesz, zero shift: $j"
        j=$(submit_run cyl$config-metric --optimiser tao-bnls-riesz --bnk-shift metric \
            --riesz regularisation --output "$R/tao-metric.json")
        echo "$config tao-bnls-riesz, metric:     $j"
        if [ "$config" = P ]; then
            # g-adopt today, exactly: ROL under the L2 inner product, production J.
            j=$(submit_run cyl$config-rol-l2 --optimiser rol-lbfgs --riesz L2 \
                --output "$R/rol-lbfgs-L2.json")
            echo "$config rol-lbfgs, L2:              $j"
        fi
    done
    ;;
*)
    echo "unknown phase $PHASE (gates, timing or runs)" >&2
    exit 2
    ;;
esac
