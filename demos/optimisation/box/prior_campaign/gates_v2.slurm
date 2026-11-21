#!/bin/bash --login
# The short correctness gates of the container of 2026-11-20 (env-hessian-v2.sh), on
# the tiny box case and the Hessian harness. Each part prints numbers that the
# earlier records hold, so that the job log can be compared with them line by line.
#
# Usage: sbatch gates_v2.slurm <code dir> <run dir>
#
# Parts, in order:
#   1. the tiny reference state (forward.py --steps 10 --resolution 16);
#   2. check_wrapper.py in the three wall settings (free/none, pinned/none,
#      pinned/dirichlet): the metric-shifted Hessian against an assembled reference;
#   3. the Hessian Taylor harness tests/hessian/cartesian_hvp_taylor.py with the
#      settings of gate 2 of NOTES/SETONIX-2026-11-18.md (Tobs, smooth-stable
#      direction, seed 42, 5 time steps, 6 levels);
#   4. check_box_hessian.py: the Taylor test and the central difference of the
#      Hessian-vector product on the tape of problem.py itself, which the harness
#      does not use;
#   5. gate 1 of the same notes: rol-lbfgs under L2 and tao-bnls-riesz under the
#      regularisation metric on the tiny case, window 5, 4 iterations.
# The job goes on after a failed part, so that one log holds every result, and exits
# with the number of failed parts.
#SBATCH --account=pawsey0821
#SBATCH --partition=debug
#SBATCH --job-name=v2-gates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
set -uo pipefail
# srun refuses to start when SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree
# ("cpus-per-task set by two different environment variables"). The login shell of a
# --login job carries a value of its own, so clear it and give srun the count.
unset SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=1

CODE_DIR=$1; RUN_DIR=$2
# ---- environment of this job (the same block in every job of the campaign) ----
export MY_GADOPT="$CODE_DIR"
source /software/projects/pawsey0821/sghelichkhani/env-hessian-v2.sh
trap 'rm -rf "$PYOP3_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT
echo "### job $SLURM_JOB_ID on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE_DIR"
cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
python3 -c "import gadopt, ufl; print('gadopt    ', gadopt.__file__); print('ufl       ', ufl.__file__)"
# ---- end of the environment block ----

BOX=$CODE_DIR/demos/optimisation/box
JOBS=$BOX/prior_campaign
FAILED=0
part() {
    # part <name> <command ...>: run one part, print its exit status, count failures.
    local name=$1; shift
    echo "### part $name: $*"
    if "$@"; then echo "### part $name: PASS (exit 0)"
    else echo "### part $name: FAIL (exit $?)"; FAILED=$((FAILED + 1)); fi
}
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

part forward srun -n 1 -c 1 python3 "$BOX/forward.py" --steps 10 --resolution 16 \
    --output tiny-state.h5
md5sum tiny-state.h5

part wrapper-free srun -n 1 -c 1 python3 "$JOBS/check_wrapper.py" tiny-state.h5 free none
part wrapper-pinned srun -n 1 -c 1 python3 "$JOBS/check_wrapper.py" tiny-state.h5 pinned none
part wrapper-dirichlet srun -n 1 -c 1 python3 "$JOBS/check_wrapper.py" tiny-state.h5 \
    pinned dirichlet

# The harness reads tests/adjoint/adjoint-demo-checkpoint-state.h5 of the code copy.
part harness srun -n 1 -c 1 python3 "$CODE_DIR/tests/hessian/cartesian_hvp_taylor.py" Tobs \
    --perturbation smooth-stable --seed 42 --timesteps 5 --levels 6

part box-hessian srun -n 1 -c 1 python3 "$JOBS/check_box_hessian.py" tiny-state.h5

# Gate 1: the same commands as /scratch/pawsey0821/sghelichkhani/hessian/gate.slurm,
# with the tiny reference state of part 1.
part gate1-rol srun -n 1 -c 1 python3 "$BOX/inverse.py" --checkpoint tiny-state.h5 \
    --optimiser rol-lbfgs --window 5 --max-it 4 --alpha-d 0 --alpha-s 0 \
    --initial-guess observed --output rol-lbfgs_L2.json
part gate1-tao srun -n 1 -c 1 python3 "$BOX/inverse.py" --checkpoint tiny-state.h5 \
    --optimiser tao-bnls-riesz --riesz regularisation --window 5 --max-it 4 \
    --alpha-d 0 --alpha-s 0 --metric-alpha-d 1e-3 --metric-alpha-s 1e-3 \
    --initial-guess observed --initial-radius 1.0 --output tao-bnls-riesz_regularisation.json

echo "### last row of each gate-1 record"
python3 - <<'PY'
import json
keep = ("J", "initial_misfit", "control_error", "gradient_norm", "forward", "adjoint",
        "hessian", "metric_c_d", "metric_c_s")
for path in ("rol-lbfgs_L2.json", "tao-bnls-riesz_regularisation.json"):
    try:
        record = json.load(open(path))
    except OSError as error:
        print(path, "missing:", error)
        continue
    last = record["iterations"][-1]
    print(path, {k: last[k] for k in keep if k in last})
    print(path, "metric", {k: record.get(k) for k in ("metric_c_d", "metric_c_s")})
PY
echo "### done, $FAILED failed parts"
exit $FAILED
