#!/bin/bash --login
# The reference run of the box case: the trajectory that the inverse problem inverts.
# Run it once per mesh.
#
# Usage: sbatch forward.slurm <code dir> <output dir> <steps> <resolution>
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --job-name=box-forward
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
set -euo pipefail
# srun refuses to start when SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree
# ("cpus-per-task set by two different environment variables"). The login shell of a
# --login job carries a value of its own, so clear it and give srun the count.
unset SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=1

CODE_DIR=$1; OUT_DIR=$2; STEPS=$3; RESOLUTION=$4
# ---- environment of this job (the same block in every job of the campaign) ----

# env-hessian-v2.sh reads MY_GADOPT, so the code copy of this job must be set first. It
# loads the hessian-fix container, puts the UFL overlay before MY_GADOPT on PYTHONPATH,
# sets per-job kernel caches in /tmp and pins every threaded library to one thread.
export MY_GADOPT="$CODE_DIR"
source /software/projects/pawsey0821/sghelichkhani/env-hessian-v2.sh

# Remove the per-job kernel caches when the job ends, also after a failure.
trap 'rm -rf "$PYOP3_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT

echo "### job $SLURM_JOB_ID on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE_DIR"
cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
echo "### provenance"
python3 - <<'PY'
import importlib.metadata as md
import gadopt, firedrake, pyadjoint, ufl
from petsc4py import PETSc
print("gadopt    ", gadopt.__file__)
print("firedrake ", firedrake.__file__)
print("pyadjoint ", pyadjoint.__file__)
print("ufl       ", ufl.__file__)
print("PETSc     ", ".".join(map(str, PETSc.Sys.getVersion())), PETSc.Sys.getVersionInfo())
for name in ("firedrake", "petsc4py", "pyadjoint-ad"):
    try:
        print(f"{name:<10s}", md.version(name))
    except Exception as error:
        print(f"{name:<10s}", "unknown:", error)
PY
# ---- end of the environment block ----

mkdir -p "$OUT_DIR"; cd "$OUT_DIR"
echo "### forward.py --steps $STEPS --resolution $RESOLUTION"
srun -n 1 -c 1 python3 "$CODE_DIR/demos/optimisation/box/forward.py" \
    --steps "$STEPS" --resolution "$RESOLUTION" --output reference-state.h5
md5sum reference-state.h5
echo "### done"
