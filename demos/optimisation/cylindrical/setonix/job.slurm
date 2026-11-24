#!/bin/bash --login
# One job of the cylindrical comparison: a script of demos/optimisation/cylindrical/
# under srun, on as many ranks as the job has tasks.
#
# Usage: sbatch [--partition --ntasks --mem --time --job-name] job.slurm \
#            <code dir> <work dir> <script> [script arguments ...]
#
#   <code dir>  a code copy made by sync_code.sh
#   <work dir>  where the job runs; relative paths in the arguments start here
#   <script>    compare.py or checks.py, run under srun on all tasks; or a shell
#               script of setonix/ (a name that ends in .sh), run with bash after the
#               environment is set, with the code directory as its first argument,
#               for a job of several steps with their own srun counts
#
# The resources are given on the sbatch command line by submit.sh, because they
# differ between the gates and the runs. The defaults below are those of a gate.
#SBATCH --account=pawsey0821
#SBATCH --partition=work
#SBATCH --job-name=cyl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
set -euo pipefail
# srun refuses to start when SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree
# ("cpus-per-task set by two different environment variables"). The login shell of a
# --login job carries a value of its own, so clear it and give srun the count.
unset SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=1

CODE_DIR=$1; WORK_DIR=$2; SCRIPT=$3; shift 3

# ---- environment of this job ----
# env-hessian-v3.sh reads MY_GADOPT, so the code copy of this job must be set first. It
# mounts private copies of the hessian-fix container of 2026-11-23 (firedrake a11e4be9,
# pyadjoint 957f265, UFL ccf1ba61, pyop3, PETSc 3.26.0) and of its PETSc overlay, so
# that a change to the shared group image cannot reach a running job. It puts only
# MY_GADOPT on PYTHONPATH (the image needs no UFL overlay), sets per-job kernel caches
# in /tmp and pins every threaded library to one thread.
export MY_GADOPT="$CODE_DIR"
source /software/projects/pawsey0821/sghelichkhani/env-hessian-v3.sh
# The per-job caches live in /tmp of the first node; a single-node job has no other.
trap 'rm -rf "$PYOP3_CACHE_DIR" "$FIREDRAKE_TSFC_KERNEL_CACHE_DIR" "$XDG_CACHE_HOME"' EXIT

echo "### job $SLURM_JOB_ID on $(hostname), $SLURM_NTASKS ranks, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE_DIR"
# The images that this job mounts, so that the log names the exact container.
echo "### container $CONTAINER_PATH overlays $CONTAINER_OVERLAY_PATH"
cat "$CODE_DIR/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE_DIR)"
echo "### provenance"
python3 - <<'PY'
import importlib.metadata as md
import gadopt, firedrake, pyadjoint, ufl, ROL
from petsc4py import PETSc
print("gadopt    ", gadopt.__file__)
print("firedrake ", firedrake.__file__)
print("pyadjoint ", pyadjoint.__file__)
print("ufl       ", ufl.__file__)
print("ROL       ", ROL.__file__)
print("PETSc     ", ".".join(map(str, PETSc.Sys.getVersion())))
for name in ("firedrake", "petsc4py", "pyadjoint-ad"):
    try:
        print(f"{name:<10s}", md.version(name))
    except Exception as error:
        print(f"{name:<10s}", "unknown:", error)
PY
# ---- end of the environment block ----

mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
cp "$CODE_DIR/CODE_STATE" . 2>/dev/null || true
echo "### $SCRIPT $*"
case $SCRIPT in
*.sh)
    # A job of several steps; the script calls srun itself.
    bash "$CODE_DIR/demos/optimisation/cylindrical/setonix/$SCRIPT" "$CODE_DIR" "$@"
    ;;
*)
    # The demo directory first on the path, so that its modules import by name.
    srun -n "$SLURM_NTASKS" -c 1 python3 "$CODE_DIR/demos/optimisation/cylindrical/$SCRIPT" "$@"
    ;;
esac
echo "### done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# The peak memory of every step of this job, for the plan's memory column.
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,NTasks -P || true
