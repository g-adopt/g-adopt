# The environment block of every Gauss-Newton box job on Gadi. Each PBS script of this
# directory sources this file after it sets CODE (the code copy of the job).
#
# ENV_FILE is the Gadi environment of step 2G (NOTES/team/gauss-newton/GADI-ENV.md):
# the private copies of the image firedrake JHopeCollins_nlvs-hessian-fix of
# 2026-11-23 (firedrake a11e4be95, pyadjoint 957f265, UFL ccf1ba61, PETSc 3.26.0,
# pyroltrilinos 0.5.7), the same pip freeze as env-hessian-v3.sh on Setonix. It reads
# MY_GADOPT, puts only the code copy on PYTHONPATH, puts the kernel caches in
# $PBS_JOBFS and pins every threaded library to one thread. Every job must request
# "-l storage=gdata/nm03+scratch/nm03+gdata/fp50" (the RPATH of petsc4py names
# /g/data/fp50/apps/petsc/3.26.0, a link that the private PETSc overlay resolves).
# >>> environment file: change this one line if step 2G moves it <<<
ENV_FILE=${ENV_FILE:-/g/data/nm03/sg8812/firedrake-hessian/env-hessian.sh}

# The root of every run directory of this campaign on Gadi.
GN_BASE=/scratch/nm03/sg8812/hessian/gauss-newton
# The S0 reference state, copied from reunion (md5 9b7927bcd3b4b61bc235d74a2b13ca1a).
GN_REFERENCE=$GN_BASE/reference/r64-s80/reference-state.h5

if [ -z "${CODE:-}" ] || [ ! -d "$CODE/demos/optimisation/box" ]; then
    echo "job_env.sh: CODE must be a code copy made by sync_code.sh (now '${CODE:-}')" >&2
    exit 2
fi
export MY_GADOPT="$CODE"
# shellcheck disable=SC1090
source "$ENV_FILE"
BOX=$CODE/demos/optimisation/box
JOBS=$BOX/prior_campaign

echo "### job ${PBS_JOBID:-interactive} on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### code $CODE"
cat "$CODE/CODE_STATE" 2>/dev/null || echo "(no CODE_STATE in $CODE)"
echo "### environment $ENV_FILE"
cat "$(dirname "$ENV_FILE")/COMMITS.txt" 2>/dev/null || echo "(no COMMITS.txt)"
echo "### CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
python3 -c "import gadopt; print('gadopt    ', gadopt.__file__)"

# A copy of the provenance in the run directory: CODE_STATE, COMMITS.txt and the node.
record_provenance() {
    # $1: the run directory.
    mkdir -p "$1"
    cp "$CODE/CODE_STATE" "$1/" 2>/dev/null || true
    cp "$(dirname "$ENV_FILE")/COMMITS.txt" "$1/" 2>/dev/null || true
    { hostname; grep -m1 'model name' /proc/cpuinfo; } > "$1/NODE"
}
