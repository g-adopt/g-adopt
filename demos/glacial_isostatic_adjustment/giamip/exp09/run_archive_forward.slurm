#!/bin/bash -l
# GIAMIP Exp09: copy a forward run and its ParaView series to Gadi, on the
# Setonix copy partition, through the Gadi data mover (project pawsey0821).
#
# Default source: the submit directory (the run directory of the forward
# run). Default destination: /g/data/xd2/sg8812/gia-mip/exp09/L0/forward-fine/.
# The job copies the surface file, the logs, the summary JSON, every job
# output (*.out) of the directory and the paraview/ series if it exists, and
# then prints the size and the file count on both sides.
#
# Submit after the ParaView job, from the run directory, for example:
#
#   cd /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/forward-fine
#   sbatch --dependency=afterok:<paraview job> --export=NONE \
#       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_archive_forward.slurm
#
# Variables (pass them with --export=NONE,NAME=value,...):
#   SRC    the run directory (default: the submit directory)
#   DEST   the directory on Gadi (default: see above)
#
#SBATCH --job-name=exp09-arch
#SBATCH --account=pawsey0821
#SBATCH --partition=copy
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=exp09-arch-%j.out

set -u
SRC=${SRC:-${SLURM_SUBMIT_DIR}}
H=sg8812@gadi-dm.nci.org.au
DEST=${DEST:-/g/data/xd2/sg8812/gia-mip/exp09/L0/forward-fine}

echo "job ${SLURM_JOB_ID}: ${SRC} -> ${H}:${DEST}"
date
ssh ${H} "mkdir -p ${DEST}" || exit 1
cd ${SRC} || exit 1
# The outputs of forward_exp09.py and the job outputs of the directory.
# The output of this job (exp09-arch-<job id>.out, see --output above) is
# not in the list: the job writes to it during the copy, so the copy on Gadi
# would be incomplete and rsync can report a changed file.
FILES=$(ls forward_exp09_surface.h5 forward_exp09.log params_forward_exp09.log \
    summary_forward.json *.out 2>/dev/null \
    | grep -v -x "exp09-arch-${SLURM_JOB_ID}.out")
rsync -a --stats ${FILES} ${H}:${DEST}/
status=$?
# The ParaView series of paraview_exp09.py, if it was made.
if [ -d paraview ]; then
    rsync -a --stats paraview ${H}:${DEST}/ || status=1
fi
echo "source:"
du -sb ${FILES} paraview 2>/dev/null | awk '{s += $1} END {print s " bytes"}'
find ${FILES} paraview -type f 2>/dev/null | wc -l
echo "destination:"
ssh ${H} "du -sb ${DEST}; find ${DEST} -type f | wc -l"
date
echo "exit code ${status}"
exit ${status}
