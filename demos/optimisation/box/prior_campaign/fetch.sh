#!/bin/bash
# Copy the records of one campaign step from Setonix scratch to reunion.
#
# Usage (on reunion): fetch.sh <step>
#
# Source: /scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/<step>/ on Setonix.
# Target: ~/runs/prior-campaign/<step>/ on reunion.
# The copy holds the JSON records, the -tao_view text, the job logs and CODE_STATE.
# Checkpoint files (*.h5) stay on Setonix, because they are large and can be made
# again. The stream goes through phoenix, which holds the only key Setonix accepts.
set -euo pipefail
STEP=$1
BASE=/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign
mkdir -p "$HOME/runs/prior-campaign"
ssh phoenix "ssh setonix 'tar -c -C $BASE --exclude=*.h5 --exclude=code-* $STEP'" \
    | tar -x -C "$HOME/runs/prior-campaign"
find "$HOME/runs/prior-campaign/$STEP" -type f | sort
