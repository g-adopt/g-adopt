#!/bin/bash
# Copy the Gauss-Newton box records from Gadi to reunion.
#
# Usage (on reunion): fetch.sh [<directory> ...]
#
# Default: gates, smoke and the six run directories. Source:
# /scratch/nm03/sg8812/hessian/gauss-newton/<directory>/ on Gadi. Target:
# ~/runs/prior-campaign/GN/<directory>/. The copy holds the JSON records, the
# -tao_view text, the logs, args, CODE_STATE, COMMITS.txt and NODE; checkpoint files
# (*.h5) stay on Gadi. A directory that does not exist on Gadi is skipped.
set -euo pipefail
BASE=/scratch/nm03/sg8812/hessian/gauss-newton
TARGET=$HOME/runs/prior-campaign/GN
DIRS=("$@")
[ ${#DIRS[@]} -gt 0 ] || DIRS=(gates smoke T-ROL T-N T-N-R T-GN-R T-GN T-GN-R-EW)
mkdir -p "$TARGET"
present=$(ssh gadi "source /etc/profile && cd $BASE && ls -d ${DIRS[*]} 2>/dev/null" || true)
[ -n "$present" ] || { echo "none of ${DIRS[*]} exists in $BASE" >&2; exit 1; }
ssh gadi "source /etc/profile && tar -c -C $BASE --exclude='*.h5' $(echo $present)" \
    | tar -x -C "$TARGET"
for d in $present; do find "$TARGET/$d" -type f | sort; done
