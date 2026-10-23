#!/bin/bash
# The use of the local job disk ($PBS_JOBFS) of one node of a job, sampled
# at a fixed interval into a text file. One copy runs on every node of a run
# (`run_invert.pbs` starts it with `pbsdsh`). The tape checkpoints of the
# inversion are the only large files there (decision R11 of the r2
# production), so the size of the directory is the size of the tape of that
# node. The PBS footer of the job gives one number for the whole job only;
# this file gives the size per node and its growth in time, from which the
# tape of a full march can be projected (`run_invert.pbs` prints the
# projection at the end of the job).
#
# Usage: node_jobfs_sampler.sh DIRECTORY INTERVAL_S STOP_FILE JOBFS
#
# It appends one line per interval to DIRECTORY/node_jobfs_<host>.txt:
#
#   <unix time> <ISO time> <host> used_gb <size of JOBFS> free_gb <free space>
#
# `used_gb` is the apparent size of all files below JOBFS (du -s -B1), in GB
# of 1e9 bytes. `free_gb` is the free space of the file system that holds
# JOBFS (df), which on an exclusive node is the room left for the tape. The
# sampler stops when STOP_FILE exists, or after 48 h. The directory JOBFS is
# the same path on every node of a job, so the caller passes its own
# $PBS_JOBFS.
set -u
DIR=${1:?directory}
INTERVAL=${2:-60}
STOP=${3:?stop file}
JOBFS=${4:?the jobfs directory}
HOST=$(hostname -s)
OUT=${DIR}/node_jobfs_${HOST}.txt
END=$(( $(date +%s) + 48 * 3600 ))
echo "# node jobfs sampler on ${HOST} for ${JOBFS}, every ${INTERVAL} s; fields: unix_time iso_time host used_gb free_gb" >> "${OUT}"
while [ ! -e "${STOP}" ] && [ "$(date +%s)" -lt "${END}" ]; do
    # The bytes below JOBFS. A file that the tape writer removes during the
    # walk only makes du warn; the warning goes to /dev/null and the sum
    # covers the files that exist.
    USED=$(du -s -B1 "${JOBFS}" 2>/dev/null | awk '{print $1}')
    # The free bytes of the file system of JOBFS (df: 1-byte blocks).
    FREE=$(df -B1 --output=avail "${JOBFS}" 2>/dev/null | tail -1 | tr -d ' ')
    printf '%s %s %s used_gb %.2f free_gb %.1f\n' \
        "$(date +%s)" "$(date +%Y-%m-%dT%H:%M:%S)" "${HOST}" \
        "$(awk -v u="${USED:-0}" 'BEGIN {print u / 1e9}')" \
        "$(awk -v f="${FREE:-0}" 'BEGIN {print f / 1e9}')" >> "${OUT}"
    sleep "${INTERVAL}"
done
echo "# jobfs sampler on ${HOST} stopped at $(date +%Y-%m-%dT%H:%M:%S)" >> "${OUT}"
