#!/bin/bash
# The memory of one node of a job, sampled at a fixed interval into a text
# file. One copy runs on every node of a run (`run_invert.pbs` starts it with
# `pbsdsh` when MEMORY_SAMPLER is set). It is a separate process outside MPI,
# so its file keeps the memory up to the last interval before a kill for out
# of memory, when the ranks themselves can write nothing.
#
# Usage: node_memory_sampler.sh DIRECTORY INTERVAL_S STOP_FILE
#
# It appends one line per interval to DIRECTORY/node_memory_<host>.txt:
#
#   <unix time> <ISO time> <host> avail_gb <MemAvailable> total_gb <MemTotal>
#   ranks <number of python processes> rss_sum_gb <sum of their RSS>
#   top <rank>:<pid>:<RSS GB> ... (the 5 largest by RSS)
#
# `<rank>` is OMPI_COMM_WORLD_RANK from the environment of the process (? if
# it cannot be read). The python processes are the processes of this user
# whose command name starts with "python": the MPI ranks of the run. All
# sizes are GB of 1e9 bytes (the kernel reports KiB). The
# sampler stops when STOP_FILE exists, or after 48 h.
set -u
DIR=${1:?directory}
INTERVAL=${2:-20}
STOP=${3:?stop file}
HOST=$(hostname -s)
OUT=${DIR}/node_memory_${HOST}.txt
END=$(( $(date +%s) + 48 * 3600 ))
echo "# node memory sampler on ${HOST}, every ${INTERVAL} s; fields: unix_time iso_time host avail_gb total_gb ranks rss_sum_gb top rank:pid:rss_gb" >> "${OUT}"
while [ ! -e "${STOP}" ] && [ "$(date +%s)" -lt "${END}" ]; do
    # MemAvailable and MemTotal in kB.
    read -r AVAIL TOTAL < <(awk '/^MemAvailable:/ {a=$2} /^MemTotal:/ {t=$2} END {print a, t}' /proc/meminfo)
    # pid and RSS (kB) of every python process of this user, largest first.
    PROCS=$(ps -u "$(id -u)" -o pid=,rss=,comm= | awk '$3 ~ /^python/ {print $1, $2}' | sort -k2,2nr)
    NRANKS=$(printf '%s\n' "${PROCS}" | grep -c . || true)
    SUM=$(printf '%s\n' "${PROCS}" | awk '{s += $2} END {printf "%.2f", s * 1024 / 1e9}')
    TOP=""
    while read -r PID RSS; do
        [ -z "${PID}" ] && continue
        RANK=$(tr '\0' '\n' < /proc/${PID}/environ 2>/dev/null | awk -F= '$1 == "OMPI_COMM_WORLD_RANK" {print $2}')
        TOP="${TOP} ${RANK:-?}:${PID}:$(awk -v r="${RSS}" 'BEGIN {printf "%.2f", r * 1024 / 1e9}')"
    done < <(printf '%s\n' "${PROCS}" | head -5)
    printf '%s %s %s avail_gb %.1f total_gb %.1f ranks %s rss_sum_gb %s top%s\n' \
        "$(date +%s)" "$(date +%Y-%m-%dT%H:%M:%S)" "${HOST}" \
        "$(awk -v a="${AVAIL}" 'BEGIN {print a * 1024 / 1e9}')" \
        "$(awk -v t="${TOTAL}" 'BEGIN {print t * 1024 / 1e9}')" \
        "${NRANKS}" "${SUM}" "${TOP}" >> "${OUT}"
    sleep "${INTERVAL}"
done
echo "# sampler on ${HOST} stopped at $(date +%Y-%m-%dT%H:%M:%S)" >> "${OUT}"
