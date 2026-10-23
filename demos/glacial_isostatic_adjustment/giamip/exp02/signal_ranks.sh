#!/bin/bash
# Send one signal to every MPI rank of this user on this node. The watchdog
# of `run_invert.pbs` runs one copy per node with `pbsdsh`.
#
# Usage: signal_ranks.sh SIGNAL [FILE]   (SIGNAL for example USR1 or TERM)
#
# A rank is a process of this user whose command name starts with "python"
# and whose environment holds OMPI_COMM_WORLD_RANK, which Open MPI sets for
# the ranks it starts. The test on the environment matters: the job also
# runs bash wrapper scripts named python3 (the container launchers of the
# Firedrake module), and a wrapper that gets SIGUSR1 ends at once, because
# bash does not catch that signal. The ranks register a stack dump for
# SIGUSR1 (`faulthandler.register` in invert_exp02.py) and continue after it.
#
# One line per signalled rank (host, pid, rank, signal) is appended to FILE,
# or goes to stdout without FILE. pbsdsh sends the stdout of a remote task
# to the job output, where these lines would mix with the stack dumps.
set -u
SIG=${1:?signal name, for example USR1}
RECORD=${2:-/dev/stdout}
HOST=$(hostname -s)
for PID in $(ps -u "$(id -u)" -o pid=,comm= | awk '$2 ~ /^python/ {print $1}'); do
    # The stderr redirect comes first, so that it also covers the input
    # redirect: a rank that exits during the loop has an unreadable environ.
    RANK=$(tr '\0' '\n' 2>/dev/null < /proc/${PID}/environ | awk -F= '$1 == "OMPI_COMM_WORLD_RANK" {print $2}')
    if [ -n "${RANK}" ]; then
        kill -s "${SIG}" "${PID}" 2>/dev/null && echo "signal_ranks ${HOST} pid ${PID} rank ${RANK} SIG${SIG}" >> "${RECORD}"
    fi
done
