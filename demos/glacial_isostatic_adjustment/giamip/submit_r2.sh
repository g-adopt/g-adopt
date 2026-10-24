#!/bin/bash
# Submit the jobs of the GIAMIP production on the mesh giamip_8re_r2 on Gadi
# for one experiment and one node count. Run it on a Gadi login node, in a
# login shell (qsub needs the PBS environment):
#
#   bash submit_r2.sh EXP NODES STAGE
#
#   EXP    01 .. 12
#   NODES  the node count of every solver job of the experiment (104 ranks
#          per node). The input file holds a mesh partitioned for that rank
#          count, so all stages of one experiment use the same NODES. The r2
#          decision (R6): 6 nodes for the Maxwell experiments (Exp01-08,
#          Exp11), 12 for Exp10, 18 for Exp09 and Exp12.
#   STAGE  one of
#     ready     prepare -> transfer -> 3-step test, each after the one
#               before (afterok)
#     transfer  transfer -> 3-step test, for an existing prepare output
#     test      the 3-step test only, for a transferred input file
#     invert    the production inversion from the transferred bed
#     restart   the next job of the production inversion after a stop at
#               the wall time (RESTART=state.h5 in the same directory)
#     forward   forward -> ParaView -> images (two halves) -> archive
#
# The run directory of the experiment is
# /scratch/xd2/$USER/giamip/expXX/8re_r2-<NODES>nodes/ with prepare/,
# transfer/, invert-3steps/ and invert/. The forward run, its ParaView
# series and its images go to /g/data/xd2/sg8812/gia-mip/expXX/r2/forward-<schedule>/,
# and the archive job copies the records and the final state of the run
# directory to /g/data/xd2/sg8812/gia-mip/expXX/r2/. Every submitted job id
# is appended to jobs_r2.txt in the run directory.
#
# Settings per experiment:
#   Exp09, Exp10, Exp12   schedule C (prepare --schedule windows, forward
#                         SCHEDULE=windows); the L0 bed is
#                         expXX/L0/invert-c/state.h5 on gdata
#   the other nine        the protocol steps (prepare --step_years 100,
#                         forward SCHEDULE=protocol); the L0 bed is
#                         expXX/L0/invert-protocol/state.h5 on gdata
# The transfer uses START=change for all twelve: the source is the final L0
# bed of the same experiment on another mesh (transfer_bed_exp01.py, WHAT
# IS WRITTEN).
#
# Resources of every solver job: ncpus = 104 NODES, mem = 500 NODES GB,
# jobfs = 400 NODES GB (the tape goes to $PBS_JOBFS, decision R11). Wall
# time of the inversion and the forward run: 48 h up to 10 nodes, 24 h
# above (the limit of normalsr above 10 nodes).
#
# The solver options of the inversion (decision of 2026-10-24): default
# solver options, --gia_rebuild_every 100 --log_memory
# --free_taping_solver, the watchdog at 30 min, the node memory sampler
# every 20 s. The stop rule is the default of invert_expXX.py: RMS misfit
# below 1 mm or 10 ROL iterations.
#
# Variables of this script (in its environment):
#   DRY_RUN=1          print the qsub lines, submit nothing, create nothing
#   FORCE=1            submit even if the stage directory holds the outputs
#                      of an earlier submission
#   SOURCE=<file>      another L0 state file for the transfer
#   RANKS_PER_NODE=<k> MPI ranks per node of every solver job (default
#                      104, one per core). With k < 104 the jobs run
#                      NP = k NODES ranks on full nodes (the same ncpus,
#                      mem and jobfs per node), spread evenly over the
#                      cores of each node (the rank file of run_invert.pbs
#                      and run_forward.pbs). The run directory is then
#                      8re_r2-<NODES>nodes-<NP>ranks. Sia's decision of
#                      2026-10-24: the 624-rank partition of the 6-node
#                      input file on 16 nodes with 39 ranks per node, for
#                      more node disk per rank for the tape, for example
#                      RANKS_PER_NODE=39 PREP=<6-node prepare dir>
#                      bash submit_r2.sh 01 16 invert
#   PREP=<dir>         the prepare directory (default <run dir>/prepare); the
#                      transfer, the test and the inversion read
#                      <PREP>/inputs_expXX.h5
#   FORWARD_RESTART=1  forward stage: continue the forward run from its last
#                      checkpoint (RESTART=1), then ParaView, images and
#                      archive as before
#   ROOT, ARCHIVE      the roots of the run directories
#                      (/scratch/xd2/$USER/giamip) and of the archive
#                      (/g/data/xd2/sg8812/gia-mip)
set -euo pipefail

EXP=${1:?usage: submit_r2.sh EXP NODES STAGE}
NODES=${2:?usage: submit_r2.sh EXP NODES STAGE}
STAGE=${3:?usage: submit_r2.sh EXP NODES STAGE}
case ${EXP} in
    01|02|03|04|05|06|07|08|09|10|11|12) ;;
    *) echo "error: EXP=${EXP}; use 01 .. 12" >&2; exit 1 ;;
esac
if ! [[ ${NODES} =~ ^[1-9][0-9]*$ ]]; then
    echo "error: NODES=${NODES} is not a positive integer" >&2
    exit 1
fi

# The scripts of the experiment: the directory of this file, in the clone
# of the branch (MY_GADOPT of the environment file names the same clone).
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp${EXP}
[ -d "${G}" ] || { echo "error: no directory ${G}" >&2; exit 1; }

# The ranks per node and the ranks of every solver job. The partition of
# the input file belongs to the rank count NP, not to the node count, so a
# run with fewer ranks per node gets its own run directory named after both.
RANKS_PER_NODE=${RANKS_PER_NODE:-104}
if ! [[ ${RANKS_PER_NODE} =~ ^[1-9][0-9]*$ ]] || [ ${RANKS_PER_NODE} -gt 104 ]; then
    echo "error: RANKS_PER_NODE=${RANKS_PER_NODE}; use 1 .. 104" >&2
    exit 1
fi
NP=$(( RANKS_PER_NODE * NODES ))

ROOT=${ROOT:-/scratch/xd2/${USER}/giamip}
ARCHIVE=${ARCHIVE:-/g/data/xd2/sg8812/gia-mip}
if [ ${RANKS_PER_NODE} -eq 104 ]; then
    R=${ROOT}/exp${EXP}/8re_r2-${NODES}nodes
    # One rank per core: the jobs take NP from the cores of the job.
    NPV=""
else
    R=${ROOT}/exp${EXP}/8re_r2-${NODES}nodes-${NP}ranks
    # The first entry of every -v list of a solver job.
    NPV="NP=${NP},"
fi
# The prepare directory: prepare/ by default, PREP for another one (Exp01
# has its r2 input file in prepare-full/; prepare/ there holds an older
# 3-ice-step test file).
PREP=${PREP:-${R}/prepare}
INPUTS=${PREP}/inputs_exp${EXP}.h5

# The mesh of the r2 production (NOTES decision of 2026-10-24).
M=/g/data/xd2/sg8812/giamip-software/meshes/giamip_8re_r2
MESH=${M}/giamip_8re_r2_linear.h5
MESH_EXTRA="--mesh_name giamip_8re_r2_linear --sidecar ${M}/giamip_8re_r2.json --mesh_md5 7bad5e7e3ca14fe4bc5c5e965d5b57eb"

# The schedule of the experiment, and with it the march of the input file,
# the L0 bed and the forward run.
case ${EXP} in
    09|10|12)
        SCHEDULE=windows
        PREP_EXTRA="${MESH_EXTRA} --schedule windows"
        L0_SOURCE=${ARCHIVE}/exp${EXP}/L0/invert-c/state.h5 ;;
    *)
        SCHEDULE=protocol
        PREP_EXTRA="${MESH_EXTRA} --step_years 100"
        L0_SOURCE=${ARCHIVE}/exp${EXP}/L0/invert-protocol/state.h5 ;;
esac
SOURCE=${SOURCE:-${L0_SOURCE}}
FWD=${ARCHIVE}/exp${EXP}/r2/forward-${SCHEDULE}

# The resources of one solver job, and the wall time of the long jobs.
RES="ncpus=$(( 104 * NODES )),mem=$(( 500 * NODES ))GB,jobfs=$(( 400 * NODES ))GB"
if [ ${NODES} -le 10 ]; then LONG=48:00:00; else LONG=24:00:00; fi
INV_EXTRA="--gia_rebuild_every 100 --log_memory --free_taping_solver"

DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

# qsub in the directory of the job (the jobs write to their submit
# directory), with the record of the job id. Prints the job id.
submit() {
    local dir=$1; shift
    if [ "${DRY_RUN}" = "1" ]; then
        echo "(cd ${dir} && qsub $*)" >&2
        echo "DRYRUN-$(( RANDOM ))"
        return
    fi
    mkdir -p "${dir}"
    local id
    # A failed qsub stops the whole script: the caller assigns the output
    # of this function, and set -e stops at a failed assignment, so no job
    # is submitted after a missing one.
    id=$(cd "${dir}" && qsub "$@") || { echo "error: qsub in ${dir} failed" >&2; return 1; }
    echo "$(date +%Y-%m-%dT%H:%M:%S) ${STAGE} ${dir} ${id} qsub $*" >> ${R}/jobs_r2.txt
    echo "${id}"
}

# Stop if a stage directory holds the outputs of an earlier submission.
refuse_if() {
    if [ "${FORCE}" != "1" ] && [ -e "$1" ]; then
        echo "error: $1 exists; this stage ran before. Use the next stage, or FORCE=1" >&2
        exit 1
    fi
}

# The three jobs of the readiness check.
prepare_job() {
    refuse_if ${INPUTS}
    submit ${PREP} -N x${EXP}-prep -l ${RES},walltime=05:00:00 \
        -v "${NPV}MESH=${MESH},EXTRA=${PREP_EXTRA}" ${G}/run_prepare.pbs
}
transfer_job() {
    local after=$1
    refuse_if ${R}/transfer/inputs_exp${EXP}_transfer_bed.json
    if [ "${DRY_RUN}" != "1" ] && [ ! -e "${SOURCE}" ]; then
        echo "error: the L0 bed ${SOURCE} does not exist" >&2
        exit 1
    fi
    submit ${R}/transfer -N x${EXP}-xfer -l ${RES},walltime=02:00:00 ${after} \
        -v "${NPV}TARGET=${INPUTS},SOURCE=${SOURCE},START=change" ${G}/run_transfer.pbs
}
test_job() {
    local after=$1
    refuse_if ${R}/invert-3steps/state.h5
    submit ${R}/invert-3steps -N x${EXP}-inv3 -l ${RES},walltime=03:00:00 ${after} \
        -v "${NPV}INPUTS=${INPUTS},RESTART=${INPUTS},MODE=invert,ITERATIONS=1,STEPS=3,WATCHDOG_MIN=30,MEMORY_SAMPLER=20,EXTRA=${INV_EXTRA}" \
        ${G}/run_invert.pbs
}

case ${STAGE} in
    ready)
        prep=$(prepare_job)
        xfer=$(transfer_job "-W depend=afterok:${prep}")
        inv3=$(test_job "-W depend=afterok:${xfer}")
        echo "exp${EXP} ${NODES} nodes: prepare ${prep} transfer ${xfer} 3-step test ${inv3}"
        ;;
    transfer)
        [ "${DRY_RUN}" = "1" ] || [ -e "${INPUTS}" ] || { echo "error: no input file ${INPUTS}" >&2; exit 1; }
        xfer=$(transfer_job "")
        inv3=$(test_job "-W depend=afterok:${xfer}")
        echo "exp${EXP} ${NODES} nodes: transfer ${xfer} 3-step test ${inv3}"
        ;;
    test)
        inv3=$(test_job "")
        echo "exp${EXP} ${NODES} nodes: 3-step test ${inv3}"
        ;;
    invert)
        refuse_if ${R}/invert/state.h5
        inv=$(submit ${R}/invert -N x${EXP}-inv -l ${RES},walltime=${LONG} \
            -v "${NPV}INPUTS=${INPUTS},RESTART=${INPUTS},MODE=invert,ITERATIONS=10,WATCHDOG_MIN=30,MEMORY_SAMPLER=20,EXTRA=${INV_EXTRA}" \
            ${G}/run_invert.pbs)
        echo "exp${EXP} ${NODES} nodes: inversion ${inv}"
        ;;
    restart)
        [ "${DRY_RUN}" = "1" ] || [ -e "${R}/invert/state.h5" ] || { echo "error: no ${R}/invert/state.h5 to continue from" >&2; exit 1; }
        inv=$(submit ${R}/invert -N x${EXP}-inv -l ${RES},walltime=${LONG} \
            -v "${NPV}INPUTS=${INPUTS},RESTART=state.h5,MODE=invert,ITERATIONS=10,WATCHDOG_MIN=30,MEMORY_SAMPLER=20,EXTRA=${INV_EXTRA}" \
            ${G}/run_invert.pbs)
        echo "exp${EXP} ${NODES} nodes: inversion restart ${inv}"
        ;;
    forward)
        if [ "${FORWARD_RESTART:-0}" = "1" ]; then
            fwd=$(submit ${FWD} -N x${EXP}-fwd -l ${RES},walltime=${LONG} \
                -v "${NPV}RESTART=1,SCHEDULE=${SCHEDULE}" ${G}/run_forward.pbs)
        else
            refuse_if ${FWD}/forward_exp${EXP}_surface.h5
            [ "${DRY_RUN}" = "1" ] || [ -e "${R}/invert/state.h5" ] || { echo "error: no ${R}/invert/state.h5" >&2; exit 1; }
            [ "${DRY_RUN}" = "1" ] || [ -e "${R}/invert/summary_invert.json" ] || \
                echo "warning: ${R}/invert has no summary_invert.json; is the inversion finished?" >&2
            fwd=$(submit ${FWD} -N x${EXP}-fwd -l ${RES},walltime=${LONG} \
                -v "${NPV}STATE=${R}/invert/state.h5,SCHEDULE=${SCHEDULE}" ${G}/run_forward.pbs)
        fi
        pv=$(submit ${FWD} -N x${EXP}-pv -W depend=afterok:${fwd} ${G}/run_paraview.pbs)
        img0=$(submit ${FWD} -N x${EXP}-img0 -W depend=afterok:${pv} -v HALF=0 ${G}/run_images.pbs)
        img1=$(submit ${FWD} -N x${EXP}-img1 -W depend=afterok:${pv} -v HALF=1 ${G}/run_images.pbs)
        arch=$(submit ${ARCHIVE}/exp${EXP}/r2 -N x${EXP}-arch -W depend=afterok:${img0}:${img1} \
            -v "RUN_ROOT=${R},DEST=${ARCHIVE}/exp${EXP}/r2" ${G}/run_archive.pbs)
        echo "exp${EXP} ${NODES} nodes: forward ${fwd} paraview ${pv} images ${img0} ${img1} archive ${arch}"
        echo "  a forward run that stops at the wall time exits with 3; the jobs after it are then deleted by PBS."
        echo "  Continue with FORWARD_RESTART=1 bash submit_r2.sh ${EXP} ${NODES} forward"
        ;;
    *)
        echo "error: STAGE=${STAGE}; use ready, transfer, test, invert, restart or forward" >&2
        exit 1 ;;
esac
