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
#     forward   forward -> ParaView -> images (IMAGE_PARTS jobs) -> archive
#
# The run directory of the experiment is
# /scratch/xd2/$USER/giamip/expXX/8re_r2-<NODES>nodes/ with prepare/,
# transfer/, invert-3steps/ and invert/. The inversions of Exp02-08 and
# Exp11 run in /g/data/xd2/sg8812/giamip-runs/ instead (8re_r2-16nodes-624ranks/
# there); the restart and forward stages find the root themselves (ROOT
# below). The forward run, its ParaView
# series and its images go to /g/data/xd2/sg8812/gia-mip/expXX/r2/forward-<schedule>/,
# and the archive job copies the records and the final state of the run
# directory to /g/data/xd2/sg8812/gia-mip/expXX/r2/. Every submitted job id
# is appended to jobs_r2.txt in the run directory.
#
# Settings per experiment:
#   Exp09, Exp10, Exp12   schedule D (prepare --schedule windows-coarse,
#                         forward SCHEDULE=windows-coarse; decision R12 of
#                         2026-10-24), or schedule C with
#                         WINDOW_SCHEDULE=windows; the L0 bed is
#                         expXX/L0/invert-c/state.h5 on gdata (schedule C at
#                         L0) for both
#   the other nine        the protocol steps (prepare --step_years 100,
#                         forward SCHEDULE=protocol); the L0 bed is
#                         expXX/L0/invert-protocol/state.h5 on gdata
# The transfer uses START=change for all twelve: the source is the final L0
# bed of the same experiment on another mesh (transfer_bed_exp01.py, WHAT
# IS WRITTEN).
#
# Resources of every solver job: ncpus = 104 NODES, mem = 500 NODES GB,
# jobfs = 400 NODES GB (the tape goes to $PBS_JOBFS, decision R11). Wall
# time of the inversion, its restarts and the forward run: the longest
# that normalsr gives for the node count of the job (NCI queue limits,
# by cores): 48 h up to 10 nodes (1040 cores), 24 h up to 20 nodes (2080
# cores), 10 h up to 40 nodes (4160 cores), 5 h up to 100 nodes (10400
# cores, the largest normalsr job). WALLTIME=HH:MM:SS replaces it.
#
# The solver options of the inversion (decision of 2026-10-24): default
# solver options, --gia_rebuild_every 100 --log_memory
# --free_taping_solver, the watchdog at 30 min, the node memory sampler
# every 20 s, the node jobfs sampler every 30 s (the watcher agents read
# its tape column, and the 3-step test projects the tape from it). The
# stop rule is the default of invert_expXX.py: RMS misfit
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
#   FORWARD_B=1        forward stage, Exp09, Exp10, Exp12 on schedule D only:
#                      the "b" forward chain (decision R14). The forward run
#                      marches schedule D with every protocol year as an
#                      extra step end (SCHEDULE=windows-coarse-protocol,
#                      572-592 steps), from the same invert/state.h5 as the
#                      schedule-D forward run, with --gia_rebuild_max_outer
#                      0 as in the inversion. Forward, ParaView and images
#                      go to ARCHIVE/expXX/r2/forward-windows-coarse-b/; the
#                      archive job runs with COPY=0 and DEST in that
#                      directory, so it copies nothing and changes no file
#                      of the existing archive (it writes the MD5 of the
#                      inversion files and the manifest of the b directory).
#                      A b leg after a wall-time stop: FORWARD_B=1
#                      FORWARD_RESTART=1 with the same other variables.
#   FORWARD_NODES=<n>  forward stage: the nodes of the forward run. The
#                      forward run is not taped, so it needs no node disk
#                      per rank and puts the NP ranks of the partition on
#                      full nodes: the default is NP / 104 if NP divides by
#                      104, else NODES. For example RANKS_PER_NODE=39
#                      bash submit_r2.sh 01 16 forward for an inversion on
#                      16 nodes with 39 ranks per node runs the forward run
#                      on 6 nodes (624 ranks, one per core). NP must divide
#                      by FORWARD_NODES and fit its cores.
#   IMAGE_PARTS=<k>    forward stage: the number of image jobs (default 2,
#                      run_images.pbs NPARTS; 1.9 h each for 926 images)
#   ROOT               the root of the run directories. Default for the
#                      stages ready, transfer, test and invert:
#                      /scratch/xd2/$USER/giamip. Default for restart and
#                      forward: the one of /scratch/xd2/$USER/giamip and
#                      /g/data/xd2/sg8812/giamip-runs whose run directory
#                      holds invert/state.h5; the script stops if both or
#                      neither do. These two stages therefore need Gadi,
#                      also with DRY_RUN=1.
#   ARCHIVE            the root of the archive (/g/data/xd2/sg8812/gia-mip)
#   WINDOW_SCHEDULE    Exp09, Exp10, Exp12 only: windows-coarse (default,
#                      schedule D) or windows (schedule C). It sets the march
#                      of the prepare job and the schedule of the forward
#                      run. The other nine experiments do not read it.
#   WALLTIME=HH:MM:SS  the wall time of the inversion, restart or forward
#                      job instead of the queue limit of its node count. It
#                      must not be above that limit.
#   XFER_DIR=<dir>     forward stage: the bed transfer directory that the
#                      archive job copies (default: transfer/ beside PREP,
#                      transfer-d/ for schedule D)
#   TEST_DIR=<dir>     forward stage: the test directory that the archive job
#                      copies, under its own name (default below)
#
# Schedule D (Exp09, Exp10, Exp12 with WINDOW_SCHEDULE=windows-coarse), the
# restart and forward stages. The readiness jobs of these three ran at the
# rank count NP on NP / 104 full nodes (T8, 2026-10-24), and the inversion
# runs on gdata with fewer ranks per node, with a layout that can change at
# a restart (Exp09: first job 26 x 72, restarts 36 x 52, all in
# 8re_r2-26nodes-1872ranks/). So for these stages:
#   run directory   the one directory under /scratch/xd2/$USER/giamip/expXX
#                   and /g/data/xd2/sg8812/giamip-runs/expXX (only ROOT if
#                   set) that is named 8re_r2-<n>nodes-<NP>ranks for any n,
#                   or 8re_r2-<NP/104>nodes, and holds invert/state.h5. The
#                   script stops if there is not exactly one. NODES and
#                   RANKS_PER_NODE set NP and the resources of a restart
#                   job, not the directory name.
#   PREP            /scratch/xd2/$USER/giamip/expXX/8re_r2-<NP/104>nodes/prepare-d
#   XFER_DIR        transfer-d/ beside PREP
#   TEST_DIR        invert-test6/ of the run directory, else the one
#                   invert-test6/ of another run directory of NP ranks (the
#                   short test of Exp12 ran at 26 x 72, its inversion at
#                   36 x 52)
# Examples (the forward run on NP / 104 full nodes, 24 h):
#   RANKS_PER_NODE=78 bash submit_r2.sh 10 16 restart     (16 x 78, 24 h)
#   RANKS_PER_NODE=52 bash submit_r2.sh 09 36 restart     (36 x 52, 10 h)
#   bash submit_r2.sh 10 12 forward                       (1248 ranks)
#   bash submit_r2.sh 09 18 forward                       (1872 ranks)
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

ARCHIVE=${ARCHIVE:-/g/data/xd2/sg8812/gia-mip}
if [ ${RANKS_PER_NODE} -eq 104 ]; then
    RUN_NAME=8re_r2-${NODES}nodes
    # One rank per core: the jobs take NP from the cores of the job.
    NPV=""
else
    RUN_NAME=8re_r2-${NODES}nodes-${NP}ranks
    # The first entry of every -v list of a solver job.
    NPV="NP=${NP},"
fi

# The schedule of the experiment, and with it the march of the input file,
# the L0 bed and the forward run.
case ${EXP} in
    09|10|12)
        # Schedule D on r2 (R12): schedule C needs more node disk for the
        # tape and more wall time per restart than a Gadi job has.
        SCHEDULE=${WINDOW_SCHEDULE:-windows-coarse}
        case ${SCHEDULE} in
            windows|windows-coarse) ;;
            *) echo "error: WINDOW_SCHEDULE=${SCHEDULE}; use windows-coarse or windows" >&2; exit 1 ;;
        esac
        L0_SOURCE=${ARCHIVE}/exp${EXP}/L0/invert-c/state.h5 ;;
    *)
        SCHEDULE=protocol
        L0_SOURCE=${ARCHIVE}/exp${EXP}/L0/invert-protocol/state.h5 ;;
esac

# The schedule of the forward run (SCHEDULE of run_forward.pbs), its extra
# options and the name of its directory. The b chain (FORWARD_B=1) changes
# all three; it is checked here, before the search for the run directory,
# so a wrong use stops with its own message.
FSCHEDULE=${SCHEDULE}
FEXTRA=""
FWD_NAME=${SCHEDULE}
FORWARD_B=${FORWARD_B:-0}
case ${FORWARD_B} in
    0) ;;
    1)
        if [ "${SCHEDULE}" != "windows-coarse" ] || [ "${STAGE}" != "forward" ]; then
            echo "error: FORWARD_B=1 is for the forward stage of Exp09, Exp10 and Exp12 on schedule D (windows-coarse)" >&2
            exit 1
        fi
        # Schedule D plus the protocol years, from the schedule-D state
        # file of the inversion. The inversion ran without the rebuild
        # guard; the forward run of the fine schedules has it on by
        # default, which changes where the preconditioner is rebuilt
        # (NOTES exp12-forward-J-check), so the b run turns it off.
        FSCHEDULE=windows-coarse-protocol
        FEXTRA=",EXTRA=--gia_rebuild_max_outer 0"
        FWD_NAME=windows-coarse-b ;;
    *) echo "error: FORWARD_B=${FORWARD_B}; use 0 or 1" >&2; exit 1 ;;
esac

# The schedule D production of the restart and forward stages (the header
# explains the layout of T8). D_RUN=1 selects its search for the run
# directory and its defaults of PREP, XFER_DIR and TEST_DIR. The other
# stages, schedule C and the nine Maxwell experiments keep the rules below
# unchanged.
D_RUN=0
if [ "${SCHEDULE}" = "windows-coarse" ]; then
    case ${STAGE} in restart|forward) D_RUN=1 ;; esac
fi
# The node count of NP ranks at one rank per core: the readiness jobs of
# schedule D ran in 8re_r2-<FULL_NODES>nodes/. Empty if NP does not fill
# whole nodes.
if [ $(( NP % 104 )) -eq 0 ]; then FULL_NODES=$(( NP / 104 )); else FULL_NODES=""; fi

# The root of the run directories. New runs go to /scratch. The restart
# and the forward run continue an existing inversion, and those of
# Exp02-08 and Exp11 run on gdata (giamip-runs), so for these two stages
# the root is the one whose run directory holds invert/state.h5. A wrong
# root would give the forward run a missing state file, and the archive
# job a run directory without invert/ (it would skip it and exit 0), so
# the script stops if the state file is under both roots or under none.
SCRATCH_ROOT=/scratch/xd2/${USER}/giamip
GDATA_ROOT=/g/data/xd2/sg8812/giamip-runs
if [ "${D_RUN}" = "1" ]; then
    # Schedule D: the run directory of NP ranks with any node count in its
    # name, because the job layout of a restart can differ from the layout
    # in the name of the directory. The partition of the input file is the
    # same for every layout of NP ranks, so any of them is a valid run
    # directory, but there must be exactly one, or the forward run could
    # start from the bed of another run.
    found=""
    for root in ${ROOT:-${SCRATCH_ROOT} ${GDATA_ROOT}}; do
        for dir in ${root}/exp${EXP}/8re_r2-*nodes-${NP}ranks ${FULL_NODES:+${root}/exp${EXP}/8re_r2-${FULL_NODES}nodes}; do
            # A pattern without a match stays as it is and fails this test.
            if [ -e "${dir}/invert/state.h5" ]; then
                found="${found} ${dir}"
            fi
        done
    done
    if [ $(echo ${found} | wc -w) -ne 1 ]; then
        echo "error: schedule D, ${NP} ranks: invert/state.h5 is in ${found:-none} of the run directories 8re_r2-*nodes-${NP}ranks${FULL_NODES:+ and 8re_r2-${FULL_NODES}nodes} of exp${EXP} under ${ROOT:-${SCRATCH_ROOT} and ${GDATA_ROOT}}; need exactly one" >&2
        exit 1
    fi
    R=${found# }
else
    if [ -z "${ROOT:-}" ]; then
        case ${STAGE} in
            restart|forward)
                found=""
                for root in ${SCRATCH_ROOT} ${GDATA_ROOT}; do
                    if [ -e "${root}/exp${EXP}/${RUN_NAME}/invert/state.h5" ]; then
                        found="${found} ${root}"
                    fi
                done
                # One word: exactly one root holds the state file.
                if [ $(echo ${found} | wc -w) -ne 1 ]; then
                    echo "error: exp${EXP}/${RUN_NAME}/invert/state.h5 is under ${found:-none} of ${SCRATCH_ROOT} and ${GDATA_ROOT}; set ROOT" >&2
                    exit 1
                fi
                ROOT=${found# } ;;
            *)
                ROOT=${SCRATCH_ROOT} ;;
        esac
    fi
    R=${ROOT}/exp${EXP}/${RUN_NAME}
fi
echo "run directory ${R}" >&2
# The prepare directory: prepare/ by default, PREP for another one (Exp01
# has its r2 input file in prepare-full/; prepare/ there holds an older
# 3-ice-step test file). Schedule D: prepare-d/ of the full-node directory
# on /scratch, where T8 wrote the input file of the production.
if [ "${D_RUN}" = "1" ] && [ -n "${FULL_NODES}" ]; then
    PREP=${PREP:-${SCRATCH_ROOT}/exp${EXP}/8re_r2-${FULL_NODES}nodes/prepare-d}
else
    PREP=${PREP:-${R}/prepare}
fi
INPUTS=${PREP}/inputs_exp${EXP}.h5

# The mesh of the r2 production (NOTES decision of 2026-10-24).
M=/g/data/xd2/sg8812/giamip-software/meshes/giamip_8re_r2
MESH=${M}/giamip_8re_r2_linear.h5
MESH_EXTRA="--mesh_name giamip_8re_r2_linear --sidecar ${M}/giamip_8re_r2.json --mesh_md5 7bad5e7e3ca14fe4bc5c5e965d5b57eb"

# The options of the prepare job for the march of the schedule.
case ${SCHEDULE} in
    protocol) PREP_EXTRA="${MESH_EXTRA} --step_years 100" ;;
    *)        PREP_EXTRA="${MESH_EXTRA} --schedule ${SCHEDULE}" ;;
esac
SOURCE=${SOURCE:-${L0_SOURCE}}
# The forward directory and the archive directory of the forward chain
# (the b chain: see FORWARD_B above).
FWD=${ARCHIVE}/exp${EXP}/r2/forward-${FWD_NAME}
if [ "${FORWARD_B}" = "1" ]; then ARCH_DIR=${FWD}; else ARCH_DIR=${ARCHIVE}/exp${EXP}/r2; fi

# The longest wall time of a normalsr job on n nodes (NCI queue limits,
# which go by cores: 1-1040 cores 48 h, 1144-2080 24 h, 2184-4160 10 h,
# 4264-10400 5 h; 104 cores per node). Prints nothing above 100 nodes,
# where normalsr has no job.
max_walltime() {
    local n=$1
    if [ ${n} -le 10 ]; then echo 48:00:00
    elif [ ${n} -le 20 ]; then echo 24:00:00
    elif [ ${n} -le 40 ]; then echo 10:00:00
    elif [ ${n} -le 100 ]; then echo 05:00:00
    fi
}
# The wall time of a long job on n nodes: WALLTIME if given, else the
# limit. Stops the script if WALLTIME is not HH:MM:SS or above the limit,
# so qsub does not refuse the first job of a chain after the others are
# planned.
long_walltime() {
    local n=$1 limit
    limit=$(max_walltime ${n})
    if [ -z "${limit}" ]; then
        echo "error: ${n} nodes is above the 100 nodes of a normalsr job" >&2
        return 1
    fi
    if [ -z "${WALLTIME:-}" ]; then
        echo ${limit}
        return
    fi
    if ! [[ ${WALLTIME} =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
        echo "error: WALLTIME=${WALLTIME} is not HH:MM:SS" >&2
        return 1
    fi
    # Both in seconds.
    local w l
    w=$(echo ${WALLTIME} | awk -F: '{print $1 * 3600 + $2 * 60 + $3}')
    l=$(echo ${limit} | awk -F: '{print $1 * 3600 + $2 * 60 + $3}')
    if [ ${w} -gt ${l} ]; then
        echo "error: WALLTIME=${WALLTIME} is above the normalsr limit ${limit} of ${n} nodes" >&2
        return 1
    fi
    echo ${WALLTIME}
}

# The resources of one solver job, and the wall time of the long jobs.
RES="ncpus=$(( 104 * NODES )),mem=$(( 500 * NODES ))GB,jobfs=$(( 400 * NODES ))GB"
LONG=$(long_walltime ${NODES})
INV_EXTRA="--gia_rebuild_every 100 --log_memory --free_taping_solver"

# The forward run: FORWARD_NODES nodes with the NP ranks of the partition.
# It is not taped, so the default is the fewest full nodes that hold NP
# ranks (one rank per core). Its jobfs holds only the kernel caches of the
# Firedrake modules (no tape), so it asks for 100 GB per node.
FORWARD_NODES=${FORWARD_NODES:-${FULL_NODES:-${NODES}}}
if ! [[ ${FORWARD_NODES} =~ ^[1-9][0-9]*$ ]] || [ $(( NP % FORWARD_NODES )) -ne 0 ] \
        || [ ${NP} -gt $(( 104 * FORWARD_NODES )) ]; then
    echo "error: FORWARD_NODES=${FORWARD_NODES} does not fit NP=${NP} ranks" >&2
    exit 1
fi
FRES="ncpus=$(( 104 * FORWARD_NODES )),mem=$(( 500 * FORWARD_NODES ))GB,jobfs=$(( 100 * FORWARD_NODES ))GB"
FLONG=$(long_walltime ${FORWARD_NODES})
# NP goes on the -v list only if it is not one rank per core.
if [ ${NP} -eq $(( 104 * FORWARD_NODES )) ]; then FNPV=""; else FNPV="NP=${NP},"; fi
IMAGE_PARTS=${IMAGE_PARTS:-2}
if ! [[ ${IMAGE_PARTS} =~ ^[1-9][0-9]*$ ]]; then
    echo "error: IMAGE_PARTS=${IMAGE_PARTS} is not a positive integer" >&2
    exit 1
fi
# The prepare, transfer and test directories of the archive. The readiness
# jobs can run in the run directory of another node count than NODES (the
# 6-node prepare of a 16-node inversion): the transfer job writes into
# transfer/ beside the prepare directory, and the 3-step test is in
# invert-3steps/ of this run directory or beside the prepare directory.
# Schedule D: T8 wrote the transfer into transfer-d/ beside prepare-d/ (the
# schedule C transfer stays in transfer/), and the short test (6 taped
# steps) into invert-test6/ of a run directory of NP ranks on gdata.
if [ "${D_RUN}" = "1" ]; then
    XFER_DIR=${XFER_DIR:-$(dirname "${PREP}")/transfer-d}
    if [ -z "${TEST_DIR:-}" ]; then
        if [ -d "${R}/invert-test6" ]; then
            TEST_DIR=${R}/invert-test6
        else
            # The short test of another layout of the same NP ranks.
            found=""
            for dir in $(dirname "${R}")/8re_r2-*nodes-${NP}ranks/invert-test6; do
                if [ -d "${dir}" ]; then found="${found} ${dir}"; fi
            done
            if [ $(echo ${found} | wc -w) -gt 1 ]; then
                echo "error: more than one short test of ${NP} ranks:${found}; set TEST_DIR" >&2
                exit 1
            fi
            # None found: the archive job reports the missing directory
            # and skips it.
            TEST_DIR=${found# }
            TEST_DIR=${TEST_DIR:-${R}/invert-test6}
        fi
    fi
else
    XFER_DIR=${XFER_DIR:-$(dirname "${PREP}")/transfer}
    if [ -z "${TEST_DIR:-}" ]; then
        TEST_DIR=${R}/invert-3steps
        [ -d "${TEST_DIR}" ] || TEST_DIR=$(dirname "${PREP}")/invert-3steps
    fi
fi

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
    mkdir -p "${dir}" "${R}"
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
        -v "${NPV}INPUTS=${INPUTS},RESTART=${INPUTS},MODE=invert,ITERATIONS=1,STEPS=3,WATCHDOG_MIN=30,MEMORY_SAMPLER=20,JOBFS_SAMPLER=30,EXTRA=${INV_EXTRA}" \
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
            -v "${NPV}INPUTS=${INPUTS},RESTART=${INPUTS},MODE=invert,ITERATIONS=10,WATCHDOG_MIN=30,MEMORY_SAMPLER=20,JOBFS_SAMPLER=30,EXTRA=${INV_EXTRA}" \
            ${G}/run_invert.pbs)
        echo "exp${EXP} ${NODES} nodes: inversion ${inv}"
        ;;
    restart)
        [ "${DRY_RUN}" = "1" ] || [ -e "${R}/invert/state.h5" ] || { echo "error: no ${R}/invert/state.h5 to continue from" >&2; exit 1; }
        inv=$(submit ${R}/invert -N x${EXP}-inv -l ${RES},walltime=${LONG} \
            -v "${NPV}INPUTS=${INPUTS},RESTART=state.h5,MODE=invert,ITERATIONS=10,WATCHDOG_MIN=30,MEMORY_SAMPLER=20,JOBFS_SAMPLER=30,EXTRA=${INV_EXTRA}" \
            ${G}/run_invert.pbs)
        echo "exp${EXP} ${NODES} nodes: inversion restart ${inv}"
        ;;
    forward)
        # Both paths need the inversion: a first run reads its state file,
        # and the archive job of both copies its run directory.
        [ -e "${R}/invert/state.h5" ] || { echo "error: no ${R}/invert/state.h5" >&2; exit 1; }
        # The archive job copies the prepare and transfer records from PREP
        # and beside it; a missing PREP would leave them out without an error.
        # The archive job of the b chain copies nothing (COPY=0), so the b
        # chain does not need PREP.
        if [ "${FORWARD_B}" != "1" ]; then
            [ -d "${PREP}" ] || { echo "error: no prepare directory ${PREP}; set PREP" >&2; exit 1; }
        fi
        if [ "${FORWARD_RESTART:-0}" = "1" ]; then
            fwd=$(submit ${FWD} -N x${EXP}-fwd -l ${FRES},walltime=${FLONG} \
                -v "${FNPV}RESTART=1,SCHEDULE=${FSCHEDULE}${FEXTRA}" ${G}/run_forward.pbs)
        else
            refuse_if ${FWD}/forward_exp${EXP}_surface.h5
            [ "${DRY_RUN}" = "1" ] || [ -e "${R}/invert/summary_invert.json" ] || \
                echo "warning: ${R}/invert has no summary_invert.json; is the inversion finished?" >&2
            fwd=$(submit ${FWD} -N x${EXP}-fwd -l ${FRES},walltime=${FLONG} \
                -v "${FNPV}STATE=${R}/invert/state.h5,SCHEDULE=${FSCHEDULE}${FEXTRA}" ${G}/run_forward.pbs)
        fi
        pv=$(submit ${FWD} -N x${EXP}-pv -W depend=afterok:${fwd} ${G}/run_paraview.pbs)
        imgs=""
        for (( p = 0; p < IMAGE_PARTS; p++ )); do
            img=$(submit ${FWD} -N x${EXP}-img${p} -W depend=afterok:${pv} \
                -v PART=${p},NPARTS=${IMAGE_PARTS} ${G}/run_images.pbs)
            imgs="${imgs}:${img}"
        done
        if [ "${FORWARD_B}" = "1" ]; then
            # The b chain: no copy, DEST is the b directory (see FORWARD_B).
            ARCH_V="RUN_ROOT=${R},COPY=0,DEST=${ARCH_DIR}"
        else
            ARCH_V="RUN_ROOT=${R},PREP_DIR=${PREP},XFER_DIR=${XFER_DIR},TEST_DIR=${TEST_DIR},DEST=${ARCH_DIR}"
        fi
        arch=$(submit ${ARCH_DIR} -N x${EXP}-arch -W depend=afterok${imgs} \
            -v "${ARCH_V}" ${G}/run_archive.pbs)
        echo "exp${EXP} forward (${FSCHEDULE}) on ${FORWARD_NODES} nodes in ${FWD}: forward ${fwd} paraview ${pv} images ${imgs#:} archive ${arch}"
        echo "  a forward run that stops at the wall time exits with 3; the jobs after it are then deleted by PBS."
        if [ "${FORWARD_B}" = "1" ]; then B_WORD=" FORWARD_B=1"; else B_WORD=""; fi
        echo "  Continue with the same variables and FORWARD_RESTART=1${B_WORD} bash submit_r2.sh ${EXP} ${NODES} forward"
        ;;
    *)
        echo "error: STAGE=${STAGE}; use ready, transfer, test, invert, restart or forward" >&2
        exit 1 ;;
esac
