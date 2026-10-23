r"""Run diagnostics of the Exp12 inversion: memory per stage, PETSc profile, solver work.

`invert_exp12.py --diagnostics` builds one `RunDiagnostics` and installs its
hooks. Without the option this module is never imported, and the run is the
run without diagnostics. The diagnostics answer two questions of the mesh
sweep (`NOTES/team/gadi-mesh-sweep/00-brief.md`): where the memory goes, and
where the solve time goes.

WHAT IS RECORDED
    1. Memory per rank at the stages of the run (`RunDiagnostics.memory`):
       the current resident size, the peak resident size of the process
       (`VmHWM`), the host, the owned degrees of freedom of the solution, and
       the owned mantle and parent cells. The stages are: after the load,
       after the solver build, after the setup, before and after the build of
       every outer preconditioner (one per solver: the taped forward solver,
       the cached replay solver, the cached adjoint solver), after every taped
       step, after the taped forward, after the replay, after the adjoint,
       and after every Taylor evaluation.
    2. A PETSc profile split by PETSc log stages (`RunDiagnostics.stage`):
       `exp12_setup`, `exp12_taped_forward`, `exp12_replay`, `exp12_adjoint`,
       `exp12_taylor` (`exp12_rol` in invert mode). At the end of every stage
       the full `-log_view` of the run so far is written again, so a job that
       is killed keeps the profile of the stages that ended. It has no
       memory columns: with `-log_view_memory` the L0 check run stopped in
       its first solve (job 182106743: `PetscMallocPopMaximumUsage` failed at
       the end of a nested event), so the memory per stage comes from 1.
    3. The solver work of every nonlinear (or linear adjoint) solve
       (`RunDiagnostics.solve`): Newton iterations, outer Krylov iterations
       per Newton step, the block-0 solves with their iterations and the
       number that stop at the iteration cap, the builds of the cached
       `Real`-block complement (`_SchurAinvBCache`) and of the low-rank DtN
       columns (`LowRankPotentialPC`), and the wall time of the
       preconditioner setup against the rest of the solve.

    4. What the preconditioner of each rank holds
       (`_block0_makeup`, `_twoblock_makeup`): at the first application of
       each block-0 preconditioner, the local memory and nonzeros of the
       four condensed blocks, the levels of the two GAMG hierarchies with
       the local rows, nonzeros and memory of each level operator, and the
       low-rank DtN columns; at each outer preconditioner build, the cached
       columns of the `Real`-block complement. With the halo dofs and the
       neighbour ranks of `set_problem`, the extra memory of a heavy rank can
       be put on one of these components.
    5. The residual history of every block-0 solve: the reduction
       `h[-1] / h[0]` of the inner FGMRES, and for a solve at the cap the
       reduction at iterations 10 and 25 and at the last iteration. A capped
       solve with a flat history stalls; one with a steady reduction is slow.

    Every record carries `t_unix` (seconds since 1970), the clock of the
    node sampler (`node_memory_sampler.sh`) and of the PBS `date` lines.

OUTPUT FILES (in --output_path, written on rank 0, flushed after every line)
    diag_memory.jsonl   one line per stage and rank
    diag_solves.jsonl   one line per solve
    diag_stages.jsonl   one line per PETSc log stage at its end: the events
                        of the stage parsed from the log view
    diag_pc.jsonl       one line per preconditioner build: the summary over
                        the ranks (mean, max, rank of the max) of every
                        field of item 4, and the full fields of the ranks
                        above twice the mean memory
    petsc_log_view.txt  the full PETSc log view, rewritten at every stage end
                        and at the end of the run

HOW THE HOOKS WORK
    The hooks wrap methods of the preconditioner classes of
    `gadopt.preconditioners` and of the coupled solve block of
    `gadopt.dtn_coupled_adjoint` at class level, as `install_block_log` of
    `invert_exp12.py` does. They read counters and clocks and change no
    numerical operation, with one exception that is equivalent: before each
    block-0 application the hook calls `PCSetUp` on the fieldsplit PC of the
    condensed block-0 KSP and `KSPSetUp` on its two split KSPs, so that the
    setup of the two GAMG hierarchies and of the low-rank columns is timed
    apart from the iterations. The solve calls the same setups first, and a
    second setup on an operator that did not change returns at once, so the
    work and its result are the same; only the moment of the setup moves.

    A SNES monitor is attached to every Newton solver (not to the `ksponly`
    adjoint solver: `SNESSolve_KSPONLY` evaluates one more residual when a
    monitor is set). It records the outer Krylov iterations of each Newton
    step. The KSP converged-reason lines of the solver preset are removed
    from the solver parameters when the diagnostics are on
    (`exp12_common.build_solver`); the solve table replaces them.
"""

import contextlib
import json
import os
import re
import socket
import time
import weakref

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from gadopt import COMM_WORLD

#: The PETSc events that the stage summary in the log always shows, when the
#: stage has them. The times of nested events overlap: an outer `KSPSolve`
#: contains the inner ones, and `PCApply` contains the block-0 `KSPSolve`.
SUMMARY_EVENTS = ("SNESSolve", "SNESFunctionEval", "SNESJacobianEval",
                  "KSPSolve", "KSPSetUp", "PCSetUp", "PCApply", "MatMult",
                  "CondensedBlockPCApply", "LowRankPotentialPCApply",
                  "ParLoopExecute")

#: The bytes in one GB, as the rest of the experiment counts them (1e9).
GB = 1.0e9


def _rss_gb():
    """The resident size of this process now, GB (`/proc/self/statm`; 0 without /proc)."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / GB
    except OSError:
        return 0.0


def _status_gb(key):
    """A `kB` field of `/proc/self/status` (`VmHWM`: the peak resident size), GB."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    # The kernel's kB is 1024 bytes.
                    return int(line.split()[1]) * 1024.0 / GB
    except OSError:
        pass
    return 0.0


def _meminfo_gb(key):
    """A `kB` field of `/proc/meminfo` (`MemAvailable`: what the node can still give), GB."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    # The kernel's kB is 1024 bytes.
                    return int(line.split()[1]) * 1024.0 / GB
    except OSError:
        pass
    return None


def _mat_info(mat):
    """The local memory (MB), used nonzeros and rows of an assembled matrix.

    `MatGetInfo` with `MAT_LOCAL` reads the storage of this rank only, so the
    numbers of different ranks can be compared directly. Its `memory` field
    is 0 in PETSc 3.26 (PETSc no longer logs object memory; seen on the
    login-node check), so the memory is estimated from the allocated
    nonzeros: one scalar and one column index per nonzero and one row
    pointer per row, which is the storage of an AIJ matrix. The halo
    vector and the column map of a parallel matrix are not counted. A
    matrix without stored entries (a Python shell matrix, such as the
    low-rank potential operator of the nest) gives None.

    Args:
      mat: a `PETSc.Mat`.

    Returns:
      `(memory_mb, nz_used, local_rows)`, or None.
    """
    try:
        if mat.getType() == PETSc.Mat.Type.PYTHON:
            return None
        info = mat.getInfo(PETSc.Mat.InfoType.LOCAL)
        rows = int(mat.getLocalSize()[0])
        index = np.dtype(PETSc.IntType).itemsize
        scalar = np.dtype(PETSc.ScalarType).itemsize
        memory = (float(info["nz_allocated"]) * (scalar + index)
                  + (rows + 1) * index)
        return memory / 1.0e6, float(info["nz_used"]), rows
    except Exception:  # noqa: BLE001  (a record, never a reason to stop)
        return None


def _gamg_makeup(pc, tag, out):
    """Add the levels of a GAMG (or other multigrid) PC to `out`.

    PETSc numbers the levels from the coarsest (level 0) to the finest
    (`levels - 1`); the finest level operator is the matrix the PC was built
    on, which the caller records itself, so it is left out of the total
    `{tag}_coarse_mem_mb`. For each level `k` it adds `{tag}_l{k}_rows`,
    `{tag}_l{k}_nz` and `{tag}_l{k}_mem_mb` (local to this rank).

    Args:
      pc: the `PETSc.PC`.
      tag: the name prefix of the fields.
      out: the dict to fill.
    """
    try:
        if pc.getType() not in ("gamg", "mg", "ml", "hypre"):
            return
        levels = int(pc.getMGLevels())
    except Exception:  # noqa: BLE001
        return
    out[f"{tag}_levels"] = levels
    coarse = 0.0
    for level in range(levels):
        try:
            matrix = pc.getMGSmoother(level).getOperators()[0]
        except Exception:  # noqa: BLE001
            continue
        info = _mat_info(matrix)
        if info is None:
            continue
        memory, nz, rows = info
        out[f"{tag}_l{level}_rows"] = rows
        out[f"{tag}_l{level}_nz"] = nz
        out[f"{tag}_l{level}_mem_mb"] = memory
        if level < levels - 1:
            coarse += memory
    out[f"{tag}_coarse_mem_mb"] = coarse


def _block0_makeup(obj, ksp):
    """What one `CondensedBlockPC` holds on this rank, as a flat dict.

    Called after the first setup of the block-0 preconditioner (the setup
    that the hook times), so the GAMG hierarchies and the low-rank columns
    exist. Fields (all local to this rank):

    - `cond_<block>_mem_mb`, `_nz`, `_rows` for the four condensed blocks
      `S_uu`, `A_upsi`, `A_psiu`, `A_psipsi`;
    - the levels of the GAMG of the displacement split (`gamg_u_...`, see
      `_gamg_makeup`) and of the GAMG inside the low-rank PC of the
      potential split (`gamg_psi_...`);
    - `lowrank_columns` (the number of DtN modes), `lowrank_z_local` (the
      local length of all columns `Z` together, one float each),
      `lowrank_u_entries` (the local entries of the mode rows `U`) and the
      private copy of the potential block (`lowrank_copy_mem_mb`);
    - `pc_mat_mem_mb`: the sum of the matrix memory above (the condensed
      blocks, the coarse GAMG levels, the low-rank copy) plus the columns
      `Z` at 8 bytes per value.

    Args:
      obj: the `CondensedBlockPC`.
      ksp: its condensed KSP.

    Returns:
      The dict.
    """
    out = {}
    total = 0.0
    for name in ("S_uu", "A_upsi", "A_psiu", "A_psipsi"):
        block = getattr(obj, name, None)
        info = _mat_info(block.petscmat) if block is not None else None
        if info is None:
            continue
        out[f"cond_{name}_mem_mb"], out[f"cond_{name}_nz"], \
            out[f"cond_{name}_rows"] = info
        total += info[0]
    try:
        subs = ksp.getPC().getFieldSplitSubKSP()
    except Exception:  # noqa: BLE001
        subs = []
    if len(subs) == 2:
        _gamg_makeup(subs[0].getPC(), "gamg_u", out)
        total += out.get("gamg_u_coarse_mem_mb", 0.0)
        psi_pc = subs[1].getPC()
        context = None
        try:
            if psi_pc.getType() == PETSc.PC.Type.PYTHON:
                context = psi_pc.getPythonContext()
        except Exception:  # noqa: BLE001
            context = None
        if context is not None and hasattr(context, "_columns"):
            columns = getattr(context, "_Z", None) or []
            out["lowrank_columns"] = len(context._columns)
            out["lowrank_z_local"] = int(sum(z.getLocalSize() for z in columns))
            out["lowrank_u_entries"] = int(sum(len(dofs) for dofs, _, _
                                               in context._columns))
            total += out["lowrank_z_local"] * 8.0 / 1.0e6
            assembled = getattr(context, "assembled", None)
            info = _mat_info(assembled) if assembled is not None else None
            if info is not None:
                out["lowrank_copy_mem_mb"] = info[0]
                total += info[0]
            inner = getattr(context, "ksp", None)
            if inner is not None:
                _gamg_makeup(inner.getPC(), "gamg_psi", out)
                total += out.get("gamg_psi_coarse_mem_mb", 0.0)
        else:
            _gamg_makeup(psi_pc, "gamg_psi", out)
            total += out.get("gamg_psi_coarse_mem_mb", 0.0)
    out["pc_mat_mem_mb"] = total
    return out


def _twoblock_makeup(obj):
    """The cached columns of the `Real`-block complement on this rank.

    `_SchurAinvBCache.Z` holds one block-0 vector per `Real` row: fields
    `ainvb_columns`, `ainvb_z_local` (local values of all columns) and
    `ainvb_z_mem_mb` (8 bytes per value). Empty without `dtn_schur_ainvb`.
    """
    cache = getattr(obj, "ainvb", None)
    columns = getattr(cache, "Z", None) if cache is not None else None
    if not columns:
        return {}
    local = int(sum(z.getLocalSize() for z in columns))
    return {"ainvb_columns": len(columns), "ainvb_z_local": local,
            "ainvb_z_mem_mb": local * 8.0 / 1.0e6}


def _safe_makeup(function, *args):
    """`function(*args)`, or a marker field if it raises.

    The make-up is a record. It runs inside a collective preconditioner
    call, so an error must not stop one rank and leave the others waiting:
    every rank returns a dict and goes on to the collective memory record.
    """
    try:
        return function(*args)
    except Exception as exc:  # noqa: BLE001
        return {"makeup_error": 1.0, "makeup_error_text": repr(exc)[:200]}


def _neighbour_ranks(mesh):
    """The number of ranks this rank receives halo points from, for one mesh.

    The leaves of the point star forest of the DMPlex are the halo points of
    this rank; their remote ranks are the neighbours that own them.
    """
    try:
        _, _, remote = mesh.topology_dm.getPointSF().getGraph()
        remote = np.asarray(remote).reshape(-1, 2)
        return int(len(np.unique(remote[:, 0])))
    except Exception:  # noqa: BLE001
        return None


def _new_counters():
    """The counters of one solve, filled by the preconditioner hooks.

    Times are seconds of wall clock on each rank; `RunDiagnostics.end_solve`
    takes the largest value over the ranks.

    - `outer_pc`: `DtNTwoBlockSchurPC`, the outer preconditioner. Its
      `initialize` and `update` hold the whole preconditioner setup of an
      outer linear solve: the re-extraction of the sub-blocks and, under
      `dtn_schur_ainvb`, the build of the cached complement.
    - `ainvb`: the builds of `_SchurAinvBCache` (n block-0 solves each).
    - `block0`: the applications of `CondensedBlockPC`, that is the block-0
      solves. `its` and `capped` count the iterations of its inner FGMRES
      (`..._dtn_fieldsplit_0_condensed_`); `build_*` are the solves made
      inside a complement build. `setup_s` is the explicit `KSPSetUp` before
      the application (GAMG on `u`, the low-rank PC on `psi`), split into
      `setup_u_s` and `setup_psi_s`. `reduction_*` are the sum, smallest and
      largest `h[-1] / h[0]` of the residual history `h` of the inner FGMRES
      (`rtol` its relative tolerance); `capped_history` holds, for the first
      20 solves at the cap, the iterations and `h[10] / h[0]`,
      `h[25] / h[0]` and `h[-1] / h[0]`.
    - `condensed`: `CondensedBlockPC.initialize` and the reassemblies of its
      four blocks in `update`.
    - `lowrank`: `LowRankPotentialPC.initialize`, its rebuilds and its column
      builds (one accurate potential solve per DtN mode).
    - `block1`: the Krylov solves of the `Real` block on the delegating path
      (none under `dtn_schur_ainvb`, where dense factors solve block 1).
    - `late_setup_s`: the block-0 setup that ran outside the outer
      preconditioner setup, inside an outer iteration.
    """
    return {
        "outer_pc": {"inits": 0, "init_s": 0.0, "updates": 0, "update_s": 0.0,
                     "applies": 0},
        "ainvb": {"builds": 0, "time_s": 0.0},
        "block0": {"solves": 0, "its": 0, "its_min": None, "its_max": 0,
                   "capped": 0, "max_it": None, "time_s": 0.0, "setup_s": 0.0,
                   "setup_u_s": 0.0, "setup_psi_s": 0.0, "build_solves": 0,
                   "build_its": 0, "build_capped": 0, "rtol": None,
                   "reduction_sum": 0.0, "reduction_min": None,
                   "reduction_max": None, "capped_history": []},
        "condensed": {"inits": 0, "init_s": 0.0, "updates": 0,
                      "reassemblies": 0, "update_s": 0.0},
        "lowrank": {"inits": 0, "init_s": 0.0, "rebuilds": 0, "rebuild_s": 0.0,
                    "column_builds": 0, "column_solves": 0, "columns_s": 0.0},
        "block1": {"krylov_solves": 0, "its": 0},
        "late_setup_s": 0.0,
    }


#: The paths of the time fields of `_new_counters`, reduced with MAX over the
#: ranks at the end of each solve.
_TIME_FIELDS = (("outer_pc", "init_s"), ("outer_pc", "update_s"),
                ("ainvb", "time_s"), ("block0", "time_s"),
                ("block0", "setup_s"), ("block0", "setup_u_s"),
                ("block0", "setup_psi_s"), ("condensed", "init_s"),
                ("condensed", "update_s"), ("lowrank", "init_s"),
                ("lowrank", "rebuild_s"), ("lowrank", "columns_s"),
                (None, "late_setup_s"))


def parse_log_view(text):
    """The event lines of a PETSc log view, per stage.

    Args:
      text: the text of `PETSc.Log.view`.

    Returns:
      `{stage name: {event: {"count", "time_s", "time_ratio", "memory_mb"}}}`.
      `time_s` is the largest time over the ranks, `time_ratio` the largest
      over the smallest. `memory_mb` holds the four memory columns of
      `-log_view_memory` (Malloc, EMalloc, MMalloc, RMI: memory summed over
      the ranks, MB) when the log view has them, else None.
    """
    stages, current = {}, None
    header = re.compile(r"^--- Event Stage \d+: (.+?)\s*$")
    for line in text.splitlines():
        match = header.match(line)
        if match:
            current = match.group(1)
            stages.setdefault(current, {})
            continue
        if current is None:
            continue
        tokens = line.split()
        # An event line has 21 columns, or 25 with the memory columns.
        if len(tokens) not in (21, 25):
            continue
        try:
            count = int(tokens[1])
            time_s = float(tokens[3])
            ratio = float(tokens[4])
        except ValueError:
            continue
        memory = None
        if len(tokens) == 25:
            try:
                memory = [float(v) for v in tokens[21:25]]
            except ValueError:
                memory = None
        stages[current][tokens[0]] = {"count": count, "time_s": time_s,
                                      "time_ratio": ratio,
                                      "memory_mb": memory}
    return stages


class RunDiagnostics:
    """The diagnostics of one run: files, stages, memory records and solve records.

    Collective: build it on every rank. The methods that gather
    (`memory`, `end_solve`, the end of `stage`) are collective too and must be
    called on every rank in the same order.

    Args:
      output_path: the directory of the output files.
      say: the log function of the run (rank 0 writes the log).
    """

    def __init__(self, output_path, say):
        self.say = say
        self.comm = COMM_WORLD
        self.rank = COMM_WORLD.rank
        self.output_path = output_path
        self.tic = time.time()
        self.host = MPI.Get_processor_name() or socket.gethostname()
        # One communicator per node, as `exp12_common.memory_now` uses: the
        # node total is what the memory limit of a node applies to.
        self.node_comm = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        self.node_leader = self.node_comm.bcast(COMM_WORLD.rank, root=0)
        self.files = {}
        if self.rank == 0:
            for key, name in (("memory", "diag_memory.jsonl"),
                              ("solves", "diag_solves.jsonl"),
                              ("stages", "diag_stages.jsonl"),
                              ("pc", "diag_pc.jsonl")):
                self.files[key] = open(os.path.join(output_path, name), "a")
        self.log_view_path = os.path.join(output_path, "petsc_log_view.txt")
        # The ownership of this rank; filled by `set_problem`.
        self.ownership = {"dofs_owned": None, "dofs_by_field": None,
                          "mantle_cells": None, "mantle_cells_halo": None,
                          "parent_cells": None, "parent_cells_halo": None,
                          "dofs_halo_by_field": None,
                          "neighbours_parent": None, "neighbours_mantle": None}
        # The PETSc log: the default handler, started here (the run sets no
        # `-log_view`). No `-log_view_memory`: see the module docstring.
        PETSc.Log.begin()
        self._petsc_stages = {}
        self._stage_stack = []
        # The label of the current point of the run, for the solve records
        # ("tape step 3", "replay", "taylor eps 0.5"), and the solve record
        # that the hooks fill (None between solves).
        self.label = "setup"
        self.current = None
        self._loose = _new_counters()
        self._in_outer_setup = 0
        self._in_ainvb_build = 0
        self._monitored = {}
        # Every outer preconditioner that is initialised, by weak reference.
        # One initialised `DtNTwoBlockSchurPC` is one preconditioner set (the
        # outer PC owns the cached complement and, through its fieldsplit,
        # the block-0 PC with its GAMG and low-rank PC). The number alive is
        # written into every memory record as `live_pc_sets`; a weak set
        # keeps nothing alive.
        self._outer_pcs = weakref.WeakSet()
        self._last_log_view = 0.0
        self._solve_index = 0
        # The start time, broadcast from rank 0, so every record and the node
        # sampler share one clock (`t_unix`).
        self.tic = self.comm.bcast(self.tic, root=0)
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.tic))
        self.say(f"  [diag] diagnostics on: started at unix {self.tic:.1f} "
                 f"({start}); files in {output_path} (diag_memory.jsonl, "
                 "diag_solves.jsonl, diag_stages.jsonl, diag_pc.jsonl, "
                 "petsc_log_view.txt)")

    # -- files ---------------------------------------------------------------
    def _write(self, key, data):
        """One JSON line into a diagnostics file (rank 0), flushed at once."""
        if self.rank == 0:
            from exp12_common import plain_json
            self.files[key].write(json.dumps(plain_json(data)) + "\n")
            self.files[key].flush()

    def write_log_view(self):
        """Write the full PETSc log view of the run so far. Collective.

        Returns:
          The text on rank 0, None elsewhere.
        """
        tmp = self.log_view_path + ".tmp"
        viewer = PETSc.Viewer().createASCII(tmp, comm=self.comm)
        PETSc.Log.view(viewer)
        viewer.destroy()
        self._last_log_view = time.time()
        text = None
        if self.rank == 0:
            # Replace the file in one step, so a kill during the write keeps
            # the previous view.
            os.replace(tmp, self.log_view_path)
            with open(self.log_view_path) as handle:
                text = handle.read()
        return text

    # -- the ownership of each rank -------------------------------------------
    def set_problem(self, parent, mantle, solution):
        """Record the owned cells and degrees of freedom of this rank.

        Args:
          parent: the parent mesh.
          mantle: the mantle mesh.
          solution: the solution `Function` of the coupled solver.
        """
        by_field, halo_by_field = [], []
        for sub in solution.subfunctions:
            with sub.dat.vec_ro as vec:
                by_field.append(int(vec.getLocalSize()))
            # The halo values: the local array with halos minus the owned
            # array (values, not nodes: a vector field counts 3 per node).
            try:
                halo_by_field.append(int(sub.dat.data_ro_with_halos.size
                                         - sub.dat.data_ro.size))
            except Exception:  # noqa: BLE001
                halo_by_field.append(None)
        with solution.dat.vec_ro as vec:
            owned = int(vec.getLocalSize())
        self.ownership = {
            "dofs_owned": owned, "dofs_by_field": by_field,
            "mantle_cells": int(mantle.cell_set.size),
            "mantle_cells_halo": int(mantle.cell_set.total_size
                                     - mantle.cell_set.size),
            "parent_cells": int(parent.cell_set.size),
            "parent_cells_halo": int(parent.cell_set.total_size
                                     - parent.cell_set.size),
            "dofs_halo_by_field": halo_by_field,
            "neighbours_parent": _neighbour_ranks(parent),
            "neighbours_mantle": _neighbour_ranks(mantle)}

    # -- memory ----------------------------------------------------------------
    def memory(self, label, extra=None):
        """Record the memory of every rank now, and log a summary. Collective.

        Writes one line per rank into `diag_memory.jsonl`, and one log line
        with the mean, median, p90, p99 and largest rank, the largest node,
        and the ranks above twice the mean with their host, degrees of
        freedom, cells, halo cells and neighbour ranks.

        Args:
          label: the stage of the run.
          extra: a flat dict of numbers of this rank (the preconditioner
            make-up), merged into its row. With it, rank 0 also writes the
            summary over the ranks into `diag_pc.jsonl` (`_pc_summary`).
        """
        rss, peak = _rss_gb(), _status_gb("VmHWM")
        available = (_meminfo_gb("MemAvailable") if self.node_comm.rank == 0
                     else None)
        now = time.time()
        row = {"stage": label, "t_s": now - self.tic, "t_unix": now,
               "rank": self.rank, "host": self.host,
               "node_leader": self.node_leader, "rss_gb": rss,
               "peak_gb": peak, "node_mem_available_gb": available,
               "live_pc_sets": self.live_pc_sets(),
               **self.ownership, **(extra or {})}
        rows = self.comm.gather(row, root=0)
        if self.rank != 0:
            return
        for r in rows:
            self._write("memory", r)
        values = np.array([r["rss_gb"] for r in rows])
        peaks = np.array([r["peak_gb"] for r in rows])
        nodes = {}
        for r in rows:
            nodes.setdefault(r["host"], 0.0)
            nodes[r["host"]] += r["rss_gb"]
        node_host = max(nodes, key=nodes.get)
        avail = [r["node_mem_available_gb"] for r in rows
                 if r["node_mem_available_gb"] is not None]
        mean = float(values.mean())
        largest = int(values.argmax())
        p50, p90, p99 = np.percentile(values, [50, 90, 99])
        line = (f"  [diag] memory {label}: rank mean {mean:.2f}, median "
                f"{p50:.2f}, p90 {p90:.2f}, p99 {p99:.2f}, max {values.max():.2f} "
                f"GB (rank {largest}, {rows[largest]['host']}); peak max "
                f"{peaks.max():.2f} GB (rank {int(peaks.argmax())}); node max "
                f"{nodes[node_host]:.0f} GB ({node_host}), node mean "
                f"{np.mean(list(nodes.values())):.0f} GB")
        if avail:
            line += f", MemAvailable min {min(avail):.0f} GB"
        sets = [r["live_pc_sets"] for r in rows]
        line += f"; live PC sets {min(sets)}-{max(sets)}"
        self.say(line)
        heavy = [r for r in rows if r["rss_gb"] > 2.0 * mean]
        if heavy:
            heavy.sort(key=lambda r: -r["rss_gb"])

            def mean_of(key):
                found = [r[key] for r in rows if r[key] is not None]
                return np.mean(found) if found else float("nan")
            text = "; ".join(
                f"{r['rank']} {r['rss_gb']:.2f} GB {r['host']} "
                f"{r['dofs_owned']} dofs {r['mantle_cells']} mantle "
                f"(+{r['mantle_cells_halo']} halo) {r['parent_cells']} parent "
                f"(+{r['parent_cells_halo']} halo) cells, "
                f"{r['neighbours_parent']} neighbours" for r in heavy[:10])
            more = f" (and {len(heavy) - 10} more)" if len(heavy) > 10 else ""
            self.say(f"  [diag]   {len(heavy)} ranks above 2x the mean: {text}"
                     f"{more}; mean rank {mean_of('dofs_owned'):.0f} dofs "
                     f"{mean_of('mantle_cells'):.0f} mantle "
                     f"{mean_of('parent_cells'):.0f} parent cells")
        if extra is not None:
            self._pc_summary(label, rows, sorted(extra))

    def _pc_summary(self, label, rows, keys):
        """The summary over the ranks of the preconditioner make-up (rank 0).

        For each field: mean, max and the rank of the max. The full rows of
        the ranks whose resident size or whose `pc_mat_mem_mb` is above
        twice the mean go into the record as they are. One line into
        `diag_pc.jsonl` and one log line.

        Args:
          label: the stage of the run.
          rows: the gathered rows of all ranks.
          keys: the fields of the make-up.
        """
        summary = {}
        for key in keys:
            # A field that is not a number (the text of a make-up error) has
            # no summary; it stays in the rows of `diag_memory.jsonl`.
            try:
                values = np.array([r.get(key) if r.get(key) is not None
                                   else np.nan for r in rows], dtype=float)
            except (TypeError, ValueError):
                continue
            if np.all(np.isnan(values)):
                continue
            at = int(np.nanargmax(values))
            summary[key] = {"mean": float(np.nanmean(values)),
                            "max": float(values[at]), "rank": rows[at]["rank"]}
        rss_mean = np.mean([r["rss_gb"] for r in rows])
        pc_mean = summary.get("pc_mat_mem_mb", {}).get("mean", np.inf)
        heavy = [r for r in rows if r["rss_gb"] > 2.0 * rss_mean
                 or (r.get("pc_mat_mem_mb") or 0.0) > 2.0 * pc_mean]
        self._write("pc", {"stage": label, "t_unix": time.time(),
                           "summary": summary, "heavy_ranks": heavy[:50],
                           "heavy_count": len(heavy)})
        shown = []
        for key in ("pc_mat_mem_mb", "cond_S_uu_mem_mb", "gamg_u_levels",
                    "gamg_u_coarse_mem_mb", "gamg_psi_coarse_mem_mb",
                    "lowrank_columns", "lowrank_z_local", "ainvb_columns",
                    "ainvb_z_mem_mb"):
            if key in summary:
                v = summary[key]
                shown.append(f"{key} {v['mean']:.4g}/{v['max']:.4g} (rank "
                             f"{v['rank']})")
        self.say(f"  [diag]   preconditioner make-up {label} (mean/max): "
                 + ", ".join(shown) + f"; {len(heavy)} ranks above 2x the "
                 "mean (full rows in diag_pc.jsonl)")

    # -- PETSc stages ------------------------------------------------------------
    @contextlib.contextmanager
    def stage(self, name, label=None):
        """A PETSc log stage around a part of the run. Collective.

        At the end of the stage the full log view is written again, and the
        events of the stage go into `diag_stages.jsonl` and into a short log
        line (the largest times over the ranks).

        Args:
          name: the PETSc stage name (`exp12_...`, no spaces).
          label: the label of the solve records in the stage (default: the
            name without the `exp12_` prefix).
        """
        if name not in self._petsc_stages:
            self._petsc_stages[name] = PETSc.Log.Stage(name)
        petsc_stage = self._petsc_stages[name]
        previous = self.label
        self.label = label or name.replace("exp12_", "")
        tic = time.time()
        petsc_stage.push()
        self._stage_stack.append(name)
        try:
            yield
        except BaseException:
            # This rank leaves the stage by an exception. The local state is
            # restored, but the collective summary of `_end_stage` (the log
            # view) is skipped: the other ranks can still be inside the
            # stage, and a collective here would turn the abort into a hang
            # (the same reason as in `solve`).
            petsc_stage.pop()
            self._stage_stack.pop()
            self.label = previous
            raise
        # A normal exit of the stage, the same on every rank.
        petsc_stage.pop()
        self._stage_stack.pop()
        self.label = previous
        self._end_stage(name, time.time() - tic)

    def _end_stage(self, name, wall):
        """Write the log view and the summary of the stage `name`. Collective."""
        self._flush_loose(name)
        text = self.write_log_view()
        if self.rank != 0:
            return
        events = parse_log_view(text).get(name, {})
        self._write("stages", {"stage": name, "wall_s": wall,
                               "t_s": time.time() - self.tic,
                               "t_unix": time.time(),
                               "events": events})
        shown = [e for e in SUMMARY_EVENTS if e in events]
        # The largest other events by time, so that an unexpected cost shows.
        others = sorted((e for e in events if e not in SUMMARY_EVENTS),
                        key=lambda e: -events[e]["time_s"])[:6]
        parts = [f"{e} {events[e]['time_s']:.1f} s/{events[e]['count']}"
                 for e in shown + others]
        self.say(f"  [diag] PETSc stage {name}: wall {wall:.1f} s; max over "
                 f"ranks (time/count, nested events overlap): "
                 + ", ".join(parts))

    def _flush_loose(self, stage_name):
        """Write the preconditioner work that fell outside any solve, if any."""
        loose, self._loose = self._loose, _new_counters()
        if loose == _new_counters():
            return
        self._write("solves", {"stage": stage_name, "kind": "outside solves",
                               "counters": loose})

    # -- solves ----------------------------------------------------------------
    def counters(self):
        """The counters the hooks fill now: the current solve, or the loose set."""
        return self.current["counters"] if self.current is not None else self._loose

    def _monitor(self, snes):
        """Attach the Newton-step monitor to `snes` once (Newton solvers only)."""
        if snes is None or id(snes) in self._monitored:
            return
        # `ksponly` evaluates one more residual when a monitor is set.
        if snes.getType() != "newtonls":
            self._monitored[id(snes)] = snes
            return
        diag = self

        def newton_monitor(snes_, its, fnorm):
            record = diag.current
            if record is None or its == 0:
                return
            total = snes_.getLinearSolveIterations()
            record["outer_per_newton"].append(total - record["_outer_seen"])
            record["_outer_seen"] = total
        snes.setMonitor(newton_monitor)
        # Keep the object, so that its id stays unique.
        self._monitored[id(snes)] = snes

    def forget(self, snes):
        """Drop the monitored wrapper of `snes`, so this object holds no reference to it."""
        self._monitored.pop(id(snes), None)

    def live_pc_sets(self):
        """The number of initialised outer preconditioners alive on this rank."""
        return sum(1 for obj in list(self._outer_pcs)
                   if getattr(obj, "initialized", False))

    @contextlib.contextmanager
    def solve(self, kind, step, snes):
        """One solve record around a nonlinear (or linear adjoint) solve. Collective.

        Args:
          kind: `forward` (taped), `replay` or `adjoint`.
          step: the step of the march.
          snes: the SNES of the solver that runs the solve, for its counts.
        """
        if self.current is not None:
            # A nested call (two wrappers around one solve): the outer one
            # records it.
            yield
            return
        self._monitor(snes)
        self._solve_index += 1
        self.current = {"stage": self.label, "kind": kind, "step": step,
                        "index": self._solve_index,
                        "prefix": snes.getOptionsPrefix() if snes else None,
                        "outer_per_newton": [], "_outer_seen": 0,
                        "counters": _new_counters()}
        events = self._event_times(("SNESFunctionEval", "SNESJacobianEval"))
        tic = time.perf_counter()
        try:
            yield
        except BaseException:
            # This rank leaves the solve by an exception. The other ranks can
            # still be inside the solve (a Python error in a PETSc callback
            # can come on the ranks of one node only), so this rank must not
            # enter the collectives of `_end_solve`: an `Allreduce` here
            # would wait for ranks that wait in the solve, and the job would
            # hang in place of an abort. The record is dropped, and the
            # exception goes up to `main`, where the rank exits and mpiexec
            # stops the job.
            self.current = None
            raise
        # A normal exit of the solve on this rank. Every rank gets here after
        # the same solve, so the collectives of `_end_solve` match.
        wall = time.perf_counter() - tic
        record, self.current = self.current, None
        self._end_solve(record, wall, snes, events)

    def _event_times(self, names):
        """The time of each named PETSc event in the current stage, this rank."""
        stage = (self._petsc_stages[self._stage_stack[-1]].id
                 if self._stage_stack else 0)
        return {name: PETSc.Log.Event(name).getPerfInfo(stage)["time"]
                for name in names}

    def _end_solve(self, record, wall, snes, events_before):
        """Finish a solve record: counts, times (max over ranks), file and log. Collective."""
        record.pop("_outer_seen", None)
        counters = record["counters"]
        events = self._event_times(tuple(events_before))
        record["residual_s"] = events["SNESFunctionEval"] - events_before["SNESFunctionEval"]
        record["jacobian_s"] = events["SNESJacobianEval"] - events_before["SNESJacobianEval"]
        if snes is not None:
            record["newton"] = int(snes.getIterationNumber())
            record["outer_total"] = int(snes.getLinearSolveIterations())
            record["snes_reason"] = int(snes.getConvergedReason())
            record["ksp_reason_last"] = int(snes.ksp.getConvergedReason())
            if not record["outer_per_newton"]:
                record["outer_per_newton"] = [record["outer_total"]]
        # The times: the largest over the ranks, in one reduction.
        local = [wall, record["residual_s"], record["jacobian_s"]]
        for group, key in _TIME_FIELDS:
            local.append(counters[key] if group is None else counters[group][key])
        local = np.array(local, dtype=float)
        largest = np.empty_like(local)
        self.comm.Allreduce(local, largest, op=MPI.MAX)
        record["wall_s"], record["residual_s"], record["jacobian_s"] = largest[:3]
        for (group, key), value in zip(_TIME_FIELDS, largest[3:]):
            if group is None:
                counters[key] = float(value)
            else:
                counters[group][key] = float(value)
        outer_pc = counters["outer_pc"]
        record["pc_setup_s"] = (outer_pc["init_s"] + outer_pc["update_s"]
                                + counters["late_setup_s"])
        record["rest_s"] = record["wall_s"] - record["pc_setup_s"]
        # Memory after the solve, reduced (no per-rank lines per solve).
        rss = _rss_gb()
        node = self.node_comm.allreduce(rss, op=MPI.SUM)
        record["memory_rank_mean_gb"] = self.comm.allreduce(rss, op=MPI.SUM) / self.comm.size
        record["memory_rank_max_gb"] = self.comm.allreduce(rss, op=MPI.MAX)
        record["memory_node_max_gb"] = self.comm.allreduce(node, op=MPI.MAX)
        record["t_s"] = time.time() - self.tic
        record["t_unix"] = time.time()
        if self.rank == 0:
            self._write("solves", record)
            self.say(self.solve_line(record))
        # A long stage keeps a recent profile in case the job is killed. The
        # log view is collective, so rank 0 decides and every rank follows:
        # with each rank reading its own clock, a rank that left the last
        # collective a few ms later could skip the view the others enter, and
        # the job would hang.
        due = self.comm.bcast(
            time.time() - self._last_log_view > 300.0 if self.rank == 0
            else None, root=0)
        if due:
            self.write_log_view()

    @staticmethod
    def solve_line(record):
        """The two log lines of one solve record (the solve table)."""
        c = record["counters"]
        b0, ab, lr, cd = c["block0"], c["ainvb"], c["lowrank"], c["condensed"]
        outer = "+".join(str(n) for n in record.get("outer_per_newton", []))
        its = (f"its {b0['its_min']}-{b0['its_max']} mean "
               f"{b0['its'] / b0['solves']:.1f}" if b0["solves"] else "its -")
        if b0["solves"] and b0["reduction_min"] is not None:
            its += (f" reduction {b0['reduction_min']:.1e}-"
                    f"{b0['reduction_max']:.1e} (rtol {b0['rtol']:.0e})")
        capped_text = ""
        if b0["capped_history"]:
            first = b0["capped_history"][0]
            capped_text = (f" [first capped: r10 {first['r10']:.1e} r25 "
                           f"{first['r25']:.1e} last {first['rlast']:.1e}]")
        line1 = (f"  [diag] {record['stage']:>14s} {record['kind']:>7s} step "
                 f"{record['step']!s:>3s}: wall {record['wall_s']:7.1f} s | "
                 f"newton {record.get('newton', '?')} outer "
                 f"{record.get('outer_total', '?')} ({outer}) | block-0 "
                 f"{b0['solves']} solves {its} capped {b0['capped']} "
                 f"(cap {b0['max_it']}){capped_text}, {b0['build_solves']} in "
                 "builds | "
                 f"block-1 Krylov {c['block1']['krylov_solves']} solves "
                 f"{c['block1']['its']} its")
        line2 = (f"  [diag] {'':>14s} {'':>7s}           setup "
                 f"{record['pc_setup_s']:7.1f} s (outer PC init "
                 f"{c['outer_pc']['init_s']:.1f} update "
                 f"{c['outer_pc']['update_s']:.1f}, late {c['late_setup_s']:.1f}) "
                 f"| ainvb {ab['builds']} builds {ab['time_s']:.1f} s | block-0 "
                 f"setup {b0['setup_s']:.1f} s (u {b0['setup_u_s']:.1f}, psi "
                 f"{b0['setup_psi_s']:.1f}), condensed init {cd['inits']} "
                 f"{cd['init_s']:.1f} s, reassembly {cd['reassemblies']} "
                 f"{cd['update_s']:.1f} s | low-rank init {lr['inits']} "
                 f"{lr['init_s']:.1f} s, columns {lr['column_builds']}x"
                 f"{lr['column_solves']} {lr['columns_s']:.1f} s | block-0 "
                 f"applications {b0['time_s']:.1f} s | rest "
                 f"{record['rest_s']:.1f} s (residual {record['residual_s']:.1f}, "
                 f"jacobian {record['jacobian_s']:.1f}) | memory mean "
                 f"{record['memory_rank_mean_gb']:.2f} max "
                 f"{record['memory_rank_max_gb']:.2f} node "
                 f"{record['memory_node_max_gb']:.0f} GB")
        return line1 + "\n" + line2

    # -- the hooks ---------------------------------------------------------------
    def install(self):
        """Wrap the preconditioner and solve-block methods at class level.

        Collective in effect: every rank installs the same wrappers. Call
        once, before the first solve. See the module docstring.
        """
        from gadopt import preconditioners as pcs
        from gadopt import dtn_coupled_adjoint as dca
        diag = self

        def timed(cls, name, group, count_key, time_key, outer_setup=False,
                  late=False):
            """Wrap `cls.name` to count calls and add its time to a counter."""
            method = getattr(cls, name)

            def wrapper(obj, *args, **kwargs):
                tic = time.perf_counter()
                if outer_setup:
                    diag._in_outer_setup += 1
                try:
                    return method(obj, *args, **kwargs)
                finally:
                    if outer_setup:
                        diag._in_outer_setup -= 1
                    elapsed = time.perf_counter() - tic
                    counters = diag.counters()
                    if count_key:
                        counters[group][count_key] += 1
                    counters[group][time_key] += elapsed
                    if late and not diag._in_outer_setup:
                        counters["late_setup_s"] += elapsed
            wrapper.__wrapped__ = method
            setattr(cls, name, wrapper)

        # The outer preconditioner. Its first `initialize` on each solver is
        # the first preconditioner build of that solver (the cached
        # complement is built inside it), so the memory is recorded before
        # and after it.
        two_block = pcs.DtNTwoBlockSchurPC
        init_two_block = two_block.initialize

        def two_block_initialize(obj, pc):
            diag._outer_pcs.add(obj)
            prefix = pc.getOptionsPrefix() or ""
            diag.memory(f"before outer PC build {prefix} ({diag.label})")
            diag._in_outer_setup += 1
            tic = time.perf_counter()
            try:
                result = init_two_block(obj, pc)
            finally:
                # Local counters only: safe on a rank that raised.
                diag._in_outer_setup -= 1
                counters = diag.counters()
                counters["outer_pc"]["inits"] += 1
                counters["outer_pc"]["init_s"] += time.perf_counter() - tic
            # The memory record is collective, so it runs only after a
            # normal exit of the build. A rank that raised above skips it,
            # and does not wait for ranks that are still in the build.
            diag.memory(f"after outer PC build {prefix} ({diag.label})",
                        extra=_safe_makeup(_twoblock_makeup, obj))
            return result
        setattr(two_block, "initialize", two_block_initialize)
        timed(two_block, "update", "outer_pc", "updates", "update_s",
              outer_setup=True)
        apply_two_block = two_block.apply

        def two_block_apply(obj, pc, x, y):
            out = apply_two_block(obj, pc, x, y)
            counters = diag.counters()
            counters["outer_pc"]["applies"] += 1
            # The delegating path solves block 1 with a Krylov method.
            if getattr(obj, "ainvb", None) is None:
                try:
                    ksp_real = obj.pc.getFieldSplitSchurGetSubKSP()[1]
                    if ksp_real.getType() != PETSc.KSP.Type.PREONLY:
                        counters["block1"]["krylov_solves"] += 1
                        counters["block1"]["its"] += ksp_real.getIterationNumber()
                except PETSc.Error:
                    pass
            return out
        setattr(two_block, "apply", two_block_apply)

        # The cached complement: n block-0 solves per build.
        cache = pcs._SchurAinvBCache
        build = cache.build

        def ainvb_build(obj, pc):
            diag._in_ainvb_build += 1
            tic = time.perf_counter()
            try:
                return build(obj, pc)
            finally:
                diag._in_ainvb_build -= 1
                counters = diag.counters()
                counters["ainvb"]["builds"] += 1
                counters["ainvb"]["time_s"] += time.perf_counter() - tic
        setattr(cache, "build", ainvb_build)

        # Block 0.
        condensed = pcs.CondensedBlockPC
        timed(condensed, "initialize", "condensed", "inits", "init_s", late=True)
        update_condensed = condensed.update

        def condensed_update(obj, pc):
            before = getattr(obj, "assembly_count", 0)
            tic = time.perf_counter()
            try:
                return update_condensed(obj, pc)
            finally:
                elapsed = time.perf_counter() - tic
                counters = diag.counters()
                counters["condensed"]["updates"] += 1
                counters["condensed"]["update_s"] += elapsed
                if getattr(obj, "assembly_count", 0) != before:
                    counters["condensed"]["reassemblies"] += 1
                if not diag._in_outer_setup:
                    counters["late_setup_s"] += elapsed
        setattr(condensed, "update", condensed_update)
        apply_condensed = condensed.apply

        def condensed_apply(obj, pc, x, y):
            ksp = obj.condensed_ksp
            # The setup of the fieldsplit PC of the condensed KSP and of its
            # two split KSPs, timed here; the solve would run the same setup
            # first (module docstring). `PCSetUp` and not `KSPSetUp` on the
            # condensed KSP: `KSPSetUp` before the first solve makes the
            # FGMRES work vectors from the nest matrix, not from the nest
            # right-hand side, and `VecCopy` then refuses them ("Nest vector
            # argument 2 not setup", L0 check job 182107720).
            tic = time.perf_counter()
            ksp.getPC().setUp()
            t_u = t_psi = 0.0
            try:
                subs = ksp.getPC().getFieldSplitSubKSP()
            except PETSc.Error:
                subs = []
            if len(subs) == 2:
                t0 = time.perf_counter()
                subs[0].setUp()
                t1 = time.perf_counter()
                subs[1].setUp()
                t_u, t_psi = t1 - t0, time.perf_counter() - t1
            setup = time.perf_counter() - tic
            # The residual history of the inner FGMRES, kept for one solve
            # at a time (`reset=True`). It records norms only and changes no
            # iteration.
            if not getattr(obj, "_diag_history", False):
                ksp.setConvergenceHistory(reset=True)
                obj._diag_history = True
            out = apply_condensed(obj, pc, x, y)
            elapsed = time.perf_counter() - tic
            its = int(ksp.getIterationNumber())
            reason = int(ksp.getConvergedReason())
            rtol, _, _, max_it = ksp.getTolerances()
            max_it = int(max_it)
            history = np.asarray(ksp.getConvergenceHistory(), dtype=float)
            # -3 is KSP_DIVERGED_ITS: the inner FGMRES stopped at its cap
            # (the preset counts no block-0 cap as convergence).
            capped = its >= max_it or reason == -3
            counters = diag.counters()
            b0 = counters["block0"]
            b0["solves"] += 1
            b0["its"] += its
            b0["its_min"] = its if b0["its_min"] is None else min(b0["its_min"], its)
            b0["its_max"] = max(b0["its_max"], its)
            b0["capped"] += int(capped)
            b0["max_it"] = max_it
            b0["time_s"] += elapsed
            b0["setup_s"] += setup
            b0["setup_u_s"] += t_u
            b0["setup_psi_s"] += t_psi
            b0["rtol"] = float(rtol)
            if len(history) > 1 and history[0] > 0.0:
                # The reduction of the residual norm; the norms are global,
                # so every rank holds the same history.
                scaled = history / history[0]
                reduction = float(scaled[-1])
                b0["reduction_sum"] += reduction
                b0["reduction_min"] = (reduction if b0["reduction_min"] is None
                                       else min(b0["reduction_min"], reduction))
                b0["reduction_max"] = (reduction if b0["reduction_max"] is None
                                       else max(b0["reduction_max"], reduction))
                if capped and len(b0["capped_history"]) < 20:
                    b0["capped_history"].append({
                        "its": its,
                        "r10": float(scaled[min(10, len(scaled) - 1)]),
                        "r25": float(scaled[min(25, len(scaled) - 1)]),
                        "rlast": reduction})
            if diag._in_ainvb_build:
                b0["build_solves"] += 1
                b0["build_its"] += its
                b0["build_capped"] += int(capped)
            if not diag._in_outer_setup:
                counters["late_setup_s"] += setup
            # The first setup of each block-0 preconditioner: its memory.
            if not getattr(obj, "_diag_setup_recorded", False):
                obj._diag_setup_recorded = True
                diag.memory(f"after first block-0 setup "
                            f"{pc.getOptionsPrefix() or ''} ({diag.label})",
                            extra=_safe_makeup(_block0_makeup, obj, ksp))
            return out
        setattr(condensed, "apply", condensed_apply)

        # The low-rank DtN columns on the potential split.
        low_rank = pcs.LowRankPotentialPC
        timed(low_rank, "initialize", "lowrank", "inits", "init_s")
        timed(low_rank, "_rebuild", "lowrank", "rebuilds", "rebuild_s")
        build_columns = low_rank._build_columns

        def lowrank_columns(obj):
            tic = time.perf_counter()
            try:
                return build_columns(obj)
            finally:
                counters = diag.counters()
                counters["lowrank"]["column_builds"] += 1
                counters["lowrank"]["column_solves"] += len(getattr(obj, "_columns", []))
                counters["lowrank"]["columns_s"] += time.perf_counter() - tic
        setattr(low_rank, "_build_columns", lowrank_columns)

        # The replay and adjoint solves of the coupled solve block.
        if dca.CACHED_SOLVER_BLOCKS:
            block_cls = dca.CachedCoupledLowRankDtNSolveBlock
            pairs = (("recompute_component", "replay", "forward_cache"),
                     ("prepare_evaluate_adj", "adjoint", "adjoint_cache"))
        else:
            block_cls = dca.CoupledLowRankDtNSolveBlock
            pairs = (("_forward_solve", "replay", None),
                     ("_adjoint_solve", "adjoint", None))
        for name, kind, cache_name in pairs:
            method = getattr(block_cls, name)

            def wrapped(obj, *args, _method=method, _kind=kind,
                        _cache=cache_name, **kwargs):
                snes = None
                if _cache is not None:
                    solver = getattr(getattr(obj, _cache, None), "solver", None)
                    snes = getattr(solver, "snes", None)
                with diag.solve(_kind, getattr(obj, "giamip_step", "?"), snes):
                    return _method(obj, *args, **kwargs)
            setattr(block_cls, name, wrapped)
        self.say("  [diag] hooks installed on DtNTwoBlockSchurPC, "
                 "_SchurAinvBCache, CondensedBlockPC, LowRankPotentialPC and "
                 f"{block_cls.__name__}")

    def close(self):
        """Write the final log view and close the files. Collective."""
        self.write_log_view()
        for handle in self.files.values():
            handle.close()
