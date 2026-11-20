"""A Quadrature state of the GIA solver restarted on another number of ranks.

`gadopt.gia_state.write_state` stores the Quadrature history with the
physical coordinates of its quadrature points, and `read_state` pairs the
stored values with the local points by position, so a restart may run on
another partition than the run that wrote the file. The test writes on one
rank and restarts on two, and the other way round, inside one two-rank
process, one rank working on `COMM_SELF` alone.

The test has its own file because it carries a `parallel` marker: with one
such test in a file, pytest-mpi refuses every unmarked test of that file
under an outer `mpiexec`, and `test_gia_solver.py` is run that way.
"""
from pathlib import Path

import pytest
from mpi4py import MPI

import gadopt  # noqa: F401  (gadopt before firedrake)
import firedrake as fd
import gia_cases as C
from gadopt.gia_state import read_state, write_state
from test_gia_solver import (FOUR, PARALLEL_RTOL, STATE_CASES, load_meshes,
                             relative, run)
from parallel_ranks import assert_marked_rank_count

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest


def _march_and_write(path, comm):
    """March the Quadrature state case two steps on `comm` and write its state.

    Returns:
      The name of the stored parent mesh.
    """
    meshes = C.annulus_meshes(comm)
    solver, _, _ = C.build_new(meshes, QUADRATURE_STATE_CASE,
                               dt=C.dt_function(meshes[1]),
                               nullspace="rigid_rotation")
    run(solver, None, FOUR[:2])
    with fd.CheckpointFile(str(path), "w", comm=comm) as chk:
        chk.save_mesh(meshes[0])
        write_state(chk, solver, idx=0)
    return meshes[0].name


def _restart_gap(path, name, comm):
    """Restore the written state on `comm` and compare with a run from rest.

    A solver on the mesh loaded from the file restores the state; a second
    solver on the same loaded mesh marches the same two steps from rest.
    Both take a third step.

    Returns:
      The largest relative difference of displacement, potential and
      history, after the restore and after the third step.
    """
    loaded = load_meshes(path, name, comm)

    def build():
        return C.build_new(loaded, QUADRATURE_STATE_CASE,
                           dt=C.dt_function(loaded[1]),
                           nullspace="rigid_rotation")

    restarted, u, psi = build()
    with fd.CheckpointFile(str(path), "r", comm=comm) as chk:
        restarted.restore(read_state(chk, restarted, idx=0))
    reference, u_ref, psi_ref = build()
    run(reference, None, FOUR[:2])
    gap = 0.0
    for _ in range(2):
        for a, b in [(u, u_ref), (psi, psi_ref),
                     (restarted.internal_variables, reference.internal_variables)]:
            gap = max(gap, relative(a, b))
        run(restarted, None, FOUR[2:3], first=2)
        run(reference, None, FOUR[2:3], first=2)
    return gap


#: The Quadrature case of `STATE_CASES`: BDF2, low-rank DtN, two Maxwell
#: elements, fluid core, centre of mass.
QUADRATURE_STATE_CASE = STATE_CASES[0]


@pytest.mark.parallel(nprocs=2)
def test_a_quadrature_state_restarts_on_another_number_of_ranks(request, tmp_path):
    """A Quadrature state written on one rank restarts on two, and the other way round.

    `write_state` stores the quadrature point coordinates of the history,
    and `read_state` pairs the values by position. The restarted run agrees
    with a run from rest on the same loaded mesh to `PARALLEL_RTOL`, after
    the restore and after one more step. Without the pairing the history
    differs by tens of percent. One rank works on `COMM_SELF` alone, and its
    result (or its error) is broadcast, so that every rank fails alike.
    """
    assert_marked_rank_count(request)
    comm = MPI.COMM_WORLD
    directory = Path(comm.bcast(str(tmp_path), root=0))

    def on_rank_zero(function, *args):
        result = None
        if comm.rank == 0:
            try:
                result = ("value", function(*args))
            except Exception as error:  # noqa: BLE001  (re-raised on every rank)
                result = ("error", error)
        kind, value = comm.bcast(result, root=0)
        if kind == "error":
            raise value
        return value

    one = directory / "state_written_on_one.h5"
    name = on_rank_zero(_march_and_write, one, MPI.COMM_SELF)
    gap = _restart_gap(one, name, comm)
    assert gap <= PARALLEL_RTOL, gap

    two = directory / "state_written_on_two.h5"
    name = _march_and_write(two, comm)
    comm.barrier()
    gap = on_rank_zero(_restart_gap, two, name, MPI.COMM_SELF)
    assert gap <= PARALLEL_RTOL, gap
