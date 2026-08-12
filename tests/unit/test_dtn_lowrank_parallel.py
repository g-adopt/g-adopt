"""`_assert_against_reference` must take the same branch on every rank.

WHAT WAS WRONG

    `_assert_against_reference` computed `scale` from rank-local `rows` and
    returned early when it was zero, in front of collective assemblies. A rank
    owning no degree of freedom on a DtN boundary took that early return and
    raced on to the next boundary's `assemble(1 * measure)` while every other
    rank waited in the reference assemblies. The build deadlocked with no output
    and no error. The `worst > rtol` comparison below it had the same shape and
    a worse consequence: a rank-local raise leaves the ranks that did not raise
    blocked in a collective for ever, so the assertion that protects the whole
    representation could not report in parallel at all.

    Measured 2026-08-12: a `GravitySolver(..., dtn_representation="lowrank")` on
    an unstructured gmsh annulus at 2 ranks hung until killed, while the same
    script at `"multiplier"` finished in one second.

HOW THIS IS TESTED, AND WHY NOT THROUGH A MESH

    The obvious test builds a mesh whose partition leaves some rank empty on a
    boundary. It was tried first and **it does not reproduce reliably**: on an
    unstructured annulus of inner radius 1.22 and outer 2.22 at 2 ranks, no rank
    owns zero dofs on either boundary, because an area-balanced cut across the
    ring is shorter than a cut around it, so both ranks touch both circles. A
    test resting on that would pass while exercising nothing, and the mesh that
    does reproduce is the demo's own annulus, which a unit test should not
    import.

    Every extruded mesh is worse still and must never be used here. Extrusion
    partitions laterally only, so every rank owns part of both the top and the
    bottom boundary and the early return is unreachable by construction. That
    covers `ExtrudedMesh` over `CircleManifoldMesh` or `CubedSphereMesh` - which
    is every other DtN mesh in this suite - and `firedrake.AnnulusMesh`, which
    is extruded despite the name. `test_dtn_lowrank_parity.py`'s
    `test_parallel_parity`, whose docstring says "an ownership mistake there is
    invisible in serial", is on exactly such a mesh and could not have caught
    this. The same defect ran clean through a 64-rank Gadi campaign for the same
    reason: partitioning decides whether it fires, not the rank count.

    So the divergence is produced directly instead. `rows` is zeroed on every
    rank but the first, which is precisely the state a rank with nothing on the
    boundary arrives in, and the function is called. That is deterministic, it
    needs no partition luck, and it exercises the exact branch that was wrong.

RUN
    python3 -m pytest tests/unit/test_dtn_lowrank_parallel.py
    # `mpi-pytest` re-launches the marked tests under MPI by itself. No
    # `mpiexec` wrapper is needed and CI's plain `pytest` runs them.
"""

import numpy as np
import pytest

import gadopt  # noqa: F401  (before firedrake)
import firedrake as fd  # noqa: E402
from gadopt import CylindricalDtN  # noqa: E402
from gadopt.dtn_lowrank import _assert_against_reference  # noqa: E402

RMIN, RMAX = 1.22, 2.22
QUAD_DEGREE = 12
SIDE, RTOL = "exterior", 1e-13


@pytest.fixture(scope="module")
def annulus():
    """Any mesh will do here: the partition is NOT what produces the divergence.

    An extruded one is used deliberately, to make the point that this test does
    not depend on ownership and therefore cannot go inert when a partitioner
    changes.
    """
    heights = list(np.diff(np.linspace(RMIN, RMAX, 5)))
    base = fd.CircleManifoldMesh(48, radius=RMIN, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=len(heights), layer_height=heights,
                           extrusion_type="radial")
    mesh.cartesian = False
    return mesh


def build_pieces(annulus, M=3):
    """`rows` assembled symbolically, plus everything the assertion needs.

    `rows` is built the way the reference is, so a clean call must find zero
    deviation. That is what makes the poisoned call below a real poison rather
    than a second source of error.
    """
    descriptor = CylindricalDtN(M)
    V = fd.FunctionSpace(annulus, "CG", 1)
    v = fd.TestFunction(V)
    measure = fd.ds_t(domain=annulus, degree=QUAD_DEGREE)
    X = fd.SpatialCoordinate(annulus)
    metadata = descriptor.mode_metadata(SIDE, RMAX)
    keys = [mode.key for mode in metadata]
    rows = np.array([
        np.asarray(fd.assemble(mode.expr * v * measure).dat.data_ro,
                   dtype=float)
        for mode in descriptor.modes(SIDE, RMAX, X)])
    return dict(rows=rows, keys=keys, descriptor=descriptor, side=SIDE,
                radius=RMAX, v=v, measure=measure, mesh=annulus,
                trace_degree=QUAD_DEGREE // 2, rtol=RTOL)


@pytest.mark.parallel(nprocs=2)
def test_clean_rows_pass_on_every_rank(annulus):
    """The positive control. Without it the poisoned call proves nothing.

    `rows` is the reference, so the deviation is zero and no rank raises. If
    this failed, the poisoned test below would be detecting the setup rather
    than the defect.
    """
    pieces = build_pieces(annulus)
    _assert_against_reference(**pieces)


@pytest.mark.parallel(nprocs=2)
def test_a_rank_with_empty_rows_does_not_strand_the_others(annulus):
    """The regression. Before the fix this HUNG; it did not fail.

    Rank 1's `rows` are zeroed, which is the state a rank owning nothing on the
    boundary arrives in. Unfixed, rank 1's local `scale` is 0, it returns
    early, and rank 0 blocks in the reference assemblies for ever.

    Fixed, `scale` is a global maximum, so both ranks enter the loop, both
    complete the collective assemblies, and both raise the same `RuntimeError`
    because zeroed rows really do disagree with the reference. The assertion
    stays unconditional, which `NOTES/poisson/HANDOVER-FAST-DTN.md` section 7
    requires: it is the only protection against `HDiv Trace` node placement
    moving under a Firedrake upgrade.

    **A hang has no exception to catch, so what is asserted here is that the
    process reaches the next line at all.** `mpi-pytest` surfaces a stuck
    subprocess as a failure rather than as a run that never ends.

    **THIS SUITE COVERS THE MECHANISM, NOT THE END-TO-END HANG.** What is
    proved here is that the branch is uniform across ranks, and that branch is
    the mechanism - an end-to-end deadlock is this branch and nothing else. It
    is not a whole solve on a mesh whose partition really produces the
    divergence. That evidence is
    `NOTES/fastdtn/repro_lowrank_deadlock.py`, which runs a `GravitySolver` on
    an unstructured gmsh annulus and hung at 2 ranks before the fix while
    completing at 1, 2 and 4 ranks after. It is deliberately kept out of the
    suite because its trigger is partitioner dependent and would go inert
    without saying so.
    """
    pieces = build_pieces(annulus)
    if fd.COMM_WORLD.rank != 0:
        pieces["rows"] = np.zeros_like(pieces["rows"])

    with pytest.raises(RuntimeError, match="disagrees with the symbolic one"):
        _assert_against_reference(**pieces)

    # Every rank raised, not just the poisoned one. Reaching this line on all
    # ranks is the whole point: it is what the early return made impossible.
    assert fd.COMM_WORLD.allreduce(1) == fd.COMM_WORLD.size


@pytest.mark.parallel(nprocs=2)
def test_every_rank_names_the_same_mode(annulus):
    """The error message must be the same everywhere.

    `worst_key` is rank-local, so before the fix each rank would have named
    whichever mode was worst in its own partition. A parallel run that raises
    two different diagnoses for one defect sends the reader after the wrong
    mode.
    """
    pieces = build_pieces(annulus)
    # Poison a DIFFERENT mode on each rank, so a rank-local `worst_key` would
    # disagree by construction.
    target = 0 if fd.COMM_WORLD.rank == 0 else len(pieces["keys"]) - 1
    pieces["rows"] = pieces["rows"].copy()
    pieces["rows"][target] *= 1.0 + (fd.COMM_WORLD.rank + 1) * 1e-3

    messages = []
    try:
        _assert_against_reference(**pieces)
    except RuntimeError as exc:
        messages.append(str(exc))
    gathered = fd.COMM_WORLD.allgather(messages)
    assert all(m for m in gathered), (
        f"not every rank raised: {[bool(m) for m in gathered]}")
    first = gathered[0][0]
    for other in gathered[1:]:
        assert other[0] == first, (
            "ranks disagree on the diagnosis:\n"
            f"  rank 0: {first}\n  other:  {other[0]}")
