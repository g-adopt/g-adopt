"""Divergence-free near-nullspace enrichment for nearly-incompressible elasticity.

`solenoidal_modes` and `near_incompressible_modes` build the basis that
`NearlyIncompressibleAssembledPC` hands to GAMG. These fast tests check
that the combined basis is orthonormal and a strict superset of the
rigid-body modes, and that the mode builders put nothing on the tape.

The `condensation-pc` entry of `tests/weak_bc_gia` (driver
`gia_condensation.py`) checks the rest: the generated fields are
divergence-free and the filter drops those that are not on a curved mesh,
the preconditioner lowers the CG count against rigid modes alone without
saturating GAMG, and the patch multigrid parameters converge.
"""

import firedrake as fd
import numpy as np
import pytest

from gadopt.nullspaces import (
    near_incompressible_modes,
    rigid_body_modes,
    solenoidal_modes,
)


def _box(dim, n=8):
    return fd.UnitSquareMesh(n, n) if dim == 2 else fd.UnitCubeMesh(n, n, n)


def _gram(basis):
    vecs = basis._petsc_vecs
    n = len(vecs)
    G = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = vecs[i].dot(vecs[j])
    return G


@pytest.mark.parametrize("build", [
    pytest.param(lambda V: rigid_body_modes(V, rotational=True,
                                            translations=[0, 1]),
                 id="rigid_body_modes"),
    pytest.param(lambda V: solenoidal_modes(V, max_degree=1, divfree_tol=1e-8),
                 id="solenoidal_modes"),
    pytest.param(lambda V: near_incompressible_modes(V, max_degree=1),
                 id="near_incompressible_modes"),
])
def test_mode_builders_leave_the_tape_alone(build):
    # Every mode is auxiliary -- handed to PETSc as a (near-)nullspace, never
    # part of a residual or a functional -- so building one must add nothing to
    # the tape. Unguarded, each of the 2-D degree-1 set left dead-end
    # `AssembleBlock`s behind: 3 from the rigid interpolations (Firedrake lowers
    # `interpolate` to an assembly, so `rigid_body_modes` leaked this way long
    # before the solenoidal work), 3 more from the solenoidal ones, and 6 from
    # the divergence filter's norms. They cannot corrupt a gradient, but they
    # grow the tape and are re-run on every replay.
    #
    # The production path is safe regardless (Firedrake wraps the whole solve in
    # `stop_annotating`, and PETSc only sets a PC up from inside `KSPSolve`), so
    # what this pins is the *driver* path: user code that builds a basis itself
    # while a tape is live, which is how every `nullspace=` argument is made.
    from firedrake.adjoint import (continue_annotation, get_working_tape,
                                   pause_annotation)

    V = fd.VectorFunctionSpace(_box(2), "CG", 2)
    continue_annotation()
    try:
        tape = get_working_tape()
        before = len(tape.get_blocks())
        build(V)
        assert len(tape.get_blocks()) == before
    finally:
        pause_annotation()


@pytest.mark.parametrize("dim", [2, 3])
def test_near_incompressible_modes_orthonormal_and_superset(dim):
    V = fd.VectorFunctionSpace(_box(dim), "CG", 2)
    n_rigid = len(rigid_body_modes(
        V, rotational=True, translations=list(range(dim)))._petsc_vecs)

    basis = near_incompressible_modes(V, max_degree=1)
    G = _gram(basis)
    assert np.allclose(G, np.eye(G.shape[0]), atol=1e-9)   # orthonormal

    # Degree-1 solenoidal fields span the complete linear divergence-free space:
    # 5 modes in 2-D, 11 in 3-D, strictly more than the rigid modes alone.
    n_modes = G.shape[0]
    assert n_modes == (5 if dim == 2 else 11)
    assert n_modes > n_rigid
