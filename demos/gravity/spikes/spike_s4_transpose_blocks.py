"""S4 -- the two cross-mesh coupling blocks are transposes of each other.

    B1 = inner(rho0*grad(psi), w) * dx_mantle     test w on the SUBMESH,
                                                  trial psi on the PARENT
    B2 = rho0 * inner(u, grad(v)) * dx_mantle     test v on the PARENT,
                                                  trial u on the SUBMESH

If the monolithic Jacobian is to be symmetric (verification V1), these two must
satisfy B1 = B2^T exactly, not approximately.  This spike measures
max|B1 - B2^T| on an assembled mixed matrix.

Note both are written on `dx_mantle`, i.e. a submesh volume measure that
declares the parent -- that is the only measure on which both directions
assemble.
"""
import os
import sys

import numpy as np

import gadopt  # noqa: F401  -- import order, see SPIKE-RESULTS.md S2(d)
from firedrake import *
from firedrake.petsc import PETSc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_mesh  # noqa: E402


def pr(*a):
    PETSc.Sys.Print(*a)


def dense(mat, rows, cols):
    """Dense numpy copy of a PETSc submatrix given by row/col index arrays."""
    out = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        cidx, cval = mat.getRow(r)
        lookup = {c: v for c, v in zip(cidx, cval)}
        for j, c in enumerate(cols):
            out[i, j] = lookup.get(c, 0.0)
    return out


def main():
    if COMM_WORLD.size > 1:
        pr("S4 compares dense blocks and is written for 1 rank only; "
           "the forms themselves are exercised in parallel by S2.")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    msh = os.path.join(here, "spike_annulus.msh")
    if not os.path.exists(msh):
        spike_mesh.generate(msh)

    parent = Mesh(msh)
    sub = Submesh(parent, 2, spike_mesh.CELL_MANTLE)

    pr("=" * 72)
    pr("S4  transposed cross-mesh coupling blocks")
    pr("=" * 72)

    # A coarse space, so the dense extraction is cheap.
    V = VectorFunctionSpace(sub, "CG", 1)
    P = FunctionSpace(parent, "CG", 1)
    Z = MixedFunctionSpace([V, P])

    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))

    # A non-constant background density, so the test is not degenerate.
    DG0 = FunctionSpace(sub, "DG", 0)
    xs = SpatialCoordinate(sub)
    rho0 = Function(DG0).interpolate(2.0 + sin(3 * xs[0]) * cos(2 * xs[1]))

    z = TrialFunctions(Z)
    t = TestFunctions(Z)
    u, psi = z
    w, v = t

    # The (u-row, psi-column) block and the (psi-row, u-column) block.
    a_B1 = inner(rho0 * grad(psi), w) * dx_m
    a_B2 = rho0 * inner(u, grad(v)) * dx_m

    pr("\n-- do they assemble at all, on their own?")
    for name, form in (("B1  inner(rho0*grad(psi), w)*dx_m", a_B1),
                       ("B2  rho0*inner(u, grad(v))*dx_m", a_B2)):
        M = assemble(form, mat_type="aij")
        pr(f"  OK   {name:<40} norm {M.petscmat.norm():.10e}")

    pr("\n-- assembled together in one mixed matrix (mat_type='aij')")
    A = assemble(a_B1 + a_B2, mat_type="aij").petscmat
    pr(f"  size {A.getSize()},  ||A||_F = {A.norm():.10e}")

    ises = Z.dof_dset.field_ises
    rows_u = ises[0].getIndices()
    rows_p = ises[1].getIndices()
    pr(f"  block sizes: u {len(rows_u)}, psi {len(rows_p)}")

    B1 = dense(A, rows_u, rows_p)   # (u-row, psi-col)
    B2 = dense(A, rows_p, rows_u)   # (psi-row, u-col)

    d = np.abs(B1 - B2.T)
    pr("\n-- the comparison")
    pr(f"  max|B1|            = {np.abs(B1).max():.10e}")
    pr(f"  max|B2|            = {np.abs(B2).max():.10e}")
    pr(f"  max|B1 - B2^T|     = {d.max():.6e}")
    pr(f"  ||B1-B2^T||_F      = {np.linalg.norm(d):.6e}")
    pr(f"  relative           = {d.max() / np.abs(B1).max():.6e}")
    pr(f"  nnz(B1) {np.count_nonzero(B1)}   nnz(B2) {np.count_nonzero(B2)}")

    # A deliberately WRONG sign, to prove the test has teeth.
    pr("\n-- control: the same measurement with B2 negated (must NOT pass)")
    A_bad = assemble(a_B1 - a_B2, mat_type="aij").petscmat
    B1b = dense(A_bad, rows_u, rows_p)
    B2b = dense(A_bad, rows_p, rows_u)
    db = np.abs(B1b - B2b.T)
    pr(f"  max|B1 - B2^T|     = {db.max():.6e}   "
       f"relative {db.max() / np.abs(B1b).max():.6e}")

    # Symmetry of the whole 2-field operator, since it has no Real blocks.
    pr("\n-- and the whole (u,psi) operator, which has no Real blocks, so aij "
       "works")
    a_full = (inner(sym(grad(u)), sym(grad(w))) * dx_m + inner(u, w) * dx_m
              + inner(grad(psi), grad(v)) * dx(domain=parent,
                                               intersect_measures=(
                                                   Measure("dx", domain=sub),))
              + a_B1 + a_B2)
    M = assemble(a_full, mat_type="aij").petscmat
    Mt = M.duplicate(copy=True)
    Mt.transpose()
    Mt.axpy(-1.0, M)
    pr(f"  ||J - J^T||_F / ||J||_F = {Mt.norm() / M.norm():.6e}")


if __name__ == "__main__":
    main()
