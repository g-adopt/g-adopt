"""Check ``ShiftedHessianMat`` against an assembled reference, with a test that can fail.

``check_s2c.py`` only checks that the Hessian operator received BNK's shift. It would
still pass if the operator added ``pert * I`` instead of ``pert * A``. This script
checks the action itself, on a small case that runs in seconds:

1. ``tao.getHessian()`` returns the wrapper and the Riesz preconditioning matrix.
2. TAO's Hessian callback sets ``pert`` to zero, whatever it was.
3. With ``pert = 0`` the wrapper equals pyadjoint's Hessian matrix.
4. ``shift`` sets the value: after ``shift(0.3)`` and ``shift(0.05)`` the action is
   ``H v + 0.05 A v``, with ``A`` assembled from the form of the metric (zero rows
   and columns on the wall nodes for the Dirichlet operator).
5. The sub-matrix on a random inactive set of 70 percent of the nodes is a
   ``MaskedShiftedHessianMat``, its ``shift`` sets the full ``pert``, and its action
   is ``Z^T (H + pert A) Z``. The same action is also compared with the identity-shifted
   ``Z^T H Z + pert I``, which must differ by much more than the tolerance; this is
   what shows that the test can fail.
6. ``InnerProductMat.mult`` equals the assembled reference ``A``.

It also prints the eigenvalues of ``(A^{-1})_ff A_ff`` with the wall nodes fixed:
on the free set the preconditioner is ``(A^{-1})_ff``, so the preconditioned metric
shift is ``pert (A^{-1})_ff A_ff`` and not exactly ``pert I``.

Usage (from a directory that holds the reference state of the tiny case,
``forward.py --steps 10 --resolution 16``):
    python check_wrapper.py REFERENCE.h5 [free|pinned] [none|dirichlet]

Exit status 0 when every check holds (relative errors <= 1e-12, and the identity
reference differs by more than 1e-3), 1 otherwise.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firedrake import TestFunction, TrialFunction, assemble  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from pyadjoint import MinimizationProblem, stop_annotating  # noqa: E402
from pyadjoint.optimization.tao_solver import TAOSolver  # noqa: E402

from optimisers import tao_parameters  # noqa: E402
from problem import build_problem  # noqa: E402
from riesz_hessian import (MaskedShiftedHessianMat, build_preconditioner_matrix,  # noqa: E402
                           build_shifted_hessian)

TOLERANCE = 1e-12


def main(reference, wall="free", bc="none"):
    """Run the checks on one wall setting and return True when all hold."""
    problem = build_problem(reference, window=5, alpha_d=0, alpha_s=0,
                            inner_product="regularisation", initial_guess="observed",
                            prior_sigma=31.594026758304967, prior_L=0.22071129807892717,
                            wall_treatment=wall, prior_wall_bc=bc)
    V = problem.Tic.function_space()
    params = tao_parameters("tao-bnls-riesz", 3, 0.2, {"tao_bnk_ksp_max_it": 100})
    pmat, pctx = build_preconditioner_matrix(problem)
    solver = TAOSolver(MinimizationProblem(problem.reduced_functional, bounds=problem.bounds),
                       params, Pmat=pmat)
    tao = solver.tao
    H0 = tao.getHessian()[0]
    h0ctx = H0.getPythonContext()
    Hw, ctx = build_shifted_hessian(tao, pctx)
    ok = True

    def report(name, value, good):
        nonlocal ok
        ok &= bool(good)
        print(f"   {name}: {value} {'ok' if good else 'FAILED'}")

    print(f"== wall treatment {wall}, prior operator {bc}")
    installed = tao.getHessian()
    report("TAO Hessian is the wrapper, preconditioner is the Riesz matrix",
           (installed[0].handle == Hw.handle, installed[1].handle == pmat.handle),
           installed[0].handle == Hw.handle and installed[1].handle == pmat.handle)

    # The reference metric: assembled from the form, with zero rows and columns on
    # the constrained nodes for the Dirichlet operator, which is what the Riesz map
    # inverts there.
    A = assemble(problem.inner_product.form(TrialFunction(V), TestFunction(V))).petscmat
    n = V.dim()
    Ad = A.convert("dense").getDenseArray().copy()
    constrained = np.concatenate([np.asarray(b.nodes) for b in problem.inner_product.bcs]
                                 ) if problem.inner_product.bcs else np.array([], dtype=int)
    if constrained.size:
        Ad[constrained, :] = 0.0
        Ad[:, constrained] = 0.0
    rng = np.random.default_rng(0)
    with stop_annotating():
        x = tao.getSolution().copy()
        x.setArray(np.clip(problem.Tic.dat.data_ro + 0.0, 0, 1))
        Hw.shift(0.7)
        tao.computeHessian(x, Hw, pmat)
        report("pert after the Hessian callback (want 0)", ctx.pert, ctx.pert == 0.0)
        v = x.duplicate()
        v.setArray(rng.standard_normal(n))
        Hv = v.duplicate()
        H0.mult(v, Hv)
        Av = Ad @ v.getArray()
        y = v.duplicate()
        Hw.mult(v, y)
        err = (y - Hv).norm() / Hv.norm()
        report("pert 0: |W v - H v| / |H v|", f"{err:.3e}", err <= TOLERANCE)

        Hw.shift(0.3)
        Hw.shift(0.05)
        Hw.mult(v, y)
        ref = Hv.getArray() + 0.05 * Av
        err = np.linalg.norm(y.getArray() - ref) / np.linalg.norm(ref)
        report("shift(0.3), shift(0.05): pert", ctx.pert, ctx.pert == 0.05)
        report("rel error against H v + 0.05 A v", f"{err:.3e}", err <= TOLERANCE)
        print(f"   size of the shift term |0.05 A v| / |H v| = "
              f"{0.05 * np.linalg.norm(Av) / Hv.norm():.3e}")

        tao.computeHessian(x, Hw, pmat)
        free = np.sort(rng.choice(n, size=int(0.7 * n), replace=False)).astype(PETSc.IntType)
        iset = PETSc.IS().createGeneral(free)
        sub = Hw.createSubMatrix(iset, iset)
        report("sub-matrix context", type(sub.getPythonContext()).__name__,
               isinstance(sub.getPythonContext(), MaskedShiftedHessianMat))
        sub.shift(0.11)
        report("after sub.shift(0.11): full pert", ctx.pert, ctx.pert == 0.11)
        xr = PETSc.Vec().createSeq(len(free))
        xr.setArray(rng.standard_normal(len(free)))
        yr = xr.duplicate()
        sub.mult(xr, yr)
        vf = v.duplicate()
        vf.set(0)
        vf.setValues(free, xr.getArray())
        vf.assemble()
        H0.mult(vf, Hv)
        Avf = Ad @ vf.getArray()
        ref = Hv.getArray()[free] + 0.11 * Avf[free]
        ref_identity = Hv.getArray()[free] + 0.11 * xr.getArray()
        err = np.linalg.norm(yr.getArray() - ref) / np.linalg.norm(ref)
        err_identity = np.linalg.norm(yr.getArray() - ref_identity) / np.linalg.norm(ref_identity)
        report("sub: rel error against Z^T (H + 0.11 A) Z", f"{err:.3e}", err <= TOLERANCE)
        report("sub: rel difference from the identity shift (must be large)",
               f"{err_identity:.3e}", err_identity > 1e-3)
        sub.shift(0.02)
        sub.mult(xr, yr)
        ref = Hv.getArray()[free] + 0.02 * Avf[free]
        err = np.linalg.norm(yr.getArray() - ref) / np.linalg.norm(ref)
        report("sub: a second shift sets, rel error against pert 0.02", f"{err:.3e}",
               err <= TOLERANCE)

        pv = v.duplicate()
        pctx.mult(None, v, pv)
        Av = Ad @ v.getArray()
        err = np.linalg.norm(pv.getArray() - Av) / np.linalg.norm(Av)
        report("InnerProductMat.mult against the assembled reference", f"{err:.3e}",
               err <= TOLERANCE)

        # The preconditioned metric shift on BNK's free set with the walls fixed.
        Afull = A.convert("dense").getDenseArray()
        walls = np.concatenate([problem.wall_nodes["bottom"], problem.wall_nodes["top"]])
        f = np.setdiff1d(np.arange(n), walls)
        if constrained.size:
            # The Dirichlet Riesz map inverts A_ff exactly on the free nodes.
            ev = np.ones(len(f))
        else:
            ev = np.sort(np.linalg.eigvals(np.linalg.inv(Afull)[np.ix_(f, f)]
                                           @ Afull[np.ix_(f, f)]).real)
        print(f"   eig of (A^-1)_ff A_ff with the walls fixed: min {ev[0]:.3f} median "
              f"{np.median(ev):.3f} max {ev[-1]:.3f}, {(ev > 2).sum()} of {len(ev)} above 2")
    return ok


if __name__ == "__main__":
    result = main(*sys.argv[1:])
    print("all checks hold" if result else "A CHECK FAILED")
    sys.exit(0 if result else 1)
