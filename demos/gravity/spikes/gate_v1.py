"""V1 — per-block Jacobian symmetry of the coupled system, rotation ON.

Spike S5 route B: `mat_type="nest"`, block by block, `max|A_ij - A_ji^T|`
against `max|A_ij|`, serial.  Route D (matfree `mult` vs `multTranspose`) is
run as well, at whatever rank count the script is launched with, as the
parallel cross-check.

Three things this gate must get right, all of them from review section D:

* **`theta_psi` and `theta_rot` are asserted, never fitted.**  A symmetry test
  is an equation *for* the row scaling, so a fitted scaling absorbs any sign
  error in the block it scales and reports success.  Both are recomputed here
  from `B_mu`, `Lambda` and `Omega^2` and compared with what the solver uses.

* **Rotation is switched on.**  The `(u, psi)` blocks certify nothing about the
  rotation blocks, which scale by `theta_rot` and not `theta_psi`.

* **Two mechanics pairs are quarantined** and compared against a G0 baseline
  measured on the same mesh rather than against zero: the `(u, m)` pair (the
  symmetric-deviatoric projector mismatch, review B5) and the `(u, u)` block
  under `un = 0` (the shipped free-slip Nitsche defect,
  `NOTES/FINDING-FREESLIP-NITSCHE-ASYMMETRY.md`).  Both are properties of
  shipped code that this project did not touch, and the comparison is
  entry-by-entry against the uncoupled operator, which is stronger than
  comparing two asymmetry scalars.

The `(psi, c)` DtN pairs are also quarantined, and the quarantine is *earned*
here rather than asserted: the predicted ratio between the two blocks is
`lam_k - alpha/R`, and that prediction is checked mode by mode.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from firedrake.petsc import PETSc  # noqa: E402
from gadopt import *  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import selfgrav_gia_annulus as demo  # noqa: E402

OMEGA_SQ = 1.566e-3


def _to_sparse(M):
    """A nest sub-block as scipy CSR.  Dense would be gigabytes at M = 5."""
    if M.getType() in ("seqaij", "mpiaij"):
        indptr, indices, data = M.getValuesCSR()
        return sp.csr_matrix((data, indices, indptr), shape=M.getSize())
    # `Real` blocks come out as PETSc `python` matrices, which have no getrow;
    # they are 1 x N, N x 1 or 1 x 1, so densifying them costs nothing.
    return sp.csr_matrix(M.convert("dense").getDenseArray())


def _amax(A):
    return 0.0 if A is None or A.nnz == 0 else float(abs(A).max())


def nest_blocks(F, z):
    """Per-block sparse arrays of `derivative(F, z)`, keyed `(i, j)`."""
    A = assemble(derivative(F, z), mat_type="nest").petscmat
    n = len(z.function_space())
    out = {}
    for i in range(n):
        for j in range(n):
            M = A.getNestSubMatrix(i, j)
            out[(i, j)] = None if M is None else _to_sparse(M)
    return out


def pair(blocks, i, j):
    """`(max|A_ij - A_ji^T|, max|A_ij|, max|A_ji|)`, treating None as zero."""
    aij, aji = blocks[(i, j)], blocks[(j, i)]
    if aij is None and aji is None:
        return 0.0, 0.0, 0.0
    if aij is None:
        return _amax(aji), 0.0, _amax(aji)
    if aji is None:
        return _amax(aij), _amax(aij), 0.0
    return _amax(aij - aji.T), _amax(aij), _amax(aji)


def matfree_asymmetry(F, z, probes=12, seed=0):
    """Route D: `||A x - A^T x|| / ||A x||`, worst over random probes.

    Runs unchanged in parallel, which is the whole reason it is here.  It gives
    a single number with no block attribution, so on this system it is
    dominated by the quarantined mechanics blocks; what it is good for is
    checking that the *parallel* assembly has the same symmetry structure as
    the serial one.
    """
    A = assemble(derivative(F, z), mat_type="matfree").petscmat
    x = A.createVecRight()
    y = A.createVecLeft()
    yt = A.createVecLeft()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(probes):
        with x as arr:
            arr[:] = rng.standard_normal(arr.shape)
        A.mult(x, y)
        A.multTranspose(x, yt)
        yt.axpy(-1.0, y)
        worst = max(worst, yt.norm() / y.norm())
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--nazim", type=int, default=64)
    p.add_argument("--truncation", type=int, default=5)
    p.add_argument("--dt", type=float, default=1.0)
    args, _ = p.parse_known_args()

    serial = COMM_WORLD.size == 1
    parent, sub = demo.build_meshes(
        args.dr, args.nazim,
        path=os.path.join(HERE, f"v1_{args.dr}_{args.nazim}.msh"))
    solver, z, layout, bcs, C = demo.build_solver(
        parent, sub, dt=args.dt, truncation=args.truncation, rotation=True)

    PETSc.Sys.Print(
        f"V1 - per-block Jacobian symmetry, rotation ON, M = {args.truncation}")
    PETSc.Sys.Print(
        f"  parent cells {parent.num_cells()}, mantle cells {sub.num_cells()}, "
        f"{len(layout.multipliers)} multipliers, dim {z.function_space().dim()}, "
        f"ranks {COMM_WORLD.size}")

    # ---- the two scaling constants, asserted from the derivation -----------
    f = float(solver.scaling_factor)
    theta_psi_expected = f * demo.B_MU / demo.LAMBDA
    theta_rot_expected = -f * demo.B_MU * OMEGA_SQ     # s_3 = -1
    theta_psi = float(solver.theta_psi)
    theta_rot = float(solver._theta_rot(2))
    PETSc.Sys.Print(
        f"\n  scaling_factor {f}\n"
        f"  theta_psi   derived {theta_psi_expected:.12e}  "
        f"solver {theta_psi:.12e}  d {abs(theta_psi - theta_psi_expected):.2e}\n"
        f"  theta_rot(m3) derived {theta_rot_expected:.12e}  "
        f"solver {theta_rot:.12e}  d {abs(theta_rot - theta_rot_expected):.2e}")
    ok_theta = (abs(theta_psi - theta_psi_expected) < 1e-14 * abs(theta_psi)
                and abs(theta_rot - theta_rot_expected)
                < 1e-14 * abs(theta_rot))
    PETSc.Sys.Print(f"  asserted, not fitted: {'OK' if ok_theta else 'FAIL'}")

    worst_mf = matfree_asymmetry(solver.F, z)
    PETSc.Sys.Print(
        f"\n  route D (matfree, 12 probes, {COMM_WORLD.size} ranks): "
        f"global ||Ax - A^T x||/||Ax|| = {worst_mf:.6e}")
    PETSc.Sys.Print(
        "    dominated by the quarantined mechanics blocks; compare across "
        "rank counts, not against zero.")
    if not serial:
        return

    # ---- route B: the per-block grid ---------------------------------------
    blocks = nest_blocks(solver.F, z)
    iu, ipsi = layout.displacement, layout.potential
    im = layout.internal_variables[0]
    i3 = layout.rotation["m3"]
    ics = list(layout.multipliers)

    # ---- G0 on the same mesh, for the quarantined mechanics blocks ---------
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    Zm = MixedFunctionSpace([V, S])
    zm = Function(Zm)
    ref = CoupledInternalVariableSolver(
        zm, demo.approximation(), dt=args.dt, bcs=bcs,
        solver_parameters="direct")
    gblocks = nest_blocks(ref.F, zm)

    PETSc.Sys.Print("\n  QUARANTINED: mechanics blocks, against the G0 "
                    "baseline on this mesh (not against zero)")
    for label, (i, j), (gi, gj) in (
            ("(u,u)", (iu, iu), (0, 0)),
            ("(u,m)/(m,u)", (iu, im), (0, 1)),
            ("(m,m)", (im, im), (1, 1))):
        d, a, b = pair(blocks, i, j)
        gd, ga, gb = pair(gblocks, gi, gj)
        # Entry by entry: adding gravity must not perturb the mechanics AT ALL.
        delta = _amax(blocks[(i, j)] - gblocks[(gi, gj)])
        PETSc.Sys.Print(
            f"    {label:<12s} coupled |A-A^T| {d:.4e}  baseline {gd:.4e}  "
            f"|A_coupled - A_G0| {delta:.4e}  (max|A| {a:.4e})")

    # ---- the blocks V1 exists to test --------------------------------------
    PETSc.Sys.Print("\n  THE COUPLING BLOCKS - must transpose to roundoff")
    verdict = []
    for label, i, j in (("(u,psi)/(psi,u)", iu, ipsi),
                        ("(u,m3)/(m3,u)", iu, i3)):
        d, a, b = pair(blocks, i, j)
        rel = d / max(a, b)
        verdict.append((label, rel))
        PETSc.Sys.Print(
            f"    {label:<16s} max|A_ij - A_ji^T| {d:.4e}   max|A_ij| {a:.4e}"
            f"   max|A_ji| {b:.4e}   relative {rel:.3e}")

    PETSc.Sys.Print("\n  NEW DIAGONAL BLOCKS")
    for label, i in (("(psi,psi)", ipsi), ("(m3,m3)", i3)):
        d, a, _ = pair(blocks, i, i)
        PETSc.Sys.Print(f"    {label:<12s} max|A - A^T| {d:.4e}  "
                        f"max|A| {a:.4e}")
    dcc = max(pair(blocks, i, i)[0] for i in ics)
    PETSc.Sys.Print(f"    (c_k,c_k)    max over {len(ics)} multipliers "
                    f"{dcc:.4e}")

    # ---- (psi, c): quarantined, and the quarantine checked -----------------
    # Predicted from `DtNGravityForm.boundary_bilinear`: the constraint row is
    # `psi e_k mu ds` and the feedback is `(lam_k - alpha/R) c_k e_k v ds`, so
    # the two blocks are the SAME integral times different constants and their
    # ratio must be exactly `lam_k - alpha/R`.  If that holds mode by mode the
    # pair is a free row scaling of an operator that is self-adjoint once the
    # multipliers are eliminated, not a defect.
    form = layout.gravity_form
    alpha = float(form.alpha)
    lam_by_key = {}
    for bc_id, dtn in form.dtn_boundaries:
        side, R = form.boundary_geometry[bc_id]
        for mode in dtn.mode_metadata(side, R):
            lam_by_key[(bc_id, mode.key)] = (mode.lam, R)

    PETSc.Sys.Print("\n  QUARANTINED: (psi, c_k) DtN pairs, against the "
                    "predicted ratio lam_k - alpha/R")
    worst_key, worst_rel = None, 0.0
    for (bc_id, key), ic in zip(form.multiplier_keys, ics):
        d, a, b = pair(blocks, ipsi, ic)
        if b == 0.0:
            continue
        lam, R = lam_by_key[(bc_id, key)]
        predicted = float(lam) - alpha / R
        col = np.asarray(blocks[(ipsi, ic)].todense()).ravel()
        row = np.asarray(blocks[(ic, ipsi)].todense()).ravel()
        measured = np.sign(np.dot(col, row)) * a / b
        rel = abs(measured - predicted) / max(abs(predicted), 1e-30)
        if rel > worst_rel:
            worst_rel, worst_key = rel, (bc_id, key)
        # A stronger form of the same statement: the two blocks must be exact
        # multiples of one another.
        resid = float(np.abs(col - predicted * row).max())
        if abs(predicted) > 1e-12:
            rel_resid = resid / a
        else:
            rel_resid = resid / max(b, 1e-30)
        PETSc.Sys.Print(
            f"    {str(bc_id) + ' ' + key:<12s} |A_psic - A_cpsi^T| {d:.3e}   "
            f"predicted ratio {predicted:+.6f}   "
            f"|A_psic - ratio A_cpsi^T|/scale {rel_resid:.3e}")
    PETSc.Sys.Print(
        f"    worst ratio mismatch {worst_rel:.3e} at {worst_key}")

    # ---- everything else ----------------------------------------------------
    quarantine = {frozenset((iu, iu)), frozenset((iu, im)),
                  frozenset((im, im))}
    quarantine |= {frozenset((ipsi, ic)) for ic in ics}
    n = len(z.function_space())
    worst, where = 0.0, None
    scale = max(_amax(b) for b in blocks.values() if b is not None)
    for i in range(n):
        for j in range(i, n):
            if frozenset((i, j)) in quarantine:
                continue
            d, a, b = pair(blocks, i, j)
            rel = d / max(a, b, scale * 1e-30)
            if rel > worst:
                worst, where = rel, (i, j)
    PETSc.Sys.Print(
        f"\n  EVERY OTHER PAIR: worst relative asymmetry {worst:.4e} at "
        f"block {where}")

    PETSc.Sys.Print("\n  VERDICT")
    for label, rel in verdict:
        PETSc.Sys.Print(
            f"    {label:<16s} {rel:.3e}   "
            f"{'PASS' if rel < 1e-13 else 'FAIL'} (threshold 1e-13)")
    PETSc.Sys.Print(f"    other pairs      {worst:.3e}   "
                    f"{'PASS' if worst < 1e-13 else 'FAIL'}")


if __name__ == "__main__":
    main()
