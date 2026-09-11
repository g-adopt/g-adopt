"""Wave 1 gate: does the low-rank `B` equal the multiplier path's eliminated action?

WHAT IS BEING COMPARED, AND WHY IT IS NOT CIRCULAR
    The multiplier representation writes two extra rows per mode. The
    constraint row defines the trace coefficient,

        (psi e_k - scale_k c_k) mu_k ds        ->   u_k . psi - scale_k A_h c_k

    and the feedback row returns it to the potential,

        (lam_k - alpha/R) c_k e_k v ds         ->   (lam_k - alpha/R) c_k u_k

    Eliminating `c` between the two gives `B = sum_k w_k u_k u_k^T`, which is
    what `gadopt.dtn_lowrank` builds. The gate performs that elimination on the
    ASSEMBLED coupled Jacobian, using nothing from the low-rank code:

      1. Probe the constraint block. Apply `J` to a vector that is 1 in one
         multiplier and 0 everywhere else, and read every multiplier row. That
         gives the `(c, c)` block by columns. It is asserted diagonal here, not
         assumed.
      2. Apply `J` to a vector carrying a random psi and no multipliers. The
         multiplier rows then hold the constraint residual at `c = 0`, so
         `c = -residual / diagonal` is the eliminated coefficient, obtained
         from the operator itself.
      3. Apply `J` to a vector carrying THAT `c` and nothing else. Its psi rows
         are the feedback, which is the eliminated action, `B psi`.

    Step 3 uses `c` from step 2, which came from `J`. No quantity on the
    reference side is computed by the code under test. `theta_psi` cancels
    between steps 1 and 2 and is carried by step 3, so the comparison is against
    `theta_psi * B0` and not against `B0`.

WHAT THIS GATE DOES NOT CLAIM
    Nothing about speed, nothing about iteration counts, and nothing in 3-D.
    Rule 2 of `NOTES/fastdtn/HISTORICAL-PLAN.md`: a 2-D annulus shows that a construction
    works and never that it helps. The annulus IS cross-mesh - displacement on
    a `Submesh`, potential on the parent - so the cross-mesh index mapping is
    exercised by construction rather than by a separate case.

THE CONTROLS, WHICH ARE THE POINT
    A parity number that is small proves nothing on its own. Four controls run
    beside it, each a deliberate defect that the same number must catch:

      weight   the heaviest mode weight scaled by 1 + eps
      (null)   an inert mode given a real weight
      dof      one owned column dropped from `C`
      mode     the heaviest mode's whole functional zeroed
      offset   the monolithic row offset shifted by one slot
      theta    the `theta_psi` prefactor dropped
      mask     one live column of `C` masked

    Each prints the parity number it produces. If a control comes back at the
    clean floor, the gate cannot see that defect and its pass is worthless.

    **The weight control had to be chosen, not taken.** `w_k` is
    `(lam_k - alpha/R) / (scale_k A_h)` and the default Robin shift is
    `alpha = 1`, so every mode with `lam_k R = 1` has weight exactly zero and
    contributes nothing to `B`. In 2-D that is `|m| = 1` on both boundaries.
    The first version of this gate perturbed mode 0 of boundary 0, which is one
    of them, and reported the clean floor while announcing a poisoned operator.
    The control now perturbs the heaviest weight and names it, and a second
    control gives an inert mode a real weight so that the inert set is measured
    rather than assumed.

RUN
    PYTHONPATH=<worktree> python3 gate_lowrank_operator.py
    PYTHONPATH=<worktree> mpiexec -np 4 python3 gate_lowrank_operator.py
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "gravity"))

# PETSc reads `sys.argv` into its options database at import.
_ARGV, sys.argv = list(sys.argv), sys.argv[:1]

import gadopt  # noqa: E402,F401  (import gadopt before firedrake)
import numpy as np  # noqa: E402
from firedrake import (COMM_WORLD, Function, Measure, Mesh,  # noqa: E402
                       SpatialCoordinate, Submesh, TrialFunction, assemble,
                       atan2, cos, derivative, dot)
from gadopt import (CompressibleInternalVariableApproximation,  # noqa: E402
                    CylindricalDtN, SelfGravitatingGIASolver,
                    self_gravitating_gia_space)
from gadopt.dtn_coupled import monolithic_rows  # noqa: E402,F401
from gadopt.dtn_lowrank import apply_dirichlet_to_rows  # noqa: E402
from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402

import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

B_MU, LAMBDA = 1.2769, 1.1116


def say(msg):
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


def build(args, dtn_representation="lowrank", gravity_dirichlet=None):
    """The 2-D coupled annulus. Cross-mesh: `u` on the submesh, `psi` on the parent."""
    path = os.path.join(args.workdir, f"gate_lowrank_{args.dr:g}.msh")
    if COMM_WORLD.rank == 0 and not os.path.exists(path):
        gen.generate(path, dr_mantle=args.dr, n_azimuthal=args.nazim)
    COMM_WORLD.barrier()

    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False

    X = SpatialCoordinate(parent)
    sigma = 1.0e-3 * cos(2 * atan2(X[1], X[0]))
    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(args.dtn_degree)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(args.dtn_degree)},
        gen.CURVE_RE: {"interior_sigma": sigma},
    }
    if gravity_dirichlet is not None:
        # An explicit datum on the potential, so that `apply_dirichlet_to_rows`
        # has something to do. Without one the annulus has no strongly
        # constrained psi dof and the bc control below cannot fire.
        gravity_bcs[gen.CURVE_RC] = {"psi": gravity_dirichlet}

    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=args.rotation,
        self_gravity_number=LAMBDA, condense_internal_variables=True)
    z = Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        bulk_shear_ratio=100.0, g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)

    Xm = SpatialCoordinate(sub)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    C = assemble(1.0 * dot(Xm, Xm) * dx_m)
    bcs = {
        gen.CURVE_RC: {"un": 0.0},
        gen.CURVE_RE: {"normal_stress":
                       B_MU * 1.0e-3 * cos(2 * atan2(Xm[1], Xm[0]))},
    }
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=1.0, bcs=bcs,
        rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH,
        dtn_representation=dtn_representation)
    return solver, z, layout, Z


def jacobian_action(solver, z, Z):
    """`x -> J x` on the monolithic mixed vector, from the solver's own residual.

    Matrix free, because a space with `Real` blocks has no monolithic `aij`
    assembly at all. The strong conditions are applied, so the constrained psi
    rows and columns are eliminated exactly as they are in a solve - which is
    what makes a `B` that still couples to them detectable.
    """
    J = derivative(solver.F, z, TrialFunction(Z))
    Jmat = assemble(J, bcs=solver.strong_bcs, mat_type="matfree")
    petsc = Jmat.petscmat

    def apply(x):
        y = Function(Z)
        with x.dat.vec_ro as xv, y.dat.vec as yv:
            petsc.mult(xv, yv)
        return y

    return apply, Jmat


def real_get(f, idx):
    """The global value of a `Real` sub-field. Replicated, so any rank has it."""
    return float(f.subfunctions[idx].dat.data_ro[0])


def eliminated_action(solver, z, Z, layout, apply_J, psi_probe):
    """`B psi` as the multiplier path itself defines it.

    Returns `(B_ref_psi_local, c, diagonal)`.
    """
    mult = list(layout.multipliers)
    n = len(mult)

    # Step 1. The (c, c) block, one column at a time, and a check that it IS
    # diagonal. `multiplier_diagonal`'s docstring says no row couples c_k to
    # c_j; this measures it rather than repeating it.
    diagonal = np.zeros(n)
    max_offdiag = 0.0
    for j, field in enumerate(mult):
        x = Function(Z)
        x.subfunctions[field].assign(1.0)
        y = apply_J(x)
        column = np.array([real_get(y, f) for f in mult])
        diagonal[j] = column[j]
        column[j] = 0.0
        max_offdiag = max(max_offdiag, float(np.abs(column).max(initial=0.0)))

    # Step 2. The constraint residual at c = 0, hence the eliminated c.
    x0 = Function(Z)
    x0.subfunctions[layout.potential].dat.data[:] = psi_probe
    y0 = apply_J(x0)
    residual = np.array([real_get(y0, f) for f in mult])
    c = -residual / diagonal

    # Step 3. The feedback those coefficients produce, with psi itself absent,
    # so what comes back on the psi rows is the eliminated action and nothing
    # else.
    x1 = Function(Z)
    for value, field in zip(c, mult):
        x1.subfunctions[field].assign(value)
    y1 = apply_J(x1)
    B_ref = np.array(y1.subfunctions[layout.potential].dat.data_ro, dtype=float)

    # The scale the parity number has to be read against. If `B_ref` is a
    # rounding error beside `A psi`, a small parity number says nothing.
    A_psi = np.array(y0.subfunctions[layout.potential].dat.data_ro, dtype=float)
    return B_ref, c, diagonal, max_offdiag, A_psi


def global_norm(comm, a):
    return float(np.sqrt(comm.allreduce(float(np.dot(a, a)))))


def parity(comm, ours, ref):
    num = global_norm(comm, ours - ref)
    den = global_norm(comm, ref)
    return num / den if den > 0 else float("nan"), den


def our_action(solver, Z, layout, psi_probe, theta=None):
    """`B psi` from `CoupledLowRankDtN`, through the monolithic vector."""
    x = Function(Z)
    x.subfunctions[layout.potential].dat.data[:] = psi_probe
    y = Function(Z)
    with x.dat.vec_ro as xv, y.dat.vec as yv:
        solver.dtn_operator.mult(xv, yv, theta=theta)
    return np.array(y.subfunctions[layout.potential].dat.data_ro, dtype=float)


def run(args, gravity_dirichlet=None, label=""):
    comm = COMM_WORLD
    # **Stage markers, and they are not decoration.** Every stage below is
    # collective, and a rank-divergent one deadlocks with no output and no
    # error - measured, 41 minutes at 4 ranks. Without markers a hang cannot be
    # located at all, and "still compiling" and "deadlocked" look identical.
    say(f"  [stage] building{label}")
    solver, z, layout, Z = build(args, "lowrank", gravity_dirichlet)
    say("  [stage] assembling the matrix-free Jacobian")
    apply_J, _ = jacobian_action(solver, z, Z)

    psi_space = Z.sub(layout.potential)
    rng = np.random.default_rng(12345 + comm.rank)
    psi_probe = rng.standard_normal(psi_space.node_set.size)

    say(f"  [stage] eliminating {len(layout.multipliers)} multipliers "
        f"({len(layout.multipliers) + 2} Jacobian applications)")
    B_ref, c, diagonal, max_offdiag, A_psi = eliminated_action(
        solver, z, Z, layout, apply_J, psi_probe)
    say("  [stage] applying the low-rank operator")
    ours = our_action(solver, Z, layout, psi_probe)
    rel, ref_norm = parity(comm, ours, B_ref)

    op = solver.dtn_operator
    say("")
    say("=" * 78)
    say(f"WAVE 1 GATE{label}   ranks={comm.size}   L={args.dtn_degree}   "
        f"rotation={args.rotation}")
    say("=" * 78)
    say(f"  modes                       {op.n_modes}   "
        f"(Real fields {len(layout.multipliers)})")
    say(f"  theta_psi                   {op.theta_value:.10f}")
    say(f"  constrained psi dofs        {len(solver.dtn_constrained_dofs)} "
        "(rank-local)")
    say(f"  psi offset, rank 0          {op.psi_offset}")
    # **The share is read BEFORE the parity number, and it gates it.** A parity
    # number is meaningless if `B` barely acts: an operator that is inert
    # agrees with a reference that is also inert. Two people hit this on
    # 2026-08-12 through different doors - four modes here carry weight exactly
    # zero, and the Forward reviewer nearly filed "M1 is fine" because a rank-1
    # mode was nearly orthogonal to the solution. So print the share, and
    # refuse to read the parity below `--min-share`.
    A_norm = global_norm(comm, A_psi)
    share = ref_norm / A_norm if A_norm > 0 else float("nan")
    say(f"  ||B psi||                   {ref_norm:.6e}")
    say(f"  ||A psi||                   {A_norm:.6e}")
    say(f"  SHARE  ||B psi|| / ||A psi||  =  {share:.4%}   "
        f"(floor for reading the parity: {args.min_share:.4%})")
    say(f"  max |off-diagonal (c,c)|    {max_offdiag:.3e}   "
        f"vs min |diagonal| {float(np.abs(diagonal).min()):.3e}")
    say("")
    if not (share >= args.min_share):
        say(f"  REFUSED: B acts at {share:.4%} of A, below the "
            f"{args.min_share:.4%} floor. The parity number below cannot "
            "distinguish a correct operator from an inert one and must not "
            "be quoted.")
    say(f"  PARITY  ||B_ours - B_ref|| / ||B_ref||   =   {rel:.6e}")
    say("")

    # ---- what the operator is made of, before the controls -----------------
    # **Some modes are exactly inert and a control that perturbs one of them
    # cannot fire.** `w_k = (lam_k - alpha/R) / (scale_k A_h)` and the default
    # Robin shift is `alpha = 1`, so any mode whose `lam_k R` equals 1 has
    # weight exactly zero: in 2-D that is `|m| = 1` on both sides, since the
    # exterior map is `lam = m/R` and the interior one `lam = m/R` as well. The
    # first version of this gate perturbed mode 0 of boundary 0, which is one
    # of them, and reported the clean floor while claiming to have poisoned the
    # operator. That is the failure this section exists to prevent.
    inert = [(b, k) for b, rows in enumerate(op.mode_rows)
             for k, w in enumerate(rows.weights) if w == 0.0]
    heavy_b, heavy_k, heavy_w = 0, 0, 0.0
    for b, rows in enumerate(op.mode_rows):
        for k, w in enumerate(rows.weights):
            if abs(w) > abs(heavy_w):
                heavy_b, heavy_k, heavy_w = b, k, w
    masked = sum(int(np.isin(rows.dofs, np.array(
        sorted(solver.dtn_constrained_dofs), dtype=np.int64)).sum())
        for rows in op.mode_rows)
    say(f"  modes with weight exactly 0 {len(inert)} of {op.n_modes}   "
        f"{[f'{op.mode_rows[b].keys[k]}' for b, k in inert]}")
    say(f"  heaviest weight             boundary {heavy_b} mode "
        f"{op.mode_rows[heavy_b].keys[heavy_k]}  w = {heavy_w:.6e}")
    say(f"  columns of C the Dirichlet elimination masked  {masked}")
    say("")

    # ---- controls, each a defect the parity number must catch --------------
    say("  controls (each must move the parity number, or the gate is blind)")

    saved = [rows.weights.copy() for rows in op.mode_rows]
    op.mode_rows[heavy_b].weights[heavy_k] *= 1.0 + args.eps
    r, _ = parity(comm, our_action(solver, Z, layout, psi_probe), B_ref)
    say(f"    weight  heaviest weight x (1 + {args.eps:g})  : {r:.6e}")
    for rows, w in zip(op.mode_rows, saved):
        rows.weights[:] = w

    if inert:
        b, k = inert[0]
        op.mode_rows[b].weights[k] = heavy_w
        r, _ = parity(comm, our_action(solver, Z, layout, psi_probe), B_ref)
        say(f"    (null)  an INERT mode given a real weight: {r:.6e}   "
            "<- must also move; a zero weight is not an excuse")
        for rows, w in zip(op.mode_rows, saved):
            rows.weights[:] = w

    # **Never put a collective behind a rank-local emptiness test.** `parity`
    # and `our_action` both reduce over the communicator, and `rows.rows.size`
    # is 0 on a rank that owns none of that boundary. The first version of this
    # section wrote `if dropped:` around the comparison, so on 4 ranks the ranks
    # that owned no boundary dofs skipped the `Allreduce` and the run hung -
    # 41 minutes with no output and no error. The poison below is local, the
    # comparison is unconditional, and `poisoned_ranks` says how many ranks the
    # poison actually reached, so a control that fired nowhere is visible
    # instead of silent.
    #
    # The magnitude of this one depends on the rank count, because each rank
    # drops its own first owned column. It is a control, not a measurement.
    saved_rows = [rows.rows.copy() for rows in op.mode_rows]
    dropped = 0
    for rows in op.mode_rows:
        if rows.rows.size:
            rows.rows[:, 0] = 0.0
            dropped = 1
            break
    poisoned = comm.allreduce(dropped)
    r, _ = parity(comm, our_action(solver, Z, layout, psi_probe), B_ref)
    say(f"    dof     one owned column dropped from C : {r:.6e}   "
        f"(poisoned on {poisoned}/{comm.size} ranks)")
    for rows, m in zip(op.mode_rows, saved_rows):
        rows.rows[:] = m

    # Globally well defined, unlike the one above: the heaviest mode's whole
    # functional is deleted, and every rank deletes its slice of the same row.
    saved_rows = [rows.rows.copy() for rows in op.mode_rows]
    if op.mode_rows[heavy_b].rows.size:
        op.mode_rows[heavy_b].rows[heavy_k, :] = 0.0
    r, _ = parity(comm, our_action(solver, Z, layout, psi_probe), B_ref)
    say(f"    mode    heaviest mode's functional zeroed: {r:.6e}")
    for rows, m in zip(op.mode_rows, saved_rows):
        rows.rows[:] = m

    # **Clipped, because the raw `+ 1` walks off the end of the local vector.**
    # The psi block is last but for the `Real` fields, which own nothing above
    # rank 0, so the highest psi row can be the final owned entry and `+ 1`
    # is then out of bounds. At 2 ranks it happened to stay in range; at 4 it
    # raised `IndexError` from inside the operator and took the gate down after
    # the parity number had already been printed - a control that crashes is
    # not a control, and it is the only one here that could hide behind the
    # numbers above it.
    x_probe = Function(Z)
    with x_probe.dat.vec_ro as xv:
        n_local = xv.getLocalSize()
    saved_mono = [m.copy() for m in op.rows_mono]
    op.rows_mono = [np.minimum(m + 1, max(n_local - 1, 0)) for m in op.rows_mono]
    r, _ = parity(comm, our_action(solver, Z, layout, psi_probe), B_ref)
    say(f"    offset  monolithic rows shifted by +1   : {r:.6e}   "
        f"(local vector {n_local} entries)")
    op.rows_mono = saved_mono

    # `theta = 1` is the same defect as losing the prefactor, and it is the one
    # the withdrawn recommendation in PLAN section 0.4 would have shipped.
    r, _ = parity(comm, our_action(solver, Z, layout, psi_probe, theta=1.0),
                  B_ref)
    say(f"    theta   prefactor dropped (theta = 1)   : {r:.6e}")

    # **The Dirichlet elimination cannot be exercised against `J` on this
    # geometry, and that is a property of the coupled solver rather than of the
    # gate.** `DtNGravityForm.set_boundary_conditions` refuses `'dtn'` and
    # `'psi'` on one boundary as mutually exclusive, and on an annulus or a
    # shell no other boundary shares a node with a DtN boundary. So the true
    # constrained set intersects `C` in 0 columns (printed above) and masking
    # is a no-op here. What CAN be shown is that the gate would see a masking
    # error if there were one: mask a column that IS in `C`'s support and watch
    # the parity number move.
    saved_rows = [rows.rows.copy() for rows in op.mode_rows]
    masked_here = 0
    for rows in op.mode_rows:
        if rows.rows.size:
            apply_dirichlet_to_rows([rows], [int(rows.dofs[0])])
            masked_here = 1
            break
    poisoned = comm.allreduce(masked_here)
    r, _ = parity(comm, our_action(solver, Z, layout, psi_probe), B_ref)
    say(f"    mask    one live column of C masked     : {r:.6e}   "
        f"(poisoned on {poisoned}/{comm.size} ranks)")
    for rows, m in zip(op.mode_rows, saved_rows):
        rows.rows[:] = m

    # ---- the coefficient check, independent of the action ------------------
    x = Function(Z)
    x.subfunctions[layout.potential].dat.data[:] = psi_probe
    with x.dat.vec_ro as xv:
        ours_c = np.concatenate(op.coefficients(xv.array_r))
    err = float(np.abs(ours_c - c).max())
    scale = float(np.abs(c).max())
    say("")
    say(f"  trace coefficients  max|c_ours - c_ref| = {err:.6e}   "
        f"max|c_ref| = {scale:.6e}   relative {err / scale:.6e}")
    say("")
    return rel, share


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--dr", type=float, default=0.15)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--rotation", action="store_true", default=False)
    p.add_argument("--eps", type=float, default=1e-3,
                   help="size of the weight perturbation control")
    p.add_argument("--workdir", default=HERE)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--min-share", type=float, default=1e-3,
                   help="the smallest ||B psi|| / ||A psi|| at which the "
                        "parity number is worth reading. Below it a correct "
                        "operator and an inert one agree equally well.")
    args = p.parse_args(_ARGV[1:])

    rels = [run(args, None, "  (no potential Dirichlet)")]
    # A second configuration with a strong condition on psi. It does NOT make
    # the Dirichlet elimination bite - the CMB shares no node with a DtN
    # boundary - but it does put constrained psi rows into `J`, so the parity
    # is measured against an operator that has an identity block in it.
    rels.append(run(args, 0.0, "  (Dirichlet psi = 0 on the CMB)"))

    ok = all(r < args.tol and s >= args.min_share for r, s in rels)
    say("=" * 78)
    say(f"  GATE {'PASS' if ok else 'FAIL'}   "
        f"parity {[f'{r:.3e}' for r, _ in rels]}   tol {args.tol:g}   "
        f"share {[f'{s:.4%}' for _, s in rels]}   min {args.min_share:.4%}")
    say("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
