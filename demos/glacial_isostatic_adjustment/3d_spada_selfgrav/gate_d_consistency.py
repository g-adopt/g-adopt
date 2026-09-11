"""D1/D2/D3: the Wave 2 consistency gate for the low-rank forward plumbing.

Spec: `NOTES/fastdtn/REVIEW-FORWARD.md` sections 14.3 and 22.

D1 - PLUMBING ONLY, AND THE DOCSTRING SAYS SO ON PURPOSE

    This does NOT prove the residual and Jacobian augmentations are consistent.
    Both call one implementation, so agreement is near-tautological. It proves
    only that the two call sites pass the same `theta`, rows and offset. The
    check that one arm is not simply ABSENT is D2, and D2 is the gate.

    Whether `B` is the RIGHT operator is a different question and was answered
    in Wave 1 by `gate_lowrank_operator.py`, at parity 4.29e-16 to 7.79e-16
    against a reference built from the assembled coupled Jacobian.

D2 - THE LOAD-BEARING HALF

    `snes_type: ksponly`, NON-ZERO initial guess, two clauses, both required:

        clause 1   ||F(z) + theta B z|| / ||F(z0) + theta B z0||  <  1e-10
        clause 2   ||F(z)||            / ||F(z0) + theta B z0||   >  1e-6

    Clause 1 says the augmented system was solved. Clause 2 says the plain
    system was NOT, so the gate cannot be passed by deleting `B`.

    **The initial guess must be non-zero and here is why.** From a zero guess
    `B z` is zero, so a residual-only augmentation reproduces the un-augmented
    answer in one Newton step, clause 1 lands at the solver floor, and the gate
    goes green on a broken solver.

    **Never assert on `snes_converged_reason`.** It reads 5 in every arm here,
    including both broken ones. That is measured, three times now, not a style
    preference.

D3 - DID IT ACTUALLY RUN

    One integer incremented inside the augmented `mult`, asserted `> 0`. It
    tests "the augmentation ran", not "it was installed". A handle COUNT is not
    assertable: PETSc reuses freed `Mat` addresses, and forty build/destroy
    cycles produced one distinct handle across forty matrices.

RULE 16
    `||theta B z|| / ||F(z)||` is printed before any other number and gates
    them. A `B` that does not move the answer hides every bug in itself while
    looking like a pass.

RUN
    PYTHONPATH=<worktree> python3 gate_d_consistency.py
    PYTHONPATH=<worktree> python3 gate_d_consistency.py --arm residual_only
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "gravity"))
_ARGV, sys.argv = list(sys.argv), sys.argv[:1]

import gadopt  # noqa: E402,F401
import numpy as np  # noqa: E402
from firedrake import (COMM_WORLD, Function, Measure,  # noqa: E402
                       Mesh, SpatialCoordinate, Submesh, TrialFunction,
                       assemble, atan2, cos, derivative, dot)
from gadopt import (CompressibleInternalVariableApproximation,  # noqa: E402
                    CylindricalDtN, SelfGravitatingGIASolver,
                    self_gravitating_gia_space)
from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402

import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

B_MU, LAMBDA = 1.2769, 1.1116


def say(msg):
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


def build(args):
    path = os.path.join(HERE, f"gate_d_{args.dr:g}.msh")
    if COMM_WORLD.rank == 0 and not os.path.exists(path):
        gen.generate(path, dr_mantle=args.dr, n_azimuthal=args.nazim)
    COMM_WORLD.barrier()
    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False

    X, Xm = SpatialCoordinate(parent), SpatialCoordinate(sub)
    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(args.dtn_degree)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(args.dtn_degree)},
        gen.CURVE_RE: {"interior_sigma": 1.0e-3 * cos(2 * atan2(X[1], X[0]))},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=True,
        self_gravity_number=LAMBDA, condense_internal_variables=True,
        dtn_representation="lowrank")
    z = Function(Z)
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        bulk_shear_ratio=100.0, g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    C = assemble(1.0 * dot(Xm, Xm) * dx_m)
    bcs = {gen.CURVE_RC: {"un": 0.0},
           gen.CURVE_RE: {"normal_stress":
                          B_MU * 1.0e-3 * cos(2 * atan2(Xm[1], Xm[0]))}}
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=1.0, bcs=bcs,
        rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH,
        dtn_representation="lowrank",
        solver_parameters=preset(args))
    return solver, z, Z, layout


def preset(args):
    """`ksponly`, and a direct-ish block 0 so the gate measures B and not GAMG."""
    return {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "fgmres",
        "ksp_rtol": args.rtol,
        # PETSc's DEFAULT stopping test is on the PRECONDITIONED
        # residual. `NOTES/HANDOVER.md` section 3.9 records the same
        # trap in this codebase. Clause 1 measures the TRUE augmented
        # residual, so a CONVERGED_RTOL at 1e-12 can mean the
        # preconditioned residual fell twelve orders while the true one
        # fell six.
        "ksp_norm_type": args.norm_type,
        "ksp_max_it": 400,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
        "dtn_pc_fieldsplit_schur_fact_type": "full",
        "dtn_fieldsplit_0_ksp_type": "preonly",
        "dtn_fieldsplit_0_pc_type": "python",
        "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "dtn_fieldsplit_0_assembled_pc_type": "lu",
        "dtn_fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
        "dtn_fieldsplit_1_ksp_type": "gmres",
        "dtn_fieldsplit_1_ksp_rtol": 1e-10,
        "dtn_fieldsplit_1_pc_type": "none",
    }


def gnorm(a):
    return float(np.sqrt(COMM_WORLD.allreduce(float(np.dot(a, a)))))


def residual(solver, homogenise=True):
    """`F(z)` as a local array, with the constrained rows zeroed.

    Zeroed so that the bc violation of a deliberately non-conforming initial
    guess does not dominate the norms; both ends of every ratio below are
    treated identically.
    """
    r = assemble(solver.F)
    if homogenise:
        for bc in solver.strong_bcs:
            bc.zero(r)
    with r.dat.vec_ro as rv:
        return np.array(rv.array_r, dtype=float)


def augmented_residual(solver, z):
    """`F(z) + theta B z`, using the SAME call site the solver uses."""
    r = assemble(solver.F)
    for bc in solver.strong_bcs:
        bc.zero(r)
    with z.dat.vec_ro as xv, r.dat.vec as rv:
        solver.augment_residual(xv, rv)
    return _vec(r)


def _vec(r):
    with r.dat.vec_ro as rv:
        return np.array(rv.array_r, dtype=float)


def randomise(z, scale=1e-3, bcs=()):
    """A NON-ZERO, BC-CONFORMING initial guess.

    **These are two separate requirements meeting two separate needs and the
    docstring says so on purpose.** Non-zero is what makes `B z` non-zero, so
    that a residual-only augmentation cannot reproduce the un-augmented answer
    in one Newton step and pass at the solver floor - that is section 14.3's
    reason. BC-conforming removes a confound: the clause-1 denominator is
    assembled with `bc.zero`, while the SNES's own residual carries `x - g` on
    the constrained rows, so a non-conforming guess makes the two normalisations
    differ by the bc violation and can put a floor under clause 1 that has
    nothing to do with the solve. Conformity cannot weaken the check.
    """
    rng = np.random.default_rng(20260812 + COMM_WORLD.rank)
    for f in z.subfunctions:
        d = f.dat.data
        if d.size:
            d[...] = scale * rng.standard_normal(d.shape)
    for bc in bcs:
        bc.apply(z)
    return z


def d1(solver, z, Z, args):
    """Plumbing: do the two call sites pass the same theta, rows and offset?"""
    op = solver.dtn_operator

    # residual arm, through the solver's own callback
    r = Function(Z)
    with z.dat.vec_ro as xv, r.dat.vec as rv:
        solver.augment_residual(xv, rv)
    b_res = _vec(r)

    # Jacobian arm, through the installed context
    J = derivative(solver.F, z, TrialFunction(Z))
    Jmat = assemble(J, bcs=solver.strong_bcs, mat_type="matfree").petscmat
    plain = Function(Z)
    with z.dat.vec_ro as xv, plain.dat.vec as yv:
        Jmat.mult(xv, yv)
    a_only = _vec(plain)
    solver.augment_jacobian(None, Jmat)
    aug = Function(Z)
    with z.dat.vec_ro as xv, aug.dat.vec as yv:
        Jmat.mult(xv, yv)
    b_jac = _vec(aug) - a_only

    if args.arm == "d1_no_theta":
        # Sabotage: one arm loses the prefactor. §14.3 measured 5.000e-01.
        b_res = b_res / op.theta_value

    num, den = gnorm(b_res - b_jac), gnorm(b_res)
    if den == 0.0:
        # **An explicit failure, not a `nan`.** `nan < gate` is False in Python,
        # so a `nan` reaches the verdict through the failure path by accident
        # rather than by design - and it would just as happily have reached a
        # PASS if the comparison had been written the other way round. A
        # zero-norm residual arm means that arm contributed nothing at all.
        say("  D1  FAILED: the residual arm produced |B_res x| = 0, so it "
            "contributed nothing. Nothing can be compared against it.")
        return float("inf"), 0.0
    return num / den, den


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtn-degree", type=int, default=5)
    p.add_argument("--dr", type=float, default=0.15)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--rtol", type=float, default=1e-12)
    p.add_argument("--norm-type", default="unpreconditioned",
                   choices=["unpreconditioned", "preconditioned"])
    p.add_argument("--arm", default="good",
                   choices=["good", "residual_only", "jacobian_only",
                            "d1_no_theta"])
    p.add_argument("--min-share", type=float, default=1e-3)
    args = p.parse_args(_ARGV[1:])

    solver, z, Z, layout = build(args)
    op = solver.dtn_operator

    # Sabotage arms, installed by disabling one call site. `residual_only` is
    # the §7.1 defect: the augmentation exists but the Jacobian never fires it.
    if args.arm == "residual_only":
        solver.augment_jacobian = lambda X, J: None
        solver.set_solver()
    elif args.arm == "jacobian_only":
        solver.augment_residual = lambda X, F: None
        solver.set_solver()

    say("=" * 78)
    say(f"D-GATE  arm={args.arm}  L={args.dtn_degree}  ranks={COMM_WORLD.size}")
    say("=" * 78)

    # ---- rule 16, before every other number ------------------------------
    # **Read from the OPERATOR, never through `solver.augment_residual`.** The
    # first version went through the callback, which the `jacobian_only` arm
    # replaces with a no-op, so the ratio came back 0.0000% there and 0.0015%
    # elsewhere - i.e. it reported which arm was running rather than any
    # property of `B`. A diagnostic that moves with the sabotage it is meant to
    # be independent of is not a diagnostic.
    randomise(z, bcs=solver.strong_bcs)
    f0 = residual(solver)
    b_only = Function(Z)
    with z.dat.vec_ro as xv, b_only.dat.vec as bv:
        op.apply_local(xv.array_r, bv.array_w)
    n_b, n_f = gnorm(_vec(b_only)), gnorm(f0)

    # `||J z||`, the UN-augmented Jacobian action on the same z. Wave 1
    # measured `||B psi|| / ||A psi||` at 0.31-0.39% on this mesh with this B,
    # so this is the number that says whether the operator is unchanged and only
    # the denominator differs, or whether B is genuinely weak in the coupled
    # context.
    J = derivative(solver.F, z, TrialFunction(Z))
    Jm = assemble(J, bcs=solver.strong_bcs, mat_type="matfree").petscmat
    jz = Function(Z)
    with z.dat.vec_ro as xv, jz.dat.vec as yv:
        Jm.mult(xv, yv)
    n_j = gnorm(_vec(jz))

    say(f"  norms    ||theta B z|| = {n_b:.6e}")
    say(f"           ||F(z)||      = {n_f:.6e}   (carries the LOAD, which is "
        "independent of z)")
    say(f"           ||J z||       = {n_j:.6e}   (un-augmented operator action)")
    share = n_b / n_f if n_f else float("nan")
    share_j = n_b / n_j if n_j else float("nan")
    del share_j
    # **The old rule-16 ratio is gone, and it was ill-posed rather than
    # miscalibrated.** Measured by the Forward reviewer: a load knob moves it
    # 3.7x and the arbitrary scale of the probe vector moves it 36x, while the
    # operator-action form is exactly constant under both - as it must be, since
    # both terms are linear in z. Worse, as the load grows the ratio FALLS while
    # the sabotage margin GROWS, 8.7 to 11.1 orders: the proxy anti-correlates
    # with the thing it proxied for. And its 0.1% floor was calibrated against
    # one miniature that itself sat at 0.016% while its test was live.
    #
    # What replaces it is the sabotage margin, in orders, computed across arms
    # by the driver - it needs no proxy, cannot be contaminated by a load and
    # cannot be gamed by a denominator, because it IS the failure being guarded
    # against, executed.
    #
    # This stays as a DIAGNOSTIC WITH NO FLOOR. The denominator recovers the
    # un-augmented action exactly and costs nothing: `J_aug z` is a matvec
    # already taken and `theta B z` is the rank-k apply already in the numerator.
    n_unaug = gnorm(_vec(jz) - _vec(b_only))
    say(f"  diagnostic  ||theta B z|| / ||J_aug z - theta B z|| = "
        f"{(n_b / n_unaug if n_unaug else float('nan')):.4%}   (no floor)")

    # ---- D1 --------------------------------------------------------------
    rel, scale = d1(solver, z, Z, args)
    say(f"  D1  |B_res x - B_jac x| / |B_res x| = {rel:.6e}   "
        f"(|B_res x| = {scale:.4e}, gate 1e-14)")
    say("      plumbing only: one implementation, so agreement is "
        "near-tautological.")

    # ---- D2 --------------------------------------------------------------
    randomise(z, bcs=solver.strong_bcs)
    r_init = gnorm(augmented_residual(solver, z))
    op.applications = op.jacobian_applications = 0
    solver.solve()
    c1 = gnorm(augmented_residual(solver, z)) / r_init
    c2 = gnorm(residual(solver)) / r_init
    say(f"  D2  clause 1  ||F+thetaBz|| / ||F0+thetaBz0|| = {c1:.6e}  "
        f"(gate < 1e-10)")
    say(f"      clause 2  ||F||        / ||F0+thetaBz0|| = {c2:.6e}  "
        f"(gate > 1e-6)")

    # ---- D3 --------------------------------------------------------------
    say(f"  D3  augmented mult fired {op.jacobian_applications} times "
        f"(gate > 0);  total apply_local {op.applications}")

    # No `min_share` term: the floor it enforced had no provenance and the
    # quantity behind it was ill-posed. The margin gate lives in the driver,
    # which is the only place that sees more than one arm.
    ok = (rel < 1e-14 and c1 < 1e-10 and c2 > 1e-6
          and op.jacobian_applications > 0)
    say("=" * 78)
    say(f"  D-GATE {'PASS' if ok else 'FAIL'}   arm={args.arm}")
    say("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
