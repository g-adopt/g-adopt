"""V2 — the null coupling: with `B_mu = 0` the mechanics must be untouched.

`B_mu = 0`, **not** `Lambda = 0`.  With `Lambda = 0` the potential block loses
its source while the momentum block keeps its self-gravity term, so the psi
half of the test becomes vacuous (review D2).  Zeroing `B_mu` switches off the
two body forces instead: the mechanics is then exactly uncoupled and must
reproduce a plain `CoupledInternalVariableSolver` on the submesh alone, while
psi stays driven by the real divergence source, so the potential half stays a
genuine one-way test.

What this certifies is that nothing in the two-mesh mixed assembly — the
intersected measures, the 21 `Real` multiplier rows, the rotation row, the
second mesh in the same space — has broken the mechanics.  It certifies nothing
about the coupling, because the coupling is what it switches off.

The last block of the run is the rejection region: the same comparison with
`B_mu` restored, which must be large or V2 is measuring nothing.

## V2 as specified does not run, and that is the first result

`theta_psi = scaling_factor * B_mu / Lambda` and
`theta_rot_i = s_i scaling_factor B_mu Omega^2` both carry `B_mu`, and both
multiply *whole residual rows*.  At `B_mu = 0` they are zero, so the potential
row, all 21 DtN constraint rows and the rotation row are annihilated — measured
here: `max|A_(psi,psi)| = 0`, `max|A_(c_0,psi)| = 0` — and the Jacobian has 23
identically zero rows.  The solve fails with `DIVERGED_LINEAR_SOLVE` at the
first Krylov iteration.  Loud rather than silent, but the message names the
linear solver and not the cause.

The gate therefore runs the null coupling in the only form that is meaningful:
`B_mu = 0` in the two body forces, with `theta_psi` and `theta_rot` held at the
values they take at the nominal `B_mu`.  That is exactly what the review asks
for — the mechanics exactly uncoupled, psi still driven by the divergence
source — and the row scaling it pins is unobservable in the solution anyway
(review D1), so nothing is being tuned.  The symmetry constraint that fixes
`theta_psi` is vacuous at `B_mu = 0`, since the block it is derived from is
zero.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import selfgrav_gia_annulus as demo  # noqa: E402


class NominalScalingSolver(SelfGravitatingGIASolver):
    """`theta_psi` and `theta_rot` at the nominal `B_mu`, whatever `B_mu` is.

    Only for the null-coupling gate.  The two constants are row scalings, which
    are unobservable in every field the solver produces; pinning them here is
    what lets `B_mu = 0` switch off the body forces without also deleting the
    potential equation.  Nothing else in the class is touched, so the mechanics
    and the potential source are the production ones.
    """

    nominal_B_mu = None

    @property
    def theta_psi(self):
        return self.scaling_factor * self.nominal_B_mu / self.Lambda

    def _theta_rot(self, i):
        return (self.CLOSURE_SIGNS[i] * self.scaling_factor
                * self.nominal_B_mu * self.Omega_sq)


def approximation_factory(B_mu):
    """A fresh approximation each call: the solvers mutate `approximation.mu`."""
    def make(density=1.0):
        return CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=density, shear_modulus=1.0,
            viscosity=1.0, g=1.0, B_mu=B_mu,
            self_gravity_number=demo.LAMBDA)
    return make


def reference(sub, bcs, dt, make_approximation):
    """`CoupledInternalVariableSolver` on the mantle alone, same everything."""
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    zm = Function(MixedFunctionSpace([V, S]))
    ref = CoupledInternalVariableSolver(
        zm, make_approximation(), dt=dt, bcs=bcs,
        solver_parameters="direct")
    ref.solve()
    return zm


def rigid_rotation_split(uc, ur, sub):
    """Split `u_coupled - u_reference` into a rigid rotation and the rest.

    The mechanics as configured — free slip `un = 0` at the CMB, traction at
    the surface — has a rigid rotation `(-y, x)` as a zero-energy mode: it is
    divergence free, strain free and tangential to both circles.  In the
    continuum the operator is therefore *singular*; discretely it is only
    saved by the facet geometry, and nothing in the solver declares a
    nullspace.  So two solvers of the same system may land on different
    multiples of it, and a raw displacement comparison measures that rather
    than anything about the coupling.  Report both halves.
    """
    V = ur.function_space()
    d = Function(V)
    d.dat.data[:] = uc.dat.data_ro - ur.dat.data_ro
    X = SpatialCoordinate(sub)
    rot = Function(V).interpolate(as_vector([-X[1], X[0]]))
    dxm = Measure("dx", domain=sub)
    c = assemble(dot(d, rot) * dxm) / assemble(dot(rot, rot) * dxm)
    rest = Function(V)
    rest.dat.data[:] = d.dat.data_ro - c * rot.dat.data_ro
    return norm(d), c, norm(rest)


def compare(z_coupled, layout, zm):
    """Relative L2 and max nodal differences, field by field."""
    out = {}
    uc = z_coupled.subfunctions[layout.displacement]
    ur = zm.subfunctions[0]
    out["u"] = (norm(assemble(uc - ur)) / norm(ur),
                float(np.abs(uc.dat.data_ro - ur.dat.data_ro).max()),
                float(np.abs(ur.dat.data_ro).max()))
    for k, i in enumerate(layout.internal_variables):
        mc = z_coupled.subfunctions[i]
        mr = zm.subfunctions[1 + k]
        out[f"m{k + 1}"] = (norm(assemble(mc - mr)) / norm(mr),
                            float(np.abs(mc.dat.data_ro
                                         - mr.dat.data_ro).max()),
                            float(np.abs(mr.dat.data_ro).max()))
    return out


def run(parent, sub, B_mu, dt, truncation, rotation=True, nominal=None):
    """The demo's own configuration, with `B_mu` swapped and nothing else.

    `demo.build_solver` builds the meshes' boundary conditions, the load, the
    gravity boundary conditions and the polar moment; only the approximation
    factory and (for the null coupling) the solver class are substituted, so
    the configuration under test is the demo's.
    """
    make = approximation_factory(B_mu)
    saved_approx, demo.approximation = demo.approximation, make
    saved_cls = demo.SelfGravitatingGIASolver
    if nominal is not None:
        cls = type("PinnedScaling", (NominalScalingSolver,),
                   {"nominal_B_mu": nominal})
        demo.SelfGravitatingGIASolver = cls
    try:
        solver, z, layout, bcs, C = demo.build_solver(
            parent, sub, dt=dt, truncation=truncation, rotation=rotation)
        # V2 is written for the rigid core and only for it. `reference` builds
        # a plain `CoupledInternalVariableSolver` from these same `bcs`, so
        # with a `FluidCore` configured the coupled side would carry a CMB
        # traction the reference has no term for, and the reference's mantle
        # would be traction-free there. That is a different mechanical problem,
        # compared silently, and it would read as a coupling defect. The guard
        # is here rather than in the demo because this is the file that owns
        # the comparison; the fluid core has its own gates in
        # `gate_fluidcore.py`.
        cmb = bcs.get(demo.gen.CURVE_RC, {})
        if solver.fluid_core is not None or "un" not in cmb:
            raise RuntimeError(
                "V2 compares the coupled mechanics against a plain "
                "CoupledInternalVariableSolver built from the same boundary "
                "conditions, which is only a like-for-like comparison with the "
                "rigid core (`un = 0` at Rc). This configuration has a fluid "
                "core, whose CMB traction the reference carries no term for. "
                "Run V2 on the rigid core, and gate_fluidcore.py on the fluid "
                "one.")
        solver.solve()
    finally:
        demo.approximation = saved_approx
        demo.SelfGravitatingGIASolver = saved_cls
    zm = reference(sub, bcs, dt, make)
    return solver, z, layout, zm, compare(z, layout, zm)


def show_degenerate(parent, sub, dt, truncation):
    """V2 exactly as specified: build at `B_mu = 0` and report what happens."""
    make = approximation_factory(0.0)
    saved, demo.approximation = demo.approximation, make
    try:
        solver, z, layout, _, _ = demo.build_solver(
            parent, sub, dt=dt, truncation=truncation, rotation=True)
    finally:
        demo.approximation = saved
    A = assemble(derivative(solver.F, z), mat_type="nest").petscmat

    def amax(i, j):
        M = A.getNestSubMatrix(i, j)
        return 0.0 if M is None else float(
            np.abs(M.convert("dense").getDenseArray()).max())

    print("\n  V2 EXACTLY AS SPECIFIED (B_mu = 0 everywhere)")
    print(f"    theta_psi      {float(solver.theta_psi):.3e}")
    print(f"    theta_rot(m3)  {float(solver._theta_rot(2)):.3e}")
    print(f"    max|A(psi,psi)|  {amax(layout.potential, layout.potential):.3e}")
    print(f"    max|A(psi,u)|    "
          f"{amax(layout.potential, layout.displacement):.3e}")
    print(f"    max|A(c_0,psi)|  "
          f"{amax(layout.multipliers[0], layout.potential):.3e}")
    try:
        solver.solve()
        print("    solve: converged (unexpected)")
    except Exception as exc:
        print(f"    solve: {type(exc).__name__}: "
              f"{str(exc).strip().splitlines()[-1].strip()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--nazim", type=int, default=64)
    p.add_argument("--truncation", type=int, default=5)
    p.add_argument("--dt", type=float, default=1.0)
    args = p.parse_args()

    parent, sub = demo.build_meshes(
        args.dr, args.nazim,
        path=os.path.join(HERE, f"v2_{args.dr}_{args.nazim}.msh"))
    print(f"V2 - null coupling, B_mu = 0.  parent {parent.num_cells()} cells, "
          f"mantle {sub.num_cells()}, M = {args.truncation}, rotation ON")

    show_degenerate(parent, sub, args.dt, args.truncation)

    print("\n  V2 AS RUN: B_mu = 0 in the body forces, theta_psi and theta_rot "
          f"pinned at the nominal B_mu = {demo.B_MU}")
    solver, z, layout, zm, diffs = run(
        parent, sub, 0.0, args.dt, args.truncation, nominal=demo.B_MU)
    print(f"\n  ||u||   coupled {norm(solver.displacement):.8e}   "
          f"reference {norm(zm.subfunctions[0]):.8e}")
    print(f"  ||psi|| {norm(solver.potential):.8e}   "
          "(nonzero: the potential is still driven by the divergence source, "
          "so the psi half of the test is not vacuous)")
    print(f"  m3 {solver.rotation_values()}   "
          f"inertia {solver.inertia_perturbation()}")
    print("\n  field   rel L2 difference   max nodal diff   max nodal value")
    worst = 0.0
    for name, (rel, dmax, vmax) in diffs.items():
        worst = max(worst, rel)
        print(f"  {name:<6s}  {rel:.6e}       {dmax:.6e}     {vmax:.6e}")
    nd, c_rot, n_rest = rigid_rotation_split(
        solver.displacement, zm.subfunctions[0], sub)
    nref = norm(zm.subfunctions[0])
    print("\n  the displacement difference, split against the rigid rotation "
          "(-y, x)")
    print(f"    ||u_c - u_r||                      {nd:.6e}")
    print(f"    rotation amplitude c               {c_rot:.6e}")
    print(f"    ||u_c - u_r - c (-y,x)||           {n_rest:.6e}")
    print(f"    fraction of the difference left    {n_rest / nd:.3e}")
    print(f"    strain-carrying part, relative     {n_rest / nref:.6e}")
    worst_real = max(n_rest / nref,
                     max(v[0] for k, v in diffs.items() if k != "u"))
    print(f"\n  worst relative difference, raw              {worst:.3e}")
    print(f"  worst relative difference, rotation removed {worst_real:.3e}   "
          f"{'PASS' if worst_real < 1e-9 else 'FAIL'} (threshold 1e-9)")

    # Rejection region: the same comparison with the coupling restored.
    _, _, _, _, coupled_diffs = run(
        parent, sub, demo.B_MU, args.dt, args.truncation)
    print("\n  rejection region - same comparison with B_mu restored to "
          f"{demo.B_MU}")
    for name, (rel, dmax, vmax) in coupled_diffs.items():
        print(f"  {name:<6s}  {rel:.6e}       {dmax:.6e}     {vmax:.6e}")
    print("  (that is also the V6 number in advance: self-gravity changes the "
          "displacement by this much.)")


if __name__ == "__main__":
    main()
