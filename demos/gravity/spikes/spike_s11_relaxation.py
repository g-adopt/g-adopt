"""Spike: is the demo configuration's fluid limit finite at all?

The first run of the time loop grew without bound. Three candidates, separated
here:

1. the *uncoupled* mechanics, a plain `CoupledInternalVariableSolver` on the
   mantle with the same load and boundary conditions - full `B_mu`, so the
   hydrostatic-prestress restoring term is at full strength and self-gravity is
   simply absent;
2. the coupled system with the loop gain turned down, `Lambda` scaled by a
   factor while `B_mu` is untouched;
3. the coupled system as the demo builds it.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
from gadopt import *  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import selfgrav_gia_annulus as demo  # noqa: E402
import generate_selfgrav_annulus as gen  # noqa: E402


def deflection(u, sub):
    X = SpatialCoordinate(sub)
    n = X / sqrt(dot(X, X))
    dss = Measure("ds", domain=sub)(gen.CURVE_RE)
    length = assemble(Constant(1.0) * dss)
    return 2 * assemble(dot(u, n) * cos(2 * atan2(X[1], X[0])) * dss) / length


def uncoupled_with_free_surface(sub, dt, steps, sign):
    """Uncoupled mechanics with an explicit Airy restoring stress at Re.

    `hydrostatic_prestress_advection_and_buoyancy_term` is a *volume* term, and
    a uniform `rho_0` has no gradient, so the only place the surface density
    contrast can enter is the free-surface condition - which
    `CompressibleInternalVariableApproximation.hydrostatic_prestress_advection`
    returns 0 for, on the grounds that it is absorbed into the volume term.
    Test whether adding it back gives the Airy fluid limit.
    """
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    Zm = MixedFunctionSpace([V, S])
    zm = Function(Zm)
    Xm = SpatialCoordinate(sub)
    n = Xm / sqrt(dot(Xm, Xm))
    u_r = dot(split(zm)[0], n)
    bcs = {
        gen.CURVE_RC: {"un": 0.0},
        gen.CURVE_RE: {"normal_stress":
                       demo.B_MU * (demo.SIGMA_HAT
                                    * cos(2 * atan2(Xm[1], Xm[0]))
                                    + sign * u_r)},
    }
    basis = VectorSpaceBasis(
        [Function(Zm.sub(0)).interpolate(as_vector([-Xm[1], Xm[0]]))])
    basis.orthonormalize()
    solver = CoupledInternalVariableSolver(
        zm, demo.approximation(), dt=dt, bcs=bcs, solver_parameters="direct",
        nullspace=MixedVectorSpaceBasis(Zm, [basis, Zm.sub(1)]))
    print(f"\nUNCOUPLED + free-surface restoring stress, sign {sign:+g}")
    prev = None
    for step in range(1, steps + 1):
        solver.solve()
        basis.orthogonalize(zm.subfunctions[0])
        z = deflection(zm.subfunctions[0], sub)
        ch = abs(z - prev) / abs(z) if prev else float("nan")
        if step in (1, 2, 5, 10, 20, 40, 80) or step == steps:
            print(f"  step {step:4d}  t {step * dt:8.2f}  deflection "
                  f"{z:16.10e}  rel change {ch:.3e}")
        prev = z
    print(f"  Airy prediction -sigma_hat/rho_0 = {-demo.SIGMA_HAT:.6e}")


def uncoupled(sub, dt, steps):
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    Zm = MixedFunctionSpace([V, S])
    zm = Function(Zm)
    Xm = SpatialCoordinate(sub)
    bcs = {
        gen.CURVE_RC: {"un": 0.0},
        gen.CURVE_RE: {"normal_stress":
                       demo.B_MU * demo.SIGMA_HAT
                       * cos(2 * atan2(Xm[1], Xm[0]))},
    }
    basis = VectorSpaceBasis(
        [Function(Zm.sub(0)).interpolate(as_vector([-Xm[1], Xm[0]]))])
    basis.orthonormalize()
    solver = CoupledInternalVariableSolver(
        zm, demo.approximation(), dt=dt, bcs=bcs, solver_parameters="direct",
        nullspace=MixedVectorSpaceBasis(Zm, [basis, Zm.sub(1)]))
    print("\nUNCOUPLED mechanics (no self-gravity, prestress term at full B_mu)")
    prev = None
    for step in range(1, steps + 1):
        solver.solve()
        basis.orthogonalize(zm.subfunctions[0])
        z = deflection(zm.subfunctions[0], sub)
        ch = abs(z - prev) / abs(z) if prev else float("nan")
        if step in (1, 2, 5, 10, 20, 40, 80) or step == steps:
            print(f"  step {step:4d}  t {step * dt:8.2f}  deflection "
                  f"{z:16.10e}  rel change {ch:.3e}")
        prev = z
    print(f"  Airy prediction -sigma_hat/rho_0 = {-demo.SIGMA_HAT:.6e}")


def coupled(parent, sub, dt, steps, lam_factor):
    lam = demo.LAMBDA * lam_factor
    saved_lam, demo.LAMBDA = demo.LAMBDA, lam
    saved_approx = demo.approximation

    def make(density=1.0):
        return CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=density, shear_modulus=1.0,
            viscosity=1.0, g=1.0, B_mu=demo.B_MU, self_gravity_number=lam)
    demo.approximation = make
    try:
        solver, z, layout, bcs, _ = demo.build_solver(
            parent, sub, dt=dt, truncation=3)
    finally:
        demo.LAMBDA = saved_lam
        demo.approximation = saved_approx

    print(f"\nCOUPLED, Lambda = {lam:.6f} ({lam_factor:g} x nominal)")
    prev = None
    for step in range(1, steps + 1):
        solver.solve()
        zeta = demo.deflection_amplitude(solver)
        ch = abs(zeta - prev) / abs(zeta) if prev else float("nan")
        if step in (1, 2, 5, 10, 20, 40, 80) or step == steps:
            print(f"  step {step:4d}  t {step * dt:8.2f}  deflection "
                  f"{zeta:16.10e}  rel change {ch:.3e}  dev stress "
                  f"{demo.fluid_limit_residual(solver):.3e}")
        prev = zeta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dr", type=float, default=0.2)
    p.add_argument("--nazim", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--full", action="store_true")
    args = p.parse_args()

    parent, sub = demo.build_meshes(
        args.dr, args.nazim,
        path=os.path.join(HERE, f"s11_{args.dr}_{args.nazim}.msh"))
    uncoupled_with_free_surface(sub, args.dt, args.steps, +1.0)
    uncoupled_with_free_surface(sub, args.dt, args.steps, -1.0)
    if args.full:
        uncoupled(sub, args.dt, args.steps)
        for f in (0.0001, 0.25, 0.5, 1.0):
            coupled(parent, sub, args.dt, args.steps, f)


if __name__ == "__main__":
    main()
