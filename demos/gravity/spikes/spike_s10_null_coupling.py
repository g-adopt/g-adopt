"""D-1 and D-2, measured together: `B_mu = 0` with no workaround.

Runs the null coupling exactly as the review specifies it - `B_mu = 0` in the
approximation, nothing pinned, nothing subclassed - and compares against a
plain `CoupledInternalVariableSolver` on the submesh alone. Both sides declare
the rigid-rotation kernel, so the comparison is of the solutions and not of an
arbitrary multiple of a zero-energy mode.

The rejection region is the same comparison with `B_mu` restored.
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


def approximation_factory(B_mu):
    def make(density=1.0):
        return CompressibleInternalVariableApproximation(
            bulk_modulus=1.0, density=density, shear_modulus=1.0,
            viscosity=1.0, g=1.0, B_mu=B_mu,
            self_gravity_number=demo.LAMBDA)
    return make


def reference(sub, bcs, dt, make, declare=True):
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    Zm = MixedFunctionSpace([V, S])
    zm = Function(Zm)
    ns = None
    if declare:
        X = SpatialCoordinate(sub)
        basis = VectorSpaceBasis(
            [Function(Zm.sub(0)).interpolate(as_vector([-X[1], X[0]]))])
        basis.orthonormalize()
        ns = MixedVectorSpaceBasis(Zm, [basis, Zm.sub(1)])
    ref = CoupledInternalVariableSolver(
        zm, make(), dt=dt, bcs=bcs, solver_parameters="direct", nullspace=ns)
    ref.solve()
    if ns is not None:
        basis.orthogonalize(zm.subfunctions[0])
    return zm


def compare(z, layout, zm):
    out = {}
    uc = z.subfunctions[layout.displacement]
    ur = zm.subfunctions[0]
    out["u"] = (norm(assemble(uc - ur)) / norm(ur),
                float(np.abs(uc.dat.data_ro - ur.dat.data_ro).max()),
                float(np.abs(ur.dat.data_ro).max()))
    for k, i in enumerate(layout.internal_variables):
        mc, mr = z.subfunctions[i], zm.subfunctions[1 + k]
        out[f"m{k + 1}"] = (norm(assemble(mc - mr)) / norm(mr),
                            float(np.abs(mc.dat.data_ro
                                         - mr.dat.data_ro).max()),
                            float(np.abs(mr.dat.data_ro).max()))
    return out


def run(parent, sub, B_mu, dt, truncation, declare=True):
    make = approximation_factory(B_mu)
    saved, demo.approximation = demo.approximation, make
    try:
        solver, z, layout, bcs, _ = demo.build_solver(
            parent, sub, dt=dt, truncation=truncation,
            declare_nullspace=declare)
        solver.solve()
    finally:
        demo.approximation = saved
    zm = reference(sub, bcs, dt, make, declare=declare)
    return solver, z, layout, zm, compare(z, layout, zm)


def rot_split(uc, ur, sub):
    V = ur.function_space()
    d = Function(V)
    d.dat.data[:] = uc.dat.data_ro - ur.dat.data_ro
    X = SpatialCoordinate(sub)
    r0 = Function(V).interpolate(as_vector([-X[1], X[0]]))
    dxm = Measure("dx", domain=sub)
    c = assemble(dot(d, r0) * dxm) / assemble(dot(r0, r0) * dxm)
    rest = Function(V)
    rest.dat.data[:] = d.dat.data_ro - c * r0.dat.data_ro
    return norm(d), c, norm(rest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--nazim", type=int, default=64)
    p.add_argument("--truncation", type=int, default=5)
    p.add_argument("--dt", type=float, default=1.0)
    args = p.parse_args()

    parent, sub = demo.build_meshes(
        args.dr, args.nazim,
        path=os.path.join(HERE, f"s10_{args.dr}_{args.nazim}.msh"))
    print(f"parent {parent.num_cells()}, mantle {sub.num_cells()}, "
          f"M = {args.truncation}, rotation ON, dt {args.dt}")

    for declare in (False, True):
        solver, z, layout, zm, diffs = run(
            parent, sub, 0.0, args.dt, args.truncation, declare=declare)
        print(f"\nB_mu = 0, nullspace declared = {declare}")
        print(f"  theta_psi      {float(solver.theta_psi):.6e}")
        print(f"  theta_rot(m3)  {float(solver._theta_rot(2)):.6e}")
        print(f"  ||u||   {norm(solver.displacement):.8e}   "
              f"reference {norm(zm.subfunctions[0]):.8e}")
        print(f"  ||psi|| {norm(solver.potential):.8e}  (nonzero: the "
              "divergence source still drives it)")
        print(f"  m3 {solver.rotation_values()}  "
              f"inertia {solver.inertia_perturbation()}")
        for name, (rel, dmax, vmax) in diffs.items():
            print(f"  {name:<4s} rel L2 {rel:.6e}   max nodal diff {dmax:.6e}"
                  f"   max value {vmax:.6e}")
        nd, c, nrest = rot_split(solver.displacement, zm.subfunctions[0], sub)
        print(f"  difference split: ||d|| {nd:.6e}, rotation amplitude "
              f"{c:.6e}, remainder {nrest:.6e}")

    _, _, _, _, coupled = run(parent, sub, demo.B_MU, args.dt, args.truncation)
    print(f"\nrejection region: B_mu restored to {demo.B_MU}")
    for name, (rel, dmax, vmax) in coupled.items():
        print(f"  {name:<4s} rel L2 {rel:.6e}   max nodal diff {dmax:.6e}"
              f"   max value {vmax:.6e}")


if __name__ == "__main__":
    main()
