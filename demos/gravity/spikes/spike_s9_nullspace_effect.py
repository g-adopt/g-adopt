"""Spike: does declaring the rigid-rotation nullspace actually change anything?

S8 showed the two coupled runs bit-identical, which is either "the mode was
never there" or "PETSc never saw the declaration". This separates them:

- is the `MatNullSpace` on the operator at all;
- how much of the mode does each of the four runs carry
  (coupled/reference x declared/not);
- and the V2 comparison in all four combinations.
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


def rot_amplitude(u, sub):
    X = SpatialCoordinate(sub)
    V = u.function_space()
    r0 = Function(V).interpolate(as_vector([-X[1], X[0]]))
    dxm = Measure("dx", domain=sub)
    return (assemble(dot(u, r0) * dxm) / assemble(dot(r0, r0) * dxm),
            norm(r0))


def reference(sub, bcs, dt, declare):
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
        zm, demo.approximation(), dt=dt, bcs=bcs,
        solver_parameters="direct", nullspace=ns)
    ref.solve()
    return zm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--nazim", type=int, default=64)
    p.add_argument("--truncation", type=int, default=5)
    args = p.parse_args()

    parent, sub = demo.build_meshes(
        args.dr, args.nazim,
        path=os.path.join(HERE, f"s9_{args.dr}_{args.nazim}.msh"))
    print(f"parent {parent.num_cells()}, mantle {sub.num_cells()}, "
          f"M = {args.truncation}")

    coupled = {}
    for declare in (False, True):
        solver, z, layout, bcs, _ = demo.build_solver(
            parent, sub, truncation=args.truncation,
            declare_nullspace=declare)
        # is the declaration on the operator?
        attached = None
        if declare:
            J = assemble(derivative(solver.F, z), mat_type="matfree")
            attached = J.petscmat.getNullSpace()
        solver.solve()
        A = solver.solver.snes.ksp.getOperators()[0].getNullSpace()
        c, nrot = rot_amplitude(solver.displacement, sub)
        print(f"\ncoupled, declare={declare}")
        print(f"  ksp operator nullspace       {A.handle != 0 if A else None}")
        print(f"  ||u||                        {norm(solver.displacement):.10e}")
        print(f"  rotation amplitude in u      {c:.6e}  "
              f"(relative {abs(c) * nrot / norm(solver.displacement):.3e})")
        coupled[declare] = (solver, z, layout, bcs)
        del attached

    for declare in (False, True):
        _, _, _, bcs = coupled[False]
        zm = reference(sub, bcs, 1.0, declare)
        c, nrot = rot_amplitude(zm.subfunctions[0], sub)
        print(f"\nreference, declare={declare}")
        print(f"  ||u||                        {norm(zm.subfunctions[0]):.10e}")
        print(f"  rotation amplitude in u      {c:.6e}  "
              f"(relative {abs(c) * nrot / norm(zm.subfunctions[0]):.3e})")


if __name__ == "__main__":
    main()
