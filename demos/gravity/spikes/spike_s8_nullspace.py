"""Spike: can a MixedVectorSpaceBasis be built over the coupled space?

Three questions, in order:

1. does `Function(Z.sub(0))` work on a mixed space spanning two meshes with 21
   `Real` blocks in it;
2. does `MixedVectorSpaceBasis(Z, [VectorSpaceBasis([rot]), Z.sub(1), ...])`
   construct;
3. does the solve still converge with it attached, through
   `DtNTwoBlockSchurPC` and a matfree operator?

Plus the measurement D-2 asks for: `||J u_rot|| / ||u_rot||`.
"""
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)

import selfgrav_gia_annulus as demo  # noqa: E402


def main():
    parent, sub = demo.build_meshes(
        0.2, 32, path=os.path.join(HERE, "s8_0.2_32.msh"))
    solver, z, layout, bcs, C = demo.build_solver(parent, sub, dt=1.0,
                                                  truncation=3)
    Z = z.function_space()
    print(f"fields {len(Z)}, dim {Z.dim()}")

    # 1. the rotation mode as a Function on the displacement sub-space
    X = SpatialCoordinate(sub)
    rot = Function(Z.sub(layout.displacement)).interpolate(
        as_vector([-X[1], X[0]]))
    print(f"1. Function(Z.sub(0)) OK, ||rot|| = {norm(rot):.6e}")

    # 2. the mixed basis
    entries = [Z.sub(i) for i in range(len(Z))]
    vsb = VectorSpaceBasis([rot])
    vsb.orthonormalize()
    entries[layout.displacement] = vsb
    ns = MixedVectorSpaceBasis(Z, entries)
    print(f"2. MixedVectorSpaceBasis OK: {ns}")

    # the measurement: J applied to the mode, embedded in the coupled space
    probe = Function(Z)
    probe.subfunctions[layout.displacement].assign(rot)
    J = assemble(derivative(solver.F, z), mat_type="matfree")
    with probe.dat.vec_ro as xv:
        yv = J.petscmat.createVecLeft()
        J.petscmat.mult(xv, yv)
        print(f"   ||J u_rot|| / ||u_rot|| = {yv.norm() / xv.norm():.6e}")

    # 3. does the solve survive it
    solver2, z2, layout2, _, _ = demo.build_solver(
        parent, sub, dt=1.0, truncation=3, nullspace=ns)
    solver2.solve()
    print(f"3. solve with nullspace OK, ||u|| = "
          f"{norm(solver2.displacement):.8e}")

    solver.solve()
    print(f"   solve without,            ||u|| = "
          f"{norm(solver.displacement):.8e}")

    d = Function(Z.sub(layout.displacement))
    d.dat.data[:] = (solver2.displacement.dat.data_ro
                     - solver.displacement.dat.data_ro)
    dxm = Measure("dx", domain=sub)
    r0 = Function(Z.sub(layout.displacement)).interpolate(
        as_vector([-X[1], X[0]]))
    c = assemble(dot(d, r0) * dxm) / assemble(dot(r0, r0) * dxm)
    rest = Function(Z.sub(layout.displacement))
    rest.dat.data[:] = d.dat.data_ro - c * r0.dat.data_ro
    print(f"   difference {norm(d):.6e}, rotation amplitude {c:.6e}, "
          f"rest {norm(rest):.6e}")
    print(f"   projection of the nullspace-declared answer onto the mode: "
          f"{assemble(dot(solver2.displacement, r0) * dxm) / assemble(dot(r0, r0) * dxm):.6e}")
    print(f"   projection of the undeclared answer onto the mode:         "
          f"{assemble(dot(solver.displacement, r0) * dxm) / assemble(dot(r0, r0) * dxm):.6e}")

    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
