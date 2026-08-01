"""S3 -- what StokesSolverBase.__init__ does to a two-mesh mixed space.

`gadopt/stokes_integrators.py:276-277` is

    self.mesh = self.solution_space.mesh()
    self.k = upward_normal(self.mesh)

and `set_boundary_conditions` (line 306) then does `is_cartesian(self.mesh)`,
`self.mesh.geometric_dimension`, and builds `bc_map` from
`self.solution_space.sub(0)`.

This spike answers, empirically and without touching gadopt: what does each of
those return or raise when the mixed space spans a parent mesh and a submesh,
and what would a subclass have to pre-set.
"""
import os
import sys
import traceback

import gadopt  # noqa: F401  -- import order, see SPIKE-RESULTS.md S2(d)
from gadopt.utility import is_cartesian, upward_normal, vertical_component
from firedrake import *
from firedrake.petsc import PETSc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_mesh  # noqa: E402


def pr(*a):
    PETSc.Sys.Print(*a)


def probe(label, fn):
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        pr(f"  RAISES  {label}")
        pr(f"          {type(exc).__name__}: {str(exc).splitlines()[0][:180]}")
        pr(f"          at {tb.filename}:{tb.lineno} in {tb.name}")
        return None
    pr(f"  OK      {label}")
    pr(f"          -> {value!r}"[:400])
    return value


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    msh = os.path.join(here, "spike_annulus.msh")
    if not os.path.exists(msh):
        spike_mesh.generate(msh)

    parent = Mesh(msh)
    sub = Submesh(parent, 2, spike_mesh.CELL_MANTLE)

    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    P = FunctionSpace(parent, "CG", 2)
    R = FunctionSpace(parent, "R", 0)

    pr("=" * 72)
    pr(f"S3  StokesSolverBase on a two-mesh mixed space  "
       f"[{COMM_WORLD.size} rank(s)]")
    pr("=" * 72)

    for name, Z in (
        ("single-mesh control  Z = [V(sub), S(sub)]",
         MixedFunctionSpace([V, S])),
        ("two-mesh             Z = [V(sub), S(sub), Psi(parent)]",
         MixedFunctionSpace([V, S, P])),
        ("two-mesh + Real      Z = [V(sub), S(sub), Psi(parent), R, R]",
         MixedFunctionSpace([V, S, P, R, R])),
    ):
        pr(f"\n### {name}")

        pr("\n-- line 276:  self.mesh = self.solution_space.mesh()")
        m = probe("Z.mesh()", lambda Z=Z: Z.mesh())
        if m is not None:
            pr(f"          type          : {type(m).__module__}.{type(m).__name__}")
            pr(f"          is parent     : {m is parent}")
            pr(f"          is sub        : {m is sub}")
            probe("len(Z.mesh())", lambda m=m: len(m))
            probe("Z.mesh().unique()", lambda m=m: m.unique())

        pr("\n-- line 277:  upward_normal(self.mesh)")
        probe("is_cartesian(Z.mesh())", lambda m=m: is_cartesian(m))
        probe("upward_normal(Z.mesh())", lambda m=m: upward_normal(m))
        probe("Z.mesh().geometric_dimension",
              lambda m=m: m.geometric_dimension)
        probe("SpatialCoordinate(Z.mesh())", lambda m=m: SpatialCoordinate(m))

        pr("\n-- line 306ff: set_boundary_conditions")
        probe("Z.sub(0)  (the bc_map['u'] entry)", lambda Z=Z: Z.sub(0))
        probe("Z.sub(0).mesh() is sub",
              lambda Z=Z: Z.sub(0).mesh() is sub)
        probe("Z.sub(0).sub(0)  (bc_map['ux'])", lambda Z=Z: Z.sub(0).sub(0))
        probe("DirichletBC(Z.sub(0), (0,0), Rc=3)  [submesh boundary]",
              lambda Z=Z: len(DirichletBC(Z.sub(0), as_vector([0., 0.]),
                                          spike_mesh.CURVE_RC).nodes))
        probe("DirichletBC(Z.sub(0), (0,0), Re=2)  [submesh boundary]",
              lambda Z=Z: len(DirichletBC(Z.sub(0), as_vector([0., 0.]),
                                          spike_mesh.CURVE_RE).nodes))
        if len(Z) > 2:
            probe("DirichletBC(Z.sub(2), 0, outer=4)  [parent boundary]",
                  lambda Z=Z: len(DirichletBC(Z.sub(2), 0.,
                                              spike_mesh.CURVE_OUTER).nodes))

        pr("\n-- other things __init__ touches")
        probe("isinstance(Z.topological, MixedFunctionSpace)",
              lambda Z=Z: isinstance(
                  Z.topological, functionspaceimpl.MixedFunctionSpace))
        probe("TestFunctions(Z) length", lambda Z=Z: len(TestFunctions(Z)))
        probe("split(Function(Z)) length", lambda Z=Z: len(split(Function(Z))))
        probe("Function(Z).copy(deepcopy=True)",
              lambda Z=Z: type(Function(Z).copy(deepcopy=True)).__name__)

    # ------------------------------------------------------------------
    # What a subclass would have to pre-set: does the mechanics mesh work?
    # ------------------------------------------------------------------
    pr("\n### does mesh.cartesian propagate parent -> Submesh?")
    parent.cartesian = False
    probe("parent.cartesian set to False; is_cartesian(parent)",
          lambda: is_cartesian(parent))
    probe("is_cartesian(sub) -- inherited from the parent?",
          lambda: is_cartesian(sub))
    sub.cartesian = False
    probe("after sub.cartesian = False; is_cartesian(sub)",
          lambda: is_cartesian(sub))
    probe("is_cartesian(Z.mesh()) once BOTH are set",
          lambda: is_cartesian(MixedFunctionSpace([V, S, P, R, R]).mesh()))

    pr("\n### the fix: pre-set self.mesh to the MECHANICS mesh")
    Z = MixedFunctionSpace([V, S, P, R, R])
    probe("is_cartesian(sub)", lambda: is_cartesian(sub))
    probe("upward_normal(sub)", lambda: upward_normal(sub))
    probe("sub.geometric_dimension", lambda: sub.geometric_dimension)
    probe("vertical_component of a submesh Function",
          lambda: vertical_component(Function(Z.sub(0))))
    dx_s = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    probe("assemble( dot(upward_normal(sub), u) * dx_s )  with u = x",
          lambda: assemble(dot(upward_normal(sub),
                               Function(Z.sub(0)).interpolate(
                                   SpatialCoordinate(sub))) * dx_s))

    # And the thing that would silently pick the WRONG mesh:
    pr("\n### what happens if the potential mesh is used instead")
    probe("upward_normal(parent) integrated over dx_s (should still work, "
          "but it is the parent's coordinate field)",
          lambda: assemble(dot(upward_normal(parent),
                               Function(Z.sub(0)).interpolate(
                                   SpatialCoordinate(sub))) * dx_s))


if __name__ == "__main__":
    main()
