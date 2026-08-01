"""S1 -- a gmsh physical curve interior to the mesh, seen from Firedrake.

The ice/ocean load sheet lives on the Re circle, which is an *interior* facet
of the parent mesh (mantle on one side, buffer on the other).  This spike asks
whether the tag survives the reader and what idiom gives 2 pi R.

`dS` visits each interior facet once, but every integrand is two-valued, so a
bare `1*dS(2)` is ambiguous and UFL will refuse it.  The point of the spike is
to establish which restriction gives the right number.

Run at 1, 2 and 4 ranks: interior facets on a rank boundary are the classic
place for a double count.
"""
import os
import sys
import traceback

import gadopt  # noqa: F401  -- import order, see SPIKE-RESULTS.md S2(d)
from firedrake import *
from firedrake.petsc import PETSc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_mesh  # noqa: E402

RE, RC = spike_mesh.CURVE_RE, spike_mesh.CURVE_RC
OUTER, INNER = spike_mesh.CURVE_OUTER, spike_mesh.CURVE_INNER


def pr(*a):
    PETSc.Sys.Print(*a)


def check(label, fn, expected=None):
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        pr(f"  RAISES  {label}")
        pr(f"          {type(exc).__name__}: {str(exc).splitlines()[0][:170]}")
        pr(f"          at {tb.filename}:{tb.lineno} in {tb.name}")
        return None
    if expected is None:
        pr(f"  OK      {label:<58} {value}")
    else:
        rel = abs(value - expected) / abs(expected)
        verdict = "MATCHES" if rel < 5e-3 else ("2x" if abs(value - 2 * expected)
                                                / abs(expected) < 5e-3 else "WRONG")
        pr(f"  OK      {label:<58} {value:>14.8f}   "
           f"expect {expected:.8f}  rel {rel:.2e}  {verdict}")
    return value


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    msh = os.path.join(here, "spike_annulus.msh")
    if not os.path.exists(msh):
        spike_mesh.generate(msh)

    parent = Mesh(msh)
    ana = spike_mesh.analytic()
    two_pi_Re, two_pi_Rc = ana["len_Re"], ana["len_Rc"]

    pr("=" * 78)
    pr(f"S1  tagged interior facet          [{COMM_WORLD.size} rank(s)]")
    pr("=" * 78)

    pr("\n-- does the tag survive the reader at all?")
    check("exterior_facets markers", lambda: sorted(
        int(t) for t in parent.exterior_facets.unique_markers))
    check("interior_facets markers", lambda: sorted(
        int(t) for t in parent.interior_facets.unique_markers))

    pr("\n-- unique_markers lists the whole label set, so cross-check which "
       "\n   facets each measure actually visits")
    check("assemble(avg(1)*dS(4))  [outer boundary; must be 0]",
          lambda: assemble(avg(Constant(1)) * dS(OUTER, domain=parent)))
    check("assemble(1*ds(2))       [Re is interior; must be 0]",
          lambda: assemble(Constant(1) * ds(RE, domain=parent)))

    pr("\n-- the outer boundaries, as a control (ds, single-valued)")
    check("assemble(1*ds(4))   [2 Re]", lambda: assemble(
        Constant(1) * ds(OUTER, domain=parent)), ana["len_outer"])
    check("assemble(1*ds(5))   [0.5 Rc]", lambda: assemble(
        Constant(1) * ds(INNER, domain=parent)), ana["len_inner"])

    pr("\n-- the interior circle Re (tag 2), the load sheet's home")
    check("assemble(1*dS(2))                 [bare, ambiguous]",
          lambda: assemble(Constant(1) * dS(RE, domain=parent)), two_pi_Re)
    check("assemble(avg(1)*dS(2))", lambda: assemble(
        avg(Constant(1)) * dS(RE, domain=parent)), two_pi_Re)
    check("assemble(Constant(1)('+')*dS(2))", lambda: assemble(
        Constant(1)("+") * dS(RE, domain=parent)), two_pi_Re)
    check("assemble(jump(1)*dS(2))           [must be ~0]", lambda: assemble(
        jump(Constant(1)) * dS(RE, domain=parent)))

    pr("\n-- and with a genuinely two-valued integrand: a DG0 region marker")
    DG0 = FunctionSpace(parent, "DG", 0)
    X = SpatialCoordinate(parent)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    # 1 inside the mantle, 0 in the buffer: jumps across the Re circle.
    chi = Function(DG0).interpolate(
        conditional(And(r >= spike_mesh.RC, r <= spike_mesh.RE), 1.0, 0.0))
    check("assemble(avg(chi)*dS(2))          [expect pi Re]",
          lambda: assemble(avg(chi) * dS(RE, domain=parent)), two_pi_Re / 2)
    check("assemble(abs(jump(chi))*dS(2))    [expect 2 pi Re]",
          lambda: assemble(abs(jump(chi)) * dS(RE, domain=parent)), two_pi_Re)
    check("assemble(chi('+')*dS(2)) [side-dependent, NOT reproducible]",
          lambda: assemble(chi("+") * dS(RE, domain=parent)))
    check("assemble(chi('-')*dS(2)) [side-dependent, NOT reproducible]",
          lambda: assemble(chi("-") * dS(RE, domain=parent)))

    pr("\n-- the same for the Rc circle (tag 3)")
    check("assemble(avg(1)*dS(3))", lambda: assemble(
        avg(Constant(1)) * dS(RC, domain=parent)), two_pi_Rc)

    pr("\n-- a CG field: is the trace on the interior facet single-valued?")
    Vc = FunctionSpace(parent, "CG", 2)
    f = Function(Vc).interpolate(r)
    check("assemble(avg(f)*dS(2))            [expect 2 pi Re * Re]",
          lambda: assemble(avg(f) * dS(RE, domain=parent)),
          two_pi_Re * spike_mesh.RE)
    check("assemble(abs(jump(f))*dS(2))      [must be ~0]",
          lambda: assemble(abs(jump(f)) * dS(RE, domain=parent)))

    pr("\n-- the FacetNormal on the tagged interior facet")
    n = FacetNormal(parent)
    check("assemble(dot(avg(X), n('+'))*dS(2))  [+/- 2 pi Re^2, sign is "
          "whichever side gmsh gave]",
          lambda: assemble(dot(avg(X), n("+")) * dS(RE, domain=parent)))
    check("assemble(abs(dot(avg(X), n('+')))*dS(2))  [expect 2 pi Re^2]",
          lambda: assemble(abs(dot(avg(X), n("+"))) * dS(RE, domain=parent)),
          two_pi_Re * spike_mesh.RE)

    pr("\n-- the SAME circle seen from the submesh, where it is exterior")
    sub = Submesh(parent, 2, spike_mesh.CELL_MANTLE)
    check("assemble(1*ds(2, domain=sub))", lambda: assemble(
        Constant(1) * ds(RE, domain=sub)), two_pi_Re)
    check("assemble(1*ds(3, domain=sub))", lambda: assemble(
        Constant(1) * ds(RC, domain=sub)), two_pi_Rc)
    n_s = FacetNormal(sub)
    X_s = SpatialCoordinate(sub)
    check("assemble(dot(X, n)*ds(2, domain=sub))  [outward: +2 pi Re^2]",
          lambda: assemble(dot(X_s, n_s) * ds(RE, domain=sub)),
          two_pi_Re * spike_mesh.RE)
    check("assemble(dot(X, n)*ds(3, domain=sub))  [inward: -2 pi Rc^2]",
          lambda: assemble(dot(X_s, n_s) * ds(RC, domain=sub)),
          -two_pi_Rc * spike_mesh.RC)


if __name__ == "__main__":
    main()
