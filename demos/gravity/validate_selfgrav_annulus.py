"""Acceptance gate for the four-region parent annulus.

Phase 2 of `NOTES/PLAN-MONOLITHIC-SELFGRAV.md`.  Two stages, and the second
only runs if the first is green, because an accuracy number measured on a mesh
whose measures are wrong is worse than no number at all.

**Stage 1, geometry.**  Areas of the three cell groups; lengths of all four
tagged circles; the two interface circles as *interior* facets of the parent
(`dS`) and as *boundary* facets of the mantle submesh (`ds`), which must agree;
`curve_mesh`'s P2 correction taking the straight-facet O(h^2) errors down to
roundoff; and the buffer's cell count as a fraction of the mantle's.

Spike S1's warning drives the shape of this: a `dS` or `ds` given the wrong
kind of tag returns **zero and a warning**, not an exception, so every measure
is asserted *positive* before it is asserted accurate.  A test that only
checked `abs(measured - expected) < tol` would pass a silently-empty subdomain
if the tolerance were ever loosened, and would fail with a misleading message
if it were not.

**Stage 2, the boundary-treatment residual.**  Road map §1.2 predicts that with
the exterior boundary stood off to 2 Re, an *untreated* azimuthal mode m leaves
a relative residual

    eps_m(r) = (r / R)^(2m) * (m - alpha) / (m + alpha),    alpha = 1

at radius r, which at r = Re and R = 2 Re is 1.7e-4 for m = 6 and *falls* with
m — the opposite of the usual truncation worry, and the reason L = 5 is
enough.

Measured by solving a single-mode shell source twice on the same mesh: once
with `CylindricalDtN(M=5)`, leaving the mode untreated, and once with `M = m`,
treating it.  The headline number is the *difference* of the two, because the
two solves share their discretisation error and it cancels; what survives is
the spurious growing branch the untreated boundary injects, which is exactly
what eps_m measures.  Absolute errors against the closed form are printed
alongside as context, but at any affordable resolution they are dominated by
CG2 discretisation (~1e-4) and could not resolve eps at m = 10 (~8e-7) at all.
A null control — two truncations that both treat the mode — pins the floor of
the difference measurement.

The reference potential is the closed form written as UFL, so it carries no
interpolation error, and it is checked against `passess`'s own `psi_m` through
an integral identity rather than by pointwise evaluation.

Measured 2026-07-30 at n_azimuthal = 256, dr_mantle = 0.025, CG2, with the
shell 50 km thick at 500 km depth:

    m   eps 2-D pred   A/B trace   ratio      abs err M=5   abs err M=m
    6      1.744e-04   1.748e-04   1.002        8.717e-05     1.817e-05
    7      4.578e-05   4.587e-05   1.002        3.394e-05     2.889e-05
    8      1.187e-05   1.190e-05   1.003        4.312e-05     4.321e-05
    9      3.052e-06   3.067e-06   1.005        6.157e-05     6.163e-05
   10      7.803e-07   7.868e-07   1.008        8.464e-05     8.466e-05

    null control (m = 6 treated by both M = 6 and M = 9): 2.287e-16

Two things that table says.  The prediction holds to under 1 % over four
orders of magnitude, and the residual improves monotonically with m.  And the
absolute error columns cannot see any of it — by m = 8 they agree to three
digits — which is why the A/B is the instrument and not a convenience: getting
the same answer from absolute errors would need the CG2 floor below 8e-7, some
hundred times the work for a worse number.

Note that road map §1.2 tabulates the **3-D** eps (exponent 2l+1) while this
is a 2-D problem (exponent 2m).  At m = 6, R = 2 Re the 2-D value is 1.7e-4
where the road map's table prints 1.0e-4.  Same order, same conclusion, but
the brief's "~1e-4 at m = 6" is the 3-D number.
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see SPIKE-RESULTS.md S2(d)
import numpy as np
from firedrake import (
    COMM_WORLD, And, Function, FunctionSpace, Measure, Mesh, SpatialCoordinate,
    Submesh, VectorFunctionSpace, assemble, atan2, avg, conditional, cos, dS,
    ds, dx, sqrt)
from gadopt import CylindricalDtN, GravitySolver
from scipy.integrate import quad

import generate_selfgrav_annulus as gen
from generate_selfgrav_annulus import (
    CELL_BUFFER, CELL_INNER, CELL_MANTLE, CURVE_INNER, CURVE_OUTER, CURVE_RC,
    CURVE_RE, RC, RE, R_OUTER)

def _poisson_polar_2d():
    """Import `passess`'s closed-form solution, lazily and at the point of use.

    **This import is deliberately not at module scope, and the reason cost a
    production job.** `passess` is a local checkout at `~/Workplace/passess`; it
    is not installed in the Firedrake venv and it does not exist on Gadi at all.
    Only `epsilon_stage` needs it. But `validate_selfgrav_sphere` imports
    `Report` from this module so that there is one definition of it rather than
    two, and a module-scope import made a 3-D *geometry* gate depend on a 2-D
    analytic Poisson library it never calls. The first cluster run of the
    displacement campaign died in the import, before a single line of physics.

    Keep any dependency that is not available everywhere this module is
    *imported from* out of module scope, however convenient it is to read.
    """
    sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))
    from passess.polar import PoissonPolar2D
    return PoissonPolar2D

#: Rewritten by every stage, so two runs of this script must not share it -
#: hence `--mesh-file`.  Under MPI only rank 0 writes; gmsh is not collective.
MESH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "selfgrav_annulus.msh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def curve_mesh(linear_mesh):
    """Remap to quadratic (P2) coordinates so mesh circles become curved.

    Verbatim from `gravity_poisson_robin.py` minus the origin guard, which
    exists there for the disc and is dead weight on an annulus: r >= 0.5 Rc
    everywhere here.  Edge midpoints are pushed radially onto the linear
    interpolant of the vertex radii, so an edge whose vertices lie on a circle
    becomes a quadratic arc on that circle.
    """
    X = SpatialCoordinate(linear_mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        (r_p1 / r) * X)
    return Mesh(X_p2)


class Report:
    """Collects checks so every number is printed even after one fails."""

    def __init__(self):
        self.failures = []

    def check(self, name, measured, expected, rtol, positive=True):
        rel = abs(measured - expected) / abs(expected)
        # S1: an empty subdomain assembles to zero and only warns.  Say so
        # explicitly rather than letting it read as a tolerance failure.
        empty = positive and abs(measured) < 1e-30
        ok = (not empty) and rel <= rtol
        flag = "ok " if ok else ("EMPTY" if empty else "FAIL")
        print(f"  [{flag}] {name:<34s} {measured: .12e}  "
              f"expected {expected: .12e}  rel {rel:.2e}")
        if not ok:
            self.failures.append(name)
        return ok

    def note(self, name, value, fmt="{: .12e}"):
        print(f"  [ -- ] {name:<34s} " + fmt.format(value))

    def done(self, stage):
        if self.failures:
            print(f"\n{stage}: FAILED — {', '.join(self.failures)}")
            return False
        print(f"\n{stage}: all checks passed")
        return True


def build_mesh(**kwargs):
    """gmsh on rank 0 only; every rank then reads the same file.

    gmsh is not collective and all ranks writing one path is a race that
    usually produces a truncated file rather than an error.
    """
    zones = None
    if COMM_WORLD.rank == 0:
        _, zones = gen.generate(MESH_FILE, **kwargs)
    zones = COMM_WORLD.bcast(zones, root=0)
    return zones


def cell_count(mesh, tag):
    """Cells carrying a given cell tag, owned only, summed over ranks."""
    owned = mesh.cell_set.size
    local = int(np.count_nonzero(mesh.cell_subset(tag).indices < owned))
    return mesh.comm.allreduce(local)


# ---------------------------------------------------------------------------
# Stage 1 — geometry
# ---------------------------------------------------------------------------
def geometry_stage(dr_mantle, n_azimuthal, quads, interfaces=()):
    ref = gen.analytic()
    print(f"\nGenerating: dr_mantle={dr_mantle}, n_azimuthal={n_azimuthal}, "
          f"quads={quads}, interfaces={interfaces}")
    zones = build_mesh(dr_mantle=dr_mantle, n_azimuthal=n_azimuthal,
                       interfaces=interfaces, quads=quads)
    for r_in, r_out, n, coef, tag in zones:
        print(f"    {r_in:7.4f} -> {r_out:7.4f}  {n:3d} layers  "
              f"progression {coef:5.3f}  tag {tag}")

    rep = Report()
    parent = Mesh(MESH_FILE)

    # --- straight facets: everything should sit at the polygon-vs-circle
    # --- O(h^2) error, which is what makes the curved comparison meaningful.
    print("\nStraight-facet parent (before curve_mesh):")
    straight = {}
    for name, tag, expect in [
            ("area mantle dx(101)", CELL_MANTLE, ref["area_mantle"]),
            ("area inner  dx(102)", CELL_INNER, ref["area_inner"]),
            ("area buffer dx(103)", CELL_BUFFER, ref["area_buffer"]),
    ]:
        straight[name] = assemble(1 * dx(tag, domain=parent))
        rep.check(name, straight[name], expect, 5e-3)
    for name, tag, expect in [
            ("len 2Re  ds(4)", CURVE_OUTER, ref["len_outer"]),
            ("len 0.5Rc ds(5)", CURVE_INNER, ref["len_inner"]),
    ]:
        straight[name] = assemble(1 * ds(tag, domain=parent))
        rep.check(name, straight[name], expect, 5e-3)
    for name, tag, expect in [
            ("len Re   dS(2) interior", CURVE_RE, ref["len_Re"]),
            ("len Rc   dS(3) interior", CURVE_RC, ref["len_Rc"]),
    ]:
        straight[name] = assemble(avg(1) * dS(tag, domain=parent))
        rep.check(name, straight[name], expect, 5e-3)

    # --- the submesh, straight off the gmsh cell tag (spike S2: no
    # --- RelabeledMesh needed).
    print("\nMantle submesh of the straight parent, Submesh(parent, 2, 101):")
    sub = Submesh(parent, 2, CELL_MANTLE)
    rep.check("area sub dx", assemble(1 * dx(domain=sub)),
              ref["area_mantle"], 5e-3)
    for name, tag, expect in [("sub ds(2) = Re", CURVE_RE, ref["len_Re"]),
                              ("sub ds(3) = Rc", CURVE_RC, ref["len_Rc"])]:
        rep.check(name, assemble(1 * ds(tag, domain=sub)), expect, 5e-3)
    # The parent's interior view and the submesh's boundary view are the same
    # facets: they must agree to the last digit, not merely to tolerance.
    for tag, label in [(CURVE_RE, "Re"), (CURVE_RC, "Rc")]:
        d = abs(assemble(avg(1) * dS(tag, domain=parent))
                - assemble(1 * ds(tag, domain=sub)))
        rep.check(f"dS({tag}) - ds({tag}) identical [{label}]", d + 1.0, 1.0,
                  1e-14, positive=False)

    # --- the same measures on the wrong kind of tag, to demonstrate the trap
    # --- rather than merely cite it.  These SHOULD be zero.
    print("\nS1's silent trap, exercised deliberately (zero expected):")
    rep.note("ds(2) on parent [Re is interior]",
             assemble(1 * ds(CURVE_RE, domain=parent)))
    rep.note("dS(4) on parent [2Re is exterior]",
             assemble(avg(1) * dS(CURVE_OUTER, domain=parent)))

    # --- P2 isoparametric correction.  The residual after curving is not a
    # --- fixed 1e-11: a quadratic arc through three points of a circle
    # --- deviates from it at O(theta^4) in the azimuthal cell angle, so the
    # --- tolerance has to follow the resolution.  Empirically the constants
    # --- are ~1.0e-3 theta^4 on a circumference and ~2.1e-3 theta^4 on an
    # --- area (see the sweep printed by --sweep-curved); a factor of five of
    # --- headroom still rejects a straight-facet mesh by three orders.
    theta = 2 * np.pi / n_azimuthal
    tol_len, tol_area = 5e-3 * theta**4, 1.1e-2 * theta**4
    print(f"\nCurved parent (curve_mesh, P2 coordinates), theta^4 tolerances "
          f"{tol_len:.2e} / {tol_area:.2e}:")
    parent_c = curve_mesh(parent)
    for name, tag, expect in [
            ("area mantle dx(101)", CELL_MANTLE, ref["area_mantle"]),
            ("area inner  dx(102)", CELL_INNER, ref["area_inner"]),
            ("area buffer dx(103)", CELL_BUFFER, ref["area_buffer"]),
    ]:
        rep.check(name, assemble(1 * dx(tag, domain=parent_c)), expect,
                  tol_area)
    for name, tag, expect in [
            ("len 2Re  ds(4)", CURVE_OUTER, ref["len_outer"]),
            ("len 0.5Rc ds(5)", CURVE_INNER, ref["len_inner"]),
    ]:
        rep.check(name, assemble(1 * ds(tag, domain=parent_c)), expect,
                  tol_len)
    for name, tag, expect in [
            ("len Re   dS(2) interior", CURVE_RE, ref["len_Re"]),
            ("len Rc   dS(3) interior", CURVE_RC, ref["len_Rc"]),
    ]:
        rep.check(name, assemble(avg(1) * dS(tag, domain=parent_c)),
                  expect, tol_len)
    # Confirm the residual is geometry and not under-integration.
    hi = assemble(1 * dx(CELL_MANTLE, domain=parent_c, degree=12))
    rep.note("area mantle at quad degree 12", hi)

    # --- does the submesh inherit the parent's P2 coordinates?  The recorded
    # --- answer in demos/gravity/CLAUDE.md is no.  Establish it here, because
    # --- under Stack B the mechanics runs on this submesh.
    print("\nDoes Submesh inherit the curved parent's P2 coordinates?")
    sub_c = Submesh(parent_c, 2, CELL_MANTLE)
    sub_deg = sub_c.coordinates.function_space().ufl_element().degree()
    par_deg = parent_c.coordinates.function_space().ufl_element().degree()
    print(f"    parent coordinate degree {par_deg}, "
          f"submesh coordinate degree {sub_deg}")
    a_sub_c = assemble(1 * dx(domain=sub_c))
    rel = abs(a_sub_c - ref["area_mantle"]) / ref["area_mantle"]
    inherits = rel < 1e-10
    print(f"    area of Submesh(curved parent) {a_sub_c: .12e}  "
          f"rel {rel:.2e}  ->  "
          f"{'INHERITS' if inherits else 'does NOT inherit'}")
    for tag, label in [(CURVE_RE, "Re"), (CURVE_RC, "Rc")]:
        v = assemble(1 * ds(tag, domain=sub_c))
        e = ref["len_Re"] if tag == CURVE_RE else ref["len_Rc"]
        print(f"    ds({tag}) [{label}] {v: .12e}  rel "
              f"{abs(v - e) / e:.2e}")

    # --- the workaround, and whether it survives cross-mesh assembly, which
    # --- is the thing that actually matters for the coupled solve.
    print("\nWorkaround: curve_mesh the submesh separately.")
    sub_cc = curve_mesh(sub_c)
    rep.check("area curve_mesh(submesh)", assemble(1 * dx(domain=sub_cc)),
              ref["area_mantle"], tol_area)
    for tag, label, expect in [(CURVE_RE, "Re", ref["len_Re"]),
                               (CURVE_RC, "Rc", ref["len_Rc"])]:
        rep.check(f"curved sub ds({tag}) [{label}]",
                  assemble(1 * ds(tag, domain=sub_cc)), expect, tol_len)

    # Does the coupled solve's cross-mesh assembly survive re-curving the
    # submesh?  A parent-mesh coefficient integrated over the submesh with an
    # intersected measure is exactly the shape of both u<->psi coupling terms,
    # and `x^2` (unlike anything odd) has a non-trivial answer, so this
    # measures the entity maps rather than only exercising them.
    want = np.pi * (RE**4 - RC**4) / 4
    for label, mesh_m in [("curved sub x curved parent", sub_cc),
                          ("plain  sub x curved parent", sub_c)]:
        dx_m = Measure("dx", domain=mesh_m,
                       intersect_measures=(Measure("dx", domain=parent_c),))
        f = Function(FunctionSpace(parent_c, "CG", 2)).interpolate(
            SpatialCoordinate(parent_c)[0]**2)
        try:
            rep.check(f"int x^2, {label}", assemble(f * dx_m), want,
                      tol_area if mesh_m is sub_cc else 5e-3)
        except Exception as exc:  # noqa: BLE001 — the point is what it raises
            print(f"    int x^2, {label}: {type(exc).__name__}: {exc}")
            rep.failures.append(f"cross-mesh {label}")

    # --- cost.
    print("\nCell counts:")
    n_mantle = cell_count(parent, CELL_MANTLE)
    n_inner = cell_count(parent, CELL_INNER)
    n_buffer = cell_count(parent, CELL_BUFFER)
    total = n_mantle + n_inner + n_buffer
    print(f"    mantle {n_mantle:8d}")
    print(f"    inner  {n_inner:8d}   ({n_inner / n_mantle:.3f} of mantle)")
    print(f"    buffer {n_buffer:8d}   ({n_buffer / n_mantle:.3f} of mantle)"
          "   <- road map §1.3 predicts < 1/3")
    print(f"    total  {total:8d}   ({total / n_mantle:.3f} of mantle)")

    return rep.done("Stage 1 (geometry)"), dict(
        n_mantle=n_mantle, n_inner=n_inner, n_buffer=n_buffer)


def curved_sweep(dr_mantle, quads, azimuthals=(64, 128, 256, 512)):
    """Shows that the post-curve_mesh residual is the O(theta^4) arc error.

    Worth printing once rather than asserting: it is what turns "the errors
    drop to ~1e-11" (a statement about one mesh, the n_azimuthal = 512 one the
    E3 record was written on) into a rule that transfers to any resolution.
    """
    ref = gen.analytic()
    print("\ncurve_mesh residual vs azimuthal resolution:")
    print(f"{'n_azim':>7s} {'theta^4':>10s} {'rel(area)':>11s} {'order':>7s} "
          f"{'rel(len Re)':>12s} {'order':>7s}")
    prev = None
    for n in azimuthals:
        build_mesh(dr_mantle=dr_mantle, n_azimuthal=n, quads=quads)
        mesh = curve_mesh(Mesh(MESH_FILE))
        ra = abs(assemble(1 * dx(CELL_MANTLE, domain=mesh))
                 - ref["area_mantle"]) / ref["area_mantle"]
        rl = abs(assemble(avg(1) * dS(CURVE_RE, domain=mesh))
                 - ref["len_Re"]) / ref["len_Re"]
        oa = ol = float("nan")
        if prev is not None:
            oa = np.log(prev[0] / ra) / np.log(2)
            ol = np.log(prev[1] / rl) / np.log(2)
        print(f"{n:7d} {(2 * np.pi / n)**4:10.2e} {ra:11.3e} {oa:7.2f} "
              f"{rl:12.3e} {ol:7.2f}")
        prev = (ra, rl)


def cost_sweep(growth=1.3):
    """Layer ladders across resolution: the buffer/mantle cell-count ratio.

    Cell counts are `4 * (n_azimuthal//4) * n_layers` per region, so the ratio
    depends only on the radial ladder and is independent of n_azimuthal.  Road
    map §1.3 predicts the buffer costs under a third of the mantle in 2-D and
    says to measure rather than trust it.
    """
    print(f"\nRadial layer ladder vs dr_mantle (growth {growth}):")
    print(f"{'dr_mantle':>10s} {'inner':>7s} {'mantle':>7s} {'buffer':>7s} "
          f"{'buf/man':>8s} {'total/man':>10s}")
    for dr in (0.1, 0.05, 0.025, 0.0125, 0.00625):
        zones = gen.radial_zones(dr, growth)
        n = {tag: 0 for tag in (CELL_INNER, CELL_MANTLE, CELL_BUFFER)}
        for _, _, layers, _, tag in zones:
            n[tag] += layers
        print(f"{dr:10.5f} {n[CELL_INNER]:7d} {n[CELL_MANTLE]:7d} "
              f"{n[CELL_BUFFER]:7d} {n[CELL_BUFFER] / n[CELL_MANTLE]:8.3f} "
              f"{sum(n.values()) / n[CELL_MANTLE]:10.3f}")


# ---------------------------------------------------------------------------
# Stage 2 — the boundary-treatment residual
# ---------------------------------------------------------------------------
def shell_potential_ufl(X, m, rho_m, r1, r2, gamma):
    """Closed-form psi for a cos(m phi) shell source, as exact UFL.

    The three branches of `passess.polar.PoissonPolar2D._unit_mneq0`, written
    for UFL so the reference carries no interpolation error of its own.  Valid
    for m != 0, 2 (m = 2 needs the logarithmic outer branch).
    """
    assert m not in (0, 2), "m = 0 and m = 2 need the log branch"
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan2(X[1], X[0])
    ei, eo = m + 2, 2 - m

    inner = (conditional(r <= r1, 0.0,
                         conditional(r >= r2, r2**ei, r**ei) - r1**ei) / ei)
    outer = ((r2**eo - conditional(r <= r1, r1**eo,
                                   conditional(r >= r2, r2**eo, r**eo))) / eo)
    unit = 2 * np.pi * (r**(-m) * inner + r**m * outer) / m
    return gamma * rho_m * unit * cos(m * phi)


def epsilon_stage(dr_mantle, n_azimuthal, degree, modes, M_untreated,
                  quads, gamma=1.0, rho_m=1.0):
    # Shell near the top of the mantle, as in the standalone gravity tests:
    # 50 km thick at 500 km depth, non-dimensionalised by D = 2891 km.
    D_km = 2891.0
    r_centre = RE - 500.0 / D_km
    r1 = r_centre - 50.0 / (2 * D_km)
    r2 = r_centre + 50.0 / (2 * D_km)

    print(f"\nGenerating for the eps study: dr_mantle={dr_mantle}, "
          f"n_azimuthal={n_azimuthal}, shell [{r1:.6f}, {r2:.6f}]")
    build_mesh(dr_mantle=dr_mantle, n_azimuthal=n_azimuthal,
               interfaces=(r1, r2), quads=quads)
    parent = curve_mesh(Mesh(MESH_FILE))
    X = SpatialCoordinate(parent)
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan2(X[1], X[0])

    print("""
Predicted eps_m(Re) = (Re / 2Re)^(2m) (m-1)/(m+1) with alpha = 1
(gravity_solver.py:698), against the 3-D form road map §1.2 tabulates.

The headline measurement is the A/B *difference* between the same problem
solved with the mode untreated (M = %d) and treated (M = m) on the same mesh.
The two solves share their discretisation error almost exactly - only the
boundary rows differ - so it cancels in the difference, and what is left is
the spurious r^m branch the untreated boundary injects, which is what eps_m
is.  The absolute errors against the closed form are printed beside it as
context: they are dominated by CG%d discretisation at these resolutions and
so cannot resolve eps below about 1e-4 on their own.
""" % (M_untreated, degree))
    print(f"{'m':>3s} {'eps 2-D':>10s} {'eps 3-D':>10s} | "
          f"{'A/B trace':>10s} {'ratio':>7s} | {'A/B L2':>10s} {'ratio':>7s} | "
          f"{'abs M=%d' % M_untreated:>10s} {'abs M=m':>10s}")

    rows = []
    PoissonPolar2D = _poisson_polar_2d()
    for m in modes:
        analytic = PoissonPolar2D(m=m, rho_m=rho_m, r1=r1, r2=r2, gamma=gamma)
        ref_ufl = shell_potential_ufl(X, m, rho_m, r1, r2, gamma)

        # Sanity: the UFL closed form against passess, without interpolating
        # either.  int_mantle psi^2 dA = pi int_Rc^Re psi_m(r)^2 r dr, the
        # right-hand side by scipy quadrature on passess's own psi_m, split at
        # the shell radii where its derivative kinks.  The residual is the
        # mesh's O(theta^4) geometry error, nothing else.
        dq_ref = 2 * m + 2 * degree + 8
        got = assemble(ref_ufl**2 * dx(CELL_MANTLE, domain=parent,
                                       degree=dq_ref))
        want = np.pi * quad(lambda t: analytic.psi_m(t).real**2 * t,
                            RC, RE, points=[r1, r2], limit=200)[0]
        assert abs(got - want) <= 1e-7 * want, (
            f"UFL reference disagrees with passess at m={m}: {got} vs {want}")

        rho = conditional(And(r >= r1, r <= r2), rho_m * cos(m * phi), 0.0)

        dq = 2 * m + 2 * degree + 6
        d_Re = dS(CURVE_RE, domain=parent, degree=dq)
        d_man = dx(CELL_MANTLE, domain=parent, degree=dq)

        errs, psis = {}, {}
        for label, M in [("untreated", M_untreated), ("treated", m)]:
            psi = Function(FunctionSpace(parent, "CG", degree))
            GravitySolver(
                psi, rho,
                bcs={CURVE_OUTER: {"dtn": CylindricalDtN(M=M)},
                     CURVE_INNER: {"dtn": CylindricalDtN(M=M)}},
                gravitational_constant=gamma,
                source_quad_degree=dq).solve()
            psis[label] = psi
            errs[label + "_trace"] = np.sqrt(
                assemble(avg((psi - ref_ufl)**2) * d_Re)
                / assemble(avg(ref_ufl**2) * d_Re))
            errs[label + "_l2"] = np.sqrt(
                assemble((psi - ref_ufl)**2 * d_man)
                / assemble(ref_ufl**2 * d_man))

        diff = psis["untreated"] - psis["treated"]
        errs["ab_trace"] = np.sqrt(
            assemble(avg(diff**2) * d_Re)
            / assemble(avg(psis["treated"]**2) * d_Re))
        errs["ab_l2"] = np.sqrt(
            assemble(diff**2 * d_man)
            / assemble(psis["treated"]**2 * d_man))

        pred2 = (RE / R_OUTER)**(2 * m) * (m - 1) / (m + 1)
        pred3 = (RE / R_OUTER)**(2 * m + 1) * m / (m + 1)
        print(f"{m:3d} {pred2:10.3e} {pred3:10.3e} | "
              f"{errs['ab_trace']:10.3e} {errs['ab_trace'] / pred2:7.3f} | "
              f"{errs['ab_l2']:10.3e} {errs['ab_l2'] / pred2:7.3f} | "
              f"{errs['untreated_l2']:10.3e} {errs['treated_l2']:10.3e}")
        rows.append((m, pred2, pred3, errs))

    print("\nThe distinctive prediction is monotone improvement with m:")
    seq = [row[3]["ab_trace"] for row in rows]
    improving = all(b < a for a, b in zip(seq, seq[1:]))
    print(f"    A/B trace residual  {['%.3e' % s for s in seq]}")
    print(f"    monotonically decreasing in m: {improving}")
    ratios = [row[3]["ab_trace"] / row[1] for row in rows]
    print(f"    measured / predicted {['%.3f' % s for s in ratios]}")

    # Null control: the same A/B between two truncations that BOTH treat the
    # mode.  Whatever this returns is the floor of the measurement, and the
    # table above is only worth reading down to it.
    m = modes[0]
    rho = conditional(And(r >= r1, r <= r2), rho_m * cos(m * phi), 0.0)
    dq = 2 * m + 2 * degree + 6
    d_Re = dS(CURVE_RE, domain=parent, degree=dq)
    ctrl = []
    for M in (m, m + 3):
        psi = Function(FunctionSpace(parent, "CG", degree))
        GravitySolver(psi, rho,
                      bcs={CURVE_OUTER: {"dtn": CylindricalDtN(M=M)},
                           CURVE_INNER: {"dtn": CylindricalDtN(M=M)}},
                      gravitational_constant=gamma,
                      source_quad_degree=dq).solve()
        ctrl.append(psi)
    floor = np.sqrt(assemble(avg((ctrl[0] - ctrl[1])**2) * d_Re)
                    / assemble(avg(ctrl[1]**2) * d_Re))
    print(f"\nNull control, m = {m} treated by both M = {m} and M = {m + 3}: "
          f"A/B trace {floor:.3e}")
    print("    (the measurement floor; the table is meaningful above it)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["geometry", "epsilon", "all"])
    ap.add_argument("--dr-mantle", type=float, default=0.05)
    ap.add_argument("--n-azimuthal", type=int, default=128)
    ap.add_argument("--eps-dr-mantle", type=float, default=0.025)
    ap.add_argument("--eps-n-azimuthal", type=int, default=256)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--modes", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--triangles", action="store_true")
    ap.add_argument("--sweeps", action="store_true",
                    help="also print the theta^4 curving sweep and the "
                         "layer-ladder cost table")
    ap.add_argument("--mesh-file", default=MESH_FILE)
    args = ap.parse_args()
    quads = not args.triangles
    MESH_FILE = args.mesh_file

    ok = True
    if args.stage in ("geometry", "all"):
        ok, _ = geometry_stage(args.dr_mantle, args.n_azimuthal, quads)
        if args.sweeps:
            cost_sweep()
            curved_sweep(args.dr_mantle, quads)
    if args.stage == "epsilon" or (args.stage == "all" and ok):
        epsilon_stage(args.eps_dr_mantle, args.eps_n_azimuthal, args.degree,
                      args.modes, args.M, quads)
    elif args.stage == "all":
        print("\nStage 2 skipped: stage 1 did not pass.")
        sys.exit(1)
