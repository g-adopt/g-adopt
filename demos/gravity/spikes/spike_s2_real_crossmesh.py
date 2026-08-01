"""S2 -- Real fields in a cross-mesh mixed space.

Question set (from NOTES/PLAN-MONOLITHIC-SELFGRAV.md, Phase 0):

  (a) does MixedFunctionSpace([V(sub), S(sub), Psi(parent), R, R, ...]) build?
  (b) does a residual mixing submesh dx, parent dx and parent ds assemble?
  (c) what does Z.dof_dset.field_ises give, and does it match what
      gadopt.DtNTwoBlockSchurPC expects?
  (d) does DtNTwoBlockSchurPC initialise on such an operator?

  and, crucially, WHICH MESH the Real spaces must live on.  Both kinds of
  constraint row are tested against both choices:

      "parent-boundary row"  ->  integrand over ds(parent)
      "submesh-volume row"   ->  integrand over dx(submesh)

Run:  PYTHONPATH=<worktree> python3 spike_s2_real_crossmesh.py
      mpiexec -n 2 ... ; mpiexec -n 4 ...
"""
import os
import sys
import traceback

import numpy as np

# gadopt (hence irksome) MUST be imported before anything runs a UFL
# multifunction, otherwise the deferred import that PETSc performs when it
# resolves `pc_python_type: gadopt.DtNTwoBlockSchurPC` dies with
# IrksomeImportOrderException, surfacing as an opaque PETSc error 101 out of
# SNESSetFromOptions.  See SPIKE-RESULTS.md, S2(d).
import gadopt  # noqa: F401
from firedrake import *
from firedrake.petsc import PETSc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_mesh  # noqa: E402

CELL_MANTLE = spike_mesh.CELL_MANTLE
CURVE_RE, CURVE_RC = spike_mesh.CURVE_RE, spike_mesh.CURVE_RC
CURVE_OUTER, CURVE_INNER = spike_mesh.CURVE_OUTER, spike_mesh.CURVE_INNER

N_REAL = 8


def pr(*a):
    PETSc.Sys.Print(*a)


FULL_ERRORS = os.environ.get("SPIKE_FULL_ERRORS", "0") == "1"


def report(label, fn):
    """Run fn, print PASS/FAIL with the exception origin, return (ok, value)."""
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 -- reporting is the point
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        pr(f"  FAIL {label}")
        if FULL_ERRORS:
            pr(traceback.format_exc())
        else:
            pr(f"       {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        pr(f"       raised at {tb.filename}:{tb.lineno}  in {tb.name}")
        return False, None
    pr(f"  PASS {label}   {value if value is not None else ''}")
    return True, value


def build_meshes(msh):
    parent = Mesh(msh)
    # Route A: straight from the gmsh cell label.
    sub = Submesh(parent, 2, CELL_MANTLE)
    return parent, sub


def build_meshes_relabelled(msh):
    """Route B: the idiom of submesh_2way_coupling.py lines 10-26."""
    parent0 = Mesh(msh)
    DG0 = FunctionSpace(parent0, "DG", 0)
    X = SpatialCoordinate(parent0)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    f_mantle = Function(DG0).interpolate(
        conditional(And(r >= spike_mesh.RC, r <= spike_mesh.RE), 1, 0))
    f_all = Function(DG0).interpolate(conditional(r >= 0, 1, 0))
    parent = RelabeledMesh(parent0, [f_mantle, f_all], [98, 99])
    sub = Submesh(parent, 2, 98)
    return parent, sub


def main():
    msh = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "spike_annulus.msh")
    if not os.path.exists(msh):
        spike_mesh.generate(msh)

    pr("=" * 72)
    pr(f"S2  Real fields in a cross-mesh mixed space   [{COMM_WORLD.size} rank(s)]")
    pr("=" * 72)

    pr("\n-- meshes")
    ok, res = report("Submesh straight from the gmsh cell tag 101",
                     lambda: build_meshes(msh))
    if ok:
        parent, sub = res
    else:
        ok, res = report("RelabeledMesh + Submesh (Dale's idiom)",
                         lambda: build_meshes_relabelled(msh))
        parent, sub = res
    report("area(submesh)", lambda: assemble(Constant(1) * dx(domain=sub)))
    report("area(parent)", lambda: assemble(Constant(1) * dx(domain=parent)))
    report("ds(2) on submesh (= 2 pi Re = %.6f)" % spike_mesh.analytic()["len_Re"],
           lambda: assemble(Constant(1) * ds(CURVE_RE, domain=sub)))
    report("ds(3) on submesh (= 2 pi Rc = %.6f)" % spike_mesh.analytic()["len_Rc"],
           lambda: assemble(Constant(1) * ds(CURVE_RC, domain=sub)))
    report("ds(4) on parent (= 2 pi 2Re = %.6f)" % spike_mesh.analytic()["len_outer"],
           lambda: assemble(Constant(1) * ds(CURVE_OUTER, domain=parent)))

    # ------------------------------------------------------------------
    # (a) construction, both choices of Real mesh
    # ------------------------------------------------------------------
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    P = FunctionSpace(parent, "CG", 2)

    spaces = {}
    for tagname, rmesh in (("R-on-parent", parent), ("R-on-sub", sub)):
        pr(f"\n-- (a) construction, {tagname}")
        R = FunctionSpace(rmesh, "R", 0)
        ok, Z = report(f"MixedFunctionSpace([V,S,Psi] + {N_REAL} x R)",
                       lambda R=R: MixedFunctionSpace([V, S, P] + [R] * N_REAL))
        if not ok:
            continue
        spaces[tagname] = Z
        report("  len(Z)", lambda Z=Z: len(Z))
        report("  Z.dim()", lambda Z=Z: Z.dim())
        report("  Function(Z)", lambda Z=Z: type(Function(Z)).__name__)
        ok2, m = report("  Z.mesh()", lambda Z=Z: Z.mesh())
        if ok2:
            pr(f"       -> {m!r}   is parent: {m is parent}  is sub: {m is sub}")

    if not spaces:
        pr("\nNo mixed space constructed; stopping.")
        return

    # ------------------------------------------------------------------
    # (b) residual mixing submesh volume, parent volume, parent boundary
    # ------------------------------------------------------------------
    dx_p = Measure("dx", domain=parent,
                   intersect_measures=(Measure("dx", domain=sub),))
    dx_s = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    ds_p_plain = Measure("ds", domain=parent)
    # A parent-boundary measure that DECLARES the submesh, so that Arguments
    # living on the submesh (a Real space built on `sub`) are admissible in a
    # parent-boundary integral.
    ds_p_int = Measure("ds", domain=parent,
                       intersect_measures=(Measure("dx", domain=sub),))

    for tagname, Z in spaces.items():
        ds_p = ds_p_plain if tagname == "R-on-parent" else ds_p_int
        pr(f"\n-- (b) residual assembly, {tagname}   "
           f"(parent ds = {'plain' if ds_p is ds_p_plain else 'intersected'})")
        z = Function(Z)
        # Non-trivial state, so that a zero residual norm cannot be mistaken
        # for a form that assembled.  Set from spatial expressions rather than
        # from dat indices, so the norms below are identical at any rank count.
        xs = SpatialCoordinate(sub)
        xp = SpatialCoordinate(parent)
        z.subfunctions[0].interpolate(as_vector([sin(xs[0]), cos(xs[1])]))
        z.subfunctions[1].interpolate(
            as_tensor([[sin(xs[0] * xs[1]), xs[0]], [xs[1], cos(xs[0])]]))
        z.subfunctions[2].interpolate(sin(xp[0]) * cos(xp[1]))
        for k, sf in enumerate(z.subfunctions[3:]):
            sf.assign(1.0 + 0.25 * k)
        tests = TestFunctions(Z)
        fields = split(z)
        u, mm, psi = fields[0], fields[1], fields[2]
        w, tau, v = tests[0], tests[1], tests[2]
        cs, nus = fields[3:], tests[3:]

        rho0 = Function(FunctionSpace(sub, "DG", 0)).assign(1.0)
        n_p = FacetNormal(parent)
        X_p = SpatialCoordinate(parent)
        X_s = SpatialCoordinate(sub)

        # mechanics on the submesh
        F_mech = (inner(sym(grad(u)), sym(grad(w))) * dx_s
                  + inner(mm, tau) * dx_s)
        # potential on the parent
        F_psi = inner(grad(psi), grad(v)) * dx_p
        # cross-mesh: parent-test against submesh trial and vice versa
        F_cross = (inner(rho0 * grad(psi), w) * dx_s
                   - rho0 * inner(u, grad(v)) * dx_s)
        # parent-boundary term on the potential row
        F_bnd = psi * v * ds_p(CURVE_OUTER)

        report("mechanics only (dx submesh)",
               lambda f=F_mech: assemble(f).dat.norm)
        report("potential only (dx parent, intersected)",
               lambda f=F_psi: assemble(f).dat.norm)
        report("cross-mesh coupling both directions",
               lambda f=F_cross: assemble(f).dat.norm)
        report("parent boundary term",
               lambda f=F_bnd: assemble(f).dat.norm)
        report("all four together",
               lambda: assemble(F_mech + F_psi + F_cross + F_bnd).dat.norm)

        # --- the two kinds of Real constraint row -------------------------
        # PARENT-BOUNDARY ROW: multiplier constrained by an integral over a
        # parent boundary, and feeding a flux back onto the potential row.
        row_bnd = (cs[0] * nus[0] * ds_p(CURVE_OUTER)
                   - psi * nus[0] * ds_p(CURVE_OUTER)
                   + cs[0] * v * ds_p(CURVE_OUTER))
        report("Real row = PARENT BOUNDARY integral",
               lambda f=row_bnd: assemble(f).dat.norm)

        # SUBMESH-VOLUME ROW: rotation closure, m_i (C-A) - Delta I_i[u]
        row_vol = (cs[1] * nus[1] * dx_s
                   - rho0 * inner(X_s, u) * nus[1] * dx_s
                   + cs[1] * rho0 * inner(X_s, w) * dx_s)
        report("Real row = SUBMESH VOLUME integral",
               lambda f=row_vol: assemble(f).dat.norm)

        # A parent-volume Real row too, for completeness.
        row_pvol = cs[2] * nus[2] * dx_p - psi * nus[2] * dx_p
        report("Real row = PARENT VOLUME integral",
               lambda f=row_pvol: assemble(f).dat.norm)

        report("FULL residual (mechanics + potential + cross + both Real rows)",
               lambda: assemble(F_mech + F_psi + F_cross + F_bnd
                                + row_bnd + row_vol).dat.norm)

        # ------------------------------------------------------------------
        # (c) field_ises
        # ------------------------------------------------------------------
        pr(f"\n-- (c) field_ises, {tagname}")
        ok, ises = report("Z.dof_dset.field_ises",
                          lambda Z=Z: Z.dof_dset.field_ises)
        if ok:
            pr(f"       n = {len(ises)} (len(Z) = {len(Z)})")
            sizes = [iset.getLocalSize() for iset in ises]
            pr(f"       local sizes per field: {sizes}")
            fams = [Vs.ufl_element().family() for Vs in Z]
            pr(f"       families: {fams}")
            realidx = [i for i, f in enumerate(fams) if f == "Real"]
            pr(f"       Real sub-fields at {realidx}; "
               f"contiguous-and-last: "
               f"{realidx == list(range(realidx[0], len(Z))) if realidx else False}")
            tot = sum(sizes)
            pr(f"       sum of local field sizes = {tot}; "
               f"Z local dofs = {Z.dof_dset.layout_vec.getLocalSize()}")

        # ------------------------------------------------------------------
        # (d) DtNTwoBlockSchurPC on a trivial solve
        # ------------------------------------------------------------------
        pr(f"\n-- (d) DtNTwoBlockSchurPC, {tagname}")
        run_pc_solve(Z, sub, parent, dx_s, dx_p, ds_p, tagname)

        # ------------------------------------------------------------------
        # (e) are the Real rows numerically RIGHT, not merely assemblable?
        # ------------------------------------------------------------------
        pr(f"\n-- (e) numerical check of the Real rows, {tagname}")
        check_real_rows(Z, sub, parent, dx_s, ds_p, tagname)

    # ------------------------------------------------------------------
    # (f) workaround attempts for R-on-sub carrying a parent-boundary row
    # ------------------------------------------------------------------
    if "R-on-sub" in spaces:
        pr("\n-- (f) can an R-on-sub multiplier be given a parent-boundary row?")
        Z = spaces["R-on-sub"]
        tests = TestFunctions(Z)
        z = Function(Z)
        psi = split(z)[2]
        v, nu0 = tests[2], tests[3]
        for name, meas in (
            ("ds(parent) intersected with dx(sub)",
             Measure("ds", domain=parent,
                     intersect_measures=(Measure("dx", domain=sub),))),
            ("ds(parent) intersected with ds(sub)",
             Measure("ds", domain=parent,
                     intersect_measures=(Measure("ds", domain=sub),))),
        ):
            # Constant integrand, so a zero answer means the intersected
            # measure found no facets -- not merely that the state was zero.
            report(f"{name}  [expect 2 pi 2Re = 27.6925]",
                   lambda m=meas: assemble(
                       Constant(1.0) * nu0 * m(CURVE_OUTER)).dat.norm)
        # And the one that DOES work: a shared facet (Re) seen from the submesh.
        ds_s = Measure("ds", domain=sub)
        report("R-on-sub row over ds(sub, Re) -- a SHARED facet",
               lambda: assemble(Constant(1.0) * nu0 * ds_s(CURVE_RE)).dat.norm)


def check_real_rows(Z, sub, parent, dx_s, ds_p, tagname):
    """Assemble one-form Real rows with a known value and compare."""
    nus = TestFunctions(Z)[3:]
    ana = spike_mesh.analytic()

    def real_value(form, k):
        # assemble() of a one-form gives a Cofunction; the Real sub-block holds
        # the single scalar.  float() only works on Functions, so read the dat
        # (safe here: a Real dat is globally replicated, no reduction needed).
        f = assemble(form)
        data = f.subfunctions[3 + k].dat.data_ro
        return float(data[0]) if data.size else 0.0

    ok, val = report("row_0 = 1 * nu_0 * ds(parent, outer): expect 2 pi 2Re "
                     f"= {ana['len_outer']:.6f}",
                     lambda: real_value(Constant(1.0) * nus[0] * ds_p(
                         CURVE_OUTER), 0))
    ok2, val2 = report("row_1 = 1 * nu_1 * dx(sub): expect area(mantle) "
                       f"= {ana['area_mantle']:.6f}",
                       lambda: real_value(Constant(1.0) * nus[1] * dx_s, 1))
    if ok and ok2:
        pr(f"       relative errors: boundary "
           f"{abs(val - ana['len_outer']) / ana['len_outer']:.3e}, "
           f"volume {abs(val2 - ana['area_mantle']) / ana['area_mantle']:.3e} "
           f"(both dominated by the straight-facet geometry, not the algebra)")


def run_pc_solve(Z, sub, parent, dx_s, dx_p, ds_p, tagname):
    """A trivial well-posed problem on Z, solved matfree + DtNTwoBlockSchurPC."""
    z = Function(Z)
    tests = TestFunctions(Z)
    fields = split(z)
    u, mm, psi = fields[0], fields[1], fields[2]
    w, tau, v = tests[0], tests[1], tests[2]
    cs, nus = fields[3:], tests[3:]

    rho0 = Function(FunctionSpace(sub, "DG", 0)).assign(1.0)
    X_s = SpatialCoordinate(sub)

    F = (inner(sym(grad(u)), sym(grad(w))) * dx_s + inner(u, w) * dx_s
         + inner(mm, tau) * dx_s - inner(sym(grad(u)), tau) * dx_s
         + inner(grad(psi), grad(v)) * dx_p + psi * v * ds_p(CURVE_OUTER)
         + inner(rho0 * grad(psi), w) * dx_s - rho0 * inner(u, grad(v)) * dx_s
         - inner(as_vector([1.0, 0.0]), w) * dx_s)
    # Real rows: half parent-boundary, half submesh-volume, so that whichever
    # mesh the Real space is on, both kinds are exercised.
    for i, (c, nu) in enumerate(zip(cs, nus)):
        if i % 2 == 0:
            F += c * nu * ds_p(CURVE_OUTER) - psi * nu * ds_p(CURVE_OUTER)
            F += c * v * ds_p(CURVE_OUTER)
        else:
            F += c * nu * dx_s - rho0 * inner(X_s, u) * nu * dx_s
            F += c * rho0 * inner(X_s, w) * dx_s

    params = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 200,
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
        "ksp_converged_reason": None,
    }

    def go():
        problem = NonlinearVariationalProblem(F, z)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        return f"||z|| = {z.dat.norm:.6e}"

    report(f"matfree solve with DtNTwoBlockSchurPC ({tagname})", go)


if __name__ == "__main__":
    main()
