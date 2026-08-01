"""V3' — the source-to-psi leg, against an analytic field rather than a rerun.

Freeze `u = grad(chi)` with `chi = (A r^n + B r^-n) cos(n phi)` harmonic, so
`div u = 0`.  The reference density is uniform over the mantle and absent
outside it, so the divergence-form Poisson source

    -Lambda int_mantle rho_0 u . grad(v) dx

integrates by parts to interface mass sheets alone:

    sigma = rho_0 (u . n),  n the mantle's OUTWARD normal,

i.e. `sigma_Re = +rho_0 u_r(Re)` and `sigma_Rc = -rho_0 u_r(Rc)`.  Each has the
closed form of `passess.polar.SheetPolar2D`, so the monolithic psi is compared
against an analytic field and not against another run of the same code.  That
is what makes this the one gate in the structural set that constrains the
*absolute* magnitude of Lambda on the psi side (review D3, D4).

How u is frozen: the displacement sub-function of the coupled mixed function is
set to the interpolant of `grad(chi)`, and the potential and multiplier rows -
which are linear in (psi, c) and affine in u - are solved exactly by a Schur
complement on the `nest` Jacobian of the production residual.  Nothing here
re-derives a form: the operator and the right-hand side both come from
`SelfGravitatingGIASolver.potential_residual`, through the production
cross-mesh assembly.

Serial only (a global dense transpose is not needed, but `getValuesCSR` on a
nest sub-block is).
"""
import argparse
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake; see the demo's note
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from gadopt import *  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, DEMOS)
sys.path.insert(0, os.path.expanduser("~/Workplace/passess"))

import generate_selfgrav_annulus as gen  # noqa: E402
from passess.polar import SheetPolar2D  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

B_MU = 1.2769
LAMBDA = 1.1116
RHO0 = 1.0
GAMMA = LAMBDA / (4.0 * np.pi)   # the form's gravitational constant


def build(dr, nazim, truncation, n_mode):
    """Meshes, coupled solver with no sheet, and the frozen displacement."""
    path = os.path.join(HERE, f"v3prime_{dr}_{nazim}.msh")
    gen.generate(path, dr_mantle=dr, n_azimuthal=nazim)
    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False

    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=False,
        self_gravity_number=LAMBDA)
    z = Function(Z)

    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=RHO0, shear_modulus=1.0, viscosity=1.0,
        g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)
    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=1.0,
        bcs={gen.CURVE_RC: {"un": 0.0}})

    # u = grad(chi), chi = (A r^n + B r^-n) cos(n phi).  A and B are chosen so
    # that both interface sheets are active and of comparable size; the overall
    # amplitude is irrelevant (everything is linear) and is set to O(1e-3) only
    # so the numbers look like a GIA displacement.
    A, B = 1.0e-3 / gen.RE ** n_mode, 1.0e-3 * gen.RC ** n_mode
    X = SpatialCoordinate(sub)
    r = sqrt(dot(X, X))
    phi = atan2(X[1], X[0])
    chi = (A * r ** n_mode + B * r ** (-n_mode)) * cos(n_mode * phi)
    u = Function(Z[layout.displacement]).interpolate(grad(chi))
    z.subfunctions[layout.displacement].assign(u)

    def u_r_amplitude(a):
        """d chi/dr at radius a, divided by cos(n phi)."""
        return n_mode * (A * a ** (n_mode - 1) - B * a ** (-n_mode - 1))

    sheets = {gen.RE: RHO0 * u_r_amplitude(gen.RE),
              gen.RC: -RHO0 * u_r_amplitude(gen.RC)}
    return parent, sub, solver, z, layout, sheets


def solve_gravity_block(solver, z, layout):
    """Solve the potential and multiplier rows exactly, with u held frozen.

    The rows are linear in `(psi, c)`, so with `psi = 0` and `c = 0` in `z` the
    assembled residual of those rows *is* the negated right-hand side, and the
    corresponding Jacobian blocks are the operator.  Schur-complement onto the
    handful of multipliers rather than converting anything dense.
    """
    if not solver.strong_bcs:
        pass
    else:  # pragma: no cover - configuration guard
        raise RuntimeError(
            f"{len(solver.strong_bcs)} strong bcs present; the reduced solve "
            "below ignores them.")

    F = solver.F
    J = assemble(derivative(F, z), mat_type="nest").petscmat
    R = assemble(F)

    ip = layout.potential
    ics = list(layout.multipliers)
    nc = len(ics)

    def block(i, j):
        M = J.getNestSubMatrix(i, j)
        return None if M is None else M

    Ap = block(ip, ip)
    indptr, indices, data = Ap.getValuesCSR()
    n = Ap.getSize()[0]
    Amat = sp.csr_matrix((data, indices, indptr), shape=(n, n))

    Bmat = np.zeros((n, nc))
    Cmat = np.zeros((nc, n))
    Dmat = np.zeros((nc, nc))
    for a, ia in enumerate(ics):
        Mb = block(ip, ia)
        if Mb is not None:
            Bmat[:, a] = Mb.convert("dense").getDenseArray()[:, 0]
        Mc = block(ia, ip)
        if Mc is not None:
            Cmat[a, :] = Mc.convert("dense").getDenseArray()[0, :]
        for b, ib in enumerate(ics):
            Md = block(ia, ib)
            if Md is not None:
                Dmat[a, b] = Md.convert("dense").getDenseArray()[0, 0]

    b1 = -R.subfunctions[ip].dat.data_ro.copy()
    b2 = -np.array([float(R.subfunctions[i].dat.data_ro[0]) for i in ics])

    lu = spla.splu(Amat.tocsc())
    AinvB = np.column_stack([lu.solve(Bmat[:, a]) for a in range(nc)])
    Ainvb1 = lu.solve(b1)
    S = Dmat - Cmat @ AinvB
    y = np.linalg.solve(S, b2 - Cmat @ Ainvb1)
    x = Ainvb1 - AinvB @ y

    z.subfunctions[ip].dat.data[:] = x
    for a, ia in enumerate(ics):
        z.subfunctions[ia].dat.data[:] = y[a]

    resid = assemble(F)
    rg = np.hstack([resid.subfunctions[ip].dat.data_ro,
                    [float(resid.subfunctions[i].dat.data_ro[0])
                     for i in ics]])
    return np.abs(rg).max(), np.abs(np.hstack([b1, b2])).max()


def sheet_ufl(r, phi, m, sigma_m, a):
    """`SheetPolar2D`'s closed form as UFL: (2 pi gamma sigma a/m)(r_< /r_>)^m."""
    r_less = conditional(lt(r, a), r, Constant(a))
    r_more = conditional(lt(r, a), Constant(a), r)
    return (2.0 * np.pi * GAMMA * sigma_m * a / m) * (r_less / r_more) ** m \
        * cos(m * phi)


def check_ufl_against_passess(m, sheets):
    """The UFL closed form must be `passess`'s, to roundoff, before it is used."""
    worst = 0.0
    for a, sigma in sheets.items():
        ref = SheetPolar2D(m=m, sigma_m=sigma, a=a, gamma=GAMMA)
        for rr in (0.7, 1.1, 1.5, 2.0, 2.5, 4.0):
            r_less, r_more = min(rr, a), max(rr, a)
            mine = (2.0 * np.pi * GAMMA * sigma * a / m) * (r_less / r_more) ** m
            worst = max(worst, abs(mine - float(ref.psi_m(rr))) / abs(mine))
    return worst


def analytic(parent, m, sheets):
    X = SpatialCoordinate(parent)
    r = sqrt(dot(X, X))
    phi = atan2(X[1], X[0])
    expr = None
    for a, sigma in sheets.items():
        term = sheet_ufl(r, phi, m, sigma, a)
        expr = term if expr is None else expr + term
    return expr


def errors(psi, exact, dx_):
    num = sqrt(assemble((psi - exact) ** 2 * dx_))
    den = sqrt(assemble(exact ** 2 * dx_))
    return num / den, den


def run(dr, nazim, truncation, n_mode, variants=False):
    parent, sub, solver, z, layout, sheets = build(dr, nazim, truncation, n_mode)
    res, rhs = solve_gravity_block(solver, z, layout)

    psi = z.subfunctions[layout.potential]
    exact = analytic(parent, n_mode, sheets)

    dx_parent = Measure("dx", domain=parent)
    dx_mantle = dx_parent(gen.CELL_MANTLE)
    e_mantle, n_mantle = errors(psi, exact, dx_mantle)
    e_parent, n_parent = errors(psi, exact, dx_parent)

    # Divergence of the CG2 interpolant: the leading term the gate is limited by.
    dx_m = Measure("dx", domain=sub)
    u = z.subfunctions[layout.displacement]
    div_l1 = assemble(abs(div(u)) * dx_m)
    u_l1 = assemble(sqrt(dot(u, u)) * dx_m)

    # The quantity the project actually reads: the cos(n phi) amplitude of psi
    # on the surface circle Re, which is the geoid up to g_0.  Re is an
    # interior facet of the parent, so this is a dS integral through the form's
    # own measure and `avg`.
    Xp = SpatialCoordinate(parent)
    phip = atan2(Xp[1], Xp[0])
    traces = {}
    for tag, radius in ((gen.CURVE_RE, gen.RE), (gen.CURVE_RC, gen.RC)):
        dSs = solver.form.dS(tag)
        length = assemble(avg(Constant(1.0)) * dSs)
        num = 2 * assemble(avg(psi * cos(n_mode * phip)) * dSs) / length
        ana = 2 * assemble(avg(exact * cos(n_mode * phip)) * dSs) / length
        traces[tag] = (num, ana, abs(num - ana) / abs(ana))

    # Where the remaining error lives, by radial band.  E5 found the coarse
    # buffer concentrating 100-200x the mantle error on the gravity-alone
    # geometry; this says whether that is still true here.
    bands = {}
    for name, tag in (("inner", gen.CELL_INNER), ("mantle", gen.CELL_MANTLE),
                      ("buffer", gen.CELL_BUFFER)):
        d = dx_parent(tag)
        bands[name] = (sqrt(assemble((psi - exact) ** 2 * d)),
                       sqrt(assemble(exact ** 2 * d)))

    out = {
        "dr": dr, "nazim": nazim, "parent_cells": parent.num_cells(),
        "psi_dofs": psi.function_space().dim(),
        "reduced_residual": res, "reduced_rhs": rhs,
        "e_mantle": e_mantle, "e_parent": e_parent,
        "norm_mantle": n_mantle,
        "div_over_u": div_l1 / u_l1,
        "sheets": sheets, "traces": traces, "bands": bands,
    }

    if variants:
        # The rejection region: what the same comparison gives for plausible
        # wrong interface bookkeeping.  If these are not much worse than the
        # correct pair, the gate is not measuring what it claims to.
        alts = {
            "Rc sheet sign flipped": {gen.RE: sheets[gen.RE],
                                      gen.RC: -sheets[gen.RC]},
            "Rc sheet omitted": {gen.RE: sheets[gen.RE]},
            "Re sheet omitted": {gen.RC: sheets[gen.RC]},
            "both signs flipped": {gen.RE: -sheets[gen.RE],
                                   gen.RC: -sheets[gen.RC]},
            "Lambda off by 10%": None,
        }
        out["variants"] = {}
        for name, alt in alts.items():
            if alt is None:
                e = errors(psi, 1.1 * exact, dx_mantle)[0]
            else:
                e = errors(psi, analytic(parent, n_mode, alt), dx_mantle)[0]
            out["variants"][name] = e
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truncation", type=int, default=5)
    p.add_argument("--mode", type=int, default=2)
    p.add_argument("--meshes", type=str, default="0.2:32,0.1:64,0.05:128")
    args = p.parse_args()

    print("V3' - monolithic psi for a frozen harmonic u, against SheetPolar2D")
    print(f"  Lambda {LAMBDA}   gamma = Lambda/4pi {GAMMA:.8f}   "
          f"rho_0 {RHO0}   mode n = {args.mode}   M = {args.truncation}")

    rows = []
    for k, spec in enumerate(args.meshes.split(",")):
        dr, nazim = spec.split(":")
        r = run(float(dr), int(nazim), args.truncation, args.mode,
                variants=(k == 1))
        rows.append(r)
        if k == 0:
            worst = check_ufl_against_passess(args.mode, r["sheets"])
            print(f"  UFL closed form vs passess.SheetPolar2D: {worst:.3e} "
                  "relative (must be roundoff)")
            print(f"  sheet strengths  sigma(Re) = {r['sheets'][gen.RE]:+.6e}"
                  f"   sigma(Rc) = {r['sheets'][gen.RC]:+.6e}")
        print(f"\n  dr {r['dr']} nazim {r['nazim']}: parent cells "
              f"{r['parent_cells']}, psi dofs {r['psi_dofs']}")
        print(f"    reduced residual after solve  {r['reduced_residual']:.3e} "
              f"(rhs scale {r['reduced_rhs']:.3e})")
        print(f"    int|div u| / int|u|           {r['div_over_u']:.3e}")
        print(f"    rel L2 error, mantle          {r['e_mantle']:.6e}")
        print(f"    rel L2 error, whole parent    {r['e_parent']:.6e}")
        for tag, (num, ana, rel) in r["traces"].items():
            print(f"    cos{args.mode} amplitude of psi on circle {tag}: "
                  f"{num:.8e} vs {ana:.8e}, rel {rel:.3e}")
        print("    error by region      abs           |psi|         rel")
        for name, (e, nrm) in r["bands"].items():
            print(f"      {name:<8s}  {e:.4e}    {nrm:.4e}    {e / nrm:.4e}")
        if "variants" in r:
            print("    rejection region (same psi, wrong reference):")
            for name, e in r["variants"].items():
                print(f"      {name:<24s} {e:.4e}")

    print("\n  refinement of the mantle error")
    for a, b in zip(rows, rows[1:]):
        order = np.log(a["e_mantle"] / b["e_mantle"]) / np.log(2.0)
        print(f"    {a['dr']} -> {b['dr']}: factor "
              f"{a['e_mantle'] / b['e_mantle']:.3f}, observed order {order:.2f}")
    print("  A common-factor error in Lambda would NOT fall with refinement.")


if __name__ == "__main__":
    main()
