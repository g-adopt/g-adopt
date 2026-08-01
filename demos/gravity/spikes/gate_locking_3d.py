"""Round two of the locking probe: the same measurement, in 3-D on tets.

`gate_locking.py` measured the h-convergence order of a surface functional at
fixed high nu in 2-D and found it degrading from 3.95 at nu = 0.28 to 2.48 at
nu = 0.4999 — half an order, reproduced across three mesh families, with an
error constant growing like `h^-1.4` rather than settling.  A concurrent
measurement of the approximation-space amplification `R = e_Z/e_V` on tets
found `R ~ h^-0.505` for CG2/P1-disc with both controls flat.  Those are
different quantities and both can be true.  **The quantity the project actually
cares about — does the answer stop converging — had only ever been measured in
2-D.**  This is that cell.

Everything is held identical to the 2-D run on purpose, because the
comparability *is* the result:

* mechanics only, no gravity coupling; the volumetric penalty lives in the
  (u,u) block and nowhere else
* `g = 0`, so the Al-Attar prestress/buoyancy pair is off and road map §2.5's
  growing mode plays no part
* elastic: one solve at `dt/tau = 1e-8`, so the internal variable sits at 1e-8
  of the strain and `R_eff = R (1 + dt/tau)` equals `R` to eight digits
* `CompressibleInternalVariableApproximation` via `MaxwellApproximation`, never
  `QuasiCompressible...` (trap §12.0); `bulk_modulus = 1` so `K_eff = ratio`
* clamped at Rc (strong `u`, which `stokes_integrators.py:326` puts in
  `strong_bcs` only, so it adds no weak term — in particular none carrying
  `bulk_modulus * bulk_shear_ratio`, unlike the `un` branch)
* a load with genuine high-degree content, because the locking error is
  proportional to `|u|_{k+1}` and a smooth load measures nothing.  Degree 2 is
  what voided round one's first pass.

Geometry: Rc/Re = 3480/6371 km, non-dimensionalised by D = Re - Rc = 2891 km,
so Rc = 1.203736 and Re = 2.203736 — the same numbers as the 2-D annulus.

## The one place this departs from the brief, and why

**Straight-sided tets would have destroyed the measurement.**  A linear
spherical shell has a domain error that enters a surface functional at O(h^2),
which caps *every* convergence order at 2 regardless of nu — the two columns
would agree at 2.0 and the run would report a confident null.  Measured on this
ladder:

    lc     cells   straight vol err   curved vol err
    0.50    1882       1.07e-02          2.77e-05
    0.36    4768       5.00e-03          8.31e-06
    0.25   12354       2.80e-03          1.44e-06

Straight-sided falls at order 2 and sits four orders above the discretisation
error being measured; curved falls at order ~3.7 and sits below it.  So the
same radial P2 remap the 2-D probe uses is applied here — it is
dimension-agnostic, and it is not "putting geometry under test", it is keeping
geometry *out* of the test.  (gmsh's own `Mesh.ElementOrder 2` is not usable:
Firedrake's reader raises `ValueError: cannot reshape array of size 23994 into
shape (1200,3)` on a quadratic tet mesh.)

## What decides it

The observed h-convergence order of `J` at fixed nu, and the error constant
ratio `C(nu)/C(0.28)` at fixed h — exactly the two numbers reported in 2-D.

* If the order holds up as it did in 2-D, the 2-D verdict extends and the
  amplification measurement is describing a constant rather than a rate.
* If the order collapses towards 1 or 2 in 3-D while 2-D held near 2.5, the
  transfer argument is confirmed and the penalty plan needs a remedy.

Not measured here, deliberately: any SRI ladder (a separate agent is measuring
whether reduced integration lowers the *true* 3-D error against a manufactured
solution), and anything about the coupled system.

Usage:

    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking_3d.py
    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking_3d.py --lcs 0.5,0.25,0.125
"""
import argparse
import os
import time

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402
from gadopt.spherical_harmonics import real_spherical_harmonic  # noqa: E402
import gmsh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# 3480 / 6371 km, non-dimensionalised by D = Re - Rc = 2891 km.  Same numbers
# as the 2-D annulus, so the two runs are directly comparable.
RC, RE = 1.203736, 2.203736
SURF_RE, SURF_RC = 2, 3
CELL_MANTLE = 101

SIGMA_0 = 1.0e-3
LOAD_L, LOAD_M = 6, 3     # degree of the surface normal stress
MU0 = 1.0
TAU = 1.0
DT = 1.0e-8

NU_LADDER = [0.28, 0.49, 0.499]

# A single-degree load on a spherically symmetric operator produces a
# single-degree response, so "high-degree load" and "degree-2 functional" are
# not simultaneously satisfiable with one harmonic: the degree-2 amplitude of
# the response to Y_6,3 is identically zero by orthogonality.  The broadband
# load carries genuine high-degree content to load the element while leaving a
# degree-2 component to read, which is what the benchmark actually reads.
LOAD_MODES = {
    "single": [(6, 3)],
    "multi": [(2, 0), (6, 3), (10, 5)],
}


def bulk_shear_ratio(nu):
    return 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))


def effective_ratio(ratio, dt):
    """R_eff = R (1 + dt/tau); see gate_locking.py, measured there to 2.1e-10."""
    return ratio * (1.0 + dt / TAU)


DIRECT = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

ITERATIVE = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-11,
    "ksp_max_it": 20000,
    "pc_type": "python",
    "pc_python_type": "gadopt.SPDAssembledPC",
    "assembled_pc_type": "gamg",
    "assembled_mg_levels_pc_type": "sor",
    "assembled_pc_gamg_threshold": 0.01,
    "assembled_pc_gamg_square_graph": 100,
    "assembled_pc_gamg_coarse_eq_limit": 1000,
    "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
}


# --------------------------------------------------------------------------
# mesh
# --------------------------------------------------------------------------

def write_shell(path, lc):
    """Unstructured tet shell Rc -> Re via OpenCASCADE, tags 2 (Re), 3 (Rc)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("shell")
    occ = gmsh.model.occ
    occ.addSphere(0, 0, 0, RE, 1)
    occ.addSphere(0, 0, 0, RC, 2)
    occ.cut([(3, 1)], [(3, 2)])
    occ.synchronize()

    inner, outer = [], []
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        radius = (bb[3] - bb[0]) / 2.0
        (outer if radius > 0.5 * (RC + RE) else inner).append(tag)
    gmsh.model.addPhysicalGroup(2, outer, SURF_RE, name="Re")
    gmsh.model.addPhysicalGroup(2, inner, SURF_RC, name="Rc")
    gmsh.model.addPhysicalGroup(
        3, [t for _, t in gmsh.model.getEntities(3)], CELL_MANTLE, name="mantle")

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path)
    gmsh.finalize()
    return path


def curve_mesh(linear_mesh):
    """Radial P2 remap: edge midpoints pushed onto the linear-interpolant radius.

    Identical in form to the 2-D probe's `curve_mesh`; nothing about it is
    two-dimensional.  On the two boundaries the vertices already sit on the
    exact sphere, so their edge midpoints land on it too.
    """
    X = SpatialCoordinate(linear_mesh)
    r = sqrt(dot(X, X))
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        (r_p1 / r) * X)
    return Mesh(X_p2)


def geometry_errors(mesh):
    volume = assemble(Constant(1.0) * dx(domain=mesh))
    a_re = assemble(Constant(1.0) * ds(SURF_RE, domain=mesh))
    a_rc = assemble(Constant(1.0) * ds(SURF_RC, domain=mesh))
    v_exact = 4.0 / 3.0 * np.pi * (RE ** 3 - RC ** 3)
    return (abs(volume - v_exact) / v_exact,
            abs(a_re - 4 * np.pi * RE ** 2) / (4 * np.pi * RE ** 2),
            abs(a_rc - 4 * np.pi * RC ** 2) / (4 * np.pi * RC ** 2))


def mesh_ladder(lcs):
    out = []
    for lc in lcs:
        path = write_shell(os.path.join(HERE, f"locking3d_{lc:g}.msh"), lc)
        m = curve_mesh(Mesh(path))
        m.cartesian = False
        out.append(m)
    return out


# --------------------------------------------------------------------------
# one solve
# --------------------------------------------------------------------------

def load_expression(X, modes):
    expr = None
    for (l, m) in modes:
        term = real_spherical_harmonic(l, m, X)
        expr = term if expr is None else expr + term
    return expr


def build(mesh, ratio, dt=DT, modes=(( 6, 3),)):
    V = VectorFunctionSpace(mesh, "CG", 2)
    S = TensorFunctionSpace(mesh, "DG", 1)
    u = Function(V, name="displacement")
    m = Function(S, name="internal variable")

    approximation = MaxwellApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=MU0, viscosity=MU0 * TAU,
        g=0.0, B_mu=1.0, bulk_shear_ratio=ratio)

    X = SpatialCoordinate(mesh)
    bcs = {
        SURF_RC: {"u": Constant((0.0, 0.0, 0.0))},
        SURF_RE: {"normal_stress": -SIGMA_0 * load_expression(X, modes)},
    }
    return u, m, approximation, bcs


def diagnostics(mesh, u, approximation, read=(6, 3)):
    """J, the norms and the divergence measure.

    `J` is the Y_lm projection coefficient of the radial displacement on the
    loaded sphere, `int u_r Y ds / int Y^2 ds`.  Normalising by the *measured*
    `int Y^2 ds` rather than the analytic `Re^2` cancels the leading surface
    quadrature error, and is the 3-D twin of the 2-D probe's boundary Fourier
    amplitude.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    Y = real_spherical_harmonic(read[0], read[1], X)
    u_r = dot(u, X / r)

    J = (assemble(u_r * Y * ds(SURF_RE))
         / assemble(Y * Y * ds(SURF_RE)))

    u_l2 = sqrt(assemble(dot(u, u) * dx))
    dev = approximation.deviatoric_strain(u)
    energy = sqrt(assemble(2.0 * MU0 * inner(dev, dev) * dx))
    div_l2 = sqrt(assemble(div(u) ** 2 * dx))
    grad_l2 = sqrt(assemble(inner(grad(u), grad(u)) * dx))

    return {"J": J, "u_l2": u_l2, "energy": energy,
            "div_over_grad_l2": div_l2 / grad_l2}


def residual_reduction(solver, u, m, m_before):
    """||F(u)||/||F(0)|| with strong-bc rows removed.

    `InternalVariableSolver.solve` overwrites the internal variable *after* the
    solve, so `m` is restored to what the solve actually saw; without that the
    measured reduction floors at dt/tau and looks like a failed solve.  This
    cost an hour in round one.
    """
    m_after = m.copy(deepcopy=True)
    m.assign(m_before)

    def norm_free():
        r = assemble(solver.F)
        for bc in solver.strong_bcs:
            r.dat.data[bc.nodes] = 0.0
        return np.linalg.norm(r.dat.data_ro)

    here = u.copy(deepcopy=True)
    final = norm_free()
    u.assign(0.0)
    initial = norm_free()
    u.assign(here)
    m.assign(m_after)
    return final / initial if initial > 0 else np.nan


def solve_cell(mesh, ratio, params, label, modes=((6, 3),), read=(6, 3)):
    u, m, approximation, bcs = build(mesh, ratio, modes=modes)
    t0 = time.perf_counter()
    solver = InternalVariableSolver(
        u, approximation, dt=DT, internal_variables=[m], bcs=bcs,
        solver_parameters=params)
    m_before = m.copy(deepcopy=True)
    solver.solve()
    wall = time.perf_counter() - t0

    out = diagnostics(mesh, u, approximation, read=read)
    out["residual_reduction"] = residual_reduction(solver, u, m, m_before)
    out["wall"] = wall
    out["dofs"] = u.function_space().dim()
    out["approx_class"] = type(approximation).__name__
    out["ratio_eff"] = effective_ratio(ratio, DT)
    out["solver"] = label
    try:
        out["its"] = solver.solver.snes.ksp.getIterationNumber()
        out["reason"] = solver.solver.snes.ksp.getConvergedReason()
    except Exception:
        out["its"], out["reason"] = None, None
    out["u"] = u
    return out


def orders(values, factors):
    """Observed order from successive differences on a general h ladder."""
    v = np.asarray(values, float)
    d = np.abs(np.diff(v))
    out = []
    for a, b, f in zip(d[:-1], d[1:], factors[1:]):
        out.append(np.log(a / b) / np.log(f) if b > 0 else np.inf)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lcs", type=str, default="0.5,0.35,0.25,0.175")
    p.add_argument("--nu", type=str, default="")
    p.add_argument("--load", choices=tuple(LOAD_MODES), default="single")
    p.add_argument("--read", type=str, default="",
                   help="l:m of the harmonic read from u_r, default = load mode")
    p.add_argument("--lu-levels", type=int, default=2,
                   help="levels solved by MUMPS LU as well as by CG, for the "
                        "cross-check; the rest are CG only")
    args = p.parse_args()

    lcs = [float(x) for x in args.lcs.split(",")]
    modes = LOAD_MODES[args.load]
    read = (tuple(int(v) for v in args.read.split(":")) if args.read
            else modes[0])
    nus = [float(x) for x in args.nu.split(",")] if args.nu else NU_LADDER
    ratios = [bulk_shear_ratio(nu) for nu in nus]

    print("Locking probe, ROUND TWO: 3-D unstructured tets")
    print("  the missing cell - functional convergence order at fixed high nu,")
    print("  in 3-D, with every confounder held identical to the 2-D run")
    print()
    print("  configuration, reported unconditionally")
    print("    element            P2 (CG2) vector on tets, no pressure partner")
    print(f"    domain             spherical shell {RC:.6f} -> {RE:.6f}")
    print("                       (3480/6371 km over D = 2891 km)")
    print("    geometry           radial P2 remap (curve_mesh).  Straight-sided")
    print("                       tets give O(h^2) domain error that would cap")
    print("                       every order at 2 - see the docstring.")
    print(f"    load               -{SIGMA_0} x sum Y_lm, lm = {modes}")
    print(f"    functional         degree {read} amplitude of u_r at Re")
    print("    CMB condition      clamped (strong u; no weak term, so none")
    print("                       carrying bulk_modulus * bulk_shear_ratio)")
    print("    g                  0 (Al-Attar prestress/buoyancy pair off)")
    print(f"    dt/tau             {DT / TAU:.1e}  -> R_eff/R = {1 + DT / TAU:.10f}")
    print()
    print("  expected before the run:")
    print("    at nu = 0.28 the order is the CG2 functional rate, 3 or better")
    print("    2-D gave 3.95 at nu = 0.28 falling to 2.48 at nu = 0.4999, with")
    print("      the error constant ratio growing like h^-1.4 (6.5 -> 103)")
    print("    OPEN, and the whole point: whether the 3-D order at nu = 0.499")
    print("      holds near the 2-D value or collapses towards 1-2")
    print()

    meshes = mesh_ladder(lcs)
    wavelength = 2 * np.pi * RE / max(l for l, _ in modes)
    hs = []
    print("  mesh ladder")
    for k, (lc, m) in enumerate(zip(lcs, meshes)):
        vol_e, re_e, rc_e = geometry_errors(m)
        h = (4.0 / 3.0 * np.pi * (RE ** 3 - RC ** 3) / m.num_cells()
             * 6.0 * np.sqrt(2.0)) ** (1.0 / 3.0)
        hs.append(h)
        print(f"    L{k}: lc {lc:6.3f}  {m.num_cells():8d} cells  "
              f"CG2 dofs {VectorFunctionSpace(m, 'CG', 2).dim():9d}  "
              f"h {h:.4f}  {wavelength / h:5.1f} cells/wavelength  "
              f"geom vol {vol_e:.2e} Re {re_e:.2e} Rc {rc_e:.2e}")
    factors = [1.0] + [hs[k - 1] / hs[k] for k in range(1, len(hs))]
    print("    refinement factors in h: " +
          "  ".join(f"{f:.3f}" for f in factors[1:]))
    print()

    table = {}
    for k, mesh in enumerate(meshes):
        for nu, ratio in zip(nus, ratios):
            use_lu = k < args.lu_levels
            r = solve_cell(mesh, ratio, ITERATIVE, "cg", modes, read)
            if use_lu:
                rlu = solve_cell(mesh, ratio, DIRECT, "lu", modes, read)
                mismatch = (sqrt(assemble(dot(r["u"] - rlu["u"],
                                              r["u"] - rlu["u"]) * dx))
                            / max(rlu["u_l2"], 1e-300))
                r["lu_J"] = rlu["J"]
                r["lu_mismatch"] = mismatch
                r["lu_wall"] = rlu["wall"]
            else:
                r["lu_J"] = None
                r["lu_mismatch"] = None
                r["lu_wall"] = None
            r.pop("u")
            table[(k, nu)] = r
            extra = ""
            if r["lu_J"] is not None:
                extra = (f"  |LU-CG|/|u| {r['lu_mismatch']:.1e}  "
                         f"dJ_lu {abs(r['J'] - r['lu_J']) / abs(r['J']):.1e}")
            print(f"    L{k} nu={nu:<7.4f} R_eff={r['ratio_eff']:9.4f}  "
                  f"J={r['J']:+.9e}  |u|={r['u_l2']:.6e}  "
                  f"div/grad={r['div_over_grad_l2']:.3e}  "
                  f"its={r['its']:>6}  reason={r['reason']}  "
                  f"t={r['wall']:7.1f}s  res={r['residual_reduction']:.1e}  "
                  f"{r['approx_class']}{extra}")
        print()

    classes = {r["approx_class"] for r in table.values()}
    print(f"  approximation classes instantiated: {sorted(classes)}")
    bad = [key for key, r in table.items()
           if r["reason"] is not None and r["reason"] < 0]
    print(f"  cells whose Krylov solve did NOT converge: {bad if bad else 'none'}")
    print()

    print("  THE LOCKING TEST - observed h-convergence order of J at fixed nu")
    for nu in nus:
        vals = [table[(k, nu)]["J"] for k in range(len(meshes))]
        ords = orders(vals, factors)
        print(f"    nu={nu:<7.4f}  J = " +
              "  ".join(f"{v:+.6e}" for v in vals) +
              "   orders " + "  ".join(f"{o:5.2f}" for o in ords))

    print()
    print("  same, for the L2 norm of u")
    for nu in nus:
        vals = [table[(k, nu)]["u_l2"] for k in range(len(meshes))]
        print(f"    nu={nu:<7.4f}  orders " +
              "  ".join(f"{o:5.2f}" for o in orders(vals, factors)))

    print()
    print("  |J(h) - J(h_finest)|/|J(h_finest)|, and C(nu)/C(nu_min) at fixed h")
    last = len(meshes) - 1
    errs = {}
    for nu in nus:
        ref = table[(last, nu)]["J"]
        errs[nu] = [abs(table[(k, nu)]["J"] - ref) / abs(ref) for k in range(last)]
        print(f"    nu={nu:<7.4f}  " + "  ".join(f"{e:.3e}" for e in errs[nu]))
    base = errs[nus[0]]
    print("    C(nu)/C(nu_min):")
    for nu in nus:
        print(f"    nu={nu:<7.4f}  " +
              "  ".join(f"{a / b:9.1f}" for a, b in zip(errs[nu], base)) +
              f"   [(1-2nu)^-1 relative to nu_min: "
              f"{(1 / max(1 - 2 * nu, 1e-12)) / (1 / (1 - 2 * nus[0])):.1f}]")

    print()
    print("  penalty enforcement: ||div u||_L2/||grad u||_L2 vs R_eff")
    print("  (a sanity check only - a locked element drives div u to zero too)")
    for k in range(len(meshes)):
        vals = [table[(k, nu)]["div_over_grad_l2"] for nu in nus]
        reffs = [table[(k, nu)]["ratio_eff"] for nu in nus]
        slopes = [np.log(b / a) / np.log(rb / ra)
                  for a, b, ra, rb in zip(vals[:-1], vals[1:],
                                          reffs[:-1], reffs[1:])]
        print(f"    L{k}: " + "  ".join(f"{v:.3e}" for v in vals) +
              "   log-log slopes " + "  ".join(f"{s:+5.2f}" for s in slopes))

    print()
    print("  Krylov cost (CG + GAMG, rtol 1e-11)")
    for k in range(len(meshes)):
        its = [table[(k, nu)]["its"] for nu in nus]
        print(f"    L{k}: its " + "  ".join(f"{i:>7}" for i in its))

    print()
    print("  VERDICT: compare the order column at nu = 0.499 against the 2-D")
    print("  value.  Holding near it extends the 2-D verdict; collapsing")
    print("  towards 1-2 confirms the 2-D -> 3-D transfer argument.")


if __name__ == "__main__":
    main()
