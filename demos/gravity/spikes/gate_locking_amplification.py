"""Does the penalty limit lose approximation power?  The amplification ratio.

`gate_infsup_3d.py` measures `beta_h`.  That number enters the penalty error
only through a **one-sided** bound,

    inf_{v in Z_h} |u - v|_1  <=  (1 + 1/beta_h) inf_{v in V_h} |u - v|_1,
    Z_h = ker(div_h),

so `beta_h` bounded below **proves** there is no locking, but `beta_h -> 0`
proves nothing at all: it withdraws the guarantee without supplying a failure.
The standing counterexample is Q1/P0 in 2-D, whose `beta_h` is identically zero
with a genuine checkerboard mode, and which is nevertheless the workhorse
nearly-incompressible element in commercial solid mechanics — because the
penalty pressure `p = -K_eff div u` lies in `range(div)` by construction and
never excites the spurious mode.

This script measures the left-hand side directly instead.  Take a smooth,
exactly divergence-free field `u`, and compare

    e_V = min over v in V_h        |u - v|_1
    e_Z = min over v in ker(div_h) |u - v|_1

`R = e_Z / e_V` **is** the amplification, on the actual meshes, rather than the
`(1 + 1/beta_h)` bound on it.  There is no eigensolver, no deflation of a
pressure kernel, no interpretation of a small number — none of the failure modes
that can corrupt `beta_h` reach this quantity.

`R` flat and `O(1)` under refinement: no locking, whatever `beta_h` does.
`R` growing with refinement: locking, and the growth rate prices it.

## How the constrained minimiser is taken, and why that is the honest way

`e_Z` comes from the regularised saddle system

    [ A      B^T   ] [v]     [g]
    [ B   -Mq/gamma] [l]  =  [0],       g_w = int grad(u) : grad(w)

whose Schur elimination is `(A + gamma B^T Mq^{-1} B) v = g` — that is
*literally* the volumetric penalty with `gamma` playing `bulk_shear_ratio`, and
`gamma -> infinity` is the constrained minimiser.  The saddle form is used
rather than the penalty form because it stays sparse and stays conditioned, and
because it is nonsingular for **any** `gamma > 0` even when `B` is rank
deficient, which it is here.  `gamma` is swept over eight orders and `e_Z` is
reported at each, so a reader can see the limit being reached rather than take
it on trust.

## Boundary conditions: the actual GIA problem, not the textbook one

No essential conditions anywhere.  The real momentum problem has
`normal_stress` at the surface and `normal_stress`/`un` at the CMB, so `V_h`
keeps every boundary velocity dof.  Clamping a boundary — the textbook Stokes
setting `gate_infsup_3d.py` used — deletes `O(N^(2/3))` of them and answers a
pessimistically biased version of the question.  Both are run here so the size
of that bias is a measured number rather than an argument.

`ker(A)` for the H1 seminorm is then exactly the three translations (rotations
have nonzero antisymmetric gradient and are *not* in it), removed by pinning one
dof per component, which is exact for right-hand sides orthogonal to the kernel
— and `g` is, since `grad(const) = 0`.

## Expected, stated before the run

* `div(u_ex)` integrates to machine zero.  If not, nothing below means anything.
* `e_V` converges at the CG2 seminorm rate, `O(h^2)`.
* **Taylor-Hood CG2/CG1: `R` close to 1 and flat.**  It is stable, so it must
  be, and if it is not the machinery is wrong.
* **CG2/P0 — the pair implied by one-point volumetric quadrature, i.e.
  selective reduced integration: `R` close to 1 and flat.**  This is the branch
  being priced: it is inf-sup stable, and in G-ADOPT it is a low-degree measure
  on the volumetric term plus a `bulk_quad_degree` argument, with no new
  function space and nothing else in the coupled solver touched.
* **CG2/P1-disc — the pair G-ADOPT's full-quadrature penalty actually implies:
  unknown, and this is the measurement.**  `beta_h` there is at or near zero,
  so the bound is vacuous and only `R` can say whether that matters.
"""

import os
import sys
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import gadopt  # noqa: F401  (import before firedrake; Irksome's order guard)
from firedrake import (
    Function,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    assemble,
    as_vector,
    cos,
    curl,
    div,
    dx,
    grad,
    inner,
    pi,
    sin,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gate_infsup_3d import (  # noqa: E402
    RC_OVER_RE,
    census,
    cell_size,
    retained_dofs,
    shell_mesh,
    to_scipy,
)

QUAD = 8  # quadrature degree for every integral involving u_ex


def target_field(mesh):
    """A smooth, exactly divergence-free field: u = curl of a vector potential.

    Trigonometric rather than polynomial, so CG2 cannot represent it exactly and
    `e_V` is a real approximation error rather than round-off.  Nothing in it is
    singular anywhere in the shell.
    """
    x, y, z = SpatialCoordinate(mesh)
    A = as_vector([sin(pi * y) * cos(pi * z),
                   sin(pi * z) * cos(pi * x),
                   sin(pi * x) * cos(pi * y)])
    return curl(A)


def amplification(mesh, q_family, q_degree, v_degree=2, bc_mode="free",
                  gammas=(1e4, 1e6, 1e8), label="", variant=None, quad=None):
    """e_V, e_Z(gamma), and the ratio."""
    t0 = time.time()
    kw = {} if variant is None else {"variant": variant}
    V = VectorFunctionSpace(mesh, "CG", v_degree, **kw)
    Q = FunctionSpace(mesh, q_family, q_degree, **kw)
    u, w = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)

    u_ex = target_field(mesh)
    dxq = dx(degree=quad or QUAD)
    u1_sq = float(assemble(inner(grad(u_ex), grad(u_ex)) * dxq))
    div_l2 = float(assemble(div(u_ex) ** 2 * dxq)) ** 0.5

    A_full = to_scipy(assemble(inner(grad(u), grad(w)) * dx)).tocsc()
    B_full = to_scipy(assemble(q * div(u) * dx)).tocsc()
    Mq = to_scipy(assemble(p * q * dx)).tocsc()
    with assemble(inner(grad(u_ex), grad(w)) * dxq).dat.vec_ro as vec:
        g_full = vec.array_r.copy()

    free, _ = retained_dofs(V, (1,), bc_mode)
    A = A_full[free][:, free].tocsc()
    B = B_full[:, free].tocsc()
    g = g_full[free]
    n_v, n_q = len(free), Q.dim()

    lu = spla.splu(A)
    v_star = lu.solve(g)
    eV_sq = u1_sq - float(g @ v_star)          # Galerkin: |u|^2 - |v*|^2

    rows = []
    for gamma in gammas:
        K = sp.bmat([[A, B.T], [B, -Mq / gamma]], format="csc")
        sol = spla.splu(K).solve(np.concatenate([g, np.zeros(n_q)]))
        v = sol[:n_v]
        eZ_sq = u1_sq - 2.0 * float(g @ v) + float(v @ (A @ v))
        divn = float(np.sqrt(max((B @ v) @ spla.spsolve(Mq, B @ v), 0.0)))
        rows.append({"gamma": gamma, "eZ": float(np.sqrt(max(eZ_sq, 0.0))),
                     "div_norm": divn})

    eV = float(np.sqrt(max(eV_sq, 0.0)))
    return {
        "label": label, "ncells": FunctionSpace(mesh, "DG", 0).dim(),
        "n_v": n_v, "n_q": n_q, "u1": float(np.sqrt(u1_sq)), "div_l2": div_l2,
        "constraint_ratio": n_q / V.dim(),
        "eV": eV, "gammas": rows, "eZ": rows[-1]["eZ"],
        "ratio": rows[-1]["eZ"] / eV if eV else float("inf"),
        "ratio_prev": rows[-2]["eZ"] / eV if eV else float("inf"),
        "seconds": time.time() - t0,
    }


def run(lcs, pairs, bc_mode="free", v_degree=2):
    out = {}
    for name, (fam, deg) in pairs.items():
        rows = []
        for lc in lcs:
            mesh = Mesh(shell_mesh(lc, os.path.join(HERE, f"infsup_shell_{lc:g}.msh")))
            r = amplification(mesh, fam, deg, v_degree=v_degree,
                              bc_mode=bc_mode, label=name)
            r["h"], r["lc"] = cell_size(mesh), lc
            rows.append(r)
            print(f"    {name:>14}  lc={lc:<5} ncells={r['ncells']:<5} "
                  f"e_V={r['eV']:.4e}  e_Z={r['eZ']:.4e}  R={r['ratio']:.4f}  "
                  f"(R at gamma/100 = {r['ratio_prev']:.4f}, {r['seconds']:.0f} s)",
                  flush=True)
        out[name] = rows
    return out


def report(name, rows):
    print(f"\n  {name}")
    print(f"  {'lc':>6} {'ncells':>7} {'n_q':>7} {'h':>7} {'dimQ/dimV':>10} "
          f"{'e_V':>11} {'e_Z':>11} {'R = e_Z/e_V':>12} {'|div v|':>10}")
    for r in rows:
        print(f"  {r['lc']:>6} {r['ncells']:>7} {r['n_q']:>7} {r['h']:>7.4f} "
              f"{r['constraint_ratio']:>10.4f} "
              f"{r['eV']:>11.4e} {r['eZ']:>11.4e} {r['ratio']:>12.4f} "
              f"{r['gammas'][-1]['div_norm']:>10.2e}")
    hs = np.array([r["h"] for r in rows])
    ratios = np.array([r["ratio"] for r in rows])
    slope = np.polyfit(np.log(hs), np.log(ratios), 1)[0]
    eVs = np.array([r["eV"] for r in rows])
    order = np.polyfit(np.log(hs), np.log(eVs), 1)[0]
    print(f"    e_V converges at O(h^{order:.2f});  R ~ h^{slope:+.3f} "
          f"(negative = growing under refinement = locking)")
    return slope


def main():
    print(__doc__)
    lcs = [float(x) for x in
           os.environ.get("AMP_LCS", "0.40,0.30,0.22,0.17,0.13").split(",")]

    print("\n=== mesh census (is this a lattice or genuine unstructured?) ===")
    print(f"  {'lc':>6} {'T':>7} {'V':>7} {'E':>7} {'F':>7} {'V-E+F-T':>8} "
          f"{'T/V':>6} {'dimQ/dimV':>10}")
    for lc in lcs:
        mesh = Mesh(shell_mesh(lc, os.path.join(HERE, f"infsup_shell_{lc:g}.msh")))
        c = census(mesh)
        nv = VectorFunctionSpace(mesh, "CG", 2).dim()
        nq = FunctionSpace(mesh, "DG", 1).dim()
        print(f"  {lc:>6} {c['T']:>7} {c['V']:>7} {c['E']:>7} {c['F']:>7} "
              f"{c['euler']:>8} {c['T_over_V']:>6.3f} {nq / nv:>10.4f}")

    pairs = {
        "CG2/P1-disc": ("DG", 1),   # what full-quadrature penalty implies
        "CG2/P0": ("DG", 0),        # what reduced volumetric quadrature implies
        "CG2/CG1 (TH)": ("CG", 1),  # stable control
    }

    print("\n=== amplification, no essential BCs (the actual GIA problem) ===",
          flush=True)
    free = run(lcs, pairs, bc_mode="free")
    slopes = {k: report(k, v) for k, v in free.items()}

    print("\n=== amplification, clamped CMB (the textbook, pessimistic, setting) ===",
          flush=True)
    clamped = run(lcs, {"CG2/P1-disc": ("DG", 1)}, bc_mode="clamped")
    report("CG2/P1-disc, clamped", clamped["CG2/P1-disc"])

    print("\n=== CG3/P2-disc: the other keyword-change remedy ===", flush=True)
    deg3 = run(lcs[:4], {"CG3/P2-disc": ("DG", 2)}, bc_mode="free", v_degree=3)
    report("CG3/P2-disc", deg3["CG3/P2-disc"])

    print("\n=== verdict ===")
    for k, rows in free.items():
        rs = [r["ratio"] for r in rows]
        print(f"  {k:>14}: R = {', '.join(f'{r:.3f}' for r in rs)}  "
              f"(R ~ h^{slopes[k]:+.3f})")
    return free, clamped


if __name__ == "__main__":
    main()
