"""Does selective reduced integration actually help?  A manufactured solution.

`gate_locking_amplification.py` showed `R = e_Z/e_V` growing like `h^(-1/2)` for
CG2/P1-disc and flat at 1.02-1.04 for CG2/P0.  That is not yet an argument for
reduced integration, and the objection is exact: **`R` is a ratio whose
denominator is common but whose numerator is taken over a different subspace.**
`Z_h` for P0 is `{v : int_K div v = 0 per cell}`, strictly larger than the
P1-disc kernel, so of course it approximates better — it constrains less.  A low
`R` could be a locking error traded for an incompressibility error, and nothing
in `R` distinguishes those.

The 2-D probe supplies direct grounds for the worry: reduced quadrature there
dropped a functional's convergence order from 2.91 to 1.90.

So this script stops measuring proxies and measures the error itself, against a
**known exact solution**.

## The manufactured problem

    sigma(u) = 2 mu dev(eps(u)) + lambda div(u) I,     eps = sym(grad),
    -div sigma(u) = f

with `u_ex = curl(A)`, `A = (sin(pi y) cos(pi z), sin(pi z) cos(pi x),
sin(pi x) cos(pi y))` — the same field the amplification study used, smooth and
**exactly divergence free**.

Two consequences make this the right test and both are worth stating, because
they are what make the comparison fair:

* `div(u_ex) = 0` kills the volumetric term identically, so `sigma_ex =
  2 mu eps(u_ex)` and `f = -div sigma_ex = -mu Laplacian(u_ex)`.  **`u_ex` is
  the exact solution for every `lambda`**, so sweeping `lambda` changes only the
  discretisation, never the target.  Nothing has to be extrapolated.
* The reduced-quadrature form is **exactly consistent** for this solution:
  `div(u_ex) = 0` pointwise, so the one-point rule integrates it to zero just as
  the full rule does.  The variational crime therefore contributes no
  consistency error here, and any order loss that shows up is a real property of
  the scheme rather than an artefact of the manufactured data.

Boundary conditions: `u = u_ex` strongly at the CMB, exact traction
`sigma_ex . n` naturally at the surface — the configuration `gate_locking.py`
uses, which has a free surface (so it is not the fully-clamped textbook setting)
and no rigid-body kernel (so no nullspace bookkeeping can bias the answer).
Clamping one boundary is where locking is worst, so the choice errs conservative.

## The three quadrature treatments

`inner(nabla_grad(v), K_eff div(u) I) = K_eff div(u) div(v)`, so the volumetric
contribution is one term and its quadrature can be changed on its own.

1. **full** — everything at the default degree.  `div(u)` is P1 for CG2, so the
   volumetric integrand is degree 2 and is integrated exactly.  This is the
   current `viscosity_term` behaviour and it implies the CG2/P1-disc pair.
2. **SRI** — the volumetric term at `dx(degree=0)`, a one-point rule, the
   deviatoric term untouched.  This implies the CG2/P0 pair, and it is the
   ~10-line change under consideration.
3. **uniform-reduced** — *both* terms at degree 0.  Not a proposal; a diagnostic.
   If the 2-D probe's order loss came from under-integrating the deviatoric term
   as well, this column will reproduce it and column 2 will not, which would
   explain the disagreement rather than leave it standing.

## Expected, stated before the run

* At `lambda/mu = 1.94` (compressible) all three deviatoric-exact columns should
  show the CG2 rates, `O(h^3)` in L2 and `O(h^2)` in the H1 seminorm.
* At `lambda/mu = 4999.7`, **full quadrature should lose order** if locking is
  real — that is the whole claim of the previous round, and if the order holds
  at 2 the amplification result has to be reconciled with this.
* **SRI should keep the order.**  If it does not, `R` was measuring the wrong
  side of a trade and the remedy fails.
* `||div u_h|| / ||grad u_h||` should fall like `1/lambda` for full quadrature.
  Under SRI only the cell averages of `div u_h` are driven to zero, so the
  pointwise divergence will be **larger**; how much larger is exactly the
  incompressibility that was given up, and it is reported rather than glossed.
"""

import os
import sys

import numpy as np

import gadopt  # noqa: F401  (import before firedrake; Irksome's order guard)
from firedrake import (
    Constant,
    DirichletBC,
    FacetNormal,
    Function,
    FunctionSpace,
    Identity,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    as_vector,
    assemble,
    cos,
    curl,
    div,
    ds,
    dot,
    dx,
    grad,
    inner,
    pi,
    sin,
    solve,
    sym,
    tr,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gate_infsup_3d import shell_mesh  # noqa: E402


def mean_cell_size(mesh):
    """Nominal `h = (volume/ncells)^(1/3)`, and why not the largest diameter.

    These meshes are generated independently per `lc`, not nested, so the
    *largest* cell diameter is a single-sliver statistic and is **not monotone
    in the cell count**: lc = 0.52 gives 0.6727 and the finer lc = 0.40 gives
    1.0005.  Fitting a convergence order against it produces nonsense (observed
    orders of -0.14, 4.02 and 7.84 in the same column).  The mean size is
    monotone by construction and is what every order below is fitted against.
    """
    vol = float(assemble(Constant(1.0) * dx(domain=mesh)))
    return (vol / FunctionSpace(mesh, "DG", 0).dim()) ** (1.0 / 3.0)

QUAD = 8
LU = {"ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}

# nu = 0.28, 0.49, 0.4999 through K/mu = 2(1+nu)/(3(1-2nu)); the same ladder the
# 2-D probe swept, so the two studies can be compared at the same stiffness
RATIOS = {"1.94": 1.94, "49.7": 49.7, "4999.7": 4999.7}


def target_field(mesh):
    x, y, z = SpatialCoordinate(mesh)
    A = as_vector([sin(pi * y) * cos(pi * z),
                   sin(pi * z) * cos(pi * x),
                   sin(pi * x) * cos(pi * y)])
    return curl(A)


def solve_case(mesh, ratio, vol_degree, dev_degree=None, degree=2):
    """One solve.  `vol_degree=None` means the default (full) rule."""
    V = VectorFunctionSpace(mesh, "CG", degree)
    u, v = TrialFunction(V), TestFunction(V)
    mu = Constant(1.0)
    lam = Constant(ratio)
    u_ex = target_field(mesh)
    n = FacetNormal(mesh)
    dxq = dx(degree=QUAD)

    def eps(w):
        return sym(grad(w))

    def dev(t):
        return t - (1.0 / 3.0) * tr(t) * Identity(3)

    dx_dev = dx if dev_degree is None else dx(degree=dev_degree)
    dx_vol = dx if vol_degree is None else dx(degree=vol_degree)

    a = (2.0 * mu * inner(dev(eps(u)), eps(v)) * dx_dev
         + lam * div(u) * div(v) * dx_vol)

    # div(u_ex) = 0, so sigma_ex is purely deviatoric and f = -mu Laplacian(u_ex)
    f = -mu * div(grad(u_ex))
    sigma_ex = 2.0 * mu * eps(u_ex)
    L = inner(f, v) * dxq + inner(dot(sigma_ex, n), v) * ds(2, degree=QUAD)

    uh = Function(V)
    solve(a == L, uh, bcs=[DirichletBC(V, u_ex, 1)], solver_parameters=LU)

    e = uh - u_ex
    e_l2 = float(assemble(inner(e, e) * dxq)) ** 0.5
    e_h1 = float(assemble(inner(grad(e), grad(e)) * dxq)) ** 0.5
    u_l2 = float(assemble(inner(u_ex, u_ex) * dxq)) ** 0.5
    u_h1 = float(assemble(inner(grad(u_ex), grad(u_ex)) * dxq)) ** 0.5

    gh = float(assemble(inner(grad(uh), grad(uh)) * dxq)) ** 0.5
    dh = float(assemble(div(uh) ** 2 * dxq)) ** 0.5

    # Two different divergences, and keeping them apart is the whole point of
    # this column.  `div(uh)` is P1 for CG2, so interpolating into DG1 is exact
    # and its nodal extremes are the **true pointwise** maximum — that is what
    # incompressibility actually costs.  The DG0 projection is the **cell mean**,
    # which is precisely and only what the one-point rule constrains, so a tiny
    # value there under SRI is the scheme working as designed and says nothing
    # about the pointwise divergence.
    DG1 = FunctionSpace(mesh, "DG", 1)
    DG2 = FunctionSpace(mesh, "DG", 2)
    P0 = FunctionSpace(mesh, "DG", 0)
    dmax = float(np.abs(Function(DG1).interpolate(div(uh)).dat.data_ro).max())
    gmax = float(Function(DG2).interpolate(
        inner(grad(uh), grad(uh))).dat.data_ro.max()) ** 0.5
    dcell = float(np.abs(Function(P0).project(div(uh)).dat.data_ro).max())

    return {"e_l2": e_l2 / u_l2, "e_h1": e_h1 / u_h1,
            "div_rel": dh / gh, "div_max_rel": dmax / gmax,
            "div_cellmean_rel": dcell / gmax, "ndof": V.dim()}


def orders(hs, errs):
    hs, errs = np.asarray(hs), np.asarray(errs)
    return [float("nan")] + [
        float(np.log(errs[i - 1] / errs[i]) / np.log(hs[i - 1] / hs[i]))
        for i in range(1, len(errs))
    ]


def sweep(lcs, cases, ratio_name, ratio):
    print(f"\n=== lambda/mu = {ratio_name} ===", flush=True)
    meshes = [Mesh(shell_mesh(lc, os.path.join(HERE, f"infsup_shell_{lc:g}.msh")))
              for lc in lcs]
    hs = [mean_cell_size(m) for m in meshes]
    out = {}
    for name, kw in cases.items():
        try:
            rows = [solve_case(m, ratio, **kw) for m in meshes]
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {name}\n    FAILED: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[-1].strip()}", flush=True)
            continue
        o_l2 = orders(hs, [r["e_l2"] for r in rows])
        o_h1 = orders(hs, [r["e_h1"] for r in rows])
        print(f"\n  {name}")
        print(f"  {'lc':>6} {'ndof':>8} {'h':>7} {'relL2':>11} {'ord':>6} "
              f"{'relH1':>11} {'ord':>6} {'L2 div/grad':>12} "
              f"{'maxdiv/maxgrad':>14} {'cellmean':>10}")
        for lc, h, r, a, b in zip(lcs, hs, rows, o_l2, o_h1):
            print(f"  {lc:>6} {r['ndof']:>8} {h:>7.4f} {r['e_l2']:>11.4e} "
                  f"{a:>6.2f} {r['e_h1']:>11.4e} {b:>6.2f} "
                  f"{r['div_rel']:>12.3e} {r['div_max_rel']:>14.3e} "
                  f"{r['div_cellmean_rel']:>10.3e}", flush=True)
        out[name] = {"rows": rows, "o_l2": o_l2, "o_h1": o_h1, "hs": hs}
    return out


def main():
    print(__doc__)
    lcs = [float(x) for x in
           os.environ.get("SRI_LCS", "0.40,0.30,0.22,0.17,0.13").split(",")]

    # a one-point rule integrates constants exactly, so degree 0 and degree 1
    # are the same rule on a simplex; both are run to confirm that, because if
    # they differ the "reduced" column is not what it claims to be
    cases = {
        "full quadrature (implies CG2/P1-disc)": {"vol_degree": None},
        "SRI, volumetric at degree 0 (implies CG2/P0)": {"vol_degree": 0},
        "SRI, volumetric at degree 1": {"vol_degree": 1},
        "uniform-reduced, BOTH at degree 0 (diagnostic)":
            {"vol_degree": 0, "dev_degree": 0},
    }

    results = {}
    for rname, ratio in RATIOS.items():
        results[rname] = sweep(lcs, cases, rname, ratio)

    print("\n=== verdict ===")
    for rname in RATIOS:
        print(f"  lambda/mu = {rname}")
        for name, d in results[rname].items():
            print(f"    {name:>48}: H1 order (finest pair) "
                  f"{d['o_h1'][-1]:+.2f},  L2 {d['o_l2'][-1]:+.2f},  "
                  f"relH1 {d['rows'][-1]['e_h1']:.3e}")
    return results


if __name__ == "__main__":
    main()
