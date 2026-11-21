"""S5 -- how to measure ||J - J^T||/||J|| when the space has Real blocks.

Monolithic aij assembly is refused (firedrake/assemble.py:1406-1407, raised out
of the SparsityFormatError handler in `_make_sparsity`), and
`gadopt/gravity_solver.py:1313-1320` refuses it deliberately for the same
reason.  Verification V1 nevertheless wants ||J - J^T||/||J|| at roundoff.

Four candidate routes are tried here on a toy mixed space that has a Real field,
in a KNOWN-SYMMETRIC and a KNOWN-ASYMMETRIC variant, so that whatever survives
is demonstrated to be sharp:

    A  mat_type="aij"      -- the baseline refusal, for the record
    B  mat_type="nest"     -- block by block, (i,j) against (j,i)^T
    C  mat_type="matfree" + column probing with unit vectors -> dense
    D  mat_type="matfree" + mult/multTranspose, no dense at all
"""
import os
import sys
import traceback

import numpy as np

import gadopt  # noqa: F401  -- import order, see SPIKE-RESULTS.md S2(d)
from firedrake import *
from firedrake.petsc import PETSc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_mesh  # noqa: E402


def pr(*a):
    PETSc.Sys.Print(*a)


def show_failure(label, exc):
    tb = traceback.extract_tb(sys.exc_info()[2])[-1]
    pr(f"  RAISES  {label}")
    pr(f"          {type(exc).__name__}: {str(exc).splitlines()[0][:180]}")
    pr(f"          at {tb.filename}:{tb.lineno} in {tb.name}")


# ----------------------------------------------------------------------
# the toy problem
# ----------------------------------------------------------------------
def build(variant, parent, sub):
    """A mixed (u, psi, c0, c1) system with a Real pair.

    variant "sym"      -> every block pair is an exact transpose
    variant "asym_uv"  -> the u/psi cross-mesh block is scaled by 1+1e-8
    variant "asym_real"-> the SUBMESH-VOLUME Real (rotation-like) row is scaled
                          by 1+1e-8, so the defect lives entirely in a Real
                          block.  This is the one the routes must not miss.
    """
    V = VectorFunctionSpace(sub, "CG", 1)
    P = FunctionSpace(parent, "CG", 1)
    R = FunctionSpace(parent, "R", 0)
    Z = MixedFunctionSpace([V, P, R, R])

    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    dx_p = Measure("dx", domain=parent,
                   intersect_measures=(Measure("dx", domain=sub),))
    ds_p = Measure("ds", domain=parent)

    DG0 = FunctionSpace(sub, "DG", 0)
    xs = SpatialCoordinate(sub)
    rho0 = Function(DG0).interpolate(2.0 + sin(3 * xs[0]) * cos(2 * xs[1]))
    xp = SpatialCoordinate(parent)
    e0 = cos(3 * atan2(xp[1], xp[0]))          # a DtN mode shape
    r_s = sqrt(dot(xs, xs))

    u, psi, c0, c1 = TrialFunctions(Z)
    w, v, nu0, nu1 = TestFunctions(Z)

    eps_uv = 1e-8 if variant == "asym_uv" else 0.0
    eps_r = 1e-8 if variant == "asym_real" else 0.0

    a = (
        # diagonal, symmetric by construction
        inner(sym(grad(u)), sym(grad(w))) * dx_m + inner(u, w) * dx_m
        + inner(grad(psi), grad(v)) * dx_p + psi * v * ds_p(spike_mesh.CURVE_OUTER)
        # cross-mesh coupling, the S4 pair
        + inner(rho0 * grad(psi), w) * dx_m
        + (1.0 + eps_uv) * rho0 * inner(u, grad(v)) * dx_m
        # Real pair 1: a PARENT-BOUNDARY (DtN-like) constraint row
        + c0 * nu0 * ds_p(spike_mesh.CURVE_OUTER)
        - e0 * psi * nu0 * ds_p(spike_mesh.CURVE_OUTER)
        - e0 * c0 * v * ds_p(spike_mesh.CURVE_OUTER)
        # Real pair 2: a SUBMESH-VOLUME (rotation-like) constraint row
        + c1 * nu1 * dx_m
        - (1.0 + eps_r) * rho0 * dot(xs, u) / r_s * nu1 * dx_m
        - rho0 * c1 * dot(xs, w) / r_s * dx_m
    )
    return Z, a


# ----------------------------------------------------------------------
# route A -- aij
# ----------------------------------------------------------------------
def route_A(Z, a, label):
    try:
        assemble(a, mat_type="aij")
    except Exception as exc:  # noqa: BLE001
        show_failure(f"[A] mat_type='aij'  ({label})", exc)
        return None
    pr(f"  OK      [A] mat_type='aij' ({label}) -- it worked after all")
    return True


# ----------------------------------------------------------------------
# route B -- nest, block by block
# ----------------------------------------------------------------------
def route_B(Z, a, label):
    try:
        A = assemble(a, mat_type="nest").petscmat
    except Exception as exc:  # noqa: BLE001
        show_failure(f"[B] mat_type='nest'  ({label})", exc)
        return None
    n = len(Z)
    worst, scale = 0.0, 0.0
    detail = []
    norms = {}
    for i in range(n):
        for j in range(n):
            try:
                Bij = A.getNestSubMatrix(i, j)
            except Exception as exc:  # noqa: BLE001
                show_failure(f"[B] getNestSubMatrix({i},{j})", exc)
                return None
            Bji = A.getNestSubMatrix(j, i)
            if Bij is None or Bji is None:
                continue
            Mi = Bij.convert("dense").getDenseArray().copy()
            Mj = Bji.convert("dense").getDenseArray().copy()
            norms[(i, j)] = np.abs(Mi).max() if Mi.size else 0.0
            d = np.abs(Mi - Mj.T).max() if Mi.size else 0.0
            scale = max(scale, norms[(i, j)])
            if d > 0:
                detail.append((i, j, d))
            worst = max(worst, d)
    rel = worst / scale if scale else float("nan")
    pr(f"  OK      [B] nest, block-by-block ({label}): "
       f"max|A_ij - A_ji^T| = {worst:.6e}, scale {scale:.6e}, rel {rel:.6e}")
    pr("            block max-abs grid (a zero here means the test does not "
       "cover that block):")
    for i in range(n):
        pr("              " + "  ".join(
            f"{norms.get((i, j), 0.0):9.3e}" for j in range(n)))
    for i, j, d in sorted(detail, key=lambda t: -t[2])[:4]:
        pr(f"            worst blocks: ({i},{j}) {d:.6e}")
    return rel


# ----------------------------------------------------------------------
# route C -- matfree, dense by column probing
# ----------------------------------------------------------------------
def matfree_dense(Z, a):
    A = assemble(a, mat_type="matfree").petscmat
    n = Z.dim()
    x, y = A.createVecRight(), A.createVecLeft()
    J = np.zeros((n, n))
    for k in range(n):
        x.set(0.0)
        x.setValue(k, 1.0)
        x.assemble()
        A.mult(x, y)
        J[:, k] = y.getArray().copy()
    return J


def route_C(Z, a, label):
    try:
        J = matfree_dense(Z, a)
    except Exception as exc:  # noqa: BLE001
        show_failure(f"[C] matfree column probe  ({label})", exc)
        return None
    d = np.linalg.norm(J - J.T) / np.linalg.norm(J)
    pr(f"  OK      [C] matfree column probe ({label}): "
       f"||J - J^T||_F/||J||_F = {d:.6e}   (max abs "
       f"{np.abs(J - J.T).max():.6e})")
    return d


# ----------------------------------------------------------------------
# route D -- matfree, mult vs multTranspose, no dense
# ----------------------------------------------------------------------
def route_D(Z, a, label, n_probe=12, seed=0):
    try:
        A = assemble(a, mat_type="matfree").petscmat
        rng = np.random.default_rng(seed)
        x, y = A.createVecRight(), A.createVecLeft()
        num, den = 0.0, 0.0
        for _ in range(n_probe):
            x.setArray(rng.standard_normal(x.getLocalSize()))
            A.mult(x, y)
            yt = y.duplicate()
            A.multTranspose(x, yt)
            yt.axpy(-1.0, y)
            num = max(num, yt.norm())
            den = max(den, y.norm())
    except Exception as exc:  # noqa: BLE001
        show_failure(f"[D] matfree mult/multTranspose  ({label})", exc)
        return None
    pr(f"  OK      [D] matfree (J - J^T)x over {n_probe} random x ({label}): "
       f"max||(J-J^T)x|| / max||Jx|| = {num / den:.6e}")
    return num / den


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    msh = os.path.join(here, "spike_annulus.msh")
    if not os.path.exists(msh):
        spike_mesh.generate(msh)
    parent = Mesh(msh)
    sub = Submesh(parent, 2, spike_mesh.CELL_MANTLE)

    pr("=" * 72)
    pr(f"S5  symmetry test with Real blocks   [{COMM_WORLD.size} rank(s)]")
    pr("=" * 72)

    variants = [("sym", "SYMMETRIC"),
                ("asym_uv", "ASYM in the u/psi block, eps=1e-8"),
                ("asym_real", "ASYM in a REAL block, eps=1e-8")]

    results = {}
    for variant, label in variants:
        pr(f"\n### {label}")
        Z, a = build(variant, parent, sub)
        pr(f"  space: {[Vs.ufl_element().family() for Vs in Z]}, "
           f"dim {Z.dim()}")
        route_A(Z, a, label)
        if COMM_WORLD.size == 1:
            results[("B", label)] = route_B(Z, a, label)
            if Z.dim() <= 4000:
                results[("C", label)] = route_C(Z, a, label)
        else:
            pr("  SKIP    [B] and [C]: both need a global dense transpose, so "
               "they are serial-only as written")
        results[("D", label)] = route_D(Z, a, label)

    pr("\n### does each route DISTINGUISH symmetric from asymmetric?")
    base = "SYMMETRIC"
    for route in ("B", "C", "D"):
        s = results.get((route, base))
        if s is None:
            pr(f"  {route}: not available")
            continue
        for _, label in variants[1:]:
            asym = results.get((route, label))
            if asym is None:
                continue
            ratio = asym / s if s else float("inf")
            pr(f"  {route}  vs {label:<36}  symmetric {s:.3e}  "
               f"asymmetric {asym:.3e}  ratio {ratio:.3e}  "
               f"{'SHARP' if asym > 100 * max(s, 1e-16) else 'NOT SHARP'}")


if __name__ == "__main__":
    main()
