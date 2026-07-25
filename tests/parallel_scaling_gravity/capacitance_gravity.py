r"""Capacitance-matrix diagnostic for the gravitational Poisson DtN Schur block.

The wall-free half of the scaling study (see ``SCALING-ANALYSIS.md``). The coupled
``GravitySolver`` fieldsplit solve is capped at L = 6 by PETSc's 128-field
``PCFIELDSPLIT`` limit, but the question the L axis exists to answer — does the
DtN Schur cost grow like L^2 or mildly with the truncation — is a property of the
Schur complement ``S = D - C A^{-1} B`` alone, and ``S`` can be built without ever
constructing the multiplier mixed space or touching ``PCFIELDSPLIT``. So this
routine reaches all L = 2..30.

Construction, matching ``gadopt.gravity_solver.GravitySolver.set_form`` term for
term so the signs are right by construction:

* ``A`` is the Robin-shifted potential Laplacian
  ``a(psi, v) = grad(psi).grad(v) dx + sum_boundaries (alpha/R) psi v ds``,
  assembled and solved with the same CG + GAMG parameters the iterative preset
  uses for its ``fieldsplit_0`` block.
* For each treated mode ``k`` (over both boundaries), the boundary mode vector is
  ``m_k = assemble(e_k v ds)``. The constraint row contributes the diagonal
  ``D_kk = -norm_k`` (the Real test function integrates ``scale_k`` to
  ``scale_k * area = norm_k``), the flux feedback contributes ``B`` column
  ``(lam_k - alpha/R_k) m_k``, and the constraint row contributes ``C`` row
  ``m_k^T``. Hence

      S_jk = -norm_j delta_jk - (lam_k - alpha/R_k) * (m_j . A^{-1} m_k).

The primary deliverable is the iteration count of a dense GMRES on ``S`` to 1e-6
for several right-hand sides — the source-independent answer, valid even though
``S`` is non-normal (the symmetric Gram core is scaled column-wise per mode).
``cond(S)``, the eigenvalue clustering, and the 2x2 same-mode block off-diagonal
weight are supporting numbers.

Run locally, e.g.

    PYTHONPATH=<worktree> <firedrake-python> capacitance_gravity.py 4 --lmax 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from gadopt import *  # noqa: E402
from gadopt.gravity_solver import SphericalDtN  # noqa: E402
from gadopt.utility import CombinedSurfaceMeasure  # noqa: E402
sys.argv = _argv

from gravity_cubed_sphere import RMIN, RMAX, LAYERS  # noqa: E402

ALPHA = 1.0  # GravitySolver.alpha, the Robin shift


# Plain-assembled CG + GAMG for the SPD Robin-shifted Laplacian. These are the
# same GAMG knobs the iterative preset carries (minus the SPDAssembledPC
# indirection, which is for the matfree fieldsplit; here A is an assembled aij
# matrix so GAMG takes the unprefixed options directly).
_A_SOLVER_PARAMETERS = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-10,          # tighter than the study's 1e-8: keeps S clean
    "ksp_max_it": 1000,
    "pc_type": "gamg",
    "mg_levels_pc_type": "sor",
    "pc_gamg_threshold": 0.01,
    "pc_gamg_square_graph": 100,
    "pc_gamg_coarse_eq_limit": 1000,
    "pc_gamg_mis_k_minimum_degree_ordering": True,
}


def boundary_geometry(mesh, X, ds_measure, bc_id):
    """Side, radius and discrete area of a DtN boundary, as GravitySolver measures them."""
    n = FacetNormal(mesh)
    r = sqrt(dot(X, X))
    dss = ds_measure(bc_id)
    area = assemble(1 * dss)
    R = assemble(r * dss) / area
    side = "exterior" if assemble(dot(n, X) * dss) > 0 else "interior"
    return side, R, area


def build_capacitance(ref_level, lmax, comm=None):
    """Assemble the dense n x n DtN Schur complement S at (ref_level, lmax).

    Returns (S, meta) where meta carries the mode table (boundary id, key, l, m,
    lam, norm, R) in the same order as S's rows, plus the worst W-symmetry
    residual and the per-mode A-solve iteration counts.
    """
    nlayers = LAYERS[ref_level]
    mesh2d = CubedSphereMesh(RMIN, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial",
                        layer_height=(RMAX - RMIN) / nlayers)
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)
    X = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    quad_degree = 2 * (max(lmax, 1) + 1)
    ds_m = CombinedSurfaceMeasure(mesh, quad_degree)

    dtn = SphericalDtN(lmax)
    bc_ids = [boundary.top, boundary.bottom]

    # Robin-shifted Laplacian A, summing the shift over both DtN boundaries.
    u, v = TrialFunction(V), TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    geom = {}
    for bc_id in bc_ids:
        side, R, area = boundary_geometry(mesh, X, ds_m, bc_id)
        geom[bc_id] = (side, R, area)
        a += (ALPHA / R) * u * v * ds_m(bc_id)
    A = assemble(a)
    A_solver = LinearSolver(A, solver_parameters=_A_SOLVER_PARAMETERS)

    # Full mode table across both boundaries, in set_form order.
    modes = []  # (bc_id, R, area, DtNMode)
    for bc_id in bc_ids:
        side, R, area = geom[bc_id]
        for mode in dtn.modes(side, R, X):
            modes.append((bc_id, R, area, mode))
    n = len(modes)

    # m_k = assemble(e_k v ds) (a cofunction) and x_k = A^{-1} m_k.
    m_vecs, x_vecs, a_iters = [], [], []
    for bc_id, R, area, mode in modes:
        m_k = assemble(mode.expr * v * ds_m(bc_id))   # cofunction over V'
        x_k = Function(V)
        A_solver.solve(x_k, m_k)
        a_iters.append(A_solver.ksp.getIterationNumber())
        m_vecs.append(m_k)
        x_vecs.append(x_k)

    # W_jk = <m_j, x_k> = m_j^T A^{-1} m_k, the natural cofunction-function
    # pairing (sum of dof products). Symmetric since A is SPD; checked below.
    W = np.empty((n, n))
    for k in range(n):
        with x_vecs[k].dat.vec_ro as xk:
            for j in range(n):
                with m_vecs[j].dat.vec_ro as mj:
                    W[j, k] = mj.dot(xk)
    w_sym = float(np.max(np.abs(W - W.T)) / max(np.max(np.abs(W)), 1e-30))

    col_factor = np.array([mode.lam - ALPHA / R for _, R, _, mode in modes])
    # Diagonal from the assembled constraint row: D_jj = -scale_j * area_j (the
    # discrete boundary area), not the analytic -norm = -R^2. They agree up to
    # the mesh area error, which is largest at the marginal level-4/high-L corner
    # this diagnostic exists to probe, so use the discrete value.
    diag = np.array([-mode.scale * area for _, _, area, mode in modes])
    S = np.diag(diag) - col_factor[np.newaxis, :] * W

    meta = {
        "n": n,
        "modes": [
            {"bc_id": str(bc_id), "key": mode.key,
             "l": int(mode.key[1:].split(",")[0]),
             "m": int(mode.key.split(",")[1]),
             "lam": float(mode.lam), "norm": float(mode.norm), "R": float(R)}
            for bc_id, R, _, mode in modes
        ],
        "w_symmetry_residual": w_sym,
        "a_solve_iterations_mean": float(np.mean(a_iters)),
        "a_solve_iterations_max": int(np.max(a_iters)),
    }
    return S, meta


def dense_gmres_count(S, b, rtol=1e-6):
    """(iterations, converged) of full GMRES on the dense S for rhs b.

    scipy's `maxiter` counts restart *cycles*, not inner iterations, so with
    restart = n (full GMRES, which terminates within one cycle) maxiter = 2 is
    one cycle plus slack; a larger value would let a stagnating S burn ~n cycles
    of n steps silently. `info == 0` means converged to rtol.
    """
    counter = {"n": 0}

    def cb(_):
        counter["n"] += 1

    n = S.shape[0]
    _, info = scipy.sparse.linalg.gmres(
        S, b, rtol=rtol, restart=n, maxiter=2, callback=cb,
        callback_type="pr_norm")
    return counter["n"], (info == 0)


def block_offdiag_weight(S, meta):
    """||S - blockdiag_2x2(S)|| / ||S|| over same-(l,m) top/bottom blocks."""
    # Group row indices by (l, m); each group is the <=2 boundaries of that mode.
    groups = {}
    for i, mode in enumerate(meta["modes"]):
        groups.setdefault((mode["l"], mode["m"]), []).append(i)
    B = np.zeros_like(S)
    for idx in groups.values():
        for j in idx:
            for k in idx:
                B[j, k] = S[j, k]
    return float(np.linalg.norm(S - B) / np.linalg.norm(S))


def offline_gmres_suite(S, seed_key):
    """GMRES counts for random, adversarial (worst eigvec mix), and flat rhs.

    Deterministic RNG seeded off a passed integer (scripts here cannot call the
    unseeded default), so repeated runs match.
    """
    n = S.shape[0]
    rng = np.random.default_rng(seed_key)
    # One non-symmetric eigendecomposition; both eigenvalues and vectors from it.
    w, vr = scipy.linalg.eig(S)
    order = np.argsort(np.abs(w))
    out = {}
    out["flat"], out["flat_converged"] = dense_gmres_count(S, np.ones(n))
    out["random"], out["random_converged"] = dense_gmres_count(
        S, rng.standard_normal(n))
    # Adversarial (a labelled heuristic, not a proven worst case): align the rhs
    # with the smallest-magnitude eigenvector directions.
    adver = vr[:, order[: max(1, n // 4)]].sum(axis=1).real
    out["adversarial"], out["adversarial_converged"] = dense_gmres_count(S, adver)
    out["cond"] = float(np.linalg.cond(S))
    out["eig_abs_min"] = float(np.min(np.abs(w)))
    out["eig_abs_max"] = float(np.max(np.abs(w)))
    return out


def run(ref_level, lmax, out_dir=None, seed=12345):
    t0 = time.perf_counter()
    S, meta = build_capacitance(ref_level, lmax)
    build_time = time.perf_counter() - t0
    n = meta["n"]

    log(f"capacitance L={lmax} level={ref_level}: n={n}, "
        f"W symmetry residual {meta['w_symmetry_residual']:.2e}, "
        f"A-solve iters mean {meta['a_solve_iterations_mean']:.1f}, "
        f"build {build_time:.1f}s")

    suite = offline_gmres_suite(S, seed)
    block_w = block_offdiag_weight(S, meta)

    log(f"  offline GMRES on S: flat={suite['flat']}, random={suite['random']}, "
        f"adversarial={suite['adversarial']} | 2(L+1)={2 * (lmax + 1)}")
    log(f"  cond(S)={suite['cond']:.3e}, eig|min|={suite['eig_abs_min']:.3e}, "
        f"eig|max|={suite['eig_abs_max']:.3e}, block-offdiag weight={block_w:.3e}")

    if not (suite["flat_converged"] and suite["random_converged"]):
        log("  WARNING: offline GMRES on S did not converge to rtol for some rhs")

    summary = {
        "level": ref_level,
        "lmax": lmax,
        "n": n,
        "gmres_flat": suite["flat"],
        "gmres_random": suite["random"],
        "gmres_adversarial": suite["adversarial"],
        "gmres_converged": bool(
            suite["flat_converged"] and suite["random_converged"]
            and suite["adversarial_converged"]),
        "cond_S": suite["cond"],
        "eig_abs_min": suite["eig_abs_min"],
        "eig_abs_max": suite["eig_abs_max"],
        "block_offdiag_weight": block_w,
        "w_symmetry_residual": meta["w_symmetry_residual"],
        "a_solve_iterations_mean": meta["a_solve_iterations_mean"],
        "a_solve_iterations_max": meta["a_solve_iterations_max"],
        "build_time": build_time,
    }

    out_dir = Path(out_dir) if out_dir else Path.cwd()
    sidecar = out_dir / f"capacitance_level{ref_level}_lmax{lmax}.json"
    if COMM_WORLD.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(sidecar, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"wrote {sidecar}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", type=int)
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    run(args.level, args.lmax, out_dir=args.out_dir, seed=args.seed)
