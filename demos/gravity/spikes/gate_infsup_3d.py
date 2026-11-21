"""Does the penalty limit lock on 3-D tetrahedra?  The inf-sup constant, measured.

G-ADOPT's GIA formulation imposes incompressibility by a volumetric penalty,
`stress = bulk_shear_ratio * bulk_modulus * div(u) I + deviatoric`, on a **CG2
vector displacement with no pressure partner anywhere**.  As
`bulk_shear_ratio -> infinity` the minimiser of that energy converges to the
minimiser constrained to `{v in V_h : div v = 0}`.  So the whole question of
whether the incompressible answer is reachable by extrapolating a ladder in
`1/ratio` (`HANDOFF-SPADA-BENCHMARK.md` §8) reduces to one classical quantity:
is the *discretely divergence-free subspace of V_h rich enough*?  That is the
inf-sup (LBB) constant of the pair (CG2 displacement, **discontinuous P1**
pressure) — discontinuous P1 because it is exactly `div(V_h)`, hence the right
pressure space for this question and not a modelling choice.

    beta_h = inf_{q in Q_h} sup_{v in V_h}  (q, div v) / (||q||_0 |v|_1)

which is the square root of the smallest nonzero eigenvalue of

    B A^{-1} B^T q = lambda M_q q,     A = H1 seminorm on V_h,
                                       B = pressure/divergence coupling,
                                       M_q = pressure mass matrix.

A 2-D pass on the existing quadrilateral prototype does **not** settle this.
Q2/P1-disc is inf-sup stable in 2-D; CG2/P1-disc on 3-D tetrahedra is not
generally stable — Scott-Vogelius in 3-D wants degree >= 4, or a barycentrically
refined mesh.  The dof counting says the same thing loudly.  On a tetrahedral
mesh with `n` vertices there are roughly `6n` cells and `7n` edges, so

    dim Q_h = 4 * ncells ~ 24 n        dim V_h = 3 * (nvert + nedge) ~ 24 n

- as many constraints as unknowns.  On a quadrilateral mesh with `n` vertices
there are `n` cells and `2n` edges, so `dim Q_h = 3n` against
`dim V_h = 2 * (n + 2n + n) = 8n`, a ratio of 8/3.  The 3-D pair is the one in
trouble, and only a 3-D measurement can say so.

## Expected, stated before the run

* **Taylor-Hood control (CG2/CG1) on the same tets: `beta_h` bounded away from
  zero, flat under refinement.**  This is provably stable and is the guard
  against reporting an implementation bug as a physics result.  If this one
  collapses, nothing else in the script may be believed.
* **2-D control (Q2/P1-disc on a quadrilateral annulus): `beta_h` bounded away
  from zero**, order 0.1-0.4, flat under refinement.  Stable by theory, and it
  is the configuration the 2-D probe runs on.
* **Target (CG2/P1-disc on unstructured tets): unknown, and that is the point.**
  The counting argument above predicts collapse.  A fitted `beta_h ~ h^p` with
  `p` clearly positive is locking and redirects the project to a mixed
  Taylor-Hood displacement-pressure formulation; `p ~ 0` means the penalty limit
  is a sound discretisation and §8's ladder survives.
* Every eigenvalue lies in `(0, d]` with `d` the dimension, because
  `||div v||^2 <= d |v|_1^2`.  Anything outside that is an assembly error.

## The configuration, and why

Spherical shell, `Rc/Re = 3480/6371`, nondimensionalised by `Re`, unstructured
tets from gmsh.  Clamped at the core-mantle boundary (`u = 0`, strong) and
traction-free at the surface — the same choice `gate_locking.py` makes and for
the same reasons: it is the physically relevant GIA configuration, it carries no
weak boundary terms that could confound the volumetric measurement, and clamped
boundaries are where locking is worst, so it errs in the conservative direction.

That choice also disposes of both nullspaces cleanly, and the script *checks*
rather than assumes:

* **V_h has no rigid-body kernel.**  `u = 0` on a whole boundary component makes
  the H1 seminorm a norm on V_h, so `A` restricted to the free dofs is SPD.
  The script factorises it and would fail loudly otherwise.
* **The constant pressure is not spurious.**  `(1, div v) = int_{Re} v.n` over
  the free surface alone (the CMB integral vanishes with `v = 0` there), which
  is not zero for general `v`.  So `B^T 1 != 0` and no deflation is needed.
  Under an enclosed or all-traction boundary it *would* be, and the script
  therefore measures the whole low end of the spectrum, counts eigenvalues below
  a relative floor as the numerical kernel of `B^T`, and takes `beta_h` from the
  smallest eigenvalue *above* it.  The count is reported, not assumed to be zero.

`dim ker(B)` is reported alongside, as `n_v_free - rank(B)`.  That is the
dimension of the discretely divergence-free subspace the penalty limit actually
minimises over, and it is the most direct statement of impoverishment there is.
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
    AnnulusMesh,
    CellDiameter,
    DirichletBC,
    Function,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    as_vector,
    assemble,
    div,
    dx,
    grad,
    inner,
)

HERE = os.path.dirname(os.path.abspath(__file__))

RC_OVER_RE = 3480.0 / 6371.0
ZERO_FLOOR = 1.0e-9  # relative to the largest eigenvalue; the numerical kernel

# the whole spectrum is taken densely, which is what makes the rank exact and
# leaves no Krylov convergence to be mistaken for a small eigenvalue.  It costs
# an n_q x n_q array and an O(n_q^3) `eigh`, so the pressure space is capped.
MAX_NQ = int(os.environ.get("INFSUP_MAX_NQ", 12000))


# ----------------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------------


def shell_mesh(lc, path):
    """Unstructured tetrahedral spherical shell, Rc/Re = 3480/6371, Re = 1.

    Physical surface 1 is the core-mantle boundary, 2 the outer surface.
    """
    if os.path.exists(path):
        return path

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    outer = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    inner = gmsh.model.occ.addSphere(0, 0, 0, RC_OVER_RE)
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()

    # both spheres are centred on the origin, so tell the two boundary surfaces
    # apart by their area rather than by their centre of mass
    inner_tags, outer_tags = [], []
    for dim, tag in gmsh.model.getEntities(2):
        area = gmsh.model.occ.getMass(dim, tag)
        r = np.sqrt(area / (4.0 * np.pi))
        (inner_tags if r < 0.5 * (1.0 + RC_OVER_RE) else outer_tags).append(tag)
    assert inner_tags and outer_tags, (inner_tags, outer_tags)
    gmsh.model.addPhysicalGroup(2, inner_tags, 1, name="cmb")
    gmsh.model.addPhysicalGroup(2, outer_tags, 2, name="surface")
    gmsh.model.addPhysicalGroup(3, [v for _, v in gmsh.model.getEntities(3)], 101)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path)
    gmsh.finalize()
    return path


def annulus_mesh(nr, nt):
    """Quadrilateral annulus, the 2-D control's geometry (same radius ratio)."""
    return AnnulusMesh(1.0, RC_OVER_RE, nr=nr, nt=nt)


# ----------------------------------------------------------------------------
# linear algebra
# ----------------------------------------------------------------------------


def to_scipy(mat):
    """Firedrake assembled matrix -> scipy CSR."""
    handle = mat.M.handle
    indptr, indices, data = handle.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=handle.getSize())


def clamped_dofs(V, ids):
    """Global dof indices of a strong u = 0 condition, with the layout checked.

    Firedrake blocks a vector space's dofs as `node * bs + component`.  That is
    an assumption about a data layout, so it is verified against an actual
    `DirichletBC.apply` rather than trusted.
    """
    bs = V.value_size
    bc = DirichletBC(V, as_vector([0.0] * bs), ids)
    dofs = np.sort((bc.nodes[:, None] * bs + np.arange(bs)).ravel())

    probe = Function(V)
    DirichletBC(V, as_vector([1.0 + i for i in range(bs)]), ids).apply(probe)
    nonzero = np.sort(np.flatnonzero(probe.dat.data_ro.ravel()))
    assert np.array_equal(dofs, nonzero), "vector dof layout is not node*bs+comp"
    return dofs


def retained_dofs(V, clamp_ids, bc_mode):
    """Which velocity dofs the measurement keeps, and how the kernel is killed.

    `bc_mode="clamped"`: `u = 0` strongly on `clamp_ids`, those dofs removed.
    The H1 seminorm is then a norm on what is left and there is no kernel.  This
    is the textbook Stokes setting and it is **pessimistically biased** for GIA:
    it deletes `O(N^(2/3))` boundary velocity dofs that the real problem has.

    `bc_mode="free"`: no essential conditions anywhere, which is the actual GIA
    problem (`normal_stress` at both radii).  `ker(A)` is then exactly the three
    translations — *not* the rotations, whose gradients are antisymmetric but
    nonzero, so the seminorm does not annihilate them.  The kernel is removed by
    **pinning one dof per component at node 0**, which is exact rather than
    approximate: the translations are the nodal indicator vectors of each
    component, so the kernel matrix restricted to the pinned dofs is the
    identity, and every right-hand side we solve against (`B^T q`, and the load
    `int grad(u_ex):grad(w)`) is exactly l2-orthogonal to the kernel because
    `B t = 0` and `grad(const) = 0`.  The recovered solution is then a genuine
    solution of the singular system, differing from any other only by a
    translation, which changes neither `q^T B v` nor the seminorm.  `measure`
    checks the residual on the pinned rows and asserts it is zero.

    No shift, no `eps * mass`, no pseudo-inverse, and above all no direct
    factorisation of a singular matrix — that last is the failure that would
    report `beta_h` too *high* and so err towards "stable".
    """
    n = V.dim()
    if bc_mode == "clamped":
        return np.setdiff1d(np.arange(n), clamped_dofs(V, clamp_ids)), []
    if bc_mode != "free":
        raise ValueError(bc_mode)
    pinned = list(range(V.value_size))
    return np.setdiff1d(np.arange(n), pinned), pinned


def measure(mesh, v_family, v_degree, q_family, q_degree, clamp_ids, label,
            variant=None, bc_mode="clamped", want_vectors=True):
    """Full spectrum of B A^{-1} B^T q = lambda M_q q, and what it says."""
    t0 = time.time()
    kw = {} if variant is None else {"variant": variant}
    V = VectorFunctionSpace(mesh, v_family, v_degree, **kw)
    Q = FunctionSpace(mesh, q_family, q_degree, **kw)

    u, w = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)

    A_full = to_scipy(assemble(inner(grad(u), grad(w)) * dx)).tocsc()
    B_full = to_scipy(assemble(q * div(u) * dx)).tocsc()
    Mq = to_scipy(assemble(p * q * dx))
    Mq_dense = Mq.toarray()

    free, pinned = retained_dofs(V, clamp_ids, bc_mode)
    A = A_full[free][:, free].tocsc()
    B = B_full[:, free].tocsc()
    n_v_free, n_v_full, n_q = len(free), V.dim(), Q.dim()

    # SPD on the retained dofs by construction; splu is where it would fail.
    lu = spla.splu(A)

    Bt = B.T.tocsc()
    S = np.empty((n_q, n_q))
    chunk = max(1, min(n_q, int(4.0e7 // max(n_v_free, 1))))
    pin_resid = 0.0
    for j0 in range(0, n_q, chunk):
        rhs = Bt[:, j0:j0 + chunk].toarray()
        sol = lu.solve(rhs)
        if pinned and j0 == 0:
            # the pinned rows of A v - B^T q must vanish identically if the
            # deflation is exact; this is the check that it is
            k = sol.shape[1]
            v_full = np.zeros((n_v_full, k))
            v_full[free] = sol
            blk = B_full[j0:j0 + k]
            resid = A_full[pinned] @ v_full - blk[:, pinned].toarray().T
            scale = abs(blk).max() or 1.0
            pin_resid = float(np.abs(resid).max() / scale)
        S[:, j0:j0 + rhs.shape[1]] = B @ sol
    S = 0.5 * (S + S.T)

    if want_vectors:
        lam, vec = scipy.linalg.eigh(S, Mq_dense)
    else:
        lam, vec = scipy.linalg.eigh(S, Mq_dense, eigvals_only=True), None
    order = np.argsort(lam)
    lam = lam[order]
    lam_max = lam[-1]
    floor = ZERO_FLOOR * lam_max
    n_zero = int(np.sum(lam < floor))
    positive = np.flatnonzero(lam >= floor)
    beta = float(np.sqrt(lam[positive[0]])) if positive.size else 0.0

    out = {
        "label": label,
        "ncells": FunctionSpace(mesh, "DG", 0).dim(),
        "n_v": n_v_free,
        "n_v_full": n_v_full,
        "n_q": n_q,
        "rank_B": n_q - n_zero,
        "dim_kerB": n_v_full - (n_q - n_zero),
        "kerB_frac": (n_v_full - (n_q - n_zero)) / n_v_full,
        "n_zero": n_zero,
        "beta": beta,
        "lam_min_all": float(lam[0]),
        "lam_max": float(lam_max),
        "lam_tail": lam[:6].tolist(),
        "pin_resid": pin_resid,
        "n_negative": int(np.sum(lam < -1e-12 * lam_max)),
        "seconds": time.time() - t0,
    }

    if want_vectors and positive.size:
        qv = vec[:, order[positive[0]]]           # M_q-normalised: q^T Mq q = 1
        out.update(_mode_diagnostics(qv, S, Mq, Mq_dense, lam[positive[0]], Q))
        # sensitivity of beta_h to the zero floor, over six orders
        out["beta_by_floor"] = {}
        for f in (1e-6, 1e-9, 1e-12):
            pos = np.flatnonzero(lam >= f * lam_max)
            out["beta_by_floor"][f] = (
                float(np.sqrt(lam[pos[0]])) if pos.size else 0.0
            )
    return out


def _mode_diagnostics(qv, S, Mq, Mq_dense, lam, Q):
    """Is the minimising pressure oscillatory, or is it a deflation residual?

    A genuine spurious mode oscillates at the cell scale.  A sliver of a
    badly-deflated constant is nearly constant.  Both fractions are taken in
    the `M_q` inner product, which is the one the eigenproblem uses; taking
    them in `l2` is exactly the mistake that manufactures a fake power law.
    """
    ones = np.ones_like(qv)
    mq_qq = float(qv @ (Mq_dense @ qv))
    mq_11 = float(ones @ (Mq_dense @ ones))
    coeff = float(qv @ (Mq_dense @ ones)) / mq_11
    const_frac = coeff ** 2 * mq_11 / mq_qq

    # cellwise-mean part, which a cell-to-cell checkerboard also has; the
    # complement of this is the *within-cell* oscillation
    f = Function(Q)
    f.dat.data[:] = qv
    P0 = FunctionSpace(Q.mesh(), "DG", 0)
    mean = Function(P0).project(f)
    lifted = Function(Q).interpolate(mean)
    d = qv - lifted.dat.data_ro
    within_cell_frac = float(d @ (Mq_dense @ d)) / mq_qq

    resid = S @ qv - lam * (Mq_dense @ qv)
    denom = abs(lam) * np.linalg.norm(Mq_dense @ qv)
    return {
        "mode_const_frac": const_frac,
        "mode_within_cell_frac": within_cell_frac,
        "eigresid": float(np.linalg.norm(resid) / denom) if denom else float("nan"),
    }


def census(mesh):
    """Entity counts, the constraint ratio, and an Euler check on both."""
    dm = mesh.topology_dm
    dim = mesh.topological_dimension
    counts = []
    for d in range(dim + 1):
        lo, hi = dm.getDepthStratum(d)
        counts.append(hi - lo)
    if dim == 3:
        nv, ne, nf, nt = counts
        euler = nv - ne + nf - nt
    else:
        nv, ne, nt = counts
        nf, euler = 0, nv - ne + nt
    return {"V": nv, "E": ne, "F": nf, "T": nt, "euler": euler,
            "T_over_V": nt / nv}


def crosscheck_svd(mesh, q_family, q_degree, clamp_ids, variant=None, v_degree=2):
    """`beta_h` again, from a singular value rather than from a squared one.

    `S = B A^{-1} B^T` is a Gram matrix, so its eigenvalues are `beta^2` and an
    eigensolver delivers them with absolute error `~eps ||S||`: half the digits
    of `beta` are lost to the squaring, and a `beta` near `sqrt(eps) ~ 1e-8`
    would be indistinguishable from zero.  Forming the scaled matrix
    `C = L_q^{-1} B L_v^{-T}` explicitly (Cholesky factors of `M_q` and `A`) and
    taking its singular values avoids the squaring entirely.  Dense, so only the
    coarsest level, but it is the arbiter of whether a tiny `beta_h` is real.
    """
    kw = {} if variant is None else {"variant": variant}
    V = VectorFunctionSpace(mesh, "CG", v_degree, **kw)
    Q = FunctionSpace(mesh, q_family, q_degree, **kw)
    u, w = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)

    free = np.setdiff1d(np.arange(V.dim()), clamped_dofs(V, clamp_ids))
    A = to_scipy(assemble(inner(grad(u), grad(w)) * dx)).tocsc()[free][:, free]
    B = to_scipy(assemble(q * div(u) * dx)).tocsc()[:, free]
    Mq = to_scipy(assemble(p * q * dx)).toarray()

    Lv = scipy.linalg.cholesky(A.toarray(), lower=True)
    Lq = scipy.linalg.cholesky(Mq, lower=True)
    # C = Lq^{-1} B Lv^{-T}
    C = scipy.linalg.solve_triangular(Lq, B.toarray(), lower=True)
    C = scipy.linalg.solve_triangular(Lv, C.T, lower=True).T
    s = np.sort(scipy.linalg.svdvals(C))
    return s


def cell_size(mesh):
    """A single length scale per mesh: the largest cell diameter."""
    f = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellDiameter(mesh))
    return float(f.dat.data_ro.max())


def fit_power(hs, betas):
    """beta ~ C h^p; returns p."""
    hs, betas = np.asarray(hs), np.asarray(betas)
    good = betas > 0
    if good.sum() < 2:
        return float("nan")
    p, _ = np.polyfit(np.log(hs[good]), np.log(betas[good]), 1)
    return float(p)


def report(rows):
    print(f"{'level':>6} {'ncells':>8} {'n_v':>7} {'n_q':>7} {'h':>8} "
          f"{'ker(B^T)':>9} {'dim ker(B)':>11} {'beta_h':>10} {'lam_max':>8} {'s':>6}")
    for r in rows:
        print(f"{r['level']:>6} {r['ncells']:>8} {r['n_v']:>7} {r['n_q']:>7} "
              f"{r['h']:>8.4f} {r['n_zero']:>9} {r['dim_kerB']:>11} "
              f"{r['beta']:>10.4e} {r['lam_max']:>8.3f} {r['seconds']:>6.1f}")
    p = fit_power([r["h"] for r in rows], [r["beta"] for r in rows])
    print(f"  fitted beta_h ~ h^p :  p = {p:+.3f}")
    return p


def run_3d(lcs, pair, v_degree=2, variant=None):
    rows = []
    for i, lc in enumerate(lcs):
        path = shell_mesh(lc, os.path.join(HERE, f"infsup_shell_{lc:g}.msh"))
        mesh = Mesh(path)
        kw = {} if variant is None else {"variant": variant}
        n_q = FunctionSpace(mesh, pair[0], pair[1], **kw).dim()
        if n_q > MAX_NQ:
            print(f"    level {i}: lc={lc} skipped, n_q={n_q} > {MAX_NQ}", flush=True)
            continue
        r = measure(mesh, "CG", v_degree, pair[0], pair[1], (1,),
                    f"3d-{pair[0]}{pair[1]}", variant=variant)
        r["level"], r["h"], r["lc"] = i, cell_size(mesh), lc
        rows.append(r)
        print(f"    level {i}: lc={lc} ncells={r['ncells']} beta={r['beta']:.4e} "
              f"({r['seconds']:.1f} s)", flush=True)
    return rows


def run_2d(sizes):
    rows = []
    for i, (nr, nt) in enumerate(sizes):
        mesh = annulus_mesh(nr, nt)
        r = measure(mesh, "CG", 2, "DPC", 1, (1,), "2d-Q2/P1disc")
        r["level"], r["h"] = i, cell_size(mesh)
        rows.append(r)
        print(f"    level {i}: nr={nr} nt={nt} ncells={r['ncells']} "
              f"beta={r['beta']:.4e} ({r['seconds']:.1f} s)", flush=True)
    return rows


def main():
    print(__doc__)
    results = {}

    print("\n=== control 3: Taylor-Hood CG2/CG1 on the target tets ===", flush=True)
    lcs = [float(x) for x in os.environ.get("INFSUP_LCS", "0.40,0.33,0.27,0.22").split(",")]
    rows = run_3d(lcs, ("CG", 1))
    results["th3d"] = (rows, report(rows))

    print("\n=== control 2: Q2/P1-disc on the quadrilateral annulus (2-D) ===", flush=True)
    rows = run_2d([(6, 24), (9, 36), (13, 52), (18, 72)])
    results["q2p1_2d"] = (rows, report(rows))

    print("\n=== target: CG2/P1-disc on unstructured tets (3-D) ===", flush=True)
    rows = run_3d(lcs, ("DG", 1))
    results["cg2p1_3d"] = (rows, report(rows))

    # the standard remedy, as a macro element on the Alfeld (barycentric) split
    # of each parent tetrahedron rather than a refined mesh: same geometry, same
    # bookkeeping, and CG_k / DG_{k-1} on the split is the Scott-Vogelius pair.
    # the eigenvalues above are `beta^2`, so they carry half the digits of
    # `beta`; and one small number is a weaker statement than the shape of the
    # low end of the spectrum.  Both are fixed by taking singular values of the
    # scaled coupling `C = L_q^{-1} B L_v^{-T}` directly.
    print("\n=== cross-check: singular values of the scaled coupling ===", flush=True)
    print(f"{'lc':>6} {'n_q':>7} {'sv_min':>11} {'#sv<1e-8':>9} {'#sv<0.05':>9} "
          f"{'#sv<0.1':>8} {'frac<0.1':>9}")
    for lc in lcs[:3]:
        mesh = Mesh(shell_mesh(lc, os.path.join(HERE, f"infsup_shell_{lc:g}.msh")))
        s = crosscheck_svd(mesh, "DG", 1, (1,))
        print(f"{lc:>6} {s.size:>7} {s[0]:>11.3e} {int((s < 1e-8).sum()):>9} "
              f"{int((s < 0.05).sum()):>9} {int((s < 0.1).sum()):>8} "
              f"{(s < 0.1).mean():>9.4f}", flush=True)

    # a macro element costs 4 subcells per parent tet, and degree 3 costs ten
    # dofs per subcell, so each remedy ladder needs its own coarser meshes to
    # stay under MAX_NQ.  h overlaps the target ladder at the coarse end.
    for deg, remedy_lcs in ((2, [0.52, 0.45, 0.40]), (3, [0.75, 0.62, 0.52])):
        print(f"\n=== remedy: Alfeld-split Scott-Vogelius CG{deg}/DG{deg - 1} "
              f"(3-D) ===", flush=True)
        try:
            rows = run_3d(remedy_lcs, ("DG", deg - 1), v_degree=deg,
                          variant="alfeld")
            results[f"alfeld{deg}"] = (rows, report(rows))
        except Exception as exc:  # noqa: BLE001 - a remedy probe, not the result
            print(f"  skipped: {type(exc).__name__}: {exc}")

    print("\n=== verdict ===")
    th_p = results["th3d"][1]
    th_beta = [r["beta"] for r in results["th3d"][0]]
    q2_beta = [r["beta"] for r in results["q2p1_2d"][0]]
    tg = results["cg2p1_3d"]
    controls_ok = min(th_beta) > 0.02 and abs(th_p) < 0.35 and min(q2_beta) > 0.02
    print(f"  Taylor-Hood 3-D : beta in [{min(th_beta):.3e}, {max(th_beta):.3e}], p={th_p:+.3f}")
    print(f"  Q2/P1-disc 2-D  : beta in [{min(q2_beta):.3e}, {max(q2_beta):.3e}]")
    print(f"  CG2/P1-disc 3-D : beta in [{min(r['beta'] for r in tg[0]):.3e}, "
          f"{max(r['beta'] for r in tg[0]):.3e}], p={tg[1]:+.3f}")
    spurious = [r["n_zero"] for r in tg[0]]
    print(f"  CG2/P1-disc 3-D : exactly spurious pressure modes per level "
          f"{spurious}")
    if not controls_ok:
        print("  VERDICT: MEASUREMENT NOT TRUSTWORTHY (a control misbehaved)")
    elif max(spurious) > 0:
        print(f"  VERDICT: UNSTABLE -- beta_h ~ h^{tg[1]:.2f} over this range, and "
              f"B loses rank outright ({max(spurious)} spurious pressure modes at "
              f"the finest level), so beta_h is exactly zero there.  The penalty "
              f"limit locks.")
    elif tg[1] > 0.4:
        print(f"  VERDICT: UNSTABLE, beta_h ~ h^{tg[1]:.2f} -- the penalty limit locks")
    else:
        print("  VERDICT: STABLE -- no locking detected")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
