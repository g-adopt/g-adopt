"""Locking or conditioning? The volumetric-penalty probe, in 2-D.

`HANDOFF-SPADA-BENCHMARK.md` §8 plans the whole incompressible benchmark around
one assumption: that G-ADOPT's penalty incompressibility
(`stress = bulk_shear_ratio * bulk_modulus * div(u) I + deviatoric`, CG2
displacement, **no pressure partner**) fails only through *conditioning*, so the
incompressible answer is reachable by fitting a quantity against `1/ratio` and
extrapolating.  The alternative is *locking*, which does not present as a solver
failure at all: the residual converges, the linear solve reports success, and
the **answer** stops converging under mesh refinement.  Three phases sit on the
distinction.

## The verdict lives on the h axis at fixed high nu, and nowhere else

**A straight 1/ratio ladder is not evidence against locking.**  The penalty
solution converges at O(1/lambda) to the constrained minimiser over
`{v in V_h : div v = 0}` whether or not that subspace is impoverished, and an
impoverished one extrapolates cleanly and confidently to an over-stiff *wrong*
limit.  Q1 is reported because Phase C needs the fit; it is not the test.

**A clean -1 slope in `||div u||/||grad u||` is not evidence either.**  A fully
locked element drives `div u` to zero beautifully — that is what locking *is*.
It is reported only to confirm the penalty is applied at all.

The test is Q2: the observed h-convergence order of a functional at fixed nu,
and whether it degrades as nu -> 0.5.

## Three things that will silently produce a meaningless null result

1. **A low-degree load.**  The locking error is proportional to `|u|_{k+1}`, and
   a degree-2 field is nearly *in* the CG2 space at any usable resolution, so
   the whole discretisation error can sit near roundoff.  Measured: at
   `cos(2 phi)` the L4 error at nu = 0.28 was 2.8e-08 relative and every order
   read >= 2.9.  At `cos(16 phi)` the same cell is at 1.5e-05 and the orders
   separate properly.  **`LOAD_DEGREE = 16`.**

2. **The nominal ratio is not the operator's ratio.**  `InternalVariableSolver`
   eliminates the backward-Euler internal variable, giving an incremental shear
   modulus `mu0 tau/(tau + dt)` while the volumetric stiffness is untouched, so

       R_eff = bulk_shear_ratio * (1 + dt/tau)

   At `dt = 100 tau` a nominal nu = 0.28 is really nu_eff = 0.4975.  Measured
   here to 2.1e-10 by `--check-dt`.  Consequences: this probe is the *elastic
   snapshot* at `dt/tau = 1e-8`; and near the fluid limit the displacement is
   set by buoyancy balance and goes locking-blind, which is the cleanest
   possible false negative.  `R_eff` is printed for every run.

3. **A ladder built by uniform refinement, refined too far.**  See
   `mesh_ladder`: snapping cannot be made hierarchical, so the ladder has a hard
   depth limit past which boundary cells invert while every circumference stays
   perfect to 1e-13.  `check_geometry_ladder` turns that into a hard stop.

## The problem, and why this one

Mechanics only, no gravity coupling.  **Locking is a property of the (u,u)
block alone**: the volumetric penalty appears in no other block, the potential
rows carry no `bulk_shear_ratio`, and the coupling blocks are O(Lambda)
regardless.  Turning the coupling off *sharpens* attribution rather than
weakening it.  `g = 0` switches off the Al-Attar prestress/buoyancy pair, so
road map §2.5's growing mode plays no part.

Elastic, one solve at `dt = 1e-8` tau — the operator §11.1's elastic snapshot
and Phase C's ladder both run on.

Clamped at Rc by default (`u = 0`, strong, contributing no weak term at all:
`stokes_integrators.py:326` puts it in `strong_bcs` only).  `--cmb un0` runs the
production condition instead, whose Nitsche pair *does* carry
`bulk_modulus * bulk_shear_ratio` (`momentum_equation.py:114-124`) and is
therefore a confounder.  **Measured: at load degree 16 the two agree in J to
1.6e-06**, because a degree-16 surface field has decayed by `(Rc/Re)^16 ~ 1e-04`
before it reaches Rc.  So this probe cannot see that confounder — the load
degree the locking test requires and the low degree the CMB confounder needs are
incompatible, and the CMB question needs its own low-degree run.

Load: `-sigma_0 cos(16 phi)` normal stress at Re.  J is its amplitude in `u_r`
on the loaded circle, `(2/L) int_Re u_r cos(16 phi) ds` — the same information
as §10's colatitude sampler, linear in the solution, and free of the
interpolation error a point evaluation on an unstructured curved mesh carries.

## Mesh, and three families rather than one

Unstructured Frontal-Delaunay triangles, never the transfinite quads of
`generate_selfgrav_annulus.py`: the 3-D target is tets and Q2 on structured
quads is markedly less locking-prone than P2 on simplices, so a quad probe
would answer the wrong question optimistically.  Boundary vertices snapped onto
the exact circles, then `curve_mesh`; geometry error 8.4e-07 falling at order 4.

Because the answer turned out to depend on the family, all three are available:

* default — `MeshHierarchy`, nested, exact factor two.
* `--distort a` — the same, with interior vertices randomly displaced by `a h`
  at each level.  Uniform refinement is not topologically neutral, so this asks
  whether the answer is a property of the element or of the refinement.
* `--independent` — a fresh gmsh mesh per level, no inherited topology.

## Caveats, stated rather than buried

`deviatoric_strain` subtracts `tr(e)/3` in every dimension, so in 2-D the
effective plane bulk modulus is `K_eff + mu/3`.  The nu column is a *label* for
the ratio, computed from §8's 3-D formula, not the plane-strain Poisson ratio of
this problem.  The controlled variable is `bulk_shear_ratio` throughout.

2-D P2 triangles are a proxy for 3-D P2 tets, and the evidence is asymmetric: a
2-D failure transfers to 3-D near-certainly, a 2-D pass transfers weakly.  The
definitive instrument is the discrete inf-sup constant on tets.

Usage:

    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking.py --check-dt
    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking.py --lc 0.125 --levels 5
    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking.py --independent
    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking.py --distort 0.25
    PYTHONPATH=$(pwd) python demos/gravity/spikes/gate_locking.py --cmb un0
"""
import argparse
import os
import time

import gadopt  # noqa: F401  BEFORE firedrake; see demos/gravity/CLAUDE.md
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402
import gmsh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RC, RE = 1.2037, 2.2037
CURVE_RE, CURVE_RC = 2, 3
CELL_MANTLE = 101

SIGMA_0 = 1.0e-3          # load amplitude; everything is linear in it
LOAD_DEGREE = 16          # cos(m phi) of the surface normal stress; see below
MU0 = 1.0                 # shear modulus
TAU = 1.0                 # Maxwell time (viscosity / shear modulus)
DT = 1.0e-8               # Maxwell times: elastic to eight digits

# nu -> K/mu = 2(1+nu)/(3(1-2nu)), the ladder of HANDOFF §8.
NU_LADDER = [0.28, 0.45, 0.49, 0.499, 0.4999]


def bulk_shear_ratio(nu):
    return 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))


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
    "ksp_rtol": 1.0e-10,
    "ksp_max_it": 5000,
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

def write_annulus(path, lc):
    """Unstructured Delaunay annulus Rc -> Re, tags 3 (Rc) and 2 (Re)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("locking_annulus")
    geo = gmsh.model.geo
    centre = geo.addPoint(0, 0, 0)
    quadrant = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    rings = []
    for radius in (RC, RE):
        pts = [geo.addPoint(radius * cx, radius * cy, 0, lc)
               for cx, cy in quadrant]
        arcs = [geo.addCircleArc(pts[j], centre, pts[(j + 1) % 4])
                for j in range(4)]
        rings.append((geo.addCurveLoop(arcs), arcs))
    surface = geo.addPlaneSurface([rings[1][0], rings[0][0]])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(1, rings[0][1], CURVE_RC, name="Rc")
    gmsh.model.addPhysicalGroup(1, rings[1][1], CURVE_RE, name="Re")
    gmsh.model.addPhysicalGroup(2, [surface], CELL_MANTLE, name="mantle")
    gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    return path


def snap_boundary(mesh):
    """Push boundary vertices radially onto the exact circles, in place.

    `MeshHierarchy` bisects the parent's straight boundary chords, so without
    this every level shares the coarse polygon and the geometry error never
    falls.  With it the domain is a genuine refinement sequence.
    """
    Vc = mesh.coordinates.function_space()
    nodes = DirichletBC(Vc, Constant((0.0, 0.0)), (CURVE_RE, CURVE_RC)).nodes
    X = mesh.coordinates.dat.data
    r = np.hypot(X[nodes, 0], X[nodes, 1])
    target = np.where(np.abs(r - RC) < np.abs(r - RE), RC, RE)
    X[nodes, 0] *= target / r
    X[nodes, 1] *= target / r
    return mesh


def curve_mesh(linear_mesh):
    """P2 coordinates with edge midpoints pushed radially onto the circle.

    Verbatim from `validate_selfgrav_annulus.py`.
    """
    X = SpatialCoordinate(linear_mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        (r_p1 / r) * X)
    return Mesh(X_p2)


def distort(mesh, amplitude, seed):
    """Randomly displace interior vertices by `amplitude` x the local spacing.

    `MeshHierarchy` splits each triangle into four similar ones, so a refined
    unstructured mesh becomes locally *more* regular at every level, and local
    regularity is exactly what a locking test should not be handed for free.
    This breaks it.  Boundary vertices are excluded so the geometry stays exact.
    The perturbation is proportional to the level's own mean spacing, so the
    family is uniform in shape and the sequence is still a refinement sequence —
    but it is an independent random field per level, so the error is no longer
    smooth in h and the observed orders are correspondingly noisier.  That is
    the price, and it is reported rather than hidden.
    """
    Vc = mesh.coordinates.function_space()
    boundary = set(DirichletBC(Vc, Constant((0.0, 0.0)),
                               (CURVE_RE, CURVE_RC)).nodes.tolist())
    X = mesh.coordinates.dat.data
    interior = np.array([i for i in range(X.shape[0]) if i not in boundary])
    h = np.sqrt(np.pi * (RE ** 2 - RC ** 2) / mesh.num_cells())
    rng = np.random.default_rng(seed)
    X[interior] += amplitude * h * rng.uniform(-1.0, 1.0, size=(len(interior), 2))
    return mesh


def mesh_ladder(lc, levels, distortion=0.0):
    """`levels` nested meshes, coarsest first, each snapped and curved.

    **The ladder has a hard depth limit and it fails silently at the bottom.**
    `MeshHierarchy` refines the DMPlex, and the DMPlex does not see mutations
    to `mesh.coordinates` — measured directly: snap level 1's boundary onto
    the circle (its minimum boundary radius goes 1.197904 -> 1.203700), refine
    that snapped mesh, and the child's minimum boundary radius is 1.197904
    again, not the 1.202250 a propagated snap would give.  So snapping cannot
    be made hierarchical, every level descends from the *base* polygon, and
    every level's boundary still carries first-generation bisection vertices
    sitting a fixed `h_b^2/(8R)` inside the circle, where `h_b` is the base
    mesh's boundary spacing.  Snapping displaces those by that fixed amount on
    a mesh whose own spacing is `h_b/2^(levels-1)`, so once

        h_b 2^(levels-1) / (8 R)  ~  1

    the boundary elements inevitably fold.  Measured: the curved area error
    fell 8.4e-07 -> 1.3e-11 at order 4 across five levels and then jumped to
    2.3e-05 on the sixth, **with both circumferences still perfect to 1e-13**.
    The boundary was exactly right and the cells behind it were inverted,
    which is why nothing about the sixth level looked wrong except this one
    number.  At `lc = 0.25`, `R = 1.2`, the ratio is 0.42 at five levels and
    0.83 at six — hence five.

    `check_geometry_ladder` below turns that into an assertion rather than a
    thing to remember.  To reach finer cells, start the ladder finer (`--lc
    0.125`) rather than adding a level.
    """
    path = write_annulus(os.path.join(HERE, f"locking_{lc}.msh"), lc)
    hierarchy = MeshHierarchy(Mesh(path), levels - 1)
    out = []
    for k, lvl in enumerate(hierarchy):
        if distortion > 0.0:
            distort(lvl, distortion, seed=1234 + k)
        m = curve_mesh(snap_boundary(lvl))
        m.cartesian = False
        out.append(m)
    return out


def independent_ladder(lc, levels):
    """`levels` *independently meshed* annuli at lc, lc/2, lc/4, ...

    Not a refinement sequence: each is a fresh Frontal-Delaunay mesh, so no
    level inherits another's topology.  This exists because uniform (red)
    refinement is not topologically neutral — it places a new vertex at the
    midpoint of every edge, and at such a vertex two of the incident edges are
    exactly collinear, which is the *singular vertex* configuration that
    degrades the Scott-Vogelius inf-sup constant for P2 with a discontinuous
    pressure partner.  A `MeshHierarchy` therefore manufactures singular
    vertices at every level however generic its base mesh was, and a locking
    measurement taken on one is measuring that and not the element.

    gmsh places boundary nodes exactly on the arcs, so no snapping is needed
    and `curve_mesh` alone gives the O(h^4) geometry.  The price is that the
    levels are not nested, so the error sequence is not smooth in h and the
    observed orders carry mesh-to-mesh noise.
    """
    out = []
    for k in range(levels):
        h = lc / 2 ** k
        path = write_annulus(os.path.join(HERE, f"locking_ind_{h:g}.msh"), h)
        m = curve_mesh(Mesh(path))
        m.cartesian = False
        out.append(m)
    return out


def check_geometry_ladder(meshes):
    """The geometry error must keep falling.  Returns the offending levels.

    A curved level should improve on its parent by ~16x (order 4).  Anything
    that does not improve at all has inverted boundary cells, per
    `mesh_ladder`'s docstring, and every number computed on it is worthless.
    """
    errs = [geometry_error(m) for m in meshes]
    return [k for k in range(1, len(errs)) if errs[k] > 0.5 * errs[k - 1]]


def geometry_error(mesh):
    area = assemble(Constant(1.0) * dx(domain=mesh))
    exact = np.pi * (RE ** 2 - RC ** 2)
    return abs(area - exact) / exact


# --------------------------------------------------------------------------
# one solve
# --------------------------------------------------------------------------

def build(mesh, ratio, degree=LOAD_DEGREE, dt=DT, cmb="clamped"):
    """Displacement, internal variable, approximation and boundary conditions."""
    V = VectorFunctionSpace(mesh, "CG", 2)
    S = TensorFunctionSpace(mesh, "DG", 1)
    u = Function(V, name="displacement")
    m = Function(S, name="internal variable")

    # g = 0 switches off the Al-Attar prestress/buoyancy pair entirely, so the
    # reference state plays no part.  bulk_modulus = 1 with the ratio carrying
    # the whole penalty keeps K_eff = ratio exactly.
    approximation = MaxwellApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=MU0, viscosity=MU0 * TAU,
        g=0.0, B_mu=1.0, bulk_shear_ratio=ratio)

    X = SpatialCoordinate(mesh)
    phi = atan2(X[1], X[0])
    load = -SIGMA_0 * cos(degree * phi)

    if cmb == "clamped":
        # Strong `u` goes into `strong_bcs` only and is NOT copied into
        # `weak_bcs` (stokes_integrators.py:326), so it contributes no weak
        # term at all - in particular none carrying `bulk_modulus *
        # bulk_shear_ratio`.
        inner_bc = {"u": Constant((0.0, 0.0))}
        nullspace = None
    elif cmb == "un0":
        # The production condition, and a deliberate confounder: the `un`
        # branch of `viscosity_term` (momentum_equation.py:114-124) adds a
        # Nitsche pair whose coefficient IS `bulk_modulus * bulk_shear_ratio`,
        # so climbing the ladder stiffens a boundary constraint as well as the
        # interior one.  Run to measure how much that matters, not because it
        # is the cleaner experiment.
        inner_bc = {"un": 0.0}
        nullspace = rigid_body_modes(V, rotational=True)
    else:                                              # pragma: no cover
        raise ValueError(f"unknown cmb condition {cmb!r}")

    bcs = {CURVE_RC: inner_bc, CURVE_RE: {"normal_stress": load}}
    return u, m, approximation, bcs, nullspace


def effective_ratio(ratio, dt):
    """The ratio the *tangent operator* actually sees.

    `InternalVariableSolver` eliminates the backward-Euler internal variable
    into the stress:

        m_new = (m_old + (dt/tau) d) / (1 + dt/tau)
        dev_stress = 2 mu0 d - 2 mu m_new
                   = 2 mu0 d / (1 + dt/tau)  -  (explicit m_old term)

    so the *incremental* shear modulus is `mu0 tau/(tau + dt)` while the
    volumetric stiffness `bulk_shear_ratio * bulk_modulus` is untouched by dt.
    The operator's bulk/shear ratio is therefore

        R_eff = R (1 + dt/tau)

    which is why this probe is run as the elastic snapshot at dt/tau = 1e-8 and
    not at the fluid limit.  Checked numerically by `check_effective_ratio`.
    """
    return ratio * (1.0 + dt / TAU)


def residual_reduction(solver, u, m, m_before):
    """||F(u)|| / ||F(0)||, with the strong-bc rows removed from both.

    `snes_type: ksponly` reports success unconditionally under `preonly` LU, so
    the only honest convergence statement is this one, assembled from the
    solver's own residual form.  `InternalVariableSolver.solve` overwrites the
    internal variable *after* the solve, so `m` is restored to what the solve
    actually saw first; without that the measured reduction floors at
    `dt/tau ~ 1e-8` and looks like a failed solve.
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


def diagnostics(mesh, u, approximation, ratio, degree=LOAD_DEGREE):
    """The functional, the norms, and the divergence measure."""
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    phi = atan2(X[1], X[0])
    n_hat = X / r
    u_r = dot(u, n_hat)

    length = assemble(Constant(1.0) * ds(CURVE_RE, domain=mesh))
    J = 2.0 * assemble(u_r * cos(degree * phi) * ds(CURVE_RE)) / length

    u_l2 = sqrt(assemble(dot(u, u) * dx))
    dev = approximation.deviatoric_strain(u)
    energy = sqrt(assemble(2.0 * MU0 * inner(dev, dev) * dx))

    div_l2 = sqrt(assemble(div(u) ** 2 * dx))
    grad_l2 = sqrt(assemble(inner(grad(u), grad(u)) * dx))

    # A pointwise measure as well as an integral one: nodal values of a DG1
    # interpolant, which for a CG2 field is exact for both quantities.
    D = FunctionSpace(mesh, "DG", 1)
    div_max = np.abs(Function(D).interpolate(abs(div(u))).dat.data_ro).max()
    grad_max = np.abs(Function(D).interpolate(
        sqrt(inner(grad(u), grad(u)))).dat.data_ro).max()

    return {
        "J": J,
        "u_l2": u_l2,
        "energy": energy,
        "div_over_grad_l2": div_l2 / grad_l2,
        "div_over_grad_max": div_max / grad_max,
    }


def solve_cell(mesh, ratio, iterative=True, degree=LOAD_DEGREE, dt=DT,
               cmb="clamped"):
    """One (h, ratio) cell: the LU answer, and the Krylov cost beside it."""
    u, m, approximation, bcs, nullspace = build(mesh, ratio, degree, dt, cmb)

    t0 = time.perf_counter()
    solver = InternalVariableSolver(
        u, approximation, dt=dt, internal_variables=[m], bcs=bcs,
        solver_parameters=DIRECT, nullspace=nullspace,
        transpose_nullspace=nullspace)
    m_before = m.copy(deepcopy=True)
    solver.solve()
    t_lu = time.perf_counter() - t0

    out = diagnostics(mesh, u, approximation, ratio, degree)
    out["residual_reduction"] = residual_reduction(solver, u, m, m_before)
    out["t_lu"] = t_lu
    out["dofs"] = u.function_space().dim()
    # §12.0: the class switch at bulk_shear_ratio > 10 silently disables the
    # volume prestress term.  Print what was actually instantiated.
    out["approx_class"] = type(approximation).__name__
    out["dt_over_tau"] = dt / TAU
    out["ratio_eff"] = effective_ratio(ratio, dt)

    out["its"] = None
    out["t_cg"] = None
    out["cg_reason"] = None
    out["cg_mismatch"] = None
    if iterative:
        u2, m2, approx2, bcs2, ns2 = build(mesh, ratio, degree, dt, cmb)
        t0 = time.perf_counter()
        s2 = InternalVariableSolver(
            u2, approx2, dt=dt, internal_variables=[m2], bcs=bcs2,
            solver_parameters=ITERATIVE, nullspace=ns2,
            transpose_nullspace=ns2, near_nullspace=ns2)
        try:
            s2.solve()
            reason = s2.solver.snes.ksp.getConvergedReason()
            its = s2.solver.snes.ksp.getIterationNumber()
        except Exception as exc:                      # divergence is a datum
            reason, its = f"raised {type(exc).__name__}", -1
        out["t_cg"] = time.perf_counter() - t0
        out["its"] = its
        out["cg_reason"] = reason
        if isinstance(reason, int) and reason > 0:
            out["cg_mismatch"] = (
                sqrt(assemble(dot(u - u2, u - u2) * dx)) / max(out["u_l2"], 1e-300))
    return out


def check_effective_ratio(mesh, degree):
    """Measure `R_eff = R (1 + dt/tau)` rather than assert it.

    Two runs with the same *effective* ratio and very different `dt` must give
    the same J.  If they do, the formula is right, and the corollary — that a
    Phase C run at large `dt` is at a far higher effective ratio than its
    nominal nu advertises — is established rather than argued.

    The incremental shear modulus is also `mu0/(1 + dt/tau)`, so with a fixed
    load `J` itself scales by `(1 + dt/tau)`.  The invariant to compare is
    `J / (1 + dt/tau)`.
    """
    rows = []
    for nominal, dt in ((49.6667, 1e-8), (24.83335, 1.0), (4.96667, 9.0),
                        (49.6667, 1.0), (49.6667, 9.0)):
        r = solve_cell(mesh, nominal, iterative=False, degree=degree, dt=dt)
        scale = 1.0 + dt / TAU
        rows.append((nominal, dt, r["ratio_eff"], r["J"], r["J"] / scale))
    return rows


# --------------------------------------------------------------------------
# the reduced-quadrature remedy (only run when locking is indicated)
# --------------------------------------------------------------------------

def solve_reduced(mesh, ratio, vol_degree, degree=LOAD_DEGREE):
    """Hand-written residual with the volumetric term under-integrated.

    `inner(nabla_grad(v), K_eff div(u) I) = K_eff div(u) div(v)`, so selective
    reduced integration is a quadrature degree on that one term.  §8 asks for
    exactly this and explicitly *not* for a new function space.  At
    `vol_degree = None` the form must reproduce `InternalVariableSolver`'s
    answer to roundoff, which is how the hand-written form is validated before
    it is trusted.
    """
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = Function(V)
    v = TestFunction(V)

    approximation = MaxwellApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=MU0, viscosity=1.0,
        g=0.0, B_mu=1.0, bulk_shear_ratio=ratio)

    X = SpatialCoordinate(mesh)
    phi = atan2(X[1], X[0])
    load = -SIGMA_0 * cos(degree * phi)
    n = FacetNormal(mesh)

    # The internal variable at dt = 1e-8 is 1e-8 of the strain; drop it, and
    # the validation against the solver path below measures what that costs.
    dev_stress = 2.0 * MU0 * approximation.deviatoric_strain(u)
    dx_vol = dx if vol_degree is None else dx(degree=vol_degree)

    F = inner(nabla_grad(v), dev_stress) * dx
    F += ratio * 1.0 * div(u) * div(v) * dx_vol
    F += dot(v, load * n) * ds(CURVE_RE)

    bc = DirichletBC(V, Constant((0.0, 0.0)), CURVE_RC)
    solve(F == 0, u, bcs=[bc], solver_parameters=DIRECT)

    return diagnostics(mesh, u, approximation, ratio, degree)


# --------------------------------------------------------------------------
# fits and orders
# --------------------------------------------------------------------------

def extrapolate(ratios, values):
    """Least-squares line in 1/ratio; returns (intercept, slope, max residual)."""
    x = 1.0 / np.asarray(ratios, dtype=float)
    y = np.asarray(values, dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coeffs
    return coeffs[0], coeffs[1], np.abs(resid).max()


def orders(values):
    """Observed order from successive differences on a factor-two ladder."""
    v = np.asarray(values, dtype=float)
    d = np.abs(np.diff(v))
    out = []
    for a, b in zip(d[:-1], d[1:]):
        out.append(np.log2(a / b) if b > 0 else np.inf)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lc", type=float, default=0.25,
                   help="gmsh target size of the coarsest level")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--load-degree", type=int, default=LOAD_DEGREE,
                   help="azimuthal degree n of the surface load; must be high")
    p.add_argument("--cmb", choices=("clamped", "un0"), default="clamped")
    p.add_argument("--independent", action="store_true",
                   help="independently meshed levels instead of MeshHierarchy")
    p.add_argument("--distort", type=float, default=0.0,
                   help="random interior-vertex perturbation, in units of h")
    p.add_argument("--nu", type=str, default="",
                   help="comma-separated subset of the nu ladder")
    p.add_argument("--no-iterative", action="store_true",
                   help="skip the CG/GAMG pass, keeping only the LU answers")
    p.add_argument("--reduced-quadrature", action="store_true",
                   help="hand-written SRI cross-check; see the docstring")
    p.add_argument("--check-dt", action="store_true",
                   help="measure R_eff = R (1 + dt/tau) instead of asserting it")
    args = p.parse_args()

    nus = ([float(x) for x in args.nu.split(",")] if args.nu else NU_LADDER)
    ratios = [bulk_shear_ratio(nu) for nu in nus]
    n = args.load_degree

    print("Locking probe for the volumetric penalty - HANDOFF-SPADA §8")
    print("  THE VERDICT IS READ OFF THE h AXIS AT FIXED HIGH nu, AND NOWHERE")
    print("  ELSE.  A straight 1/ratio ladder is not evidence against locking:")
    print("  the penalty solution converges at O(1/lambda) to the constrained")
    print("  minimiser over {v in V_h : div v = 0} whether or not that subspace")
    print("  is impoverished, and an impoverished one extrapolates cleanly and")
    print("  confidently to an over-stiff WRONG limit.  Q1 below is reported")
    print("  because Phase C needs the fit; it is not the locking test.")
    print()
    print("  configuration, reported unconditionally")
    print(f"    element            P2 (CG2) vector on triangles, no pressure "
          "partner")
    print(f"    domain             annulus {RC} -> {RE}, single mesh, no "
          "submesh")
    print(f"    curving            curve_mesh applied to every level (there is")
    print("                       only one mesh here; the parent/submesh pair")
    print("                       of the coupled solver does not arise)")
    print(f"    load               -{SIGMA_0} cos({n} phi) normal stress at Re")
    print(f"    CMB condition      {args.cmb}")
    print(f"    distortion         {args.distort} h")
    print(f"    ladder             "
          f"{'independent gmsh meshes' if args.independent else 'MeshHierarchy (uniform red refinement)'}")
    print(f"    g                  0 (Al-Attar prestress/buoyancy pair off)")
    print(f"    dt/tau             {DT / TAU:.1e}  -> R_eff/R = "
          f"{1.0 + DT / TAU:.10f}")
    print()
    print("  expected before the run:")
    print("    at nu = 0.28 the observed h-order of J is the CG2 functional rate")
    print("    LOCKING would show as that order degrading with nu, and as the")
    print("      error-constant ratio C(0.499)/C(0.28) approaching (1-2nu)^-1 ~ 220")
    print("    CONDITIONING-ONLY would show as the order preserved and the")
    print("      constant bounded at O(1)-O(10)")
    print("    ||div u||_L2/||grad u||_L2 falls like 1/R_eff, slope -1 on log-log")
    print("      -- a sanity check that the penalty is applied at all, and NOT")
    print("      evidence against locking: a fully locked element drives div u")
    print("      to zero beautifully, which is what locking IS")
    print("    CG+GAMG iteration slope on log-log against R_eff in [0.5, 1.0]")
    print("    OPEN: the h-order at R = 4999.7")
    print()
    print("  ladder      nu     K/mu (nominal)    R_eff")
    for nu, ratio in zip(nus, ratios):
        print(f"            {nu:7.4f}  {ratio:12.4f}  "
              f"{effective_ratio(ratio, DT):12.4f}")
    print()

    if args.independent:
        meshes = independent_ladder(args.lc, args.levels)
    else:
        meshes = mesh_ladder(args.lc, args.levels, args.distort)
    print("  mesh ladder")
    for k, m in enumerate(meshes):
        # Surface cells per load wavelength: the number that says whether a
        # level can carry cos(n phi) at all.
        n_surf = len(m.exterior_facets.subset(CURVE_RE).indices)
        print(f"    L{k}: {m.num_cells():8d} cells, "
              f"CG2 vector dofs {VectorFunctionSpace(m, 'CG', 2).dim():8d}, "
              f"{n_surf:5d} facets on Re = {n_surf / n:6.1f} per wavelength, "
              f"area err {geometry_error(m):.2e}")
    bad = check_geometry_ladder(meshes)
    if bad:
        print(f"    *** LEVELS {bad} HAVE INVERTED BOUNDARY CELLS - their")
        print("    *** geometry error stopped falling.  See mesh_ladder's")
        print("    *** docstring: start finer with --lc, do not add levels.")
        raise SystemExit(1)
    print()

    if args.check_dt:
        print("  R_eff = R (1 + dt/tau), measured not asserted.  Rows with the")
        print("  same R_eff must agree in J/(1 + dt/tau).")
        rows = check_effective_ratio(meshes[min(1, len(meshes) - 1)], n)
        print("      R_nominal      dt/tau        R_eff             J        "
              "  J/(1+dt/tau)")
        for nominal, dt, reff, J, Jn in rows:
            print(f"    {nominal:11.5f}  {dt / TAU:9.1e}  {reff:12.4f}  "
                  f"{J:+.9e}  {Jn:+.9e}")
        base = rows[0][4]
        worst = max(abs(r[4] - base) / abs(base) for r in rows[:3])
        print(f"    the three same-R_eff rows agree to {worst:.2e} relative")
        print("    the last two rows share R_nominal and differ in R_eff by")
        print("    2x and 10x: nominal nu is NOT what the operator sees at")
        print("    large dt, which is why this probe is the elastic snapshot")
        print()

    table = {}
    for k, mesh in enumerate(meshes):
        for nu, ratio in zip(nus, ratios):
            r = solve_cell(mesh, ratio, iterative=not args.no_iterative,
                           degree=n, dt=DT, cmb=args.cmb)
            table[(k, nu)] = r
            flag = "" if r["residual_reduction"] < 1e-8 else "  <-- RESIDUAL"
            its = "-" if r["its"] is None else str(r["its"])
            print(f"    L{k} nu={nu:<7.4f} R={ratio:9.4f} "
                  f"R_eff={r['ratio_eff']:9.4f}  "
                  f"J={r['J']:+.9e}  |u|={r['u_l2']:.6e}  "
                  f"E={r['energy']:.6e}  "
                  f"div/grad L2={r['div_over_grad_l2']:.3e} "
                  f"max={r['div_over_grad_max']:.3e}  "
                  f"its={its:>5s}  t_lu={r['t_lu']:6.2f}s  "
                  f"res={r['residual_reduction']:.1e}  "
                  f"{r['approx_class']}{flag}")
        print()

    classes = {r["approx_class"] for r in table.values()}
    print(f"  approximation classes instantiated across the whole grid: "
          f"{sorted(classes)}")
    print("  (§12.0: a switch to QuasiCompressible... would show as a step in J")
    print("   between the 9.67 and 49.7 rungs.  One class means no switch.)")
    print()

    # --- THE LOCKING TEST: the h axis at fixed nu -------------------------
    print("  Q2  THE LOCKING TEST - observed h-convergence order of J at fixed nu")
    print("      a rate that degrades as nu -> 0.5 IS locking - UNLESS h has")
    print("      not yet resolved the incompressibility boundary layer at the")
    print("      Dirichlet boundary, whose width is sqrt(mu/K_eff).  h and that")
    print("      width are printed together below; a rate that recovers once")
    print("      h drops below the layer was never locking.")
    hs = [np.sqrt(np.pi * (RE ** 2 - RC ** 2) / m.num_cells()) for m in meshes]
    print("      h        " + "  ".join(f"{v:9.2e}" for v in hs))
    for nu, ratio in zip(nus, ratios):
        layer = 1.0 / np.sqrt(effective_ratio(ratio, DT))
        print(f"      nu={nu:<7.4f} layer sqrt(mu/K_eff) = {layer:.2e}, "
              f"resolved from L{next((k for k, v in enumerate(hs) if v < layer), None)}")
    for nu in nus:
        vals = [table[(k, nu)]["J"] for k in range(len(meshes))]
        ords = orders(vals)
        txt = "  ".join(f"{o:5.2f}" for o in ords)
        print(f"    nu={nu:<7.4f}  J = " +
              "  ".join(f"{v:+.6e}" for v in vals) + f"   orders {txt}")

    print()
    print("  Q2b same, for the L2 norm of u (so the verdict is not one number)")
    for nu in nus:
        vals = [table[(k, nu)]["u_l2"] for k in range(len(meshes))]
        print(f"    nu={nu:<7.4f}  orders " +
              "  ".join(f"{o:5.2f}" for o in orders(vals)))

    print()
    print("  Q2c |J(h) - J(h_finest)|/|J(h_finest)| at fixed nu, and the")
    print("      error-constant ratio C(nu)/C(0.28) at each h.  Bounded O(1)-O(10)")
    print("      means no locking; approaching (1-2nu)^-1 means locking.")
    last = len(meshes) - 1
    errs = {}
    for nu in nus:
        ref = table[(last, nu)]["J"]
        errs[nu] = [abs(table[(k, nu)]["J"] - ref) / abs(ref)
                    for k in range(last)]
        print(f"    nu={nu:<7.4f}  " + "  ".join(f"{e:.3e}" for e in errs[nu]))
    if nus[0] in errs:
        base = errs[nus[0]]
        print("    C(nu)/C(nu_min) at each h:")
        for nu in nus:
            print(f"    nu={nu:<7.4f}  " +
                  "  ".join(f"{a / b:9.1f}" for a, b in zip(errs[nu], base)) +
                  f"   [(1-2nu)^-1 = {1.0 / max(1.0 - 2 * nu, 1e-12):.1f}, "
                  f"relative to nu_min: "
                  f"{(1.0 / max(1.0 - 2 * nu, 1e-12)) / (1.0 / (1.0 - 2 * nus[0])):.1f}]")

    # --- Q1, reported for Phase C, NOT as the locking test ----------------
    print()
    print("  Q1  fit J against 1/R_eff at fixed h (for Phase C's extrapolation")
    print("      only - see the header: this is not evidence about locking)")
    J_inc = []
    for k in range(len(meshes)):
        vals = [table[(k, nu)]["J"] for nu in nus]
        reffs = [table[(k, nu)]["ratio_eff"] for nu in nus]
        c0, c1, res = extrapolate(reffs, vals)
        span = abs(vals[0] - vals[-1])
        inv = 1.0 / np.asarray(reffs)
        slopes = [(vals[i] - vals[i + 1]) / (inv[i] - inv[i + 1])
                  for i in range(len(vals) - 1)]
        rich = vals[-1] - slopes[-1] * inv[-1] if len(vals) > 1 else vals[-1]
        J_inc.append(rich)
        print(f"    L{k}: 5-point fit {c0:+.9e}  residual {res:.3e} "
              f"({res / max(span, 1e-300):.2%} of span)")
        print(f"         secant slopes " + "  ".join(f"{s:.4e}" for s in slopes))
        print(f"         two-rung Richardson (tightest pair) {rich:+.9e}")
    if len(J_inc) > 2:
        print("    J_inc(h) = " + "  ".join(f"{v:+.6e}" for v in J_inc) +
              "   orders " + "  ".join(f"{o:5.2f}" for o in orders(J_inc)))

    # --- the penalty itself ------------------------------------------------
    print()
    print("  penalty enforcement: ||div u||_L2/||grad u||_L2 against R_eff.")
    print("  Expected log-log slope -1.  NOT evidence against locking.")
    for k in range(len(meshes)):
        vals = [table[(k, nu)]["div_over_grad_l2"] for nu in nus]
        reffs = [table[(k, nu)]["ratio_eff"] for nu in nus]
        slopes = [np.log(b / a) / np.log(rb / ra)
                  for a, b, ra, rb in zip(vals[:-1], vals[1:],
                                          reffs[:-1], reffs[1:])]
        print(f"    L{k}: " + "  ".join(f"{v:.3e}" for v in vals) +
              "   log-log slopes " + "  ".join(f"{s:+5.2f}" for s in slopes))

    if not args.no_iterative:
        print()
        print("  Krylov cost.  This is a genuine iterative solve - CG on a")
        print("  matfree operator with GAMG through gadopt.SPDAssembledPC, the")
        print("  3d_spada options list, rtol 1e-10 - and NOT the coupled")
        print("  solver's MUMPS-on-block-0 fieldsplit, whose flat counts would")
        print("  say nothing about conditioning.  Every count below is a count")
        print("  to an answer that agrees with LU.")
        for k in range(len(meshes)):
            row = [table[(k, nu)] for nu in nus]
            reffs = [r["ratio_eff"] for r in row]
            its = [r["its"] for r in row]
            print(f"    L{k}: its " + "  ".join(f"{i:>6}" for i in its) +
                  "   reason " + "  ".join(f"{str(r['cg_reason']):>3}"
                                           for r in row))
            if all(isinstance(i, int) and i > 0 for i in its):
                slopes = [np.log(b / a) / np.log(rb / ra)
                          for a, b, ra, rb in zip(its[:-1], its[1:],
                                                  reffs[:-1], reffs[1:])]
                print(f"         log-log slope vs R_eff " +
                      "  ".join(f"{s:5.2f}" for s in slopes) +
                      "   (expected bracket 0.5-1.0)")
            mism = [r["cg_mismatch"] for r in row]
            print(f"         |u_cg - u_lu|/|u| " +
                  "  ".join("   n/a" if v is None else f"{v:.0e}" for v in mism))
        print("    and with h at fixed nu (GAMG h-independence)")
        for nu in nus:
            its = [table[(k, nu)]["its"] for k in range(len(meshes))]
            if any(i is None or i < 0 for i in its):
                continue
            print(f"    nu={nu:<7.4f} " + "  ".join(f"{i:6d}" for i in its))

    if args.reduced_quadrature:
        print()
        print("  SRI cross-check.  §8 calls reduced quadrature on the volumetric")
        print("  term a cheap remedy; it is NOT a parameter.  `stress` is")
        print("  assembled whole on one measure at the equation's quad_degree,")
        print("  and nothing selective reaches the volumetric term alone, so the")
        print("  form below is hand-written and the change in the library would")
        print("  be small but real.  Run against a no-locking verdict this is a")
        print("  confirmation: if under-integrating barely moves the converged J,")
        print("  the full rule was not over-constraining.")
        ratio = ratios[-1]
        base = table[(len(meshes) - 1, nus[-1])]
        full = solve_reduced(meshes[-1], ratio, None, n)
        print(f"    hand-written form at full quadrature vs the solver path: "
              f"J {full['J']:+.9e} vs {base['J']:+.9e}, "
              f"rel {abs(full['J'] - base['J']) / abs(base['J']):.2e} "
              "(the internal variable is dropped, so ~dt/tau is expected)")
        for deg in (2, 1):
            print(f"    volumetric quadrature degree {deg}:")
            vals = []
            for k, m in enumerate(meshes):
                r = solve_reduced(m, ratio, deg, n)
                vals.append(r["J"])
                print(f"      L{k}: J={r['J']:+.9e}  "
                      f"div/grad L2={r['div_over_grad_l2']:.3e}")
            print(f"      orders " + "  ".join(f"{o:5.2f}" for o in orders(vals)))

    print()
    print("  VERDICT is read off Q2 and Q2c: order preserved and the error")
    print("  constant bounded means CONDITIONING-ONLY; order degrading or the")
    print("  constant tracking (1-2nu)^-1 means LOCKING.")


if __name__ == "__main__":
    main()
