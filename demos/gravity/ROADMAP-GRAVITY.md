# Road map: boundary treatment for the gravitational Poisson equation — Dirichlet-to-Neumann maps, extended domains, and how to combine them

## 0. Progress checkpoint (solver shipped into gadopt; only E6 outstanding)

Status as of the gadopt integration pass (2026-07-21). Every validation
experiment except the cost study E6 is done, and the generalised solver of
Step 7 has landed in gadopt proper: `gadopt/gravity_solver.py`
(`GravitySolver` with `CylindricalDtN`/`SphericalDtN` boundary-condition
objects) plus `gadopt/spherical_harmonics.py`, commits `b8112d66`,
`565adb47`, `d8c160cb` on this branch; the surface-sheet analytics are
passess commit `6cf46f1` and the sheet validation experiment is `4cf8ce6b`
(`gravity_poisson_sheet.py`). The class reproduces every recorded
benchmark below to the printed digits. Detailed lessons live in
`GRAVITY-LESSONS-LEARNED.md` (part II covers the gadopt class and the
unification mathematics); the demo prototypes (`gravity_dtn.py` and the
E-scripts) are retained unchanged as the reference implementations.

| exp | what | status | headline number |
|---|---|---|---|
| E1 | Dirichlet vs single-mode Robin, truncated disc | **done** | Robin at R=2*rmax recovers the 1.06e-4 floor; Dirichlet degrades to 1.08e-2 (m=3), tracks (r_s/R)^m exactly in the m-sweep |
| E2 | modal DtN via R-space multipliers, blind to the density | **done** | single-mode snaps to the E1 floor at M>=mode; superposed {2,3,5} drops stepwise, all coeffs recovered, inactive ~1e-11; reproduced through `gadopt.GravitySolver` to all printed digits (1.063867e-4, c3/exact 0.999843) |
| E3 | interior DtN on a structured annulus, no centre disc | **done** | discretisation floor for m in {1,2,3,5}, no low-m penalty; both-boundary coeffs match analytical; through the gadopt class: m=1 at 3.181e-5, inactive multipliers ~1e-14 |
| E4 | config D extruded mechanics (ds_t/ds_b, no Submesh) | **done** | R spaces work on extruded meshes (the deferred smoke test): 3D modal DtN validated on the extruded cubed sphere, and 2D modal DtN runs on a genuine Firedrake extruded annulus in `tests/unit/test_gravity_solver.py` — the last untested corner is closed |
| E5 | refinement study, exact source, config D | **done** | optimal O(h^{p+1}): observed order 2.00 (CG1), 3.00 (CG2), CG2 to 6.8239e-8 at dr=0.01, reproduced through the gadopt class |
| E7 | 3D single-(l,m) Robin, extruded cubed sphere | **done** | order 2.0 (CG1), 3.0 (CG2) on the production extruded mesh, config D, no Submesh; rerun through the full modal `SphericalDtN(L=2)`: orders 2.03/1.92 (CG1), 3.00 (CG2) |
| E8 | 3D modal Y_lm DtN | **core done** | full (l,m) multiplier machinery at L=2-3 on the extruded cubed sphere: optimal orders, Y_lm coefficient recovered to 1e-5, inactive coefficients ~1e-14; the L-vs-R_buf trade study for production L remains open |
| S | surface sheets on DtN boundaries (section 5.4) | **done** | sheet enters as the single term -4 pi G sigma v ds on either boundary; validated against passess SheetPolar2D on both boundaries (7.1e-12 for m=1 outer, floors elsewhere), 3D interior-monopole sheet (nonzero total mass) in the unit tests |
| E6 | cost study A vs D | not started | DOF/wall-clock/Krylov-count comparison at matched mantle resolution still to be measured |

Key deviations from what this document predicted, all recorded in
`CLAUDE.md` with source references:

- **The naive form (section 6.2) diverges.** It leaves the psi-psi block a
  singular pure-Neumann Laplacian, fatal for the mandatory R-block
  fieldsplit. The fix is a **Robin-shifted DtN** (flux written as
  `-(alpha/R) psi + sum_m ((alpha-m)/R) c_m e_m`), which moves a pointwise
  Robin term into the psi-psi block and makes it SPD. Section 6.2 should be
  read with this shift applied. As a bonus the exterior monopole needs no
  separate multiplier (the shifted m=0 is exactly the zero-mean condition).
- **Vector-valued R Arguments are unsupported** in this Firedrake, so the
  packed `VectorFunctionSpace(mesh, "R", 0, dim=2M)` of section 6.2 cannot
  be an unknown; use separate scalar R fields (section 6.3's "wide rows"
  still describes the linear algebra correctly).
- **R blocks forbid monolithic aij assembly**, forcing the Schur
  fieldsplit of section 6.3 from the start (not just for iterative solves).
- **Submesh does not inherit curved P2 coordinates** — a refinement-study
  caveat for E5 not anticipated here.
- **The DG0 density is the accuracy floor, at O(h^2).** Centroid-sampling
  cos(m phi) caps the potential at O(h^2); integrating the analytic source
  exactly removes it. Not a boundary-treatment issue, but it is what let
  E5's convergence study exist at all.
- **The extended-but-truncated buffer caps accuracy and rate.** A coarse
  exterior annulus carries most of the error and bleeds it back through
  rmax; the DtN must sit *at* rmax (config D), not beyond a buffer, or the
  study never converges cleanly. This reframes "truncated domain" as
  "no domain beyond the DtN boundary".
- **passess is closed-form** (machine precision), so it never limits any
  convergence study — the observed floors are all discretisation.

### Section-4 hypothesis: confirmed

Exterior infinity, interior core, multi-mode discovery, optimal
convergence, and the extruded production mesh have all been demonstrated
(E1-E3, E5, E7). The gravitational Poisson equation solves on the
unmodified extruded cubed sphere with no gmsh, no Submesh, and no exterior
buffer, at optimal order. The extended-domain/submesh baseline survives
only as the cross-validation reference and the fallback for non-separable
geometries, exactly as section 4 proposed.

### Delivered: the generalised solver (Step 7 done)

The two prototype code paths (`gravity_dtn.py`, 2D modal;
`gravity_poisson_3d_robin.py`, 3D Robin) are unified in
`gadopt/gravity_solver.py`. The design differs from the sketch above in
one important way, settled during the API review: the DtN map is not a
solver mode but a **boundary condition object** passed through the
standard G-ADOPT bcs dictionary, exactly like the other solvers:

```python
psi = Function(V, name="Gravitational potential")
solver = GravitySolver(psi, rho, bcs={
    "top":    {"dtn": SphericalDtN(L=4), "sigma": delta_rho_h},
    "bottom": {"dtn": SphericalDtN(L=4)},
})
solver.solve()                    # updates psi in place
solver.coefficients()             # trace spectrum per boundary (the geoid)
```

Key decisions, each recorded in GRAVITY-LESSONS-LEARNED.md part II:

- Every DtN boundary reduces to a mode table (e_k, lambda_k, N_k) plus the
  Robin shift; one geometry-blind core serves 2D, 3D and (later) Cartesian.
  The interior monopole is not a special case (it is the lambda = 0 mode),
  single-mode Robin is the truncation-zero degenerate case (no
  `boundary_treatment` enum), and the exact-source path collapsed into
  `rho` accepting any UFL expression (+ `source_quad_degree`).
- Boundary orientation (exterior vs interior map) and radius are measured
  from the marked boundary (sign of the boundary integral of n.x; mean
  facet radius), not user-specified.
- `"psi"` (strong), `"flux"`, and `"sigma"` (surface mass sheet, the
  inhomogeneous map of 5.4) ride the same dictionary; an unspecified
  boundary keeps natural Neumann. The config-A Dirichlet reference runs
  through the same interface.
- rho may live on a Submesh of the solver mesh (cross-mesh coupling
  auto-detected) — needed for the extended/buffer configurations where the
  mantle fields live on a submesh of the gravity mesh (dynamic-topography
  case, avoiding high production L).
- The 2D nonzero-net-mass monopole datum is implemented on both DtN
  representations: `∂ψ/∂n|_{m=0} = −2GM/R` on a 2-D exterior boundary, with
  `M` summing volume, sheet and prescribed-flux (cavity) mass, and the
  potential returned in the gauge `∮ψ ds = 0`. `check_net_mass` no longer
  refuses; it warns about monopole leakage and refuses only the two gauges
  that cannot exist (a strong `psi` condition alongside nonzero mass, and two
  exterior DtN boundaries on one 2-D mesh). See `NOTES/PLAN-MONOPOLE-C0.md`.
- Not implemented, explicitly guarded: Cartesian geometry (a future
  `CartesianDtN` is a pure addition, and would reuse the same enclosed-mass
  scalars for its own `n = 0` datum); the coupled self-gravitating extraction
  (section 12, deliberately deferred — `solver.F` and the mixed space are
  exposed).

Validation: unit tests in `tests/unit/test_gravity_solver.py` and
`tests/unit/test_spherical_harmonics.py` (17 tests, Firedrake utility
meshes only, inline closed forms, no gmsh/passess dependency), plus
reproduction of the full E2/E3/E5/E7 record through the class (section 0
table). Reviewed in two stages: a math review that re-derived every
closed form (all items approved) and an independent adversarial pass
(no blocking findings; its one should-fix, the monolithic-override guard,
is commit `d8c160cb`).

## 1. Who this document is for

You are a software engineer who has just joined the G-ADOPT project. You know
Python and some numerics, but you do not know geodynamics, you have never
heard of Firedrake, and you have never met a Dirichlet-to-Neumann map. This
document explains the problem, the mathematics of the fix, exactly how to
express it in Firedrake/UFL, and the sequence of experiments that decides
which configuration we ship. By the end you should have a ticket-to-PR path.

Relevant files in this worktree:

- `demos/gravity/gravity_poisson_test.py` — the working baseline: Poisson on
  an extended disc with density on a submesh, validated against analytical
  solutions to 0.33% relative L2 error in the mantle.
- `demos/gravity/generate_gravity_disc.py` — the gmsh generator for that
  baseline. Read it to understand what "extended domain" costs in practice.
- `demos/gravity/CLAUDE.md` — condensed notes on the baseline (mesh-conforming
  shell finding) AND the full DtN lessons-learned section (E1-E3, the
  Robin-shift fix, the R-space/Firedrake pitfalls). Read this before E4.
- `demos/gravity/gravity_dtn.py` — the `GravityPoissonSolver` class:
  reusable modal DtN, interior + exterior, the seed for the gadopt/ solver.
- `demos/gravity/gravity_poisson_robin.py`, `gravity_poisson_m_sweep.py`,
  `gravity_poisson_dtn_modal.py`, `gravity_poisson_dtn_multimode.py`,
  `generate_gravity_annulus.py`, `gravity_poisson_dtn_annulus.py` — the
  E1-E3 experiment scripts (see the section 0 table).
- `demos/gravity/ROAD-MAP-STOKES-COUPLE.md` — the companion roadmap for the
  monolithic Stokes–Poisson coupling with mean-free dynamic topography. This
  document feeds into that one; neither replaces the other. Section 12 below
  specifies the interface between them.
- `~/Workplace/passess` — the analytical-solutions package
  (`PoissonPolar2D`, `PoissonCartesian2D`, spherical). Every experiment
  below is judged against it.
- `demos/glacial_isostatic_adjustment/2d_cylindrical/2d_cylindrical.py:68-75`
  and `demos/mantle_convection/3d_spherical/3d_spherical.py:32-33` — the
  production mesh constructions (extruded annulus, extruded cubed sphere)
  that the winning configuration ultimately has to run on.

## 2. The physics in one paragraph

Density anomalies ρ inside a planet perturb the gravitational potential ψ
through Poisson's equation, ∇²ψ = −4πγρ (γ is the gravitational constant;
in our 2D reductions ρ is mass per unit out-of-plane length). The correct
boundary condition lives at infinity: ψ must match the decaying free-space
solution. But a finite-element mesh is finite. Everything in this document
is about answering one question: **how do we impose "the domain is actually
infinite" on a finite mesh, accurately and cheaply, in a way that survives
being embedded in a coupled nonlinear solve?**

## 3. What the code does today

The baseline (`gravity_poisson_test.py`) answers the question by brute
force:

1. Mesh a full disc out to R_grav = 10·rmax, coarsening rapidly outside the
   mantle (`generate_gravity_disc.py`). At R_grav the analytical potential
   of an m = 10 anomaly is ~1e-10 of its mantle value, so a homogeneous
   Dirichlet condition there (line 111) is an excellent approximation.
2. Restrict the density to a mantle `Submesh` (lines 62-66) and couple the
   two meshes through `intersect_measures` and a `MixedFunctionSpace`
   (lines 94-108).
3. Compare against `passess.polar.PoissonPolar2D`. Result: 0.33% relative
   L2 error in the mantle at lc_mantle = 0.04, m = 10.

This works and it stays as our reference. But it has three structural costs:

- **Mesh generation.** The extended disc needs gmsh, five conforming
  circles, and two hand-tuned coarsening ramps. Every geometry change means
  re-tuning size fields.
- **Wasted DOFs.** The exterior annulus and the inner disc exist only to
  emulate infinity and regularity; they carry no physics we care about.
- **The extruded-mesh incompatibility.** This is the decisive one.
  Firedrake's `Submesh` does not work on extruded meshes, and *all* G-ADOPT
  production meshes are extruded (radially extruded `CircleManifoldMesh` in
  2D cylindrical, radially extruded `CubedSphereMesh` in 3D). A gravity
  formulation that requires `Submesh` cannot run on the meshes our GIA and
  mantle-convection solvers actually use. It forces gmsh meshes everywhere,
  which is a heavy price in 3D.

## 4. The strategic picture: two tools, one decision

We have two tools:

- **Extended domain + submesh** (what we have): approximate infinity by
  distance; restrict sources by submeshing. Geometry-agnostic, conceptually
  simple, validated. Requires unstructured meshes; incompatible with
  extruded production meshes.
- **Dirichlet-to-Neumann (DtN) boundary condition** (what this roadmap
  builds): impose the *exact* far-field behaviour on a boundary placed at —
  or very near — the edge of the physical domain. Requires the boundary to
  be a coordinate surface of a separable geometry (circle, sphere,
  periodic-box top/bottom), which is exactly what our production meshes
  have.

They are not mutually exclusive, and the end state is a *combination*: DtN
where the geometry permits it, extended domains where it does not or as a
cross-check. The central hypothesis this roadmap tests is:

> **If exterior and interior DtN conditions on the mantle's own boundaries
> reach the same accuracy as the extended mesh, then the gravity solve
> needs no submesh, no exterior, no inner disc, and no gmsh: it runs on the
> unmodified extruded production mesh, and ρ lives on the whole mesh.
> Extended-domain-plus-submesh then survives only as the cross-validation
> reference and as the fallback for non-separable geometries.**

If the hypothesis fails (accuracy or solver cost), the fallback ladder is:
thin-buffer meshes with DtN at a slightly enlarged boundary, then the full
extended domain. Every rung is benchmarked in section 8 so the decision is
made by numbers, not taste.

## 5. The mathematics of the DtN map

### 5.1 The key fact: the map needs the trace of ψ, not ρ

The obvious worry is that a far-field condition needs to know the source —
i.e. that we must Fourier-transform the spatial density field. **We do
not.** The DtN map is a property of the source-free exterior alone. Expand
the boundary trace of ψ in the angular eigenfunctions of the boundary; each
mode extends uniquely into the exterior as the decaying homogeneous
solution, whose normal derivative at the boundary is proportional to its
own trace. The density enters nowhere. The only requirement is that the
boundary encloses all sources.

Per mode, on a circle/sphere of radius R with sources inside:

| geometry | exterior solution | exact condition at r = R |
|---|---|---|
| 2D polar, mode m ≥ 1 | r^(−m) | ∂ψ/∂r + (m/R) ψ = 0 |
| 2D polar, mode m = 0 | ln r | ∂ψ/∂r = −2γM/R, M = ∫ρ dA |
| 3D spherical, degree l ≥ 0 | r^(−l−1) | ∂ψ/∂r + ((l+1)/R) ψ = 0 |
| 2D Cartesian (x-periodic, width L), mode k_n = 2πn/L, n ≥ 1 | e^(−k_n z) | ∂ψ/∂z + k_n ψ = 0 (top) |
| 2D Cartesian, n = 0 | linear in z | ∂ψ/∂z = −2πγM/L (top) |

Note the two exceptional monopole rows. In 2D the m = 0 (and Cartesian
n = 0) exterior does not decay, so the trace alone does not determine the
flux — but the flux is determined by a *single scalar*, the total enclosed
mass, which is one UFL assembly (`assemble(rho*dx)`), not a Fourier
transform. In 3D there is no exception at all: l = 0 decays as 1/r and the
trace-based map is complete. 3D is the *clean* case.

### 5.2 The interior DtN: why an inner boundary is legitimate after all

Intuition says a gravitational problem cannot have an inner boundary — a
naive Neumann or Dirichlet condition at r = rmin is simply wrong, which is
why the baseline meshes all the way through r = 0. But the same
construction works pointing inward. If the core (r < R_i) is source-free,
the solution there is spanned by the *regular* homogeneous solutions
(r^(+m) in 2D, r^(+l) in 3D), and matching across R_i gives, on the annulus
side:

- 2D, mode m ≥ 1:  ∂ψ/∂r − (m/R_i) ψ = 0, which in the weak form (outward
  normal n = −r̂) contributes with the **same positive sign** as the
  exterior term (see 6.1).
- 2D, mode m = 0: the interior solution is constant, so ∂ψ/∂r = 0 —
  homogeneous Neumann is *exact* for the monopole with no enclosed mass.
- 3D, degree l: ∂ψ/∂r − (l/R_i) ψ = 0; l = 0 again reduces to natural
  Neumann.

So an annulus (or spherical shell) equipped with interior DtN at rmin and
exterior DtN at rmax is an *exact* reduction of the free-space problem —
no meshing through the coordinate singularity at r = 0, no coarse inner
disc, no worry about whether the coarse core pollutes low-m modes. This is
what makes the pure-extruded-mesh configuration possible: the production
annulus/shell mesh *is* the whole computational domain.

### 5.3 Truncation replaces distance as the accuracy knob

A practical DtN implementation truncates the mode sum at some M (section
6.2); modes above M see whatever the untreated boundary condition is
(homogeneous Neumann). The error this makes is controlled by how much
high-mode content reaches the boundary: a source feature at radius r_s
contributes to boundary mode m with amplitude ~(r_s/R)^m, so the neglected
content is O((r_s/R)^(M+1)). Concretely, with our density shell at
r_s ≈ 2.05 and the DtN boundary at R = 3.0, M = 30 leaves a relative error
of (2.05/3)^31 ≈ 8e-6 — three orders below the current discretisation
error, with the boundary at 3.0 instead of 22.2. The trade is explicit:
pull the boundary in and pay with more modes, or push it out and truncate
earlier. Section 8's experiments map this trade instead of guessing it.

### 5.4 Surface masses sitting on the DtN boundary

In the coupled problems, boundary mass sheets are sources: dynamic
topography h at the surface and CMB in mantle convection, load-induced
topography in GIA. If such a sheet σ(φ) sits *exactly on* the DtN boundary,
the map still works: the sheet enters through the flux jump
[∂ψ/∂r] = −4πγσ, so the interior-side condition becomes inhomogeneous,

    ∂ψ/∂r + (m/R) ψ = 4πγ σ   at r = R,

which in the weak form is the usual DtN term plus the completely standard
surface-source term −4πγ∫σ v ds. No buffer region is needed above the
surface. This is what lets configuration D in section 7 put the DtN
boundary directly at rmax even in the fully coupled problem.

## 6. Implementation in Firedrake/UFL

### 6.1 Level 0: single-mode Robin — one line

For a single-mode benchmark (the current m = 10 test; any single-(l,m) 3D
test), the boundary trace contains only that mode, so the DtN map collapses
to a *local* Robin condition. Weak form of −∇²ψ = 4πγρ:

    ∫∇ψ·∇v dx − ∫_Γ v ∂ψ/∂n ds = 4πγ ∫ρ v dx.

Substituting the exterior map at the outer boundary (n = +r̂) and the
interior map at the inner boundary (n = −r̂) gives — with the same sign for
both, do the two-line derivation yourself to believe it —

```python
F += (m / R_out) * psi * v * ds_outer     # exterior DtN, exact for mode m
F += (m / R_in)  * psi * v * ds_inner     # interior DtN, exact for mode m
```

That is the entire implementation. It is exact for the mode present and
wrong for modes that are absent, which is fine in a single-mode benchmark.
Two bonuses: the pointwise Robin term penalises constants, so the
pure-Neumann nullspace problem does not arise; and being a plain positive
`ds` term it is symmetric, coercive, assembles into any solver, and is
trivially differentiable/adjointable.

### 6.2 Level 1: the modal DtN via R-space multipliers

For general (multi-mode) density the map is nonlocal: it acts mode-by-mode
on the trace. Written out, the boundary operator is a low-rank sum

    a_DtN(ψ, v) = Σ_{m=1..M} (m/(πR²)) [ (∫_Γ ψ cos mφ ds)(∫_Γ v cos mφ ds)
                                        + (∫_Γ ψ sin mφ ds)(∫_Γ v sin mφ ds) ],

a product of boundary integrals — which is *not* a UFL form (UFL forms are
integrals of pointwise expressions). The cure is the same R-space trick
that ROAD-MAP-STOKES-COUPLE.md uses for the mean-free dynamic topography:
introduce one global scalar unknown per mode coefficient and let scalar
constraint equations define them as the Fourier coefficients of the trace.
`FunctionSpace(mesh, "R", 0)` is a space with exactly one DOF on the whole
mesh; a vector R-space packs all 2M coefficients into one field.

```python
x = SpatialCoordinate(mesh)
phi = atan2(x[1], x[0])

M = 30                                       # mode truncation
RM = VectorFunctionSpace(mesh, "R", 0, dim=2*M)
Z = V * RM
z = Function(Z)
psi, c = split(z)
v, mu = split(TestFunction(Z))

F = inner(grad(psi), grad(v))*dx - 4*pi*gamma*rho*v*dx

for m in range(1, M + 1):
    cm, sm   = c[2*m-2],  c[2*m-1]
    muc, mus = mu[2*m-2], mu[2*m-1]
    # Constraint rows: since mu is globally constant, each of these is ONE
    # scalar equation.  The "- cm/2" works because ∫ds = 2πR, so it
    # enforces  c_m = (1/(πR)) ∫ psi cos(mφ) ds  — the Fourier coefficient.
    F += (psi*cos(m*phi) - cm/2) * muc * ds_outer
    F += (psi*sin(m*phi) - sm/2) * mus * ds_outer
    # DtN flux fed back into the psi equation:
    F += (m/R_out) * (cm*cos(m*phi) + sm*sin(m*phi)) * v * ds_outer
```

Sanity check against 6.1: for a pure trace ψ = cos(mφ), the constraint
gives c_m = 1 and the flux term reduces to (m/R)∫ψ v ds — the Robin form.
The interior version is identical with (m/R_in) on `ds_inner`. The 2D
monopole row, when total mass is nonzero (or solution-dependent, see
section 12), is one more scalar: M_tot ∈ R constrained by
∫ρ dx − M_tot·(…) = 0, feeding the Neumann datum −2γM_tot/R.

Everything here is a genuine UFL form. Firedrake differentiates it
symbolically, SNES sees the full coupling when this block is embedded in a
larger system, and pyadjoint tapes it. This is the answer to "how do we
compute the Fourier expansion in Firedrake": we never compute it — the
solver *enforces* it as 2M scalar constraint equations.

### 6.3 What this does to the linear algebra

The Jacobian gains 2M rows/columns that couple only to boundary DOFs. Each
is a "wide row" — dense within the boundary, zero elsewhere — exactly like
the pressure-nullspace and λ_top multipliers we already use; sparse AIJ
handles it, MUMPS handles it. As written above the off-diagonal blocks are
unsymmetric (constraint rows are unscaled); multiplying the constraint rows
by m/R symmetrises the system at the cost of a diagonal −πm block for the
coefficients — a standard saddle structure, Schur-eliminable. For iterative
solves, isolate the R-field in its own fieldsplit with `preonly`/`lu` (a
2M×2M dense solve, microseconds) and precondition the ψ block with AMG as
usual; the pattern is the `pc_fieldsplit_N_fields` nesting already worked
out in ROAD-MAP-STOKES-COUPLE.md section 6. (That pattern is what shipped,
and it has since been replaced in `gadopt.GravitySolver`: describing the split
by field number makes PETSc enumerate every scalar R sub-field and it refuses
past 128 of them, so the two blocks are now described by index set instead.
See `SOLVE-STRATEGIES.md`, "How the split is described to PETSc".)

One genuine nullspace subtlety, worth knowing before it bites: the *modal*
DtN terms annihilate constant traces (∫cos mφ ds = 0 for m ≥ 1), so with
DtN on all boundaries and a zero-total-mass density the constant nullspace
of the Neumann problem survives. Fix it the usual way (pass the constant
`VectorSpaceBasis` nullspace, or pin the boundary mean with one more R
multiplier). The pointwise Robin of 6.1 does *not* have this issue.

### 6.4 Pitfalls checklist

- **Quadrature degree.** UFL's automatic degree estimation is meaningless
  for `cos(30*atan2(x[1], x[0]))` on straight or quadratic facets. Set it
  explicitly, `ds_outer = ds(OUTER_ID, degree=…)`, and verify by checking
  that assembled orthogonality ∫cos(mφ)cos(m'φ) ds is diagonal to
  tolerance. Rule of thumb: the boundary mesh must resolve mode M anyway
  (facet size ≲ R/M), after which modest degrees (8–12) suffice per facet.
- **The atan2 branch cut** at φ = ±π is harmless for integer m — cos(mφ)
  and sin(mφ) are both continuous through it. If paranoid, use the
  polynomial forms cos(mφ) = Re[(x+iy)^m]/r^m.
- **Extruded-mesh boundary labels.** On radially extruded meshes the inner
  and outer boundaries are "bottom" and "top": use `ds_b`/`ds_t` (this is
  what the GIA demos already do for their boundary conditions).
- **R-spaces on extruded meshes.** Expected to work; verify with a
  five-line smoke test (assemble one constraint row on the extruded
  annulus) before building anything on it. This is Step 3's first commit.
- **Mesh-conforming density boundaries still matter.** The 230%-error
  lesson from the baseline (DG0 straddling the shell boundary leaks m = 0
  content) is independent of the boundary treatment. On extruded meshes
  the fix is free: place extrusion layer interfaces at the shell radii.

## 7. The candidate configurations

The decision space, from most conservative to most aggressive:

| | mesh | boundary treatment | needs Submesh | needs gmsh | runs on extruded | status |
|---|---|---|---|---|---|---|
| A | disc to 10·rmax | Dirichlet at R_grav | yes | yes | no | baseline, validated |
| B | disc to ~1.5·rmax | exterior DtN at R_Γ | yes | yes | no | cheap validation of DtN itself |
| C | annulus, thin buffers | DtN both sides | yes | yes | no | validates interior DtN |
| D | mantle only (rmin–rmax) | DtN both sides, surface masses in the BC | **no** | **no** | **yes** | the production target |

A is the reference every other row must reproduce. B isolates "does the
exterior DtN work" from every other change. C adds the interior map and
drops the centre. D is the hypothesis of section 4: the unmodified
production mesh, density on the whole mesh (no restriction needed — the
mesh *is* the mantle), DtN carrying both infinity and the core, surface
sheets entering through the boundary condition (5.4). If D matches A's
accuracy at production resolutions, the combination question is answered:
DtN everywhere the geometry is separable, and A's machinery is retained
only as cross-validation and for exotic geometries.

## 8. The benchmark matrix

Each experiment has a pass criterion; together they force the A-vs-D
decision by measurement. All 2D experiments validate against
`passess.polar.PoissonPolar2D`, mantle-restricted relative L2 as in the
baseline (`gravity_poisson_test.py:137-141`).

- **E1 — Robin truncation sweep (config B).** Single mode m = 10, Robin at
  R_Γ ∈ {2.4, 3.0, 4.0}, mesh otherwise as baseline. Pass: reproduces the
  0.33% error at every R_Γ, at a fraction of the DOFs. This is the
  first-hard-number experiment and it is nearly free.
- **E2 — modal DtN and the truncation law.** Multi-mode density (e.g.
  m ∈ {2, 10, 25} superposed), modal DtN, sweep M at fixed R_Γ and R_Γ at
  fixed M. Pass: error floor scales as (r_s/R_Γ)^(M+1) until it hits the
  discretisation floor.
- **E3 — interior treatment shoot-out.** Sweep m ∈ {1, 2, 5, 10}: coarse
  meshed-through centre (A) vs interior DtN annulus (C). Low m is where the
  coarse core must prove itself — the interior field ~r^m is only
  negligible for high m. Pass for C: error independent of the core's
  existence. This experiment also decides whether A's inner coarsening ramp
  was ever safe.
- **E4 — the production-mesh run (config D).** The headline. Native
  extruded annulus (`CircleManifoldMesh` + radial `ExtrudedMesh`, layer
  interfaces at the shell radii), density on the full mesh, DtN via
  `ds_t`/`ds_b`, no gmsh anywhere. Pass: matches A's accuracy; mesh setup
  is ~5 lines.
- **E5 — refinement study.** On the E4 winner, refine with the truncation
  knobs (M, quadrature) held far below the discretisation error. Pass:
  clean O(h²) for CG1 (and O(h³) for CG2), no floor within the swept range.
  This is the study the baseline never had, and it is only meaningful once
  the three error sources (truncation, interior, discretisation) are
  independently dialable — which DtN is what makes possible.
- **E6 — cost.** DOF counts, assembly + solve wall-clock, and Krylov
  iteration counts with the R-block fieldsplit, A vs D at matched mantle
  resolution. Pass for D: cheaper across the board (expected: ~all exterior
  and core DOFs deleted, 2M scalars added).
- **E7 — 3D single-(l,m) Robin.** Extruded cubed sphere, Robin
  ((l+1)/R_out, l/R_in), validated against passess spherical. The 3D
  entry point; no multipliers needed.
- **E8 — 3D modal DtN.** Real spherical harmonics up to degree L,
  (L+1)² − 1 multiplier pairs plus monopole-free l = 0 handled by the
  trace map (no 3D exception, 5.1). Y_lm as UFL expressions — polynomial in
  (x, y, z)/r, generated once by sympy codegen, not hand-written.

## 9. Step-by-step implementation plan

Each step is one PR-sized unit that compiles and validates on its own.

### Step 0: reproduce the baseline

Regenerate the meshes (`generate_gravity_disc.py`; the `.msh` files are
deliberately untracked), run `gravity_poisson_test.py`, confirm 0.33%.
Read sections 5–6 of this document with the code open.

### Step 1: Robin-truncated disc (E1) — DONE

`gravity_poisson_robin.py` (four-case matrix + shared helpers) and
`gravity_poisson_m_sweep.py`. Switched to m=3/CG2 so the truncation error
is visible above the discretisation floor (see CLAUDE.md). Added the
`curve_mesh` P2 trick to keep coarse outer boundaries circular under Robin.

### Step 2: modal DtN prototype (E2) — DONE

`gravity_dtn.py` (`GravityPoissonSolver`), driven by
`gravity_poisson_dtn_modal.py` (single mode, M-sweep) and
`gravity_poisson_dtn_multimode.py` (superposed {2,3,5}). Uses the
Robin-shifted form and separate scalar R fields (section 0). Quadrature
sanity check wired in as `check_boundary_quadrature()`. Nullspace handled
by the shift, not a separate exterior multiplier.

### Step 3: interior DtN and the annulus (E3) — DONE

Built as a *gmsh structured extruded-style* annulus
(`generate_gravity_annulus.py`), NOT a Firedrake ExtrudedMesh, because
Submesh cannot handle extrusion — so the "R-on-extruded smoke test" is
deferred to Step 4, where the real ExtrudedMesh first appears.
`gravity_poisson_dtn_annulus.py` runs both maps and the m in {1,2,5}
sweep; passes with no low-m penalty.

### Step 4: the production-mesh configuration (E4 + E5 + E6) — E4/E5 DONE, E6 open

E4 and E5 are done (section 0): the 2D extruded-annulus configuration with
zero gmsh dependence runs in `tests/unit/test_gravity_solver.py`
(CircleManifoldMesh + radial ExtrudedMesh, DtN via top/bottom markers),
and E5's clean-order study passed and reproduces through the gadopt class.
Only the E6 cost measurement remains. Original plan:

Extruded-annulus script with zero gmsh dependence; then the refinement and
cost studies. Deliverable: the A-vs-D decision table of section 7, filled
with numbers, and a short written recommendation. **This is the go/no-go
gate for the section 4 hypothesis** — everything after assumes it passed;
if it failed, the fallback ladder (thin-buffer B/C) gets promoted instead.

### Step 5: 3D spherical (E7, then E8) — DONE up to the E8 trade study

E7 done; the E8 machinery (real Y_lm from `gadopt/spherical_harmonics.py`,
(L+1)^2 multipliers per boundary) is delivered and validated at L = 2-3 on
the extruded cubed sphere with inactive coefficients at machine zero. What
remains of E8 is the production question only: mapping the L-vs-R_buf
residual surface (see the hybrid note in CLAUDE.md) and the assembly-cost
scaling before choosing production L. Original plan:

Robin single-(l,m) on the extruded cubed sphere first — it is a
five-line delta on E4's script plus the passess spherical solution. Then
the sympy-generated Y_lm modal machinery. Keep L modest (≤ 32; ~10⁳
multipliers, dense block still trivial next to the volume solve).

### Step 6: Cartesian design (see section 11)

Periodic-box modal DtN in 2D against `PoissonCartesian2D`; a short design
note on the non-periodic case (where extended domains stay the tool of
choice — box meshes are cheap, so DtN buys little there).

### Step 7: promote into gadopt/ — DONE

Delivered as `gadopt/gravity_solver.py` (see the section-0 "Delivered"
note). The design review replaced the `boundary_treatment=` enum sketched
here with per-boundary DtN objects in the bcs dictionary: Dirichlet-far is
`{"psi": 0}`, Robin is the truncation-zero degenerate case, and the modal
map is `{"dtn": CylindricalDtN(M)/SphericalDtN(L)}` — one formulation,
one knob. The R-space block is exposed through `solver.F` and
`solver.mixed_space` for the coupled solvers of section 12. Unit tests
cover both geometries on utility meshes at coarse resolution; the
refinement studies live in the demo scripts.

## 10. 3D spherical specifics

Everything transfers with m → (l, m) and the coefficients from the 5.1
table; the monopole exception disappears; the boundary eigenfunctions are
real spherical harmonics. Only two genuinely new items:

1. **Generating Y_lm as UFL.** Each real Y_lm is a homogeneous polynomial
   in (x, y, z) divided by r^l. Hand-writing them beyond l ≈ 4 is
   error-prone; generate them once with sympy into a Python module and
   import. Orthogonality on the discrete sphere is the built-in test.
2. **Multiplier count.** (L+1)² coefficients per boundary. At L = 32 that
   is 1089 scalars against ~10⁶–10⁷ volume DOFs — negligible memory, and
   the dense elimination block is still tiny. The real cost scaling to
   watch is assembly: each mode adds a boundary integral, so assembly of
   the DtN block is O(L² · N_facets). E8 measures whether this needs
   batching (a single vector-valued constraint form) before production.

## 11. Cartesian specifics

Cartesian is genuinely different because a box has no separable exterior —
*unless* it is periodic laterally, which is exactly the setting of
`passess.cartesian.PoissonCartesian2D` (single wavenumber k, slab between
z1 and z2). For an x-periodic box of width L, the modes are k_n = 2πn/L
and the 5.1 table applies on the top and bottom boundaries with the same
R-space machinery (cos/sin(k_n x) in place of cos/sin(mφ)); the n = 0 row
is the sheet monopole, a flux datum ∓2πγM/L. A single-k benchmark again
reduces to a Robin one-liner, `+ k*psi*v*ds_top` (and bottom).

For non-periodic boxes there is no modal boundary and the honest options
are the extended domain (cheap here — box meshing is trivial and there is
no submesh pressure, since Cartesian production meshes are not extruded in
a way that conflicts) or embedding in a periodic supercell. The roadmap
deliberately leaves non-periodic Cartesian as extended-domain territory:
this is the "combination" answer of section 4 in its clearest form.

## 12. Interface to the coupled solvers

This roadmap ends with a validated, geometry-appropriate boundary
treatment for the standalone Poisson problem. ROAD-MAP-STOKES-COUPLE.md
owns the monolithic coupling. The contract between them:

1. **Everything is a UFL residual contribution.** Robin terms, modal
   constraint rows, and flux feedback terms are plain forms; the coupled
   solvers add them to their residual F and extend their mixed space by
   the R-block. Nothing here performs a solve or a Python-level
   post-processing step, so SNES differentiates through all of it — the
   same design invariant the mean-free-h multiplier obeys.
2. **Surface masses enter through the BC** (5.4). Dynamic topography
   σ = Δρ·(h_ufl − λ_top) at rmax/rmin plugs into the inhomogeneous DtN
   term as −4πγ∫σ v ds. Since h_ufl is itself a UFL expression in (u, p),
   the gravity boundary condition becomes solution-dependent
   automatically and correctly — no buffer region above the surface is
   required.
3. **Solution-dependent monopole.** In GIA, ρ₁ = −∇·(ρ₀ u) makes the 2D
   enclosed mass an unknown. The same R-trick handles it: M_tot ∈ R
   constrained by ∫ρ₁ dx, feeding the Neumann datum. (In 3D this is moot —
   the trace map covers l = 0.)
4. **Fieldsplit placement.** The R-block joins the "gravity" outer field
   alongside Φ (ROAD-MAP-STOKES-COUPLE.md section 6); the Stokes inner
   split remains verbatim untouched.
5. **Adjoint.** Because every term is taped UFL, pyadjoint support should
   be automatic; verifying the tape through R-space constraint rows is a
   follow-up ticket shared with the λ_top work — do it once, both roadmaps
   benefit.

One trick, used four times across the two documents: mean-free h, modal
DtN, interior DtN, solution-dependent monopole. All are "a global scalar
defined by a boundary/volume integral of the unknowns, enforced by an
R-space constraint row". Implement it well once (Step 7) and the coupled
work inherits it.

## 13. Out of scope

- Removing the extended-domain/submesh baseline. It is the reference for
  every experiment above and the fallback for non-separable geometries.
- Fast spherical-harmonic transforms or FMM-grade scalability for L ≫ 32.
  Measure first (E8); optimise only if assembly cost demands it.
- Local high-order absorbing conditions (Bayliss–Turkel-type rational
  approximations of the DtN symbol). They avoid nonlocality but need
  auxiliary boundary fields that Firedrake has no clean home for; the
  exact modal map is both simpler and better here.
- The coupled Stokes–Poisson solve itself — that is the other roadmap.

## 14. What success looks like

- E1 reproduces the baseline accuracy with the boundary at r = 3 instead
  of 22.2, and E2 shows the (r_s/R_Γ)^(M+1) floor moving as predicted —
  the truncation knob demonstrably replaces the distance knob.
- E4: the gravitational Poisson equation solves on the *unmodified*
  production extruded annulus — no gmsh, no Submesh, no exterior — at the
  baseline's accuracy, and E6 shows it strictly cheaper.
- E5: the first clean second-order refinement study this problem has had.
- A filled-in section 7 table and a one-paragraph recommendation stating
  which configuration each application (2D/3D GIA, 2D/3D convection,
  Cartesian) uses, signed off by the numbers.
- A `gadopt/` solver class whose boundary-treatment option makes the
  choice explicit, tested, and inheritable by the coupled solvers.

## 15. Reading list

In order of decreasing priority:

1. `demos/gravity/gravity_poisson_test.py` and `CLAUDE.md` — the baseline
   and its lessons (especially the mesh-conforming-shell finding).
2. Section 5 of this document, next to
   `~/Workplace/passess/solution-dim2-polar.md` — the same expansion, done
   analytically; the DtN coefficients fall out of the exterior branch of
   the Green's function.
3. `demos/glacial_isostatic_adjustment/2d_cylindrical/2d_cylindrical.py:46-75`
   — the extruded production mesh the target configuration runs on.
4. ROAD-MAP-STOKES-COUPLE.md sections 5–6 — the R-space multiplier and
   fieldsplit patterns this document reuses.
5. Firedrake docs: `FunctionSpace(mesh, "R", 0)`, extruded meshes and
   `ds_t`/`ds_b`, and setting quadrature degree on measures.
6. Givoli, *Numerical Methods for Problems in Infinite Domains* — the
   classical reference for truncated DtN maps, if you want the theory
   with proofs.
