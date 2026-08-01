# Gravity Poisson Equation with Submesh Coupling

> On branch `sghelichkhani/selfgravity` the work has moved past gravity alone.
> This document and `GRAVITY-LESSONS-LEARNED.md` remain the record of the
> standalone Poisson/DtN development and are still accurate for it, but the
> active design is `ROADMAP-GIA-SELFGRAV.md`: the shipped `GravitySolver`
> coupled to the viscoelastic GIA solvers, with rotational feedback and a
> time-varying geoid feeding sea level. One position below is superseded there —
> the "DtN degree vs buffer" trade in `ROADMAP-GRAVITY.md` uses the wrong
> exponent for the far-field residual, and §1.2 of the new road map corrects it.
> The coupled solver now exists in 2-D and is verified; **"The monolithic
> self-gravitating GIA solver" at the bottom of this file** is what to read if
> that is what you are here for.

## What this directory contains

A minimal working example that solves the gravitational Poisson equation on an extended disc mesh, with density sources restricted to a mantle submesh. The purpose is to validate the Firedrake submesh coupling machinery and the `intersect_measures` mechanism for the gravity equation that will eventually be coupled to viscoelastic GIA simulations.

There are two scripts that form the test:

- `generate_gravity_disc.py` — gmsh mesh generator for a full disc (r=0 to R_grav) with mesh-conforming density shell boundaries.
- `gravity_poisson_test.py` — solves -nabla^2 psi = 4 pi gamma rho and compares against the analytical solution from the `passess` package.

The original scripts by Dale (`submesh_2way_coupling.py` and `unstructured_annulus.py`) are the fully coupled viscoelastic GIA problem that this test is working toward.

## The equation

We solve the 2D gravitational Poisson equation in polar coordinates:

    nabla^2 psi = -4 pi gamma rho

where rho is a density anomaly localised to a thin radial shell inside the mantle, with azimuthal structure cos(m phi). The analytical solution comes from `passess.polar.PoissonPolar2D`, which provides closed-form Green's function convolutions for each azimuthal Fourier mode.

## Mesh geometry

The gmsh mesh is a full disc with five concentric regions separated by mesh-conforming circles:

    r = 0  -->  rmin  -->  r1_shell  -->  r2_shell  -->  rmax  -->  R_grav

- Inner disc (0 to rmin): no sources, no physics of interest.
- Mantle (rmin to rmax): where the Poisson source lives and where we assess the error. Extracted as a Firedrake `Submesh`.
- Exterior (rmax to R_grav): extended domain for the infinite-boundary approximation. R_grav = 10 * rmax.

The density shell [r1_shell, r2_shell] is a 50 km thick layer at 500 km depth (non-dimensionalised by D = rmax - rmin = 1.0, corresponding to 2891 km mantle depth). Its boundaries are explicit circles in the gmsh geometry so that element edges align with them exactly.

## Submesh workflow

The Firedrake submesh construction follows the same pattern as the fully coupled script:

1. Load the full mesh from gmsh.
2. Mark mantle cells with a DG0 indicator function (conditional on rmin <= r <= rmax).
3. `RelabeledMesh(full_mesh, [F_mantle, F_all], [98, 99])` to attach cell labels.
4. `Submesh(mesh, 2, 98)` to extract the mantle as a standalone mesh.

## Cross-mesh variational form

The potential psi lives on the full mesh (CG1), while the density rho lives on the submesh (DG0). To couple them, we use a `MixedFunctionSpace` with a dummy field on the submesh:

    W = V_fullmesh * V_dummy_submesh

with intersect_measures:

    dx_full = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", domain=subm),))
    dx_sub  = Measure("dx", domain=subm, intersect_measures=(Measure("dx", domain=mesh),))

The Laplacian integrates over the full mesh (via dx_full), the source term integrates over the submesh (via dx_sub). The MixedFunctionSpace is necessary because Firedrake's cross-mesh assembly requires it to set up the entity maps between parent and child mesh DOFs.

Firedrake does support cross-mesh coefficient forms without MixedFunctionSpace (there is a test `test_assemble_parent_coefficient` in the submesh test suite), but for solving a coupled variational problem the MixedFunctionSpace route is the tested and reliable path.

## Critical finding: mesh-conforming shell boundaries

When the density shell boundaries do NOT align with element edges, DG0 introduces spurious low-order azimuthal content. For a cos(m phi) density with m=10, even a small m=0 leakage dominates the global error because the m=0 potential decays as ln(r) while the m=10 potential decays as r^{-10}. In early testing this produced 230% relative error despite the solver converging correctly.

The fix is to include the shell radii r1_shell and r2_shell as explicit circles in the gmsh geometry. With mesh-conforming boundaries, every DG0 cell is entirely inside or entirely outside the shell, so the integral of rho over the domain is machine-zero (no m=0 leakage) and the relative L2 error drops to 0.3%.

## Error assessment

We measure the L2 error only within the mantle submesh:

    error = sqrt(integral over mantle of (psi_h - psi_exact)^2 dx)

This is the physically meaningful region. The exterior is just a buffer for the infinite-domain approximation and the potential there is negligible for high azimuthal modes.

Current results with lc_mantle = 0.04, m = 10:

- Relative L2 error over mantle: 0.33%
- Point-wise numerical/analytical ratio: 0.95 to 1.00 throughout the mantle
- integral(rho) = O(1e-18) (machine zero, no m=0 leakage)

## Analytical solutions (passess)

The `passess` package lives at `~/Workplace/passess` and provides `PoissonPolar2D` for 2D polar, `PoissonCartesian2D` for 2D Cartesian, and spherical harmonic solutions for 3D. It is not installed in the Firedrake venv; the test script adds it to sys.path.

The class interface:

    from passess.polar import PoissonPolar2D
    solver = PoissonPolar2D(m=10, rho_m=1.0, r1=r1, r2=r2, gamma=1.0)
    psi_at_r = solver.psi_m(r)          # radial mode coefficient
    psi_spatial = solver.to_spatial(r, phi)  # full spatial field (complex)

For real rho_m the spatial potential is Re[psi_m(r) exp(i m phi)] = psi_m(r) cos(m phi).

For m=0 (axisymmetric) the potential has a gauge freedom; pass r_ref to fix psi(r_ref)=0.

## Non-dimensionalisation

Matches the G-ADOPT 2D cylindrical demo: rmin=1.22, rmax=2.22, D=1.0. Physical mantle depth is 2891 km, so 1 non-dimensional unit = 2891 km. The density shell at 500 km depth, 50 km thick corresponds to:

    r_center = rmax - 500/2891 = 2.047
    r1_shell = r_center - 50/(2*2891) = 2.038
    r2_shell = r_center + 50/(2*2891) = 2.056

## Dependencies

- Firedrake (with gmsh Python bindings: `pip install gmsh` in the venv)
- passess (at ~/Workplace/passess, added to sys.path)
- MUMPS (for the direct LU solver; could switch to iterative)

## What came next

This standalone Poisson test validated the submesh coupling for the gravity equation in isolation, and the coupling to the momentum and viscoelastic internal variable equations has since been built: `gadopt/gia_gravity.py`, documented at the bottom of this file. Two things it changed about the sketch that used to stand here. The source is written as a divergence, `Lambda int rho_0 u . grad(v)`, and not as `-Lambda int rho_1 v` with jump terms, so every interface mass comes along automatically and discrete mass conservation is exact rather than `O(h^p)`. And the body force is `+rho_0 grad(psi)` with a **plus**, not `-rho grad(psi)`, because the shipped solver's psi is minus the Newtonian potential; the residual then carries its negative, since `momentum_equation.py` writes every term as if on the left-hand side.

---

# DtN boundary treatment: lessons learned

This section records what we found while replacing the extended-domain
Dirichlet approximation with Dirichlet-to-Neumann (DtN) boundary
conditions. See `ROADMAP-GRAVITY.md` for the plan; the experiments below
are its E1-E3, E5 and E7. Scripts, in build order:

- `gravity_poisson_robin.py` — E1: Dirichlet vs single-mode Robin, four-case
  matrix at m=3, plus the m-sweep helpers. Also home to the shared helpers
  (`curve_mesh`, `make_mantle_submesh`, `shell_density`,
  `mantle_relative_error`) the later scripts import.
- `gravity_poisson_m_sweep.py` — E1 continued: Dirichlet-vs-Robin gap vs m.
- `gravity_dtn.py` — the reusable `GravityPoissonSolver` class (2D modal
  DtN, interior + exterior, R-space multipliers, exact-source option).
- `gravity_poisson_dtn_modal.py` — E2: blind single-mode modal DtN, M-sweep.
- `gravity_poisson_dtn_multimode.py` — E2: superposed 3-mode density.
- `generate_gravity_annulus.py` + `gravity_poisson_dtn_annulus.py` — E3:
  interior DtN on a structured annulus, no centre disc.
- `gravity_poisson_exact_source.py` — exact-source vs DG0 A/B (the DG0
  floor is O(h_phi^2)).
- `gravity_poisson_convergence.py` — E5: 2D config-D convergence study,
  CG1 -> order 2, CG2 -> order 3.
- `gravity_poisson_3d_robin.py` — E7: 3D single-(l,m) Robin on the extruded
  cubed sphere, config D, optimal convergence.
- `generate_gravity_disc_structured.py` — SUPERSEDED early structured-mesh
  generator (Dirichlet-only outer boundary); use `generate_gravity_annulus.py`
  instead, which adds interior DtN, shell-conforming layers and the
  no-buffer config-D option. Kept only for reference.

- `gravity_poisson_sheet.py` — surface-sheet (sigma) validation: mass
  sheets ON the DtN boundaries via -4 pi gamma sigma v ds, against the
  passess SheetPolar2D delta-sheet solution.

Handover COMPLETE (2026-07-21): `gravity_dtn.py` (2D modal) and
`gravity_poisson_3d_robin.py` (3D Robin) are unified into
`gadopt/gravity_solver.py` — `GravitySolver` with
`CylindricalDtN`/`SphericalDtN` boundary-condition objects in the standard
bcs dictionary, real Y_lm from `gadopt/spherical_harmonics.py`, unit tests
in `tests/unit/`. All recorded numbers below reproduce through the class;
see GRAVITY-LESSONS-LEARNED.md part II for the unification mathematics and
the full record. The scripts in this directory are retained as the
reference implementations and gmsh-based cross-checks.

## Why DtN cannot improve the original m=10 test

The computed error is the sum of two independent sources: boundary
*truncation* error (how wrong the outer BC is) and *discretisation* error
(mesh resolving cos(m phi)). At R_grav = 10*rmax the truncation error for
m=10 is (r_s/R)^10 ~ 1e-10, so the 0.33% is pure discretisation. Replacing
Dirichlet by DtN there polishes 1e-10 to 0 and changes nothing visible.

DtN only *demonstrably* wins when the boundary is pulled in, so that
truncation error would otherwise dominate. Two knobs make it visible:
lower the mode (Dirichlet truncation ~(r_s/R)^m grows as m drops) and/or
lower the discretisation floor (CG2 + finer mesh) so the truncation error
is no longer masked. We use **m=3, CG2** for all DtN experiments.

## E1: single-mode Robin is exact at any radius

`dpsi/dr + (m/R) psi = 0` is the exact exterior condition for a single
mode; as one UFL term `(m/R)*psi*v*ds` it is symmetric, coercive, kills the
Neumann nullspace, and needs no solver changes (plain MUMPS LU). Results:

- Four-case matrix (m=3): truncating 10*rmax -> 2*rmax breaks Dirichlet to
  1.08e-2 (~1% boundary amplitude) but Robin recovers the 1.06e-4
  discretisation floor at the truncated boundary. On the 10*rmax mesh the
  two treatments are indistinguishable, confirming the error-budget claim.
- m-sweep at R=2*rmax: Dirichlet error tracks (r_s/R)^m exactly (22% at
  m=1!); Robin stays at the discretisation floor for every m. By m=10 the
  truncation error has dropped below discretisation and the two coincide —
  which is exactly why the original m=10 test never revealed any of this.

## Curved (P2) meshes: the coarse-boundary polygon fix

With `lc_exterior` large, gmsh renders the outer "circle" as a ~4-chord
polygon — harmless under Dirichlet (psi ~ 0 there) but it corrupts a Robin
term applied on that boundary. The fix is the isoparametric trick from the
cylindrical Stokes benchmarks (`curve_mesh`): interpolate a P2 coordinate
field that pushes edge midpoints radially onto the linear-interpolant
radius, so straight edges become quadratic arcs. Guard the origin with
`conditional(r > 1e-12, r_p1/r, 1.0)` (the disc has a centre; annuli don't).

- Effect at lc=0.04: modest (~10-15%), because the floor there is already
  field discretisation, not boundary geometry. Its real value is upstream:
  it decouples boundary-circle fidelity from resolution (coarse outer rings
  stop being polygons) and it is what makes a clean CG2 O(h^3) refinement
  study possible (linear facets cap convergence at O(h^2)).
- **Submesh does NOT inherit the parent's P2 coordinates.** A mantle
  submesh of a curved parent still reports the straight-sided ~2.5e-5 area
  error. Harmless at current floors; will surface in refinement studies.

## E2: the modal DtN and its solver pitfalls

For general (multi-mode) density the map is nonlocal; implement it with one
global R-space scalar per trace Fourier coefficient (`FunctionSpace(mesh,
"R", 0)`), defined by scalar constraint rows and fed back as boundary flux.
The solver is blind to the density — it *discovers* the modes. Findings:

- **Vector-valued R Arguments are unsupported** in this Firedrake
  (`firedrake/ufl_expr.py` raises `NotImplementedError` for `Argument on a
  vector-valued Real space`). So `VectorFunctionSpace(mesh, "R", 0,
  dim=2M)` cannot be an unknown; use 2M+ *separate scalar* R fields in the
  mixed space instead. (Noted upstream gap; scalar workaround is equivalent.)
- **Monolithic aij assembly is impossible with R blocks**
  (`firedrake/assemble.py:1407`). Must use a fieldsplit: full Schur
  complement eliminating onto the R fields, psi block via `AssembledPC` +
  MUMPS, dense R-R Schur by GMRES. This is what Firedrake's
  `solving_utils.set_defaults` auto-generates for Real blocks; we spell it
  out in the class so it is visible and tunable.
- **The naive section-6.2 form diverges.** Putting *all* boundary stiffness
  in the psi<->R coupling leaves the psi-psi block a singular pure-Neumann
  Laplacian; MUMPS returns garbage pivots (inner residual ~1e13) and the
  solve diverges at M>=2. The cure is a **Robin-shifted DtN**: write the
  flux as `-(alpha/R) psi + sum_m ((alpha - m)/R) c_m e_m` (alpha=1),
  mathematically identical for treated modes but with a pointwise Robin
  term that lands in the psi-psi block and makes it SPD. Bonuses: untreated
  modes > M see Robin(alpha/R) not Neumann; the exterior m=0 Robin is exact
  for zero-total-mass sources (boundary mean must vanish), so no separate
  monopole/mean multiplier is needed on the exterior; and at M=1 the scheme
  degenerates to exactly the validated E1 Robin term.
- Never read R coefficients via `.dat.data_ro`; cast the scalar-R
  subfunctions with `float(f)` (Firedrake's `Function.__float__` handles
  Real scalars). Raw `.dat` bypasses the ghosting/reduction machinery.

Results: single-mode M-sweep snaps from ~5e-3 (M<3) to the E1 floor
(1.06e-4, bit-identical to hand-coded Robin) at M>=3 and stays flat; solved
c_3 matches the analytical trace amplitude, other coefficients ~1e-11.
Superposed {2,3,5} density: error drops stepwise as M crosses each active
mode (ordered by amplitude), all three coefficients recovered, inactive
ones ~1e-11 — a genuine multi-mode validation, still exact by linearity.

## E3: interior DtN replaces the meshed-through centre

The same construction points inward: on a source-free core the interior
solution ~ r^m, giving `dpsi/dr - (m/R_in) psi = 0`, which enters the weak
form with the *same positive sign* as the exterior term. The interior m=0
is exact homogeneous Neumann (not Robin), so the Robin shift there is
undone by one mean multiplier. This matters at **low m** (interior field is
largest); it is where the coarse baseline disc could have polluted.

- **Mesh: gmsh, not Firedrake extrusion.** `Submesh` raises
  `NotImplementedError` on `ExtrudedMesh` (`firedrake/mesh.py:4964`), and
  all G-ADOPT production meshes are extruded. To keep the submesh route we
  build a *structured, extruded-style* annulus in gmsh:
  `generate_gravity_annulus.py` lays transfinite quad layers between
  explicit concentric radii — uniform fine layers through the mantle, one
  conforming layer for the density shell, geometric coarsening outside.
  Boundary IDs 1=outer, 2=inner. Geometry validates to 1e-11 (area, both
  circumferences) after `curve_mesh`; shell mass 1.5e-17.
- Results (annulus rmin->2*rmax, interior + exterior DtN, M=5): mantle
  error at the discretisation floor for every m in {1,2,3,5} with **no
  low-m penalty** — m=1 has the *smallest* error (3.2e-5). Coefficients at
  both boundaries match analytical to ~1e-4; inactive/mean multipliers at
  1e-14 (an order cleaner than the unstructured disc, from the structured
  mesh's near-perfect discrete orthogonality). This also retroactively
  clears the baseline's coarse inner disc: unnecessary, but not harmful.

## Exact source and the convergence study (E5)

`gravity_poisson_exact_source.py` (A/B) and `gravity_poisson_convergence.py`.

- **The DG0 density was the floor, and it is O(h_phi^2).** Sampling
  cos(m phi) as cell-wise constants is O(h) in L2, propagating to O(h^2) in
  the potential; the exact analytic source (integrated over the shell
  subdomain with high quadrature, `GravityPoissonSolver(source_expr=...,
  source_id=...)`) removes it entirely. A/B on the annulus, sweeping
  *azimuthal* resolution at fixed radial resolution: DG0 error falls as
  h_phi^2 (2.3e-4 -> 5.8e-5 -> 1.6e-5 for n_azim 256/512/1024) while the
  exact-source error stays flat at ~1.7e-6. So whenever the source has
  known analytic azimuthal structure, integrate it exactly -- it buys ~2
  orders. (In coupled GIA the density is a computed field with no analytic
  form, so there DG0 would cap you at O(h^2): argue for an azimuthally
  higher-order density representation when we get there.)
- **The extended-but-truncated buffer is itself the accuracy bottleneck.**
  With an exterior annulus rmax -> 2*rmax, a radial-band decomposition of
  the error showed it living almost entirely in that coarse buffer
  (100-200x the mantle error) and bleeding back through rmax, capping the
  mantle accuracy at ~1.5e-6 and reducing the convergence rate to ~1. The
  fix is **config D: exterior DtN directly at rmax, no buffer** (the
  exterior of rmax is source-free, so the map is exact there). This is the
  whole point of DtN -- do not keep a buffer you then under-resolve.
  `generate_gravity_annulus` builds the no-buffer mesh when
  `R_grav_factor <= 1`.
- **Clean convergence at last.** Config D, exact source, interior +
  exterior DtN, curved P2 geometry, error assessed in CG(deg+3) against the
  closed-form passess solution: the finest-pair observed order is **2.00
  for CG1 and 3.00 for CG2**, i.e. optimal O(h^{p+1}), with CG2 reaching
  6.8e-8 (no floor in range). Coarse levels are pre-asymptotic. passess
  psi_m is a closed-form expression (machine precision), so it never limits
  the study.

## 3D: single-(l,m) Robin on the extruded cubed sphere (E7)

`gravity_poisson_3d_robin.py`. The 3D entry point and, at the same time,
the first run on an actual Firedrake `ExtrudedMesh` (config D, no buffer).

- **Mesh mechanics (all validated in a smoke test first).**
  `CubedSphereMesh(radius=rmin, refinement_level=k, degree=2)` +
  `ExtrudedMesh(base, layers=N, layer_height=list(diffs),
  extrusion_type="radial")` with the radial node list conforming to
  r1_shell/r2_shell. Variable `layer_height` as a list works. Boundaries
  are `ds_t` (rmax) and `ds_b` (rmin); areas match 4 pi R^2 to ~1e-4 at
  refinement_level 2, improving with refinement. **No Submesh** -- the
  exact source (a radial `conditional` indicator, exact per cell because
  layers conform) removes the need, which is exactly why the extruded
  production mesh is usable here where the submesh route was not.
- **Single-mode Robin, no multipliers.** Exterior `(l+1)/rmax` on `ds_t`,
  interior `l/rmin` on `ds_b` (3D coefficients; both positive sign in the
  weak form). SPD for l >= 1, so plain MUMPS LU, no R-space, no nullspace.
  The class in `gravity_dtn.py` is 2D-specific (cos(m phi), m/R), so this
  is a dedicated script; 3D modal Y_lm (E8) is where the class gets
  generalised.
- **Y_lm as UFL.** For (l,m)=(2,0), the real orthonormal
  Y_20 = sqrt(5/(16 pi)) (3 (z/r)^2 - 1) matches scipy `sph_harm_y(2,0,...)`
  used by passess; a runtime assert checks passess `to_spatial/psi_lm`
  against it before trusting the comparison.
- **Result:** optimal convergence, observed order 2.0 (CG1) and 3.0 (CG2),
  validated against `passess.spherical.PoissonSpherical3D`.

## Production strategy note: DtN degree vs buffer (the L-R trade)

For the coupled problem the density is arbitrary (all degrees up to the
mesh's l ~ O(100s)); modal DtN needs ~(L+1)^2 blocks, so only low L (~4-5)
is affordable. The far-field residual of an untreated degree l is
(r_s/R_buf)^(l+1): fine for deep sources, but for shallow sources
(r_s ~ rmax, the dynamic-topography case) a thin buffer leaves several
percent (e.g. (0.83)^6 ~ 33% at l=5, R_buf=1.2 rmax). So the production
config is a **hybrid**: modal DtN for low L at a buffer edge, the buffer
sized so (r_s/R_buf)^(L+1) is below tolerance for the untreated tail. L and
R_buf trade off through that one formula; the single-mode config-D tests
validate the DtN itself, and a dedicated multi-degree experiment (the 3D
E2-analogue) must map the L-R_buf-residual surface before sizing the real
buffer. Config D (DtN at rmax) is the accuracy reference, not the
production geometry.

## The GravityPoissonSolver class (`gravity_dtn.py`)

`GravityPoissonSolver(mesh, rho=None, M=None, *, outer=(id, R),
inner=(id, R)|None, gamma, degree, quad_degree, source_expr=None,
source_id=None, source_degree=None)`. **2D only** (cos(m phi) basis, m/R
coefficients on both boundaries). Provide exactly one of:

- `rho` — a density Function on `mesh` or on a Submesh of it (cross-mesh
  via `intersect_measures` + dummy field, auto-detected); or
- `source_expr` — a UFL density expression integrated over the full mesh
  (optionally only subdomain `source_id`) with quadrature `source_degree`.
  No density field, no Submesh. This is the exact-source path (E5).

`quad_degree` defaults to `2*(M+degree)` — UFL's estimate for
`cos(M*atan2)` is unreliable, so it is set explicitly and checked by
`check_boundary_quadrature()` (asserts `integral cos(m phi)^2 ds = pi R`).
Both of those still describe this class accurately; **both were superseded
in `gadopt.GravitySolver` and should not be copied forward.** The default
is `O(M)` in the truncation where the quantity that decides whether a rule
resolves a mode is `M h / R`, so it keeps buying degree after the mesh has
stopped needing it; and asserting against the analytic `pi R` measures how
far the discrete boundary is from a circle, not whether the rule
integrates. The replacements are a mesh-aware calibrated default and a
self-convergence in the degree — see
`NOTES/FINDING-QUADRATURE-DEGREE-FORM.md` and
`NOTES/FINDING-QUADRATURE-CANCELLATION.md`. (In the shipped solver the
degree also sets the low-rank build's trace degree, where the same change
takes that build from `O(L^4)` to `O(L^2)`; this class has no low-rank
path, so that consequence is not one it shares.)
Solver defaults to the scalar-R Schur fieldsplit (Robin-shifted form).
`.solve()` returns psi; `.coefficients()` returns the per-boundary trace
Fourier coefficients (and the interior mean multiplier). SUPERSEDED as of
2026-07-21 by `gadopt.GravitySolver` (see the handover note above), which
generalises this class to 2D+3D, adds sheets/flux/Dirichlet through the
bcs dictionary, and measures boundary orientation and radius from the
mesh. This class stays as the validated 2D reference.

---

# The monolithic self-gravitating GIA solver

Added 2026-07-30, on branch `sghelichkhani/selfgravity`. Everything above this
line is the gravity-alone record and is still accurate for it. What follows is
Track 2 of `ROADMAP-GIA-SELFGRAV.md`: the viscoelastic mechanics, the
gravitational Poisson equation with its DtN treatment, and the rotational
closure, solved together in one mixed space and one Newton solve. The 2-D
prototype exists and has passed ten verification gates; the road map's §9.5
records their numbers and §9.6 records what they do not establish.

## What exists

    gadopt/dtn_form.py                        DtNGravityForm, the boundary
                                              treatment as a reusable form
    gadopt/gia_gravity.py                     self_gravitating_gia_space,
                                              SelfGravitatingGIASolver,
                                              rigid_rotation_nullspace
    demos/gravity/generate_selfgrav_annulus.py   the four-region parent mesh
    demos/gravity/validate_selfgrav_annulus.py   its acceptance gate, and
                                              `curve_mesh`
    demos/gravity/selfgrav_gia_annulus.py     the driver
    demos/gravity/spikes/gate_*.py            the verification gates
    tests/unit/test_gia_gravity.py            37 structural tests
    tests/unit/test_dtn_form.py               the extraction's own tests
    tests/unit/test_self_gravity_terms.py     the two body-force terms
    tests/unit/test_gravity_interior_sheet.py the interior-facet sheet

`DtNGravityForm` is `GravitySolver`'s boundary mathematics moved out one class,
with the same behaviour: every existing gravity test passes untouched behind a
shim of read-only properties that return the form's own objects, so
`solver.ds is solver.form.ds` and the two cannot drift. The 2-D monopole and
enclosed-mass bookkeeping deliberately did **not** move; it is the only taped
code in `GravitySolver` and it stays where its tests can reach it. The coupled
solver reimplements the datum on its own terms instead, because in the coupled
system the volume source contributes exactly zero mass and the sheets carry all
of it.

## The geometry, and its tags

    0.5 Rc ---- inner (101/102/103 = mantle/inner/buffer) ---- 2 Re

    r = 0.6019 --- inner --- 1.2037 --- mantle --- 2.2037 --- buffer --- 4.4074
        |                      |                     |                     |
     curve 5                curve 3               curve 2               curve 4
    interior DtN         Rc, INTERIOR          Re, INTERIOR          exterior DtN
                         facet of parent       facet of parent

Non-dimensionalised by D = Re − Rc = 2891 km. The mantle is extracted as
`Submesh(parent, 2, 101)`, straight off the gmsh cell tag — no `RelabeledMesh`,
which is only needed when the mesh carries no cell label. Curves 2 and 3 are
interior facets of the parent and *boundary* facets of the submesh, which is the
entire point of the stack: every existing mechanics form and boundary condition
works verbatim on the submesh, and only the two u↔ψ coupling terms are
cross-mesh. Tags follow the 3-D convention of road map §1.5 so nothing is
renamed on promotion.

## How to run it

Firedrake interpreter, `PYTHONPATH` at this worktree — the rules in the worktree
`CLAUDE.md` apply in full.

    PYTHONPATH=$(pwd) python demos/gravity/selfgrav_gia_annulus.py --gates

    --dr 0.1          radial spacing through the mantle
    --nazim 64        azimuthal cells, divisible by 4
    --truncation 5    DtN truncation M on both boundaries
    --dt 1.0          time step, in Maxwell times
    --steps 1         1 is a single backward-Euler step; more runs to steady state
    --rtol 1e-7       steady-state tolerance on the surface deflection
    --fluid-limit     the configuration that HAS a fluid limit; see below
    --lam-factor 1.0  scale Lambda, keeping B_mu
    --no-nullspace    do not declare the rigid-rotation kernel
    --no-rotation     drop the m_3 row
    --gates           run G0 and V4
    --output          two VTK files, one per mesh
    --monitor         SNES and KSP convergence

The verified relaxation, on the coarse mesh in about forty seconds:

    --dr 0.2 --nazim 32 --truncation 3 --dt 0.5 --steps 300 \
      --fluid-limit --lam-factor 0.25

reaching Airy isostasy `zeta = -sigma_hat/rho_0` in 187 steps, −9.9955e-04
against −1.0e-03.

The gates are separate scripts in `demos/gravity/spikes/`, not in the driver:
`gate_v1.py`, `gate_v2.py`, `gate_v3prime.py`, `gate_v7.py`, `gate_v8.py`,
`gate_v9.py`. Each states its expected numbers before it runs them.

## The traps

Every one of these was found by paying for it. Most are silent.

**Import `gadopt` before `from firedrake import *`.** Any script that reaches a
G-ADOPT python PC only through a solver-options string —
`pc_python_type: gadopt.DtNTwoBlockSchurPC` — makes PETSc import `gadopt`
lazily at `SNESSetFromOptions` time, long after a UFL multifunction has run.
Irksome's import-order guard fires, PETSc swallows it, and what you see is a
bare `petsc4py.PETSc.Error: error code 101` out of `SNES.setFromOptions` with
nothing in the traceback naming either package. One line at the top of the file
fixes it.

**Curve BOTH meshes.** `Submesh` does not inherit the parent's P2 coordinates,
so a submesh of a curved parent is straight-sided and reports the polygon error,
4.02e-04 relative on areas. `curve_mesh` on the submesh recovers the parent's
accuracy to the same digits, and the cross-mesh entity maps survive `Mesh(X_p2)`
— measured, 1.2e-08 against 4.02e-04 for a coupling-shaped integral. The error
the un-recurved version makes is concentrated exactly at Rc and Re, which is
where the interface mass sheets carrying the entire source live. `Submesh` does
not inherit `cartesian` either, and `is_cartesian` raises `AttributeError` on a
mesh that has none.

**An intersected measure that finds nothing assembles to zero**, with no
exception. And it is needed more often than the obvious rule suggests: in a
mixed space spanning two meshes the *arguments* carry both domains, so every
measure needs intersecting, including the parent-side Laplacian (which mentions
no submesh field) and the internal-variable source (which mentions no parent
field). The residual assembles happily without it; the failure arrives later,
when `AssembledPC` inside `DtNTwoBlockSchurPC` extracts the non-`Real` sub-block
and compiles that, again as `PETSc.Error: error code 101`, this time out of
`PCApply`. Diagnosing it means pulling the block out with `ExtractSubBlock` and
assembling its integrals one at a time. Intersecting the parent cell measure
does *not* restrict it to the mantle, and `check_geometry` asserts that.

**`ds` on an interior tag gives zero and a warning**, not an error — and so does
`dS` on an exterior tag. Both `exterior_facets.unique_markers` and
`interior_facets.unique_markers` return the whole physical-group label set
regardless of which facets each holds, so the tag tells you nothing and the
marker list cannot be trusted. A load sheet written the shipped exterior way is
simply absent: the solve converges, the potential is missing the largest single
term in the geoid, and no symmetry or Picard-consistency test looks, because a
sheet is a right-hand side with no Jacobian contribution and two solvers sharing
the form omit it identically. Hence the `interior_sigma` bc key, and hence
`check_sheet_measures` raising at construction. Write interior-facet integrands
with `avg`, never `'+'`: restriction sides are consistent only by gmsh's cell
ordering.

**`snes_atol = 1e-10` silently returns the zero solution for a small load.**
`newton_stokes_solver_parameters` sets an *absolute* SNES tolerance. A
configuration whose entire forcing is below it converges at iteration zero on
every step and returns exactly 0.0 forever, reporting `SNES converged`, with no
warning. This is a general trap for any gate that scales a load down to keep a
problem linear, and it cost a full sweep. It has a live consequence too: near a
steady state the per-step residual falls, and once it falls below 1e-10 the time
loop freezes rather than converging.

**A sheet amplitude that folds to UFL `Zero` raises a misleading error.**
`sigma_hat * cos(2 phi)` at `sigma_hat = 0.0` is `Zero`, the enclosed-mass form
then has no integrals, and `update_total_mass`'s `solve(identity == mass, ...)`
raises `ValueError: Provided RHS is not a linear form` on the first step, from a
traceback that names `firedrake/variational_solver.py` and nothing about sheets.
Use a tiny nonzero amplitude, or guard the empty form.

**Per-block symmetry measurements are serial, and an unguarded parallel run
deadlocks.** `getNestSubMatrix(...).convert("dense")` builds a global dense
block; at two ranks one rank fails its local comparison and the other waits in
the next collective. Mark them serial. The parallel instrument is matfree
`mult` against `multTranspose`, which is as sharp and gives one number with no
block attribution.

**Declaring a nullspace is not enough on this solver.** FGMRES is
right-preconditioned, PETSc removes the kernel from the right-hand side but not
from the preconditioner's output, and `DtNTwoBlockSchurPC` is nearly an exact
inverse here — the outer solve converges in one iteration — so the answer *is*
the preconditioner's output, kernel and all. `project_out_nullspace()` after
each solve is what actually removes it.

## The one physics trap

**The production configuration has no fluid limit, and self-gravity is not
why.** Stepped past a few Maxwell times the surface deflection grows
exponentially, and so does a plain `CoupledInternalVariableSolver` on the same
mantle with the same load and no coupling at all, at half the rate. The cause is
the reference state: a uniform density in a constant gravity field is not a
hydrostatic equilibrium, and the operator linearised about it has a growing
mode with an e-folding time of about fifty Maxwell times — invisible over the
ten a glacial cycle runs, fatal to a gate that steps to a hundred.
`--fluid-limit` is the configuration that does relax: `g = 0` plus an explicit
Airy restoring stress at Re. Road map §2.5 has the rest, including why the
degree-one mode then goes unstable above `--lam-factor 0.8` and why projecting
out a translation does not help.
