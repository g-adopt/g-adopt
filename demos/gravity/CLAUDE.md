# Gravity Poisson Equation with Submesh Coupling

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

## What comes next

This standalone Poisson test validates the submesh coupling for the gravity equation in isolation. The next step is coupling it to the momentum and viscoelastic internal variable equations as in Dale's `submesh_2way_coupling.py`, where the density perturbation rho1 = -u dot grad(rho) - rho div(u) comes from the displacement field and the gravitational potential feeds back as a body force -rho grad(psi) in the momentum equation.

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
is `O(L)` in the truncation where the quantity that decides whether a rule
resolves a mode is `L h / R`, which is the difference between an `O(L^4)`
and an `O(L^2)` boundary build; and asserting against the analytic `pi R`
measures how far the discrete boundary is from a circle, not whether the
rule integrates. The replacements are a mesh-aware calibrated default and
a self-convergence in the degree — see `NOTES/FINDING-QUADRATURE-DEGREE-FORM.md`
and `NOTES/FINDING-QUADRATURE-CANCELLATION.md`.
Solver defaults to the scalar-R Schur fieldsplit (Robin-shifted form).
`.solve()` returns psi; `.coefficients()` returns the per-boundary trace
Fourier coefficients (and the interior mean multiplier). SUPERSEDED as of
2026-07-21 by `gadopt.GravitySolver` (see the handover note above), which
generalises this class to 2D+3D, adds sheets/flux/Dirichlet through the
bcs dictionary, and measures boundary orientation and radius from the
mesh. This class stays as the validated 2D reference.
