# Lessons learned: DtN boundary treatment for the gravity Poisson equation

Companion to `ROADMAP-GRAVITY.md`. That document says what we planned; this
one records what we found out while doing it, so nobody has to rediscover
these the hard way.

**Part I** (sections 1-6) was established on the m = 3 shell benchmark
against `passess.polar.PoissonPolar2D`, in the scripts
`gravity_poisson_robin.py`, `gravity_poisson_m_sweep.py`, `gravity_dtn.py`
and `gravity_poisson_dtn_modal.py`, during roadmap Steps 1-2; the E3-E7
findings that followed are condensed in `CLAUDE.md` ("DtN boundary
treatment: lessons learned").

**Part II** (sections 7-13) covers the gadopt integration (roadmap Step 7,
2026-07-21): the unification mathematics behind `gadopt/gravity_solver.py`,
the surface-sheet (dynamic topography) term, the real-Y_lm construction,
the second batch of Firedrake facts, the full validation record through
the new class, and what remains open. Status: the solver is shipped
(commits `b8112d66`, `565adb47`, `d8c160cb`; passess `6cf46f1`; sheet
experiment `4cf8ce6b`); only the E6 cost study and the E8 L-vs-buffer
trade study remain from the original experiment matrix.

## 1. The error budget, and why "add DtN and it gets more accurate" is wrong

The computed solution carries two independent errors: boundary truncation
(how wrong the condition at the artificial boundary is) and discretisation
(mesh resolution). For a mode-m source at radius r_s, the truncation error
of a homogeneous Dirichlet condition at radius R scales as (r_s/R)^m. On
the baseline mesh (R = 10·rmax = 22.2, m = 10) that is ~4e-11: completely
buried under the 0.33% discretisation error. Adding DtN there changes
nothing measurable. The value of DtN is that it holds the truncation error
at zero while the boundary comes *in* — the correct demonstrations are
either "same accuracy with the boundary at 2·rmax and the exterior DOFs
deleted" or "visible improvement at low m, where (r_s/R)^m is large even
for distant boundaries". We used m = 3, where Dirichlet at 2·rmax is wrong
at the 10% level.

Measured (CG2, lc_mantle = 0.04, curved mesh, m = 3): Dirichlet at
R = 22.2 and Robin at R = 22.2 both give 1.1e-4; Dirichlet at R = 4.44
gives 1.1e-2; Robin at R = 4.44 gives 1.1e-4. The m-sweep at R = 4.44
follows the (r_s/R)^m law exactly: the Dirichlet/Robin error ratio is
~3900 at m = 1, ~600 at m = 2, ~90 at m = 3, ~3 at m = 5, and 1.0 at
m = 10 — the last row being precisely why the original m = 10 test could
never have seen any of this.

## 2. The curved-mesh (superparametric P2) trick

Linear meshes approximate every circle by chords: with lc_exterior = 50
the outer boundary of the baseline mesh is essentially a 4-chord polygon.
Harmless under Dirichlet (psi ≈ 0 there anyway), fatal for any boundary
condition that uses the boundary's radius. The fix, borrowed from the
cylindrical Stokes benchmarks, is a P2 coordinate remap: interpolate
`(x,y)/r * r_p1` into a CG2 vector space, where r_p1 is the CG1
interpolant of r, and wrap it in `Mesh(...)`. Midpoint nodes land on the
circle through their edge's endpoints, so all mesh-conforming circles
become piecewise-quadratic arcs (boundary geometry error O(h^4) instead of
O(h^2)). Two findings:

- On a *disc* mesh the remap needs a guard at the coordinate singularity:
  `conditional(r > 1e-12, r_p1/r, 1.0) * X` (the Stokes benchmarks are
  annuli and never hit this).
- `RelabeledMesh`/`Submesh` and the cross-mesh `intersect_measures`
  coupling all work on the P2-coordinate mesh. This was not guaranteed and
  is worth knowing.

At lc_mantle = 0.04 the accuracy gain is only 10–15% — the floor there is
field resolution, not geometry. The trick matters for what comes next: a
clean O(h^3) CG2 refinement study (E5) is impossible with linear facets,
and coarse outer boundaries are only legitimate with curved geometry.

## 3. Firedrake facts about R spaces (the hard-won ones)

1. **A vector-valued Real space cannot be an unknown.** `Argument.__init__`
   (`firedrake/ufl_expr.py:43`) raises `NotImplementedError` for Real
   spaces with non-empty shape. The roadmap's sketch of packing all 2M
   coefficients into one `VectorFunctionSpace(mesh, "R", 0, dim=2M)` is
   therefore not implementable: each multiplier must be its own scalar
   `FunctionSpace(mesh, "R", 0)` field in the MixedFunctionSpace. All
   fields are the same space object type; the cost is per-field
   bookkeeping in the mixed space, which is why we keep M modest (≤ 5)
   for now.
2. **Monolithic (aij) assembly is unsupported with R blocks.** The sparsity
   builder rejects it outright (`assemble.py`, "Monolithic matrix assembly
   not supported for systems with R-space blocks"; there is a regression
   test asserting this failure). Forcing `mat_type: aij` + LU — the
   obvious thing to try — can never work. When R blocks are present
   Firedrake's default `mat_type` becomes `nest`.
3. **Firedrake already ships the right solver strategy.** With no solver
   parameters, `solving_utils.set_defaults` detects Real blocks and
   generates: matfree fgmres, fieldsplit with a *full Schur complement
   eliminating onto the R fields*, the PDE block applied via `AssembledPC`
   and factorised by MUMPS, the tiny dense R Schur complement handled by
   GMRES/`pc_type none`. `GravityPoissonSolver` spells this configuration
   out explicitly (visible, tunable) rather than relying on the
   warning-emitting default.
4. **Read R values with `float(f)`**, which is supported for scalar Real
   Functions (`function.py`), not by poking `.dat.data`.
5. A silly one that cost a debugging round: naming a keyword argument
   `inner` shadows `ufl.inner` inside the method. The class keeps the
   `inner=` API and simply avoids calling `inner()` in that scope.

## 4. The coupling finding: the naive modal DtN is solver-hostile, the Robin-shifted form is not

This is the important one. Write the modal DtN the obvious way —
constraint rows defining the trace Fourier coefficients c_m, plus flux
feedback Σ (m/R) c_m e_m — and every bit of boundary stiffness lives in
the psi↔R *coupling* blocks. The psi–psi diagonal block is then a pure
Neumann Laplacian: **singular**. The Schur fieldsplit hands exactly that
block to MUMPS, which factorises it with garbage pivots; the symptom is
inner residuals of order 1e13, an outer Krylov that limps (40 iterations
at M = 1) and diverges outright at M ≥ 2. Nothing is wrong with the
mathematics; the *placement* of the terms breaks the solver.

The cure is an exact algebraic rearrangement, the **Robin-shifted modal
DtN**. For any shift α, split the flux as

    dpsi/dn ≈ −(α/R) psi + Σ_{m=1..M} ((α − m)/R) c_m e_m(φ).

For the treated modes the two pieces recombine to the exact map — the
formulation is unchanged where it matters. But the pointwise Robin term
(α/R)ψv now sits in the psi–psi block, which becomes SPD; after the fix
the fieldsplit converges in a handful of iterations at any M we tried.
Three side effects, all benign or better:

- Untreated modes (> M) feel Robin(α/R) instead of homogeneous Neumann —
  a decaying far field rather than a reflecting one; no worse, arguably
  better.
- The exterior m = 0 also feels the Robin term, which for zero-total-mass
  density is *exact* (the boundary mean of psi must vanish). The separate
  monopole/mean-pinning multiplier of the original design became
  unnecessary and was deleted, and the constant nullspace is gone for
  free. (Nonzero total mass still needs the −2γM/R flux datum; not yet
  implemented.)
- With α = 1, the M = 1 system *is* the validated single-mode Robin
  scheme, which gives a built-in consistency check between the two
  implementations (they agree to the printed digit).

On an interior DtN boundary the situation flips: the exact interior
monopole condition is homogeneous Neumann, so there the shift must be
*undone* on the mean — one extra R multiplier holding the trace mean,
subtracted as (α/R)c₀ from the flux. Implemented in `gravity_dtn.py`,
untested until an annulus mesh exists (E3).

The general lesson, which will apply verbatim to the coupled
Stokes–Poisson system and to the other R-multiplier constructions
(mean-free h, solution-dependent monopole): **when a global multiplier
carries stiffness that the eliminated block needs for invertibility, move
a pointwise part of it onto the diagonal and let the multiplier carry only
the correction.** Check the diagonal block's invertibility *before*
blaming the solver.

## 5. Validation results to reproduce

Blind modal DtN (density m = 3, solver not told; R = 2·rmax, CG2,
lc_mantle = 0.04, curved mesh; `gravity_poisson_dtn_modal.py`):

| M | rel L2 (mantle) | c_3 / exact | max other coeff |
|---|---|---|---|
| 1 | 5.23e-3 | — | 6e-14 |
| 2 | 5.23e-3 | — | 4e-12 |
| 3 | 1.0639e-4 | 0.999843 | 1e-11 |
| 4 | 1.0639e-4 | 0.999843 | 2e-11 |
| 5 | 1.0639e-4 | 0.999843 | 2e-11 |

The M ≥ 3 rows are bit-identical to the single-mode Robin run — the
truncation knob works, the solver finds the mode, and surplus modes cost
nothing and stay at machine zero. Boundary quadrature is set explicitly
(default 2(M + degree)) and verified by `check_boundary_quadrature`,
which asserts ∫cos²(mφ) ds = πR for all treated modes.

That describes `gravity_dtn.py`'s own method and still does, accurately —
the prototype is unchanged. But **the technique is now known not to
measure what its name claims**, and a reader should not carry it
forward: comparing against the analytic πR is dominated by the discrete
boundary not being a circle, so it reports boundary shape rather than
whether the rule integrates. Both the prototype's default and its check
were superseded in `gadopt.GravitySolver`; see section 8 and
`NOTES/FINDING-QUADRATURE-CANCELLATION.md`.

## 6. Open items part I did not cover

(All of these are now done and recorded below or in `CLAUDE.md`, except
the nonzero-total-mass monopole datum.) Multi-mode superposed density
(E2); interior DtN on an annulus (E3); extruded-mesh smoke test for R
spaces and the production configuration D (E4); everything 3D.

---

# Part II: the gadopt GravitySolver (roadmap Step 7)

Everything below was established while unifying `gravity_dtn.py` (2D
modal) and `gravity_poisson_3d_robin.py` (3D Robin) into
`gadopt/gravity_solver.py`, validating the surface-sheet term, and
reviewing the result (one companion math review that re-derived every
closed form, one independent adversarial review; both approved, and every
confirmed finding is folded in below).

## 7. The unification: every DtN boundary is a mode table

The load-bearing observation: in every geometry, a truncated DtN boundary
is fully described by the Robin shift alpha plus a list of modes, each a
tuple

    (e_k, lambda_k, N_k)  =  (angular eigenfunction as a UFL expression,
                              DtN eigenvalue, analytic value of
                              integral(e_k^2 ds)),

and the entire Robin-shifted machinery of section 4 becomes geometry-blind:

    F += (alpha/R) psi v ds                                # shift
    for each mode k:
        F += (psi e_k - (N_k/|Gamma|) c_k) mu_k ds         # constraint row
        F += (lambda_k - alpha/R) c_k e_k v ds             # flux feedback

The constraint scaling N_k/|Gamma| makes c_k the true trace coefficient of
e_k: 1/2 for cos/sin(m phi) (pi R over 2 pi R), 1 for the boundary mean,
1/(4 pi) for orthonormal Y_lm (R^2 over 4 pi R^2). The mode tables are:

| | exterior side | interior side |
|---|---|---|
| `CylindricalDtN(M)` | cos/sin(m phi), lambda = m/R, N = pi R, m = 1..M; **m = 0 excluded** (log monopole; see 7.2) | the same, **plus** the mean mode (1, 0, 2 pi R) |
| `SphericalDtN(L)` | Y_lm, lambda = (l+1)/R, N = R^2, l = 0..L, **l = 0 included** | Y_lm, lambda = l/R, N = R^2, l = 0..L |

Three collapses fall out, all verified term-by-term against the validated
prototype:

1. **The interior monopole is not a special case.** The hand-coded
   "mean multiplier undoing the Robin shift" (`gravity_dtn.py:172-179`) is
   exactly the lambda = 0 mode of the generic loop: constraint row defines
   the trace mean, feedback -(alpha/R) c_0 is the shift-undo. In 3D the
   interior l = 0 (lambda = l/R = 0) plays the same role automatically —
   zero special-cased lines in the class.
2. **The exterior 2D m = 0 stays excluded; the exterior 3D l = 0 is a
   genuine mode.** In 2D the exterior monopole is logarithmic (trace does
   not determine flux), so the mode list omits it and the Robin shift is
   exact under zero net mass. Nonzero net mass is supplied separately, as
   the flux datum -2 G M / R (see 7.3). In 3D the l = 0 exterior
   decays as 1/r and belongs in the table with lambda = 1/R. At alpha = 1
   its feedback coefficient (lambda - alpha/R) vanishes — the pointwise
   Robin term alone is the exact monopole map — but the constraint row
   keeps a nonzero diagonal (-scale |Gamma|), so the R-block Schur
   complement stays nonsingular and the multiplier survives as a pure
   diagnostic (the monopole trace coefficient). The adversarial review
   attacked exactly this corner and confirmed well-posedness; the
   interior-monopole unit test exercises it.
3. **Single-mode Robin is the truncation-zero degenerate case.** With
   M = 0 (or L = 0) the mode list is empty (or l = 0-only) and the scheme
   reduces to the pointwise Robin term: SPD, no multipliers, plain LU.
   There is no `boundary_treatment` enum and no separate "robin" mode —
   one formulation, one knob. Likewise the exact-source option of the
   prototype collapsed into `rho` accepting any UFL expression (a Function
   IS an expression), with `source_quad_degree` as the only extra knob.

## 8. DtN as boundary conditions: the API decision

The map is a property of a boundary, not of the solver. The class
therefore follows the StokesSolver/EnergySolver contract exactly:

    solver = GravitySolver(psi, rho, bcs={
        "top":    {"dtn": SphericalDtN(L=4), "sigma": sheet_expr},
        "bottom": {"dtn": SphericalDtN(L=4)},
    })

- `"psi"` (strong Dirichlet), `"flux"` (prescribed d psi/dn), `"dtn"` and
  `"sigma"` ride one dictionary; an unspecified boundary keeps natural
  Neumann (the user's responsibility, as everywhere in G-ADOPT). The
  config-A Dirichlet reference is `{"psi": 0}` through the same interface
  — no special casing, and it reproduces the recorded 1.08e-2 truncation
  error at R = 2*rmax.
- **Orientation and radius are measured, not declared.** The sign of
  assemble(dot(n, x) ds) over the marked boundary decides exterior
  (positive: domain inside the boundary) vs interior; the radius is the
  facet mean of r, which on curved/discrete boundaries is also the
  consistent discrete radius. A boundary deviating from constant radius
  (rms > 1e-3 R) warns but solves at the mean radius; DtN is only valid
  on origin-centred coordinate circles/spheres and the docstring says so.
- **Consistency of measured R with analytic constraint scales** (the
  adversarial reviewer's sharpest question): the hardcoded scales (1/2,
  1, 1/(4 pi)) with measured R are self-consistent because c_k is
  *defined* by the constraint row. The measured R enters only lambda
  and N.
  This used to add "and `check_boundary_quadrature` bounds the deviation
  of assemble(e_k^2 ds) from N_k — which for Y_lm doubles as a
  discrete-orthonormality test for free". **That second half is no longer
  true, and the reason it stopped being true is the more useful lesson.**
  The deviation from N_k is dominated by the discrete boundary not being
  a sphere, not by quadrature: on a level-2 extruded cubed sphere at
  L = 8 it moves from 2.328711e-05 to 2.328713e-05 as the degree goes
  from 12 to 40, while one refinement level moves it nearly two orders
  (2.33e-05 to 4.98e-07, a factor of 47). So it
  bounded boundary sphericity while carrying a name and a docstring
  claiming quadrature. `check_boundary_quadrature` now differences two
  degrees on one mesh, which cancels the geometry error because it is
  common to both, and its return value is a self-convergence rather than
  a deviation from N_k. Sphericity is still measured, by the rms radius
  warning in `set_boundary_geometry` above, which is the direct
  instrument for it. The free orthonormality check is genuinely lost:
  nothing now compares assemble(e_k^2 ds) against N_k, and folding it
  back in would put geometry error into the one instrument built to
  exclude it. See `NOTES/FINDING-QUADRATURE-CANCELLATION.md`.
- **The mixed space is internal** (psi block + optional cross-mesh dummy +
  one scalar R field per mode): its size depends on the bcs, so it cannot
  predate the solver. The user's psi is assigned back after each solve,
  like EnergySolver updating T. `solver.coefficients()` returns the trace
  spectrum per boundary keyed by mode label ("cos3", "Y2,-1", "mean") —
  the geoid coefficients when read at the surface.
- **rho is a single argument**: Function, UFL expression, or number. If
  its mesh differs from psi's (a Submesh), the intersect_measures +
  dummy-field coupling engages automatically. This is not test scaffolding:
  in the dynamic-topography configuration the mantle fields live on a
  submesh of the extended gravity mesh, so cross-mesh IS the coupling
  geometry. Verified by a consistency test (same-mesh vs Submesh paths
  agree to 1e-8) and by the E2 reproduction.

## 9. Surface sheets on DtN boundaries: mathematics and validation

Roadmap 5.4 in practice. A sheet sigma on a DtN boundary enters through
the flux jump [d psi/dr] = -4 pi G sigma. Deriving the weak form on BOTH
boundaries (outward normal +r_hat at the outer, -r_hat at the inner) gives
the SAME single extra term,

    F -= 4 pi G sigma v ds(marker),

with the modal DtN machinery completely untouched — the constraint rows
still define trace coefficients, the feedback is unchanged. Validated
blind (M = 5, solver not told the mode) against the new passess
delta-sheet solutions on the config-D annulus
(`gravity_poisson_sheet.py`):

| sheet position | m | rel L2 | c_m/exact | max inactive |
|---|---|---|---|---|
| outer (rmax) | 1 | 7.1e-12 | 1.00000000 | 2e-13 |
| outer | 2 | 3.3e-8 | 1.00000000 | 2e-13 |
| outer | 3 | 1.4e-7 | 1.00000000 | 1e-13 |
| outer | 5 | 7.5e-7 | 1.00000000 | 4e-14 |
| inner (rmin) | 1-5 | 7.9e-8 .. 4.1e-6 | 1 to 8 digits | ~1e-13 |

The m = 1 outer row is near machine precision because the domain-side
solution of an outer sheet is the harmonic polynomial ~ r cos(phi),
exactly representable in CG2 — a useful smoke case. The analytic
references are `passess.polar.SheetPolar2D` and
`passess.spherical.SheetSpherical3D` (thin-shell limits of the volumetric
classes; closed forms, jump conditions and DtN relations all unit-tested
in passess itself):

    2D, m != 0:  psi_m(r) = (2 pi gamma sigma a / |m|) (r_</r_>)^|m|
    2D, m == 0:  psi_0(r) = -4 pi gamma sigma a ln(r_>)  (+ gauge)
    3D:          psi_lm(r) = (4 pi G sigma a^2 / (2l+1)) r_<^l / r_>^{l+1}

Two corollaries worth remembering:

- **The monopole datum must include sheet mass, and flux-implied mass.**
  The 2D exterior monopole is set by the TOTAL enclosed mass: volumetric,
  plus sheets, plus the cavity mass `int g ds / (4 pi G)` implied by any
  prescribed-flux boundary. The last of those was missed by the source
  analysis and costs a mesh-independent 85% if omitted. `check_net_mass`
  no longer refuses nonzero mass - the datum handles it - and warns only
  in the band 1e-8 < |M|/scale < 1e-4, where a mass is nonzero but too
  small to look deliberate. If it fires on a density that should be
  zero-mean, suspect DG0 leakage from
  density boundaries that do not conform to cell edges — the original
  230%-error lesson — which is an accuracy problem in its own right.
- **In 3D, nonzero total mass is legal and tested.** A uniform (Y_00)
  sheet on the inner boundary — psi = 4 pi G sigma a^2 / r, the shell
  theorem — exercises the interior l = 0 mean multiplier and the exterior
  l = 0 trace map simultaneously (`test_interior_monopole`).

## 10. Real Y_lm as UFL: the Legendre-derivative construction

`gadopt/spherical_harmonics.py` avoids both hand-written tables and
runtime sympy expression trees. With u = z/r and the solid-harmonic
factors A_m + i B_m = (x + i y)^m (built by the two-term recurrence),

    Y_l0 = N_l0 P_l(u),
    Y_lm = sqrt(2) N_l|m| (d^|m| P_l / du^|m|)(u) * (A or B)_|m| / r^|m|,

with N_lm = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!). Points that were verified
rather than assumed:

- **The Condon-Shortley phases cancel exactly** in this form:
  (1-u^2)^{m/2} d^m P_l/du^m = (-1)^m P_l^m, so the expression equals
  sqrt(2) (-1)^m Re/Im[Y_l^m(scipy)] with no leftover sign. Confirmed
  against `scipy.special.sph_harm_y` to 2e-15 for all modes l <= 6, and
  by both reviews independently.
- sympy is used only to generate the Legendre-derivative coefficients
  (cached per (l, m)); the returned expression is plain UFL — polynomial
  in (x, y, z) and 1/r — so it differentiates, tapes under pyadjoint, and
  is homogeneous of degree zero (radius-independent; tested on a
  non-unit sphere).
- Full Gram matrix orthonormality to 1e-14; UFL-vs-numpy agreement on a
  cubed sphere to 3e-15. The numpy twin
  (`real_spherical_harmonic_numpy`) exists precisely so error assessments
  never re-derive the convention.
- Pole safety: for m >= 1 the azimuthal factor is A_m/r^m = 0 (not 0/0)
  on the z-axis; nothing is singular for r > 0. Factorial overflow would
  appear only beyond l + |m| ~ 170, far past the multiplier-count warning
  (n > 200, i.e. L ~ 13).

## 11. Firedrake facts, second batch

1. **R spaces work on extruded meshes.** The smoke test the roadmap
   deferred is done in anger: 3D modal DtN (18-32 scalar R fields) on the
   radially extruded cubed sphere, and 2D modal DtN on a radially extruded
   `CircleManifoldMesh` annulus (the production GIA construction), both at
   optimal order with inactive coefficients ~1e-14. Config D therefore
   runs on the unmodified production meshes in both dimensions.
2. `ufl_element().degree()` returns a (horizontal, vertical) TUPLE on
   extruded tensor-product elements; take max() before arithmetic.
3. G-ADOPT's `CombinedSurfaceMeasure` resolves "top"/"bottom"/integer
   markers uniformly, so one `self.ds(marker)` covers gmsh ids and
   extruded boundaries; `DirichletBC` accepts "top"/"bottom" directly.
4. **A full `solver_parameters` override can silently discard the
   mandatory R-block fieldsplit** (the adversarial reviewer's should-fix):
   monolithic aij assembly is impossible with R blocks, so the class now
   rejects `mat_type: aij` overrides with a pointer to
   `solver_parameters_extra`. Partial tuning belongs in
   `solver_parameters_extra`; full replacement is for expert fieldsplit
   configs only.
5. The solver integrates `SolverConfigurationMixin` like the other
   solvers (options_prefix "Gravity", update callbacks); defaults are the
   validated full-Schur-onto-R fieldsplit (AssembledPC + MUMPS on the PDE
   block, GMRES/none on the tiny dense R Schur) when multipliers exist,
   plain MUMPS LU otherwise.

## 12. The validation record through gadopt.GravitySolver

Every recorded prototype number reproduces through the new class:

| benchmark | recorded (prototype) | through GravitySolver |
|---|---|---|
| E2 M-sweep floor (M >= 3) | 1.0639e-4, c3/exact 0.999843 | 1.063867e-4, 0.999843 (all printed digits) |
| E2 below truncation (M < 3) | 5.23e-3 | 5.232587e-3 |
| E1 Dirichlet at 2*rmax | 1.08e-2 | 1.0767e-2 via `{"psi": 0}` |
| E3 annulus m = 1 | 3.2e-5 | 3.181e-5; inactive ~1e-14 (cleaner than the recorded 1e-11 disc run) |
| E5 orders (finest pair) | 2.00 / 3.00, CG2 6.8e-8 | 2.00 / 3.00, 6.8239e-8 |
| E7 3D orders | 2.0 / 3.0 (hand Robin) | 2.03, 1.92 / 3.00 via full modal SphericalDtN(L=2) |
| 2D sheet outer m = 3 | 1.3737e-7 (prototype script) | 1.3737e-7 (bit-identical) |

Test inventory **as it stood at Step 7** (2026-07-21):
`tests/unit/test_gravity_solver.py` (13 tests: geometry inference, bcs
validation, quadrature guard, monolithic-override guard, net-mass guard,
sheets on both boundaries, volumetric shell, manufactured Dirichlet/flux,
cross-mesh consistency, 3D sheet, 3D interior monopole) and
`tests/unit/test_spherical_harmonics.py` (4 tests). All references are
inline closed forms; no gmsh, no passess, ~45 s total — CI-safe. The
passess cross-checks stay in `demos/gravity/` and the passess suite
itself (62 sheet tests among 397).

Two of those entries have since been overtaken and the count with them.
`test_gravity_solver.py` now collects 37, and the **net-mass guard is
gone as a guard**: the 2-D monopole datum is implemented, so the tests
that asserted a `NotImplementedError` now assert that the mass is
carried. What is still refused is narrower — a strong `psi` condition
alongside nonzero mass, and two exterior DtN boundaries on one 2-D mesh.
The **quadrature guard** is likewise a different instrument now (section
8 above). Dated rather than rewritten, because this section is a record
of what Step 7 shipped; for the current inventory read the suite.

## 13. Open items after part II

- **E6** (cost, A vs D at matched resolution) — the only unstarted
  experiment.
- **E8 production sizing** — machinery done; the L-vs-R_buf residual
  surface and assembly-cost scaling (O(L^2 N_facets)) still to be mapped
  before choosing production L. The scalar-R bookkeeping (vector-valued R
  Arguments still unsupported upstream) caps practical L at ~5, which is
  what the hybrid low-L-plus-buffer strategy assumed anyway.
- **2D nonzero-net-mass monopole datum** (-2 G M / R flux): implemented
  on both DtN representations, with the enclosed mass carried in taped
  `Real` functions rather than the R multiplier once envisaged, and the
  potential returned in the gauge `int psi ds = 0` on the exterior
  boundary. See `NOTES/PLAN-MONOPOLE-C0.md`. What remains open is the
  *solution-dependent* monopole of roadmap section 12.3, where the
  enclosed mass is a function of the Stokes unknowns and has to be
  symbolic in a monolithic residual.
- **Cartesian** (`CartesianDtN` for periodic boxes): a pure addition to
  the mode-table abstraction — cos/sin(k_n x), lambda = k_n both sides,
  plus the n = 0 sheet-mass datum.
- **The coupled self-gravitating extraction** (a function taking e.g. a
  mantle solver and returning the coupled system): deliberately deferred;
  `solver.F` and `solver.mixed_space` are exposed for it, and every term
  is plain UFL so SNES/pyadjoint see through the whole construction.
