# Road map: monolithic Stokes–Poisson coupling with mean-free dynamic topography

## 1. Who this document is for

You are a software engineer who has just joined the G-ADOPT project. You know
Python and a bit of numerics, but you do not know geodynamics, you have never
heard of Firedrake, and you have never touched PETSc. This document walks you
through what the problem is, why it is a problem, what the fix looks like, and
how to land it. By the end you should have a clear ticket-to-PR path.

Relevant files in this worktree:

- `gadopt/stokes_integrators.py` — the Stokes solver and the current
  `BoundaryNormalStressSolver` class. The thing we are going to change lives
  here.
- `demos/gravity/submesh_2way_coupling.py` — Dale's prototype that already
  wires Poisson for gravity into a single mixed system. It is the structural
  template for the coupled solver we want.
- `tests/analytical_comparisons/delta_cylindrical_zeroslip.py` — the
  convergence test for the radial-stress field. This is our yardstick.

## 2. The physics in one paragraph

G-ADOPT solves mantle convection: viscous Stokes flow inside a spherical
shell driven by buoyancy. On the Earth's outer surface the normal component
of the stress tensor, divided by a density-gravity factor, is what geodynamicists
call *dynamic topography* `h` — it tells you how much the surface would bulge
if it were free to move. In the coupled problem we also solve Poisson's
equation for the perturbation gravitational potential `Φ`; `h` enters that
Poisson problem on the right-hand side as a surface mass anomaly.

Two physical facts drive everything below:

1. The mean radius of the Earth does not change. Mass is conserved. So the
   dynamic topography on each closed boundary must integrate to zero:
   `∫ h dS = 0` on the top, and separately on the CMB.
2. Gravity is coupled back into the momentum equation as a body force,
   so Stokes and Poisson must be solved together, not in separate steps.

## 3. What the code does today

`BoundaryNormalStressSolver` at `gadopt/stokes_integrators.py:1069` defines a
small standalone solver on the surface mesh facets:

- Weak form: find `h ∈ Q` such that `∫ φ h ds = -∫ φ (σ·n)·n ds` for all test
  `φ`. The left-hand side is just a boundary mass matrix, so this is
  essentially an L2 projection of the radial stress onto the surface DOFs.
- After the solve, the mean is subtracted as a Python post-processing step:

  ```python
  vave = fd.assemble(self.force * self.ds) / fd.assemble(1 * self.ds)
  self.force.assign(self.force - vave)
  ```

  (See `gadopt/stokes_integrators.py:1161`.)

This works in isolation. The output is a `Function` you can hand around.
The test `tests/analytical_comparisons/delta_cylindrical_zeroslip.py` runs
it after the Stokes solve and compares against `assess`'s analytical radial
stress, reporting an L2 error on the top boundary (line 161).

## 4. Why this needs to change

We want to couple Stokes to a Poisson solve for `Φ` inside a single
monolithic mixed system, and drive it with SNES (Newton) so that PETSc's
fieldsplit preconditioning and the Jacobian assembled by Firedrake do the
work. That means every quantity that appears on the right-hand side of
either equation must be a UFL expression in the unknowns of the mixed
system. `h` is one of those quantities: it enters the Poisson RHS as a
surface-mass term.

The current design breaks this in three ways:

1. `h` is the solution of a *separate* linear solve. You cannot put
   `force_on_boundary(...)` into a UFL form and expect SNES to differentiate
   through it.
2. The mean subtraction is a Python statement between two solves. If you try
   to do that inside a Newton iteration you get a non-differentiable step
   and a broken Jacobian.
3. Even if you were willing to nest a solve inside the residual evaluation,
   you would lose the ability to use fieldsplit across the full coupled
   system, because `h`'s values depend implicitly on `(u, p)` in a way PETSc
   cannot see.

Dale's prototype at `demos/gravity/submesh_2way_coupling.py` already solves
the coupling-through-a-mixed-space problem for the GIA (viscoelastic) case.
There the mixed space is `[V, S, V_grav]` and Poisson is just another block
in `F`. Look at lines 57, 494–506: `Z = MixedFunctionSpace([V, S, V_grav])`,
`F_grav = L_grav - R_grav`, then one `NonlinearVariationalSolver`. That is
the shape we want for Stokes–Poisson, but with a twist: we need to enforce
`∫ h dS = 0`.

## 5. The mathematical idea

Instead of computing `h` with its own solve and subtracting the mean
afterwards, we write

```
h = σ_rr(u, p) / (g Δρ)
```

as a pure UFL expression in the Stokes unknowns, and we enforce the zero-mean
constraint *inside the mixed system* using one scalar Lagrange multiplier
per constrained boundary. Concretely we add a new unknown

```
λ_top ∈ R-space, one global DOF on the whole mesh
```

and we augment the coupled residual with:

1. **The constraint equation.** For a test function `μ` in the R-space,

   ```
   ∫ (h_ufl - λ_top) μ ds_top = 0
   ```

   Because `μ` is constant, this single scalar equation says
   `λ_top = (1/|S_top|) ∫ h_ufl ds_top`, i.e. `λ_top` is exactly the mean
   of `h` on the top boundary.

2. **The corrected coupling to Poisson.** Anywhere the Poisson residual
   previously used `h`, we substitute `(h_ufl - λ_top)`. This is the
   zero-mean dynamic topography, by construction.

If you also need zero-mean at the CMB, repeat with a `λ_bot` and
`ds_bot`.

Key properties:

- `h` is never carried as a separate solved Function. It is a UFL expression
  that Firedrake symbolically differentiates when it builds the Jacobian.
  Newton sees the full coupling.
- The constraint is linear in `(u, p, λ)`, so it adds zero nonlinear burden.
- No post-processing step, no outer Picard loop.

## 6. Why fieldsplit preconditioning still works

This is the part people worry about. The concern is that adding a
"Lagrange multiplier row" produces a saddle-point system that could blow up
the condition number or break the Schur-complement preconditioner that
makes Stokes tractable in 3D. It does not, provided you structure the
fieldsplit correctly.

The current iterative Stokes preconditioner
(`gadopt/stokes_integrators.py:40-70`) is a two-field Schur split:

- `fieldsplit_0`: velocity `V`, preconditioned with GAMG via `SPDAssembledPC`.
- `fieldsplit_1`: pressure `W`, preconditioned by a viscosity-scaled mass
  matrix via `firedrake.MassInvPC`.

After we add `λ_top` and `Φ` to the mixed space, the subspaces are
`[V, W, Φ, λ_top]` (index 0, 1, 2, 3). We want:

- The existing Stokes Schur preconditioner to be **untouched**.
- `Φ` to be preconditioned as a Poisson problem (GAMG or AMG-on-assembled).
- `λ_top` to cost almost nothing — it is one scalar DOF.

PETSc + Firedrake support this through the
`pc_fieldsplit_<N>_fields` option. A worked example lives in the Firedrake
test suite at
`firedrake/_check/tests/firedrake/regression/test_matrix_free.py:148-167`
which does nested fieldsplit with `"pc_fieldsplit_0_fields": "1"` and
`"pc_fieldsplit_1_fields": "0,2"`. Dale's prototype at
`demos/gravity/submesh_2way_coupling.py:219-267` also shows a three-field
`symmetric_multiplicative` split in production.

The clean structure for us:

```
outer split  (pc_fieldsplit_type = schur or symmetric_multiplicative)
  field "stokes"  = subspaces {0, 1}       # V and W, together
    inner split = existing Stokes Schur from stokes_integrators.py
      fieldsplit_0 = V, GAMG
      fieldsplit_1 = W, MassInvPC
  field "gravity" = subspaces {2, 3}       # Φ and λ_top
    inner split (additive or schur)
      subfield for Φ    = AssembledPC with AMG
      subfield for λ_top = preonly/lu  (1×1, trivial)
```

Mapped to PETSc options:

```python
"pc_type": "fieldsplit",
"pc_fieldsplit_type": "schur",
"pc_fieldsplit_0_fields": "0,1",   # Stokes lives here
"pc_fieldsplit_1_fields": "2,3",   # Gravity + multiplier
"fieldsplit_0": {
    # ... literally the current iterative_stokes_solver_parameters dict
    #     from stokes_integrators.py:40-70, nested as a sub-solve
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": { ... GAMG velocity block ... },
    "fieldsplit_1": { ... MassInvPC pressure block ... },
},
"fieldsplit_1": {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0": { ... AMG for Φ ... },
    "fieldsplit_1": { "ksp_type": "preonly", "pc_type": "lu" },
},
```

This is the important design invariant:

> **The Stokes preconditioner is used verbatim, just one level of nesting
> deeper. We are not re-tuning GAMG, we are not changing MassInvPC, we are
> not touching the Schur setup. We are only asking PETSc to treat Stokes as
> one sub-block of a bigger problem.**

About the "dense row" worry. The multiplier equation `∫(h_ufl - λ) μ ds_top`
couples `λ_top` to every pressure and velocity DOF *that touches the top
boundary* — not to the global interior. In a sparse AIJ matrix this is one
wider row, nothing pathological. Once `λ_top` is isolated in its own
sub-field with `preonly/lu`, it is Schur-complemented out as a 1×1 system.
Condition-number-wise this is the same trick everyone uses for the
pressure-nullspace constant in incompressible Stokes; it has been working
for 30 years.

## 7. Step-by-step implementation plan

Below is the sequence of PRs a new engineer should submit. Each step
compiles and passes tests on its own.

### Step 0: reproduce and understand the baseline

1. Check out this worktree and set `PYTHONPATH` per the project's
   `CLAUDE.md`.
2. Run
   `pytest tests/analytical_comparisons/delta_cylindrical_zeroslip.py -v`.
   You should see L2 convergence on velocity, pressure, and normal stress.
3. Read `BoundaryNormalStressSolver` at `gadopt/stokes_integrators.py:1069`
   end to end. Make sure you understand what each of the four lines of
   `solve()` does (lines 1158–1168).
4. Read `demos/gravity/submesh_2way_coupling.py` lines 47–65 (mixed space
   construction) and 494–514 (coupled residual + single
   `NonlinearVariationalSolver`). The Stokes–Poisson coupling you are
   about to build looks like this.

### Step 1: add a UFL-only helper for `h`

In `gadopt/stokes_integrators.py`, add a function next to
`BoundaryNormalStressSolver` that returns the UFL expression for `h`
without doing any solve or projection:

```python
def dynamic_topography_ufl(u, p, approximation, subdomain_id, g_delta_rho):
    """Return -σ·n·n / (g Δρ) as a pure UFL expression on the given boundary.

    This is the pointwise dynamic topography before mean subtraction.
    Callers are responsible for enforcing ∫ h dS = 0 when required.
    """
    mesh = u.function_space().mesh()
    n = fd.FacetNormal(mesh)
    dim = mesh.geometric_dimension
    stress_with_pressure = approximation.stress(u) - p * fd.Identity(dim)
    return -fd.dot(fd.dot(stress_with_pressure, n), n) / g_delta_rho
```

This is just a refactor of the RHS of the existing solver. Keep
`BoundaryNormalStressSolver` intact for backward compatibility and for use
in decoupled diagnostic workflows. Anyone who just wants a plottable `h`
Function still uses the class.

### Step 2: prototype the mean-free coupling in a demo

Do not touch `gadopt/` yet. Instead, copy
`demos/gravity/submesh_2way_coupling.py` to a new file
`demos/gravity/stokes_poisson_meanfree.py` and modify it to implement:

- Mixed space `Z = MixedFunctionSpace([V, W, V_grav, R])` where the last
  entry is the multiplier.
- `u, p, phi, lam_top = split(z)`.
- Residual terms:
  - Stokes momentum and continuity, with gravity body force `ρ_0 α T +
    ρ ∇Φ`, using the existing G-ADOPT infrastructure.
  - Poisson: `-∫ ∇ψ·∇Φ dx = 4π ∫ ρ₁ ψ dx + ∫ ψ (h_ufl - λ_top) ds_top`.
  - Constraint: `∫ (h_ufl - λ_top) μ ds_top = 0`.
- One `NonlinearVariationalSolver` with the nested fieldsplit parameters
  from section 6.
- A simple analytical forcing (pick something with a known surface stress,
  e.g. the delta forcing used by
  `tests/analytical_comparisons/delta_cylindrical_zeroslip.py`).

Validate three things:

1. `assemble((h_ufl - lam_top) * ds_top)` is zero to solver tolerance.
2. If you pass `(h_ufl - lam_top)` to the existing
   `BoundaryNormalStressSolver` flow as a comparison, the values match the
   post-processed `force` up to a constant. Cross-check against
   `assess.CylindricalStokesSolutionDeltaZeroSlip` the way
   `delta_cylindrical_zeroslip.py` does at lines 110–141, 161.
3. SNES converges to `snes_rtol` in a small number of Newton iterations,
   and each inner linear solve converges in the fieldsplit in a reasonable
   number of Krylov iterations.

If any of those three fail, the bug is almost certainly in the fieldsplit
options, not in the physics. The most common mistakes are (a) forgetting
`pc_fieldsplit_0_fields`/`_1_fields`, which silently causes PETSc to make
its own split, and (b) leaving `snes_type: ksponly` when you actually want
Newton. Turn on `-ksp_view -snes_view` to inspect what PETSc is actually
doing.

### Step 3: promote the prototype into a reusable solver class

Once Step 2 works, the pattern becomes a new class
`CoupledStokesPoissonSolver` in `gadopt/stokes_integrators.py`, alongside
`StokesSolver` and `ViscoelasticStokesSolver`. Its job is to assemble the
mixed space, build the residual, and configure the default fieldsplit
parameters. Its constructor accepts the same `approximation`, `bcs`,
`solver_parameters_extra` interface as `StokesSolver` so that user code
looks familiar.

Critical: **do not duplicate** the iterative Stokes parameters. Import
`iterative_stokes_solver_parameters` from the module and nest it as the
`fieldsplit_0` block of the new coupled default. That enforces the "Stokes
preconditioner is verbatim" invariant at code level.

### Step 4: regression and convergence tests

1. Add `tests/unit/test_dynamic_topography_meanfree.py` that checks, on a
   small cylindrical mesh, that the mean of `h_ufl - λ_top` is machine-zero
   after a solve.
2. Add `tests/analytical_comparisons/stokes_poisson_meanfree.py` modelled
   on `delta_cylindrical_zeroslip.py` (cylinder, delta forcing, zero slip).
   The success criterion is that the L2 error of the recovered
   mean-subtracted radial stress against the analytical solution
   converges at the expected rate under mesh refinement, *and* that the
   L2 error on `Φ` also converges.
3. Before merging, confirm with `pytest -m "not longtest"` that existing
   tests still pass. The old `BoundaryNormalStressSolver`-based test must
   stay green; we are not replacing it, we are adding a monolithic
   alternative.

### Step 5: documentation

Update the docstrings of the new helper and class, and add a short section
to `gadopt/__init__.py` exports. Add one demo under
`demos/mantle_convection/` that uses the coupled solver end to end.

## 8. Things that are out of scope

- Removing `BoundaryNormalStressSolver`. It is still useful as a
  post-processing diagnostic and for decoupled workflows. Leave it alone.
- Spherical / 3-D performance tuning. Get the coupling correct first on
  2-D cylindrical. 3-D is a separate optimisation pass.
- The adjoint-mode story. Adjoint support for the coupled solve requires
  verifying that pyadjoint tapes the R-space multiplier correctly; that
  is a follow-up ticket.
- Handling a CMB constraint at the same time as the top. The extension is
  mechanical — add `λ_bot` with its own `ds_bot` equation — but do it in a
  second PR after the top-only version is green.

## 9. What success looks like

- Running `pytest tests/analytical_comparisons/stokes_poisson_meanfree.py`
  shows the expected convergence rates.
- A single `NonlinearVariationalSolver` drives the whole
  Stokes–Poisson–multiplier system. No nested Python-level solve, no
  explicit mean-subtraction statement anywhere in the time loop.
- `-snes_view` shows one SNES wrapping one KSP wrapping the nested
  fieldsplit tree described in section 6.
- The per-timestep cost is within ~10% of the equivalent uncoupled Stokes
  solve plus a separate Poisson solve, confirming that the multiplier
  adds negligible overhead.

## 10. Reading list

In order of decreasing priority:

1. `gadopt/stokes_integrators.py:40-175` — existing solver parameters.
2. `gadopt/stokes_integrators.py:1069-1220` — the solver we are replacing.
3. `demos/gravity/submesh_2way_coupling.py` — the structural template.
4. `tests/analytical_comparisons/delta_cylindrical_zeroslip.py` — the
   verification pattern and what a "convergence test" looks like in this
   project.
5. Firedrake docs on mixed function spaces, `FunctionSpace(mesh, "R", 0)`,
   and nested fieldsplit.
6. Davies et al. (2022) Section 4.3 — the Schur-complement rationale for
   the current Stokes preconditioner, which is what we are preserving.
