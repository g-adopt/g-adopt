# Physics and Mathematical Coupling Analysis

## Overview

The script `submesh_2way_coupling.py` implements a **fully coupled 2D viscoelastic self-gravitating** model for Glacial Isostatic Adjustment (GIA). Three equations are solved simultaneously in a mixed system:

1. **Momentum equation** (on submesh/mantle)
2. **Internal variable evolution** (Maxwell viscoelasticity, on submesh/mantle)
3. **Gravity Poisson equation** (on full mesh, sources from mantle only)

---

## 1. The Gravity Equation (Poisson Equation) — Lines 494–498

### Mathematical Formulation

```
L_grav = -∫ ∇ψ · ∇v dx_all
R_grav = 4π ∫ ρ₁ v dx_m  +  0 · ∫ v dx_all
F_grav = L_grav - R_grav
```

where:
- `ψ` (phi) is the gravitational potential perturbation
- `v` (test_funcs[2]) is the test function for the gravity equation
- `ρ₁` (rho1_grav) is the linearised density perturbation

In physical (dimensional) form this is: **−∇²ψ = 4πGρ₁**

### The Density Perturbation (Line 495)

```python
rho1_grav = -dot(u, grad(density)) - density * div(u)
```

This is the **linearised Eulerian density perturbation** from the continuity equation:

- **`-dot(u, grad(density))`**: Advection of the background density by displacement u. When stratified material is displaced vertically, the local density changes because the material at position x came from position x−u, which had a different background density.
- **`-density * div(u)`**: Compression/dilation effect. Positive divergence means local expansion, which decreases density; negative divergence means compression.

Together: ρ₁ = −u·∇ρ₀ − ρ₀∇·u, which is the time-integrated linearised continuity equation.

### Boundary Condition (Line 491)

```python
grav_bcs = [DirichletBC(Z.sub(2), 0.0, Rg_id)]
```

Dirichlet zero at the gravity mesh outer boundary (Rg ≈ 31,855 km). This approximates the infinite-domain condition ψ → 0 as r → ∞.

### Measure Usage

- The Laplacian (`L_grav`) uses `dx_all` — it must be integrated over the full gravity domain because ψ is defined everywhere from Rc to Rg.
- The source (`R_grav`) uses `dx_m` — density perturbations only exist in the mantle (the submesh).
- The dummy `Constant(0) * test_funcs[2] * dx_all` ensures the residual has contributions from the full mesh for proper FEM assembly.

---

## 2. Infinite Boundary Condition Strategy

### Mesh Geometry (from `unstructured_annulus.py`)

| Boundary | Radius | Role |
|----------|--------|------|
| Core (Rc) | 3,480 km | Core–mantle boundary |
| Surface (Re) | 6,371 km | Earth's surface |
| Gravity (Rg) | 31,855 km | Truncated "infinity" |

The non-dimensionalisation uses D = Re − Rc = 2,891 km.

### Why This Works

The Poisson equation for gravity is naturally posed on all of R³ (or R² in this 2D case). The implementation approximates this by truncating the domain at a very large radius Rg ≈ 5Re and imposing ψ = 0 there. Since the gravitational potential from a finite-sized source decays as 1/r (monopole) or faster (higher multipoles), placing the boundary at ~5× planetary radius introduces only a small error for mantle-scale features.

The gmsh mesh coarsens dramatically in the exterior region (cell size ~5000 km vs ~500 km in the mantle), keeping computational cost manageable while still resolving the potential decay in the far field.

---

## 3. Coupling Between Equations

### Field Variables (Line 57–60)

```python
Z = MixedFunctionSpace([V, S, V_grav])
u, m, phi = split(z)
```

- `u`: Displacement vector (CG2 on submesh)
- `m`: Deviatoric stress internal variable (DG1 tensor on submesh)
- `phi`: Gravitational potential (CG2 on full mesh)

### A. Gravity → Momentum (Line 332)

```python
source_grav = -Vi * density / g * grad_phi
```

The gradient of the gravitational potential is the gravitational acceleration perturbation. This enters the momentum equation as a body force: material with density ρ experiences an extra force −ρ∇ψ from the perturbed gravity field. This is the self-gravitation feedback.

### B. Momentum → Gravity (Line 495)

```python
rho1_grav = -dot(u, grad(density)) - density * div(u)
```

Displacement `u` creates density perturbations that source the Poisson equation. This closes the feedback loop: deformation changes density, changed density changes gravity, changed gravity changes forces, changed forces change deformation.

### C. Momentum Equation — Viscoelastic Stress

The stress tensor (lines 305–313):

```
σ = κ(∇·u)I + 2Nμ(ε_d − Σmᵢ)
```

where:
- κ = β·μ (bulk modulus, β ≈ 1.94)
- ε_d = sym(∇u) − (1/3)tr(sym(∇u))I (deviatoric strain)
- m is the internal variable (accumulated viscous deviatoric strain)
- N is the number of internal variables (here 1)

The momentum source includes:
- **Buoyancy** (line 325): `−Vi·g·(−u·∇ρ)·n̂` — advection of density creates buoyancy forces
- **Self-gravity** (line 332): `−Vi·(ρ/g)·∇ψ` — gravitational acceleration from potential perturbation
- **Hydrostatic prestress advection** (lines 344–367): accounts for how vertical displacement changes the background hydrostatic pressure acting on material, including jump terms at density discontinuities

### D. Internal Variable Evolution (Maxwell Rheology)

```
dm/dt = ε_d/τ_M − m/τ_M
```

where τ_M = η/μ is the Maxwell time. This represents:
- **Source**: strain rate drives stress buildup
- **Sink**: stress relaxes exponentially on the Maxwell timescale

Time discretisation is semi-implicit (backward Euler-like, line 502):
```
F = ... − 0.5·(m − m_old)/dt + 0.5·(source + sink)
```

### Coupling Summary

```
  u (displacement) ──────→ ρ₁ = −u·∇ρ − ρ∇·u ──────→ −∇²ψ = 4πρ₁
       ↑                                                      │
       │                                                      │
       └──────── σ includes −ρ∇ψ body force ←── ∇ψ ←────────┘
```

The system is monolithically coupled: all three fields are solved simultaneously in a single NonlinearVariationalProblem.

---

## 4. Mesh Strategy and Domain Separation

### Construction (Lines 10–26)

```
full_mesh (from gmsh: Rc to Rg)
    ↓
mesh = RelabeledMesh(full_mesh, [F_mantle, F_rest], [98, 99])
    ↓
subm = Submesh(mesh, 2, 98)   # mantle only: Rc to Re
```

- `F_mantle`: indicator function = 1 for Rc ≤ r ≤ Re (label 98)
- `F_rest`: indicator function = 1 everywhere (label 99)
- `Submesh(mesh, 2, 98)`: extracts cells with label 98 as a standalone mesh

### Why Separate?

- **Momentum + internal variable** live only in the mantle (Rc to Re). The core is rigid and doesn't participate in GIA dynamics. No need to solve for displacement there.
- **Gravity** requires the extended domain (Rc to Rg) to approximate the infinite boundary condition. The potential is defined everywhere but only sourced by mantle density perturbations.
- This separation avoids wasting DOFs on velocity/stress in the exterior gravity region while allowing the gravity equation to "see" the full domain.

---

## 5. Measure Coupling (Lines 286–291)

This is the most subtle aspect. Firedrake's `intersect_measures` control where integrals are evaluated when dealing with nested meshes.

```python
dx_all  = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", domain=subm),))
dx_m    = Measure("dx", domain=subm, intersect_measures=(Measure("dx", domain=mesh),))
dS_m    = Measure("dS", domain=subm, intersect_measures=(Measure("ds", domain=mesh),))
dS_full = Measure("dS", domain=mesh, intersect_measures=(Measure("dS", domain=subm),))
ds_mt   = Measure("ds", domain=subm, intersect_measures=(Measure("dS", domain=mesh),))
ds_mb   = Measure("ds", domain=subm, intersect_measures=(Measure("ds", domain=mesh),))
```

### What Each Measure Does

| Measure | Primary domain | Constraint | Physical use |
|---------|---------------|------------|--------------|
| `dx_all` | full mesh (cells) | intersected with submesh cells | Gravity Laplacian: ∫∇ψ·∇v over full domain |
| `dx_m` | submesh (cells) | intersected with full mesh cells | Momentum/stress integrals, gravity source: ∫ρ₁v over mantle |
| `dS_m` | submesh interior facets | intersected with full mesh boundary facets | Hydrostatic prestress jumps at density discontinuities |
| `dS_full` | full mesh interior facets | intersected with submesh interior facets | (Available but not heavily used in this script) |
| `ds_mt` | submesh boundary facets | intersected with full mesh interior facets | Surface integrals at Re (Earth surface, which is an interior facet of the full mesh) |
| `ds_mb` | submesh boundary facets | intersected with full mesh boundary facets | Boundary condition integrals (CMB, core boundary) |

### Why This Complexity?

The Earth's surface (Re) is a **boundary** of the submesh but an **interior facet** of the full mesh. This dual nature requires careful measure selection:

- When applying boundary conditions on the submesh at Re, we need `ds_mb` (submesh boundary facets that are also full mesh boundary facets — this captures the CMB at Rc) and the surface traction terms use the submesh boundary measure.
- The `dS_m` measure captures interior facets of the submesh that align with boundaries of the full mesh — needed for jump terms in the hydrostatic prestress advection where density is discontinuous.

Without `intersect_measures`, integrating a submesh function over the full mesh would incorrectly accumulate contributions from exterior cells (which have no submesh DOFs), and vice versa.

---

## 6. Boundary Conditions Summary

### Momentum (on submesh)

| Boundary | ID | Condition | Physical meaning |
|----------|----|-----------|-----------------|
| CMB (Rc) | 1 | `un = 0` | Rigid core, no radial flow |
| Surface (Re) | 2 | normal_stress = ice_load + hydrostatic prestress | Ice loading + isostatic pressure adjustment |
| Surface (Re) | 2 | free_surface = {} | Free surface (surface can deform) |

### Gravity (on full mesh)

| Boundary | ID | Condition | Physical meaning |
|----------|----|-----------|-----------------|
| Gravity outer (Rg) | 3 | ψ = 0 | Infinite domain approximation |

---

## 7. Time Stepping and Physical Parameters

- **Maxwell time**: τ_M = η/μ ≈ 10²¹/10¹¹ ≈ 10¹⁰ s ≈ 317 years
- **Time step**: 100 years (well-resolved relative to Maxwell time)
- **Total simulation**: 10,000 years (captures multiple Maxwell times of relaxation)
- **Ice load**: Two disc-shaped loads with smooth tanh transitions, applied at t=0
- **Viscosity**: Heterogeneous, with Gaussian anomalies representing Antarctica (low visc), LLSVPs (low visc), a slab (high visc), and a craton (high visc)

The simulation computes the viscoelastic response of Earth's mantle to ice loading, including the self-gravitational feedback that modifies the geoid and sea level.
