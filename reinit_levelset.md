# Coupled DAE Formulation for Level-Set Reinitialisation

## Overview

This document describes a coupled Differential-Algebraic Equation (DAE) formulation for level-set reinitialisation that eliminates the need for `update_forcings` callbacks in time integration.

## Background: Traditional Approach

In the standard level-set reinitialisation approach, we solve:

```
∂φ/∂t = sharpen_term(φ) + balance_term(φ, ∇φ)
```

where:
- `sharpen_term(φ) = -φ(1-φ)(1-2φ)` maintains the interface sharpness
- `balance_term(φ, ∇φ) = ε(1-2φ)|∇φ|` ensures proper interface width

### Problem with Traditional Approach

The gradient `∇φ` needs to be updated at every stage of the time integrator (e.g., at each Runge-Kutta stage). This was previously handled by:

1. Computing `∇φ` via L² projection before starting the time integrator
2. Using an `update_forcings` callback to recompute `∇φ` at each intermediate stage

While functional, this approach:
- Requires managing callback mechanisms
- Creates tight coupling between the solver and time integrator
- Can be error-prone when changing time integration schemes

## New Coupled DAE Formulation

### Mathematical Formulation

Instead of treating `∇φ` as a derived quantity, we solve for both φ and g = ∇φ simultaneously as a coupled system:

```
∂φ/∂t = sharpen_term(φ) + balance_term(φ, g)    [differential equation]
0 = gradient_reconstruction(φ, g)                 [algebraic constraint]
```

The gradient reconstruction ensures g = ∇φ through a weak L² projection:

```
∫ g · v_g dx = -∫ φ ∇·v_g dx + ∫ φ (v_g·n) ds
```

This is the weak form of g = ∇φ obtained via integration by parts.

### System Structure

This creates a **Differential-Algebraic Equation (DAE)** system because:
- φ has a time derivative (differential equation)
- g has NO time derivative (algebraic constraint)

The mass term only applies to φ:

```
M(Dt(φ), Dt(g)) = ∫ Dt(φ) · v_φ dx
```

Note that there is no term involving Dt(g), making the mass matrix singular.

## Implementation

### Mixed Function Space

We create a mixed function space combining the level-set and gradient spaces:

```python
grad_space_degree = level_set_space.ufl_element().degree()
gradient_space = VectorFunctionSpace(mesh, "CG", grad_space_degree)
mixed_space = level_set_space * gradient_space
```

### Residual Form

The combined residual includes both equations:

```python
def coupled_reinitialisation_term(eq, trial):
    phi, g = split(trial)
    v_phi, v_g = split(eq.test)

    # Reinitialisation equation for phi
    sharpen_term = -phi * (1 - phi) * (1 - 2 * phi) * v_phi * eq.dx
    balance_term = eq.epsilon * (1 - 2 * phi) * sqrt(inner(g, g)) * v_phi * eq.dx
    reinit_residual = sharpen_term + balance_term

    # Gradient reconstruction (algebraic constraint)
    grad_lhs = inner(g, v_g) * eq.dx
    grad_rhs = -phi * div(v_g) * eq.dx + phi * dot(v_g, eq.n) * eq.ds
    grad_residual = grad_lhs - grad_rhs

    return reinit_residual + grad_residual
```

### Mass Term

The mass term only includes the φ component:

```python
def coupled_reinitialisation_mass_term(eq, trial):
    # Extract the operand if trial is Dt(mixed_function)
    if isinstance(trial, TimeDerivative):
        trial = trial.ufl_operands[0]

    phi, g = split(trial)
    v_phi, v_g = split(eq.test)

    # Only phi has time derivative (DAE structure)
    return inner(phi, v_phi) * eq.dx
```

## Critical Requirement: Implicit Time Integration

**Important:** Because this is a DAE system with a singular mass matrix, **explicit time integrators will fail**. The system requires **implicit time integration methods**.

### Why Explicit Methods Fail

With explicit methods (e.g., `eSSPRKs3p3`), at each stage we need to solve:
```
M * k_i = F(...)
```

But since M is singular (zero block for g), this system has no unique solution.

### Solution: Use Implicit Methods

Implicit methods (e.g., `ImplicitMidpoint`) solve:
```
(M - α*dt*J) * Δu = residual
```

The Jacobian J is non-singular for the coupled system, making the overall system solvable.

## Solver Configuration

The coupled system requires appropriate preconditioners for the mixed space:

```python
"reini_coupled": {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    },
    "fieldsplit_1": {
        "ksp_type": "cg",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    },
}
```

## Usage

```python
reini_kwargs = {
    "epsilon": epsilon,
    "use_coupled_formulation": True,
    "time_integrator": ImplicitMidpoint  # Must use implicit!
}

level_set_solver = LevelSetSolver(
    psi,
    adv_kwargs=adv_kwargs,
    reini_kwargs=reini_kwargs
)
```

## Benefits

1. **No callbacks needed:** The gradient is automatically consistent with φ throughout the solve
2. **Cleaner interface:** No need to manage `update_forcings` callbacks
3. **More robust:** The coupling is enforced by the variational formulation

## Trade-offs

1. **Implicit integration required:** May be slower than explicit methods
2. **Larger system:** Mixed space increases DOFs (adds gradient variables)
3. **More complex preconditioning:** Need fieldsplit or block preconditioners
4. **Memory overhead:** Storing both φ and g during the solve

## Implementation Files

- `gadopt/level_set_tools.py`:
  - `coupled_reinitialisation_term()`: Combined residual form
  - `coupled_reinitialisation_mass_term()`: DAE mass term
  - `LevelSetSolver.set_up_solvers()`: Coupled formulation setup (lines 661-729)

- `demos/multi_material/thermochemical_buoyancy/thermochemical_buoyancy.py`:
  - Example usage with coupled formulation

## Future Considerations

- Performance comparison: coupled DAE vs. callback approach
- Optimal preconditioner strategies for the mixed system
- Extension to 3D and different mesh types
- Compatibility with adaptive mesh refinement

## Backward Compatibility

The original callback-based approach is still available by setting `use_coupled_formulation=False` (the default). Both formulations are maintained for flexibility and comparison purposes.
