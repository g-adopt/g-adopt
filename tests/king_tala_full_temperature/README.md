# Full-temperature Cartesian TALA benchmark

By default, the evolved variable is (Kelvin temperature - 273 K) / 3000 K.
Its top and bottom boundary values are 0 and 1. The absolute reference temperature is
`0.091*exp(Di*(1-y))` and density is `exp(Di*(1-y))`; all other thermodynamic
coefficients, viscosity and conductivity are one. Heating is zero. `Ra` uses
the full 3000 K contrast. The unit square has free-slip walls and insulated sides.

The physical problem is from King et al. (2010), *A community benchmark for 2-D
Cartesian compressible convection in the Earth's mantle*, GJI 180, 73-87,
[doi:10.1111/j.1365-246X.2009.04413.x](https://doi.org/10.1111/j.1365-246X.2009.04413.x).
`king_reference.json` transcribes the six code rows for four selected cases from
Tables 5, 6 and 7 of the supplied online supplement. Those rows are published
comparison data, not new G-ADOPT regression values.

## Approximation interface

```python
from gadopt import FullTemperatureTruncatedAnelasticLiquidApproximation

approximation = FullTemperatureTruncatedAnelasticLiquidApproximation(
    Ra, Di, reference_temperature=reference, temperature_offset=0.091, rho=density,
)
```

Pass the same full-temperature Function to both StokesSolver and EnergySolver.
The approximation uses `T+temperature_offset-reference_temperature` in buoyancy,
`T+temperature_offset` in adiabatic heating and T in diffusion. Its legacy `Tbar` diffusion offset
remains zero. The corresponding EBA and ALA classes use the same explicit
temperature-offset convention; ALA retains pressure-dependent buoyancy and its
nonconstant pressure nullspace. Only TALA is benchmarked here.

`--formulation full` selects absolute scaled temperature, with top/bottom BCs
0.091/1.091 and zero offset. The approximation default offset is zero for
compatibility; the benchmark default is `--formulation surface-relative`.
`reference_temperature` always means the absolute reference adiabat. The offset
must be a stationary scalar number, Firedrake Constant, or Function in the Real
(R, 0) space. Use the Real-space Function for an adjoint control. Spatial/time-dependent
shifts would require additional advection, diffusion and time-derivative terms.
Use `approximation.absolute_temperature(T)` in absolute-temperature material laws.
`energy_source(u)` includes the negative offset contribution to adiabatic work;
`viscous_dissipation(u)` remains pure dissipation, and `work_against_gravity(u,T)`
returns physical work including the offset.

`--formulation perturbation` uses the existing approximation and evolves theta.
It diffuses `theta+reference-0.091`, which has the same gradient as absolute
temperature. Its top and bottom BCs are 0 and `1-0.091*(exp(Di)-1)`.

## Running and solving

```sh
python benchmark.py --ra 10000 --di 0.5 --n 16 --output king_result
python -m pytest test_result.py
python benchmark.py --ra 10000 --di 0.5 --n 32 --initial king_result.h5 --output refined
```

The `meta.py` registers a two-core pilot with `doit run_case` and `doit check`.
Use `doit list --all` to discover its exact task names. Each output prefix gets
a JSON diagnostic record, a collectively written HDF5 checkpoint and gathered
nodal NPZ data for plotting. No output files are scientific reference updates.

The steady mixed Q2/Q1/Q2 solve assembles existing G-ADOPT momentum, continuity
and scalar-energy terms. Momentum is multiplied by 1/Ra for algebraic scaling.
The constant TALA pressure nullspace is supplied explicitly; mean pressure is
removed after convergence. Newton with backtracking and direct MUMPS solves
the steady coupled problem. This is not a time integration or a stability test.

Direct Newton can find other convective branches or fail from a poor initial
guess. For Di=1, continue from Di=0.5 through 0.75. For Ra=1e5, continue from
Ra=1e4 through 2e4 and 5e4. Inspect the flow, not just the residual. These
continuation steps are part of reproducing the intended single-roll branch.

`transient_check.py` provides an additional standard-solver integration check.
It loads the Ra=1e4, Di=0.5 checkpoint, perturbs the temperature by a 0.005
cosine/sine field, and advances both formulations with the ordinary
StokesSolver/EnergySolver and ImplicitMidpoint. Defaults: 50 steps at dt=1e-4.
Its relative-difference bound allows the O(h^3) interpolation defect of the
reference adiabat on the intended 32-cell Q2 mesh; it is not a universal bound
for arbitrary meshes or reference states.

## Diagnostics

- Nu from the boundary gradient and Nu from weak energy-equation reactions are
  both reported. Reaction fluxes use affine top/bottom test lifts and converge
  faster here. Since the unit layer has unit conductivity and total contrast,
  these heat-flow values are nondimensional Nusselt numbers.
- Temperature means reported for the King comparison are surface-relative,
  `mean(T_absolute-0.091)`. Absolute means are stored separately.
- RMS velocity is `sqrt(integral(u.u)/volume)`. Surface RMS is not mean speed
  or maximum surface speed; it must not be compared to those table columns.
- Viscous heating includes Di/Ra. Both full-temperature and perturbation
  adiabatic-work integrals are recorded. TALA does not enforce equality of
  viscous heating and adiabatic work.
- The reaction energy residual includes the integral of discrete advection.
  Taylor-Hood velocity enforces anelastic continuity weakly, so this integral
  is not identically zero at finite resolution. Strong mass-divergence L2 norms
  should converge with refinement, not be mistaken for algebraic solver errors.

Analytic reference coefficients and quadrature degree eight are used for the
paired study. Although the two forms are equivalent continuously, an exponential
reference cannot be represented exactly in Q2. Agreement should converge under
refinement; it need not be bitwise exact.

To compare the two total-temperature conventions through standard solvers, run
`transient_check.py --initial <32-cell-checkpoint.h5> --comparison surface-relative`.
