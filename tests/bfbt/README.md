# Density-aware BFBT benchmarks

`benchmark.py` compares Firedrake's pressure-mass Schur preconditioner with
`gadopt.DensityAwareBFBTPC`. Both configurations use the same full Schur
factorisation, velocity solver and outer pressure tolerance. The linear cases
use one `ksponly` solve; the viscoplastic case uses the same Newton tolerances
for both pressure preconditioners.

For the zero pressure-pressure block used by these Stokes systems, PETSc's
Schur complement is `S = -D_rho A^-1 G`. The implemented approximation to the
inverse of the corresponding positive pressure operator, `-S`, is

\[
\widetilde S^{-1} =
(D_\rho C^{-1}G)^{-1}
(D_\rho C^{-1}AC^{-1}G)
(D_\rho C^{-1}G)^{-1},
\]

where `G` and `D_rho` are the actual PETSc Jacobian blocks. They are not
assumed to be transposes. This is required for TALA and ALA. For ALA, the
assembled auxiliary pressure operator also includes the pressure-buoyancy
part of `G` and receives the non-constant right pressure nullspace.

## Reproducing the comparison

Run one configuration per process from the repository root. Prepending the
checkout to `PYTHONPATH` is deliberate: the benchmark fails fast if Python
imports G-ADOPT from a different editable checkout. The JSON output
reports maximum-rank cold timing, the median and samples of repeated warm
timings, fieldsplit solve/iteration counts for every warm sample,
convergence failures, MPI size, equation residuals, imported package paths,
and the Git commit. The benchmark uses communicator barriers and only rank
zero writes JSON.

For BFBT, the JSON additionally reports both inner pressure solves and their
iterations. It does not currently instrument the MassInvPC baseline's private
`Mp_ksp`, so nested iteration-work totals are not symmetric between those two
arms. Wall time and the common fieldsplit counters remain comparable.

```bash
PYTHONPATH=$PWD python tests/bfbt/benchmark.py \
  --case tala --contrast 1e10 --pc mass --n 24 --velocity-pc gamg \
  --warm-repeats 5
PYTHONPATH=$PWD python tests/bfbt/benchmark.py \
  --case tala --contrast 1e10 --pc bfbt --n 24 --velocity-pc gamg \
  --warm-repeats 5

PYTHONPATH=$PWD python tests/bfbt/benchmark.py \
  --case viscoplastic --pc mass --n 12 --velocity-pc gamg
PYTHONPATH=$PWD python tests/bfbt/benchmark.py \
  --case viscoplastic --pc bfbt --n 12 --velocity-pc gamg
```

The available cases are `linear`, `tala`, `ala`, and `viscoplastic`. BFBT
tuning options are exposed as command-line arguments. PETSc options can also
be appended and are left in `sys.argv` for PETSc to consume.

## Local reference measurements

The following single-rank Apple-silicon measurements used five warm repeats,
a 20-by-20 mesh, GAMG for the velocity block and the current instrumented
harness. Times are medians. They are local trend evidence, not a production
speedup claim.

| Case | PC | Pressure work | Velocity work | BFBT inner work | Warm s |
| --- | --- | ---: | ---: | ---: | ---: |
| TALA, contrast 1 | Mass | 5 | 84 | - | 0.087 |
| TALA, contrast 1 | BFBT | 4 | 74 | 27 | 0.101 |
| TALA, contrast 1e3 | Mass | 14 | 235 | - | 0.231 |
| TALA, contrast 1e3 | BFBT | 8 | 144 | 53 | 0.185 |
| TALA, contrast 1e6 | Mass | 22 | 379 | - | 0.374 |
| TALA, contrast 1e6 | BFBT | 11 | 188 | 78 | 0.247 |
| TALA, contrast 1e10 | Mass | 32 | 517 | - | 0.505 |
| TALA, contrast 1e10 | BFBT | 14 | 205 | 100 | 0.282 |
| Viscoplastic, 5 Newton steps | Mass | 56 | 725 | - | 0.615 |
| Viscoplastic, 5 Newton steps | BFBT | 34 | 471 | 204 | 0.547 |

At unit contrast, BFBT's two auxiliary solves cost more than the work they
save. The crossover in this suite lies below contrast 1e3. Its advantage then
grows with viscosity contrast: the TALA warm-time reduction is about 20% at
1e3, 34% at 1e6 and 44% at 1e10. A two-rank, 24-by-24 TALA run at contrast
1e10 showed a similar 45% reduction; the two-rank viscoplastic case improved
by 17%. These timings are not scaling results, but the consistent serial/MPI
trend motivates the production GPlates test.

The DG0-interpolated square-root-viscosity weight avoids passing the
viscoplastic rheology's large symbolic polynomial degree into the auxiliary
forms. In this test it removes the TSFC quadrature-degree warning without
changing the nonlinear iteration count or final residual.

The conservative defaults remain diagonal mass lumping, a DG0 weight and an
inner FGMRES tolerance of `1e-2`. A later cylindrical TALA sweep at larger,
high-contrast configurations identified a better opt-in setting: a DG1
weight, inner FGMRES tolerance `1e-4`, no aggressive GAMG coarsening and two
smoothed-aggregation prolongator smoothing steps. This reduced outer and
velocity work and produced warm speedups on 64-by-16 and 128-by-32 meshes. It
remained substantially slower than pressure mass at mild viscosity contrast,
so it is not a replacement default for easy cases. Looser inner tolerances
and fixed four-step Richardson reduced inner Krylov work but increased outer
work in the larger local cases.

Inner BFBT failure is fatal by default. Each application records both inner
solve reasons and total work. ``bfbt_raise_on_inner_failure false`` is
intended only for controlled failure diagnostics. Exact pressure-nullspace
attachment is verified by default. The benchmark explicitly opts into
``bfbt_nullspace_policy=schur`` for ALA so it can compare G-ADOPT's existing
non-exact analytical quotient; the output reports that discrepancy and gives
momentum and continuity residuals separately.

BFBT reuses pressure work vectors but always starts each inner solve from
zero. If `bfbt_ksp_initial_guess_nonzero=true` is supplied, the preconditioner
overrides it and records that fact in its diagnostics, preventing the result
from depending on a stale work vector from the previous left or right solve.

ALA use remains experimental until the non-exact analytical gauge and the
full projected residual have been validated on the target discretisation.
Its pressure-buoyancy term also makes the assembled inner GAMG operator
nonsymmetric, which requires production-rank hierarchy and convergence tests.
The preconditioner is currently forward-only: transpose application fails
clearly instead of assuming that every selectable inner PC supplies a valid
numerical transpose.

The unit suite now includes a three-dimensional, extruded cubed-sphere TALA
regression with imposed velocity on the top and weak free slip on the bottom,
matching the boundary-condition structure of the GPlates case. A production
claim still requires repeated MPI runs on the full three-dimensional mesh, a
frozen fully plastic checkpoint, and an end-to-end nonlinear slice. Report
total fieldsplit_0, fieldsplit_1, and BFBT inner work,
GAMG setup and operator complexity, maximum-rank time, memory, and full
nonlinear residuals. The pressure-mass and BFBT configurations should also be
tuned to comparable final accuracy; their inner tolerances need not have the
same numerical value to achieve that.

On very small problems with a direct velocity solve, BFBT can be slower even
when it reduces pressure iterations. Its two auxiliary pressure solves have a
fixed cost. Production assessment should therefore use representative MPI
meshes, AMG velocity solves and warm timings.

## Testing the GPlates TALA case

Apply the same pressure configuration to the Newton and frozen-viscosity
Picard solvers:

```python
{
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_rtol": 5e-3,
    "pc_type": "python",
    "pc_python_type": "gadopt.DensityAwareBFBTPC",
    "bfbt_ksp_type": "fgmres",
    "bfbt_ksp_rtol": 1e-4,
    "bfbt_ksp_max_it": 1000,
    "bfbt_pc_type": "gamg",
    "bfbt_weight_degree": 1,
    "bfbt_pc_gamg_aggressive_coarsening": 0,
    "bfbt_pc_gamg_agg_nsmooths": 2,
}
```

TALA should retain the default `bfbt_nullspace_policy=verified`; do not copy
the benchmark's experimental ALA `schur` setting. Test BFBT and the conformal
near-nullspace branch independently before combining them. On Gadi, sync or
clone the complete branch and verify `python -c "import gadopt;
print(gadopt.__file__)"` points to it rather than the centrally installed
package.

`DensityAwareBFBTPC` supports a scalar pressure Schur block. Coupled implicit
free-surface systems must retain `FreeSurfaceMassInvPC` until a surface-aware
BFBT operator is implemented.
