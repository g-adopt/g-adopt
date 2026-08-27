# Density-aware BFBT benchmarks

`benchmark.py` compares Firedrake's pressure-mass Schur preconditioner with
`gadopt.DensityAwareBFBTPC`. Both configurations use the same full Schur
factorisation, velocity solver and outer pressure tolerance. The linear cases
use one `ksponly` solve; the viscoplastic case uses the same Newton tolerances
for both pressure preconditioners.

The implemented inverse Schur approximation is

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

Run one configuration per process from the repository root. The JSON output
reports maximum-rank cold timing, the median and samples of repeated warm
timings, fieldsplit solve/iteration counts for every warm sample,
convergence failures, MPI size, and equation residuals. The benchmark uses
communicator barriers and only rank zero writes JSON.

For BFBT, the JSON additionally reports both inner pressure solves and their
iterations. It does not currently instrument the MassInvPC baseline's private
`Mp_ksp`, so nested iteration-work totals are not symmetric between those two
arms. Wall time and the common fieldsplit counters remain comparable.

```bash
python tests/bfbt/benchmark.py \
  --case tala --contrast 1e10 --pc mass --n 24 --velocity-pc gamg \
  --warm-repeats 5
python tests/bfbt/benchmark.py \
  --case tala --contrast 1e10 --pc bfbt --n 24 --velocity-pc gamg \
  --warm-repeats 5

python tests/bfbt/benchmark.py \
  --case viscoplastic --pc mass --n 12 --velocity-pc gamg
python tests/bfbt/benchmark.py \
  --case viscoplastic --pc bfbt --n 12 --velocity-pc gamg
```

The available cases are `linear`, `tala`, `ala`, and `viscoplastic`. BFBT
tuning options are exposed as command-line arguments. PETSc options can also
be appended and are left in `sys.argv` for PETSc to consume.

## Local reference measurements

The following earlier single-rank Apple-silicon measurements used one warm
sample and recorded only the final nested solve's iterations. They are
retained as historical smoke evidence, not as a production speedup claim.
Re-run the current harness and compare ``warm_work_samples`` and repeated
wall times before making a performance decision.

| Case | Pressure PC | Pressure iterations | Warm seconds | Residual |
| --- | --- | ---: | ---: | ---: |
| Boussinesq, contrast 1e10 | Mass | 19 | 0.426 | 3.45e-9 |
| Boussinesq, contrast 1e10 | BFBT | 10 | 0.314 | 3.43e-9 |
| TALA, contrast 1e10 | Mass | 32 | 0.922 | 2.14e-9 |
| TALA, contrast 1e10 | BFBT | 14 | 0.501 | 2.10e-9 |
| ALA, contrast 1e10 | Mass | 30 | 0.816 | 8.76e-11 (continuity) |
| ALA, contrast 1e10 | BFBT | 14 | 0.512 | 4.68e-11 (continuity) |
| Viscoplastic, 5 Newton steps | Mass | 12 | 0.906 | 3.56e-9 |
| Viscoplastic, 5 Newton steps | BFBT | 7 | 0.820 | 3.56e-9 |

The DG0-interpolated square-root-viscosity weight avoids passing the
viscoplastic rheology's large symbolic polynomial degree into the auxiliary
forms. In this test it removes the TSFC quadrature-degree warning without
changing the nonlinear iteration count or final residual.

The tuned defaults are diagonal mass lumping, a DG0 weight and an inner
FGMRES tolerance of `1e-2`. Row-sum lumping and a DG1 weight did not improve
the tested TALA solve. An inner tolerance of `1e-1` reduced inner work but
increased outer pressure iterations from 15 to 20 in the tuning case.

Inner BFBT failure is fatal by default. Each application records both inner
solve reasons and total work. ``bfbt_raise_on_inner_failure false`` is
intended only for controlled failure diagnostics. Exact pressure-nullspace
attachment is verified by default. The benchmark explicitly opts into
``bfbt_nullspace_policy=schur`` for ALA so it can compare G-ADOPT's existing
non-exact analytical quotient; the output reports that discrepancy and gives
momentum and continuity residuals separately.

ALA use remains experimental until the non-exact analytical gauge and the
full projected residual have been validated on the target discretisation.
Its pressure-buoyancy term also makes the assembled inner GAMG operator
nonsymmetric, which requires production-rank hierarchy and convergence tests.
The preconditioner is currently forward-only: transpose application fails
clearly instead of assuming that every selectable inner PC supplies a valid
numerical transpose.

The current local tests are two-dimensional smoke and regression cases. A
production claim still requires repeated MPI runs on the three-dimensional
extruded spherical mesh, a frozen fully plastic checkpoint, and an end-to-end
nonlinear slice. Report total fieldsplit_0, fieldsplit_1, and BFBT inner work,
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
    "bfbt_ksp_rtol": 1e-2,
    "bfbt_ksp_max_it": 200,
    "bfbt_pc_type": "gamg",
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
