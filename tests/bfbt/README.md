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
reports cold and warm timing, iteration counts, and equation residuals.

```bash
python tests/bfbt/benchmark.py \
  --case tala --contrast 1e10 --pc mass --n 24 --velocity-pc gamg
python tests/bfbt/benchmark.py \
  --case tala --contrast 1e10 --pc bfbt --n 24 --velocity-pc gamg

python tests/bfbt/benchmark.py \
  --case viscoplastic --pc mass --n 12 --velocity-pc gamg
python tests/bfbt/benchmark.py \
  --case viscoplastic --pc bfbt --n 12 --velocity-pc gamg
```

The available cases are `linear`, `tala`, `ala`, and `viscoplastic`. BFBT
tuning options are exposed as command-line arguments. PETSc options can also
be appended and are left in `sys.argv` for PETSc to consume.

## Local reference measurements

The following single-rank Apple-silicon measurements use the commands above.
They are smoke-performance evidence, not production scaling results.

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

The projected DG0 square-root-viscosity weight avoids passing the
viscoplastic rheology's large symbolic polynomial degree into the auxiliary
forms. In this test it removes the TSFC quadrature-degree warning without
changing the nonlinear iteration count or final residual.

The tuned defaults are diagonal mass lumping, a DG0 weight and an inner
FGMRES tolerance of `1e-2`. Row-sum lumping and a DG1 weight did not improve
the tested TALA solve. An inner tolerance of `1e-1` reduced inner work but
increased outer pressure iterations from 15 to 20 in the tuning case.

On very small problems with a direct velocity solve, BFBT can be slower even
when it reduces pressure iterations. Its two auxiliary pressure solves have a
fixed cost. Production assessment should therefore use representative MPI
meshes, AMG velocity solves and warm timings.
