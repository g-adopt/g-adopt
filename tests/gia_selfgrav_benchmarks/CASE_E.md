# Martinec E: infer the initial topography

Case E uses D's growing ice cap, moving coastline and floating ice. Basin
B2 is prescribed at **15 kyr**, rather than at the initial time. The case
specification comes from `giamip.case("martinec2018-E")` and Martinec et al.
(2018), equations 32–34, DOI: https://doi.org/10.1093/gji/ggy280.

With topography positive upward, the terminal field is

```
zeta_terminal = zeta_initial + U - N - h_UF = -SL_terminal
```

The desired field is the CG2 interpolation of the analytic B2 basin, using
the same basin representation as D. The reported RMS compares with that
represented target, rather than measuring its interpolation error. The physical relative sea-level
change `N + h_UF - U` need not vanish. Both fitting methods minimise the
**terminal topography error**, allowing nonzero physical sea-level change.

## Methods and numerical choices

`--topography_method picard` is the default. Every iteration starts a fresh
zero-deformation, zero-ice history and advances all 300 steps (50 yr) to
15 kyr. After testing the terminal error, it applies equation 34:

```
zeta_initial_next = zeta_initial + zeta_target - zeta_terminal
```

The default relaxation is 1; `--topography_relaxation` can reduce it. The
default iteration limit is six full histories, following the paper's report
of at most six iterations. This is a limit, not a promise of convergence for
this discretisation. `--topography_max_iterations` can increase it.

`--topography_method rol` uses Firedrake adjoint and G-ADOPT's Lin-More ROL
optimiser. The control is a CG2 correction on the outer mantle surface,
scaled by 1000 m; interior topography DOFs are not optimisation variables.
The objective is half the area-normalised squared terminal error divided by
`(1000 m)^2`. The same maximum-iterations option limits ROL iterations;
each iteration may require several forward/adjoint histories. An independent
forward history evaluates the final control. Optimiser termination alone
does not establish convergence. The optimiser also stops when its objective
reaches the physical RMS threshold. The ROL binding requires bounds; these
are set to ±1e20 in control units (±1e23 m), far outside meaningful corrections.

Both methods start from the terminal basin and use **whole-surface,
area-normalised RMS < 0.01 m**. The paper specifies an L2 threshold of
0.01 m but leaves its normalisation and starting estimate unspecified.
The smooth-mask slope is computed from the target basin and held fixed
throughout both fits. This makes the two methods solve the same discretised
problem and avoids differentiating through an unannotated slope recovery.
The Earth model, production mesh, compressibility (`K/mu=100`), DtN degree
(5) and inner solver follow D.

Use `--time_step_yr 500` with either method for 30 steps to 15 kyr. Omitting
the option retains the 50 yr production default (300 steps). An explicit
timestep also overrides E's smoke timestep. The uniform timestep is adjusted
to divide the terminal time exactly if necessary; the actual value is logged
and saved in `summary_E.json`. Restarts must use the same timestep as the
checkpoint, including staged Picard passes.

## Smoke runs

From this directory in a compatible Firedrake environment:

```bash
python3 martinec.py --case E --smoke --topography_one_pass \
    --output_path smoke/martinec_E
python3 martinec.py --case E --smoke --topography_one_pass \
    --topography_restart smoke/martinec_E/checkpoint_E_next.h5 \
    --output_path smoke/martinec_E
```

By default, E smoke covers all 15 kyr in six 2500 yr steps on the smoke mesh;
`--time_step_yr` can override this grid. It exercises the topography update
but supplies no benchmark-quality numbers. Smoke
runs retain the measured convergence flag even when the iteration limit is
reached. Production runs fail if the fit remains above tolerance.

## Staged Picard runs on Gadi

Submit from this directory. Set `OUTPUT` to a shared filesystem directory
that persists between jobs. As for C/D, provide `EXTRA_PYTHONPATH` if the
module does not include `giamip` (see README).

```bash
qsub -N martinec_E_01 -l ncpus=1872 -l mem=9000GB -l walltime=15:00:00 \
    -v "DRIVER=martinec,CASE=E,OUTPUT=$PWD/outputs/martinec_E,EXTRA=--topography_one_pass" \
    run_benchmark.pbs
```

After it finishes, inspect `summary_E.json`. If `topography.converged` is
false, start the next full history using the saved next guess:

```bash
qsub -N martinec_E_02 -l ncpus=1872 -l mem=9000GB -l walltime=15:00:00 \
    -v "DRIVER=martinec,CASE=E,OUTPUT=$PWD/outputs/martinec_E,EXTRA=--topography_one_pass --topography_restart $PWD/outputs/martinec_E/checkpoint_E_next.h5" \
    run_benchmark.pbs
```

Repeat until converged; the checkpoint records the next pass number.
The suggested resources are based on D's approximately ten-hour history;
E has not yet been timed on Gadi. E is excluded from `meta.py`'s automatic
single-job suite because a multi-history fit can exceed the 24-hour limit.

If a job stops **inside** a history, resume its dynamic checkpoint instead:

```bash
qsub -l ncpus=1872 -l mem=9000GB -l walltime=15:00:00 \
    -v "DRIVER=martinec,CASE=E,OUTPUT=$PWD/outputs/martinec_E,RESTART=$PWD/outputs/martinec_E/topography_02/checkpoint_E.h5,EXTRA=--topography_one_pass" \
    run_benchmark.pbs
```

Use the same rank count and recorded physics, method, tolerance, relaxation
and smoke setting. `--restart` resumes a partial history;
`--topography_restart` starts a fresh history from a saved topography.
`initial_topography_E.h5` holds the guess used by the latest evaluated
history; it can also be supplied as a topography restart to repeat that fit.

## Adjoint/ROL

This route additionally requires the ROL Python module used by
`gadopt.inverse` (for example `pyroltrilinos`). First exercise the derivative
against independent fresh forward solves:

```bash
python3 martinec.py --case E --smoke --topography_method rol \
    --check_topography_gradient --topography_max_iterations 12 \
    --output_path smoke/martinec_E_rol
```

`topography_gradient.json` reports an adjoint directional derivative, a
central difference of fresh histories, and their relative difference. The
check fails above 1% or if the taped and fresh objective disagree. It tests
a constant surface direction; it is not a full mesh/time convergence study.

Use the same options without `--smoke` for production. Revolve uses
`--adjoint_checkpoints` RAM snapshots (default five) and recomputes states
between them. Memory includes each stored coupled state, and runtime includes
replays, adjoints and optimisation evaluations. Budget a separate job after
measuring the smoke/production pilot. ROL supports a saved-topography warm
start via `--topography_restart`, but does not resume a partially taped
history or retain optimiser state across jobs.

## Outputs and reference comparisons

- `summary_E.json`: terminal RMS, convergence, pass history, settings and cost.
- `topography_XX/`: each evaluated full history's time series, parameter log
  and optional within-history checkpoints.
- `martinec-E-timeseries.npz`: the latest evaluated history, even when the fit
  is unconverged. Check the summary before treating it as a benchmark result.
- `initial_topography_E.h5`: initial sea-level field used by that history;
  negate and multiply by the displacement scale to obtain topography in m.
- `checkpoint_E_next.h5`: updated Picard guess for the next history, when needed.
- `--write_output`: mechanics/potential VTK files plus initial and terminal
  topography in metres for each pass.
- ROL additionally writes `topography_rol_evaluations.json`.

The published E comparisons are ocean area (Fig. 18) and `h_UF` (Fig. 19)
over 0–15 kyr. `giamip` supplies these via `BenchmarkResult.reference("E")`.
It supplies no E meridian profiles. Contributor `ZM_case_D` in the E archive
is the paper's D overlay; exclude it when computing the E contributor spread.

`test_benchmarks.py::test_martinec_E` requires a converged, non-smoke,
complete history, terminal `h_UF` within 1% of VEGA and terminal ocean area
inside the E contributor range. The scalar tolerances extend this repo's
D checks; they are local criteria, not pass tolerances stated by the paper.
The test is skipped when E's summary is absent, so the existing automated
B/C/D suite still runs. To check a run in another output directory, copy its
summary/time series into this directory or use `martinec_result("E", directory)`
for analysis without copying. A passed fit does not demonstrate mesh/time
convergence or agreement over the entire published time series.

To report RMS and maximum differences for both complete time series on the
login node, run from this directory with the reference cache available:

```bash
python3 - <<'PY'
from pathlib import Path
import giamip
from test_benchmarks import martinec_result
result = martinec_result("E", Path("outputs/martinec_E"))
for row in giamip.case("martinec2018-E").score(result):
    print(row)
PY
```

This reports differences; it does not assign a paper-defined pass tolerance.

## Local validation

Picard, staged topography restarts and within-history restarts were exercised
in a local Firedrake container. A modified workflow used a coarser 5689-cell
mantle, three 5000 yr steps, a tenfold reference relaxation time and direct
inner preconditioners. These overrides are confined to the external test
runner; production settings remain those of D. Picard RMS decreased from
34.69 m to 1.086 m to 0.0721 m, then reached a coarse discretisation floor
near 0.0616 m. This is a workflow check, not a converged Case E result.

Resuming a partial history reproduced terminal h_UF within 1.1e-12 m.
Surface control/projection and adjoint assembly passed distributed checks,
including a rank with no surface cells. The coupled adjoint directional
derivative agreed with fresh central differences to a relative difference
of 4.8e-7 in the modified workflow. One actual ROL iteration reduced RMS
from 34.6878 m to 33.9982 m; an independent final history confirmed that
value and wrote the summary and warm-start checkpoint. This verifies the
optimiser workflow, not convergence of the fit. Output validation tests
also passed.

The container used Firedrake 2026.10.0 with PETSc 3.25.5; that Firedrake
release requests PETSc 3.26, so this environment is not a fully matched
production stack. A production run on Gadi, convergence below 0.01 m and
comparison with the published E time series remain to be established.
