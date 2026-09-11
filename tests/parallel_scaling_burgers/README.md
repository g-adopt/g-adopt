# Weak scaling of the Burgers internal-variable solvers

This directory measures five solver configurations for the Burgers
viscoelastic sphere of `tests/3d_sphere_burgers` at cubed-sphere levels 5, 6
and 7. Each job runs the substituted reference solver and one coupled
configuration on the same mesh and reports GAMG V-cycles and wall time for
each.

Configurations (see `tests/3d_sphere_burgers/coupled_solver_variants.py`;
the names and the option layout are in `solver_configs.py` next to it):

| name | solver |
|---|---|
| `substituted` | `InternalVariableSolver`, the reference. Runs in every job. |
| `multiplicative` | `CoupledInternalVariableSolver` with the symmetric multiplicative fieldsplit that preceded static condensation (`multiplicative_gia_solver_parameters`), outer iterations capped at 40. |
| `schur-a11` | Schur fieldsplit, internal variables eliminated exactly, GAMG on the elastic block. |
| `schur-substituted` | Same layout, GAMG on the substituted operator (`gadopt.SubstitutedDisplacementPC`). |
| `static-condensation` | Slate static condensation (`gadopt.InternalVariableSCPC`), CG and GAMG on the condensed operator. This is the shipped preset of `CoupledInternalVariableSolver`. |

## Running

Locally (levels 1 to 3 are for smoke tests):

    python burgers_scaling.py 1 schur-a11 -n 3
    mpiexec -n 4 python burgers_scaling.py 2 static-condensation -n 3

On Gadi, one job for each level and configuration:

    qsub -v LEVEL=5,CONFIG=schur-a11 -l ncpus=104 -l mem=480GB -l walltime=01:00:00 pbs_job.sh

`pbs_job.sh` lists the resources per level. It runs a one-step warmup that
pays the compilation cost, then the measured run with a PETSc log. Two
optional variables: `RUN_TAG=name` adds `_name` to every output file so a
rerun does not overwrite an earlier one, and `LOG_FORMAT=ascii_xml` writes
the nested PETSc log instead of the flat text table.

`meta.py` and `run.template` describe the same jobs for `gadopt_hpc_helper`,
which is how the long tests launch them.

## Reading the results

`burgers_scaling.get_data(level, config, path)` needs no Firedrake. It returns the
V-cycles per step of both solvers, the outer iteration counts, the PCSetUp
and stage times from the flat PETSc log, and the JSON summary that the
driver prints on the line that starts with `GADOPT_BURGERS_SCALING`.

`test_burgers_scaling.py` checks those numbers against `expected.csv` (V-cycles) and
`gadi_expected.csv` (times); both hold the values of the 2026-09-09
campaign. Run with `pytest -m longtest` once the outputs are in place.

`plot_breakdown.py` draws where the time goes. It needs the nested log, so
run the jobs with `RUN_TAG=nested LOG_FORMAT=ascii_xml` first. The flat
text log cannot be used for a breakdown: its events nest inside each other
and each time is the maximum over ranks, so they do not add up to the
total. The script writes `results/breakdown.pdf` (stacked cost per solve,
one panel per solver) and `results/iterations.pdf` (V-cycles against time
per solve).

## Results of the 2026-09-09 campaign

Warm-step wall time per solve relative to the substituted reference, with
V-cycles per step in brackets:

| level | ranks | substituted | multiplicative | schur-a11 | schur-substituted | static-condensation |
|---|---|---|---|---|---|---|
| 5 | 104 | 1.78 s (19) | 13.4 (361) | 3.34 (28) | 2.56 (20) | 1.59 (20) |
| 6 | 832 | 2.05 s (20) | 12.9 (392) | 3.13 (29) | 2.43 (21) | 1.56 (23) |
| 7 | 6656 | 2.8 s (22) | 10.3 (436) | 2.67 (32) | 1.94 (23) | 1.52 (26) |

Static condensation matches the reference's V-cycle count within a few
cycles and costs 1.5 to 1.6 times its wall time at every level. Its extra
cost is the Slate condensation assembly, the elimination and back
substitution, and the residual evaluation of the full coupled form.

Every coupled configuration keeps both Maxwell elements in one
internal-variable field of shape `(2, 3, 3)`, so the internal variables are
field 1 of the mixed space whatever the number of elements.

Measured on 2026-09-11 (jobs 178727788 to 178728386, module
`firedrake/main-20260902`), warm steps 2 to 4, seconds per step:

| level | ranks | substituted | static-condensation | ratio | schur-substituted | schur-a11 | multiplicative |
|---|---|---|---|---|---|---|---|
| 5 | 104 | 1.78 | 1.75 | 0.98 | 3.98 | 5.19 | 22 |
| 6 | 832 | 2.05 | 1.78 | 0.87 | 4.32 | 5.56 | 26 |
| 7 | 6656 | 2.55 | 2.16 | 0.85 | 5.07 | 6.55 | 30 |

V-cycles per step on the condensed field (CG): 21, 22, 25 against 19, 20,
22 for the substituted solver. The condensed operator and its GAMG hierarchy
are built once per job (fixed time step, Newtonian rheology); the substituted
solver rebuilds its operator every step. `expected.csv` and
`gadi_expected.csv` hold this campaign's numbers, read from the nested log.
The figure `results/breakdown.pdf` is regenerated with
`python plot_breakdown.py --tag ""` once the jobs' outputs are in `results/`
(`--tag nested` for jobs submitted with `RUN_TAG=nested`).

## Open items

- **The scaling test's time tolerance is 10 percent** at levels 6 and 7;
  the rerun of 2026-09-11 reproduced the 2026-09-09 stage times of the
  unchanged routes to within 3 to 10 percent, so a failure at that margin
  is noise until more campaigns exist.
- **`gadopt.SubstitutedDisplacementPC` has no unit test.** It reads the
  elastic block's form from the matrix-free context; a Firedrake change
  there would only show up in this scaling test. `InternalVariableSCPC` is
  covered by `tests/unit/test_internal_variable_history.py`.
- **Two timing loops.** `burgers_sphere.model` and
  `benchmark_internal_variable_solvers.main` contain the same solve loop,
  failure handling, L2 difference and JSON summary. One
  `run_comparison(problem, configs, steps)` in the variants module would
  serve both.
- **Power-law rheology is not measured here.** With every Maxwell element
  in one field, the cross coupling that the power-law Jacobian introduces
  sits inside the internal-variable block, so both the Schur layout and
  static condensation invert it exactly; the unit tests cover a small
  power-law solve under Newton. The cost on the sphere is unmeasured.
- **GPU path untested** for the Schur and static-condensation layouts.
  `StokesSolverBase._configure_iterative_solver` now offloads the condensed
  matrix directly through `OffloadPC`; nobody has run it.
