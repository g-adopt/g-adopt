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
| `multiplicative` | `CoupledInternalVariableSolver` with the shipped preset, outer iterations capped at 40. |
| `schur-a11` | Schur fieldsplit, internal variables eliminated exactly, GAMG on the elastic block. |
| `schur-substituted` | Same layout, GAMG on the substituted operator (`gadopt.SubstitutedDisplacementPC`). |
| `static-condensation` | Slate static condensation (`gadopt.InternalVariableSCPC`), GMRES and GAMG on the condensed operator. |

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
substitution, and the residual evaluation of the full three-field form.

## Open items

- **The shipped preset is still the multiplicative fieldsplit.** The
  static-condensation configuration should become the iterative preset of
  `CoupledInternalVariableSolver`. Its option set is
  `StaticCondensationCoupledInternalVariableSolver._iterative_preset` in the
  variants module; moving it changes the default for every user of the
  coupled solver and the expectations in
  `tests/unit/test_stokes_solver_configuration.py`.
- **No tests cover the new preconditioners.** `gadopt.InternalVariableSCPC`
  overrides `condensed_system` and `local_solver_calls` of Firedrake's
  `SCPC` and reads its private attributes, and
  `gadopt.SubstitutedDisplacementPC` reads the elastic block's form from the
  matrix-free context. Both need a small test (level 1, serial, a few
  steps) that checks the V-cycle count against the reference, so that a
  Firedrake update that breaks them is caught.
- **Two timing loops.** `burgers_sphere.model` and
  `benchmark_internal_variable_solvers.main` contain the same solve loop,
  failure handling, L2 difference and JSON summary. One
  `run_comparison(problem, configs, steps)` in the variants module would
  serve both.
- **Power-law rheology is not covered.** `InternalVariableSCPC` drops the
  cross coupling between internal variables that the power-law Jacobian
  introduces, and the Schur layout inverts each internal-variable block on
  its own. Both are exact only for a Newtonian rheology.
- **GPU path untested** for the Schur and static-condensation layouts.
  `StokesSolverBase._configure_iterative_solver` refuses the GPU offload
  when the displacement block is an assembled matrix (static condensation).
