# Prior-covariance campaign on the box case

This directory holds the job scripts of the box campaign of the branch
`sghelichkhani/hessian`. The campaign runs on Setonix, account `pawsey0821`.
The run outputs are not in git.

## Files

| file | purpose |
|---|---|
| `sync_code.sh` | Copies one commit to a directory on Setonix and compares the md5 of the tracked files. Runs on reunion. |
| `forward.slurm` | Makes the reference state of one mesh. |
| `inverse.slurm` | Runs `inverse.py` once, serial, with the arguments you give. |
| `smoke.slurm` | Runs the tiny box case with two code copies and compares the two records. |
| `submit_s0.sh` | Submits step S0. Runs on a Setonix login node. |
| `submit_s1.sh` | Submits acceptance test 3 of step S1. Runs on a Setonix login node. |
| `submit_s2.sh` | Submits step S2, the wall treatment: a smoke job, a regression run and five long runs. Runs on a Setonix login node. |
| `smoke_pinned.slurm` | Runs the two pinned cases of S2 for two iterations and checks them. |
| `check_pinned.py` | Checks that pinned wall nodes stay in BNK's fixed set and do not move. |
| `summarise_walls.py` | Prints the S2 tables: misfit against work by wall row, Krylov counts, active sets, wall-node entries and exits. |
| `submit_s2b.sh` | Submits step S2b, the Hessian shift of BNK: two smoke jobs and four long runs. Runs on a Setonix login node. |
| `smoke_s2b.slurm` | Checks the shift recording and the shift options on short runs, and checks a default run bitwise against its S2 record. |
| `check_shift.py` | Checks that every recorded shift follows the update rules of BNK. |
| `submit_s2c.sh` | Submits step S2c, BNK's shift in the metric: the 150-cell reference, three smoke jobs, five 64-cell runs and two 150-cell runs. Runs on a Setonix login node. |
| `smoke_s2c.slurm` | Runs one short 64-cell case, compares it with its S2 record and runs `check_s2c.py`. |
| `check_wrapper.py` | Checks the metric-shifted Hessian against an assembled reference on the tiny case, and that it differs from an identity shift. Runs in seconds. |
| `check_s2c.py` | Checks the shift mode, that the Hessian operator applied BNK's shift, and the curvature record of every negative-curvature solve. |
| `submit_s2d.sh` | Submits step S2d, the wall treatment under the zero shift: an optional smoke job, five 64-cell runs and the 150-cell run Z150. Takes the step directory. Runs on a Setonix login node. |
| `submit_v2.sh` | Submits step V2, the move to the container of 2026-11-20. Phase `gates` submits the short checks of the container, phase `runs` calls `submit_s2d.sh` without a smoke job. Runs on a Setonix login node. |
| `gates_v2.slurm` | Runs the tiny-case gates of a container: `check_wrapper.py` in three wall settings, the Hessian Taylor harness, `check_box_hessian.py` and gate 1. |
| `check_box_hessian.py` | Checks the Hessian-vector product of the tape of `problem.py` on the tiny case: second-order Taylor remainders and a central difference of the gradient. Runs in about a minute. |
| `summarise_shift.py` | Prints the S2b tables: work to reach values of `J`, `initial_misfit` there, negative-curvature solves, `ritz_min` and the shift per solve. |
| `fetch.sh` | Copies the records of one step from Setonix to reunion. Runs on reunion. |
| `compare_records.py` | Compares two JSON records row by row. |
| `check_gauss_newton.py` | Checks G1 to G13 of the Gauss-Newton product on the tiny case, G11 on the 64-cell state (`--g11`), and the product prices of the box rule (`--timing`). |
| `summarise_gn.py` | The decision rule of the Gauss-Newton box runs (tests A, B, C, the floor guard and the side readings). |
| `test_summarise_gn.py` | Tests of `summarise_gn.py` on synthetic records. Pure Python, not part of CI. |
| `gadi/` | The job scripts of the Gauss-Newton box runs on Gadi; see `gadi/README.md`. |

## Locations

| item | location |
|---|---|
| code copy of the campaign | `/software/projects/pawsey0821/sghelichkhani/g-adopt-hessian/` |
| other code copies | `/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/<step>/code-<commit>/` |
| reference states | `/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/reference/` |
| run outputs on Setonix | `/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/<step>/` |
| run outputs on reunion | `~/runs/prior-campaign/<step>/` |
| environment file | `/software/projects/pawsey0821/sghelichkhani/env-hessian-v2.sh` (the container of 2026-11-20); `env-hessian.sh` holds the environment of the runs before 2026-11-21 |

Setonix accepts the ssh key of phoenix only. Every command from reunion goes through
phoenix: `ssh phoenix 'ssh setonix "<command>"'`.

## Sync the code

1. Commit the change on reunion.
2. Run `sync_code.sh <commit> <Setonix directory>` from the worktree.
3. Read the last lines of the output.
4. If a file differs, do not submit a job.
5. If a line starts with "untracked on Setonix", remove that file or explain it.

The script replaces the directory. It keeps the old copy as `<directory>.previous`.
It moves `tests/adjoint/adjoint-demo-checkpoint-state.h5` into the new copy.

CAUTION: do not change a code copy while a job uses it. Python can load a module
late, and the job then runs a mix of two commits.

## Submit a step

1. Log in to Setonix through phoenix.
2. Run the submit script of the step, for example
   `submit_s0.sh <code directory D1> <code directory D2>`.
3. Write down the job numbers that the script prints.
4. Monitor the jobs with `squeue -u $USER`.

Every job script holds the four lines that `srun` needs on Setonix:
`#!/bin/bash --login`, `#SBATCH --export=NONE`, `unset SLURM_CPUS_PER_TASK` and
`export SRUN_CPUS_PER_TASK=1`. Keep them in every new job script.

Every job prints the path of `gadopt` before it starts. The path must be in the code
copy that you gave to the job.

## Fetch the records

1. Wait until the jobs of the step stop.
2. Run `fetch.sh <step>` on reunion.
3. Find the records in `~/runs/prior-campaign/<step>/`.

The script does not copy `.h5` files and code copies.

## Compare two records

Run `python compare_records.py <reference.json> <run.json>`.

The default fields are `J`, `initial_misfit`, `gradient_norm` and the three sweep
counts. Add `--all-fields` to compare every numeric field that both records hold.
Add `--rows both` to compare the gradient rows as well as the iteration rows. Add
`--solves` to compare the Krylov solve records of the shared iteration rows. Add
`--max-rows N` to compare only the first N rows. The
exit status is 0 only when every compared field is bitwise equal.

The field `work` uses the measured wall time of a Hessian-vector product. Thus
`work` is the one old field that differs between two runs with bitwise-equal paths.
The script does not compare it. The work budget tests `work`, so two runs with the
same path can stop at different iterations. The script compares only the rows that
both records hold.
