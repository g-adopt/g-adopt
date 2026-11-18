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
| `fetch.sh` | Copies the records of one step from Setonix to reunion. Runs on reunion. |
| `compare_records.py` | Compares two JSON records row by row. |

## Locations

| item | location |
|---|---|
| code copy of the campaign | `/software/projects/pawsey0821/sghelichkhani/g-adopt-hessian/` |
| other code copies | `/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/<step>/code-<commit>/` |
| reference states | `/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/reference/` |
| run outputs on Setonix | `/scratch/pawsey0821/sghelichkhani/hessian/prior-campaign/<step>/` |
| run outputs on reunion | `~/runs/prior-campaign/<step>/` |
| environment file | `/software/projects/pawsey0821/sghelichkhani/env-hessian.sh` |

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
Add `--rows both` to compare the gradient rows as well as the iteration rows. The
exit status is 0 only when every compared field is bitwise equal.

The field `work` uses the measured wall time of a Hessian-vector product. Thus
`work` is the one old field that differs between two runs with bitwise-equal paths.
The script does not compare it. The work budget tests `work`, so two runs with the
same path can stop at different iterations. The script compares only the rows that
both records hold.
