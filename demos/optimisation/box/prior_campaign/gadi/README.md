# Gauss-Newton box runs on Gadi

This directory holds the job scripts of the Gauss-Newton box runs of the branch
`sghelichkhani/hessian` on Gadi, project `nm03`, queue `normal`. The plan, the costs
and the decision rule are in `NOTES/team/gauss-newton/PLAN.md` of the reunion
worktree (revision 3, sections 3 and 4). The run outputs are not in git.

## Files

| file | purpose |
|---|---|
| `job_env.sh` | The environment block of every job. It sets `ENV_FILE`, sources it after `MY_GADOPT`, and prints the code, the versions and the node. |
| `run_flags.sh` | The flags of the runs: `COMMON`, `NEWTON`, `ROL` and one array per arm. |
| `gates_gn.pbs` | The tiny-case gates: G0 against the parent commit, `check_gauss_newton.py` and `check_box_hessian.py`. 16 GB, 1 h. |
| `smoke_gn.pbs` | The 64-cell smoke: `timing.json`, case (c) against Setonix, G11, and short runs of T-ROL and T-N (2 iterations) and of T-GN-R-EW and T-GN-R (2 iterations, Krylov cap 5). 24 GB, 4 h. |
| `inverse.pbs` | One run of `inverse.py` with the arguments of `<run directory>/args`. 24 GB, 20 h. |
| `submit_gn.sh` | Submits phase `gates`, `runs` or `optional <arm>`. Runs on a Gadi login node. The header holds the decision rule in words. |
| `sync_code.sh` | Copies one commit to Gadi and compares the md5 of every tracked file under `gadopt/` and `demos/optimisation/box/`. Runs on reunion. |
| `fetch.sh` | Copies the records from Gadi to `~/runs/prior-campaign/GN/` on reunion. Runs on reunion. |

## Locations

| item | location |
|---|---|
| environment file | `/g/data/nm03/sg8812/firedrake-hessian/env-hessian.sh` (`ENV_FILE` in `job_env.sh`) |
| code copies | `/scratch/nm03/sg8812/hessian/gauss-newton/code-<commit>/` |
| parent code copy for G0 | `/scratch/nm03/sg8812/hessian/gauss-newton/code-84ade15f/` |
| reference state | `/scratch/nm03/sg8812/hessian/gauss-newton/reference/r64-s80/reference-state.h5` |
| Setonix records for the container gate | `/scratch/nm03/sg8812/hessian/gauss-newton/env/setonix-ref/` |
| run directories | `/scratch/nm03/sg8812/hessian/gauss-newton/{gates,smoke,T-ROL,T-N,T-N-R,T-GN-R,T-GN,T-GN-R-EW}/` |
| records on reunion | `~/runs/prior-campaign/GN/` |

Every PBS script requests `-l storage=gdata/nm03+scratch/nm03+gdata/fp50`. The
environment reads its images from private copies in `gdata/nm03`, but petsc4py names
`/g/data/fp50/apps/petsc/3.26.0` in its RPATH, so `gdata/fp50` must be visible.

## Procedure

Every command to Gadi starts with `source /etc/profile`, without which Gadi does not
find `qsub`, `qstat` or `module`.

1. Commit the reviewed code on reunion.
2. Run `sync_code.sh <commit>` from the worktree. If a file differs, do not submit.
3. Submit the gates:
   `ssh gadi 'source /etc/profile && bash <code>/demos/optimisation/box/prior_campaign/gadi/submit_gn.sh <code> gates'`.
4. Wait for state `F` with `qstat -x <job id>`. Do not use a `pgrep` loop.
5. Read `gates/gates.status`, `smoke/smoke.status` and the logs of each part. Write
   `NOTES/team/gauss-newton/GATES.md` with one line per pass item, the measured value
   and PASS or FAIL.
6. Write `r_GN = gn_over_full`, `full_per_forward` and the forward time of
   `smoke/timing.json` into GATES.md. If the forward time gives an expected run time
   above 1.5 times the estimate of PLAN.md section 4.4, report to Sia first.
7. Submit the runs: `submit_gn.sh <code> runs`. Write the job ids into GATES.md.
8. When every run has state `F`, run `fetch.sh` on reunion, then `summarise_gn.py`
   with `--timing-json ~/runs/prior-campaign/GN/smoke/timing.json`.

The smoke job runs the parts that the rule needs first (timing, the container gate,
G11). The Gauss-Newton smoke runs have a Krylov cap of 5, so the job ends inside its
4 h limit. Each job exits with status 1 when one of its parts failed.

CAUTION: do not change a code copy while a job uses it. `sync_code.sh` refuses an
existing directory for this reason.
