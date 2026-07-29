# Gravitational Poisson DtN scaling study

Weak/stress-scaling harness for `gadopt.GravitySolver`, across both DtN
representations. Design and reasoning live in `NOTES/SCALING-ANALYSIS.md` (that
directory is a local research record and is not tracked); this README is the
operational summary.

## What the study asks

Four questions, and every table the summarizer prints answers one of them.

*Do the two representations compute the same answer?* Everything else is
conditional on this, so it is printed first. `multiplier` promotes every treated
mode to a scalar Real unknown eliminated by a Schur complement; `lowrank`
eliminates them by hand and applies a rank-`n` update to the Robin-shifted
stiffness. They discretise the same operator, so their potentials should agree to
the Krylov tolerance and their errors against the closed form should be
indistinguishable. `--parity` measures exactly that, and it is the first job
generated at each level.

*What does each representation cost?* Read the ratio tables. Steady solve time,
first solve, construction, multigrid work and memory, multiplier over low-rank at
the same `(level, L)`.

*How well does the solve scale at a fixed truncation?* Read down a column. Each
level multiplies both the mesh and the rank count by eight and holds the potential
block near thirteen thousand degrees of freedom per core.

*How high a truncation can a given resolution carry?* Read along a row of the
accuracy table. The error against the closed-form potential falls by about four
per level while the mesh resolves the modes being asked for, and stops falling
once it does not.

## The one metric that needs care

Iteration counts are **not** comparable between the paths and the summarizer does
not compare them. The multiplier path inverts the potential block once per
application of the Schur complement; the low-rank path inverts it once, full
stop. So one gravity solve costs a *sum* over invocations on one path and a
single count on the other.

The comparable quantity is `amg_applications_per_solve`: multigrid V-cycles per
gravity solve. Measured on a toy case, iterations-per-invocation reads 7.4
(multiplier) against 11 (low-rank), which says the fast path is worse;
`amg_applications_per_solve` on the same case reads 192 against 11. Both numbers
are correct and only one is a comparison.

Tolerances differ in *where* they bind and are recorded in every sidecar rather
than assumed: the multiplier path is an outer FGMRES at `rtol` 1e-11 over an
inner CG at 1e-8, the low-rank path a single CG at 1e-11 where that tolerance is
the solution accuracy. Delivered accuracy is therefore comparable, and the
evidence for that is the parity gate, not the presets.

## The vehicle

`gravity_cubed_sphere.py` runs one `(level, L, representation)` case. Correctness
comes from `analytic_gravity.py`, which gives the closed-form potential of the
study's own source. Because the source is band-limited to degree `L` and the DtN
maps are exact for every mode they treat, that is the exact solution of the
continuous problem, so the difference is discretisation error with no truncation
component. This is not decoration: a matrix-free mixed operator with Real blocks
has been observed in this code area to converge cleanly, in parallel, to a wrong
answer, so "it converged" is not evidence and an error against a known answer is.
`check_analytic_vs_passess.py` pins that reference against `passess`.

One caveat on that instrument, unresolved and worth holding in mind when reading
the accuracy tables: the discrete density is the CG1 interpolant of the analytic
one, while the reference potential is the closed form of the *analytic* density.
The two diverge as `L` rises, and the checkerboard weights amplify high degrees
on purpose (16.7x at `l = 30` on the inner bump). The previous campaign read a
rising error past `L = 10` as a resolution ceiling; that reading has a confound in
it and the sidecars now carry enough to separate the two.

## Cold and warm

Each case runs twice in one job. The first invocation meets an empty kernel
cache, the second meets the cache the first left behind, which separates three
costs that are otherwise reported as one:

| regime | pays |
| ------ | ---- |
| cold first solve | C compilation, TSFC form compilation, per-process symbolic work, the solve |
| warm first solve | per-process symbolic work, the solve |
| steady solve | the solve |

The second difference is the interesting one, and it is where the two paths are
expected to separate most: the multiplier path's per-process symbolic work grows
with the mode count and recurs in every fresh interpreter no matter how warm the
on-disk cache is, while the low-rank path's first solve was measured independent
of the mode count. Caches live on `$PBS_JOBFS`, node-local and wiped at job end,
so every job's cold phase is genuinely cold and no two nodes race.

## PETSc counters

Every sidecar carries `petsc_perf`: count, flops, time, message count, message
volume and reduction count for the events the study's questions turn on, split by
the setup / first-solve / solve stages, taken from `getPerfInfo` rather than
parsed back out of a `-log_view` table. `logging_active` is recorded beside them,
because the counters only exist when `-log_view` is set and an empty record would
otherwise be indistinguishable from zero cost.

`ParLoopExecute` in the solve stage is the assembly cost itself, and it is the
clearest single statement of what the low-rank path removes.

## Running

```bash
# one case, one path (the PBS scripts do both phases for you)
python3 gravity_cubed_sphere.py 4 --lmax 5 --representation lowrank \
    --cache-phase cold --out-dir results/

# the gate: same case both ways, compared against each other and the closed form
python3 gravity_cubed_sphere.py 4 --lmax 5 --parity --out-dir results/
mpiexec -np 8 python3 gravity_cubed_sphere.py 4 --lmax 5 --parity  # and in parallel

# generate the Gadi PBS grid (dry run; --submit to qsub)
python3 submit_gravity_scaling.py --levels 4 5 6

# read the tables
python3 summarize_gravity_scaling.py --results-dir results/
```

Levels 2 and 3 exist in `LAYERS` for laptop smoke tests and are not part of the
ladder. Run against a specific worktree with `PYTHONPATH=<worktree>` so the
editable G-ADOPT install does not shadow it. On Gadi the submit script pins
`PYTHONPATH`, the `fp50` Firedrake module, project `xd2`, the `normalsr` queue and
the kernel caches automatically.

Submission is staged on purpose: levels 4 to 6 all fit on one node; level 7 is
eight nodes and the rung where communication rather than arithmetic sets the
time, so its walltime is sized from the level-6 measurement rather than guessed.

## Files

| file | role |
| ---- | ---- |
| `gravity_cubed_sphere.py` | one case, either representation, plus the parity gate |
| `analytic_gravity.py` | closed-form potential of the study's source |
| `check_analytic_vs_passess.py` | pins that reference against passess |
| `capacitance_gravity.py`  | wall-free capacitance L-cost diagnostic (superseded) |
| `submit_gravity_scaling.py` | Gadi PBS generator (dry run / `--submit`) |
| `summarize_gravity_scaling.py` | reads sidecars, prints the tables, enforces the floor |
| `scaling.py`, `meta.py`, `run.template`, `test_scaling.py` | layer-two doit/longtest integration |
| `expected.csv`, `expected_capacitance.csv` | regression references, **currently empty** |

## The empty reference files

Both CSVs hold headers and no rows, deliberately. Every number they carried was
measured under the previous boundary-quadrature default — a rule with no mesh
dependence in it, since replaced by one in `L h / R` — and under the pre-recursion
`real_spherical_harmonic`, whose replacement deliberately changed the default
path's numerical output. Asserting against them would be asserting against a
different solver.

`test_scaling.py` therefore *skips* a case with no reference row rather than
failing it: a suite that is red for a known reason stops being read, which is
worse than one that says plainly what it has not yet been told. Repopulating the
files from a campaign turns the assertions back on with no other change.

The previous campaign is archived at
`/scratch/xd2/sg8812/gravity-scaling-SUPERSEDED-2026-07-29` on Gadi, with a
`SUPERSEDED.md` explaining what changed under it. It is kept so the fixes can be
measured against it, not so its numbers can be quoted.

Do **not** add this suite to the weekly `longtest.yml` rotation. A CI regression
test and a scaling study want different things from the same harness, and
conflating them has already cost this study once.
