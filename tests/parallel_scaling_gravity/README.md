# Gravitational Poisson DtN scaling study

Weak/stress-scaling harness for `gadopt.GravitySolver`. Design and reasoning live
in `../../SCALING-ANALYSIS.md`; this README is the operational summary.

## What the study asks

Three questions, and every table the summarizer prints answers one of them.

*How well does the solve scale at a fixed truncation?* Read down a column: the
`fieldsplit_0` count per invocation and the steady solve time should stay flat as
the level rises, because each level multiplies both the mesh and the rank count
by eight and holds the potential block near thirteen thousand degrees of freedom
per core.

*How high a truncation can a given resolution carry?* Read along a row of the
accuracy table. The error against the closed-form potential falls by about four
per level while the mesh resolves the modes being asked for, and stops falling
once it does not. That is a resolution limit on `L`, measured rather than
argued.

*What does one more angular mode cost?* Read the marginal-cost table. It is not
a constant: each matrix-free application re-assembles the boundary mode forms at
`O(L^4)`, and the per-process symbolic form processing grows with the mode count
too.

## The vehicle

`gravity_cubed_sphere.py` runs one `(level, L)` case of the coupled solve. The
128-field `PCFIELDSPLIT` wall that used to cap it at `L = 6` is gone -
`gadopt.DtNTwoBlockSchurPC` describes the two Schur blocks by index set, so the
sub-field enumeration that carried the cap never runs - and the coupled solve now
reaches every truncation in the study. `capacitance_gravity.py` remains in the
tree: it was the wall-free diagnostic that answered the `L`-cost question while
the wall stood, and it is no longer on the critical path.

Correctness comes from `analytic_gravity.py`, which gives the closed-form
potential of the study's own source. Because the source is band-limited to
degree `L` and the DtN maps are exact for every mode they treat, that is the
exact solution of the continuous problem, so the difference is discretisation
error with no truncation component. Every case is checked against it. This is
not decoration: a matrix-free mixed operator with Real blocks has been observed
in this code area to converge cleanly, in parallel, to a wrong answer, so "it
converged" is not evidence and an error against a known answer is.
`check_analytic_vs_passess.py` pins that reference against `passess`, which
solves the constant-density case in closed form.

## Cold and warm

Each case runs twice in one job. The first invocation meets an empty kernel
cache, the second meets the cache the first left behind, which separates three
costs that are otherwise reported as one:

| regime | pays |
| ------ | ---- |
| cold first solve | C compilation, TSFC form compilation, per-process symbolic work, the solve |
| warm first solve | per-process symbolic work, the solve |
| steady solve | the solve |

The second difference is the interesting one. Turning an `n`-term boundary mode
sum into kernels is per-process work that recurs in every fresh interpreter no
matter how warm the on-disk cache is, and it grows with the truncation. Caches
live on `$PBS_JOBFS`, which is node-local and wiped at job end, so every job's
cold phase is genuinely cold and no two nodes race to write the same file.

## Running

```bash
# one case, both phases (the PBS scripts do this for you)
python3 gravity_cubed_sphere.py 4 --lmax 5 --cache-phase cold --out-dir results/
python3 gravity_cubed_sphere.py 4 --lmax 5 --cache-phase warm --out-dir results/

# the direct-vs-iterative anchor
python3 gravity_cubed_sphere.py 4 --lmax 5 --anchor --out-dir results/

# generate the Gadi PBS grid (dry run; --submit to qsub)
python3 submit_gravity_scaling.py --levels 4 5

# read the tables
python3 summarize_gravity_scaling.py --results-dir results/

# check the closed-form reference against passess (developer machine only)
python3 check_analytic_vs_passess.py
```

Run against a specific worktree with `PYTHONPATH=<worktree>` so the editable
G-ADOPT install does not shadow it. On Gadi the submit script pins `PYTHONPATH`,
the `fp50` Firedrake module, project `xd2`, the `normalsr` queue, and the kernel
caches automatically.

Submission is staged on purpose: run levels 4 and 5, read the measured cold and
warm costs, then size and submit levels 6 and 7 from those numbers rather than
from a guess. The cold cost at several hundred modes had never been measured
when this study was designed.

## Files

| file | role |
| ---- | ---- |
| `gravity_cubed_sphere.py` | coupled-solve model + correctness anchor |
| `analytic_gravity.py` | closed-form potential of the study's source |
| `check_analytic_vs_passess.py` | pins that reference against passess |
| `capacitance_gravity.py`  | wall-free capacitance L-cost diagnostic (superseded) |
| `submit_gravity_scaling.py` | Gadi PBS generator (dry run / `--submit`) |
| `summarize_gravity_scaling.py` | reads sidecars, prints the tables, enforces the floor |
| `scaling.py`, `meta.py`, `run.template`, `test_scaling.py` | layer-two doit/longtest integration |
| `expected.csv`, `expected_capacitance.csv` | regression references |

The JSON sidecars each run writes are the stable interface the summarizer and
tests read; the raw PETSc logs remain the source of truth.

## Layer-two staging

`test_scaling.py` parametrizes over every level, and `expected.csv` is seeded
with level-4 numbers only, so most cases have no reference row and would fail.
Do **not** add this suite to the weekly `longtest.yml` rotation. The reference
CSVs are repopulated from the Gadi run; wiring the suite into CI is deliberately
left alone, because a CI regression test and a scaling study want different
things from the same harness and conflating them has already cost this study
once.
