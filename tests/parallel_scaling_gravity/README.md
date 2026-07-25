# Gravitational Poisson DtN scaling study

Weak/stress-scaling harness for `gadopt.GravitySolver`. Design and reasoning live
in `../../SCALING-ANALYSIS.md`; this README is the operational summary.

## The two vehicles, and why there are two

A hard PETSc limit shapes the study. `PCFIELDSPLIT` refuses more than **128
fields**, and the DtN mixed space carries `1 + 2(L+1)^2` fields (one scalar Real
multiplier per mode, both boundaries), which crosses 128 at **L = 7**. So the
coupled end-to-end fieldsplit solve runs only at `L <= 6`. The vector-valued Real
space that would collapse the field count is unsupported at the form level in this
Firedrake build. The study therefore uses two vehicles:

* **`gravity_cubed_sphere.py`** — the coupled solve, at `L in {2, 5, 6}` across
  levels 4-7. This is the weak-scaling AMG axis: the number to watch is
  `fieldsplit_0` iterations per invocation staying flat across levels. Locally at
  level 4 it is ~7.0 and L-independent.
* **`capacitance_gravity.py`** — the wall-free L-cost diagnostic. It builds the
  `n x n` DtN Schur complement `S` through a standalone potential-block solver
  (no `PCFIELDSPLIT`), so it reaches all `L = 2..30`. The number to watch is the
  offline GMRES-on-`S` count and `cond(S)` versus L.

The two meet at the cross-check: the offline GMRES count on `S` (flat rhs) at
level 4 / L = 5 reproduces the coupled solve's measured `fieldsplit_1` count (13),
which validates the hand-assembled `S`.

## Running

```bash
# one coupled case (L <= 6)
python3 gravity_cubed_sphere.py 4 --lmax 5 --out-dir results/

# one capacitance case (any L)
python3 capacitance_gravity.py 4 --lmax 20 --out-dir results/

# the direct-vs-iterative correctness anchor
python3 gravity_cubed_sphere.py 4 --lmax 5 --anchor --out-dir results/

# generate the Gadi PBS grid (dry run; --submit to qsub)
python3 submit_gravity_scaling.py --results-dir results/

# read the tables and enforce the excitation floor
python3 summarize_gravity_scaling.py --results-dir results/
```

Run against a specific worktree with `PYTHONPATH=<worktree>` so the editable
G-ADOPT install does not shadow it. On Gadi the submit scripts pin `PYTHONPATH`,
the `fp50` Firedrake module, project `xd2`, and the kernel cache automatically.

## Files

| file | role |
| ---- | ---- |
| `gravity_cubed_sphere.py` | coupled-solve model + correctness anchor (layer one) |
| `capacitance_gravity.py`  | wall-free capacitance L-cost diagnostic (layer one) |
| `submit_gravity_scaling.py` | Gadi PBS generator (dry run / `--submit`) |
| `summarize_gravity_scaling.py` | reads sidecars, prints tables, enforces the floor |
| `scaling.py`, `meta.py`, `run.template`, `test_scaling.py` | layer-two doit/longtest integration |
| `expected.csv`, `expected_capacitance.csv` | regression references, populated from the layer-one run |

The JSON sidecars each run writes are the stable interface the summarizer and
tests read; the raw PETSc logs remain the source of truth.

## Layer-two staging (before wiring into the weekly longtest)

`expected.csv` and `expected_capacitance.csv` are seeded with level-4 (and a
level-5 drift row) numbers measured locally. `test_scaling.py` parametrizes over
every level, so until the layer-one Gadi run repopulates the CSVs and produces the
level 5-7 sidecars, most cases have no reference row and would fail. Do **not**
add this suite to the weekly `longtest.yml` rotation until the Gadi layer-one run
has repopulated both CSVs. One caveat to check when wiring it in: the level-4/L=30
capacitance case can need up to two hours, so confirm `gadopt_hpc_helper` grants
that doit step enough walltime (the bare layer-one generator sets it explicitly;
the doit layer inherits the helper default).
