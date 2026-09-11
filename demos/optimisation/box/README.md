# Box optimisation benchmark

A twin experiment on the isoviscous convection box of the adjoint tutorial, built to
compare optimisers on one problem with one cost measure. The unknown is the
temperature at the start of an inversion window. The observation is the temperature
at the end of the window from a reference run. The measure of a run is the error to
the true initial temperature against the number of tape sweeps.

## Files

| file | purpose |
|---|---|
| `forward.py` | Reference run. Writes `reference-state.h5` with the temperature at every step. |
| `problem.py` | Tapes the forward model once and builds the reduced functional. Owns the tape. |
| `optimisers.py` | ROL and TAO runners, and the recorder that counts sweeps. |
| `inverse.py` | Command-line driver. One optimiser per run, one JSON file per run. |
| `riesz_hessian.py` | Preconditioning matrix and preconditioner that put the control inner product into the TAO bounded Newton solve, under bound constraints. |
| `plot_campaign.py` | Draws the three panels of Fig. 5 of the paper from a directory of run JSON files. |
| `run_campaign.sh` | Runs every optimiser for both inner products, several at a time. |
| `calibrate_radius.sh` | Sweeps the initial trust-region radius per optimiser on a coarse mesh. |
| `summarise_calibration.py` | Names the best radius per optimiser from that sweep. |
| `meta.py` | Demo runner metadata. |

## Run

```bash
python forward.py --steps 80 --resolution 64            # once
# The full trajectory, no regularisation, the first guess of the paper:
python inverse.py --optimiser rol-lbfgs --window 80 --alpha-d 0 --alpha-s 0 \
    --initial-guess observed --gradient-rtol 1e-6
# The same with the regularisation inner product. Its weights are separate from the
# objective weights, so the metric still exists when the objective has none:
python inverse.py --optimiser tao-bntr --window 80 --alpha-d 0 --alpha-s 0 \
    --metric-alpha-d 1e-3 --metric-alpha-s 1e-3 --initial-guess observed \
    --initial-radius 1.0 --gradient-rtol 1e-6
```

A run that must finish in a minute uses `--steps 10 --resolution 16` and
`--window 5 --max-it 4`.

The output file `<optimiser>_<riesz>.json` holds one row per gradient evaluation and
one row per optimiser iteration. The file is rewritten after every iteration, so a run
that is stopped keeps its history.

Each row has these measures, and the sweep counts at that moment:

| field | meaning |
|---|---|
| `J` | the objective functional. Panel (a) of Fig. 5 of Ghelichkhan et al. (2024, GMD). |
| `final_misfit` | `int (T(t_F) - Tobs)^2 dx`, the paper's final misfit, panel (b). |
| `initial_misfit` | `int (Tic - Tic_true)^2 dx`, the paper's initial misfit, panel (c). |
| `control_error` | the same as `initial_misfit` as a relative norm, `sqrt(initial_misfit / norm_ic)`. |
| `relative_final_misfit` | `final_misfit` divided by `int Tobs^2 dx`. |
| `gradient_norm` | the gradient norm in the control inner product. |
| `cost` | forward replays plus adjoint sweeps. This is the paper's `#(FWD+GRAD)` axis. |
| `forward`, `adjoint`, `hessian` | the three sweep counts. A Hessian-vector product is not part of `cost`, because the paper never computes one. |

The record also holds `forward_times`, `gradient_times` and `hessian_times`, the wall
time of every single call, and `call_cost`, their means and the two ratios. The ratio
of a Hessian-vector product to a forward call is what turns a Hessian count into
work, and `plot_campaign.py --x work` uses each run's own measured ratio.

Compare runs at equal `cost`, and read `hessian` next to it, because for the Newton
variants that is where most of the work goes.

`plot_campaign.py --runs <directory>` draws the panels for every JSON below a
directory.

## Optimisers

| name | method | Hessian-vector products |
|---|---|---|
| `rol-lbfgs` | ROL Lin-More trust region, L-BFGS as the Hessian | none |
| `rol-newton` | ROL Lin-More, true Hessian, L-BFGS as CG preconditioner | yes |
| `rol-newton-noprec` | ROL Lin-More, true Hessian, no preconditioner | yes |
| `tao-blmvm` | TAO bounded L-BFGS, Riesz map as initial inverse Hessian | none |
| `tao-bqnktr` | TAO bounded quasi-Newton-Krylov: Krylov on the LMVM matrix | none |
| `tao-bqnkls` | the same with a line search | none |
| `tao-bntr` | TAO bounded Newton trust region, true Hessian, LMVM preconditioner | yes |
| `tao-bntr-noprec` | the same with no preconditioner | yes |
| `tao-bntr-riesz` | the same with the control inner product as the preconditioner | yes |
| `tao-bnls` | TAO bounded Newton line search, true Hessian, LMVM preconditioner | yes |
| `tao-bnls-riesz` | the same with the control inner product as the preconditioner | yes |

`rol-lbfgs` is the g-adopt default. The only difference to `rol-newton` is two
entries of the ROL parameter list, `Use as Hessian` and `Use as Preconditioner`.

The TAO `bqnk` types set the LMVM matrix as the operator of the Krylov solve, so they
never call the true Hessian. The true-Hessian types in TAO are `bnls`, `bntr` and
`bntl`.

## Regularisation, and the inner product when there is none

`--alpha-d` and `--alpha-s` weight the damping and the smoothing terms. Zero on both
leaves the temperature misfit alone. Each term is left off the tape when its weight is
zero.

With no regularisation the problem is ill-posed: the observation does not determine
the control in every direction. Expect the initial misfit to fall and then rise again
once the iterations start fitting what the data cannot see, so report the lowest value
and the work at which it occurs, not only the last value.

`--metric-alpha-d` and `--metric-alpha-s` weight the `regularisation` inner product.
They default to the objective weights, which makes the metric the Hessian of the
regularisation terms. Set them when the objective has no regularisation: the default
would then be a zero form and the Riesz solve would be singular, and the run stops
with a message that says so.

## Stopping

`--gradient-rtol` stops a run when its convergence measure falls to that fraction of
its starting value. A relative test is used because an absolute one is not comparable
across these methods: `blmvm` measures the gradient norm in the control inner product,
while the TAO bounded Newton types report the Fischer-Burmeister residual of the
bound-constrained conditions in the plain Euclidean norm (`bnk.c:69-72`), and the two
differ by a mesh-dependent factor near 150 on a 150-cell mesh. ROL's own gradient
tolerance is absolute and the starting norm is not known before the run, so the test
is applied in the status test instead.

## Initial trust-region radius

One radius for every method measures the radius and not the method. With 1e-2, which
is what g-adopt uses for ROL, the TAO bounded Newton types spend about six iterations
growing the radius before they can take a useful step, because BNK grows it by about a
factor of two per accepted step and the truncated conjugate-gradient solve stops at
the boundary after one step until then. `calibrate_radius.sh` sweeps the radius per
optimiser on a coarse mesh with the settings of the campaign, and
`summarise_calibration.py` names the best value for each.

## Inner product

The `-riesz` variants are the only TAO Newton runs where `--riesz` changes anything.
pyadjoint sets the initial LMVM Hessian from the Riesz map for `blmvm` only, and its
own `RieszMapPC` cannot be used here, because BNK restricts the Hessian to the
inactive set of the bounds and a Python matrix without `createSubMatrix` becomes an
opaque virtual matrix. `riesz_hessian.py` supplies both missing pieces. Without it,
`tao-bntr` and `tao-bnls` give the same numbers for both inner products.

`--riesz L2` uses the mass matrix on the control space, as g-adopt does. `--riesz
regularisation` uses the Hessian of the regularisation terms,
`2 c_d (u, v) + 2 c_s (grad u, grad v)`, with the weights of the objective. The
Control carries the map, so the gradient, the gradient norm, ROL's vector dot product,
TAO's initial LMVM matrix and the Newton preconditioner all use the same operator.

## Taping

`problem.build_problem` clears the tape, annotates one forward run and pauses
annotation before it returns. Every optimiser runs under `stop_annotating`. Callbacks
run inside pyadjoint calls that carry `no_annotations`. No other code turns annotation
on.

## Environment

This demo needs the PR #4638 firedrake environment for Hessian-vector products on the
nonlinear cases; the box case is linear and runs in either.

```bash
source ~/Workplace/firedrake-pr4638/activate
export PYTHONPATH=<g-adopt worktree>:<g-adopt worktree>/demos/optimisation/box
```
