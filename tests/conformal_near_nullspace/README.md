# Conformal near-nullspace benchmark

`benchmark.py` compares candidate velocity spaces for the 3-D TALA
fieldsplit/GAMG solve on a curved, radially extruded cubed sphere. It mirrors
the GPlates model's nonzero strong top velocity and weak bottom normal-velocity
condition. The frozen viscosity has a selectable radial contrast.

Run each arm in a fresh process or PBS job:

```bash
python tests/conformal_near_nullspace/benchmark.py \
  --modes rigid-raw --refinement-level 2 --layers 4 --warm-repeats 5
python tests/conformal_near_nullspace/benchmark.py \
  --modes conformal-raw --refinement-level 2 --layers 4 --warm-repeats 5
python tests/conformal_near_nullspace/benchmark.py \
  --modes rigid-constrained --refinement-level 2 --layers 4 --warm-repeats 5
python tests/conformal_near_nullspace/benchmark.py \
  --modes conformal-constrained --refinement-level 2 --layers 4 --warm-repeats 5
```

Use `--modes none` as an overall control and `--modes rotations` to test the
only raw conformal candidates intrinsically tangent to both spherical
boundaries. The four primary arms isolate adding dilation/special-conformal
modes from strong-boundary restriction. Do not compare only `rigid-raw` with
`conformal-constrained`, because that changes both factors.

The JSON output contains maximum-rank barrier-to-barrier timings, repeated
warm samples, total velocity and pressure KSP calls/iterations, convergence
failures, assembled-matrix candidate count, residuals, and candidate strain
and boundary energies. Append PETSc logging options for GAMG setup, hierarchy,
operator-complexity and memory analysis, for example:

```text
-log_view :petsc_case.json:ascii_json
-Stokes_fieldsplit_0_assembled_pc_view
-info :pc
```

Laptop measurements are mechanism and regression evidence only. A production
decision requires repeated Gadi runs from the same checkpoint and allocation,
including a frozen fully plastic operator and an end-to-end Picard/Newton
slice. Compare total Stokes time, all fieldsplit_0 and fieldsplit_1 work,
GAMG setup/coarse-grid cost and memory, and final nonlinear residuals. Ten
candidates increase interpolation width and can reduce iterations while still
increasing time or memory.

The default refinement level is 2. On the level-1 shell the six-mode arms run,
but the ten-mode GAMG setup is under-resolved locally: some aggregates cannot
represent ten scalar columns (at least four three-component nodes are needed),
and the PETSc process can terminate during setup. This is a useful guardrail,
not evidence against the production mesh. Keep the level-2 minimum for the
ten-mode local comparison and inspect PETSc's aggregate/coarse-grid report on
every target MPI layout.

## Testing the GPlates case

Use the same specification for every Stokes solver that acts on the same
velocity operator, including both the frozen-viscosity Picard solver and the
Newton solver:

```python
velocity_near_nullspace = ConformalKillingNearNullspace(
    rotational=True,
    translations=(0, 1, 2),
    dilation=True,
    special_conformal=True,
    constrain_strong_bcs=False,
)

solver = StokesSolver(
    # Existing arguments remain unchanged.
    near_nullspace=velocity_near_nullspace,
)
```

`constrain_strong_bcs=False` is the recommended first test. It supplies the
smooth, unmodified conformal fields to GAMG even though the GPlates velocity
is prescribed strongly at the top. Use `True` as a separate experimental arm;
it zeros the top-boundary correction degrees of freedom before
orthonormalising the modes, which introduces candidate strain near that
boundary.

For a controlled comparison from one checkpoint, run at least these fresh-job
arms with otherwise identical PETSc options and process placement:

1. the current near-nullspace;
2. rotations only;
3. six rigid modes, raw and constrained;
4. all ten conformal modes, raw and constrained.

Compare maximum-rank Stokes time, total `fieldsplit_0` and `fieldsplit_1`
iterations and calls, GAMG hierarchy/operator complexity and memory, and the
final nonlinear residual. Do not select an arm from `fieldsplit_0` iterations
alone: broader interpolation can reduce iterations while increasing setup,
coarse-grid, communication, and memory costs.
