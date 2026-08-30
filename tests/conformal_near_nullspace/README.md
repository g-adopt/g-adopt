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

The warm samples reuse the already constructed basis and GAMG hierarchy. They
measure repeated solve/application cost, not candidate construction, global
orthonormalisation, or hierarchy rebuilds. The cold value includes lazy GAMG
setup but is also affected by JIT compilation. Use the PETSc event log and a
fresh process for each arm when comparing total setup and nonlinear-update
cost.

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

## Cartesian TALA and ALA comparison

`benchmark_cartesian_compressible.py` is the direct three-dimensional
Cartesian analogue of the square TALA and ALA demos. It retains their
reference density, reference temperature, thermal perturbation, Rayleigh
number, dissipation number, free-slip walls, and Taylor-Hood discretisation.
Only the temperature is frozen and the normally direct Stokes solve is
replaced by the same full-Schur, assembled-velocity GAMG configuration in
every arm.

Run each arm in a fresh process:

```bash
python tests/conformal_near_nullspace/benchmark_cartesian_compressible.py \
  --approximation tala --modes none --n 8 --viscosity-contrast 1e4
python tests/conformal_near_nullspace/benchmark_cartesian_compressible.py \
  --approximation tala --modes rigid-raw --n 8 --viscosity-contrast 1e4
python tests/conformal_near_nullspace/benchmark_cartesian_compressible.py \
  --approximation tala --modes conformal-raw --n 8 \
  --viscosity-contrast 1e4
```

Use `--approximation ala` for the matched ALA cases. As in the spherical
benchmark, `rotations`, `rigid-constrained`, and `conformal-constrained` are
also available. The output records all nested velocity and pressure solves,
candidate energy quotients, convergence failures, maximum-rank timings, and
the assembled residual components.

The first single-rank Apple-silicon measurements below used PETSc 3.25.0,
Firedrake 2026.4.1, an `8 x 8 x 8` hexahedral mesh, a radial viscosity contrast
of `1e4`, and three warm repeats. They demonstrate an algebraic effect, not a
production speedup:

| Approximation | Candidates | Velocity iterations | Warm seconds |
| --- | --- | ---: | ---: |
| TALA | none | 96 | 1.467 |
| TALA | six rigid | 92 | 1.429 |
| TALA | ten conformal | 80 | 1.477 |
| ALA | none | 95 | 1.451 |
| ALA | six rigid | 91 | 1.497 |
| ALA | ten conformal | 79 | 1.541 |

The ten modes reduce nested velocity iterations by about 17 percent relative
to no candidates, but their wider GAMG interpolation offsets that saving on
one rank. On a `12 x 12 x 12` TALA mesh the work fell from 119 to 97
iterations, while median warm time increased from 6.20 to 6.93 seconds. At a
viscosity contrast of `1e6`, the tested ten-mode hierarchy failed during its
first velocity preconditioner application with `DIVERGED_NANORINF`; the zero-
and six-mode controls converged. This robustness limit must be understood
before enabling the ten modes by default.

ALA uses G-ADOPT's analytical right pressure gauge, which is not an exact
nullspace of the discrete operator. Its raw assembled momentum residual
therefore includes a gauge component even after Krylov convergence. Compare
the continuity residual and matched solver work, and do not interpret that
known gauge residual as a velocity near-nullspace failure.

### Why there is no two-dimensional conformal benchmark

The extra conformal modes are a specifically three-dimensional result for the
TALA/ALA viscous operator. G-ADOPT's two-dimensional compressible demos retain
the physical three-dimensional coefficient `2/3` in the stress tensor. With
that operator, two translations and one rotation are zero-energy modes, but
dilation and the quadratic conformal fields are not. A two-dimensional test
could therefore assess rigid-body candidates only; it could not validate or
demonstrate the additional modes implemented by
`ConformalKillingNearNullspace`. The class deliberately continues to reject
two-dimensional meshes.

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

On Gadi, first sync or clone this complete branch and verify the import before
submitting a full allocation:

```bash
python -c "import gadopt; print(gadopt.__file__)"
```

The printed path must be this branch, not the centrally installed G-ADOPT.
For the first checkpoint comparison, use fresh output directories, one
timestep, identical rank placement and PETSc options, and separate PETSc JSON
logs for each arm.
