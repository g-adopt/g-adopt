# Spada centre-of-mass study

`spada_frame_study.py` prepares a focused paired study, using the existing
Spada driver and Gadi launcher. Preparation writes a manifest and inspectable
PBS files; it submits nothing. This is separate from the original
`spada_study.py` parameter sweep.

## Core matrix

Each row runs at **250 and 125 km**, with the library's **centre-of-mass
constraint off and on**: **16 production runs**.

| Scenario | Load | Rotation | Load centre (colatitude, longitude) |
|---|---|---|---|
| `cap-off` | Cap | Off | 0°, 0° |
| `disc-off` | Disc | Off | 0°, 0° |
| `cap-on` | Cap | On | 25°, 75° |
| `disc-on` | Disc | On | 25°, 75° |

The original Spada centres are retained. Comparing the core rotation-on/off
cases changes both rotation and load placement; optional matched-location
controls below isolate rotation.

Every pair differs only by `--centre_of_mass`. The constraint includes the
mantle, fluid core and prescribed load. The rigid-rotation nullspace remains
declared in both frame choices. Neither profiles nor geoid fields are
recentered, and no fitted dipoles are removed.

Common settings: M3-L70-V01 Earth, held step load, spatial load with analytic
degrees 0 and 1 removed, two lithosphere subdivisions, K/μ = 1000, DtN degree
5, projected diagnostic degree 10, outer/block-0 tolerances 1e-6/1e-4, and
the original 10/50/100/500 yr ladder to 20 kyr (138 marched steps plus the
elastic solve). The nominal mesh size changes radial and horizontal sizes
outside the explicitly subdivided lithosphere.

There are also **eight smoke jobs**, one per shape/rotation/frame combination,
at 1000 km with one lithosphere subdivision. Timestep scale 10 gives one
100 yr marched step after the elastic solve. These check construction and
outputs, not benchmark accuracy.

## Optional checks

Enable any combination at preparation with `--checks`:

| Check | Runs | Purpose |
|---|---|---|
| `tolerance` | 2 | Nonrotating cap, 125 km, COM off/on; tighten outer/block-0 tolerances to 1e-8/1e-6 to test solver sensitivity |
| `timestep` | 2 | Same pair with half timesteps (276 marched steps) to assess rate sensitivity |
| `location` | 8 | Nonrotating cap/disc at 25°/75°, both meshes and COM choices; match rotating load positions and profile geometries |

All checks enabled gives **28 production + 8 smoke = 36 jobs**. Start with
the core matrix and use the checks to investigate a remaining discrepancy.
The tolerance pair is especially useful if COM-off differences remain large;
inner solver iteration caps and residuals must be examined alongside reported
outer convergence. The timestep pair tests backward-difference rates; its
rate intervals are shorter, so they must not be treated as identical to core
intervals when comparing to an instantaneous reference.

## Prepare and launch on Gadi

Run from `tests/gia_selfgrav_benchmarks` after syncing the code. The generator
uses only Python's standard library. Default plan and result roots are
`outputs/spada-frame-study-plan` and `outputs/spada-frame-study`.

```bash
python3 spada_frame_study.py prepare
python3 spada_frame_study.py show --stage core
python3 spada_frame_study.py show --stage pilot
```

To include checks, prepare a single plan with the desired groups, for example:

```bash
python3 spada_frame_study.py prepare --checks tolerance timestep location
```

Choose the preparation command before creating the plan: an existing nonempty
plan is refused. Use a new `--plan-dir` and `--output-root` for a separate
study. Inspect the generated manifest before explicit submission:

```bash
python3 spada_frame_study.py submit --stage smoke
# Inspect successful smoke logs and outputs before continuing.
python3 spada_frame_study.py submit --stage pilot
# Inspect the two 125 km nonrotating-cap production runs before continuing.
python3 spada_frame_study.py submit --stage core
# Optional, if included in this plan:
python3 spada_frame_study.py submit --stage tolerance
python3 spada_frame_study.py submit --stage timestep
python3 spada_frame_study.py submit --stage location
```

`pilot` selects the existing 125 km nonrotating-cap core pair, without
duplicating runs. The submission journal is shared: subsequent core or
production submission skips recorded jobs, including the pilot pair.
`--stage production` selects every nonsmoke run in the plan, including checks.
Source-content hashes are verified at submission and job startup. Source
changes require a freshly prepared plan. Failed jobs are still recorded as
submitted; inspect them before preparing a recovery job with matching
restart settings. Output directories with existing contents are refused.

Default module and exact required package version are `firedrake/2026.10.0`
and `2026.10.0`. Both are configurable at preparation, and the package version
is checked before execution. See [SPADA_COMPATIBILITY.md](SPADA_COMPATIBILITY.md).

Production jobs request 1560 cores, 7500 GB memory and 100 GB jobfs. Walltime
caps are 4 h at 250 km, 12 h at 125 km, and 18 h for the tolerance/timestep
checks. Smoke jobs request 104 cores, 500 GB memory and 1 h. These use
`normalsr`, project `vo05`, and the same storage declarations as the original
study; CLI options can change these declarations. Existing unconstrained
250 km jobs took about 2.8 h and the available 125 km jobs about 6.8–7.6 h.
These caps allow headroom but are not measured COM-on runtimes. Use the pilot
to reassess cost and walltime before launching the remainder; rotating
125 km disc performance is not yet measured. The core walltime allocation
cap is 399,360 SU at 2 SU/core-hour, plus 1,664 SU for smoke jobs; this is a
requested-walltime bound, not a runtime estimate.

## Saved diagnostics and comparisons

All production runs save direct profiles and collective surface HDF5, without
volume VTK. Common epochs are 0, 0.1, 1, 2, 5, 10 and 20 kyr. Extra epochs
4.9, 9.8, 9.9 and 10.5 kyr bracket the previously observed anomalies and the
10 kyr timestep transition. `--extra_epochs_kyr` rejects epochs that do not
coincide with already solved timesteps; adding these outputs leaves every
step time and step size unchanged. Rates are adjacent-step backward
differences, with their actual intervals saved in the profile/surface data.
The 10 kyr core rate uses 9.9–10 kyr (100 yr); the first 500 yr rate is at
10.5 kyr. Elastic t=0 rates remain undefined.

`--write_frame_diagnostics` records these assembled quantities:

- Summary/checkpoint metadata: background whole-body mass, prescribed-load
  mass and three first moments, in kg and kg m; load dipole divided by body
  mass, in m. These use the actual gravity source forms and quadrature, so
  analytic removal of load degrees 0/1 does not force them to vanish.
- Every-step parameter log: `com_x_m`, `com_y_m`, `com_z_m`, the total
  mantle + fluid-core + prescribed-load first moment divided by background
  body mass. This is the linearised centre-of-mass displacement in metres.
- Constrained logs additionally contain `com_lambda_x/y/z` in the library's
  nondimensional convention. Unconstrained logs have no multiplier columns.
- Every summary epoch contains `mass_dipole_kg_m`, `centre_of_mass_m` and
  `centre_of_mass_multipliers` (an empty list when unconstrained).

The background mass is assembled from the x-directed unit rigid-translation
response of the same moment form, including the fluid core. These assembled
moments describe the source of the potential equation; they are not a separate
measurement of load-traction interpolation on the mechanics mesh.
`--write_frame_diagnostics` and the extra epoch list are recorded restart
settings; they must match on restart. Old checkpoints default to diagnostics
off and no extra epochs.

For each COM pair, first check matching source hashes, mesh hashes, rank
counts, load moments, sampling coordinates, found masks and rate intervals.
Then compare raw displacement/geoid and rates at 5 and 10 kyr, the assembled
COM histories, multipliers, and solver iterations/residuals. Compare the
nonrotating degree-2–10 diagnostics and profiles against TABOO as well. A
constant displacement difference with a matching geoid dipole supports a
frame interpretation; differences in higher spatial degrees require further
checks. Do not subtract a fitted dipole from a published benchmark curve.
Use `location` pairs for rotation-only comparisons.

Keep the complete per-run directories when transferring results: summaries,
parameter/PBS logs, all profiles, `.msh`, surface `.h5`, and checkpoint `.h5`,
plus the plan manifest and submission journal. Checkpoints are written every
50 marched steps; the last checkpoint is not necessarily the terminal state.
Restarts require the original rank count and all matching physical/numerical
settings. The existing profile plotter can inspect each run independently:

```bash
python3 plot_spada_profiles.py --help
```

The original study analyzer expects its original directory/variant matrix;
it does not automatically discover this frame-study layout. This setup
prepares the experiments and diagnostics; no production result or conclusion
about COM-on accuracy is implied before those runs are completed.

## Validation of the setup

Development checks used Firedrake 2026.10.0: 94 configuration, manifest,
submission and surface-output tests passed (the surface round trips require
the external converter described in the README). These include exact analytic
body/sheet moment checks, restart defaults, pair equivalence, pilot/core
submission deduplication, and unchanged timestep schedules. All generated PBS
scripts passed `bash -n`.

A separate four-rank check restored the existing constrained cap smoke
checkpoint, assembled a background mass of 5.97013e24 kg, and measured a COM
residual norm of approximately 9.0e-10 m. Imposed rigid translations in each
coordinate produced the expected physical COM shifts, including the fluid-core
contribution. Diagnostics also constructed and sampled successfully for an
unconstrained rotating disc. These are implementation checks on a coarse
mesh, not production accuracy or runtime measurements.
