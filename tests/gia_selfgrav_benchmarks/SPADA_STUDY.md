# Spada paper study: run matrix and figure requirements

This is the specification for the planned study, including the profile-rate,
LOD, off-axis controls and surface HDF5 additions after commit `deebb1ea`.
`spada_study.py` prepares a manifest and PBS files for all 27 production runs
and four smoke checks, and submits them in explicit stages. Preparation
never submits jobs. `analyse_spada_study.py` audits transferred summaries,
profiles and logs and generates sensitivity/reference figures. Surface-map
assembly and the full checkpoint audit remain separate work.

## Run matrix

Four physical scenarios use the M3-L70-V01 Earth, a held step load, and a
20 kyr end time:

| Scenario ID | Driver options | Load centre | Rotation |
|---|---|---|---|
| `cap-off` | `--case cap --load_shape cap` | North pole | Off |
| `disc-off` | `--case cap --load_shape disc` | North pole | Off |
| `cap-on` | `--case polar-motion --load_shape cap` | Colatitude 25°, longitude 75° | On |
| `disc-on` | `--case polar-motion --load_shape disc` | Colatitude 25°, longitude 75° | On |

Run the baseline and each of the following five variants for every
scenario. Change only the indicated parameter from baseline.

| Variant ID | Baseline | Variant | Current control |
|---|---|---|---|
| `baseline` | All baseline settings | None | Existing defaults; specify explicitly in run records |
| `dt2` | Steps of 10/50/100/500 yr | 20/100/200/1000 yr over the same intervals | `--timestep_scale 2` |
| `mesh500` | Bulk size 250 km | Bulk size 500 km | `--mesh_h_km 500` |
| `litho1` | Two 35 km lithosphere layers | One 70 km layer | `--litho_layers 1` |
| `bulk100` | K/μ = 1000 | K/μ = 100 | `--bulk_shear_ratio 100` |
| `dtn2` | DtN degree 5 | DtN degree 2 on both spheres | `--dtn_degree 2` |

Add just one `cap-on/loose` run with `--outer_rtol 1e-5 --block0_rtol 1e-3`.
Baseline tolerances are 1e-6/1e-4. The cached Schur preconditioner is the
same in both runs.

The main matrix has **4 × (1 baseline + 5 variants) + 1 tolerance check = 25 runs**.
Add two off-axis nonrotating baseline controls, one per shape, using
`--case cap --load_centre 25 75`. This gives **27 runs** including matched
rotation-on/off spatial comparisons. No extra sensitivity sweep is needed
for these two controls. All 27 runs now have explicit CLI controls, with
effective mesh and timestep settings recorded and checked on restart.
Smoke checks are additional execution checks and do not count as study runs.

All runs use the spatial load with degrees 0 and 1 removed, diagnostic
degree 10, CG3 displacement, DG2 internal variables, CG2 potential and
quadratic geometry. Keep material interfaces and load history fixed. The
bulk mesh-size variation changes both horizontal and radial cell sizes
outside the explicitly constrained lithosphere; it is not a purely
horizontal refinement study. Lithosphere subdivision leaves its 70 km
physical thickness unchanged.

The timestep intervals are 0–0.1, 0.1–1, 1–10 and 10–20 kyr. Preserve the
common output epochs at 0, 0.1, 1, 2, 5, 10 and 20 kyr when changing steps.
With this schedule the baseline has 138 marched steps and `dt2` has 69,
in addition to the elastic diagnostic solve.

## Files required from every run

Use one directory per scenario and variant, for example
`outputs/spada-study/cap-on/baseline/`. The nonrotating disc still writes
`summary_cap.json`; its directory distinguishes it from a cap.

Enable `--write_profiles` for **all 27 runs**. Retain:

- `summary_<case>.json`: settings, mesh provenance and epoch diagnostics.
- `profiles_<case>_<time>kyr.npz` at every common epoch: direct U, V, W,
  along/across displacement and gravitational geoid, adjacent-step rates
  in m/yr, interval metadata, coordinates, bases and found mask.
- `params_<case>.log`: every-step timing, iteration counts and, for rotating
  runs, both polar-motion components, m3 and signed LOD in ms.
- PBS stdout and stderr, job ID, exact arguments, source revision and
  resolved Firedrake module version. The launcher already prints revision
  and module information; preserve the job logs with the run.
- Mesh file and checkpoint for provenance/restart. A restarted run needs
  the original rank count and matching physical/numerical settings.

Use identical profile sampling in a scenario's baseline and variants.
The defaults are three pole-to-pole meridians for pole-centred nonrotating
runs and two perpendicular great circles through the load centre for
rotating runs and off-axis nonrotating controls.
Both use 0.1° spacing; rotating profiles cover signed distance ±180°.
The archive then supports both global and near-load panels without a rerun.

Enable `--write_surface_output` for the four baselines and both off-axis
nonrotating controls to support maps and matched rotation-on/off differences.
This writes one collective `spada_<case>_surface.h5` on the actual curved surface mesh,
with displacement, gravitational/rotational/total geoid and adjacent-step
rates in m and m/yr. Geometry is in metres and times in kyr. The t=0
rates are undefined (NaN with `rate_valid=0`). NPZ profiles support line
panels but cannot reconstruct arbitrary spatial maps. Archive the HDF5 file,
then use `convert_spada_surface.py` with fewer MPI ranks for selected fields
and frames. This adapter calls the external `dsroberts/firedrake_h5_to_vtk`
script unchanged; install/select it as documented in README.md. It preserves
physical PVD times and the saved rates. Record the external converter revision
with the plotting commands. Preserve all
converted VTK pieces with their PVD indices if distributing the VTK results. Add
`--write_output` only if interior volume-field visualisations are needed.
Direct surface VTK is available with `--surface_output_format vtk` for debugging.

## Figure coverage

| Figure | Available solver output | Independent reference | Readiness |
|---|---|---|---|
| Nonrotating cap/disc U, V, N profiles and residuals at selected times | Direct profiles plus projected coefficients | Loading Love numbers and relaxation spectra, degrees 2–256 | Existing profile plotter supports both shapes |
| Degree-by-degree nonrotating response | Summary coefficients through diagnostic degree 10 | Matching TABOO degrees | Data available; paper panel assembly still needed |
| Polar-motion components, amplitude and phase versus time | Every-step log plus elastic/epoch summary values | Detailed cap time series; disc scales by its degree-2 excitation factor | `analyse_spada_study.py` plots epoch amplitude/error/phase; continuous component panel assembly remains |
| Polar-motion rate | Finite differences of every-step components | TABOO cap rate series and cap/disc tabulated rate magnitudes | Analysis/plotting needed; label numerical derivative and evaluate timestep sensitivity |
| Rotating U, along, across, N spatial profiles | Direct profiles on two load-centred great circles | No rotating spatial reference in the supplied archive | Existing direct profile plotter works; these are field illustrations and sensitivity comparisons |
| Numerical sensitivity errors and cost | Matched baseline/variant profiles, summaries and step logs | TABOO for nonrotating fields and polar motion; baseline differences for rotating fields | `analyse_spada_study.py` audits and compares transferred runs, including the 125 km cap checks |
| Spatial deformation/geoid rates | Adjacent-step differences at every saved profile epoch, in m/yr | Instantaneous analytic TABOO derivatives, rotated into the saved profile basis | `plot_spada_profiles.py --rates` works for both cases; t=0 is omitted |
| Surface maps and rotation-on/off differences | Surface HDF5 displacement, gravitational/rotational/total geoid and rates on simulation boundary facets; offline VTK conversion | No independent rotating spatial reference in the supplied archive | Output/conversion implemented; map panel assembly still needed |
| Length of day | m3 and signed `lod_ms` at every step and output epoch | `lod_eq33_ref_ms`, reconstructed independently from TABOO loading k2(t) | Output implemented; signed comparison requires resolving the Eq.33/inertia sign convention |

The U/V/N reference uses finite degrees through 256, even for a spatial
load. State that cutoff in captions, especially for the sharp disc edge.
The current projected diagnostics stop at degree 10; direct profiles are
needed to assess the higher spatial detail resolved by the mesh.

Polar-motion reference curves exclude Chandler wobble. The existing C−A
convention produces an approximately 2.4% excitation difference from the
TABOO reference. Show raw errors and explain the convention rather than
quietly rescaling the solver. Historical pass bounds/error corrections were
calibrated at K/μ = 100 with a spectral load; do not treat them as calibrated
acceptance thresholds for this new baseline.

The four standard scenarios change load orientation as well as rotation.
Use the two additional off-axis controls for a controlled rotation-on/off
spatial difference. Their location, mesh settings and profile bases match
the rotating baselines. Load location is independent of rotation through
`--load_centre COLAT LON` and is checked on restart. Nonrotating harmonic
diagnostics are always measured about the load axis.

Rates use the previous solved timestep, not the previous saved epoch.
Intervals and validity are recorded in the NPZ files. t=0 rates are NaN
because of the instantaneous jump; the first marched interval subtracts
the approximate elastic diagnostic, and restarts subtract the restored
state. The individual profile plotter overlays instantaneous TABOO rates;
`analyse_spada_study.py` instead averages the analytic reference over each
saved interval before scoring rate errors. State which comparison is used
and report timestep sensitivity. N and its rate are gravitational contributions
only (`psi/g0`), excluding the centrifugal potential.

LOD uses `lod_ms = -86400e3*m3`, positive for a longer day. The printed
Spada equation 33 reference has an opposite geometrical sign to the
degree-0-free signed inertia integral. Both are retained under distinct
names; no LOD relative-error score or sign correction is applied. The
reference reconstruction matches the local converted GS archive values.
This convention issue must be resolved before claiming a signed LOD
benchmark match.

Optional cumulative polar-motion departure or time-integrated reference
error can be computed from the step logs without another simulation.
Include the elastic t=0 value from the summary and report quadrature and
units. These are response diagnostics, not dissipated energy. An energy
budget would need additional solver diagnostics.

## Reproducible execution and analysis

Run these commands from `tests/gia_selfgrav_benchmarks` in the **Gadi checkout**:

```bash
module use /g/data/fp50/modules
module load firedrake/2026.10.0
python3 spada_study.py prepare
python3 spada_study.py show --stage production
python3 estimate_spada_cost.py
```

This creates `outputs/spada-study-plan/manifest.json`, 31 PBS files under
`jobs/`, and cost reports/CSVs beside the manifest. Every run has an ID,
exact arguments, resources, output path, expected epochs and sampling
options. The manifest includes source hashes and the `reference.npz`
checksum, including relevant uncommitted source contents. Both submission
and execution check the source hashes, so changing code while jobs are
queued requires a new plan rather than mixing source versions silently.
Do not modify the checkout during an active run; the startup hash check
cannot protect against edits made after Python starts.

Defaults are project `vo05`, queue `normalsr`, storage
`scratch/xd2+scratch/vo05+gdata/vo05+gdata/fp50`, and the stable release module
`firedrake/2026.10.0` on Gadi. Each job checks and prints the
Firedrake package version before executing; a version other than the plan's
`firedrake_version` (default `2026.10.0`) stops the job before a solve.
Use `prepare --module ... --firedrake-version ...` for another tested build, and pass
`--project`, `--queue`, `--storage`, `--output-root` or `--plan-dir` as needed.
The four smoke jobs still check the target Gadi installation and MPI build.
The pinned Docker environment, compatibility fixes and local verification
commands are recorded in [SPADA_COMPATIBILITY.md](SPADA_COMPATIBILITY.md).
Preparation refuses a nonempty plan directory. Use a fresh plan/output root
for another study. The existing `meta.py` is independent of this matrix.

After updating code or changing the software version, preserve any failed
plan and logs and prepare a new plan and output root. Previously submitted
IDs remain in the old journal even when the job failed. For example:

```bash
python3 spada_study.py prepare --plan-dir outputs/spada-study-plan-2026.10 \
    --output-root outputs/spada-study-2026.10
python3 spada_study.py submit --plan-dir outputs/spada-study-plan-2026.10 --stage smoke
```

Use that same `--plan-dir` for later submission stages.

Submit in stages, after inspecting each preceding stage's logs and outputs:

```bash
python3 spada_study.py submit --stage smoke
# Once all four smoke checks finish successfully:
python3 spada_study.py submit --stage baselines
# Once the six baselines/controls finish, recalibrate cost and inspect fields:
python3 spada_study.py submit --stage variants
```

Only `submit` invokes `qsub`. It records PBS IDs and commands in
`submissions.jsonl`, skips already submitted IDs, rejects nonempty output
directories and serializes concurrent submissions. The job itself claims
its output directory to prevent accidental duplicate execution. These are
fresh-run launchers; continue a failed run explicitly through the existing
`run_benchmark.pbs` restart interface, matching the manifest's arguments
and rank count. Stage completion is checked by the operator, not inferred
from submission order. `submit --stage production` submits all 27 together
if that is deliberately desired after the smoke checks.

All production jobs use the historical 15-node allocation: 1,560 cores,
7,500 GB, 100 GB jobfs, with an initial three-hour walltime. This retains
the same hardware allocation for cost/sensitivity comparisons, including
coarser meshes. Smoke checks use 104 cores, 500 GB, 100 GB jobfs, one hour.
Surface HDF5 is enabled for the six baselines/controls and four smoke jobs;
all 31 jobs write profiles and rates. No job requests direct volume VTK.

The local completed degree-10 runs imply 2 SU/core-hour and a median
66.4 minutes / 3,452 SU for the old K/μ=100 baseline. The generated estimate
is about **95,000 SU**, with a broad **75,000–170,000 SU** planning range
and about **120,000 SU** including a 25% margin on the central estimate.
This includes smoke checks, but excludes retries and offline conversion.
K/μ=1000, disc runs and new output overhead are not measured in those logs;
mesh/lithosphere savings are explicit assumptions in the report. Requested
walltimes are limits rather than measured costs: all jobs hitting their
limits would be about 253,552 SU. Review the six measured baselines before
committing the remaining sweep's allocation.

`estimate_spada_cost.py` reads actual PBS footer accounting, excludes failed
and high-diagnostic-degree runs from calibration, and retains per-log
evidence and per-run estimates as CSV/JSON. Its defaults read the logs in
this directory; use `--logs-dir` for another archive. The report records
all multipliers. It does not assume a full 8× cost reduction when doubling
mesh size, or a full 2× reduction when halving the step count.

The workflow should support these stages:

1. Generate and inspect all submission commands without launching jobs.
   Record the source revision, reference-file checksum and actual resolved
   software versions. Use a consistent resolved Firedrake build across runs.
2. Smoke-check both rotation cases and both load shapes on the target
   environment, including profile export, before full production runs.
3. Submit each manifest entry into its own directory and record its PBS ID.
   Keep explicit baseline settings in the manifest even where they equal
   defaults. Prevent accidental overwrites of completed runs.
4. Audit all 27 runs before analysis: successful job exit, non-smoke status,
   matching settings, all seven epochs, all required profiles, finite
   values (except intentionally undefined t=0 rates), found masks, mesh checks and solver convergence. Compare mesh
   MD5s for variants meant to share a mesh. Flag changed software versions.
5. Generate PDF/PNG figures and a CSV of raw errors/costs using exactly the
   audited manifest, with no automatic discovery of unrelated archived
   runs. Retain units, normalization, reference cutoffs and the plotting
   command/software versions with each figure set.

Existing commands for individual baseline panels, from this directory:

```bash
python3 plot_spada_profiles.py outputs/spada-study/cap-off/baseline/summary_cap.json \
    --profiles-dir outputs/spada-study/cap-off/baseline --max-colatitude 25 \
    --output outputs/spada-study/cap-off/baseline/profiles_near.pdf

python3 plot_spada_profiles.py outputs/spada-study/disc-on/baseline/summary_polar-motion.json \
    --max-distance 25 \
    --output outputs/spada-study/disc-on/baseline/profiles_near.pdf
```

For a six-panel nonrotating baseline figure, with cap/disc columns and radial
displacement, tangential displacement and geoid rows:

```bash
python3 plot_spada_baselines.py outputs/spada-study
```

This overlays direct baseline curves and the Spada/TABOO degree-2–256
reference at 0 kyr (elastic), 1 kyr, 5 kyr and 20 kyr. It selects the saved
0° longitude meridian over 0–25° from the north-pole load centre; tangential
displacement is positive away from that centre. Both columns use the same
vertical scale within each row. The 10° load edge is marked. No meridian
averaging or degree-10 projection is applied. Use `--longitude`, `--times-kyr`,
`--max-colatitude` and `--output` to change these choices. PDF/PNG and an
input hash/settings JSON are saved as
`outputs/spada-study-analysis/spada_nonrotating_baselines.*`.

Figure preferences agreed on 26 October 2026, to preserve in future revisions:

- Use a colour-blind-friendly scientific palette. The current Okabe–Ito
  choices are blue `#0072B2` for elastic, orange `#E69F00` for 1 kyr,
  bluish green `#009E73` for 5 kyr and reddish purple `#CC79A7` for 20 kyr.
  Use the same time colour for numerical and reference values, at full opacity.
- Draw G-ADOPT as solid lines and the reference as crosses without connecting
  lines. Label the reference **TABOO** in the legend.
- Put the combined time/method legend in the bottom-right corner of the
  top-left panel. Use a rectangular white background and a thin **black**
  border to match the plot outlines.
- Put all panel labels `(a)`–`(f)` in the same top-right position, slightly
  inset from both edges: axes coordinates `(0.96, 0.96)`, right/top aligned.
  Preserve extra headroom in the shared radial-displacement y-axis so the
  top-row labels remain clear of the curves. The current upper-limit padding
  adds 7% of the original axis range.
- Keep explanatory text out of the figure footer. Describe rotation status,
  mesh resolution, profile location, reference cutoff and the load-edge line
  in the paper caption. Retain the axis labels, units, column titles and legend.
- Export both vector PDF for the paper and PNG for preview; retain the JSON
  recording times, colours, input hashes and reference cutoff.

To generate a separate six-panel figure for each completed nonrotating
comparison, keeping the same layout, time colours and TABOO crosses:

```bash
python3 plot_spada_baselines.py outputs/spada-study --all-variants
```

Each figure overlays the **solid G-ADOPT baseline**, one **dashed changed
run**, and TABOO crosses at elastic, 1, 5 and 20 kyr. The legend names the
changed setting. PDF/PNG and input-hash/settings JSON files are saved under
`outputs/spada-study-analysis/nonrotating_variants`, with an `index.md`
linking the figures and recording cap/disc availability. Use `--output-dir`
to choose another batch directory. To regenerate just one comparison:

```bash
python3 plot_spada_baselines.py outputs/spada-study --variant mesh500
```

The single-variant command writes
`outputs/spada-study-analysis/spada_nonrotating_mesh500.*`; `--output`
overrides that stem. Running without either comparison option still
generates the original baseline figure.

| Variant | Changed setting | Completed nonrotating runs |
|---|---|---|
| `dt2` | Doubled timestep | Cap and disc |
| `mesh500` | 500 km horizontal mesh size | Cap and disc |
| `litho1` | One lithosphere layer | Cap and disc |
| `bulk100` | K/μ = 100 | Cap and disc |
| `dtn2` | DtN degree 2 | Cap and disc |
| `mesh125` | 125 km horizontal mesh size | Cap only |
| `offaxis` | Load centre at colatitude 25°, longitude 75°; rotation off | Cap and disc |

The 125 km figure retains baseline/TABOO curves in the disc column and
labels that column as having no comparison result. The solver-tolerance
check is a rotating cap run, so it is outside this nonrotating figure set.
The one-layer runs also regenerate the bulk mesh, as described below.

The off-axis figure is a load-placement control. It uses the positive
0–25° branch of the saved bearing-0° great circle through the off-axis load
centre; `--bearing` selects another saved bearing. Tangential displacement
is positive away from each run's own load centre. The spherical nonrotating
reference is invariant under rotating the load and profile together.

Before drawing a dashed curve, the plotter checks that the saved settings
match the advertised variant and the baseline's remaining controls. It
requires matching sample distances and depths, plus matching spatial bases
for comparisons at the same load location. Profiles are neither interpolated
nor averaged. Missing comparisons are labelled explicitly; other invalid
inputs stop generation. Each figure's JSON records the files used, script
hashes, software versions, reference cutoff and comparison availability.

The existing `plot_spada_load_convergence.py` compares load cutoffs. For this
parameter study use the separate analysis command, which requires NumPy and
Matplotlib but does not import Firedrake:

```bash
python3 analyse_spada_study.py outputs/spada-study
```

It writes `outputs/spada-study-analysis/analysis.md`, PDF/PNG figures,
CSV metrics and `provenance.json`. The provenance records input/reference
hashes, analysis/helper script hashes and analysis software versions; the
run table retains source/module strings and mesh hashes/seeds. The script
checks saved controls against the planned matrix, all seven epochs,
profile validity, mesh-quality flags, timestep logs and outer convergence.
Inner iteration caps are reported separately. Surface HDF5 files and full
checkpoints are outside this audit.

Sensitivity metrics compare matching physical scenarios on identical saved
profile points and bases. They include absolute/RMS errors and peak-normalized
differences within 25° of the load centre, plus global and load-edge-excluded
metrics. These are line-profile norms, not area-weighted surface norms.
TABOO spatial reference curves use degrees 2–256; rate errors use matched
interval averages. Polar-motion errors retain the raw C−A convention
difference, and signed Eq.33 LOD errors are not scored. One-at-a-time variants
do not establish an error bound for combined settings. Lithosphere changes
also regenerate the bulk mesh: the transferred one-layer runs used seed 10
versus baseline seed 9, so their effects include remeshing.

The transferred October 2026 dataset contains 28 completed production runs:
26 of the planned 27 plus `cap-off/mesh125` and `cap-on/mesh125`. All seven
profiles are present for each completed run. `disc-on/dtn2` failed with an
MPI buffer error and has no completed summary. No 125 km disc run is
available. Completed production runs cost 214,238.28 SU; including the failed
DtN attempt gives 214,572.81 SU, excluding smoke checks.

The generated report separates large-scale degree-2–10 diagnostics from
direct-field spatial accuracy. In particular, the 125 km cap-off geoid rate
at 10 kyr has a dipole-shaped discrepancy: `low_mode_diagnostics.csv`
records a fit on the saved profiles, without removing that component from
the scored errors. A controlled centre-of-mass-frame check is needed before
claiming convergence of those rates. The scientific interpretation of this
transferred dataset is in [SPADA_STUDY_ANALYSIS.md](SPADA_STUDY_ANALYSIS.md).
