# Self-gravitating GIA benchmarks in 3-D

This directory holds two published benchmarks for glacial isostatic
adjustment (GIA) on a self-gravitating, spherically layered Maxwell Earth.
Both use the Earth model M3-L70-V01 and the same 3-D tetrahedral mesh and
solver.

- Spada, G., et al. (2011), A benchmark study for glacial isostatic
  adjustment codes, Geophys. J. Int. 185(1), 106-132. The reference is the
  normal-mode code TABOO.
- Martinec, Z., et al. (2018), A benchmark study of numerical
  implementations of the sea level equation in GIA modelling, Geophys. J.
  Int. 215(1), 389-414. The reference is VEGA.

Each case runs on Gadi as one job. Case B needs a second job at present
(see "Walltime and cost"). Case E uses successive full-history jobs to fit
the initial topography; see [CASE_E.md](CASE_E.md). `test_benchmarks.py` then compares the output
with the published criteria.

## The cases

| driver | case | load | what the test compares |
|---|---|---|---|
| `spada.py` | `cap` | 1500 m ice cap on the pole, switched on at t = 0 | U, N and V against TABOO at 0 to 20 kyr |
| `spada.py` | `polar-motion` | the same cap at colatitude 25, longitude 75, rotation on | polar motion against TABOO at 0 to 20 kyr |
| `martinec.py` | `B` | 1500 m cap, switched on at t = 0; fixed coastline, basin B1 | U, N, S and h_UF against VEGA at 10 kyr |
| `martinec.py` | `C` | 500 m cap, grows over 10 kyr; fixed coastline, basin B2 | as B, plus the grounded ice mass, at 15 kyr |
| `martinec.py` | `D` | as C, with a moving coastline and floating ice | as C, plus the floating ice mass and the ocean area |
| `martinec.py` | `E` | as D, with B2 prescribed at 15 kyr; fit the initial topography | terminal topography mismatch, ocean area and h_UF; Picard default or adjoint/ROL |

The docstring of each driver describes the physics, the scales, the
discretisation and the output. `selfgrav_common.py` holds the parts that
both drivers use: the Earth model, the mesh, the time steps and the solver
settings.

The planned Spada paper run matrix, required outputs, figure coverage and
remaining workflow work are recorded in [SPADA_STUDY.md](SPADA_STUDY.md).
The Firedrake 2026.10.0 container recipe and local compatibility checks are
recorded in [SPADA_COMPATIBILITY.md](SPADA_COMPATIBILITY.md).

## Files

| file | content |
|---|---|
| `spada.py`, `martinec.py` | the two drivers |
| `selfgrav_common.py` | the shared module |
| `martinec_e.py`, `CASE_E.md` | Case E topography fitting and staged-job instructions |
| `test_benchmarks.py` | the pass criteria; it reads the output files and needs `giamip` and no Firedrake |
| `meta.py`, `run.template` | the steps for `doit run_case_hpc` |
| `run_benchmark.pbs` | a PBS script to submit one case by hand |
| `reference.npz` | the TABOO spectra and polar-motion series of Spada et al. (2011) |

## Install giamip

The Martinec driver reads every physical constant of a case from the Python
package `giamip` (version 0.1.0 or later). The test reads the VEGA reference
data and the scoring functions from `giamip` too. The package is in the
`demos` extra of G-ADOPT, next to `assess`.

1. If your Python has no `giamip`, install it: `python3 -m pip install giamip`.
2. Optional: set `GIAMIP_DATA_ROOT` to a shared directory, for example on
   `/g/data`. The default is `~/.cache/giamip`.
3. Fill the data cache once, on a machine with network access:
   `python3 -m giamip.data --stage converted`.

On Gadi, until the weekly Firedrake module has `giamip`, install it into a
directory of your own. The module Python is a virtual environment, so
`pip install --user` does not work there. numpy, scipy, h5py and packaging
are in the module, so only `giamip` and `h5netcdf` go into the directory:

```bash
module use /g/data/fp50/modules
module load firedrake/2026.10.0
python3 -m pip install --no-deps --target /g/data/vo05/$USER/python-extra \
    giamip h5netcdf
export PYTHONPATH=/g/data/vo05/$USER/python-extra:$PYTHONPATH
python3 -m giamip.data --stage converted
```

Give the same directory to each Martinec job as `EXTRA_PYTHONPATH` (see
"With qsub"), and keep the `export` line in the login shell where you run
the test.

The run itself needs no data file and no network, because the case files are
inside the package. Only the test needs the reference files, and it
downloads them on first use if the cache does not have them. Gadi compute
nodes have no network, so run the test on a login node.

## Run the cases

The three procedures below all need a Firedrake installation with gmsh.
Each driver generates its mesh in the job.

On Gadi, the system git (`/bin/git`) has no `git-lfs`, and the repository
has LFS files. Load the git module before you clone, because it has
`git-lfs`:

```bash
module load git/2.39.2
git clone https://github.com/g-adopt/g-adopt.git
```

`run_benchmark.pbs` puts the repository root first on `PYTHONPATH`. Its
jobs then use the `gadopt` of the clone and not the one in the module.

### With doit

Use this procedure to run all five cases on Gadi when the Firedrake module
has `giamip`. `run.template` adds nothing to `PYTHONPATH`, so until then the
Martinec jobs of this procedure stop at `import giamip`. In that case, run
the Martinec cases with qsub.

1. Install `giamip` (see above).
2. From the repository root, run `doit 'run_case_hpc:tests/gia_selfgrav_benchmarks:*'`.
3. After the jobs end, run `doit check_hpc:tests/gia_selfgrav_benchmarks`.

A bare `doit run_case_hpc` submits the HPC cases of every directory in the
repository.

Each step in `meta.py` submits one job through `gadopt_hpcrun`, the
launcher of the package `gadopt_hpc_helper`. The weekly long-test CI of
G-ADOPT uses this procedure in its own environment, which has the launcher.
The Firedrake module on Gadi does not have it, so for a run by hand use
qsub. The `check_hpc` task runs `test_benchmarks.py` in this directory.

### With qsub

Use this procedure to run one case by hand. Submit from this directory.

```bash
qsub -v DRIVER=spada,CASE=cap run_benchmark.pbs
qsub -v DRIVER=spada,CASE=polar-motion run_benchmark.pbs
qsub -l ncpus=1872 -l mem=9000GB -l walltime=13:00:00 \
     -v DRIVER=martinec,CASE=B,EXTRA_PYTHONPATH=/g/data/vo05/$USER/python-extra \
     run_benchmark.pbs
```

Leave out `EXTRA_PYTHONPATH` if the module has `giamip`. The header of
`run_benchmark.pbs` lists all its variables. The script uses the project
`vo05` and loads the stable release `firedrake/2026.10.0` on Gadi. Set
`FIREDRAKE_MODULE` to select another module. Study plans additionally check
the required Firedrake package version before solving. For another project,
edit the `-P` and storage lines.

For Spada, `--load_shape cap` (default) selects the paper's 1500 m parabolic
cap; `--load_shape disc` selects its uniform 1000 m disc. Both have a 10°
angular radius and ice density 931 kg m⁻³ (Table 4). This choice is independent
of `--case`: `cap` remains the name of the non-rotating case, with its
default centre at the pole, even for a disc; `polar-motion` defaults to
colatitude 25°, longitude 75° and enables rotation. `--load_centre COLAT LON`
overrides either location. The TABOO U/V/N comparison uses the
selected load coefficients. The disc polar-motion reference scales the cap
series by `sigma_2(disc)/sigma_2(cap) = 0.9953950073`, following equation 31.
The Earth and loading history are unchanged, so this scaling is exact within
the linear reference model.

For example, run the non-rotating disc with the default spatial load:

```bash
qsub -N spada_disc_spatial \
    -v "DRIVER=spada,CASE=cap,OUTPUT=$PWD/outputs/disc-spatial,EXTRA=--load_shape disc --diagnostic_nmax 10 --write_profiles --write_output" \
    run_benchmark.pbs
```

For a disc polar-motion run, use `CASE=polar-motion` and
`EXTRA=--load_shape disc --write_output`, with a separate
output directory. Output filenames still follow the case (`summary_cap.json`
for the non-rotating disc), so keep different load shapes in separate folders.
Summaries and checkpoints record `load_shape`; older records imply `cap`.
Restarting a cap checkpoint as a disc is rejected. Existing error tolerances
and compressibility/time-step floors were calibrated for the cap only.
Spada now defaults to `--bulk_shear_ratio 1000`; use 100 for the coarser
compressibility approximation. The archived runs and existing numerical
error floors used 100, so those floor corrections need recalculating for
the new baseline. Restart an archived run with its original ratio explicitly.

For the DtN boundary sensitivity comparison, keep the mesh, time steps,
load and compressibility fixed and add `--dtn_degree 2` to `EXTRA` in a
separate output directory. The baseline is degree 5 on both the inner and
outer gravity boundary spheres. Degree 2 retains the quadrupole needed for
polar motion and reduces the number of harmonic modes from 36 to 9 per
sphere. It changes the gravity boundary approximation, independently of
`--load_nmax` and `--diagnostic_nmax`. Summaries and checkpoints record the
degree; older checkpoints imply 5, and restarting at another degree is
rejected. This provides a fifth one-at-a-time sensitivity variant alongside
time step, bulk mesh size, lithosphere layer count and compressibility.

A sixth sensitivity check loosens the coupled solver tolerance from 1e-6 to
1e-5 and its inner mechanics-potential tolerance from 1e-4 to 1e-3. For one
cap polar-motion comparison, with the other baseline settings unchanged:

```bash
qsub -N spada_pm_loose \
    -v "DRIVER=spada,CASE=polar-motion,OUTPUT=$PWD/outputs/cap-pm-loose,EXTRA=--outer_rtol 1e-5 --block0_rtol 1e-3" \
    run_benchmark.pbs
```

The cached Schur preconditioner is retained explicitly for both Spada cases.
The preset normally guards its automatic selection above an inner tolerance
of 1e-4, because inaccurate Schur columns can cause stagnation. This
experiment uses its supported explicit selection; check convergence reasons
and iteration counts as well as the benchmark errors. Start with a smoke
run before the full comparison. Tolerances are recorded in summaries and
checkpoints; older checkpoints imply 1e-6/1e-4 and cannot be continued with
different tolerances.

The spatial load is the default and has no upper harmonic cutoff before
mesh interpolation. Spectral loading is available explicitly with
`--load_representation spectral`; its cutoff defaults to degree 10.
The diagnostic cutoff independently defaults to 10. For example, retain
spectral load degrees 2–128 while projecting only degrees 0–10 for diagnostics:

```bash
qsub -N spada_cap_n128 \
    -v "DRIVER=spada,CASE=cap,OUTPUT=$PWD/outputs/cap-n128,EXTRA=--load_representation spectral --load_nmax 128 --diagnostic_nmax 10 --write_profiles --write_output" \
    run_benchmark.pbs
```

By default, evaluate the chosen load in space and subtract only its analytic
degree-0 and degree-1 components (`sigma_0 + sigma_1 cos(gamma)`):

```bash
qsub -N spada_cap_spatial \
    -v "DRIVER=spada,CASE=cap,OUTPUT=$PWD/outputs/cap-spatial,EXTRA=--load_representation spatial --diagnostic_nmax 10 --write_profiles --write_output" \
    run_benchmark.pbs
```

`--load_representation spatial` is the default; `--load_nmax` is unused
unless `--load_representation spectral` is also selected. Both modes
omit degrees 0 and 1 before CG2 interpolation and use the same mesh and
load prescription for the mechanics traction and gravity mass sheet. The
spatial load differs from a degree-128 series; mesh interpolation can introduce
small low-degree components in either mode. Add `--load_shape disc` to use
the disc's step profile and its own degree-0/1 subtraction in spatial mode.
Its sharp edge needs a load/mesh convergence check.

`--diagnostic_nmax` must be at least 2 and, for the cap comparison, no larger
than the available TABOO reference (currently 256). It controls the projected
coefficient arrays and the reconstructed pole values and maximum V, not the
detail retained by the solver or by direct profiles. If it exceeds a spectral
load's cutoff, reference coefficients above that cutoff are zero. Larger
diagnostic cutoffs still cost more surface assemblies. Use direct profiles
to compare the full computed fields with the appropriate load reference.
The existing benchmark tolerances were calibrated for the original degree-10
spectral configuration at K / mu = 100. Archived checkpoints without a load
representation are interpreted as spectral, and must be restarted with
`--load_representation spectral` and their original bulk/shear ratio.

The summary and checkpoint record `load_representation`, `load_nmax` (`null`
for spatial mode), and `diagnostic_nmax`. The legacy summary key `nmax` now
means the diagnostic cutoff. Restart with the same recorded settings. Old
cap checkpoints infer their shared cutoff from the recorded diagnostic
degrees; old polar-motion checkpoints are interpreted as degree-10 runs.

After the jobs end, run the test on a login node in this directory, with the
same module and the same `PYTHONPATH` as for the install:

```bash
python3 -m pytest -m longtest -rfxX test_benchmarks.py
```

### A smoke run

The flag `--smoke` runs a case on a very coarse mesh for 10 time steps. Use it
to check an installation before a full run. The numbers of a smoke run are
not results, and the test refuses them.

```bash
mpiexec -n 4 python3 spada.py --case cap --smoke --output_path smoke/
mpiexec -n 4 python3 martinec.py --case D --smoke --output_path smoke/
```

On a laptop with a debug build of PETSc, one step takes 45 to 100 s.

### Flags

| flag | meaning |
|---|---|
| `--case` | the case |
| `--load_centre COLAT LON` | Spada: load location in degrees, independent of rotation; defaults 0/0 for `cap`, 25/75 for `polar-motion` |
| `--bulk_shear_ratio` | K / mu in every layer; default 1000 for Spada and Martinec case B, 100 for other Martinec cases |
| `--dtn_degree` | Spada: harmonic cutoff on both gravity boundary spheres; default 5, minimum 2 |
| `--outer_rtol`, `--block0_rtol` | Spada: coupled/inner relative tolerances; defaults 1e-6/1e-4 |
| `--timestep_scale` | Spada: multiply the graded timestep ladder; default 1, coarse check 2; output epochs remain solved states |
| `--time_step_yr` | Martinec: uniform step in years; defaults B 10, C/D/E 50 (E smoke 2500); adjusted to land on output times |
| `--time_step_ladder END_YR:STEP_YR [...]` | Martinec B only: experimental graded steps in years; increasing end times must cover 10 kyr; mutually exclusive with `--time_step_yr` |
| `--mesh_h_km` | Spada: bulk mesh size in km; default 250, or 1000 for `--smoke` |
| `--litho_layers` | Spada: subdivisions of the 70 km lithosphere; default 2, or 1 for `--smoke` |
| `--write_output` | also write VTK files at every output epoch |
| `--write_surface_output` | Spada: surface-only displacement, geoid and rates in physical units; independent of `--write_output` and `--write_profiles` |
| `--surface_output_format` | Spada: `hdf5` (default), one collective surface file per run; `vtk` for direct parallel VTK output |
| `--output_path` | the directory for every output file (default `./`) |
| `--smoke` | the coarse mesh and 10 steps |
| `--checkpoint_every` | the number of steps between two checkpoints (default 50; 0 writes none) |
| `--restart` | continue a run after the last checkpoint in the given file |

### A restart

Every run writes a checkpoint every 50 steps into `checkpoint_<case>.h5`,
a Firedrake `CheckpointFile`. The file holds the mesh once and one entry for
each checkpoint. If a job stops before its end, continue it from the last
entry:

```bash
qsub -l ncpus=1872 -l mem=9000GB -l walltime=01:30:00 \
     -v DRIVER=martinec,CASE=B,RESTART=checkpoint_B.h5 run_benchmark.pbs
```

Set the walltime from the steps that remain; 50 Martinec steps take about
40 minutes on 18 nodes. Use the same number of ranks as the job that wrote
the file. The restarted
run loads the mesh from the file, continues after the last checkpoint and
appends its own checkpoints to the same file. It writes the parameter log,
the time series and the summary again, with the rows of the earlier job.
If the earlier job stopped during a save, the restart uses the last
complete checkpoint. A new run without `RESTART` renames an existing
`checkpoint_<case>.h5` to `checkpoint_<case>.previous.h5` first.
The summary records every restart and the wall time of all jobs together.

## Output

Every file name contains the case, so all five cases can run in one
directory.

| file | content |
|---|---|
| `params_<case>.log` | one line per time step: time, step length, wall time, iteration counts and a few numbers of the state |
| `summary_<case>.json` | the run settings, the mesh record and, for Spada, the comparison with TABOO at every epoch |
| `spada_<case>.msh`, `martinec_<case>.msh` | the mesh of the run |
| `martinec-<case>-profiles_<t>kyr.npz` | U, N, S and RSL along the two comparison meridians, at 1801 colatitudes from 0 to 180 degrees, at every epoch |
| `martinec-<case>-timeseries.npz` | h_UF, ocean area, ice masses and iteration counts at every step |
| `checkpoint_<case>.h5` | the checkpoints of the run, for `--restart` |

`test_benchmarks.py` converts the two Martinec npz files to a `giamip`
`BenchmarkResult`. `giamip` evaluates the VEGA reference at the run's own
colatitudes with a cubic spline.

## Meshes

Each driver generates its mesh with gmsh at a fixed resolution. The job log
gives the gmsh version, the random seed, the MD5 of the mesh file and the
cell counts. Another gmsh version or another machine gives other
tetrahedra, so two runs with different MD5 values used different meshes.

The driver then curves the mesh. It interpolates the coordinates into CG2
and moves the midpoint of each edge whose two vertices lie on one mesh
sphere onto that sphere. The surface, the core-mantle boundary, the density
interfaces and the two DtN spheres are then piecewise quadratic surfaces. The displacement is CG3, the potential
CG2 and the internal variables DG2.

| mesh | lateral size | lithosphere | cells (gmsh 4.15.2, Gadi x86, the seeds of `MESHES`) | unknowns | nodes |
|---|---|---|---|---|---|
| `spada` | 250 km | two 35 km layers | 595 633 | 30.3 million | 15 |
| `martinec` | 78 km in the ice cap and along both coastlines, 250 km elsewhere | two 35 km layers | 742 357 | 37.6 million | 18 |
| `smoke` | 1000 km | one 70 km layer | 19 330 | 1.0 million | 4 ranks |

The node counts give about 20 000 unknowns per core on `normalsr` nodes.

The driver distributes the mesh with its own partition
(`selfgrav_common.balanced_partition`). A third of the cells are in the
buffer shell and in the inner shell, where only the potential lives. The
default partitioner cuts the mesh into compact pieces of equal cell count,
so on hundreds of ranks many pieces hold no mantle cell. Firedrake then
fails on those ranks, and the mantle work, which is most of the cost, is
unbalanced. The driver orders the cells along a Hilbert curve of their
direction from the centre and splits the mantle cells into equal counts.
The mantle work is balanced to one cell. The buffer and inner work, the
potential alone, is not balanced. Each rank then owns a solid-angle sector
through all the shells.

The job stops with an error if the mesh has a defective cell. A tetrahedron
with all four vertices on one mesh sphere (a flat cell) folds when the driver
curves the mesh. The curved cell then has a Jacobian determinant that
changes sign, which gives a wrong answer without an error message. The
Delaunay algorithm of gmsh makes flat cells at random, and the same seed
gives other tetrahedra on another machine. The driver therefore starts at
the seed of `selfgrav_common.MESHES` and tries the next seeds, up to
`SEED_TRIES`, until the mesh has no flat cell. After the curving, it checks
every cell for a Jacobian determinant of one sign.

## Pass criteria

### Spada et al. (2011)

The paper gives no pass tolerance. Each tolerance in `test_benchmarks.py` is
a known error floor of this model plus a margin for the mesh and the solver.
Three floors are known:

- K / mu = 100. The model is compressible and TABOO is incompressible. At
  t = 0 this gives U(0) 2.5 percent high and the largest V 3.6 percent low.
  The effect decreases with time.
- Backward Euler on the step sequence of the cap case. N(0) is about 1.1
  percent high at 5 kyr.
- The moment difference. The model uses C - A = 2.6952e35 kg m^2, the value
  that goes with the secular Love number of the reference. The reference
  excitation uses 2.63e35 kg m^2, so |m| / |m_ref| is near 0.976.

The test applies these bounds:

| quantity | bound |
|---|---|
| U(0), N(0), largest V, ratio to TABOO | within 3, 1.5 and 4 percent of 1, at every epoch |
| the same ratios, divided by the predicted floor of the epoch | within 0.5 percent |
| degree-0 uplift, \|U_0\| / \|U_2\| | below 1e-5 |
| phase of the polar motion | within 0.01 degrees of -105 degrees |
| \|m\| / \|m_ref\| | between 0.965 and 0.980 |

The fixed bounds must hold at every epoch. So the compressibility floor at
t = 0 sets the bounds on U(0) and the largest V. At 20 kyr that floor is
small, and a late-time error of 2 to 3 percent passes the fixed bounds. The
second check removes this gap. It divides each ratio by the floor that
lovejx, a Love-number code that is not in this repository, predicts for
that epoch (`SPADA_CAP_FLOOR`). The remainder, the error of the mesh and the
solver, must stay within the bound. If the step sequence or K / mu of the
cap case changes, the predicted floors must be computed again. The comments
in `test_benchmarks.py` give the numbers behind each bound.

### Martinec et al. (2018)

The test uses the rules of the earlier Martinec runs on the branch
`sghelichkhani/sea-level`:

1. h_UF at the terminal time is within 1 percent of VEGA.
2. The test compares each quantity along each meridian with every other
   published code. Our largest difference from VEGA is not larger than
   theirs, and our root mean square difference is not larger than theirs.
   Both sides exclude a band of 6.74 degrees around the cap margin and
   around the coastline. The test calls `case.envelope_check` of `giamip`.
3. Case C and case D: the grounded ice mass is within 0.5 percent of VEGA.
   Case D also: the floating ice mass is within 0.5 percent, and the ocean
   area is inside the range of the published codes.

Near the cap margin and the coastline all codes disagree strongly, and the
paper does not give the width of the excluded band. At 1.05 degrees instead of
6.74 degrees, the earlier case C run fails rule 2 on U along the basin
meridian by 4 percent.

Case B is sensitive to the time step. The load is a step that is held, so
at 10 kyr most of the geoid change is still to come. Backward Euler is first
order in time. The graded steps of the Spada cases are 10 yr to 0.1 kyr,
50 yr to 1 kyr and 100 yr to 10 kyr. On these steps case B fails rule 2 on
U and N along the load meridian. The driver therefore takes uniform 10 yr
steps for case B (1000 steps). On these steps lovejx predicts a largest
load-meridian N difference of 0.0250 m, with an uncertainty of about 4 mm.
The published codes reach 0.0256 m. The run on these steps brings load U
inside the spread of the published codes. Load N stays 3.5 mm outside:

| load meridian | graded steps | 10 yr steps | published codes |
|---|---|---|---|
| U, largest difference | 0.6516 m | 0.0825 m | 0.3266 m |
| N, largest difference | 0.0917 m | 0.0291 m | 0.0256 m |
| N, root mean square difference | | 0.0075 m | 0.0171 m |

The largest N difference is at the cap centre, colatitude 25 degrees. It is
4 mm above the prediction, and the cause of that remainder is not known.
The test for rule 2 of case B is therefore marked `xfail`.

Case B also accepts a configurable timestep ladder for convergence studies.
For example, to reproduce the Spada schedule through 10 kyr (118 marched
steps plus the separate elastic solve):

```bash
mpiexec -n 4 python3 martinec.py --case B \
    --time_step_ladder 100:10 1000:50 10000:100 --output_path case_B_ladder/
```

Each pair gives a segment's end time and step length in years. End times
must be finite, positive and strictly increasing; step lengths must be
finite and positive. The final segment must reach at least 10,000 yr and
is truncated there if necessary. Steps are adjusted within each segment to
land exactly on output epochs, so every profile is a solved state. With
`--smoke`, the run still stops after ten marched steps.

Uniform 10 yr steps remain the default. The example above is the coarser
schedule that failed the case B comparison, so check accuracy against the
uniform run and VEGA before adopting another ladder. The summary records
`time_step_ladder_yr` as pairs of end and step times, and `time_segments_yr`
as `(start_yr, end_yr, actual_step_yr, step_count, is_output_epoch)` tuples.
For a ladder run, `time_step_yr` is `null`. Restart with the same effective
ladder; changing it is rejected. Earlier uniform-step checkpoints remain
compatible with the matching uniform settings.

## Walltime and cost

For the 27-run Spada paper study plus four smoke checks, see
[SPADA_STUDY.md](SPADA_STUDY.md). `spada_study.py prepare` writes the full
manifest and Gadi PBS files without submitting anything;
`estimate_spada_cost.py` produces a reproducible SU estimate from the local
logs. The current central estimate is about 95,000 SU, with the assumptions
and uncertainty documented in the generated report.

The runs below are on Gadi with `firedrake/main-20260918`, whole `normalsr`
nodes, gmsh 4.15.2. The walltimes in `meta.py` and `run_benchmark.pbs` are
about 1.5 times these. The normalsr queue allows at most 24 h for a job of
1144 to 2080 cores. A case that needs more time continues in a second job
from its last checkpoint (see "A restart").

| case | job | nodes | wall | SU | memory | mesh MD5 | verdict |
|---|---|---|---|---|---|---|---|
| Spada cap | 179921930 | 15 | 1 h 01 min | 3 182 | 1.16 TB | `22d2602d87e6029327cf910cad6a76ad` | pass |
| Spada polar motion | 179923050 | 15 | 1 h 01 min | 3 162 | 0.93 TB | `22d2602d87e6029327cf910cad6a76ad` | pass |
| Martinec C | 179921931 | 18 | 1 h 53 min | 7 055 | 1.18 TB | `c3782d68cbd1babd889baa2c2566b66c` | pass |
| Martinec D | 179923052 | 18 | 9 h 57 min | 37 244 | 1.79 TB | `c3782d68cbd1babd889baa2c2566b66c` | pass |
| Martinec B | 180015010, then 180015049 | 18 | 11 h 55 min + 43 min | 44 646 + 2 678 | 1.76 TB | `c3782d68cbd1babd889baa2c2566b66c` | rule 1 passes; rule 2 fails on load N (`xfail`) |

Case B does not finish in one job at present. Three jobs crashed after step
998 of 1000, at the same step. Each time the first rank of every node had a
segmentation fault. The backtrace ends in HCOLL, the collectives library of
the Gadi Open MPI. The call is an `MPI_Allreduce` of the low-rank DtN
operator (`gadopt/preconditioners.py`, `_LowRankPotentialOperator.mult`).
Inside HCOLL it goes to the multicast broadcast
(`hmca_mcast_vmc_bcast_multiroot`, `vmc_bcast_multiroot`). A second job
with `RESTART=checkpoint_B.h5` finishes the case from the checkpoint at step
950. At step 998 the restarted job gives the same h_UF as the first job
(-119.9284 m). The module `petsc/main-20260920`, which `firedrake/main`
loads, sets `OMPI_MCA_coll=^holl`. This value does not switch HCOLL off.
Without HCOLL the Martinec steps take twice as long.

## Direct Spada profiles and field plots

For either case and load shape, `--write_profiles` saves direct
finite-element displacement and geoid profiles at every output epoch as
`profiles_<case>_<time>kyr.npz`. Non-rotating runs retain the original three
meridians at 0°, 90° and 180° longitude, with 0.1° colatitude spacing.
Off-axis non-rotating runs and rotating runs default to two great circles through the load centre
(colatitude 25°, longitude 75°): north–south and east–west at that centre.
Their coordinate is signed angular distance from the centre, from −180°
to +180° at 0.1° spacing. Positive distance follows the selected bearing;
the east–west great circle is not a constant-latitude line.

All displacement arrays are in metres and have shape `(profile, sample)`:
`U_m` is radial, `V_m` geographic southward and `W_m` geographic eastward.
`along_m` follows increasing profile coordinate, and `across_m` points to
its right (along-direction crossed with radial direction). Rotating plots
show U, along, across and N, capturing both horizontal components. Default
cap meridian files preserve the original `colatitude_deg`, `longitude_deg`,
`U_m`, `V_m`, `N_m` and `found` arrays for existing plotting scripts.

Every file includes the profile labels, coordinate, sample colatitudes and
longitudes, and vector bases. Great-circle files also record their centre
and bearings. Summaries and checkpoints include compact `profile_sampling`
metadata. Sampling options are output choices, independent of solver settings.
Points sit 1e-6 Earth radii (about
6.4 m) below the nominal surface to help locate them in the curved mesh.
The depth fraction and point-found mask are saved; missing values are NaN
and counted in the job output. N uses gravity at the nominal Earth surface.

`--write_output` also saves displacement and potential as VTK at each epoch.
These fields retain nondimensional units: multiply displacement by `D_SCALE`
and potential by `G_BAR * D_SCALE` for physical units.

For maps, use **`--write_surface_output`**. It writes
one collective Firedrake `CheckpointFile`, `spada_<case>_surface.h5`, for either
case and either load shape. The mesh is stored once and fields are appended
at every output epoch. Omit `--write_output` to avoid volume VTK;
`--write_profiles` is optional and independent. For example:

```bash
python3 spada.py --case polar-motion --write_surface_output --output_path outputs/polar-surface
```

This output uses a submesh of the simulation's actual surface triangles,
with the same resolution and quadratic curved geometry. Displacement is
copied exactly at its CG3 nodes and potential at its CG2 nodes, through
the submesh topology maps. There is no inward sampling offset or point
search. VTK uses degree-3 triangles to retain the cubic displacement.
The undeformed output coordinates are in metres and PVD times are in kyr;
simulation coordinates are unchanged.

Convert afterwards in a Firedrake environment, using fewer MPI ranks or
serially if memory permits. This produces fewer VTK pieces without changing
the saved resolution. Conversion uses
[dsroberts/firedrake_h5_to_vtk](https://github.com/dsroberts/firedrake_h5_to_vtk)
unchanged, through a Spada adapter. It needs the surface HDF5 file,
Firedrake and the upstream script; the volume mesh, restart checkpoint and
G-ADOPT are not required. Set up the external converter once:

```bash
git clone https://github.com/dsroberts/firedrake_h5_to_vtk.git "$HOME/src/firedrake_h5_to_vtk"
git -C "$HOME/src/firedrake_h5_to_vtk" checkout ba77f27fcf2d8cc4531e5547489924187c4308af
export FIREDRAKE_H5_TO_VTK="$HOME/src/firedrake_h5_to_vtk/firedrake_h5_to_vtk.py"
```

The commit above is the version used in our integration tests. Alternatively,
put its checkout on `PYTHONPATH` or pass `--converter /path/to/firedrake_h5_to_vtk.py`.
Then convert with the existing commands:

```bash
python3 convert_spada_surface.py -i outputs/polar-surface/spada_polar-motion_surface.h5 --list
mpiexec -n 2 python3 convert_spada_surface.py \
    -i outputs/polar-surface/spada_polar-motion_surface.h5 \
    -o outputs/polar-surface/maps.pvd
python3 convert_spada_surface.py \
    -i outputs/polar-surface/spada_polar-motion_surface.h5 \
    -o outputs/polar-surface/final-geoid.pvd -t last -f geoid_total_m
```

`-t` accepts `all`, `last`, or comma-separated frame indices from `--list`;
`-f` accepts `all` or comma-separated field names. PVD timestamps retain the
physical times in kyr even when only a subset is converted. The adapter
passes selected complete, active frame indices to upstream's `main()`;
upstream performs mesh/field discovery, loading and VTK export. Afterwards
the adapter replaces frame-number timestamps in the PVD collection with
the recorded kyr values. It does not change the VTU pieces. Upstream uses
the maximum degree of the selected fields and geometry, preserving CG3
when displacement is selected. No upstream source is copied or patched here.

For small debugging runs, `--write_surface_output --surface_output_format vtk`
instead writes `spada_<case>_surface.pvd` and parallel `.pvtu`/`.vtu` pieces
during the solve. The default HDF5 path creates no VTK pieces during the run.

Surface fields are:

- `displacement_m`: Cartesian displacement vector, plus
  `radial_displacement_m`, `northward_displacement_m` and `eastward_displacement_m`.
- `geoid_gravitational_m`: gravitational potential divided by nominal
  surface gravity, matching the profile geoid definition.
- `geoid_rotational_m`: the rotational contribution, using the solver's
  negated centrifugal potential divided by the same gravity. It is zero
  when rotation is disabled.
- `geoid_total_m`: the sum of the gravitational and rotational contributions.
- Corresponding rates, replacing `_m` with `_rate_m_per_yr`, including
  `displacement_rate_m_per_yr` and `geoid_total_rate_m_per_yr`.
- `rate_valid`, `rate_dt_yr`, `rate_interval_start_kyr` and
  `rate_interval_end_kyr`: the adjacent solved timestep used for each rate.

Rates are backward differences over the last simulation timestep, rather
than the interval between saved frames. They are stored during the solve;
the converter does not recompute them. At t=0 they are undefined (NaN,
with `rate_valid=0`). Archive the surface HDF5 file. When copying converted
VTK results, keep all referenced pieces with the PVD index. The summary
records the surface mesh and field metadata.

The surface archive is separate from `checkpoint_<case>.h5`: it contains
visualisation fields at output epochs, not the full state needed for restart.
A fresh run preserves an existing surface archive as `*.previous.h5`.
A restart appends to the existing archive, ignores incomplete frames, and
marks frames later than the restored checkpoint as obsolete. The converter
uses only complete, active frames. Appending requires the original rank
count/partition and matching surface nodes (checked before writing). Nodes
reordered by loading the volume checkpoint are matched to the stored
surface nodes and their values copied without field interpolation.
Reading/conversion can use a different rank count. If surface output is
first enabled on restart or the archive is absent, its history starts at
the next output epoch.

Start with a smoke run in a new output folder, submitted from this directory:

```bash
mkdir -p smoke/spada-profiles
qsub -N spada_profiles_smoke \
    -l storage=scratch/xd2+scratch/vo05+gdata/vo05+gdata/fp50 \
    -l ncpus=104 -l mem=500GB -l walltime=01:00:00 \
    -v "DRIVER=spada,CASE=cap,SMOKE=1,OUTPUT=$PWD/smoke/spada-profiles,EXTRA=--write_profiles --write_surface_output" \
    run_benchmark.pbs
```

Adjust storage projects to the checkout/output locations. Check missing-point
counts before a full run; the coarse mesh may need a larger inward offset.
Set `--profile_depth_fraction` to adjust it, and check sensitivity to that
offset before interpreting the samples as surface values.

Plot with NumPy and Matplotlib installed (no Firedrake needed):

```bash
python3 plot_spada_profiles.py smoke/spada-profiles/summary_cap.json \
    --profiles-dir smoke/spada-profiles --max-colatitude 25 \
    --output smoke/spada-profiles/profiles_near.png
```

The plot overlays direct meridians, projected G-ADOPT profiles, matching
truncated TABOO and all available TABOO degrees. No correction factors are
applied. Omit `--profiles-dir` for existing summaries without sampled profiles.
The summary's `load_shape` selects the cap or disc TABOO curves automatically.

For a rotating run, enable the default great-circle profiles with:

```bash
qsub -N spada_pm_profiles \
    -v "DRIVER=spada,CASE=polar-motion,OUTPUT=$PWD/outputs/cap-pm-profiles,EXTRA=--write_profiles" \
    run_benchmark.pbs
```

To choose locations, `--profile_bearings 0 45 90` selects great circles
clockwise from geographic north at the profile centre. The default centre
is the load location; override it with `--profile_centre COLAT LON`.
`--profile_extent_deg 30` samples −30° to +30° from that centre, while
`--profile_spacing_deg 0.25` sets a maximum spacing of 0.25° (the grid
includes the centre and both endpoints). Alternatively,
`--profile_longitudes 75 165` selects geographic meridians, sampled from
colatitude 0° to `--profile_extent_deg`. Longitude selection cannot be
combined with bearings or a great-circle centre. These options also work
for non-rotating cases. For example:

```bash
python3 spada.py --case polar-motion --write_profiles \
    --profile_bearings 0 45 90 --profile_extent_deg 30

python3 plot_spada_profiles.py outputs/cap-pm-profiles/summary_polar-motion.json \
    --max-distance 25 --output outputs/cap-pm-profiles/profiles_near.png
```

For rotating runs, the plotting script automatically reads direct profile
files beside the summary; use `--profiles-dir` to override their location.
It shows sampled fields without a non-rotating TABOO overlay. The same
direct plotting path is selected for non-rotating great-circle output.
The pole-centred meridian comparison continues to overlay TABOO as before.
The rotating spatial fields require a rotational spatial reference for an
independent benchmark comparison; Spada's Figure 13 provides polar-motion
time series rather than those spatial profiles.

### Adjacent-step rates and matched rotation controls

`--write_profiles` also saves `U_m_per_yr`, `V_m_per_yr`, `W_m_per_yr`,
`along_m_per_yr`, `across_m_per_yr` and `N_m_per_yr`. These are backward
differences between consecutive solved timesteps, sampled at each output
epoch. The files record `rate_interval_start_kyr`, `rate_interval_end_kyr`,
`rate_dt_yr`, `rate_method` and `rate_valid`. The t=0 elastic jump has no
finite difference rate: those arrays are NaN and `rate_valid` is false.
The first marched interval uses the separate elastic diagnostic as its
previous state. Later intervals, including the first after a restart, use
the previous marched solution. This avoids dividing the elastic jump by
the first timestep. The elastic diagnostic itself uses the small existing
`DT_ELASTIC`, so it is an approximation to the instantaneous response.

Plot rates with:

```bash
python3 plot_spada_profiles.py outputs/cap-spatial/summary_cap.json \
    --rates --max-distance 25 --output rates_near.pdf
```

The rate plot omits t=0 and shows U, along, across and N in m/yr. For
nonrotating runs it overlays the instantaneous analytic TABOO derivative,
expressed in the saved profile basis. A timestep difference is an interval
average, so assess its difference from the instantaneous reference with
the timestep sensitivity run. Older profiles contain no rate arrays and
must be regenerated. As before, N is the gravitational contribution
`psi/g0`, excluding centrifugal potential; this definition is saved in the
profile file and applies to its rate too.

For a nonrotating control at exactly the rotating load location:

```bash
qsub -N spada_offaxis \
    -v "DRIVER=spada,CASE=cap,OUTPUT=$PWD/outputs/cap-offaxis,EXTRA=--load_centre 25 75 --write_profiles --write_output" \
    run_benchmark.pbs
```

Add `--load_shape disc` for its disc counterpart, in another directory.
Both controls use the same default sampling as `--case polar-motion`.
Load location is recorded in summaries/checkpoints; legacy checkpoints
imply the historical location of their case, and a changed location is
rejected on restart. For nonrotating off-axis loads, the summary's
Legendre spectra and U0/U180/Vmax are measured about the load axis; V
points away from the load centre. The saved `V_m` profile array remains
geographic southward, with `along_m`/`across_m` in the profile basis.

### Length of day

Rotating step logs now include `m3` and `lod_ms`; epoch summaries include
the same quantities. `m3 = delta(Omega)/Omega = -delta(I33)/C`, and
`lod_ms = -86400e3*m3`: a positive value means a longer day. The conversion
is linearized and uses the paper's nominal 86400 s day.

Epoch summaries separately save `lod_eq33_ref_ms`, reconstructed from
TABOO's degree-2 loading Love number with the geometrical factor of Spada
equation 33. It reproduces the supplied GS LOD archive curve, which was
absent from `reference.npz`. No new external dataset is needed to compute
this reference. There is a sign-convention discrepancy to resolve before
scoring a signed LOD benchmark: the printed equation 33 factor is positive
for the default load, whereas the degree-0-free inertia contribution is
negative under the solver's signed angular-momentum convention. The
solver output and published formula are therefore labelled separately,
with no sign adjustment or relative-error score applied.

On restarting an older rotating checkpoint, historical log rows without
m3/LOD receive NaNs in the added columns; only newly solved states have
these diagnostics. Earlier profile epochs do not acquire rates on restart.

### Compare load prescriptions

To compare load degrees and the spatial cap, put each full cap run's
`summary_cap.json` and all its `profiles_cap_<time>kyr.npz` files in a separate
subdirectory of `outputs/`, then run:

```bash
python3 plot_spada_load_convergence.py --outputs outputs --max-colatitude 25
```

Use `--load-shape disc` for the disc series. Runs are filtered by shape,
so cap and disc results cannot be mixed in one convergence comparison.
Disc figure and CSV filenames start with `disc_`.

This needs NumPy and Matplotlib, with no Firedrake. It writes PNG and PDF
figures for the load comparison, differences from TABOO, RMS error versus
load cutoff, and the degree-128 run against TABOO at all common epochs.
It also writes `load_convergence_metrics.csv` with errors in metres and
normalised by the reference peak amplitude. RMS errors pool the sampled
meridians with uniform colatitude weighting. The profile comparison shows
the 0-degree meridian; the degree-128 figure shows all three meridians.
If degree 128 is absent, the latter figure uses the highest available cutoff.

All comparisons use the fixed TABOO reference, degrees 2–256, without
compressibility or timestep corrections. Direct G-ADOPT samples sit below
the surface as described above. If this directory's legacy degree-10
`summary_cap.json` is present, its reconstructed spectrum is included as a
labelled baseline when no direct degree-10 dataset is available.
