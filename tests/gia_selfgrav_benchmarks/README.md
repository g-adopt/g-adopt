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

Each case runs on Gadi as one job. `test_benchmarks.py` then compares the
output with the published criteria.

## The cases

| driver | case | load | what the test compares |
|---|---|---|---|
| `spada.py` | `cap` | 1500 m ice cap on the pole, switched on at t = 0 | U, N and V against TABOO at 0 to 20 kyr |
| `spada.py` | `polar-motion` | the same cap at colatitude 25, longitude 75, rotation on | polar motion against TABOO at 0 to 20 kyr |
| `martinec.py` | `B` | 1500 m cap, switched on at t = 0; fixed coastline, basin B1 | U, N, S and h_UF against VEGA at 10 kyr |
| `martinec.py` | `C` | 500 m cap, grows over 10 kyr; fixed coastline, basin B2 | as B, plus the grounded ice mass, at 15 kyr |
| `martinec.py` | `D` | as C, with a moving coastline and floating ice | as C, plus the floating ice mass and the ocean area |

The docstring of each driver describes the physics, the scales, the
discretisation and the output. `selfgrav_common.py` holds the parts that
both drivers use: the Earth model, the mesh, the time steps and the solver
settings.

## Files

| file | content |
|---|---|
| `spada.py`, `martinec.py` | the two drivers |
| `selfgrav_common.py` | the shared module |
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
module load firedrake/main
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
qsub -l ncpus=1872 -l mem=9000GB \
     -v DRIVER=martinec,CASE=B,EXTRA_PYTHONPATH=/g/data/vo05/$USER/python-extra \
     run_benchmark.pbs
```

Leave out `EXTRA_PYTHONPATH` if the module has `giamip`. The header of
`run_benchmark.pbs` lists all its variables. The script uses the project
`vo05` and loads `firedrake/main`, the newest weekly build of the Firedrake
module. For another project, edit the `-P` and storage lines.

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
| `--bulk_shear_ratio` | K / mu in every layer; default 100, and 1000 for Martinec case B |
| `--write_output` | also write VTK files at every output epoch |
| `--output_path` | the directory for every output file (default `./`) |
| `--smoke` | the coarse mesh and 10 steps |

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
load-meridian N difference of 0.0250 m. The published codes reach 0.0256 m,
so the margin is small. The margin is 0.6 mm. The uncertainty of the
prediction is about 4 mm, and the prediction is calibrated on the earlier
run on the coarser mesh. Only the run decides. The test for case B is marked
`xfail` until a run on these steps confirms the pass.

## Walltime and cost

No run on these meshes exists yet. The walltimes in `meta.py` and
`run_benchmark.pbs` are provisional. Set each one from the first timed run.
The normalsr queue allows at most 24 h for a job of 1144 to 2080 cores. The
drivers cannot restart, so each case must finish inside 24 h.

On the earlier, coarser Martinec mesh, case C took 1 h 26 min. Case D took
6 h 17 min in two jobs of 2 nodes each.
