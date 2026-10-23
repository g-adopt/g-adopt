# GIAMIP Exp11: the bed at 122 ka by adjoint inversion

This directory finds the bed at the start of GIAMIP experiment Exp11. The
run of the coupled self-gravitating sea-level solver must carry that bed to
the observed present-day bed at year 0. The inversion minimises the misfit
between the predicted bed at year 0 and the observed bed.

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. It is a copy of the Exp01 directory
(`../exp01/`) with only the changes in "Changes for another experiment".
Exp11 is Exp03 with the 3-D viscosity of Lloyd et al. (2024).

## The experiment

The values come from `giamip.mip.inputs.GiamipExperiment("Exp11")`, the
case file `giamip/resources/cases/giamip/exp11.json` and the model file
`giamip/resources/models/lloyd2024.json` of gia-mip (0cf5f39 or later). The
protocol leaves the earth model and the topography of Exp11 open. The values
marked "G-ADOPT choice" come from the `gadopt` block of the case file.

| item | value |
|---|---|
| ice | `iceHistory-GLAC3c_profile3.nc` |
| topography | `bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc` (observed present-day bed; G-ADOPT choice `bedmap3`) |
| start | 122 ka |
| march | 61 steps of 2 kyr to year 0 |
| viscosity | Lloyd et al. (2024), 3-D (G-ADOPT choice `lloyd2024`): g-drift dataset `3d_earthmodel_Lloyd_et_al_2024_GJI`, field `log10_viscosity`, the unmodified inference (1e18 to 1e39 Pa s); lower bound 1e20 Pa s from the surface to 670 km and 1e21 Pa s below 670 km, applied to log10 at the cell centroid; no upper bound; no elastic lid |
| elastic structure and density | PREM-stable (G-ADOPT choice) |
| rheology | Maxwell (G-ADOPT choice) |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables, P2 potential |
| time stepping | backward Euler |

## The files

| file | purpose |
|---|---|
| `exp11_common.py` | code that both scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp11.py` | reads giamip, g-drift and the mesh, and writes `inputs_exp11.h5` |
| `invert_exp11.py` | loads `inputs_exp11.h5`, tapes the march, and runs a Taylor test or the ROL inversion |
| `run_prepare.slurm` | the prepare job on Setonix |
| `run_invert.slurm` | the inversion job on Setonix |

`invert_exp11.py` does not import `giamip`, `gdrift`, `h5netcdf` or
`netCDF4`. The prepare script makes every field that comes from data. The
tape of the inversion holds only the operations from the control to the
misfit.

## The mesh

All GIAMIP experiments use the coarse mesh `n2_giamip_all_o1.msh` (MD5
`a4da84206dc82f5b458be66e4fd33844`) and its JSON sidecar. On Setonix the
mesh is in `/software/projects/pawsey0821/sghelichkhani/meshes/n2_giamip_all_o1/`.
The prepare script stops if the MD5 of the mesh file is different.

The mesh has spheres at 60, 90, 120, 400, 670, 980, 1260, 1470, 1950 and
2400 km depth, at the CMB and at the surface. In Exp11 only the 670 km
sphere matters for the viscosity: the lower bound changes there. The 60, 90
and 120 km spheres have no special role, because no lid is added.

## The 3-D viscosity

The prepare script reads the model file `lloyd2024.json` from the installed
giamip package (`importlib.resources`). It takes the dataset name, the field,
the SHA-256, the bounds and the lithosphere rule from that file. It stops if
the file asks for a rule that the script does not implement (an upper bound,
an added lithosphere, bounds that do not act on log10), or if the SHA-256 of
the dataset is not the one of the model file.

g-drift interpolates the stored log10 viscosity at the centroid of each cell:
inverse-distance weights over the 8 nearest points of each of the two
bracketing depth layers, then linear in radius. The lower bound applies to
the interpolated log10 value, by the depth of the centroid. The cell then
takes the power 10 of that value. Every cell takes its viscosity from the
model, also in the top 120 km. The stiff top of the model (crust 1.5e26 Pa
s, cratons up to 1e39 Pa s) acts as the lithosphere.

The dataset has 201 depth layers from 0 km (radius 6371 km) to 2891 km
(radius 3480 km, the CMB of PREM-stable), each a Fibonacci lattice of 65 341
points (about 88 km apart). It covers the whole mantle, so no value is
extrapolated. The script stops if the dataset does not reach the surface or
the CMB of the elastic structure.

The record of the prepare job (`viscosity_3d`) holds the model file and its
SHA-256, the dataset file and its SHA-256, the g-drift version, and per mesh
layer the min, max and volume mean of log10 viscosity before and after the
bound and the volume fraction of the cells at the lower bound.

The g-drift dataset uses about 420 MB of memory. One rank per node loads it
and computes the values for the other ranks of its node.

## How to run

The input file holds a mesh that is partitioned for the number of ranks of
the prepare job. The inversion job must use the same nodes and ranks.

1. Send the code to the Setonix checkout.
2. In a new run directory, submit the prepare job:

   ```bash
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp11/run_prepare.slurm
   ```

3. Read `inputs_exp11.json`. Compare the values with the table above, the
   statistics per layer, and the ice volume check against the grid.
4. In a second run directory, submit the Taylor test with 3 steps:

   ```bash
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp11.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp11/run_invert.slurm
   ```

   The R1 rates must be near 2.
5. Submit the inversion with 3 steps and 3 iterations
   (`MODE=invert,STEPS=3,ITERATIONS=3`). J must decrease at every iteration.
6. Submit the full inversion (`MODE=invert`).
7. If the job stops at the time limit, submit it again in the same run
   directory with `RESTART=state.h5`.

Add `--partition=debug` to `sbatch` for the short tests of steps 2, 4 and 5.

## The control and the node map

The control is the bed at the start of the run, in metres, a CG2 field on
the surface Re. The surface is a codim-1 `Submesh` of the mantle. Its
geometry is P1 (flat triangles), because a submesh takes its coordinates
from the DMPlex of the mantle, which holds only the vertices. The mantle
has P2 geometry. The solver reads the initial sea level `SL_init`, a CG2
field on the mantle, on the facets of Re only.

The CG2 nodes of the surface are the CG2 nodes of the mantle on Re.
`exp11_common.SurfaceNodeMap` pairs these nodes once, by position, and
stores the pairing as a sparse matrix of zeros and ones. The inversion
uses this map for every move of a field between the surface and the
mantle:

- `extend` copies `-b / D` onto the mantle nodes on Re and sets all other
  nodes to zero. This is the first block on the tape
  (`SurfaceExtendBlock`). Its adjoint is the transpose of the copy.
- `restrict` copies the mantle nodes on Re onto the surface. The
  inversion uses it for the observed bed and for the bed of a restart.

`restrict(extend(b))` is equal to `b` bit for bit. The script does not use
`interpolate` between the surface and the mantle. For a codim-1 submesh,
Firedrake finds each target node with a point search and evaluates the
source field there. The curved midpoint nodes of the mantle are not on the
flat surface triangles, so that route does not return the nodal values.

## The stop rule

The inversion stops after `--iterations` ROL iterations (10), or earlier
when the RMS bed misfit `sqrt(2 J / A)` is less than `--stop_rms_m` (1 mm).

## Restart

`gadopt.inverse.LinMoreOptimiser.restore` cannot restore this control.
Firedrake loads a function only onto a mesh that it loaded from the same
file, and the control is on a `Submesh` of the mantle. The restart
therefore uses one file for the meshes and the control:

- An inversion copies the input file into its run directory as `state.h5`
  and loads the meshes and fields from it.
- After each ROL iteration, the script copies the bed onto the mantle
  nodes on Re (`extend`), copies that field onto a CG2 field of the parent
  mesh with `Function.assign`, and appends it to `state.h5` as
  `bed_start_parent_cg2_m`. Both steps copy nodal values.
- A new run with `RESTART=state.h5` copies that file, loads the meshes and
  the last bed from it (`assign` to the mantle, then `restrict`), and
  starts ROL from that bed. The bed is the stored bed bit for bit. The
  first J of the new run equals the last J of the earlier run up to the
  reproducibility of the forward solve (5.7e-8 relative in the restart
  test).

The L-BFGS memory of the earlier run is lost. The script renames an earlier
`state.h5` and `controls.h5` to `state_previous<n>.h5` and
`controls_previous<n>.h5` before it writes new ones.

A state file of an earlier version of the script stores the bed as a broken
copy named `bed_start_m`. That version moved the bed onto the mantle with
`interpolate`, so the stored mantle values are the values that its solver
used. A restart from such a file reads them and continues from the bed that
the earlier solver used.

## Outputs

| file | content |
|---|---|
| `inputs_exp11.h5` | the input file (mesh, earth fields, observed sea level, slope of the ocean mask, ice of every step, record) |
| `inputs_exp11.json` | the record of the prepare job |
| `inputs_exp11_earth.pvd` | ParaView: the earth fields in SI units |
| `inputs_exp11_ice.pvd` | ParaView: the ice at the first, middle and last step, m |
| `controls.h5` | the bed of every ROL iteration on the surface, with J and the RMS misfit |
| `state.h5` | a copy of the input file with the bed of every ROL iteration appended (`bed_start_parent_cg2_m`, CG2 on the parent mesh, nonzero only on Re); the file of a restart |
| `summary_invert.json` | the settings, the forward steps, the Taylor test or the ROL iterations |
| `prepare_exp11.log`, `invert_exp11.log` | the logs |

## The forward run at the protocol steps

`forward_exp11.py --schedule protocol` runs the same forward solve at every
model year of the GIAMIP protocol: 500 yr steps from 122 ka to year -26000,
then 100 yr steps to year +1000. That is 462 steps, and year 0 is step 452.
The input file holds the ice of the 2 kyr steps only. The script reads the
ice of each step from giamip in the job, with `prepare_exp11.SurfaceForcing`,
the transfer of the prepare script. Before the march it makes the ice of the
62 years that the input file also holds and compares it with the input
file. The run stops if one differs by more than 1e-9 relative.

The bed was inverted with 2 kyr steps, so the bed at year 0 differs from the
observed bed by the change of the time-discretisation error. The log line
`MISFIT AT YEAR 0` and `checks.misfit_year0` and `checks.bed_misfit_year0` in
`summary_forward.json` give J, the RMS misfit on the solver's measure and
the nodal min, max and RMS of the bed minus the observed bed. The surface
file also holds `bed_observed_m` and `bed_misfit_year0_m`, saved once.

The preconditioner is rebuilt every 20 solves (`REBUILD_EVERY`) and at the
first solve after the change of dt at year -26000. The run has no restart.
At L0 on 3 nodes it takes about 3-6 h; the job asks for 10 h.

```bash
mkdir -p /scratch/pawsey0821/$USER/giamip/exp11/L0-3nodes/forward-protocol
cd /scratch/pawsey0821/$USER/giamip/exp11/L0-3nodes/forward-protocol
sbatch --export=NONE,STATE=../invert-full/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp11/run_forward_protocol.slurm
# After the forward job: the ParaView series (1 rank), then the copy to Gadi.
sbatch --dependency=afterok:<forward job> --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp11/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp11/run_archive_forward.slurm
```

The archive job copies the surface file, the logs, the summary, the job
outputs and `paraview/` to
`/g/data/xd2/sg8812/gia-mip/exp11/L0/forward-protocol/`.

## Changes for another experiment

A directory for experiment XX is a copy of this directory. Only these
changes are allowed:

1. Rename the files: `exp01_common.py` to `expXX_common.py`,
   `prepare_exp01.py` to `prepare_expXX.py`, `invert_exp01.py` to
   `invert_expXX.py`. Change the imports, the log and output file names,
   the job names and the paths in the Slurm scripts to match.
2. In `prepare_expXX.py`, set `EXPERIMENT` and `STEM`.
3. Change the text of the docstrings and of this README to the values of the
   experiment.
4. If the case file gives no topography file (Exp08, Exp11), map the
   G-ADOPT choice of the `gadopt` block to the file name.
5. Exp08 and Exp11 only: add the g-drift step to `earth_fields` in the
   prepare script. The inversion script does not change.

In this directory, against `../exp01/`:

1. The renames of item 1. The environment file `env-exp01-pr4638-pyop2fix.sh`
   and the variable `EXP01_MPLCONFIGDIR` keep their names, because they
   belong to the Setonix environment.
2. `EXPERIMENT = "Exp11"` and `STEM = "inputs_exp11"`.
3. The text of the docstrings and of this README.
4. `experiment_topography` in `prepare_exp11.py` opens the file of the
   G-ADOPT choice `bedmap3` through `giamip.cases.with_gadopt_choices`.
5. In `prepare_exp11.py`: the earth model `exp11_earth` (PREM-stable from
   `giamip.earth.ElasticProfile`, the 3-D viscosity `Viscosity3D` from
   g-drift), the viscosity block of `earth_fields`, the per-layer log10
   statistics and the `viscosity_3d` entry of the record. In
   `run_prepare.slurm`: `PYTHONPATH` gets
   `/software/projects/pawsey0821/sghelichkhani/python-extra-3d` (g-drift
   at 9273a8c, giamip at gia-mip 0cf5f39), and `GDRIFT_DATA_DIR` is set to
   `/software/projects/pawsey0821/sghelichkhani/gdrift-data`.

`exp11_common.py`, `invert_exp11.py` and `run_invert.slurm` differ from the
Exp01 files only by the renames.

The march length comes from the case file through `coarse_step_indices`.
An experiment that starts at 80 ka has 40 steps.

## On Gadi: the production on giamip_8re_r2

The production of 2026-10 runs on Gadi (NCI) on the mesh `giamip_8re_r2`:
an outer sphere at 8 Re and 2 rounds of local refinement near the coasts
and the ice margins, 4 957 665 cells. The decisions of this phase are in
`NOTES/team/r2-production/00-brief.md` of the worktree (not in the
repository). Exp11 runs on 6 `normalsr` nodes (624 ranks). The
input file holds a mesh partitioned for that rank count, so every solver
job of the experiment uses it.

| file | purpose |
|---|---|
| `run_prepare.pbs` | the input file (`prepare_exp11.py`) |
| `run_transfer.pbs` | the final L0 bed onto the r2 input file (`transfer_bed_exp11.py`) |
| `run_invert.pbs` | the inversion, with the watchdog and the node memory and jobfs samplers |
| `run_forward.pbs` | the forward run from the final bed, with checkpoints and a restart |
| `run_paraview.pbs` | the ParaView series of the forward run (1 rank) |
| `run_images.pbs` | the 4K images of the forward run, one half of the steps per job (project vo05) |
| `run_archive.pbs` | the records and the final state from /scratch to /g/data |
| `node_memory_sampler.sh`, `node_jobfs_sampler.sh`, `signal_ranks.sh` | the helpers of `run_invert.pbs` |
| `probe_partition.py` | what each part of the mantle-balanced partition of a mesh holds |
| `../submit_r2.sh` | submits the jobs of one stage for one experiment and node count |

The job scripts and helpers of this table (except `../submit_r2.sh`) and
`exp11_common.py`, `exp11_diagnostics.py`, `invert_exp11.py`,
`forward_exp11.py`, `paraview_exp11.py` and `transfer_bed_exp11.py` are
copies of the Exp01 files with `exp01`, `Exp01` and `EXP01` changed to
`exp11`, `Exp11` and `EXP11`, except `env-exp01-pr4638-pyop2fix.sh`,
`EXP01_MPLCONFIGDIR`, `EXP01_ALLOW_CORE_DUMPS`, `tests/giamip/exp01.py`,
`views_exp01_L1.json` and the Exp01 measurements in the comments. Do not
edit them here: a change goes into the Exp01 file and is copied out.

The march is the protocol steps, as in the final L0 run: the prepare job
gets `--step_years 100` (500 yr steps to year -26000, then 100 yr steps),
and the forward run `SCHEDULE=protocol`. The transfer writes the r2
observed bed plus the change of the final L0 inversion (`START=change`);
the source is
`/g/data/xd2/sg8812/gia-mip/exp11/L0/invert-protocol/state.h5`. The tape of
the inversion goes to `$PBS_JOBFS` of each node and nowhere else.

The steps, on a Gadi login node in a login shell, from this directory:

1. `bash ../submit_r2.sh 11 6 ready` submits the prepare job, the
   transfer and the 3-step test, each after the one before.
2. Read the records. Prepare: `prepare/inputs_exp11.json` (the cell counts
   against the sidecar, the statistics per layer, the ice volume check).
   Transfer: `transfer/inputs_exp11_transfer_bed.json` (the coverage, the
   RMS of the bed change against the L0 value, the largest distance of a
   target node from the source surface). 3-step test: J, the memory per
   node, and the jobfs lines at the end of the job output (the tape per
   taped step and its projection to the full march, which must fit in
   400 GB per node).
3. `bash ../submit_r2.sh 11 6 invert` submits the inversion
   (48 h wall time). It stops at an RMS misfit below 1 mm or after 10 ROL
   iterations.
4. If the inversion stops at the wall time, `bash ../submit_r2.sh 11
   6 restart` continues it with `RESTART=state.h5` in the same
   directory.
5. `bash ../submit_r2.sh 11 6 forward` submits the forward run, the
   ParaView job, the two image jobs and the archive job, each after the
   one before. If the forward run stops at the wall time (exit code 3),
   PBS deletes the jobs after it; `FORWARD_RESTART=1 bash ../submit_r2.sh
   11 6 forward` continues the run and submits them again.

| item | path |
|---|---|
| run directory | `/scratch/xd2/$USER/giamip/exp11/8re_r2-6nodes/` (`prepare/`, `transfer/`, `invert-3steps/`, `invert/`, `jobs_r2.txt`) |
| forward run, ParaView files, images | `/g/data/xd2/sg8812/gia-mip/exp11/r2/forward-protocol/` |
| archive of the records and the final state | `/g/data/xd2/sg8812/gia-mip/exp11/r2/` |

## Run log

| date | job | what | result |
|---|---|---|---|
| 2026-10-19 | 51123803 (work, 30 min limit; submitted to debug and moved to work in the queue) | prepare, L0, 2 nodes, 256 ranks, `/scratch/pawsey0821/sghelichkhani/giamip/exp11/L0-2nodes/prepare` | COMPLETED in 9 min 31 s, peak 1.71 GB per rank. 61 steps of 2 kyr (-122000 to 0). Ice volume on the mesh within 2.4e-3 of the grid. Dataset 0-2891 km, 201 layers, SHA-256 checked, g-drift 0.0.0+g9273a8c. log10 viscosity after the bound: 0-60 km 20.0 to 39.08 (mean 29.20), 120-400 km 20.0 to 30.49 (mean 20.55, 42.6 % at the bound), 2400-2891 km 21.0 to 30.39 (mean 22.05, 26.7 % at the bound). 11.6 % of the mantle volume at the lower bound. |
| 2026-10-19 | 51141583 (work, 3 nodes, 384 ranks, `--time=00:30:00`) | prepare, L0, `/scratch/pawsey0821/sghelichkhani/giamip/exp11/L0-3nodes/prepare` | COMPLETED in 15 min 30 s, peak 1.72 GB per rank. 61 steps, -122000 to 0. Largest ice volume error 2.35e-03. Distribution `mantle_balanced_384`, 1138-1139 mantle cells per rank, mesh MD5 checked. Rheology maxwell, compressible. Against the 2-node record (`L0-2nodes/prepare`): inputs, march, scales, solver constants and viscosity per shell are equal, the statistics agree to 4.5e-14 and the ice volume check to 7.5e-13 relative. The `earth_by_layer` means agree to 1e-15 (viscosity: 1.2e-14). The min and max of density, mu, K and gravity differ, because this run takes them at the nodes of an equispaced DG2 space, which include the vertices: the range is wider in every layer (mu 0-60 km min 7.4e9 Pa against 1.8e10 Pa). One exception is the gravity min in 1470-1950 km (9.93863 against 9.93826 m/s^2): gravity is CG2, and its minimum in that shell is between the nodes of both sets. `viscosity_3d` agrees with the 2-node record to 1e-15 relative. |
| 2026-10-19 | 51145299 (work, 3 nodes, 384 ranks) | full inversion, L0, to year 0, `/scratch/pawsey0821/sghelichkhani/giamip/exp11/L0-3nodes/invert-full` | COMPLETED in 6 h 29 min. Stopped by the RMS misfit rule after 8 ROL iterations: RMS bed misfit 32.36 m -> 0.7 mm. RMS change of the bed from the observed bed 35.75 m. Largest node 188-193 GB from iteration 1. Full history: `NOTES/L0-INVERSION-RESULTS-2026-10-20.md`. |
