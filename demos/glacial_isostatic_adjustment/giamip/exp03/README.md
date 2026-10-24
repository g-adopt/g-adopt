# GIAMIP Exp03: the bed at 122 ka by adjoint inversion

This directory finds the bed at the start of GIAMIP experiment Exp03. The
run of the coupled self-gravitating sea-level solver must carry that bed to
the observed present-day bed at year 0. The inversion minimises the misfit
between the predicted bed at year 0 and the observed bed.

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. It is a copy of the Exp01 directory
(`../exp01/`) with only the changes in "Changes for another experiment".

## The experiment

The values come from `giamip.mip.inputs.GiamipExperiment("Exp03")` and the
case file `giamip/resources/cases/giamip/exp03.json` of gia-mip.

| item | value |
|---|---|
| ice | `iceHistory-GLAC3c_profile1.nc` |
| topography | `bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc` (observed present-day bed) |
| start | 122 ka |
| march | 61 steps of 2 kyr to year 0 |
| viscosity | midpoint of the GIAMIP suggested range (`giamip-open-range-midpoint`, G-ADOPT choice): elastic lid 0-90 km, 10^20.5 = 3.16e20 Pa s to 670 km, 1.0e22 Pa s to the CMB |
| elastic structure and density | PREM-stable (G-ADOPT choice; the protocol leaves it open) |
| rheology | Maxwell (G-ADOPT choice, `gadopt` block of the case file; the protocol leaves it open) |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables, P2 potential |
| time stepping | backward Euler |

## The files

| file | purpose |
|---|---|
| `exp03_common.py` | code that both scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp03.py` | reads giamip and the mesh, and writes `inputs_exp03.h5` |
| `invert_exp03.py` | loads `inputs_exp03.h5`, tapes the march, and runs a Taylor test or the ROL inversion |
| `run_prepare.slurm` | the prepare job on Setonix |
| `run_invert.slurm` | the inversion job on Setonix |

`invert_exp03.py` does not import `giamip`, `gdrift`, `h5netcdf` or
`netCDF4`. The prepare script makes every field that comes from data. The
tape of the inversion holds only the operations from the control to the
misfit.

## The mesh

All GIAMIP experiments use the coarse mesh `n2_giamip_all_o1.msh` (MD5
`a4da84206dc82f5b458be66e4fd33844`) and its JSON sidecar. On Setonix the
mesh is in `/software/projects/pawsey0821/sghelichkhani/meshes/n2_giamip_all_o1/`.
The prepare script stops if the MD5 of the mesh file is different.

The mesh has spheres at 60, 90, 120, 400, 670, 980, 1260, 1470, 1950 and
2400 km depth, at the CMB and at the surface. For the midpoint profile
the base of the elastic lid at 90 km and the jump at 670 km are on spheres,
so every viscosity boundary is exact. The prepare record gives the volume
fraction of each viscosity value in each shell (`viscosity_by_layer`). Each
shell must hold one value.

## How to run

The input file holds a mesh that is partitioned for the number of ranks of
the prepare job. The inversion job must use the same nodes and ranks.

1. Send the code to the Setonix checkout.
2. In a new run directory, submit the prepare job:

   ```bash
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp03/run_prepare.slurm
   ```

3. Read `inputs_exp03.json`. Compare the values with the table above, the
   statistics per layer, and the ice volume check against the grid.
4. In a second run directory, submit the Taylor test with 3 steps:

   ```bash
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp03.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp03/run_invert.slurm
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
`exp03_common.SurfaceNodeMap` pairs these nodes once, by position, and
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
| `inputs_exp03.h5` | the input file (mesh, earth fields, observed sea level, slope of the ocean mask, ice of every step, record) |
| `inputs_exp03.json` | the record of the prepare job |
| `inputs_exp03_earth.pvd` | ParaView: the earth fields in SI units |
| `inputs_exp03_ice.pvd` | ParaView: the ice at the first, middle and last step, m |
| `controls.h5` | the bed of every ROL iteration on the surface, with J and the RMS misfit |
| `state.h5` | a copy of the input file with the bed of every ROL iteration appended (`bed_start_parent_cg2_m`, CG2 on the parent mesh, nonzero only on Re); the file of a restart |
| `summary_invert.json` | the settings, the forward steps, the Taylor test or the ROL iterations |
| `prepare_exp03.log`, `invert_exp03.log` | the logs |

## The forward run at the protocol steps

`forward_exp03.py --schedule protocol` runs the same forward solve at every
model year of the GIAMIP protocol: 500 yr steps from 122 ka to year -26000,
then 100 yr steps to year +1000. That is 462 steps, and year 0 is step 452.
The input file holds the ice of the 2 kyr steps only. The script reads the
ice of each step from giamip in the job, with `prepare_exp03.SurfaceForcing`,
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
mkdir -p /scratch/pawsey0821/$USER/giamip/exp03/L0-3nodes/forward-protocol
cd /scratch/pawsey0821/$USER/giamip/exp03/L0-3nodes/forward-protocol
sbatch --export=NONE,STATE=../invert-full/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp03/run_forward_protocol.slurm
# After the forward job: the ParaView series (1 rank), then the copy to Gadi.
sbatch --dependency=afterok:<forward job> --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp03/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp03/run_archive_forward.slurm
```

The archive job copies the surface file, the logs, the summary, the job
outputs and `paraview/` to
`/g/data/xd2/sg8812/gia-mip/exp03/L0/forward-protocol/`.

## Changes for another experiment

A directory for experiment XX is a copy of the Exp01 directory
(`../exp01/`). Only these changes are allowed:

1. Rename the files: `exp01_common.py` to `expXX_common.py`,
   `prepare_exp01.py` to `prepare_expXX.py`, `invert_exp01.py` to
   `invert_expXX.py`. Change the imports, the log and output file names,
   the job names and the paths in the Slurm scripts to match. Do not rename
   `env-exp01-pr4638-pyop2fix.sh` and `EXP01_MPLCONFIGDIR`: they are the
   shared Setonix environment.
2. In `prepare_expXX.py`, set `EXPERIMENT` and `STEM`.
3. Change the text of the docstrings and of this README to the values of the
   experiment.
4. If the case file gives no topography file (Exp08, Exp11), map the
   G-ADOPT choice of the `gadopt` block to the file name.
5. Exp08 and Exp11 only: add the g-drift step to `earth_fields` in the
   prepare script. The inversion script does not change.

The march length comes from the case file through `coarse_step_indices`.
An experiment that starts at 80 ka has 40 steps.

## On Gadi: the production on giamip_8re_r2

The production of 2026-10 runs on Gadi (NCI) on the mesh `giamip_8re_r2`:
an outer sphere at 8 Re and 2 rounds of local refinement near the coasts
and the ice margins, 4 957 665 cells. The decisions of this phase are in
`NOTES/team/r2-production/00-brief.md` of the worktree (not in the
repository). Exp03 runs on 6 `normalsr` nodes (624 ranks). The
input file holds a mesh partitioned for that rank count, so every solver
job of the experiment uses it.

| file | purpose |
|---|---|
| `run_prepare.pbs` | the input file (`prepare_exp03.py`) |
| `run_transfer.pbs` | the final L0 bed onto the r2 input file (`transfer_bed_exp03.py`) |
| `run_invert.pbs` | the inversion, with the watchdog and the node memory and jobfs samplers |
| `run_forward.pbs` | the forward run from the final bed, with checkpoints and a restart |
| `run_paraview.pbs` | the ParaView series of the forward run (1 rank) |
| `run_images.pbs` | the 4K images of the forward run, one part of the steps per job (project vo05) |
| `run_archive.pbs` | the records and the final state from /scratch to /g/data |
| `node_memory_sampler.sh`, `node_jobfs_sampler.sh`, `signal_ranks.sh` | the helpers of `run_invert.pbs` |
| `probe_partition.py` | what each part of the mantle-balanced partition of a mesh holds |
| `../submit_r2.sh` | submits the jobs of one stage for one experiment and node count |

The job scripts and helpers of this table (except `../submit_r2.sh`) and
`exp03_common.py`, `exp03_diagnostics.py`, `invert_exp03.py`,
`forward_exp03.py`, `paraview_exp03.py` and `transfer_bed_exp03.py` are
copies of the Exp01 files with `exp01`, `Exp01` and `EXP01` changed to
`exp03`, `Exp03` and `EXP03`, except `env-exp01-pr4638-pyop2fix.sh`,
`EXP01_MPLCONFIGDIR`, `EXP01_ALLOW_CORE_DUMPS`, `tests/giamip/exp01.py`,
`views_exp01_L1.json` and the Exp01 measurements in the comments. Do not
edit them here: a change goes into the Exp01 file and is copied out.

The march is the protocol steps, as in the final L0 run: the prepare job
gets `--step_years 100` (500 yr steps to year -26000, then 100 yr steps),
and the forward run `SCHEDULE=protocol`. The transfer writes the r2
observed bed plus the change of the final L0 inversion (`START=change`);
the source is
`/g/data/xd2/sg8812/gia-mip/exp03/L0/invert-protocol/state.h5`. The tape of
the inversion goes to `$PBS_JOBFS` of each node and nowhere else.

The steps, on a Gadi login node in a login shell, from this directory:

1. `bash ../submit_r2.sh 03 6 ready` submits the prepare job, the
   transfer and the 3-step test, each after the one before.
2. Read the records. Prepare: `prepare/inputs_exp03.json` (the cell counts
   against the sidecar, the statistics per layer, the ice volume check).
   Transfer: `transfer/inputs_exp03_transfer_bed.json` (the coverage, the
   RMS of the bed change against the L0 value, the largest distance of a
   target node from the source surface). 3-step test: J, the memory per
   node, and the jobfs lines at the end of the job output (the tape per
   taped step and its projection to the full march, which must fit in
   400 GB per node).
3. `bash ../submit_r2.sh 03 6 invert` submits the inversion
   (48 h wall time). It stops at an RMS misfit below 1 mm or after 10 ROL
   iterations.
4. If the inversion stops at the wall time, `bash ../submit_r2.sh 03
   6 restart` continues it with `RESTART=state.h5` in the same
   directory.
5. `bash ../submit_r2.sh 03 6 forward` submits the forward run, the
   ParaView job, the image jobs (`IMAGE_PARTS`, default 2) and the archive
   job, each after the one before. If the forward run stops at the wall
   time (exit code 3), PBS deletes the jobs after it;
   `FORWARD_RESTART=1 bash ../submit_r2.sh 03 6 forward` continues
   the run and submits them again. If the inversion ran with fewer ranks
   per node (`RANKS_PER_NODE`), give the forward stage the same
   `RANKS_PER_NODE` and `PREP`, and `FORWARD_NODES` for full nodes: the
   forward run has no tape, so for example the 624 ranks of a 16-node
   inversion with 39 ranks per node run on 6 full nodes
   (`RANKS_PER_NODE=39 FORWARD_NODES=6`). The restart and forward stages
   find the run directory of the inversion under `/scratch/xd2/$USER/giamip`
   or `/g/data/xd2/sg8812/giamip-runs` (the one that holds
   `invert/state.h5`; `ROOT` overrides it), so their dry run needs Gadi.

| item | path |
|---|---|
| run directory | `/scratch/xd2/$USER/giamip/exp03/8re_r2-6nodes/` (`prepare/`, `transfer/`, `invert-3steps/`, `invert/`, `jobs_r2.txt`) |
| forward run, ParaView files, images | `/g/data/xd2/sg8812/gia-mip/exp03/r2/forward-protocol/` |
| archive of the records and the final state | `/g/data/xd2/sg8812/gia-mip/exp03/r2/` |

## Run log

| date | job | what | result |
|---|---|---|---|
| 2026-10-19 | 51123490 (work, 2 nodes, 256 ranks) | prepare, L0 | COMPLETED in 8 min 45 s, peak memory 0.85 GB (record `peak_memory_gb`). 61 steps, -122000 to 0. Each shell has one viscosity value: 1e40 Pa s to 90 km, 3.16e20 Pa s to 670 km, 1e22 Pa s to the CMB. Largest ice volume error 3.41e-3. |
| 2026-10-19 | 51141577 (work, 3 nodes, 384 ranks, `--time=00:30:00`) | prepare, L0, `/scratch/pawsey0821/sghelichkhani/giamip/exp03/L0-3nodes/prepare` | COMPLETED in 6 min 02 s, peak 0.85 GB per rank. 61 steps, -122000 to 0. Largest ice volume error 3.41e-03. Distribution `mantle_balanced_384`, 1138-1139 mantle cells per rank, mesh MD5 checked. Rheology maxwell, compressible. Against the 2-node record (`L0-2nodes/prepare`): inputs, march, scales, solver constants and viscosity per shell are equal, the statistics agree to 4.5e-14 and the ice volume check to 4.5e-13 relative. The `earth_by_layer` means agree to 1e-15. The min and max of density, mu, K and gravity differ, because this run takes them at the nodes of an equispaced DG2 space, which include the vertices: the range is wider in every layer (mu 0-60 km min 7.4e9 Pa against 1.8e10 Pa). One exception is the gravity min in 1470-1950 km (9.93863 against 9.93826 m/s^2): gravity is CG2, and its minimum in that shell is between the nodes of both sets. |
| 2026-10-19 | 51144129 (work, 3 nodes, 384 ranks) | full inversion, L0, to year 0, `/scratch/pawsey0821/sghelichkhani/giamip/exp03/L0-3nodes/invert-full` | COMPLETED in 6 h 57 min. Stopped by the RMS misfit rule after 9 ROL iterations: RMS bed misfit 26.82 m -> 0.3 mm. RMS change of the bed from the observed bed 28.15 m. Largest node 188-193 GB from iteration 1. Full history: `NOTES/L0-INVERSION-RESULTS-2026-10-20.md`. |
