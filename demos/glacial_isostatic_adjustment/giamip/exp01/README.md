# GIAMIP Exp01: the bed at 122 ka by adjoint inversion

This directory finds the bed at the start of GIAMIP experiment Exp01. The
run of the coupled self-gravitating sea-level solver must carry that bed to
the observed present-day bed at year 0. The inversion minimises the misfit
between the predicted bed at year 0 and the observed bed.

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. The other GIAMIP experiments copy this
directory and change only the items in "Changes for another experiment".

## The experiment

The values come from `giamip.mip.inputs.GiamipExperiment("Exp01")` and the
case file `giamip/resources/cases/giamip/exp01.json` of gia-mip.

| item | value |
|---|---|
| ice | `iceHistory-ICE-6G_D.nc` |
| topography | `bedtopo_Bedmap2ANT_BedMach5GRL_GEBCO_GLQ256.nc` (observed present-day bed) |
| start | 122 ka |
| march | 61 steps of 2 kyr to year 0 |
| viscosity | VM5a: elastic lid 0-60 km, 1.0e22 Pa s to 100 km, 5.0e20 Pa s to 670 km, 1.57e21 Pa s to 1260 km, 3.23e21 Pa s to the CMB |
| elastic structure and density | PREM-stable |
| rheology | Maxwell |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables, P2 potential |
| time stepping | backward Euler |

## The files

| file | purpose |
|---|---|
| `exp01_common.py` | code that both scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp01.py` | reads giamip and the mesh, and writes `inputs_exp01.h5` |
| `invert_exp01.py` | loads `inputs_exp01.h5`, tapes the march, and runs a Taylor test or the ROL inversion |
| `run_prepare.slurm` | the prepare job on Setonix |
| `run_invert.slurm` | the inversion job on Setonix |
| `forward_exp01.py` | a forward run from the bed of an inversion, with the surface fields and the GIAMIP scalars of every step |
| `paraview_exp01.py` | turns the surface file of the forward run into ParaView files |
| `run_forward.slurm` | the forward job on Setonix |
| `transfer_bed_exp01.py` | moves the bed of an inversion at one mesh level onto the input file of another level |
| `run_transfer.slurm` | the transfer job on Setonix |

`invert_exp01.py` does not import `giamip`, `gdrift`, `h5netcdf` or
`netCDF4`. The prepare script makes every field that comes from data. The
tape of the inversion holds only the operations from the control to the
misfit.

## The mesh

All GIAMIP experiments use the coarse mesh `n2_giamip_all_o1.msh` (MD5
`a4da84206dc82f5b458be66e4fd33844`) and its JSON sidecar. On Setonix the
mesh is in `/software/projects/pawsey0821/sghelichkhani/meshes/n2_giamip_all_o1/`.
The prepare script stops if the MD5 of the mesh file is different.

The mesh has spheres at 60, 90, 120, 400, 670, 980, 1260, 1470, 1950 and
2400 km depth, at the CMB and at the surface. For VM5a the jumps at 60, 670
and 1260 km are on spheres. The jump at 100 km is in the 90-120 km shell. A
cell takes the viscosity at its centroid, so the effective boundary is a
cell face near 100 km. The prepare record gives the volume fraction of each
viscosity value in each shell (`viscosity_by_layer`).

## How to run

The input file holds a mesh that is partitioned for the number of ranks of
the prepare job. The inversion job must use the same nodes and ranks.

1. Send the code to the Setonix checkout.
2. In a new run directory, submit the prepare job:

   ```bash
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_prepare.slurm
   ```

3. Read `inputs_exp01.json`. Compare the values with the table above, the
   statistics per layer, and the ice volume check against the grid.
4. In a second run directory, submit the Taylor test with 3 steps:

   ```bash
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp01.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_invert.slurm
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
`exp01_common.SurfaceNodeMap` pairs these nodes once, by position, and
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
| `inputs_exp01.h5` | the input file (mesh, earth fields, observed sea level, slope of the ocean mask, ice of every step, record) |
| `inputs_exp01.json` | the record of the prepare job |
| `inputs_exp01_earth.pvd` | ParaView: the earth fields in SI units |
| `inputs_exp01_ice.pvd` | ParaView: the ice at the first, middle and last step, m |
| `controls.h5` | the bed of every ROL iteration on the surface, with J and the RMS misfit |
| `state.h5` | a copy of the input file with the bed of every ROL iteration appended (`bed_start_parent_cg2_m`, CG2 on the parent mesh, nonzero only on Re); the file of a restart |
| `summary_invert.json` | the settings, the forward steps, the Taylor test or the ROL iterations |
| `prepare_exp01.log`, `invert_exp01.log` | the logs |

## The forward run and its ParaView files

`forward_exp01.py` runs the solver of the inversion forward, once, from the
bed of the last ROL iteration in a `state.h5`. It uses the solver options
and defaults of `invert_exp01.py` and the exact node map for `SL_init`, so
the run is the replay of the inversion at its last iterate. It marches the
61 steps of 2 kyr to year 0 and then one step of 1000 yr to year +1000
(3000 CE), the end of the protocol. The input file has no ice for +1000 yr,
so the script reads it from giamip with `prepare_exp01.SurfaceForcing`. It
tapes nothing. It must run on the rank count of the state file.

The script checks four things and writes them to the log and to
`summary_forward.json`:

1. The bed through the node map: `extend(restrict(stored))` equals the
   stored field and `restrict(extend(b))` equals `b`, both exactly. The min
   and max of the bed equal those of the last control in `controls.h5`.
   The script does not compare the integral, because on the rank count of
   the writer the surface submesh of `controls.h5` loads with a wrong
   distribution. On 1 and on 16 ranks the integral of the last control in
   `controls.h5` is -149474.13223, equal to the value of the state file.
   On the writer's 384 ranks the loaded surface has an area of 248.4
   instead of 61.0, so the integral is wrong there.
2. The forcing of giamip continues the march of the input file with one
   step, and that step is the last protocol step (+1000 yr).
3. The ice of year 0 and the observed bed, made again from giamip, equal
   the fields of the input file (a test of the transfer).
4. The misfit J at year 0 against the last J of the state file.

The outputs:

| file | content |
|---|---|
| `forward_exp01_surface.h5` | `CheckpointFile` on the surface submesh: the surface fields of every step in SI units (`idx` = step, with step, year, Shift and m1-m3), and the curved node positions `coordinates_m` once |
| `params_forward_exp01.log` | `ParameterLog`: one row of scalars per step (Shift, m1-m3, `mean_delta_g`, `grd_ice_mass`, `total_ice_mass`, `ocean_area`, `ocean_area_grdice`, `maf`) |
| `forward_exp01.log`, `summary_forward.json` | the log, the checks and every row |

The module docstring of `forward_exp01.py` gives the definition of every
field and every scalar. CG2 fields move to the surface through
`SurfaceNodeMap.restrict`. The masks of the solver (ocean function,
grounded and land ice) contain the facet area, so the file holds their mean
over each facet (DG0) from a facet integral.

`paraview_exp01.py` reads the surface file on any rank count other than
the writer's (1 is enough at L0) and writes two series with `time` = model year:
`forward_exp01_fields.pvd` (the continuous fields on the curved surface in
metres) and `forward_exp01_fractions.pvd` (the facet fractions).

```bash
sbatch --export=NONE,STATE=../invert-full-exactmap/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_forward.slurm
python3 paraview_exp01.py --input forward_exp01_surface.h5 --output_path paraview
```

For a short test, add `EXTRA="--steps 3 --plus_step always"`: the run then
does 3 steps of the march and the step of 1000 yr after them.

## The forward run at the protocol steps

`forward_exp01.py --schedule protocol` runs the same forward solve at every
model year of the GIAMIP protocol: 500 yr steps from 122 ka to year -26000,
then 100 yr steps to year +1000. That is 462 steps, and year 0 is step 452.
The input file holds the ice of the 2 kyr steps only. The script reads the
ice of each step from giamip in the job, with `prepare_exp01.SurfaceForcing`,
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
mkdir -p /scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/forward-protocol
cd /scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/forward-protocol
sbatch --export=NONE,STATE=../invert-full-exactmap/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_forward_protocol.slurm
# After the forward job: the ParaView series (1 rank), then the copy to Gadi.
sbatch --dependency=afterok:<forward job> --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_archive_forward.slurm
```

The archive job copies the surface file, the logs, the summary, the job
outputs and `paraview/` to
`/g/data/xd2/sg8812/gia-mip/exp01/L0/forward-protocol/`.

## The bed transfer between mesh levels and the L2 workflow

An inversion at a fine level (L1 or L2) starts from the bed of the
inversion at a coarser level, through the restart path of
`invert_exp01.py`. `invert_exp01.py` does not change.
`transfer_bed_exp01.py` writes the coarse bed into the input file of the
fine level, in the form that the restart reads.

The script loads the meshes of the target input file, on the rank count of
that file. It loads the parent mesh and the last `bed_start_parent_cg2_m`
entry of the source `state.h5` (or the `bed_start_m` entry of an earlier
version of the inversion). If the source was written on another rank count,
Firedrake partitions its mesh again with its default partitioner. The
script reads only the facets of Re of the source, so any partition works.
It cuts no `Submesh` from the source mesh.

For each target node on Re, the script takes the point of the curved
source surface on the ray from the centre through the node. It evaluates
the P2 interpolant of the source bed on that facet at that point. The
target nodes lie on the sphere, and the source P2 surface lies off the
sphere between its nodes by O(h^3). A point evaluation in the 3-D source
mesh (`VertexOnlyMesh`, `interpolate`) is not used. The bed on the parent
is zero at the nodes off Re, so a point off the source surface gets a value
that is too small by a fraction of the bed itself. In the tests of a
smooth bed, the RMS error of that point evaluation was about 1.8 times the
RMS error of the ray rule at both levels. The script gives these results:

- a target node on a source node (a vertex, or a source edge node, which
  giamsh puts on the sphere along the same ray) gets the source value to
  round-off;
- a smooth bed is reproduced at the other nodes to O(h^3);
- a target node that no source facet contains stops the script with an
  error. The script never writes a zero for a missing node.

The bed goes into the target file as entry `idx = 0` of
`bed_start_parent_cg2_m`, written by `exp01_common.append_control`. The
timestepping keys of the entry are `J` and `rms_misfit_m`, which are the
keys that `invert_exp01.py` appends. The values are those of the source
iterate on the source mesh. The dataset `/giamip_inputs/bed_transfer` in
the target file and the file `inputs_exp01_transfer_bed.json` next to it
record the source, the sizes and dates of both files, the SHA-256 of the
source, the number of target nodes on Re, the bed range, the values at the
coincident nodes, and these area-weighted RMS values over Re:

| key | meaning |
|---|---|
| `bed_minus_target_observed_m` | the transferred bed against the observed bed of the target |
| `bed_minus_source_observed_m` | the transferred bed against the transferred observed bed of the source: the change of the source inversion (12.33 m for the L0 bed of Exp01) |
| `source_observed_minus_target_observed_m` | the part of the observed bed that the source mesh does not resolve |

`--start` (`START` in the job script) selects the field that is written.
`change` (the default) writes the observed bed of the target plus the
change of the source inversion (source bed minus source observed bed),
both at the target nodes. `bed` writes the source bed. The observed bed of L1
resolves topography that L0 does not: for Exp01 the two observed beds
differ by 147 m RMS, and the L0 inversion changed its bed by 12.3 m RMS.
With `bed`, a finer inversion therefore starts with that missing
topography in its misfit. With `change`, it starts from the observed
topography of its own mesh plus the correction of the coarse inversion.
The L1 test of the transfer used `change`, and Exp01 uses `change`.

The script changes the target file. It refuses a target that already holds
a bed entry. `run_transfer.slurm` copies the target to `<target>.bak` in
the same directory before the run, and removes the copy only after the
script exits with 0, which includes its reload check. At L2 the copy needs
about 100 GB of disk for the time of the job. If the job fails, move the
`.bak` file back over the target before the next transfer.

The L2 workflow (decided 2026-10-20: L2 starts from the L0 bed, with no L1
inversion):

1. Prepare at L2 on the node count of the inversion, for example 32 nodes:
   `sbatch --nodes=32 --export=NONE,REFINE=2 .../run_prepare.slurm`. Before
   the prepare job, set `lfs setstripe -c 16 -S 4M` on the run directory.
2. Transfer the L0 bed on the same node count, with `START=change`. The
   observed bed of L2 differs from the L0 observed bed by at least the
   147 m RMS of L1. `change` starts the inversion 12.3 m RMS from its own
   observed bed, as at L0. `change` is the default of the job script, and
   the command gives it explicitly:

   ```bash
   sbatch --nodes=32 --export=NONE,START=change,TARGET=$PWD/inputs_exp01.h5,SOURCE=/scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/invert-full-exactmap/state.h5 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp01/run_transfer.slurm
   ```

   Read `inputs_exp01_transfer_bed.json`. `target_nodes_on_re` must be
   2121282 at L2 (530322 at L1). The coincident values must agree
   to round-off. `bed_minus_source_observed_m` must be near the RMS change
   of the source.
3. Start the inversion from the transferred bed:
   `sbatch --nodes=32 --export=NONE,INPUTS=../prepare/inputs_exp01.h5,RESTART=../prepare/inputs_exp01.h5,MODE=invert ... run_invert.slurm`.
   The restart path copies the target file to `state.h5`. Its iterates
   follow as `idx = 1, 2, ...`.

`MODE=taylor` ignores `RESTART`: `invert_exp01.py` uses the restart only in
`MODE=invert`. A Taylor test therefore starts from the observed bed. A
short test that starts from the transferred bed is
`MODE=invert,STEPS=3,ITERATIONS=1` with `RESTART`.

## Changes for another experiment

A directory for experiment XX is a copy of this directory. Only these
changes are allowed:

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
repository). Exp01 runs on 6 `normalsr` nodes (624 ranks). The
input file holds a mesh partitioned for that rank count, so every solver
job of the experiment uses it.

| file | purpose |
|---|---|
| `run_prepare.pbs` | the input file (`prepare_exp01.py`) |
| `run_transfer.pbs` | the final L0 bed onto the r2 input file (`transfer_bed_exp01.py`) |
| `run_invert.pbs` | the inversion, with the watchdog and the node memory and jobfs samplers |
| `run_forward.pbs` | the forward run from the final bed, with checkpoints and a restart |
| `run_paraview.pbs` | the ParaView series of the forward run (1 rank) |
| `run_images.pbs` | the 4K images of the forward run, one part of the steps per job (project vo05) |
| `run_archive.pbs` | the records and the final state from /scratch to /g/data |
| `node_memory_sampler.sh`, `node_jobfs_sampler.sh`, `signal_ranks.sh` | the helpers of `run_invert.pbs` |
| `probe_partition.py` | what each part of the mantle-balanced partition of a mesh holds |
| `../submit_r2.sh` | submits the jobs of one stage for one experiment and node count |

The job scripts and helpers of this table (except `../submit_r2.sh`) and
`exp01_common.py`, `exp01_diagnostics.py`, `invert_exp01.py`,
`forward_exp01.py`, `paraview_exp01.py` and `transfer_bed_exp01.py` are the
masters of the copies in `../exp02/` .. `../exp12/`: each copy is the Exp01
file with `exp01`, `Exp01` and `EXP01` changed to the experiment, except
`env-exp01-pr4638-pyop2fix.sh`, `EXP01_MPLCONFIGDIR`,
`EXP01_ALLOW_CORE_DUMPS`, `tests/giamip/exp01.py`, `views_exp01_L1.json`
and the Exp01 measurements in the comments. A change goes into the Exp01
file and is copied out.

The march is the protocol steps, as in the final L0 run: the prepare job
gets `--step_years 100` (500 yr steps to year -26000, then 100 yr steps),
and the forward run `SCHEDULE=protocol`. The transfer writes the r2
observed bed plus the change of the final L0 inversion (`START=change`);
the source is
`/g/data/xd2/sg8812/gia-mip/exp01/L0/invert-protocol/state.h5`. The tape of
the inversion goes to `$PBS_JOBFS` of each node and nowhere else.

The steps, on a Gadi login node in a login shell, from this directory:

1. `bash ../submit_r2.sh 01 6 ready` submits the prepare job, the
   transfer and the 3-step test, each after the one before.
2. Read the records. Prepare: `prepare/inputs_exp01.json` (the cell counts
   against the sidecar, the statistics per layer, the ice volume check).
   Transfer: `transfer/inputs_exp01_transfer_bed.json` (the coverage, the
   RMS of the bed change against the L0 value, the largest distance of a
   target node from the source surface). 3-step test: J, the memory per
   node, and the jobfs lines at the end of the job output (the tape per
   taped step and its projection to the full march, which must fit in
   400 GB per node).
3. `bash ../submit_r2.sh 01 6 invert` submits the inversion
   (48 h wall time). It stops at an RMS misfit below 1 mm or after 10 ROL
   iterations.
4. If the inversion stops at the wall time, `bash ../submit_r2.sh 01
   6 restart` continues it with `RESTART=state.h5` in the same
   directory.
5. `bash ../submit_r2.sh 01 6 forward` submits the forward run, the
   ParaView job, the image jobs (`IMAGE_PARTS`, default 2) and the archive
   job, each after the one before. If the forward run stops at the wall
   time (exit code 3), PBS deletes the jobs after it;
   `FORWARD_RESTART=1 bash ../submit_r2.sh 01 6 forward` continues
   the run and submits them again. If the inversion ran with fewer ranks
   per node (`RANKS_PER_NODE`), give the forward stage the same
   `RANKS_PER_NODE` and `PREP`, and `FORWARD_NODES` for full nodes: the
   forward run has no tape, so for example the 624 ranks of a 16-node
   inversion with 39 ranks per node run on 6 full nodes
   (`RANKS_PER_NODE=39 FORWARD_NODES=6`).

| item | path |
|---|---|
| run directory | `/scratch/xd2/$USER/giamip/exp01/8re_r2-6nodes/` (`prepare/`, `transfer/`, `invert-3steps/`, `invert/`, `jobs_r2.txt`) |
| forward run, ParaView files, images | `/g/data/xd2/sg8812/gia-mip/exp01/r2/forward-protocol/` |
| archive of the records and the final state | `/g/data/xd2/sg8812/gia-mip/exp01/r2/` |

## Run log

| date | job | what | result |
|---|---|---|---|
| 2026-10-19 | 51131967 | Taylor test, 3 steps, exact node map | replay 1.3e-15 relative; R1 rates 1.989, 1.994, 1.996, 1.997, 1.998 |
| 2026-10-19 | 51131968, 51131969 | restart pair, 3 steps, 2 + 1 iterations | restored bed equal bit for bit; first J 10.968244923 against last J 10.968244302 (5.7e-8 relative, the forward solve) |
| 2026-10-19 | 51135637 | restart from a state file of the earlier version | first J 10.969422547 against 10.969421918 of that run (5.7e-8 relative) |
| 2026-10-19 | 51141575 (work, 3 nodes, 384 ranks, `--time=00:30:00`) | prepare, L0, `/scratch/pawsey0821/sghelichkhani/giamip/exp01/L0-3nodes/prepare` | COMPLETED in 6 min 35 s, peak 0.86 GB per rank. 61 steps, -122000 to 0. Largest ice volume error 1.29e-03. Distribution `mantle_balanced_384`, 1138-1139 mantle cells per rank, mesh MD5 checked. Rheology maxwell, compressible. Against the 2-node record (`L0-2nodes/prepare`): inputs, march, scales, solver constants and viscosity per shell are equal, the statistics agree to 4.5e-14 and the ice volume check to 8.9e-13 relative. The `earth_by_layer` means agree to 1e-15. The min and max of density, mu, K and gravity differ, because this run takes them at the nodes of an equispaced DG2 space, which include the vertices: the range is wider in every layer (mu 0-60 km min 7.4e9 Pa against 1.8e10 Pa). One exception is the gravity min in 1470-1950 km (9.93863 against 9.93826 m/s^2): gravity is CG2, and its minimum in that shell is between the nodes of both sets. The record of job 51123593 in this directory is kept as `inputs_exp01_job51123593.json` and `prepare_exp01_job51123593.log`. |
| 2026-10-19 | 51142261 (work, 3 nodes, 384 ranks) | full inversion, L0, to year 0, `/scratch/pawsey0821/sghelichkhani/giamip/exp01/L0-3nodes/invert-full-exactmap` | COMPLETED in 5 h 14 min. Stopped by the RMS misfit rule after 6 ROL iterations: RMS bed misfit 12.35 m -> 0.3 mm. RMS change of the bed from the observed bed 12.33 m. Largest node 188-193 GB from iteration 1. Full history: `NOTES/L0-INVERSION-RESULTS-2026-10-20.md`. |
| 2026-10-19 | 51126689 (work, 3 nodes, 384 ranks) | full inversion, L0, with the old point-search bed map, `/scratch/pawsey0821/sghelichkhani/giamip/exp01/L0-3nodes/invert-full` | COMPLETED in 5 h 13 min, 6 ROL iterations, RMS misfit 12.85 m -> 0.3 mm. Kept as the test of memory and time on 3 nodes only; its control is not the bed the solver used, so it is not a result (use 51142261). |
