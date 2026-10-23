# GIAMIP Exp08: the bed at 80 ka by adjoint inversion

This directory finds the bed at the start of GIAMIP experiment Exp08. The
run of the coupled self-gravitating sea-level solver must carry that bed to
the observed present-day bed at year 0. The inversion minimises the misfit
between the predicted bed at year 0 and the observed bed.

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. It is a copy of the Exp01 directory
with only the changes in "Changes for another experiment" (items 1 to 5).

## The experiment

Exp08 is Exp05 with a laterally varying elastic structure. The protocol
leaves the earth and the topography open. The values come from
`giamip.mip.inputs.GiamipExperiment("Exp08")`, the case file
`giamip/resources/cases/giamip/exp08.json` and its `gadopt` block, and the
model files `prem-glad-m25.json` and `gowan2021.json` of gia-mip (0cf5f39).

| item | value |
|---|---|
| ice | `iceHistory-PaleoMIST_1a.nc` |
| topography | `bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc` (observed present-day bed; G-ADOPT choice `bedmach3`) |
| start | 80 ka |
| march | 40 steps of 2 kyr to year 0 |
| viscosity | Gowan et al. (2021) (G-ADOPT choice): elastic lid 0-120 km, 4e20 Pa s to 670 km, 4e22 Pa s to the CMB |
| density | PREM-stable |
| shear and bulk modulus | PREM-stable with the GLAD-M25 relative change of vs and vp (G-ADOPT choice `prem-glad-m25`, see below) |
| rheology | Maxwell (G-ADOPT choice) |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables, P2 potential |
| time stepping | backward Euler |

Exp08 and Exp05 differ only in the shear and bulk modulus. The density, the
viscosity, gravity, the observed sea level, the slope of the ocean mask and
the ice are the same.

## The GLAD-M25 elastic structure

The model file `prem-glad-m25.json` names the g-drift dataset
`3d_seismic_GLAD-M25` (SHA-256
`b9b37f2628c42c4ed5d09f94ed1bd44a5f4d701a8d18ce1efa53e24a0c5a3f95`), its
fields `vs` and `vp` (Voigt averages), and the rule. The prepare script
reads these items from the model file. It stops if the rule texts of the
file differ from the texts that the code implements.

The rule (Sia Ghelichkhani, 2026-10-18):

1. At each depth of the dataset, `dlnvs = vs / <vs>(r) - 1` and
   `dlnvp = vp / <vp>(r) - 1`. `<.>(r)` is the plain mean over the points
   of that depth. g-drift stores each depth on the same equal-area point
   set, so the plain mean is the mean over the sphere.
2. The relative change goes to the DG1 nodes by the layered interpolation
   of `gdrift.EarthModel3D`: inverse distance squared over the 8 nearest
   directions on the two depths around the node, then linear in radius.
   The dataset covers 6 to 2891 km. Above 6 km the 6 km values hold.
3. With P the PREM-stable value at the node,
   `mu = mu_P (1 + dlnvs)^2` and
   `K = rho_P (vp_P^2 (1 + dlnvp)^2 - 4/3 vs_P^2 (1 + dlnvs)^2)`.
4. The density stays PREM-stable, so the reference state stays hydrostatic.

The relative change is lateral at each depth. The jumps of GLAD-M25 (650 km,
and between 405 and 410 km) do not enter mu and K. The jumps of mu and K at
400 and 670 km are those of PREM-stable, on mesh spheres.

The G-ADOPT choice of 2026-10-19 (Sia Ghelichkhan): mu and K in the
0-60 km shell are the values of the GLAD-M25 rule at the cell centroid, so
that the linear DG1 interpolant does not extrapolate the steep crustal
changes of GLAD-M25 to negative or near-zero values at the cell vertices.
The DG1 nodes are interior points of each cell, and the linear field of a
cell extrapolates from them to its vertices. The prepare script does these
steps:

1. Make `f_dg0`, the rule at the centroid of every cell, and `f_dg1`, the
   rule at the DG1 nodes, for mu and for K.
2. Interpolate `conditional(r_centroid > R_60, f_dg0, f_dg1)` into the DG1
   space. `R_60` is the radius of the 60 km mesh sphere, read from the mesh
   sidecar (`radial.radius("d60")`). No cell crosses that sphere, so mu and
   K are constant per cell in 0-60 km and linear below.
3. Stop unless mu and K are positive at every centroid in 0-60 km, at
   every vertex below 60 km, and at every vertex of the written fields.

The record holds the rule, `R_60`, the number of cells of each rule and the
vertex min and max of mu and K per layer (`moduli_lid_rule`). It also holds
the per-layer min, max and mean of `dlnvs`, `dlnvp`, mu and K
(`earth_by_layer`), and the per-depth check near 400 and 670 km
(`tomography.sphere_check`). The min and max of `earth_by_layer` are taken
at the nodes of an equispaced DG2 space, which include the vertices of each
cell. The `dlnvs` and `dlnvp` fields of `earth_by_layer` and of the ParaView
file are the values at the DG1 nodes everywhere, also in 0-60 km, where the
written mu and K use the centroid values.

Only rank 0 reads the dataset. The other ranks send their DG1 node
positions to rank 0 one rank at a time.

## The files

| file | purpose |
|---|---|
| `exp08_common.py` | code that both scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp08.py` | reads giamip, g-drift and the mesh, and writes `inputs_exp08.h5` |
| `invert_exp08.py` | loads `inputs_exp08.h5`, tapes the march, and runs a Taylor test or the ROL inversion |
| `run_prepare.slurm` | the prepare job on Setonix |
| `run_invert.slurm` | the inversion job on Setonix |

`invert_exp08.py` does not import `giamip`, `gdrift`, `h5netcdf` or
`netCDF4`. The prepare script makes every field that comes from data. The
tape of the inversion holds only the operations from the control to the
misfit.

## The mesh

All GIAMIP experiments use the coarse mesh `n2_giamip_all_o1.msh` (MD5
`a4da84206dc82f5b458be66e4fd33844`) and its JSON sidecar. On Setonix the
mesh is in `/software/projects/pawsey0821/sghelichkhani/meshes/n2_giamip_all_o1/`.
The prepare script stops if the MD5 of the mesh file is different.

The mesh has spheres at 60, 90, 120, 400, 670, 980, 1260, 1470, 1950 and
2400 km depth, at the CMB and at the surface. For the Gowan viscosity the
jumps at 120 and 670 km are on spheres, so no cell crosses a viscosity
boundary. The prepare record gives the volume fraction of each viscosity
value in each shell (`viscosity_by_layer`).

## Setonix environment

`run_prepare.slurm` puts `/software/projects/pawsey0821/sghelichkhani/python-extra-3d`
first on `PYTHONPATH`. It holds g-drift (commit 9273a8c) and giamip at
gia-mip 0cf5f39 with the model files. `GDRIFT_DATA_DIR` is
`/software/projects/pawsey0821/sghelichkhani/gdrift-data`, which holds the
GLAD-M25 file. The compute nodes have no internet access, so the file must
be there before the job starts. `run_invert.slurm` needs neither.

## How to run

The input file holds a mesh that is partitioned for the number of ranks of
the prepare job. The inversion job must use the same nodes and ranks.

1. Send the code to the Setonix checkout.
2. In a new run directory, submit the prepare job:

   ```bash
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp08/run_prepare.slurm
   ```

3. Read `inputs_exp08.json`. Compare the values with the table above, the
   statistics per layer, and the ice volume check against the grid.
4. In a second run directory, submit the Taylor test with 3 steps:

   ```bash
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp08.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp08/run_invert.slurm
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
`exp08_common.SurfaceNodeMap` pairs these nodes once, by position, and
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
| `inputs_exp08.h5` | the input file (mesh, earth fields, observed sea level, slope of the ocean mask, ice of every step, record) |
| `inputs_exp08.json` | the record of the prepare job |
| `inputs_exp08_earth.pvd` | ParaView: the earth fields in SI units, with `dlnvs` and `dlnvp` |
| `inputs_exp08_ice.pvd` | ParaView: the ice at the first, middle and last step, m |
| `controls.h5` | the bed of every ROL iteration on the surface, with J and the RMS misfit |
| `state.h5` | a copy of the input file with the bed of every ROL iteration appended (`bed_start_parent_cg2_m`, CG2 on the parent mesh, nonzero only on Re); the file of a restart |
| `summary_invert.json` | the settings, the forward steps, the Taylor test or the ROL iterations |
| `prepare_exp08.log`, `invert_exp08.log` | the logs |

## The forward run at the protocol steps

`forward_exp08.py --schedule protocol` runs the same forward solve at every
model year of the GIAMIP protocol: 500 yr steps from 80 ka to year -26000,
then 100 yr steps to year +1000. That is 378 steps, and year 0 is step 368.
The input file holds the ice of the 2 kyr steps only. The script reads the
ice of each step from giamip in the job, with `prepare_exp08.SurfaceForcing`,
the transfer of the prepare script. Before the march it makes the ice of the
41 years that the input file also holds and compares it with the input
file. The run stops if one differs by more than 1e-9 relative.

The bed was inverted with 2 kyr steps, so the bed at year 0 differs from the
observed bed by the change of the time-discretisation error. The log line
`MISFIT AT YEAR 0` and `checks.misfit_year0` and `checks.bed_misfit_year0` in
`summary_forward.json` give J, the RMS misfit on the solver's measure and
the nodal min, max and RMS of the bed minus the observed bed. The surface
file also holds `bed_observed_m` and `bed_misfit_year0_m`, saved once.

The preconditioner is rebuilt every 20 solves (`REBUILD_EVERY`) and at the
first solve after the change of dt at year -26000. The run has no restart.
At L0 on 3 nodes it takes about 2.5-5 h; the job asks for 8 h.

```bash
mkdir -p /scratch/pawsey0821/$USER/giamip/exp08/L0-3nodes/forward-protocol
cd /scratch/pawsey0821/$USER/giamip/exp08/L0-3nodes/forward-protocol
sbatch --export=NONE,STATE=../invert-full/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp08/run_forward_protocol.slurm
# After the forward job: the ParaView series (1 rank), then the copy to Gadi.
sbatch --dependency=afterok:<forward job> --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp08/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp08/run_archive_forward.slurm
```

The archive job copies the surface file, the logs, the summary, the job
outputs and `paraview/` to
`/g/data/xd2/sg8812/gia-mip/exp08/L0/forward-protocol/`.

## Changes for another experiment

This directory is a copy of the Exp01 directory. Only these changes against
Exp01 are allowed:

1. Rename the files: `exp01_common.py` to `exp08_common.py`,
   `prepare_exp01.py` to `prepare_exp08.py`, `invert_exp01.py` to
   `invert_exp08.py`. Change the imports, the log and output file names,
   the job names and the paths in the Slurm scripts to match. The names of
   the Setonix environment (`env-exp01-pr4638-pyop2fix.sh` and
   `EXP01_MPLCONFIGDIR`) stay, because they name the shared environment.
2. In `prepare_exp08.py`, set `EXPERIMENT` and `STEM`.
3. Change the text of the docstrings and of this README to the values of the
   experiment.
4. The case file gives no topography file, so `topography_of` in the
   prepare script maps the G-ADOPT choice of the `gadopt` block
   (`bedmach3`) to the file name.
5. The g-drift step in `earth_fields` of the prepare script, with the
   centroid rule for the moduli in the 0-60 km shell and the 1D
   earth built from the giamip model files (`radial_earth`), because
   `GiamipExperiment("Exp08").earth()` raises `NotImplementedError` for
   `prem-glad-m25`. `run_prepare.slurm` adds the g-drift and giamip folder
   to `PYTHONPATH` and sets `GDRIFT_DATA_DIR`. The inversion script and
   `exp08_common.py` do not change.

The march length comes from the case file through `coarse_step_indices`.
An experiment that starts at 80 ka has 40 steps.

## Run log

| date | job | what | result |
|---|---|---|---|
| 2026-10-19 | 51124425 (work, 2 nodes, 256 ranks, `--time=00:30:00`) | prepare, L0, `/scratch/pawsey0821/sghelichkhani/giamip/exp08/L0-2nodes/prepare` | COMPLETED in 8 min 12 s, peak 2.83 GB per rank. 40 steps -80000 to 0. Viscosity per shell: 1e40 Pa s (lid) 0-120 km, 4e20 Pa s 120-670 km, 4e22 Pa s 670 km to the CMB, 100 % in each shell. GLAD-M25 sha256 checked, g-drift 0.0.0+g9273a8c. mu and K positive at every DG1 node (min 3.8e9 and 1.1e10 Pa). The vertex values of the linear DG1 field are negative in some 0-60 km cells (mu down to -4.2e9 Pa, K down to -3.1e9 Pa). Against Exp05 (job 51124434): density, viscosity, gravity, SL_obs, slope and ice are identical; only mu and K differ. Job 51124394 (debug) was cancelled before it started. |
| 2026-10-19 | 51125405 (work, 2 nodes, 256 ranks, `--time=00:30:00`) | prepare, L0, same run directory, with the cell-mean rule for the moduli, the rheology stop and the new coastline text | COMPLETED in 10 min 16 s, peak 2.83 GB per rank. 27 869 of 437 028 mantle cells take the cell mean of mu and K, all in 0-60 km (centroid depth 15 to 45 km). After the change, mu and K are positive at every vertex. The smallest vertex values are mu 2.85e5 Pa and K 9.56e5 Pa (0-60 km), and mu 3.9e10 Pa and K 8.4e10 Pa below 60 km. Rheology maxwell, compressible. Against Exp05 (job 51124434): density, viscosity, gravity, SL_obs, slope, ice, march, scales, solver constants, inputs and mesh are identical; only mu and K differ. |
| 2026-10-19 | 51127675 (work, 2 nodes, 256 ranks, `--time=00:30:00`) | prepare, L0, same run directory, with the centroid rule for mu and K in every 0-60 km cell | COMPLETED in 5 min 08 s, peak 2.83 GB per rank. 151 272 cells (all cells of 0-60 km) take mu and K at the centroid, 285 756 cells the DG1 rule. Smallest centroid values in 0-60 km: mu 1.43e10 Pa, K 2.76e10 Pa. Smallest vertex values of the written fields: mu 1.43e10 Pa and K 2.76e10 Pa (0-60 km), mu 3.94e10 Pa and K 8.45e10 Pa (60-90 km). The script on Setonix had the default DG2 nodes, so the `earth_by_layer` min and max of this record are interior-node values. This record replaces the record of job 51125405. |
| 2026-10-19 | 51141582 (work, 3 nodes, 384 ranks, `--time=00:30:00`) | prepare, L0, `/scratch/pawsey0821/sghelichkhani/giamip/exp08/L0-3nodes/prepare` | COMPLETED in 14 min 31 s, peak 2.83 GB per rank. 40 steps, -80000 to 0. Largest ice volume error 1.69e-03. Distribution `mantle_balanced_384`, 1138-1139 mantle cells per rank, mesh MD5 checked. Rheology maxwell, compressible. Against the 2-node record (`L0-2nodes/prepare`): inputs, march, scales, solver constants and viscosity per shell are equal, the statistics agree to 4.5e-14 and the ice volume check to 1.2e-11 relative. The `earth_by_layer` means agree to 1e-15 (`dlnvs` and `dlnvp`: 4e-12). The min and max of density, mu, K and gravity differ, because this run takes them at the nodes of an equispaced DG2 space, which include the vertices: the range is wider in every layer except mu and K in 0-60 km. One exception is the gravity min in 1470-1950 km (9.93863 against 9.93826 m/s^2): gravity is CG2, and its minimum in that shell is between the nodes of both sets. `moduli_lid_rule` and `tomography` are equal to the 2-node record (151 272 centroid-rule cells, 285 756 DG1-rule cells). In 0-60 km mu and K are constant per cell, so their min and max are equal too. Below 60 km the min and max are now vertex values (mu 60-90 km min 3.94e10 Pa against 4.45e10 Pa), and the `dlnvs` range in 0-60 km is -0.83 to 0.52 (2-node: -0.70 to 0.27). |
| 2026-10-19 | 51144667 (work, 3 nodes, 384 ranks) | full inversion, L0, to year 0, `/scratch/pawsey0821/sghelichkhani/giamip/exp08/L0-3nodes/invert-full` | COMPLETED in 3 h 06 min. Stopped by the RMS misfit rule after 6 ROL iterations: RMS bed misfit 24.73 m -> 0.4 mm. RMS change of the bed from the observed bed 24.21 m. Largest node 188-193 GB from iteration 1. Full history: `NOTES/L0-INVERSION-RESULTS-2026-10-20.md`. |
