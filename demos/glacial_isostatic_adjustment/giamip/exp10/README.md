# GIAMIP Exp10: the bed at 122 ka by adjoint inversion, and the forward run

This directory finds the bed at the start of GIAMIP experiment Exp10 and
runs the forward model from it. The run of the coupled self-gravitating
sea-level solver must carry that bed to the observed present-day bed at
year 0. The inversion minimises the misfit between the predicted bed at
year 0 and the observed bed.

Exp10 is Exp04 (same ice, topography, viscosity and mesh) with a power-law
rheology. The design is in the private handout
`NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-2026-10-20.md` (decisions S3,
S18-S21, S24, S25, S27, S29; sections 7, 8 and 12).

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. The shared files (`exp10_common.py`,
`invert_exp10.py`, `forward_exp10.py`, `paraview_exp10.py`,
`transfer_bed_exp10.py` and the job scripts of the forward run, ParaView,
the archive and the transfer) are the Exp01 files (`../exp01/`) with the
names changed. The prepare script, this README, `run_prepare.slurm` and
`run_invert.slurm` come from `../exp04/` with the changes in "Changes for
Exp10".

## The experiment

The values come from `giamip.mip.inputs.GiamipExperiment("Exp10")` and the
case file `giamip/resources/cases/giamip/exp10.json` of gia-mip.

| item | value |
|---|---|
| ice | `iceHistory-GLAC3c_profile2.nc` |
| topography | `bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc` (observed present-day bed). The case file leaves it open; this is the file of the G-ADOPT choice `bedmap3`, the file of Exp04 |
| start | 122 ka |
| march | 61 steps of 2 kyr to year 0 |
| viscosity | midpoint of the GIAMIP suggested range (`giamip-open-range-midpoint`, G-ADOPT choice): elastic lid 0-90 km, 10^20.5 = 3.16e20 Pa s to 670 km, 1.0e22 Pa s to the CMB |
| elastic structure and density | PREM-stable (G-ADOPT choice; the protocol leaves it open) |
| rheology | power law with one Maxwell element (G-ADOPT choice `power-law`, `gadopt` block of the case file; the protocol leaves it open). See "The power law" |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables, P2 potential |
| time stepping | backward Euler |

## The power law

`gadopt` (`InternalVariableApproximation.power_law_factor`, Kang et al.
2021, Eq. 12) multiplies the Maxwell time of the element by

    f = (1 + (sigma_b / sigma_T)^(n-1)) / (1 + ((sigma + sigma_b) / sigma_T)^(n-1))

with sigma = sqrt(J2) of the deviatoric stress. With sigma_b = 0 the
effective viscosity is eta_0 / (1 + (sigma / sigma_T)^(n-1)).

| item | value | decision |
|---|---|---|
| n | 3.5 in the power-law cells, exactly 1.0 elsewhere | P13 (Kang et al. 2021) |
| sigma_T | 1 MPa in every cell (non-dimensional 1e-5), also where n = 1 | P14 |
| sigma_b | 0 | P15 |
| stress measure | sqrt(J2) | P19 |
| region | 90 km < depth < 670 km and eta_0 < 1e22 Pa s, at the cell centroid | S21 |

For the midpoint profile the region is the whole upper mantle below the
elastic lid: the shells 90-120, 120-400 and 400-670 km. The lid and the
lower mantle are linear. The prepare script writes the DG0 fields
`exponent` and `transition_stress` and the record entry `power_law`. The
common file reads the two fields when the record has that entry, snaps the
exponent to exactly 1.0 or 3.5, and builds the solver with the power law
(Newton, GMRES on block 0, uncondensed layout). The record gives the
exponent per shell (`exponent_by_layer`) and the power-law volume fraction
per shell (`power_law.power_law_volume_fraction_by_layer`).

Expected effect (handout 7.3, sigma_b = 0): f = 1.6 at 0.8 MPa and f = 17
at 3 MPa (under the LGM ice). The Maxwell time of the upper mantle, about
140 yr, then falls to about 9 yr under the largest stresses.

## The files

| file | purpose |
|---|---|
| `exp10_common.py` | code that all scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp10.py` | reads giamip and the mesh, and writes `inputs_exp10.h5` with the power-law fields |
| `transfer_bed_exp10.py` | appends the Exp04 L0 bed to the input file (the warm start) |
| `invert_exp10.py` | loads `inputs_exp10.h5`, tapes the march, and runs a Taylor test or the ROL inversion |
| `forward_exp10.py` | the forward run from the inverted bed (`--schedule fine`, `fine_b`, `protocol` or `inversion`) |
| `paraview_exp10.py` | the ParaView series from the surface file of a forward run |
| `run_prepare.slurm`, `run_transfer.slurm`, `run_invert.slurm` | the jobs of the inversion on Setonix, 6 nodes |
| `run_forward_fine.slurm` | the forward run at the fine steps, with checkpoints and resubmission |
| `run_paraview.slurm`, `run_archive_forward.slurm` | ParaView (1 rank) and the copy to Gadi |
| `run_forward_protocol.slurm` | the forward run at the protocol steps (not used for Exp10) |

`invert_exp10.py` does not import `giamip`, `gdrift`, `h5netcdf` or
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

Every job of Exp10 runs on 6 nodes, 768 ranks (S24): the input file holds a
mesh partitioned for the ranks of the prepare job, and every later job must
use the same ranks. Run directories are under
`/scratch/pawsey0821/$USER/giamip/exp10/L0-6nodes/`. The debug partition
has too few nodes; every job runs on `work`.

1. Send the code to the Setonix checkout (`NOTES/sync_setonix.sh`) and
   compare the MD5 sums.
2. The prepare job:

   ```bash
   mkdir -p /scratch/pawsey0821/$USER/giamip/exp10/L0-6nodes/prepare
   cd /scratch/pawsey0821/$USER/giamip/exp10/L0-6nodes/prepare
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_prepare.slurm
   ```

   Read `inputs_exp10.json`: the values of the tables above, the viscosity
   and the exponent per shell, the ice volume check against the grid.
3. The warm start (S20, P20): append the bed of the Exp04 L0 inversion
   (same mesh, `START=bed`) to the input file. The source is another
   experiment, so the script needs `--allow_other_experiment`:

   ```bash
   sbatch --export=NONE,TARGET=$PWD/inputs_exp10.h5,SOURCE=/scratch/pawsey0821/$USER/giamip/exp04/L0-3nodes/invert-full/state.h5,START=bed,EXTRA=--allow_other_experiment \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_transfer.slurm
   ```

4. The Taylor test with 3 steps. The Taylor mode starts from the observed
   bed (it reads no restart bed):

   ```bash
   mkdir -p ../taylor-3steps && cd ../taylor-3steps
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp10.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_invert.slurm
   ```

   The R1 rates must be near 2. Read the node memory (`sacct` MaxRSS, and
   the memory lines of the log): above about 210 GB per node, stop.
5. The inversion with 3 steps and 3 iterations, from the transferred bed
   (`INPUTS=../prepare/inputs_exp10.h5,MODE=invert,STEPS=3,ITERATIONS=3,RESTART=../prepare/inputs_exp10.h5`). J must decrease at every iteration.
   Watch the Newton counts of the power law in the log.
6. The full inversion
   (`INPUTS=../prepare/inputs_exp10.h5,MODE=invert,RESTART=../prepare/inputs_exp10.h5`),
   in `invert-full/`: at most 10 ROL iterations, stop at 1 mm RMS. After
   the first ROL iteration, project the wall time; above 2 days, stop
   (S29).
7. If the job stops at the time limit, submit it again in the same run
   directory with `RESTART=state.h5`.

## The control and the node map

The control is the bed at the start of the run, in metres, a CG2 field on
the surface Re. The surface is a codim-1 `Submesh` of the mantle. Its
geometry is P1 (flat triangles), because a submesh takes its coordinates
from the DMPlex of the mantle, which holds only the vertices. The mantle
has P2 geometry. The solver reads the initial sea level `SL_init`, a CG2
field on the mantle, on the facets of Re only.

The CG2 nodes of the surface are the CG2 nodes of the mantle on Re.
`exp10_common.SurfaceNodeMap` pairs these nodes once, by position, and
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
| `inputs_exp10.h5` | the input file (mesh, earth fields, observed sea level, slope of the ocean mask, ice of every step, record) |
| `inputs_exp10.json` | the record of the prepare job |
| `inputs_exp10_earth.pvd` | ParaView: the earth fields in SI units |
| `inputs_exp10_ice.pvd` | ParaView: the ice at the first, middle and last step, m |
| `controls.h5` | the bed of every ROL iteration on the surface, with J and the RMS misfit |
| `state.h5` | a copy of the input file with the bed of every ROL iteration appended (`bed_start_parent_cg2_m`, CG2 on the parent mesh, nonzero only on Re); the file of a restart |
| `summary_invert.json` | the settings, the forward steps, the Taylor test or the ROL iterations |
| `prepare_exp10.log`, `invert_exp10.log` | the logs |

## The forward run at the fine steps

`forward_exp10.py --schedule fine` (schedule A, S19): 50 yr steps from
122 ka to year -26000, then 10 yr steps to year +1000, 4620 steps. The ice
is linear in time between the protocol years. The surface fields and the
parameter log are written at the protocol years only (S12). The
preconditioner is rebuilt every 100 solves (P17), at the first solve after
the change of dt, and when a solve takes more than 6 outer iterations per
Newton iteration. The run writes a checkpoint into its copy of the state
file and resubmits itself before the 24 h limit (`run_forward_fine.slurm`).

```bash
mkdir -p /scratch/pawsey0821/$USER/giamip/exp10/L0-6nodes/forward-fine
cd /scratch/pawsey0821/$USER/giamip/exp10/L0-6nodes/forward-fine
sbatch --export=NONE,STATE=../invert-full/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_forward_fine.slurm
# After the last leg: the ParaView series (1 rank), then the copy to Gadi.
sbatch --dependency=afterok:<last leg> --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE,DEST=/g/data/xd2/sg8812/gia-mip/exp10/L0/forward-fine \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp10/run_archive_forward.slurm
```

Rule S25: after the first 50 steps, compare the time per step with a
Maxwell step on the same mesh and nodes. If the ratio is above 2, stop the
run and start it again with `SCHEDULE=fine_b` (schedule B: 100 yr, then
25 yr, and 10 yr from 15 to 14 ka and from 11 to 10 ka). Watch the Newton
and outer counts in the first fast melt window (GLAC3c profile 2: 14.6 to
14.1 ka and 10.4 to 9.9 ka).

## Changes for Exp10

Against `../exp04/` (prepare, README, `run_prepare.slurm`,
`run_invert.slurm`) and `../exp01/` (the shared files):

1. The names: `exp04` and `exp01` to `exp10`, `Exp04` and `Exp01` to
   `Exp10`, except `env-exp01-pr4638-pyop2fix.sh` and `EXP01_MPLCONFIGDIR`
   (the shared Setonix environment).
2. `prepare_exp10.py`: `EXPERIMENT`, `STEM`; the topography of the G-ADOPT
   choice (`experiment_topography`, as in Exp11), because the case file
   leaves it open; the rheology check accepts `power-law` only; the
   power-law fields (`power_law_fields`), their statistics per shell and
   the record entry `power_law`.
3. The job scripts of the prepare, the transfer and the inversion: 6 nodes;
   the prepare job asks for 2 h.
4. The shared files are not changed. A change goes into the Exp01 file and
   is copied out.

## Run log

| date | job | what | result |
|---|---|---|---|
