# GIAMIP Exp12: the bed at 122 ka by adjoint inversion, and the forward run

This directory finds the bed at the start of GIAMIP experiment Exp12 and
runs the forward model from it. The run of the coupled self-gravitating
sea-level solver must carry that bed to the observed present-day bed at
year 0. The inversion minimises the misfit between the predicted bed at
year 0 and the observed bed.

Exp12 combines the 3-D viscosity of Exp11, the GLAD-M25 elastic structure
of Exp08, the Extended Burgers (EB) rheology of Exp09 and the power law of
Exp10. It is the strongest combination of the G-ADOPT runs and has no
Maxwell control run (S23). The design is in the private handout
`NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-2026-10-20.md` (decisions S6,
S10, S18-S25, S29, S31, S32; sections 5-8 and 12).

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. It reads the EB fit file
`../eb_fit/eb_fit.json`. The shared files (`exp12_common.py`,
`invert_exp12.py`, `forward_exp12.py`, `paraview_exp12.py`,
`transfer_bed_exp12.py` and the job scripts of the forward run, ParaView,
the archive and the transfer) are the Exp01 files (`../exp01/`) with the
names changed. The prepare script combines `../exp11/prepare_exp11.py`
and `../exp08/prepare_exp08.py` and adds the EB elements, the drop rule
and the power law (see "Changes for Exp12").

## The experiment

The values come from `giamip.mip.inputs.GiamipExperiment("Exp12")`, the
case file `giamip/resources/cases/giamip/exp12.json` and the model files
`lloyd2024.json` and `prem-glad-m25.json` of gia-mip (0cf5f39 or later). The
protocol leaves the earth model and the topography open. The values marked
"G-ADOPT choice" come from the `gadopt` block of the case file.

| item | value |
|---|---|
| ice | `iceHistory-ICE-6G_D.nc` |
| topography | `bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc` (observed present-day bed; G-ADOPT choice `bedmap3`) |
| start | 122 ka |
| march | 61 steps of 2 kyr to year 0 (inversion); 4620 steps (forward run, schedule A) |
| viscosity eta_0 | Lloyd et al. (2024), 3-D (G-ADOPT choice `lloyd2024`), as Exp11: lower bound 1e20 Pa s to 670 km and 1e21 Pa s below, on log10 at the cell centroid; no upper bound; no elastic lid |
| elastic structure and density | `prem-glad-m25` (G-ADOPT choice), as Exp08: PREM-stable density, PREM-stable mu and K scaled by the GLAD-M25 relative change of vs and vp; centroid values in 0-60 km |
| rheology | `burgers-power-law` (G-ADOPT choice): EB with N = 3 Maxwell elements below 90 km, one Maxwell element above, and the power law. See "The rheology" |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables (N = 3), P2 potential |
| time stepping | backward Euler |

## The rheology

### The EB elements

The normalised EB fit is `free_time_fits/3` of `../eb_fit/eb_fit.json`
(S31: N = 3, free element times, largest modulus error 2.1 % over the fit
band). EB parameters: alpha 0.274, Delta 1.9, tau_H / tau_M = 1, tau_L = 0
(S22, P2-P4).

| element j | tau_bar_j (units of tau_M) | mu_bar_j (units of mu_inf) |
|---|---|---|
| 0 | 0.01488 | 0.1935 |
| 1 | 0.1895 | 0.1773 |
| 2 | 2.933 | 0.3285 |

The sum of mu_bar_j is 0.699 (the instant modulus of the fit) and the sum
of mu_bar_j tau_bar_j is 1 (the long-term viscosity is eta_0). In each cell
below 90 km, element j has mu_j = mu_inf mu_bar_j (DG1) and eta_j = eta_0
mu_bar_j tau_bar_j (DG0), with the local tau_M = eta_0 / mu_inf. Above
90 km (S10) the cell has one Maxwell element with the local tau_M in N
parts: mu_j = mu_inf w_j and eta_j = eta_0 w_j, w_j = mu_bar_j / sum mu_bar.
Exp12 has no elastic lid, so these cells keep the Lloyd viscosity.

### The drop rule (S32)

The gate test of the shared code found that Newton stalls with the power
law when the fastest element is 5 times faster than the step, and converges
at 2.5 times. At the Lloyd lower bound the fastest element has a Maxwell
time of about 0.7 yr, 15 times shorter than the 10 yr step. The rule:

- Element j is dropped in a cell below 90 km where 10 yr / tau_j > 2, that
  is tau_j < 5 yr. tau_j is the shortest Maxwell time of the element in the
  cell (eta_j over the largest mu_j at the cell vertices). 10 yr is the
  shortest step of every schedule (S18), so one set of fields serves the
  inversion and the forward run.
- The slowest element (j = 2) is never dropped.
- A dropped element is fully relaxed: it carries no elastic stress. Its
  viscous part mu_bar_j tau_bar_j is added to the slowest element, so the
  sum constraint holds and the long-term viscosity stays eta_0 in every
  cell. The slot of a dropped element carries an equal share of the slowest
  element at the same time, which is the same as one element with the full
  modulus.
- Element 0 is dropped where tau_M < 336 yr (at mu_inf = 70 GPa, eta_0
  below about 7e20 Pa s). Element 1 is dropped where tau_M < 26 yr. The
  instant modulus of the cell is then 0.506 or 0.329 mu_inf instead of
  0.699 mu_inf.

The prepare record (`burgers.drop_rule`, `burgers.by_layer`) gives the
volume fraction of the mantle and of each shell where each element is
dropped, and the instant modulus with and without the drop. The values of
the L0 run are in the run log below.

The rule tests the linear Maxwell times. In a power-law cell the factor f
divides every Maxwell time at run time, so a kept element can become faster
than the limit of the rule under a large stress. At the lower bound of the
viscosity, element 1 has a Maxwell time of about 9 yr: it reaches the
5 yr limit at f of about 2 and the ratio at which the gate stalled
(dt / tau = 5) at f of about 5. The record entry
`burgers.drop_rule.power_law_effective_times` gives, for f = 2, 5 and 17
(17 is the factor at 3 MPa, design handout 7.3), the volume fraction of the
mantle where this occurs. These numbers are the prior for the Newton watch
of the forward run. The rule does not use f: a rule on the effective times
would drop element 1 in most of the upper mantle and remove most of the
transient response that Exp12 is to show.

The option `--eb_drop_max_ratio` of the prepare script changes the limit
(`inf` keeps every element). A new value needs a new prepare job and a new
inversion.

### The power law

`gadopt` (`InternalVariableApproximation.power_law_factor`, Kang et al.
2021, Eq. 12) multiplies the Maxwell time of every element by

    f = (1 + (sigma_b / sigma_T)^(n-1)) / (1 + ((sigma + sigma_b) / sigma_T)^(n-1))

with sigma = sqrt(J2) of the deviatoric stress. The factor therefore shifts
the whole EB spectrum in a power-law cell.

| item | value | decision |
|---|---|---|
| n | 3.5 in the power-law cells, exactly 1.0 elsewhere | P13 |
| sigma_T | 1 MPa in every cell (non-dimensional 1e-5), also where n = 1 | P14 |
| sigma_b | 0 | P15 |
| stress measure | sqrt(J2) | P19 |
| region | 90 km < depth < 670 km and eta_0 < 1e22 Pa s, at the cell centroid | S21 |

The viscosity is 3-D, so the region is a set of cells and not whole shells.
The record gives the power-law volume fraction per shell
(`power_law.power_law_volume_fraction_by_layer`).

## The files

| file | purpose |
|---|---|
| `exp12_common.py` | code that all scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp12.py` | reads giamip, g-drift (Lloyd, GLAD-M25), the EB fit and the mesh, and writes `inputs_exp12.h5` |
| `transfer_bed_exp12.py` | appends the Exp01 L0 bed change to the input file (the warm start) |
| `invert_exp12.py` | loads `inputs_exp12.h5`, tapes the march, and runs a Taylor test or the ROL inversion |
| `forward_exp12.py` | the forward run from the inverted bed (`--schedule fine`, `fine_b`, `protocol` or `inversion`) |
| `paraview_exp12.py` | the ParaView series from the surface file of a forward run |
| `run_prepare.slurm`, `run_transfer.slurm`, `run_invert.slurm` | the jobs of the inversion on Setonix, 6 nodes |
| `run_forward_fine.slurm` | the forward run at the fine steps, with checkpoints and resubmission |
| `run_paraview.slurm`, `run_archive_forward.slurm` | ParaView (1 rank) and the copy to Gadi |

`invert_exp12.py` does not import `giamip`, `gdrift`, `h5netcdf` or
`netCDF4`. The forward run imports `prepare_exp12.py` for the ice between
the protocol years, so it runs in the environment of the prepare job.

## The mesh

All GIAMIP experiments use the coarse mesh `n2_giamip_all_o1.msh` (MD5
`a4da84206dc82f5b458be66e4fd33844`) and its JSON sidecar. On Setonix the
mesh is in `/software/projects/pawsey0821/sghelichkhani/meshes/n2_giamip_all_o1/`.
The prepare script stops if the MD5 of the mesh file is different. The
spheres at 60 km (moduli rule), 90 km (EB and power-law limit) and 670 km
(lower bound of the viscosity, power-law limit) are mesh spheres, so no
cell crosses a limit of a rule.

## How to run

The final L0 run (2026-10-21/22) is schedule C (the protocol steps plus 25
yr and 10 yr steps in the fast melt windows of the ice history, decisions
S33-S35). The steps below are the inversion at 2 kyr steps, whose bed is
the start of schedule C. The schedule-C chain
(`NOTES/team/schedule-c/submit_chain_c.sh` of the worktree, not in the
repository) submits each job after the one before: (a) the prepare job with
`EXTRA="--schedule windows"` in `L0-6nodes/prepare-c/`; (b) the transfer of
the converged 2 kyr bed of Exp12 (`invert-full/state.h5`, `START=bed`: same
mesh, same experiment) into that input file; (c) the inversion on schedule
C in `invert-c/` on 12 nodes of 64 ranks (`--nodes=12 --ntasks=768`) with
`EXTRA="--gia_rebuild_every 100 --log_memory --drop_block_adjoint_copy"`
and the tape on /scratch; (d) the forward run with `SCHEDULE=windows`
(`run_forward_fine.slurm`) in `forward-windows/`, a gate that passes only a
finished march, the ParaView job and the archive to
`/g/data/xd2/sg8812/gia-mip/exp12/L0/forward-windows/`. The inversion on
schedule C ended at 0.9 mm RMS after 3 ROL iterations. The production on
the r2 mesh is in the section "On Gadi: the production on giamip_8re_r2".

Every job of Exp12 runs on 6 nodes, 768 ranks (S24): the input file holds a
mesh partitioned for the ranks of the prepare job, and every later job must
use the same ranks. Run directories are under
`/scratch/pawsey0821/$USER/giamip/exp12/L0-6nodes/`. Every job runs on the
`work` partition.

1. Send the code to the Setonix checkout and compare the MD5 sums.
2. The prepare job:

   ```bash
   mkdir -p /scratch/pawsey0821/$USER/giamip/exp12/L0-6nodes/prepare
   cd /scratch/pawsey0821/$USER/giamip/exp12/L0-6nodes/prepare
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_prepare.slurm
   ```

   Read `inputs_exp12.json`: the viscosity and moduli per shell, the drop
   rule (`burgers.drop_rule`), the power-law volume fractions, the ice
   volume check against the grid.
3. The warm start (S20, P20): append the Exp01 L0 bed with `START=change`
   (the Exp12 observed bed plus the change that the Exp01 inversion found):

   ```bash
   sbatch --export=NONE,TARGET=$PWD/inputs_exp12.h5,SOURCE=/scratch/pawsey0821/$USER/giamip/exp01/L0-3nodes/invert-full-exactmap/state.h5,START=change,EXTRA=--allow_other_experiment \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_transfer.slurm
   ```

   The transfer stops for a source of another experiment unless
   `--allow_other_experiment` is given.

4. The Taylor test with 3 steps:

   ```bash
   mkdir -p ../taylor-3steps && cd ../taylor-3steps
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp12.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_invert.slurm
   ```

   The R1 rates must be near 2. Read the node memory (`sacct` MaxRSS, and
   the memory lines of the log): above about 210 GB per node, stop.
5. The inversion with 3 steps and 3 iterations, from the transferred bed
   (`INPUTS=../prepare/inputs_exp12.h5,MODE=invert,STEPS=3,ITERATIONS=3,RESTART=../prepare/inputs_exp12.h5`).
   J must decrease at every iteration. Watch the Newton counts.
6. The full inversion
   (`INPUTS=../prepare/inputs_exp12.h5,MODE=invert,RESTART=../prepare/inputs_exp12.h5`),
   in `invert-full/`: at most 10 ROL iterations, stop at 1 mm RMS. After
   the first ROL iteration, project the wall time; above 2 days, stop
   (S29).
7. If the job stops at the time limit, submit it again in the same run
   directory with `RESTART=state.h5`.

For the serial check of the drop rule on the real fields, run the prepare
script on one rank with `--earth_only` (`EXTRA=--earth_only`, `--nodes=1
--ntasks-per-node=1`). It writes `inputs_exp12_earth_only.json` and no
input file.

## The control, the stop rule and the restart

These are the same as in Exp01 (`../exp01/README.md`): the control is the
bed at the start of the run on the surface `Submesh`, moved with
`exp12_common.SurfaceNodeMap`; the inversion stops after 10 ROL iterations
or at 1 mm RMS bed misfit; a restart reads the last bed of `state.h5`.

## Outputs

| file | content |
|---|---|
| `inputs_exp12.h5` | the input file (mesh, earth fields, the N element fields `shear_moduli` and `viscosities`, `exponent`, `transition_stress`, observed sea level, slope of the ocean mask, ice of every step, record) |
| `inputs_exp12.json` | the record of the prepare job |
| `inputs_exp12_earth.pvd` | ParaView: the earth fields in SI units, dlnvs, dlnvp, the instant modulus ratio, the number of dropped elements, the exponent |
| `inputs_exp12_ice.pvd` | ParaView: the ice at the first, middle and last step, m |
| `controls.h5`, `state.h5`, `summary_invert.json`, `invert_exp12.log` | the outputs of the inversion |
| `forward_exp12_surface.h5`, `params_forward_exp12.log`, `forward_exp12.log`, `summary_forward.json`, `paraview/` | the outputs of the forward run |

## The forward run at the fine steps

Schedule A below was the plan before schedule C. The final forward run is
`--schedule windows` (step (d) of the schedule-C chain in "How to run"); it
uses the same restart, output and preconditioner rules as the fine
schedules.

`forward_exp12.py --schedule fine` (schedule A, S19): 50 yr steps from
122 ka to year -26000, then 10 yr steps to year +1000, 4620 steps. The ice
is linear in time between the protocol years. The surface fields and the
parameter log are written at the protocol years only (S12). The
preconditioner is rebuilt every 100 solves (P17), at the first solve after
the change of dt, and when a solve takes more than 6 outer iterations per
Newton iteration. The run writes a checkpoint into its copy of the state
file and resubmits itself before the 24 h limit.

```bash
mkdir -p /scratch/pawsey0821/$USER/giamip/exp12/L0-6nodes/forward-fine
cd /scratch/pawsey0821/$USER/giamip/exp12/L0-6nodes/forward-fine
sbatch --export=NONE,STATE=../invert-full/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_forward_fine.slurm
# After the last leg: the ParaView series (1 rank), then the copy to Gadi.
sbatch --dependency=afterok:<last leg> --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp12/run_archive_forward.slurm
```

The archive goes to `/g/data/xd2/sg8812/gia-mip/exp12/L0/forward-fine/`.

Rule S25: after the first 50 steps, compare the time per step with a
Maxwell step on the same mesh and nodes. If the ratio is above 2, stop the
run and start it again with `SCHEDULE=fine_b` (schedule B: 100 yr, then
25 yr, and 10 yr from 15 to 13 ka). Watch the Newton and outer counts in
the fast melt window of ICE-6G_D (MWP-1A, 14.5 to 14.0 ka). If Newton
fails, report at once.

## Changes for Exp12

Against `../exp11/` and `../exp08/` (prepare, README, slurm files) and
`../exp01/` (the shared files):

1. The names: `exp11` and `exp01` to `exp12`, `Exp11` and `Exp01` to
   `Exp12`, except `env-exp01-pr4638-pyop2fix.sh` and `EXP01_MPLCONFIGDIR`
   (the shared Setonix environment).
2. `prepare_exp12.py`: `EXPERIMENT`, `STEM`; the rheology check accepts
   `burgers-power-law` only; the viscosity of `prepare_exp11.py`
   (`Viscosity3D`, `viscosity_fields`); the GLAD-M25 moduli of
   `prepare_exp08.py` (`tomography_model_file`, `RelativeChange`,
   `elastic_fields`); the EB elements with the drop rule (`load_eb_fit`,
   `drop_fast_elements`, `burgers_elements`, `burgers_by_layer`); the
   power-law fields (`power_law_fields`); the record entries `n_elements`,
   `burgers`, `power_law`; the option `--earth_only`.
3. The job scripts: 6 nodes; `run_forward_protocol.slurm` is not in this
   directory (Exp12 runs the fine schedule only); the archive goes to
   `forward-fine` by default.
4. The shared files are not changed. A change goes into the Exp01 file and
   is copied out.

## On Gadi: the production on giamip_8re_r2

The production of 2026-10 runs on Gadi (NCI) on the mesh `giamip_8re_r2`:
an outer sphere at 8 Re and 2 rounds of local refinement near the coasts
and the ice margins, 4 957 665 cells. The decisions of this phase are in
`NOTES/team/r2-production/00-brief.md` of the worktree (not in the
repository). Exp12 runs on 18 `normalsr` nodes (1872 ranks). The
input file holds a mesh partitioned for that rank count, so every solver
job of the experiment uses it.

| file | purpose |
|---|---|
| `run_prepare.pbs` | the input file (`prepare_exp12.py`) |
| `run_transfer.pbs` | the final L0 bed onto the r2 input file (`transfer_bed_exp12.py`) |
| `run_invert.pbs` | the inversion, with the watchdog and the node memory and jobfs samplers |
| `run_forward.pbs` | the forward run from the final bed, with checkpoints and a restart |
| `run_paraview.pbs` | the ParaView series of the forward run (1 rank) |
| `run_images.pbs` | the 4K images of the forward run, one part of the steps per job (project vo05) |
| `run_archive.pbs` | the records and the final state from /scratch to /g/data |
| `node_memory_sampler.sh`, `node_jobfs_sampler.sh`, `signal_ranks.sh` | the helpers of `run_invert.pbs` |
| `probe_partition.py` | what each part of the mantle-balanced partition of a mesh holds |
| `../submit_r2.sh` | submits the jobs of one stage for one experiment and node count |

The job scripts and helpers of this table (except `../submit_r2.sh`) and
`exp12_common.py`, `exp12_diagnostics.py`, `invert_exp12.py`,
`forward_exp12.py` and `paraview_exp12.py` are copies of the Exp01 files
with `exp01`, `Exp01` and `EXP01` changed to `exp12`, `Exp12` and `EXP12`,
except `env-exp01-pr4638-pyop2fix.sh`, `EXP01_MPLCONFIGDIR`,
`EXP01_ALLOW_CORE_DUMPS`, `tests/giamip/exp01.py`, `views_exp01_L1.json`
and the Exp01 measurements in the comments. Do not edit them here: a change
goes into the Exp01 file and is copied out.

The march is schedule C, as in the final L0 run: the prepare job gets
`--schedule windows`, and the forward run `SCHEDULE=windows`. The transfer
writes the r2 observed bed plus the change of the final L0 inversion
(`START=change`); the source is
`/g/data/xd2/sg8812/gia-mip/exp12/L0/invert-c/state.h5`. The tape of the
inversion goes to `$PBS_JOBFS` of each node and nowhere else.

The steps, on a Gadi login node in a login shell, from this directory:

1. `bash ../submit_r2.sh 12 18 ready` submits the prepare job, the
   transfer and the 3-step test, each after the one before.
2. Read the records. Prepare: `prepare/inputs_exp12.json` (the cell counts
   against the sidecar, the statistics per layer, the ice volume check).
   Transfer: `transfer/inputs_exp12_transfer_bed.json` (the coverage, the
   RMS of the bed change against the L0 value, the largest distance of a
   target node from the source surface). 3-step test: J, the memory per
   node, and the jobfs lines at the end of the job output (the tape per
   taped step and its projection to the full march, which must fit in
   400 GB per node).
3. `bash ../submit_r2.sh 12 18 invert` submits the inversion
   (24 h wall time). It stops at an RMS misfit below 1 mm or after 10 ROL
   iterations.
4. If the inversion stops at the wall time, `bash ../submit_r2.sh 12
   18 restart` continues it with `RESTART=state.h5` in the same
   directory.
5. `bash ../submit_r2.sh 12 18 forward` submits the forward run, the
   ParaView job, the image jobs (`IMAGE_PARTS`, default 2) and the archive
   job, each after the one before. If the forward run stops at the wall
   time (exit code 3), PBS deletes the jobs after it;
   `FORWARD_RESTART=1 bash ../submit_r2.sh 12 18 forward` continues
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
| run directory | `/scratch/xd2/$USER/giamip/exp12/8re_r2-18nodes/` (`prepare/`, `transfer/`, `invert-3steps/`, `invert/`, `jobs_r2.txt`) |
| forward run, ParaView files, images | `/g/data/xd2/sg8812/gia-mip/exp12/r2/forward-windows/` |
| archive of the records and the final state | `/g/data/xd2/sg8812/gia-mip/exp12/r2/` |

## Run log

| date | job | what | result |
|---|---|---|---|
| 2026-10-21 | 51261940, 51262158 (1 rank, `--earth_only`) | serial check of the drop rule, `L0-serial/earth-check*` | COMPLETED in under 2 min, 3.9 GB. Element 0 dropped in 76 619 of 187 603 cells below 90 km (41.1 % of the mantle volume; 82.9 % of 120-400 km, 85.5 % of 400-670 km); elements 1 and 2 never. Instant modulus -9.6 % (mantle mean), -22.8 % and -23.6 % in 120-400 and 400-670 km. Power law in 26.7 % of the mantle. |
| 2026-10-21 | 51262180 (6 nodes, 768 ranks) | prepare, `L0-6nodes/prepare` | COMPLETED in 5 min 14 s, peak 2.85 GB per rank. Drop numbers as the serial check. Ice volume within 1.29e-3 of the grid. |
| 2026-10-21 | 51262335 (51262181 failed at the guard of another experiment) | transfer of the Exp01 L0 bed, `START=change`, `--allow_other_experiment` | COMPLETED. Written bed 12.33 m RMS from the Exp12 observed bed. |
| 2026-10-21 | 51262336 | Taylor test, 3 steps, `L0-6nodes/taylor-3steps` | COMPLETED in 30 min. R1 rates 1.990 to 1.997. Largest node 213 GB (tape in /tmp). |
| 2026-10-21 | 51262337 | inversion, 3 steps, 3 ROL iterations | COMPLETED in 27 min. RMS misfit 19.0 m to 15 mm. |
| 2026-10-21 | 51263474 | full inversion, 61 steps of 2 kyr, tape on /scratch, `L0-6nodes/invert-full` | COMPLETED in 4 h 59 min. Stopped by the RMS rule after 5 ROL iterations: 10.91 m to 0.5 mm. RMS change of the bed from the observed bed 17.69 m. Newton 2-3 per step. Largest node 228 GB of 230 GB. |
| 2026-10-21/22 | 51290801-51290807 (schedule C, `submit_chain_c.sh 12`) | prepare `--schedule windows`, transfer of the 2 kyr bed, inversion on 12 x 64 ranks, forward `--schedule windows`, ParaView, archive | Inversion: 722 taped steps, stopped after 3 ROL iterations, 0.517 m to 0.9 mm, 19.7 h, 116 GB per node. Forward: 4 h 08 min, year-0 RMS 0.8727 mm (J relative difference 5.3e-5 to the inversion), Newton at most 3 (500 yr) and 2 (25 and 10 yr windows). ParaView MaxRSS 15.4 GiB. Gadi `/g/data/xd2/sg8812/gia-mip/exp12/L0/forward-windows`: 935 files, 20.5 GB, equal to Setonix. |
