# GIAMIP Exp09: Exp03 with the Extended Burgers rheology

This directory runs GIAMIP experiment Exp09 with the coupled
self-gravitating sea-level solver. Exp09 is Exp03 with a transient
rheology: the Extended Burgers model (EB) of Lau (2023) and Ivins et al.
(2022), represented by N = 3 Maxwell elements in parallel. The work has two
parts. The inversion finds the bed at 122 ka so that the run carries it to
the observed present-day bed at year 0. The forward run then marches from
that bed at fine steps and writes the GIAMIP outputs at the protocol years.

The directory is self-contained. It imports nothing from another experiment
directory and nothing from `tests/`. The files `exp09_common.py`,
`invert_exp09.py`, `forward_exp09.py`, `paraview_exp09.py`,
`transfer_bed_exp09.py` and the Slurm scripts are copies of the Exp01
master files (`../exp01/`) with the rename exp01 to exp09 (see "Changes for
another experiment"). Only `prepare_exp09.py`, `test_prepare_exp09.py` and
this README are specific to Exp09. The design is
`NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-2026-10-20.md` (private record;
decisions S1-S32).

## The experiment

The values come from `giamip.mip.inputs.GiamipExperiment("Exp09")`, the
case file `giamip/resources/cases/giamip/exp09.json` of gia-mip, and the fit
file `../eb_fit/eb_fit.json`.

| item | value |
|---|---|
| ice | `iceHistory-GLAC3c_profile1.nc` (as Exp03) |
| topography | `bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc`, the G-ADOPT choice `bedmap3` (the case file leaves it open; `giamip.cases.with_gadopt_choices` gives the file) |
| start | 122 ka |
| viscosity eta_0 | midpoint of the GIAMIP suggested range (`giamip-open-range-midpoint`, G-ADOPT choice): elastic lid 0-90 km, 3.16e20 Pa s to 670 km, 1.0e22 Pa s to the CMB |
| elastic structure and density | PREM-stable (G-ADOPT choice) |
| rheology | Extended Burgers (`burgers`, G-ADOPT choice), N = 3 Maxwell elements below 90 km, one elastic element in the lid |
| EB parameters | alpha 0.274, Delta 1.9, tau_H / tau_M 1, tau_L 0, one set in the whole mantle below 90 km (S22) |
| EB fit | `free_time_fits/3` of `eb_fit.json` (S31, free element times): tau_bar 0.01488, 0.1895, 2.933; mu_bar 0.1935, 0.1773, 0.3285; sum mu_bar 0.6993; sum mu_bar tau_bar 1; largest modulus error 2.1 % over the band |
| compressibility | compressible (G-ADOPT choice) |
| constants | ice 917 kg/m^3, ocean water 1025 kg/m^3 |
| rotation | on, `k_s` 0.9422, C = 0.3307007 M Re^2 |
| core | fluid, mean density of the PREM-stable core |
| elements | P2 displacement, DG1 internal variables (N = 3, one (3, 3, 3) field), P2 potential |
| time stepping | backward Euler; inversion 61 steps of 2 kyr; forward schedule A (`--schedule fine`) |
| nodes | 6 nodes, 768 ranks, for the prepare, the transfer, the inversion and the forward run (S24) |
| warm start | the L0 bed of Exp03, `START=bed` (S20, P20) |

## The Extended Burgers elements

The fit is normalised: the moduli mu_bar_j are in units of the unrelaxed
modulus mu_inf, and the element times tau_bar_j are in units of the local
Maxwell time tau_M = eta_0 / mu_inf. The constraint
sum_j mu_bar_j tau_bar_j = 1 makes the long-term viscosity of the elements
equal to eta_0, the viscosity of the Maxwell twin Exp03.

`prepare_exp09.py` makes the element fields from the fit:

| region | modulus of element j (DG1) | viscosity of element j (DG0) |
|---|---|---|
| below 90 km (centroid depth) | mu_inf mu_bar_j | eta_0 mu_bar_j tau_bar_j |
| lid, 0-90 km | mu_inf w_j | ETA_LID w_j |

Here w_j = mu_bar_j / sum_k mu_bar_k. Below the lid, the Maxwell time of
element j is tau_bar_j tau_M at every node. In the lid, the sum of the
moduli is mu_inf and every element has the Maxwell time of the Exp03 lid,
so the lid is elastic, as in Exp03 (S10). The script stops if the cells
above 90 km are not the cells with the lid viscosity ETA_LID.

Expected Maxwell times of the elements in the midpoint model (tau_M about
140 yr in the upper mantle and 1.6 kyr in the lower mantle): about 2, 27
and 410 yr in the upper mantle, and about 24, 300 and 4700 yr in the lower
mantle. The record gives the measured values per shell
(`burgers.by_layer`).

The instant modulus below the lid is 0.70 mu_inf (sum mu_bar). Under a new
load increment, the elastic response of Exp09 is therefore about 1.4 times
that of Exp03. This is the intended effect of EB at a 10 yr step (design
handout 6.3 and 13).

The input file holds `shear_moduli` (DG1, 3 components) and `viscosities`
(DG0, 3 components) in addition to the Maxwell fields. The record has
`n_elements` = 3, so `exp09_common.read_inputs` reads the element fields
and `build_solver` builds 3 internal variables. `shear_modulus` and
`viscosity` stay in the file as mu_inf and eta_0 (the Maxwell reference).

## The files

| file | purpose |
|---|---|
| `exp09_common.py` | code that all scripts use: scales, mesh partition and cutting, input file, solver |
| `prepare_exp09.py` | reads giamip, the fit file and the mesh, and writes `inputs_exp09.h5` |
| `test_prepare_exp09.py` | serial tests of the EB part of the prepare script |
| `transfer_bed_exp09.py` | appends the Exp03 L0 bed to the input file (warm start) |
| `invert_exp09.py` | loads the input file, tapes the march, and runs a Taylor test or the ROL inversion |
| `forward_exp09.py` | the forward run from the inverted bed (here `--schedule fine`) |
| `paraview_exp09.py` | the ParaView series of the surface file of a forward run |
| `run_prepare.slurm`, `run_transfer.slurm`, `run_invert.slurm` | the prepare, transfer and inversion jobs on Setonix (6 nodes by default) |
| `run_forward_fine.slurm` | the forward run at the fine steps, with checkpoints and resubmission at the wall limit (6 nodes) |
| `run_paraview.slurm`, `run_archive_forward.slurm` | the ParaView job (1 rank) and the copy to Gadi |

## How to run

The final L0 run (2026-10-21/22) is schedule C (the protocol steps plus 25
yr and 10 yr steps in the fast melt windows of the ice history, decisions
S33-S35). The steps below are the inversion at 2 kyr steps, whose bed is
the start of schedule C. The schedule-C chain
(`NOTES/team/schedule-c/submit_chain_c.sh` of the worktree, not in the
repository) submits each job after the one before: (a) the prepare job with
`EXTRA="--schedule windows"` in `L0-6nodes/prepare-c/`; (b) the transfer of
the converged 2 kyr bed of Exp09 (`invert-full/state.h5`, `START=bed`: same
mesh, same experiment) into that input file; (c) the inversion on schedule
C in `invert-c/` on 12 nodes of 64 ranks (`--nodes=12 --ntasks=768`) with
`EXTRA="--gia_rebuild_every 100 --log_memory --drop_block_adjoint_copy"`
and the tape on /scratch; (d) the forward run with `SCHEDULE=windows`
(`run_forward_fine.slurm`) in `forward-windows/`, a gate that passes only a
finished march, the ParaView job and the archive to
`/g/data/xd2/sg8812/gia-mip/exp09/L0/forward-windows/`. The inversion on
schedule C ended at 0.45 mm RMS after 6 ROL iterations in two legs. The
production on the r2 mesh is in the section "On Gadi: the production on
giamip_8re_r2".

All jobs except ParaView and the archive use 6 nodes and 768 ranks. The
input file holds a mesh partitioned for that rank count, and every later
job must use it. Before each job, send the code to the Setonix checkout
with `NOTES/sync_setonix.sh` and compare the MD5 of the files.

1. The prepare job, in a new run directory:

   ```bash
   mkdir -p /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/prepare
   cd /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/prepare
   sbatch --export=NONE $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_prepare.slurm
   ```

2. Read `inputs_exp09.json`. Compare the values with the tables above:
   `n_elements`, `burgers.fit`, `burgers.sum_check_relative_errors`,
   `burgers.by_layer` (instant modulus ratio 1 in the lid and 0.6993 below
   it), the viscosity per shell, and the ice volume check.
3. The warm start: append the Exp03 L0 bed to the input file. Exp03 and
   Exp09 have the same mesh, ice and observed bed, so `START=bed`. The
   record of the source says Exp03, so the transfer needs
   `--allow_other_experiment`. This is correct because the mesh (same MD5),
   the ice and the observed bed are the same:

   ```bash
   sbatch --export=NONE,START=bed,EXTRA=--allow_other_experiment,TARGET=$PWD/inputs_exp09.h5,SOURCE=/scratch/pawsey0821/$USER/giamip/exp03/L0-3nodes/invert-full/state.h5 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_transfer.slurm
   ```

4. The Taylor test with 3 steps. `MODE=taylor` ignores `RESTART`, so the
   test starts from the observed bed:

   ```bash
   mkdir ../taylor-3steps && cd ../taylor-3steps
   sbatch --export=NONE,INPUTS=../prepare/inputs_exp09.h5,MODE=taylor,STEPS=3 \
       $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_invert.slurm
   ```

   The R1 rates must be near 2. Read the node memory from the log. If the
   largest node is above about 210 GB, stop.
5. The inversion with 3 steps and 3 ROL iterations
   (`MODE=invert,STEPS=3,ITERATIONS=3`, `RESTART=` as above). J must
   decrease at every iteration. This is the first job that uses the
   transferred bed: its first J is the J of the Exp03 bed under the EB
   rheology.
6. The full inversion (`MODE=invert`, `RESTART=../prepare/inputs_exp09.h5`)
   in `invert-full/`. It stops at 1 mm RMS bed misfit or after 10 ROL
   iterations. If it stops at the time limit, submit it again in the same
   run directory with `RESTART=state.h5`.
7. The forward run at the fine steps (next section), then ParaView and the
   archive.

Add `--partition=debug` to `sbatch` only for short tests that fit its
limits.

## The forward run at the fine steps

Schedule A below was the plan before schedule C. The final forward run is
`--schedule windows` (step (d) of the schedule-C chain in "How to run"); it
uses the same restart, output and preconditioner rules as the fine
schedules.

`forward_exp09.py --schedule fine` (schedule A, S19) marches from 122 ka
with 50 yr steps to year -26000 and 10 yr steps to year +1000: 1920 + 2700
= 4620 steps. The ice is linear in time between the protocol years (P9).
The surface file and the parameter log have one entry per protocol year
only (S12), so the ParaView series is the protocol series. The text log
has one line per step (Newton and outer iterations, solve time, dt).

The preconditioner is rebuilt every 100 solves (P17), at the change of dt
at year -26000, and after a solve with more than 6 outer iterations per
Newton iteration. The run writes a checkpoint every 200 steps (at the next
protocol year) into its copy of the state file (`forward_state.h5`). Before
the wall limit (the job limit minus 1 h), it writes a checkpoint, leaves
`RESTART_NEEDED`, and the job submits the next leg with `RESTART=1`.

```bash
mkdir -p /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/forward-fine
cd /scratch/pawsey0821/$USER/giamip/exp09/L0-6nodes/forward-fine
sbatch --export=NONE,STATE=../invert-full/state.h5 \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_forward_fine.slurm
```

After the first 50 steps, compare the time per step with a Maxwell step on
the same mesh and nodes. If the ratio is above 2, stop the run and start it
again with `SCHEDULE=fine_b` (schedule B, S25).

When the last leg ends, make the ParaView series on 1 rank and copy the run
to Gadi. The resubmitted legs have new job ids, so submit these two jobs
after the last leg, not with a dependency on the first leg:

```bash
sbatch --ntasks=1 --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_paraview.slurm
sbatch --dependency=afterok:<paraview job> --export=NONE \
    $MY_GADOPT/demos/glacial_isostatic_adjustment/giamip/exp09/run_archive_forward.slurm
```

The archive job copies the surface file, the logs, the summary, the job
outputs and `paraview/` to
`/g/data/xd2/sg8812/gia-mip/exp09/L0/forward-fine/`. It does not copy
`forward_state.h5` (the input file plus the checkpoints).

## Mesh, control, stop rule and restart of the inversion

These are the same as in Exp03 (`../exp03/README.md`, sections "The mesh",
"The control and the node map", "The stop rule" and "Restart"). The mesh is
`n2_giamip_all_o1.msh` (MD5 `a4da84206dc82f5b458be66e4fd33844`). The 90 km
and 670 km spheres are mesh spheres, so the base of the lid and the jump of
eta_0 at 670 km are exact.

## Changes for another experiment

The shared files of this directory come from `../exp01/` by this rule:
replace `exp01` by `exp09` and `Exp01` by `Exp09`, but keep
`env-exp01-pr4638-pyop2fix.sh`, `EXP01_MPLCONFIGDIR`, "Exp01 job 51182503"
and `tests/giamip/exp01.py`. In the Slurm scripts, the default node count
of the prepare, transfer and inversion jobs is 6, and the example paths are
those of Exp09. Do not edit the shared files here. Edit the Exp01 master
and copy it out again.

## On Gadi: the production on giamip_8re_r2

The production of 2026-10 runs on Gadi (NCI) on the mesh `giamip_8re_r2`:
an outer sphere at 8 Re and 2 rounds of local refinement near the coasts
and the ice margins, 4 957 665 cells. The decisions of this phase are in
`NOTES/team/r2-production/00-brief.md` of the worktree (not in the
repository). Exp09 runs on 18 `normalsr` nodes (1872 ranks). The
input file holds a mesh partitioned for that rank count, so every solver
job of the experiment uses it.

| file | purpose |
|---|---|
| `run_prepare.pbs` | the input file (`prepare_exp09.py`) |
| `run_transfer.pbs` | the final L0 bed onto the r2 input file (`transfer_bed_exp09.py`) |
| `run_invert.pbs` | the inversion, with the watchdog and the node memory and jobfs samplers |
| `run_forward.pbs` | the forward run from the final bed, with checkpoints and a restart |
| `run_paraview.pbs` | the ParaView series of the forward run (1 rank) |
| `run_images.pbs` | the 4K images of the forward run, one part of the steps per job (project vo05) |
| `run_archive.pbs` | the records and the final state from /scratch to /g/data |
| `node_memory_sampler.sh`, `node_jobfs_sampler.sh`, `signal_ranks.sh` | the helpers of `run_invert.pbs` |
| `probe_partition.py` | what each part of the mantle-balanced partition of a mesh holds |
| `../submit_r2.sh` | submits the jobs of one stage for one experiment and node count |

The job scripts and helpers of this table (except `../submit_r2.sh`) and
`exp09_common.py`, `exp09_diagnostics.py`, `invert_exp09.py`,
`forward_exp09.py`, `paraview_exp09.py` and `transfer_bed_exp09.py` are
copies of the Exp01 files with `exp01`, `Exp01` and `EXP01` changed to
`exp09`, `Exp09` and `EXP09`, except `env-exp01-pr4638-pyop2fix.sh`,
`EXP01_MPLCONFIGDIR`, `EXP01_ALLOW_CORE_DUMPS`, `tests/giamip/exp01.py`,
`views_exp01_L1.json` and the Exp01 measurements in the comments. Do not
edit them here: a change goes into the Exp01 file and is copied out.

The march is schedule C, as in the final L0 run: the prepare job gets
`--schedule windows`, and the forward run `SCHEDULE=windows`. The transfer
writes the r2 observed bed plus the change of the final L0 inversion
(`START=change`); the source is
`/g/data/xd2/sg8812/gia-mip/exp09/L0/invert-c/state.h5`. The tape of the
inversion goes to `$PBS_JOBFS` of each node and nowhere else.

The steps, on a Gadi login node in a login shell, from this directory:

1. `bash ../submit_r2.sh 09 18 ready` submits the prepare job, the
   transfer and the 3-step test, each after the one before.
2. Read the records. Prepare: `prepare/inputs_exp09.json` (the cell counts
   against the sidecar, the statistics per layer, the ice volume check).
   Transfer: `transfer/inputs_exp09_transfer_bed.json` (the coverage, the
   RMS of the bed change against the L0 value, the largest distance of a
   target node from the source surface). 3-step test: J, the memory per
   node, and the jobfs lines at the end of the job output (the tape per
   taped step and its projection to the full march, which must fit in
   400 GB per node).
3. `bash ../submit_r2.sh 09 18 invert` submits the inversion
   (24 h wall time). It stops at an RMS misfit below 1 mm or after 10 ROL
   iterations.
4. If the inversion stops at the wall time, `bash ../submit_r2.sh 09
   18 restart` continues it with `RESTART=state.h5` in the same
   directory.
5. `bash ../submit_r2.sh 09 18 forward` submits the forward run, the
   ParaView job, the image jobs (`IMAGE_PARTS`, default 2) and the archive
   job, each after the one before. If the forward run stops at the wall
   time (exit code 3), PBS deletes the jobs after it;
   `FORWARD_RESTART=1 bash ../submit_r2.sh 09 18 forward` continues
   the run and submits them again. If the inversion ran with fewer ranks
   per node (`RANKS_PER_NODE`), give the forward stage the same
   `RANKS_PER_NODE` and `PREP`, and `FORWARD_NODES` for full nodes: the
   forward run has no tape, so for example the 624 ranks of a 16-node
   inversion with 39 ranks per node run on 6 full nodes
   (`RANKS_PER_NODE=39 FORWARD_NODES=6`).

| item | path |
|---|---|
| run directory | `/scratch/xd2/$USER/giamip/exp09/8re_r2-18nodes/` (`prepare/`, `transfer/`, `invert-3steps/`, `invert/`, `jobs_r2.txt`) |
| forward run, ParaView files, images | `/g/data/xd2/sg8812/gia-mip/exp09/r2/forward-windows/` |
| archive of the records and the final state | `/g/data/xd2/sg8812/gia-mip/exp09/r2/` |

## Run log

| date | job | what | result |
|---|---|---|---|
| 2026-10-21 | 51261982, 51261983 | prepare (2 kyr march), transfer of the Exp03 L0 bed, `L0-6nodes/prepare` | COMPLETED |
| 2026-10-21 | 51261984, 51261985 | Taylor test 3 steps; inversion 3 steps x 3 ROL | R1 rates 2.000; J decreases; largest node 176 GB |
| 2026-10-21 | 51262594 | full inversion at 2 kyr steps, `invert-full` | COMPLETED 3 h 17 min, 5 ROL iterations, 0.574 m -> 0.6 mm |
| 2026-10-21 | 51266333 | forward schedule A | cancelled before the start: Sia changed Exp09 to schedule C |
| 2026-10-21/22 | 51290773, 51290774 | prepare `--schedule windows` (752 steps), transfer of the 2 kyr bed, `prepare-c` | COMPLETED |
| 2026-10-21/22 | 51290775, 51337512 | inversion on schedule C, 12 nodes x 64 ranks, `invert-c` | leg 1 FAILED after 4 iterations (network fault, 3.7 mm); leg 2 COMPLETED, 0.45 mm after 6 iterations in total |
| 2026-10-22 | 51337513-51337516 | forward `--schedule windows`, gate, ParaView, archive, `forward-windows` | COMPLETED; year-0 RMS 0.4478 mm = inversion; Gadi `/g/data/xd2/sg8812/gia-mip/exp09/L0/forward-windows/`, 935 files, 20.5 GB |
