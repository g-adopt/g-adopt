# The Martinec et al. (2018) sea-level benchmark in 3-D

This directory contains the sea-level benchmark of Martinec et al. (2018) for
G-ADOPT, cases B, C and D.

Martinec, Z., et al. (2018), A benchmark study of numerical implementations of
the sea level equation in GIA modelling, Geophys. J. Int. 215(1), 389-414.

## The benchmark

The benchmark drives the Earth model M3-L70-V01 of Spada et al. (2011) with an
ice cap and solves the sea-level equation on a prescribed ocean basin. The
reference solution is VEGA. The comparison is against U, N and S along two
meridians at the terminal time, and against the time series of the uniform
water layer, the ice masses and the ocean area.

| case | ocean | ice | time | basin |
|---|---|---|---|---|
| B | fixed coastline | L1, 1500 m | T0, a step at t = 0 | B1, far from the ice |
| C | fixed coastline | L2, 500 m | T1, growth over 10 kyr | B2, under the ice |
| D | moving coastline, floating ice | L2, 500 m | T1 | B2 |

Case A has no ocean and the Spada driver next door covers it. Case E
prescribes the topography at the terminal time, which is an inverse problem.
Both are out of scope here.

The driver `martinec_benchmark.py` solves the mechanics, the gravitational
potential, the centre-of-mass frame and the sea-level equation in one coupled
system. Its module docstring describes the equation, the load, the scales, the
discretisation and the solver.

## The physical constants come from gia-mip

Every physical constant of a case comes from the gia-mip files
`cases/benchmarks/martinec2018/<case>.json` and `loads/martinec2018.json`.
Martinec et al. (2018) and GIAMIP prescribe different densities, so a constant
written into the driver can disagree with the case. Give the checkout with
`--gia-mip`, or set the environment variable `GIA_MIP`. The driver reads JSON
only, so the checkout needs no installation.

The file `martinec_profiles.json` holds the colatitudes of the VEGA comparison
profiles. `write_profile_json.py` writes it from the converted reference data
of gia-mip. That script needs `netCDF4`; the driver does not.

## Files

| File | Content |
|---|---|
| `martinec_benchmark.py` | The driver for cases B, C and D. |
| `generate_martinec_sphere.py` | The mesh generator: the Spada sphere with local refinement in the cap and the two coastline bands. |
| `martinec_profiles.json` | The colatitudes of the VEGA comparison profiles. |
| `write_profile_json.py` | Writes `martinec_profiles.json` from the gia-mip reference data. |
| `run_martinec.pbs` | The PBS job script for the NCI Gadi cluster. |

The driver imports `build_meshes`, `spada_approximation`, `truncated_ladder`
and `time_ladder` from `../3d_spada_selfgrav/`, and the mesh
generator imports the geometry of `generate_selfgrav_sphere.py` from the same
directory. Nothing is copied, so the two benchmarks keep the same Earth model
and the same mesh construction.

## The meshes

Two meshes exist. Copy both files to the machine that runs the benchmark. Do
not generate them again there.

| mesh | cells | use | MD5 |
|---|---|---|---|
| `../3d_spada_selfgrav/b2_coarse_ar7.msh` | 99 059 | the laptop checks and the first comparison job | `f5072078dbf75567d9352b9cf64eec33` |
| `martinec_h78_cap12_band4_gl0.5_depth500_base500_grade2_ll1_mc32_alg1_seed7.msh` | 185 030 | the comparison jobs on the refined mesh | `6bd1f84eb53de7ec3e84a0b038ac489a` |

CAUTION: do not generate a mesh again for a production run. Another gmsh
version gives other tetrahedra. A tetrahedron with all four vertices on one
mesh sphere folds when the driver curves the mesh, and a folded cell gives a
wrong answer without an error message. Both files above have no such cell.

The refined mesh has a lateral spacing of 78 km in the ice cap and in the two
coastline bands, and 500 km elsewhere. The command that wrote it is

```bash
python generate_martinec_sphere.py
```

The generator needs the `gmsh` Python module. It counts the flat tetrahedra
after it writes the file, and it stops with a non-zero exit status if there is
one. Use another `--random-seed` in that case.

## Run the benchmark

If G-ADOPT is not installed, put the repository root on `PYTHONPATH`.

To check an installation, run the dry run. The dry run builds the meshes, the
load and the solver, assembles the residual one time, and stops before the
first solve:

```bash
python martinec_benchmark.py --case B --dry-run \
    --mesh ../3d_spada_selfgrav/b2_coarse_ar7.msh
```

A full run needs many ranks. Case B marches 118 steps to 10 kyr, and cases C
and D march 300 steps of 50 years to 15 kyr:

```bash
mpiexec -np 96 python martinec_benchmark.py --case B \
    --mesh ../3d_spada_selfgrav/b2_coarse_ar7.msh
```

On Gadi, submit `run_martinec.pbs`:

```bash
qsub -v CASE=B run_martinec.pbs
qsub -v CASE=B,STEPS=2,LABEL=B-smoke -l walltime=02:00:00 run_martinec.pbs
```

## The options

| option | default | meaning |
|---|---|---|
| `--case` | `B` | the benchmark case, B, C or D |
| `--gia-mip` | `$GIA_MIP` or `~/Workplace/gia-mip` | the gia-mip checkout |
| `--mesh` | the 78 km refined mesh | the gmsh file |
| `--dtn-representation` | `lowrank` | the representation of the exterior DtN condition |
| `--multiplier-pc` | `gadopt.DtNMultiplierDenseSchurPC` | the preconditioner of the `Real` block |
| `--epochs` | per time scenario | the output times in kyr |
| `--dt-yr`, `--ladder` | the ladder of the case | the time steps |
| `--steps` | all | stop after this many steps |
| `--block0-max-it` | `200` | the iteration cap of the mechanics-potential block solve. Lower it for a cheap check only |
| `--restart`, `--restart-file` | none | continue from a checkpoint state; the two are given together |
| `--export-ocean-function` | none | write `C0` on the radopt grid |
| `--label`, `--output` | the case letter, this directory | the output files |
| `--dry-run` | off | build everything and stop before the first solve |

`python martinec_benchmark.py --help` lists every option.

## The output

Every file name carries the label, which is the case letter by default.

| file | content |
|---|---|
| `martinec-<label>.h5` | the meshes and the state at every output time and every 50 steps |
| `martinec-<label>-profiles_<t>kyr.npz` | U, N, S and RSL in metres along the two meridians |
| `martinec-<label>-timeseries.npz` | the uniform water layer, the shift, the net sheet mass, the ocean area, the two ice masses, the mass dipole with its moment scale `load_moment` and its scaled value `dipole_rel`, and the Newton and outer iteration counts, at every step |
| `martinec-<label>-steptimes.npz` | the end time of every step in kyr, as the array `t_kyr` |
| `martinec-<label>-ocean_function_C0_nglv<N>.npz` | the fixed ocean function on the radopt grid |

The driver prints one `TIMESTEP` line and one `FRAME` line for each step. The
`TIMESTEP` line gives the cost: the Newton iterations, the outer iterations,
the block-0 applications and the three preconditioner counters, the wall-clock
time, and the sea-level quantities. The `FRAME` line gives the centre-of-mass
multipliers, the first mass moment of the perturbation `D`, its norm `abs_D`,
the moment scale `load_moment` and the scaled moment `rel_D`.

`load_moment` is `Re int |sigma| dS`, the largest first moment that this
surface load could carry. `rel_D` is `abs_D` divided by it, so it says which
fraction of the load's own moment the centre of mass still carries. Read
`rel_D` to judge the frame, because `abs_D` alone grows with the load. In
cases C and D the load grows from zero, so `rel_D` divides by a small number in
the first steps and is larger there for a reason that is not a frame error.
Read `abs_D` beside it in those cases.

The four preconditioner counters (`block0`, `assembly`, `columns` and
`dense_builds`) are running totals from the build of the preconditioner. The
cost of one step is the difference between two `TIMESTEP` lines. `newton`,
`outer` and `wall_s` are of the step itself.

The comparison with VEGA is not here. `scripts/martinec_gadopt.py` of gia-mip
reads these npz files, writes the benchmark NetCDF and scores it with
`scripts/martinec_compare.py`.

## Restart

`--restart INDEX` continues from state INDEX of a checkpoint. `--restart-file`
names that checkpoint and is required with `--restart`. The driver writes
`martinec-<label>.h5` and refuses to write over the file it reads, so the
continued run also needs its own label:

```bash
python martinec_benchmark.py --case B --restart 3 \
    --restart-file martinec-B.h5 --label B-continued
```

State 0 of a case-B run is the elastic response at t = 0, which a fresh run
writes and then throws away before it marches (the first marched step
reproduces it). A restart from state 0 therefore starts the march from the
elastic state and counts it twice. Restart case B from state 1 or later.

The continued run writes its own time series, which holds its own steps only.
Concatenate the two files to get the whole march. The step-time file holds the
whole ladder in both runs, because it describes the ladder and not the job.

The checkpoint holds the displacement and the internal variables twice: on the
mantle mesh for a reader, and on the parent mesh for the restart. The restart
needs the parent copies. A mesh that comes back from a checkpoint is an
independent mesh, and the coupled system needs the mantle to be a submesh of
the parent. The driver therefore reads the parent mesh from the file, cuts the
mantle from it, and interpolates the two fields down. That interpolation is
exact in both directions.

CAUTION: restart on the rank count that wrote the checkpoint. The parent copy
of the displacement leaves a parent dof that the mantle does not cover at zero,
and which dofs those are comes from point location, which is a local operation.
The set therefore depends on the partition. The driver stores the rank count
with each state and prints a `WARNING` when the restart runs on another one.
Measured in W8 on the 900 km mesh, 27 dofs of 355 890 differ between one rank
and two, which is 3.5e-05 in the L2 norm of the displacement copy and above the
outer tolerance. A checkpoint written before the rank count was stored is read
without the warning.

## The matched radopt runs

The matched radopt runs of cases B and C use the ocean function of G-ADOPT, so
that the width of the mask leaves the comparison. Write it with

```bash
python martinec_benchmark.py --case B --dry-run \
    --mesh ../3d_spada_selfgrav/b2_coarse_ar7.msh \
    --export-ocean-function 512
```

The grid of the file is the grid of the radopt run, so each grid size needs its
own export. The file holds `colatitude_deg`, `longitude_deg` and `C0` of shape
`(nglv, 2 nglv)`, which `read_ocean_function` of
`gia-mip/scripts/martinec_radopt.py` reads.

Each value is the mask that G-ADOPT integrates, at the grid point itself: the
facet size comes from the surface facet that contains the point, and the
initial sea level and the frozen slope are evaluated at the point. A point can
land in a cell that touches the surface along one edge only and therefore has
no facet there (4 of the 8 192 points at `nglv = 64` on the coarse mesh). The
facet size of such a point is the mean over the facets that meet at the nodes
of its cell on the surface, and the driver prints how many points took that
route. A point that no surface facet reaches at all is written as `nan` and
the driver prints a warning.

The driver also samples a facet-wise constant mask and prints the difference
between the two. That difference is large across a coast, because the mask
goes from 0.1 to 0.9 in about 1.5 facet sizes.
