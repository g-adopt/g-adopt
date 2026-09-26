# Gravity and self-gravity in G-ADOPT

This directory holds the demos and the design documents for the gravitational
Poisson solver. The solver computes the perturbation gravitational potential on
a truncated domain. It uses a Dirichlet-to-Neumann map on the outer boundary,
so the exterior problem stays exact.

Three bodies of work use this directory. The list below gives their state on
2026-08-18.

## What is complete

**The gravitational Poisson solver.** `gadopt/gravity_solver.py` provides
`GravitySolver`. It supports two representations of the Dirichlet-to-Neumann
map: a multiplier form with one `Real` unknown for each mode, and a low-rank
form that removes those unknowns. Both forms pass the 2-D monopole test.
`gadopt/spherical_harmonics.py` evaluates real orthonormal spherical harmonics
by recursion. A parallel scaling study is in `tests/parallel_scaling_gravity/`.

**The self-gravitating GIA solver.** `gadopt/gia_gravity.py` provides
`SelfGravitatingGIASolver`. It solves displacement, the internal variable, the
potential, the fluid-core pressure, and the rotation scalars in one mixed
space. The exterior condition enters in one of two representations,
`dtn_representation="multiplier"` (one `Real` unknown for each spherical
harmonic mode) or `"lowrank"` (a rank-k update on the potential rows, no
multiplier unknowns). The default is low-rank on the full layout and
multiplier on the condensed layout (`resolve_dtn_representation`); on the
3-D Spada benchmark the low-rank arm gives the same state as the multiplier
arm at every truncation, at 0.69 of the warm step, with a setup that does
not grow with the truncation. `DtNTwoBlockSchurPC` preconditions the system,
with `CondensedBlockPC` on block 0 and, on the low-rank representation,
`LowRankPotentialPC` on its potential split. The 3-D Spada and Martinec benchmarks are in
`tests/gia_selfgrav_benchmarks/`: the driver `spada.py` runs the cap case (U,
V and N against TABOO) and the polar-motion case, and `martinec.py` runs the
sea-level cases B, C and D, with the library's default solver configuration.
Its `README.md` gives the run commands, the meshes and the pass criteria.

**Power-law rheology in 3-D.** `SelfGravitatingGIASolver` solves a power-law
rheology on the full layout with Newton. The 3-D Spada restart driver
that measured it, `b5_restart_condensation.py`, is in git history at commit
`a8df4939` and not in the benchmark directory. It takes `--exponent`,
`--transition-stress-mpa` and `--power-law-layers`. The exponent is a DG0
field that is 3 in the two upper-mantle shells (70 km to 670 km) and 1 in the other shells, as in
`tests/3d_weerdesteijn_coupled`. The transition stress uses the sqrt(2 J2)
convention of that test. The driver option `--stress-report` prints the
deviatoric stress of each checkpoint state per shell, and the step size that
keeps `dt / (tau f)` below 5, where `f` is the power-law factor. On 2026-09-17
the low-rank arm ran three restart steps at exponent 3 and 0.2 MPa, at
20 kyr with 500 yr steps and at 1 kyr with 25 yr steps (Gadi jobs 179212840
and 179212841). Every Newton solve converged in 2 to 4 iterations. Each
Newton iteration costs the same linear work as one Newtonian step (6 or 7
block-0 applications, none at the cap). At 1 kyr a power-law step takes
2.8 times the wall time of the Newtonian step. The record is
`NOTES/team/power-law-3d/STRESS-CHECK.md`.

**The fluid-core volume constraint.** One uniform pressure multiplier enforces
zero integrated CMB flux. It removes the degree-zero core-mass mode that broke
the B5 march. The local implementation and review are complete.

**A near-incompressible preconditioner.** `NearlyIncompressibleAssembledPC`
gives GAMG the low-degree divergence-free modes together with the rigid-body
modes. A displacement-only operator becomes hard to solve as the bulk/shear
ratio increases, because its slow modes move into the divergence-free space.
The enrichment makes the solve possible at a bulk/shear ratio of 100 or more,
where plain GAMG fails.

## What is open

**The branch migration.** The current branch contains gravity, coupled GIA,
fluid-core physics, and mechanics preconditioning. The private migration plan
is `NOTES/PLAN.md`.

**Cartesian gravity.** The 2-D periodic rectangular and 3-D periodic Cartesian
DtN maps do not exist. Both can use one Fourier-map implementation.

**Formal convergence.** The radial and Cartesian cases need one consistent
convergence campaign against independent `passess` references.

**Cost study E6** in `ROADMAP-GRAVITY.md`. Nobody measured degrees of freedom,
wall-clock time and Krylov counts for configuration A against configuration D.

**Power-law time-step control.** No solver changes the step size when the
stress rises. Newton failed on the 2-D annulus when `dt / (tau f)` was about
25. No 3-D job has found the step at which Newton fails.

**Coupled mechanics cost.** The P3 B5 result passes the accuracy gate. Its 351
capped block-0 solves identify an open preconditioner problem.

**Condensed history.** The condensed update differs from the uncondensed DG
mass projection by approximately `7.3e-4`.

**Monolithic Stokes-Poisson coupling for mantle convection**, described in
`ROAD-MAP-STOKES-COUPLE.md`. That document is a plan, not a record. Do not
confuse it with the GIA coupling above: the GIA coupling is complete, the
mantle-convection coupling is not started.

## The documents

| document | type | what it gives you |
|---|---|---|
| `CLAUDE.md` | record | the submesh workflow, the cross-mesh form, and the mesh-conforming shell requirement |
| `GRAVITY-LESSONS-LEARNED.md` | record | the error budget, the curved-mesh trick, Firedrake `Real`-space behaviour, and the validation record |
| `ROADMAP-GRAVITY.md` | design record | the radial DtN mathematics and benchmark matrix. Cartesian and E6 remain open |
| `ROAD-MAP-STOKES-COUPLE.md` | plan | mean-free dynamic topography for mantle convection. Not started |
| `exploration_*.md` | record | four investigations into submesh and cross-mesh options in Firedrake and PETSc |
| `spikes/` | code | exploratory drivers and local diagnostic gates |

Read the records for the reasoning behind a decision. Read the plans for work
that is not complete.

## Two results worth knowing before you change anything

The naive modal Dirichlet-to-Neumann form is hostile to the solver. The
Robin-shifted form is not. `GRAVITY-LESSONS-LEARNED.md` section 4 gives the
measurement.

More boundary treatment does not always give more accuracy. The error budget in
`GRAVITY-LESSONS-LEARNED.md` section 1 shows why. Quadrature or the density
discretisation can set the accuracy floor instead.

## The private record

Measurements, campaign data and handover notes are in `NOTES/`, which stays out
of the repository. `NOTES/PLAN.md` is the current plan.
`NOTES/HANDOVER.md` is the current evidence record.
