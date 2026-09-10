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
potential, the multipliers, the fluid-core pressure, and the rotation scalars
in one mixed space.
`DtNTwoBlockSchurPC` preconditions that system. The 3-D Spada benchmark is in
`demos/glacial_isostatic_adjustment/3d_spada_selfgrav/`.

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
