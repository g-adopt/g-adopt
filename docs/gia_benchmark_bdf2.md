# BDF2 and material elimination in the GIA experiment drivers

These changes live in `w3168/giamip-bdf2`, in the
`g-adopt-giamip-bdf2` worktree. The original benchmark and GIAMIP checkouts
are unchanged. Codex assisted.

## Selecting the formulation

The GIAMIP forward/inversion drivers for experiments 01–12, the Spada and
Martinec benchmark drivers (including Martinec E), and `tests/giamip/exp01.py`
accept:

```text
--time_scheme {backward_euler,bdf2}   default: backward_euler
--material_solver {coupled,eliminated}  default: coupled
--material_quad_degree N            default: 4; used by eliminated
```

Use both `--time_scheme bdf2 --material_solver eliminated` for the new
memory-reduction strategy. Selecting BDF2 alone keeps global DG history
unknowns. The eliminated formulation removes those unknowns from the mixed
system, retaining displacement, potential and physical Real constraint fields.
Its material histories live at the mechanics integration points. Degree N
selects a quadrature rule, not a polynomial history approximation. The solver
uses that same volume integration degree; quadrature convergence needs checking
for each spatial discretisation and nonlinear model.

The direct local constitutive solve supports nonlinear Maxwell rheology,
including spatial DG0 exponents, and multiple modes. These driver options use
the current Maxwell approximations; they do not change an experiment into the
separate recovering-backstress model. Low-rank DtN remains enabled. There is
no Firedrake external operator or sparse constitutive matrix composition.

## Commands

Run in the Firedrake environment from the worktree root, with the local `giamip`
and `giamsh` packages on `PYTHONPATH` if they are not installed.

```bash
python tests/gia_selfgrav_benchmarks/spada.py --case cap \
    --time_scheme bdf2 --material_solver eliminated \
    --output_path /path/to/new-spada-output

python tests/gia_selfgrav_benchmarks/martinec.py --case B \
    --time_scheme bdf2 --material_solver eliminated \
    --output_path /path/to/new-martinec-output

python tests/gia_selfgrav_benchmarks/martinec.py --case E \
    --time_scheme bdf2 --material_solver eliminated \
    --output_path /path/to/new-martinec-e-output

python demos/glacial_isostatic_adjustment/giamip/exp01/forward_exp01.py \
    --state /path/to/state.h5 --time_scheme bdf2 --material_solver eliminated \
    --output_path /path/to/new-giamip-output

python demos/glacial_isostatic_adjustment/giamip/exp01/invert_exp01.py \
    --inputs /path/to/prepared-inputs.h5 --mode taylor --steps 4 \
    --time_scheme bdf2 --material_solver eliminated \
    --output_path /path/to/new-taylor-output
```

Replace exp01 with another experiment number for its own prepared input and
driver. All existing data dependencies still apply; some experiment modules
also require `gdrift`. Use a fresh output directory when changing formulation.

Spada uses CG3 displacement / DG2 histories in coupled mode and CG3 displacement
with quadrature histories in eliminated mode; potential is CG2. Its
centre-of-mass constraint is on by default. `--no_centre_of_mass` disables it,
including for legacy unconstrained restarts. Comparison study generators name
both frame choices explicitly. GIAMIP uses CG2 displacement and CG2 potential;
its coupled histories remain DG1.

## Time history, loading and restarts

Only the material histories have time derivatives. Ice/load and all algebraic
fields are evaluated at the new endpoint; BDF2 needs no intermediate stage
ice evaluations. BE supplies the first accepted step. Drivers explicitly
request another BE startup when a prescribed timestep increase violates
`h_new/h_old < 1+sqrt(2)`, preserving the forcing/output grid. This is a
prescribed schedule, not adaptive error control. Resolve loading jumps on that
grid; second-order accuracy across unresolved nonsmooth forcing is not implied.

The benchmark elastic diagnostic is followed by a reset of the mixed state,
external material histories and both accepted time-history levels. The first
physical step therefore starts from the intended undeformed history.

Restart files store the mixed state, external histories, both accepted BDF2
levels and the previous accepted timestep. Quadrature values are encoded
losslessly as DG0 vectors holding every independent component at every local
point. This storage encoding does not replace the constitutive history with a
DG0 approximation. The parent/submesh mapping transfers whole cell records;
restarts use the saved mesh and rank distribution. No projection of quadrature
values into a lower-order polynomial space is used.

Restart validation rejects a scheme/formulation/quadrature change or missing
required BDF history. Legacy BE files remain coupled BE files. A checkpoint
cannot turn an existing coupled run into eliminated BDF2 without a separate,
explicit history-conversion procedure.

Elimination still stores accepted histories, and quadrature can use more
history values per cell than DG1. Removing global unknowns primarily reduces
matrix, preconditioner and differentiated-solve costs; it does not guarantee
smaller checkpoint files or a particular total memory saving.

## Benchmark provenance and physics

The selectively ported benchmark sources come from `w3168/martinec-E` at
`a7e64589` (2 November 2026). At the audit, local Martinec E had four commits
not on its remote; local Martinec B had five. The benchmark checkout also had
many untracked result files and several untracked tests/notes. Those result
folders were not copied or modified. The current GIAMIP source branch was
`02a75382` (10 November), matching its remote. The benchmark and GIAMIP branches
had 32 commits each beyond their shared ancestor `1305b4d5`.

Martinec now uses the current GIAMIP 1 m softplus sea-level sheet, including
case E, rather than restoring the historical mask/frozen-slope library code.
Historical mask outputs are not exact reproductions of this model; the sheet
choice is recorded and incompatible legacy continuation is rejected. Existing
backward-Euler benchmark error factors are not BDF2 acceptance criteria.

## Validation scope

The library already has serial/MPI tests of variable-step nonlinear material
elimination with low-rank DtN, physical Real fields, moving sea-level masks,
first reverse adjoints and disk replay; see [the solver notes](gia_bdf2.md) and
[the low-rank investigation](gia_eliminated_lowrank.md).

Driver continuation tests exercise real checkpoint save, mesh reload, submesh
reconstruction and continued unequal-step solves for both storage formats and
both material formulations. They include a spatial nonlinear exponent,
elastic-history reset and rejection of incompatible restart metadata. Five
eliminated checkpoint/reset cases also pass on two MPI ranks, including both
checkpoint formats and both BE/BDF2. The topography-only E checkpoint has a
separate save/reload regression.

Current driver validation includes 169 configuration/schedule/helper checks,
15 real checkpoint/timestep/selection checks, and a separate test of the actual
Martinec E taped history loop on the small two-mesh geometry. That E test uses
spatial nonlinear rheology and moving coastlines, agrees with an independently
rebuilt perturbed forward history, and passes a second-order Taylor test for
the topography control.

A two-rank Spada CG3 cap smoke run with COM enabled, low-rank DtN and degree-4
quadrature elimination completed its elastic diagnostic and two 50-year steps,
including the full BDF2 step and both application checkpoints. Reloading its
completed 3D checkpoint also succeeded. This reload checks the saved terminal
state; continued unequal-step evolution is covered by the smaller serial/MPI
tests. Each solve took
two outer iterations. The global system had 222,751 unknowns, versus roughly
1.05 million in the coupled smoke layout. Many inner solves hit their
200-iteration cap, so this remains a solver-performance concern. Reproduce
from the worktree root with:

```bash
mpiexec -n 2 python tests/gia_selfgrav_benchmarks/spada.py \
    --case cap --smoke --time_scheme bdf2 --material_solver eliminated \
    --timestep_scale 5 --load_representation spectral --load_nmax 2 \
    --diagnostic_nmax 2 --dtn_degree 2 --checkpoint_every 1 \
    --output_path /path/to/new-smoke-output
```

This coarse mesh, low harmonic cutoff and short history validate driver
operation. Full GIAMIP production histories/inversions, benchmark spatial/time
convergence, 3D nonlinear benchmark validation and 3D memory scaling remain
outstanding. No new memory-saving percentage is inferred from the unknown
count. Output from the validation run is in
`/tmp/gia-spada-eliminated-bdf2-cg3/summary_cap.json`.
