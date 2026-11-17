# Variable-step BDF2 and local constitutive integration for GIA

`SelfGravitatingGIASolver(..., time_scheme="bdf2")` selects prescribed
variable-step BDF2, with backward-Euler startup. The default remains
`time_scheme="backward_euler"`. Both the coupled DG history formulation and
linear pointwise substitution support BDF2. The global displacement, potential,
rotation, sea-level and frame constraints remain coupled at the new time.
Nonlinear quadrature elimination and an illustrative recovering-backstress
model are also available through the opt-in interface below. These changes live
on `w3168/giamip-bdf2`, based on `sghelichkhani/giamip-exp01` at `02a75382`.

```python
# Use a Real-space Function so assignments are checkpointed on the adjoint tape.
dt = Function(FunctionSpace(mechanics_mesh, "R", 0)).assign(first_dt)
solver = SelfGravitatingGIASolver(
    state, approximation, layout=layout, dt=dt,
    time_scheme="bdf2", **existing_gia_options,
)
for h, forcing in prescribed_schedule:
    dt.assign(h)
    load.assign(forcing)
    solver.solve()  # prepares coefficients, solves and accepts material history
```

There is no separate `set_step` call in the integrated API. The solver prepares
coefficients automatically. With $r=h/h_{previous}$,

$$
\alpha=\frac{1+2r}{1+r},\qquad
\beta=1+r,\qquad \gamma=-\frac{r^2}{1+r},
$$

$$
\alpha m^{n+1}-\beta m^n-\gamma m^{n-1}
=h\frac{e(u^{n+1})-m^{n+1}}{\tau(s^{n+1})}.
$$

Only material histories have BDF derivatives; algebraic constraints use their
existing new-time residuals. Coefficients and both accepted history levels are
Functions recorded by pyadjoint. Constant Jacobian mode is refused because
startup and subsequent steps have different tangents. Timestep increases must
satisfy $h/h_{previous}<1+\sqrt2$; all steps must be positive and finite.

At a known load discontinuity, finish at the event, update the load, then call
`solver.restart_time_integration()` before the next solve to request BE startup.
There is no automatic event detection, error controller, step rejection/rollback
or differentiation with respect to timestep sizes or a control-dependent
timestep controller. Step ratios are evaluated in Python and recorded as
prescribed coefficient assignments.

The BDF history Functions are additional checkpoint state. An application-level
restart must save/restore both levels, previous accepted step size and the
current mixed state; the existing BE-only restart file is not sufficient.

## Nonlinear elimination at mechanics quadrature points

Create the usual mixed GIA space with the material histories outside the global
unknown vector. Their quadrature degree must match the residual integration
degree. For example, combining material elimination with low-rank DtN:

```python
Z, layout = self_gravitating_gia_space(
    mechanics_mesh, potential_mesh,
    gravity_bcs=gravity_bcs,
    n_internal_variables=2,
    condense_internal_variables=True,
    internal_variable_family="Quadrature",
    internal_variable_degree=4,
    dtn_representation="lowrank",
    # Include the existing rotation, fluid-core, COM and sea-level options.
)
state = Function(Z)
solver = SelfGravitatingGIASolver(
    state, approximation, layout=layout, dt=dt,
    time_scheme="bdf2", constitutive_method="direct", quad_degree=4,
    solver_parameters="iterative",
    **existing_gia_options,
)
```

The quadrature points are those used to integrate the mechanics residual.
For Maxwell modes with a common nonlinear fluidity $q=1/f(s)$, let
$H_i=\beta m_i^n+\gamma m_i^{n-1}$. At each point the local update is

$$
m_i^{n+1}=\frac{H_i+h q\,e(u^{n+1})/\tau_i}
                    {\alpha+h q/\tau_i},\qquad
s=2\sum_i\mu_i(e(u^{n+1})-m_i^{n+1}),\qquad
E(q)=q-1/f(s(q))=0.
$$

A damped scalar Newton solve determines $q$, shared by the Maxwell modes.
Spatial DG0 exponents and transition stresses are accepted by the Maxwell
expressions; the integration tests exercise a DG0 exponent changing from 1 to 3.
The implementation uses the implicit sensitivity $dq/dz=-E_z/E_q$ to assemble
the global tangent directly. There is no Firedrake external operator and no
composition of assembled constitutive derivative matrices.

A custom pyadjoint block records the converged full mixed solve and the material
updates together. Its reverse solves the transpose of the full mixed tangent
and assembles gradients with respect to Function inputs. Private replay
coefficients prevent later timesteps from overwriting the state needed by the
reverse calculation. Rigid-rotation projection is included in the recorded
map. Ordinary Firedrake forms still supply the gravity, fluid-core, sea-level,
rotation and centre-of-mass equations and their derivatives.

This removes histories from the global linear system, but still checkpoints
their accepted values. BDF2 needs two accepted levels. No new memory or timing
claim has been established for the full GIA implementation.

## Recovering backstress

`BackstressApproximation` uses two tensors: creep strain $p$ and strain-like
backstress $a$. It is a specified illustrative model, without experimental
calibration:

$$
s=2\mu(e-p),\quad X=2\mu_b a,\quad z=s-X,
\qquad \dot p=\frac{\chi(z)z}{2\eta},\qquad
\dot a=\dot p-\frac{a}{\tau_r}.
$$

For exponent $n>1$, $\chi(z)=1+(J_2(z)/s_t^2+10^{-24})^{(n-1)/2}$;
for $n=1$, $\chi=1$. In 2D, the invariant includes the implied out-of-plane
deviatoric component $z_{zz}=-\operatorname{tr}(z_{2D})$.
The same BDF coefficients advance both tensors. Eliminating them leaves one
scalar fluidity root per point; the coupled route retains both tensor equations.

```python
approximation = BackstressApproximation(
    bulk_modulus=K, density=rho, shear_modulus=mu, viscosity=eta,
    backstress_modulus=mu_b, recovery_time=tau_r,
    exponent=3, transition_stress=stress_scale,
    g=g, B_mu=B_mu, self_gravity_number=self_gravity_number,
)
# Use n_internal_variables=2 in the space factory.
# Use the direct/quadrature options above for elimination, or coupled DG histories.
```

The exponent and transition stress are fixed scalars in this backstress model.
Viscosity and recovery time may depend on spatial Function controls. Initial
backstress is represented by the tensor history, rather than a scalar background
stress parameter.

## Current limits

- Direct elimination supports both explicit DtN multipliers and low-rank DtN.
  See [the combined solver and block investigation](gia_eliminated_lowrank.md).
  Condensed spaces keep their existing multiplier default; select low-rank
  explicitly for the reduced potential preconditioner.
- Direct elimination supports natural displacement boundaries, including fluid
  core coupling. Strong and weak/Nitsche displacement boundaries, extra forcing
  callbacks and user-supplied Jacobians are refused. Backstress also requires
  natural displacement boundaries.
- Native coupled Quadrature histories encounter a Firedrake cross-mesh compiler
  limitation. Use DG histories for the normal coupled solver. The test-only
  independent quadrature reference uses separate coefficient fields and a
  serial unreduced Newton solve to bypass this limitation.
- The custom block implements first reverse derivatives for Function inputs;
  tangent-linear and Hessian actions are not implemented. Material Constants
  should be fixed; use Function controls for differentiated material parameters.
- The direct path binds low-rank scaling and cache inputs per replay/reverse
  step and uses a deterministic zero guess for replay.
  The GIAMIP and benchmark drivers now expose coupled and direct eliminated
  BE/BDF2 with application restart history; see the
  [driver guide](gia_benchmark_bdf2.md). Automatic adaptivity and full 3D
  production validation remain outstanding.

## Validation

`tests/unit/test_gia_bdf2.py` compares full two-mesh mixed solutions against
independently constructed BE history problems with time $h/\alpha$ and old
history $(\beta m^n+\gamma m^{n-1})/\alpha$. It also checks unequal-step
reverse adjoints and equality of startup with the existing BE solver.
It also tests the production-style nonlinear DG0 rheology with low-rank DtN,
fluid core, rotation, COM and a moving coastline. Replay/reverse cache decisions
use the saved effective history step $h/\alpha$, not the current live timestep.

`tests/unit/test_gia_direct_bdf2.py` compares nonlinear eliminated Maxwell and
backstress updates against simultaneous unreduced quadrature equations on a
curved 2D two-mesh annulus. It exercises all mixed global fields, unequal steps,
history-dependent objectives, guarded adjoint Taylor tests, disk checkpointing
and iterative solves. MPI checks use the adjoint cases; the independent
unreduced reference is serial only.

Validation on 17 November 2026, using Firedrake 2026.10:

| Check | Result |
| --- | --- |
| Coupled/linear-eliminated BDF2 and low-rank moving-coastline adjoint | 6 passed |
| Full mixed direct Maxwell/backstress, RAM/disk adjoints, iterative comparison | 10 passed, 1 skipped |
| Existing GIA suite together with BDF2 tests | 134 passed, 1 expected failure |
| Maxwell and backstress RAM adjoints on 2 MPI ranks | 2 passed on each rank |
| Maxwell and backstress RAM adjoints on 4 MPI ranks | 2 passed on each rank |
| Maxwell disk-checkpoint adjoint on 2 MPI ranks | 1 passed on each rank |

The combined eliminated/low-rank suite adds 13 passing serial checks, including
nonlinear Maxwell/backstress RAM and disk adjoints, moving coastlines, a gravity
control and constant-step replay. Fixed-coastline Maxwell/backstress RAM adjoints
also pass on four MPI ranks. On two ranks, moving-coastline, Maxwell disk and
gravity-control adjoints pass with the retained-kernel projection; constant-step
replay also passed in the preceding validation run.
The existing low-rank block suite has 30 passing serial checks, with its two-rank
multiplier comparison passing when launched explicitly. The automatic MPMD test
launcher failed in this environment; the explicit command is:

```bash
mpiexec -n 2 env _PYTEST_MPI_CHILD_PROCESS=1 python -m pytest \
  tests/unit/test_gia_lowrank_block0.py::test_two_ranks_match_the_multiplier_path -q
```

The direct-suite skip is the intentionally unsupported spatial backstress
exponent. MPI tests verify replay and guarded Taylor convergence on each
partition; this is not a performance-scaling or memory benchmark.

Reproduce the serial checks in the Firedrake 2026.10 environment:

```bash
source /data/will/codex_sbx/firedrake-2026.10/activate.sh
python -m pytest tests/unit/test_gia_bdf2.py tests/unit/test_gia_direct_bdf2.py \
  tests/unit/test_gia_eliminated_lowrank.py -q
mpiexec -n 2 python -m pytest \
  'tests/unit/test_gia_direct_bdf2.py::test_full_mixed_direct_adjoint[memory-maxwell]' \
  'tests/unit/test_gia_direct_bdf2.py::test_full_mixed_direct_adjoint[memory-backstress]' -q
# Repeat the MPI command with -n 4 for four ranks.
```

Reproduce the combined low-rank MPI checks:

```bash
mpiexec -n 2 python -m pytest \
  tests/unit/test_gia_eliminated_lowrank.py::test_lowrank_moving_coastline_adjoint \
  'tests/unit/test_gia_eliminated_lowrank.py::test_lowrank_adjoint[disk-maxwell]' \
  tests/unit/test_gia_eliminated_lowrank.py::test_lowrank_gravity_control_adjoint -q
mpiexec -n 4 python -m pytest \
  'tests/unit/test_gia_eliminated_lowrank.py::test_lowrank_adjoint[memory-maxwell]' \
  'tests/unit/test_gia_eliminated_lowrank.py::test_lowrank_adjoint[memory-backstress]' -q
```

An existing separate regression,
`test_gia_dt_on_tape.py::test_taylor_and_replay_across_a_change_of_dt`, fails
with `DIVERGED_LINEAR_SOLVE` in this environment on both this worktree and the
untouched `02a75382` checkout. It is recorded as a baseline solver-convergence
failure rather than counted as a passing check of this port.

Codex assisted.
