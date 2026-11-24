# Material elimination with low-rank DtN

The Quadrature history family keeps displacement, potential and physical
global scalars in the mixed Newton vector, with material histories stored
separately. With the low-rank DtN representation (the default) the DtN modal
multipliers are eliminated too:

```python
solver = SelfGravitatingGIASolver(
    u, psi, approximation, dt=dt, gravity_bcs=gravity_bcs,
    internal_variable_family="Quadrature", quad_degree=4,
    dtn_representation="lowrank",
    couplings=couplings,  # FluidCore, Rotation, CentreOfMass, SeaLevel
    time_scheme="bdf2", solver_parameters="iterative",
)
```

The coupled DG history family and the multiplier representation remain
available.

## Operator, preconditioner and reverse calculation

After the local material solve, the retained tangent has the form

$$
J=\begin{pmatrix}
 A_{uu} & A_{u\psi} & B_u\\
 A_{\psi u} & A_{\psi\psi}+\theta C_D^T W C_D & B_\psi\\
 C_u & C_\psi & D
\end{pmatrix},
\qquad \theta=\frac{f B_\mu}{\Lambda}.
$$

Here the final block contains core pressure, rotation, centre-of-mass and
sea-level-shift fields, when enabled. Their count does not increase with DtN
truncation. The boundary mode rows $C_D$ and weights $W$ are the existing
low-rank DtN representation, applied in factored form.

`ReducedGIABlockPC` assembles the four displacement/potential blocks directly
and puts the low-rank update in its potential block. It reuses the existing
`LowRankPotentialPC` Woodbury correction and the retained-field solve used by
`CondensedBlockPC`. It performs no Slate material elimination and forms no
constitutive derivative-matrix compositions. The outer matrix-free operator
and the preconditioner both carry the DtN correction. When rigid modes are
declared, the inner Krylov action projects out the declared left kernel just
as the outer augmented operator does; fieldsplit still uses the assembled
nest. On curved facets rigid rotations are only approximate discrete kernels.
Merely attaching the basis to the displacement diagonal left the two-rank
moving-coastline solve stagnating; matching the outer projection resolved it.

The displacement block of `ReducedGIABlockPC` holds the whole consistent
tangent, the weight of the ocean column of the sea-level load included. That
term cancels the restoring force on a rigid translation of the mantle with its
induced potential, so with a large ocean the retained system has $d$
eigenvalues close to zero. The `Real` rows of `CentreOfMass` fix the
translation in the full system, but the retained solve does not see them. The
retained solve therefore deflates the translations: with
$Z=[(e_k,\psi_k)]$, $\psi_k=-(A_{\psi\psi}+B)^{-1}A_{\psi u}e_k$ and
$E=Z^TAZ$, it solves $PA\tilde x=Pb$ with $P=I-AZE^{-1}Z^T$ and returns
$x=Qb+(I-QA)\tilde x$, $Q=ZE^{-1}Z^T$. This identity is exact, so the
converged block-0 solution does not change; the Krylov method no longer
sees the near-null directions. The option `condensed_deflate_translations`
(default true) switches it off.

The direct constitutive block owns forward/replay and the first reverse solve.
Its transpose matrix receives the same symmetric DtN correction. Because that
term is outside UFL, the gradient additionally includes

$$
-\frac{\partial\theta}{\partial c}\,
 \lambda_\psi^T C_D^T W C_D\,\psi.
$$

Each replay/reverse operation binds the DtN prefactor and preconditioner cache
to the saved private Function coefficients and $h/\alpha$ for that step, then
restores the live context. A private solve counter also forces cache refresh
when controls change while the timestep stays constant. Replay uses a zero algebraic initial guess rather
than the shared workspace left by a later reverse step. This changes iteration
work, not the converged timestep map.

DtN scaling Function controls must be scalar Real Functions. Global gravity
parameters should live on the potential mesh, whose boundary extends outside
the mantle submesh. Geometry and mode rows remain fixed. Material Constants
remain fixed inputs of this first-reverse implementation.

## Why group displacement and potential before the Real fields?

Write $A$ for the retained displacement/potential block, $r$ for the Real fields:

$$
\begin{pmatrix} A&B\\C&D\end{pmatrix}
\begin{pmatrix}\delta x\\\delta r\end{pmatrix}
=\begin{pmatrix}b_x\\b_r\end{pmatrix},\qquad
S_r=D-CA^{-1}B.
$$

The core-pressure and centre-of-mass constraints have zero diagonal blocks.
Their response therefore comes through the displacement/potential solve;
solving $D$ alone is not a viable independent scalar solve. Grouping
$(u,\psi)$ provides an approximation to their mutual gravity/mechanics feedback
inside $A^{-1}$, then a small dense Schur complement handles the global
constraints. The outer FGMRES still couples every field. This grouping is a
preconditioning choice, not a physical sequence or a requirement to solve
$A$ to high accuracy.

A four-step nonlinear Maxwell experiment compared three inner choices on the
same curved 2D annulus, with two material modes, variable BDF2 steps and five
physical Real fields:

| Inner displacement/potential choice | Outer iterations | Block applications | Total inner iterations | Warm mean seconds/step |
| --- | ---: | ---: | ---: | ---: |
| Coupled FGMRES, relative tolerance $10^{-4}$ | 37 | 57 | 818 | 0.514 |
| Coupled FGMRES, relative tolerance $10^{-2}$ | 63 | 83 | 782 | 0.549 |
| One multiplicative displacement/potential sweep | 248 | 268 | 268 | 0.587 |

All choices took 12 Newton iterations in total and agreed in final mixed state
to relative coefficient-vector error below $8\times10^{-16}$. The additional
20 block applications are the five Real-response columns built once per step.
The potential Woodbury columns were built only once in each run. Warm means
exclude the first step, which includes setup/compilation. Run
`python tests/benchmarks/gia_reduced_block.py` to reproduce.

Repeating the experiment before and after the retained-kernel projection gave
identical work counts. Timings varied; both runs are saved in
[the raw results](gia_reduced_block_results.json).

The single sweep can reduce total inner work, but spends many more outer
iterations and was slower here. A looser coupled solve was competitive. These
small-mesh results do not select a production tolerance: refinement, 3D,
stronger gravity feedback and moving coastlines can change the balance.
Wall times are illustrative, not isolated timing measurements. The default
remains the existing $10^{-4}$ tolerance.

A moving coastline or a power law makes the Jacobian depend on the state.
On this route the preconditioner blocks and the Real complement then follow
the default rebuild rule of the solver, `gia_rebuild="lagged"`: they are built
again every `gia_rebuild_every` nonlinear solves (default 5) and at a change
of the time step. A taped replay starts each step from the saved start of the
forward step, so it repeats the Newton iterations of the forward run under
this rule. A caller can select `per_solve` or `always` with `gia_rebuild`.

## Validation and remaining scope

`tests/unit/test_gia_eliminated_lowrank.py` contains independent multiplier
comparisons, nonlinear Maxwell/backstress adjoints, disk replay, a gravity
Function control, moving-coastline checks and a no-Real-field iterative solve.
The multiplier comparisons include displacement, potential, physical Real
fields and material histories over unequal timesteps. The reverse objective
includes displacement, potential and histories. The complete combined suite
has 13 passing serial checks. Two-rank tests pass
for the moving-coastline adjoint, Maxwell disk replay and the gravity Function
control; fixed-coastline Maxwell/backstress RAM adjoints also pass on four
ranks. MPI validation and commands are recorded in the main
[BDF2 notes](gia_bdf2.md).

Natural displacement boundaries, prescribed BE/BDF2 steps, fixed geometry and
first reverse derivatives remain the supported direct-path scope.
Production-driver selection and restart support (`examples/common/gia_benchmark_bdf2.md`
of the giamip repository) are
now available. Full 3D nonlinear validation, memory/performance scaling, adaptive
step selection, strong/weak displacement boundaries and second derivatives
remain separate work.

Codex assisted.
