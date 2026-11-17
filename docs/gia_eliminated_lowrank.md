# Material elimination with low-rank DtN

The opt-in combination now keeps displacement, potential and physical global
scalars in the mixed Newton vector, with material histories stored separately.
The DtN modal multipliers are eliminated too. The default representation of a
condensed space remains `"multiplier"` for compatibility; select low-rank
explicitly:

```python
Z, layout = self_gravitating_gia_space(
    mechanics_mesh, potential_mesh, gravity_bcs=gravity_bcs,
    n_internal_variables=2, condense_internal_variables=True,
    internal_variable_family="Quadrature", internal_variable_degree=4,
    dtn_representation="lowrank",
    # existing fluid_core, rotation, centre_of_mass and sea_level switches
)
solver = SelfGravitatingGIASolver(
    Function(Z), approximation, layout=layout, dt=dt,
    time_scheme="bdf2", constitutive_method="direct", quad_degree=4,
    solver_parameters="iterative", **existing_gia_options,
)
```

The layout carries the representation to the solver automatically. Linear
`constitutive_method="pointwise"` also supports low-rank DtN. The existing
coupled DG and multiplier routes remain available.

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

With a moving coastline, a tangent frozen at the zero replay guess failed to
converge during a Taylor perturbation. The direct low-rank route therefore
selects `gia_rebuild="always"` by default for a moving coastline, refreshing
preconditioner blocks and the Real complement at each Newton iteration.
Explicit user rebuild settings still take precedence. Fixed-coastline runs
retain the existing per-solve/lagged policy.

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
first reverse derivatives remain the supported direct-path scope. Production
restart files, 3D nonlinear validation, memory/performance scaling, adaptive
step selection, strong/weak displacement boundaries and second derivatives
remain separate work.

Codex assisted.
