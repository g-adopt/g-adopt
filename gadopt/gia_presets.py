r"""Solver presets and constants of the self-gravitating GIA solver.

The PETSc option dictionaries of `gadopt.gia_solver.SelfGravitatingGIASolver`:
`selfgrav_dtn_schur_solver_parameters` and its low-rank variants (2-D, direct
on block 0 of the two-block Schur split) and
`selfgrav_dtn_iterative_solver_parameters` (3-D, iterative), which the solver
calls itself with the counts it knows and the caller's `preset_options`. Also
the row-scale floor `NULL_COUPLING_ROW_SCALE` and the Earth's non-dimensional
rotation rate `OMEGA_SQ_EARTH`.
"""

from .solver_options_manager import gamg_parameters

__all__ = [
    "DEFAULT_DISPLACEMENT_PC",
    "NULL_COUPLING_ROW_SCALE",
    "OMEGA_SQ_EARTH",
    "resolve_dtn_representation",
    "selfgrav_dtn_iterative_solver_parameters",
    "selfgrav_dtn_lowrank_direct_solver_parameters",
    "selfgrav_dtn_schur_lowrank_solver_parameters",
    "selfgrav_dtn_schur_solver_parameters",
]


NULL_COUPLING_ROW_SCALE = 1.0
r"""The row scaling `theta_psi` and `theta_rot` fall back to at `B_mu = 0`.

Both scalings are `B_mu` times something, and both multiply *whole residual
rows*, so following `B_mu` to zero deletes the potential equation, every DtN
constraint row and the rotation row rather than decoupling them: the Jacobian
acquires 23 identically zero rows on the 2-D prototype and the solve fails with
`DIVERGED_LINEAR_SOLVE`. `B_mu = 0` is a *supported* configuration - it is how
the null-coupling gate switches the two body forces off while leaving the
potential driven by the real divergence source - so the scalings are floored
here instead.

Any nonzero value would do, and that is the point: the symmetry condition the
scalings are derived from is `0 = 0` at `B_mu = 0`, because the block it
constrains is itself zero. `1.0` is the value that needs no bookkeeping -
`theta_psi` becomes `scaling_factor / Lambda`, the scaling the potential row
would carry in a system with no mechanics at all.

`gadopt.gia_solver.SelfGravitatingGIASolver.row_scale_B_mu` applies it.
"""


OMEGA_SQ_EARTH = 1.566e-3
r"""The squared rotation rate, non-dimensionalised by `g_bar / L`.

    Omega^2 L / g_bar = (7.292e-5)^2 * 2.891e6 / 9.815 = 1.566e-3

for the Earth's rotation rate, the mantle depth `L = D = 2891 km` and
`g_bar = 9.815`. It appears nowhere in either design document as a number,
which is why it is here: `theta_rot` depends on it, the symmetry test depends
on `theta_rot` being an *asserted* constant, and a fitted `Omega^2` would
silently absorb a sign error in the rotational closure exactly as a fitted
`theta_psi` absorbs one in the self-gravity coupling.

**Its smallness is not a statement about how much rotation matters**, and the
module docstring records at length why: `C - A` is proportional to `Omega^2`
too, so the `Omega^2` cancels out of the closure, the loop gain is the O(1)
ratio `k_T/k_s`, and switching the rotational body force off changes `|m|` by
32 % on the benchmark. Do not use `Omega_sq << Lambda` to justify a tolerance.

It belongs on `BaseGIAApproximation` next to `B_mu` and `self_gravity_number`
once that class grows a rotation section; until then the solver takes it as a
keyword defaulting to this value.
"""


selfgrav_dtn_schur_solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-10,
    "pc_type": "python",
    "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
    "dtn_pc_fieldsplit_schur_fact_type": "full",
    "dtn_fieldsplit_0_ksp_type": "preonly",
    "dtn_fieldsplit_0_pc_type": "python",
    "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "dtn_fieldsplit_0_assembled_pc_type": "lu",
    "dtn_fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
    "dtn_fieldsplit_1_ksp_type": "gmres",
    # **A stopping test, because PETSc's generic default is not one here.**
    # This was the only dictionary in the tree that left block 1 at PETSc's
    # `rtol 1e-5` with no iteration cap, while the iterative preset, both of
    # b1's paths, b4's and the B2 spike all pin it at 1e-4/200. Block 1 carries
    # the 72 DtN multipliers the geoid is read from as well as the rotation
    # scalars, and its rows span five orders of magnitude, so a generic
    # relative tolerance on it means different things for different rows.
    #
    # **It was investigated as the cause of B4's 140x error and EXONERATED BY
    # MEASUREMENT** - `max_it 3` (binding: every solve exiting DIVERGED_ITS at
    # 6-8e-01 relative) and `rtol 1e-2` both return |m| = 0.0121270, identical
    # to an `rtol 1e-12` control to seven digits, because block 1 is a
    # preconditioner application inside a *flexible* outer Krylov and degrading
    # it costs outer iterations (2 -> 25) rather than accuracy. **This change is
    # not the fix for that; nobody should read it as one and stop looking.**
    "dtn_fieldsplit_1_ksp_rtol": 1e-4,
    "dtn_fieldsplit_1_ksp_max_it": 200,
    # The dense Schur complement of the `Real` block, built once by block-0
    # solves and factorised. Block 0 is an exact LU solve here, so the
    # complement is exact and GMRES on block 1 converges in one iteration.
    # Without a preconditioner, GMRES on block 1 must resolve rows that span
    # several orders of magnitude. With the centre-of-mass rows, which have a
    # zero diagonal, that took 66 block-0 solves per warm step against 9 with
    # the complement (`NOTES/findings/FINDING-centre-of-mass-frame.md`). The build
    # costs one block-0 solve per `Real` column, which is cheap at 2-D size.
    "dtn_fieldsplit_1_pc_type": "python",
    "dtn_fieldsplit_1_pc_python_type": "gadopt.DtNMultiplierDenseSchurPC",
    # **Auditability, and it is not optional instrumentation.** GMRES stops on
    # its recurrence estimate of the residual, not on a recomputed one; under
    # loss of orthogonality that estimate can drift below the truth and the KSP
    # reports CONVERGED_RTOL on a residual that has barely moved. The true
    # residual monitor is the only thing that distinguishes that from real
    # convergence after the fact, and a solve whose exit status cannot be
    # audited afterwards is how a fortnight went into a defect that no log
    # recorded. `snes_converged_reason` is here for the same reason: this
    # system is linear, so anything other than one Newton step is a signal, and
    # that signal was unreadable in every run this preset produced.
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "snes_converged_reason": None,
}
"""**2-D ONLY. Do not use this in 3-D** - see `selfgrav_dtn_iterative_solver_parameters`.

The 2-D development default: a direct solve on everything but the multipliers.

Monolithic `aij` assembly is impossible with `Real` blocks
(`firedrake/assemble.py`, "Monolithic matrix assembly not supported for systems
with R-space blocks"), so "direct LU" for this system means LU on the *first
block* of `DtNTwoBlockSchurPC`'s two-block Schur split, with the multiplier
Schur complement taken by GMRES. Spike S2 measured this dictionary converging
in a single FGMRES iteration on the coupled space, at 1, 2 and 4 ranks.

Block 1 is preconditioned by `gadopt.DtNMultiplierDenseSchurPC`, the dense
Schur complement of the `Real` block. With LU on block 0 that complement is
exact, so it is the natural pairing for this preset. It is necessary once the
`Real` block contains Lagrange multiplier rows with a zero diagonal (the core
pressure, the centre-of-mass frame), where unpreconditioned GMRES needs many
more block-0 solves. `selfgrav_dtn_iterative_solver_parameters` still defaults
to no block-1 preconditioner and takes the dense complement through
`multiplier_pc`.

**`fieldsplit_0` is not the potential block.** The preconditioner merges every
non-`Real` sub-field into block 0, so here that is displacement + internal
variables + potential together - a saddle-ish coupled operator, not the scalar
Laplacian that `gravity_solver.py`'s `SPDAssembledPC` + GAMG preset is tuned
for. That preset is **not** transferable and must not be copied across.

This has no 3-D successor and is not a fallback there: at production size
block 0 is tens of millions of displacement dofs plus hundreds of millions of
DG1 tensor dofs plus the potential. The pointwise history layout removes those
unknowns, but it is not exact elimination of the mixed weak DG formulation on
curved cells. Slate local condensation cannot represent the general weak and
facet forms required by GIA and must not be retried. The remaining candidate is
the segregated Picard iteration as the preconditioner.

Note also that reaching for a plain `fieldsplit` while debugging brings back
PETSc's 128-field cap, which registering the two blocks as index sets is what
avoids.
"""


selfgrav_dtn_schur_lowrank_solver_parameters = {
    key: value for key, value in selfgrav_dtn_schur_solver_parameters.items()
    if key != "dtn_fieldsplit_1_pc_python_type"
} | {"dtn_fieldsplit_1_pc_type": "none"}
"""The 2-D direct preset for the low-rank DtN path WITH `Real` sub-fields.

The same dictionary as `selfgrav_dtn_schur_solver_parameters`, with no
preconditioner on block 1. The low-rank adjoint and tangent solves in
`gadopt.dtn_coupled_adjoint` pass the forward solver parameters to a solve on
an assembled matrix. Any block-1 preconditioner other than `none` makes
`DtNTwoBlockSchurPC.initialize` ask Firedrake for a sub-DM of the `Real` fields,
and on that matrix the request ends in a segmentation fault
(`tests/unit/test_gia_gravity_adjoint_lowrank.py`, the fluid-core adjoint and
tangent tests). `DtNTwoBlockSchurPC.initialize` records the mechanism. The
low-rank path carries at most the core pressure and the rotation scalars in its
`Real` block, so GMRES without a preconditioner there costs little.
"""


selfgrav_dtn_lowrank_direct_solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-11,
    # No `Real` sub-fields, so there is nothing for `DtNTwoBlockSchurPC` to
    # split off and no two-block Schur. The whole matrix-free operator `A + B`
    # goes to `firedrake.AssembledPC`, which assembles the STOCK form `A` out of
    # the operator context (the low-rank `B` is a `Python` update that carries
    # no form and is left out of the assembly), LU-factorises it, and lets the
    # flexible outer Krylov take up the rank-`k` difference in a few iterations.
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps",
    "ksp_converged_reason": None,
    "snes_converged_reason": None,
}
"""The 2-D direct preset for the low-rank DtN path with NO `Real` sub-fields.

The low-rank representation eliminates the DtN multiplier `Real` fields, so with
rotation off the mixed space carries no `Real` block at all. `DtNTwoBlockSchurPC`
then has nothing to split off and refuses, which is correct: the whole operator
assembles as one block. This preset LU-factorises that block through
`firedrake.AssembledPC` and takes the low-rank `B` on the outer FGMRES. With one
or more `Real` fields (rotation on) the two-block Schur presets still apply.
"""


# The GAMG settings for the displacement and potential blocks used to be a
# sixth literal copy of the six that `gadopt.solver_options_manager` now owns --
# the same ones the Stokes velocity block and `GravitySolver` use. The sweep
# builder below calls `gamg_parameters("assembled_")` directly.


def _prefixed(d, prefix):
    return {prefix + k: v for k, v in d.items()}


#: The preconditioner the displacement split of block 0 runs by default.
#:
#: `gadopt.NearlyIncompressibleAssembledPC` seeds GAMG with the six rigid-body
#: modes **and** the low-degree divergence-free fields. The internal-variable
#: stress carries a volumetric penalty `lambda * integral (div u)(div v)`, and
#: the slow modes of that operator sit in the divergence-free space once the
#: effective bulk/shear ratio `bulk_shear_ratio * (1 + dt/tau)` is large. GAMG
#: builds coarse spaces that reproduce whatever near-nullspace it is handed, so
#: the rigid modes alone leave it nothing to coarsen the slow modes onto.
#:
#: Measured on Gadi: two 500 yr restart steps on `b2_coarse_ar7.msh`, 96 ranks,
#: CG3 displacement, DG2 internal variables, bulk/shear 100, block-0 cap 400,
#: `block0_rtol` 1e-4, dense Schur complement on the `Real` block. Job
#: `178765557`, rigid modes: 33 699 GAMG V-cycles, 64 of 124 block-0 solves
#: stopped at the cap, 565 s for the warm step. Job `178765558`, this class:
#: 16 853 V-cycles, 8 of 102 at the cap, 270 s. The two states after two steps
#: agree to 5e-7 in the displacement norm, so this is a preconditioner change
#: and nothing else.
#:
#: Those two jobs ran four iterations of CG on the split above this class, at
#: PETSc's GMRES restart of 30 on the block-0 FGMRES. The preset now writes one
#: GAMG V-cycle there (`u_ksp_max_it`, default `0`) at a restart equal to the
#: block-0 cap, which is faster by a factor 1.94 on the polar-motion case of
#: the Spada benchmark and 1.44 on the cap case (jobs `179496714` and
#: `179483713` against `179423870` and `179423871`, 96 ranks, the full 138-step
#: ladder, every printed benchmark number unchanged). The two control runs were
#: made from a package that predates `c24685ac`; job `179482312` is what ties
#: their walls to this tree, at 73.7 s per 100 yr step against the control
#: arm's 72.8 and 25.0 against the option arm's 25.1, each the mean of the warm
#: steps as every arm figure in this record is. The choice of modes this
#: constant makes is unaffected: both arms of the pair above run the same split.
#:
#: Pass `u_pc="gadopt.RigidBodyAssembledPC"` to reproduce a run made with the
#: rigid modes alone.
DEFAULT_DISPLACEMENT_PC = "gadopt.NearlyIncompressibleAssembledPC"

#: The block-1 preconditioner that forms the exact Schur complement and factors
#: it. Named here because the iterative preset both selects it by default on the
#: low-rank representation and treats it specially: it is the one block-1
#: preconditioner that is an exact inverse, so the block-1 KSP above it is
#: `preonly`.
_DENSE_MULTIPLIER_PC = "gadopt.DtNMultiplierDenseSchurPC"


def _displacement_krylov(max_it: int, rtol: float) -> dict:
    """The Krylov options of the displacement split, with no option prefix.

    The default is `max_it = 0`: one GAMG V-cycle per block-0 iteration, with
    no Krylov method on the split. Block 0 is a flexible FGMRES, so its
    preconditioner can differ from one application to the next and a truncated
    Krylov solve on the displacement split is also a valid preconditioner
    inside it. That is the option `max_it > 0` buys: the split becomes a few
    steps of CG with GAMG, which brings the block-0 FGMRES to `block0_rtol` in
    far fewer of its own iterations and lets the tolerance decide when block 0
    stops instead of the cap. It costs wall clock at the restart the preset
    ships; see `u_ksp_max_it` in
    `selfgrav_dtn_iterative_solver_parameters` for the measurement.

    `ksp_converged_maxits` makes PETSc count a solve that reaches `max_it` as
    `CONVERGED_ITS`. The truncation is deliberate here; without the option the
    log fills with `DIVERGED_ITS` lines from a split that did exactly what it
    was asked to do, and the block-0 counters of the campaign then have to be
    read around them.

    Args:
      max_it: the iteration cap on the displacement split. `0`, the preset's
        default, selects one GAMG V-cycle per block-0 iteration
        (`ksp_type preonly`), which is also the route the P3 march ran.
      rtol: the relative tolerance of the truncated CG. It has no effect when
        `max_it` is `0`.

    Returns:
      The Krylov options of one split, for the caller to prefix.
    """
    if max_it <= 0:
        return {"ksp_type": "preonly", "ksp_converged_reason": None}
    return {
        "ksp_type": "cg",
        "ksp_max_it": max_it,
        "ksp_rtol": rtol,
        "ksp_converged_maxits": None,
        "ksp_converged_reason": None,
    }


def resolve_dtn_representation(dtn_representation, *, condensed: bool) -> str:
    """The representation a caller left unset, decided by the layout.

    The default is `"lowrank"` on the full layout and `"multiplier"` on the
    condensed layout for compatibility. Explicit low-rank selection also works
    after material elimination through `ReducedGIABlockPC`. Decided on the
    3-D scan of 2026-09-17 (`NOTES/HANDOVER-2026-09-17.md` section 3.4): the
    low-rank arm reproduces the multiplier arm's state at every truncation,
    its warm step is 0.69 of the multiplier's at truncation 5 and does not
    change with the truncation, and the multiplier arm's setup grows as the
    square of the truncation and does not fit one node at truncation 20.

    Args:
      dtn_representation: `"multiplier"`, `"lowrank"` or `None`.
      condensed: whether the internal variables are eliminated pointwise.

    Returns:
      `"multiplier"` or `"lowrank"`.

    Raises:
      ValueError: an explicit value that is neither name.
    """
    if dtn_representation is None:
        return "multiplier" if condensed else "lowrank"
    if dtn_representation not in ("multiplier", "lowrank"):
        raise ValueError(
            "dtn_representation must be 'multiplier' or 'lowrank' (or None "
            f"for the layout's default), got {dtn_representation!r}.")
    return dtn_representation


def _potential_split(dtn_representation: str) -> dict:
    """Options for the potential split of `gadopt.CondensedBlockPC`'s nest.

    On the multiplier representation the potential block is the assembled
    Laplacian with the DtN boundary terms, which GAMG handles in one V-cycle.

    On the low-rank representation that block also carries
    `B = theta_psi * sum_b C_b^T W_b C_b`, applied in factored form, and GAMG
    alone does not see it: the block is then a Python operator and its
    preconditioner is `gadopt.LowRankPotentialPC`, which applies one V-cycle on
    the assembled part and corrects it with the Woodbury identity. The V-cycle's
    own options move down one prefix, to `lowrank_`, because the class creates
    that Krylov solve itself.

    Args:
      dtn_representation: `"multiplier"` or `"lowrank"`.

    Returns:
      The options of the split, unprefixed.
    """
    if dtn_representation == "multiplier":
        return {"ksp_type": "preonly", "ksp_converged_reason": None,
                "pc_type": "gamg", **gamg_parameters()}
    return {
        "ksp_type": "preonly",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "gadopt.LowRankPotentialPC",
        **_prefixed({"ksp_type": "preonly", "pc_type": "gamg",
                     **gamg_parameters()}, "lowrank_"),
    }


def selfgrav_dtn_iterative_solver_parameters(
    *, condensed: bool = True, block0_rtol: float = 1e-4,
    outer_rtol: float = 1e-6, block0_max_it: int = 200,
    snes_rtol: float = 1e-4,
    outer_atol: float = 0.0, outer_max_it: int = 200,
    outer_accept_capped: bool = False,
    snes_type: str = "newtonls",
    u_pc: str = DEFAULT_DISPLACEMENT_PC,
    multiplier_pc: str | None = None,
    u_ksp_max_it: int = 0,
    u_ksp_rtol: float = 1e-2,
    block0: str = "condensed",
    dtn_representation: str | None = None,
    ainvb: bool | None = None,
    n_real: int | None = None,
    dense_schur_max_rows: int = 16,
) -> dict:
    r"""The 3-D configuration that works, and the one the measurements select.

    **This exists because its absence cost two production jobs in one day.**
    Until now `gadopt` shipped exactly one preset for this system,
    `selfgrav_dtn_schur_solver_parameters`, whose own docstring says it has no
    3-D successor - and the working configuration lived in a spike. B1 was
    OOM-killed after 90 minutes and a phase-tolerance ladder ran six hours into
    a walltime kill, both on the direct default with block 0 capping at 200
    iterations. Every driver was rediscovering this dictionary and two of them
    got it wrong.

        outer FGMRES
         +- DtNTwoBlockSchurPC, schur_fact_type full with the cached apply,
         |  which is the default on a narrow `Real` block; lower otherwise
             +- block 0, condensed layout with multiplier DtN: FGMRES + multiplicative fieldsplit
             |   +- u  : one V-cycle of NearlyIncompressibleAssembledPC
             |   +- psi: SPDAssembledPC + GAMG   (GravitySolver's preset)
             +- block 0, condensed layout with low-rank DtN:
             |  ReducedGIABlockPC, assembled (u, psi) nest with low-rank potential PC
             +- block 0, full layout, block0="condensed" (the default):
             |  preonly + gadopt.CondensedBlockPC, which eliminates `M` once
             |  with Slate and solves the assembled (u, psi) system
             |   +- FGMRES rtol block0_rtol + multiplicative fieldsplit
             |       +- u  : one GAMG V-cycle on S_uu, near-incompressible
             |       |         modes
             |       +- psi: one GAMG V-cycle on A_psipsi
             +- block 0, full layout, block0="pair": FGMRES + multiplicative
             |  fieldsplit
             |   +- (u, M): InternalVariableSCPC, which eliminates `M` on
             |   |          every inner iteration; one GAMG V-cycle on the
             |   |          condensed operator under `condensed_field_`
             |   +- psi   : SPDAssembledPC + GAMG
             +- block 1, low-rank representation: preonly +
             |  gadopt.DtNMultiplierDenseSchurPC, the exact complement formed
             |  once and factored (`multiplier_pc`)
             +- block 1, multiplier representation: GMRES on the Real block,
                pc_type none; under `ainvb` the block-1 KSP is never entered

    **This preset carries no iteration count of its own, and the "flat at 3"
    that used to stand here was another configuration's.**  That 3 was measured
    with a 3-D probe driver of the Spada benchmark (in git history at commit
    `a8df4939`), in its `cond` configuration: an *algebraic* Schur split on the **uncondensed** space,
    `-pc_fieldsplit_type schur` with `pc_fieldsplit_0_fields "1"` (`m`, inverted
    exactly by `bjacobi`/ILU(0), which is exact on a block-diagonal matrix) and
    `pc_fieldsplit_1_fields "0,2"` (`u` and `psi`).  The counter that read 3,
    flat across `--coarse`, `--medium` and `--fine` - 3.35e6 to 1.23e8 dofs, 37x,
    256 ranks - is `dtn_fieldsplit_0_ksp`, i.e. **the layer above that Schur
    complement**, and it is the u-psi coupling alone precisely because `m` has
    been eliminated below it.

    This preset has no such layer.  With `condensed=True` there is no `m` field
    to eliminate, so its `dtn_fieldsplit_0_ksp` *is* the `[u, psi]` sweep -
    structurally `b2_probe`'s inner `dtn_fieldsplit_0_fieldsplit_1_ksp`, which
    measured **174-388 at `--coarse`, 7-19 at `--medium`, 11-296 at `--fine`**:
    non-monotone and large, because the difficulty did not go away when `m` was
    eliminated, it moved down one level into a single-physics AMG problem on
    A2's anisotropic lithosphere (`NOTES/HANDOVER-PETSC.md` §§1-2).

    So: **budget against the 174-388 band, not against 3.**  A production
    estimate built on the 3 is an estimate for a configuration this dictionary
    does not implement.  The uncondensed alternative below was separately
    measured at outer FGMRES 5 with a block-0 median of 9 (`--coarse`) and 10
    (`--medium`).

    Two traps, both of which have already been paid for:

    - **`fieldsplit_N_` indexes the SPLIT, not the field.** The sweep order is
      `m, u, psi` while the mixed space's field order is `u = 0, m = 1,
      psi = 2`, so `pc_fieldsplit_0_fields: "1"` makes split *0* the *m* field
      and the options under `fieldsplit_0_` then go to `m`. Writing the two out
      by hand put GAMG on the DG1 tensor block and `bjacobi/ilu` on the
      displacement; it does not raise, does not warn, and shows up only as
      block 0 hitting its iteration cap. The sweep is built from one list here
      so the two cannot disagree.
    - **The multiplier block takes no matrix-based preconditioner.** `jacobi`
      needs `MatGetDiagonal`, which on a matfree block goes through TSFC's
      `diagonal=True` path and cannot handle `Real` arguments, and
      `AssembledPC` is structurally unavailable for the same family of
      reasons. What works there is a python PC that forms its own data from
      the operator, which is what `gadopt.DtNMultiplierDenseSchurPC` does with
      one `A.mult` per column. On the multiplier representation that block is
      about 76 columns wide at L = 5 and the default leaves it at
      `pc_type: none`, so its iteration counts are an **upper bound** rather
      than a tuned figure.

    Args:
      condensed: whether the history is eliminated at the quadrature points
        (the Quadrature family of the GIA solver, which passes this itself).
        The internal variable is ~85 % of block 0. On the condensed layout it
        is not in the space, and block 0 is a two-way sweep over `u` (through
        `u_pc`) and `psi`. On the uncondensed layout it is field 1, and `M` is
        eliminated cell by cell with Slate on either block-0 route below.
        **Keep this argument and the solver's flag in step** - that is the
        whole reason it is an argument rather than an assumption.
      block0: which block-0 route the uncondensed layout takes. It has no
        meaning on the condensed layout, where there is no `M` to eliminate,
        and a non-default value there raises rather than being ignored.
      dtn_representation: `"multiplier"`, `"lowrank"` or `None`. `None` is
        the default and resolves through `resolve_dtn_representation`:
        low-rank on the full layout, multiplier on the condensed one. The
        value must agree with the space and the solver; the solver refuses a
        mismatch.

        `"condensed"`, the default, puts `gadopt.CondensedBlockPC` on block 0
        behind a `preonly` KSP. That class eliminates `M` **once** per block-0
        application and solves the assembled `(u, psi)` system
        `[[S_uu, A_upsi], [A_psiu, A_psipsi]]` with its own FGMRES (options
        under `condensed_`) around a multiplicative `u`/`psi` fieldsplit. So
        `M` - about 85 percent of the block-0 unknowns - rides in no Krylov
        vector at all.

        `"pair"` is the earlier route: block 0 is an FGMRES over all three
        fields around a two-split sweep whose split 0 is the pair `(u, M)`
        under `gadopt.InternalVariableSCPC`, which eliminates `M` on every
        block-0 inner iteration (about 900 times per production step) and runs
        the displacement Krylov solve of `u_ksp_max_it` with GAMG on the exact
        condensed displacement operator (options under `condensed_field_`), so
        one GAMG V-cycle by default. It is kept because the T2 Gadi
        measurements were made on it - 472 s per 500 yr step at 96 ranks, job
        `178936321`, two steps from the P3 20 kyr state at block-0 cap 200 and
        PETSc's GMRES restart of 30 with the four-iteration CG on the
        displacement split, which `u_ksp_max_it=4` reproduces - and a job that
        compares the two arms selects one by this flag alone. Both routes take
        their displacement split from the same argument, which is what keeps
        such a comparison a measurement of the block-0 route.
      block0_rtol, outer_rtol, block0_max_it, snes_rtol: the tolerances.
        `block0_rtol` is 1e-4 because the dense complement on block 1 is only
        as linear as the block-0 solve that builds its columns: at 1e-2 that
        arm stagnates, with 642 non-convergent block-0 calls and a wall worse
        than no block-1 preconditioner at all (Gadi job 176078939). At 1e-4 it
        is 19 block-0 calls and 69 s per marching step against `none`'s 354
        and 1036 s (job 176103130, medium rung, L = 5, 104 ranks). See
        NOTES/fastdtn/HANDOVER.md. On the multiplier and condensed arms, where
        the complement is off, the same tolerance is a choice and not a
        measurement: the `none` arm costs 354 block-0 calls and 1036 s at 1e-4
        (job 176103130) against 265 calls and about 891 s at 1e-2 (job
        176078939), about 15 percent, and one tolerance for the whole preset
        is what keeps the two arms comparable.
        `block0_max_it` caps the block-0 FGMRES at 200 iterations, which is
        what A2's anisotropic lithosphere needs: the condensed `[u, psi]`
        sweep sits in the 174-388 band and a smaller cap binds on every
        application. The cap is not the knob that makes block 0 converge --
        raising it to 400 with one V-cycle on the displacement split bought
        three outer iterations for 28 percent more GAMG V-cycles and 15
        percent more wall (job `178765557` against `178763503`). At the
        restart this preset ships, the cap binds on a small part of the solves
        either way and `u_ksp_max_it` moves that part by about a fifth: 117 of
        523 block-0 solves reached it over the full ladder with one V-cycle on
        the displacement split (job `179496714`) against 98 of 527 with four
        iterations of CG (job `179423870`), never more than two solves in one
        step in either run.
        `snes_rtol` is relative to the norm of the **whole** mixed residual,
        which the mechanics rows dominate; rows scaled by `Omega_sq = 1.566e-3`
        are converged to correspondingly fewer digits, which is a live question
        for the rotational closure.
      outer_atol, outer_max_it, outer_accept_capped: the stopping rule of the
        outer FGMRES besides `outer_rtol`. `outer_rtol` is relative to the
        residual a Newton iteration starts from. When that residual is tiny,
        the relative target can fall below a floor the outer residual cannot
        go under, and the solve runs to its cap. One such floor is the part of
        the residual along a declared transpose nullspace that is not an
        exact left kernel of the discrete operator: the rigid rotation, which
        curved facets annihilate only to about 2e-6 relative, gave a floor of
        about 1.2e-12 on GIAMIP Exp01 at L0 (NOTES/EXP01-HANDOVER.md, section
        17.4 and 18.9, in the giamip-exp01 worktree). The low-rank operator
        removes the declared spaces from its action
        (`gadopt.dtn_coupled.AugmentedImplicitMatrixContext`), so that floor
        does not arise on the low-rank path. `outer_atol` (`ksp_atol`) is an
        absolute floor for any other case: the solve stops at the larger of
        the relative target and this value. It is 0 (off) by default because
        a useful value depends on the scaling of the residual and on the mesh.
        `outer_accept_capped` (`ksp_converged_maxits`) counts a solve that
        reaches `outer_max_it` as converged, so Newton takes its update, which
        has already reduced the residual by several orders, and `snes_rtol`
        ends the step at the next iteration. It needs no scale and catches a
        floor higher than `outer_atol`; with it, a cap near 30 bounds the cost
        of such a solve (normal steps take 2-7 outer iterations per Newton
        iteration). It is off by default, so a solve that cannot reach its
        tolerance still fails loudly unless the caller asks otherwise.
      snes_type: the outer method. `"newtonls"` is the default because
        this preset serves a power-law rheology as well as a Newtonian one.
        With `exponent != 1` the residual is nonlinear and Newton is the method
        that solves it, and the nested preconditioner follows the
        state-dependent Jacobian without any help from the caller: the inner
        fieldsplit keeps its own setup state against the preconditioning
        matrix's object state and re-runs `PCSetUp` inside `apply` after every
        Jacobian reassembly, so every sub-preconditioner's `update` runs once
        per Newton iteration. Measured on the 2-D annulus at exponent 3 and
        transition stress 1e-3: `gadopt.InternalVariableSCPC.assembly_count`
        equals the Newton iteration count of the solve, and the nested route
        reproduces the direct route's residual history to four digits at every
        Newton step (`NOTES/PLAN-POWER-LAW-SELFGRAVITY.md` section 2).
        **For `exponent = 1` pass `"ksponly"`.** The residual is then linear -
        which is the same fact that lets `DtNTwoBlockSchurPC.update` be a no-op -
        and `newtonls` spends one extra residual evaluation per timestep to
        rediscover that. Newton does not pay for a second linear solve here:
        the residual after the first step sits at roundoff, below the
        `snes_atol` of either preset, so SNES exits `CONVERGED_FNORM_ABS` at
        iteration 1 with one outer linear solve per step (measured on every
        Newtonian configuration of both presets, 2026-09-14). So the saving is
        one residual evaluation and the linesearch around it, which is small
        and real. Note that `snes_rtol` is then inert and `outer_rtol` alone
        controls the accuracy. The reverse
        combination, `"ksponly"` with `exponent != 1`, is refused by
        `SelfGravitatingGIASolver.set_solver_options`: one linear solve on a
        nonlinear residual reports `CONVERGED` and returns a state that is
        wrong by the whole nonlinearity.
        One trap carries over from the Newtonian case unchanged. The absolute
        `snes_atol` of 1e-10 in `newton_stokes_solver_parameters` stops Newton
        at iteration zero for any configuration whose whole forcing is smaller
        than that, and it stops it for a power law exactly as it does for a
        Newtonian one. This preset names `snes_atol` (1e-15) and so overrides
        it; `selfgrav_dtn_schur_solver_parameters` does not name the key and so
        inherits it.
      n_real: the number of `Real` rows in the mixed space, which a caller
        takes as `len(layout.real_fields)`. When it is given, the sentinel
        `multiplier_pc=None` chooses the block-1 preconditioner from this width
        and not from the name of the DtN representation; see
        `dense_schur_max_rows` for the limit and for what is measured. When it
        is `None`, the default, the choice is made by `dtn_representation`
        exactly as it always was, so **every caller that passes no count keeps
        its dictionary**.

        The width is the direct quantity and the name of the representation is
        an indirect one. The cost of `gadopt.DtNMultiplierDenseSchurPC` is `n`
        block-0 solves for each build, so in block-0 applications per step:

            full  + Krylov on block 1 : 2*K + I1 + n*b
            lower + preonly + dense   : K'      + n*b
            ainvb                     : K       + n*b

        with `K` the outer iteration count, `K'` the larger count that `lower`
        pays, `I1` the block-1 iteration count and `b` the number of builds in
        a step. `n` enters in one term only, and the dense complement pays for
        itself while `n*b` stays below the block-1 iterations it removes, which
        puts the break-even near `n = K`, at 3 to 12 outer iterations per step.
        The name tracks `n` on this branch only because the two cases here are
        1 or 4 rows on the low-rank representation and about 76 on the
        multiplier one. They separate as soon as another kind of `Real` row
        exists: `sghelichkhani/sea-level` adds three centre-of-mass rows and a
        sea-level `Shift`, which make a low-rank block 5 or 8 rows wide and a
        multiplier one 80 at L = 5.
      dense_schur_max_rows: the widest `Real` block the preset will form the
        exact Schur complement on when `n_real` is given - through the cached
        apply of `gadopt.DtNTwoBlockSchurPC` by default, and through
        `gadopt.DtNMultiplierDenseSchurPC` when the caller names
        `ainvb=False`. One limit covers both because both objects are that
        same complement and both cost `n` block-0 solves to form. It has no
        meaning without a count.

        **What is measured.** 4 rows win: 88.2 s per step against 269 s with
        block 1 unpreconditioned (arm B4 against arms B0 and C0, job
        179385036, `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 24). With
        the cached apply at 4 rows (core pressure and the three
        centre-of-mass rows), the frame costs 1.20 times a run without it,
        against 6.33 times on the delegating path (Spada cap, jobs 179535948
        and 179535947 against 179271032 and 179271031). The complement there
        has a condition number of 1.019 after `_DenseLU` scales it by its
        diagonal, so the rows are almost uncoupled.

        5 rows (4 rows plus the sea-level `Shift`) also win with the cached
        apply. On the 78 km Martinec mesh a 50 yr step of case C costs 16.0 s
        at 2 outer iterations (job 179535949), and case C and case D march to
        15 kyr on it (jobs 179535950, 179581634 and 179603338). Case B at a
        bulk modulus of 1000 times the shear modulus converges in 2 or 3
        outer iterations with every block-0 solve that forms the cache
        stopped at its iteration cap (job 179535952), so a cache formed from
        capped block-0 solves still works. The 16.0 s compares with 75.3 s
        for a 50 yr step of case B with `DtNMultiplierDenseSchurPC` named
        (job 179449169). That is a different case, and no arm runs the cached
        apply against `ainvb=False` on the same case at 5 rows.

        About 76 rows lose, because a build then costs more than a whole
        outer solve and nobody has measured over how many steps that
        amortises. **No arm measures a width between 6 and 75.** 8 rows
        (core pressure, three rotation rows, three centre-of-mass rows and
        `Shift`) need rotation and sea level in one solver, and no driver
        builds that. So the default 16 is a choice with a margin above the
        widest measured win and not a measurement; the argument exists so
        that a caller can move it without a new release. The dense
        complement at these widths is reached by naming `ainvb=False`, and a
        campaign that wants both arms must name it.
      multiplier_pc: the preconditioner on the block-1 (`Real`) split, named
        as a `pc_python_type` string, or `"none"`. The default `None` is a
        sentinel meaning "the preset chooses". It chooses `"none"` whenever
        the cached apply of `ainvb` owns block 1, which is the default on a
        narrow `Real` block, because nothing enters the block-1 KSP there.
        Otherwise it chooses by `n_real` when a count was given, and by the
        resolved `dtn_representation` when none was:

        * `"gadopt.DtNMultiplierDenseSchurPC"` on the low-rank representation
          with `ainvb=False` named. The `Real` block is then 4 rows with rotation and
          1 without, so one build of the exact complement is 4 block-0 solves,
          and those are the cheap kind: a solve whose right-hand side is a
          column of `A01` takes 63 inner iterations and stops at the cap in 5
          of 628 solves, against 131 to 138 iterations for one carrying the
          mechanics residual (100 yr step, 96 ranks; a labelled pass over the
          production log, `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 4,
          which separates the three block-0 positions and so names no arm).
          The block-1 KSP above it is `preonly`, because the factored
          complement is an exact inverse; the whole arm costs 88.2 s per step
          (arm B4, job 179385036, section 24 of the same record) against 269 s
          for the unpreconditioned block 1 (arms B0 and C0, the same
          section).
        * `"none"` on the multiplier representation. The block is about 76
          columns wide at L = 5, so a build costs more than a whole outer
          solve and has to amortise over a segment of the march before it
          pays. Nobody has measured over how long, so the default stays off
          and a caller who wants it names it.
        * `"none"` under `ainvb`, because the cached apply solves block 1 with
          its own dense factors and a preconditioner there would be built,
          configured and never applied. This case is tested first, so a narrow
          `n_real` cannot override it.

        With `n_real` given, the first two of those three cases are replaced by
        one rule on the width: `"gadopt.DtNMultiplierDenseSchurPC"` when the
        block is between 1 and `dense_schur_max_rows` rows, `"none"` at every
        other width, on either representation. A block of zero rows takes
        `"none"`, because there is nothing for the class to form.

        A named string is taken as written on either representation and at
        every width, and naming one together with `ainvb=True` is refused.

        **The sentinel refuses the dense complement at a loose block-0
        tolerance.** When the preset would choose it and `block0_rtol` is
        looser than 1e-4, the call raises a `ValueError`: the columns of the
        complement are block-0 solves, and at 1e-2 the arm stagnates with 642
        non-convergent block-0 calls and a wall worse than no block-1
        preconditioner at all (Gadi job 176078939). The refusal scopes to the
        preset's own choice and not to the caller's, so
        `multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"` named explicitly is
        accepted at any tolerance, and `multiplier_pc="none"` is the way to
        keep a loose tolerance. The 1e-4 threshold is one measurement at one
        configuration and not a law of the method.

        The block-1 KSP follows from the choice. The dense complement is an
        exact inverse of the block, so it runs under `preonly` and the preset
        writes no block-1 tolerance and no iteration cap: a Krylov method
        there would spend one Schur-complement `MatMult`, and so one block-0
        solve, per extra iteration, which measures 120.4 s per 100 yr step
        against `preonly`'s 99.8 under `full` (arms B1 and B2, job 179385036,
        `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 24, 96 ranks).
        `"none"` and an approximate inverse such as
        `"gadopt.DtNMultiplierDiagPC"` keep the GMRES at `rtol` 1e-4 and 200
        iterations, because neither solves the block on its own.
      ainvb: whether `gadopt.DtNTwoBlockSchurPC` owns the apply and caches the
        columns `Z = A00^{-1} A01`. The default `None` is a sentinel meaning
        "the preset chooses", the convention `multiplier_pc` and
        `dtn_representation` use on this same function. `False` keeps the
        delegating path, whose one block-0 solve per outer iteration comes from
        `dtn_pc_fieldsplit_schur_fact_type lower` instead. With `True`
        the class runs `n` block-0 solves at every operator change, keeps from
        them both `Z` and the exact complement `S = A11 - A10 Z`, and then
        spends ONE block-0 solve per outer iteration while keeping the outer
        iteration count of `full`, which this argument therefore writes above
        it. The two routes are alternatives, not a stack: `lower` buys the
        same single solve by spending more outer iterations.

        **What the preset chooses, and why.** It chooses the cached apply when
        it would form the exact complement at all: when the caller named no
        `multiplier_pc` and the `Real` block is narrow by the rule of
        `n_real` and `dense_schur_max_rows`. One width test decides both
        objects, because both are the same complement of the same block and
        both cost `n` block-0 solves to form, and the cached one adds only `n`
        AXPYs of block-0 length per apply. A caller who named a block-1
        preconditioner keeps the delegating path, `"none"` included, because
        under the cached apply the class they named would be built, configured
        and never applied.

        **What is measured.** Both cases of the Spada benchmark, 138 steps,
        96 ranks, from the preset alone, against the same preset with the
        delegating path: **45 min 44 s against 1 h 18 32** on the polar-motion
        case (job `179511971` against `179496714`) and **45 min 45 s against
        1 h 08 54** on the cap case (`179511972` against `179483713`), with
        every printed benchmark number unchanged - the seven polar-motion
        ratios agree to 5.6e-09 and the cap table to 4.0e-10. At one step size
        with one thing changed, 16.78 s per warm step against 25.6 at
        dt = 100 yr, and 2 outer iterations against 3 (arms Z2 and Z1 of job
        `179510821`). The cache built five times in each full run, one per
        operator change, and the whole run then costs `n * builds + outer
        iterations` block-0 solves, which is what the logs count: 298 on the
        polar-motion case, 20 of them in five four-column builds and 278 in
        the 139 solves at two outer iterations each (job `179511971`), and 283
        on the cap case, where the `Real` block is one row without rotation,
        so a build is one solve and five builds cost five (`179511972`).

        **What is not measured.** No arm runs the cached apply on the
        multiplier representation, where a build is about 76 block-0 solves,
        and the width rule keeps the preset off that path. No 3-D arm runs it
        with a power-law rheology, where the rebuild rule falls to
        `gia_solve_index` and builds once per nonlinear solve; the unit tests
        cover that branch at toy size.

        **A note for whoever couples the sea-level equation.** A moving
        coastline changes the ocean function, which enters the `Real` rows of
        the sea-level constraint and therefore `A01`, `A10` and `A11`, the
        blocks this cache is built from. An ocean function that enters the
        residual as a `Function` needs no action: the solver fingerprints
        every Jacobian coefficient at each `solve()` and invalidates the
        Jacobian itself. One modified through a view of its data, or one that
        reaches the solve only through a hand-supplied `J`, needs
        `invalidate_jacobian`. **A coastline that depends on the solution
        inside one solve, on a Newtonian rheology, is caught by neither**,
        because `operator_version` is published as `None` from the rheology
        alone; such a solver must publish `None` itself, which gives the
        power-law treatment of one build per nonlinear solve. A stale cache
        changes no converged answer, because this is a preconditioner and the
        residual is untouched; it costs outer iterations, quietly while the
        outer solve still converges inside its cap and loudly past it.

        Block 1 is solved by
        the dense factors of `S`, so this argument writes `preonly` and
        `pc_type none` there and writes no block-1 tolerance, iteration cap or
        converged-reason key: no block-1 Krylov solve runs, and a tolerance in
        the log for a solve that never happens is worse than no line at all.
        The cost model of a log therefore becomes
        `block-0 applications = n * builds + outer iterations`.
        `n` is the number of `Real` rows, which is what decides whether the
        option pays for itself: about 4 with `dtn_representation="lowrank"`
        (the three rotation rows and the core pressure) and about 76 with
        `dtn_representation="multiplier"` (72 DtN multipliers at L = 5, plus
        those four). On the low-rank path a build is 4 block-0 solves and the
        option saves one solve per outer iteration from the first step. On the
        multiplier path a build is the same 76 block-0 solves the dense
        complement spends, so the saving has to amortise a build that costs
        more than a whole outer solve.
        The columns are only as good as the block-0 solve that produced them,
        which is the other half of why `block0_rtol` defaults to 1e-4; a
        caller who loosens it loosens the cached complement with it.
        `ainvb=True` together with a named `multiplier_pc` is refused: the cached
        path solves block 1 with its own factors and a preconditioner named for
        that block would be built, configured and never applied.
      u_pc: the preconditioner on the displacement split, condensed layout
        only. The default `DEFAULT_DISPLACEMENT_PC` builds the rigid-body
        modes **and** the low-degree divergence-free fields on the block
        itself; see that constant for the measurement behind it (job
        `178765558` against `178765557`: 16 853 GAMG V-cycles against 33 699,
        8 capped block-0 solves against 64, 270 s against 565 s for a 500 yr
        step) and `gadopt.NearlyIncompressibleAssembledPC` for why a
        `near_nullspace` declared on the outer mixed space never reaches GAMG
        here. `"gadopt.RigidBodyAssembledPC"` gives the six rigid-body modes
        alone, which span the slow space only while the effective bulk/shear
        ratio is small; it is the route every run before this default took,
        and naming it is how a caller reproduces one. **The only reason to
        pass `"firedrake.AssembledPC"` is to reproduce a run that predates
        both classes**, which is a measurement of the defect and not a
        configuration - a driver that names it is silently coarsening the
        elasticity block with no modes at all. On the uncondensed layout the
        displacement operator is the condensed matrix and GAMG runs on it
        directly, so this argument has no meaning there and a non-default
        value raises: the near-nullspace is chosen by the solver's
        `near_nullspace` argument instead.
      u_ksp_max_it, u_ksp_rtol: the Krylov solve on the displacement split, on
        both layouts. The default is `0`: one GAMG V-cycle per block-0
        iteration, written as `ksp_type preonly` with no tolerance and no cap
        attached. `u_ksp_max_it=4` selects the alternative, four iterations of
        CG at `u_ksp_rtol`, counting the cap as convergence.

        Measured end to end on Gadi from this preset alone, 96 ranks, both
        cases of the Spada benchmark over the full 138-step ladder, with the
        four-iteration CG as the control: the polar-motion case falls from
        2 h 32 15 to **1 h 18 32** (job `179496714` against `179423870`) and
        the cap case from 1 h 39 01 to **1 h 08 54** (`179483713` against
        `179423871`), and every printed benchmark number is unchanged, the
        seven polar-motion ratios to 1.5e-08 and the cap table to 6.1e-10.
        Per step at dt = 100 yr, where the ladder spends 90 of its 138 steps,
        34.0 s against 75.9.

        **The gain belongs to the block-0 restart as much as to this option.**
        At PETSc's default GMRES restart of 30 the same option loses: 354 s
        per 100 yr step against 269 for the CG (arms C3 and B0 of
        `NOTES/team/rotation-pc/03-CAMPAIGN.md`). This preset has shipped a
        restart equal to the block-0 cap since `c24685ac`, and at that restart
        the option is the largest single gain of that campaign. The two are
        not independent.

        The older pair in the record measures a different corner and stands:
        at dt = 500 yr, block-0 cap 400 and restart 30, the CG needs 46
        block-0 inner iterations per application against the V-cycle's 165
        and 20 156 GAMG V-cycles against 16 853, for a warm step of 271 s
        against 270 (jobs `178765560` and `178765558`,
        `NOTES/T2-GADI-GATE-2026-09-11.md` sections 5.5 and 5.6). What the CG
        buys there is that every block-0 solve reaches `block0_rtol`, so the
        tolerance decides instead of the cap; at the shipped restart that is
        no longer worth its wall clock. See `_displacement_krylov` for the
        keys and for `ksp_converged_maxits`.

    Every sub-KSP that runs a Krylov method reports `ksp_converged_reason`.
    That is not optional instrumentation: block 0 and block 1 are
    preconditioner applications inside a flexible outer Krylov, so degrading
    either costs outer iterations rather than accuracy, and the only way to see
    which one degraded is its own exit status. The per-split lines are also the
    whole of B2's cost model, so which line to count depends on the route. On
    the `"pair"` route block 0 runs its own Krylov solve and one block-0
    application is one `dtn_fieldsplit_0_` line. On the `"condensed"` route
    block 0's KSP is `preonly` and prints nothing at all: the line for one
    block-0 application is the class's inner `(u, psi)` solve at
    `dtn_fieldsplit_0_condensed_`, and its two splits print at
    `dtn_fieldsplit_0_condensed_fieldsplit_N_`.
    """
    if block0 not in ("condensed", "pair"):
        raise ValueError(
            f"block0 must be 'condensed' or 'pair', got {block0!r}.")
    dtn_representation = resolve_dtn_representation(
        dtn_representation, condensed=condensed)
    if dtn_representation == "lowrank" and block0 == "pair":
        raise ValueError(
            "block0='pair' has no low-rank route. The nested block-0 sweep "
            "solves over (u, M) and psi separately, and the low-rank DtN "
            "update lives on the potential rows of an operator that sweep "
            "never assembles, so there is nowhere to put it and the outer "
            "FGMRES would pay for its absence. Use the default "
            "block0='condensed', whose potential split is a matrix this "
            "update can be added to, or name dtn_representation='multiplier' "
            "here and on the space and the solver to run the nested route.")
    if ainvb and multiplier_pc not in (None, "none"):
        raise ValueError(
            "ainvb=True solves block 1 with gadopt.DtNTwoBlockSchurPC's own "
            "dense factors of the exact Schur complement, so the block-1 KSP "
            "is never entered and a preconditioner there would be built, "
            f"configured and never applied; got multiplier_pc={multiplier_pc!r}"
            ". Choose one of the two.")
    if condensed and block0 != "condensed":
        raise ValueError(
            f"block0={block0!r} has no meaning with condensed=True: the "
            "condensed layout holds no internal-variable field, so there is "
            "nothing for block 0 to eliminate and its sweep is over `u` and "
            "`psi` alone. Drop the argument, or pass condensed=False for the "
            "DG family.")
    # `None` is the sentinel that hands the choice to the preset; a caller who
    # names a string, `"none"` included, gets exactly that string. The dense
    # complement is worth its build only where the `Real` block is small: on
    # the low-rank representation it is 4 rows with rotation and 1 without, so
    # a build is 4 block-0 solves of the cheap kind (63 inner iterations for a
    # column of `A01` against 131-138 for the mechanics residual, 100 yr step;
    # `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 4, a labelled pass over
    # the production log, which names no arm).
    # On the multiplier representation it is about 76 rows at L = 5, so a
    # build costs more than a whole outer solve and nobody has measured over
    # how many steps that amortises. Under `ainvb` the cached apply solves
    # block 1 with its own factors, so a preconditioner there would never be
    # applied.
    #
    # The name of the representation is an INDIRECT test. The direct quantity
    # is `n`, the width of the `Real` block, because the cost of the class is
    # `n` block-0 solves for each build and `n` enters the cost in that one
    # term only. The name tracks `n` on this branch because the two cases here
    # are 1 or 4 rows on the low-rank representation and about 76 on the
    # multiplier one, and the two separate as soon as another kind of `Real`
    # row exists: the centre-of-mass rows and the sea-level `Shift` of
    # `sghelichkhani/sea-level` make a low-rank block 5 or 8 rows wide. So a
    # caller that knows its own width passes it and gets one rule that covers
    # every combination of the switches; a caller that passes nothing keeps
    # the choice by representation and so keeps its dictionary.
    if n_real is not None and n_real < 0:
        raise ValueError(
            f"n_real={n_real!r} is negative. It is the number of `Real` rows "
            "in the mixed space, which a caller normally takes as "
            "len(layout.real_fields), so a negative value is a bug in "
            "whatever computed it.")
    if dense_schur_max_rows < 0:
        raise ValueError(
            f"dense_schur_max_rows={dense_schur_max_rows!r} is negative. It is "
            "the widest `Real` block the preset will form the dense complement "
            "on; pass 0 to switch that choice off entirely.")
    chosen_by_preset = multiplier_pc is None
    # One predicate decides both objects, because both are the same exact
    # complement of the same `Real` block and both cost `n` block-0 solves to
    # form: the preset forms it when the caller named no block-1
    # preconditioner and the block is narrow. Writing the width test once is
    # what keeps the two choices from drifting apart, and it is what lets the
    # tolerance refusal below test one thing.
    if n_real is not None:
        # The rule on `n`. A block of zero rows has nothing to form, so it
        # falls through with everything above the limit.
        narrow_real_block = 0 < n_real <= dense_schur_max_rows
    else:
        # No count given: the earlier rule, kept so that every caller which
        # passes no count keeps exactly the choice it had.
        narrow_real_block = dtn_representation == "lowrank"
    forms_exact_complement = chosen_by_preset and narrow_real_block
    # What the caller said about `ainvb`, kept because the refusal below turns
    # on it: `True` is the caller taking the decision, `False` leaves the
    # preset choosing the dense complement, and `None` is the preset choosing
    # both.
    named_ainvb = ainvb
    if ainvb is None:
        # The sentinel. The cached apply is the faster of the two ways to use
        # that complement: it keeps the outer iteration count of the full
        # factorisation while spending one block-0 solve per outer iteration,
        # where the delegating path under `lower` spends the same one solve
        # and more outer iterations. Measured end to end on both cases of the
        # Spada benchmark, 96 ranks, the full 138-step ladder, from the preset
        # alone: 45 min 44 s against 1 h 18 32 on the polar-motion case (job
        # `179511971` against `179496714`) and 45 min 45 s against 1 h 08 54 on
        # the cap case (`179511972` against `179483713`), with every printed
        # benchmark number unchanged.
        #
        # A caller who named a block-1 preconditioner keeps the delegating
        # path, because under the cached apply the class they named would be
        # built, configured and never applied, and the preset must not take
        # that decision away without saying so. `multiplier_pc="none"` is a
        # naming, so a driver that sweeps `block0_rtol` keeps its own
        # configuration and its tolerance ladder.
        ainvb = forms_exact_complement
    if multiplier_pc is None:
        # Under `ainvb` the cached apply solves block 1 with its own dense
        # factors, so the block-1 KSP is never entered and no preconditioner
        # belongs there, at any width.
        multiplier_pc = ("none" if ainvb
                         else _DENSE_MULTIPLIER_PC if narrow_real_block
                         else "none")
    # The refusal scopes to the preset's own choice, not to the caller's. The
    # columns of the exact complement come out of block-0 solves, whichever of
    # the two objects holds it, so a loose block-0 tolerance makes the
    # factored complement the complement of a different operator, and the
    # preset must not SELECT a pair it documents as stagnating. A caller who
    # names the class, or `"none"`, or `ainvb`, has taken that decision and is
    # accepted at any tolerance. 1e-4 is where the one measurement sits and not
    # a law.
    if (forms_exact_complement and named_ainvb is not True
            and block0_rtol > 1e-4):
        chosen = ("gadopt.DtNTwoBlockSchurPC's cached apply" if ainvb
                  else _DENSE_MULTIPLIER_PC)
        raise ValueError(
            f"block0_rtol={block0_rtol!r} is looser than 1e-4, and the preset "
            f"would form the exact Schur complement of the `Real` block here, "
            f"through {chosen}. The columns of that complement are block-0 "
            "solves, so at 1e-2 the arm stagnates: 642 non-convergent block-0 "
            "calls and a wall worse than no block-1 preconditioner at all "
            "(Gadi job 176078939). Pass multiplier_pc='none' to keep the loose "
            "tolerance, or tighten block0_rtol to 1e-4, or name ainvb=True. "
            f"Naming multiplier_pc='{_DENSE_MULTIPLIER_PC}' is accepted at any "
            "tolerance too, because then the pair is the caller's decision and "
            "not the preset's.")
    # How block 1 is solved follows from what preconditions it, so the three
    # cases are written out together.
    if ainvb:
        # The cached apply owns block 1: it applies the dense factors of the
        # exact complement itself, nothing enters the block-1 KSP, and
        # `DtNTwoBlockSchurPC` refuses anything but `preonly` and `none` there
        # so that no arm can configure a solve that never runs.
        block1 = {"dtn_schur_ainvb": True,
                  "dtn_fieldsplit_1_ksp_type": "preonly",
                  "dtn_fieldsplit_1_pc_type": "none"}
    elif multiplier_pc == _DENSE_MULTIPLIER_PC:
        # The dense complement is the exact inverse of block 1, up to the
        # linearity of the block-0 solves that built its columns, so one
        # application solves the block and a Krylov method above it spends a
        # Schur-complement `MatMult` -- one block-0 solve -- per extra
        # iteration for nothing. Measured at 96 ranks on the 100 yr step:
        # GMRES there costs 120.4 s per step against `preonly`'s 99.8 under
        # `full` (arms B1 and B2, job 179385036,
        # `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 24). With no Krylov
        # solve on the block, a
        # relative tolerance and an iteration cap would describe a solve that
        # never runs, so neither is written.
        block1 = {"dtn_fieldsplit_1_ksp_type": "preonly",
                  "dtn_fieldsplit_1_ksp_converged_reason": None,
                  "dtn_fieldsplit_1_pc_type": "python",
                  "dtn_fieldsplit_1_pc_python_type": _DENSE_MULTIPLIER_PC}
    else:
        # An unpreconditioned block 1, or one carrying an approximate inverse
        # such as `gadopt.DtNMultiplierDiagPC`, needs the Krylov method: the
        # preconditioner does not solve the block on its own.
        block1 = {"dtn_fieldsplit_1_ksp_type": "gmres",
                  "dtn_fieldsplit_1_ksp_rtol": 1e-4,
                  "dtn_fieldsplit_1_ksp_max_it": 200,
                  "dtn_fieldsplit_1_ksp_converged_reason": None,
                  **({"dtn_fieldsplit_1_pc_type": "none"}
                     if multiplier_pc == "none" else
                     {"dtn_fieldsplit_1_pc_type": "python",
                      "dtn_fieldsplit_1_pc_python_type": multiplier_pc})}
    # The default route does the whole block-0 solve inside
    # `gadopt.CondensedBlockPC`, so block 0's own KSP is `preonly`. The
    # tolerances and the converged-reason line move down to the class's own
    # `(u, psi)` Krylov solve: leaving a reason line on a `preonly` KSP as
    # well would put a second line in the log for one block-0 application,
    # and `bench_dtn_baseline.parse_counts` counts an application per line.
    new_block0 = block0 == "condensed" and (not condensed or dtn_representation == "lowrank")
    if outer_atol < 0 or outer_max_it < 1:
        raise ValueError(
            f"outer_atol must be >= 0 and outer_max_it >= 1; they are "
            f"{outer_atol} and {outer_max_it}")
    p = {
        "mat_type": "matfree",
        "snes_type": snes_type,
        "snes_linesearch_type": "l2",
        "snes_max_it": 100,
        "snes_atol": 1e-15,
        "snes_rtol": snes_rtol,
        "snes_converged_reason": None,

        "ksp_type": "fgmres",
        "ksp_rtol": outer_rtol,
        "ksp_max_it": outer_max_it,
        "ksp_converged_reason": None,

        "pc_type": "python",
        "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
        # `full` applies the block-0 inverse twice per outer iteration and
        # `lower` applies it once (PETSc
        # `src/ksp/pc/impls/fieldsplit/fieldsplit.c:1184-1330`, confirmed by
        # counting: 2 block-0 applications per outer iteration against 1).
        # `lower` pays 14 outer iterations where `full` spends 8 over the same
        # four solves and still wins, 88.2 s against 99.8 s per 100 yr step at
        # 96 ranks (arms B4 and B2, job 179385036,
        # `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 24). Under `ainvb`
        # the cached apply IS the `full`
        # factorisation with its second block-0 solve replaced by the cached
        # columns, so it reaches one solve per outer iteration while keeping
        # `full`'s iteration count: 16.78 s per warm step against the
        # delegating path's 25.6 at dt = 100 yr, 2 outer iterations against 3
        # (arms Z2 and Z1 of job 179510821), and 45 min 44 s against 1 h 18 32
        # over the full ladder (job 179511971 against 179496714).
        # The two routes are alternatives and the class refuses anything but
        # `full`.
        "dtn_pc_fieldsplit_schur_fact_type": "full" if ainvb else "lower",

        **({"dtn_fieldsplit_0_ksp_type": "preonly",
            "dtn_fieldsplit_0_pc_type": "python",
            "dtn_fieldsplit_0_pc_python_type": ("gadopt.ReducedGIABlockPC" if condensed
                                                    else "gadopt.CondensedBlockPC")}
           if new_block0 else
           {"dtn_fieldsplit_0_ksp_type": "fgmres",
            "dtn_fieldsplit_0_ksp_rtol": block0_rtol,
            "dtn_fieldsplit_0_ksp_max_it": block0_max_it,
            # The restart equals the cap, so the solve is not restarted: FGMRES
            # restarts every 30 by default
            # (`src/ksp/ksp/impls/gmres/fgmres/fgmres.c:15`) and a solve
            # allowed 200 iterations would throw away its Krylov space six
            # times. Measured on the condensed route below, with the restart
            # as its only change: 126 s per 100 yr step against 269, and
            # block-0 solves stopping at the cap fall from 11 to 2 over five
            # steps (arm F1, job 179386296,
            # `NOTES/team/rotation-pc/03-CAMPAIGN.md` section 7; the 269 is
            # arms B0 and C0 of section 24). The cost is `block0_max_it`
            # Krylov vectors on block 0 instead of 30.
            "dtn_fieldsplit_0_ksp_gmres_restart": block0_max_it,
            "dtn_fieldsplit_0_ksp_converged_reason": None,
            "dtn_fieldsplit_0_pc_type": "fieldsplit",
            "dtn_fieldsplit_0_pc_fieldsplit_type": "multiplicative"}),

        **block1,
    }
    if outer_atol > 0:
        # The absolute floor of the outer solve; see `outer_atol` above. Left
        # out at 0, so the default dictionary is the one callers had before.
        p["ksp_atol"] = outer_atol
    if outer_accept_capped:
        # PETSc counts an outer solve that reaches `outer_max_it` as converged
        # (`CONVERGED_ITS`); see `outer_accept_capped` above.
        p["ksp_converged_maxits"] = None

    def inner(pc_python, extra=None):
        d = {"ksp_type": "preonly", "pc_type": "python",
             "pc_python_type": pc_python, "ksp_converged_reason": None}
        d.update(_prefixed(extra, "assembled_") if extra
                 else gamg_parameters("assembled_"))
        return d

    # The displacement split carries the same Krylov settings on both
    # layouts, so that the uncondensed layout's counts stay comparable with
    # the condensed one's and the wall ratio between them measures the cost of
    # the coupled residual alone (2.5 times at matched settings, job
    # `178765563` against `178765560`). The `psi` split keeps one V-cycle:
    # the potential block is a Laplacian that GAMG handles in one sweep.
    displacement_krylov = _displacement_krylov(u_ksp_max_it, u_ksp_rtol)

    # (field index in the mixed space, options). Order IS the sweep order:
    # `m` first because it is exact and nearly free, `u` before `psi` because
    # the load enters through `u`. With the internal variable condensed away
    # the space has no `m` field and the indices shift, which is why this list
    # is built rather than written out.
    if condensed and not new_block0:
        sweep = [("0", {**inner(u_pc), **displacement_krylov}),
                 ("1", inner("gadopt.SPDAssembledPC"))]
    else:
        if condensed and new_block0 and u_pc != DEFAULT_DISPLACEMENT_PC:
            raise ValueError(
                "The reduced low-rank block uses GAMG directly on the assembled displacement tangent; "
                "configure dtn_fieldsplit_0_condensed_fieldsplit_0_ rather than u_pc.")
        if not condensed and u_pc != DEFAULT_DISPLACEMENT_PC:
            raise ValueError(
                "u_pc has no meaning on the uncondensed layout: the "
                "displacement operator there is the condensed matrix that "
                "gadopt.InternalVariableSCPC assembles, and GAMG runs on it "
                "directly. Choose the near-nullspace with "
                "SelfGravitatingGIASolver(near_nullspace=...) "
                f"instead; got u_pc={u_pc!r}.")
        if new_block0:
            # `gadopt.CondensedBlockPC` owns the whole block-0 solve, so there
            # is no block-0 fieldsplit and no `_fields` key: the class builds
            # the `(u, psi)` nest itself and the fieldsplit below it takes its
            # index sets from that nest. Split 0 is the displacement block and
            # split 1 the potential block, in that order.
            #
            # `block0_rtol` and `block0_max_it` land on this inner Krylov
            # solve, because it is the one that does the work now, and its
            # converged-reason line is what the Gadi counters read one block-0
            # application from.
            #
            # GAMG runs directly on the two assembled blocks: neither is
            # matrix-free, so no `AssembledPC` wraps them and the GAMG keys
            # carry no `assembled_` prefix.
            # `SelfGravitatingGIASolver._adjust_options` replaces the
            # displacement split's CG by a GMRES of the same length when
            # `rheology_nonlinear` says the condensed operator is not
            # symmetric, which every power-law rheology makes
            # it. On the default split there is no CG to replace: `preonly`
            # runs one GAMG V-cycle, which is symmetric or not with the
            # operator and needs no Krylov method to match it.
            p.update(_prefixed({
                "ksp_type": "fgmres",
                "ksp_rtol": block0_rtol,
                "ksp_max_it": block0_max_it,
                # Not restarted: the restart equals the cap, for the reason and
                # the measurement given on the other block-0 route above.
                "ksp_gmres_restart": block0_max_it,
                "ksp_converged_reason": None,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
                **_prefixed({**displacement_krylov, "pc_type": "gamg",
                             **gamg_parameters()}, "fieldsplit_0_"),
                **_prefixed(_potential_split(dtn_representation),
                            "fieldsplit_1_"),
            }, "dtn_fieldsplit_0_condensed_"))
            return p
        # Split 0 is the pair `(u, M)`, fields 0 and 1, under static
        # condensation. The displacement operator is the condensed matrix
        # `gadopt.InternalVariableSCPC` assembles, GAMG runs on it directly,
        # and the Krylov options under `condensed_field_` are the ones the
        # condensed layout puts on its displacement split: one GAMG V-cycle by
        # default, and the same truncated CG under `u_ksp_max_it=4`. Both
        # routes read one argument so that a job comparing them measures the
        # route. `SelfGravitatingGIASolver._adjust_options` replaces a CG
        # by a GMRES of the same length when `rheology_nonlinear` says the
        # condensed operator is not symmetric, which every power-law rheology does; with `preonly` there
        # is nothing for it to replace.
        condensation = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "gadopt.InternalVariableSCPC",
            "pc_sc_eliminate_fields": "1",
            **_prefixed(displacement_krylov, "condensed_field_"),
            "condensed_field_pc_type": "gamg",
            **_prefixed(gamg_parameters(), "condensed_field_"),
        }
        sweep = [("0,1", condensation),
                 ("2", inner("gadopt.SPDAssembledPC"))]
    # `fields` is the PETSc field string of the split: one index, or a
    # comma-separated pair for the condensed `(u, M)` block.
    for split, (fields, opts) in enumerate(sweep):
        p[f"dtn_fieldsplit_0_pc_fieldsplit_{split}_fields"] = fields
        p.update(_prefixed(opts, f"dtn_fieldsplit_0_fieldsplit_{split}_"))
    return p
