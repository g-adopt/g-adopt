r"""The monolithic self-gravitating GIA system: displacement, internal
variables, potential, DtN multipliers and rotation, in one mixed space.

`SelfGravitatingGIASolver` solves the viscoelastic momentum equation, the
internal-variable evolution, the gravitational Poisson equation with its
Dirichlet-to-Neumann boundary treatment, and the rotational closure,
simultaneously. The design is `NOTES/PLAN-MONOLITHIC-SELFGRAV.md` and
`demos/gravity/ROADMAP-GIA-SELFGRAV.md`; what follows is what a reader of the
code needs.

## The sign convention, before anything else

> **The coupled `psi` is `GravitySolver`'s potential, which is *minus* the
> Newtonian one.** `grad^2 psi = -4 pi G rho`, the perturbation gravity is
> `g_1 = +grad psi`, and a positive point mass has `psi = +G M / r`.

The convention is forced rather than chosen. Everything in the potential
residual after the source line - the Robin shift, the DtN constraint rows, the
sheets, the 2-D monopole datum - is `DtNGravityForm`, which is
`GravitySolver.set_form` moved out one class, and both the monopole flux
(`-2 G M / R`) and the sheet term are anchored to that convention and are not
sign-symmetric in `psi`. Three consequences, each of which contradicts what an
older revision of the road map printed:

1. **Both body-force terms carry a MINUS**, `-B_mu int rho_0 grad(psi) . w`.
   The requirement is that the two off-diagonal blocks carry the *same*
   constant, not opposite ones.
2. **`psi_rot` is the *negated* centrifugal perturbation**, so that it shares
   `psi`'s convention and the two adjacent body forces read alike.
3. **The geoid is `N = +psi/g_0`** (`BaseGIAApproximation.geoid`), which the
   road map had right all along.

## The system

The mixed space, built by `self_gravitating_gia_space`, is

    Z = [ V(sub), S_1..S_N(sub), Psi(parent), R x n_mult, R x n_rot ]

with the mechanics on a `Submesh` of the mantle and the potential on the whole
parent annulus/shell, whose stand-off buffer carries the DtN boundaries away
from the sources. The residual, with every term written as if on the left-hand
side (the convention of `gadopt.momentum_equation`):

    F_u   = mechanics, unchanged
          - c B_mu int rho_0 grad(psi) . w dx_m
          - c B_mu int rho_0 grad(psi_rot) . w dx_m
    F_m   = unchanged internal-variable triple, per internal variable
    F_psi = theta_psi [ int grad(psi) . grad(v) dx_g
                        - Lambda int rho_0 u . grad(v) dx_m
                        + DtNGravityForm.boundary_residual(psi, v, mult) ]
    F_c   = theta_psi x the DtN constraint rows (inside boundary_residual)
    F_rot = theta_rot_i ( K_i m_i - s_i dI_i3[u, sigma] ) nu_i

and, with a `FluidCore`, the variation of one CMB energy added to the first
and third of those:

    F   += c d/dz int_Rc B_mu [ rho_core (u.n) psi
                                + 0.5 rho_core g_0 (u.n)^2 ] ds

with `c` the solver's `scaling_factor`, `K = (C-A, C-A, C)` and
`s = (+1, +1, -1)` - the `m_3` closure `m_3 = -dI_33/C` has a different
constant *and* a different sign from the polar-wander pair, and since a disc
has no polar wander the 2-D prototype exercises **only** that row.

## The three scaling constants, and why there are three

`theta_psi` and `theta_rot` are multiplicative scalings of whole residual rows.
Scaling a row changes the conditioning of the Jacobian and *nothing* about the
solution, so they are unobservable in every field the solver produces, and they
exist for two reasons: to make the Jacobian symmetric, and (road-map §5.3) to
stop SNES's single residual norm over the whole mixed vector being dominated by
one block. They are computed here from the approximation's own numbers and are
**never to be fitted to make a symmetry test pass** - a symmetry test is an
equation *for* the row scaling, so a fitted scaling absorbs any sign error in
the block it scales and reports success.

    theta_psi   = scaling_factor * B_mu / Lambda
    theta_rot_i = s_i * scaling_factor * B_mu * Omega_sq

`Omega_sq` is the squared rotation rate non-dimensionalised by `g_bar / L`;
`OMEGA_SQ_EARTH` below is 1.566e-3. This docstring used to add that it is
"three orders below `Lambda` and `B_mu`, which is the quantitative statement
that rotational feedback is a small correction riding on a leading-order
coupling". **That inference is wrong and the sentence has been removed**
(2026-08-01). It is the kind of wrong that licenses a tolerance, or a `Real`
block scaling, generous enough to swallow the very effect the rotation gate
exists to measure, so it is corrected here rather than deleted quietly.

The error is to compare `Omega_sq` with `Lambda` while treating `C - A` as an
independent constant. `C - A` is *itself* caused by rotation and is itself
proportional to `Omega^2`, so the `Omega^2` cancels out of the closure.
Eliminating `u` analytically leaves the classical secular Liouville relation

    m = (1 + k_L) dI_direct / [ (C - A)(1 - k_T/k_s) ]

in which `k_T/k_s` is **O(1)** and carries no `Omega^2` at all. Measured on the
benchmark: switching the rotational body force off drops `|m|` by **32 %** at
`t = 0`, an amplification of `1/(1 - 0.318) = 1.466`, and by more at later
epochs. The feedback is a third of the answer, not a perturbation. What
`Omega_sq` being small does control is the size of `psi_rot` itself, which is a
statement about the centrifugal potential and not about the loop gain.

The derivations of the two scalings are in
`SelfGravitatingGIASolver.theta_psi` and `SelfGravitatingGIASolver._theta_rot`;
both are one line and neither is the line the road map originally claimed.

The `scaling_factor` in both is easy to miss: `CoupledInternalVariableSolver`
multiplies the entire momentum residual by it, so it multiplies the two body
forces too, and a `theta_psi` that ignored it would make the coupled Jacobian
asymmetric by exactly that factor on any run that used one.

## Traps this module asserts against rather than documents

Two failure modes in this geometry are *silent*, and both have cost a day
already elsewhere in the project:

- **A measure that finds nothing assembles to zero.** A `ds` given an interior
  facet tag, or an intersected measure whose intersection is empty, returns
  `0.0` with at most a warning. A load sheet written that way is simply absent:
  the solve converges, the potential is missing the largest single term in the
  geoid, and no symmetry or Picard-consistency test looks, because a sheet is a
  right-hand side with no Jacobian contribution and two solvers sharing the
  form omit it identically. `check_geometry` therefore measures every DtN
  boundary and every sheet against `2 pi R` (`4 pi R^2` in 3-D) *at
  construction*, and checks the cross-mesh measure by integrating a parent
  coefficient over the submesh.
- **Restriction sides on a tagged interior facet are consistent only by gmsh's
  cell ordering.** Every sheet integrand goes through
  `DtNGravityForm.sheet_integral`, which uses `avg` and never a hard-coded
  `'+'`.

## What this module does not do

No time loop, no output, no checkpointing: it is a solver, and the driver is
the demo's. The 2-D enclosed-mass bookkeeping *is* here, mirroring
`GravitySolver`, because the coupled system needs the monopole datum for any
configuration whose load carries net mass - but it is 2-D-lifetime code that
3-D will not execute at all, so it is confined to `set_monopole_datum`,
`update_total_mass` and `check_net_mass` and nothing else in the class reasons
about enclosed mass.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Number
from typing import Any
from warnings import warn

import numpy as np
from firedrake import *
from ufl import Form

from .approximations import BaseGIAApproximation
from .preconditioners import RigidBodyAssembledPC
from .dtn_form import DtNGravityForm
from .equations import Equation
from .gravity_solver import real_scalar_solver_parameters
from .momentum_equation import (
    compressible_viscoelastic_terms,
    rotational_potential,
    rotational_potential_term,
    self_gravity_term,
)
from .scalar_equation import mass_term, sink_term, source_term
from .solver_options_manager import ConfigType, gamg_parameters
from .stokes_integrators import (
    CoupledInternalVariableSolver,
    newton_stokes_solver_parameters,
)
from .utility import ensure_constant

__all__ = [
    "FluidCore",
    "RigidBodyAssembledPC",
    "GIASpaceLayout",
    "NULL_COUPLING_ROW_SCALE",
    "OMEGA_SQ_EARTH",
    "SelfGravitatingGIASolver",
    "rigid_rotation_nullspace",
    "self_gravitating_gia_space",
    "selfgrav_dtn_iterative_solver_parameters",
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

See `SelfGravitatingGIASolver._row_scale_B_mu` for the full argument.
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
    "dtn_fieldsplit_1_pc_type": "none",
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

**`fieldsplit_0` is not the potential block.** The preconditioner merges every
non-`Real` sub-field into block 0, so here that is displacement + internal
variables + potential together - a saddle-ish coupled operator, not the scalar
Laplacian that `gravity_solver.py`'s `SPDAssembledPC` + GAMG preset is tuned
for. That preset is **not** transferable and must not be copied across.

This has no 3-D successor and is not a fallback there: at production size
block 0 is tens of millions of displacement dofs plus hundreds of millions of
DG1 tensor dofs plus the potential. The candidates, neither yet measured, are
static condensation of the internal variables (the `(m, m)` block is
block-diagonal per cell, so it inverts exactly and for free) and using the
segregated Picard iteration as the preconditioner. The 2-D prototype is the
last cheap place to find out.

Note also that reaching for a plain `fieldsplit` while debugging brings back
PETSc's 128-field cap, which registering the two blocks as index sets is what
avoids.
"""


# The GAMG settings for the displacement and potential blocks used to be a
# sixth literal copy of the six that `gadopt.solver_options_manager` now owns --
# the same ones the Stokes velocity block and `GravitySolver` use. The sweep
# builder below calls `gamg_parameters("assembled_")` directly.


def _prefixed(d, prefix):
    return {prefix + k: v for k, v in d.items()}


# `RigidBodyAssembledPC` now lives in `gadopt.preconditioners`, alongside
# `SPDAssembledPC` and the `DtNTwoBlockSchurPC` whose merged index sets are the
# reason it has to exist. It was defined here only because that file was under
# concurrent edit when it was packaged. It is imported at the top of this
# module and stays in `__all__`, so `gadopt.RigidBodyAssembledPC` and
# `gadopt.gia_gravity.RigidBodyAssembledPC` both still resolve -- which matters
# because every caller reaches it through the *string*
# `"gadopt.RigidBodyAssembledPC"` in a solver-options dictionary, and a broken
# name there surfaces as `PETSc.Error: error code 101` naming nothing.


def selfgrav_dtn_iterative_solver_parameters(
    *, condensed: bool = True, block0_rtol: float = 1e-2,
    outer_rtol: float = 1e-6, block0_max_it: int = 60,
    snes_rtol: float = 1e-4,
    u_pc: str = "gadopt.RigidBodyAssembledPC",
) -> dict:
    r"""The 3-D configuration that works, and the only one that does.

    **This exists because its absence cost two production jobs in one day.**
    Until now `gadopt` shipped exactly one preset for this system,
    `selfgrav_dtn_schur_solver_parameters`, whose own docstring says it has no
    3-D successor - and the working configuration lived in a spike. B1 was
    OOM-killed after 90 minutes and a phase-tolerance ladder ran six hours into
    a walltime kill, both on the direct default with block 0 capping at 200
    iterations. Every driver was rediscovering this dictionary and two of them
    got it wrong.

        outer FGMRES
         +- DtNTwoBlockSchurPC, schur_fact_type full
             +- block 0: FGMRES rtol 1e-2 + multiplicative fieldsplit
             |   +- m  : AssembledPC + bjacobi/ilu   (condensed: absent)
             |   +- u  : RigidBodyAssembledPC + GAMG
             |   +- psi: SPDAssembledPC + GAMG   (GravitySolver's preset)
             +- block 1: GMRES on the Real block, pc_type none

    **This preset carries no iteration count of its own, and the "flat at 3"
    that used to stand here was another configuration's.**  That 3 was measured
    in `demos/glacial_isostatic_adjustment/3d_spada_selfgrav/b2_probe.py`,
    `--config cond`: an *algebraic* Schur split on the **uncondensed** space,
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
    - **The multiplier block runs at `pc_type: none`** because `jacobi` needs
      `MatGetDiagonal`, which on a matfree block goes through TSFC's
      `diagonal=True` path and cannot handle `Real` arguments, and
      `AssembledPC` is structurally unavailable for the same family of reasons.
      So its iteration counts are an **upper bound** rather than a tuned
      figure.

    Args:
      condensed: whether the caller passes `condense_internal_variables=True`.
        The internal variable is ~85 % of block 0, so condensing is what makes
        it affordable; when it is condensed there is no `m` field to sweep and
        including one would index the splits wrongly.  (It is *not* what takes
        any count in this preset to 3 - see the attribution above.) **Keep this argument and
        the solver's flag in step** - that is the whole reason it is an
        argument rather than an assumption.
      block0_rtol, outer_rtol, block0_max_it, snes_rtol: the tolerances.
        `snes_rtol` is relative to the norm of the **whole** mixed residual,
        which the mechanics rows dominate; rows scaled by `Omega_sq = 1.566e-3`
        are converged to correspondingly fewer digits, which is a live question
        for the rotational closure.
      u_pc: the preconditioner on the displacement split. The default builds
        the rigid-body near-nullspace on the block itself; see
        `gadopt.RigidBodyAssembledPC` for why a `near_nullspace` declared on
        the outer mixed space never reaches GAMG here. **The only reason to
        pass `"firedrake.AssembledPC"` is to reproduce a run that predates
        that class**, which is a measurement of the defect and not a
        configuration - a driver that names it is silently coarsening the
        elasticity block with no rigid modes.

    Every sub-KSP reports `ksp_converged_reason`. That is not optional
    instrumentation: block 0 and block 1 are preconditioner applications inside
    a flexible outer Krylov, so degrading either costs outer iterations rather
    than accuracy, and the only way to see which one degraded is its own exit
    status. The per-split lines are also the whole of B2's cost model - block-0
    applications per solve are read from them and from nowhere else.
    """
    p = {
        "mat_type": "matfree",
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 100,
        "snes_atol": 1e-15,
        "snes_rtol": snes_rtol,
        "snes_converged_reason": None,

        "ksp_type": "fgmres",
        "ksp_rtol": outer_rtol,
        "ksp_max_it": 200,
        "ksp_converged_reason": None,

        "pc_type": "python",
        "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
        "dtn_pc_fieldsplit_schur_fact_type": "full",

        "dtn_fieldsplit_0_ksp_type": "fgmres",
        "dtn_fieldsplit_0_ksp_rtol": block0_rtol,
        "dtn_fieldsplit_0_ksp_max_it": block0_max_it,
        "dtn_fieldsplit_0_ksp_converged_reason": None,
        "dtn_fieldsplit_0_pc_type": "fieldsplit",
        "dtn_fieldsplit_0_pc_fieldsplit_type": "multiplicative",

        # `pc_type: none` is not laziness; see the docstring.
        "dtn_fieldsplit_1_ksp_type": "gmres",
        "dtn_fieldsplit_1_ksp_rtol": 1e-4,
        "dtn_fieldsplit_1_ksp_max_it": 200,
        "dtn_fieldsplit_1_ksp_converged_reason": None,
        "dtn_fieldsplit_1_pc_type": "none",
    }

    def inner(pc_python, extra=None):
        d = {"ksp_type": "preonly", "pc_type": "python",
             "pc_python_type": pc_python, "ksp_converged_reason": None}
        d.update(_prefixed(extra, "assembled_") if extra
                 else gamg_parameters("assembled_"))
        return d

    # (field index in the mixed space, options). Order IS the sweep order:
    # `m` first because it is exact and nearly free, `u` before `psi` because
    # the load enters through `u`. With the internal variable condensed away
    # the space has no `m` field and the indices shift, which is why this list
    # is built rather than written out.
    if condensed:
        sweep = [(0, inner(u_pc)),
                 (1, inner("gadopt.SPDAssembledPC"))]
    else:
        sweep = [(1, inner("firedrake.AssembledPC",
                           {"pc_type": "bjacobi", "sub_pc_type": "ilu"})),
                 (0, inner(u_pc)),
                 (2, inner("gadopt.SPDAssembledPC"))]
    # `field_index` and not `field`: `dataclasses.field` is imported above and
    # a loop variable would shadow it.
    for split, (field_index, opts) in enumerate(sweep):
        p[f"dtn_fieldsplit_0_pc_fieldsplit_{split}_fields"] = str(field_index)
        p.update(_prefixed(opts, f"dtn_fieldsplit_0_fieldsplit_{split}_"))
    return p


@dataclass(frozen=True)
class FluidCore:
    r"""An inviscid, homogeneous, hydrostatic core, eliminated onto its boundary.

    The alternative to the legacy `un = 0` at the CMB, and the switch between
    them is the presence of this object: pass one to
    `SelfGravitatingGIASolver` for the fluid core, leave it out and keep
    `{Rc: {"un": 0.0}}` in `bcs` for the rigid one. The two cannot be combined,
    and the solver refuses the combination rather than adding the traction to a
    boundary whose normal displacement is pinned to zero.

    The core is not meshed. It cannot be: its interior Eulerian density
    perturbation is `-rho_c div(u) = 0` and its displacement field is genuinely
    indeterminate, since any divergence-free tangential shuffle costs no
    energy. A "soft solid" core with a small shear modulus approximates
    something that can be written down exactly and degrades in conditioning as
    it improves in fidelity. What the core does to the mantle is push on it and
    carry mass across the displaced interface, and both are boundary terms.

    ## What it contributes, and where the density contrast comes from

    Everything is the variation of one energy on the CMB facet,
    `SelfGravitatingGIASolver.fluid_core_energy`:

        E = B_mu [ rho_core (u.n) psi + 0.5 rho_core g_0 (u.n)^2 ] ds

    with `n` the **mantle's outward facet normal**, which at its inner boundary
    points *inward*, so `dot(u, n) = -u_r` there. `vertical_component(u)` - the
    codebase's own idiom for the radial component - is therefore the **wrong
    vector here**, and using it costs an anti-restoring CMB with no error
    message and a 5.3e-05 asymmetry in a term that is otherwise symmetric to
    1e-16 (the two factors then use different vectors and agree only to the
    angle between the facet normal and the analytic radial).

    **The buoyancy half carries `rho_core` alone, NOT the density contrast
    `(rho_core - rho_0)`.** This is a correction (2026-08-05) to what the design
    documents *and this docstring* said in between: the spring is the exact
    parallel of the `psi` half, which always correctly used the core's full
    density on top of a mantle half the volume terms supply automatically. The
    "contrast" reasoning below is preserved only long enough to name why it is
    wrong, because it is intuitive and cost time twice.

    The claim that "nothing in the volume terms supplies the mantle-side half at
    the CMB" is FALSE, and was measured false.
    `hydrostatic_prestress_advection_and_buoyancy_term`, evaluated on a
    divergence-free radial field, equals a pure boundary integral

        a_vol(u, u) = B_mu oint rho_0 (u.n)(g_0 u_r) ds

    to relative residual 2.9e-15: it supplies `-0.5 rho_0 g_0 u_r^2` at Rc and
    `+0.5 rho_0 g_0 u_r^2` at Re (the free-surface spring b1 relies on and adds
    no explicit term for - the control that proves the mechanism). The earlier
    "1.8e-11 cancellation" measured the *antisymmetric* combination
    `[(u.n)(w.gv) - (gv.u)(w.n)]`, which is identically zero at `(u, u)` whether
    or not `n || g` - the wrong quantity. The surviving *symmetric* boundary
    term is the spring, and it is the mantle half. Reproduce:
    `NOTES/measurements/cmb_prestress_check.py`, assembly only, no solve.

    So the volume term already contributes `-0.5 rho_0 g_0 u_r^2` at the CMB; the
    physical interface stiffness is `+0.5 (rho_core - rho_0) g_0 u_r^2`; the
    explicit spring must therefore supply the remainder
    `+0.5 rho_core g_0 u_r^2` - the core's density alone. Written with the
    contrast, the assembled net is `0.5 (rho_core - 2 rho_0)`, only 13.8 % of
    physical (7.3x too soft) for the Spada model.

    The *sheet* half is the exact parallel and was always right: the
    divergence-form Poisson source supplies its mantle side. Integrating
    `-B_mu int rho_0 u.grad(psi) dx` by parts leaves `-B_mu oint rho_0 (u.n) psi
    ds` on the whole submesh boundary, an automatic sheet `sigma_auto = rho_0
    (u.n)`, which at Rc is `-rho_0 u_r`. The physical total is `(rho_core -
    rho_0) u_r`, so the missing piece is `+rho_core u_r` and the `psi` energy
    carries the core's density alone. The buoyancy half now matches it.

    ## Magnitudes, for the Spada M3-L70-V01 model

    Net physical CMB interface stiffness `(10750 - 4978) x 10.457 = 60357.8
    Pa/m`, **positive and therefore stabilising** (heavy below light), and
    2.0249 times the surface contrast's `3037 x 9.815 = 29808.2`.
    Non-dimensionally, with `rho/rhobar`, `g/gbar` and `B_mu = 1.564037`, that
    net stiffness is **1.744945** - but it is the sum of `-0.5 rho_0 g` (the
    volume term) and `+0.5 rho_core g` (this spring), so the SPRING coefficient
    this class writes is `rho_core g = 3.249855`, not the net. The old code
    wrote the net `(rho_core - rho_0) g` as the spring and assembled
    `(rho_core - 2 rho_0) g` in total. `buoyancy_density="contrast"` restores
    that old (wrong) coefficient for baseline runs; default `"core"` is correct.

    Attributes:
      boundary: the CMB facet tag, as seen from the *mechanics* mesh, where it
        is an exterior facet. On the parent it is an interior facet and carries
        the same tag; see `SelfGravitatingGIASolver.fluid_core_measure` for why
        the mechanics side is the one that can carry this term at all.
      rho_core: the core's density, in the reference state's own units.
      rho_mantle: the mantle-side reference density at the interface. `None`
        takes `approximation.density`, which is the right thing for a layered
        `rho_0`: the mechanics mesh has only mantle cells, so the facet trace of
        `rho_0` is the mantle value by construction.
      g: the reference gravity at the interface. `None` takes
        `approximation.g`.
      buoyancy_density: which density the buoyancy spring carries. `"core"`
        (default, correct) writes `rho_core`; `"contrast"` writes the old,
        7.3x-too-soft `(rho_core - rho_mantle)` and exists only to reproduce
        pre-2026-08-05 baseline numbers. See the class docstring for the
        measurement.
    """

    boundary: int | str
    rho_core: Any
    rho_mantle: Any = None
    g: Any = None
    buoyancy_density: str = "core"


@dataclass(frozen=True)
class GIASpaceLayout:
    """Where each field of the coupled mixed space lives, by name.

    Returned by `self_gravitating_gia_space` alongside the space itself, so
    that nothing downstream counts sub-fields by hand: the multiplier count is
    a property of the boundary conditions, and the rotation count is a property
    of the dimension, so the indices of the later blocks are not knowable
    without both.

    The `Real` sub-fields - multipliers then rotation - are **contiguous and
    last**, which `DtNTwoBlockSchurPC.initialize` asserts and which is worth
    keeping: anything else would leave sub-fields out of both of its blocks,
    which is a silently wrong split rather than an error.

    Attributes:
      displacement: index of the CG vector displacement block.
      internal_variables: indices of the DG tensor internal-variable blocks.
      potential: index of the potential block, which is on the *parent* mesh.
      multipliers: indices of the DtN multiplier `Real` blocks.
      rotation: `{}` with rotation off; otherwise `{"m3": i}` in 2-D and
        `{"m1": i, "m2": i+1, "m3": i+2}` in 3-D. Keyed by name and never by
        position, because the 2-D field is `m_3` - index **2** of the rotation
        triple, not index 0 - and a positional record would silently reindex on
        promotion to 3-D.
      gravity_form: the `DtNGravityForm` the multiplier count came from. It is
        returned rather than rebuilt by the solver because rebuilding it would
        re-run every boundary measurement and could pick a different quadrature
        degree, leaving the space sized against one boundary treatment and the
        residual written against another.
      mechanics_mesh, potential_mesh: the two meshes, the same object when the
        system is not cross-mesh.
      self_gravity_number: `Lambda`, recorded so the solver can check it
        against the approximation's; see `self_gravitating_gia_space`.
    """

    displacement: int
    internal_variables: tuple[int, ...]
    potential: int
    multipliers: tuple[int, ...]
    rotation: dict[str, int]
    gravity_form: DtNGravityForm
    mechanics_mesh: Any
    potential_mesh: Any
    self_gravity_number: Any = None
    #: Names of the rotation components, in the fixed 3-D order.
    rotation_names: tuple[str, ...] = field(default=("m1", "m2", "m3"))
    #: The DG tensor space the internal variables live on. Normally this is
    #: `Z.sub(i)` for `i` in `internal_variables` and is redundant; under static
    #: condensation `internal_variables` is empty and the variables are stored
    #: `Function`s outside the mixed space, so this is the only record of it.
    internal_variable_space: Any = None
    #: Whether the internal variables were condensed out of the mixed space.
    condensed: bool = False

    @property
    def cross_mesh(self) -> bool:
        return self.mechanics_mesh is not self.potential_mesh

    @property
    def n_fields(self) -> int:
        return (2 + len(self.internal_variables) + len(self.multipliers)
                + len(self.rotation))

    @property
    def real_fields(self) -> tuple[int, ...]:
        """Every `Real` sub-field index, in space order."""
        return tuple(self.multipliers) + tuple(sorted(self.rotation.values()))

    @property
    def real_fields(self) -> tuple[int, ...]:
        """Every `Real` sub-field index, multipliers then rotation, in order.

        **The one place the `Real` block's composition is written down.**
        Anything that needs to describe that block per row -- a diagonal
        preconditioner, say -- must key off this rather than off position
        within the contiguous run, so that an interleaved or rotation-first
        layout becomes a *construction* error rather than something a count
        check would wave through. `DtNTwoBlockSchurPC` separately requires the
        run to be contiguous and last, and `block1_diagonal` asserts that this
        tuple is exactly that run.
        """
        return tuple(self.multipliers) + tuple(
            self.rotation[name] for name in self.rotation_names
            if name in self.rotation)

    def rotation_slots(self) -> tuple[int | None, int | None, int | None]:
        """`(m1, m2, m3)` indices, `None` where the component does not exist.

        The one accessor rotation code should use, so that every expression is
        written in three components unconditionally and the 3-D promotion adds
        two `Real` fields rather than revisiting every rotation expression.
        """
        return tuple(self.rotation.get(name) for name in self.rotation_names)


def self_gravitating_gia_space(
    mechanics_mesh,
    potential_mesh,
    /,
    *,
    gravity_bcs: dict[int | str, dict[str, Any]],
    n_internal_variables: int = 1,
    rotation: bool = False,
    self_gravity_number: Number | Constant | None = None,
    displacement_degree: int = 2,
    internal_variable_degree: int = 1,
    potential_degree: int = 2,
    quad_degree: int | None = None,
    alpha: Number | Constant | None = None,
    condense_internal_variables: bool = False,
) -> tuple[MixedFunctionSpace, GIASpaceLayout]:
    r"""Builds the coupled mixed space and the layout that describes it.

        Z = [ V(sub), S_1..S_N(sub), Psi(parent), R x n_mult, R x n_rot ]

    The multiplier count depends on the boundary conditions - a
    `CylindricalDtN(M)` contributes `2M` multipliers on an exterior boundary and
    `2M + 1` on an interior one, and which side a boundary is on is *measured*
    from the mesh rather than declared - so the user cannot build this space
    before knowing them. That is why the factory takes `gravity_bcs` and builds
    the `DtNGravityForm` itself, returning it on the layout.

    **The `Real` spaces go on the parent mesh** (spike S2, measured). Both
    choices construct and give bit-identical answers, but on the parent *both*
    families of constraint row - parent-boundary DtN rows and submesh-volume
    rotation rows - assemble against the measures the mechanics already needs,
    whereas on the submesh every parent-boundary row needs a specially
    intersected `ds`, and an intersected measure that finds nothing assembles
    to zero without raising.

    `n_rot` is 1 in 2-D and 3 in 3-D. The single 2-D component is `m_3`, the
    rotation-rate change: a disc has no polar wander, because `m_1` and `m_2`
    tilt the rotation axis out of a plane that has no third direction. That is
    a genuine physical statement rather than a degenerate case - `m_3` is
    angular-momentum conservation of the disc, `dOmega/Omega = -dI_33/C` - which
    is why the 2-D prototype can test the rotation machinery at all.

    Args:
      mechanics_mesh: the mantle, usually a `Submesh` of `potential_mesh`.
        Pass the same object twice for a single-mesh system.
      potential_mesh: the parent, carrying the buffer and the DtN boundaries.
      gravity_bcs: boundary conditions for the potential, in
        `DtNGravityForm`'s dictionary form.
      n_internal_variables: number of viscoelastic internal variables, matching
        the approximation's `maxwell_times`.
      rotation: whether to carry the rotational closure.
      self_gravity_number: `Lambda`. The sheets carry an explicit `4 pi G` while
        the volume source has `4 pi G` absorbed into `Lambda`, so the form's
        gravitational constant must be `Lambda / (4 pi)` for the two to be
        consistent; passing `Lambda` here is what makes that happen in one
        place. Getting it wrong scales the sheet against the volume source by a
        constant, which no symmetry test can see. `None` leaves the form's
        `G = 1`, which is right only for a configuration with no sheets.
      displacement_degree, internal_variable_degree, potential_degree:
        element degrees; the defaults are the CG2 / DG1 / CG2 of road-map §5.1.
      quad_degree: boundary quadrature degree for the DtN treatment; `None`
        takes `DtNGravityForm`'s mesh-aware calibrated default.
      alpha: the Robin shift, `None` for the form's default of 1.

    Returns:
      `(Z, layout)`.
    """
    if n_internal_variables < 1:
        raise ValueError(
            f"n_internal_variables must be at least 1, got {n_internal_variables}.")

    # `Submesh` does not inherit `cartesian` from its parent, and G-ADOPT reads
    # that attribute through `is_cartesian` on every mesh it touches - so the
    # first thing that asks the submesh whether it is Cartesian raises
    # `AttributeError: 'MeshTopology' object has no attribute 'cartesian'`, from
    # inside `upward_normal`, several frames from anything that mentions a
    # submesh. Carry it across here, where both meshes are in scope.
    if not hasattr(potential_mesh, "cartesian"):
        raise ValueError(
            "potential_mesh has no `cartesian` attribute. G-ADOPT requires it "
            "on every mesh (`mesh.cartesian = False` for an annulus or shell); "
            "a mesh read from a file does not have one.")
    if not hasattr(mechanics_mesh, "cartesian"):
        mechanics_mesh.cartesian = potential_mesh.cartesian

    Psi = FunctionSpace(potential_mesh, "CG", potential_degree)
    form = DtNGravityForm(
        Psi, gravity_bcs,
        gravitational_constant=(1.0 if self_gravity_number is None
                                else self_gravity_number / (4.0 * np.pi)),
        quad_degree=quad_degree, alpha=alpha)

    V = VectorFunctionSpace(mechanics_mesh, "CG", displacement_degree)
    S = TensorFunctionSpace(mechanics_mesh, "DG", internal_variable_degree)
    R = FunctionSpace(potential_mesh, "R", 0)

    n_rot = 0
    if rotation:
        n_rot = 1 if potential_mesh.geometric_dimension == 2 else 3

    # Static condensation: the (m, m) block is `mass + sink` with DG test and
    # trial over volume integrals only, so it is block-diagonal per cell and
    # eliminating `m` is exact and local. Doing it is not an optimisation -
    # measured on the 3-D benchmark mesh the internal variable is **85 % of the
    # whole system** (2 860 092 dofs of 3 348 411 at the coarse rung), so every
    # Krylov vector, residual evaluation and reduction in the uncondensed
    # system spends most of its traffic on a variable that never needed to be
    # an unknown. The elimination is exactly the backward-Euler substitution
    # `InternalVariableSolver` already performs; see
    # `SelfGravitatingGIASolver.set_equations`.
    spaces = [V] + ([] if condense_internal_variables
                    else [S] * n_internal_variables) + [Psi]
    i_potential = len(spaces) - 1
    i_R = len(spaces)
    spaces.extend([R] * (form.n_multipliers + n_rot))

    multipliers = tuple(range(i_R, i_R + form.n_multipliers))
    i_rot = i_R + form.n_multipliers
    # Named, and named `m3` in 2-D: the 2-D component is index *2* of the
    # rotation triple, and a layout that recorded it as "the first rotation
    # field" would silently change meaning on promotion to 3-D.
    names = ("m3",) if n_rot == 1 else ("m1", "m2", "m3")
    rotation_map = {name: i_rot + k for k, name in enumerate(names[:n_rot])}

    layout = GIASpaceLayout(
        displacement=0,
        internal_variables=(() if condense_internal_variables
                            else tuple(range(1, 1 + n_internal_variables))),
        internal_variable_space=S,
        condensed=condense_internal_variables,
        potential=i_potential,
        multipliers=multipliers,
        rotation=rotation_map,
        gravity_form=form,
        mechanics_mesh=mechanics_mesh,
        potential_mesh=potential_mesh,
        self_gravity_number=(None if self_gravity_number is None
                             else ensure_constant(self_gravity_number)),
    )
    return MixedFunctionSpace(spaces), layout


def rigid_rotation_nullspace(
    Z: MixedFunctionSpace,
    layout: GIASpaceLayout,
) -> MixedVectorSpaceBasis:
    r"""The rigid-rotation kernel of the coupled operator, as a mixed basis.

    **Declare this whenever the tangential displacement is unconstrained**,
    which for GIA is the ordinary case: the CMB condition is free slip
    (`un = 0`) and the surface carries a traction, so nothing anywhere fixes a
    rotation of the whole mantle about the centre. In 2-D the kernel is spanned
    by `u = (-y, x)`; in 3-D by the three generators `e_i x x`.

    The mode is a genuine kernel of the *continuum* operator, and of the whole
    coupled system rather than of the mechanics alone:

    - it is strain free and divergence free, so the viscoelastic bilinear form
      annihilates it and so does the internal-variable source;
    - `u . n = 0` on every concentric circle or sphere, so both Nitsche free-slip
      terms vanish and the divergence-form Poisson source
      `-Lambda int rho_0 u . grad(v)` integrates to zero against every `v`;
    - `dI_i3 = int rho_0 grad(p_i) . u = 0` for each `i`, so the rotation rows
      do not see it either.

    Discretely it is annihilated only to about `2e-06` relative, because
    `u . n_h` is not exactly zero on a piecewise-quadratic approximation to a
    circle. That is the whole reason to declare it rather than leave it: the
    operator is nonsingular *through facet geometry error*, so the multiple of
    the mode the solver lands on is set by nothing physical, and two runs of the
    same system - a coupled one and its uncoupled reference, say - differ by an
    arbitrary rigid rotation. Measured on the development annulus, that is a
    `1.1e-07` relative displacement difference where the internal variable it
    drives agrees to `7e-13`. Declaring the nullspace projects it out and makes
    the two comparable; the radial quantities that carry the physics - `u_r`,
    the geoid, `dI_33` - are a rigid rotation's blind spot and are identical
    either way.

    **Why declared and not pinned.** Pinning would mean either a strong
    condition on a single tangential degree of freedom, which is mesh-dependent
    and puts a spurious point force into an otherwise smooth solution, or a
    Lagrange multiplier, which changes the mixed space and therefore the
    `Real`-block layout `DtNTwoBlockSchurPC` asserts on. A `MixedVectorSpaceBasis`
    changes neither: PETSc removes the mode from the residual and from the
    iterates, the answer becomes the minimum-norm one, and the system the solver
    reports on is the one that was written down.

    **When not to declare it.** If any boundary carries a strong `u` condition,
    or a `un` condition on something that is not a concentric sphere, the mode
    is not in the kernel and declaring it would project out a piece of the real
    answer. The existing production GIA driver passes `nullspace=None` and
    relies on `un = 0` at the CMB to pin translations, which it does; it does
    not pin rotations, and road-map §3.4 discusses the consequence.

    Args:
      Z: the coupled mixed space from `self_gravitating_gia_space`.
      layout: its `GIASpaceLayout`.

    Returns:
      A `MixedVectorSpaceBasis` over `Z` carrying the orthonormalised rotation
      generators in the displacement block and every other block untouched.
    """
    V = Z.sub(layout.displacement)
    X = SpatialCoordinate(layout.mechanics_mesh)
    if len(X) == 2:
        modes = [as_vector([-X[1], X[0]])]
    else:
        modes = [as_vector([Constant(0.0), -X[2], X[1]]),
                 as_vector([X[2], Constant(0.0), -X[0]]),
                 as_vector([-X[1], X[0], Constant(0.0)])]

    basis = VectorSpaceBasis([Function(V).interpolate(m) for m in modes])
    # Required by PETSc: `MatNullSpace` takes an orthonormal set, and an
    # un-orthonormalised basis is accepted and then silently projects wrongly.
    basis.orthonormalize()

    entries = [Z.sub(i) for i in range(len(Z))]
    entries[layout.displacement] = basis
    return MixedVectorSpaceBasis(Z, entries)


class SelfGravitatingGIASolver(CoupledInternalVariableSolver):
    r"""Viscoelastic GIA coupled monolithically to self-gravitation and rotation.

    Extends `CoupledInternalVariableSolver` with the gravitational potential,
    its DtN boundary treatment and the rotational closure, all in one mixed
    space and one Newton solve. The module docstring states the sign convention,
    the residual and the three scaling constants; this docstring is about what
    the class had to override and why.

    **`StokesSolverBase.__init__` cannot see a two-mesh mixed space.** It does
    `self.mesh = self.solution_space.mesh()`, which succeeds and quietly returns
    a `MeshSequenceGeometry`, and then `upward_normal(self.mesh)`, which raises
    `NonUniqueMeshSequenceError` from `.unique()`. That is one line later, so
    re-assigning `self.mesh` after `super().__init__` returns is not an option -
    it never returns. `mesh` is therefore a property here whose setter keeps the
    mechanics mesh and discards anything that is not a real mesh; everything
    else in `set_boundary_conditions`, including strong conditions on fields
    living on *different* meshes in one space, works untouched (spike S3).

    **`set_equations` and `set_form` are both overridden.** The parent's
    `set_form` is `sum(eq.residual(sol) for eq, sol in zip(...))`, one
    `Equation` per sub-field, and this space has twenty-odd multiplier
    sub-fields that are not equations in that sense. The mechanics goes through
    the base machinery unchanged - which is the point, since the coupling must
    not perturb it - and the gravity and rotation residuals are built separately
    and added.

    **`solve` refreshes the enclosed mass first**, exactly as
    `GravitySolver.solve` does, because the 2-D monopole datum is a coefficient
    in the boundary form rather than an unknown.

    Args:
      solution: `Function` on the space `self_gravitating_gia_space` returned.
        Zero-initialised, and it is worth being explicit that it should be: the
        base class takes `solution_old = solution.copy(deepcopy=True)`, so a
        nonzero internal variable at construction is a silently prestressed
        initial state. A nonzero `psi` or multiplier is harmless, because the
        time weighting is backward Euler.
      approximation: a `BaseGIAApproximation` carrying `B_mu`, `density`, `g`
        and `self_gravity_number`.
      layout: the `GIASpaceLayout` from the factory.
      dt: time step.
      rotation_moments: `{"C": ..., "C_minus_A": ...}`, the principal moments of
        the reference hydrostatic figure. `C_minus_A` is required only in 3-D,
        where it is the dynamical ellipticity and *cannot* be computed from a
        spherically symmetric reference density at all (`C = A` identically),
        so it is an input. In 2-D, `C` is the disc's polar second moment
        `int rho_0 r^2 dV`, which the caller can assemble; it is still an input
        here so that the 3-D promotion supplies one more key rather than
        rewriting the row.
      Omega_sq: the squared rotation rate non-dimensionalised by `g_bar / L`;
        see `OMEGA_SQ_EARTH`.
      fluid_core: a `FluidCore` (or its keyword mapping) to give the core its
        buoyancy and its mass sheet, or `None` for the legacy rigid core, which
        the caller then spells as `{Rc: {"un": 0.0}}` in `bcs`. The two are
        alternatives and the constructor refuses both on one boundary; keeping
        `un = 0` reachable is what makes the difference between the treatments a
        measured quantity rather than an inherited approximation.
      Any remaining keyword goes to `CoupledInternalVariableSolver`, notably
      `bcs`, `scaling_factor`, `quad_degree`, `solver_parameters` and
      `solver_parameters_extra`.

    A note on `sigma_load`, because it appears in three places - as
    `normal_stress` in the mechanics boundary conditions, as the sheet in the
    potential residual, and in the inertia row. They must be *one object at one
    time level*. The sheet and the inertia row both read
    `layout.gravity_form.sigma_bcs`, so those two cannot drift; the mechanics
    condition is the caller's `bcs` and is not checked. If the load is time
    dependent, make it a `Function` (or an expression in a `Constant`) updated
    in place, so every form follows it; rebuilding it per step leaves two of the
    three uses stale.
    """

    name = "SelfGravitatingGIA"

    def __init__(
        self,
        solution: Function,
        approximation: BaseGIAApproximation,
        /,
        *,
        layout: GIASpaceLayout,
        dt: float,
        rotation_moments: dict[str, Any] | None = None,
        Omega_sq: Number | Constant = OMEGA_SQ_EARTH,
        fluid_core: "FluidCore | Mapping | None" = None,
        internal_variables: "list | None" = None,
        **kwargs,
    ) -> None:
        self.layout = layout
        self._check_block0_split_matches_layout(kwargs.get("solver_parameters"))
        # Under static condensation the internal variables are not unknowns:
        # they are stored `Function`s carried between steps, exactly as the
        # segregated `InternalVariableSolver` carries them.
        if layout.condensed:
            if internal_variables is None:
                internal_variables = [
                    Function(layout.internal_variable_space,
                             name=f"internal_variable_{k}")
                    for k in range(len(approximation.maxwell_times))]
            if len(internal_variables) != len(approximation.maxwell_times):
                raise ValueError(
                    f"{len(internal_variables)} internal variables for "
                    f"{len(approximation.maxwell_times)} Maxwell times.")
            kwargs["condense_internal_variables"] = True
        elif internal_variables is not None:
            raise ValueError(
                "`internal_variables` is only meaningful when the space was "
                "built with `condense_internal_variables=True`; without it the "
                "internal variables are sub-fields of the mixed space.")
        self.internal_variables = internal_variables
        if isinstance(fluid_core, Mapping):
            fluid_core = FluidCore(**fluid_core)
        self.fluid_core = fluid_core
        self.form = layout.gravity_form
        # Set before delegating upward: the base constructor writes `self.mesh`
        # and the property setter below needs somewhere to have already put the
        # mechanics mesh.
        self._mesh = layout.mechanics_mesh
        self.potential_mesh = layout.potential_mesh
        self.Omega_sq = ensure_constant(Omega_sq)
        self.rotation_moments = dict(rotation_moments or {})

        if approximation.self_gravity_number is None:
            raise ValueError(
                "The approximation carries no `self_gravity_number`, so it "
                "describes a system with no gravitational potential; "
                "SelfGravitatingGIASolver needs Lambda = 4 pi G rho_bar L / "
                "g_bar.")
        self.Lambda = approximation.self_gravity_number
        self._check_self_gravity_number()

        self.set_measures()
        # Before the base class, which builds a residual and a solver: a
        # constructor that is going to refuse a mesh should refuse it before
        # paying for either.
        self.check_geometry()
        self.check_fluid_core(kwargs.get("bcs") or {})
        self.form.warn_on_quadrature_rule_limits()
        self.set_monopole_datum()
        super().__init__(solution, approximation, dt=dt, **kwargs)

    def _check_block0_split_matches_layout(self, solver_parameters) -> None:
        """Refuses a block-0 fieldsplit that disagrees with the space it acts on.

        `selfgrav_dtn_iterative_solver_parameters(condensed=...)` and the space's
        own `condense_internal_variables` are independent arguments, and if they
        disagree **nothing else notices**. A caller who builds an uncondensed
        space and asks the preset for `condensed=True` gets a two-way fieldsplit
        over a three-field block: `m` is swept by nothing, `u` and `psi` are
        given each other's preconditioners, and the only symptom is block 0
        running to its iteration cap - which is exactly the failure that cost a
        run while this solver was being built, and exactly the failure mode that
        killed two production jobs on the direct preset. A docstring asking the
        caller to keep the two in step is not enough, because the two are set in
        different places by different people.

        The number of `dtn_fieldsplit_0_pc_fieldsplit_N_fields` entries is the
        preset's own statement of how many fields it thinks block 0 has, so it
        is compared against how many the space actually has. Anything that does
        not set those keys - a hand-written dictionary, a string preset - is not
        making a claim and is left alone.
        """
        if not isinstance(solver_parameters, Mapping):
            return
        prefix = "dtn_fieldsplit_0_pc_fieldsplit_"
        n_split = sum(1 for k in solver_parameters
                      if k.startswith(prefix) and k.endswith("_fields"))
        if n_split == 0:
            return
        n_space = 2 + len(self.layout.internal_variables)  # u, psi, and any m
        if n_split == n_space:
            return
        condensed = self.layout.condensed
        raise ValueError(
            f"The block-0 fieldsplit describes {n_split} fields but the mixed "
            f"space has {n_space} in block 0 (displacement, potential and "
            f"{len(self.layout.internal_variables)} internal variable(s)); the "
            f"space was built with condense_internal_variables="
            f"{condensed}. Change ONE of:\n"
            f"  - the space, via self_gravitating_gia_space("
            f"condense_internal_variables={n_split == 2}), or\n"
            f"  - the parameters, via selfgrav_dtn_iterative_solver_parameters("
            f"condensed={condensed}).\n"
            "They are independent arguments and must agree: a mismatched split "
            "sweeps some fields with the wrong preconditioner and others with "
            "none, which does not raise and shows up only as block 0 running "
            "to its iteration cap.")

    # -- The one attribute the base class gets wrong on two meshes ----------

    @property
    def mesh(self):
        """The *mechanics* mesh, never the mixed space's mesh sequence."""
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        # `StokesSolverBase.__init__` assigns `self.solution_space.mesh()`,
        # which on a space spanning two meshes is a `MeshSequenceGeometry` -
        # not a mesh, and not something `upward_normal` or `is_cartesian` can
        # use, since both go through `.unique()`. Discard it and keep the
        # mechanics mesh. A genuine mesh is still accepted, so the property is
        # transparent to anything but the two-mesh case.
        if isinstance(value, MeshGeometry):
            self._mesh = value

    # -- Construction -------------------------------------------------------

    def _check_self_gravity_number(self) -> None:
        """Refuses a `Lambda` that disagrees with the one the sheets were built with.

        The factory turns `Lambda` into the boundary form's gravitational
        constant `Lambda / (4 pi)`, because the sheets carry an explicit
        `4 pi G sigma` while the volume source has the `4 pi G` absorbed. If the
        two `Lambda`s disagree, the sheet is scaled against the volume source by
        a constant - the one error class a symmetry test provably cannot see,
        since it is not a symmetry statement at all - and the geoid is wrong by
        that factor with a perfectly plausible magnitude.
        """
        declared = self.layout.self_gravity_number
        if declared is None:
            if self.form.sigma_bcs:
                raise ValueError(
                    "The space factory was given no `self_gravity_number`, so "
                    "the boundary form's gravitational constant is 1, but this "
                    "configuration has mass sheets. The sheets would then be "
                    "scaled against the volume source by 4 pi / Lambda. Pass "
                    "`self_gravity_number=Lambda` to "
                    "`self_gravitating_gia_space`.")
            return
        if not np.isclose(float(declared), float(self.Lambda)):
            raise ValueError(
                f"self_gravity_number disagrees between the space factory "
                f"({float(declared)!r}) and the approximation "
                f"({float(self.Lambda)!r}). The first scales the mass sheets "
                "and the second the volume source; they must be the same "
                "number.")

    def set_measures(self) -> None:
        """The two volume measures of road-map §5.4.

        `dx_g` is the potential's, over the whole parent; `dx_m` is the
        mantle's. Both are intersected with the other mesh when the two differ,
        and the Laplacian's needs it even though the Laplacian mentions no
        submesh field. That is not obvious and it is not an optimisation missed:
        the *arguments* of this residual live on a mixed space spanning both
        meshes, so a plain parent measure assembles the whole residual happily
        and then fails inside `AssembledPC`, which extracts the non-`Real` block
        - displacement, internal variables and potential together - and compiles
        it as a form whose argument still carries the submesh. The error is
        `MismatchingDomainError` raised from `tsfc/driver.py`, at
        preconditioner setup rather than at assembly, and its message names the
        remedy.

        Intersecting does **not** restrict the Laplacian to the mantle:
        Firedrake keeps the parent's own cells and uses the intersection only to
        widen the set of admissible domains. `check_geometry` asserts exactly
        that, because the alternative reading - a Laplacian solved on the mantle
        alone with the buffer silently absent - is a converged solve with a
        wrong potential.

        The boundary measures belong to `DtNGravityForm`, at its own calibrated
        quadrature degree. That matters and is easy to lose: a DtN constraint
        row silently integrated at `Equation`'s default degree instead would be
        a *wrong constraint with no warning*, in the same family as a silently
        empty measure. `check_boundary_quadrature` is the instrument that
        measures whether the rule actually resolves the modes, and it is
        reachable on this class.
        """
        parent, sub = self.potential_mesh, self.mesh
        #: Plain, un-intersected, and used only by the enclosed-mass system,
        #: whose arguments are `Real` fields on the parent and nothing else.
        self.dx_parent = Measure("dx", domain=parent)
        if self.layout.cross_mesh:
            self.dx_g = Measure(
                "dx", domain=parent,
                intersect_measures=(Measure("dx", domain=sub),))
            self.dx_m = Measure(
                "dx", domain=sub,
                intersect_measures=(Measure("dx", domain=parent),))
        else:
            self.dx_g = self.dx_parent
            self.dx_m = Measure("dx", domain=sub)

    def check_geometry(self) -> None:
        """Measures every boundary, sheet and cross-mesh measure. Called from `__init__`.

        Three checks, all of them for failure modes that produce a converged
        solve rather than an exception:

        1. Every DtN boundary and every sheet is measured against the perimeter
           (2-D) or area (3-D) of a circle/sphere of its own measured radius. A
           tag that matched nothing gives zero; a tag that matched half a circle
           gives half. The tolerance is loose (1%) because the point is to catch
           a missing or partial boundary, not to grade the mesh: straight facets
           on a coarse annulus already cost 4e-4 relative and a P2-curved one
           1e-8, and both are fine.
        2. The intersected `dx_m` assembles to the submesh's own volume.
        3. A *parent* coefficient integrated over `dx_m` gives the same volume.
           This is the one that tests the cross-mesh entity maps rather than the
           measure, and it is the shape of both coupling terms. Phase 2 measured
           1.2e-08 against the closed form for the curved pair and 4.02e-04 for
           an un-recurved submesh, the latter being what a caller gets who
           curves the parent and forgets that `Submesh` does not inherit P2
           coordinates - an error concentrated exactly at Rc and Re, which is
           where the interface mass sheets carrying the entire source live.
        """
        dim = self.potential_mesh.geometric_dimension
        rtol = 1e-2

        X = SpatialCoordinate(self.potential_mesh)
        r = sqrt(dot(X, X))

        def circle(tag, label, integral_type="exterior_facet"):
            """Measure one tagged surface and compare it with its own radius."""
            if integral_type == "exterior_facet":
                dss = self.form.ds(tag)
                extent = assemble(Constant(1.0) * dss)
                radius = assemble(r * dss)
            else:
                # `avg`, never a hard-coded `'+'`: the restriction sides on a
                # tagged interior facet are consistent only by gmsh's cell
                # ordering, and `avg` of a single-valued trace is exact.
                dss = self.form.dS(tag)
                extent = assemble(avg(Constant(1.0)) * dss)
                radius = assemble(avg(r) * dss)
            if extent <= 0.0:
                raise ValueError(
                    f"{label} {tag}: the measure is empty. Firedrake reports "
                    "every physical-group label in both facet sets, so a tag "
                    "alone does not say whether its facets are interior or "
                    "exterior; asking for the wrong kind gives zero and a "
                    "warning rather than an error.")
            radius /= extent
            expected = (2 * np.pi * radius if dim == 2
                        else 4 * np.pi * radius ** 2)
            if abs(extent - expected) > rtol * expected:
                raise ValueError(
                    f"{label} {tag} measures {extent:.6e} at mean radius "
                    f"{radius:.6g}, against {expected:.6e} for a complete "
                    f"circle/sphere of that radius. A tag that matches only "
                    "part of the intended surface is not otherwise detectable.")
            return extent

        for bc_id, _ in self.form.dtn_boundaries:
            circle(bc_id, "DtN boundary")
        for bc_id, _, integral_type in self.form.sigma_bcs:
            circle(bc_id, "Sheet", integral_type)

        volume = assemble(Constant(1.0) * Measure("dx", domain=self.mesh))
        intersected = assemble(Constant(1.0) * self.dx_m)
        if not np.isclose(intersected, volume, rtol=1e-12):
            raise ValueError(
                f"The intersected mantle measure assembles to {intersected:.6e} "
                f"against the submesh's own volume {volume:.6e}. An intersected "
                "measure whose intersection is empty returns zero silently.")

        parent_volume = assemble(Constant(1.0) * self.dx_parent)
        if not np.isclose(assemble(Constant(1.0) * self.dx_g), parent_volume,
                          rtol=1e-12):
            raise ValueError(
                "The intersected parent measure does not cover the whole "
                "parent mesh, so the Laplacian would be solved on the mantle "
                "alone with the stand-off buffer and the DtN boundaries "
                "silently absent.")

        if self.layout.cross_mesh:
            probe = Function(
                FunctionSpace(self.potential_mesh, "CG", 1)).assign(1.0)
            crossed = assemble(probe * self.dx_m)
            if not np.isclose(crossed, volume, rtol=1e-10):
                raise ValueError(
                    f"A parent coefficient integrated over the mantle gives "
                    f"{crossed:.6e} against the mantle volume {volume:.6e}. "
                    "The cross-mesh entity maps are not what the coupling terms "
                    "assume.")

    # -- The fluid core -----------------------------------------------------

    def check_fluid_core(self, bcs: Mapping) -> None:
        """Measures the CMB facet and refuses a doubly-specified boundary.

        Two failure modes, both of them quiet:

        1. **The tag asked for as the wrong kind gives zero and a warning**, not
           an error, so a fluid core written on a tag that carries no exterior
           facet of the *mechanics* mesh contributes nothing: the solve
           converges, the CMB is free, and the difference from a correct run is
           a plausible-looking factor in the deep response. Measured here
           against `2 pi R` (`4 pi R^2` in 3-D) of the facet's own mean radius,
           exactly as `check_geometry` measures every sheet.
        2. **`un` or `u` on the same boundary.** The fluid core's whole content
           is a normal traction, and a strong or Nitsche condition on the normal
           displacement makes it inert - `un = 0` sets the very quantity the
           traction is proportional to. The two are the *switch*, not layers:
           the point of keeping `un = 0` is to measure the difference between
           the treatments, which is only a measurement if exactly one is active.
        """
        if self.fluid_core is None:
            return

        tag = self.fluid_core.boundary
        clash = sorted(k for k, v in bcs.items()
                       if k == tag and ({"u", "un"} & set(v)))
        if clash:
            raise ValueError(
                f"Boundary {tag} carries both a `fluid_core` and a "
                f"{sorted({'u', 'un'} & set(bcs[tag]))} condition. The fluid "
                "core is a traction proportional to the normal displacement, "
                "which `un` pins; they are alternatives, and `un = 0` is the "
                "rigid-core switch the fluid core replaces.")

        dss = Measure("ds", domain=self.mesh)(tag)
        try:
            extent = assemble(Constant(1.0) * dss)
        except (KeyError, LookupError):
            # A tag the mechanics mesh has never heard of raises out of PyOP2
            # as a bare `KeyError: (4,)`, several frames from anything naming a
            # boundary condition. A tag it knows as the *other* kind of facet
            # assembles to zero instead. Both mean the same thing here.
            extent = 0.0
        if extent <= 0.0:
            raise ValueError(
                f"The fluid core's boundary {tag} has an empty measure on the "
                "mechanics mesh. Firedrake reports every physical-group label "
                "in both facet sets, so a tag alone does not say whether its "
                "facets are interior or exterior; asking for the wrong kind "
                "gives zero and a warning rather than an error. On the parent "
                "mesh this tag is an interior facet - the fluid core is built "
                "on the mechanics mesh, where it is an exterior one.")
        X = SpatialCoordinate(self.mesh)
        radius = assemble(sqrt(dot(X, X)) * dss) / extent
        dim = self.mesh.geometric_dimension
        expected = 2 * np.pi * radius if dim == 2 else 4 * np.pi * radius ** 2
        if abs(extent - expected) > 1e-2 * expected:
            raise ValueError(
                f"The fluid core's boundary {tag} measures {extent:.6e} at mean "
                f"radius {radius:.6g}, against {expected:.6e} for a complete "
                "circle/sphere of that radius. A tag that matches only part of "
                "the interface is not otherwise detectable.")

    def fluid_core_measure(self) -> Measure:
        r"""The CMB measure: the mantle's own `ds`, intersected with the parent's `dS`.

        This is the resolution of the one structural obstacle the fluid core
        raises, and it is worth stating in full, because **two** of the obvious
        routes are wrong and one of them is wrong *silently*.

        The CMB sheet is the **first sheet whose density depends on the
        displacement**. Every other sheet in this system - the ice load at Re
        included - is a pure spatial expression on the parent, so it goes
        through `DtNGravityForm.sheet_integral`, which restricts with `avg` on
        the parent's `dS(tag)`. That route is not available as it stands: the
        two sides of the parent's Rc facet are a mantle cell and an
        inner-region cell, `u` exists on only one of them, and UFL raises
        `Inconsistent restrictions: current restriction = -, while default
        restriction = None`.

        **The trap is the intersection, and it does not raise.** A facet
        integral must be intersected with the other mesh's *facet* measure -
        `Measure("dS", domain=parent)` here - and not with its cell measure.
        Intersecting a `ds` on the submesh with `Measure("dx", domain=parent)`
        assembles perfectly happily and evaluates the parent's field at the
        wrong points: measured on the development annulus, a parent CG2 `x^2`
        integrated over the Re circle gives **26.42 against an exact
        33.62** - a 21 % error, mesh-independent in character, with no warning
        of any kind. It is invisible to any check whose parent-side integrand is
        constant, which is exactly what a measure check against `2 pi R` is. The
        same pairing is what `Equation` builds for the momentum equation's
        boundary measure, so **a CMB traction written as a `normal_stress`
        boundary condition containing `psi` would be silently wrong by that
        21 %**; that alone is reason enough for this term to live on the solver
        rather than in a driver's `bcs`.

        With the facet-to-facet pairing, both spellings work and agree to the
        last bit. Measured, `int (u.n) x^2 ds` at Rc with `u = X`, against the
        closed form `-6.59512323`:

            submesh ds, avg on the parent field   -6.59479797
            parent  dS, avg on the parent field   -6.59479797

        and their assembled `(u, psi)` blocks differ by exactly `0.0`. The
        mantle's own `ds` is the one used here, for three reasons:

        - **`n` is unambiguous.** `FacetNormal` of the mechanics mesh is the
          mantle's outward normal, so `dot(u, n)` needs no restriction and no
          sign convention beyond the one the geometry gives it.
        - **Both blocks share the measure**, necessarily, because they are two
          variations of one energy written once. A transpose test on blocks
          built over different measures fails for a reason unrelated to the
          physics.
        - It is the same facet set the mechanics' own boundary conditions use,
          so the fluid core and a `normal_stress` at Re are integrated over
          consistent geometry.

        The parent's `psi` is still two-valued on that facet as far as the
        parent is concerned, so it is restricted with `avg` in
        `fluid_core_energy` - never a hard-coded `'+'`, exactly as
        `DtNGravityForm.sheet_integral` argues.
        """
        # `DtNGravityForm`'s **calibrated** boundary degree, not the solver's
        # volume `quad_degree`. Every other sheet in this system is integrated
        # at that degree, and calibrating it was a piece of work in its own
        # right (`NOTES/FINDING-QUADRATURE-DEGREE-FORM.md`); the one sheet whose
        # density is an unknown is not the one that should opt out of it. The
        # difference is silent either way - a boundary rule that under-resolves
        # is a wrong coefficient, not an error - which is the argument for
        # having a single answer rather than two.
        kwargs = {"domain": self.mesh, "degree": self.form.quad_degree}
        if self.layout.cross_mesh:
            # The parent's *facet* measure. Pairing this `ds` with the parent's
            # cell measure instead is the silent 21 % error above.
            kwargs["intersect_measures"] = (
                Measure("dS", domain=self.potential_mesh),)
        # `ds(...)` and not `Measure("ds", ...)`: only the call on the measure
        # object takes `degree` and `intersect_measures`.
        return ds(**kwargs)

    def fluid_core_sheet(self):
        r"""`sigma = rho_core (u . rhat) = -rho_core (u . n)`, the CMB mass sheet.

        The core's half of the interface mass redistribution, in the same
        convention as every `interior_sigma`: a surface density, positive where
        mass has been added. The mantle's half is *not* here and does not need
        to be - the divergence-form volume source generates it automatically,
        at Rc as `sigma_auto = rho_0 (u . n) = -rho_0 u_r`, which is the mantle
        vacating the shell a rising CMB sweeps out. The two together are the
        physical `(rho_core - rho_0) u_r`, and FC-1 measures their cancellation
        at 1.9e-16 when the two densities are equal.

        **Why this is not registered in `DtNGravityForm.sigma_bcs`**, which is
        where every other sheet lives and where `inertia_form` finds them.
        Three reasons, of which the first two are hard:

        1. **It would double count.** `boundary_source` adds
           `4 pi G sigma v` for every entry, and this sheet is already in the
           potential row as the `psi`-variation of `fluid_core_energy`. Taking
           it out of the energy instead is not an option: the energy is what
           makes the `(u, psi)` and `(psi, u)` blocks transposes by
           construction, and splitting them across two mechanisms is exactly
           the arrangement whose sign nobody can check.
        2. **`sheet_integral` cannot integrate it.** It restricts with `avg` on
           the *parent's* `dS(bc_id)`, whose two sides are a mantle cell and an
           inner-region cell, and `u` exists on only one of them; UFL raises
           `Inconsistent restrictions`. This sheet needs the facet-to-facet
           intersected measure `fluid_core_measure` builds, and that measure
           belongs to the solver, not to the boundary form.
        3. `sigma_bcs` is filled in `DtNGravityForm.__init__`, which runs
           inside `self_gravitating_gia_space` - before the mixed function
           exists, so there is no `u` to write the density with.

        So the sheet is a first-class thing on the solver instead, and every
        consumer that has to see it goes through this method or
        `fluid_core_sheet_integral`. `inertia_form` is the one that matters.
        """
        if self.fluid_core is None:
            return None
        u = self.solution_split[self.layout.displacement]
        return -ensure_constant(self.fluid_core.rho_core) * dot(
            u, FacetNormal(self.mesh))

    def fluid_core_sheet_integral(self, integrand) -> Form:
        """`int_Rc integrand * sigma ds` on the fluid core's own measure.

        An empty form with no fluid core, so callers need no special case. The
        integrand is evaluated on the *mechanics* mesh, which is what the
        measure integrates over: a polynomial of position is the same
        polynomial on either mesh at the same point, and writing it on the
        mantle's coordinates keeps the only cross-mesh object in the form the
        `Real` test function, whose basis function is the global constant 1.
        """
        sigma = self.fluid_core_sheet()
        if sigma is None:
            return Form([])
        dss = self.fluid_core_measure()(self.fluid_core.boundary)
        return integrand * sigma * dss

    def fluid_core_energy(self) -> Form:
        r"""The CMB energy whose variation is the whole fluid-core condition.

            c B_mu int_Rc [ rho_core (u.n) psi
                            + 0.5 rho_core g_0 (u.n)^2 ] ds

        **One energy and not three additions**, which is what makes the three
        blocks it produces - `(u, psi)`, `(psi, u)` and `(u, u)` - symmetric by
        construction rather than by anybody's algebra. The sign convention is
        then fixed by the energy too: `F = c dE/dz` is the convention the rest
        of this residual already obeys, as `self_gravity_term`'s
        `-B_mu int rho_0 grad(psi).w` is the `u`-variation of
        `E_div = -B_mu int rho_0 u.grad(psi) dx` and the potential row's
        `-Lambda int rho_0 u.grad(v)`, scaled by `theta_psi = c B_mu / Lambda`,
        is that same energy's `psi`-variation.

        `FluidCore` documents the physics, the density contrast and the sign
        trap in `dot(u, n)`. Two implementation notes belong here instead:

        - **The `c` is `scaling_factor`**, which
          `CoupledInternalVariableSolver` multiplies the whole momentum residual
          by and which `theta_psi` carries as well. Omitting it here would make
          the coupled Jacobian asymmetric by exactly `c` on any run that used
          one, and that reads as a sign error in new code.
        - **At `B_mu = 0` this term disappears entirely**, body force and sheet
          together, while the potential row keeps its floored `theta_psi` and
          its volume source. That is the honest consequence of writing the pair
          as one energy, and it is not a configuration worth patching around: a
          fluid core whose traction is `B_mu` times something is not present at
          `B_mu = 0` in the first place. The null-coupling gate switches off the
          body forces, and with a fluid core it switches off the core with them.
        """
        fc = self.fluid_core
        u = self.solution_split[self.layout.displacement]
        psi = self.solution_split[self.layout.potential]
        # The mantle's OUTWARD normal. At its inner boundary this points inward,
        # so `dot(u, n) = -u_r`, and `vertical_component(u)` is the wrong vector
        # here - see `FluidCore`.
        un = dot(u, FacetNormal(self.mesh))

        rho_core = ensure_constant(fc.rho_core)
        rho_mantle = (self.approximation.density if fc.rho_mantle is None
                      else ensure_constant(fc.rho_mantle))
        g0 = self.approximation.g if fc.g is None else ensure_constant(fc.g)
        B_mu = self.approximation.B_mu

        dss = self.fluid_core_measure()(fc.boundary)
        # `avg(psi)` and never `psi('+')`: this facet is interior to the parent,
        # so the parent's field is formally two-valued there, and which side is
        # `'+'` is gmsh's cell ordering and nothing else. `psi` is CG, so `avg`
        # of its trace is exact. `u` and `n` are the mantle's own and are
        # single-valued on the measure's own domain.
        psi_face = avg(psi) if self.layout.cross_mesh else psi
        # `-B_mu sigma psi` with `sigma = fluid_core_sheet()`, written through
        # that method so the sheet the inertia row reads and the sheet the
        # potential row carries are one object and cannot drift in sign. It is
        # `+B_mu rho_core (u.n) psi` once expanded, and the minus is the same
        # one every sheet carries into the scaled potential row.
        E = -B_mu * self.fluid_core_sheet() * psi_face * dss
        # The spring carries `rho_core` ALONE (default). The prestress volume
        # term already supplies the mantle half `-0.5 rho_0 g u_r^2` at the CMB
        # (measured to 2.9e-15 relative, `NOTES/measurements/cmb_prestress_check.py`), so
        # the net becomes `0.5 (rho_core - rho_0) g u_r^2` = the physical contrast
        # spring. Writing the contrast HERE double-counts the mantle half and is
        # 7.3x too soft; `buoyancy_density="contrast"` restores it for baselines.
        if fc.buoyancy_density == "contrast":
            spring_rho = rho_core - rho_mantle
        elif fc.buoyancy_density == "core":
            spring_rho = rho_core
        else:
            raise ValueError(
                f"FluidCore.buoyancy_density must be 'core' or 'contrast', "
                f"not {fc.buoyancy_density!r}")
        E += 0.5 * B_mu * spring_rho * g0 * un * un * dss
        return self.scaling_factor * E

    def fluid_core_rotational_traction(self) -> Form:
        r"""`+c B_mu rho_core (w.n) psi_rot ds`: the centrifugal traction on the CMB.

        **The transpose partner of the fluid core's contribution to `dI`, and
        without it the coupled Jacobian is asymmetric by 97 %.** Measured on the
        development annulus before this term existed: the `(u, m_3)` and
        `(m_3, u)` blocks differed by 9.07e-04 against a block maximum of
        9.32e-04, where the rigid core gives 2.6e-15. It went unseen because
        FC-2 ran with rotation off; it does not any more.

        The physics is the statement the module docstring already makes about
        the two body forces. `psi_rot` is the *negated* centrifugal
        perturbation precisely so that it enters alongside `psi` with the same
        sign, and that extends to the interface: the changed centrifugal
        potential presses on the core boundary exactly as the changed
        gravitational one does. The total CMB traction is therefore

            tau = B_mu [ rho_core (psi + psi_rot)
                         + (rho_core - rho_0) g_0 (u.n) ]

        and only the `psi_rot` part is written here.

        **Why it is not in `fluid_core_energy`, where it obviously belongs.**
        Because it would then be counted twice. Adding `psi_rot` to that energy
        gives `derivative` both variations: the `(u, m_i)` block wanted here
        *and* an `(m_i, u)` block equal to `-c B_mu Omega_sq int sigma p_i`,
        which the rotation row already carries through `inertia_form`. The two
        are identical - `theta_rot_i s_i = c B_mu Omega_sq` exactly - so the
        sheet's contribution to the closure would be doubled and the
        transposition would break in the other direction. Keeping `dI` complete
        in one place and writing this one variation by hand is the arrangement
        with a single meaning per form; the price is that its sign is algebra
        rather than construction, which is why FC-2 now measures it with
        rotation on.
        """
        if self.fluid_core is None or not self.layout.rotation:
            return Form([])
        w = self.tests[self.layout.displacement]
        n = FacetNormal(self.mesh)
        psi_rot = self.rotational_potential_expression(self.mesh)
        dss = self.fluid_core_measure()(self.fluid_core.boundary)
        return (self.scaling_factor * self.approximation.B_mu
                * ensure_constant(self.fluid_core.rho_core)
                * dot(w, n) * psi_rot * dss)

    def fluid_core_residual(self) -> Form:
        """The whole fluid-core contribution, or an empty form without one.

        `dE/dz` of `fluid_core_energy` - `derivative` rather than three
        hand-written terms, deliberately, since it is the only spelling in
        which the `(u, psi)`/`(psi, u)` transpose cannot be got wrong - plus
        the rotational traction, which cannot go in that energy without being
        counted twice. `fluid_core_rotational_traction` explains why.
        """
        if self.fluid_core is None:
            return Form([])
        return (derivative(self.fluid_core_energy(), self.solution)
                + self.fluid_core_rotational_traction())

    # -- The scaling constants ---------------------------------------------

    @property
    def _row_scale_B_mu(self):
        r"""`B_mu`, floored to `NULL_COUPLING_ROW_SCALE` when it is exactly zero.

        **This is the fix for the `B_mu = 0` singular Jacobian**, and the
        reasoning matters more than the line. `theta_psi` and `theta_rot` are
        multiplicative scalings of *whole residual rows* - the potential row,
        every DtN constraint row, the rotation row. Following `B_mu` to zero
        multiplies twenty-odd rows of the Jacobian by zero, and the result is
        not a decoupled system but a structurally singular one:
        `max|A(psi,psi)| = 0`, `DIVERGED_LINEAR_SOLVE` at the first Krylov
        iteration, with a message that names the linear solver and not the
        cause.

        There is no mathematical reason for the scalings to vanish. Both are
        *derived* from a symmetry condition on a coupling block, and at
        `B_mu = 0` that block is identically zero, so the condition reads
        `0 = 0` and fixes nothing at all. Any nonzero row scaling is then
        equally correct, because a row scaling is unobservable in the solution
        (see the module docstring): the system is genuinely one-way at
        `B_mu = 0` - `u` and the internal variables solve on their own, and
        `psi` and `m_3` are driven by them - and one-way systems do not have
        symmetric Jacobians whatever is done to their rows.

        `NULL_COUPLING_ROW_SCALE = 1.0` is chosen rather than, say, the
        approximation's nominal `B_mu`, because 1 is the only value that needs
        no explanation and no bookkeeping: `theta_psi` becomes
        `scaling_factor / Lambda`, which is the scaling the potential row would
        carry in a system with no mechanics at all.

        The floor is a strict equality test on zero and not a tolerance, so a
        run with a small-but-nonzero `B_mu` is untouched and continues to scale
        continuously - a tolerance would put a discontinuity in the middle of
        the parameter range and a continuation study would walk straight into
        it. For every `B_mu != 0` this property returns the approximation's own
        `Constant`, unchanged, so nothing about a production run differs by a
        single bit.

        One caveat, stated because the value is read once. `B_mu` is a
        `Constant` and the residual is built at construction, so the decision is
        frozen there: an approximation whose `B_mu` is zero at construction and
        assigned a nonzero value afterwards keeps the floored scaling. The
        solution stays correct - it is still only a row scaling - but the
        Jacobian is then asymmetric by exactly `B_mu`, so build a second solver
        rather than reassigning if symmetry matters.
        """
        B_mu = self.approximation.B_mu
        if float(B_mu) == 0.0:
            return NULL_COUPLING_ROW_SCALE
        return B_mu

    @property
    def theta_psi(self):
        r"""`scaling_factor * B_mu / Lambda`: the potential rows' scaling.

        Derived, not quoted. Write the two off-diagonal blocks as
        `J_upsi[w, psi] = c int rho_0 grad(psi) . w` and
        `J_psiu[v, u] = d int rho_0 u . grad(v)`. Transposition of a bilinear
        form pairs `c` against `d` with the *same* sign, so symmetry is `c = d`,
        not `c = -d`.

        With the minus of `self_gravity_term` and the solver's own residual
        scaling, `c = -f B_mu` where `f` is `scaling_factor`
        (`CoupledInternalVariableSolver` multiplies the whole momentum residual
        by it, this term included), and the potential row gives
        `d = -theta_psi Lambda`. Hence `theta_psi = f B_mu / Lambda`, which is
        1.1487 for the Earth values at `f = 1`.

        The `f` is the part that is easy to lose, and losing it makes the
        Jacobian asymmetric by exactly `f` on any run with a rescaled momentum
        residual - which reads as a sign error in the new code.

        At `B_mu = 0` the numerator is `_row_scale_B_mu`'s floor rather than
        zero, because a row multiplied by zero is a *deleted* equation and not
        a decoupled one; see that property.
        """
        return self.scaling_factor * self._row_scale_B_mu / self.Lambda

    def _theta_rot(self, i: int):
        r"""`s_i * scaling_factor * B_mu * Omega_sq`: the `i`th rotation row's scaling.

        **A third independent constant, and not `theta_psi`**: it carries an
        `Omega^2` that `theta_psi` has none of. The derivation, in full, because
        the road map asserted the wrong answer once:

        `rotational_potential` builds `psi_rot = sum_i m_i P_i(x)` with
        `P_i = Omega_sq p_i`, `p_3 = x^2 + y^2`, `p_1 = -x z`, `p_2 = -y z`. For
        each `i`, `int rho_0 grad(p_i) . u` is exactly the inertia perturbation
        `dI_i3[u]` of road-map §3.1 - check `i = 3`:
        `grad(p_3) . u = 2(x u_1 + y u_2) = 2(x.u - z u_3)`, which is §3.1's
        `dI_33`. The body force and the inertia perturbation are the *same*
        bilinear form, which is the reason to prefer the volume integral over a
        rescaled DtN coefficient.

        So, with `f` the residual scaling factor:

            J[u, m_i] = -f B_mu Omega_sq dI_i3[w]        (from the MINUS body force)
            J[m_i, u] = -theta_rot_i s_i dI_i3[du]       (from the closure row)

        and they transpose when `theta_rot_i s_i = f B_mu Omega_sq`, i.e.

            theta_rot_i = s_i f B_mu Omega_sq

        since `s_i = +-1`. **Negative on the `m_3` row**, whose closure sign
        `s_3 = -1` differs from the polar-wander pair - and `m_3` is the only
        component a disc has, so the 2-D prototype exercises only the row with
        the negative scaling.

        Never fit this to make a symmetry test pass. The `(m_3, m_3)` diagonal
        is 1x1 and symmetric for any value, so a flip in the closure sign is
        exactly compensated by a flip in the row scaling and the test reports
        success.

        At `B_mu = 0` the factor is `_row_scale_B_mu`'s floor rather than zero,
        for the reason given there: the closure row `K_3 m_3 = -dI_33` is still
        a real equation when the body force it transposes against is absent,
        and scaling it by zero deletes it.
        """
        return (self.CLOSURE_SIGNS[i] * self.scaling_factor
                * self._row_scale_B_mu * self.Omega_sq)

    #: `s_i` of the closure `K_i m_i = s_i dI_i3`. The polar-wander pair is
    #: `m_1 = dI_13/(C-A)`, `m_2 = dI_23/(C-A)`; the rotation-rate change is
    #: `m_3 = -dI_33/C`, a different constant *and* a different sign.
    CLOSURE_SIGNS = (1.0, 1.0, -1.0)

    def _closure_constant(self, i: int):
        """`K_i`: `C - A` for the polar-wander pair, `C` for `m_3`."""
        key = "C" if i == 2 else "C_minus_A"
        if key not in self.rotation_moments:
            raise ValueError(
                f"The rotational closure for {self.layout.rotation_names[i]} "
                f"needs `rotation_moments[{key!r}]`. In 2-D `C` is the disc's "
                "polar second moment `int rho_0 r^2 dV`, which you can "
                "assemble; in 3-D `C - A` is the dynamical ellipticity of the "
                "hydrostatic figure and cannot be computed from a spherically "
                "symmetric reference density at all.")
        return ensure_constant(self.rotation_moments[key])

    # -- Fields -------------------------------------------------------------

    @property
    def displacement(self) -> Function:
        """The displacement sub-function, on the mechanics mesh."""
        return self.solution.subfunctions[self.layout.displacement]

    @property
    def potential(self) -> Function:
        """The gravitational potential sub-function, on the parent mesh."""
        return self.solution.subfunctions[self.layout.potential]

    def rotation_values(self) -> dict[str, float]:
        """The solved rotation scalars, by name.

        `Real` data is replicated on every rank, so `float()` on the
        sub-function is a local operation that agrees everywhere.
        """
        return {name: float(self.solution.subfunctions[i])
                for name, i in self.layout.rotation.items()}

    def coefficients(self) -> dict[int | str, dict[str, float]]:
        """Solved trace coefficients of every DtN boundary, keyed by marker.

        The spectrum of `psi` on each boundary; at the surface, the geoid
        coefficients. Same contract as `GravitySolver.coefficients`.
        """
        out = {bc_id: {} for bc_id, _ in self.form.dtn_boundaries}
        for (bc_id, key), i in zip(
                self.form.multiplier_keys, self.layout.multipliers):
            out[bc_id][key] = float(self.solution.subfunctions[i])
        return out

    def geoid(self, *, include_rotation: bool = True):
        r"""The geoid height as UFL on the parent mesh, `N = +(psi + psi_rot)/g_0`.

        Evaluate at the surface for the geoid proper. The plus sign is derived
        in `BaseGIAApproximation.geoid` and is the one sign in this project that
        reaches the science rather than the algebra: it feeds
        `SL = SL_0 + dphi - du_r` in the sea-level equation, so getting it wrong
        inverts self-attraction and loading while leaving the magnitude entirely
        plausible, and no symmetry test can see it.

        The rotational contribution enters with the same sign, which is the
        payoff for `psi_rot` being the negated centrifugal perturbation. It is
        included by default because the geoid a sea-level solver wants is the
        total one; pass `include_rotation=False` for the gravitational part
        alone.
        """
        psi = self.solution_split[self.layout.potential]
        total = self.approximation.geoid(psi)
        if include_rotation and self.layout.rotation:
            total = total + self.approximation.geoid(
                self.rotational_potential_expression(self.potential_mesh))
        return total

    # -- Rotation -----------------------------------------------------------

    @staticmethod
    def inertia_polynomial(i: int, X):
        r"""`p_i(x)`, whose gradient contracts with `u` to give `dI_i3`.

        `p_3 = x^2 + y^2`, `p_1 = -x z`, `p_2 = -y z`, written in three
        components unconditionally and indexing `X[2]` only where it exists, so
        that 2-D and 3-D share one code path. `psi_rot = Omega_sq sum_i m_i p_i`
        exactly, which is what makes the rotational body force the transpose of
        the inertia row rather than merely resembling it.

        The sheet contribution to the inertia perturbation is the *same*
        polynomial: road-map §3.1's `int sigma (R^2 delta_ij - x_i x_j) dS` at
        `i3` is `int sigma p_i dS` for every `i` - at `i = j = 3`,
        `R^2 - z^2 = x^2 + y^2 = p_3`, and at `i = 1`, `-x z = p_1`.
        """
        if i == 2:
            return X[0] ** 2 + X[1] ** 2
        if len(X) < 3:
            raise ValueError(
                f"The inertia component {i} needs a third coordinate; a 2-D "
                "disc has only the rotation-rate change m_3 (index 2).")
        return -X[i] * X[2]

    def rotational_potential_expression(self, mesh):
        """`psi_rot` on the given mesh, from the solved rotation scalars.

        Built on the mechanics mesh for the body force and on the parent for the
        geoid; the polynomial is the same and the two meshes' coordinates agree
        pointwise where they overlap.
        """
        slots = self.layout.rotation_slots()
        n_rot = 1 if mesh.geometric_dimension == 2 else 3
        # A component the space does not carry is the literal zero, not a Real
        # unknown, so every rotation expression can be written in three
        # components whatever the dimension.
        values = [Constant(0.0) if i is None else self.solution_split[i]
                  for i in slots]
        return rotational_potential(
            values[-n_rot:], mesh, Omega_sq=self.Omega_sq)

    def inertia_form(self, i: int, u=None, test=None):
        r"""The inertia perturbation `dI_i3` as a form in `u` plus the sheets.

        Road-map §3.1, in divergence form:

            dI_i3 = int_mantle rho_0 grad(p_i) . u dx + sum_sheets int sigma p_i dS

        and **never** the rescaled DtN coefficient. Two reasons, and the second
        is the operative one: the volume integral does not depend on where the
        DtN boundary sits, and it is the same bilinear form as the rotational
        body force, so its transpose is that body force and the coupled operator
        stays symmetric. (Computing it both ways and comparing is a free
        verification test, and a 3-D one.)

        Args:
          i: component index, 0-based, so `i = 2` is the `dI_33` a disc has.
          u: displacement; the current solution's by default.
          test: a test function to weight the integrand by, which is how the
            closure row gets it. `None` gives the plain scalar functional, i.e.
            a 0-form that `assemble` turns into the number.
        """
        if u is None:
            u = self.solution_split[self.layout.displacement]
        weight = Constant(1.0) if test is None else test
        rho0 = self.approximation.density

        p_m = self.inertia_polynomial(i, SpatialCoordinate(self.mesh))
        form = weight * rho0 * dot(grad(p_m), u) * self.dx_m

        p_g = self.inertia_polynomial(i, SpatialCoordinate(self.potential_mesh))
        for bc_id, sigma, integral_type in self.form.sigma_bcs:
            form = form + self.form.sheet_integral(
                weight * ensure_constant(sigma) * p_g, bc_id, integral_type)

        # The fluid core's own sheet. It is a genuine mass redistribution - the
        # core boundary moves and the core's mass moves with it - so it enters
        # dI exactly as the load sheet does, and omitting it would leave the
        # potential right and the core's contribution to polar motion silently
        # absent. Nothing in FC-1, FC-2 or FC-4 looks at dI, so this would
        # surface only as a polar-motion answer wrong by an unattributable
        # amount. See `fluid_core_sheet` for why it is not in `sigma_bcs`.
        form = form + self.fluid_core_sheet_integral(weight * p_m)
        return form

    def inertia_perturbation(self) -> dict[str, float]:
        """The three `dI_i3` of the current state, as numbers.

        Diagnostic. Keyed `dI_13`, `dI_23`, `dI_33`; in 2-D the first two are
        identically zero and are not computed, since they index a coordinate
        that does not exist.
        """
        out = {}
        for i, name in enumerate(("dI_13", "dI_23", "dI_33")):
            if i < 2 and self.mesh.geometric_dimension == 2:
                out[name] = 0.0
                continue
            out[name] = float(assemble(self.inertia_form(i)))
        return out

    # -- The residual -------------------------------------------------------

    def set_equations(self) -> None:
        """The mechanics equations, with the two body forces added to the momentum one.

        Deliberately *not* the parent's `set_equations`: that does
        `u, *internal_variables = self.solution_split`, which on this space
        would take the potential and every multiplier for an internal variable.
        Everything else about the mechanics is the parent's, term for term and
        scaling for scaling, because a coupling that perturbed the mechanics
        would not be detectable by any gate that compares the two.

        **Every** equation gets `intersect_measures` pointing at the parent,
        including the internal-variable ones, whose integrands mention nothing
        on the parent at all. The reason is the *arguments* rather than the
        integrands: the test and trial functions live on a mixed space spanning
        both meshes, so a plain submesh measure assembles the full residual
        happily and then raises `MismatchingDomainError` from `tsfc/driver.py`
        the first time something extracts a sub-block of the Jacobian and
        compiles it - which is what `AssembledPC` does inside the
        preconditioner, several frames away from anything that mentions an
        internal variable. Measured while building this: the failing integral
        was the internal-variable source term, not either coupling term.
        """
        assert self._theta == 1.0, (
            "SelfGravitatingGIASolver assumes backward Euler. A theta-weighted "
            "psi against an unweighted u in the Poisson source would break the "
            "(u, psi)/(psi, u) transposition by exactly theta, and a symmetry "
            "test would report it as a sign error.")

        u = self.solution_split[self.layout.displacement]
        psi = self.solution_split[self.layout.potential]

        if self.layout.condensed:
            # The backward-Euler update substituted into the stress *before*
            # differentiation, which is what makes the elimination exact:
            #   m_new = (m_old + (dt/tau) d(u)) / (1 + dt/tau)
            # This is `InternalVariableSolver.update_m` verbatim. Two
            # consequences worth stating because both are silent if wrong.
            # The u-tangent of the stress becomes
            # `sum_i eta_i/(tau_i + dt) = effective_viscosity(dt)`, which is why
            # `CoupledInternalVariableSolver.__init__` hands the Nitsche pair
            # that coefficient rather than `mu0` in this configuration. And the
            # power-law factor would become a function of `u` alone rather than
            # of an independent `m`, which is a different Newton linearisation,
            # so condensation is refused for `exponent != 1`.
            strain_u = self.approximation.deviatoric_strain(u)
            if float(getattr(self.approximation, "exponent", 1)) != 1:
                raise NotImplementedError(
                    "Static condensation of the internal variables is "
                    "implemented for Newtonian rheology only (exponent = 1). "
                    "For a power law the Maxwell times depend on the deviatoric "
                    "stress, hence on m, and substituting m(u) changes the "
                    "Newton linearisation rather than merely eliminating a "
                    "block.")
            internal_variables = [
                (m + self.dt / mt * strain_u) / (1 + self.dt / mt)
                for m, mt in zip(self.internal_variables,
                                 self.approximation.maxwell_times)]
            self._internal_variables_update = internal_variables
        else:
            internal_variables = [self.solution_split[i]
                                  for i in self.layout.internal_variables]

        intersect = self.potential_mesh if self.layout.cross_mesh else None

        stress = self.approximation.stress(
            u, internal_variables=internal_variables)
        source = self.approximation.buoyancy(u) * self.k
        strain = self.approximation.deviatoric_strain(u)
        dev_stress = self.approximation.deviatoric_stress(u, internal_variables)
        visc_factor = self.approximation.power_law_factor(dev_stress)
        maxwell_times = [mt * visc_factor
                         for mt in self.approximation.maxwell_times]

        momentum_terms = list(compressible_viscoelastic_terms) + [self_gravity_term]
        momentum_attrs = {"stress": stress, "source": source, "psi": psi}
        if self.layout.rotation:
            momentum_terms.append(rotational_potential_term)
            momentum_attrs["psi_rot"] = self.rotational_potential_expression(
                self.mesh)

        self.equations.append(
            Equation(
                self.tests[self.layout.displacement],
                self.solution_space[self.layout.displacement],
                momentum_terms,
                eq_attrs=momentum_attrs,
                approximation=self.approximation,
                bcs=self.weak_bcs,
                quad_degree=self.quad_degree,
                scaling_factor=self.scaling_factor,
                intersect_measures=intersect,
            )
        )

        for k, maxwell_time in enumerate([] if self.layout.condensed
                                          else maxwell_times):
            i = self.layout.internal_variables[k]
            self.equations.append(
                Equation(
                    self.tests[i],
                    self.solution_space[i],
                    [mass_term, sink_term, source_term],
                    eq_attrs={
                        "source": strain / maxwell_time,
                        "sink_coeff": 1 / maxwell_time,
                        "dt": self.dt,
                        "trial_old": self.solution_old_split[i],
                        "use_irksome": False,
                    },
                    approximation=self.approximation,
                    bcs=self.weak_bcs,
                    quad_degree=self.quad_degree,
                    scaling_factor=-self._theta * self.scaling_factor,
                    intersect_measures=intersect,
                )
            )

    def set_form(self) -> None:
        """Mechanics through the base machinery, gravity and rotation added on.

        The parent's `set_form` zips `self.equations` against
        `self.solution_split`, one `Equation` per sub-field. Here there are
        `1 + N` equations and `2 + N + n_mult + n_rot` sub-fields, and `zip`
        would truncate to the shorter - which happens to pair correctly, and is
        exactly the kind of accident that stops being true when somebody
        reorders the space. Write it out.
        """
        mechanics = zip(self.equations,
                        (self.solution_split[self.layout.displacement],
                         *(self.solution_split[i]
                           for i in self.layout.internal_variables)))
        self.F = sum(eq.residual(sol) for eq, sol in mechanics)
        self.F += self.potential_residual()
        self.F += self.fluid_core_residual()
        if self.layout.rotation:
            self.F += self.rotation_residual()

        self.strong_bcs.extend(
            DirichletBC(self.solution_space.sub(self.layout.potential),
                        val, bc_id)
            for bc_id, val in self.form.dirichlet_bcs)

    def potential_residual(self) -> Form:
        r"""`F_psi` and `F_c`: the Poisson equation, its source, and the DtN rows.

            theta_psi [ int grad(psi).grad(v) dx_g
                        - Lambda int rho_0 u.grad(v) dx_m
                        + boundary_residual(psi, v, multipliers) ]

        **The source is written as a divergence** and not as `-Lambda int rho_1 v`
        with jump terms. Three reasons, of which the operative one is that
        discrete mass conservation is then *exact*: with `v = 1` the form is
        identically zero because `grad(v)` is, and `v = 1` genuinely lies in the
        CG test space, so the discrete perturbation carries exactly zero net
        mass to roundoff regardless of mesh and quadrature - not `O(h^p)`. In
        2-D that keeps the solver inside the regime its monopole and log-gauge
        treatment was built for. The other two: every interface mass is included
        automatically, including the free surface, with no list of interfaces to
        keep in step with the mesh; and it needs one derivative fewer of both
        `rho_0` (so a layered DG0 density is legal with no `jump()`) and `u`.

        Everything after the source line is `DtNGravityForm`, at its own
        calibrated boundary quadrature degree, and `theta_psi` multiplies the
        whole row - the DtN constraint rows included, which is what keeps the
        `(psi, c)` pair scaled consistently with the `(psi, u)` one.
        """
        psi = self.solution_split[self.layout.potential]
        v = self.tests[self.layout.potential]
        u = self.solution_split[self.layout.displacement]
        rho0 = self.approximation.density

        multipliers = [(self.solution_split[i], self.tests[i])
                       for i in self.layout.multipliers]

        F = dot(grad(psi), grad(v)) * self.dx_g
        F -= self.Lambda * rho0 * dot(u, grad(v)) * self.dx_m
        F += self.form.boundary_residual(
            psi, v, multipliers, extra_flux=self.monopole_fluxes())
        return self.theta_psi * F

    def source_mass_form(self):
        """The Poisson source contracted against `v = 1`, which is zero by construction.

        Returned rather than asserted so that a gate can assemble it: this is
        the whole content of the divergence form's exact mass conservation, and
        `int rho_0 u . grad(1) dx` is zero because the gradient is, not because
        anything cancels.

        The constant is built as a `Function` in the potential's own space and
        assigned 1, rather than as a `Constant`, deliberately: `grad` of a
        `Constant` folds to symbolic zero and the test would then be a statement
        about UFL rather than about the discretisation. This way the gradient is
        the real one, evaluated from a full set of unit degrees of freedom, and
        the zero it returns is the discrete statement that `v = 1` lies in the
        test space.
        """
        u = self.solution_split[self.layout.displacement]
        one = Function(
            self.solution_space[self.layout.potential]).assign(1.0)
        return self.Lambda * self.approximation.density * dot(
            u, grad(one)) * self.dx_m

    def rotation_residual(self) -> Form:
        r"""`F_rot`: the rotational closure, one row per component.

            theta_rot_i ( K_i m_i - s_i dI_i3[u, sigma] ) nu_i

        written per component with `K = (C-A, C-A, C)` and `s = (+1, +1, -1)`,
        the 2-D path selecting index 2. That is not a 3-D nicety: `m_3` is the
        only component a disc has, its closure `m_3 = -dI_33/C` carries a
        different constant *and* a different sign from the polar-wander pair,
        and an earlier revision of the design stated the pair's closure for all
        three - so the one row the prototype exercises is the one that was
        stated incorrectly.

        `nu_i` is a `Real` test function, whose basis function is the global
        constant 1, so an integrand multiplied by it assembles to the plain
        integral. The `K_i m_i` term is therefore divided by the mantle volume
        that same measure gives, so that the row means `K_i m_i - s_i dI_i3 = 0`
        and not that times a volume.
        """
        F = Form([])
        volume = assemble(Constant(1.0) * self.dx_m)
        for i, index in enumerate(self.layout.rotation_slots()):
            if index is None:
                continue
            m_i, nu_i = self.solution_split[index], self.tests[index]
            row = (self._closure_constant(i) * m_i / volume) * nu_i * self.dx_m
            row -= self.CLOSURE_SIGNS[i] * self.inertia_form(i, test=nu_i)
            F += self._theta_rot(i) * row
        return F

    # -- The 2-D monopole datum. Mirrors `GravitySolver`; 3-D runs none of it.

    def set_monopole_datum(self) -> None:
        """Prepares the 2-D exterior monopole flux datum.

        A 2-D exterior DtN boundary is the only kind whose `m = 0` mode the
        trace does not determine, because the exterior monopole solution is
        logarithmic. The datum supplies its flux, `-2 G M / R`, with `M`
        everything the boundary encloses. In 3-D the `l = 0` exterior map
        handles net mass exactly and nothing here is built - not even the `Real`
        space - so the 3-D path pays nothing for any of it.

        This is the same construction as `GravitySolver.set_monopole_datum`, and
        deliberately so, but the mass form is not: in the coupled system the
        volume source is the divergence form, whose net mass is *identically*
        zero (see `source_mass_form`), so the enclosed mass is the sheets alone
        - **every** sheet, including the fluid core's, which is not in
        `sigma_bcs` and which `enclosed_mass_forms` therefore adds by hand.
        That is a fact to assert rather than to assume, which is what
        `check_net_mass` is for.
        """
        self.monopole_boundaries = [
            bc_id for bc_id, _ in self.form.dtn_boundaries
            if self.potential_mesh.geometric_dimension == 2
            and self.form.boundary_geometry[bc_id][0] == "exterior"]

        self._real_space = None
        self.mesh_volume = None
        self.source_mass = None
        self.cavity_flux = None
        if not self.monopole_boundaries:
            return

        if len(self.monopole_boundaries) > 1:
            raise ValueError(
                f"2-D mesh with {len(self.monopole_boundaries)} exterior DtN "
                f"boundaries {self.monopole_boundaries}: the monopole datum "
                "needs the mass enclosed by the exterior boundary, which is "
                "not defined when there is more than one of them.")
        clash = sorted(
            {bc_id for bc_id, _ in self.form.flux_bcs}.intersection(
                self.monopole_boundaries), key=str)
        if clash:
            raise ValueError(
                f"Boundary {clash[0]}: 'flux' and a 2-D exterior 'dtn' both "
                "prescribe the m = 0 normal derivative there, so the two are a "
                "double specification of one quantity.")

        self._real_space = FunctionSpace(self.potential_mesh, "R", 0)
        self.mesh_volume = assemble(Constant(1.0) * self.dx_parent)
        self.source_mass = Function(self._real_space, name="source_mass")
        self.cavity_flux = Function(self._real_space, name="cavity_flux")

    def enclosed_mass_forms(self) -> tuple:
        """`(identity, mass, flux)`: the one-by-one systems on the `Real` space.

        `mu` and `nu` are globally constant, so dividing the left-hand side by
        the volume that same measure assembles to makes the system the identity
        and `source_mass` hold the mass its name claims.

        The volume source contributes **nothing**, which is the one difference
        from `GravitySolver`: the divergence form carries exactly zero net mass
        by construction, so the datum is driven by the sheets alone.
        """
        mu = TestFunction(self._real_space)
        nu = TrialFunction(self._real_space)
        identity = (nu / self.mesh_volume) * mu * self.dx_parent

        mass = None
        for bc_id, sigma, integral_type in self.form.sigma_bcs:
            term = self.form.sheet_integral(
                ensure_constant(sigma) * mu, bc_id, integral_type)
            mass = term if mass is None else mass + term

        # The fluid core's sheet, which is *not* in `sigma_bcs` (see
        # `fluid_core_sheet`) and which the datum would otherwise be blind to.
        # Its net mass is the core's volume change, `rho_core oint (u.rhat) ds`,
        # and nothing in the formulation makes that zero: the eliminated core
        # has no pressure degree of freedom with which to enforce its own
        # incompressibility, so the degree-0 CMB deformation is free. Measured
        # on the development annulus with a `cos 2 phi` load it is 8.3e-16,
        # i.e. 1.6e-14 of the sheet's own scale - zero because the load has no
        # degree-0 content, not because the term cannot carry mass.
        #
        # **This contribution is lagged by one solve** and that is the honest
        # price of including it. `update_total_mass` runs at the *start* of
        # `solve`, so `u` is the previous iterate's; for the linear system a
        # single Newton step then sees a datum built from the last step's
        # displacement. Leaving it out instead is not the safer option - that
        # is a datum missing a term outright rather than one evaluated a step
        # late - and `check_net_mass` reports when the term is large enough for
        # the difference to matter. Making it implicit would need a further
        # `Real` unknown in the mixed space, which changes the layout
        # `DtNTwoBlockSchurPC` asserts on, and this is 2-D-lifetime code: in
        # 3-D the `l = 0` exterior map handles net mass exactly and none of
        # this runs.
        core = self.fluid_core_sheet_integral(mu)
        if core.integrals():
            mass = core if mass is None else mass + core

        flux = None
        for bc_id, value in self.form.flux_bcs:
            term = ensure_constant(value) * mu * self.form.ds(bc_id)
            flux = term if flux is None else flux + term
        return identity, mass, flux

    def monopole_fluxes(self) -> dict:
        """`{bc_id: flux}` for `DtNGravityForm.boundary_source`'s `extra_flux`.

        `-2 G M / R` with `G = Lambda / (4 pi)`, plus the cavity term implied by
        any prescribed-flux boundary, whose own `G` cancels by Gauss and is
        therefore written without one.
        """
        out = {}
        for bc_id in self.monopole_boundaries:
            _, R = self.form.boundary_geometry[bc_id]
            flux = -2.0 * self.form.G * self.source_mass / R
            if self.form.flux_bcs:
                flux = flux - self.cavity_flux / (2.0 * np.pi * R)
            out[bc_id] = flux
        return out

    def update_total_mass(self) -> None:
        """Refreshes the enclosed-mass scalars the monopole datum reads.

        Called from `solve` and never from `__init__`, so that the assembles and
        the assigns land on the tape. `GravitySolver.update_total_mass` explains
        at length why the spelling matters - a `Constant.assign(assemble(...))`
        severs the adjoint silently while every Taylor test still passes, and a
        *cached* solver on a `Real` space with a facet integral returns garbage
        after its first solve in parallel - and every word of it applies here.
        Do not simplify this without reading it.
        """
        if not self.monopole_boundaries:
            return

        identity, mass, flux = self.enclosed_mass_forms()
        if mass is not None:
            solve(identity == mass, self.source_mass,
                  solver_parameters=real_scalar_solver_parameters)
        if flux is not None:
            solve(identity == flux, self.cavity_flux,
                  solver_parameters=real_scalar_solver_parameters)
        self.check_net_mass()

    def total_enclosed_mass(self) -> float:
        """Everything the 2-D exterior DtN boundary encloses, as a number.

        Diagnostic only; the differentiable quantities are `source_mass` and
        `cavity_flux`, and reading a float off them is the sever documented in
        `GravitySolver.update_total_mass`.
        """
        if not self.monopole_boundaries:
            return 0.0
        return (float(self.source_mass) + float(self.cavity_flux)
                / (4.0 * np.pi * float(self.form.G)))

    def check_net_mass(self) -> None:
        """Refuses a doubly-anchored gauge; scales the leakage test on the sheets.

        `GravitySolver` measures the net mass against the density's own scale,
        `int |rho| dx`. There is no density here - the volume source is a
        divergence with identically zero mass - so that scale would be zero and
        the relative test would degenerate to `mass > 0`, which is not a test.
        The scale here is the sheets' own, `sum int |sigma|`, and with no sheets
        at all there is nothing whose mass could leak and the check is skipped.

        The refusal is the gauge argument's caveat and is kept verbatim in
        substance: with a strong `psi` condition present, `v = 1` is not
        admissible, so the Robin monopole relation and the Dirichlet condition
        become two competing anchors for the same constant.

        **With a fluid core the warning band is less trustworthy than it
        reads.** The CMB sheet's net mass is the core's volume change, which
        this formulation does not constrain to zero (`enclosed_mass_forms`
        says why), so a genuinely small degree-0 CMB deformation lands in the
        1e-8 to 1e-4 band and is reported as though it were quadrature leakage
        from a non-conforming load. The two are distinguishable by switching
        the fluid core off; the band is left as it is because the leakage it
        was written for is the more common fault.
        """
        if not self.monopole_boundaries:
            return

        scale = 0.0
        for bc_id, sigma, integral_type in self.form.sigma_bcs:
            scale += assemble(self.form.sheet_integral(
                abs(ensure_constant(sigma)), bc_id, integral_type))
        core_scale = self.fluid_core_sheet_integral(Constant(1.0))
        if core_scale.integrals():
            # `abs` of the density rather than of the integral: the scale is
            # "how much mass this sheet moves", not "how much it moves net",
            # and a sheet that redistributes mass without changing it must
            # still set the scale the leakage test is read against.
            sigma_core = self.fluid_core_sheet()
            scale += assemble(
                abs(sigma_core)
                * self.fluid_core_measure()(self.fluid_core.boundary))
        mass = abs(float(self.source_mass))
        relative = mass / scale if scale > 0.0 else 0.0
        anchored = relative > 1e-8 or float(self.cavity_flux) != 0.0

        if anchored and self.form.dirichlet_bcs:
            raise ValueError(
                f"Net enclosed mass {self.total_enclosed_mass():.3e} is nonzero "
                "on a 2-D mesh that has both an exterior DtN boundary and a "
                f"strong 'psi' condition on boundary "
                f"{self.form.dirichlet_bcs[0][0]}. The monopole datum fixes the "
                "potential's additive constant through the Robin term and the "
                "Dirichlet condition fixes it again: the two anchors disagree "
                "and the system is over-constrained.")

        if 1e-8 < relative < 1e-4:
            warn(
                f"Net sheet mass {float(self.source_mass):.3e} is "
                f"{relative:.1e} of the sheets' own scale - nonzero, but too "
                "small to look deliberate. This is the signature of leakage "
                "from a load that does not conform to cell edges.")

    # -- Solver -------------------------------------------------------------

    def set_solver_options(
        self,
        solver_preset: ConfigType | str | None,
        solver_extras: ConfigType | None,
    ) -> None:
        """PETSc options; both presets are two-block DtN Schur splits.

        `"direct"` gives `selfgrav_dtn_schur_solver_parameters` and `"iterative"`
        gives `selfgrav_dtn_iterative_solver_parameters`, and **the default is
        chosen by dimension** - direct in 2-D, iterative in 3-D - mirroring
        `GravitySolver` and `StokesSolverBase`, so the scalable path is what a
        caller who states nothing gets at production scale.

        "Direct" is honest rather than literal: with `Real` blocks in the space
        there *is* no monolithic direct solve to fall back on, so the LU lives
        on block 0 of the Schur split and the multiplier complement is still
        taken by GMRES.

        **This method used to hand every caller the direct preset**, including
        in 3-D and including when asked for `"iterative"` - which the string was
        accepted for and then ignored. The iterative dictionary existed and was
        exported, but nothing in this class could reach it, so every 3-D driver
        pasted its own copy; that is how four copies came to exist and how two
        of them drifted. The direct preset's own docstring says "2-D ONLY. Do
        not use this in 3-D" and "has no 3-D successor", which was true of the
        preset and false of the class.

        `condensed` is taken from the layout rather than from the caller, so the
        block-0 sweep cannot disagree with the space it acts on - the mismatch
        `_check_block0_split_matches_layout` exists to catch is unreachable on
        this path by construction.

        A `Mapping` is honoured verbatim, through the base class.
        """
        # **The appctx is built on EVERY path, including the `Mapping` one, and
        # that placement is the fix for a defect rather than tidiness.** It
        # used to be built only after the `Mapping` early return below, so a
        # caller who passed their own dictionary got an appctx without
        # `dtn_block1_diagonal` -- and every 3-D driver passes a dictionary
        # (`b1_elastic.condensed_solver_parameters`,
        # `b4_polar_motion.coupled_solver_parameters`, the B2 probe). So the
        # shipped `DtNMultiplierDiagPC` was **unreachable from every driver
        # that would use it**: naming it produced a bare
        # `PETSc.Error: error code 101`, with the real cause only in the
        # `_loud` line above it.
        #
        # Measured, job 175339746 (coarse, condensed, `--mult-pc gadopt_diag`):
        # `ValueError: DtNMultiplierDiagPC needs the block-1 diagonal in the
        # appctx`, once per rank, then 101. The one recorded number for that
        # preconditioner (111 -> 48) came from the probe's *own* copy in
        # `b2_pc.py`, which takes its diagonal from a module global -- so the
        # shipped class had never run in 3-D at all.
        #
        # Building it unconditionally costs nothing: `block1_diagonal` does no
        # assembly and no solves, and returns `None` when the space has no
        # `Real` fields. It changes no default -- both presets still run block
        # 1 at `pc_type: none`, so the entry is inert unless a caller names the
        # preconditioner.
        #
        # This is also the argument for the design rule the successor obeys: a
        # preconditioner that reads its data off the operator has no such
        # failure mode, and the dense-complement arm of the same job ran
        # unmodified.
        #
        # **The entry is ADDED to the base class's appctx, never used to build
        # one before calling it.** `StokesSolverBase.set_solver_options`
        # *assigns* `self.appctx = {"mu": ...}` as its first statement
        # (`stokes_integrators.py:375`), so anything set beforehand is
        # discarded. A first attempt at this fix built the dictionary above the
        # `Mapping` branch and was silently undone by exactly that line -- the
        # retry (job 175340812) failed identically, with the remote md5
        # confirming the fixed file was the one that ran. Set it after every
        # path that can reach the base class, and read it back rather than
        # rebuilding it, so the `"mu"` entry keeps whatever the base class
        # decided it should be.
        # The string/None path below never reaches a base class, so it may have
        # no `appctx` at all; the `Mapping` path always has one by the time
        # this runs. Handle both rather than assuming either.
        def _attach_block1_diagonal():
            if getattr(self, "appctx", None) is None:
                self.appctx = {
                    "mu": self.approximation.mu / self.rho_continuity}
            self.appctx["dtn_block1_diagonal"] = self.block1_diagonal()

        if isinstance(solver_preset, Mapping):
            super().set_solver_options(solver_preset, solver_extras)
            _attach_block1_diagonal()
            return
        if solver_preset not in (None, "direct", "iterative"):
            raise ValueError("Solver type must be 'direct' or 'iterative'.")

        # Deliberately NOT moved above the `Mapping` branch: its own docstring
        # names "pass an explicit solver_parameters dictionary" as the escape
        # hatch for a caller whose preconditioner does handle a state-dependent
        # Jacobian, so the dictionary path is exactly where the refusal must
        # not fire.
        self.refuse_stale_preconditioner()

        if solver_preset is None:
            solver_preset = (
                "direct" if self.mesh.topological_dimension == 2
                else "iterative")

        # Added first so the preset wins every key it names. That ordering is
        # load-bearing for one key in particular: `newton_stokes_solver_
        # parameters` sets an ABSOLUTE `snes_atol` of 1e-10, and a configuration
        # whose entire forcing is smaller than that converges at iteration zero
        # and returns exactly 0.0 on every step, reporting SNES converged. The
        # iterative preset sets 1e-15 and so overrides it; the direct one does
        # not name the key and so inherits the trap, which is recorded in
        # `demos/gravity/CLAUDE.md` and is not fixed here.
        self.add_to_solver_config(newton_stokes_solver_parameters)
        if solver_preset == "direct":
            self.add_to_solver_config(selfgrav_dtn_schur_solver_parameters)
        else:
            self.add_to_solver_config(selfgrav_dtn_iterative_solver_parameters(
                condensed=self.layout.condensed))
        if solver_extras:
            self.add_to_solver_config(solver_extras)
        _attach_block1_diagonal()
        self.register_update_callback(self.set_solver)

    def refuse_stale_preconditioner(self) -> None:
        """`DtNTwoBlockSchurPC.update` is a no-op, which needs a constant Jacobian.

        Correct for Newtonian GIA, and false the moment the rheology is
        non-Newtonian: with `exponent > 1` the viscosity depends on the state,
        the assembled block-0 operator inside the preconditioner goes *stale*
        rather than failing, and Newton then converges slowly or to nothing with
        no diagnostic. Refused rather than warned about, because a silently
        stale preconditioner is this project's recurring failure mode.
        """
        exponent = getattr(self.approximation, "exponent", 1)
        try:
            newtonian = float(exponent) == 1.0
        except (TypeError, ValueError):
            newtonian = False
        if not newtonian:
            raise ValueError(
                f"The approximation has exponent={exponent!r}, so the Jacobian "
                "depends on the state, but DtNTwoBlockSchurPC.update is a "
                "no-op and its assembled block would go stale silently between "
                "Newton steps. Pass an explicit solver_parameters dictionary "
                "if you have a preconditioner that handles this.")

    def check_boundary_quadrature(self, *args, **kwargs):
        """Measures whether the boundary rule resolves the DtN modes.

        Forwarded from the boundary form, and worth running on the *coupled*
        system rather than only on a standalone gravity solve: a DtN constraint
        row integrated at the wrong degree is a wrong constraint with no
        warning.
        """
        return self.form.check_boundary_quadrature(*args, **kwargs)

    def block1_diagonal(self):
        """The exact diagonal of the `Real` block, or `None` if there is none.

        `DtNMultiplierDiagPC` reads this out of the appctx. Supplied
        unconditionally, costs nothing to compute -- no assembly, no solves --
        and **changes no default**: block 1 stays at `pc_type: none` in both
        shipped presets, so this is inert unless a caller selects the
        preconditioner by name.

        Two contributions:

        * the **multipliers**, `theta_psi * (-scale_k * A_h)`, from
          `DtNGravityForm.multiplier_diagonal`, which derives the sign and
          explains why the *discrete* boundary measure is the right one;
        * the **rotation closure rows**, `theta_rot_i * K_i`, present only when
          rotation is on -- one row in 2-D (`m_3` alone) and three in 3-D. Not
          guessed: verified against the assembled diagonal to 1.5e-15 relative
          on two machines and two rank counts, `+5.9313525844e-04` for the
          polar-wander pair and `-1.7692384969e-01` for `m_3`, whose closure
          sign is negative.

        **`theta_psi` is read as the property, never recomputed.** At
        `B_mu = 0` -- a supported configuration, and the one the null-coupling
        gate uses -- `theta_psi` is floored through `_row_scale_B_mu`. A
        reimplementation here as `scaling_factor * B_mu / Lambda` would give
        zero, and this method would hand the preconditioner a diagonal of zeros
        to divide by. Same for `_theta_rot`, which carries the same floor.

        ## Order is not assumed, and the count is only the tripwire

        The entries are built **keyed by sub-field index** off
        `layout.multipliers` and `layout.rotation`, then placed by position
        within `layout.real_fields`. An interleaved or rotation-first block
        therefore cannot produce a wrongly-ordered diagonal; it produces a
        failed assertion. That matters because the count check alone would wave
        such a layout through, and *the missing block-0 guard cost this project
        a week*.

        The count check stays as the tripwire for a **foreign** `Real`
        sub-field -- one added for something that is neither a multiplier nor a
        rotation row, which no accounting here could describe. Today none
        exists: `self_gravitating_gia_space` does
        `spaces.extend([R] * (form.n_multipliers + n_rot))`, and the 2-D
        monopole datum deliberately builds its `Real` space **outside** the
        mixed space.

        Returns `None` when the block is empty, so building a solver never
        fails on account of a preconditioner nobody selected.

        Raises:
          RuntimeError: if the `Real` sub-fields are not exactly
            `layout.real_fields`, in that order, as a contiguous trailing run.
        """
        form = getattr(self.layout, "gravity_form", None)
        space = self.solution.function_space()
        real_in_space = tuple(
            i for i in range(len(space))
            if space.sub(i).ufl_element().family() == "Real")
        if not real_in_space:
            return None

        i_R = real_in_space[0]
        expected = self.layout.real_fields
        if expected != real_in_space or expected != tuple(
                range(i_R, len(space))):
            raise RuntimeError(
                f"block1_diagonal cannot describe this space. The Real "
                f"sub-fields are at {real_in_space}, the layout accounts for "
                f"{expected}, and a contiguous trailing run would be "
                f"{tuple(range(i_R, len(space)))}. The diagonal read requires "
                "all three to agree: it assumes the Real block is exactly the "
                "DtN multipliers followed by the rotation closure rows, in "
                "that order and last. A Real field added for anything else, or "
                "a reordering, invalidates it. Fix the accounting here before "
                "using DtNMultiplierDiagPC on this configuration.")

        by_index = {}
        if form is not None and getattr(form, "multiplier_keys", None):
            mult = float(self.theta_psi) * form.multiplier_diagonal()
            if len(mult) != len(self.layout.multipliers):
                raise RuntimeError(
                    f"the form describes {len(mult)} multiplier rows but the "
                    f"layout has {len(self.layout.multipliers)}.")
            by_index.update(zip(self.layout.multipliers, mult))
        for k, name in enumerate(self.layout.rotation_names):
            idx = self.layout.rotation.get(name)
            if idx is not None:
                by_index[idx] = (float(self._theta_rot(k))
                                 * float(self._closure_constant(k)))

        missing = [i for i in expected if i not in by_index]
        if missing:
            raise RuntimeError(
                f"no diagonal entry was derived for Real sub-fields {missing}; "
                "they are in the layout but neither a multiplier nor a "
                "rotation row described it.")
        return np.array([by_index[i] for i in expected])

    def project_out_nullspace(self) -> bool:
        """Removes any declared kernel from the solution. Returns whether it did.

        **Declaring a `nullspace` is not enough on this solver's default Krylov
        method, and that is a PETSc fact rather than a Firedrake one.** The
        outer method is FGMRES, which is right-preconditioned by construction.
        PETSc removes the declared kernel from the *right-hand side*, and for a
        left-preconditioned method it also removes it from the preconditioner's
        output on every application; for a right-preconditioned one it does
        not. Since `DtNTwoBlockSchurPC` is nearly an exact inverse here - the
        outer solve converges in one iteration - the answer is essentially the
        preconditioner's output, kernel component and all, and the declaration
        changes the result by not one bit. Measured, with
        `rigid_rotation_nullspace` declared and the mesh at `dr 0.2, nazim 32`:
        the rotation content of `u` stayed at exactly `1.000410e-13` where the
        same declaration on the LU reference took it from `-3.5e-13` to
        `2.6e-19`.

        So the projection is applied here, once, after each solve. It is
        legitimate precisely when the declaration is: a genuine kernel admits
        no solution at all unless the residual is orthogonal to it, so removing
        it selects the minimum-norm representative and discards nothing. The
        converse is the warning - a *wrongly* declared kernel now silently
        deletes part of the answer instead of merely being ignored, which is
        why `rigid_rotation_nullspace` documents when the rigid rotation is and
        is not one.

        Only blocks carrying an actual `VectorSpaceBasis` are touched; the
        entries that are plain sub-spaces mean "no kernel here" and are skipped.
        """
        if self.nullspace is None:
            return False
        projected = False
        for i, basis in enumerate(self.nullspace):
            if isinstance(basis, VectorSpaceBasis):
                basis.orthogonalize(self.solution.subfunctions[i])
                projected = True
        return projected

    def solve(self) -> None:
        """Refreshes the enclosed mass, solves, then projects out the kernel."""
        self.update_total_mass()
        super().solve()
        if self.project_out_nullspace():
            # `StokesSolverBase.solve` has already copied the solution into
            # `solution_old`; the projection happens afterwards, so the old
            # state has to follow it or the next step's internal-variable
            # source would carry the kernel component the current one dropped.
            self.solution_old.assign(self.solution)
        if self.layout.condensed:
            self.recover_internal_variables()

    def recover_internal_variables(self) -> None:
        """Rebuild the eliminated `m` from the solved displacement.

        Called **after** `project_out_nullspace`, so that `m` is built from the
        displacement actually returned; doing it before would leave the stored
        state carrying a kernel component the solution no longer has, and the
        disagreement would only surface as drift several steps later.

        The expression is rebuilt here from `self.solution.subfunctions[...]`
        rather than reused from `set_equations`, and that is not tidiness. The
        residual's `u` comes from `split()` of a mixed function spanning **two**
        meshes, so an expression built on it carries both domains and
        `interpolate` refuses it with

            NotImplementedError: Interpolating an expression with no arguments
            defined on multiple meshes is not implemented yet.

        The sub-function is a genuine `Function` on the mechanics mesh and
        carries one domain, so the same algebra interpolates.
        """
        u = self.solution.subfunctions[self.layout.displacement]
        strain = self.approximation.deviatoric_strain(u)
        for m, maxwell_time in zip(self.internal_variables,
                                   self.approximation.maxwell_times):
            ratio = self.dt / maxwell_time
            # into a temporary first: `m` appears on the right-hand side, and
            # interpolating a function into itself is not worth relying on.
            updated = Function(m.function_space()).interpolate(
                (m + ratio * strain) / (1 + ratio))
            m.assign(updated)
