r"""The monolithic self-gravitating GIA system: displacement, internal
variables, potential, DtN multipliers, core pressure, rotation, and the
centre-of-mass frame multipliers.

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

The same relation says that `C - A` and `k_s` are not independent inputs. The
solver's feedback is `Q k_T(t)` with `Q = a^5 Omega^2 / (3 G)`, an identity of
the degree-2 exterior field (MacCullagh), so the closure it actually solves,
`[(C - A) - Q k_T(t)] m = dI_direct`, is the secular relation above only when
`C - A = Q k_s`. A caller can state either constant;
`SelfGravitatingGIASolver._resolve_rotation_moments` computes the other and
refuses a pair that disagrees.

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
from mpi4py import MPI
from ufl import Form

from .approximations import BaseGIAApproximation
from .preconditioners import RigidBodyAssembledPC, near_nullspace_basis
from .dtn_form import DtNGravityForm
from pyadjoint.tape import annotate_tape, get_working_tape

from .dtn_coupled import (CoupledLowRankDtN, install_augmented_context,
                          taped_coupled_trace_coefficients)
from .dtn_lowrank import apply_dirichlet_to_rows
from .nullspaces import check_kernel_basis, project_out_kernel
from .equations import Equation
from .gravity_solver import real_scalar_solver_parameters
from .momentum_equation import (
    compressible_viscoelastic_terms,
    rotational_potential,
    rotational_potential_term,
    self_gravity_term,
)
from .internal_variable_equation import (
    assign_history_slices,
    history_slices,
    internal_variable_history_terms,
    internal_variable_space,
)
from .solver_options_manager import ConfigType
from .stokes_integrators import (
    CoupledInternalVariableSolver,
    _basis_provider,
    _displacement_basis,
    newton_stokes_solver_parameters,
)
from .utility import (CombinedSurfaceMeasure, ensure_constant,
                      restrict_real_coefficients_on_interior_facets,
                      scalar_value)
from .sea_level_sheet import ocean_fraction, softplus_column, water_fraction
from ufl import derivative as ufl_derivative
from ufl.algorithms.ad import expand_derivatives


__all__ = [
    "FluidCore",
    "RigidBodyAssembledPC",
    "GIASpaceLayout",
    "NULL_COUPLING_ROW_SCALE",
    "OMEGA_SQ_EARTH",
    "SelfGravitatingGIASolver",
    "rigid_rotation_nullspace",
    "self_gravitating_gia_space",
    "resolve_dtn_representation",
    "selfgrav_dtn_iterative_solver_parameters",
    "selfgrav_dtn_lowrank_direct_solver_parameters",
    "selfgrav_dtn_schur_lowrank_solver_parameters",
    "selfgrav_dtn_schur_solver_parameters",
]


from .gia_presets import (  # noqa: E402,F401  (re-exported for the old solver)
    DEFAULT_DISPLACEMENT_PC, NULL_COUPLING_ROW_SCALE, OMEGA_SQ_EARTH,
    _DENSE_MULTIPLIER_PC, _displacement_krylov, _potential_split, _prefixed,
    resolve_dtn_representation, selfgrav_dtn_iterative_solver_parameters,
    selfgrav_dtn_lowrank_direct_solver_parameters,
    selfgrav_dtn_schur_lowrank_solver_parameters,
    selfgrav_dtn_schur_solver_parameters)


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

        E = [ B_mu rho_core (u.n) psi
              + 0.5 B_mu rho_core g_0 (u.n)^2
              + beta_core p_core (u.n) ] ds

    Here `p_core` is one uniform pressure multiplier and `beta_core` is a
    nonzero row scale. Its variation enforces `int_CMB u.n ds = 0`, so the
    eliminated incompressible core cannot change volume or mass.

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
class SeaLevel:
    r"""The sea-level equation on the outer surface, with a gravitationally consistent ocean.

    The switch for the sea-level load, in the style of `FluidCore`: pass one
    to `SelfGravitatingGIASolver` together with a space built with
    `sea_level=True`. The surface load is then no longer a prescribed sheet.
    It is the sheet of the ocean and the grounded ice, and the ocean depends
    on the solution through the sea level

        SL = SL_init + (N - N_init) - (u_r - ur_init) + Shift

    with `N = (psi + psi_rot) / g_surface` the geoid height, `u_r = u . n` the
    radial displacement at the surface and `Shift` a `Real` unknown, the
    uniform sea-level change that conserves the mass of water and ice. With
    the centre-of-mass multipliers the conservation holds to order of those
    multipliers, which is discretisation error (see `sea_level_energy`).

    **The sheet.** With the ocean function `C` and the grounded-ice function
    `B` of Martinec et al. (2018) as sharp 0/1 fields and a non-negative ice
    thickness, the column of water and ice at one point is
    `rho_w B C SL + rho_i (1 - B) I = max(rho_w SL, rho_i I)`. The sheet is
    that maximum rounded by a softplus of a fixed width `eps =
    softplus_width` in units of ice thickness, minus the same column at the
    reference state:

        sigma = S(I, SL) - S(I_init, SL_init),
        S(I, SL) = rho_w SL + rho_i eps softplus((I_+ - r SL) / eps)

    with `r = rho_w / rho_i` and `I_+` the ice thickness clipped at zero
    (`gadopt.sea_level_sheet`). The sheet is exact wherever `|I_+ - r SL|` is
    more than a few `eps`, on any slope and any mesh, its derivatives with
    respect to `I` and `SL` lie in `[0, rho_i]` and `[0, rho_w]`, and every
    derivative is continuous. The coastline and the grounding line move with
    the sea level and the ice, and floating ice loads as the water it
    displaces. `SelfGravitatingGIASolver.sea_level_energy` documents the
    energy whose variation gives every row.

    **The fixed coastline.** With `fixed_ocean=True` the ocean function is
    held at its value in the reference state, Martinec et al. (2018), eq. 7
    and 8 (their sea-level equation of level 1, `fixed_ocean=1` in radopt):

        C0    = H(r SL_init / eps)            (`ocean_fraction`)
        sigma = rho_w C0 (SL - SL_init) + rho_i (1 - C0) (I - I_init)

    `C0` is the rounded ocean function of the reference state, with the
    rounding of the live sheet at a coast without ice. The factor `1 - C0`
    removes the ice over the reference ocean: that ice does not load the
    Earth and does not enter the mass balance. The grounded-ice function `B`
    is not used, because all ice on the reference land rests on the bed, and
    `C0` is zero there, so the water term vanishes on land. `C0` depends on
    the prescribed `SL_init` only, so `sigma` is affine in
    `Delta = SL - SL_init`. The sea-level rows are then linear in the
    unknowns, and the Jacobian does not change with the state. A Newtonian
    rheology converges in one Newton iteration when the linear solve reduces
    the residual below `snes_rtol`, and can use `snes_type ksponly`. A badly
    conditioned system, for example a stiff Earth with a degree-1 load, can
    need more iterations and then stops on the step-size test. Such a count is
    not a sign of nonlinearity.

    Attributes:
      boundary: the tag of the outer surface, as seen from the mechanics mesh
        (an exterior facet there, an interior facet of the parent).
      rho_w, rho_i: the water and ice densities, in the units of the
        reference density.
      g_surface: the surface gravity that divides the potential in the geoid
        and weights the load. A number: in the 2-D configuration the
        approximation's `g` is a mantle-mesh expression of radius, which is
        not the surface value where it is read.
      SL_init: the sea level of the reference state, a `Function` on the
        mechanics mesh.
      I, I_init: the current and the reference ice thickness.
      N_init, ur_init: the geoid height and radial displacement of the
        reference state. Zero for a run that starts undeformed.
      softplus_width: `eps`, the width of the rounding, in the units of the ice
        thickness and the sea level (a number or a `Constant`). The error of
        the sheet is at most `rho_i eps log(2)` at the coastline and the
        grounding line and falls as `exp(-|I_+ - r SL| / eps)` away from them.
        The second derivative of the sheet is nonzero in a band about
        `eps / (r s)` wide along a coast of slope `s`, so a Hessian, or a
        Taylor test of second order, needs `eps` large enough for the surface
        quadrature to sample that band (`gadopt.sea_level_sheet`).
      fixed_ocean: hold the ocean function at `C0 = H(r SL_init / eps)` and
        remove the ice over that ocean, as described above. `False`, the
        default, gives the live sheet of the current sea level (a moving
        coastline and floating ice).

    Raises:
      ValueError: for a `softplus_width` that is not positive.
    """

    boundary: int | str
    rho_w: Any
    rho_i: Any
    g_surface: Any
    SL_init: Any
    I: Any
    I_init: Any
    N_init: Any
    ur_init: Any
    softplus_width: Any
    fixed_ocean: bool = False

    def __post_init__(self):
        """Refuses a width that is not positive, at construction."""
        # A Constant or a number: read the value only to check the sign.
        if scalar_value(self.softplus_width) <= 0.0:
            raise ValueError(
                f"SeaLevel.softplus_width must be positive, got "
                f"{scalar_value(self.softplus_width)}.")


@dataclass(frozen=True)
class GIASpaceLayout:
    """Where each field of the coupled mixed space lives, by name.

    Returned by `self_gravitating_gia_space` alongside the space itself, so
    that nothing downstream counts sub-fields by hand: the multiplier count is
    a property of the boundary conditions, and the rotation count is a property
    of the dimension, so the indices of the later blocks are not knowable
    without both.

    The `Real` sub-fields - DtN multipliers, core pressure, rotation, then the
    centre-of-mass multipliers, then the sea-level `Shift` - are
    **contiguous and last**, which `DtNTwoBlockSchurPC.initialize` asserts and
    which is worth keeping: anything else would leave sub-fields out of both of
    its blocks, which is a silently wrong split rather than an error.

    Attributes:
      displacement: index of the CG vector displacement block.
      internal_variables: the index of the DG tensor internal-variable block,
        as a tuple of length one, or `()` on the condensed layout. One block
        holds every Maxwell element: the field has shape `(n, d, d)` and is
        built by `internal_variable_space`, so the number of elements never
        appears in a field index or a fieldsplit option. The tuple form is
        kept so that `n_fields`, `set_form` and the block-0 layout check read
        it the same way whether the block exists or not;
        `internal_variable_field` gives the bare index.
      potential: index of the potential block, which is on the *parent* mesh.
      multipliers: indices of the DtN multiplier `Real` blocks.
      core_pressure: index of the fluid-core pressure `Real` block, or `None`
        for a system without a fluid core.
      rotation: `{}` with rotation off; otherwise `{"m3": i}` in 2-D and
        `{"m1": i, "m2": i+1, "m3": i+2}` in 3-D. Keyed by name and never by
        position, because the 2-D field is `m_3` - index **2** of the rotation
        triple, not index 0 - and a positional record would silently reindex on
        promotion to 3-D.
      centre_of_mass: indices of the `Real` blocks that hold the Lagrange
        multipliers of the centre-of-mass frame, one per coordinate direction,
        or `()` when the frame is not constrained. See
        `SelfGravitatingGIASolver.centre_of_mass_energy`.
      sea_level: index of the `Real` block that holds `Shift`, the spatially
        uniform offset of the sea-level equation, or `None` when sea level is
        off. It is the last field of the space. See `SeaLevel`.
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
    core_pressure: int | None
    rotation: dict[str, int]
    gravity_form: DtNGravityForm
    mechanics_mesh: Any
    potential_mesh: Any
    self_gravity_number: Any = None
    #: Names of the rotation components, in the fixed 3-D order.
    rotation_names: tuple[str, ...] = field(default=("m1", "m2", "m3"))
    #: The combined DG tensor space, shape `(n, d, d)`, the internal
    #: variables live on. On the uncondensed layout this is
    #: `Z.sub(internal_variable_field)` and is redundant; on the condensed
    #: layout the variables are a stored `Function` outside the mixed space,
    #: and this is the only record of the space they are built on.
    internal_variable_space: Any = None
    #: Whether pointwise history substitution omits the variables from the space.
    condensed: bool = False
    #: Which DtN representation the space was SIZED for, `"multiplier"` or
    #: `"lowrank"`. Recorded because the space and the solver take it as
    #: independent arguments and a disagreement is otherwise invisible: a
    #: `"lowrank"` space handed to a `"multiplier"` solver has no `Real` fields
    #: for the constraint rows to be written into, and the reverse leaves
    #: `n_multipliers` unknowns in the space with nothing constraining them.
    #: `SelfGravitatingGIASolver` refuses the mismatch rather than running it.
    #: Always set by `self_gravitating_gia_space`; the field default is never
    #: what a built layout carries.
    dtn_representation: str = "multiplier"
    #: The centre-of-mass multiplier fields, one per coordinate direction, or
    #: `()`. A default so that a layout built without the frame constraint
    #: keeps its existing field count and order.
    centre_of_mass: tuple[int, ...] = ()
    #: The index of the `Real` field `Shift` of the sea-level equation, or
    #: `None`. It is the LAST field of the space, after the centre-of-mass
    #: multipliers, so every other field keeps its index when sea level is
    #: switched on.
    sea_level: int | None = None

    @property
    def cross_mesh(self) -> bool:
        return self.mechanics_mesh is not self.potential_mesh

    @property
    def internal_variable_field(self) -> int | None:
        """Index of the combined internal-variable block, `None` when condensed."""
        return self.internal_variables[0] if self.internal_variables else None

    @property
    def n_fields(self) -> int:
        return (2 + len(self.internal_variables) + len(self.multipliers)
                + (self.core_pressure is not None) + len(self.rotation)
                + len(self.centre_of_mass) + (self.sea_level is not None))

    @property
    def real_fields(self) -> tuple[int, ...]:
        """All `Real` field indices, in their mixed-space order.

        **The one place the `Real` block's composition is written down.**
        Anything that needs to describe that block per row -- a diagonal
        preconditioner, say -- must key off this rather than off position
        within the contiguous run, so that an interleaved or rotation-first
        layout becomes a *construction* error rather than something a count
        check would wave through. `DtNTwoBlockSchurPC` separately requires the
        run to be contiguous and last, and `block1_diagonal` asserts that this
        tuple is exactly that run.
        """
        core = (() if self.core_pressure is None else (self.core_pressure,))
        return tuple(self.multipliers) + core + tuple(
            self.rotation[name] for name in self.rotation_names
            if name in self.rotation) + tuple(self.centre_of_mass) + (
            () if self.sea_level is None else (self.sea_level,))

    def rotation_slots(self) -> tuple[int | None, int | None, int | None]:
        """`(m1, m2, m3)` indices, `None` where the component does not exist.

        The one accessor rotation code should use, so that every expression is
        written in three components unconditionally and the 3-D promotion adds
        two `Real` fields rather than revisiting every rotation expression.
        """
        return tuple(self.rotation.get(name) for name in self.rotation_names)


# `scalar_value` is `gadopt.utility.scalar_value`, imported above.


def self_gravitating_gia_space(
    mechanics_mesh,
    potential_mesh,
    /,
    *,
    gravity_bcs: dict[int | str, dict[str, Any]],
    n_internal_variables: int = 1,
    fluid_core: bool = False,
    rotation: bool = False,
    self_gravity_number: Number | Constant | None = None,
    displacement_degree: int = 2,
    internal_variable_degree: int = 1,
    internal_variable_family: str = "DG",
    potential_degree: int = 2,
    quad_degree: int | None = None,
    alpha: Number | Constant | None = None,
    condense_internal_variables: bool = False,
    dtn_representation: str | None = None,
    centre_of_mass: bool = False,
    sea_level: bool = False,
) -> tuple[MixedFunctionSpace, GIASpaceLayout]:
    r"""Builds the coupled mixed space and the layout that describes it.

        Z = [ V(sub), S_1..S_N(sub), Psi(parent),
              R x n_mult, R_core, R x n_rot, R x n_com, R_shift ]

    The field `R_shift` exists only with `sea_level=True`.

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
      n_internal_variables: number of Maxwell elements, matching the length of
        the approximation's `maxwell_times`. They share one DG tensor field of
        shape `(n, d, d)`, symmetric on the last two indices.
      fluid_core: add one uniform core-pressure field and its constraint row.
        The solver must receive a matching `FluidCore` object.
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
      internal_variable_family: "DG" (default) or "Quadrature". Quadrature
        histories are intended for direct constitutive elimination; their degree
        must match the solver's volume quadrature degree.
      quad_degree: boundary quadrature degree for the DtN treatment; `None`
        takes `DtNGravityForm`'s mesh-aware calibrated default.
      alpha: the Robin shift, `None` for the form's default of 1.
      dtn_representation: `"multiplier"`, `"lowrank"` or `None`. `None` is
        the default: low-rank on the full layout, multiplier when
        `condense_internal_variables` is set (`resolve_dtn_representation`).
        The layout records the value for the solver to follow.
      centre_of_mass: add one `Real` field per coordinate direction, the
        Lagrange multipliers that hold the centre of mass of the whole system
        (mantle, core and surface load) at the origin. It requires
        `fluid_core=True`; see `SelfGravitatingGIASolver.centre_of_mass_energy`
        for the reasons. It works on both DtN representations. With
        `sea_level=False` and no named representation it resolves to
        `"multiplier"`.
      sea_level: add the `Real` field `Shift` of the sea-level equation, last
        in the space. It requires `centre_of_mass=True`, because an ocean load
        always has degree-1 content. It works on both DtN representations.
        With no named representation it resolves to the layout default:
        `"lowrank"` on the full layout, `"multiplier"` on the condensed
        layout. Either layout accepts explicit low-rank selection. The solver
        must receive a matching `SeaLevel` object.

    Returns:
      `(Z, layout)`.
    """
    if centre_of_mass and not fluid_core:
        # With a rigid core the solver pins `u.n = 0` on the core boundary, a
        # circle or sphere about the origin. A rigid translation violates that
        # condition, so translations are not a kernel of the operator and the
        # frame is already fixed. A second constraint on the same freedom makes
        # the system over-determined: the multiplier then has to supply a real
        # force to reconcile the two.
        raise ValueError(
            "centre_of_mass=True requires fluid_core=True. A rigid core (un = 0 "
            "at the core boundary) already fixes the translation of the mantle, "
            "so a centre-of-mass constraint would over-determine the system.")
    if sea_level and not centre_of_mass:
        # The ocean load follows the geoid and the uplift, so it has degree-1
        # content whatever the ice load is. Without the frame constraint that
        # content drives the translation near-kernel of the coupled operator.
        raise ValueError(
            "sea_level=True requires centre_of_mass=True. The ocean load has "
            "degree-1 content, and only the centre-of-mass multipliers fix the "
            "translation that it drives.")
    if sea_level and dtn_representation is None:
        # Sea level with no named representation follows the layout default
        # of `resolve_dtn_representation`: low-rank on the full layout, the
        # layout that the 3-D sea-level runs use. The multiplier path costs a
        # dense complement of one block-0 solve per multiplier at every build,
        # which does not fit one node at a truncation of 20. The condensed
        # layout keeps its existing multiplier default; callers can now
        # explicitly select low-rank DtN after material elimination too.
        #
        # This rule comes before the centre-of-mass rule below on purpose.
        # Sea level requires the frame, so the frame rule would otherwise see
        # every sea-level caller first and name the multiplier path for them.
        dtn_representation = resolve_dtn_representation(
            None, condensed=condense_internal_variables)
    if centre_of_mass and dtn_representation is None:
        # The frame alone, with no named representation, stays on the
        # multiplier path. Both representations carry the frame rows: they are
        # UFL terms of `Real` fields, and the low-rank update acts on the
        # potential rows only. The decision that puts sea level on the
        # low-rank path by default covers sea level alone, not the frame alone
        # (`NOTES/DECISIONS.md`, 2026-09-17).
        dtn_representation = "multiplier"
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
    # One field for every Maxwell element, shape (n, d, d), symmetric on the
    # last two indices. Static condensation then always eliminates field 1,
    # whatever `n`, and the `Real` fields keep one fixed offset.
    if internal_variable_family not in ("DG", "Quadrature"):
        raise ValueError("internal_variable_family must be 'DG' or 'Quadrature'")
    S = internal_variable_space(
        mechanics_mesh, n_internal_variables, internal_variable_degree,
        family=internal_variable_family)
    R = FunctionSpace(potential_mesh, "R", 0)

    n_rot = 0
    if rotation:
        n_rot = 1 if potential_mesh.geometric_dimension == 2 else 3

    # Pointwise history layout: omit `M` from the mixed space and substitute the
    # same backward-Euler expression that `InternalVariableSolver` uses. This is
    # not exact elimination of the mixed weak DG formulation on curved cells.
    # It is important in 3-D because the history field is 85 percent of the
    # coarse benchmark system: 2 860 092 of 3 348 411 degrees of freedom. The
    # uncondensed layout carries `M` as field 1 and eliminates it inside the
    # preconditioner instead (`selfgrav_dtn_iterative_solver_parameters`).
    spaces = [V] + ([] if condense_internal_variables else [S]) + [Psi]
    i_potential = len(spaces) - 1
    i_R = len(spaces)

    # **The low-rank path carries no multiplier unknowns at all.** They are
    # eliminated by hand into `form.build_mode_rows()` and applied as a rank-n
    # update, so the space stops growing with the DtN truncation: `L = 20` costs
    # the same fields as `L = 2`. `form.n_multipliers` keeps the true count,
    # because the form still knows how many modes it treats.
    #
    # The core-pressure and rotation scalars are unaffected and stay. The
    # low-rank path therefore still has a `Real` block when either feature is
    # active. With neither feature there is no `Real` block.
    dtn_representation = resolve_dtn_representation(
        dtn_representation, condensed=condense_internal_variables)
    n_mult = 0 if dtn_representation == "lowrank" else form.n_multipliers
    n_core = int(fluid_core)
    # One centre-of-mass multiplier per coordinate direction: two in 2-D,
    # three in 3-D. They go last, after rotation, so every existing field keeps
    # its index when the frame constraint is switched on.
    n_com = potential_mesh.geometric_dimension if centre_of_mass else 0
    # The sea-level `Shift`, last of all, so that the frame multipliers keep
    # their indices when sea level is switched on.
    n_sl = int(sea_level)
    spaces.extend([R] * (n_mult + n_core + n_rot + n_com + n_sl))

    multipliers = tuple(range(i_R, i_R + n_mult))
    core_pressure = i_R + n_mult if fluid_core else None
    i_rot = i_R + n_mult + n_core
    i_com = i_rot + n_rot
    centre_of_mass_fields = tuple(range(i_com, i_com + n_com))
    # Named, and named `m3` in 2-D: the 2-D component is index *2* of the
    # rotation triple, and a layout that recorded it as "the first rotation
    # field" would silently change meaning on promotion to 3-D.
    names = ("m3",) if n_rot == 1 else ("m1", "m2", "m3")
    rotation_map = {name: i_rot + k for k, name in enumerate(names[:n_rot])}

    layout = GIASpaceLayout(
        displacement=0,
        internal_variables=(() if condense_internal_variables else (1,)),
        internal_variable_space=S,
        condensed=condense_internal_variables,
        potential=i_potential,
        multipliers=multipliers,
        core_pressure=core_pressure,
        rotation=rotation_map,
        centre_of_mass=centre_of_mass_fields,
        sea_level=(i_com + n_com) if sea_level else None,
        dtn_representation=dtn_representation,
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

    **Rheology: where a power law is supported.** With `exponent != 1` in the
    approximation the Maxwell times carry `power_law_factor`, the residual is
    nonlinear and both string presets run `newtonls`. The three configurations
    differ:

    - **Uncondensed layout, `dtn_representation="multiplier"`: supported.**
      The internal variables are field 1 of the mixed space, the history
      equation carries the factor, and the nested preconditioner follows the
      state-dependent Jacobian because the inner fieldsplit re-runs its setup
      after every Jacobian reassembly. Measured on the 2-D annulus at exponent
      3: the nested preset reproduces the direct route at every Newton step
      and `gadopt.InternalVariableSCPC` rebuilds once per Newton iteration
      (`NOTES/PLAN-POWER-LAW-SELFGRAVITY.md` section 2).
    - **Condensed layout (`condense_internal_variables=True`):** the legacy
      `constitutive_method="pointwise"` is linear only. The opt-in `"direct"`
      method solves the implicit nonlinear constitutive equation at quadrature
      points and directly assembles its consistent tangent and reverse adjoint.
      It requires natural displacement boundaries and supports both DtN representations.
    - **Uncondensed layout, `dtn_representation="lowrank"`: supported.** The
      update `B` does not depend on the rheology, the forward Newton solve
      reinstalls it after every Jacobian assembly through `augment_jacobian`,
      and the adjoint form `adjoint(dFdu)` carries the power-law tangent from
      UFL. Verified by
      `tests/unit/test_gia_gravity_adjoint_lowrank.py::test_taylor_with_a_power_law`
      at exponent 3 on both presets (Taylor rate 2, replay equal to a fresh
      solve), and by
      `tests/unit/test_gia_lowrank_block0.py::test_the_columns_survive_newton_on_a_power_law`,
      which pins that the potential-split preconditioner is built once per
      run under Newton while `gadopt.CondensedBlockPC` reassembles its blocks
      at every linear solve (`operator_version` is `None` for a power law).

    A **fluid core** and a power law work together on the supported
    configuration, in 2-D and in 3-D.
    `tests/unit/test_gia_nested_condensation.py::TestPowerLaw` solves the
    annulus with a `FluidCore` at the CMB on both routes and gets the same
    state, with the core-pressure row inside the dense multiplier complement
    that saddle needs. `TestThreeDimensions` does the 3-D counterpart on the
    24-cell extruded sphere, where `fluid_core_measure` reaches the CMB through
    `gadopt.utility.CombinedSurfaceMeasure` because that boundary is `ds_b` and
    carries no facet tag.

    Every power law takes GMRES on the condensed field, because
    `condensed_operator_symmetric` reads the rheology and nothing else. On
    these configurations the operator is nonsymmetric in fact as well as by
    rule. Measured asymmetry `|S - S^T| / |S|` of the condensed operator at the
    converged state: 3.6e-3 on the 2-D annulus with a rigid core, 1.0e-2 with a
    fluid core, 1.4e-2 on the 3-D sphere with a rigid core and 9.4e-2 with a
    fluid core, against 1.7e-16 for the same 3-D fluid-core configuration under
    a Newtonian rheology. The fluid core adds no asymmetry of its own -
    `fluid_core_energy` carries no stress, no `mu` and no Nitsche term, so it
    contributes nothing to the `(u, M)` block - and the Newtonian number is
    what says so; the 2-D trace of the deviatoric operator and the hexahedral
    cells of an extruded sphere are what do.

    Both are small cases that establish the construction works; nothing at
    production rank counts is measured (parent plan S4).

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
      dt: time step. A Real-space Function supports taped step changes.
      time_scheme: "backward_euler" (default) or prescribed variable-step "bdf2".
        BDF2 uses BE startup and stores two accepted material history levels.
      constitutive_method: "pointwise" retains the legacy linear substitution;
        "direct" uses nonlinear quadrature integration and a first-reverse
        timestep block on the condensed layout with either DtN representation. Weak
        displacement boundaries are not yet supported by the direct path.
      rotation_moments: `{"C": ..., "C_minus_A": ..., "k_s": ...}`, the
        reference hydrostatic figure. `C_minus_A` is required only in 3-D,
        where it is the dynamical ellipticity and *cannot* be computed from a
        spherically symmetric reference density at all (`C = A` identically),
        so it is an input. In 2-D, `C` is the disc's polar second moment
        `int rho_0 r^2 dV`, which the caller can assemble; it is still an input
        here so that the 3-D promotion supplies one more key rather than
        rewriting the row. `k_s` is the secular (fluid-limit) tidal Love number
        of the same figure, and it is the *consistent* way to state `C - A`:
        see `tidal_inertia_factor` and `_resolve_rotation_moments` for the
        identity `C - A = Q k_s` that ties the two together, and for what
        happens when both keys are given. No other key is accepted, because a
        misspelt `k_s` would be a silently unchecked `C - A`.
      Omega_sq: the squared rotation rate non-dimensionalised by `g_bar / L`;
        see `OMEGA_SQ_EARTH`.
      surface_radius: the non-dimensional radius `ahat = a / L` of the free
        surface. Required exactly when `rotation_moments["k_s"]` is given,
        because `Q = a^5 Omega^2 / (3 G)` needs it; ignored otherwise. It is a
        separate argument and not measured from the mesh, because the mesh the
        solver holds does not say which of its radii is the *figure* radius of
        MacCullagh's relation - the parent carries a stand-off buffer past the
        surface, and the mechanics submesh stops at the surface only by the
        convention of these GIA problems. What the value is checked against is
        the mechanics mesh's outer radius, which is that convention made
        explicit rather than assumed.
      fluid_core: a `FluidCore` (or its keyword mapping) to give the core its
        buoyancy and its mass sheet, or `None` for the legacy rigid core, which
        the caller then spells as `{Rc: {"un": 0.0}}` in `bcs`. The two are
        alternatives and the constructor refuses both on one boundary; keeping
        `un = 0` reachable is what makes the difference between the treatments a
        measured quantity rather than an inherited approximation.
      internal_variables: the stored history on the condensed layout, in any
        layout `gadopt.internal_variable_equation.history_slices` accepts: one
        combined `(n, d, d)` `Function` on `layout.internal_variable_space`
        (the default, built here when `None`), one `(d, d)` `Function` for a
        single Maxwell element, or a list of `(d, d)` `Function`s, one per
        element, which is the layout the B5 restart checkpoints hold. On the
        uncondensed layout the history is field 1 of the mixed space and this
        argument must be `None`.
      condensed_near_nullspace: what GAMG is seeded with on the condensed
        displacement operator of the uncondensed iterative preset: `"rigid"`,
        `"incompressible"` or `"none"`, the axis `gadopt.near_nullspace_basis`
        takes. The default `"incompressible"` adds the low-degree
        divergence-free fields to the six rigid-body modes, because the
        volumetric penalty of the internal-variable stress puts the slow modes
        of this operator in the divergence-free space and GAMG can only
        coarsen onto the modes it is given. It is the choice
        `gadopt.gia_gravity.DEFAULT_DISPLACEMENT_PC` makes on the condensed
        layout, and giving both layouts the same modes and the same
        displacement split is what keeps the ratio between them a measurement
        of the coupled
        residual: job `178765563`, uncondensed, 107 block-0 applications at 55
        inner iterations and 670 s for a 500 yr step, against job `178765560`,
        condensed, 111 at 46 and 271 s. A `near_nullspace` passed to the
        solver wins over this argument, and
        `b1_elastic.build_solver` declares one unless it is called with
        `near_nullspace=False`.
      Any remaining keyword goes to `CoupledInternalVariableSolver`, notably
      `bcs`, `scaling_factor`, `quad_degree`, `solver_parameters` and
      `solver_parameters_extra`.
      dtn_representation: `"multiplier"`, `"lowrank"` or `None`. `None`
        follows the layout, which is the library default (low-rank on the
        full layout, multiplier on the condensed one). An explicit value
        that disagrees with the layout is refused.

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
        surface_radius: Number | None = None,
        fluid_core: "FluidCore | Mapping | None" = None,
        internal_variables: "Function | list | None" = None,
        condensed_near_nullspace: str = "incompressible",
        dtn_representation: str | None = None,
        sea_level: "SeaLevel | None" = None,
        time_scheme: str = "backward_euler",
        constitutive_method: str = "pointwise",
        **kwargs,
    ) -> None:
        # `None` follows the space: the layout records what it was built for,
        # and a solver that names nothing cannot disagree with it. An explicit
        # value is checked against the layout below, as before.
        if dtn_representation is None:
            dtn_representation = layout.dtn_representation
        dtn_representation = resolve_dtn_representation(
            dtn_representation, condensed=layout.condensed)
        self.dtn_representation = dtn_representation
        # The space and the solver take the representation as INDEPENDENT
        # arguments, exactly as `condense_internal_variables` and the preset's
        # `condensed` are, and a disagreement between those two cost a run
        # before `_check_block0_split_matches_layout` existed. Refuse it here
        # for the same reason: a `"lowrank"` space in a `"multiplier"` solver
        # has no `Real` fields for the constraint rows, and the reverse leaves
        # `n_multipliers` unknowns that nothing constrains - a singular system
        # rather than a wrong one, but neither says which argument was wrong.
        if layout.dtn_representation != dtn_representation:
            raise ValueError(
                f"The space was built for dtn_representation="
                f"{layout.dtn_representation!r} but the solver was given "
                f"{dtn_representation!r}. They are independent arguments and "
                f"must agree. Change ONE of:\n"
                f"  - the space, via self_gravitating_gia_space("
                f"dtn_representation={dtn_representation!r}), or\n"
                f"  - the solver, via SelfGravitatingGIASolver("
                f"dtn_representation={layout.dtn_representation!r}).")
        if time_scheme not in ("backward_euler", "bdf2"):
            raise ValueError("time_scheme must be 'backward_euler' or 'bdf2'")
        if time_scheme == "bdf2" and kwargs.get("constant_jacobian", False):
            raise ValueError("BDF2 changes the startup Jacobian; constant_jacobian must be False")
        if constitutive_method not in ("pointwise", "direct"):
            raise ValueError("constitutive_method must be 'pointwise' or 'direct'")
        if constitutive_method == "direct":
            if not layout.condensed:
                raise ValueError("Direct constitutive integration requires the condensed layout")
            if kwargs.get("constant_jacobian", False) or kwargs.get("J") is not None:
                raise ValueError("Direct constitutive integration supplies its own changing Jacobian")
            if any(bc.keys() & {"u", "un"} for bc in (kwargs.get("bcs") or {}).values()):
                raise NotImplementedError(
                    "Quadrature integration does not yet implement weak displacement boundary traces; "
                    "use the coupled DG formulation, or natural boundaries with a fluid core")
            if kwargs.get("additional_forcing_term") is not None:
                raise NotImplementedError("Direct integration currently requires forcing in the GIA residual")
        from .backstress import BackstressApproximation
        self._backstress_model = isinstance(approximation, BackstressApproximation)
        if self._backstress_model:
            if layout.internal_variable_space.ufl_element().value_shape()[0] != 2:
                raise ValueError("Backstress requires n_internal_variables=2")
            if layout.condensed and constitutive_method != "direct":
                raise ValueError("Eliminated backstress requires constitutive_method='direct'")
            if any(bc.keys() & {"u", "un"} for bc in (kwargs.get("bcs") or {}).values()):
                raise NotImplementedError("Backstress currently requires natural displacement boundaries")
        self.constitutive_method = constitutive_method
        self._direct_backend = None
        self._quadrature_update = None
        self.time_scheme = time_scheme
        self.time_integrator = None
        self.layout = layout
        self._check_fluid_core_matches_layout(fluid_core)
        self._check_sea_level_matches_layout(sea_level)
        #: The `SeaLevel` settings, or `None`. Not called `sea_level`, because
        #: that name is the method that returns the sea-level expression.
        self.sea_level_parameters = sea_level
        self._mixed_space = solution.function_space()
        self._check_block0_split_matches_layout(kwargs.get("solver_parameters"))
        self._check_representation_matches_parameters(
            kwargs.get("solver_parameters"))
        # In the pointwise history layout the internal variables are not unknowns:
        # they are stored `Function`s carried between steps, exactly as the
        # segregated `InternalVariableSolver` carries them.
        if layout.condensed:
            if internal_variables is None:
                internal_variables = Function(
                    layout.internal_variable_space, name="internal_variables")
            n_stored = len(history_slices(internal_variables))
            n_expected = getattr(approximation, "n_history_tensors", len(approximation.maxwell_times))
            if n_stored != n_expected:
                raise ValueError(
                    f"{n_stored} stored history tensor(s), expected {n_expected} "
                    f"for {type(approximation).__name__}.")
        elif internal_variables is not None:
            raise ValueError(
                "`internal_variables` is only meaningful when the space was "
                "built with `condense_internal_variables=True`; without it the "
                "internal variables are sub-fields of the mixed space.")
        self.internal_variables = internal_variables
        if condensed_near_nullspace not in ("rigid", "incompressible", "none"):
            raise ValueError(
                "condensed_near_nullspace must be 'rigid', 'incompressible' "
                f"or 'none', got {condensed_near_nullspace!r}.")
        self.condensed_near_nullspace = condensed_near_nullspace
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
        self.surface_radius = surface_radius
        self.rotation_moments = dict(rotation_moments or {})

        if approximation.self_gravity_number is None:
            raise ValueError(
                "The approximation carries no `self_gravity_number`, so it "
                "describes a system with no gravitational potential; "
                "SelfGravitatingGIASolver needs Lambda = 4 pi G rho_bar L / "
                "g_bar.")
        self.Lambda = approximation.self_gravity_number
        self._check_self_gravity_number()
        # After `Lambda`, `Omega_sq` and `surface_radius`, because the rotation
        # constant `Q` is built from all three, and before any form is built,
        # because `_closure_constant` reads the resolved dictionary.
        self._resolve_rotation_moments()

        self.set_measures()
        # The sea-level refusals read only the settings and the boundary form,
        # so they run before the geometry measurements.
        self.check_sea_level(kwargs.get("bcs") or {})
        # Before the base class, which builds a residual and a solver: a
        # constructor that is going to refuse a mesh should refuse it before
        # paying for either.
        self.check_geometry()
        self.check_fluid_core(kwargs.get("bcs") or {})
        # Once, here, so that a basis that is not orthonormal is refused in
        # forward-only runs too. `project_out_nullspace` and PETSc both assume
        # an orthonormal set and are silently wrong without one.
        check_kernel_basis(kwargs.get("nullspace"))
        check_kernel_basis(kwargs.get("transpose_nullspace"))
        self.form.warn_on_quadrature_rule_limits()
        self.set_monopole_datum()
        super().__init__(solution, approximation, dt=dt, **kwargs)
        #: The low-rank DtN operator, or `None` on the multiplier path.
        self.dtn_operator = None
        #: The adopted low-rank solve block (route 1.5b), set on every annotated
        #: solve and `None` before the first one; only on the low-rank path.
        self.adjoint_block = None
        #: Whether the adjoint and tangent carry the `(d theta_psi/dm) B0 psi`
        #: term. Read at EACH use so a caller can flip it between gradient
        #: evaluations on one solver. With three overrides only, the adjoint and
        #: tangent agree with each other and are 93.65% wrong; this term is the
        #: difference (REVIEW-ADJOINT L11b).
        self._include_theta_derivative = True
        if self.dtn_representation == "lowrank":
            self.build_dtn_operator()

    def _check_fluid_core_matches_layout(self, fluid_core) -> None:
        """Refuses a fluid-core field without its equation, or the reverse."""
        space_has_core = self.layout.core_pressure is not None
        solver_has_core = fluid_core is not None
        if space_has_core == solver_has_core:
            return
        raise ValueError(
            "The space was built with fluid_core="
            f"{space_has_core}, but the solver received fluid_core="
            f"{'a FluidCore object' if solver_has_core else 'None'}. "
            "These settings must agree. Pass fluid_core=True to "
            "self_gravitating_gia_space() exactly when the solver receives a "
            "FluidCore object. A missing Real field omits the core-volume "
            "constraint. An unused Real field makes the Jacobian singular.")

    def _check_representation_matches_parameters(self, solver_parameters) -> None:
        """Refuses a potential-split preconditioner the representation cannot serve.

        `selfgrav_dtn_iterative_solver_parameters(dtn_representation=...)` and
        the solver's own `dtn_representation` are independent arguments, and a
        disagreement is silent in both directions.

        A `"lowrank"` preset on a multiplier solver selects
        `gadopt.LowRankPotentialPC`, whose preconditioning matrix would then be
        the plain assembled potential block: that raises inside
        `PCSetUp`, which is late and reported as an unhandled Python exception
        under a PETSc error code, so it is caught here instead.

        A `"multiplier"` preset on a low-rank solver fails late as well:
        `gadopt.CondensedBlockPC.initialize` installs the Python potential
        block `A_psipsi + B` whenever `dtn_operator` is in the context,
        whatever the preset names on the split, and plain GAMG then refuses
        that block inside `PCSetUp_GAMG` (`No method getinfo for Mat of type
        python`). Caught here with a message that names the fix.

        The combination this check does NOT see is a block-0 route without a
        potential split at all, `block0="pair"` or a hand-written dictionary,
        on a low-rank solver: there the update reaches block 0's operator
        through no split, nothing raises, and the outer FGMRES pays 8 or 9
        iterations where it should pay 3 (measured, annulus, truncation 3).
        `_refuse_block0_without_potential_split` refuses the named route.
        """
        if not isinstance(solver_parameters, Mapping):
            return
        keys = ("dtn_fieldsplit_0_condensed_fieldsplit_1_pc_python_type",
                # With no `Real` field the two-block Schur split has nothing to
                # split and `gadopt.CondensedBlockPC` is the outer
                # preconditioner, so its potential split loses the
                # `dtn_fieldsplit_0_` prefix.
                "condensed_fieldsplit_1_pc_python_type")
        preset_is_lowrank = any(
            solver_parameters.get(key) == "gadopt.LowRankPotentialPC"
            for key in keys)
        solver_is_lowrank = self.dtn_representation == "lowrank"
        if preset_is_lowrank == solver_is_lowrank:
            return
        if not preset_is_lowrank:
            # Only a dictionary that HAS a potential split can be wrong about
            # it. A direct preset solves block 0 with LU and has no split at
            # all, and a hand-written dictionary may do anything; neither is a
            # mistake and neither is refused here. The claim that there is a
            # split to be wrong about is the block-0 class, the same claim
            # `_check_block0_split_matches_layout` reads.
            block0_class = solver_parameters.get(
                "dtn_fieldsplit_0_pc_python_type",
                solver_parameters.get("pc_python_type"))
            if block0_class not in ("gadopt.CondensedBlockPC", "gadopt.ReducedGIABlockPC"):
                self._refuse_block0_without_potential_split(solver_parameters)
                return
        if preset_is_lowrank:
            raise ValueError(
                "The solver parameters put gadopt.LowRankPotentialPC on the "
                "potential split of block 0, which preconditions the low-rank "
                "DtN update, but the solver was built with "
                f"dtn_representation={self.dtn_representation!r} and carries "
                "no such update. Pass dtn_representation='lowrank' to "
                "SelfGravitatingGIASolver, or drop it from "
                "selfgrav_dtn_iterative_solver_parameters.")
        raise ValueError(
            "The solver was built with dtn_representation='lowrank', but the "
            "solver parameters leave the potential split of block 0 on plain "
            "GAMG. gadopt.CondensedBlockPC hands that split the Python block "
            "A_psipsi + B, which GAMG refuses inside PCSetUp with 'No method "
            "getinfo for Mat of type python'. Pass "
            "dtn_representation='lowrank' to "
            "selfgrav_dtn_iterative_solver_parameters as well.")

    def _refuse_block0_without_potential_split(self, solver_parameters) -> None:
        """Refuses the nested block-0 route on a low-rank solver.

        The nested route (`block0="pair"`) is a multiplicative fieldsplit on
        block 0 with `gadopt.InternalVariableSCPC` on its `(u, M)` half and a
        plain assembled potential block on the other. The low-rank update
        reaches block 0's operator through the outer matrix, so nothing raises
        and every solve converges; the outer FGMRES pays 8 or 9 iterations
        where the potential split of `gadopt.CondensedBlockPC` brings it to 3.
        That is the cost that decides a Gadi campaign, so the route is refused
        on the low-rank representation. A direct preset (no block-0 Python
        class) and a hand-written dictionary are not refused: neither names a
        block-0 route this class can reason about.

        The route is recognised by the key the preset writes for it,
        `dtn_fieldsplit_0_fieldsplit_0_pc_python_type`, and its no-Real-field
        spelling without the `dtn_fieldsplit_0_` prefix.

        Args:
          solver_parameters: the dictionary under test.

        Raises:
          ValueError: block 0 is the nested route on a low-rank solver.
        """
        keys = ("dtn_fieldsplit_0_fieldsplit_0_pc_python_type",
                "fieldsplit_0_pc_python_type")
        if not any(solver_parameters.get(key) == "gadopt.InternalVariableSCPC"
                   for key in keys):
            return
        raise ValueError(
            "The solver was built with dtn_representation='lowrank', but the "
            "solver parameters put gadopt.InternalVariableSCPC (block0='pair') "
            "on block 0, which has no potential split for the low-rank DtN "
            "update to be preconditioned on. Nothing would raise: every solve "
            "would converge and the outer FGMRES would cost about three times "
            "the iterations. Pass block0='condensed' and "
            "dtn_representation='lowrank' to "
            "selfgrav_dtn_iterative_solver_parameters.")

    def _check_sea_level_matches_layout(self, sea_level) -> None:
        """Refuses a `Shift` field without its equation, or the reverse."""
        space_has_shift = self.layout.sea_level is not None
        solver_has_sea_level = sea_level is not None
        if space_has_shift == solver_has_sea_level:
            return
        raise ValueError(
            "The space was built with sea_level="
            f"{space_has_shift}, but the solver received sea_level="
            f"{'a SeaLevel object' if solver_has_sea_level else 'None'}. "
            "These settings must agree. Pass sea_level=True to "
            "self_gravitating_gia_space() exactly when the solver receives a "
            "SeaLevel object. Without the Shift field there is no row for the "
            "mass conservation of water and ice. An unused Shift field makes "
            "the Jacobian singular.")

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

        The field indices named across the `dtn_fieldsplit_0_pc_fieldsplit_N_fields`
        entries (a split may name a comma-separated pair, as the condensed
        `(u, M)` split does) are the preset's own statement of how many fields
        it thinks block 0 has, so their count is compared against how many the
        space actually has. The method reads a second claim as well:
        `dtn_fieldsplit_0_pc_python_type == "gadopt.CondensedBlockPC"` says
        block 0 holds three fields, because that class eliminates an
        internal-variable field, so it is refused on a condensed layout whose
        block 0 holds only `(u, psi)`. Anything that makes neither claim - a
        hand-written dictionary, a string preset - is left alone.
        """
        if not isinstance(solver_parameters, Mapping):
            return
        # The default uncondensed route names no `_fields` key at all: block 0
        # is one python preconditioner that splits the fields itself. So the
        # field count below cannot see it, and the class name is the claim
        # instead. On the condensed layout that class looks for an
        # internal-variable field the space does not hold.
        block0_class = solver_parameters.get("dtn_fieldsplit_0_pc_python_type")
        if (block0_class == "gadopt.CondensedBlockPC"
                and self.layout.condensed):
            raise ValueError(
                "Block 0 runs gadopt.CondensedBlockPC, which eliminates the "
                "internal-variable field of a three-field block 0, but the "
                "space was built with condense_internal_variables=True and "
                "its block 0 holds two fields (displacement and potential). "
                "Change ONE of:\n"
                "  - the space, via self_gravitating_gia_space("
                "condense_internal_variables=False), or\n"
                "  - the parameters, via "
                "selfgrav_dtn_iterative_solver_parameters(condensed=True).\n"
                "They are independent arguments and must agree.")
        prefix = "dtn_fieldsplit_0_pc_fieldsplit_"
        named = [
            index.strip()
            for k, v in solver_parameters.items()
            if k.startswith(prefix) and k.endswith("_fields")
            for index in str(v).split(",")]
        if not named:
            return
        n_split = len(named)
        n_space = 2 + len(self.layout.internal_variables)  # u, psi, and any M
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
        if not np.isclose(scalar_value(declared), scalar_value(self.Lambda)):
            raise ValueError(
                f"self_gravity_number disagrees between the space factory "
                f"({scalar_value(declared)!r}) and the approximation "
                f"({scalar_value(self.Lambda)!r}). The first scales the mass sheets "
                "and the second the volume source; they must be the same "
                "number.")

    # -- The rotational closure's two constants -----------------------------

    #: The keys `rotation_moments` accepts. Anything else is refused rather
    #: than ignored: the whole point of `k_s` is that it is *checked* against
    #: `C_minus_A`, so a misspelt key would restore in silence exactly the
    #: unchecked `C - A` this machinery exists to remove.
    ROTATION_MOMENT_KEYS = ("C", "C_minus_A", "k_s")

    #: Relative tolerance of `C_minus_A` against `Q k_s`.
    #:
    #: The number is set by how far an error in `C - A` travels, not by
    #: floating point. Eliminating the displacement from the closure gives the
    #: secular Liouville relation `m = (1 + k_L) dI_direct / [(C - A) - Q k_T(t)]`,
    #: so a fractional error `eps` in `C - A` changes `m` by
    #: `eps (C - A) / [(C - A) - Q k_T(t)]`. Near the fluid limit that
    #: amplification is large: on the Spada M3-L70-V01 benchmark
    #: `C - A = 0.2421`, `Q = 0.2505` and `k_T(20 kyr) = 0.957`, so the
    #: denominator is 0.0024 and the amplification is about 100.
    #:
    #: `1e-4` therefore bounds the induced error in the polar motion at about
    #: 1 % at 20 kyr, while still admitting the four-to-five-digit rounding a
    #: caller does when it writes a constant down - the benchmark driver's
    #: `2.6952e35` against the exact `Q k_s = 2.6952068e35` is 2.5e-6, forty
    #: times inside this tolerance. It refuses the benchmark's own
    #: inconsistent pair (2.4 %) by a factor of 240. A caller that needs
    #: better than 1 % at the fluid limit must give `k_s` alone and let the
    #: solver compute `C - A`, which is exact by construction.
    ROTATION_CONSISTENCY_RTOL = 1e-4

    #: Relative tolerance of `surface_radius` against the mechanics mesh's own
    #: outer radius. Loose, because it is a check on the *meaning* of the
    #: argument and not on the mesh: it must catch a radius given in metres, or
    #: the CMB radius given by mistake, and must not grade a coarse annulus
    #: whose straight facets already cost 4e-4 (`check_geometry`). Two percent
    #: is ten times the largest faceting error measured there, and a `Q` built
    #: from a radius 2 % wrong is 10 % wrong, since `Q ~ ahat^5`.
    #:
    #: So this check does not bound the accuracy of `Q`, and the "exact by
    #: construction" of `ROTATION_CONSISTENCY_RTOL` holds only as far as the
    #: caller's own `surface_radius` is exact. A radius 1 % wrong passes here
    #: and leaves `C - A` 5 % wrong.
    SURFACE_RADIUS_RTOL = 2e-2

    @property
    def tidal_inertia_factor(self) -> float:
        r"""`Q`: the inertia perturbation per unit rotation-vector change and Love number.

        MacCullagh's relation for the degree-2 response. A body whose inertia
        tensor is perturbed by `dI_13` has exterior potential
        `3 G dI_13 x z / r^5`; setting that equal to `k_T` times the applied
        centrifugal perturbation `Omega^2 m_1 x z` at the surface `r = a` gives

            dI_13 = a^5 Omega^2 k_T m_1 / (3 G) = Q k_T m_1,  Q = a^5 Omega^2 / (3 G)

        so `Q` is an identity of the degree-2 exterior field, not a model
        parameter, and the solver's rotational feedback is `Q k_T(t)` whatever
        `rotation_moments` says. In the solver's non-dimensional variables,
        with `a = ahat L`, `Omega^2 = Omega_sq g_bar / L` and
        `G = Lambda g_bar / (4 pi rho_bar L)`, the moment scale `rho_bar L^5`
        divides out and

            Qhat = Q / (rho_bar L^5) = 4 pi ahat^5 Omega_sq / (3 Lambda)

        which is what this returns. On the Spada benchmark's constants it is
        0.25047547, and `Qhat k_s = 0.2421406` against the driver's
        `C - A = 0.2421400`.

        Raises:
          ValueError: if no `surface_radius` was given, since `ahat` is the one
            factor the solver cannot get from `Lambda` and `Omega_sq`.
        """
        if self.surface_radius is None:
            raise ValueError(
                "`tidal_inertia_factor` needs `surface_radius`, the "
                "non-dimensional radius of the free surface: Q = a^5 Omega^2 "
                "/ (3 G) carries the fifth power of it. Pass "
                "`surface_radius=a / L` to SelfGravitatingGIASolver.")
        ahat = scalar_value(self.surface_radius)
        return (4 * np.pi * ahat ** 5 * scalar_value(self.Omega_sq)
                / (3 * scalar_value(self.Lambda)))

    def _measured_surface_radius(self) -> float:
        """The mechanics mesh's largest vertex radius, over all ranks.

        Used only to check `surface_radius`. The maximum over the *vertices* is
        the nominal radius exactly on any mesh whose outer vertices sit on the
        sphere, whether or not the facets between them are straight, so it is a
        sharper probe of the intended radius than an area would be. The halo
        vertices a rank owns are included and would only repeat a value another
        rank already has, so the reduction is a plain maximum.
        """
        coords = self.mesh.coordinates.dat.data_ro
        local = float(np.sqrt((coords ** 2).sum(axis=1)).max()) if coords.size \
            else 0.0
        return self.mesh.comm.allreduce(local, MPI.MAX)

    def _resolve_rotation_moments(self) -> None:
        r"""Ties `C - A` to the solver's own rotational feedback. Called from `__init__`.

        The rotation row closes as `K_i m_i = s_i dI_i3`, and `dI_i3` contains
        the solver's *own* degree-2 response to the centrifugal perturbation,
        which by `tidal_inertia_factor` is `Q k_T(t) m_i`. So the polar-wander
        closure the solver actually solves is

            [(C - A) - Q k_T(t)] m_i = dI_direct,i

        against the classical secular form `(C - A)(1 - k_T/k_s) m_i =
        dI_direct,i`. The two are the same equation **only** when
        `C - A = Q k_s`, because `Q` is an identity that the caller cannot
        change and `k_s` is the fluid limit of the same `k_T`. A `C - A` that
        does not satisfy it describes a body whose hydrostatic flattening and
        whose fluid-limit tidal response disagree, and the error it makes is
        not the fractional error in `C - A`: it is amplified by
        `(C - A) / [(C - A) - Q k_T(t)]`, which grows without bound as the
        model relaxes towards its fluid limit.

        The Spada et al. (2011) benchmark contains exactly that inconsistency -
        `C - A = 2.63e35 kg m^2` in its load excitation against
        `Q k_s = 2.6952e35 kg m^2` implied by the `k_s = 0.96672389` of its
        transfer function, a 2.4 % difference that shows as 3.6 % in `|m|` at
        `t = 0` and more later (`NOTES/SPADA-BENCHMARK-2026-09-17.md`). It
        cannot be expressed through this solver once `k_s` is named.

        Three ways to spell the rotation moments, and what each does:

        - `C_minus_A` alone: taken as given, unchecked. This is what the
          solver has always done and what the benchmark driver still does. The
          solver has no second opinion to check it against, since `k_s` appears
          nowhere else in the system.
        - `k_s` alone, with `surface_radius`: `C - A = Q k_s` is computed here.
          Consistent by construction, and the route to prefer.
        - Both, with `surface_radius`: refused unless they agree to
          `ROTATION_CONSISTENCY_RTOL`. The message carries both values, their
          relative difference and the `k_s` that the given `C - A` implies,
          because which of the two the caller meant is not knowable here.

        `C` (the polar moment, which closes the `m_3` row) is untouched: it is
        a moment of the reference density that the caller can assemble, and no
        Love number constrains it.
        """
        unknown = set(self.rotation_moments) - set(self.ROTATION_MOMENT_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown rotation_moments key(s) {sorted(unknown)!r}. The "
                f"accepted keys are {list(self.ROTATION_MOMENT_KEYS)!r}.")

        # Kept in the dictionary, not consumed: `k_s` is part of the record of
        # what the caller asked for, and a second call of this method must see
        # the same inputs and reach the same answer.
        k_s = self.rotation_moments.get("k_s")
        if self.surface_radius is not None:
            # A `Q` built from the wrong radius would make the check below
            # meaningless rather than absent, so the radius is checked first.
            ahat = scalar_value(self.surface_radius)
            measured = self._measured_surface_radius()
            if abs(ahat - measured) > self.SURFACE_RADIUS_RTOL * measured:
                raise ValueError(
                    f"surface_radius={ahat:.6g} against the mechanics mesh's "
                    f"outer radius {measured:.6g}. Q = a^5 Omega^2 / (3 G) "
                    f"carries the fifth power of this radius, so the "
                    f"difference is a factor {(ahat / measured) ** 5:.4g} in "
                    "the rotational feedback. Give the *non-dimensional* "
                    "radius of the free surface, a / L.")
        elif k_s is not None:
            raise ValueError(
                "rotation_moments['k_s'] needs `surface_radius`, the "
                "non-dimensional radius a / L of the free surface: the "
                "identity that ties k_s to C - A is C - A = Q k_s with "
                "Q = a^5 Omega^2 / (3 G), and the radius is the one factor "
                "that is neither `Lambda` nor `Omega_sq`.")

        if k_s is None:
            return

        Q = self.tidal_inertia_factor
        consistent = Q * scalar_value(ensure_constant(k_s))
        given = self.rotation_moments.get("C_minus_A")
        if given is None:
            self.rotation_moments["C_minus_A"] = consistent
            return

        given = scalar_value(ensure_constant(given))
        relative = abs(given - consistent) / abs(consistent)
        if relative > self.ROTATION_CONSISTENCY_RTOL:
            raise ValueError(
                f"rotation_moments gives C_minus_A={given!r} and "
                f"k_s={scalar_value(ensure_constant(k_s))!r}, which disagree "
                f"by {relative:.3%}: the k_s implies "
                f"C - A = Q k_s = {consistent!r} with "
                f"Q = 4 pi ahat^5 Omega_sq / (3 Lambda) = {Q!r}, and the given "
                f"C - A implies k_s = {given / Q!r}. The solver's rotational "
                "feedback is Q k_T(t) by MacCullagh's relation and is not a "
                "free parameter, so the two spellings are the same closure "
                "only when C - A = Q k_s. Give one of them, not both, or "
                "reconcile them. (The tolerance is "
                f"{type(self).__name__}.ROTATION_CONSISTENCY_RTOL = "
                f"{self.ROTATION_CONSISTENCY_RTOL:g}, set by the "
                "amplification of this error near the fluid limit.)")

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

        dss = self._plain_fluid_core_measure(tag)
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

    def _plain_fluid_core_measure(self, tag) -> Measure:
        """The CMB facet measure for a *geometric* integral, un-intersected.

        `check_fluid_core` measures the facet's own area and mean radius, which
        are properties of the mechanics mesh alone. It therefore wants the
        mantle's plain `ds(tag)` and never the cross-mesh pairing
        `fluid_core_measure` builds: intersecting with the parent's facet
        measure answers a different question, and the check has to be able to
        report an empty measure rather than fail inside an intersection.

        The extruded branch is the one thing the two share, and it must be
        shared, because that is the whole content of this helper: an extruded
        sphere's CMB is `ds_b` and a plain `ds("bottom")` refuses the name. So
        the tag is resolved through `gadopt.utility.CombinedSurfaceMeasure`
        here as well, and a `"bottom"` that the mesh does not carry still
        measures zero and reaches the empty-measure message above.

        `CombinedSurfaceMeasure` requires a degree, where the plain `ds` branch
        takes UFL's estimate. The form's calibrated boundary degree is the one
        used, so that this integral is taken at the degree every other CMB
        integral uses. The choice is immaterial to the check itself, which
        compares an area against `4 pi r^2` at a 1 percent tolerance.

        Args:
          tag: the fluid core's boundary, an integer tag or `"bottom"`/`"top"`.

        Returns:
          A `Measure` already called on `tag`.
        """
        if self.mesh.extruded:
            return CombinedSurfaceMeasure(
                domain=self.mesh, degree=self.form.quad_degree)(tag)
        return Measure("ds", domain=self.mesh)(tag)

    def _on_parent_facet(self, value):
        """`value`, a scalar of the potential (parent) mesh, on a facet-to-facet measure.

        The fluid-core and sea-level measures integrate over facets that are
        exterior facets of the mechanics mesh and interior facets of the
        parent (`fluid_core_measure`, `sea_level_measure`). UFL requires every
        `Argument` of the parent to carry a restriction there, including the
        test and trial functions of a `Real` field, whose single basis function
        is the global constant 1 and so has the same value on both sides. A
        `Real` coefficient may stay unrestricted, but its derivative is an
        `Argument`, so the Jacobian and every adjoint derivative would then
        need one. `avg` is the restriction: it is exact for any field that is
        continuous across the facet, it does not depend on which side UFL
        calls `'+'`, and it is the spelling the solver already uses for the
        parent's CG potential on these measures. For a `Real` field it adds
        half of the facet integral to the single global entry from each side,
        so the assembled value is the facet integral once.

        Only the `Real` fields of the mixed space, their test functions and
        constants go through here. A field of the mechanics mesh must not:
        that mesh's facets are exterior facets, and UFL refuses a restriction
        on them (`Inconsistent restrictions`).

        Args:
          value: A `Real` field, its test function, or a constant, all on the
            parent mesh.

        Returns:
          `avg(value)` when the two meshes differ, `value` otherwise.
        """
        return avg(value) if self.layout.cross_mesh else value

    def _restrict_parent_real_coefficients(self, form: Form) -> Form:
        """Restrict the parent's `Real` coefficients on its interior facets.

        `_on_parent_facet` restricts the `Real` fields of the mixed space where
        a term is written. A `Real` coefficient that comes from outside the
        solver, such as `B_mu`, `Lambda` or a sheet scale given as a `Function`
        on the parent's `Real` space, can appear in the same integrals through
        the approximation and the gravity form, and pyadjoint differentiates
        with respect to it when it is a control. This applies
        `gadopt.utility.restrict_real_coefficients_on_interior_facets` with
        the parent, which explains the rule.

        Args:
          form: A form of this solver.

        Returns:
          The form with the parent's `Real` coefficients restricted on its
          interior facets, or `form` itself when the meshes are the same.
        """
        if not self.layout.cross_mesh:
            return form
        return restrict_real_coefficients_on_interior_facets(
            form, self.potential_mesh)

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

        ## Extruded meshes, where the CMB is the bottom surface

        A radially extruded sphere - the 3-D production geometry - reaches its
        CMB through `ds_b`, not through a tagged side facet, so a plain
        `ds(tag)` refuses `"bottom"` with `Invalid subdomain_id bottom` and the
        fluid core is unreachable on it. `gadopt.utility.CombinedSurfaceMeasure`
        is the helper the rest of the solver already uses for this
        (`Equation.__init__` builds one on `trial_space.extruded`): it maps
        `"bottom"` to `ds_b`, `"top"` to `ds_t` and every integer tag to
        `ds_v`, so every caller keeps writing `fluid_core_measure()(tag)` and
        the tag itself decides. Returned here for an extruded mesh, so that the
        energy, the sheet, the volume constraint and the diagnostics all follow
        one measure on both geometries.

        An extruded mesh is never cross-mesh: the two-mesh coupling comes from
        `Submesh`, which the extruded path does not use, and
        `CombinedSurfaceMeasure` takes no `intersect_measures` and so cannot
        express one. The assertion below states that rather than leaving the
        combination to produce a measure that silently drops the intersection.
        """
        if self.mesh.extruded:
            if self.layout.cross_mesh:
                # Raised and not asserted: an `assert` disappears under
                # `python -O`, and this one guards a silently wrong
                # integration rather than an internal invariant.
                # `Equation.__init__` raises the same way for the same gap.
                raise NotImplementedError(
                    "An extruded mechanics mesh with a separate potential "
                    "mesh is not supported by the fluid core. "
                    "CombinedSurfaceMeasure takes no `intersect_measures`, so "
                    "there is no way to pair the parent's facet measure with "
                    "it and the CMB sheet loses the facet-to-facet "
                    "intersection it needs - which is a 21 percent error with "
                    "no warning, as this method's docstring records. Extruded "
                    "self-gravity runs pass one mesh for both roles; widen "
                    "CombinedSurfaceMeasure if that ever stops being true.")
            # The same calibrated boundary degree as the non-extruded branch
            # below, for the same reason.
            return CombinedSurfaceMeasure(domain=self.mesh,
                                          degree=self.form.quad_degree)
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
        The caller restricts that test function (`_on_parent_facet`), because
        the facets are interior facets of the parent.
        """
        sigma = self.fluid_core_sheet()
        if sigma is None:
            return Form([])
        dss = self.fluid_core_measure()(self.fluid_core.boundary)
        return integrand * sigma * dss

    def fluid_core_energy(self) -> Form:
        r"""The CMB energy whose variation is the whole fluid-core condition.

            c int_Rc [ B_mu rho_core (u.n) psi
                       + 0.5 B_mu rho_core g_0 (u.n)^2
                       + beta_core p_core (u.n) ] ds

        **One energy and not three additions**, which is what makes the three
        blocks it produces - `(u, psi)`, `(psi, u)` and `(u, u)` - symmetric by
        construction rather than by anybody's algebra. The sign convention is
        then fixed by the energy too: `F = c dE/dz` is the convention the rest
        of this residual already obeys, as `self_gravity_term`'s
        `-B_mu int rho_0 grad(psi).w` is the `u`-variation of
        `E_div = -B_mu int rho_0 u.grad(psi) dx` and the potential row's
        `-Lambda int rho_0 u.grad(v)`, scaled by `theta_psi = c B_mu / Lambda`,
        is that same energy's `psi`-variation.

        `p_core` is a `Real` field. Its variation gives the zero-volume row,
        and the displacement variation gives the uniform pressure traction.
        The two blocks are transposes because both come from this energy.

        `FluidCore` documents the physics, the density contrast and the sign
        trap in `dot(u, n)`. Three implementation notes belong here instead:

        - **The `c` is `scaling_factor`**, which
          `CoupledInternalVariableSolver` multiplies the whole momentum residual
          by and which `theta_psi` carries as well. Omitting it here would make
          the coupled Jacobian asymmetric by exactly `c` on any run that used
          one, and that reads as a sign error in new code.
        - **At `B_mu = 0` the gravity and spring terms disappear, but the volume
          constraint remains.** Core incompressibility does not depend on
          gravity. `beta_core = _row_scale_B_mu` equals `B_mu` for every nonzero
          value and uses its nonzero floor at zero. Any nonzero common factor
          gives the same constrained displacement and only rescales `p_core`.
          Using zero would leave an empty `Real` row and column in the Jacobian.
        - The constraint uses `fluid_core_measure`, exactly as the other CMB
          terms do. A second measure could use different geometry or quadrature
          and break the transpose relation on a cross-mesh system.
        - `p_core` follows the algebraic row scale and the CMB normal. To report
          pressure from an assembled traction coefficient, divide that
          coefficient by `scaling_factor * B_mu`. This conversion applies only
          when `B_mu` is nonzero. The mantle normal points inward at the CMB.
        """
        fc = self.fluid_core
        u = self.solution_split[self.layout.displacement]
        psi = self.solution_split[self.layout.potential]
        # The core pressure is a `Real` field of the parent, restricted on the
        # facet-to-facet measure (`_on_parent_facet`).
        p_core = self._on_parent_facet(
            self.solution_split[self.layout.core_pressure])
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
        # The incompressible core's missing uniform pressure. `_row_scale_B_mu`
        # is exactly `B_mu` for every nonzero value. Its floor at zero keeps the
        # physical volume constraint without leaving a structurally empty Real
        # row and column. The common scale changes only the multiplier value.
        E += self._row_scale_B_mu * p_core * un * dss
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
        psi_rot = self.rotational_potential_expression(self.mesh, facet=True)
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

    # -- The sea-level equation -----------------------------------------------

    def check_sea_level(self, bcs: Mapping) -> None:
        """Refuses a surface load given twice and an under-resolved surface rule.

        With sea level on, the ice load and the ocean load are one sheet,
        `surface_load_sheet`, that enters the mechanics, the potential and every
        mass moment through `sea_level_energy`. Three refusals:

        1. A `normal_stress` on the sea-level boundary in the mechanics `bcs`.
           It would add a second surface load to the momentum row that the
           potential row and the mass conservation row do not see.
        2. A `sigma` or `interior_sigma` sheet on the same boundary in the
           gravity boundary conditions, for the same reason on the other row.
        3. A surface quadrature degree below `2 p`, with `p` the highest degree
           of the displacement and the potential. The sheet contains
           polynomials of degree `p` and products of them, and the quadrature
           has to integrate them.

        Raises:
          ValueError: for any of the three.
          NotImplementedError: on an extruded mechanics mesh, which the
            surface measure does not support yet.
        """
        sl = self.sea_level_parameters
        if sl is None:
            return
        tag = sl.boundary
        if tag in bcs and "normal_stress" in bcs[tag]:
            raise ValueError(
                f"Boundary {tag} carries both the sea-level load and a "
                "`normal_stress`. With sea level on, the ice and ocean load is "
                "the sea-level sheet, which already enters the momentum row. "
                "Remove the normal_stress, and give the ice thickness to "
                "SeaLevel.I instead.")
        if any(bc_id == tag for bc_id, _, _ in self.form.sigma_bcs):
            raise ValueError(
                f"Boundary {tag} carries both the sea-level load and a gravity "
                "sheet (`sigma` or `interior_sigma`). With sea level on, the "
                "ice and ocean load is the sea-level sheet, which already enters "
                "the potential row. Remove the sheet from gravity_bcs.")
        if self.mesh.extruded:
            raise NotImplementedError(
                "The sea-level equation is implemented on a non-extruded "
                "mechanics mesh only (the 2-D annulus).")
        p = self._sea_level_polynomial_degree()
        q = self.form.quad_degree
        if q < 2 * p:
            raise ValueError(
                f"The sea-level measure uses the boundary quadrature degree "
                f"{q}, below 2 p = {2 * p} for polynomial degree p = {p} of the "
                "displacement and the potential. The sheet is not integrated "
                "accurately at that degree. Pass a larger quad_degree to "
                "self_gravitating_gia_space, or None for the calibrated default.")

    def _sea_level_polynomial_degree(self) -> int:
        """`p`, the highest polynomial degree of the displacement and the potential.

        The sea level contains `u . n` and `psi`, so its degree on a facet is
        the larger of the two. The quadrature check reads it.
        """
        # The space of the solution Function, read directly, because the check
        # runs before the base constructor sets `solution_space`.
        Z = self._mixed_space
        return max(Z[self.layout.displacement].ufl_element().embedded_superdegree,
                   Z[self.layout.potential].ufl_element().embedded_superdegree)

    def sea_level_measure(self) -> Measure:
        """The surface measure of the sea-level sheet, not yet restricted to a tag.

        The mantle's own `ds`, intersected with the parent's interior facet
        measure `dS`, at the boundary form's calibrated quadrature degree. This
        is the construction of `fluid_core_measure` for the outer surface, and
        the reasons are the same: the displacement exists only on the mantle
        side, `FacetNormal` of the mechanics mesh is the unambiguous outward
        normal, and the facet-to-facet intersection evaluates the parent's
        `psi` at the right points. Intersecting with the parent's cell measure
        instead assembles without an error and is 21 percent wrong for a
        parent field (`fluid_core_measure` docstring).

        Call the result on `SeaLevel.boundary`.
        """
        kwargs = {"domain": self.mesh, "degree": self.form.quad_degree}
        if self.layout.cross_mesh:
            kwargs["intersect_measures"] = (
                Measure("dS", domain=self.potential_mesh),)
        return ds(**kwargs)

    def _sea_level_ds(self) -> Measure:
        """`sea_level_measure()` restricted to the sea-level boundary."""
        return self.sea_level_measure()(self.sea_level_parameters.boundary)

    def _sea_level_expression(self, split) -> Any:
        """`SL` built from the sub-functions `split` of a solution.

        `sea_level` passes the live solution. `sea_level_energy` passes a
        frozen copy, at which it linearises the sheet.

        Args:
          split: the `split()` of a `Function` on the mixed space.

        Returns:
          `SL_init + (N - N_init) - (u . n - ur_init) + Shift` as UFL on the
          sea-level measure.
        """
        sl = self.sea_level_parameters
        u = split[self.layout.displacement]
        psi = split[self.layout.potential]
        # `Shift` and the rotation scalars are `Real` fields of the parent,
        # restricted on this facet-to-facet measure (`_on_parent_facet`).
        shift = self._on_parent_facet(split[self.layout.sea_level])
        g_surface = ensure_constant(sl.g_surface)
        # `avg(psi)`: the surface is an interior facet of the parent, so the
        # parent's field is formally two-valued there. `psi` is CG, so the
        # average of its trace is exact and does not depend on the side order.
        psi_face = avg(psi) if self.layout.cross_mesh else psi
        # Bruns: N = +(psi + psi_rot) / g. The sign is derived in
        # `BaseGIAApproximation.geoid`: a mass excess raises the geoid.
        N = psi_face / g_surface
        if self.layout.rotation:
            # The polynomial on the mechanics mesh, which the measure
            # integrates over. It is the same polynomial as on the parent.
            values = [Constant(0.0) if i is None
                      else self._on_parent_facet(split[i])
                      for i in self.layout.rotation_slots()]
            n_rot = 1 if self.mesh.geometric_dimension == 2 else 3
            N = N + rotational_potential(
                values[-n_rot:], self.mesh, Omega_sq=self.Omega_sq) / g_surface
        # The mechanics mesh's outward normal is the radial direction at the
        # outer surface, so `u . n` is the uplift.
        u_r = dot(u, FacetNormal(self.mesh))
        return (sl.SL_init + (N - sl.N_init) - (u_r - sl.ur_init) + shift)

    def sea_level(self):
        """The sea level `SL = SL_init + (N - N_init) - (u_r - ur_init) + Shift`.

        UFL on the sea-level measure. `N = (psi + psi_rot) / g_surface` with
        rotation, `psi / g_surface` without. `g_surface` is a number from
        `SeaLevel` and never the approximation's `g`, which in the 2-D
        configuration is an expression of radius on the mechanics mesh.
        """
        return self._sea_level_expression(self.solution_split)

    def _sheet_of(self, SL):
        r"""The load sheet for the sea level `SL`.

        The live sheet (`SeaLevel.fixed_ocean` false):

            sigma = S(I, SL) - S(I_init, SL_init)

        with `S = gadopt.sea_level_sheet.softplus_column`, the rounded maximum
        of the water column and the ice column (`SeaLevel` docstring). The
        reference column is subtracted so that a state without change carries
        no load. With a fixed coastline (`SeaLevel.fixed_ocean` true):

            sigma = rho_w C0 (SL - SL_init) + rho_i (1 - C0) (I - I_init)

        with `C0 = H(r SL_init / eps)` (`_fixed_ocean_sheet`).

        Args:
          SL: the sea level of the current column.

        Returns:
          The UFL expression of `sigma`.
        """
        sl = self.sea_level_parameters
        if sl.fixed_ocean:
            return self._fixed_ocean_sheet(SL)
        eps = ensure_constant(sl.softplus_width)
        now = softplus_column(sl.I, SL, eps, sl.rho_w, sl.rho_i)
        reference = softplus_column(sl.I_init, sl.SL_init, eps, sl.rho_w,
                                    sl.rho_i)
        return now - reference

    def sea_level_water_fraction(self):
        """The fraction of the surface that takes up water when the sea rises.

        `d sigma / d SL` divided by `rho_w` at the current state: 1 over open
        water and floating ice, 0 on land and under grounded ice. Its integral
        on the sea-level measure is the ocean area of the barystatic sea-level
        change, the area that `- (rho_i / rho_w) dV_ice / A_ocean` divides by.
        For the live sheet it is `1 - H(f / eps)` with `f = I_+ - r SL`
        (`gadopt.sea_level_sheet.water_fraction`); with a fixed coastline it
        is `C0`.

        Returns:
          A UFL expression on the sea-level measure, with values in `[0, 1]`.
        """
        sl = self.sea_level_parameters
        if sl.fixed_ocean:
            return self._fixed_ocean_mask()
        return water_fraction(sl.I, self.sea_level(),
                              ensure_constant(sl.softplus_width),
                              sl.rho_w, sl.rho_i)

    def _fixed_ocean_mask(self):
        """`C0 = H(r SL_init / eps)`, the ocean function of the reference state.

        `gadopt.sea_level_sheet.ocean_fraction` at the reference sea level,
        with the width of the live sheet: the rounding of the live sheet at a
        coast without ice. A switch between a fixed and a live coastline then
        changes only whether the coastline follows the solution.

        Returns:
          The UFL expression of `C0` on the sea-level measure.
        """
        sl = self.sea_level_parameters
        return ocean_fraction(sl.SL_init, ensure_constant(sl.softplus_width),
                              sl.rho_w, sl.rho_i)

    def _fixed_ocean_sheet(self, SL):
        r"""The sheet of a fixed coastline, `rho_w C0 Delta + rho_i (1 - C0) dI`.

            I_eff      = (1 - C0) I,   I_eff_init = (1 - C0) I_init
            sigma      = rho_w C0 (SL - SL_init) + rho_i (I_eff - I_eff_init)

        Martinec et al. (2018), eq. 7 and 8. The water term is the change of
        the water column over the fixed ocean. The ice term is the change of
        the ice on the reference land only: the factor `1 - C0` removes the
        ice over the ocean, which floats or is replaced by water in this
        approximation. `sigma` is affine in `SL`, so its derivative with
        respect to the solution is the constant `rho_w C0` times the variation
        of `SL`, and the residual is linear.

        Args:
          SL: the sea level in the water term.

        Returns:
          The UFL expression of `sigma`.
        """
        sl = self.sea_level_parameters
        C0 = self._fixed_ocean_mask()
        # The water column added over the reference ocean.
        water = sl.rho_w * C0 * (SL - sl.SL_init)
        # The ice change on the reference land, `rho_i (I_eff - I_eff_init)`
        # with `I_eff = (1 - C0) I` as in the docstring.
        ice = sl.rho_i * ((1 - C0) * sl.I - (1 - C0) * sl.I_init)
        return water + ice

    def surface_load_sheet(self):
        """The surface mass density `sigma` of ocean and ice at the current state.

        Positive where mass has been added, in the convention of every
        `interior_sigma`. UFL on the sea-level measure. The coastline is live,
        or fixed at the reference state when `SeaLevel.fixed_ocean` is true
        (`_sheet_of`). Every consumer of the sheet (the sea-level rows, the
        centre-of-mass moment, the rotation row, the scale of the net mass
        check) reads it here, so the setting reaches all of them.
        """
        return self._sheet_of(self.sea_level())

    def surface_load_sheet_integral(self, integrand) -> Form:
        """`int_Re integrand * sigma dS` on the sea-level measure.

        An empty form without sea level, so callers need no special case. Write
        a position polynomial on the mechanics mesh's coordinates: the only
        cross-mesh objects in the form are then `psi` inside `sigma` and any
        `Real` test function. The caller restricts that test function
        (`_on_parent_facet`), because the facets are interior facets of the
        parent.
        """
        if self.sea_level_parameters is None:
            return Form([])
        return integrand * self.surface_load_sheet() * self._sea_level_ds()

    def _sheet_test_variation(self):
        r"""`(d sigma / d SL) delta Delta`, in the directions of the test functions.

            delta Delta = avg(v) / g_s - (w . n) + s + Omega_sq p_i nu_i / g_s

        with `w`, `v`, `s` and `nu_i` the test functions of the displacement,
        the potential, `Shift` and the rotation scalars, and the live sheet in
        `d sigma / d SL = rho_w (1 - H(f / eps))`. With `SeaLevel.fixed_ocean` the sheet is affine, and
        the same derivative gives `d sigma / d SL = rho_w C0`.

        Built as the Gateaux derivative of `surface_load_sheet` with respect to
        the solution, in the direction of the mixed test function, and
        expanded at once by `expand_derivatives`. Two reasons for this
        spelling:

        - The expression is the same one that the Jacobian assembly derives for
          the transpose block, so the two blocks agree to round-off without a
          hand-derived derivative of the sheet.
        - No unevaluated derivative node stays in the form, so `split_form`
          removes the parent-mesh terms from the blocks that do not contain
          these test functions (`NOTES/DESIGN-SEA-LEVEL.md`).
        """
        direction = TestFunction(self._mixed_space)
        return expand_derivatives(ufl_derivative(
            self.surface_load_sheet(), self.solution, direction))

    def _sea_level_scale(self):
        """`c B_mu g_s`, the factor of the sea-level energy."""
        return (self.scaling_factor * self.approximation.B_mu
                * ensure_constant(self.sea_level_parameters.g_surface))

    def sea_level_energy(self) -> Form:
        r"""The sea-level energy, with the sheet linearised at the current state.

            E_sl = -c B_mu g_s int_Re G(Delta) dS,   dG/dDelta = sigma

        with `Delta = SL - SL_init = (N - N_init) - (u_r - ur_init) + Shift`.
        Its variations are the three sea-level rows:

        - `u`, direction `w`: `+c B_mu g_s int sigma (w . n) dS`, the weight of
          the load, the same as a `normal_stress` of `B_mu g_s sigma`;
        - `psi`, direction `v`: `-c B_mu int sigma v dS`, the same as an
          `interior_sigma` of density `sigma` in the potential row, which
          carries `-theta_psi 4 pi G sigma v = -c B_mu sigma v`;
        - `Shift`, direction `s`: `-c B_mu g_s int sigma dS`, the conservation
          of the mass of water and ice. The centre-of-mass energy adds
          `c B_mu sum_i lambda_i int x_i (dsigma/dShift) s dS` to the same row,
          because the sheet moves the centre of mass. Keeping that column keeps
          the residual the gradient of one Lagrangian and the Jacobian
          symmetric. At convergence the net sheet mass is therefore
          `sum_i lambda_i int x_i dsigma/dShift dS / g_s`: of order the
          multipliers, which are discretisation error, and not zero to the
          solver tolerance (measured 7.4e-9 on the coarse annulus, about 7e-8
          of the ice mass).

        The sign of `E_sl` is fixed by the first two. `g_s` equals the
        reference gravity at the surface, so the potential variation cancels
        it and the displacement variation keeps it.

        **The live sheet.** The exact `G` of the softplus sheet needs the
        dilogarithm, which UFL does not have. The energy linearises the sheet
        at a copy of the solution made at the call (`SL_f`) instead:

            sigma_lin(SL) = sigma(SL_f) + rho_w W_f (SL - SL_f),
            G(Delta)      = sigma_lin(SL_init) Delta + 0.5 rho_w W_f Delta^2

        with `W_f = 1 - H(f_f / eps)` the water fraction at the frozen state.
        `dG/dDelta = sigma_lin(SL)`, which equals `sigma(SL)` at the frozen
        state. So `derivative(E_sl, solution)` equals the written residual at
        the state where the energy is built, for any state. The residual itself
        is the variation of `int G(Delta)` for the exact `G`, because `sigma`
        depends pointwise on the one scalar `Delta`; it is written directly in
        `sea_level_residual`, and this energy is the check of it.

        **Fixed coastline.** With `SeaLevel.fixed_ocean` the sheet is
        `sigma = rho_i (I_eff - I_eff_init) + rho_w C0 Delta` with
        `I_eff = (1 - C0) I` and `C0 = H(r SL_init / eps)`, so

            G(Delta) = rho_i (I_eff - I_eff_init) Delta + 0.5 rho_w C0 Delta^2

        for any state. `C0` depends on the prescribed `SL_init` only, so no
        frozen copy of the solution is made, and `derivative(E_sl, solution)`
        equals the written residual exactly.

        **Rotation.** With rotation on, `Delta` contains `psi_rot / g_s`, and
        the `m_i` variation of this energy duplicates the sheet term that
        `inertia_form` already puts into the rotation row. This energy is a
        diagnostic; the residual does not include that variation.
        """
        sl = self.sea_level_parameters
        if sl is None:
            return Form([])
        if sl.fixed_ocean:
            C0 = self._fixed_ocean_mask()
            delta = self.sea_level() - sl.SL_init
            # The ice term of the sheet, `rho_i (I_eff - I_eff_init)`, which
            # does not depend on the solution, and the coefficient of the
            # quadratic water term over the fixed ocean.
            ice = sl.rho_i * ((1 - C0) * sl.I - (1 - C0) * sl.I_init)
            G = ice * delta + 0.5 * sl.rho_w * C0 * delta * delta
            return -self._sea_level_scale() * G * self._sea_level_ds()
        frozen = self.solution.copy(deepcopy=True)
        SL_f = self._sea_level_expression(split(frozen))
        eps = ensure_constant(sl.softplus_width)
        # The water fraction `d sigma / d SL / rho_w` at the frozen state.
        W_f = water_fraction(sl.I, SL_f, eps, sl.rho_w, sl.rho_i)
        # The tangent of the sheet at the frozen state, evaluated at `SL_init`
        # (that is, at `Delta = 0`).
        sigma_0 = self._sheet_of(SL_f) + sl.rho_w * W_f * (sl.SL_init - SL_f)
        delta = self.sea_level() - sl.SL_init
        G = sigma_0 * delta + 0.5 * sl.rho_w * W_f * delta * delta
        return -self._sea_level_scale() * G * self._sea_level_ds()

    def sea_level_residual(self) -> Form:
        r"""The sea-level rows, written term by term.

            -c B_mu g_s int sigma(Delta) delta Delta dS
            delta Delta = avg(v) / g_s - (w . n) + s

        The `u`, `psi` and `Shift` rows of the variation of `sea_level_energy`,
        with `sigma` the live sheet, or the sheet of the fixed ocean function
        `C0` when `SeaLevel.fixed_ocean` is true. The Jacobian of these rows is
        symmetric with the live sheet too, because `sigma` depends on the
        unknowns only through `Delta`. With the fixed ocean the rows are
        linear in the unknowns.

        **Why it is written out and not `derivative(E, solution)`**: the energy
        contains `avg(psi)` on the parent's facets. `derivative` leaves an
        unevaluated node whose zero variations still carry the parent mesh
        into every Jacobian block after `split_form`, and
        `InternalVariableSCPC` then fails in Slate
        (`centre_of_mass_residual` docstring). Here each term carries its test
        function explicitly.

        **No rotation row.** With rotation on, the sheet already enters the
        rotation row through `inertia_form`, which is the transpose partner of
        the `psi_rot / g_s` part of `Delta` in these rows. Adding the
        `m_i`-variation of the energy here as well would count that coupling
        twice, the trap that `fluid_core_rotational_traction` records.

        An empty form without sea level.
        """
        if self.sea_level_parameters is None:
            return Form([])
        sl = self.sea_level_parameters
        g_surface = ensure_constant(sl.g_surface)
        sigma = self.surface_load_sheet()
        dss = self._sea_level_ds()
        scale = self._sea_level_scale()
        w = self.tests[self.layout.displacement]
        v = self.tests[self.layout.potential]
        # The `Real` test function of `Shift`, restricted on this measure.
        s = self._on_parent_facet(self.tests[self.layout.sea_level])
        v_face = avg(v) if self.layout.cross_mesh else v
        # The displacement row: the weight of the load on the surface.
        F = scale * sigma * dot(w, FacetNormal(self.mesh)) * dss
        # The potential row: the mass sheet, with the sign of every other sheet.
        F -= scale * sigma * (v_face / g_surface) * dss
        # The Shift row: the net mass of water and ice is zero, up to the
        # centre-of-mass column that `centre_of_mass_residual` adds to it.
        F -= scale * sigma * s * dss
        return F

    # -- The centre-of-mass frame ---------------------------------------------

    def mass_dipole_form(self, i: int, u=None, weight=None, *,
                         scale=None, include_load: bool = True) -> Form:
        r"""`D_i`, the `i`th component of the first mass moment of the perturbation.

            D_i = int_mantle rho_0 u_i dx
                  + int_Rc x_i sigma_core ds
                  + sum_sheets int x_i sigma dS
                  + int_Re x_i sigma_sl dS   (only with sea_level)

        `D` is the total mass of the system times the displacement of its
        centre of mass. The centre of mass stays at the origin exactly when
        `D = 0`.

        **Why the mantle term is `rho_0 u` and not `x delta_rho`.** The Eulerian
        density change inside the mantle is `-div(rho_0 u)`, and the moving
        boundaries add the sheets `rho_0 (u.n)` at Rc and Re. Integration by
        parts of `x_i` against both gives `int rho_0 u_i dx` exactly, for any
        layered `rho_0`, with no jump term. This is the same identity that makes
        the potential source a divergence (`potential_residual`), so the moment
        constrained here is the moment of the mass that the potential row sees.

        The fluid core adds its own sheet, `sigma_core = rho_core (u.rhat)` at
        Rc (`fluid_core_sheet`). The load sheets in `DtNGravityForm.sigma_bcs`
        are prescribed, so they contribute a constant to `D`. A load with a
        degree-1 pattern therefore moves the solid Earth by the opposite amount,
        which is the physics of the centre-of-mass frame.

        The sea-level sheet `sigma_sl` (`surface_load_sheet`) is different. It
        depends on the solution through `u`, `psi`, `Shift` and the rotation
        potential, so its contribution to `D` is not constant.

        A check of the construction: a rigid translation `u = e` gives
        `D = (rho_0 V_mantle + rho_core V_core) e`, the whole mass of the body
        times `e`.

        Args:
          i: coordinate direction, 0-based.
          u: displacement; the current solution's by default.
          weight: a factor that multiplies every integrand. `None` gives the
            plain functional, which `assemble` turns into a number. The
            residual passes a multiplier or its test function here. It must be
            a `Real` field of the parent, its test function or a constant,
            because the fluid-core and sea-level terms restrict it
            (`_on_parent_facet`).
          scale: a further factor of every integrand that is not restricted,
            for example `c B_mu`. `None` is one.
          include_load: whether to add the moment of the load sheets. The
            prescribed sheets do not depend on `u`, so the variation of `D`
            with respect to the displacement leaves them out. The sea-level
            sheet depends on the solution. `centre_of_mass_residual` writes
            its variation as a separate column through
            `_sheet_test_variation`.

        Returns:
          A 0-form (or a form in whatever arguments `weight` carries).
        """
        if u is None:
            u = self.solution_split[self.layout.displacement]
        weight = Constant(1.0) if weight is None else weight
        scale = Constant(1.0) if scale is None else scale
        # The weight on the fluid-core and sea-level measures, where the
        # parent's `Real` fields need a restriction (`_on_parent_facet`).
        facet_weight = scale * self._on_parent_facet(weight)
        weight = scale * weight
        rho0 = self.approximation.density

        # The mantle term, on the intersected mantle measure. The `Real`
        # factors live on the parent mesh; that pairing is the one
        # `rotation_residual` already uses.
        form = weight * rho0 * u[i] * self.dx_m

        # The prescribed load sheets, written on the parent's coordinates
        # because the sheets are parent expressions on the parent's facets.
        if include_load:
            X_g = SpatialCoordinate(self.potential_mesh)
            for bc_id, sigma, integral_type in self.form.sigma_bcs:
                form = form + self.form.sheet_integral(
                    weight * ensure_constant(sigma) * X_g[i], bc_id,
                    integral_type)

        # The core sheet `sigma_core = -rho_core (u.n)` of the displacement
        # passed in, on the core's own facet-to-facet measure, with the
        # mechanics mesh's coordinates. It is written here and not through
        # `fluid_core_sheet_integral`, because that method reads the solution's
        # displacement: the residual evaluates this form at the test function,
        # and reading the solution there would drop the core traction from the
        # column and break the transpose.
        if self.fluid_core is not None:
            X_m = SpatialCoordinate(self.mesh)
            sigma_core = -ensure_constant(self.fluid_core.rho_core) * dot(
                u, FacetNormal(self.mesh))
            dss = self.fluid_core_measure()(self.fluid_core.boundary)
            form = form + facet_weight * X_m[i] * sigma_core * dss

        # The sea-level sheet of the current solution. It depends on `u`,
        # `psi`, `Shift` and the rotation scalars, so it is not linear in the
        # `u` passed in, and it is part of the load: the column of
        # `centre_of_mass_residual` writes its variation separately.
        if include_load and self.sea_level_parameters is not None:
            X_m = SpatialCoordinate(self.mesh)
            form = form + self.surface_load_sheet_integral(facet_weight * X_m[i])
        return form

    def centre_of_mass_energy(self) -> Form:
        r"""`c beta sum_i lambda_i D_i`: the energy whose variation fixes the frame.

        **What it does.** A rigid translation of the whole body, with the
        potential it carries, is a kernel of the continuous coupled problem
        when the reference gravity is consistent with the reference density
        and the core is fluid. The discrete operator then annihilates it up to
        discretisation error, measured on the 2-D annulus at `1.1e-4`,
        `2.0e-5`, `3.5e-6` relative on three refinements
        (`NOTES/frame/m1_translation_kernel.py`). Nothing else in the system
        picks a translation, so a load with a degree-1 pattern leaves the
        solver close to singular and the degree-1 displacement and geoid
        without meaning. The multipliers `lambda_i` enforce `D = 0`, which puts
        the centre of mass of mantle, core and load at the origin. That is the
        centre-of-mass frame the sea-level benchmarks use (Martinec et al.
        2018).

        **Why an energy.** Its `lambda`-variation is the constraint row
        `beta D_i[u] = 0`. Its `u`-variation is the column
        `beta lambda . (int rho_0 w dx + int x sigma_core(w) ds)`, a uniform
        acceleration applied to all of the moving mass. The two are transposes
        by construction, as for `fluid_core_energy`.

        **What the multiplier value means.** In the continuous problem a
        rigid translation costs no energy, so a load can exert no net force on
        the body and `lambda = 0`. Discretely `lambda` is the force that
        discretisation error leaves, and it must converge to zero with mesh
        refinement. If it does not, the reference gravity is not consistent
        with the density: with a constant `g` on the annulus the translation
        residual is 19 percent and stays there.

        **The scale `beta`.** `c * _row_scale_B_mu`, the same factor the
        core-pressure row carries. The column then has the magnitude of the
        body-force terms of the momentum row, and a floored `B_mu = 0` keeps
        the row non-empty.
        """
        beta = self.scaling_factor * self._row_scale_B_mu
        E = Form([])
        for i, index in enumerate(self.layout.centre_of_mass):
            lam_i = self.solution_split[index]
            E += self.mass_dipole_form(i, weight=lam_i, scale=beta)
        return E

    def centre_of_mass_residual(self) -> Form:
        r"""The variation of `centre_of_mass_energy`, written out term by term.

            c beta sum_i [ nu_i D_i[u]  +  lambda_i D_i^u[w] ]

        with `nu_i` the test function of `lambda_i` and `D_i^u` the part of
        `D_i` that depends on the displacement (the load moment left out). It
        equals `derivative(centre_of_mass_energy(), solution)`, and
        `tests/unit/test_gia_gravity.py` asserts that on the assembled vectors.

        **Why it is not written as `derivative(E, solution)`**, as
        `fluid_core_residual` is. `derivative` leaves the variation as an
        unevaluated UFL node. The load moment in `E` is an integral over the
        parent's interior facets, and its variation is zero. Firedrake's
        `split_form` extracts a sub-block by replacing the test function
        inside that node, and UFL cannot see that the result is zero until the
        derivative is evaluated. So every sub-block of the Jacobian keeps an
        integral on the parent mesh. `gadopt.InternalVariableSCPC` compiles the
        `(u, m)` block with Slate, which needs a single mesh, and fails with
        `ValueError: too many values to unpack (expected 1)` in
        `slate/slac/compiler.py`. Written out, the load moment carries the test
        function `nu_i` explicitly, and the split removes it from every block
        that does not contain `nu_i`. The fluid-core energy does not hit this,
        because all its integrals are on the mechanics mesh.
        """
        if not self.layout.centre_of_mass:
            return Form([])
        beta = self.scaling_factor * self._row_scale_B_mu
        w = self.tests[self.layout.displacement]
        F = Form([])
        for i, index in enumerate(self.layout.centre_of_mass):
            lam_i = self.solution_split[index]
            nu_i = self.tests[index]
            # The constraint row: the whole moment, load included.
            F += self.mass_dipole_form(i, weight=nu_i, scale=beta)
            # The column: the variation of D_i with respect to u, in the
            # direction of the displacement test function. D_i is linear in u,
            # so this is D_i evaluated at `w` without the constant load term.
            F += self.mass_dipole_form(i, u=w, weight=lam_i, scale=beta,
                                       include_load=False)
        if self.sea_level_parameters is not None:
            # With sea level on, `D_i` also contains `int x_i sigma dS`, and
            # `sigma` depends on every unknown in `Delta`. The column is its
            # variation, `lambda_i int x_i (d sigma/d SL) delta Delta dS`, in the
            # directions of the displacement, potential, Shift and rotation
            # test functions. The rotation part is included here: the rotation
            # row reads `D` through nothing else, so the `(m_i, lambda)` block
            # would otherwise be missing its transpose.
            X_m = SpatialCoordinate(self.mesh)
            variation = self._sheet_test_variation()
            dss = self._sea_level_ds()
            for i, index in enumerate(self.layout.centre_of_mass):
                lam_i = self._on_parent_facet(self.solution_split[index])
                F += beta * lam_i * X_m[i] * variation * dss
        return F

    def mass_dipole(self) -> list[float]:
        """The first mass moment `D` of the current state, one number per direction.

        Diagnostic. With the centre-of-mass multipliers active it is zero to
        the solver tolerance.
        """
        dim = self.mesh.geometric_dimension
        return [float(assemble(self.mass_dipole_form(i))) for i in range(dim)]

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

    @property
    def theta_psi_value(self) -> float:
        """`theta_psi` as a number, computed from its factors, never as a whole.

        `float(self.theta_psi)` raises on a `Real` `Function` control; see
        `scalar_value`. This is the only thing that should ever be used where a
        number is genuinely needed - the low-rank operator's per-application
        read, and the preconditioner diagonal. The UFL `theta_psi` stays the
        thing that goes into the residual, so `Lambda`, `B_mu` and
        `scaling_factor` remain live coefficients on the tape.
        """
        return (scalar_value(self.scaling_factor)
                * scalar_value(self._row_scale_B_mu)
                / scalar_value(self.Lambda))

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
                "symmetric reference density at all - give it directly, or "
                "give `k_s` and `surface_radius` and let "
                "`_resolve_rotation_moments` compute the consistent value.")
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

    @property
    def core_pressure(self) -> Function | None:
        """The uniform fluid-core pressure, or `None` without a fluid core."""
        index = self.layout.core_pressure
        return None if index is None else self.solution.subfunctions[index]

    def centre_of_mass_multipliers(self) -> list[float]:
        """The solved centre-of-mass multipliers `lambda_i`, or `[]` without them.

        `Real` data is replicated on every rank, so `float()` is local and
        agrees on every rank.
        """
        return [float(self.solution.subfunctions[i])
                for i in self.layout.centre_of_mass]

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

        **The two paths get the same numbers from different places.** On the
        multiplier path each `c_k` IS a solved unknown and is read straight out
        of its `Real` sub-field. On the low-rank path there is no such unknown:
        `c = C psi / (scale_k A_h)` is recovered from the trace, which is what
        `gadopt.dtn_adjoint.taped_trace_coefficients` does for the scalar
        solver. B5 reports `N(0)` and `N(180)` against TABOO, so this is on the
        critical path and not a diagnostic.

        **The old body zipped `form.multiplier_keys` against
        `layout.multipliers` and that is silent on the low-rank path.** The
        form still lists all 21 modes it treats while the layout has 0 unknowns,
        so `zip` truncates to nothing and every boundary comes back `{}` - no
        error, no warning, and a geoid of zero. `gravity_solver.py:648-652`
        records the same trap from the other side: `n_multipliers` on the form
        and in the space are two different numbers with one name. Every pairing
        below is length-checked rather than zipped.
        """
        if self.dtn_representation != "lowrank":
            keys, fields = self.form.multiplier_keys, self.layout.multipliers
            if len(keys) != len(fields):
                raise RuntimeError(
                    f"the form lists {len(keys)} multiplier keys but the "
                    f"layout has {len(fields)} multiplier fields; a zip here "
                    "would silently return the shorter of the two.")
            out = {bc_id: {} for bc_id, _ in self.form.dtn_boundaries}
            for (bc_id, key), i in zip(keys, fields):
                out[bc_id][key] = float(self.solution.subfunctions[i])
            return out

        # **Taped, and not read off a numpy array.** An earlier version of this
        # branch did `float()` of `dtn_operator.coefficients(...)`, which is
        # correct in value and severs the tape by construction: `geoid()` reads
        # this, B5 reports `N(0)` and `N(180)` through `geoid()`, and the method
        # that returned a geoid of exactly zero before the `zip` fix would then
        # have returned correct numbers with a gradient of exactly zero. The
        # scalar solver already solved this; `taped_coupled_trace_coefficients`
        # is the port of `gadopt.dtn_adjoint.taped_trace_coefficients` and the
        # maths is not re-derived here.
        return taped_coupled_trace_coefficients(self)

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

    def rotational_potential_expression(self, mesh, facet: bool = False):
        """`psi_rot` on the given mesh, from the solved rotation scalars.

        Built on the mechanics mesh for the body force and on the parent for the
        geoid; the polynomial is the same and the two meshes' coordinates agree
        pointwise where they overlap.

        Args:
          mesh: The mesh whose coordinates the polynomial is written in.
          facet: `True` for a term on the fluid-core or sea-level measure. The
            rotation scalars are `Real` fields of the parent, and on those
            measures they need a restriction (`_on_parent_facet`).
        """
        slots = self.layout.rotation_slots()
        n_rot = 1 if mesh.geometric_dimension == 2 else 3
        # A component the space does not carry is the literal zero, not a Real
        # unknown, so every rotation expression can be written in three
        # components whatever the dimension.
        values = [Constant(0.0) if i is None else self.solution_split[i]
                  for i in slots]
        if facet:
            values = [self._on_parent_facet(value) for value in values]
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
        # The weight on the fluid-core and sea-level measures, where the
        # rotation test function, a `Real` field of the parent, needs a
        # restriction (`_on_parent_facet`).
        facet_weight = self._on_parent_facet(weight)
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
        form = form + self.fluid_core_sheet_integral(facet_weight * p_m)
        # The sea-level sheet enters dI as any load sheet does. It depends on
        # the solution, so in the rotation row its variation is the transpose
        # partner of the `psi_rot / g_s` part of the sea level in the
        # sea-level rows (`sea_level_residual`).
        form = form + self.surface_load_sheet_integral(facet_weight * p_m)
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
            "SelfGravitatingGIASolver requires new-time algebraic coupling (BE/BDF2). A theta-weighted "
            "psi against an unweighted u in the Poisson source would break the "
            "(u, psi)/(psi, u) transposition by exactly theta, and a symmetry "
            "test would report it as a sign error.")

        u = self.solution_split[self.layout.displacement]
        psi = self.solution_split[self.layout.potential]

        if self.time_scheme == "bdf2" or self.constitutive_method == "direct":
            from .viscoelastic_timestepping import BDF2History
            histories = (list(self.internal_variables) if isinstance(self.internal_variables, (list, tuple))
                         else [self.internal_variables]) if self.layout.condensed else [
                             self.solution_old.subfunctions[self.layout.internal_variable_field]]
            self.time_integrator = BDF2History(
                histories, self.dt, order=2 if self.time_scheme == "bdf2" else 1)
        if self.constitutive_method == "direct":
            histories = self.time_integrator.current
            for history in histories:
                if history.function_space().ufl_element().family() != "Quadrature":
                    raise ValueError("Direct integration requires Quadrature history Functions")
                if history.function_space().ufl_element().degree() != self.quad_degree:
                    raise ValueError("History quadrature degree must equal solver quad_degree")
                if history.function_space().mesh() is not self.mesh:
                    raise ValueError("Direct integration histories must be on the mechanics mesh")
            from .constitutive_integrators import MaxwellQuadratureUpdate, BackstressQuadratureUpdate
            update = BackstressQuadratureUpdate if self._backstress_model else MaxwellQuadratureUpdate
            self._quadrature_update = update(self)

        if self.layout.condensed:
            # `CoupledInternalVariableSolver.__init__` configures the
            # approximation for the mixed formulation, where the displacement
            # block at fixed internal variables is elastic and the Nitsche
            # penalty and the preconditioner scale are `mu0`. The condensed
            # layout substitutes the history into the stress, so its
            # displacement tangent is the effective viscosity
            # `sum_i eta_i / (tau_i + dt)`, and that is the coefficient the
            # Nitsche pair must carry (the same choice
            # `PointwiseHistoryFormulation.configure_approximation` makes for
            # the substituted solver). Set it here, before the momentum terms
            # read `approximation.mu` below.
            alpha = self.time_integrator.alpha if self.time_integrator else 1
            self.approximation.mu = sum(
                mu * alpha / (alpha + self.dt / tau)
                for mu, tau in zip(self.approximation.shear_modulus,
                                   self.approximation.maxwell_times))
            # This legacy path substitutes the backward-Euler update into the
            # stress before differentiation:
            #   m_new = (m_old + (dt/tau) d(u)) / (1 + dt/tau)
            # This is `InternalVariableSolver.update_m` verbatim. It is a
            # pointwise reduction, not exact elimination of the weak DG history
            # block on a curved mesh. Two consequences remain important.
            # The u-tangent of the stress becomes
            # `sum_i eta_i/(tau_i + dt) = effective_viscosity(dt)`, which is why
            # the assignment above hands the Nitsche pair
            # that coefficient rather than `mu0` in this configuration. And the
            # legacy explicit update is insufficient for stress-dependent
            # viscosity. The direct constitutive path below supplies the
            # implicit local solve and consistent tangent instead.
            strain_u = self.approximation.deviatoric_strain(u)
            if self.constitutive_method != "direct" and float(getattr(self.approximation, "exponent", 1)) != 1:
                raise NotImplementedError(
                    "Pointwise substitution of the internal variables is "
                    "implemented for Newtonian rheology only (exponent = 1). "
                    "For a power law the Maxwell times depend on the deviatoric "
                    "stress, hence on m, and substituting m(u) changes the "
                    "Newton linearisation rather than merely eliminating a "
                    "block.")
            histories = (self.time_integrator.combinations() if self.time_integrator
                         else [self.internal_variables])
            internal_variables = [
                (m + self.dt / mt * strain_u) / (alpha + self.dt / mt)
                for m, mt in zip(history_slices(histories),
                                 self.approximation.maxwell_times)]
            if self._quadrature_update:
                internal_variables = [self._quadrature_update.on_mixed_state(g)
                                      for g in self._quadrature_update.updates]
        else:
            # One `(d, d)` slice per Maxwell element of the combined field.
            internal_variables = history_slices(
                self.solution_split[self.layout.internal_variable_field])

        intersect = self.potential_mesh if self.layout.cross_mesh else None

        stress = self.approximation.stress(
            u, internal_variables=internal_variables)
        source = self.approximation.buoyancy(u) * self.k
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

        if not self.layout.condensed:
            # One history equation for every Maxwell element, on the combined
            # field: the same three terms and the same sign convention as
            # `MixedHistoryFormulation.add_history_equations`. The residual
            # (mass + relaxation - strain) is negated through the scaling
            # factor so that the `(u, M)` and `(M, u)` blocks are transposes
            # of each other; `history_strain_term` carries the boundary term
            # on the weak `un` boundary (the CMB) that keeps that true, and
            # the condensed displacement operator is then symmetric for a
            # Newtonian rheology. `maxwell_times` carries the power-law
            # factor, so the relaxation is nonlinear in `u` for `exponent != 1`
            # exactly as the old per-field equations were.
            i = self.layout.internal_variable_field
            history_terms = internal_variable_history_terms
            history_attrs = {"maxwell_times": maxwell_times}
            if self._backstress_model:
                from .backstress import backstress_history_term
                history_terms = [backstress_history_term]
                history_attrs = {}
            self.equations.append(
                Equation(
                    self.tests[i],
                    self.solution_space[i],
                    history_terms,
                    eq_attrs={
                        **history_attrs,
                        "displacement": u,
                        "dt": self.dt,
                        "trial_old": (self.time_integrator.combinations()[0]
                                      if self.time_integrator else self.solution_old_split[i]),
                        "time_alpha": self.time_integrator.alpha if self.time_integrator else 1,
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
        one or two mechanics equations (displacement, and the combined
        internal-variable field on the uncondensed layout) against
        `n_fields` sub-fields, and `zip` would truncate to the shorter - which
        happens to pair correctly, and is exactly the kind of accident that
        stops being true when somebody reorders the space. Write it out.
        """
        mechanics = zip(self.equations,
                        (self.solution_split[self.layout.displacement],
                         *(self.solution_split[i]
                           for i in self.layout.internal_variables)))
        self.F = sum(eq.residual(sol) for eq, sol in mechanics)
        self.F += self.potential_residual()
        self.F += self.fluid_core_residual()
        self.F += self.sea_level_residual()
        if self.layout.rotation:
            self.F += self.rotation_residual()
        self.F += self.centre_of_mass_residual()
        # A `Real` control on the parent (`B_mu`, `Lambda`) in a facet term
        # needs a restriction for its derivative to exist.
        self.F = self._restrict_parent_real_coefficients(self.F)

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

        # **`None`, not an empty list, on the low-rank path.**
        # `boundary_residual` reads `multipliers is None` as "write the Robin
        # shift alone"; an empty *list* means "write the modal rows too, and
        # here are zero pairs for them", which raises a length mismatch. The
        # DtN feedback the modal rows would have supplied is then added as
        # `theta_psi * B0 psi` by the two callbacks, never here, because a form
        # cannot express a dense rank-n update without one term per mode -
        # which is the cost this path exists to remove.
        if self.dtn_representation == "lowrank":
            multipliers = None
        else:
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
        - every sheet, including the fluid core's, which is not in
        `sigma_bcs` and which `enclosed_mass_forms` therefore adds by hand.
        The one exception is the sea-level sheet, whose net mass the `Shift`
        row sets to zero; `enclosed_mass_forms` records why including it with
        a lag of one solve gives a wrong answer. That the enclosed mass is
        what it should be is a fact to assert, which is what `check_net_mass`
        is for.
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
        by construction, so the datum is driven by the sheets alone, except
        the sea-level sheet, whose net mass the `Shift` row sets to zero.
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
        # `mu` is the test function of a `Real` space on the parent, restricted
        # on the core's facet-to-facet measure (`_on_parent_facet`).
        core = self.fluid_core_sheet_integral(self._on_parent_facet(mu))
        if core.integrals():
            mass = core if mass is None else mass + core

        # The sea-level sheet is deliberately left out of the datum. The
        # `Shift` row makes its net mass zero at a converged state (to order
        # of the centre-of-mass multipliers, which is discretisation error), so
        # its correct contribution is zero. Including it would evaluate it one
        # solve late, like the core sheet above, and on the first solve after
        # the ice changes the previous state holds the net mass of the ice
        # change that the ocean has not yet balanced. Measured on the 2-D
        # annulus: a datum of 1.29e-2 and `Shift` and the geoid wrong by
        # 1.44e-3, 15 percent of the eustatic value, on that solve only
        # (`NOTES/findings/FINDING-2d-sea-level-build.md`). The sheet still
        # sets the scale of `check_net_mass`.

        flux = None
        for bc_id, value in self.form.flux_bcs:
            term = ensure_constant(value) * mu * self.form.ds(bc_id)
            flux = term if flux is None else flux + term
        # The same restriction as in the residual, for the `Real` controls
        # that the sheets can carry (`_restrict_parent_real_coefficients`).
        if mass is not None:
            mass = self._restrict_parent_real_coefficients(mass)
        if flux is not None:
            flux = self._restrict_parent_real_coefficients(flux)
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
        # `scalar_value`, not `float()`, on `self.form.G`. The form's `G` is
        # `self_gravity_number / (4 pi)` built at `self_gravitating_gia_space`
        # BEFORE any `ensure_constant`, and `ensure_constant` wraps only `float`
        # and `int` - everything else passes through. So with a `Function`
        # control this attribute is a UFL `Division` and `float()` of it raises
        # `Division.__float__ returned non-float`, on the solve path, via
        # `update_total_mass` -> `check_net_mass`. Reachable only in 2-D with an
        # exterior DtN boundary, since `monopole_boundaries` is empty otherwise -
        # which is why a direct conversion test could not see it.
        return (scalar_value(self.source_mass) + scalar_value(self.cavity_flux)
                / (4.0 * np.pi * scalar_value(self.form.G)))

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
        if self.sea_level_parameters is not None:
            # The mass that the sea-level sheet moves, for the same reason.
            scale += assemble(
                abs(self.surface_load_sheet()) * self._sea_level_ds())
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
        gpu_extras: ConfigType | None = None,
        iterative_preset: ConfigType | None = None,
        direct_preset: ConfigType | None = None,
    ) -> None:
        """PETSc options; both presets are two-block DtN Schur splits.

        `gpu_extras`, `iterative_preset` and `direct_preset` exist because the
        base class signature carries them. The two-block DtN presets have no
        GPU variant, and the Real-field splits are built here, so the two
        presets are ignored and `gpu_extras` is only forwarded on the
        `Mapping` path.

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

        A `Mapping` is honoured verbatim, through the base class, and refused
        when it names `ksponly` for a power law: that one combination reports
        `CONVERGED` on a state that is wrong by the whole nonlinearity, so
        there is nothing a caller can read to find out. See
        `_refuse_ksponly_on_a_nonlinear_residual`.
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
        # `Real` fields. No preset names `gadopt.DtNMultiplierDiagPC`, the one
        # class that reads this entry, so it is inert unless a caller names
        # that class.
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
            super().set_solver_options(solver_preset, solver_extras, gpu_extras)
            # The options are final on this path here, so the outer-method
            # check runs here too: it reads `self.solver_parameters`.
            self._refuse_ksponly_on_a_nonlinear_residual()
            _attach_block1_diagonal()
            self._attach_condensation_context(solver_extras)
            return
        if solver_preset not in (None, "direct", "iterative"):
            raise ValueError("Solver type must be 'direct' or 'iterative'.")

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
        if (self.dtn_representation == "lowrank"
                and not self.layout.real_fields):
            # With no global scalars the retained u/psi block is the whole
            # system. Keep the outer FGMRES and remove only the Real split.
            if self.layout.condensed and solver_preset == "iterative":
                options = selfgrav_dtn_iterative_solver_parameters(
                    condensed=True, dtn_representation="lowrank", n_real=0)
                reduced = {key: value for key, value in options.items() if not key.startswith("dtn_")}
                prefix = "dtn_fieldsplit_0_condensed_"
                reduced.update({"condensed_"+key[len(prefix):]: value for key, value in options.items()
                                if key.startswith(prefix)})
                reduced.update(pc_type="python", pc_python_type="gadopt.ReducedGIABlockPC")
                self.add_to_solver_config(reduced)
            else:
                self.add_to_solver_config(selfgrav_dtn_lowrank_direct_solver_parameters)
        elif solver_preset == "direct":
            # The low-rank path keeps block 1 unpreconditioned: its adjoint and
            # tangent solves reuse these parameters on an assembled matrix,
            # where the dense complement's sub-DM request crashes.
            self.add_to_solver_config(
                selfgrav_dtn_schur_lowrank_solver_parameters
                if self.dtn_representation == "lowrank"
                else selfgrav_dtn_schur_solver_parameters)
        else:
            # The representation has to be passed, and not left at the
            # preset's default: this is the path a 3-D driver takes when it
            # names no solver_parameters dictionary of its own, so leaving it
            # at "multiplier" would put plain GAMG on the potential split of a
            # low-rank solver, which `gadopt.CondensedBlockPC` hands the
            # Python block `A_psipsi + B`, and GAMG refuses that inside
            # `PCSetUp` with a PETSc type error.
            #
            # The width of the `Real` block is passed as well, so that the
            # block-1 choice is made on the quantity the cost depends on: a
            # build of the dense complement is one block-0 solve per row. The
            # layout is the one place that composition is written down, so
            # `len(real_fields)` covers every combination of the switches -
            # core pressure, rotation, DtN multipliers, and the centre-of-mass
            # and sea-level rows that `sghelichkhani/sea-level` adds - without
            # this call having to know which of them are on.
            #
            # This moves one case that the earlier rule decided by name: a
            # multiplier run whose block is at most `dense_schur_max_rows` wide
            # now gets the dense complement where it got `pc_type: none`. That
            # is a low degree only. The production arms do not move: low-rank
            # is 1 or 4 rows and already had the class, and the multiplier arm
            # at L = 5 is about 76 rows and keeps `"none"`.
            self.add_to_solver_config(selfgrav_dtn_iterative_solver_parameters(
                condensed=self.layout.condensed,
                dtn_representation=self.dtn_representation,
                n_real=len(self.layout.real_fields)))
        if solver_extras:
            self.add_to_solver_config(solver_extras)
        # The extras are the last thing that can name `snes_type`, so the
        # outer method is final here and the check runs on the value a solve
        # will actually use.
        self._refuse_ksponly_on_a_nonlinear_residual()
        _attach_block1_diagonal()
        self._attach_condensation_context(solver_extras)
        self.register_update_callback(self.set_solver)

    def _live_coastline(self) -> bool:
        """True when the surface load sheet has a coastline that follows the state.

        With `SeaLevel(fixed_ocean=False)` the sheet is the rounded maximum of
        the water and ice columns of the *current* sea level
        `SL = SL_init + dN - du_r + Shift` (`_sheet_of`,
        `gadopt.sea_level_sheet`). It is not affine in the unknowns, and the
        Jacobian of the sea-level rows moves inside one nonlinear solve. With
        `fixed_ocean=True` the sheet uses `C0 = H(r SL_init / eps)`, a frozen
        field, and the rows are linear.

        `fixed_ocean` is a fixed boolean on the `SeaLevel` dataclass, identical
        on every rank and unchanged for the life of the solver, so an answer
        taken from it is rank-consistent and can be published once into the
        application context.

        Returns:
            True when sea level is on and its coastline is free to move; False
            when there is no sea level at all, or its coastline is frozen.
        """
        sea_level = getattr(self, "sea_level_parameters", None)
        return sea_level is not None and not sea_level.fixed_ocean

    def _jacobian_depends_on_solution(self) -> bool:
        """True when the Jacobian changes within one nonlinear solve.

        Two independent causes on this solver, either of which is enough:

        - the rheology, which is what the base class tests: a power-law
          exponent puts the solution into the mechanics tangent;
        - a live coastline, where the surface load sheet follows the sea level
          and so follows the unknowns (`_live_coastline`).

        The second cause reaches more blocks than the first. `Shift` is a
        `Real` unknown, so the sheet sits in `A01`, `A10` and `A11` as well as
        in `A00`, and every cached factor of the two-block Schur
        preconditioner describes a state the solve has already left.

        The consequences of the answer are one correctness rule and one cost
        rule. `_refuse_ksponly_on_a_nonlinear_residual` refuses a single
        linear solve on a residual that is not linear. And
        `CoupledInternalVariableSolver._refresh_operator_version` publishes
        `operator_version = None`, which moves every cache keyed on it
        (`gadopt.CondensedBlockPC`, `_RebuildRule`) off the "nothing changed"
        branch it would otherwise sit on for the whole march.

        This does **not** change `condensed_operator_symmetric`, which reads
        the rheology alone. The sea-level rows keep a symmetric Jacobian with
        a live coastline, because the sheet depends on the unknowns only through the
        one scalar `Delta = dN - du_r + Shift`, so one energy `E_sl` generates
        all three rows (`NOTES/DESIGN-SEA-LEVEL.md` section 3). Block 0 is
        `(u, psi)` and the sheet enters it through that same energy, so the
        condensed block-0 operator is as symmetric with a live coastline as
        without one.

        Returns:
            True when a Newton method is required and no operator version can
            describe the Jacobian; False when the residual is linear in the
            unknowns.
        """
        return super()._jacobian_depends_on_solution() or self._live_coastline()

    def _refuse_ksponly_on_a_nonlinear_residual(self) -> None:
        """Refuse `snes_type ksponly` when the Jacobian depends on the state.

        `ksponly` takes one linear solve at the initial state and reports
        `CONVERGED`, which is the right thing for a linear residual and a
        wrong answer with no diagnostic otherwise: the returned state is off
        by the whole nonlinearity and nothing in the log says so. Both
        docstrings that recommend `ksponly` recommend it for `exponent = 1`,
        so one misuse this guard catches is a copy of such a dictionary into a
        power-law run.

        **Two causes, and the message names whichever applies.** A power-law
        exponent is one. A live coastline is the other: with
        `SeaLevel(fixed_ocean=False)` the surface load sheet follows the sea
        level, so the sea-level rows are
        nonlinear even on a Newtonian Earth, and a message that named only the
        exponent would send the reader to look at a rheology that is already
        correct. A configuration can carry both.

        Called on both options paths, after the options are final, because
        `snes_type` can arrive from the preset, from a caller's `Mapping` or
        from `solver_parameters_extra`, and only the merged dictionary says
        which method a solve will use.

        Raises:
          ValueError: the outer method is `ksponly` and the rheology, the
            moving coastline, or both make the Jacobian state-dependent.
        """
        if self.solver_parameters.get("snes_type") != "ksponly":
            return
        if not self._jacobian_depends_on_solution():
            return
        exponent = getattr(self.approximation, "exponent", 1)
        # Each cause contributes the sentence that describes it and the way
        # out of it, so a reader with both gets both and a reader with one is
        # not sent to look at the other.
        causes, remedies = [], []
        if super()._jacobian_depends_on_solution():
            causes.append(f"the power-law factor of exponent={exponent!r}")
            remedies.append(
                "set exponent=1 if the rheology is meant to be Newtonian")
        if self._live_coastline():
            causes.append(
                "the live sea-level sheet of SeaLevel(fixed_ocean=False), "
                "which follows the sea level and so follows the unknowns,")
            remedies.append(
                "set SeaLevel(fixed_ocean=True) if the coastline is meant to "
                "be frozen")
        # The predicate and this list read the same two branches, so the list
        # cannot be empty today. It would become empty if a third cause of a
        # state-dependent Jacobian were added to the base class without a
        # branch here, and the message would then be a sentence with a hole in
        # it. A diagnostic that describes nothing is worse than a general one.
        if not causes:
            raise ValueError(
                "snes_type='ksponly' is refused because the Jacobian depends "
                "on the solution, so one linear solve at the initial state "
                "converges by definition and returns a state that is wrong by "
                "the whole nonlinearity. Use snes_type='newtonls'. The cause "
                "is a branch of _jacobian_depends_on_solution that "
                "_refuse_ksponly_on_a_nonlinear_residual does not name; add "
                "it there.")
        verb = "makes" if len(causes) == 1 else "make"
        raise ValueError(
            f"snes_type='ksponly' is refused because {' and '.join(causes)} "
            f"{verb} the residual nonlinear, so one linear solve at the "
            "initial state converges by definition and returns a state that "
            "is wrong by the whole nonlinearity. Use snes_type='newtonls' "
            "(the default of both self-gravity presets), or "
            f"{', or '.join(remedies)}.")

    def _attach_condensation_context(self, extras) -> None:
        """Publish what the block-0 condensations read, on every options path.

        The keys serve both `gadopt.InternalVariableSCPC` (the `"pair"` value
        of the preset's `block0`) and `gadopt.CondensedBlockPC` (the
        `"condensed"` value) alike.

        The base class publishes these on its own path
        (`CoupledInternalVariableSolver.set_solver_options`), which the string
        presets of this solver never reach. The keys are fixed names in the
        shared application context, whatever option prefix the condensation
        sits under, so publishing them once serves the nested split of the
        iterative preset and any hand-written dictionary alike:

        - `approximation`, `dt`, `scaling_factor`: what a preconditioner that
          rebuilds the displacement operator from the material needs.
        - `operator_version`: the reuse counter; `_refresh_operator_version`
          bumps it before every solve when `dt` or a Jacobian coefficient
          changed, and the condensation skips its reassembly otherwise.
        - the three `condensed_field_*nullspace` providers: the displacement
          part of a nullspace the caller declared on the mixed space; and, for
          the near-nullspace, the modes named by `condensed_near_nullspace`
          built on the condensed displacement space when the caller declared
          none. GAMG coarsens the slow modes onto whatever it is given, and
          the condensed operator carries the volumetric penalty of the
          internal-variable stress, so it needs the rigid-body modes and the
          low-degree divergence-free fields, which is what the default
          `"incompressible"` builds.

        Two further keys serve `gadopt.DtNMultiplierDenseSchurPC`, which forms
        the whole multiplier Schur complement by one block-0 solve per column
        and must know when that complement is worth forming again. That
        preconditioner keys its rebuild on `operator_version` above, the same
        value the condensation uses, so the two caches of pieces of one
        operator agree by construction. These two keys cover what the version
        cannot say:

        - `gia_solve_index`: the number of `solve()` calls this solver has made,
          starting at 0 and incremented by `solve` before the nonlinear solve
          runs, so the first solve carries index 1. A power law publishes
          `operator_version = None`, because no version can describe a Jacobian
          that moves inside one nonlinear solve; the solve index is what turns
          "rebuild once per solve" into a test a preconditioner can make
          without knowing about Newton at all.
        - `gia_jacobian_depends_on_solution`: a `bool`, fixed for the life of
          the solver, true for a power law and true for a live coastline
          (`_jacobian_depends_on_solution`). `operator_version` carries the same
          fact from the first `solve` onwards, by going `None`, and this key
          carries it **before** that: at construction the version is still the
          integer 0 on every rheology, so a caller or a preconditioner that has
          to know the answer before any solve reads this one.

        Both keys are rank-consistent, which the collective rebuild needs:
        `gia_solve_index` is incremented by the collective `solve()` on every
        rank alike, and `gia_jacobian_depends_on_solution` is a deterministic
        function of the exponent and of `SeaLevel.fixed_ocean`, both of which
        hold the same value on every rank. So the rank-local rebuild decision
        agrees on every rank and the collective `_build` of the dense
        complement is entered by all ranks together.

        On the condensed layout nothing reads these; they are harmless there.

        Also selects the Krylov method on the condensed field, when there is
        one to select. A short CG acts as a preconditioner inside the flexible
        block-0 FGMRES and is valid only for a symmetric operator. The
        rheology alone decides that (`condensed_operator_symmetric`): a
        Newtonian rheology keeps the CG, and every power law takes an equally
        short GMRES whose restart is the cap the preset wrote.

        **The switch is inert on the preset's default**, which is
        `u_ksp_max_it=0` and writes `ksp_type preonly` on that split. The test
        below is for `"cg"`, so a default run of a power-law rheology carries
        no `ksp_gmres_restart` key and none is needed: one GAMG V-cycle is not
        a Krylov method and does not assume symmetry. The switch fires for the
        caller who asks for the truncated CG with `u_ksp_max_it`. The self-gravity
        configurations are all on the nonsymmetric side of that rule by more
        than one route anyway - measured 3.6e-3 asymmetry on the 2-D annulus
        with a rigid core, 9.4e-2 on the 3-D sphere with a fluid core - so the
        switch fires for a power law here whatever the mesh. The `Mapping` a
        caller passes is normally
        `selfgrav_dtn_iterative_solver_parameters(...)` itself, where a `cg`
        comes from `u_ksp_max_it` and not from a named `ksp_type`, so a `cg`
        found there is switched like the string path's.

        The displacement split sits at a different prefix on each block-0
        route, so the switch runs over both: `block0="pair"` puts it under
        `dtn_fieldsplit_0_fieldsplit_0_condensed_field_`, and the default
        `block0="condensed"` puts it under
        `dtn_fieldsplit_0_condensed_fieldsplit_0_`. The one way to keep CG on
        an operator this rule calls nonsymmetric is to name `ksp_type` under
        the prefix of the route in use in `solver_parameters_extra`, which
        wins; a key written under the other route's prefix is never read.

        Args:
          extras: the caller's `solver_parameters_extra`, or `None`.
        """
        if (self.constitutive_method == "direct" and self.dtn_representation == "lowrank"
                and self._live_coastline() and "gia_rebuild" not in self.solver_parameters):
            # A moving coastline can defeat a tangent frozen at the replay's
            # zero initial guess. Honour any explicitly chosen lag policy.
            self.add_to_solver_config({"gia_rebuild": "always"})
        if getattr(self, "appctx", None) is None:
            self.appctx = {"mu": self.approximation.mu / self.rho_continuity}
        self.appctx["approximation"] = self.approximation
        self.appctx["dt"] = self.dt
        self.appctx["scaling_factor"] = self.scaling_factor
        self.appctx["operator_version"] = self._operator_version
        # Fixed for the life of the solver: the rheology cannot change under a
        # built residual, so the dense complement can read this once per update
        # and trust it.
        self.appctx["gia_jacobian_depends_on_solution"] = (
            self._jacobian_depends_on_solution())
        # `solve` increments this, so the first solve runs at index 1 and a
        # preconditioner that initialises during that solve records 1 and does
        # not rebuild inside it.
        self.appctx["gia_solve_index"] = 0
        for key, basis in (
            ("condensed_field_nullspace", self.nullspace),
            ("condensed_field_transpose_nullspace", self.transpose_nullspace),
        ):
            displacement_basis = _displacement_basis(basis)
            if displacement_basis is not None:
                self.appctx[key] = _basis_provider(displacement_basis)
        near = _displacement_basis(self.near_nullspace)
        if near is not None:
            self.appctx["condensed_field_near_nullspace"] = _basis_provider(near)
        elif self.condensed_near_nullspace != "none":
            modes = self.condensed_near_nullspace

            def near_nullspace_provider(space, modes=modes):
                # Built on the condensed displacement space the
                # preconditioner hands over. `CondensedBlockPC` calls this
                # once and attaches the same basis on every operator rebuild.
                return near_nullspace_basis(space, modes)

            self.appctx["condensed_field_near_nullspace"] = near_nullspace_provider

        # The displacement split sits at a different prefix on each block-0
        # route, and the switch has to reach whichever one the caller selected:
        # `"pair"` puts the condensed displacement solve under
        # `dtn_fieldsplit_0_fieldsplit_0_condensed_field_`, and
        # `gadopt.CondensedBlockPC` puts it under
        # `dtn_fieldsplit_0_condensed_fieldsplit_0_`. A route whose prefix this
        # loop does not name keeps CG on a nonsymmetric operator, which does
        # not raise and converges to the wrong thing or not at all. Both are
        # visited because a hand-written dictionary may name either.
        for prefix in ("dtn_fieldsplit_0_fieldsplit_0_condensed_field_",
                       "dtn_fieldsplit_0_condensed_fieldsplit_0_",
                       # With no `Real` field the two-block Schur split has
                       # nothing to split, `gadopt.CondensedBlockPC` sits as the
                       # outer preconditioner, and its displacement split loses
                       # the `dtn_fieldsplit_0_` prefix entirely.
                       "condensed_fieldsplit_0_"):
            key = prefix + "ksp_type"
            max_it_key = prefix + "ksp_max_it"
            restart_key = prefix + "ksp_gmres_restart"
            user_named = bool(extras) and key in extras
            if (self.solver_parameters.get(key) == "cg"
                    and not user_named
                    and not self.condensed_operator_symmetric()):
                # The restart is the preset's own iteration cap, so the GMRES
                # that replaces the CG costs the same Krylov vectors and the
                # same number of GAMG V-cycles per block-0 iteration. A restart
                # longer than the cap would allocate vectors the solve never
                # reaches. 50 is the fallback for a hand-written dictionary
                # that names no cap, where the condensed solve runs to its own
                # tolerance instead.
                self.add_to_solver_config({
                    key: "gmres",
                    restart_key: self.solver_parameters.get(max_it_key, 50)})

    def check_boundary_quadrature(self, *args, **kwargs):
        """Measures whether the boundary rule resolves the DtN modes.

        Forwarded from the boundary form, and worth running on the *coupled*
        system rather than only on a standalone gravity solve: a DtN constraint
        row integrated at the wrong degree is a wrong constraint with no
        warning.
        """
        return self.form.check_boundary_quadrature(*args, **kwargs)

    # -- The low-rank DtN operator ------------------------------------------

    def build_dtn_operator(self) -> CoupledLowRankDtN:
        """`B = theta_psi * C^T W C` on the psi rows, built and returned.

        **This builds an operator and changes nothing else.** The space, the
        form, the residual, the Jacobian and the preconditioner are exactly
        what the multiplier path produces, so a solver constructed with
        `dtn_representation="lowrank"` today solves the multiplier system and
        carries `B` alongside it. That is deliberate: if `B` is wrong, finding
        out here means finding out as an operator problem rather than as a
        solver problem.

        Three things this method must get right, all previously paid for:

        1. **The index shift.** `build_mode_rows` returns owned local indices
           into the potential space; the coupled operator acts on the
           monolithic mixed vector. `CoupledLowRankDtN` shifts them with
           `psi_local_offset`, which reads the field index sets. Summing
           `subfunctions[i].dat` sizes instead is wrong on every rank above 0.
        2. **`apply_dirichlet_to_rows` is not optional.** `A` has its
           constrained rows and columns eliminated, so a `B` that still couples
           to a prescribed degree of freedom is inconsistent with it and is not
           symmetric either. Writing into a constrained row violates the
           boundary condition by exactly the amount written, silently.
        3. **`theta_psi` is not folded in.** It multiplies the whole potential
           row, the DtN constraint and feedback rows included, so it multiplies
           `B`. It is passed as a callable and read at every application, since
           `scaling_factor`, `B_mu` and `Lambda` are all `Constant`s that an
           adjoint can control, and a frozen product cannot be corrected at
           replay time.

        Returns:
          The `CoupledLowRankDtN`, also stored as `self.dtn_operator`.
        """
        mode_rows = self.form.build_mode_rows()

        # **This guard used to compare the rebuilt `multiplier_keys` against
        # the ones `boundary_bilinear` left behind, and on this path that list
        # is always empty - so the check never executed.** It read as
        # protection and was decoration. What matters here is the alignment the
        # operator and `coefficients()` actually rely on: `mode_rows[i]` must
        # belong to `dtn_boundaries[i]`, and within it `keys[k]` must be the
        # k-th mode of that boundary's own descriptor. Recomputed from the
        # descriptor below, which is an independent source, so this runs and
        # can fail.
        boundaries = self.form.dtn_boundaries
        if len(mode_rows) != len(boundaries):
            raise RuntimeError(
                f"build_mode_rows returned {len(mode_rows)} row sets for "
                f"{len(boundaries)} DtN boundaries; they are paired by "
                "position everywhere downstream.")
        for (bc_id, dtn), rows in zip(boundaries, mode_rows):
            side, radius = self.form.boundary_geometry[bc_id]
            expected = [mode.key for mode in dtn.mode_metadata(side, radius)]
            if list(rows.keys) != expected:
                raise RuntimeError(
                    f"boundary {bc_id}: the mode rows are ordered "
                    f"{list(rows.keys)} but the descriptor's own order is "
                    f"{expected}. Every trace coefficient would be attributed "
                    "to the wrong mode, and the weights would be applied to "
                    "the wrong functionals.")

        # Rebuilt from the form rather than filtered out of `self.strong_bcs`,
        # so this cannot pick up a mechanics condition and cannot miss a
        # potential one; `set_form` builds them from exactly this list.
        psi_space = self.solution_space.sub(self.layout.potential)
        constrained = set()
        for bc_id, val in self.form.dirichlet_bcs:
            constrained.update(np.asarray(
                DirichletBC(psi_space, val, bc_id).nodes, dtype=np.int64
            ).tolist())
        apply_dirichlet_to_rows(mode_rows, constrained)
        self.dtn_constrained_dofs = constrained

        self.dtn_operator = CoupledLowRankDtN(
            self.solution_space, self.layout.potential, mode_rows,
            # A callable, not a float. See point 3 above.
            lambda: self.theta_psi_value, self.potential_mesh.comm)
        # Published so that `gadopt.CondensedBlockPC` can put the update inside
        # block 0's potential split, where the outer Jacobian's augmentation
        # cannot reach: Firedrake's `createSubMatrix` builds every fieldsplit
        # sub-block as a plain `ImplicitMatrixContext`, so without this key
        # block 0 preconditions a system with no `B` in it and the outer FGMRES
        # pays 8 or 9 iterations where the multiplier representation pays 3.
        # The key is absent on the multiplier representation, and its absence
        # is what keeps that path unchanged.
        #
        # `_attach_condensation_context` runs inside `super().__init__` and
        # creates `self.appctx`, and this method runs after it, so the
        # dictionary exists here. It is the same dictionary the solver hands to
        # the preconditioners, and it is read at their `initialize`, which is
        # the first solve, so publishing here is early enough.
        if getattr(self, "appctx", None) is None:
            # Unreachable by construction, and a hard error rather than a new
            # dictionary: a dictionary made here would not be the one the
            # solver already handed to Firedrake, so the key would reach no
            # preconditioner and block 0 would lose `B` with nothing to say so.
            raise RuntimeError(
                "SelfGravitatingGIASolver.build_dtn_operator ran before the "
                "application context existed. The context is created by "
                "set_solver_options during the base constructor, and this "
                "method runs after it; reaching this means the construction "
                "order changed and the low-rank update would not reach "
                "block 0.")
        self.appctx["dtn_operator"] = self.dtn_operator
        return self.dtn_operator

    # -- The two augmentations, which must stay consistent -------------------

    def augment_residual(self, X, F) -> None:
        """`post_function_callback`: add `theta_psi * B0 psi` to the psi rows.

        `X` is the current iterate and `F` the residual, both as monolithic
        PETSc vectors. The callback receives `ctx._F`'s vec **writable** and the
        copy-out happens afterwards, so modifying it in place is correct.

        The constrained psi rows are safe: `_assemble_residual` zeroes them and
        `apply_dirichlet_to_rows` zeroed `B`'s rows there, so nothing is written
        into a row whose value is prescribed. Writing into one violates the
        boundary condition by exactly the amount written, silently.
        """
        self.dtn_operator.apply_local(X.array_r, F.array_w)

    def augment_jacobian(self, X, Jmat) -> None:
        """`post_jacobian_callback`: give the matrix-free action the same `B`.

        **Both augmentations or neither.** Under `snes_type: "ksponly"`, which
        `selfgrav_dtn_iterative_solver_parameters` recommends for the
        production `exponent = 1`, augmenting the residual alone is inert from a
        zero initial guess - `B z` is zero, one Newton step reproduces the
        un-augmented answer, and PETSc reports CONVERGED - and simply wrong from
        any other guess. Measured `||z_cb - z_plain|| = 0.000e+00` at every eps
        (`NOTES/fastdtn/REVIEW-FORWARD.md` section 1.2). So under production
        settings this callback **is** the correctness, not an optimisation.
        """
        install_augmented_context(Jmat, self.dtn_operator)

    def adjoint_solve_keywords(self) -> dict:
        """No adjoint keywords from here on the low-rank representation.

        `StokesSolverBase` hands the cached adjoint solver the forward solver's
        own keywords with the options replaced, which is what makes a Taylor
        test complete at Earth scale on the multiplier representation.

        The low-rank representation cannot take those keywords. Its forward
        options put `gadopt.LowRankPotentialPC` on the potential split, and
        that class needs a matrix-free operator with the low-rank update. The
        derivative solves of that representation need both augmentation
        callbacks, the application context with `dtn_operator` and the
        swapped nullspaces instead. `LowRankVariationalSolver.solve` supplies
        those keywords itself, for the adjoint and the tangent solver
        (`gadopt.dtn_coupled_adjoint.derivative_solver_kwargs`), so this
        method returns nothing there and `StokesSolverBase.solve` passes
        nothing on top.

        Returns:
          `{}` on the low-rank representation, else the keywords of
          `StokesSolverBase.adjoint_solve_keywords`.
        """
        if self.dtn_representation == "lowrank":
            return {}
        return super().adjoint_solve_keywords()

    def set_solver(self) -> None:
        """The base solver, rebuilt with the two callbacks on the low-rank path.

        Rebuilt and not patched: `post_function_callback` and
        `post_jacobian_callback` are public constructor arguments of both
        `NonlinearVariationalSolver` and `LinearVariationalSolver`, and reaching
        into `solver._ctx` afterwards would be a private-API dependency for no
        gain. Constructing a solver assembles nothing, so the discarded first
        object costs no work. The class is `LowRankVariationalSolver`, which
        supplies the adjoint solver's keywords on every annotated solve, so
        the derivative solves carry the same two callbacks, this application
        context and `snes_type: ksponly`.
        """
        if self.time_integrator:
            self.appctx["time_alpha"] = self.time_integrator.alpha
        super().set_solver()
        if self._quadrature_update:
            if self.strong_bcs:
                raise NotImplementedError("Direct GIA adjoint currently requires natural boundary conditions")
            from .constitutive_integrators import DirectConstitutiveBackend
            self._direct_backend = DirectConstitutiveBackend(self, self._quadrature_update)
            self.solver = self._direct_backend.solver
            self.problem = self._direct_backend.problem
            self.J = self._direct_backend.J
            return  # The direct block owns low-rank replay and reverse hooks.
        if self.dtn_representation != "lowrank":
            return
        if self.constant_jacobian:
            # The base builds a `LinearVariationalSolver` on a
            # `LinearVariationalProblem` in this case, and the class below
            # derives from `NonlinearVariationalSolver`, whose default
            # `snes_type` is `newtonls` against the linear solver's `ksponly`.
            # A forward solve would then run a Newton loop on a linear system
            # with a preset that names no `snes_type`, one residual evaluation
            # per solve for nothing. No driver passes this combination; it is
            # refused instead of being silently reshaped.
            raise NotImplementedError(
                "dtn_representation='lowrank' with constant_jacobian=True: the "
                "low-rank path rebuilds the solver as a "
                "NonlinearVariationalSolver subclass to name its adjoint "
                "solver's keywords, which the LinearVariationalProblem the "
                "constant-Jacobian path builds does not fit. Use "
                "constant_jacobian=False; the operator-version reuse of the "
                "block-0 preconditioner already avoids the reassembly a "
                "constant Jacobian would save.")
        # `LowRankVariationalSolver` and not the base class: it names the
        # adjoint solver's keywords (and, with Firedrake PR 4638, the tangent
        # solver's) on every annotated solve, so the derivative solves carry
        # the two callbacks, this context and `ksponly`
        # (`gadopt.dtn_coupled_adjoint`, "Both augmentations, and ksponly").
        from .dtn_coupled_adjoint import LowRankVariationalSolver
        self.solver = LowRankVariationalSolver(
            self.problem,
            solver_parameters=self.solver_parameters,
            nullspace=self.nullspace,
            transpose_nullspace=self.transpose_nullspace,
            near_nullspace=self.near_nullspace,
            appctx=self.appctx,
            options_prefix=self.name,
            post_function_callback=self.augment_residual,
            post_jacobian_callback=self.augment_jacobian,
        )
        self.solver.gia_solver = self

    def block1_diagonal(self):
        """The exact diagonal of the `Real` block, or `None` if there is none.

        `DtNMultiplierDiagPC` reads this out of the appctx. Supplied
        unconditionally, costs nothing to compute -- no assembly, no solves --
        and **no preset names that class**, so the entry is inert unless a
        caller selects it by name. The block-1 preconditioner the iterative
        preset does select on the low-rank representation,
        `gadopt.DtNMultiplierDenseSchurPC`, reads nothing from the appctx: it
        forms its data from the operator it is handed.

        Three contributions:

        * the **multipliers**, `theta_psi * (-scale_k * A_h)`, from
          `DtNGravityForm.multiplier_diagonal`, which derives the sign and
          explains why the *discrete* boundary measure is the right one;
        * the **core pressure**, whose diagonal is exactly zero because it is a
          Lagrange multiplier rather than a compressibility law;
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
        sub-field -- one added for something that is not a DtN multiplier, core
        pressure, or rotation row. The 2-D monopole datum deliberately builds
        its `Real` space **outside** the mixed space.

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
                "DtN multipliers, the optional core pressure, the rotation "
                "closure rows, then the optional centre-of-mass multipliers, "
                "then the optional sea-level Shift, in that order and last. Another Real field or a "
                "reordering invalidates it. Fix the accounting here before "
                "using DtNMultiplierDiagPC on this configuration.")

        by_index = {}
        # **`multiplier_keys` is NOT the test for whether multiplier rows
        # exist, on the low-rank path.** `build_mode_rows` fills that list with
        # every treated mode, because the form still knows what it treats -
        # while `layout.multipliers` is empty, because none of them is an
        # unknown. Keying off the form here would then find `n_multipliers`
        # diagonal entries for zero `Real` rows and raise the length mismatch
        # below. The layout is the authority on what is in the space.
        has_multiplier_rows = (self.layout.dtn_representation == "multiplier"
                               and bool(self.layout.multipliers))
        if form is not None and has_multiplier_rows and getattr(
                form, "multiplier_keys", None):
            mult = self.theta_psi_value * form.multiplier_diagonal()
            if len(mult) != len(self.layout.multipliers):
                raise RuntimeError(
                    f"the form describes {len(mult)} multiplier rows but the "
                    f"layout has {len(self.layout.multipliers)}.")
            by_index.update(zip(self.layout.multipliers, mult))
        if self.layout.core_pressure is not None:
            by_index[self.layout.core_pressure] = 0.0
        for k, name in enumerate(self.layout.rotation_names):
            idx = self.layout.rotation.get(name)
            if idx is not None:
                by_index[idx] = (self.CLOSURE_SIGNS[k] * scalar_value(self.scaling_factor)
                                 * scalar_value(self._row_scale_B_mu) * scalar_value(self.Omega_sq)
                                 * scalar_value(self._closure_constant(k)))
        # The centre-of-mass multipliers are Lagrange multipliers, like the
        # core pressure, so their diagonal is exactly zero. `getattr` keeps a
        # layout that predates the field working.
        for idx in getattr(self.layout, "centre_of_mass", ()):
            by_index[idx] = 0.0
        # The sea-level Shift row. Its assembled diagonal is
        # `-c B_mu g_s rho_w int (1 - H(f / eps)) dS`, the water fraction of the
        # softplus sheet (or `C0` for a fixed coastline), which depends
        # on the state. No assembly is allowed here, so the entry is zero, as
        # for a Lagrange multiplier. `DtNMultiplierDiagPC` is therefore not
        # suitable for a space with sea level.
        if getattr(self.layout, "sea_level", None) is not None:
            by_index[self.layout.sea_level] = 0.0

        missing = [i for i in expected if i not in by_index]
        if missing:
            raise RuntimeError(
                f"no diagonal entry was derived for Real sub-fields {missing}; "
                "they are in the layout but no DtN multiplier, core-pressure, "
                "rotation or centre-of-mass row described them.")
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

        So the projection is applied here, after each solve. On the
        quadrature path (`constitutive_method="direct"`) the solve block has
        already applied it inside its own map, and this is the second
        application, which changes nothing beyond round-off. It is
        legitimate precisely when the declaration is: a genuine kernel admits
        no solution at all unless the residual is orthogonal to it, so removing
        it selects the minimum-norm representative and discards nothing. The
        converse is the warning - a *wrongly* declared kernel now silently
        deletes part of the answer instead of merely being ignored, which is
        why `rigid_rotation_nullspace` documents when the rigid rotation is and
        is not one.

        Only blocks carrying an actual `VectorSpaceBasis` are touched; the
        entries that are plain sub-spaces mean "no kernel here" and are skipped.

        The projection is a recorded operation (`gadopt.nullspaces.
        project_out_kernel`). The coefficient change goes through the PETSc
        vector, which pyadjoint cannot see, so the function records a
        `KernelProjectionBlock` around it. Replay then returns the projected
        solution, as the forward run did, and the adjoint, the tangent linear
        model and the Hessian apply the same projection. The forward values
        are those of `VectorSpaceBasis.orthogonalize`, bit for bit.
        """
        return project_out_kernel(self.nullspace, self.solution)

    def solve(self) -> None:
        """Refreshes the enclosed mass, solves, then projects out the kernel."""
        if self.time_integrator:
            self.time_integrator.prepare()
        self.update_total_mass()
        # Count the solve BEFORE it runs, so that every preconditioner setup
        # inside it sees one index and a preconditioner built during this solve
        # keeps its complement for the whole of it. The counter is what lets
        # `gadopt.DtNMultiplierDenseSchurPC` rebuild once per time step on a
        # state-dependent Jacobian without counting Newton iterations, which
        # costs three times the block-0 work of a step. A pyadjoint replay
        # goes through `_forward_solve` and never through this method, so a
        # replayed solve leaves the index where the forward solve left it.
        self.appctx["gia_solve_index"] = self.appctx.get("gia_solve_index", 0) + 1
        # Two marker blocks around the taped solve. A replay or a tangent
        # sweep reaches the first before the solve, an adjoint or a Hessian
        # sweep the second, and each raises `gia_tape_sweep` when a new sweep
        # starts, so the rebuild rule of the preconditioners rebuilds for a
        # replay at a new control value (`TapeSweepMarkerBlock`).
        from .dtn_coupled_adjoint import TapeSweepMarkerBlock
        markers = annotate_tape()
        if markers:
            get_working_tape().add_block(TapeSweepMarkerBlock(
                self, self.appctx["gia_solve_index"], before=True))
        # Route 1.5b: on the low-rank path, let the stock annotated solve run,
        # then take its solve block off the tape and re-class it so the adjoint
        # and tangent carry `A + B` and the theta derivative. Record the block
        # count BEFORE the solve; `update_total_mass` has already added its own
        # block, and `project_out_nullspace` adds more after, so the block is
        # found by output identity among what the solve itself added.
        adopt = (self.dtn_representation == "lowrank" and self._direct_backend is None
                 and annotate_tape())
        if adopt:
            tape = get_working_tape()
            n0 = len(tape.get_blocks())
        if self._direct_backend:
            self._refresh_operator_version()
            self._direct_backend.solve()
        else:
            super().solve()
        if markers:
            get_working_tape().add_block(TapeSweepMarkerBlock(
                self, self.appctx["gia_solve_index"], before=False))
        # Publish the outer linear iterations and the Newton iterations of
        # this solve. `gia_rebuild = lagged` reads their ratio at the
        # next solve to decide whether its lagged complement of the `Real`
        # block has become poor enough to rebuild. Both counts come from the
        # SNES, which runs collectively, so they are the same on every rank.
        # PETSc resets the linear count at the start of every SNESSolve.
        snes = self.solver.snes
        self.appctx["gia_last_solve_linear_its"] = int(
            snes.getLinearSolveIterations())
        self.appctx["gia_last_solve_newton_its"] = max(
            1, int(snes.getIterationNumber()))
        if adopt:
            from .dtn_coupled_adjoint import adopt_coupled_lowrank_block
            adopt_coupled_lowrank_block(self, tape, n0)
        # The projection is recorded, and so is the copy below, which depends
        # on the projected value. On the quadrature path the solve block has
        # already projected inside its own map (`DirectConstitutiveBackend.
        # forward`), so this applies P a second time. P is idempotent up to
        # round-off, and its adjoint is applied twice as well, which is
        # correct. The second application is part of the forward map that the
        # GIA reference runs are compared against bit for bit.
        self.project_out_nullspace()
        # `solution_old` holds the accepted state when `solve` returns, on
        # every path: with or without a nullspace, and on the direct path,
        # which does not go through `StokesSolverBase.solve`. The next solve
        # does not need the copy, because `StokesSolverBase.solve` copies
        # `solution` into `solution_old` before it solves. The copy is here
        # for the code that reads `solution_old` between two steps: a
        # checkpoint, or a diagnostic that assembles the residual. It comes
        # after the projection, so it holds the projected state.
        self.solution_old.assign(self.solution)
        if self.layout.condensed:
            self.recover_internal_variables()
        if self.time_integrator:
            histories = (list(self.internal_variables) if isinstance(self.internal_variables, (list, tuple))
                         else [self.internal_variables]) if self.layout.condensed else [
                             self.solution.subfunctions[self.layout.internal_variable_field]]
            self.time_integrator.accept(histories)

    def restart_time_integration(self) -> None:
        """Request BE startup on the next step after a resolved forcing event."""
        if self.time_integrator:
            self.time_integrator.restart()

    def require_controls_reach_residual(self, *controls) -> None:
        """D1: raise if a control never enters the low-rank solve's residual.

        A control absent from the residual gives a silent `0.0` gradient with
        only a `WARNING:root:Adjoint value is None` on stderr. The solver cannot
        discover its own controls (pyadjoint does not mark a `Control` on the
        tape), so a caller that knows them passes them here after an annotated
        solve, and an unreachable one raises before any wrong number is produced
        (REVIEW-ADJOINT S6.5 D1).
        """
        if self.adjoint_block is None:
            raise RuntimeError(
                "require_controls_reach_residual needs an adopted low-rank "
                "adjoint block; run one annotated solve first.")
        from .dtn_coupled_adjoint import require_controls_reach_block
        require_controls_reach_block(self.adjoint_block, controls)

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
        if self._direct_backend:
            # Rigid-kernel projection leaves strain and material updates invariant.
            stored = self.internal_variables
            sources = self._direct_backend.buffers
            targets = list(stored) if isinstance(stored, (list, tuple)) else [stored]
            for target, source in zip(targets, sources):
                target.assign(source)
            return
        u = self.solution.subfunctions[self.layout.displacement]
        strain = self.approximation.deviatoric_strain(u)
        stored = self.internal_variables
        histories = self.time_integrator.combinations() if self.time_integrator else [stored]
        alpha = self.time_integrator.alpha if self.time_integrator else 1
        updates = [
            (m + self.dt / mt * strain) / (alpha + self.dt / mt)
            for m, mt in zip(history_slices(histories),
                             self.approximation.maxwell_times)]
        # Into a temporary of the same layout first: every stored field
        # appears on the right-hand side, and interpolating a function into
        # itself is not worth relying on. `assign_history_slices` handles the
        # combined field, a single `(d, d)` field and the list layout alike.
        if isinstance(stored, (list, tuple)):
            scratch = [Function(m.function_space()) for m in stored]
        else:
            scratch = Function(stored.function_space())
        assign_history_slices(scratch, updates)
        if isinstance(stored, (list, tuple)):
            for m, tmp in zip(stored, scratch):
                m.assign(tmp)
        else:
            stored.assign(scratch)
