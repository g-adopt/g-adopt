"""The sea-level equation on the coupled self-gravitating solver, 2-D annulus.

These are the acceptance tests of the sea-level equation. The design they check
is in `NOTES/DESIGN-SEA-LEVEL.md`, and the decisions behind it in
`NOTES/DECISIONS.md` and `NOTES/findings/FINDING-2d-sea-level-build.md`:

    SL      = SL_init + (N - N_init) - (u_r - ur_init) + Shift
    Delta   = SL - SL_init
    sigma   = S(I, SL) - S(I_init, SL_init)
    S(I, SL) = rho_w SL + rho_i eps softplus((I_+ - r SL) / eps)
    E_sl    = -c B_mu g_0 int_Re G(Delta) dS,   dG/dDelta = sigma

with `r = rho_w / rho_i`, `I_+` the ice clipped at zero and `eps = EPS` the
width of the rounded maximum of the water and ice columns
(`gadopt.sea_level_sheet`). The `Shift` row of the residual is mass
conservation, `int sigma dS = 0`, up to the frame column `lambda . dD/dShift`
that keeps the residual the gradient of one Lagrangian. At a converged state
the net sheet mass is therefore of order `lambda`, which is discretisation
error, and not zero to the solver tolerance
(`NOTES/findings/FINDING-2d-sea-level-build.md`, decision 2). The functions of
the sheet and the tests that need the rounding itself are in
`test_sea_level_sheet.py`.

The solver is `gadopt.gia_solver.SelfGravitatingGIASolver` with the couplings
of `gadopt.gia_couplings`, and the sea-level equation is its `SeaLevel`
coupling. The configuration is the centre-of-mass frame configuration of
`test_gia_gravity.py`: the coarse P2-curved annulus (dr 0.2, 32 azimuthal
cells), a `FluidCore`, the `CentreOfMass` multipliers and the reference gravity
of the 2-D Gauss law for the test's own density (`gravity_of_r`). Rotation is
off except where a test says otherwise. `build` returns the solver, its mixed
solution and the `SeaLevel` coupling, which holds the sea-level diagnostics
(`sea_level()`, `sheet()`, `measure()`, `energy()`, `shift()`).

## Four surface states

Most tests use one of four prescribed surface states. Three of them keep every
quadrature point far from the coastline and the grounding line, so that the
sheet is the sharp maximum there and the answer of a test can be written
without the rounding:

- `"ocean"`: a deep uniform ocean, `SL_init = DEEP`, no ice.
- `"cap"`: the same ocean with grounded ice on the quarter circle
  `0 < phi < pi/2`. The ice is a DG0 field, so its edge sits on mesh vertices
  and no quadrature point sees a transition. The ocean is then asymmetric in
  both `x` and `y`, which makes every frame coupling nonzero.
- `"caps"`: grounded ice on the two opposite quarter circles centred on
  `phi = 0` and `phi = pi`. The load has no degree-1 content, so the rigid
  translation of the frame does not enter the uplift.
- `"shoreline"`: a smooth sea level `cos(phi - PHASE) + 0.2` with land and a
  melting ice sheet on it. The coast lies inside facets, so this state tests
  the live coastline and the rounding.

`DEEP = 25` and `GROUNDED = 100` give `|I_+ - r SL|` of at least 26.9 at every
quadrature point of the saturated states (73 under the ice, `-26.9` in the
open ocean), which is 538 widths `EPS`. The rounding is `exp(-538)` there, zero
in double precision.

## Two DtN representations

The tests that assert physics run on both DtN representations through the
`representation` fixture: the multiplier path with the direct preset, and the
low-rank path with the iterative preset, letting the preset choose block 1
itself (`solve_settings`), which at this width is the cached apply of
`gadopt.DtNTwoBlockSchurPC` under `schur_fact_type full`. `TestRepresentationParity` compares the two
converged shoreline states. The tests of `TestLayout` that check the position
of `Shift` and of the frame multipliers in the solver's mixed space run on both
representations, because the two spaces differ in their multiplier fields. The
other structural tests (sheet, transposes, the energy) read `Real` rows of the
multiplier space and stay on that path.

## Properties checked by the GIA cases

The eustatic limit, the fixed-coastline limits (eustatic shift, saturated
coastline, ice over the ocean) and the repeated solve are rows of the `sea`
entry of the `limits` case in `tests/gia` (driver `gia_limits.py`). The
transposes of the sheet blocks, the mechanics row against a normal stress
and the nonzero shift-rotation block are rows of the `sea-transposes` entry
of the `operators` case (`gia_operators.py`). The rotating solve on both
representations is the `sea-multiplier` and `sea-iterative` entries of the
`forward` case. These drivers import the builders and constants of this
module.

## What these tests are not

They are 2-D structural and consistency checks. Nothing here is a benchmark:
Martinec et al. (2018) is 3-D and out of scope, and so are Will Scott's
two-disc moving-shoreline case and Irksome.
"""

import importlib

import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import firedrake as fd
import numpy as np
import pytest
import ufl
from ufl.algorithms import expand_derivatives
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from pyadjoint.tape import annotate_tape

from gadopt.gia_couplings import (
    CentreOfMass,
    FluidCore,
    Rotation,
    SeaLevel,
    SurfaceLoad,
)
from gadopt.gia_presets import (
    OMEGA_SQ_EARTH,
    selfgrav_dtn_iterative_solver_parameters,
)
from gadopt.gia_solver import SelfGravitatingGIASolver
from gadopt.utility import scalar_value
import test_gia_gravity
from test_gia_gravity import (  # noqa: E402  (module-level fixture and helpers)
    B_MU,
    CURVE_OUTER,
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    RC,
    RE,
    approximation,
    gravity_bcs,
)
# The module fixture of `test_gia_gravity`, bound by assignment so that pytest finds
# it here and flake8 does not read the test arguments of the same name as
# redefinitions of an import.
meshes = test_gia_gravity.meshes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Water and ice densities in the units of the test's mantle density, which is
#: 1. The mantle stands for the Spada upper mantle, 4978 kg/m^3, and the water
#: and ice densities are 1000 and 931 kg/m^3.
RHO_W = 1000.0 / 4978.0
RHO_I = 931.0 / 4978.0

#: The reference state of the centre-of-mass frame tests of
#: `test_gia_gravity.py`: a core of density `RHO_CORE` inside Rc and a
#: unit-density mantle.
RHO_CORE = 2.0


def gravity_of_r(r):
    """`g(r) = Lambda M(<r) / (2 pi r)`, the 2-D Gauss law of the reference state.

    With the constant `g = 1` of other tests a rigid translation costs energy
    and the frame multipliers would carry a real force. With this `g` the
    translation is a near-kernel, which is the configuration the frame exists
    for.

    Args:
      r: a radius, a number or UFL.

    Returns:
      The reference gravity at `r`, of the same kind as `r`.
    """
    mass = RHO_CORE * np.pi * RC ** 2 + np.pi * (r ** 2 - RC ** 2)
    return LAMBDA * mass / (2 * np.pi * r)


#: The surface gravity of the reference state, about 1.19 in the
#: non-dimensional units of the solver.
G_RE = gravity_of_r(RE)

#: The name of the `Shift` field of the mixed space (`solver.field_index`).
SHIFT = "sea_level_shift"

#: A deep uniform ocean and a thick grounded ice sheet, in units of the mantle
#: thickness D. The values are not physical. They keep every quadrature point
#: far from the rounding of the sheet, see the module docstring.
DEEP = 25.0
GROUNDED = 100.0

#: The width `eps` of the softplus sheet, in units of D like every length of
#: the test. The sea level of the shoreline state is of order 1 and the facets
#: of the coarse annulus are about 0.2 long, so with a shoreline slope of order
#: 1 the band where the second derivative of the sheet is nonzero, about
#: `eps / (r s)`, is a quarter of a facet wide.
EPS = 0.05

#: Rotation of the smooth shoreline state, so that it has no mirror symmetry in
#: `x` or `y` and every first-moment integral of the ocean is nonzero.
PHASE = 0.5

#: Fraction of the ice sheet that melts in the shoreline state.
MELT = 0.1

#: Tight solver tolerances for tests that read a converged state. `ksp_rtol`
#: is the outer FGMRES of the direct preset. `snes_atol` is set explicitly so that the
#: mass-conservation threshold below is computed from known numbers.
SNES_RTOL = 1e-10
SNES_ATOL = 1e-15
TIGHT = {"snes_rtol": SNES_RTOL, "snes_atol": SNES_ATOL, "ksp_rtol": 1e-12}

#: The two DtN representations that the physics tests run on.
REPRESENTATIONS = ("multiplier", "lowrank")

#: An Earth with an elastic step (viscosity 1e18, so the Maxwell time is far
#: above `dt = 1`) with shear and bulk modulus 30.
#: Chosen so that the ocean means of the geoid change and of the uplift are
#: both a large part of the non-eustatic remainder of the shift. Measured on
#: the current solver with the fixed-shoreline load `rho_i dI (2 caps - 1)`
#: given as a sheet: `|mean_dN| / |rest| = 0.59`, `|mean_ur| / |rest| = 0.41`,
#: `|rest| / |eustatic| = 0.086`. The default Earth (viscosity 1, modulus 1,
#: `dt = 1`) gave 0.058 for the geoid, below the guard of 0.1; modulus 10 gave
#: 0.35 and 0.65, modulus 100 gave 0.82 and 0.18.
ELASTIC_EARTH = dict(shear_modulus=30.0, bulk_modulus=30.0, viscosity=1e18)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_tape():
    """A fresh tape for every test, and annotation off afterwards.

    The Taylor tests annotate. A test that runs after them must not find a
    tape that still records, because every assemble would then be taped.
    """
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


@pytest.fixture(scope="module", params=REPRESENTATIONS)
def representation(request):
    """The DtN representation of one physics test: `"multiplier"` or `"lowrank"`.

    Module scope, so that the module-scoped `shoreline_solved` can depend on
    it and solves the shoreline state once per representation.
    """
    return request.param


def solve_settings(representation):
    """The solver keywords for a converged solve on one representation.

    The multiplier path uses the direct preset with `TIGHT`, which needs two
    outer iterations per Newton step there.

    The low-rank path uses the iterative preset (`solver_parameters=
    "iterative"`, which on the DG family is the full layout) and names no
    block-1 preconditioner, so the preset chooses one. The solver supplies
    the `Real` width, `condensed` and the representation to the preset; the
    test gives the tolerances through `preset_options`. At the width of this
    `Real` block that choice is the cached apply of
    `gadopt.DtNTwoBlockSchurPC` under `schur_fact_type full`, which is the
    configuration the Martinec driver runs in production. Naming a class here
    would pin the test to a route production no longer takes, and it would
    hide a change of the library default; leaving the key out lets the test
    see whatever that default becomes.

    The direct preset also converges on the low-rank path, but its LU
    factorisation of block 0 does not contain the low-rank update `B`, so it
    needs 7 outer iterations per Newton step against 2 on the multiplier path.

    The delegating path under `schur_fact_type lower` does not converge here.
    On 2026-09-21, after the rebase onto `bd414346`,
    `TestFixedOcean::test_the_taylor_rate_with_ice_thickness_control` on the
    low-rank arm ran the adjoint solve to the outer cap of 200 iterations
    without converging, while the forward solve of the same configuration
    converged. The cause is not settled.
    `NOTES/team/rebase-2026-09-21/10-REVIEW.md` holds the hypothesis and the
    experiment that would test it. The earlier measurement of that path, 147
    block-0 applications for the shoreline solve against 233 with `pc_type
    none` (`NOTES/PLAN-LOWRANK-SEA-LEVEL.md` section 1), was made before the
    preset defaults moved and does not describe what runs today.

    The tolerances are those of `TIGHT`: the Newton tolerances are the same
    numbers, so the thresholds that the tests compute from `SNES_RTOL` and
    `SNES_ATOL` hold on both paths. The outer FGMRES stops at a relative
    residual of 1e-12, as the outer Krylov solve of the direct preset. Block 0
    is solved to 1e-4 only, because the outer FGMRES corrects an inexact block
    0 and a tighter inner tolerance costs applications without changing the
    converged state.

    Args:
      representation: `"multiplier"` or `"lowrank"`.

    Returns:
      A new dictionary of `SelfGravitatingGIASolver` keywords:
      `solver_parameters`, `solver_parameters_extra` and, on the low-rank
      path, `preset_options`. Every call returns fresh dictionaries, so a
      caller can add an option to `solver_parameters_extra` (for example
      `snes_type`) without changing another test's settings.
    """
    if representation == "multiplier":
        return dict(solver_parameters="direct",
                    solver_parameters_extra=dict(TIGHT))
    return dict(
        solver_parameters="iterative",
        preset_options=dict(outer_rtol=TIGHT["ksp_rtol"], block0_rtol=1e-4,
                            snes_rtol=SNES_RTOL),
        # The preset sets `snes_atol` itself. It is set here again from the
        # module constant, so that the thresholds of the tests read the value
        # the solver uses.
        solver_parameters_extra={"snes_atol": SNES_ATOL})


def sheet():
    """The sea-level sheet module, imported at call time and not at module level.

    A module-level import of a module that does not exist stops collection of
    the whole file. At call time each test fails on its own.
    """
    return importlib.import_module("gadopt.sea_level_sheet")


def polar_angle(mesh):
    """`phi = atan2(y, x)` as UFL on `mesh`."""
    X = fd.SpatialCoordinate(mesh)
    return fd.atan2(X[1], X[0])


def indicator(mesh, which):
    """A DG0 0/1 field of an angular sector, evaluated at cell centroids.

    Every sector boundary is at a multiple of `pi/4`, and the Re vertices of
    this mesh are at multiples of `pi/16` (measured), so the edge of a sector
    is always a mesh vertex and a facet is either fully inside or fully
    outside. A cell centroid is never on a sector boundary.

    Args:
      mesh: the mechanics submesh or a circle manifold mesh.
      which: `"cap"` for `0 < phi < pi/2`, `"caps"` for `|cos phi| > cos(pi/4)`,
        `"near"` for the ocean facets next to the caps
        (`cos(5 pi/16) < |cos phi| < cos(pi/4)`), `"far"` for the ocean facets
        farthest from the caps (`|cos phi| < cos(7 pi/16)`).

    Returns:
      A DG0 `Function` on `mesh`.
    """
    phi = polar_angle(mesh)
    c = fd.cos(phi)
    if which == "cap":
        expr = fd.conditional(fd.cos(phi - np.pi / 4) > np.cos(np.pi / 4),
                              1.0, 0.0)
    elif which == "caps":
        expr = fd.conditional(abs(c) > np.cos(np.pi / 4), 1.0, 0.0)
    elif which == "near":
        expr = fd.conditional(
            fd.And(abs(c) < np.cos(np.pi / 4), abs(c) > np.cos(5 * np.pi / 16)),
            1.0, 0.0)
    elif which == "far":
        expr = fd.conditional(abs(c) < np.cos(7 * np.pi / 16), 1.0, 0.0)
    else:
        raise ValueError(which)
    return fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(expr)


def surface_fields(sub, state, *, d_ice=0.0, sl_init=None):
    """The prescribed fields of one surface state, as `SeaLevel` keywords.

    Args:
      sub: the mechanics submesh.
      state: `"ocean"`, `"cap"`, `"caps"`, `"shoreline"` or `"land"` (a
        uniform `SL_init = -DEEP` with no ice).
      d_ice: the ice thickness change on the grounded sectors of `"cap"` and
        `"caps"`, in units of D. Positive means ice growth.
      sl_init: a constant that replaces `SL_init` of the `"ocean"` state.

    Returns:
      `dict(SL_init=..., I=..., I_init=..., N_init=..., ur_init=...)`, every
      entry a `Function` on `sub`. `N_init` and `ur_init` are zero: the run
      starts undeformed.
    """
    V = fd.FunctionSpace(sub, "CG", 2)
    phi = polar_angle(sub)
    zero = fd.Function(V)
    if state in ("ocean", "land"):
        value = -DEEP if state == "land" else (DEEP if sl_init is None
                                               else sl_init)
        SL_init = fd.Function(V).assign(value)
        I_init = fd.Function(V)
        I = fd.Function(V)
    elif state in ("cap", "caps"):
        ice = indicator(sub, state)
        SL_init = fd.Function(V).assign(DEEP)
        I_init = fd.Function(ice.function_space()).interpolate(GROUNDED * ice)
        I = fd.Function(ice.function_space()).interpolate(
            (GROUNDED + d_ice) * ice)
    elif state == "shoreline":
        # Ocean where cos(phi - PHASE) > -0.2, land around phi = PHASE + pi,
        # and an ice sheet on the land that is 0.36 thick at its centre. The
        # coast lies inside facets, so the quadrature samples the rounding of
        # the sheet there, which is the point of this state.
        SL_init = fd.Function(V).interpolate(fd.cos(phi - PHASE) + 0.2)
        I_init = fd.Function(V).interpolate(
            0.6 * fd.max_value(0.0, -fd.cos(phi - PHASE) - 0.4))
        I = fd.Function(V).interpolate((1.0 - MELT) * I_init)
    else:
        raise ValueError(state)
    return dict(SL_init=SL_init, I=I, I_init=I_init, N_init=zero.copy(True),
                ur_init=zero.copy(True))


def make_sea_level(fields, **overrides):
    """A `SeaLevel` coupling on the Re tag with the test's densities and gravity.

    Args:
      fields: `surface_fields(...)` or `coastline_fields(...)`.
      overrides: extra or replaced `SeaLevel` keywords, for example
        `fixed_ocean` or `softplus_width`.

    Returns:
      An unbound `gadopt.gia_couplings.SeaLevel`.
    """
    settings = dict(boundary=CURVE_RE, rho_w=RHO_W, rho_i=RHO_I,
                    g_surface=G_RE, softplus_width=EPS)
    settings.update(fields)
    settings.update(overrides)
    return SeaLevel(**settings)


def unit_time_step(mesh):
    """`dt = 1` as a `Real` Function on `mesh`, so a test can change it with `assign`."""
    return fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(1.0)


def build(meshes, fields=None, *, rotation=False, lam=LAMBDA, stiff=False,
          earth=None, sea_level=True, surface_bcs=None, surface_load=None,
          quad_degree=None, sea_level_overrides=None,
          representation="multiplier", centre_of_mass=True, **solver_kwargs):
    """The centre-of-mass frame configuration with sea level, on the new solver.

    The mechanics live on the mantle submesh and the potential on the parent;
    the couplings are a `FluidCore` that carries the reference gravity at Rc,
    the `CentreOfMass` multipliers, and on request `Rotation`, `SeaLevel` and
    a prescribed `SurfaceLoad` on Re. No nullspace is declared: with rotation
    off the rigid rotation of the mantle is fixed only by the facet geometry
    error, which `remove_rigid_rotation` accounts for where two solves are
    compared.

    Args:
      meshes: the module fixture.
      fields: `surface_fields(...)`; required when `sea_level` is true.
      rotation: carry the rotation closure, with the polar moment of the
        unit-density mantle and the Earth's `Omega_sq`.
      lam: `Lambda` of the approximation. The reference gravity `g(r)` keeps
        the value of the frame tests, so a small `lam` removes
        self-attraction and keeps the load weight.
      stiff: shear and bulk modulus 1e6 and a viscosity of 1e18, so the
        Maxwell time is 1e12 against `dt = 1` and the step is elastic. The
        uplift then scales as 1e-6 of the default Earth.
      earth: extra keywords for the approximation (moduli, viscosity), applied
        after `stiff`. `None` keeps the default Earth.
      sea_level: whether the solver carries a `SeaLevel` coupling.
      surface_bcs: mechanics `bcs`; empty by default because the ice load is
        part of the sea-level sheet.
      surface_load: a prescribed surface density on Re (UFL on the mechanics
        mesh), given as `SurfaceLoad(g_surface=G_RE)`, or `None`. It loads
        the momentum equation with `B_mu G_RE sigma` and the potential with
        the sheet `sigma`.
      quad_degree: the boundary quadrature degree (`dtn_quad_degree`), or
        `None` for the DtN form's default.
      sea_level_overrides: extra `SeaLevel` keywords.
      representation: the DtN representation. `"multiplier"` by default, so
        the structural tests that read the `Real` rows of the multiplier space
        name nothing. The physics tests pass the `representation` fixture.
        `None` leaves the solver's own default.
      centre_of_mass: carry the `CentreOfMass` coupling. `False` only for the
        refusal of a sea level without the frame.
      solver_kwargs: passed to `SelfGravitatingGIASolver`.

    Returns:
      `(solver, z, sea)`: the solver, its mixed solution `solver.solution`,
      and the `SeaLevel` coupling (`None` without sea level).
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")

    Xm = fd.SpatialCoordinate(sub)
    rm = fd.sqrt(fd.dot(Xm, Xm))
    approximation_kwargs = dict(g=gravity_of_r(rm), self_gravity_number=lam)
    if stiff:
        approximation_kwargs.update(shear_modulus=1e6, bulk_modulus=1e6,
                                    viscosity=1e18)
    if earth is not None:
        approximation_kwargs.update(earth)

    # The core boundary carries the reference gravity at Rc, as in the frame
    # tests of `test_gia_gravity.py`.
    couplings = [FluidCore(boundary=CURVE_RC, rho_core=RHO_CORE,
                           g=fd.Constant(gravity_of_r(RC)))]
    if centre_of_mass:
        couplings.append(CentreOfMass())
    if rotation:
        # The polar moment `C = int r^2 dx` of the unit-density mantle.
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        couplings.append(Rotation(C=fd.assemble(fd.dot(Xm, Xm) * dx_m),
                                  Omega_sq=OMEGA_SQ_EARTH))
    sea = None
    if sea_level:
        sea = make_sea_level(fields, **(sea_level_overrides or {}))
        couplings.append(sea)
    if surface_load is not None:
        couplings.append(SurfaceLoad(boundary=CURVE_RE, sigma=surface_load,
                                     g_surface=G_RE))
    if representation is not None:
        solver_kwargs["dtn_representation"] = representation
    if quad_degree is not None:
        solver_kwargs["dtn_quad_degree"] = quad_degree
    # The sea level and the frame are shared code: the history family is the
    # Quadrature family unless the caller names another.
    solver_kwargs.setdefault("internal_variable_family", "Quadrature")
    solver = SelfGravitatingGIASolver(
        u, psi, approximation(**approximation_kwargs), dt=unit_time_step(sub),
        gravity_bcs=gravity_bcs(parent, sheet=False), bcs=surface_bcs or {},
        couplings=couplings, **solver_kwargs)
    return solver, solver.solution, sea


def coupling(solver, cls):
    """The one coupling of class `cls` that `solver` carries."""
    found = [c for c in solver.couplings if isinstance(c, cls)]
    assert len(found) == 1, f"{solver} carries {len(found)} {cls.__name__}"
    return found[0]


def re_measure(solver):
    """The solver's sea-level measure, already restricted to the Re tag."""
    return coupling(solver, SeaLevel).measure()


def real_value(field):
    """The one number of a `Real` Function, for example `sea.shift()`.

    Collective: `float` gathers the value from the rank that owns it, so
    every rank must call it. On the pyop3 Firedrake only the owner holds
    the entry in `dat.data_ro`; the other ranks hold an empty array.
    """
    return float(field)


def perturb(z, amplitude, seed):
    """Fills every sub-field of `z` with scaled standard normal values."""
    rng = np.random.default_rng(seed)
    for sub_z in z.subfunctions:
        sub_z.dat.data[:] = amplitude * rng.standard_normal(
            sub_z.dat.data.shape)


def dense_block(A, i, j):
    """A dense copy of block `(i, j)` of a nest matrix, `None` if absent."""
    block = A.getNestSubMatrix(i, j)
    if block is None:
        return None
    return block.convert("dense").getDenseArray()


def vector_norm(cofunction):
    """The l2 norm of an assembled mixed residual."""
    with cofunction.dat.vec_ro as v:
        return v.norm()


def solver_converged(solver):
    """Whether the last nonlinear solve reported convergence."""
    return solver.solver.snes.getConvergedReason() > 0


# ---------------------------------------------------------------------------
# The mixed space and refusals
# ---------------------------------------------------------------------------

_LAYOUT_SOLVERS = {}


def layout_solver(meshes, *, rotation=False, representation="multiplier"):
    """An unsolved sea-level solver on the ocean state, cached per configuration.

    The tests of `TestLayout` read the mixed space of the solver and nothing
    else, so one solver per `(rotation, representation)` serves all of them.

    Args:
      meshes: the module fixture.
      rotation: carry the rotation closure.
      representation: the DtN representation, or `None` for the solver's
        default.

    Returns:
      `(solver, sea)`.
    """
    key = (id(meshes[0]), rotation, representation)
    if key not in _LAYOUT_SOLVERS:
        solver, _, sea = build(meshes, surface_fields(meshes[1], "ocean"),
                               rotation=rotation,
                               representation=representation)
        _LAYOUT_SOLVERS[key] = (solver, sea)
    return _LAYOUT_SOLVERS[key]


def real_indices_of_space(solver):
    """The indices of the `Real` sub-spaces of the solver's mixed space, in order."""
    return tuple(i for i, V in enumerate(solver.solution_space)
                 if V.ufl_element().family() == "Real")


class TestLayout:
    """Where `Shift` and the frame multipliers sit in the solver's mixed space.

    `DtNTwoBlockSchurPC` needs the `Real` fields contiguous and last. Their
    order is fixed by the couplings' `order`: the core pressure, the rotation
    scalar, the frame multipliers, then `Shift`.

    The first four tests name the representation explicitly and run on both.
    The multiplier space carries the DtN multiplier fields in front of the
    frame multipliers and `Shift`, and the position of `Shift` must hold there
    too.
    """

    @pytest.mark.parametrize("dtn_representation", REPRESENTATIONS)
    @pytest.mark.parametrize("rotation", [False, True])
    def test_shift_is_the_last_field(self, meshes, rotation,
                                     dtn_representation):
        """`Shift` is last in the space, after the centre-of-mass multipliers."""
        solver, _ = layout_solver(meshes, rotation=rotation,
                                  representation=dtn_representation)
        assert solver.dtn_representation == dtn_representation
        assert solver.field_index[SHIFT] == len(solver.solution_space) - 1

    @pytest.mark.parametrize("dtn_representation", REPRESENTATIONS)
    @pytest.mark.parametrize("rotation", [False, True])
    def test_the_frame_multipliers_stay_just_before_the_shift(
            self, meshes, rotation, dtn_representation):
        """The two frame multipliers are the two fields just before `Shift`."""
        solver, _ = layout_solver(meshes, rotation=rotation,
                                  representation=dtn_representation)
        n = len(solver.solution_space)
        assert solver.dtn_representation == dtn_representation
        assert (solver.field_index["centre_of_mass_0"],
                solver.field_index["centre_of_mass_1"]) == (n - 3, n - 2)

    @pytest.mark.parametrize("dtn_representation", REPRESENTATIONS)
    @pytest.mark.parametrize("rotation", [False, True])
    def test_shift_is_a_real_field_in_the_accounting(self, meshes, rotation,
                                                     dtn_representation):
        """`field_names` and `real_indices` count the new field, in space order."""
        solver, _ = layout_solver(meshes, rotation=rotation,
                                  representation=dtn_representation)
        assert solver.dtn_representation == dtn_representation
        real = real_indices_of_space(solver)
        assert len(solver.solution_space) == len(solver.field_names)
        assert real == solver.real_indices
        assert real[-1] == solver.field_index[SHIFT]

    @pytest.mark.parametrize("dtn_representation", REPRESENTATIONS)
    def test_shift_lives_on_the_parent(self, meshes, dtn_representation):
        """Like every other `Real` field, so its rows assemble on parent measures."""
        solver, _ = layout_solver(meshes, representation=dtn_representation)
        assert solver.dtn_representation == dtn_representation
        space = solver.solution_space[solver.field_index[SHIFT]]
        assert space.mesh() is meshes[0]

    def test_absent_by_default(self, meshes):
        """A solver without a `SeaLevel` coupling has no `Shift` field."""
        solver, _, sea = build(meshes, sea_level=False)
        assert sea is None
        assert SHIFT not in solver.field_index

    def test_refuses_sea_level_without_the_frame(self, meshes):
        """An ocean load always has degree-1 content; the frame must fix it."""
        with pytest.raises(ValueError, match="CentreOfMass"):
            build(meshes, surface_fields(meshes[1], "ocean"),
                  centre_of_mass=False)

    @pytest.mark.parametrize("rotation", [False, True])
    def test_the_lowrank_space(self, meshes, rotation):
        """The low-rank space: no DtN multipliers, and the `Real` run contiguous and last.

        The low-rank path eliminates the DtN multipliers into a low-rank
        update of the potential block, so the space has none. The `Real`
        fields are then the core pressure, the rotation scalar if present, the
        two frame multipliers and `Shift`, in that order, contiguous and last.
        `DtNTwoBlockSchurPC` requires that run contiguous and last.
        """
        solver, _ = layout_solver(meshes, rotation=rotation,
                                  representation="lowrank")
        assert solver.dtn_representation == "lowrank"
        assert solver.multiplier_indices == ()
        names = (["core_pressure"] + (["m3"] if rotation else [])
                 + ["centre_of_mass_0", "centre_of_mass_1", SHIFT])
        real = real_indices_of_space(solver)
        assert real == tuple(solver.field_index[name] for name in names)
        assert real == tuple(range(real[0], len(solver.solution_space)))
        assert real == solver.real_indices
        assert solver.field_index[SHIFT] == len(solver.solution_space) - 1

    def test_sea_level_resolves_to_the_lowrank_representation(self, meshes):
        """A sea-level solver that names no representation is on the low-rank path.

        Sia's decision of 2026-09-17 (`NOTES/DECISIONS.md`): the 3-D sea-level
        runs use the low-rank path, because the multiplier path does not fit
        one node at a truncation of 20.
        """
        solver, _ = layout_solver(meshes, representation=None)
        assert solver.dtn_representation == "lowrank"
        assert solver.multiplier_indices == ()


class TestRefusals:
    """The ice load is part of the sea-level sheet and must not be given twice."""

    def test_refuses_a_normal_stress_on_re(self, meshes):
        _, sub = meshes
        with pytest.raises(ValueError, match=r"(?i)sea.level"):
            build(meshes, surface_fields(sub, "ocean"),
                  surface_bcs={CURVE_RE: {"normal_stress": B_MU * 1e-3}})

    def test_refuses_a_surface_load_on_re(self, meshes):
        """A prescribed load sheet on Re would be a second copy of the ice load."""
        _, sub = meshes
        with pytest.raises(ValueError, match=r"(?i)sea.level"):
            build(meshes, surface_fields(sub, "ocean"),
                  surface_load=fd.Constant(1e-3))

    def test_refuses_a_boundary_quadrature_degree_below_2p(self, meshes):
        """Degree 3 is below `2 p = 4` for the P2 displacement and potential.

        The sheet contains polynomials of degree `p` and their products, which
        the surface quadrature must integrate. `dtn_quad_degree` sets the
        boundary degree that the sea-level measure uses.
        """
        _, sub = meshes
        with pytest.raises((AssertionError, ValueError), match=r"(?i)degree"):
            build(meshes, surface_fields(sub, "ocean"), quad_degree=3)


class TestSeaLevelDataclass:

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_a_width_that_is_not_positive_is_refused(self, meshes, width):
        _, sub = meshes
        with pytest.raises(ValueError, match="must be positive"):
            make_sea_level(surface_fields(sub, "ocean"),
                           softplus_width=width)

    def test_the_width_is_required(self, meshes):
        _, sub = meshes
        fields = surface_fields(sub, "ocean")
        with pytest.raises(TypeError, match="softplus_width"):
            SeaLevel(boundary=CURVE_RE, rho_w=RHO_W, rho_i=RHO_I,
                     g_surface=G_RE, **fields)

    def test_a_width_given_as_a_constant_is_accepted(self, meshes):
        _, sub = meshes
        sea_level = make_sea_level(surface_fields(sub, "ocean"),
                                   softplus_width=fd.Constant(EPS))
        assert float(sea_level.softplus_width) == EPS


# ---------------------------------------------------------------------------
# The measure, the sea level and the sheet
# ---------------------------------------------------------------------------

class TestMeasure:

    def test_the_measure_is_the_re_circle(self, meshes):
        """`2 pi Re` to 1e-3: the P2-curved circle, on the facets of the sheet."""
        _, sub = meshes
        solver, _, _ = build(meshes, surface_fields(sub, "ocean"))
        length = fd.assemble(fd.Constant(1.0) * re_measure(solver))
        assert abs(length - 2 * np.pi * RE) < 1e-3 * 2 * np.pi * RE

    def test_the_measure_pairs_with_the_parent_facets(self, meshes):
        """A parent field integrates over the sea-level measure as on the parent.

        The facet-to-facet intersection is what makes this true. The parent's
        cell measure gives 26.42 against 33.62 for `x^2` over Re, with no
        warning (`CouplingContext.surface_measure` docstring). The reference is the same
        integral on the parent's own interior facets at the same degree.
        Measured on this mesh at degree 10, the two agree to 0.0, so the
        tolerance is round-off, 1e-12 relative.
        """
        parent, sub = meshes
        solver, _, _ = build(meshes, surface_fields(sub, "ocean"))
        dss = re_measure(solver)
        degree = dss.metadata()["quadrature_degree"]
        X = fd.SpatialCoordinate(parent)
        x2 = fd.Function(fd.FunctionSpace(parent, "CG", 2)).interpolate(
            X[0] ** 2)
        got = fd.assemble(fd.avg(x2) * dss)
        want = fd.assemble(fd.avg(x2) * fd.dS(CURVE_RE, domain=parent,
                                              degree=degree))
        assert got == pytest.approx(want, rel=1e-12)

    def test_the_measure_degree_is_at_least_2p(self, meshes):
        _, sub = meshes
        solver, _, _ = build(meshes, surface_fields(sub, "ocean"))
        assert re_measure(solver).metadata()["quadrature_degree"] >= 4


class TestSeaLevelExpression:
    """`sea_level() = SL_init + (N - N_init) - (u_r - ur_init) + Shift`.

    One unknown or initial field is set at a time, and the mean of
    `sea_level()` over Re is compared with the mean of the term that field
    should produce. Each term carries a different sign or factor, so a wrong
    sign on any one of them fails its own case. The surface gravity is set to
    2, different from the reference gravity at Re, so that the geoid term also
    checks that `SL` divides by `g_surface` and not by the approximation's `g`.
    """

    G_SURFACE = 2.0

    @pytest.mark.parametrize("which", ["SL_init", "psi", "u", "Shift",
                                       "N_init", "ur_init"])
    def test_each_term_enters_with_its_sign(self, meshes, which):
        _, sub = meshes
        fields = surface_fields(sub, "ocean", sl_init=0.0)
        solver, z, sea = build(
            meshes, fields, sea_level_overrides={"g_surface": self.G_SURFACE})
        dss = re_measure(solver)
        n = fd.FacetNormal(sub)
        Xm = fd.SpatialCoordinate(sub)
        if which == "SL_init":
            fields["SL_init"].assign(0.3)
            expected = fd.Constant(0.3)
        elif which == "psi":
            # N = psi / g_surface with rotation off.
            z.subfunctions[solver.potential_index].assign(0.2)
            expected = fd.Constant(0.2 / self.G_SURFACE)
        elif which == "u":
            u = z.subfunctions[solver.field_index["displacement"]]
            u.interpolate(0.1 * Xm / fd.sqrt(fd.dot(Xm, Xm)))
            expected = -fd.dot(u, n)
        elif which == "Shift":
            sea.shift().assign(0.05)
            expected = fd.Constant(0.05)
        elif which == "N_init":
            fields["N_init"].assign(0.07)
            expected = fd.Constant(-0.07)
        else:
            fields["ur_init"].assign(0.04)
            expected = fd.Constant(0.04)
        got = fd.assemble(sea.sea_level() * dss)
        want = fd.assemble(expected * dss)
        # 1e-12 relative: each side is an assembled integral of the same
        # field on the same measure.
        assert got == pytest.approx(want, rel=1e-12)

    def test_the_rotational_geoid_enters_with_a_plus_sign(self, meshes):
        """With rotation on, `N = (psi + psi_rot) / g_surface`, as `geoid()` says.

        Handover A defines the geoid change with `N = geoid()`, rotation
        included, and the sheet enters `inertia_form`, so the rotation row
        depends on `Delta`. The sea level must then depend on the rotation
        scalar through `+psi_rot / g_surface`, or the geoid is incomplete and
        the `(Shift, m3)` and `(psi, m3)` blocks are not transposes.

        Only the rotation scalar `m3` is set; every other unknown and initial
        field is zero, so `sea_level()` is the rotational geoid alone. The
        value `m3 = 10` gives `psi_rot = Omega_sq m3 r^2`, about
        `1.566e-3 * 10 * 4.86 = 0.076` at Re with the default `Omega_sq`, and
        `0.038` after the division by `g_surface = 2`. Tolerance 1e-12
        relative, as for the other terms: the same field on the same measure.
        A missing term gives 0 and a flipped sign gives a relative error of 2.
        """
        _, sub = meshes
        fields = surface_fields(sub, "ocean", sl_init=0.0)
        solver, z, sea = build(
            meshes, fields, rotation=True,
            sea_level_overrides={"g_surface": self.G_SURFACE})
        coupling(solver, Rotation).values()["m3"].assign(10.0)
        dss = re_measure(solver)
        # The rotational potential that the couplings share, on the facets of
        # the sheet: `m3` is a `Real` field of the parent, restricted there.
        expected = (solver.coupling_context.rotational_potential(
            sub, facet=True) / self.G_SURFACE)
        got = fd.assemble(sea.sea_level() * dss)
        want = fd.assemble(expected * dss)
        # Guard: the reference is not zero, so the comparison is not vacuous.
        assert abs(want) > 1e-3 * 2 * np.pi * RE
        assert got == pytest.approx(want, rel=1e-12)


class TestSurfaceLoadSheet:
    """`SeaLevel.sheet()` in saturated states, where it has a closed form."""

    S = 1e-2

    @pytest.mark.parametrize("state, density_per_unit", [
        # Open ocean, raised by Shift: a water sheet rho_w * Shift.
        ("ocean", RHO_W),
        # Land (C = 0): no water load whatever Shift is.
        ("land", 0.0),
    ])
    def test_the_water_sheet(self, meshes, state, density_per_unit):
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, state))
        sea.shift().assign(self.S)
        dss = re_measure(solver)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(sea.sheet() * dss)
        # Absolute 1e-12 of the water sheet scale: the DEEP = 25 datum cancels
        # in `B C SL - B_init C_init SL_init`, which costs 25 * 1e-16 relative
        # to a sea-level change of 1e-2, that is 3e-13.
        assert abs(got - density_per_unit * self.S * length) \
            <= 1e-12 * RHO_W * self.S * length

    def test_floating_ice_counts_as_water(self, meshes):
        """Ice thinner than `(rho_w/rho_i) SL` floats, and only water loads."""
        _, sub = meshes
        fields = surface_fields(sub, "ocean")
        fields["I"].assign(10.0)
        fields["I_init"].assign(10.0)
        solver, z, sea = build(meshes, fields)
        sea.shift().assign(self.S)
        dss = re_measure(solver)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(sea.sheet() * dss)
        assert got == pytest.approx(RHO_W * self.S * length, rel=1e-10)

    def test_grounded_ice_loads_with_its_thickness_change(self, meshes):
        """Under grounded ice the sheet is `rho_i dI` and no water."""
        _, sub = meshes
        d_ice = 0.5
        fields = surface_fields(sub, "cap", d_ice=d_ice)
        solver, z, sea = build(meshes, fields)
        sea.shift().assign(self.S)
        dss = re_measure(solver)
        cap = indicator(sub, "cap")
        expected = fd.assemble(
            (RHO_I * d_ice * cap + RHO_W * self.S * (1 - cap)) * dss)
        got = fd.assemble(sea.sheet() * dss)
        assert got == pytest.approx(expected, rel=1e-10)

    def test_the_sheet_is_the_softplus_column_difference_at_a_shoreline(
            self, meshes):
        """The live sheet is `S(I, SL) - S(I_init, SL_init)`, pointwise.

        The expected sheet is built from `gadopt.sea_level_sheet` with the
        same width. A random pointwise weight makes the comparison pointwise
        instead of a single mean. 1e-11 relative: the two sides are the same
        algebra evaluated by two kernels.
        """
        _, sub = meshes
        m = sheet()
        fields = surface_fields(sub, "shoreline")
        solver, z, sea = build(meshes, fields)
        shift = 0.05
        sea.shift().assign(shift)
        SL0, I, I0 = fields["SL_init"], fields["I"], fields["I_init"]
        expected = (m.softplus_column(I, SL0 + shift, EPS, RHO_W, RHO_I)
                    - m.softplus_column(I0, SL0, EPS, RHO_W, RHO_I))
        weight = fd.Function(fd.FunctionSpace(sub, "CG", 2))
        weight.dat.data[:] = np.random.default_rng(7).uniform(
            0.5, 1.5, weight.dat.data.shape)
        dss = re_measure(solver)
        got = fd.assemble(weight * sea.sheet() * dss)
        want = fd.assemble(weight * expected * dss)
        assert got == pytest.approx(want, rel=1e-11)

    def test_the_sheet_integral_is_the_integrand_times_the_sheet(self, meshes):
        """The sheet the consumers integrate is `sheet()` on the sea-level measure.

        The rotation rows, the frame rows and the 2-D monopole datum read the
        sea level through the `MassSheet` it publishes in the solver's
        context. A weighted integral of that record must equal the same
        integral of `sheet()` on `measure()`, to round-off.
        """
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "shoreline"))
        sea.shift().assign(0.05)
        X = fd.SpatialCoordinate(sub)
        published = [record for record in solver.coupling_context.sheets
                     if record.name == "sea level"]
        assert len(published) == 1
        record = published[0]
        got = fd.assemble(X[0] ** 2 * record.density * record.measure)
        want = fd.assemble(X[0] ** 2 * sea.sheet()
                           * re_measure(solver))
        assert got == pytest.approx(want, rel=1e-12)


# ---------------------------------------------------------------------------
# The consumers of the sheet
# ---------------------------------------------------------------------------

class TestConsumers:
    """The sheet reaches the mass dipole, the inertia row and the net-mass scale.

    It does not reach the 2-D monopole datum (`NOTES/findings/FINDING-2d-sea-level-build.md`, decision 1). The
    datum is refreshed at the start of `solve`, so it would hold the net sheet
    mass of the previous state, and at a converged state the `Shift` row makes
    that net mass zero. The correct contribution is therefore zero.

    Each test sets `Shift` alone, with `u = 0`, so the mantle and core terms of
    each consumer are zero and the whole value is the sea-level sheet. The
    net-mass scale test is the exception and says why.
    """

    S = 1e-2

    def test_the_sheet_reaches_the_mass_dipole(self, meshes):
        """`D_i = int x_i sigma dS` from the sheet, in both directions.

        The `"cap"` state has an asymmetric ocean, so both moments are nonzero.
        """
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "cap"))
        sea.shift().assign(self.S)
        X = fd.SpatialCoordinate(sub)
        ocean = 1 - indicator(sub, "cap")
        dss = re_measure(solver)
        for i in range(2):
            expected = fd.assemble(RHO_W * self.S * ocean * X[i] * dss)
            got = fd.assemble(coupling(solver, CentreOfMass).dipole_form(
                solver.coupling_context, i))
            assert abs(expected) > 1e-3 * RHO_W * self.S
            assert got == pytest.approx(expected, rel=1e-10)

    def test_the_sheet_reaches_the_inertia_row(self, meshes):
        """`dI_33 = int sigma (x^2 + y^2) dS` from the sheet, rotation on."""
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "ocean"),
                               rotation=True)
        sea.shift().assign(self.S)
        X = fd.SpatialCoordinate(sub)
        expected = fd.assemble(RHO_W * self.S * (X[0] ** 2 + X[1] ** 2)
                               * re_measure(solver))
        got = fd.assemble(coupling(solver, Rotation).inertia_form(
            solver.coupling_context, 2))
        assert got == pytest.approx(expected, rel=1e-10)

    def test_the_sea_level_sheet_is_not_in_the_monopole_datum(self, meshes):
        """The 2-D enclosed mass does not contain `int sigma dS` of the sheet.

        The state has a large net sheet mass, `rho_w S L` with `L = 2 pi Re`,
        about 2.8e-2 in the test's units, because `Shift` was set by hand and
        not solved for. With `u = 0` the core sheet is zero and the gravity
        boundary conditions carry no sheet, so the only candidate source of
        enclosed mass is the sea-level sheet. The datum must be zero.

        Tolerance 1e-12 of `rho_w S L`: with the sheet left out, the datum is
        a one-by-one solve of an exactly zero right-hand side. With the sheet
        in, the datum equals `rho_w S L`, a relative value of 1.
        """
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "ocean"))
        sea.shift().assign(self.S)
        sheet_mass = fd.assemble(RHO_W * self.S * re_measure(solver))
        # Guard: the hand-set state really carries a net sheet mass, so a zero
        # datum is a statement about the datum and not about the state.
        assert fd.assemble(sea.sheet() * re_measure(solver)) \
            == pytest.approx(sheet_mass, rel=1e-10)
        solver.update_total_mass()
        assert abs(float(solver.gravity.source_mass)) <= 1e-12 * sheet_mass

    def test_the_sheet_sets_the_scale_of_the_net_mass_check(self, meshes):
        """`check_net_mass` reads a net mass against a scale that has the sheet.

        The net mass comes from the fluid-core sheet, which stays in the
        datum. A radial displacement `a (Re - r) / (Re - Rc) rhat` moves the
        core boundary out by `a` and leaves Re fixed, so the core sheet
        `rho_core u_r` has one sign and its net mass equals its own scale,
        `rho_core a 2 pi Rc`, about 1.5e-8 for `a = 1e-9`. Without the
        sea-level sheet in the scale, the relative value is 1, above the
        warning band `(1e-8, 1e-4)`, and nothing is reported.

        The sea-level sheet is the `"caps"` state with `d_ice = 1e-2` and the
        `Shift` that conserves mass. Its net mass is round-off (measured
        1.9e-14) and its scale `int |sigma| dS` is 2.6e-2. With it in the scale,
        the relative value is about 6e-7, inside the band, and the warning
        fires. The test does not depend on whether the datum contains the
        sea-level sheet, because that sheet has no net mass here.

        The two guards compute both relative values from the sheets, so that
        the configuration is known to separate the two cases before the
        warning is read.
        """
        _, sub = meshes
        d_ice = 1e-2
        solver, z, sea = build(meshes, surface_fields(sub, "caps",
                                                      d_ice=d_ice))
        dss = re_measure(solver)
        caps = indicator(sub, "caps")
        ice_length = fd.assemble(caps * dss)
        ocean_length = fd.assemble((1 - caps) * dss)
        # The uniform ocean rise that balances the ice growth on the caps.
        conserving = -RHO_I * d_ice * ice_length / (RHO_W * ocean_length)
        sea.shift().assign(conserving)
        # A CMB displacement of 1e-9 (units of D) that vanishes at Re, so the
        # sea-level sheet does not see it through `u_r`.
        X = fd.SpatialCoordinate(sub)
        r = fd.sqrt(fd.dot(X, X))
        z.subfunctions[solver.field_index["displacement"]].interpolate(
            1e-9 * (RE - r) / (RE - RC) * X / r)

        core = coupling(solver, FluidCore)
        core_measure = core.measure()
        core_sheet = core.sheet_density(solver.field("displacement"))
        core_net = abs(fd.assemble(core_sheet * core_measure))
        core_scale = fd.assemble(abs(core_sheet) * core_measure)
        sheet_scale = fd.assemble(abs(sea.sheet()) * dss)
        # Guard: without the sea-level scale, no warning (relative near 1).
        assert core_net / core_scale > 1e-4
        # Guard: with the sea-level scale, the relative value is in the band.
        relative = core_net / (core_scale + sheet_scale)
        assert 1e-8 < relative < 1e-4
        with pytest.warns(UserWarning, match="Net sheet mass"):
            solver.update_total_mass()


# ---------------------------------------------------------------------------
# The monopole datum is not lagged by the ocean load
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Signs against the existing terms
# ---------------------------------------------------------------------------

class TestSheetSigns:
    """The ocean sheet enters the rows as an existing sheet of the same density.

    The comparison solver has no sea level. For the potential row its load is
    a `SurfaceLoad` of density `rho_w S`, whose potential row is the
    convention of every prescribed sheet; for the mechanics row it is a plain
    `normal_stress` of `B_mu g(Re) rho_w S` in `bcs`, the convention of every
    load in `test_gia_gravity.py` (the solver accepts it with a warning). The
    sea-level solver has the uniform deep ocean and `Shift = S`, so its sheet
    is `rho_w S`. At `u = psi = 0` the residual rows of both are the load
    terms alone.

    This is the check that `NOTES/DESIGN-SEA-LEVEL.md` asks for: a flipped
    geoid sign in `sea_level()` flips the sign of the ocean sheet in the
    potential row against every other sheet, while all symmetry tests still
    pass.
    """

    S = 1e-2

    def test_the_potential_row_matches_a_prescribed_sheet(self, meshes):
        _, sub = meshes
        solver, _, sea = build(meshes, surface_fields(sub, "ocean"))
        sea.shift().assign(self.S)
        row_sea = fd.assemble(sea.residual(sea.context)).subfunctions[
            solver.potential_index].dat.data_ro

        plain, _, _ = build(meshes, sea_level=False,
                            surface_load=fd.Constant(RHO_W * self.S))
        load = coupling(plain, SurfaceLoad)
        row_plain = fd.assemble(load.residual(plain.coupling_context)) \
            .subfunctions[plain.potential_index].dat.data_ro
        scale = np.abs(row_plain).max()
        assert scale > 0.0
        # 1e-10 relative: the same facets, the same `avg` and the same degree.
        # The two spellings agreed to the last bit for the fluid core
        # (`CouplingContext.surface_measure`). The DEEP datum costs 3e-13.
        assert np.abs(row_sea - row_plain).max() <= 1e-10 * scale


# ---------------------------------------------------------------------------
# Symmetry and the energy
# ---------------------------------------------------------------------------

_JACOBIANS = {}


def jacobian(meshes, state, rotation=False):
    """The assembled nest Jacobian of the whole residual at a perturbed state.

    Cached per `(state, rotation)`, because one assembly takes about 7 s and
    the transpose tests read many blocks of it. With `rotation=True` the space
    carries the rotation scalar `m3`, and the perturbation also sets it.

    `"cap"`: every quadrature point far from the rounding, so the second
    derivative of the sheet vanishes and the Jacobian is that of the sharp
    maximum. `"shoreline"`: a live coastline. The sheet depends on
    `(u, psi, Shift)` only through the one scalar `Delta`, so the exact
    Jacobian is still symmetric (`NOTES/DESIGN-SEA-LEVEL.md`).

    The perturbation is 1e-3, so the shoreline coast stays inside its facets
    and the saturated states stay saturated.

    Returns:
      `(A, solver)`: the nest matrix and the solver whose `field_index` names
      its blocks.
    """
    key = (state, rotation)
    if key not in _JACOBIANS:
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, state,
                                                      d_ice=0.5),
                               rotation=rotation)
        perturb(z, 1e-3, seed=11)
        A = fd.assemble(fd.derivative(solver.F, z), mat_type="nest").petscmat
        _JACOBIANS[key] = (A, solver)
    return _JACOBIANS[key]


#: The short names of the block pairs, mapped to the solver's field names.
_PAIR_FIELDS = {"u": "displacement", "psi": "potential", "Shift": SHIFT,
                "lambda_x": "centre_of_mass_0", "lambda_y": "centre_of_mass_1",
                "rot": "m3"}


def _pair_indices(solver, pair):
    """Field indices of a named block pair. `"rot"` is the 2-D scalar `m3`.

    Args:
      solver: the solver whose mixed space the indices refer to.
      pair: two short names from `"u"`, `"psi"`, `"Shift"`, `"lambda_x"`,
        `"lambda_y"` and `"rot"`.

    Returns:
      `(i, j)`, indices into `solver.field_index`.
    """
    return tuple(solver.field_index[_PAIR_FIELDS[name]] for name in pair)


@pytest.mark.longtest
@pytest.mark.skipif(fd.COMM_WORLD.size > 1,
                    reason="the per-block instrument needs global dense blocks")
@test_gia_gravity.REAL_BLOCK_ON_PARENT_FACETS
class TestRotationTransposes:
    """The rotation row against the sheet, on the live-coastline shoreline state.

    With rotation on, the sheet enters `inertia_form`, so the closure row of
    `m3` depends on `(u, psi, Shift)` through `Delta`. Its transpose partner is
    the `psi_rot / g_surface` term of `sea_level()` in the sea-level residual.
    Both are variations of `E_sl`: the `m3` column is
    `-c B_mu g_0 sigma Omega_sq p_3 / g_surface` and the closure row is
    `-theta_rot s_3 int sigma p_3 = -c B_mu Omega_sq int sigma p_3`, equal when
    `g_0 = g_surface`, which the fixture has.

    1e-14 relative to the larger block, as `TestTransposes` and the existing
    `test_the_rotation_pair_transposes` in `test_gia_gravity.py` use.
    """

    @pytest.mark.parametrize("pair", [("u", "rot"), ("psi", "rot"),
                                      ("Shift", "rot")])
    def test_the_rotation_pair_transposes(self, meshes, pair):
        A, solver = jacobian(meshes, "shoreline", rotation=True)
        i, j = _pair_indices(solver, pair)
        ij = dense_block(A, i, j)
        ji = dense_block(A, j, i)
        assert ij is not None and ji is not None, (
            f"block {pair} or its transpose is structurally absent")
        scale = max(np.abs(ij).max(), np.abs(ji).max())
        assert scale > 0.0
        assert np.abs(ij - ji.T).max() <= 1e-14 * scale


class TestResidualIsTheVariationOfTheEnergy:

    def test_written_residual_equals_the_derivative_of_the_energy(self, meshes):
        """At a saturated sheet, `SeaLevel.residual = derivative(E_sl)`.

        The residual is written term by term for the Slate reason of handover
        B trap 3.4.2. This pins that it is still the variation of the energy,
        on a state where every unknown is nonzero. The `"cap"` state with an
        ice change exercises the ice term of `G`. Amplitude 0.1, so `Delta` is
        of order 0.1 and the DEEP datum costs 25 * 1e-16 / 0.1 = 3e-14
        relative. Tolerance 1e-11 relative per sub-field.
        """
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "cap",
                                                      d_ice=0.5))
        perturb(z, 1e-1, seed=5)
        written = fd.assemble(sea.residual(sea.context))
        derived = fd.assemble(fd.derivative(sea.energy(),
                                            solver.solution))
        checked = 0
        for i, (a, b) in enumerate(zip(written.subfunctions,
                                       derived.subfunctions)):
            scale = np.abs(b.dat.data_ro).max()
            if i in (solver.field_index["displacement"],
                     solver.potential_index, solver.field_index[SHIFT]):
                assert scale > 0.0, f"sub-field {i} has no sea-level term"
            tolerance = 1e-11 * max(scale, 1e-300)
            assert np.abs(a.dat.data_ro - b.dat.data_ro).max() <= tolerance
            checked += 1
        assert checked == len(z.subfunctions)


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

#: The solved shoreline states, one per representation, so that
#: `TestRepresentationParity` reads the same solves as `TestLiveCoastlineSolve`
#: instead of solving each state a second time.
_SHORELINE = {}


def solve_shoreline(meshes, representation):
    """One live-coastline solve of the shoreline state on one representation, cached.

    `solve_settings(representation)` gives the tolerances. The residual norm
    at the zero state is recorded before the solve, for the Newton part of the
    thresholds of the `Shift` row and of the net sheet mass.

    Args:
      meshes: the module fixture.
      representation: `"multiplier"` or `"lowrank"`.

    Returns:
      `(solver, z, sea, initial_norm)`, with `sea` the `SeaLevel` coupling.
    """
    if representation not in _SHORELINE:
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "shoreline"),
                               representation=representation,
                               **solve_settings(representation),
                               # The DG family: the probe values of
                               # `TestRepresentationParity` were recorded on it.
                               internal_variable_family="DG")
        initial_norm = vector_norm(fd.assemble(solver.F))
        solver.solve()
        iterations = solver.solver.snes.getIterationNumber()
        print(f"shoreline live-coastline solve, {representation}: {iterations} "
              f"Newton iterations, reason "
              f"{solver.solver.snes.getConvergedReason()}")
        _SHORELINE[representation] = (solver, z, sea, initial_norm)
    return _SHORELINE[representation]


@pytest.fixture(scope="module")
def shoreline_solved(meshes, representation):
    """The shoreline solve of `solve_shoreline`, shared by read-only tests."""
    return solve_shoreline(meshes, representation)


@pytest.mark.longtest
class TestLiveCoastlineSolve:

    def test_it_converges(self, shoreline_solved):
        solver, _, _, _ = shoreline_solved
        assert solver_converged(solver)

    def test_newton_takes_at_most_five_iterations(self, shoreline_solved):
        """`NOTES/DECISIONS.md`: more than 4 to 5 means the exact live-coastline Jacobian is wrong.

        Logged as well, by the fixture.
        """
        solver, _, _, _ = shoreline_solved
        assert 1 <= solver.solver.snes.getIterationNumber() <= 5

    def test_mass_is_conserved_to_order_lambda(self, shoreline_solved):
        """`|int sigma dS|` is bounded by the frame multipliers, not by Newton.

        The `Shift` row of the residual is

            -c B_mu g_s int sigma dS + c B_mu sum_i lambda_i int x_i dsigma/dShift dS

        because the `CentreOfMass` rows keep the column `lambda . dD/dShift`
        (`NOTES/findings/FINDING-2d-sea-level-build.md`, decision 2), so the residual stays the
        gradient of one Lagrangian. At convergence the row is zero, and the
        common factor `c B_mu` of both terms cancels:

            int sigma dS = sum_i lambda_i int x_i dsigma/dShift dS / g_s

        so the net sheet mass is of order `lambda`, which is discretisation
        error. The test bounds it by the triangle inequality on that identity,

            |int sigma| <= C sum_i |lambda_i| int |x_i dsigma/dShift| dS / g_s
                           + 10 max(snes_rtol ||F_0||, snes_atol) / (c B_mu g_s)

        The second term is the part of the row that Newton leaves, with the
        same factor 10 as `test_the_shift_row_converges`. It is 1e-11 here and
        does not set the bound.

        `C = 2`. On this fixture with the softplus sheet,
        `lambda = (4.27e-9, 2.33e-9)` and `int sigma = 5.82e-9`, of the same
        size as the frame term. A missing `Shift` row or a wrong scale on
        either term gives a net mass of order the ice change, 1e-2 of the sheet
        scale, far above the bound. A flipped sign on one term alone stays
        inside the bound; `TestTransposes` and
        `TestResidualIsTheVariationOfTheEnergy` catch that.

        The bound is checked to be small against the sheet itself, so the test
        cannot pass with a large multiplier.
        """
        solver, z, sea, initial_norm = shoreline_solved
        scale_factor = scalar_value(solver.scaling_factor)
        row_scale = scale_factor * B_MU * G_RE
        dss = re_measure(solver)
        sigma = sea.sheet()
        # `dsigma/dShift`: the Gateaux derivative of the sheet in the direction
        # of the unit `Shift`, with a live coastline.
        direction = fd.Function(z.function_space())
        direction.subfunctions[solver.field_index[SHIFT]].assign(1.0)
        dsigma_dshift = expand_derivatives(
            ufl.derivative(sigma, solver.solution, direction))
        X = fd.SpatialCoordinate(solver.mesh)
        lam = [real_value(f) for f in coupling(solver, CentreOfMass).multipliers()]
        frame_term = sum(abs(lam_i) * fd.assemble(abs(X[i] * dsigma_dshift)
                                                  * dss)
                         for i, lam_i in enumerate(lam))
        newton_term = 10 * max(SNES_RTOL * initial_norm, SNES_ATOL)
        # The frame term divides by `g_s` only: `c B_mu` multiplies both the
        # sea-level part and the frame column of the row, so it cancels.
        bound = 2.0 * frame_term / G_RE + newton_term / row_scale
        net = fd.assemble(sigma * dss)
        magnitude = fd.assemble(abs(sigma) * dss)
        assert bound < 1e-6 * magnitude
        assert abs(net) <= bound

    def test_the_shift_row_converges(self, shoreline_solved):
        """The `Shift` entry of the assembled residual is within the Newton tolerance.

        Newton stops when `||F|| <= max(snes_rtol ||F_0||, snes_atol)`, and one
        entry is at most the norm. A factor 10 covers the difference between
        the norm that the fixture records before the solve and the one that
        SNES computes. Measured entry 1.3e-17 against a bound of 7e-12.

        The bound is also checked against the sea-level part of the row alone,
        `c B_mu g_s int |sigma| dS`, so that the test cannot pass on a loose
        tolerance.
        """
        solver, z, sea, initial_norm = shoreline_solved
        residual = fd.assemble(solver.F)
        row = float(residual.subfunctions[solver.field_index[SHIFT]])
        bound = 10 * max(SNES_RTOL * initial_norm, SNES_ATOL)
        scale_factor = scalar_value(solver.scaling_factor)
        row_scale = scale_factor * B_MU * G_RE
        magnitude = fd.assemble(abs(sea.sheet())
                                * re_measure(solver))
        assert bound < 1e-6 * row_scale * magnitude
        assert abs(row) <= bound

    def test_the_centre_of_mass_includes_the_sheet_and_holds(
            self, shoreline_solved):
        """After the solve `D = 0`, with `D` including the moving ocean.

        Scale: the first moment of the sheet alone, `int |x_i sigma| dS`, which
        the displacement has to cancel. 1e-8 of it, as the frame test uses.
        """
        solver, _, sea, _ = shoreline_solved
        X = fd.SpatialCoordinate(solver.mesh)
        sigma = sea.sheet()
        dss = re_measure(solver)
        D = [fd.assemble(form)
             for form in coupling(solver, CentreOfMass).dipole()]
        for i in range(2):
            moment = fd.assemble(abs(X[i] * sigma) * dss)
            assert moment > 0.0
            assert abs(D[i]) <= 1e-8 * moment


@pytest.mark.longtest
class TestRepresentationParity:
    """The shoreline state is the same on the multiplier and the low-rank path.

    Both paths discretise the same equations: the low-rank path eliminates
    the DtN multipliers into a rank-`n` update of the potential block, which
    is exact algebra and not an approximation. So the converged states agree
    to the solver tolerance. The quantities compared are the ones that sea
    level and the frame add: `Shift`, the frame multipliers, the net sheet
    mass and the degree-1 trace coefficients of the exterior DtN boundary,
    which carry the degree-1 response that the frame fixes.

    A probe of the shoreline state with the softplus sheet (`EPS` = 0.05) gave,
    on both paths, `Shift` 5.178121256074e-03, frame multipliers
    (4.274742e-09, 2.333784e-09) and net sheet mass 5.822e-09. Each path is
    also checked against the probe `Shift`, so that the two paths cannot agree
    on a wrong value.
    """

    #: The probe values of the shoreline state.
    PROBE_SHIFT = 5.178121256074e-03
    PROBE_MULTIPLIERS = (4.274742e-09, 2.333784e-09)

    @pytest.fixture(scope="class")
    def states(self, meshes):
        """The two solved shoreline states, keyed by representation."""
        return {name: solve_shoreline(meshes, name)
                for name in REPRESENTATIONS}

    def test_both_solves_converge(self, states):
        for solver, _, _, _ in states.values():
            assert solver_converged(solver)

    def test_the_shift_agrees(self, states):
        """`Shift` to 1e-9 relative between the paths and against the probe.

        The Newton tolerance is `snes_rtol = 1e-10` and both outer Krylov
        solves run to 1e-12, so the error of `Shift` is far below 1e-9 of its
        value. The probe printed 13 significant digits, and a relative
        tolerance of 1e-9 is well above the last printed digit. Measured
        difference between the paths: 1.0e-14 relative.
        """
        shifts = {name: real_value(sea.shift())
                  for name, (_, _, sea, _) in states.items()}
        for value in shifts.values():
            assert value == pytest.approx(self.PROBE_SHIFT, rel=1e-9)
        assert shifts["lowrank"] == pytest.approx(shifts["multiplier"],
                                                  rel=1e-9)

    def test_the_frame_multipliers_agree(self, states):
        """The frame multipliers to 1e-6 relative, and to the probe's 5 digits.

        The multipliers are of order 1e-9, the discretisation error of the
        centre of mass. They are the smallest solved quantity here, so their
        relative accuracy is lower than that of `Shift`: an absolute residual
        error of order 1e-15 in their rows is 1e-6 of their value. The probe
        printed 5 digits, so the comparison with it is to 5e-5 relative.
        Measured difference between the paths: 2.4e-10 and 9.3e-12 relative.
        """
        lam = {name: [real_value(f) for f in
                      coupling(solver, CentreOfMass).multipliers()]
               for name, (solver, _, _, _) in states.items()}
        for values in lam.values():
            for got, probe in zip(values, self.PROBE_MULTIPLIERS):
                assert got == pytest.approx(probe, rel=5e-5)
        for a, b in zip(lam["lowrank"], lam["multiplier"]):
            assert a == pytest.approx(b, rel=1e-6)

    def test_the_net_sheet_mass_agrees(self, states):
        """`int sigma dS` to 1e-6 relative.

        The net mass is of order the multipliers (`TestLiveCoastlineSolve`), so it
        has the same relative accuracy. Measured difference: 5.2e-9 relative.
        """
        net = {name: fd.assemble(sea.sheet() * re_measure(solver))
               for name, (solver, _, sea, _) in states.items()}
        assert abs(net["multiplier"]) > 0.0
        assert net["lowrank"] == pytest.approx(net["multiplier"], rel=1e-6)

    def test_the_trace_coefficients_agree(self, states):
        """Every DtN trace coefficient agrees, the exterior degree 1 included.

        The multiplier path reads each coefficient from its `Real` field; the
        low-rank path computes it from the trace of `psi`
        (`SelfGravitatingGIASolver.coefficients`). The tolerance is 1e-7 of
        the largest coefficient on the boundary: the coefficients of degree 2
        are of order 1e-4 on the exterior boundary, so the tolerance there is
        of order 1e-11, and the exterior degree-1 coefficients of about 4e-10
        are then compared to a few percent. A missing frame gives a degree-1
        coefficient that is not small, and a wrong recovery on the low-rank
        path gives an error of order the coefficient itself. Measured largest
        difference over scale: 1.4e-14 on the exterior boundary, 2.3e-15 on
        the interior one.

        The multiplier path returns `Real` Functions and the low-rank path
        taped floats; both are read as numbers here.
        """
        coefficients = {
            name: {bc_id: {key: float(value) for key, value in modes.items()}
                   for bc_id, modes in solver.coefficients().items()}
            for name, (solver, _, _, _) in states.items()}
        multiplier, lowrank = coefficients["multiplier"], coefficients["lowrank"]
        assert multiplier.keys() == lowrank.keys()
        for bc_id in multiplier:
            assert multiplier[bc_id].keys() == lowrank[bc_id].keys()
            scale = max(abs(v) for v in multiplier[bc_id].values())
            assert scale > 0.0
            for key, value in multiplier[bc_id].items():
                assert abs(lowrank[bc_id][key] - value) <= 1e-7 * scale, (
                    bc_id, key, value, lowrank[bc_id][key])
        # The exterior degree-1 coefficients are small against the degree-2
        # response, which is what the frame does to them. The bound is 1e-4
        # of the largest coefficient on that boundary, and the measured ratio
        # is about 4e-6.
        exterior = multiplier[CURVE_OUTER]
        scale = max(abs(v) for v in exterior.values())
        for key in ("cos1", "sin1"):
            assert abs(exterior[key]) <= 1e-4 * scale


@pytest.mark.longtest
class TestSignAgainstPhysics:

    def test_water_piles_up_towards_added_ice_on_a_stiff_earth(
            self, meshes, representation):
        """The geoid bulges towards added mass, so the sea rises near the ice.

        Stiff Earth, so the uplift is about 1e-9 and the sea-level change is
        the geoid change plus the uniform shift. Ice grows on the two caps.
        The mean of `Delta = sea_level() - SL_init` on the ocean facets next to
        the caps must exceed the mean on the facets farthest from them. With
        the geoid sign flipped it is the other way round. The margin, 1e-3 of
        the eustatic value, is above the solver tolerance by many orders and
        below the measured degree-2 geoid variation (about 5 percent of the
        eustatic value in the probe run).
        """
        _, sub = meshes
        d_ice = 1e-2
        fields = surface_fields(sub, "caps", d_ice=d_ice)
        solver, z, sea = build(meshes, fields, stiff=True,
                               representation=representation,
                               **solve_settings(representation),
                               # The DG family: on the Quadrature family the
                               # low-rank iterative preset does not converge on
                               # this stiff Earth (outer cap of 200 at the
                               # first Newton step).
                               internal_variable_family="DG")
        solver.solve()
        assert solver_converged(solver)
        dss = re_measure(solver)
        delta = sea.sea_level() - fields["SL_init"]
        near, far = indicator(sub, "near"), indicator(sub, "far")
        mean_near = fd.assemble(near * delta * dss) / fd.assemble(near * dss)
        mean_far = fd.assemble(far * delta * dss) / fd.assemble(far * dss)
        eustatic = RHO_I / RHO_W * d_ice
        assert mean_near - mean_far > 1e-3 * eustatic


# ---------------------------------------------------------------------------
# The Jacobian and the preconditioner caches under a moving coastline
# ---------------------------------------------------------------------------

def rebuild_parameters(preset=None, **overrides):
    """The iterative preset on the low-rank path, for the cache-rule tests.

    The same settings `solve_settings("lowrank")` builds, so the tolerances
    and the converged state are those every other live-coastline test of this
    file reads. They are built here rather than taken from `solve_settings`
    because each of these tests needs its own preset arguments and options,
    and because they all name the low-rank representation: that is the path whose
    block 0 is `gadopt.CondensedBlockPC` with `gadopt.LowRankPotentialPC` on
    its potential split, which is where both caches of this section live. The
    multiplier path here runs the direct preset, whose block 0 is one LU with
    no cache to rebuild.

    Args:
      preset: extra keyword arguments of
        `selfgrav_dtn_iterative_solver_parameters`, for example
        `u_ksp_max_it` or `multiplier_pc`, given as `preset_options`.
      overrides: extra options added after the preset
        (`solver_parameters_extra`), for example `lowrank_reuse_rtol` or
        `gia_rebuild`. These are plain PETSc options that no preset argument
        names.

    Returns:
      A dictionary of `SelfGravitatingGIASolver` keywords
      (`representation`, `solver_parameters`, `preset_options`,
      `solver_parameters_extra`) for `build`.
    """
    settings = solve_settings("lowrank")
    settings["preset_options"].update(preset or {})
    settings["solver_parameters_extra"].update(overrides)
    settings["representation"] = "lowrank"
    # The cache rules of this section are read from the DG family's block 0
    # (`CondensedBlockPC` and its option prefixes).
    settings["internal_variable_family"] = "DG"
    return settings


def block0_and_potential(solver):
    """The block-0 preconditioner and the potential-split one it nests.

    Imported at call time, like `condensation_context` above, so that the
    other test module is not executed at collection of this one.

    Args:
      solver: a solved `SelfGravitatingGIASolver` on `rebuild_parameters`.

    Returns:
      `(CondensedBlockPC instance, LowRankPotentialPC instance)`.
    """
    from test_gia_lowrank_block0 import block0_context, potential_pc

    block0 = block0_context(solver)
    return block0, potential_pc(block0)


def block1_context(solver):
    """The `Real`-block preconditioner of the two-block DtN Schur split.

    Args:
      solver: a solved `SelfGravitatingGIASolver` whose block-1 preconditioner
        is a Python one.

    Returns:
      The preconditioner instance.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    _, block1 = outer.pc.getFieldSplitSchurGetSubKSP()
    return block1.getPC().getPythonContext()


class TestLiveCoastlineRebuildRules:
    """W4: the moving coastline makes the Jacobian state-dependent.

    With `SeaLevel(fixed_ocean=False)` the sheet is the rounded maximum of the
    water and ice columns of the current sea level, so the surface load sheet
    follows the unknowns and the Jacobian moves inside
    one nonlinear solve, exactly as it does for a power law.
    `SelfGravitatingGIASolver.jacobian_depends_on_solution` says so, which has
    one correctness consequence and one cost consequence.

    The correctness one is that `snes_type ksponly` - one linear solve at the
    initial state - is refused instead of reporting `CONVERGED` on a state
    that is wrong by the whole motion of the coastline.

    The cost one is that `operator_version` is published as `None`, so every
    cache keyed on it rebuilds instead of holding the state the march started
    from for the whole run. `Shift` is a `Real` unknown, so the sheet sits in
    `A01`, `A10` and `A11` as well as in `A00` and every cached factor is
    stale together. Two options let a run trade accuracy of those caches
    against their cost: `lowrank_reuse_rtol` and `gia_rebuild`. Which
    value is right at production scale is measured on Gadi and not here; these
    tests pin what each value does.
    """

    @pytest.fixture(scope="class")
    def live_exact(self, meshes):
        """One live-coastline shoreline solve that rebuilds block 0 at every update.

        `gia_rebuild = always`, so the condensed blocks and with them the
        low-rank columns follow the coastline at every Newton iteration, which
        is the case the tests below need. The default (`lagged`) keeps the
        blocks of the first Newton iteration for the whole solve.

        Class-scoped: three tests read it and the solve is the expensive part.

        Returns:
            `(solver, z, sea)`.
        """
        _, sub = meshes
        solver, z, sea = build(
            meshes, surface_fields(sub, "shoreline"),
            **rebuild_parameters(gia_rebuild="always"))
        solver.solve()
        assert solver_converged(solver)
        return solver, z, sea

    @pytest.mark.longtest
    def test_the_jacobian_is_declared_state_dependent(self, live_exact):
        """The predicate is true with a live coastline on a Newtonian Earth.

        Both the attribute itself and the copy of its answer that the solver
        publishes in the application context for the preconditioners to read
        before the first solve. The rheology is Newtonian
        (`rheology_nonlinear` is false), so the live coastline is the whole of
        the reason.
        """
        solver, _, sea = live_exact
        assert not sea.fixed_ocean
        assert solver.jacobian_depends_on_solution
        assert solver.appctx["gia_jacobian_depends_on_solution"] is True
        # The rheology is Newtonian, so the coastline is the whole of the reason.
        assert not solver.rheology_nonlinear

    @pytest.mark.longtest
    def test_block_zero_symmetry_is_unchanged(self, meshes, live_exact):
        """A live coastline does not make the condensed block-0 operator nonsymmetric.

        This is the check the design rests on. The two predicates read the
        same rheology today, so separating them could have moved the Krylov
        method on block 0 as a side effect, and CG on a nonsymmetric operator
        does not raise: it converges to the wrong thing or not at all.

        It cannot, and the reason is physical.  The sheet depends on the
        unknowns only through the one scalar `Delta = dN - du_r + Shift`, so
        one energy `E_sl` generates all three sea-level rows and its second
        variation is symmetric whatever the sheet does
        (`NOTES/DESIGN-SEA-LEVEL.md` section 3).
        The symmetry of the condensed operator follows `rheology_nonlinear`,
        which reads the material alone and not `jacobian_depends_on_solution`,
        so the live coastline reaches it by no path at all.

        The assertion is made on the answer and on its consequence. The
        consequence is the Krylov method the options carry on the condensed
        displacement split, which is where the choice is spent. The preset's
        default writes `preonly` there - one multigrid V-cycle, which is not a
        Krylov method and assumes nothing - so `u_ksp_max_it` is named, which
        is what makes the preset ask for the truncated CG.
        """
        _, sub = meshes
        solver, _, _ = live_exact
        assert not solver.rheology_nonlinear

        key = "dtn_fieldsplit_0_condensed_fieldsplit_0_ksp_type"
        settings = rebuild_parameters(preset=dict(u_ksp_max_it=5))
        # Guard: the preset asks for the truncated CG with these arguments.
        preset = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, dtn_representation="lowrank",
            **settings["preset_options"])
        assert preset[key] == "cg"
        truncated, _, _ = build(meshes, surface_fields(sub, "shoreline"),
                                **settings)
        assert truncated.jacobian_depends_on_solution
        assert not truncated.rheology_nonlinear
        assert truncated.solver_parameters[key] == "cg"
        # And solve it. Without this the test pins the code path and not the
        # mathematics: every assertion above would hold if the live-coastline
        # block-0 operator were in fact nonsymmetric, which is the failure the
        # docstring says cannot happen. CG on a nonsymmetric operator does not
        # raise, so the evidence is that the solve converges.
        truncated.solve()
        assert solver_converged(truncated)

    @pytest.mark.longtest
    def test_the_condensed_block_reassembles_at_every_newton_iteration(
            self, live_exact):
        """Under `always`, `assembly_count` equals the Newton iteration count.

        `operator_version` is `None` and `gia_rebuild = always`, so the class
        reassembles its blocks at every linear solve. Newton takes one linear
        solve per iteration, and the count starts at 1 at `initialize`, so the
        two numbers are equal.

        Item 5 of the handout. Without the predicate change the version is a
        fixed integer, the count stays at 1 for the whole march, and the
        condensed operator, the low-rank columns `Z` and the `Real`-block
        complement all describe the coastline the step started from.
        """
        solver, _, _ = live_exact
        block0, _ = block0_and_potential(solver)
        iterations = solver.solver.snes.getIterationNumber()
        print(f"a live coastline: {iterations} Newton iterations, "
              f"assembly_count {block0.assembly_count}")
        assert iterations > 1, "the live-coastline solve must take Newton steps"
        assert block0.assembly_count == iterations

    @pytest.mark.longtest
    def test_a_fixed_coastline_assembles_once_over_two_solves(self, meshes):
        """With `fixed_ocean` the count stays at 1 across two solves at one `dt`.

        The other half of the pair. A frozen coastline makes the residual
        linear, `operator_version` stays the integer it started at, and
        nothing in the Jacobian moves between two solves at the same time
        step, so the condensed operator and the low-rank columns are built
        once and serve both solves. That is what the reuse rule exists for,
        and it must survive the change that switches it off under a live coastline.
        """
        _, sub = meshes
        solver, _, _ = build(meshes, surface_fields(sub, "shoreline"),
                             sea_level_overrides=fixed(),
                             **rebuild_parameters())
        assert not solver.jacobian_depends_on_solution
        solver.solve()
        assert solver_converged(solver)
        block0, potential = block0_and_potential(solver)
        assert block0.assembly_count == 1
        assert potential.column_builds == 1
        solver.solve()
        assert solver_converged(solver)
        assert block0.assembly_count == 1
        assert potential.column_builds == 1

    def test_ksponly_with_a_live_coastline_is_refused_and_the_message_says_why(
            self, meshes):
        """The guard fires on the live sheet, and names it rather than the exponent.

        Without this, a Newtonian Earth with a moving coastline accepts
        `ksponly`: one linear solve at the initial state reports `CONVERGED`
        and returns a state wrong by the whole motion of the coastline, with
        nothing in the log. The refusal is raised while the options are set,
        so it arrives before any solve.

        The message is asserted on as well as the refusal. It must name the
        live sea-level sheet among the causes and `SeaLevel(fixed_ocean=True)`
        as the way out. The exponent is 1 here, so a message
        that named the power-law exponent would send the reader to look at a
        rheology that is already right.
        """
        _, sub = meshes
        with pytest.raises(ValueError) as raised:
            build(meshes, surface_fields(sub, "shoreline"),
                  **rebuild_parameters(snes_type="ksponly"))
        message = str(raised.value)
        assert "ksponly" in message
        assert "SeaLevel(fixed_ocean=True)" in message
        assert "sea-level sheet" in message
        assert "exponent" not in message

    @pytest.mark.longtest
    def test_the_reuse_tolerance_holds_the_low_rank_columns(
            self, meshes, live_exact):
        """`lowrank_reuse_rtol = 1e-3` builds `Z` once and changes no answer.

        `gadopt.LowRankPotentialPC` keys the columns `Z = A_psipsi^-1 U` on the
        Frobenius norm of `A_psipsi`. With a live coastline that block moves at every
        Newton iteration, because the ocean load carries `psi` through the
        geoid, so the exact-equality rule rebuilds the columns at each one: one
        accurate solve per DtN mode, thrown away one iteration later.

        The motion is small against the Laplacian part of the block, so a
        relative tolerance of 1e-3 absorbs it and one build serves the solve.
        The columns are preconditioner data, so the converged state is the
        same to the outer tolerance either way - that is the claim this test
        makes, and it is what lets the production value be chosen on cost
        alone.
        """
        exact_solver, _, _ = live_exact
        _, sub = meshes
        _, exact_potential = block0_and_potential(exact_solver)
        # Vacuity guard: with the exact rule the columns really are rebuilt,
        # so a tolerance that holds them at 1 is doing something.
        assert exact_potential.column_builds > 1

        # The same `always` rule as `live_exact`, so the blocks move at every
        # Newton iteration and the tolerance is what holds the columns.
        solver, _, _ = build(
            meshes, surface_fields(sub, "shoreline"),
            **rebuild_parameters(lowrank_reuse_rtol=1e-3,
                                 gia_rebuild="always"))
        solver.solve()
        assert solver_converged(solver)
        _, potential = block0_and_potential(solver)
        print(f"lowrank_reuse_rtol: column_builds {potential.column_builds} "
              f"against {exact_potential.column_builds} with the exact rule")
        assert potential.column_builds == 1
        assert solved_fields_agree(solver, exact_solver, rel=1e-6) == []

    @pytest.mark.longtest
    @pytest.mark.parametrize("rule,builds", [
        ("lagged", 1), ("per_solve", 1), ("always", 3)])
    def test_the_gia_rebuild_rule(self, meshes, rule, builds):
        """`gia_rebuild` selects one build per solve or one per update.

        `gadopt.DtNMultiplierDenseSchurPC` is named here, because the preset
        leaves block 1 to the cached apply of `gadopt.DtNTwoBlockSchurPC` at
        this width and that class owns no complement to rebuild.

        `per_solve` is the rule for a Jacobian no version number can
        describe: build at the first Newton iteration of a solve, from the
        state that solve starts at, and keep it for the rest of the solve.
        `lagged`, the default, is the same within one solve and differs only
        across solves. `always` builds at every `update`, which is one build
        per linear solve and so one per Newton iteration.

        The expected counts are therefore 1 and the Newton iteration count.
        The iteration count is asserted to be the 3 the parametrisation was
        written against, so that a solve which took fewer steps could not turn
        the second case into the first without failing.
        """
        _, sub = meshes
        settings = rebuild_parameters(
            preset=dict(multiplier_pc="gadopt.DtNMultiplierDenseSchurPC"),
            gia_rebuild=rule)
        solver, _, _ = build(meshes, surface_fields(sub, "shoreline"),
                             **settings)
        solver.solve()
        assert solver_converged(solver)
        iterations = solver.solver.snes.getIterationNumber()
        complement = block1_context(solver)
        print(f"gia_rebuild={rule}: build_count "
              f"{complement.build_count}, {iterations} Newton iterations")
        assert complement._rebuild_rule == rule
        assert iterations == 3
        assert complement.build_count == builds


def taylor_forward(control, meshes, *, rotation, representation,
                   fixed_ocean=False):
    """Tape one solve of the shoreline state with ice thickness `control`.

    The functional reads the degree-1 radial displacement at Re and `Shift`,
    so that it depends on the sheet through mass conservation and through the
    ocean load. The weights make the two parts of similar size.

    Args:
      control: the ice thickness, a CG2 `Function` on the mechanics mesh.
      meshes: the module fixture.
      rotation: carry the rotation closure.
      representation: `"multiplier"` or `"lowrank"`.
      fixed_ocean: the `SeaLevel.fixed_ocean` setting. `False` tapes the
        live-coastline solve.

    Returns:
      The functional, a taped `AdjFloat`.
    """
    parent, sub = meshes
    with stop_annotating():
        fields = surface_fields(sub, "shoreline")
        fields["I"] = control
        solver, _, _ = build(meshes, fields, rotation=rotation,
                             representation=representation,
                             sea_level_overrides=dict(
                                 fixed_ocean=fixed_ocean),
                             **solve_settings(representation))
    solver.solve()
    # The UFL components of the solved mixed solution, so the functional
    # depends on the solve on the tape.
    u = solver.field("displacement")
    shift = solver.field(SHIFT)
    X = fd.SpatialCoordinate(sub)
    rhat = X / fd.sqrt(fd.dot(X, X))
    ds_re = fd.Measure("ds", domain=sub)(CURVE_RE)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    return fd.assemble(
        1e3 * fd.dot(u, rhat) * fd.cos(fd.atan2(X[1], X[0])) * ds_re
        + 1e4 * shift * shift * dx_m)


def run_taylor(meshes, rotation, representation, fixed_ocean=False):
    """The guarded Taylor test with control `I` and a direction that moves the grounding line.

    On the low-rank path the tape carries a `CoupledLowRankDtNSolveBlock`, and
    the adjoint goes through the hand-written low-rank adjoint, which carries
    the `Shift` and frame rows as UFL terms.

    With `fixed_ocean` the functional is affine in the control apart from
    the `Shift^2` term, because the residual is linear. The live direction
    `sin(phi - PHASE)` is odd about the centre of the land at
    `phi = PHASE + pi`, so it adds no net ice on the land, `Shift` does not
    move and the Taylor remainder is round-off (measured 5e-15 to 4e-11
    against `J = -77`, rates -2.8 and -3.7, `NOTES/team/s3-w2-fixed-ocean/
    logs/fixed_round2.log`). The fixed arm therefore uses the direction
    `-0.1 cos(phi - PHASE)`, which is 0.1 at the centre of the land and
    changes the net land ice, so `Shift` and its square move with the
    control.
    """
    from test_gravity_adjoint import assert_taylor_with_guards

    _, sub = meshes
    V = fd.FunctionSpace(sub, "CG", 2)
    phi = polar_angle(sub)
    with stop_annotating():
        control = fd.Function(V).interpolate(
            (1.0 - MELT) * 0.6 * fd.max_value(0.0, -fd.cos(phi - PHASE) - 0.4))
        if fixed_ocean:
            # Thickening centred on the land, of 0.1 at most, so the net ice
            # on the reference land changes and `Shift` responds. The part
            # over the reference ocean is removed by `1 - C0`.
            h = fd.Function(V).interpolate(-0.1 * fd.cos(phi - PHASE))
        else:
            # Thickening on one flank and thinning on the other, of 0.1 at
            # most, scaled by the ice thickness over its maximum. The sheet
            # clips the ice at zero, and a direction that thinned ice-free
            # ground would make the ice negative there and put a kink into the
            # functional (Taylor rate 1, `nonnegative_ice`). Scaled by the ice,
            # `I + h dm` keeps the sign of `I`; the coast still moves through
            # `Shift`.
            peak = control.dat.data_ro.max()
            h = fd.Function(V).interpolate(
                0.1 * fd.sin(phi - PHASE) * control / peak)
    continue_annotation()
    try:
        m = Control(control)
        J = taylor_forward(control, meshes, rotation=rotation,
                           representation=representation,
                           fixed_ocean=fixed_ocean)
        Jhat = ReducedFunctional(J, m)
    finally:
        pause_annotation()
    # min_rate 1.9: "rate close to 2" of the task, the threshold the frame
    # adjoint test uses. The guards also check that the gradient is nonzero
    # and that the same test without the gradient gives rate 1.
    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)


@pytest.mark.longtest
class TestTaylor:

    def test_live_coastline_taylor_rate_with_ice_thickness_control(
            self, meshes, representation):
        run_taylor(meshes, rotation=False, representation=representation)


@pytest.mark.longtest
class TestRotation:
    """Rotation on. Acceptance (`NOTES/DECISIONS.md`): it solves and the Taylor test passes."""

    def test_the_rotating_case_passes_the_taylor_test(
            self, meshes, representation):
        run_taylor(meshes, rotation=True, representation=representation)


# ---------------------------------------------------------------------------
# The fixed coastline
# ---------------------------------------------------------------------------

def coastline_fields(sub, land, *, d_ice=0.0, ocean_ice=0.0):
    """A deep ocean with land sectors under grounded ice, as `SeaLevel` keywords.

    The fields are DG0, so a sector edge sits on mesh vertices and no
    quadrature point sees a transition (`indicator`). `SL_init` is `+DEEP` in
    the ocean and `-DEEP` on the land, so both the fixed ocean function
    `C0 = H(r SL_init / eps)` and the live sheet are saturated. This is what
    makes a fixed and a live coastline the same problem: the ice rests on
    land, so the live water fraction is 0 there, and the ocean is deep and
    free of grounded ice, so the live water fraction is 1 there.

    Args:
      sub: the mechanics submesh.
      land: `"cap"` or `"caps"`, the sectors of `indicator` that are land.
      d_ice: the ice thickness change on the land, in units of D.
      ocean_ice: the ice thickness that grows over the whole ocean from zero,
        in units of D. A value above `(rho_w / rho_i) DEEP = 26.9` grounds
        that ice on the sea floor under a live coastline. A fixed coastline removes
        it through `1 - C0`.

    Returns:
      `dict(SL_init=..., I=..., I_init=..., N_init=..., ur_init=...)`.
    """
    on_land = indicator(sub, land)
    Q = on_land.function_space()
    zero = fd.Function(fd.FunctionSpace(sub, "CG", 2))
    SL_init = fd.Function(Q).interpolate(DEEP * (1.0 - 2.0 * on_land))
    I_init = fd.Function(Q).interpolate(GROUNDED * on_land)
    I = fd.Function(Q).interpolate((GROUNDED + d_ice) * on_land
                                   + ocean_ice * (1.0 - on_land))
    return dict(SL_init=SL_init, I=I, I_init=I_init, N_init=zero.copy(True),
                ur_init=zero.copy(True))


def fixed(**overrides):
    """`sea_level_overrides` of a fixed coastline, with extra `SeaLevel` keywords."""
    return dict(fixed_ocean=True, **overrides)


def remove_rigid_rotation(du, mesh):
    """The displacement `du` with its L2-best rigid rotation of the mantle removed.

    A rigid rotation of the whole mantle about the centre is strain free,
    divergence free and tangential on every concentric circle or sphere
    (the solver's `nullspace="rigid_rotation"`). With rotation off and no
    nullspace declared, only the facet geometry error of the curved mesh acts
    on it. So the multiple of that mode in a solution is set by the Krylov
    path and by round-off, and not by the physics. Two solves of the same
    problem can differ by it far above their solver tolerance.

    The fit is a least-squares fit in L2 over the mechanics mesh: the Gram
    matrix `G_ij = int r_i . r_j dx` of the rotation generators `r_i` and the
    moments `b_i = int r_i . du dx` give the coefficients `c = G^-1 b`. In 2-D
    there is one generator `(-y, x)`, in 3-D the three generators `e_i x x`.
    Rigid translations are not removed: the centre-of-mass frame rows fix
    them, and the fit measured them at about 1e-18 in the agreement test.

    Args:
      du: a displacement `Function` on the mechanics mesh.
      mesh: the mechanics mesh.

    Returns:
      A new `Function` in the space of `du` with the fitted rotation removed.
    """
    V = du.function_space()
    X = fd.SpatialCoordinate(mesh)
    if len(X) == 2:
        generators = [fd.as_vector((-X[1], X[0]))]
    else:
        generators = [fd.as_vector((0.0, -X[2], X[1])),
                      fd.as_vector((X[2], 0.0, -X[0])),
                      fd.as_vector((-X[1], X[0], 0.0))]
    # Interpolate the generators into the displacement space, so that the
    # removal acts on the same degrees of freedom that the max norm reads.
    modes = [fd.Function(V).interpolate(r) for r in generators]
    dx = fd.dx(domain=mesh)
    gram = np.array([[fd.assemble(fd.inner(a, b) * dx) for b in modes]
                     for a in modes])
    moments = np.array([fd.assemble(fd.inner(a, du) * dx) for a in modes])
    coefficients = np.linalg.solve(gram, moments)
    residual = du.copy(deepcopy=True)
    for coefficient, mode in zip(coefficients, modes):
        residual.dat.data[:] -= coefficient * mode.dat.data_ro
    return residual


def solved_fields_agree(solver_a, solver_b, rel):
    """Whether the displacement, potential and `Shift` of two solved states agree.

    Each field is compared in the max norm, relative to the max norm of the
    same field of `solver_b`. The displacement difference is compared
    after `remove_rigid_rotation`. The rigid rotation of the mantle is not
    fixed by any physical term of these equations. Its multiple follows the
    Krylov path, so a comparison that keeps it tests round-off and not the
    physics. The rotation does not change `u . n`, the geoid or `Shift`, so
    the removal hides no physical difference.

    The frame multipliers and the DtN multipliers are not compared. They are
    discretisation error of order 1e-9, so the solver tolerance is not small
    against their size.

    Args:
      solver_a, solver_b: two solved solvers on the same meshes.
      rel: the relative tolerance of each field.

    Returns:
      A list of `(name, difference, scale)` for the fields that fail. The
      measured relative differences are printed for every field, so a run
      with `-s` records the margin.
    """
    failures = []
    for name in ("displacement", "potential", SHIFT):
        f_a = solver_a.solution.subfunctions[solver_a.field_index[name]]
        f_b = solver_b.solution.subfunctions[solver_b.field_index[name]]
        b = f_b.dat.data_ro
        scale = np.abs(b).max()
        if name == "displacement":
            du = fd.Function(f_b.function_space()).assign(f_a - f_b)
            difference = np.abs(
                remove_rigid_rotation(du, solver_b.mesh).dat.data_ro).max()
        else:
            difference = np.abs(f_a.dat.data_ro - b).max()
        print(f"solved_fields_agree: {name}, relative difference "
              f"{difference / scale if scale > 0.0 else np.inf:.3e}",
              flush=True)
        if not (scale > 0.0 and difference <= rel * scale):
            failures.append((name, difference, scale))
    return failures


class TestFixedOcean:
    """`SeaLevel(fixed_ocean=True)`: the coastline of Martinec et al. (2018), cases B and C.

        C0    = H(r SL_init / eps)
        sigma = rho_w C0 (SL - SL_init) + rho_i (1 - C0) (I - I_init)

    Work item W2 of `NOTES/DESIGN-MARTINEC-3D.md`. Every test runs on both
    DtN representations.
    """

    @pytest.mark.longtest
    def test_newton_converges_in_one_iteration(self, meshes, representation):
        """The fixed-coastline residual is linear: one Newton step from zero.

        The shoreline state, whose `C0` is fractional over a wide arc, so the
        test does not depend on a saturated sheet. `newtonls` from the zero
        initial guess with the tolerances of `solve_settings`. The outer
        Krylov solve reduces the residual by 1e-12, below `snes_rtol = 1e-10`,
        so a linear residual stops at iteration 1. A live-coastline solve of the
        same state takes more (`TestLiveCoastlineSolve`).
        """
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "shoreline"),
                               representation=representation,
                               sea_level_overrides=fixed(),
                               **solve_settings(representation))
        assert solver.solver.snes.getType() == "newtonls"
        solver.solve()
        assert solver_converged(solver)
        assert solver.solver.snes.getIterationNumber() <= 1
        assert abs(real_value(sea.shift())) > 0.0

    @pytest.mark.longtest
    def test_the_jacobian_is_not_state_dependent_and_ksponly_is_accepted(
            self, meshes, representation):
        """With a fixed coastline and a Newtonian Earth, `ksponly` solves the step.

        Item 5 of W2: the solver does not declare the Jacobian state-dependent
        (`jacobian_depends_on_solution`), so its `ksponly` refusal accepts
        `snes_type ksponly`, and the one linear solve gives the `newtonls`
        solution to 1e-8 relative.
        """
        _, sub = meshes
        states = {}
        for snes_type in ("newtonls", "ksponly"):
            settings = solve_settings(representation)
            settings["solver_parameters_extra"]["snes_type"] = snes_type
            solver, z, sea = build(meshes, surface_fields(sub, "shoreline"),
                                   representation=representation,
                                   sea_level_overrides=fixed(), **settings)
            assert not solver.jacobian_depends_on_solution
            solver.solve()
            assert solver_converged(solver)
            assert solver.solver.snes.getType() == snes_type
            states[snes_type] = solver
        assert solved_fields_agree(states["ksponly"], states["newtonls"],
                                   rel=1e-8) == []

    def test_written_residual_equals_the_derivative_of_the_energy(
            self, meshes, representation):
        """`SeaLevel.residual = derivative(SeaLevel.energy(), solution)` exactly.

        With a fixed coastline the energy
        `G = rho_i (I_eff - I_eff_init) Delta + 0.5 rho_w C0 Delta^2` is exact
        for any `C0`, so the shoreline state with its fractional `C0` and its
        melting ice is used, at a random state of amplitude 0.1. Tolerance
        1e-11 relative per sub-field, as in
        `TestResidualIsTheVariationOfTheEnergy`.
        """
        _, sub = meshes
        solver, z, sea = build(meshes, surface_fields(sub, "shoreline"),
                               representation=representation,
                               sea_level_overrides=fixed(),
                               **solve_settings(representation))
        perturb(z, 1e-1, seed=7)
        written = fd.assemble(sea.residual(sea.context))
        derived = fd.assemble(fd.derivative(sea.energy(),
                                            solver.solution))
        checked = 0
        for i, (a, b) in enumerate(zip(written.subfunctions,
                                       derived.subfunctions)):
            scale = np.abs(b.dat.data_ro).max()
            if i in (solver.field_index["displacement"],
                     solver.potential_index, solver.field_index[SHIFT]):
                assert scale > 0.0, f"sub-field {i} has no sea-level term"
            tolerance = 1e-11 * max(scale, 1e-300)
            assert np.abs(a.dat.data_ro - b.dat.data_ro).max() <= tolerance
            checked += 1
        assert checked == len(z.subfunctions)

    @pytest.mark.longtest
    def test_the_taylor_rate_with_ice_thickness_control(
            self, meshes, representation):
        """The guarded Taylor test of `run_taylor` on a fixed coastline."""
        run_taylor(meshes, rotation=False, representation=representation,
                   fixed_ocean=True)
