"""The sea-level equation on the coupled self-gravitating solver, 2-D annulus.

These are the acceptance tests of the sea-level equation. The design they check
is in `NOTES/DESIGN-SEA-LEVEL.md`, and the decisions behind it in
`NOTES/DECISIONS.md` and `NOTES/findings/FINDING-2d-sea-level-build.md`:

    SL      = SL_init + (N - N_init) - (u_r - ur_init) + Shift
    Delta   = SL - SL_init
    sigma   = rho_w (B C SL - B_init C_init SL_init)
              + rho_i ((1 - B) I - (1 - B_init) I_init)
    E_sl    = -c B_mu g_0 int_Re G(Delta) dS,   dG/dDelta = sigma

with `C = smooth_step(SL, k)` the ocean function, `B = 1 -
smooth_step(I - (rho_w/rho_i) SL, k)` the grounded-ice function and
`k = alpha p / h`. The `Shift` row of the residual is mass conservation,
`int sigma dS = 0`, up to the frame column `lambda . dD/dShift` that keeps the
residual the gradient of one Lagrangian. At a converged state the net sheet
mass is therefore of order `lambda`, which is discretisation error, and not
zero to the solver tolerance (`NOTES/findings/FINDING-2d-sea-level-build.md`, decision 2).

The configuration is the one of `TestCentreOfMassFrame.build` in
`test_gia_gravity.py`: the coarse P2-curved annulus (dr 0.2, 32 azimuthal
cells), a fluid core, the centre-of-mass multipliers and the reference gravity
of the 2-D Gauss law for the test's own density. Rotation is off except where a
test says otherwise.

## Four surface states

Most tests use one of four prescribed surface states. Three of them make every
mask exactly 0 or 1 at every quadrature point, so that the answer of a test can
be written without the mask functions:

- `"ocean"`: a deep uniform ocean, `SL_init = DEEP`, no ice.
- `"cap"`: the same ocean with grounded ice on the quarter circle
  `0 < phi < pi/2`. The ice is a DG0 field, so its edge sits on mesh vertices
  and no quadrature point sees a transition. The ocean is then asymmetric in
  both `x` and `y`, which makes every frame coupling nonzero.
- `"caps"`: grounded ice on the two opposite quarter circles centred on
  `phi = 0` and `phi = pi`. The load has no degree-1 content, so the rigid
  translation of the frame does not enter the uplift.
- `"shoreline"`: a smooth sea level `cos(phi - PHASE) + 0.2` with land and a
  melting ice sheet on it. The masks are fractional over a wide arc, so this
  state tests the live masks.

`DEEP = 25` is chosen so that `0.5 k DEEP` is about 29 with
`k = 0.5 * 2 / FacetArea` (about 2.3 on this mesh). `tanh(29)` is 1 to double
precision, so `C = 1` exactly. The grounded ice is `GROUNDED = 100`, so
`I - (rho_w/rho_i) SL = 73` under the ice and `-26.9` in the open ocean, which
saturates `B` in the same way.

## What these tests are not

They are 2-D structural and consistency checks. Nothing here is a benchmark:
Martinec et al. (2018) is 3-D and out of scope, and so are the low-rank DtN
representation, Will Scott's two-disc moving-shoreline case and Irksome.
"""

import importlib
import importlib.util
from pathlib import Path

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

from gadopt import SelfGravitatingGIASolver, self_gravitating_gia_space
from gadopt import gia_gravity
from gadopt.gia_gravity import FluidCore, selfgrav_dtn_iterative_solver_parameters
import test_gia_gravity
from test_gia_gravity import (  # noqa: E402  (module-level fixture and helpers)
    B_MU,
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    RC,
    RE,
    approximation,
    gravity_bcs,
    meshes,  # noqa: F401  (pytest fixture)
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Water and ice densities in the units of the test's mantle density, which is
#: 1. The mantle stands for the Spada upper mantle, 4978 kg/m^3, and Will
#: Scott's code uses 1000 and 931 kg/m^3. The ratio rho_i/rho_w = 0.931 must
#: match Will's class attributes for the comparison test.
RHO_W = 1000.0 / 4978.0
RHO_I = 931.0 / 4978.0

#: The reference state of `TestCentreOfMassFrame`: a core of density 2 inside
#: Rc and a unit-density mantle. `G_RE` is the surface gravity of that state,
#: about 1.19 in the non-dimensional units of the solver.
# Reached through the module and not imported by name: a `Test*` class in this
# module's namespace would be collected and run a second time here.
FRAME = test_gia_gravity.TestCentreOfMassFrame()
G_RE = FRAME.gravity_of_r(RE)

#: A deep uniform ocean and a thick grounded ice sheet, in units of the mantle
#: thickness D. The values are not physical. They saturate the masks, see the
#: module docstring.
DEEP = 25.0
GROUNDED = 100.0

#: The mask steepness factor of the task, `k = alpha * p / h`.
ALPHA_MASK = 0.5

#: Rotation of the smooth shoreline state, so that it has no mirror symmetry in
#: `x` or `y` and every first-moment integral of the ocean is nonzero.
PHASE = 0.5

#: Fraction of the ice sheet that melts in the shoreline state.
MELT = 0.1

#: Tight solver tolerances for tests that read a converged state. `ksp_rtol`
#: is the outer FGMRES of the direct preset, as in
#: `test_gia_gravity_adjoint.py`. `snes_atol` is set explicitly so that the
#: mass-conservation threshold below is computed from known numbers.
SNES_RTOL = 1e-10
SNES_ATOL = 1e-15
TIGHT = {"snes_rtol": SNES_RTOL, "snes_atol": SNES_ATOL, "ksp_rtol": 1e-12}

#: Path of Will Scott's `SeaLevelSolver`, copied into the untracked NOTES.
WILL_SEA_LEVEL = (Path(__file__).resolve().parents[2] / "NOTES" / "will"
                  / "sea_level.py")


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


def masks():
    """The new masks module. Imported at call time and not at module level.

    A module-level import of a module that does not exist yet stops collection
    of the whole file. At call time each test fails on its own.
    """
    return importlib.import_module("gadopt.sea_level_masks")


def sea_level_class():
    """`gadopt.gia_gravity.SeaLevel`, read at call time for the same reason."""
    return gia_gravity.SeaLevel


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
        # and an ice sheet on the land that is 0.36 thick at its centre. With
        # k about 2.3 the transition of C is about 0.43 wide in SL, so the
        # masks are fractional on a wide arc, which is the point of this state.
        SL_init = fd.Function(V).interpolate(fd.cos(phi - PHASE) + 0.2)
        I_init = fd.Function(V).interpolate(
            0.6 * fd.max_value(0.0, -fd.cos(phi - PHASE) - 0.4))
        I = fd.Function(V).interpolate((1.0 - MELT) * I_init)
    else:
        raise ValueError(state)
    return dict(SL_init=SL_init, I=I, I_init=I_init, N_init=zero.copy(True),
                ur_init=zero.copy(True))


def make_sea_level(fields, **overrides):
    """A `SeaLevel` on the Re tag with the test's densities and gravity."""
    settings = dict(boundary=CURVE_RE, rho_w=RHO_W, rho_i=RHO_I,
                    g_surface=G_RE, alpha_mask=ALPHA_MASK)
    settings.update(fields)
    settings.update(overrides)
    return sea_level_class()(**settings)


def build(meshes, fields=None, *, rotation=False, lam=LAMBDA, stiff=False,
          earth=None, sea_level=True, surface_bcs=None, gravity_sheet=None,
          quad_degree=None, sea_level_overrides=None, **solver_kwargs):
    """The frame configuration of `TestCentreOfMassFrame.build`, with sea level.

    Args:
      meshes: the module fixture.
      fields: `surface_fields(...)`; required when `sea_level` is true.
      rotation: carry the rotation closure.
      lam: `Lambda`, given to the factory and to the approximation. The
        reference gravity `g(r)` keeps the value of the frame tests, so a small
        `lam` removes self-attraction and keeps the load weight.
      stiff: shear and bulk modulus 1e6 and a viscosity of 1e18, so the
        Maxwell time is 1e12 against `dt = 1` and the step is elastic. The
        uplift then scales as 1e-6 of the default Earth.
      earth: extra keywords for the approximation (moduli, viscosity), applied
        after `stiff`. `None` keeps the default Earth.
      sea_level: whether the space and the solver carry the sea-level equation.
        With `False` the solver is built without the new keyword at all, so
        the comparison solvers of the sign tests run on the current code.
      surface_bcs: mechanics `bcs`; empty by default because the ice load is
        part of the sea-level sheet.
      gravity_sheet: an `interior_sigma` on Re for the potential, or `None`.
      quad_degree: the boundary quadrature degree of the factory.
      sea_level_overrides: extra `SeaLevel` keywords.
      solver_kwargs: passed to `SelfGravitatingGIASolver`.

    Returns:
      `(solver, z, layout)`.
    """
    parent, sub = meshes
    bcs_psi = gravity_bcs(parent, sheet=False)
    if gravity_sheet is not None:
        bcs_psi[CURVE_RE] = {"interior_sigma": gravity_sheet}
    space_kwargs = dict(gravity_bcs=bcs_psi, rotation=rotation,
                        fluid_core=True, centre_of_mass=True,
                        self_gravity_number=lam)
    if quad_degree is not None:
        space_kwargs["quad_degree"] = quad_degree
    if sea_level:
        space_kwargs["sea_level"] = True
    Z, layout = self_gravitating_gia_space(sub, parent, **space_kwargs)
    z = fd.Function(Z)

    Xm = fd.SpatialCoordinate(sub)
    rm = fd.sqrt(fd.dot(Xm, Xm))
    approximation_kwargs = dict(g=FRAME.gravity_of_r(rm),
                                self_gravity_number=lam)
    if stiff:
        approximation_kwargs.update(shear_modulus=1e6, bulk_modulus=1e6,
                                    viscosity=1e18)
    if earth is not None:
        approximation_kwargs.update(earth)
    moments = {}
    if rotation:
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        moments["C"] = fd.assemble(fd.dot(Xm, Xm) * dx_m)
    if sea_level:
        solver_kwargs["sea_level"] = make_sea_level(
            fields, **(sea_level_overrides or {}))
    solver = SelfGravitatingGIASolver(
        z, approximation(**approximation_kwargs), layout=layout, dt=1.0,
        bcs=surface_bcs or {}, rotation_moments=moments,
        # The core boundary carries the reference gravity at Rc, as in
        # `TestCentreOfMassFrame.build`.
        fluid_core=FluidCore(boundary=CURVE_RC, rho_core=FRAME.RHO_CORE,
                             g=fd.Constant(FRAME.gravity_of_r(RC))),
        **solver_kwargs)
    return solver, z, layout


def re_measure(solver):
    """The solver's sea-level measure, restricted to the Re tag.

    The task describes `sea_level_measure()` as the generalisation of
    `fluid_core_measure`, which returns a measure that the caller restricts to
    a tag. This helper accepts either spelling: a measure that is not yet
    restricted is called on `CURVE_RE`.
    """
    measure = solver.sea_level_measure()
    if measure.subdomain_id() == "everywhere":
        measure = measure(CURVE_RE)
    return measure


def real_value(z, index):
    """The one number of a `Real` sub-field."""
    return float(z.subfunctions[index].dat.data_ro[0])


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
# The masks module
# ---------------------------------------------------------------------------

class TestMasks:
    """`gadopt.sea_level_masks`: the smooth step, the steepness and the masks.

    The numbers are evaluated as integrals of a constant over the unit square,
    whose area is 1, so an assembled value is the value of the expression.
    """

    @staticmethod
    def value(expr):
        mesh = fd.UnitSquareMesh(1, 1)
        return float(fd.assemble(expr * fd.dx(domain=mesh)))

    def test_smooth_step_is_one_half_at_zero(self):
        # 1e-15 absolute: the quadrature weights of the unit square sum to 1
        # only to round-off.
        got = self.value(masks().smooth_step(fd.Constant(0.0), 3.0))
        assert abs(got - 0.5) < 1e-15

    def test_smooth_step_is_the_logistic_function(self):
        """`0.5 (1 + tanh(k x / 2)) = 1 / (1 + exp(-k x))`, to round-off.

        1e-14 absolute: both forms are O(1) and double precision gives 1e-16.
        """
        x, k = 0.3, 2.0
        expected = 1.0 / (1.0 + np.exp(-k * x))
        got = self.value(masks().smooth_step(fd.Constant(x), k))
        assert abs(got - expected) < 1e-14

    @pytest.mark.parametrize("x, expected", [(1e3, 1.0), (-1e3, 0.0)])
    def test_smooth_step_saturates_exactly(self, x, expected):
        """Far from the switch the mask is exactly 0 or 1, not close to it.

        The saturated test states of this file depend on this. The tolerance
        1e-15 absolute is the round-off of the unit-square quadrature weights.
        The test also checks the side: positive arguments give the ocean.
        """
        got = self.value(masks().smooth_step(fd.Constant(x), 1.0))
        assert abs(got - expected) < 1e-15

    def test_smooth_step_derivative_is_finite_at_a_huge_argument(self):
        """The clamp keeps the first derivative finite for any `k x`."""
        mesh = fd.UnitSquareMesh(1, 1)
        x = fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(1e6)
        form = masks().smooth_step(x, 1e3) * fd.dx(domain=mesh)
        derivative = fd.assemble(fd.derivative(form, x))
        assert np.all(np.isfinite(derivative.dat.data_ro))

    def test_the_clamp_is_inside_the_bounds_that_make_it_correct(self):
        """`19 < SMOOTH_STEP_CLAMP < 177.8`.

        This is the guard on the constant, because no unit test of
        `smooth_step` alone can reproduce the failure it prevents. The failure
        is in the second derivative of the mask, which the Jacobian of the
        centre-of-mass frame column holds: UFL writes the derivative of
        `tanh(z)` as `(2 cosh(z) / (1 + cosh(2 z)))^2`, so differentiating
        again carries `cosh(2 z)^2`. Squared, that overflows above
        `z = 177.8`; an infinite value there multiplies the exact zero
        derivative of the clamped branch and gives a NaN, which stops the
        solve. Whether a given kernel forms the square or divides twice in
        sequence is a TSFC grouping decision, and the kernel of this unit test
        divides twice, so it stays finite at every clamp up to 350 and cannot
        fail. What the failure was measured on is the solver: at `grad_floor`
        7e-5 on the 78 km shelf, clamp 350 fails and clamp 170 converges.

        The lower bound is the value: `tanh` reaches exactly 1.0 in double
        precision at an argument of 19, so a smaller clamp would make the
        saturated masks inexact. Both bounds and the solver control are in
        `NOTES/findings/FINDING-alpha-floor-mass.md`.
        """
        assert 19.0 < masks().SMOOTH_STEP_CLAMP < 177.8

    def test_ocean_function_is_the_smooth_step_of_sea_level(self):
        m = masks()
        sl, k = 0.2, 2.0
        assert self.value(m.ocean_function(fd.Constant(sl), k)) == \
            pytest.approx(self.value(m.smooth_step(fd.Constant(sl), k)),
                          abs=1e-15)

    @pytest.mark.parametrize("ice, sl, expected", [
        # Thick ice on land: grounded, B = 0. Argument 50 + 10.7 = 60.7.
        (50.0, -10.0, 0.0),
        # Ice thinner than the water depth times rho_w/rho_i: floating, B = 1.
        # Argument 10 - 53.7 = -43.7.
        (10.0, 50.0, 1.0),
        # No ice in the open ocean: B = 1. Argument -53.7.
        (0.0, 50.0, 1.0),
    ])
    def test_grounded_ice_function_is_zero_only_under_grounded_ice(
            self, ice, sl, expected):
        """`B = 1 - H(I - (rho_w/rho_i) SL)` in the three saturated cases.

        `k = 1` and arguments of at least 40 in magnitude, so `|k x / 2|` is
        at least 20 and the tanh is within 1e-17 of 1.
        """
        got = self.value(masks().grounded_ice_function(
            fd.Constant(ice), fd.Constant(sl), 1.0, RHO_W, RHO_I))
        # 1e-14 absolute: the saturated tanh plus the round-off of the
        # unit-square quadrature weights.
        assert abs(got - expected) < 1e-14

    def test_grounded_ice_function_uses_the_density_ratio(self):
        """At `I = (rho_w/rho_i) SL` the ice is at flotation and `B = 1/2`."""
        sl = 3.0
        ice = RHO_W / RHO_I * sl
        got = self.value(masks().grounded_ice_function(
            fd.Constant(ice), fd.Constant(sl), 5.0, RHO_W, RHO_I))
        assert abs(got - 0.5) < 1e-12

    def test_mask_steepness_is_alpha_p_over_the_facet_length(self, meshes):
        """`k h = alpha p` at every point of Re, with `h = FacetArea` in 2-D.

        Integrated against `FacetArea` over Re, so the check needs no facet
        length of its own: `int k h ds = alpha p L`. 1e-12 relative, the
        round-off of an assembled integral of a constant.
        """
        _, sub = meshes
        dss = fd.Measure("ds", domain=sub)(CURVE_RE)
        k = masks().mask_steepness(sub, 2, ALPHA_MASK)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(k * fd.FacetArea(sub) * dss)
        assert got == pytest.approx(ALPHA_MASK * 2 * length, rel=1e-12)

    def test_a_frozen_slope_divides_the_steepness(self, meshes):
        """With a slope `s`, `k h s = alpha p`: the transition is fixed in space."""
        _, sub = meshes
        dss = fd.Measure("ds", domain=sub)(CURVE_RE)
        slope = fd.Function(fd.FunctionSpace(sub, "DG", 0)).assign(2.0)
        k = masks().mask_steepness(sub, 2, ALPHA_MASK, slope=slope)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(k * fd.FacetArea(sub) * slope * dss)
        assert got == pytest.approx(ALPHA_MASK * 2 * length, rel=1e-12)

    def test_a_zero_slope_is_floored(self, meshes):
        """A flat region must give a finite steepness, not a division by zero."""
        _, sub = meshes
        dss = fd.Measure("ds", domain=sub)(CURVE_RE)
        slope = fd.Function(fd.FunctionSpace(sub, "DG", 0))
        k = masks().mask_steepness(sub, 2, ALPHA_MASK, slope=slope)
        got = fd.assemble(k * dss)
        assert np.isfinite(got) and got > 0.0

    def test_a_live_slope_expression_is_refused(self, meshes):
        """The slope must be a frozen `Function`.

        A live `grad(SL)` on the tape adds a `1/|grad SL|^2` term to the
        adjoint that is not physics (handover section 4.2).
        """
        _, sub = meshes
        V = fd.FunctionSpace(sub, "CG", 1)
        sl = fd.Function(V).interpolate(fd.SpatialCoordinate(sub)[0])
        with pytest.raises((TypeError, ValueError)):
            masks().mask_steepness(sub, 2, ALPHA_MASK,
                                   slope=fd.sqrt(fd.dot(fd.grad(sl),
                                                        fd.grad(sl))))


# ---------------------------------------------------------------------------
# Layout and refusals
# ---------------------------------------------------------------------------

class TestLayout:
    """`self_gravitating_gia_space(..., sea_level=True)` and its refusals."""

    @staticmethod
    def space(meshes, **kwargs):
        parent, sub = meshes
        settings = dict(gravity_bcs=gravity_bcs(parent, sheet=False),
                        fluid_core=True, centre_of_mass=True,
                        self_gravity_number=LAMBDA)
        settings.update(kwargs)
        return self_gravitating_gia_space(sub, parent, **settings)

    @pytest.mark.parametrize("rotation", [False, True])
    def test_shift_is_the_last_field(self, meshes, rotation):
        """`Shift` is last in the space, after the centre-of-mass multipliers."""
        Z, layout = self.space(meshes, rotation=rotation, sea_level=True)
        assert layout.sea_level == len(Z) - 1

    @pytest.mark.parametrize("rotation", [False, True])
    def test_the_frame_multipliers_stay_just_before_the_shift(
            self, meshes, rotation):
        Z, layout = self.space(meshes, rotation=rotation, sea_level=True)
        assert layout.centre_of_mass == (len(Z) - 3, len(Z) - 2)

    @pytest.mark.parametrize("rotation", [False, True])
    def test_shift_is_a_real_field_in_the_accounting(self, meshes, rotation):
        """`n_fields` and `real_fields` count the new field, in space order."""
        Z, layout = self.space(meshes, rotation=rotation, sea_level=True)
        real = tuple(i for i, V in enumerate(Z)
                     if V.ufl_element().family() == "Real")
        assert len(Z) == layout.n_fields
        assert real == layout.real_fields
        assert real[-1] == layout.sea_level

    def test_shift_lives_on_the_parent(self, meshes):
        """Like every other `Real` field, so its rows assemble on parent measures."""
        Z, layout = self.space(meshes, sea_level=True)
        assert Z[layout.sea_level].mesh() is meshes[0]

    def test_absent_by_default(self, meshes):
        _, layout = self.space(meshes)
        assert layout.sea_level is None

    def test_refuses_sea_level_without_the_frame(self, meshes):
        """An ocean load always has degree-1 content; the frame must fix it."""
        with pytest.raises(ValueError, match="centre_of_mass"):
            self.space(meshes, centre_of_mass=False, sea_level=True)

    def test_refuses_the_lowrank_representation(self, meshes):
        """Sea level runs on the multiplier representation only.

        The centre-of-mass frame, which sea level requires, already refuses
        the low-rank path, so this test cannot tell which of the two refusals
        fired. It records that the combination is refused.
        """
        with pytest.raises((NotImplementedError, ValueError)):
            self.space(meshes, sea_level=True, dtn_representation="lowrank")

    def test_block1_diagonal_accounts_for_the_shift(self, meshes):
        """`block1_diagonal` describes the new row and does not raise.

        The value on the row is not asserted: the task asks for zero, but the
        assembled `(Shift, Shift)` entry is `-c B_mu g_0 rho_w int B C dS`,
        which is not zero. See the open questions of the handoff file.
        """
        solver, _, layout = build(meshes, surface_fields(meshes[1], "ocean"))
        diagonal = solver.block1_diagonal()
        assert len(diagonal) == len(layout.real_fields)
        assert np.all(np.isfinite(diagonal))


class TestRefusals:
    """The ice load is part of the sea-level sheet and must not be given twice."""

    def test_refuses_a_normal_stress_on_re(self, meshes):
        _, sub = meshes
        with pytest.raises(ValueError, match=r"(?i)sea.level"):
            build(meshes, surface_fields(sub, "ocean"),
                  surface_bcs={CURVE_RE: {"normal_stress": B_MU * 1e-3}})

    def test_refuses_a_gravity_sheet_on_re(self, meshes):
        parent, sub = meshes
        with pytest.raises(ValueError, match=r"(?i)sea.level"):
            build(meshes, surface_fields(sub, "ocean"), gravity_sheet=1e-3)

    def test_refuses_a_boundary_quadrature_degree_below_2p(self, meshes):
        """Degree 3 is below `2 p = 4` for the P2 displacement and potential.

        The calibration (`NOTES/findings/FINDING-mask-steepness.md`) needs `q >= 2 p` for the gradient
        through the masks. The factory's `quad_degree` sets the boundary
        degree that the sea-level measure uses.
        """
        _, sub = meshes
        with pytest.raises((AssertionError, ValueError), match=r"(?i)degree"):
            build(meshes, surface_fields(sub, "ocean"), quad_degree=3)

    def test_a_space_without_the_field_refuses_a_sea_level(self, meshes):
        """A `SeaLevel` with no `Shift` field has no row for mass conservation."""
        _, sub = meshes
        parent = meshes[0]
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent, sheet=False),
            fluid_core=True, centre_of_mass=True, self_gravity_number=LAMBDA)
        Xm = fd.SpatialCoordinate(sub)
        with pytest.raises(ValueError):
            SelfGravitatingGIASolver(
                fd.Function(Z),
                approximation(g=FRAME.gravity_of_r(fd.sqrt(fd.dot(Xm, Xm)))),
                layout=layout, dt=1.0, bcs={},
                fluid_core=FluidCore(boundary=CURVE_RC,
                                     rho_core=FRAME.RHO_CORE),
                sea_level=make_sea_level(surface_fields(sub, "ocean")))


class TestSeaLevelDataclass:

    @staticmethod
    def default_sea_level(sub):
        """A `SeaLevel` with every mask setting left at its default."""
        fields = surface_fields(sub, "ocean")
        return sea_level_class()(boundary=CURVE_RE, rho_w=RHO_W, rho_i=RHO_I,
                                 g_surface=G_RE, **fields)

    def test_the_mask_defaults_are_the_ones_the_masks_module_carries(
            self, meshes):
        """The defaults live in `sea_level_masks`, so there is one copy of each.

        The values are `alpha = 1` and `grad_floor = 1e-4`, from steps S1 and
        S2 of `NOTES/findings/FINDING-alpha-floor-mass.md`. The reasons are in
        the docstrings of the two constants.
        """
        _, sub = meshes
        m, sea_level = masks(), self.default_sea_level(meshes[1])
        assert sea_level.alpha_mask == m.DEFAULT_ALPHA_MASK == 1.0
        assert sea_level.grad_floor == m.DEFAULT_GRAD_FLOOR == 1e-4

    @pytest.mark.parametrize("floor", [1e-3, 1e-4])
    def test_grad_floor_reaches_the_mask_steepness(self, meshes, floor):
        """The `grad_floor` of `SeaLevel` reaches `mask_steepness`.

        With a frozen slope that is zero everywhere, `max(s, grad_floor)` is
        the floor, so `k = alpha p / (h grad_floor)` and
        `int k h ds = alpha p L / grad_floor` over the surface of length `L`.
        Two floors a factor of 10 apart are checked, so a steepness that
        ignored the keyword could not pass both. 1e-12 relative: the round-off
        of an assembled integral of a constant.

        Without this keyword a caller cannot set the floor at all, which is why
        the measurement scripts of step S2 had to reach it by scaling `alpha`
        and the slope field together.
        """
        _, sub = meshes
        fields = surface_fields(sub, "ocean")
        # Zero everywhere, so the floor is what the steepness divides by.
        slope = fd.Function(fd.FunctionSpace(sub, "DG", 0))
        solver, _, _ = build(meshes, fields, sea_level_overrides=dict(
            slope=slope, grad_floor=floor))
        dss = fd.Measure("ds", domain=sub)(CURVE_RE)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(solver._sea_level_steepness() * fd.FacetArea(sub)
                          * dss)
        assert got == pytest.approx(ALPHA_MASK * 2 * length / floor,
                                    rel=1e-12)


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
        warning (`fluid_core_measure` docstring). The reference is the same
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
        solver, z, layout = build(
            meshes, fields, sea_level_overrides={"g_surface": self.G_SURFACE})
        dss = re_measure(solver)
        n = fd.FacetNormal(sub)
        Xm = fd.SpatialCoordinate(sub)
        if which == "SL_init":
            fields["SL_init"].assign(0.3)
            expected = fd.Constant(0.3)
        elif which == "psi":
            # N = psi / g_surface with rotation off.
            z.subfunctions[layout.potential].assign(0.2)
            expected = fd.Constant(0.2 / self.G_SURFACE)
        elif which == "u":
            u = z.subfunctions[layout.displacement]
            u.interpolate(0.1 * Xm / fd.sqrt(fd.dot(Xm, Xm)))
            expected = -fd.dot(u, n)
        elif which == "Shift":
            z.subfunctions[layout.sea_level].assign(0.05)
            expected = fd.Constant(0.05)
        elif which == "N_init":
            fields["N_init"].assign(0.07)
            expected = fd.Constant(-0.07)
        else:
            fields["ur_init"].assign(0.04)
            expected = fd.Constant(0.04)
        got = fd.assemble(solver.sea_level() * dss)
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
        solver, z, layout = build(
            meshes, fields, rotation=True,
            sea_level_overrides={"g_surface": self.G_SURFACE})
        z.subfunctions[layout.rotation["m3"]].assign(10.0)
        dss = re_measure(solver)
        expected = (solver.rotational_potential_expression(sub)
                    / self.G_SURFACE)
        got = fd.assemble(solver.sea_level() * dss)
        want = fd.assemble(expected * dss)
        # Guard: the reference is not zero, so the comparison is not vacuous.
        assert abs(want) > 1e-3 * 2 * np.pi * RE
        assert got == pytest.approx(want, rel=1e-12)


class TestSurfaceLoadSheet:
    """`surface_load_sheet()` in saturated states, where it has a closed form."""

    S = 1e-2

    @pytest.mark.parametrize("state, density_per_unit", [
        # Open ocean, raised by Shift: a water sheet rho_w * Shift.
        ("ocean", RHO_W),
        # Land (C = 0): no water load whatever Shift is.
        ("land", 0.0),
    ])
    def test_the_water_sheet(self, meshes, state, density_per_unit):
        _, sub = meshes
        solver, z, layout = build(meshes, surface_fields(sub, state))
        z.subfunctions[layout.sea_level].assign(self.S)
        dss = re_measure(solver)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(solver.surface_load_sheet() * dss)
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
        solver, z, layout = build(meshes, fields)
        z.subfunctions[layout.sea_level].assign(self.S)
        dss = re_measure(solver)
        length = fd.assemble(fd.Constant(1.0) * dss)
        got = fd.assemble(solver.surface_load_sheet() * dss)
        assert got == pytest.approx(RHO_W * self.S * length, rel=1e-10)

    def test_grounded_ice_loads_with_its_thickness_change(self, meshes):
        """Under grounded ice the sheet is `rho_i dI` and no water."""
        _, sub = meshes
        d_ice = 0.5
        fields = surface_fields(sub, "cap", d_ice=d_ice)
        solver, z, layout = build(meshes, fields)
        z.subfunctions[layout.sea_level].assign(self.S)
        dss = re_measure(solver)
        cap = indicator(sub, "cap")
        expected = fd.assemble(
            (RHO_I * d_ice * cap + RHO_W * self.S * (1 - cap)) * dss)
        got = fd.assemble(solver.surface_load_sheet() * dss)
        assert got == pytest.approx(expected, rel=1e-10)

    def test_the_sheet_is_the_kendall_form_at_a_shoreline(self, meshes):
        """Live masks: the sheet is equations (2)-(3) of the handover, pointwise.

        The expected sheet is built from the masks module with
        `k = mask_steepness(sub, 2, 0.5)`, where 2 is the highest degree of
        the displacement and the potential. A random pointwise weight makes
        the comparison pointwise instead of a single mean. 1e-11 relative: the
        two sides are the same algebra evaluated by two kernels.
        """
        _, sub = meshes
        m = masks()
        fields = surface_fields(sub, "shoreline")
        solver, z, layout = build(meshes, fields)
        shift = 0.05
        z.subfunctions[layout.sea_level].assign(shift)
        k = m.mask_steepness(sub, 2, ALPHA_MASK)
        SL0, I, I0 = fields["SL_init"], fields["I"], fields["I_init"]
        SL = SL0 + shift

        def B(ice, sl):
            return m.grounded_ice_function(ice, sl, k, RHO_W, RHO_I)

        expected = (RHO_W * (B(I, SL) * m.ocean_function(SL, k) * SL
                             - B(I0, SL0) * m.ocean_function(SL0, k) * SL0)
                    + RHO_I * ((1 - B(I, SL)) * I - (1 - B(I0, SL0)) * I0))
        weight = fd.Function(fd.FunctionSpace(sub, "CG", 2))
        weight.dat.data[:] = np.random.default_rng(7).uniform(
            0.5, 1.5, weight.dat.data.shape)
        dss = re_measure(solver)
        got = fd.assemble(weight * solver.surface_load_sheet() * dss)
        want = fd.assemble(weight * expected * dss)
        assert got == pytest.approx(want, rel=1e-11)

    def test_the_sheet_integral_is_the_integrand_times_the_sheet(self, meshes):
        _, sub = meshes
        solver, z, layout = build(meshes, surface_fields(sub, "shoreline"))
        z.subfunctions[layout.sea_level].assign(0.05)
        X = fd.SpatialCoordinate(sub)
        got = fd.assemble(solver.surface_load_sheet_integral(X[0] ** 2))
        want = fd.assemble(X[0] ** 2 * solver.surface_load_sheet()
                           * re_measure(solver))
        assert got == pytest.approx(want, rel=1e-12)


# ---------------------------------------------------------------------------
# The consumers of the sheet
# ---------------------------------------------------------------------------

class TestConsumers:
    """The sheet reaches `mass_dipole_form`, `inertia_form` and the net-mass scale.

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
        solver, z, layout = build(meshes, surface_fields(sub, "cap"))
        z.subfunctions[layout.sea_level].assign(self.S)
        X = fd.SpatialCoordinate(sub)
        ocean = 1 - indicator(sub, "cap")
        dss = re_measure(solver)
        for i in range(2):
            expected = fd.assemble(RHO_W * self.S * ocean * X[i] * dss)
            got = fd.assemble(solver.mass_dipole_form(i))
            assert abs(expected) > 1e-3 * RHO_W * self.S
            assert got == pytest.approx(expected, rel=1e-10)

    def test_the_sheet_reaches_the_inertia_row(self, meshes):
        """`dI_33 = int sigma (x^2 + y^2) dS` from the sheet, rotation on."""
        _, sub = meshes
        solver, z, layout = build(meshes, surface_fields(sub, "ocean"),
                                  rotation=True)
        z.subfunctions[layout.sea_level].assign(self.S)
        X = fd.SpatialCoordinate(sub)
        expected = fd.assemble(RHO_W * self.S * (X[0] ** 2 + X[1] ** 2)
                               * re_measure(solver))
        got = fd.assemble(solver.inertia_form(2))
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
        solver, z, layout = build(meshes, surface_fields(sub, "ocean"))
        z.subfunctions[layout.sea_level].assign(self.S)
        sheet_mass = fd.assemble(RHO_W * self.S * re_measure(solver))
        # Guard: the hand-set state really carries a net sheet mass, so a zero
        # datum is a statement about the datum and not about the state.
        assert fd.assemble(solver.surface_load_sheet() * re_measure(solver)) \
            == pytest.approx(sheet_mass, rel=1e-10)
        solver.update_total_mass()
        assert abs(float(solver.source_mass)) <= 1e-12 * sheet_mass

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
        solver, z, layout = build(meshes, surface_fields(sub, "caps",
                                                         d_ice=d_ice))
        dss = re_measure(solver)
        caps = indicator(sub, "caps")
        ice_length = fd.assemble(caps * dss)
        ocean_length = fd.assemble((1 - caps) * dss)
        # The uniform ocean rise that balances the ice growth on the caps.
        conserving = -RHO_I * d_ice * ice_length / (RHO_W * ocean_length)
        z.subfunctions[layout.sea_level].assign(conserving)
        # A CMB displacement of 1e-9 (units of D) that vanishes at Re, so the
        # sea-level sheet does not see it through `u_r`.
        X = fd.SpatialCoordinate(sub)
        r = fd.sqrt(fd.dot(X, X))
        z.subfunctions[layout.displacement].interpolate(
            1e-9 * (RE - r) / (RE - RC) * X / r)

        core_measure = solver.fluid_core_measure()(CURVE_RC)
        core_net = abs(fd.assemble(solver.fluid_core_sheet() * core_measure))
        core_scale = fd.assemble(abs(solver.fluid_core_sheet()) * core_measure)
        sheet_scale = fd.assemble(abs(solver.surface_load_sheet()) * dss)
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

class TestRepeatedSolve:
    """A second solve with nothing changed gives the first solve's answer.

    `update_total_mass` runs at the start of `solve`. With the sea-level sheet
    in the datum, the first solve after an ice change reads the net mass of
    the unbalanced ice change, and `Shift` and the geoid are wrong on that
    solve only. The correctness review measured this on the configuration
    below (`review-implementation-review-correctness-r3.md`):

        solve 0: datum 1.29e-2   Shift -7.054760e-03
        solve 1: datum 2.36e-14  Shift -8.495220e-03
        solve 2: datum 2.73e-14  Shift -8.495220e-03

    so the defect is 1.44e-3 in `Shift`, 17 percent of its value. The geoid
    moved by the same amount in its ocean mean.
    """

    def test_a_second_solve_gives_the_same_shift_and_geoid(self, meshes):
        """`Shift` and `psi` agree between two successive solves to 1e-8 relative.

        Configuration of the correctness probe: `"caps"` state with
        `d_ice = 1e-2`, `WILL_EARTH`, full `Lambda`, direct preset, `TIGHT`.

        Tolerance 1e-8 relative. The Newton tolerance is `snes_rtol = 1e-10`
        and the outer Krylov solve runs to 1e-12, so a converged solve is
        repeatable far below 1e-8. The datum of the second solve differs from
        the first by the converged net mass of the core sheet (about 1e-14,
        against a sheet scale of 1e-2), which moves `Shift` by about 1e-15.
        The defect is 1.7e-1 relative, seven orders above the tolerance.

        The geoid is `psi / g_surface` with rotation off, so the whole
        potential field is compared. The datum sets the degree-0 part of
        `psi`, which is where a lagged datum shows.
        """
        _, sub = meshes
        d_ice = 1e-2
        solver, z, layout = build(meshes, surface_fields(sub, "caps",
                                                         d_ice=d_ice),
                                  earth=WILL_EARTH, solver_parameters="direct",
                                  solver_parameters_extra=TIGHT)
        solver.solve()
        assert solver_converged(solver)
        shift_first = real_value(z, layout.sea_level)
        psi_first = z.subfunctions[layout.potential].dat.data_ro.copy()

        solver.solve()
        # A second solve that starts converged can stop at iteration 0, which
        # PETSc reports as a positive reason as well.
        assert solver_converged(solver)
        shift_second = real_value(z, layout.sea_level)
        psi_second = z.subfunctions[layout.potential].dat.data_ro

        # Guard: a nonzero Shift, so a relative comparison has meaning.
        assert abs(shift_first) > 1e-3
        assert abs(shift_second - shift_first) <= 1e-8 * abs(shift_first)
        psi_scale = np.abs(psi_first).max()
        assert psi_scale > 0.0
        assert np.abs(psi_second - psi_first).max() <= 1e-8 * psi_scale


# ---------------------------------------------------------------------------
# Signs against the existing terms
# ---------------------------------------------------------------------------

class TestSheetSigns:
    """The ocean sheet enters the rows as an existing sheet of the same density.

    The comparison solver has no sea level: its load is an `interior_sigma` of
    density `rho_w S` on the potential and a `normal_stress` of
    `B_mu g(Re) rho_w S` on the mechanics, which is the convention of every
    load in `test_gia_gravity.py`. The sea-level solver has the uniform deep
    ocean and `Shift = S`, so its sheet is `rho_w S`. At `u = psi = 0` the
    residual rows of both are the load terms alone.

    This is the check that `NOTES/DESIGN-SEA-LEVEL.md` asks for: a flipped
    geoid sign in `sea_level()` flips the sign of the ocean sheet in the
    potential row against every other sheet, while all symmetry tests still
    pass.
    """

    S = 1e-2

    def test_the_potential_row_matches_an_interior_sigma(self, meshes):
        _, sub = meshes
        sea, z_sea, layout = build(meshes, surface_fields(sub, "ocean"))
        z_sea.subfunctions[layout.sea_level].assign(self.S)
        row_sea = fd.assemble(sea.sea_level_residual()).subfunctions[
            layout.potential].dat.data_ro

        plain, _, plain_layout = build(meshes, sea_level=False,
                                       gravity_sheet=RHO_W * self.S)
        row_plain = fd.assemble(plain.potential_residual()).subfunctions[
            plain_layout.potential].dat.data_ro
        scale = np.abs(row_plain).max()
        assert scale > 0.0
        # 1e-10 relative: the same facets, the same `avg` and the same degree.
        # The two spellings agreed to the last bit for the fluid core
        # (`fluid_core_measure` docstring). The DEEP datum costs 3e-13.
        assert np.abs(row_sea - row_plain).max() <= 1e-10 * scale

    def test_the_mechanics_row_matches_a_normal_stress(self, meshes):
        _, sub = meshes
        sea, z_sea, layout = build(meshes, surface_fields(sub, "ocean"))
        z_sea.subfunctions[layout.sea_level].assign(self.S)
        row_sea = fd.assemble(sea.sea_level_residual()).subfunctions[
            layout.displacement].dat.data_ro

        plain, _, plain_layout = build(
            meshes, sea_level=False,
            surface_bcs={CURVE_RE: {"normal_stress": B_MU * G_RE * RHO_W
                                    * self.S}})
        row_plain = fd.assemble(plain.F).subfunctions[
            plain_layout.displacement].dat.data_ro
        scale = np.abs(row_plain).max()
        assert scale > 0.0
        # 1e-6 relative: `Equation` integrates the normal stress at degree 6
        # and the sea-level measure uses the boundary degree 10. Measured for a
        # `cos 2 phi` sheet the two rows differ by 4.0e-9 relative. A sign
        # error is a relative difference of 2.
        assert np.abs(row_sea - row_plain).max() <= 1e-6 * scale


# ---------------------------------------------------------------------------
# Symmetry and the energy
# ---------------------------------------------------------------------------

_JACOBIANS = {}


def jacobian(meshes, state, rotation=False):
    """The assembled nest Jacobian of the whole residual at a perturbed state.

    Cached per `(state, rotation)`, because one assembly takes about 7 s and
    the transpose tests read many blocks of it. With `rotation=True` the space
    carries the rotation scalar `m3`, and the perturbation also sets it.

    `"cap"`: masks saturated, so the mask derivatives vanish (below 1e-40) and
    the Jacobian is the fixed-mask one. `"shoreline"`: live masks. The sheet
    depends on `(u, psi, Shift)` only through the one scalar `Delta`, so the
    exact Jacobian is still symmetric (`NOTES/DESIGN-SEA-LEVEL.md`).

    The perturbation is 1e-3, so the shoreline masks stay fractional and the
    saturated masks stay saturated.
    """
    key = (state, rotation)
    if key not in _JACOBIANS:
        _, sub = meshes
        solver, z, layout = build(meshes, surface_fields(sub, state,
                                                         d_ice=0.5),
                                  rotation=rotation)
        perturb(z, 1e-3, seed=11)
        A = fd.assemble(fd.derivative(solver.F, z), mat_type="nest").petscmat
        _JACOBIANS[key] = (A, layout)
    return _JACOBIANS[key]


def _pair_indices(layout, pair):
    """Field indices of a named block pair. `"rot"` is the 2-D scalar `m3`."""
    names = {"u": layout.displacement, "psi": layout.potential,
             "Shift": layout.sea_level,
             "lambda_x": layout.centre_of_mass[0],
             "lambda_y": layout.centre_of_mass[1]}
    if layout.rotation:
        names["rot"] = layout.rotation["m3"]
    return names[pair[0]], names[pair[1]]


@pytest.mark.skipif(fd.COMM_WORLD.size > 1,
                    reason="the per-block instrument needs global dense blocks")
class TestTransposes:
    """Every coupling block the sheet creates is the transpose of its partner.

    1e-14 relative to the larger block, as the frame and fluid-core saddle
    tests use: both blocks are variations of one energy and differ by
    round-off only.
    """

    @pytest.mark.parametrize("state", ["cap", "shoreline"])
    @pytest.mark.parametrize("pair", [
        ("u", "psi"), ("u", "Shift"), ("psi", "Shift"),
        ("u", "lambda_x"), ("u", "lambda_y"),
        ("psi", "lambda_x"), ("psi", "lambda_y"),
        ("Shift", "lambda_x"), ("Shift", "lambda_y"),
    ])
    def test_the_pair_transposes(self, meshes, state, pair):
        A, layout = jacobian(meshes, state)
        i, j = _pair_indices(layout, pair)
        ij = dense_block(A, i, j)
        ji = dense_block(A, j, i)
        assert ij is not None and ji is not None, (
            f"block {pair} or its transpose is structurally absent")
        scale = max(np.abs(ij).max(), np.abs(ji).max())
        assert scale > 0.0
        assert np.abs(ij - ji.T).max() <= 1e-14 * scale


@pytest.mark.skipif(fd.COMM_WORLD.size > 1,
                    reason="the per-block instrument needs global dense blocks")
class TestRotationTransposes:
    """The rotation row against the sheet, on the live-mask shoreline state.

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
        A, layout = jacobian(meshes, "shoreline", rotation=True)
        i, j = _pair_indices(layout, pair)
        ij = dense_block(A, i, j)
        ji = dense_block(A, j, i)
        assert ij is not None and ji is not None, (
            f"block {pair} or its transpose is structurally absent")
        scale = max(np.abs(ij).max(), np.abs(ji).max())
        assert scale > 0.0
        assert np.abs(ij - ji.T).max() <= 1e-14 * scale

    def test_the_shift_rotation_block_is_nonzero(self, meshes):
        """`(Shift, m3)` and `(m3, Shift)` both carry the sheet.

        Without the sheet in `inertia_form` the `(m3, Shift)` entry is zero,
        and without `psi_rot` in `sea_level()` the `(Shift, m3)` entry is zero.
        The transpose test above only needs one of them to be nonzero, so this
        test asserts both. The size of either entry is about
        `c B_mu Omega_sq rho_w int B C p_3 dS`, of order 1e-3 before the
        residual scaling. The threshold is 1e-10 of the `(Shift, Shift)`
        diagonal, far below that and far above round-off.
        """
        A, layout = jacobian(meshes, "shoreline", rotation=True)
        i_s, i_r = _pair_indices(layout, ("Shift", "rot"))
        shift_shift = abs(dense_block(A, i_s, i_s)[0, 0])
        assert shift_shift > 0.0
        assert abs(dense_block(A, i_s, i_r)[0, 0]) > 1e-10 * shift_shift
        assert abs(dense_block(A, i_r, i_s)[0, 0]) > 1e-10 * shift_shift


class TestResidualIsTheVariationOfTheEnergy:

    def test_written_residual_equals_the_derivative_of_the_energy(self, meshes):
        """At saturated masks, `sea_level_residual() = derivative(E_sl)`.

        The residual is written term by term for the Slate reason of handover
        B trap 3.4.2. This pins that it is still the variation of the energy,
        on a state where every unknown is nonzero. The `"cap"` state with an
        ice change exercises the ice term of `G`. Amplitude 0.1, so `Delta` is
        of order 0.1 and the DEEP datum costs 25 * 1e-16 / 0.1 = 3e-14
        relative. Tolerance 1e-11 relative per sub-field.
        """
        _, sub = meshes
        solver, z, layout = build(meshes, surface_fields(sub, "cap",
                                                         d_ice=0.5))
        perturb(z, 1e-1, seed=5)
        written = fd.assemble(solver.sea_level_residual())
        derived = fd.assemble(fd.derivative(solver.sea_level_energy(),
                                            solver.solution))
        checked = 0
        for i, (a, b) in enumerate(zip(written.subfunctions,
                                       derived.subfunctions)):
            scale = np.abs(b.dat.data_ro).max()
            if i in (layout.displacement, layout.potential, layout.sea_level):
                assert scale > 0.0, f"sub-field {i} has no sea-level term"
            tolerance = 1e-11 * max(scale, 1e-300)
            assert np.abs(a.dat.data_ro - b.dat.data_ro).max() <= tolerance
            checked += 1
        assert checked == len(z.subfunctions)


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

class TestEustaticLimit:

    def test_the_shift_is_the_eustatic_value(self, meshes):
        """`Shift = -(rho_i/rho_w) int dI dS / L_ocean` on a stiff, non-gravitating Earth.

        `C = 1` everywhere (deep ocean), grounded ice melts on the two
        opposite caps (`"caps"`, no degree-1 content), `Lambda = 1e-6` and a
        stiff elastic Earth. `L_ocean` is the length of Re outside the caps,
        which is half of `2 pi Re` here: the ocean is where water can stand,
        and under the grounded ice `B = 0`.

        Tolerance 1e-4 relative. Measured with the same load applied as a
        normal stress and a sheet on the current solver: the ocean mean of
        `du_r` is 6e-10 at shear modulus 1e6, and the geoid change scales with
        `Lambda` to about 7e-11, against a eustatic value of 9e-4. The
        remaining error is below 1e-6. A missing `rho_i/rho_w` is 7 percent,
        and dividing by `2 pi Re` instead of the ocean length is a factor 2.
        """
        _, sub = meshes
        d_ice = -1e-2
        solver, z, layout = build(
            meshes, surface_fields(sub, "caps", d_ice=d_ice), lam=1e-6,
            stiff=True, solver_parameters="direct",
            solver_parameters_extra=TIGHT)
        solver.solve()
        assert solver_converged(solver)
        dss = re_measure(solver)
        caps = indicator(sub, "caps")
        eustatic = (-(RHO_I / RHO_W) * fd.assemble(d_ice * caps * dss)
                    / fd.assemble((1 - caps) * dss))
        assert eustatic > 0.0  # melting raises the sea
        assert real_value(z, layout.sea_level) == pytest.approx(eustatic,
                                                                rel=1e-4)


@pytest.fixture(scope="module")
def shoreline_solved(meshes):
    """One live-mask solve of the shoreline state, shared by read-only tests.

    Direct preset with `TIGHT` tolerances. The residual norm at the zero state
    is recorded before the solve, for the Newton part of the thresholds of the
    `Shift` row and of the net sheet mass.
    """
    _, sub = meshes
    solver, z, layout = build(meshes, surface_fields(sub, "shoreline"),
                              solver_parameters="direct",
                              solver_parameters_extra=TIGHT)
    initial_norm = vector_norm(fd.assemble(solver.F))
    solver.solve()
    iterations = solver.solver.snes.getIterationNumber()
    print(f"shoreline live-mask solve: {iterations} Newton iterations, "
          f"reason {solver.solver.snes.getConvergedReason()}")
    return solver, z, layout, initial_norm


class TestLiveMaskSolve:

    def test_it_converges(self, shoreline_solved):
        solver, _, _, _ = shoreline_solved
        assert solver_converged(solver)

    def test_newton_takes_at_most_five_iterations(self, shoreline_solved):
        """`NOTES/DECISIONS.md`: more than 4 to 5 means the exact live-mask Jacobian is wrong.

        Logged as well, by the fixture.
        """
        solver, _, _, _ = shoreline_solved
        assert 1 <= solver.solver.snes.getIterationNumber() <= 5

    def test_mass_is_conserved_to_order_lambda(self, shoreline_solved):
        """`|int sigma dS|` is bounded by the frame multipliers, not by Newton.

        The `Shift` row of the residual is

            -c B_mu g_s int sigma dS + c B_mu sum_i lambda_i int x_i dsigma/dShift dS

        because `centre_of_mass_residual` keeps the column `lambda . dD/dShift`
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

        `C = 2`. Calibration on this fixture: `lambda = (4.45e-9, 2.43e-9)`,
        `int |x_i dsigma/dShift| = (2.34, 2.27)`, `int sigma = 7.36e-9`, which
        the identity above reproduces to four digits and which is 0.74 of the
        first term with `C = 1`. A missing `Shift` row or a wrong scale on
        either term gives a net mass of order the ice change, 1e-2 of the sheet
        scale, far above the bound. A flipped sign on one term alone stays
        inside the bound; `TestTransposes` and
        `TestResidualIsTheVariationOfTheEnergy` catch that.

        The bound is checked to be small against the sheet itself, so the test
        cannot pass with a large multiplier.
        """
        solver, z, layout, initial_norm = shoreline_solved
        scale_factor = gia_gravity.scalar_value(solver.scaling_factor)
        row_scale = scale_factor * B_MU * G_RE
        dss = re_measure(solver)
        sigma = solver.surface_load_sheet()
        # `dsigma/dShift`: the Gateaux derivative of the sheet in the direction
        # of the unit `Shift`, with live masks.
        direction = fd.Function(z.function_space())
        direction.subfunctions[layout.sea_level].assign(1.0)
        dsigma_dshift = expand_derivatives(
            ufl.derivative(sigma, solver.solution, direction))
        X = fd.SpatialCoordinate(solver.mesh)
        lam = [real_value(z, index) for index in layout.centre_of_mass]
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
        solver, z, layout, initial_norm = shoreline_solved
        residual = fd.assemble(solver.F)
        row = float(residual.subfunctions[layout.sea_level].dat.data_ro[0])
        bound = 10 * max(SNES_RTOL * initial_norm, SNES_ATOL)
        scale_factor = gia_gravity.scalar_value(solver.scaling_factor)
        row_scale = scale_factor * B_MU * G_RE
        magnitude = fd.assemble(abs(solver.surface_load_sheet())
                                * re_measure(solver))
        assert bound < 1e-6 * row_scale * magnitude
        assert abs(row) <= bound

    def test_the_centre_of_mass_includes_the_sheet_and_holds(
            self, shoreline_solved):
        """After the solve `D = 0`, with `D` including the moving ocean.

        Scale: the first moment of the sheet alone, `int |x_i sigma| dS`, which
        the displacement has to cancel. 1e-8 of it, as the frame test uses.
        """
        solver, _, _, _ = shoreline_solved
        X = fd.SpatialCoordinate(solver.mesh)
        sigma = solver.surface_load_sheet()
        dss = re_measure(solver)
        D = solver.mass_dipole()
        for i in range(2):
            moment = fd.assemble(abs(X[i] * sigma) * dss)
            assert moment > 0.0
            assert abs(D[i]) <= 1e-8 * moment


class TestSignAgainstPhysics:

    def test_water_piles_up_towards_added_ice_on_a_stiff_earth(self, meshes):
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
        solver, z, layout = build(meshes, fields, stiff=True,
                                  solver_parameters="direct",
                                  solver_parameters_extra=TIGHT)
        solver.solve()
        assert solver_converged(solver)
        dss = re_measure(solver)
        delta = solver.sea_level() - fields["SL_init"]
        near, far = indicator(sub, "near"), indicator(sub, "far")
        mean_near = fd.assemble(near * delta * dss) / fd.assemble(near * dss)
        mean_far = fd.assemble(far * delta * dss) / fd.assemble(far * dss)
        eustatic = RHO_I / RHO_W * d_ice
        assert mean_near - mean_far > 1e-3 * eustatic


def load_will_sea_level():
    """Will Scott's `sea_level.py` as a module, or a skip if it is absent."""
    if not WILL_SEA_LEVEL.exists():
        pytest.skip(f"Will Scott's SeaLevelSolver is not at {WILL_SEA_LEVEL}. "
                    "It is untracked comparison code under NOTES/; copy it "
                    "from ~/Workplace/sl_testing/globe_test/sea_level.py.")
    spec = importlib.util.spec_from_file_location("will_sea_level",
                                                  WILL_SEA_LEVEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The Earth of the Will Scott comparison: an elastic step (viscosity 1e18,
#: so the Maxwell time is far above `dt = 1`) with shear and bulk modulus 30.
#: Chosen so that the ocean means of the geoid change and of the uplift are
#: both a large part of the non-eustatic remainder of the shift. Measured on
#: the current solver with the fixed-shoreline load `rho_i dI (2 caps - 1)`
#: given as a sheet: `|mean_dN| / |rest| = 0.59`, `|mean_ur| / |rest| = 0.41`,
#: `|rest| / |eustatic| = 0.086`. The default Earth (viscosity 1, modulus 1,
#: `dt = 1`) gave 0.058 for the geoid, below the guard of 0.1; modulus 10 gave
#: 0.35 and 0.65, modulus 100 gave 0.82 and 0.18.
WILL_EARTH = dict(shear_modulus=30.0, bulk_modulus=30.0, viscosity=1e18)


class TestAgainstWillScott:

    def test_the_shift_matches_the_separate_surface_solver(self, meshes):
        """Given our `dN` and `du_r`, Will's shift equals our `Shift`.

        Fixed shorelines: a uniform deep ocean (Will's `C` is 1) and grounded
        ice growing on the two caps, a degree-2 load. The Earth is
        `WILL_EARTH` and `Lambda` is full. The coupled solve gives `Shift`,
        `psi` and `u`. Will's `SeaLevelSolver` runs on a 32-segment circle
        whose vertices are the Re vertices, with `dphi = psi / g(Re)` and
        `du_r = u . rhat` interpolated onto CG2, and returns its `shift`.

        The two shifts share the eustatic part `-(rho_i/rho_w) dI L_ice /
        L_ocean`, which is identical on both meshes because the caps cover
        half the circle on both. The comparison is on the remainder
        `rest = -mean_ocean(dN - du_r)`, which carries the sign of the geoid
        and of the uplift.

        Both codes read the same `psi` and `u`. A flipped sign of `dN` in our
        `sea_level()` therefore moves our remainder by `2 |mean_ocean(dN)|`,
        and a flipped `du_r` by `2 |mean_ocean(du_r)|`. The guards require
        each ocean mean to exceed 0.1 of `|rest|`, so either flip moves the
        remainder by at least 20 percent, four times the tolerance. The guards
        are computed from the solved fields, so they hold for the state the
        comparison uses and not only for the probe.

        Tolerance 5 percent of the remainder: the two computations differ by
        the polygon geometry and the CG2 interpolation, both of order `h^2`
        (about 1e-3 relative of the eustatic value, that is about 1 percent
        of the remainder here).
        """
        will = load_will_sea_level()
        parent, sub = meshes
        d_ice = 1e-2
        fields = surface_fields(sub, "caps", d_ice=d_ice)
        solver, z, layout = build(meshes, fields, earth=WILL_EARTH,
                                  solver_parameters="direct",
                                  solver_parameters_extra=TIGHT)
        solver.solve()
        assert solver_converged(solver)
        shift_ours = real_value(z, layout.sea_level)

        circle = fd.CircleManifoldMesh(32, radius=RE, degree=1)
        V = fd.FunctionSpace(circle, "CG", 2)
        Vv = fd.VectorFunctionSpace(circle, "CG", 2)
        psi_c = fd.assemble(fd.interpolate(
            solver.solution.subfunctions[layout.potential], V))
        u_c = fd.assemble(fd.interpolate(
            solver.solution.subfunctions[layout.displacement], Vv))
        Xc = fd.SpatialCoordinate(circle)
        du_r = fd.Function(V).interpolate(
            fd.dot(u_c, Xc / fd.sqrt(fd.dot(Xc, Xc))))
        dphi = fd.Function(V).interpolate(psi_c / G_RE)
        caps_c = indicator(circle, "caps")
        Ih0 = fd.Function(caps_c.function_space()).interpolate(
            GROUNDED * caps_c)
        Ih = fd.Function(caps_c.function_space()).interpolate(
            (GROUNDED + d_ice) * caps_c)
        SL = fd.Function(V).assign(DEEP)
        will_solver = will.SeaLevelSolver(SL, Ih, Ih0, du_r, dphi, ALPHA_MASK)
        will_solver.update_shorelines()
        shift_will = float(will_solver.shift)

        ratio = will.SeaLevelSolver.rho_ice / will.SeaLevelSolver.rho_water
        assert ratio == pytest.approx(RHO_I / RHO_W, rel=1e-12)
        eustatic = -ratio * d_ice  # the caps are half of the circle
        rest_ours = shift_ours - eustatic
        rest_will = shift_will - eustatic
        assert abs(rest_will) > 1e-2 * abs(eustatic)

        # Guards: each term of the remainder alone is large enough that a flip
        # of its sign in `sea_level()` fails the 5 percent comparison below.
        # The ocean is the complement of the caps, with the weight that Will's
        # `B C` has for these fixed shorelines.
        ocean_c = 1 - caps_c
        ocean_length = fd.assemble(ocean_c * fd.dx(domain=circle))
        mean_dN = fd.assemble(ocean_c * dphi * fd.dx(domain=circle)) \
            / ocean_length
        mean_ur = fd.assemble(ocean_c * du_r * fd.dx(domain=circle)) \
            / ocean_length
        assert abs(mean_dN) > 0.1 * abs(rest_will)
        assert abs(mean_ur) > 0.1 * abs(rest_will)

        assert abs(rest_ours - rest_will) <= 5e-2 * abs(rest_will)


# ---------------------------------------------------------------------------
# Preconditioner and rotation
# ---------------------------------------------------------------------------

class TestNestedCondensation:

    def test_internal_variable_scpc_builds_and_solves(self, meshes):
        """The iterative preset condenses `(u, M)` in Slate with sea level on.

        With live masks the `(u, u)` block carries parent-mesh coefficients
        through the masks. `InternalVariableSCPC` compiles that block with
        Slate, which needs a single mesh (`NOTES/DESIGN-SEA-LEVEL.md`), so this is
        where a residual that is not written term by term fails.
        """
        from test_gia_nested_condensation import condensation_context

        _, sub = meshes
        # Two settings are named, and both are named for a reason.
        #
        # `dtn_representation`: sea level runs on the multiplier representation,
        # and the preset's own default is the low-rank one on the full layout.
        # Leaving it unset puts `gadopt.LowRankPotentialPC` on the potential
        # split of a solver that carries no low-rank update, and
        # `_check_representation_matches_parameters` refuses that pairing.
        #
        # `block0`: `InternalVariableSCPC` is reached through the `"pair"`
        # route, whose block 0 is a multiplicative fieldsplit. The default
        # route is `gadopt.CondensedBlockPC`, a single preconditioner with no
        # fieldsplit to descend into, so `condensation_context` below cannot
        # find its sub-KSP there. `test_gia_nested_condensation.py` names the
        # same value for the same reason. The default route has its own test
        # underneath this one.
        parameters = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
            outer_rtol=1e-8, block0_rtol=1e-4, snes_rtol=1e-8,
            dtn_representation="multiplier", block0="pair")
        solver, z, layout = build(meshes, surface_fields(sub, "shoreline"),
                                  solver_parameters=parameters)
        solver.solve()
        assert solver_converged(solver)
        condensation_context(solver)

    def test_the_condensed_block0_route_solves_with_sea_level(self, meshes):
        """The default block-0 route condenses `(u, M)` in Slate with sea level on.

        `gadopt.CondensedBlockPC` eliminates the internal variables once per
        application, on the `(u, M)` sub-form of block 0. With sea level on that
        sub-form carries the ocean column of the load, an exterior-facet
        integral of the submesh whose integrand holds `avg(psi)` from the parent
        mesh. Slate compiles kernels of one mesh and meets that restriction as
        `KeyError: 'facet_1'`, which inside a pytest process arrives as a
        segmentation fault, so the failure this test guards against takes the
        whole run with it.

        `gadopt.preconditioners.single_domain_slate_form` leaves that one
        integral out of the form the class hands to Slate. Everything the class
        computes is preconditioner work, so the missing term changes the
        convergence rate and not the solution.
        """
        _, sub = meshes
        parameters = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
            outer_rtol=1e-8, block0_rtol=1e-4, snes_rtol=1e-8,
            dtn_representation="multiplier")
        assert parameters["dtn_fieldsplit_0_pc_python_type"] == \
            "gadopt.CondensedBlockPC"
        solver, z, layout = build(meshes, surface_fields(sub, "shoreline"),
                                  solver_parameters=parameters)
        solver.solve()
        assert solver_converged(solver)


def taylor_forward(control, meshes, *, rotation):
    """Tape one live-mask solve of the shoreline state with ice thickness `control`.

    The functional reads the degree-1 radial displacement at Re and `Shift`,
    so that it depends on the masks through mass conservation and through the
    ocean load. The weights make the two parts of similar size.
    """
    parent, sub = meshes
    with stop_annotating():
        fields = surface_fields(sub, "shoreline")
        fields["I"] = control
        solver, _, layout = build(meshes, fields, rotation=rotation,
                                  solver_parameters="direct",
                                  solver_parameters_extra=TIGHT)
    solver.solve()
    u = fd.split(solver.solution)[layout.displacement]
    shift = fd.split(solver.solution)[layout.sea_level]
    X = fd.SpatialCoordinate(sub)
    rhat = X / fd.sqrt(fd.dot(X, X))
    ds_re = fd.Measure("ds", domain=sub)(CURVE_RE)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    return fd.assemble(
        1e3 * fd.dot(u, rhat) * fd.cos(fd.atan2(X[1], X[0])) * ds_re
        + 1e4 * shift * shift * dx_m)


def run_taylor(meshes, rotation):
    """The guarded Taylor test with control `I` and a direction that moves the grounding line."""
    from test_gravity_adjoint import assert_taylor_with_guards

    _, sub = meshes
    V = fd.FunctionSpace(sub, "CG", 2)
    phi = polar_angle(sub)
    with stop_annotating():
        control = fd.Function(V).interpolate(
            (1.0 - MELT) * 0.6 * fd.max_value(0.0, -fd.cos(phi - PHASE) - 0.4))
        # Thickening on one flank and thinning on the other, of 0.1 at most,
        # so the grounding line moves and the B mask derivative is exercised.
        h = fd.Function(V).interpolate(0.1 * fd.sin(phi - PHASE))
    continue_annotation()
    try:
        m = Control(control)
        J = taylor_forward(control, meshes, rotation=rotation)
        Jhat = ReducedFunctional(J, m)
    finally:
        pause_annotation()
    # min_rate 1.9: "rate close to 2" of the task, the threshold the frame
    # adjoint test uses. The guards also check that the gradient is nonzero
    # and that the same test without the gradient gives rate 1.
    assert_taylor_with_guards(Jhat, control, h, J, min_rate=1.90)


class TestTaylor:

    def test_live_mask_taylor_rate_with_ice_thickness_control(self, meshes):
        run_taylor(meshes, rotation=False)


class TestRotation:
    """Rotation on. Acceptance (`NOTES/DECISIONS.md`): it solves and the Taylor test passes."""

    def test_the_rotating_case_solves(self, meshes):
        _, sub = meshes
        solver, z, layout = build(meshes, surface_fields(sub, "shoreline"),
                                  rotation=True, solver_parameters="direct",
                                  solver_parameters_extra=TIGHT)
        solver.solve()
        assert solver_converged(solver)
        assert np.isfinite(real_value(z, layout.sea_level))

    def test_the_rotating_case_passes_the_taylor_test(self, meshes):
        run_taylor(meshes, rotation=True)
