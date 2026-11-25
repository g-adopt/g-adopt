"""Helpers of `tests/unit/test_sea_level.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_sea_level.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import importlib
import firedrake as fd
import numpy as np
from gadopt.gia_couplings import (
    CentreOfMass,
    FluidCore,
    Rotation,
    SeaLevel,
    SurfaceLoad,
)
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.gia_solver import SelfGravitatingGIASolver
from gia_models.annulus import (
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    RC,
    RE,
    approximation,
    gravity_bcs,
)


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

    The delegating path under `schur_fact_type lower` does not converge here:
    on the low-rank arm of
    `TestFixedOcean::test_the_taylor_rate_with_ice_thickness_control` the
    adjoint solve reaches the outer cap of 200 iterations, while the forward
    solve of the same configuration converges. The cause is not known.

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
