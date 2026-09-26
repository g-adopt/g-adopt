r"""The Martinec et al. (2018) sea-level benchmark in 3-D, cases B, C and D.

THE BENCHMARK
    Martinec, Z., et al. (2018), A benchmark study of numerical
    implementations of the sea level equation in GIA modelling, Geophys. J.
    Int. 215(1), 389-414.

    The benchmark loads the Earth model M3-L70-V01 of Spada et al. (2011)
    with an ice cap and solves the sea-level equation on a prescribed ocean
    basin. This driver solves the coupled problem in one monolithic system:
    viscoelastic mechanics, the perturbed gravitational potential, the
    centre-of-mass frame and the sea-level equation. The Earth model, the
    mesh, the time steps and the solver settings are in `selfgrav_common.py`.

        case   ocean            ice            time        basin
        B      fixed coastline  L1, 1500 m     T0, step    B1, far from the ice
        C      fixed coastline  L2, 500 m      T1, growth  B2, under the ice
        D      moving coastline L2, 500 m      T1, growth  B2

    Case A has no ocean, and the Spada driver covers it. Case E prescribes
    the topography at the end and is an inverse problem.

    Every physical constant of a case comes from the PyPI package `giamip`
    (`giamip.case("martinec2018-<case>")`), never from this file, because
    Martinec and GIAMIP use different densities. The case files are inside
    the package, so the run needs no data file and no network.

THE SEA-LEVEL EQUATION AS THIS CODE SOLVES IT
    `gadopt.SeaLevel` puts the ocean and the ice into one surface load and
    solves the sea level inside the one system, with the uniform shift as a
    `Real` unknown:

        SL    = SL_init + (N - N_init) - (u_r - ur_init) + Shift
        sigma = rho_w C0 (SL - SL_init) + rho_i (1 - C0) (I - I_init)

    for a fixed coastline (cases B and C, `C0 = C(SL_init)`, Martinec eq. 7
    and 8), and with the live ocean and grounded-ice masks for a moving
    coastline (case D). `N = psi / g_s` is the geoid height and `u_r = u . n`
    the uplift. The `Shift` row is the conservation of the mass of water and
    ice. The centre-of-mass multipliers hold the first mass moment of mantle,
    core and load at zero, which is the frame of the benchmark. Rotation is
    off.

        giamip    meaning                       here
        U         vertical displacement, up     dot(u, n) on Re
        N         geoid displacement, up        psi / g_surface
        S         sea-surface variation         N + Shift
        RSL       relative sea level            S - U
        h_UF      uniform water layer           Shift

THE LOAD AND THE BASIN
    The ice cap is the Spada cap, moved and rescaled:

        h(gamma, t) = h0(t) sqrt((cos gamma - cos alpha(t)) / (1 - cos alpha(t)))

    for gamma < alpha(t) and zero outside, with gamma the angular distance
    from the cap centre at colatitude 25, longitude 75 degrees. Scenario T0
    applies the full cap at t = 0 and holds it. Scenario T1 grows h0 and
    alpha linearly from zero to their full values over 10 kyr and then holds
    them. The thickness is a UFL expression of position and of a time
    `Constant`, so it is exact at every quadrature point, the infinite slope
    at the cap margin included.

    The initial topography is the ocean basin of the benchmark,
    zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2)), with psi the angular
    distance from the basin centre and sigma_b 26 degrees. The initial sea
    level is SL_init = -zeta0.

DISCRETISATION
    Mesh      `selfgrav_common.MESHES["martinec"]`, generated in the job:
              250 km lateral size with two 35 km lithosphere layers, refined
              to 78 km in the ice cap and along both initial coastlines.
    Unknowns  As the Spada driver, plus the three centre-of-mass rows and
              the `Shift` row.
    Elastic   K / mu = 1000 for case B and 100 for cases C and D
              (`--bulk_shear_ratio` overrides it). At 100, case B fails its
              geoid criterion: the error of the compressible model at
              K / mu = 100 adds to the time error of backward Euler.
    Time      Case B: the elastic solve at t = 0, then uniform 10 yr steps
              to 10 kyr (1000 steps, `T0_DT_YR`). Cases C and D: 50 yr steps
              from 0 to 15 kyr, with no elastic solve, because the load
              grows from zero.
    Solver    Cases B and C have a fixed coastline, so the residual is linear
              in the unknowns and one linear solve per step is exact
              (`ksponly`). Case D has live masks and runs Newton.

RUN
    mpiexec -np <N> python3 martinec.py --case B

    `--smoke` runs the same code on a very coarse mesh for 10 steps. It
    checks that the code runs; its numbers are not results. Case D needs the
    10 steps: at the end of the first 50 yr step the load is still too small
    to assemble to a nonzero sheet.

OUTPUT (in --output_path)
    params_<case>.log          One line per time step: the time, the wall
                               time, the iteration counts, h_UF, the ocean
                               area, the ice masses and the mass moment.
    martinec-<case>-profiles_<t>kyr.npz
                               U, N, S and RSL in metres along the two
                               comparison meridians, at 1801 colatitudes
                               from 0 to 180 degrees (a 0.1 degree step),
                               at every epoch.
    martinec-<case>-timeseries.npz
                               Every step: h_UF, the ocean area, the ice
                               masses and the iteration counts.
    summary_<case>.json        The run settings, the mesh record, the
                               epochs and the names of the files above.
    martinec_<case>.msh        The mesh of the run.
    martinec_<case>_*.pvd      VTK files, with `--write_output` only.

    `test_benchmarks.py` converts the two npz files to a `giamip`
    `BenchmarkResult` and scores the run against VEGA with the benchmark
    criteria. The conversion runs after the job, so the compute nodes need
    only Firedrake and `giamip`.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import giamip  # noqa: E402
import numpy as np  # noqa: E402
from gadopt import (COMM_WORLD, Constant, Function,  # noqa: E402
                    FunctionSpace, ParameterLog, SpatialCoordinate,
                    SphericalDtN, VertexOnlyMesh, VTKFile, acos, assemble,
                    cos, dot, exp, interpolate, log, max_value, min_value,
                    sqrt)
from gadopt.gia_gravity import (SeaLevel,  # noqa: E402
                                SelfGravitatingGIASolver,
                                rigid_rotation_nullspace,
                                self_gravitating_gia_space)
from gadopt.sea_level_masks import (DEFAULT_ALPHA_MASK,  # noqa: E402
                                    DEFAULT_GRAD_FLOOR, grounded_ice_function,
                                    ocean_function, surface_slope)

import selfgrav_common as common  # noqa: E402

# --------------------------------------------------------------------------
# Settings of the benchmark
# --------------------------------------------------------------------------

#: K / mu of each case. Case B fails its geoid criterion at 100.
BULK_SHEAR_RATIO = {"B": 1000.0, "C": 100.0, "D": 100.0}

#: Output epochs in kyr, per time scenario. The time of the published
#: profiles is added to the list in `main`.
EPOCHS_KYR = {"T0": (0.0, 1.0, 2.0, 5.0, 10.0),
              "T1": (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)}

#: The time step of the growing load (scenario T1), years.
T1_DT_YR = 50.0

#: The time step of the held step load (scenario T0, case B), years, from
#: t = 0 to the terminal time. Backward Euler is first order in time. A held
#: step load leaves most of the geoid change still to come at 10 kyr, so the
#: time error of case B is large. On the graded Spada steps (10 yr to
#: 0.1 kyr, 50 yr to 1 kyr, 100 yr to 10 kyr) the load-meridian N is 0.98
#: percent high at 10 kyr. Criterion 2 then fails on load U and N.
#:
#: lovejx, a Love-number code that is not in this repository, gives the
#: backward-Euler response of the same Earth model on any step sequence. With
#: it, the error of this driver on a new sequence is predicted from the
#: parent's run on the graded steps. For uniform 10 yr steps the prediction
#: is a largest load-meridian N difference of 0.0250 m. The published codes
#: reach 0.0256 m. Steps of 12 yr after 0.1 kyr fail. Every sequence that keeps
#: each step above half the graded one fails as well. At a vanishing step
#: 0.0173 m remains, the spatial error of the mesh.
#:
#: The margin is 0.6 mm. The uncertainty of the prediction is about 4 mm, and
#: the prediction is calibrated on the earlier run on the coarser mesh. Only
#: the run decides. The prediction is in `NOTES/team/case-b-dt/REPORT.md`,
#: and the 0.98 percent is in `NOTES/geoid-b/opus/REPORT.md` of the parent
#: branch `sghelichkhani/sea-level`.
T0_DT_YR = 10.0

#: The number of steps of a smoke run.
SMOKE_STEPS = 10

#: The colatitudes of every comparison profile, in degrees: a uniform grid
#: of 0.1 degree over the whole meridian. `giamip` evaluates the VEGA
#: reference at the run's own points with a cubic spline, so the run does not
#: need VEGA's points. The whole meridian covers the interval of every case
#: (5 to 45, 65 to 135 and 0 to 70 degrees) and leaves the interval choice to
#: the scoring. 0.1 degree is 11 km at the surface, seven points per facet
#: of the 78 km refinement, so the grid resolves every feature the mesh can
#: carry.
PROFILE_COLATITUDE_DEG = np.linspace(0.0, 180.0, 1801)

#: How far below Re the profile points sit, as a fraction of Re. A point
#: exactly on Re is outside the P2-curved cells wherever the curved surface
#: dips inward between its nodes, and the point location then drops it.
PROFILE_DEPTH_FRACTION = 1.0e-6

#: The floor of `1 - cos(alpha)` in the cap profile. At alpha = 0 the cap
#: has no width and the profile is 0/0. The height h0(t) is zero there as
#: well, so any finite floor gives the right limit of zero ice.
CAP_DENOMINATOR_FLOOR = 1.0e-12


# --------------------------------------------------------------------------
# The case: every physical constant comes from `giamip`
# --------------------------------------------------------------------------
#
# `giamip.case("martinec2018-<L>")` returns a `giamip` `MartinecCase`. The
# functions below read its fields and give the rest of the driver plain
# numbers, so this block is the only place that knows the field names of the
# package.


def read_case(letter):
    """The `giamip` case of one benchmark letter.

    Args:
      letter: `"B"`, `"C"` or `"D"`.

    Returns:
      The `giamip.benchmarks.inputs.MartinecCase` of the letter.
    """
    return giamip.case(f"martinec2018-{letter}")


def is_fixed_ocean(case):
    """Whether the coastline is fixed (sea-level equation level 1).

    Level 1 (cases B and C) keeps the initial ocean function C0 for the whole
    run. Level 2 (case D) moves the coastline with the sea level and lets ice
    float.
    """
    return int(case.sea_level_equation) == 1


def cap_centre(case):
    """(colatitude, longitude) of the ice-cap centre, degrees."""
    return (float(case.cap["centre_colatitude_deg"]),
            float(case.cap["centre_longitude_deg"]))


def basin_centre(case):
    """(colatitude, longitude) of the ocean-basin centre, degrees."""
    return (float(case.basin["centre_colatitude_deg"]),
            float(case.basin["centre_longitude_deg"]))


def growth_end_kyr(case):
    """The end of the growth of scenario T1 in kyr, or `None` for T0."""
    scenario = case.load_table["time_scenarios"][case.time_scenario]
    end = scenario.get("growth_end_kyr")
    return None if end is None else float(end)


def rho_ice_nondim(case):
    """The ice density of the case in units of rho_bar."""
    return float(case.ice_density) / common.RHO_BAR


def rho_water_nondim(case):
    """The water density of the case in units of rho_bar."""
    return float(case.water_density) / common.RHO_BAR


def describe(case):
    """A few lines that name every constant the run uses."""
    return (f"  case {case.letter}: "
            f"{'fixed' if is_fixed_ocean(case) else 'moving'} coastline, "
            f"scenario {case.time_scenario}, terminal time "
            f"{case.terminal_time_kyr:g} kyr (giamip {giamip.__version__})\n"
            f"  ice: {case.cap['height_m']:g} m at {cap_centre(case)}, "
            f"angular radius {case.cap['radius_deg']:g} deg, "
            f"{case.ice_density:g} kg/m^3\n"
            f"  basin: centre {basin_centre(case)}, "
            f"bmax {case.basin['bmax_m']:g} m, b0 {case.basin['b0_m']:g} m, "
            f"sigma_b {case.basin['sigma_b_deg']:g} deg; "
            f"water {case.water_density:g} kg/m^3")


# --------------------------------------------------------------------------
# Geometry: the cap, the basin and the closed-form cap mass
# --------------------------------------------------------------------------


def cos_angular_distance(mesh, centre_deg):
    """cos(psi) of the angular distance from a centre, as UFL on `mesh`.

    The expression depends on direction only, so it is defined at every
    radius and can be read on a surface facet or inside a cell alike.

    Args:
      mesh: the mesh whose coordinates the expression reads.
      centre_deg: (colatitude, longitude) of the centre, degrees.

    Returns:
      A UFL expression with values in [-1, 1].
    """
    X = SpatialCoordinate(mesh)
    axis = Constant(common._unit_vector(*centre_deg))
    return dot(X, axis) / sqrt(dot(X, X))


def basin_sea_level(mesh, case):
    """SL_init = -zeta0 / D of the case's ocean basin, as a CG2 `Function`.

    The benchmark prescribes the initial topography

        zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2)),

    with psi the angular distance from the basin centre. The topography is
    measured up from the geoid, so the initial sea level is -zeta0: positive
    (water) inside the coastline, negative (land) outside. The field is
    continuous because the mask steepness reads its slope.

    Args:
      mesh: the mantle mesh.
      case: the `giamip` case.

    Returns:
      A CG2 `Function` on `mesh`, non-dimensional.
    """
    # The cosine is clamped before `acos`, so that a rounding error just
    # outside [-1, 1] on the basin axis cannot give a NaN.
    cos_psi = cos_angular_distance(mesh, basin_centre(case))
    psi = acos(max_value(min_value(cos_psi, 1.0), -1.0))
    # sigma_b and psi in the same unit, radians. The zero of zeta0 is then
    # at 24.85 degrees for both basins, the value the benchmark gives.
    sigma_b = np.radians(float(case.basin["sigma_b_deg"]))
    zeta0 = (Constant(float(case.basin["bmax_m"]))
             - Constant(float(case.basin["b0_m"]))
             * exp(-psi * psi / (2.0 * sigma_b**2)))
    return Function(FunctionSpace(mesh, "CG", 2), name="SL_init").interpolate(
        -zeta0 / common.D_SCALE)


def growth_fraction_ufl(case, t_kyr):
    """The fraction of the full cap at `t_kyr`, as UFL of the time `Constant`.

    Args:
      case: the `giamip` case.
      t_kyr: the time as a `Constant`, kyr.

    Returns:
      A UFL expression in [0, 1].
    """
    # Scenario T0 applies the whole cap at t = 0 and holds it. The time loop
    # solves the elastic state at t = 0 with the load on, so the fraction is
    # 1 at every time the loop reads it, t = 0 included.
    if case.time_scenario == "T0":
        return Constant(1.0)
    fraction = t_kyr / Constant(growth_end_kyr(case))
    return min_value(max_value(fraction, 0.0), 1.0)


def cap_profile(mesh, centre_deg, height, alpha):
    r"""The parabolic spherical cap of Spada et al. (2011), as UFL on `mesh`.

        h(gamma) = height sqrt((cos gamma - cos alpha) / (1 - cos alpha))

    inside the cap and zero outside. The profile has an infinite slope at
    the margin, so an interpolation into a polynomial space would smooth the
    margin over one cell and move the edge of the load. As an expression it
    is exact at every quadrature point. The outside is cut by a `max_value`
    under the square root, which also keeps the argument non-negative at a
    rounding distance from the margin.

    Args:
      mesh: the mesh whose coordinates the expression reads.
      centre_deg: (colatitude, longitude) of the cap centre, degrees.
      height: the thickness at the centre, a number or UFL.
      alpha: the angular radius in radians, a number or UFL.

    Returns:
      A UFL expression of the thickness, in the unit of `height`.
    """
    cos_alpha = cos(alpha)
    numerator = max_value(cos_angular_distance(mesh, centre_deg) - cos_alpha,
                          0.0)
    denominator = max_value(1.0 - cos_alpha, CAP_DENOMINATOR_FLOOR)
    return height * sqrt(numerator / denominator)


def cap_thickness(mesh, case, t_kyr):
    """The ice thickness of the case at time `t_kyr`, as UFL, divided by D.

    The height and the angular radius are both the full values times the
    growth fraction, so the expression follows the time `Constant`.
    """
    fraction = growth_fraction_ufl(case, t_kyr)
    return cap_profile(
        mesh, cap_centre(case),
        Constant(float(case.cap["height_m"]) / common.D_SCALE) * fraction,
        Constant(np.radians(float(case.cap["radius_deg"]))) * fraction)


def cap_thickness_integral(radius, height, alpha):
    r"""The integral of `cap_profile` over a sphere, in closed form.

        int h dS = 2 pi R^2 height int_{cos alpha}^{1}
                       sqrt((u - cos alpha) / (1 - cos alpha)) du
                 = (4 pi / 3) R^2 height (1 - cos alpha)

    by the substitution s = (u - cos alpha) / (1 - cos alpha), whose
    integral of sqrt(s) over [0, 1] is 2/3.

    Args:
      radius: the sphere radius.
      height: the thickness at the centre.
      alpha: the angular radius, radians.

    Returns:
      The integral, in the unit of `height` times the unit of `radius`
      squared.
    """
    return (4.0 * np.pi / 3.0) * radius**2 * height * (1.0 - np.cos(alpha))


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


def surface_gravity(parent):
    """The model's reference gravity at Re, non-dimensional.

    The sea-level equation divides the potential by g_s to make the geoid
    and multiplies the load by g_s to make its weight, so g_s must be the
    reference gravity that the mechanics uses at the surface. The case
    file's 9.8155 m s^-2 is the same quantity rounded; the two differ by
    about 5e-6.
    """
    value = Function(FunctionSpace(parent, "R", 0)).interpolate(
        common.gravity_exact_ufl(Constant(common.RE)))
    return float(value)


def build_solver(case, parent, mantle, dt, t_kyr, bulk_shear_ratio):
    """The coupled self-gravitating sea-level solver of one case.

    Args:
      case: the `giamip` case.
      parent, mantle: the two meshes.
      dt: the time step `Constant`, in Maxwell times.
      t_kyr: the time `Constant`, kyr. The ice thickness reads it.
      bulk_shear_ratio: K / mu.

    Returns:
      `(solver, layout, pieces)` with `pieces` a dictionary of the objects
      the output needs: `SL_init`, `ice` and `g_surface`.
    """
    # The Poisson equation: DtN maps on the two truncation spheres and no
    # sheet on Re. The ice and the ocean are the sea-level sheet, which
    # enters the potential row through the sea-level terms.
    gravity_bcs = {
        common.SURF_OUTER: {"dtn": SphericalDtN(L=common.DTN_DEGREE)},
        common.SURF_INNER: {"dtn": SphericalDtN(L=common.DTN_DEGREE)},
    }
    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=False,
        fluid_core=True, self_gravity_number=common.LAMBDA,
        displacement_degree=common.DISPLACEMENT_DEGREE,
        internal_variable_degree=common.INTERNAL_VARIABLE_DEGREE,
        potential_degree=common.POTENTIAL_DEGREE,
        centre_of_mass=True, sea_level=True,
        dtn_representation=common.DTN_REPRESENTATION)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    # The initial topography and its frozen slope. The slope sets the width
    # of the mask in arc length. Without it the width is fixed in sea level,
    # and it spreads over many facets on the gentle B1 shelf (slope 2.51e-4)
    # while it falls inside one facet on the steep B2 one (1.26e-3).
    SL_init = basin_sea_level(mantle, case)
    ice = cap_thickness(mantle, case, t_kyr)
    g_surface = surface_gravity(parent)
    sea_level = SeaLevel(
        boundary=common.SURF_RE,
        rho_w=rho_water_nondim(case), rho_i=rho_ice_nondim(case),
        g_surface=g_surface, SL_init=SL_init, I=ice,
        # The run starts from the undeformed state, so the initial ice, geoid
        # and uplift are all zero.
        I_init=Constant(0.0), N_init=Constant(0.0), ur_init=Constant(0.0),
        alpha_mask=DEFAULT_ALPHA_MASK, slope=surface_slope(SL_init),
        grad_floor=DEFAULT_GRAD_FLOOR, fixed_ocean=is_fixed_ocean(case))

    # A rigid rotation of the mantle stays in the kernel with sea level on:
    # its u . n is zero, so it moves no water, and its mass moment is zero
    # for a spherically symmetric density, so the frame rows do not see it.
    nullspace = rigid_rotation_nullspace(Z, layout)

    # With a fixed coastline the sheet is affine in the unknowns and one
    # linear solve is exact. With live masks (case D) Newton is the method.
    snes_type = "ksponly" if is_fixed_ocean(case) else "newtonls"
    solver = SelfGravitatingGIASolver(
        z, common.maxwell_approximation(mantle, bulk_shear_ratio),
        layout=layout, dt=dt,
        # No normal stress on Re: the ice is inside the sea-level sheet.
        bcs={}, fluid_core=common.fluid_core(), sea_level=sea_level,
        dtn_representation=common.DTN_REPRESENTATION,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=common.solver_parameters(layout, snes_type))
    pieces = {"SL_init": SL_init, "ice": ice, "g_surface": g_surface,
              "snes_type": snes_type}
    return solver, layout, pieces


# --------------------------------------------------------------------------
# Numbers of one solved state
# --------------------------------------------------------------------------


def ocean_and_ice(solver, case, pieces):
    """The ocean areas and the ice masses of the current state.

    The masks are the solver's own, with its own steepness, so these
    numbers read the same masks that the sheet integrates.

    Three areas are returned, because with a moving coastline three
    integrals count three different regions:

    * int B dS, the ocean area of the benchmark (Martinec eq. 23): the
      points with water and no grounded ice, floating ice counted as ocean.
      B = 1 - H_k(I - (rho_w / rho_i) SL) is 1 in open water and under
      floating ice, and 0 under grounded ice and on land.
    * int C dS, every point where the sea level is positive. It also counts
      the bed under grounded marine ice as ocean.
    * int B C dS, the product that multiplies the sea level in the water
      term of the sheet. It loses a strip of about 1 / k along every
      ice-free coast, which holds almost no water.

    With a fixed coastline B does not exist and all three are int C0 dS.
    With a fixed coastline the ice over the initial ocean is removed by
    1 - C0 and nothing floats (Martinec eq. 7 and 8).

    Args:
      solver: the solved solver.
      case: the `giamip` case.
      pieces: the dictionary from `build_solver`.

    Returns:
      `(area_B, area_C, area_BC, grounded_kg, floating_kg)`: the three areas
      as fractions of the sphere, and the two ice masses in kg.
    """
    dss = solver.sea_level_measure()(common.SURF_RE)
    k = solver._sea_level_steepness()
    ice = pieces["ice"]
    if is_fixed_ocean(case):
        C = ocean_function(pieces["SL_init"], k)
        B, water, floating = C, C, None
        grounded = (1 - C) * ice
    else:
        SL = solver.sea_level()
        C = ocean_function(SL, k)
        B = grounded_ice_function(ice, SL, k, rho_water_nondim(case),
                                  rho_ice_nondim(case))
        water = B * C
        grounded, floating = (1 - B) * ice, B * ice
    sphere = 4.0 * np.pi * common.RE**2
    # A non-dimensional mass times rho_bar D^3 is kilograms: the densities
    # are in units of rho_bar, the thickness in D and the area in D^2.
    scale = common.RHO_BAR * common.D_SCALE**3 * rho_ice_nondim(case)
    return (float(assemble(B * dss)) / sphere,
            float(assemble(C * dss)) / sphere,
            float(assemble(water * dss)) / sphere,
            scale * float(assemble(grounded * dss)),
            0.0 if floating is None else scale * float(assemble(floating
                                                                * dss)))


def state_row(solver, layout, case, pieces, t_kyr, dt_yr, step, wall_s):
    """Every per-step number of one solved state, as a dictionary.

    Collective: every value is an assembled integral or a `Real` field, so
    every rank must call this.
    """
    shift = float(solver.solution.subfunctions[layout.sea_level])
    dipole = solver.mass_dipole()
    dss = solver.sea_level_measure()(common.SURF_RE)
    sheet = solver.surface_load_sheet()
    # The scale of the first mass moment: the moment the load would have if
    # all of its mass sat on one side of the sphere, Re int |sigma| dS. The
    # ocean and the ice carry opposite signs, and the net mass of the sheet
    # is zero by construction, so the scale takes |sigma|.
    load_moment = float(assemble(common.RE * abs(sheet) * dss))
    abs_dipole = float(np.linalg.norm(dipole))
    area, area_C, area_water, grounded, floating = ocean_and_ice(
        solver, case, pieces)
    newton, outer = common.iteration_counts(solver)
    return {"step": step, "t_kyr": t_kyr, "dt_yr": dt_yr, "wall_s": wall_s,
            "newton": newton, "outer": outer,
            "shift": shift, "h_UF_m": shift * common.D_SCALE,
            "net_sheet_mass": float(assemble(sheet * dss)),
            "ocean_area": area, "ocean_area_C": area_C,
            "ocean_area_water": area_water,
            "ice_mass_grounded_kg": grounded,
            "ice_mass_floating_kg": floating,
            "abs_dipole": abs_dipole, "load_moment": load_moment,
            # The fraction of the load's own first moment that the centre
            # of mass still carries. It is zero by definition before a load
            # exists.
            "dipole_rel": (abs_dipole / load_moment if load_moment > 0.0
                           else 0.0)}


#: The per-step quantities, in the order of `params_<case>.log` and of the
#: time-series file. `test_benchmarks.py` reads `t_kyr`, `h_UF_m`,
#: `ocean_area` and the two ice masses from the time series.
ROW_KEYS = ("step", "t_kyr", "dt_yr", "wall_s", "newton", "outer", "shift",
            "h_UF_m", "net_sheet_mass", "ocean_area", "ocean_area_C",
            "ocean_area_water", "ice_mass_grounded_kg",
            "ice_mass_floating_kg", "abs_dipole", "load_moment", "dipole_rel")


# --------------------------------------------------------------------------
# Profiles: U, N, S and RSL along the two comparison meridians
# --------------------------------------------------------------------------


def profile_grid(case):
    """The sample points of every comparison profile of a case.

    Each profile of `case.profiles` is a meridian at a fixed longitude. The
    driver samples every one on the uniform grid `PROFILE_COLATITUDE_DEG` and
    writes the profiles at the time of the published figures, which the
    driver adds to its output epochs.

    Args:
      case: the `giamip` case.

    Returns:
      `{name: {"longitude_deg", "time_kyr", "colatitude_deg"}}` with one
      entry per profile ("load" and "basin").
    """
    return {spec.name: {"longitude_deg": float(spec.longitude_deg),
                        "time_kyr": float(spec.time_kyr),
                        "colatitude_deg": PROFILE_COLATITUDE_DEG.copy()}
            for spec in case.profiles}


class PointSampler:
    """Point evaluation of expressions of one mesh, in the input order.

    A `VertexOnlyMesh` keeps only the points that its parent mesh contains,
    in the order of the mesh partition. The sampler evaluates into the
    `input_ordering` mesh, which restores the order the points were given in,
    so that the output stays aligned with the published colatitudes.

    A missing point reads back as zero and not as an error. The constructor
    therefore interpolates the constant 1, which is exactly 1.0 at a located
    point and 0.0 at a missing one, and every evaluation writes `nan` where
    that mask is false.

    Attributes:
      found: a boolean array, `True` where the mesh contains the point.
    """

    def __init__(self, mesh, points):
        """Locate the points.

        Args:
          mesh: the mesh to evaluate on.
          points: an array of shape `(n, 3)`.
        """
        # "warn" and not "error": a missing point is counted in the output,
        # and `test_benchmarks.py` refuses a profile with one.
        self.vom = VertexOnlyMesh(mesh, points,
                                  missing_points_behaviour="warn")
        self._space = FunctionSpace(self.vom, "DG", 0)
        self._ordered = FunctionSpace(self.vom.input_ordering, "DG", 0)
        self.found = self._raw(Constant(1.0)) == 1.0

    def _raw(self, expression):
        """The point values in the input order, zero at a missing point."""
        at_points = assemble(interpolate(expression, self._space))
        ordered = Function(self._ordered)
        ordered.interpolate(at_points)
        return np.array(ordered.dat.data_ro, dtype=float)

    def __call__(self, expression):
        """A scalar UFL expression at the points, `nan` at a missing point."""
        return np.where(self.found, self._raw(expression), np.nan)


def build_profile_samplers(parent, mantle, profiles):
    """One `PointSampler` per profile and per mesh.

    The displacement lives on the mantle submesh and the potential on the
    parent, so each profile needs a sampler on each mesh at the same points.
    The points sit `PROFILE_DEPTH_FRACTION` below Re.
    """
    radius = common.RE * (1.0 - PROFILE_DEPTH_FRACTION)
    out = {}
    for name, entry in profiles.items():
        theta = np.radians(entry["colatitude_deg"])
        phi = np.radians(entry["longitude_deg"])
        points = radius * np.column_stack((np.sin(theta) * np.cos(phi),
                                           np.sin(theta) * np.sin(phi),
                                           np.cos(theta)))
        out[name] = {"mantle": PointSampler(mantle, points),
                     "parent": PointSampler(parent, points)}
        found = int(np.count_nonzero(out[name]["mantle"].found
                                     & out[name]["parent"].found))
        log(f"  profile {name}: {found} of {len(points)} points located on "
            "both meshes")
    return out


def write_profiles(path, solver, layout, samplers, profiles, g_surface,
                   t_kyr):
    """U, N, S and RSL in metres along both meridians, as an npz file.

    U is the uplift dot(u, rhat) on the mantle mesh, N the geoid
    psi / g_surface on the parent, S = N + Shift the sea-surface variation
    and RSL = S - U. g_surface divides the potential, so that N is the
    geoid that the sea-level equation solves with.

    Args:
      path: the output file.
      solver: the solved solver.
      layout: the `GIASpaceLayout`.
      samplers: the dictionary from `build_profile_samplers`.
      profiles: the dictionary from `profile_grid`.
      g_surface: the reference gravity at Re, non-dimensional.
      t_kyr: the epoch, kyr.
    """
    D = common.D_SCALE
    shift = float(solver.solution.subfunctions[layout.sea_level])
    Xm = SpatialCoordinate(layout.mechanics_mesh)
    # The two sub-functions on their own mesh and not the split of the mixed
    # function: a split component of a space on two meshes carries both
    # domains, and an interpolation of it raises.
    uplift = dot(solver.displacement, Xm / sqrt(dot(Xm, Xm)))
    geoid = solver.potential / Constant(g_surface)
    arrays = {"t_kyr": np.array(t_kyr), "shift_nondim": np.array(shift),
              "h_UF_m": np.array(shift * D)}
    for name, sampler in samplers.items():
        U = sampler["mantle"](uplift) * D
        N = sampler["parent"](geoid) * D
        S = N + shift * D
        arrays[f"{name}_colatitude_deg"] = profiles[name]["colatitude_deg"]
        arrays[f"{name}_longitude_deg"] = np.array(
            profiles[name]["longitude_deg"])
        arrays[f"{name}_U_m"] = U
        arrays[f"{name}_N_m"] = N
        arrays[f"{name}_S_m"] = S
        arrays[f"{name}_RSL_m"] = S - U
        arrays[f"{name}_found"] = (sampler["mantle"].found
                                   & sampler["parent"].found)
    if COMM_WORLD.rank == 0:
        np.savez(path, **arrays)
    log(f"      profiles written: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["B", "C", "D"], default="B",
                   help="the benchmark case")
    p.add_argument("--bulk_shear_ratio", type=float, default=None,
                   help="K / mu in every layer; default 1000 for case B and "
                        "100 for cases C and D")
    p.add_argument("--smoke", action="store_true",
                   help=f"run {SMOKE_STEPS} steps on a very coarse mesh, to "
                        "check that the code runs; the numbers are not "
                        "results")
    p.add_argument("--write_output", action="store_true",
                   help="write Paraview VTK files at every epoch")
    p.add_argument("--output_path", default="./",
                   help="the directory of every output file")
    return p.parse_args()


def main():
    """Run one Martinec case."""
    args = parse_args()
    tic_run = time.time()
    os.makedirs(args.output_path, exist_ok=True)
    letter = args.case
    bulk_shear_ratio = (args.bulk_shear_ratio if args.bulk_shear_ratio
                        is not None else BULK_SHEAR_RATIO[letter])
    case = read_case(letter)
    profiles = profile_grid(case)
    epochs = sorted(set(EPOCHS_KYR[case.time_scenario])
                    | {entry["time_kyr"] for entry in profiles.values()})

    log("=" * 78)
    log(f"Martinec et al. (2018) sea-level benchmark, case {letter}"
        f"{' (smoke run)' if args.smoke else ''}")
    log("=" * 78)
    log(describe(case))
    log(f"  K/mu {bulk_shear_ratio:g}, SphericalDtN(L={common.DTN_DEGREE})")

    parent, mantle, mesh_info = common.build_meshes(
        "smoke" if args.smoke else "martinec", args.output_path,
        f"martinec_{letter}")

    # Scenario T0 (case B) is a load switched on at t = 0: the elastic solve
    # and then uniform steps of `T0_DT_YR`, which the time error of backward
    # Euler on a held load needs (see `T0_DT_YR`). A growing load (T1) has no
    # fast early response and takes uniform steps of `T1_DT_YR`.
    t_end_yr = max(epochs) * 1000.0
    if case.time_scenario == "T0":
        ladder = ((t_end_yr, T0_DT_YR),)
    else:
        ladder = ((t_end_yr, T1_DT_YR),)
    segments = common.time_segments(epochs, ladder)
    log(f"  epochs (kyr) {epochs}, {sum(s[3] for s in segments)} steps")

    dt = Constant(common.DT_ELASTIC)
    t_kyr = Constant(0.0)
    tic = time.time()
    solver, layout, pieces = build_solver(case, parent, mantle, dt, t_kyr,
                                          bulk_shear_ratio)
    log(f"  solver built ({time.time() - tic:.1f} s): "
        f"{solver.solution.function_space().dim()} unknowns, "
        f"{len(layout.real_fields)} Real rows, {pieces['snes_type']}")
    log(f"  g_surface: model {pieces['g_surface'] * common.G_BAR:.6f} m/s^2, "
        f"case file {case.surface_gravity:.6f} m/s^2")

    # The ice volume of the cap at the terminal time against its closed
    # form, a check of the load expression and of the surface quadrature on
    # this mesh. The height and the radius at that time come from
    # `case.cap_size`, the definition of `giamip`, so the check also compares
    # the UFL growth rule of this driver with the package.
    t_kyr.assign(case.terminal_time_kyr)
    dss = solver.sea_level_measure()(common.SURF_RE)
    assembled = float(assemble(pieces["ice"] * dss))
    height_m, radius_deg = case.cap_size(case.terminal_time_kyr)
    closed = cap_thickness_integral(common.RE, height_m / common.D_SCALE,
                                    np.radians(radius_deg))
    cap_volume_error = abs(assembled - closed) / closed
    log(f"  cap volume at {case.terminal_time_kyr:g} kyr: relative "
        f"difference to the closed form {cap_volume_error:.3e}")
    t_kyr.assign(0.0)

    samplers = build_profile_samplers(parent, mantle, profiles)
    # `test_benchmarks.py` refuses a profile with a point that is not on the
    # mesh. The points are fixed before the first step, so a full run stops
    # here and not after a day of steps. A smoke run only warns: its coarse
    # mesh curves the surface less exactly, and its numbers are not results.
    missing = {
        name: int(np.count_nonzero(~(s["mantle"].found & s["parent"].found)))
        for name, s in samplers.items()}
    if any(missing.values()):
        message = (f"profile points not located on the mesh: {missing}; "
                   "increase PROFILE_DEPTH_FRACTION")
        if not args.smoke:
            raise RuntimeError(message)
        log(f"  WARNING: {message}")

    vtk = None
    if args.write_output:
        vtk = (VTKFile(os.path.join(args.output_path,
                                    f"martinec_{letter}_mechanics.pvd")),
               VTKFile(os.path.join(args.output_path,
                                    f"martinec_{letter}_potential.pvd")))

    plog = ParameterLog(os.path.join(args.output_path,
                                     f"params_{letter}.log"), parent)
    plog.log_str(" ".join(ROW_KEYS))
    rows, profile_files = [], []
    timeseries_path = os.path.join(args.output_path,
                                   f"martinec-{letter}-timeseries.npz")

    def write_timeseries():
        """The time series of every step so far, rewritten at each epoch."""
        if COMM_WORLD.rank == 0 and rows:
            np.savez(timeseries_path,
                     **{key: np.array([row[key] for row in rows])
                        for key in ROW_KEYS})

    def on_step(t_step_kyr, dt_yr, step, wall_s):
        row = state_row(solver, layout, case, pieces, t_step_kyr, dt_yr,
                        step, wall_s)
        rows.append(row)
        plog.log_str(" ".join(str(row[key]) for key in ROW_KEYS))
        log(f"  step {step:4d}  t {t_step_kyr:9.4f} kyr  newton "
            f"{row['newton']}  outer {row['outer']:3d}  h_UF "
            f"{row['h_UF_m']:10.4f} m  wall {wall_s:8.1f} s")

    def on_epoch(t_epoch_kyr):
        path = os.path.join(args.output_path,
                            f"martinec-{letter}-profiles_{t_epoch_kyr:g}kyr"
                            ".npz")
        write_profiles(path, solver, layout, samplers, profiles,
                       pieces["g_surface"], t_epoch_kyr)
        profile_files.append(os.path.basename(path))
        write_timeseries()
        if vtk is not None:
            vtk[0].write(solver.displacement, time=t_epoch_kyr)
            vtk[1].write(solver.potential, time=t_epoch_kyr)

    common.march(solver, dt, segments, elastic=case.time_scenario == "T0",
                 on_step=on_step, on_epoch=on_epoch,
                 before_step=t_kyr.assign,
                 max_steps=SMOKE_STEPS if args.smoke else None)
    plog.close()
    write_timeseries()

    common.write_json(
        os.path.join(args.output_path, f"summary_{letter}.json"),
        {"benchmark": "Martinec et al. (2018)", "case": letter,
         "smoke": args.smoke, "bulk_shear_ratio": bulk_shear_ratio,
         "dtn_degree": common.DTN_DEGREE, "ranks": parent.comm.size,
         "mesh": mesh_info, "wall_s": time.time() - tic_run,
         "steps": len(rows), "epochs_kyr": epochs,
         "profile_files": profile_files,
         "timeseries_file": os.path.basename(timeseries_path),
         "terminal_time_kyr": case.terminal_time_kyr,
         "giamip_version": giamip.__version__,
         "cap_volume_error": cap_volume_error,
         "profile_points_found": {
             name: int(np.count_nonzero(s["mantle"].found
                                        & s["parent"].found))
             for name, s in samplers.items()},
         "h_UF_m_last": rows[-1]["h_UF_m"] if rows else None})


if __name__ == "__main__":
    main()
