r"""The Martinec et al. (2018) sea-level benchmark in 3-D, cases B, C and D.

THE BENCHMARK
    Martinec, Z., et al. (2018), A benchmark study of numerical
    implementations of the sea level equation in GIA modelling, Geophys. J.
    Int. 215(1), 389-414.

    The benchmark drives a spherically layered, self-gravitating Maxwell Earth
    (the Spada et al. 2011 model M3-L70-V01) with an ice cap and solves the
    sea-level equation on a prescribed ocean basin. This driver solves the
    coupled problem with G-ADOPT on the four-region tetrahedral sphere of the
    Spada driver next door: viscoelastic mechanics, the perturbed
    gravitational potential, the centre-of-mass frame and the sea-level
    equation, all in one monolithic system.

    Three of the five cases are in scope. Case A has no ocean and the Spada
    driver covers it. Case E prescribes the topography at the end and is an
    inverse problem.

        case   ocean            ice            time        basin
        B      fixed coastline  L1, 1500 m     T0, step    B1, far from the ice
        C      fixed coastline  L2, 500 m      T1, growth  B2, under the ice
        D      moving coastline L2, 500 m      T1, growth  B2

    Every physical constant comes from the gia-mip case files
    (`cases/benchmarks/martinec2018/<case>.json` and `loads/martinec2018.json`),
    never from this file, because Martinec and GIAMIP use different densities.
    Pass the gia-mip checkout with `--gia-mip`.

THE SEA-LEVEL EQUATION AS THIS CODE SOLVES IT
    `gadopt.SeaLevel` puts the ocean and the ice into one surface load sheet
    and solves the sea level inside the one nonlinear system, with the uniform
    shift as a `Real` unknown:

        SL    = SL_init + (N - N_init) - (u_r - ur_init) + Shift
        sigma = rho_w C0 (SL - SL_init) + rho_i (1 - C0) (I - I_init)

    for a fixed coastline (cases B and C, `C0 = C(SL_init)`, Martinec eq. 7
    and 8), and with the live ocean and grounded-ice masks for a moving
    coastline (case D). `N = psi / g_s` is the geoid height, `u_r = u . n` the
    uplift. The `Shift` row is the conservation of the mass of water and ice.
    The centre-of-mass multipliers hold the first mass moment of mantle, core
    and load at zero, which is the frame of the benchmark. Rotation is off.

    The quantities of the benchmark map to this code as follows.

        gia-mip   meaning                       here
        U         vertical displacement, up     dot(u, n) on Re
        N         geoid displacement, up        psi / g_surface
        S         sea-surface variation         N + Shift
        RSL       relative sea level            S - U
        zeta      topography, up                -SL_init at t = 0
        h_UF      uniform water layer           Shift

    The mask steepness is set in arc length by a frozen surface slope
    (`gadopt.sea_level_masks.surface_slope`), because the bed slopes of the
    two basins differ by a factor of five: 2.51e-4 (B1) and 1.26e-3 (B2).

THE LOAD AND THE BASIN
    The ice cap is the Spada cap, moved and rescaled:

        h(gamma, t) = h0(t) sqrt((cos gamma - cos alpha(t)) / (1 - cos alpha(t)))

    for gamma < alpha(t) and zero outside, with gamma the angular distance
    from the cap centre at colatitude 25, longitude 75 degrees. Scenario T0
    applies the full cap as a step at t = 0 and holds it. Scenario T1 grows
    `h0` and `alpha` linearly from zero to their full values over 10 kyr and
    then holds them. The thickness is a UFL expression of position and of a
    time `Constant`, never an interpolated field, so it is exact at every
    quadrature point including the infinite slope at the cap margin.

    The initial topography is the ocean basin of the benchmark,

        zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2)),

    with psi the angular distance from the basin centre and `sigma_b` 26
    degrees. The initial sea level is `SL_init = -zeta0`, interpolated into
    CG2 on the mantle mesh off the tape.

NON-DIMENSIONAL SCALES
    The scales of the Spada driver, unchanged: length D = 2891 km, density
    rho_bar = 5511.68 kg m^-3, gravity g_bar = 9.81555 m s^-2, stress
    mu_bar = 1e11 Pa, viscosity eta_bar = 1e21 Pa s, time t_bar = 316.8809 yr.

    `g_surface` of the sea-level energy is the model's own reference gravity
    at Re and not the case file's 9.8155 m s^-2. The energy needs the gravity
    that divides the potential to be the reference gravity at the surface, or
    the displacement row and the potential row weigh the same load
    differently. The two values differ by about 5e-6 relative and the driver
    prints both.

DISCRETISATION AND SOLVER
    As the Spada driver: CG3 displacement and DG2 internal variables on the
    mantle submesh, CG2 potential on the whole mesh, one Maxwell element per
    cell, a bulk modulus of `--bulk-shear-ratio` times the shear modulus,
    backward Euler in time, and the iterative preset
    `selfgrav_dtn_iterative_solver_parameters` with `gadopt.CondensedBlockPC`
    on the mechanics-potential block.

    Two settings differ. The DtN condition uses the low-rank representation by
    default, because the multiplier path costs one block-0 solve per
    multiplier in the dense Schur complement and does not fit one node at
    L = 20. And block 1 is left to the preset, which chooses by the width of
    the `Real` block: 5 rows here (core pressure, three centre-of-mass rows,
    the sea-level `Shift`), and at that width the choice is the cached apply
    of `gadopt.DtNTwoBlockSchurPC` under `schur_fact_type full`. Something
    has to own that block: with the frame rows and the `Shift` row present,
    `pc_type none` needed 66 block-0 solves against 9 on the 2-D annulus.
    `--multiplier-pc` names a class instead, which keeps the older delegating
    path under `schur_fact_type lower`. Every job before 2026-09-21 ran that
    path, so an arm that compares its counts against one of those jobs must
    name the class.

    Cases B and C have a fixed coastline, so the sea-level rows are linear in
    the unknowns and the driver runs `snes_type ksponly`. Case D has live
    masks and runs `newtonls`.

TIME
    Case B (scenario T0) takes the Spada ladder truncated at 10 kyr: an
    elastic solve at 1e-4 Maxwell times for the t = 0 state, then the state
    and the history reset to zero, then 10 yr to 0.1 kyr, 50 yr to 1 kyr and
    100 yr to 10 kyr. Cases C and D (scenario T1) take 50 yr steps from 0 to
    15 kyr with no elastic solve, because the load grows from zero. radopt's
    case C load profile still moved between 100 yr and 50 yr steps.

RUN
    Copy the mesh (see `--mesh`), then:

        mpiexec -np 96 python3 martinec_benchmark.py --case B \
            --gia-mip ~/Workplace/gia-mip --mesh b2_coarse_ar7.msh

    Check an installation before a long run. This builds everything,
    assembles the residual once and exits before any solve:

        python3 martinec_benchmark.py --case B --dry-run

    A smoke run of the first two steps, for a comparison of two settings:

        python3 martinec_benchmark.py --case B --steps 2 --label smoke

OUTPUT
    All file names carry `--label`, which defaults to the case letter.

    martinec-<label>.h5            A `CheckpointFile` with both meshes and the
                                   state at every epoch and every 50 steps.
                                   `--restart INDEX` continues from state
                                   INDEX of such a file.
    martinec-<label>-profiles_<t>kyr.npz
                                   U, N, S and RSL in metres along the two
                                   comparison meridians, at the colatitudes of
                                   the VEGA profiles.
    martinec-<label>-timeseries.npz
                                   Every step: `h_UF`, ocean area, grounded
                                   and floating ice mass, and the solver
                                   counters.
    martinec-<label>-steptimes.npz The end time of every step in kyr, as the
                                   array `t_kyr`, for the matched radopt runs.
    martinec-<label>-ocean_function_C0_nglv<N>.npz
                                   With `--export-ocean-function N`: the fixed
                                   ocean function on the radopt
                                   Gauss-Legendre grid.
    stdout                         One `TIMESTEP` line per step and one
                                   `FRAME` line per solve.

WHAT IS NOT HERE
    The comparison with VEGA. The npz files above are read on the laptop by
    `scripts/martinec_gadopt.py` of gia-mip, which writes the benchmark
    NetCDF and scores it with `scripts/martinec_compare.py`.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPADA = os.path.join(HERE, "..", "3d_spada_selfgrav")
# The Spada driver and its helpers are siblings and not a package. Both
# directories go on the path, this one first, exactly as `spada_benchmark.py`
# puts its own directory there. Nothing is copied from them: the meshes, the
# rheology and the reference state are the Spada ones, and a copy here would
# drift out of step with the benchmark the parent branch validates.
sys.path.insert(0, SPADA)
sys.path.insert(0, HERE)

import gadopt  # noqa: E402,F401  (import gadopt before firedrake)
import numpy as np  # noqa: E402
from firedrake import (COMM_WORLD, CheckpointFile, Constant,  # noqa: E402
                       FacetArea, Function, FunctionSpace, SpatialCoordinate,
                       Submesh, TestFunction, VertexOnlyMesh, acos, assemble,
                       cos, dot, ds, exp, interpolate, max_value, min_value,
                       sqrt)
from gadopt import SphericalDtN  # noqa: E402
from gadopt.gia_gravity import (FluidCore, SeaLevel,  # noqa: E402
                                SelfGravitatingGIASolver,
                                rigid_rotation_nullspace,
                                selfgrav_dtn_iterative_solver_parameters,
                                self_gravitating_gia_space)
from gadopt.sea_level_masks import (DEFAULT_ALPHA_MASK,  # noqa: E402
                                    DEFAULT_GRAD_FLOOR, SMOOTH_STEP_CLAMP,
                                    grounded_ice_function, ocean_function,
                                    surface_slope)

from pyadjoint.tape import stop_annotating  # noqa: E402

import generate_selfgrav_sphere as gen  # noqa: E402
import reference_state as refstate  # noqa: E402
import spada_benchmark as spada  # noqa: E402

# --------------------------------------------------------------------------
# Scales, files and fixed settings
# --------------------------------------------------------------------------

#: The length scale D = Re - Rc, in m.
D_M = refstate.D_SCALE
#: The time scale eta_bar / mu_bar, in years.
T_BAR_YR = spada.T_BAR_YR
#: Buoyancy stress over elastic stress, rho_bar g_bar D / mu_bar.
B_MU = spada.B_MU
#: The self-gravity number 4 pi G rho_bar D / g_bar.
LAMBDA = spada.LAMBDA
#: The time step of the elastic solve at t = 0, in Maxwell times.
DT_ELASTIC = spada.DT_ELASTIC

#: The default gia-mip checkout. It holds the case files, the load file and
#: the converted reference data. The driver reads JSON only, so the checkout
#: needs no install and no Python environment of its own.
DEFAULT_GIA_MIP = os.environ.get(
    "GIA_MIP", os.path.expanduser("~/Workplace/gia-mip"))

#: The colatitudes of the VEGA comparison profiles, written from the
#: converted reference data by `write_profile_json.py`.
PROFILE_JSON = os.path.join(HERE, "martinec_profiles.json")

#: The refined mesh of W6, the default for the production runs. The coarse
#: `b2_coarse_ar7.msh` of the Spada directory is the mesh of the smoke runs
#: and of the first comparison job.
DEFAULT_MESH = os.path.join(
    HERE, "martinec_h78_cap12_band4_gl0.5_depth500_base500_grade2_ll1_mc32"
          "_alg1_seed7.msh")

#: Output epochs in kyr, per time scenario. The terminal time of each case is
#: the time of the published profiles and is always in the list.
DEFAULT_EPOCHS = {"T0": (0.0, 1.0, 2.0, 5.0, 10.0),
                  "T1": (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)}

#: Write a checkpoint state every this many steps, on top of the epochs, so
#: that a job killed by the walltime can be restarted from close to its end.
CHECKPOINT_EVERY = 50

#: How far below Re the profile points sit, as a fraction of Re. A point
#: exactly on Re is outside the P2-curved cells wherever the curved surface
#: dips inward between its nodes, and the point location then drops it.
PROFILE_DEPTH_FRACTION = 1.0e-6

#: The floor of `1 - cos(alpha)` in the cap profile. At `alpha = 0` the cap
#: has no width and the profile is 0/0; the height `h0(t)` is zero there as
#: well, so any finite floor gives the right limit of zero ice. It only keeps
#: the expression finite.
CAP_DENOMINATOR_FLOOR = 1.0e-12

#: Names of the two meshes in the checkpoint file.
PARENT_NAME, MANTLE_NAME = "martinec_parent", "martinec_mantle"

#: Where the per-state attributes of the checkpoint live.
STATE_GROUP = "/martinec_state"


def say(msg):
    """Print on rank 0 only; every quantity printed is collective."""
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# The case: every physical constant comes from the gia-mip files
# --------------------------------------------------------------------------

class MartinecCase:
    """One benchmark case, read from the gia-mip case and load files.

    The gia-mip rule (`AGENTS.md` of that repository) is that a physical
    constant of a benchmark comes from the case file and never from a module:
    Martinec et al. (2018) and GIAMIP prescribe different densities for the
    same-looking quantities. This class is the only place in the driver that
    reads them, and every other function takes plain numbers.

    Attributes:
      letter: `"B"`, `"C"` or `"D"`.
      sea_level_level: 1 for a fixed ocean geometry, 2 for a moving one.
      fixed_ocean: whether `SeaLevel.fixed_ocean` is set, that is level 1.
      rho_ice, rho_water: densities in kg m^-3.
      g_case: the surface gravity of the case file, m s^-2. The driver uses
        the model's own reference gravity instead; see `g_surface_nondim`.
      cap_centre: (colatitude, longitude) of the ice cap, degrees.
      cap_height_m: `h0`, the full centre thickness, m.
      cap_radius_deg: `alpha0`, the full angular radius, degrees.
      scenario: `"T0"` (a step at t = 0) or `"T1"` (linear growth).
      growth_end_kyr: the end of the growth of T1, or `None` for T0.
      terminal_time_kyr: the end of the run.
      basin_centre: (colatitude, longitude) of the ocean basin, degrees.
      bmax_m, b0_m, sigma_b_deg: the basin shape,
        `zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2))`.
      coast_psi_deg: the angular radius of the initial coastline, where
        `zeta0 = 0`.
    """

    def __init__(self, letter, gia_mip):
        """Read `cases/benchmarks/martinec2018/<letter>.json` and its load file.

        Args:
          letter: the case letter.
          gia_mip: the path of the gia-mip checkout.

        Raises:
          FileNotFoundError: if either file is missing, with the path it
            looked for, because the checkout is a run-time argument and a
            missing one must not look like a physics error later.
        """
        case_path = os.path.join(gia_mip, "cases", "benchmarks", "martinec2018",
                                 f"{letter}.json")
        load_path = os.path.join(gia_mip, "loads", "martinec2018.json")
        for path in (case_path, load_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} does not exist. Pass the gia-mip checkout with "
                    "--gia-mip; clone github.com/g-adopt/gia-mip next to the "
                    "worktrees on a machine that has no copy.")
        with open(case_path) as handle:
            case = json.load(handle)
        with open(load_path) as handle:
            loads = json.load(handle)

        self.letter = letter
        self.case_path, self.load_path = case_path, load_path
        specification = case["specification"]
        self.sea_level_level = int(specification["sea_level_equation"])
        self.fixed_ocean = self.sea_level_level == 1
        self.terminal_time_kyr = float(specification["terminal_time_kyr"])

        constants = loads["constants"]
        self.rho_ice = float(constants["ice_density_kg_m3"])
        self.rho_water = float(constants["water_density_kg_m3"])
        self.g_case = float(constants["surface_gravity_m_s2"])

        ice = loads["ice_height_models"][specification["ice_height_model"]]
        self.ice_model = specification["ice_height_model"]
        self.cap_centre = (float(ice["centre_colatitude_deg"]),
                           float(ice["centre_longitude_deg"]))
        self.cap_height_m = float(ice["height_m"])
        self.cap_radius_deg = float(loads["load_shape"]["angular_radius_deg"])

        self.scenario = specification["time_scenario"]
        scenario = loads["time_scenarios"][self.scenario]
        self.growth_end_kyr = (float(scenario["growth_end_kyr"])
                               if "growth_end_kyr" in scenario else None)

        self.basin_name = specification["ocean_basin"]
        basin = loads["ocean_basins"][self.basin_name]
        self.basin_centre = (float(basin["centre_colatitude_deg"]),
                             float(basin["centre_longitude_deg"]))
        self.bmax_m = float(basin["bmax_m"])
        self.b0_m = float(basin["b0_m"])
        self.sigma_b_deg = float(loads["ocean_basins"]["sigma_b_deg"])
        self.coast_psi_deg = float(basin["radius_deg"])

    @property
    def rho_ice_nondim(self):
        """The ice density in units of the reference density rho_bar."""
        return self.rho_ice / refstate.RHO_BAR

    @property
    def rho_water_nondim(self):
        """The water density in units of the reference density rho_bar."""
        return self.rho_water / refstate.RHO_BAR

    def growth_fraction(self, t_kyr):
        """The fraction of the full cap at time `t_kyr`, as a plain number.

        Scenario T0 applies the whole cap as a step at t = 0 and holds it, so
        the fraction is 1 at every time of the run, the elastic solve at t = 0
        included. Scenario T1 grows the cap linearly from zero at t = 0 to the
        full cap at `growth_end_kyr` and then holds it.

        `cap_thickness` needs the same rule as a UFL expression of the time
        `Constant`; `growth_fraction_ufl` writes it there. This one is for
        printing and for the closed-form cap mass.
        """
        if self.scenario == "T0":
            return 1.0
        return min(1.0, max(0.0, t_kyr / self.growth_end_kyr))

    def describe(self):
        """A few lines that name every constant the run uses."""
        return (
            f"  case {self.letter}: sea-level equation level "
            f"{self.sea_level_level} "
            f"({'fixed' if self.fixed_ocean else 'moving'} coastline), ice "
            f"{self.ice_model}, scenario {self.scenario}, basin "
            f"{self.basin_name}\n"
            f"  ice: {self.cap_height_m:g} m at colatitude "
            f"{self.cap_centre[0]:g}, longitude {self.cap_centre[1]:g}, "
            f"angular radius {self.cap_radius_deg:g} deg, density "
            f"{self.rho_ice:g} kg/m^3\n"
            f"  basin: centre colatitude {self.basin_centre[0]:g}, longitude "
            f"{self.basin_centre[1]:g}, bmax {self.bmax_m:g} m, b0 "
            f"{self.b0_m:g} m, sigma_b {self.sigma_b_deg:g} deg, coastline at "
            f"{self.coast_psi_deg:g} deg\n"
            f"  water density {self.rho_water:g} kg/m^3, terminal time "
            f"{self.terminal_time_kyr:g} kyr\n"
            f"  files: {self.case_path}, {self.load_path}")


# --------------------------------------------------------------------------
# Geometry: the cap, the basin and the closed-form cap mass
# --------------------------------------------------------------------------

def unit_vector(colatitude_deg, longitude_deg):
    """The Cartesian unit vector of a point given in geographic degrees.

    Args:
      colatitude_deg: colatitude, degrees from the north pole (the +z axis).
      longitude_deg: longitude, degrees east from the +x axis.

    Returns:
      A NumPy array of shape (3,) with unit length.
    """
    theta, phi = np.radians(colatitude_deg), np.radians(longitude_deg)
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def cos_angular_distance(mesh, centre_deg):
    """`cos(psi)` of the angular distance from a centre, as UFL on `mesh`.

    The expression depends on direction only, so it is defined at every
    radius and can be read on a surface facet or inside a cell alike.

    Args:
      mesh: the mesh whose coordinates the expression reads.
      centre_deg: (colatitude, longitude) of the centre, degrees.

    Returns:
      A UFL expression with values in [-1, 1].
    """
    X = SpatialCoordinate(mesh)
    axis = Constant(unit_vector(*centre_deg))
    return dot(X, axis) / sqrt(dot(X, X))


def basin_sea_level(mesh, case, degree=2):
    """`SL_init = -zeta0 / D` of the case's ocean basin, as a CG `Function`.

    The benchmark prescribes the topography at the initial time,

        zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2)),

    with `psi` the angular distance from the basin centre. The topography is
    measured up from the geoid, so the sea level of the reference state is
    `-zeta0`: positive (water) inside the coastline, negative (land) outside.
    Both are divided by the length scale D.

    The field is interpolated into a continuous space, because the mask
    steepness reads its slope and `surface_slope` differentiates it. The
    interpolation runs inside `stop_annotating()`: the initial topography is
    data of the benchmark and is not a control of any gradient this driver
    takes, so the taped run of W11 must not carry it. The block is a no-op
    while no tape is active, which is every run of this driver today.

    Args:
      mesh: the mantle mesh, which carries the displacement and the measure.
      case: the `MartinecCase`.
      degree: the polynomial degree of the CG space. The default 2 matches the
        potential and is one below the displacement.

    Returns:
      A CG `Function` on `mesh`, non-dimensional.
    """
    # The cosine is clamped before `acos`, so that a round-off value just
    # outside [-1, 1] at a point on the basin axis cannot give a NaN.
    cos_psi = cos_angular_distance(mesh, case.basin_centre)
    psi = acos(max_value(min_value(cos_psi, 1.0), -1.0))
    # `sigma_b` and `psi` must be in the same unit. Radians here, which puts
    # the zero of zeta0 at 24.85 degrees for both basins, the value the
    # benchmark prints.
    sigma_b = np.radians(case.sigma_b_deg)
    zeta0 = (Constant(case.bmax_m)
             - Constant(case.b0_m) * exp(-psi * psi / (2.0 * sigma_b ** 2)))
    space = FunctionSpace(mesh, "CG", degree)
    with stop_annotating():
        return Function(space, name="SL_init").interpolate(-zeta0 / D_M)


def growth_fraction_ufl(case, t_kyr):
    """The fraction of the full cap at time `t_kyr`, as UFL.

    The twin of `MartinecCase.growth_fraction`, written against the live time
    `Constant` so that the ice expression follows the step without being
    rebuilt. The forms read the `Constant`, so the driver assigns the end time
    of each step to it and every use of the load follows.

    Args:
      case: the `MartinecCase`.
      t_kyr: the time as a `Constant`, in kyr.

    Returns:
      A UFL expression in [0, 1], or the number 1.0 for scenario T0.
    """
    if case.scenario == "T0":
        return Constant(1.0)
    fraction = t_kyr / Constant(case.growth_end_kyr)
    return min_value(max_value(fraction, 0.0), 1.0)


def cap_profile(mesh, centre_deg, height, alpha):
    r"""The parabolic spherical cap of Spada et al. (2011), as UFL on `mesh`.

        h(gamma) = height sqrt((cos gamma - cos alpha) / (1 - cos alpha))

    inside the cap and zero outside, with `gamma` the angular distance from
    the cap centre.

    **An expression and not an interpolated field.** The profile has an
    infinite slope at the margin, so any interpolation of it into a polynomial
    space smooths the margin over one cell and moves the edge of the load. As
    an expression it is exact at every quadrature point, which is where the
    solver reads it.

    The outside is cut by `max_value(cos gamma - cos alpha, 0)` under the
    square root instead of by a `conditional`. The two give the same value,
    and this form also keeps the argument of `sqrt` non-negative at round-off
    distance from the margin.

    Args:
      mesh: the mesh whose coordinates the expression reads.
      centre_deg: (colatitude, longitude) of the cap centre, degrees.
      height: the centre thickness, a number or a UFL expression, in the unit
        the result is wanted in.
      alpha: the angular radius in radians, a number or a UFL expression.

    Returns:
      A UFL expression of the thickness, in the unit of `height`.
    """
    cos_alpha = cos(alpha)
    cos_gamma = cos_angular_distance(mesh, centre_deg)
    numerator = max_value(cos_gamma - cos_alpha, 0.0)
    # At `alpha = 0` the cap has no width and the ratio is 0/0. Every caller
    # that can reach `alpha = 0` has `height = 0` there as well, so the floor
    # only keeps the expression finite.
    denominator = max_value(1.0 - cos_alpha, CAP_DENOMINATOR_FLOOR)
    return height * sqrt(numerator / denominator)


def cap_thickness(mesh, case, t_kyr):
    """The ice thickness of the case's cap at time `t_kyr`, as UFL on `mesh`.

    `cap_profile` with the height and the angular radius of the time
    scenario: both are the full values times `growth_fraction_ufl`, so the
    expression follows the time `Constant` without being rebuilt.

    Args:
      mesh: the mesh whose coordinates the expression reads.
      case: the `MartinecCase`.
      t_kyr: the time as a `Constant`, in kyr.

    Returns:
      A UFL expression of the thickness, non-dimensional (divided by D).
    """
    fraction = growth_fraction_ufl(case, t_kyr)
    return cap_profile(mesh, case.cap_centre,
                       Constant(case.cap_height_m / D_M) * fraction,
                       Constant(np.radians(case.cap_radius_deg)) * fraction)


def cap_thickness_integral(radius, height, alpha):
    r"""`int h dS` of `cap_profile` over a sphere, in closed form.

        int h dS = 2 pi R^2 height int_{cos alpha}^{1}
                       sqrt((u - cos alpha) / (1 - cos alpha)) du
                 = (4 pi / 3) R^2 height (1 - cos alpha)

    by the substitution `s = (u - cos alpha) / (1 - cos alpha)`, whose
    integral of `sqrt(s)` over [0, 1] is 2/3.

    Args:
      radius: the sphere radius.
      height: the centre thickness.
      alpha: the angular radius, radians.

    Returns:
      The integral, in the unit of `height` times the square of the unit of
      `radius`.
    """
    return (4.0 * np.pi / 3.0) * radius ** 2 * height * (1.0 - np.cos(alpha))


def cap_mass_closed_form(case, t_kyr, radius):
    """The mass of the case's cap at time `t_kyr`, in closed form.

    `cap_thickness_integral` of the height and the angular radius of the time
    scenario, times the ice density. The driver prints it next to the
    assembled ice mass, where the two must agree to the quadrature error of
    the mesh, and the unit test of `cap_profile` compares against the same
    closed form.

    Args:
      case: the `MartinecCase`.
      t_kyr: the time in kyr, a plain number.
      radius: the sphere radius, non-dimensional.

    Returns:
      The mass, non-dimensional (in units of rho_bar D^3).
    """
    fraction = case.growth_fraction(t_kyr)
    return case.rho_ice_nondim * cap_thickness_integral(
        radius, case.cap_height_m / D_M * fraction,
        np.radians(case.cap_radius_deg) * fraction)


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------

def surface_gravity(parent):
    """The model's reference gravity at Re, non-dimensional.

    The sea-level energy divides the potential by `g_s` to make the geoid and
    multiplies the load by `g_s` to make its weight, so `g_s` must be the
    reference gravity that the mechanics uses at the surface. The case file's
    9.8155 m s^-2 is the same quantity rounded, and the two differ by about
    5e-6 relative; the driver prints both.

    The value is read from the very expression the approximation uses,
    `reference_state.gravity_exact_ufl`, through a `Real` space, so that a
    change of the layered density reaches this number without an edit here.

    Args:
      parent: any mesh; the expression depends on the radius only.

    Returns:
      The gravity at Re in units of g_bar, as a float.
    """
    real = FunctionSpace(parent, "R", 0)
    value = Function(real).interpolate(
        refstate.gravity_exact_ufl(Constant(gen.RE)))
    return float(value)


def build_solver(parent, mantle, case, args, dt, t_kyr):
    """The coupled self-gravitating sea-level solver of one Martinec case.

    Args:
      parent, mantle: the two meshes.
      case: the `MartinecCase`.
      args: the parsed command line.
      dt: the time step as a `Constant`, in Maxwell times. The driver assigns
        new values to it between segments; the forms read the live value.
      t_kyr: the time as a `Constant`, in kyr. The ice thickness reads it, so
        the driver assigns the end time of each step before solving it
        (backward Euler).

    Returns:
      `(solver, z, layout, pieces)` with `pieces` a dictionary of the objects
      the output needs: `SL_init`, `slope`, `ice`, `g_surface` and the
      `SeaLevel` itself.
    """
    # The Poisson equation: DtN maps on the two truncation spheres and no
    # sheet on Re. The ice and the ocean are the sea-level sheet, which
    # enters the potential row through `sea_level_residual`;
    # `check_sea_level` refuses a second sheet on the same boundary.
    gravity_bcs = {
        gen.SURF_OUTER: {"dtn": SphericalDtN(L=args.dtn_degree)},
        gen.SURF_INNER: {"dtn": SphericalDtN(L=args.dtn_degree)},
    }

    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=False,
        fluid_core=True, self_gravity_number=LAMBDA,
        displacement_degree=args.displacement_degree,
        internal_variable_degree=args.internal_variable_degree,
        potential_degree=args.potential_degree,
        centre_of_mass=True, sea_level=True,
        dtn_representation=args.dtn_representation)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    approximation = spada.spada_approximation(mantle, args.bulk_shear_ratio)

    # The initial topography, its frozen slope and the ice thickness. The
    # slope sets the width of the mask in arc length; without it the width is
    # fixed in sea level and spreads over many facets on the gentle B1 shelf
    # (2.51e-4) while it collapses inside one facet on the steep B2 one
    # (1.26e-3).
    SL_init = basin_sea_level(mantle, case, degree=args.sea_level_degree)
    slope = surface_slope(SL_init)
    cap = cap_thickness(mantle, case, t_kyr)
    # `I = m cap(t)` with `m` a CG1 field equal to one. The product changes no
    # value, because the interpolant of the constant 1 is exactly 1, and it
    # gives the gradient check of W11 a control field to differentiate with
    # respect to without another load expression.
    control = Function(FunctionSpace(mantle, "CG", 1),
                       name="ice_control").assign(1.0)
    ice = control * cap

    g_surface = surface_gravity(parent)
    sea_level = SeaLevel(
        boundary=gen.SURF_RE,
        rho_w=case.rho_water_nondim, rho_i=case.rho_ice_nondim,
        g_surface=g_surface, SL_init=SL_init, I=ice,
        # The run starts from the undeformed reference state, so the reference
        # ice, geoid and uplift are all zero.
        I_init=Constant(0.0), N_init=Constant(0.0), ur_init=Constant(0.0),
        alpha_mask=args.alpha_mask, slope=slope, grad_floor=args.grad_floor,
        fixed_ocean=case.fixed_ocean)

    core = FluidCore(boundary=gen.SURF_RC, rho_core=spada.RHO_CORE,
                     g=refstate.gravity_exact_ufl(Constant(gen.RC)))

    # A rigid rotation of the mantle stays in the kernel with sea level on:
    # its `u . n` is zero, so it moves no water, and its mass moment vanishes
    # for a spherically symmetric reference density, so the frame rows do not
    # see it either.
    nullspace = rigid_rotation_nullspace(Z, layout)

    # With a fixed coastline the sheet is affine in the unknowns and one
    # linear solve is exact to the outer tolerance. With live masks (case D)
    # the masks follow the solution and Newton is the method.
    snes_type = "ksponly" if case.fixed_ocean else "newtonls"
    # `n_real` is what the preset selects the block-1 treatment by when
    # `multiplier_pc` is left at the sentinel `None`: the width of the `Real`
    # block, not the name of the DtN representation. Here that block holds the
    # core pressure, the three centre-of-mass rows and the sea-level `Shift`,
    # and it gains the three rotation rows when polar motion is on, so the
    # count has to come from the layout and not from a constant in this file.
    solver_parameters = selfgrav_dtn_iterative_solver_parameters(
        condensed=layout.condensed, outer_rtol=args.outer_rtol,
        block0_rtol=args.block0_rtol, block0_max_it=args.block0_max_it,
        snes_type=snes_type, multiplier_pc=args.multiplier_pc,
        dtn_representation=layout.dtn_representation,
        n_real=len(layout.real_fields))
    # The two rebuild rules of W4. They are PETSc options of
    # `LowRankPotentialPC` and `DtNMultiplierDenseSchurPC`, so they are set
    # only when the caller names one and the default behaviour of those
    # classes is untouched otherwise.
    if args.lowrank_reuse_rtol is not None:
        solver_parameters["lowrank_reuse_rtol"] = args.lowrank_reuse_rtol
    if args.dense_schur_rebuild is not None:
        solver_parameters["dense_schur_rebuild"] = args.dense_schur_rebuild

    solver = SelfGravitatingGIASolver(
        z, approximation, layout=layout, dt=dt,
        # No `normal_stress` on Re: the ice is inside the sea-level sheet.
        bcs={}, fluid_core=core, sea_level=sea_level,
        dtn_representation=args.dtn_representation,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=solver_parameters)
    # The resolved block-1 route, read back out of the dictionary the preset
    # returned, so that the run records which of the three routes it took
    # rather than the sentinel the caller passed in. `dtn_schur_ainvb` is the
    # cached apply of `gadopt.DtNTwoBlockSchurPC`, which owns block 1 itself.
    if solver_parameters.get("dtn_schur_ainvb"):
        block1 = "gadopt.DtNTwoBlockSchurPC cached apply (ainvb)"
    else:
        block1 = solver_parameters.get("dtn_fieldsplit_1_pc_python_type",
                                       "none")
    pieces = {"SL_init": SL_init, "slope": slope, "ice": ice,
              "control": control, "g_surface": g_surface,
              "sea_level": sea_level, "snes_type": snes_type,
              "n_real": len(layout.real_fields), "block1": block1,
              "schur_fact_type":
                  solver_parameters["dtn_pc_fieldsplit_schur_fact_type"]}
    return solver, z, layout, pieces


# --------------------------------------------------------------------------
# Diagnostics of one solved state
# --------------------------------------------------------------------------

def python_context(pc):
    """The Python preconditioner behind `pc`, or `None` if there is not one.

    Ask the type first. `getPythonContext` on another type kills the process.
    `PETSc.PC.getPythonContext` on a preconditioner whose type is not `python`
    reads a pointer that holds no Python object and kills the process with
    `SIGSEGV`. Measured with petsc4py 3.25.5: a bare
    `PETSc.PC().create(); setType("gamg"); getPythonContext()` segmentation
    faults, while `none`, `jacobi`, `lu` and `fieldsplit` return `None`. So a
    `try/except` around the call catches nothing: there is no exception, the
    run dies. The same trap is recorded in `tests/unit/test_gia_condensed_block0.py`.

    This is reached in the driver because the potential split of the block-0
    nest carries `gadopt.LowRankPotentialPC` on the low-rank representation and
    plain GAMG on the multiplier representation
    (`gadopt.gia_gravity._potential_split`), and the driver runs both.

    Args:
      pc: a `PETSc.PC`, at any point of the nest.

    Returns:
      The Python context, or `None` when the preconditioner is of another type
      or carries no context.
    """
    try:
        if pc.getType() != "python":
            return None
        return pc.getPythonContext()
    except Exception:
        return None


def fieldsplit_sub_ksps(pc):
    """The sub-KSPs of a fieldsplit preconditioner, or `()` if it is not one.

    Same reasoning as `python_context`: ask the type first, then the question.
    A fieldsplit query on a preconditioner of another type, or on one that has
    not been set up, is not something to find out by trying.

    Args:
      pc: a `PETSc.PC`.

    Returns:
      A tuple of `PETSc.KSP`, empty when `pc` is not a set-up fieldsplit.
    """
    try:
        if pc.getType() != "fieldsplit":
            return ()
        return tuple(pc.getFieldSplitSubKSP())
    except Exception:
        return ()


def preconditioner_counters(solver):
    """The four rebuild and application counters of the preconditioners.

    `CondensedBlockPC` counts its block assemblies and its eliminations (one
    per block-0 application); `LowRankPotentialPC` counts its column builds;
    whichever class holds the exact complement of the `Real` block counts its
    builds, and `dense_builds` reports that count under either class.

    **The four counters are cumulative.** Each one counts from the build of
    its preconditioner and no run resets it, so the `TIMESTEP` line prints a
    running total next to a per-step wall time. The cost of one step is the
    difference between two lines. The counters say whether the rebuild rules
    of W4 did what they were set to do.

    The nest is walked through `python_context` and `fieldsplit_sub_ksps`,
    which ask each preconditioner's type before they ask it anything else.
    Every entry is `None` when that preconditioner is not in this
    configuration, because the driver runs both DtN representations and both
    block-1 settings and a missing counter must not stop a run. A missing
    counter prints as `-`.

    Args:
      solver: the `SelfGravitatingGIASolver` after a solve.

    Returns:
      A dictionary with the keys `assembly`, `block0`, `columns` and
      `dense_builds`.
    """
    out = {"assembly": None, "block0": None, "columns": None,
           "dense_builds": None}
    outer = python_context(solver.solver.snes.ksp.pc)
    if outer is None:
        return out
    try:
        block0_ksp, block1_ksp = outer.pc.getFieldSplitSchurGetSubKSP()
    except Exception:
        return out
    condensed = python_context(block0_ksp.getPC())
    out["assembly"] = getattr(condensed, "assembly_count", None)
    out["block0"] = getattr(condensed, "elimination_count", None)
    # The potential split of the block-0 nest carries `LowRankPotentialPC` on
    # the low-rank representation and GAMG on the multiplier one, so only the
    # first has a column counter. `block0="pair"` has no `condensed_ksp` at
    # all, which is why the attribute is asked for and not assumed.
    condensed_ksp = getattr(condensed, "condensed_ksp", None)
    if condensed_ksp is not None:
        splits = fieldsplit_sub_ksps(condensed_ksp.getPC())
        if len(splits) > 1:
            out["columns"] = getattr(python_context(splits[1].getPC()),
                                     "column_builds", None)
    # Two classes can hold the factored complement of the `Real` block.
    # `DtNMultiplierDenseSchurPC` sits on block 1 of the fieldsplit, which is
    # what an arm that passes `ainvb=False` or names the class gets. The
    # preset itself selects the cached apply of `DtNTwoBlockSchurPC` wherever
    # it forms the exact complement at all, and at the 5 `Real` rows of this
    # benchmark it does. That cache lives on the OUTER preconditioner and
    # leaves block 1 at `pc_type none`, so the block-1 query below finds
    # nothing there. Reading the outer cache first is what makes a
    # `dense_schur_rebuild` arm report the builds it actually paid for;
    # without it every row of such a run prints `-`.
    #
    # The field keeps the name `dense_builds` under both classes, because the
    # counted thing is the same -- one exact complement of the `Real` block,
    # `n` block-0 applications -- and the notes and the scripts that read the
    # `TIMESTEP` lines key on that name.
    cached_apply = getattr(outer, "ainvb", None)
    if cached_apply is not None:
        out["dense_builds"] = getattr(cached_apply, "build_count", None)
    else:
        out["dense_builds"] = getattr(python_context(block1_ksp.getPC()),
                                      "build_count", None)
    return out


def surface_measure(solver):
    """The sea-level measure restricted to Re, for the driver's own integrals."""
    return solver.sea_level_measure()(gen.SURF_RE)


def ocean_and_ice(solver, case, pieces):
    """Ocean area and ice masses of the current state, in SI units.

    The ocean function is `C0 = C(SL_init)` with a fixed coastline and the
    live `C(SL)` otherwise, always with the solver's own steepness, so these
    numbers read the same mask that the sheet integrates.

    The grounded-ice mass is the ice that rests on the bed. With a fixed
    coastline the ice over the reference ocean is removed by `1 - C0` and
    nothing floats, which is the approximation of Martinec eq. 7 and 8. With
    live masks the grounded-ice function `B` splits the two: `1 - B` is
    grounded and `B` is floating.

    Args:
      solver: the solved solver.
      case: the `MartinecCase`.
      pieces: the dictionary from `build_solver`.

    Three areas are returned, because with a moving coastline three
    natural integrals count three different regions:

    * `int B dS`, the ocean area of the benchmark (Martinec eq. 23): the
      points with a positive water column and no grounded ice on them, with
      floating ice counted as ocean. `B = 1 - H_k(I - (rho_w / rho_i) SL)`
      is 1 in open water and under floating ice, and 0 under grounded ice
      and on land, because on land `SL < 0` makes the argument positive
      whether or not ice is there. So `B` alone is the benchmark's ocean
      function, with one smooth step at each coast and at each grounding
      line.
    * `int C dS`, every point where the sea level is positive. It counts
      the bed under grounded marine ice, and land pushed below sea level
      under the cap, as ocean. On case D at 15 kyr that is 1.26e-3 of the
      sphere, 2.85 percent of the area.
    * `int B C dS`, the product that multiplies the sea level in the water
      term of the sheet. Where there is no ice, `B` is a second smooth step
      centred on the same coastline as `C`, and the product of two steps
      is narrower than either one by about `1 / k`. So this integral loses
      a strip about 17 km wide along every ice-free coast on the 78 km
      mesh: 5.42e-4 of the sphere at t = 0, 1.2 percent of the area, and
      the strip scales with the facet size. The strip holds almost no water,
      because `SL` is zero at the coast, so the sheet is barely affected;
      the area is. `NOTES/ocean-area/` has the two measurements.

    With a fixed coastline `B` does not exist and all three are `int C0 dS`.

    Returns:
      `(ocean_area, ocean_area_C, ocean_area_water, grounded_kg,
      floating_kg)`: the three areas above, in that order, each a fraction
      of the whole sphere, `int ... dS / (4 pi Re^2)`, and the two ice
      masses in kilograms.
    """
    dss = surface_measure(solver)
    k = solver._sea_level_steepness()
    ice = pieces["ice"]
    if case.fixed_ocean:
        C = ocean_function(pieces["SL_init"], k)
        grounded_integrand = (1 - C) * ice
        floating_integrand = None
        benchmark = C
        water = C
    else:
        SL = solver.sea_level()
        C = ocean_function(SL, k)
        B = grounded_ice_function(ice, SL, k, case.rho_water_nondim,
                                  case.rho_ice_nondim)
        grounded_integrand = (1 - B) * ice
        floating_integrand = B * ice
        # The benchmark's ocean function is `B` alone; see the docstring.
        benchmark = B
        # The product `B C` that multiplies the sea level in the water term
        # of the sheet, kept as a diagnostic of the coastal strip.
        water = B * C
    sphere = 4.0 * np.pi * gen.RE ** 2
    area = assemble(benchmark * dss) / sphere
    area_C = assemble(C * dss) / sphere
    water_area = assemble(water * dss) / sphere
    # Non-dimensional mass times rho_bar D^3 is kilograms: the densities are
    # in units of rho_bar, the thickness in units of D and the area in D^2.
    scale = refstate.RHO_BAR * D_M ** 3 * case.rho_ice_nondim
    grounded = scale * assemble(grounded_integrand * dss)
    floating = (0.0 if floating_integrand is None
                else scale * assemble(floating_integrand * dss))
    return (float(area), float(area_C), float(water_area), float(grounded),
            float(floating))


def state_row(solver, layout, case, pieces, t_kyr, dt_yr, step, wall_s):
    """Every per-step number of one solved state, as a dictionary.

    Collective: every value here is an assembled integral or a `Real` field,
    so every rank must call this.

    Args:
      solver: the solved solver.
      layout: the `GIASpaceLayout`.
      case: the `MartinecCase`.
      pieces: the dictionary from `build_solver`.
      t_kyr: the time at the end of the step.
      dt_yr: the step length in years.
      step: the global step index, counting from 1.
      wall_s: the wall-clock time of the solve.

    Returns:
      A dictionary of plain numbers and lists.
    """
    shift = float(solver.solution.subfunctions[layout.sea_level])
    multipliers = solver.centre_of_mass_multipliers()
    dipole = solver.mass_dipole()
    dss = surface_measure(solver)
    sheet = solver.surface_load_sheet()
    net_mass = float(assemble(sheet * dss))
    # The scale of the first mass moment: the largest moment that this load
    # could carry if all of its mass sat on one side of the sphere. The
    # moment arm is Re and the mass is the total absolute surface mass, so
    # `load_moment` has the units of `mass_dipole` and is never smaller than
    # the moment of the load itself. `abs(sigma)` and not `sigma`, because the
    # ocean and the ice carry opposite signs and the net mass of the sheet is
    # zero by construction, which would make a ratio against it meaningless.
    load_moment = float(assemble(gen.RE * abs(sheet) * dss))
    area, area_C, water_area, grounded, floating = ocean_and_ice(
        solver, case, pieces)
    counters = preconditioner_counters(solver)
    snes = solver.solver.snes
    abs_dipole = float(np.linalg.norm(dipole))
    return dict(
        step=step, t_kyr=t_kyr, dt_yr=dt_yr, wall_s=wall_s,
        newton=int(snes.getIterationNumber()),
        outer=int(snes.ksp.getIterationNumber()),
        block0=counters["block0"], assembly=counters["assembly"],
        columns=counters["columns"], dense_builds=counters["dense_builds"],
        shift=shift, h_UF_m=shift * D_M,
        multipliers=[float(v) for v in multipliers],
        dipole=[float(v) for v in dipole],
        abs_dipole=abs_dipole, load_moment=load_moment,
        # The frame criterion of W8.2. A bare `abs_D` has no scale: it is a
        # non-dimensional moment whose size follows the load. The ratio is the
        # fraction of the load's own moment that the centre of mass still
        # carries, so it is comparable between two cases, two meshes and two
        # DtN representations. Before any load exists the ratio is 0 by
        # definition.
        dipole_rel=(abs_dipole / load_moment if load_moment > 0.0 else 0.0),
        net_sheet_mass=net_mass, ocean_area=area, ocean_area_C=area_C,
        ocean_area_water=water_area,
        ice_mass_grounded_kg=grounded, ice_mass_floating_kg=floating)


def print_step(row):
    """One `TIMESTEP` line and one `FRAME` line, in the style of the Spada driver.

    The two lines are what a log is read for: the first is cost and the second
    is the frame. Both are one line each so that a log can be filtered with
    `grep` and parsed by a column reader.

    `newton`, `outer` and `wall_s` are of this step. `block0`, `assembly`,
    `columns` and `dense_builds` are running totals of the preconditioners
    (see `preconditioner_counters`), so the cost of one step is the difference
    between two lines. A `-` means that this configuration has no such
    counter.

    The `FRAME` line carries the frame multipliers, the mass dipole `D`, its
    norm, the moment scale `Re int |sigma| dS` and `rel_D`, the norm divided
    by that scale. `rel_D` is the number to read: it says which fraction of
    the load's own first moment the centre of mass still carries, and it is
    comparable between cases, meshes and DtN representations.
    """
    def count(value):
        """A counter that the configuration may not carry."""
        return "-" if value is None else str(value)

    say(f"TIMESTEP t_kyr={row['t_kyr']:.9g} dt_yr={row['dt_yr']:.9g} "
        f"step={row['step']} newton={row['newton']} outer={row['outer']} "
        f"block0={count(row['block0'])} assembly={count(row['assembly'])} "
        f"columns={count(row['columns'])} "
        f"dense_builds={count(row['dense_builds'])} "
        f"wall_s={row['wall_s']:.6f} shift={row['shift']:.12e} "
        f"h_UF_m={row['h_UF_m']:.9g} "
        f"net_sheet_mass={row['net_sheet_mass']:.6e} "
        f"ocean_area={row['ocean_area']:.9g} "
        f"ocean_area_C={row['ocean_area_C']:.9g} "
        f"ocean_area_water={row['ocean_area_water']:.9g} "
        f"ice_grounded_kg={row['ice_mass_grounded_kg']:.9g} "
        f"ice_floating_kg={row['ice_mass_floating_kg']:.9g}")
    say(f"FRAME t_kyr={row['t_kyr']:.9g} "
        f"lambda=({', '.join(f'{v:.6e}' for v in row['multipliers'])}) "
        f"D=({', '.join(f'{v:.6e}' for v in row['dipole'])}) "
        f"abs_D={row['abs_dipole']:.6e} "
        f"load_moment={row['load_moment']:.6e} "
        f"rel_D={row['dipole_rel']:.6e}")


# --------------------------------------------------------------------------
# Profiles: U, N and S along the two comparison meridians
# --------------------------------------------------------------------------

def read_profiles(case_letter):
    """The VEGA profile colatitudes of one case, from the JSON next to this file.

    Args:
      case_letter: `"B"`, `"C"` or `"D"`.

    Returns:
      A dictionary `{"load": {...}, "basin": {...}}`, each with the keys
      `longitude_deg`, `time_kyr` and `colatitude_deg` (a NumPy array).

    Raises:
      FileNotFoundError: if the JSON is missing.
    """
    if not os.path.exists(PROFILE_JSON):
        raise FileNotFoundError(
            f"{PROFILE_JSON} does not exist. Write it with "
            "`python write_profile_json.py --gia-mip <path>` on a machine "
            "that has the converted reference data.")
    with open(PROFILE_JSON) as handle:
        data = json.load(handle)
    out = {}
    for name, entry in data["cases"][case_letter].items():
        out[name] = {
            "longitude_deg": float(entry["longitude_deg"]),
            "time_kyr": float(entry["time_kyr"]),
            "colatitude_deg": np.asarray(entry["colatitude_deg"], dtype=float)}
    return out


def profile_points(colatitude_deg, longitude_deg, radius):
    """Cartesian points of one meridian profile, shape `(n, 3)`."""
    theta = np.radians(np.asarray(colatitude_deg, dtype=float))
    phi = np.radians(float(longitude_deg))
    return radius * np.column_stack((np.sin(theta) * np.cos(phi),
                                     np.sin(theta) * np.sin(phi),
                                     np.cos(theta)))


class PointSampler:
    """Point evaluation of expressions of one mesh, in the input order.

    A `VertexOnlyMesh` keeps only the points its parent mesh contains, and it
    keeps them in the order of the mesh partition. Both matter here: the
    profile points sit just below Re, where a point can fall outside the
    P2-curved cells, and the output must stay aligned with the published
    colatitudes. The class therefore evaluates into the `input_ordering` mesh,
    which restores the order the points were given in.

    **A missing point reads back as zero and not as an error.** The
    `default_missing_val` keyword of `interpolate` is not applied on this
    route (measured on this Firedrake), so a point outside the mesh is
    indistinguishable from a value of zero in the returned array. The
    constructor therefore locates the points by interpolating the constant 1,
    which is exactly 1.0 at a located point and 0.0 at a missing one, and
    every later evaluation writes `nan` where that mask is false.

    Build one of these per (mesh, profile) pair once, before the first solve,
    and evaluate at every epoch. Point location is the expensive half and it
    happens in the constructor.

    Attributes:
      n_points: how many points were asked for.
      found: a boolean array, `True` where the mesh contains the point.
    """

    def __init__(self, mesh, points):
        """Locate the points and record which ones were found.

        Args:
          mesh: the mesh to evaluate on.
          points: array of shape `(n, 3)`.
        """
        self.n_points = len(points)
        # "warn" and not "error": a missing point is reported as a count by
        # the caller, and a run must not stop at an epoch because one profile
        # point of several thousand fell outside a curved cell.
        self.vom = VertexOnlyMesh(mesh, points, missing_points_behaviour="warn")
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
        """Evaluate a scalar UFL expression at the points, in the input order.

        Args:
          expression: scalar UFL on the mesh of this sampler.

        Returns:
          A NumPy array of length `n_points`, `nan` at a missing point.
        """
        return np.where(self.found, self._raw(expression), np.nan)


def build_profile_samplers(parent, mantle, profiles, radius):
    """One `PointSampler` per profile and per mesh.

    The displacement lives on the mantle submesh and the potential on the
    parent, so each profile needs a sampler on each mesh at the same points.

    Args:
      parent, mantle: the two meshes.
      profiles: the dictionary from `read_profiles`.
      radius: the radius of the profile points, non-dimensional.

    Returns:
      `{name: {"points": ..., "mantle": PointSampler, "parent": PointSampler}}`.
    """
    out = {}
    for name, entry in profiles.items():
        points = profile_points(entry["colatitude_deg"],
                                entry["longitude_deg"], radius)
        out[name] = {"points": points,
                     "mantle": PointSampler(mantle, points),
                     "parent": PointSampler(parent, points)}
    return out


def write_profiles(path, solver, layout, samplers, profiles, g_surface, t_kyr):
    """U, N, S and RSL along both meridians at one epoch, as an npz file.

    U is the uplift `dot(u, rhat)` on the mantle mesh, N the geoid
    `psi / g_surface` on the parent and S the sea-surface variation
    `N + Shift`. RSL is `S - U`. All four are written in metres, which is what
    the benchmark compares.

    `g_surface` and not the approximation's `g(r)` divides the potential, so
    that the geoid written here is the geoid the sea-level equation solves
    with. The two differ by the change of `g` over the 1e-6 Re that the
    evaluation points sit below the surface.

    Args:
      path: the output file.
      solver: the solved solver.
      layout: the `GIASpaceLayout`.
      samplers: the dictionary from `build_profile_samplers`.
      profiles: the dictionary from `read_profiles`.
      g_surface: the reference gravity at Re, non-dimensional.
      t_kyr: the epoch, kyr.
    """
    shift = float(solver.solution.subfunctions[layout.sea_level])
    mantle = layout.mechanics_mesh
    Xm = SpatialCoordinate(mantle)
    # The two sub-functions and not `solution_split`: a split component of a
    # mixed space that spans two meshes carries both domains, and an
    # interpolation of it raises "Found multiple domains, cannot return just
    # one". `solver.displacement` and `solver.potential` are the sub-functions
    # on their own mesh.
    uplift = dot(solver.displacement, Xm / sqrt(dot(Xm, Xm)))
    geoid = solver.potential / Constant(g_surface)

    arrays = {"t_kyr": np.array(t_kyr), "shift_nondim": np.array(shift),
              "h_UF_m": np.array(shift * D_M)}
    for name, sampler in samplers.items():
        U = sampler["mantle"](uplift) * D_M
        N = sampler["parent"](geoid) * D_M
        S = N + shift * D_M
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
    say(f"      profiles written: {path}")


# --------------------------------------------------------------------------
# The ocean function on the radopt Gauss-Legendre grid
# --------------------------------------------------------------------------

def gauss_legendre_colatitudes(nglv):
    """The `nglv` Gauss-Legendre colatitudes of the radopt grid, north to south.

    `radopt.sealevel.sht.gauss_legendre` takes the nodes of
    `numpy.polynomial.legendre.leggauss` and polishes them with two Newton
    steps on the three-term recursion, then flips them so that index 0 is the
    ring nearest the north pole. The colatitude is then
    `90 - degrees(arcsin(x))`, which is the expression that
    `gia_mip.scripts.martinec_radopt.read_ocean_function` compares against.
    The same spelling is used here, because that reader refuses a coordinate
    that differs by more than 1e-8 degrees.

    Args:
      nglv: the number of latitude rings.

    Returns:
      An array of `nglv` colatitudes in degrees, ascending.
    """
    x = np.polynomial.legendre.leggauss(nglv)[0]
    for _ in range(2):
        previous, current = np.zeros_like(x), np.ones_like(x)
        for k in range(1, nglv + 1):
            previous, current = current, (
                (2.0 * k - 1.0) * x * current - (k - 1.0) * previous) / k
        derivative = nglv * (x * current - previous) / (x * x - 1.0)
        x = x - current / derivative
    x = x[::-1].copy()
    return 90.0 - np.degrees(np.arcsin(x))


def facet_size_field(mantle, boundary):
    """`h = sqrt(FacetArea)` of each Re facet, as a DG0 field on the cells.

    The mask steepness contains `FacetArea`, which exists only inside a facet
    integral. An export at an arbitrary point needs the size of the facet that
    contains the point, and every cell of the mantle mesh has at most one
    facet on Re, so a cell-wise DG0 field carries it without loss.

    The field is the facet-area weighted average of `sqrt(FacetArea)` over the
    boundary facets of each cell, which for one facet per cell is that facet's
    value exactly. Cells with no facet on the boundary keep zero.

    **A zero is not always an interior cell.** Most cells of the mantle are
    interior cells (59 940 of the 64 878 of the production coarse mesh), and a
    point on Re normally lands in one of the 4 938 cells that carry a facet
    there. A tetrahedral layer also holds cells that touch Re along one edge
    or one vertex only. Such a cell has no facet on Re, keeps zero here, and
    still contains points at the radius of the export: 4 of the 8 192 grid
    points at `nglv = 64` and 34 of the 131 072 at `nglv = 256`, measured on
    that mesh. `nodal_facet_size_field` covers those points, and the caller
    must not divide by the zero of this field.

    Args:
      mantle: the mechanics mesh.
      boundary: the tag of Re on that mesh.

    Returns:
      A DG0 `Function` of the facet size, non-dimensional.
    """
    space = FunctionSpace(mantle, "DG", 0)
    q = TestFunction(space)
    measure = ds(boundary, domain=mantle)
    numerator = assemble(q * sqrt(FacetArea(mantle)) * measure)
    denominator = assemble(q * measure)
    out = Function(space, name="facet_size")
    area = denominator.dat.data_ro
    values = np.zeros_like(area)
    nonzero = area > 0.0
    values[nonzero] = numerator.dat.data_ro[nonzero] / area[nonzero]
    out.dat.data_wo[:] = values
    return out


def nodal_facet_size_field(mantle, boundary):
    """The two CG1 fields of the facet size averaged over the nodes of Re.

    This covers the grid points of the export that sit in a cell with no facet
    on Re (see `facet_size_field`). Both fields are assembled over the facets
    of Re only:

        numerator[i]   = int phi_i sqrt(FacetArea) dS(Re),
        denominator[i] = int phi_i dS(Re),

    so a node on Re carries the area-weighted mean facet size of the facets
    that meet there, and every other node carries zero in both.

    The caller evaluates both fields at the point and divides. The weights of
    the nodes off Re cancel in that ratio, so the result is the mean facet
    size over the nodes of Re that the cell has, weighted by the basis
    functions at the point. It is the size of the facets next to the point and
    not of one facet, which is what the mask needs there.

    Measured on the production coarse mesh at `nglv = 64`: at the 8 188 points
    with a facet of their own the ratio reproduces that facet's size with a
    median relative difference of 1.5e-2, and at the 4 points without one it
    gives 0.1137 to 0.1179 against a range of 0.0594 to 0.1384 over the mesh.
    The largest differences (up to 0.82) are at the transition of the two
    refinement bands, where the facet sizes themselves jump.

    Args:
      mantle: the mechanics mesh.
      boundary: the tag of Re on that mesh.

    Returns:
      `(numerator, denominator)`, two CG1 `Function`s. The ratio is the facet
      size, non-dimensional, wherever the denominator is positive.
    """
    space = FunctionSpace(mantle, "CG", 1)
    q = TestFunction(space)
    measure = ds(boundary, domain=mantle)
    numerator = Function(space, name="facet_size_nodal_numerator")
    denominator = Function(space, name="facet_size_nodal_denominator")
    numerator.dat.data_wo[:] = assemble(
        q * sqrt(FacetArea(mantle)) * measure).dat.data_ro
    denominator.dat.data_wo[:] = assemble(q * measure).dat.data_ro
    return numerator, denominator


def dg0_ocean_function(solver, pieces, boundary):
    """`C0` averaged over each Re facet, as a DG0 field on the cells.

    This is the staircase export that the review of W12 rejected: the mask
    goes from 0.1 to 0.9 across about 1.5 facets, so a facet-wise constant is
    two or three levels wide across a coast. It is built here only so that the
    driver can print how far the accepted export is from it.

    Args:
      solver: the solver (for its own steepness and masks).
      pieces: the dictionary from `build_solver`.
      boundary: the tag of Re on the mechanics mesh.

    Returns:
      A DG0 `Function`, non-dimensional.
    """
    mantle = pieces["SL_init"].function_space().mesh()
    space = FunctionSpace(mantle, "DG", 0)
    q = TestFunction(space)
    measure = solver.sea_level_measure()(boundary)
    C0 = ocean_function(pieces["SL_init"], solver._sea_level_steepness())
    numerator = assemble(q * C0 * measure)
    denominator = assemble(q * measure)
    out = Function(space, name="C0_dg0")
    area = denominator.dat.data_ro
    values = np.zeros_like(area)
    nonzero = area > 0.0
    values[nonzero] = numerator.dat.data_ro[nonzero] / area[nonzero]
    out.dat.data_wo[:] = values
    return out


def export_ocean_function(path, solver, pieces, args, nglv):
    r"""Write `C0 = C(SL_init)` on the radopt Gauss-Legendre grid.

    The matched radopt runs of cases B and C use G-ADOPT's own ocean function,
    so that the width of the mask leaves the comparison
    (`gia-mip/NOTES/martinec-radopt/03-gadopt-matched.md`). The file holds
    `colatitude_deg` (nglv values, north to south), `longitude_deg`
    (`2 nglv` values, east from 0) and `C0` of shape `(nglv, 2 nglv)`, which
    is what `read_ocean_function` of `gia-mip/scripts/martinec_radopt.py`
    reads.

    **How each value is built.** G-ADOPT integrates the mask at the facet
    quadrature points, where `SL_init` (CG2) and the frozen slope both change
    inside a facet. The export therefore evaluates

        k  = alpha p / (h max(s, grad_floor)),
        C0 = 0.5 (1 + tanh(clamp(0.5 k SL_init)))

    at the grid point itself, with `h` the size of the Re facet that contains
    the point (a cell-wise DG0 field; a boundary cell has one Re facet) and
    `s` the frozen slope at the point. A point can also land in a cell that
    touches Re along one edge only, which has no facet there and no `h`. Such
    a point takes `h` from the facets that meet at the nodes of its cell on Re
    (`nodal_facet_size_field`), and the driver prints how many points took
    that route. A point that no facet of Re reaches is written as `nan` and
    counted in a warning, so that a count of zero means that every value of
    the file is a real one. Sampling a facet-wise `C0` instead
    gives a staircase of two or three levels across a coast, because the mask
    goes from 0.1 to 0.9 in about 1.5 facet sizes. The difference between the
    two constructions is printed.

    The grid is the grid of the radopt run, so each `nglv` needs its own
    export: 512 for the degree-256 run and a smaller one for the coarse run.

    Args:
      path: the output file.
      solver: the solver, for the steepness settings and the mesh.
      pieces: the dictionary from `build_solver`.
      args: the parsed command line, for `alpha_mask` and `grad_floor`.
      nglv: the number of latitude rings of the radopt grid.
    """
    mantle = pieces["SL_init"].function_space().mesh()
    colatitude = gauss_legendre_colatitudes(nglv)
    nlon = 2 * nglv
    longitude = np.degrees(2.0 * np.pi * np.arange(nlon) / nlon)
    radius = gen.RE * (1.0 - PROFILE_DEPTH_FRACTION)
    theta = np.radians(colatitude)[:, None]
    phi = np.radians(longitude)[None, :]
    points = np.column_stack((
        (radius * np.sin(theta) * np.cos(phi)).ravel(),
        (radius * np.sin(theta) * np.sin(phi)).ravel(),
        (radius * np.cos(theta) * np.ones_like(phi)).ravel()))

    sampler = PointSampler(mantle, points)
    say(f"  ocean-function export: {nglv} x {nlon} points, "
        f"{int(np.count_nonzero(sampler.found))} located")
    facet_size = facet_size_field(mantle, gen.SURF_RE)
    nodal_numerator, nodal_denominator = nodal_facet_size_field(
        mantle, gen.SURF_RE)
    sl_at = sampler(pieces["SL_init"])
    slope_at = sampler(pieces["slope"])
    h_at = sampler(facet_size)
    staircase = sampler(dg0_ocean_function(solver, pieces, gen.SURF_RE))

    # The facet size at the points that landed in a cell with no facet on Re,
    # where the cell-wise field holds zero. Without this the steepness is
    # infinite, the smooth step saturates at exactly 0.0 or 1.0 and nothing
    # separates that value from a legitimate saturated one. The nodal average
    # is the size of the facets next to the point.
    #
    # The two nodal fields are sampled on every rank and not only where a
    # point failed: a point evaluation is collective, and a rank that holds no
    # point must take part in it as well.
    no_facet = np.isfinite(h_at) & (h_at <= 0.0)
    nodal_h = sampler(nodal_numerator)
    nodal_weight = sampler(nodal_denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        fallback = np.where(nodal_weight > 0.0, nodal_h / nodal_weight, np.nan)
    h_at = np.where(no_facet, fallback, h_at)
    # What is left is a point that no facet of Re reaches at all. It is
    # written as `nan` and counted, and it must never come out of the tanh as
    # a finite number.
    unusable = ~np.isfinite(h_at) | (h_at <= 0.0)

    # The same steepness and the same clamped smooth step as the solver, in
    # NumPy. `p` is the highest polynomial degree of the displacement and the
    # potential, which is what `mask_steepness` is given.
    degree = solver._sea_level_polynomial_degree()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (args.alpha_mask * degree
             / (h_at * np.maximum(slope_at, args.grad_floor)))
        argument = np.clip(0.5 * k * sl_at, -SMOOTH_STEP_CLAMP,
                           SMOOTH_STEP_CLAMP)
        C0 = 0.5 * (1.0 + np.tanh(argument))
    C0 = np.where(np.isfinite(C0) & ~unusable & np.isfinite(k), C0, np.nan)

    # The number the review of W12 asked for: how far the accepted export is
    # from the staircase a DG0 facet sample would give. Only the points both
    # constructions have are compared, so a point the mesh does not contain
    # does not enter the number.
    #
    # Every reduction below runs on the rank that holds the points. The
    # `input_ordering` mesh of `PointSampler` is redundant, so rank 0 holds
    # every point and every other rank holds none: a reduction of an empty
    # array raises, and Python builds the string of `say` on every rank.
    both = np.isfinite(C0) & np.isfinite(staircase)
    if np.any(both):
        difference = np.abs(C0[both] - staircase[both])
        say(f"  export against a DG0 facet sample, over {int(both.sum())} "
            f"points: max {difference.max():.6e}, rms "
            f"{np.sqrt(np.mean(difference ** 2)):.6e}, mean C0 "
            f"{C0[both].mean():.9f} against {staircase[both].mean():.9f}")
    recovered = int(np.count_nonzero(no_facet & np.isfinite(C0)))
    if recovered:
        say(f"  {recovered} grid points sit in a cell with no facet on Re; "
            "their mask width comes from the facet sizes at the nodes of "
            "that cell on Re.")
    missing = int(np.count_nonzero(~np.isfinite(C0)))
    if missing:
        say(f"  WARNING: {missing} grid points have no value and are written "
            "as nan. The mesh did not contain them, or no facet of Re "
            "reaches them. Raise PROFILE_DEPTH_FRACTION and export again.")
    if COMM_WORLD.rank == 0:
        np.savez(path, colatitude_deg=colatitude, longitude_deg=longitude,
                 C0=C0.reshape(nglv, nlon),
                 C0_dg0_sample=staircase.reshape(nglv, nlon))
    say(f"  ocean function written: {path}")


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

def case_ladder(case, args, end_yr):
    """The time-step ladder of a case, as `(t_end_yr, dt_yr)` segments.

    Scenario T0 (case B) takes the graded Spada ladder: 10 yr to 0.1 kyr,
    50 yr to 1 kyr and 100 yr to 10 kyr. The load is a step at t = 0, so the
    early relaxation is fast and needs the short steps.

    Scenario T1 (cases C and D) takes uniform 50 yr steps. The load grows over
    10 kyr, so there is no fast early transient, and radopt's case C load
    profile still moved between 100 yr and 50 yr steps.

    Both ladders stop at `end_yr`, the last epoch, because `time_ladder`
    marches over the whole ladder it is given: a ladder that runs past the
    last epoch marches to the end of the case and writes nothing after that
    epoch.

    `--dt-yr` replaces either ladder with one uniform step, and `--ladder`
    replaces it with the segments given on the command line. `--ladder` is
    used as given and is not cut.

    Args:
      case: the `MartinecCase`.
      args: the parsed command line.
      end_yr: the last epoch, in years.

    Returns:
      A tuple of `(t_end_yr, dt_yr)` pairs.
    """
    if args.ladder:
        return tuple(args.ladder)
    if args.dt_yr is not None:
        return ((end_yr, args.dt_yr),)
    if case.scenario == "T0":
        return spada.truncated_ladder(end_yr)
    return ((end_yr, 50.0),)


def parse_ladder_segment(text):
    """One `--ladder` argument, `t_end_yr:dt_yr`, as a pair of floats."""
    try:
        end, step = text.split(":")
        return (float(end), float(step))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--ladder takes segments of the form t_end_yr:dt_yr, got {text!r}")


def step_end_times(segments):
    """The end time of every step of the march, in kyr.

    The matched radopt runs take this ladder, through the array `t_kyr` of the
    step-time file (`read_step_times` of `gia-mip/scripts/martinec_radopt.py`
    refuses a file without that name). The elastic solve of scenario T0 is not
    a step of the march and is not listed; the reader adds the time 0 in front
    by itself.

    Args:
      segments: the segments from `spada.time_ladder`.

    Returns:
      A NumPy array of step end times in kyr.
    """
    times = []
    for t0, _, dt_yr, nsteps, _ in segments:
        times.extend((t0 + (k + 1) * dt_yr) / 1000.0 for k in range(nsteps))
    return np.asarray(times, dtype=float)


# --------------------------------------------------------------------------
# Checkpoint: write and restart
# --------------------------------------------------------------------------

def save_state(chk, solver, layout, index, t_kyr, step, dt_yr):
    """Write one state, in a form a restart can read back exactly.

    **Why the mantle fields are written on the parent mesh.** A restart has to
    rebuild the coupled system, and the coupled system needs the mantle to be
    a `Submesh` of the parent: the sea-level and DtN measures intersect the
    mantle's `ds` with the parent's `dS`, which only a submesh relation
    supports. A mesh read back with `CheckpointFile.load_mesh` is an
    independent mesh and carries no such relation, and a submesh cut again
    from the loaded parent is not a mesh that `load_function` accepts
    (`AttributeError: 'MeshTopology' object has no attribute 'sfXC'`). What
    does work is to load the parent, cut the submesh from it, and read the
    mantle fields from parent-mesh copies: interpolation between a submesh and
    its parent is exact in both directions (measured: 0.0 error for CG3 and
    for DG2 tensors).

    So each state is written three times on the parent mesh - displacement,
    potential and internal variables - and the two mantle fields are written
    on the mantle as well, for a reader that wants them there. The `Real`
    fields are not written: they are not state that a step carries forward,
    and the next solve recomputes them.

    **The displacement copy on the parent depends on the rank count**, which is
    why the rank count is stored with the state and `load_state` warns when it
    changes. `allow_missing_dofs=True` leaves a parent dof that the mantle does
    not cover at zero, and the set of covered dofs comes from point location,
    which is a local operation. Measured in W8 on the 900 km mesh, the same
    state written on one rank and on two ranks: the displacement copy holds
    113 493 zero dofs of 355 890 on one rank and 113 520 on two, a difference of
    27 dofs, all of them on the mantle boundary where a CG3 dof is shared with
    a cell outside the mantle. The internal-variable copy is DG2, so a dof
    belongs to exactly one cell and the two counts are identical (538 080). The
    potential is a genuine parent field and has no such copy at all.

    The same measurement puts the effect at 3.5e-05 in the L2 norm of the
    displacement copy, against 5.6e-09 for the potential. A restart therefore
    reads a state that is right to that order when the rank count changed, and
    exactly right when it did not (measured at 5.3e-22 in
    `team/s6-w7-driver/review-round2.md` section 2, and at 5.8e-11 in `Shift`
    across a change from 8 ranks to 4 in W8 check 3). Restart on the rank count
    that wrote the checkpoint.

    Args:
      chk: the open `CheckpointFile`.
      solver: the solver whose `solution` holds the state.
      layout: the `GIASpaceLayout`.
      index: the checkpoint index.
      t_kyr: the time of the state.
      step: the global step index of the state.
      dt_yr: the step length that produced it.
    """
    z = solver.solution
    parent = layout.potential_mesh

    # The displacement and the internal variables, on their own mesh.
    for index_field, name in ((layout.displacement, "displacement"),
                              (layout.internal_variable_field,
                               "internal_variables")):
        if index_field is None:
            continue
        function = z.subfunctions[index_field]
        function.rename(name)
        chk.save_function(function, idx=index)
    potential = z.subfunctions[layout.potential]
    potential.rename("potential")
    chk.save_function(potential, idx=index)

    # The same two fields on the parent mesh, which is what a restart reads.
    for index_field, name in ((layout.displacement, "displacement_on_parent"),
                              (layout.internal_variable_field,
                               "internal_variables_on_parent")):
        if index_field is None:
            continue
        source = z.subfunctions[index_field]
        target = Function(_parent_space(source, parent), name=name)
        target.interpolate(source, allow_missing_dofs=True)
        chk.save_function(target, idx=index)

    # The time and the step index of the state, so that `--restart` continues
    # the ladder at the right place instead of being told where it is.
    chk.require_group(STATE_GROUP)
    chk.set_attr(STATE_GROUP, f"t_kyr_{index}", float(t_kyr))
    chk.set_attr(STATE_GROUP, f"step_{index}", int(step))
    chk.set_attr(STATE_GROUP, f"dt_yr_{index}", float(dt_yr))
    # The rank count of the run that wrote the state. `load_state` warns when
    # the restart uses another one, because the displacement copy above is
    # partition dependent on the mantle boundary.
    chk.set_attr(STATE_GROUP, f"ranks_{index}", int(COMM_WORLD.size))
    chk.set_attr(STATE_GROUP, "n_states", int(index + 1))


def _parent_space(function, parent):
    """The same element as `function`, built on the parent mesh.

    The displacement is a vector CG space and the internal variables a tensor
    DG space of shape `(n, d, d)`. Both are rebuilt from the element of the
    source function, so that a change of degree or of the element count in the
    solver reaches the checkpoint without an edit here.
    """
    element = function.ufl_element().reconstruct(cell=parent.ufl_cell())
    return FunctionSpace(parent, element)


def load_state(path, index, args):
    """Read the meshes and one state of a checkpoint, for `--restart`.

    The parent mesh comes from the file, so that `load_function` has the
    topology it was written against. The mantle is cut from it and curved
    again, exactly as `spada.build_meshes` does for a fresh run, which is what
    keeps the submesh relation the coupled measures need.

    Args:
      path: the checkpoint file.
      index: the state index inside it.
      args: the parsed command line (unused today, kept so that a future
        option such as a different mesh name has somewhere to go).

    Returns:
      `(parent, mantle, fields, t_kyr, step, dt_yr)` with `fields` a
      dictionary of parent-mesh `Function`s keyed by
      `"displacement"`, `"potential"` and `"internal_variables"`.

    Raises:
      FileNotFoundError: if the checkpoint is missing.
      IndexError: if the state index is not in the file.
    """
    del args
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--restart needs {path}, which does not exist. Pass the "
            "checkpoint of the earlier run with --restart-file.")
    with CheckpointFile(path, "r") as chk:
        n_states = int(chk.get_attr(STATE_GROUP, "n_states"))
        if not 0 <= index < n_states:
            raise IndexError(
                f"{path} holds {n_states} states, so --restart {index} is out "
                f"of range (0 to {n_states - 1}).")
        parent = chk.load_mesh(PARENT_NAME)
        # The geometry is a sphere, so the "vertical" direction of G-ADOPT's
        # terms is radial and not the last Cartesian axis.
        parent.cartesian = False
        # `Submesh` does not inherit the parent's P2 coordinates, so the cut
        # submesh is curved again with the same radial map, exactly as
        # `spada.build_meshes` does for a fresh run.
        mantle = gen.curve_mesh(Submesh(parent, 3, gen.CELL_MANTLE),
                                name=MANTLE_NAME)
        mantle.cartesian = False
        fields = {
            "displacement": chk.load_function(parent, "displacement_on_parent",
                                              idx=index),
            "potential": chk.load_function(parent, "potential", idx=index),
            "internal_variables": chk.load_function(
                parent, "internal_variables_on_parent", idx=index)}
        t_kyr = float(chk.get_attr(STATE_GROUP, f"t_kyr_{index}"))
        step = int(chk.get_attr(STATE_GROUP, f"step_{index}"))
        dt_yr = float(chk.get_attr(STATE_GROUP, f"dt_yr_{index}"))
        # The displacement copy on the parent is partition dependent on the
        # mantle boundary (`save_state`), so a restart on another rank count
        # reads a state that differs there. The measured size is 3.5e-05 in the
        # L2 norm of the displacement, which is above the outer tolerance, so
        # the log says it rather than leaving it to be rediscovered. A file
        # written before this attribute existed has no rank count and the
        # warning is skipped.
        try:
            written_ranks = int(chk.get_attr(STATE_GROUP, f"ranks_{index}"))
        except Exception:
            written_ranks = None
    if written_ranks is not None and written_ranks != COMM_WORLD.size:
        say(f"  WARNING: state {index} of {path} was written on "
            f"{written_ranks} ranks and this restart runs on "
            f"{COMM_WORLD.size}. The displacement copy on the parent mesh is "
            "partition dependent on the mantle boundary, so the restored "
            "state differs there by about 3.5e-05 relative (see save_state). "
            "Restart on the rank count that wrote the checkpoint.")
    return parent, mantle, fields, t_kyr, step, dt_yr


def install_state(solver, layout, fields):
    """Put a restored state into `solution` and into the history.

    Backward Euler reads the internal variables of the previous step from
    `solution_old`, so both copies have to carry the restored state: the
    solver copies `solution` into `solution_old` only after a solve.

    Args:
      solver: the freshly built solver.
      layout: the `GIASpaceLayout`.
      fields: the parent-mesh functions from `load_state`.
    """
    for target in (solver.solution, solver.solution_old):
        # The two mantle fields come down from the parent copies. Between a
        # submesh and its parent the interpolation is exact in both
        # directions, and `allow_missing_dofs` is what covers the parent
        # nodes outside the mantle, which the submesh does not ask for.
        target.subfunctions[layout.displacement].interpolate(
            fields["displacement"], allow_missing_dofs=True)
        if layout.internal_variable_field is not None:
            target.subfunctions[layout.internal_variable_field].interpolate(
                fields["internal_variables"], allow_missing_dofs=True)
        # The potential is already on the parent mesh, in the same element.
        # `interpolate` and not `assign`: the loaded function's space is built
        # by the checkpoint reader, and `assign` requires the two spaces to be
        # the same object. Between identical spaces on one mesh the
        # interpolation reproduces every node exactly.
        target.subfunctions[layout.potential].interpolate(
            fields["potential"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    """The command line; every default is the production configuration."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["B", "C", "D"], default="B",
                   help="the benchmark case")
    p.add_argument("--gia-mip", default=DEFAULT_GIA_MIP,
                   help="the gia-mip checkout, for the case and load files "
                        "(default: $GIA_MIP or ~/Workplace/gia-mip)")
    p.add_argument("--mesh", default=DEFAULT_MESH,
                   help="the gmsh file. The default is the 78 km refined mesh "
                        "of this directory; the smoke runs and the first "
                        "comparison job use ../3d_spada_selfgrav/"
                        "b2_coarse_ar7.msh. Copy both files, do not "
                        "regenerate them.")
    p.add_argument("--dtn-representation", choices=["multiplier", "lowrank"],
                   default="lowrank",
                   help="the representation of the exterior DtN condition")
    p.add_argument("--dtn-degree", type=int, default=5,
                   help="degree L of SphericalDtN on both truncation spheres")
    p.add_argument("--bulk-shear-ratio", type=float, default=100.0,
                   help="K / mu in every layer")
    p.add_argument("--displacement-degree", type=int, default=3,
                   help="polynomial degree of the CG displacement")
    p.add_argument("--internal-variable-degree", type=int, default=2,
                   help="polynomial degree of the DG internal variables")
    p.add_argument("--potential-degree", type=int, default=2,
                   help="the polynomial degree of the gravitational potential "
                        "space. The geoid N = psi / g_surface is read from "
                        "this space, so it is what sets the accuracy of the "
                        "N profiles, and the default 2 is one degree below "
                        "the displacement.")
    p.add_argument("--sea-level-degree", type=int, default=2,
                   help="polynomial degree of the CG initial sea level")
    p.add_argument("--dt-yr", type=float, default=None,
                   help="replace the ladder of the case with one uniform "
                        "step, in years")
    p.add_argument("--ladder", type=parse_ladder_segment, nargs="+",
                   default=None, metavar="T_END_YR:DT_YR",
                   help="replace the ladder of the case with these segments")
    p.add_argument("--epochs", type=float, nargs="+", default=None,
                   help="output epochs in kyr; the default is the list of the "
                        "time scenario plus the time of the published "
                        "profiles")
    p.add_argument("--outer-rtol", type=float, default=1e-6,
                   help="relative tolerance of the outer FGMRES")
    p.add_argument("--block0-rtol", type=float, default=1e-4,
                   help="relative tolerance of the mechanics-potential block "
                        "solve inside the outer FGMRES")
    # The outer Krylov is FGMRES, so block 0 is allowed to stop early: an
    # inexact block-0 solve costs outer iterations and does not move the
    # converged answer, because the outer tolerance is what the answer is
    # measured against. The preset caps block 0 at 200 by default, and on a
    # coarse or badly shaped mesh that cap is reached at every application,
    # which makes the wall clock of one step a multiple of the cap. Lower the
    # cap to trade block-0 iterations for outer iterations, and record the
    # setting with the counts.
    #
    # CAUTION: this flag is for a cheap check and not for a production run. A
    # capped block-0 solve builds the dense Schur complement of block 1 from an
    # inexact operator, in the same way a loose `--block0-rtol` does. Measured
    # in W8 on the 900 km mesh: the relative asymmetry of the 5x5 complement is
    # 5.1e-06 at cap 200 and 6.0e-04 at cap 20, and at the low cap the step
    # after the elastic solve cost 56 block-0 applications where the run at cap
    # 200 needed 17. A low enough cap can stop the outer solve converging at
    # all.
    p.add_argument("--block0-max-it", type=int, default=200,
                   help="iteration cap of the mechanics-potential block "
                        "solve (the preset default is 200)")
    p.add_argument("--multiplier-pc", default=None,
                   help="the preconditioner of the Real block. The default "
                        "leaves the preset to choose it from the number of "
                        "Real rows: at a narrow block that is the cached "
                        "apply of gadopt.DtNTwoBlockSchurPC under "
                        "schur_fact_type full, and nothing is put in the "
                        "block-1 KSP. Naming a class here, or 'none', keeps "
                        "the older delegating path under schur_fact_type "
                        "lower.")
    p.add_argument("--lowrank-reuse-rtol", type=float, default=None,
                   help="W4: rebuild the low-rank columns only when the "
                        "potential block moved by more than this relative "
                        "amount")
    p.add_argument("--dense-schur-rebuild", choices=["always", "per_solve"],
                   default=None,
                   help="W4: when the dense Schur complement on block 1 is "
                        "rebuilt")
    p.add_argument("--alpha-mask", type=float, default=DEFAULT_ALPHA_MASK,
                   help="the factor of the mask steepness k = alpha p / (h s)")
    p.add_argument("--grad-floor", type=float, default=DEFAULT_GRAD_FLOOR,
                   help="the lower bound of the frozen surface slope")
    p.add_argument("--restart", type=int, default=None, metavar="INDEX",
                   help="continue from state INDEX of the checkpoint that "
                        "--restart-file names")
    p.add_argument("--restart-file", default=None,
                   help="the checkpoint that --restart reads. It is required "
                        "with --restart, and it must not be the output file "
                        "of the continued run, so the continued run also "
                        "needs its own --label")
    p.add_argument("--steps", type=int, default=None, metavar="N",
                   help="stop after N steps, for a smoke run")
    p.add_argument("--label", default=None,
                   help="output file label; default the case letter")
    p.add_argument("--output", default=HERE,
                   help="directory for the checkpoint and the npz files")
    p.add_argument("--export-ocean-function", type=int, default=None,
                   metavar="NGLV",
                   help="write C0 on the radopt Gauss-Legendre grid of NGLV "
                        "latitude rings and 2 NGLV longitudes")
    p.add_argument("--dry-run", action="store_true",
                   help="build the meshes, the load and the solver, assemble "
                        "the residual once, and exit before any solve")
    args = p.parse_args(argv)
    # A restart reads one file and writes another, and the driver has no way
    # to guess the first one: a default that follows --label moves with the
    # label of the continued run and can therefore only name the file that
    # this run is about to write. The caller names the earlier file.
    if args.restart is not None and args.restart_file is None:
        p.error("--restart needs --restart-file, the checkpoint of the "
                "earlier run. Give the continued run its own --label as "
                "well, so that it does not write over that file.")
    return args


def epochs_of(case, args, profiles):
    """The output epochs in kyr.

    Without `--epochs` the list is the default of the time scenario plus the
    time of every published profile, because a default run that does not stop
    at the profile time has nothing to compare. With `--epochs` the list is
    exactly what the caller asked for: the run stops at the last epoch, so
    adding the profile time to a short list would march the whole case.

    Args:
      case: the `MartinecCase`.
      args: the parsed command line.
      profiles: the dictionary from `read_profiles`.

    Returns:
      A sorted list of epochs in kyr, without duplicates.
    """
    if args.epochs is not None:
        wanted = list(args.epochs)
    else:
        wanted = list(DEFAULT_EPOCHS[case.scenario])
        wanted.extend(entry["time_kyr"] for entry in profiles.values())
    return sorted({round(float(value), 9) for value in wanted})


def main(argv=None):
    """Run one Martinec case."""
    args = parse_args(argv)
    label = args.label or args.case
    case = MartinecCase(args.case, args.gia_mip)
    profiles = read_profiles(args.case)
    epochs = epochs_of(case, args, profiles)

    say("=" * 78)
    say(f"Martinec et al. (2018) sea-level benchmark, case {args.case}")
    say("=" * 78)
    say(case.describe())
    say(f"  mesh {args.mesh}")
    say(f"  DtN {args.dtn_representation}, SphericalDtN(L={args.dtn_degree}), "
        f"block-1 preconditioner "
        f"{args.multiplier_pc if args.multiplier_pc is not None else 'chosen by the preset'}")
    say(f"  displacement CG{args.displacement_degree}, internal variables "
        f"DG{args.internal_variable_degree}, potential "
        f"CG{args.potential_degree}, initial sea level "
        f"CG{args.sea_level_degree}, K/mu {args.bulk_shear_ratio:g}")
    say(f"  tolerances: outer {args.outer_rtol:g}, block 0 "
        f"{args.block0_rtol:g}, block-0 iteration cap {args.block0_max_it}")
    say(f"  masks: alpha {args.alpha_mask:g}, grad_floor {args.grad_floor:g}, "
        f"clamp {SMOOTH_STEP_CLAMP:g}")
    # The two rebuild rules of W4 are what job G4-rule compares, so the output
    # file has to name the values it ran with. `library default` is printed
    # when the flag was not given, because the default lives in
    # `gadopt/preconditioners.py` and not here, and a number printed for it
    # would go stale the moment that default moved.
    say("  cache rules: lowrank_reuse_rtol "
        f"{'library default' if args.lowrank_reuse_rtol is None else format(args.lowrank_reuse_rtol, 'g')}"
        ", dense_schur_rebuild "
        f"{args.dense_schur_rebuild or 'library default'}")
    say(f"  Lambda {LAMBDA:.6f}, B_mu {B_MU:.6f}, t_bar {T_BAR_YR} yr")

    # Meshes: a fresh run cuts them from the gmsh file, a restart takes the
    # parent from the checkpoint so that the state can be read back.
    # `--restart-file` is required with `--restart` (see `parse_args`), so
    # `restart_path` is what the caller named and never a guess.
    restart_path = args.restart_file
    output_path = os.path.join(args.output, f"martinec-{label}.h5")
    restored = None
    tic = time.time()
    if args.restart is None:
        parent, mantle = spada.build_meshes(args.mesh)
        parent.name = PARENT_NAME
        mantle.name = MANTLE_NAME
        start_t_kyr, start_step = 0.0, 0
    else:
        if os.path.abspath(restart_path) == os.path.abspath(output_path):
            raise ValueError(
                f"--restart would overwrite the checkpoint it reads, "
                f"{output_path}. Give the continued run its own --label, so "
                "that its output file has another name than the file "
                "--restart-file names.")
        (parent, mantle, restored_fields, start_t_kyr, start_step,
         _) = load_state(restart_path, args.restart, args)
        restored = restored_fields
        say(f"  restart: state {args.restart} of {restart_path}, "
            f"t = {start_t_kyr:g} kyr after {start_step} steps")
    say(f"  meshes: parent {parent.comm.allreduce(parent.cell_set.size)} "
        f"cells, mantle {mantle.comm.allreduce(mantle.cell_set.size)} cells "
        f"({time.time() - tic:.1f} s)")

    # The ladder stops at the last epoch, so that a run asked for two epochs
    # does not march the whole case and write nothing after the second one.
    ladder = case_ladder(case, args, max(epochs) * 1000.0)
    segments = spada.time_ladder(epochs, ladder)
    all_steps = step_end_times(segments)
    say(f"  epochs (kyr) {epochs}")
    say(f"  ladder (t_end_yr, dt_yr) {ladder}")
    say(f"  {len(segments)} segments, {len(all_steps)} steps")

    # The elastic solve exists only for a Heaviside load that is held: the
    # t = 0 state is then the instantaneous elastic response. A growing load
    # (T1) is zero at t = 0 and needs no such solve, and a restart continues a
    # march that already has one.
    solve_elastic = (case.scenario == "T0" and args.restart is None
                     and any(abs(e) < 1e-12 for e in epochs))
    first_dt = (DT_ELASTIC if solve_elastic or not segments
                else segments[0][2] / T_BAR_YR)
    dt = Constant(first_dt)
    t_kyr = Constant(start_t_kyr)

    tic = time.time()
    solver, z, layout, pieces = build_solver(parent, mantle, case, args, dt,
                                             t_kyr)
    say(f"  solver built ({time.time() - tic:.1f} s): layout "
        f"{'condensed' if layout.condensed else 'full'}, DtN representation "
        f"{layout.dtn_representation}, {len(layout.multipliers)} DtN "
        f"multipliers, frame fields {tuple(layout.centre_of_mass)}, Shift "
        f"field {layout.sea_level}, snes_type {pieces['snes_type']}")
    say(f"  block 1: {pieces['n_real']} Real rows, preconditioner "
        f"{pieces['block1']}, schur_fact_type {pieces['schur_fact_type']}")
    say(f"  unknowns: {z.function_space().dim()}")
    say(f"  g_surface: model {pieces['g_surface']:.9f} (non-dimensional), "
        f"{pieces['g_surface'] * refstate.G_BAR:.6f} m/s^2; case file "
        f"{case.g_case:.6f} m/s^2; relative difference "
        f"{abs(pieces['g_surface'] * refstate.G_BAR - case.g_case) / case.g_case:.3e}")
    if restored is not None:
        install_state(solver, layout, restored)
        say("  restart state installed in `solution` and `solution_old`")

    # The ice mass of the terminal cap against its closed form, as a check of
    # the load expression and of the surface quadrature on this mesh.
    t_kyr.assign(case.terminal_time_kyr)
    dss = surface_measure(solver)
    assembled = float(assemble(Constant(case.rho_ice_nondim)
                               * pieces["ice"] * dss))
    closed = cap_mass_closed_form(case, case.terminal_time_kyr, gen.RE)
    say(f"  cap mass at {case.terminal_time_kyr:g} kyr: assembled "
        f"{assembled * refstate.RHO_BAR * D_M ** 3:.9e} kg, closed form "
        f"{closed * refstate.RHO_BAR * D_M ** 3:.9e} kg, relative difference "
        f"{abs(assembled - closed) / abs(closed):.3e}")
    t_kyr.assign(start_t_kyr)

    samplers = build_profile_samplers(parent, mantle, profiles,
                                      gen.RE * (1.0 - PROFILE_DEPTH_FRACTION))
    for name, sampler in samplers.items():
        found = int(np.count_nonzero(sampler["mantle"].found
                                     & sampler["parent"].found))
        say(f"  profile {name}: {found} of {sampler['mantle'].n_points} "
            f"points located on both meshes")

    # The step times of the matched radopt runs, known before the first solve.
    steptimes_path = os.path.join(args.output,
                                  f"martinec-{label}-steptimes.npz")
    if COMM_WORLD.rank == 0:
        np.savez(steptimes_path, t_kyr=all_steps)
    say(f"  step times written: {steptimes_path}")

    if args.export_ocean_function is not None:
        if not case.fixed_ocean:
            raise ValueError(
                f"--export-ocean-function is for the fixed-coastline cases. "
                f"Case {case.letter} solves the sea-level equation on a "
                "moving ocean geometry, so there is no single C0 to hand to "
                "radopt, and `solve_sea_level` accepts a prescribed ocean "
                "function only with fixed_ocean = 1.")
        nglv = args.export_ocean_function
        export_ocean_function(
            os.path.join(args.output,
                         f"martinec-{label}-ocean_function_C0_nglv{nglv}.npz"),
            solver, pieces, args, nglv)

    if args.dry_run:
        tic = time.time()
        residual = assemble(solver.F)
        with residual.dat.vec_ro as vec:
            norm = vec.norm()
        # The state is the undeformed one in a fresh run and the restored one
        # after `--restart`, so the line names which of the two it measured.
        state_name = "the zero state" if args.restart is None else (
            f"the restored state at {start_t_kyr:g} kyr")
        say(f"  residual of {state_name} assembled "
            f"({time.time() - tic:.1f} s): l2 norm {norm:.6e}")
        # The areas and ice masses of the same state. After `--restart` this
        # scores a saved state with the current diagnostics and no solve,
        # which is how a checkpoint written before a diagnostic changed is
        # read again.
        area, area_C, water_area, grounded, floating = ocean_and_ice(
            solver, case, pieces)
        say(f"STATE t_kyr={start_t_kyr:.9g} ocean_area={area:.9g} "
            f"ocean_area_C={area_C:.9g} ocean_area_water={water_area:.9g} "
            f"ice_grounded_kg={grounded:.9g} ice_floating_kg={floating:.9g}")
        say("DRY RUN complete: no solve was run.")
        return

    rows = []
    epoch_index = [0]

    def record(t_step_kyr, dt_yr, step, wall_s):
        """One solved step: the row, the two lines and the time series."""
        row = state_row(solver, layout, case, pieces, t_step_kyr, dt_yr, step,
                        wall_s)
        rows.append(row)
        print_step(row)
        return row

    def write_timeseries():
        """The time series of every step so far, overwritten at each epoch."""
        path = os.path.join(args.output, f"martinec-{label}-timeseries.npz")
        if COMM_WORLD.rank == 0 and rows:
            keys = ("t_kyr", "dt_yr", "wall_s", "shift", "h_UF_m",
                    "net_sheet_mass", "ocean_area", "ocean_area_C",
                    "ocean_area_water",
                    "ice_mass_grounded_kg",
                    "ice_mass_floating_kg", "abs_dipole", "load_moment",
                    "dipole_rel", "newton", "outer")
            arrays = {key: np.array([row[key] for row in rows], dtype=float)
                      for key in keys}
            arrays["step"] = np.array([row["step"] for row in rows], dtype=int)
            np.savez(path, **arrays)
        return path

    def report_epoch(chk, t_epoch_kyr, step, dt_yr):
        """Profiles, time series and a checkpoint state at one epoch."""
        write_profiles(
            os.path.join(args.output,
                         f"martinec-{label}-profiles_{t_epoch_kyr:g}kyr.npz"),
            solver, layout, samplers, profiles, pieces["g_surface"],
            t_epoch_kyr)
        write_timeseries()
        save_state(chk, solver, layout, epoch_index[0], t_epoch_kyr, step,
                   dt_yr)
        say(f"      checkpoint index {epoch_index[0]}, t = "
            f"{t_epoch_kyr:g} kyr, step {step}")
        epoch_index[0] += 1

    with CheckpointFile(output_path, "w") as chk:
        chk.save_mesh(mantle)
        chk.save_mesh(parent)

        step = start_step
        stop = False

        if solve_elastic:
            # The load is a step held from t = 0, so the t = 0 state is the
            # instantaneous elastic response: the limit dt -> 0 of one step.
            say(f"\n  t = 0: the elastic response, one step of "
                f"{DT_ELASTIC:g} Maxwell times")
            dt.assign(DT_ELASTIC)
            t_kyr.assign(0.0)
            tic = time.time()
            solver.solve()
            record(0.0, DT_ELASTIC * T_BAR_YR, step, time.time() - tic)
            report_epoch(chk, 0.0, step, DT_ELASTIC * T_BAR_YR)
            # The march starts from rest. With the load held from t = 0 the
            # first marched step reproduces the elastic response by itself, so
            # starting from the elastic state would count it twice.
            # `solution_old` carries the history term of backward Euler and
            # must be reset with `solution`.
            z.assign(0.0)
            solver.solution_old.assign(0.0)
            # The row of the elastic solve stays in the time series: it is the
            # t = 0 entry of `h_UF` and of the ice masses, which the benchmark
            # compares from t = 0. It is not a step of the ladder, so it is
            # not in the step-time file the matched radopt runs read.

        previous_dt_yr = None
        for t0, t1, dt_yr, nsteps, is_epoch in segments:
            if t1 / 1000.0 <= start_t_kyr + 1e-12:
                # Already marched in the run this one restarts from.
                continue
            if dt_yr != previous_dt_yr:
                # The forms read the live `Constant`, and the preconditioners
                # rebuild the operators they cache when its value changes.
                dt.assign(dt_yr / T_BAR_YR)
                previous_dt_yr = dt_yr
            say(f"\n  {t0 / 1000:7.3f} -> {t1 / 1000:7.3f} kyr   dt "
                f"{dt_yr:7.2f} yr ({float(dt):.6g} Maxwell times)   "
                f"{nsteps:4d} steps")
            for k in range(nsteps):
                t_step_kyr = (t0 + (k + 1) * dt_yr) / 1000.0
                if t_step_kyr <= start_t_kyr + 1e-12:
                    continue
                # Backward Euler: the load is read at the end of the step.
                t_kyr.assign(t_step_kyr)
                step += 1
                tic = time.time()
                solver.solve()
                record(t_step_kyr, dt_yr, step, time.time() - tic)
                if step % CHECKPOINT_EVERY == 0 and not (
                        is_epoch and k == nsteps - 1):
                    report_epoch(chk, t_step_kyr, step, dt_yr)
                if args.steps is not None and step - start_step >= args.steps:
                    say(f"\n  stopping after {args.steps} steps (--steps)")
                    stop = True
                    break
            if is_epoch and not stop:
                report_epoch(chk, t1 / 1000.0, step, dt_yr)
            if stop:
                if rows:
                    report_epoch(chk, rows[-1]["t_kyr"], step, dt_yr)
                break

    timeseries_path = write_timeseries()
    say("\n" + "=" * 78)
    say(f"SUMMARY case {args.case}, label {label}")
    say("=" * 78)
    if rows:
        say(f"  {'t (kyr)':>10}{'h_UF (m)':>14}{'ocean area':>14}"
            f"{'grounded (kg)':>18}{'wall (s)':>10}")
        for row in rows[::max(1, len(rows) // 20)]:
            say(f"  {row['t_kyr']:>10g}{row['h_UF_m']:>14.6f}"
                f"{row['ocean_area']:>14.9f}"
                f"{row['ice_mass_grounded_kg']:>18.6e}"
                f"{row['wall_s']:>10.1f}")
    say(f"\n  checkpoint:  {output_path}")
    say(f"  time series: {timeseries_path}")
    say(f"RESULT case={args.case} label={label} steps={len(rows)} "
        f"epochs={','.join(f'{e:g}' for e in epochs)} "
        f"checkpoint={output_path}")


if __name__ == "__main__":
    main()
