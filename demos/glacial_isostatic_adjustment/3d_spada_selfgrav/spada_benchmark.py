r"""The Spada et al. (2011) self-gravitating GIA benchmark in 3-D.

THE BENCHMARK
    Spada, G., et al. (2011), A benchmark study for glacial isostatic
    adjustment codes, Geophys. J. Int. 185(1), 106-132.

    The benchmark compares GIA codes against the normal-mode code TABOO on a
    spherically layered, self-gravitating Maxwell Earth under a surface ice
    load. This driver solves the full problem on a 3-D tetrahedral mesh with
    G-ADOPT: viscoelastic mechanics, the perturbed gravitational potential,
    and (in one case) the rotational feedback, all in one monolithic system.

THE EARTH MODEL: M3-L70-V01
    Four mantle layers over an inviscid, homogeneous core (radius 3480 km,
    density 10750 kg m^-3):

        layer              radius (km)    rho (kg m^-3)  mu (Pa)       eta (Pa s)
        lithosphere        6301 - 6371    3037           0.50605e11    elastic
        upper mantle 1     5951 - 6301    3438           0.70363e11    1e21
        upper mantle 2     5701 - 5951    3871           1.05490e11    1e21
        lower mantle       3480 - 5701    4978           2.28340e11    2e21

    The lithosphere viscosity is 1e19 in viscosity units (1e40 Pa s), which
    makes its Maxwell time far longer than the run. The core is eliminated
    onto the core-mantle boundary by `gadopt.FluidCore`: a buoyancy spring, a
    mass sheet in the Poisson equation, and one pressure multiplier that
    keeps the core volume fixed. The reference gravity g_0(r) is the closed
    form of the layered density (`reference_state.gravity_exact_ufl`).

    The reference model is incompressible. This driver is compressible, with
    a bulk modulus of `--bulk-shear-ratio` times the shear modulus (default
    100, Poisson ratio 0.495), because the displacement element is a
    penalty-type discretisation of the volumetric constraint. The residual
    compressibility is one source of the remaining difference to TABOO.

THE LOAD
    A parabolic ice cap, 1500 m thick at its centre, 10 degrees in half
    width, ice density 931 kg m^-3:

        sigma(gamma) = rho_ice h sqrt((cos gamma - cos alpha) / (1 - cos alpha))

    with gamma the angular distance from the cap centre. The load is applied
    as a Heaviside step at t = 0 and then held. It enters as the Legendre
    series sum_{n=2}^{nmax} sigma_n P_n(cos gamma), in two places that must
    agree: as a normal traction on the surface of the mechanics, and as a
    mass sheet in the Poisson equation. Degrees 0 and 1 are excluded, as in
    the benchmark, which removes the reference-frame question. The TABOO
    reference is synthesised at the same `--nmax`, so both sides see the same
    truncated load and the ringing of the truncated series cancels in the
    comparison.

THE TWO CASES
    --case cap
        The cap centred on the north pole, rotation off. At each epoch the
        surface radial displacement U, the horizontal displacement V and the
        geoid N are projected onto Legendre polynomials in cos(colatitude),
        and each degree n = 2..nmax is compared against TABOO. The table also
        gives U(0), N(0), U(180), N(180) and the maximum of V.

    --case polar-motion
        The same cap centred at colatitude 25 degrees, longitude 75 degrees,
        rotation on. At each epoch the polar motion m = (m_x, m_y), its
        magnitude |m| and its phase are compared against the TABOO series
        `pm_cap_*` in `reference.npz` (test T02-03 of the benchmark, Chandler
        wobble excluded). The reference phase is exactly
        lambda_c + 180 deg = -105 degrees at all epochs: it depends on the load
        geometry and on nothing else.

    The two cases need separate runs. Rotational feedback adds a degree-2,
    order-1 signal to U, V and N, so a rotating run does not reproduce the
    non-rotating cap reference, and an off-axis load has no zonal spectrum.

NON-DIMENSIONAL SCALES
    length       D = 2891 km, the mantle thickness Re - Rc
    density      rho_bar = 5511.68 kg m^-3, the mean density of the model
    gravity      g_bar = 9.81555 m s^-2
    stress       mu_bar = 1e11 Pa
    viscosity    eta_bar = 1e21 Pa s
    time         t_bar = eta_bar / mu_bar = 1e10 s = 316.8809 yr

    Two numbers couple the scaled equations. B_mu = rho_bar g_bar D / mu_bar
    = 1.564 is the ratio of buoyancy stress to elastic stress, and
    Lambda = 4 pi G rho_bar D / g_bar = 1.361 is the self-gravity number. The
    surface load density is scaled by rho_bar D.

DISCRETISATION
    Mesh      `b2_coarse_ar7.msh`: about 99 000 tetrahedra, 65 000 of them in
              the mantle, 500 km lateral spacing, one 70 km lithosphere layer,
              curved to P2 geometry. Every density interface is a mesh surface.
    Unknowns  CG3 displacement and DG2 internal variables on the mantle
              submesh; CG2 potential on the whole mesh; `Real` unknowns for
              the core pressure and, in the polar-motion case, the three
              rotation components.
    Rheology  One Maxwell element per cell, written as an internal variable
              (`gadopt.CompressibleInternalVariableApproximation`).
    Time      Backward Euler with a graded time step: 10 yr to 0.1 kyr, 50 yr
              to 1 kyr, 100 yr to 10 kyr and 500 yr to 20 kyr. The t = 0 state
              is the instantaneous elastic response, solved on its own with a
              time step of 1e-4 Maxwell times.
    Solver    The library defaults of `gadopt.SelfGravitatingGIASolver`: the
              internal variables in the mixed space ("full layout"), the
              low-rank representation of the exterior DtN condition, and the
              iterative preset `selfgrav_dtn_iterative_solver_parameters`, which
              puts `gadopt.CondensedBlockPC` on the mechanics-potential block.
              The outer FGMRES tolerance is 1e-6 and the block-0 tolerance
              is 1e-4.

THE EXTERIOR GRAVITY CONDITION
    The potential perturbation extends to infinity, and the mesh does not.
    The mesh stops at a sphere at 2 Re, and an inner sphere at 0.5 Rc bounds
    it from below. On each sphere, a Dirichlet-to-Neumann (DtN) map replaces
    the unmeshed region: for each spherical-harmonic degree l, the exterior
    solution decays as r^-(l+1) and the interior solution grows as r^l, so
    the normal derivative of each harmonic component is a known multiple of
    its value. `SphericalDtN(L)` applies that relation exactly for degrees
    0..L. Degrees above L see a Robin condition only. Both spheres are a
    factor of two away from the sources, so a degree-l field is already
    small there: a field from sources at Re falls by 2^-(l+1) between Re and
    2 Re. The truncation error therefore decreases quickly with L.

RUN
    Generate the mesh once (see `generate_selfgrav_sphere.py`), then:

        mpiexec -np 96 python3 spada_benchmark.py --case cap
        mpiexec -np 96 python3 spada_benchmark.py --case polar-motion

    Check an installation before a long run. This builds everything,
    assembles the residual once, and exits before any solve:

        python3 spada_benchmark.py --case cap --dry-run

    A smaller, faster, less accurate run for a first look:

        python3 spada_benchmark.py --case cap --displacement-degree 2 \
            --internal-variable-degree 1 --epochs 0 1

OUTPUT
    <output>/spada-<label>.h5   A Firedrake `CheckpointFile` with both meshes
                                and, at each epoch in order, the displacement,
                                the potential and the internal variables.
    <output>/spada-<label>-*.pvd
                                VTK files for Paraview, with `--vtk` only. The
                                displacement and the potential live on
                                different meshes, so they go to two files.
    stdout                      The per-epoch comparison with the reference
                                and a summary table at the end.

EXPECTED RESULTS
    Cap case, 96 ranks, to 20 kyr (model / TABOO). This run used an earlier
    solver configuration with the same physics and discretisation: the
    internal variables were eliminated pointwise (the condensed layout) and
    the exterior DtN condition used one Real multiplier per harmonic mode.
    No run from t = 0 with the current defaults (the full layout and the
    low-rank DtN representation) exists yet:

        t (kyr)   U(0)     N(0)     max V
        0         1.0251   0.9971   0.9641
        0.1       1.0192   0.9972   0.9690
        1         1.0032   0.9997   0.9803
        2         0.9992   1.0027   0.9820
        5         0.9992   1.0054   0.9862
        10        1.0011   1.0025   0.9900
        20        1.0017   0.9989   0.9916

    Per degree at 20 kyr, n = 2..10, the U ratios are 1.0004 to 1.0028, the
    V ratios 0.9888 to 0.9926 and the N ratios 0.9967 to 1.0030; the
    degree-10 geoid ratio is 1.0030. The model is compressible and the
    reference is incompressible, which is part of the remaining difference.

    Evidence that the defaults reproduce these numbers is restart evidence
    only: two 500 yr steps from the 20 kyr state of that run. The full layout
    differed from the condensed layout by 5e-7 relative in the displacement
    norm, and the low-rank representation agreed with the multiplier
    representation to 4e-11. Those restarts began from a history field that
    was transferred between layouts, so they do not replace a run from t = 0.
    For a first run with the defaults, run `--epochs 0 0.1` and compare with
    the 0 and 0.1 kyr rows above.

    Polar-motion case: the phase must be -105 degrees at every epoch. No
    validated time series of |m| from this solver exists yet.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gadopt  # noqa: E402,F401  (import gadopt before firedrake)
import numpy as np  # noqa: E402
from firedrake import (COMM_WORLD, CheckpointFile, Constant,  # noqa: E402
                       Function, FunctionSpace, Mesh, SpatialCoordinate,
                       Submesh, VTKFile, as_vector, assemble, avg,
                       conditional, dot, ds, sqrt)
from gadopt import (CompressibleInternalVariableApproximation,  # noqa: E402
                    SphericalDtN)
from gadopt.gia_gravity import (FluidCore, SelfGravitatingGIASolver,  # noqa: E402
                                rigid_rotation_nullspace,
                                selfgrav_dtn_iterative_solver_parameters,
                                self_gravitating_gia_space)

import generate_selfgrav_sphere as gen  # noqa: E402
import reference_state as refstate  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402

# --------------------------------------------------------------------------
# Scales and material
# --------------------------------------------------------------------------

#: The length scale D = Re - Rc, in m.
D_M = refstate.D_SCALE
#: The stress scale, in Pa.
MU_BAR = 1.0e11
#: The time scale eta_bar / mu_bar = 1e21 Pa s / 1e11 Pa = 1e10 s, in years.
T_BAR_YR = 316.8809
#: Buoyancy stress over elastic stress, rho_bar g_bar D / mu_bar (1.564037).
B_MU = refstate.RHO_BAR * refstate.G_BAR * D_M / MU_BAR
#: The self-gravity number 4 pi G rho_bar D / g_bar (1.361324).
LAMBDA = refstate.LAMBDA

#: The mantle layers of M3-L70-V01, outermost first:
#: (r_outer, r_inner, rho / rho_bar, mu / mu_bar, eta / eta_bar). Radii are
#: non-dimensional. The lithosphere viscosity 1e19 gives a Maxwell time of
#: about 2e19 t_bar, so the lithosphere is elastic over the whole run.
MANTLE_LAYERS = [
    (gen.RE, 2.179523, 3037.0 / refstate.RHO_BAR, 0.50605, 1.0e19),
    (2.179523, 2.058457, 3438.0 / refstate.RHO_BAR, 0.70363, 1.0),
    (2.058457, 1.971982, 3871.0 / refstate.RHO_BAR, 1.05490, 1.0),
    (1.971982, gen.RC, 4978.0 / refstate.RHO_BAR, 2.28340, 2.0),
]
#: The core density, non-dimensional (1.950402).
RHO_CORE = 10750.0 / refstate.RHO_BAR

#: The time step of the instantaneous elastic response at t = 0, in Maxwell
#: times (about 0.03 yr). The Maxwell times of the mantle are 0.3 kyr and
#: longer, so this step gives no measurable relaxation.
DT_ELASTIC = 1.0e-4

#: The graded time-step ladder, as (t_end_yr, dt_yr) segments. The time step
#: that a backward-Euler step can take for a given accuracy grows with t,
#: because at time t the relaxation modes that still change the answer have
#: Maxwell times of order t. The rule dt <= 2 e t gives a relative
#: time-stepping error of about e, so this ladder gives 2.5 percent at 1 and
#: 2 kyr, 1 percent at 5 kyr, 0.5 percent at 10 kyr and 1.25 percent at
#: 20 kyr. The last segment's 500 yr is 1.58 Maxwell times.
DT_LADDER_YR = ((100.0, 10.0), (1000.0, 50.0), (10000.0, 100.0),
                (20000.0, 500.0))

#: The load centre of each case, as (colatitude, longitude) in degrees.
LOAD_CENTRE_DEG = {"cap": (0.0, 0.0), "polar-motion": (25.0, 75.0)}
#: The reference phase of the polar motion, lambda_c + 180 deg, wrapped to
#: (-180, 180]. It is exact and does not depend on time.
REFERENCE_PHASE_DEG = LOAD_CENTRE_DEG["polar-motion"][1] + 180.0 - 360.0


def say(msg):
    """Print on rank 0 only; every quantity printed is collective."""
    if COMM_WORLD.rank == 0:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Geometry and material
# --------------------------------------------------------------------------

def build_meshes(path):
    """Read the parent sphere, cut the mantle submesh, and curve both.

    `Submesh` does not inherit the parent's P2 coordinates, so the submesh is
    cut from the curved parent and then curved again with the same radial
    map. The two coordinate fields then agree on every shared facet. The
    meshes get distinct names so that a `CheckpointFile` can hold both.

    Args:
      path: the gmsh file written by `generate_selfgrav_sphere.py`.

    Returns:
      `(parent, mantle)`, both with P2 coordinates.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--mesh {path!r} does not exist. Generate it first with\n"
            "  python generate_selfgrav_sphere.py --configuration coarse "
            "--litho-layers 1 --min-cells 32 --output b2_coarse_ar7.msh")
    parent = gen.curve_mesh(Mesh(path), name="spada_parent")
    # The geometry is a sphere, so the "vertical" direction of G-ADOPT's
    # terms is radial and not the last Cartesian axis.
    parent.cartesian = False
    mantle = gen.curve_mesh(Submesh(parent, 3, gen.CELL_MANTLE),
                            name="spada_mantle")
    mantle.cartesian = False
    return parent, mantle


def layered(mesh, column, name):
    """A DG0 field that takes `MANTLE_LAYERS[*][column]` inside each layer.

    The innermost layer's value is the default, and each layer overrides it
    above its own inner radius, so the layers are applied from the inside
    out. The mesh conforms to every interface, so each cell lies inside one
    layer and the DG0 field is exact.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    expr = Constant(MANTLE_LAYERS[-1][column])
    for row in reversed(MANTLE_LAYERS):
        expr = conditional(r >= Constant(row[1]), Constant(row[column]), expr)
    return Function(FunctionSpace(mesh, "DG", 0), name=name).interpolate(expr)


def spada_approximation(mesh, bulk_shear_ratio):
    """The layered Maxwell rheology of M3-L70-V01 on the mantle mesh.

    One Maxwell element per cell. The bulk modulus is `bulk_shear_ratio`
    times the shear modulus in every layer, so the Poisson ratio is uniform.
    Gravity is the closed form g_0(r) of the layered density.

    Args:
      mesh: the mantle submesh.
      bulk_shear_ratio: K / mu, dimensionless.

    Returns:
      A `CompressibleInternalVariableApproximation`.
    """
    rho = layered(mesh, 2, "density")
    mu = layered(mesh, 3, "shear_modulus")
    eta = layered(mesh, 4, "viscosity")
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    return CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=rho, shear_modulus=[mu], viscosity=[eta],
        bulk_shear_ratio=bulk_shear_ratio, g=refstate.gravity_exact_ufl(r),
        B_mu=B_MU, self_gravity_number=LAMBDA)


# --------------------------------------------------------------------------
# Legendre series: the load and the projection of the answer
# --------------------------------------------------------------------------

def legendre_ufl(nmax, x):
    """P_0..P_nmax as UFL expressions in `x`, by Bonnet's recurrence."""
    P = [Constant(1.0), x]
    for n in range(1, nmax):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return P[:nmax + 1]


def dlegendre_ufl(nmax, x, P):
    """dP_n / dtheta for x = cos(theta), as UFL expressions.

    From (1 - x^2) P_n'(x) = n (P_{n-1} - x P_n) and dx/dtheta = -sin(theta):
    dP_n/dtheta = -n (P_{n-1} - x P_n) / sin(theta). sin(theta) is bounded
    below by 1e-7 so that the expression stays finite at the poles, where
    the tangential displacement of an axisymmetric field vanishes anyway.
    """
    s = sqrt(conditional(1.0 - x * x > 1e-14, 1.0 - x * x, 1e-14))
    return [Constant(0.0)] + [
        -n * (P[n - 1] - x * P[n]) / s for n in range(1, nmax + 1)]


def load_axis(case):
    """Unit vector from the centre of the Earth to the centre of the load."""
    colat, lon = np.deg2rad(LOAD_CENTRE_DEG[case])
    return np.array([np.sin(colat) * np.cos(lon),
                     np.sin(colat) * np.sin(lon), np.cos(colat)])


def load_field(mesh, nmax, sigma_n, axis):
    """The truncated cap series sum_{n=2}^{nmax} sigma_n P_n(cos gamma), in CG2.

    cos(gamma) = (X . axis) / |X| is the cosine of the angular distance from
    the load centre, so the field depends only on direction and is defined at
    every radius; the solver reads it on the Re facets. The series is
    interpolated into CG2, so that the load the solver sees is the load the
    mesh can carry.

    Args:
      mesh: the mesh to build the field on.
      nmax: the truncation degree.
      sigma_n: the non-dimensional coefficients, indexed from degree 0.
      axis: unit vector to the load centre.

    Returns:
      A CG2 `Function` of the non-dimensional surface density.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    cos_gamma = dot(X, Constant(axis)) / r
    P = legendre_ufl(nmax, cos_gamma)
    expr = Constant(0.0)
    for n in range(2, nmax + 1):
        expr = expr + Constant(sigma_n[n]) * P[n]
    return Function(FunctionSpace(mesh, "CG", 2),
                    name="sigma_load").interpolate(expr)


def project_surface(field_expr, mesh, nmax, measure, interior, basis="P",
                    quad_degree=None):
    """Legendre coefficients f_n = int f B_n dS / int B_n^2 dS on a sphere.

    B_n is P_n(cos theta) for `basis="P"` and dP_n/dtheta for `basis="dP"`,
    with theta the colatitude. The analytic denominator for P_n is
    4 pi R^2 / (2n + 1). The assembled one is used, so the discretisation
    error of the surface cancels between numerator and denominator.

    An explicit quadrature degree is needed. P_n(z / r) is a rational
    expression, and the automatic degree estimate for it runs to several
    hundred at n = 20, which exhausts memory. The integrand has a polynomial
    degree of about 2n after the geometry map.

    Args:
      field_expr: the UFL expression to project.
      mesh: the mesh of `measure`.
      nmax: the highest degree.
      measure: `ds(tag)` for an exterior facet set, `dS(tag)` for interior.
      interior: whether `measure` is an interior-facet measure, which needs
        `avg`.
      basis: `"P"` or `"dP"`.
      quad_degree: the quadrature degree.

    Returns:
      A numpy array of the coefficients for degrees 0..nmax.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    P = legendre_ufl(nmax, X[2] / r)
    if basis == "dP":
        P = dlegendre_ufl(nmax, X[2] / r, P)
    if quad_degree is not None:
        measure = measure(metadata={"quadrature_degree": quad_degree})
    out = np.zeros(nmax + 1)
    for n in range(nmax + 1):
        if interior:
            num = assemble(avg(field_expr * P[n]) * measure)
            den = assemble(avg(P[n] * P[n]) * measure)
        else:
            num = assemble(field_expr * P[n] * measure)
            den = assemble(P[n] * P[n] * measure)
        out[n] = num / den if abs(den) > 0.0 else 0.0
    return out


def series_from(coeffs, theta, nmin=2, kind="P"):
    """Sum_{n >= nmin} f_n P_n(cos theta), or the same sum over dP_n/dtheta.

    The sum starts at `nmin = 2` by default because the load and the TABOO
    reference have no degree 0 or 1. Any degree-0 or degree-1 content of the
    model would otherwise enter U(0), U(180) and max V with nothing on the
    reference side to match it.

    Args:
      coeffs: coefficients for degrees 0..nmax.
      theta: colatitudes in radians.
      nmin: the lowest degree in the sum.
      kind: `"P"` or `"dP"`.

    Returns:
      The series at `theta`.
    """
    c = np.array(coeffs, dtype=float, copy=True)
    c[:nmin] = 0.0
    nmax = len(c) - 1
    P, dP = taboo.legendre_and_dtheta(nmax, np.asarray(theta, dtype=float))
    return c @ (P if kind == "P" else dP)


def surface_spectra(solver, parent, mantle, nmax, quad_degree):
    """U_n, V_n and N_n in metres, from a solved state of the cap case.

    U is the radial displacement, V the colatitudinal (southward)
    displacement expanded in dP_n/dtheta, and N the geoid psi / g_0 on Re.
    The geoid is built from parent coordinates, because psi lives on the
    parent and a quotient with a g_0 built on the mantle mesh would mix two
    domains in one expression.

    Args:
      solver: the solved `SelfGravitatingGIASolver`.
      parent, mantle: the two meshes.
      nmax: the highest degree of the projection.
      quad_degree: the surface quadrature degree.

    Returns:
      `(U_n, V_n, N_n)`, numpy arrays for degrees 0..nmax, in metres.
    """
    ds_mantle = ds(gen.SURF_RE, domain=mantle)
    dS_parent = solver.form.dS(gen.SURF_RE)

    u = solver.displacement
    Xm = SpatialCoordinate(mantle)
    rm = sqrt(dot(Xm, Xm))
    U_n = project_surface(dot(u, Xm / rm), mantle, nmax, ds_mantle,
                          interior=False, quad_degree=quad_degree)

    # e_theta |X| sin(theta) = (z x, z y, -(x^2 + y^2)): the unnormalised
    # colatitude direction, divided by r and by the cylindrical radius. The
    # cylindrical radius is bounded below to keep the expression finite on
    # the axis.
    e_theta = as_vector((Xm[2] * Xm[0], Xm[2] * Xm[1], -(Xm[0]**2 + Xm[1]**2)))
    rho_cyl = sqrt(Xm[0]**2 + Xm[1]**2)
    u_theta = dot(u, e_theta) / (rm * conditional(rho_cyl > 1e-12, rho_cyl,
                                                  Constant(1e-12)))
    V_n = project_surface(u_theta, mantle, nmax, ds_mantle, interior=False,
                          basis="dP", quad_degree=quad_degree)

    Xp = SpatialCoordinate(parent)
    geoid = solver.potential / refstate.gravity_exact_ufl(sqrt(dot(Xp, Xp)))
    N_n = project_surface(geoid, parent, nmax, dS_parent, interior=True,
                          quad_degree=quad_degree)

    # Non-dimensional lengths to metres.
    return U_n * D_M, V_n * D_M, N_n * D_M


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------

def build_solver(parent, mantle, args, dt):
    """The coupled self-gravitating solver of one benchmark case.

    Args:
      parent, mantle: the two meshes.
      args: the parsed command line.
      dt: the time step as a `Constant`, in Maxwell times. The driver assigns
        new values to it between segments; the forms read the live value.

    Returns:
      `(solver, z, layout, sigma_parent)`: the solver, its mixed solution,
      the `GIASpaceLayout`, and the load on the parent mesh.
    """
    rotation = args.case == "polar-motion"
    axis = load_axis(args.case)
    # Non-dimensional load coefficients, sigma_n / (rho_bar D).
    sigma_n = taboo.cap_load(args.nmax) / (refstate.RHO_BAR * D_M)
    sigma_parent = load_field(parent, args.nmax, sigma_n, axis)
    sigma_mantle = load_field(mantle, args.nmax, sigma_n, axis)

    # The Poisson equation: DtN maps on the two truncation spheres, and the
    # ice as a mass sheet on the interior facets of Re. Without the sheet, U
    # is unchanged and N changes sign.
    gravity_bcs = {
        gen.SURF_OUTER: {"dtn": SphericalDtN(L=args.dtn_degree)},
        gen.SURF_INNER: {"dtn": SphericalDtN(L=args.dtn_degree)},
        gen.SURF_RE: {"interior_sigma": sigma_parent},
    }

    # The library defaults: internal variables in the mixed space, and the
    # DtN representation that the library chooses for that layout.
    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        fluid_core=True, self_gravity_number=LAMBDA,
        displacement_degree=args.displacement_degree,
        internal_variable_degree=args.internal_variable_degree)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    approximation = spada_approximation(mantle, args.bulk_shear_ratio)

    # The ice load as a normal traction on the surface, in stress units
    # mu_bar: B_mu sigma_hat = rho_bar g_bar D sigma_hat / mu_bar.
    bcs = {gen.SURF_RE: {"normal_stress": B_MU * sigma_mantle}}

    core = FluidCore(boundary=gen.SURF_RC, rho_core=RHO_CORE,
                     g=refstate.gravity_exact_ufl(Constant(gen.RC)))

    # A rigid rotation of the mantle is in the kernel of the whole coupled
    # operator: the core is fluid, the surface carries a traction, and the
    # rotation is strain-free, divergence-free and invisible to the inertia
    # rows. The discrete operator annihilates it only to about 2e-6, so it is
    # declared and projected out after each solve.
    nullspace = rigid_rotation_nullspace(Z, layout)

    # The iterative preset with the tolerances of the validated run. The
    # rheology is Newtonian, so the residual is linear in the unknowns and
    # one linear solve per step (`ksponly`) is exact to the outer tolerance.
    solver_parameters = selfgrav_dtn_iterative_solver_parameters(
        condensed=layout.condensed, outer_rtol=args.outer_rtol,
        block0_rtol=args.block0_rtol, snes_type="ksponly",
        dtn_representation=layout.dtn_representation)

    solver = SelfGravitatingGIASolver(
        z, approximation, layout=layout, dt=dt, bcs=bcs, fluid_core=core,
        rotation_moments={"C": refstate.C_NONDIM,
                          "C_minus_A": refstate.C_MINUS_A_PRIMARY},
        Omega_sq=refstate.OMEGA_SQ,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=solver_parameters)
    return solver, z, layout, sigma_parent


def time_ladder(epochs_kyr, ladder):
    """Segments (t0_yr, t1_yr, dt_yr, nsteps, is_epoch) that cover the epochs.

    Each ladder segment is cut at every requested epoch inside it, and dt is
    adjusted so that an integer number of steps lands exactly on each cut.
    Every reported state is therefore a solved state, with no interpolation
    in time. Epoch 0 is excluded: the t = 0 state is the elastic response
    and is solved separately.

    Args:
      epochs_kyr: the output epochs.
      ladder: (t_end_yr, dt_yr) segments.

    Returns:
      The list of segments.
    """
    want = sorted(e * 1000.0 for e in epochs_kyr if e > 0)
    out, t0 = [], 0.0
    for t_end, dt in ladder:
        if t_end <= t0:
            continue
        marks = [w for w in want if t0 < w <= t_end]
        cuts = sorted(set(marks + [t_end]))
        for c in cuts:
            n = max(1, int(round((c - t0) / dt)))
            out.append((t0, c, (c - t0) / n, n, c in marks))
            t0 = c
    return out


def truncated_ladder(t_end_yr, dt_yr=None):
    """The graded ladder cut at the last epoch, or one uniform segment.

    `time_ladder` marches over the whole ladder it receives, so the ladder
    must stop at the last requested epoch, or the run continues to 20 kyr
    and writes nothing after that epoch.
    """
    if dt_yr is not None:
        return ((t_end_yr, dt_yr),)
    ladder = []
    for upto, dt in DT_LADDER_YR:
        if upto >= t_end_yr:
            ladder.append((t_end_yr, dt))
            break
        ladder.append((upto, dt))
    return tuple(ladder)


# --------------------------------------------------------------------------
# Comparison with the reference
# --------------------------------------------------------------------------

def compare_cap_epoch(t_kyr, U_n, V_n, N_n, ref, sigma_dim, nmax, theta_fine):
    """Print the cap-case comparison at one epoch and return its summary row.

    Args:
      t_kyr: the epoch.
      U_n, V_n, N_n: the model coefficients in metres, degrees 0..nmax.
      ref: the `TabooReference`.
      sigma_dim: the dimensional load coefficients, kg m^-2.
      nmax: the truncation degree.
      theta_fine: colatitudes for the spatial series, radians.

    Returns:
      A dictionary with U(0), N(0), max V and its colatitude, for the model
      and the reference.
    """
    U_ref, V_ref, N_ref = ref.synthesise(t_kyr, theta_fine, sigma_dim, nmax=nmax)

    # Degree 0 is the breathing mode. The load has no degree-0 content and the
    # fluid core keeps its volume, so |U_0| / |U_2| must be small.
    u0_over_u2 = abs(U_n[0]) / max(abs(U_n[2]), 1.0e-300)
    n0_over_n2 = abs(N_n[0]) / max(abs(N_n[2]), 1.0e-300)
    say(f"DEGREE_ZERO t_kyr={t_kyr:g} "
        f"U_0={U_n[0]:.16e} U0_over_U2={u0_over_u2:.16e} "
        f"N_0={N_n[0]:.16e} N0_over_N2={n0_over_n2:.16e}")

    Um = series_from(U_n, theta_fine, nmin=2, kind="P")
    Nm = series_from(N_n, theta_fine, nmin=2, kind="P")
    Vm = series_from(V_n, theta_fine, nmin=2, kind="dP")
    # The signed maximum of V, not the maximum of |V|: the reference peaks
    # positive, and a maximum of |V| would hide a sign error in the model.
    jm, jr = int(np.argmax(Vm)), int(np.argmax(V_ref))

    say(f"\n  t = {t_kyr:g} kyr")
    say(f"    {'quantity':<12}{'model':>13}{'TABOO':>13}{'ratio':>9}")
    for name, mod, rf in (("U(0)", Um[0], U_ref[0]),
                          ("N(0)", Nm[0], N_ref[0]),
                          ("U(180)", Um[-1], U_ref[-1]),
                          ("N(180)", Nm[-1], N_ref[-1])):
        r = mod / rf if abs(rf) > 1e-30 else float("nan")
        say(f"    {name:<12}{mod:>13.5f}{rf:>13.5f}{r:>9.4f}")
    say(f"    {'max V':<12}{Vm[jm]:>13.5f}{V_ref[jr]:>13.5f}"
        f"{Vm[jm] / V_ref[jr]:>9.4f}"
        f"   at {np.rad2deg(theta_fine[jm]):.2f} vs "
        f"{np.rad2deg(theta_fine[jr]):.2f} deg")

    # Per degree. This separates an error flat in n (compressibility, a
    # scale factor) from one that grows with n (mesh resolution) or one
    # concentrated at low degree. The spatial values above cannot: U(0) sums
    # P_n(1) = 1 with no cancellation, while U(180) sums (-1)^n with strong
    # cancellation. The reference coefficients are rebuilt as `synthesise`
    # builds them: (3 / rho_bar) sigma_n / (2n + 1) times the Love number.
    hb, lb, kb = ref.love_time(t_kyr, nmax)
    nn = np.arange(ref.nmin_available, nmax + 1)
    cc = 3.0 / taboo.RHO_BAR * sigma_dim[ref.nmin_available:nmax + 1] / (2 * nn + 1)
    say(f"    per-degree   {'n':>3} {'U_n model':>12} {'U_n TABOO':>12} "
        f"{'ratio':>8} {'V_n ratio':>10} {'N_n ratio':>10}")
    for j, n in enumerate(nn):
        um, ur = U_n[n], cc[j] * hb[j]
        vm, vr = V_n[n], cc[j] * lb[j]
        # `love_time` already includes the direct term 1 in kbar, so the geoid
        # coefficient is cc * kbar and not cc * (1 + kbar).
        nm_, nr = N_n[n], cc[j] * kb[j]
        say(f"    {'':<12} {n:>3} {um:>12.6f} {ur:>12.6f} "
            f"{um / ur if abs(ur) > 1e-30 else float('nan'):>8.4f} "
            f"{vm / vr if abs(vr) > 1e-30 else float('nan'):>10.4f} "
            f"{nm_ / nr if abs(nr) > 1e-30 else float('nan'):>10.4f}")

    return dict(t_kyr=t_kyr, U0=Um[0], U0_ref=U_ref[0], N0=Nm[0],
                N0_ref=N_ref[0], Vmax=Vm[jm], Vmax_ref=V_ref[jr],
                Vth=np.rad2deg(theta_fine[jm]),
                Vth_ref=np.rad2deg(theta_fine[jr]))


def polar_motion_deg(solver):
    """(m_x, m_y, |m|, phase), all in degrees, from the solved rotation scalars.

    m_1 and m_2 are the direction cosines of the displaced rotation axis
    along x and y. At the magnitude of 1e-4 radians they are angles in
    radians to far better than the comparison needs.
    """
    values = solver.rotation_values()
    mx = np.rad2deg(values["m1"])
    my = np.rad2deg(values["m2"])
    return (float(mx), float(my), float(np.hypot(mx, my)),
            float(np.degrees(np.arctan2(my, mx))))


def reference_polar_motion(path, t_kyr):
    """The T02-03 reference (m_x, m_y, |m|, phase) at `t_kyr`, in degrees.

    The series is tabulated at 35 epochs between 0 and 20 kyr, including all
    the default epochs. Other epochs are linearly interpolated.
    """
    d = np.load(path)
    t = d["pm_cap_t_kyr"]
    mx = float(np.interp(t_kyr, t, d["pm_cap_mx_deg"]))
    my = float(np.interp(t_kyr, t, d["pm_cap_my_deg"]))
    return mx, my, float(np.hypot(mx, my)), float(np.degrees(np.arctan2(my, mx)))


def compare_polar_motion_epoch(t_kyr, solver, npz_path):
    """Print the polar-motion comparison at one epoch and return its row."""
    mx, my, absm, phase = polar_motion_deg(solver)
    mxr, myr, absr, phr = reference_polar_motion(npz_path, t_kyr)
    say(f"\n  t = {t_kyr:g} kyr   polar motion (degrees)")
    say(f"    {'quantity':<12}{'model':>14}{'TABOO':>14}{'ratio':>9}")
    for name, mod, rf in (("m_x", mx, mxr), ("m_y", my, myr),
                          ("|m|", absm, absr)):
        say(f"    {name:<12}{mod:>14.7f}{rf:>14.7f}{mod / rf:>9.4f}")
    say(f"    {'phase':<12}{phase:>14.4f}{phr:>14.4f}"
        f"   difference {phase - REFERENCE_PHASE_DEG:+.4f} deg")
    return dict(t_kyr=t_kyr, mx=mx, my=my, absm=absm, phase=phase,
                mx_ref=mxr, my_ref=myr, absm_ref=absr)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def save_state(chk, z, layout, idx):
    """Write the displacement, the potential and the internal variables.

    The mixed function spans two meshes: the displacement and the internal
    variables live on the mantle submesh, the potential and every `Real`
    field on the parent. `CheckpointFile.save_function` accepts one mesh per
    call, so each sub-function is saved on its own. The `Real` fields (core
    pressure, rotation) are not state that a time step carries forward; the
    next solve recomputes them from the saved fields.
    """
    fields = [(layout.displacement, "displacement"),
              (layout.potential, "potential"),
              (layout.internal_variable_field, "internal_variables")]
    for index, name in fields:
        f = z.subfunctions[index]
        f.rename(name)
        chk.save_function(f, idx=idx)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    """The command line; every default is the validated cap-case run."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["cap", "polar-motion"], default="cap",
                   help="cap: the axisymmetric cap, U, V and N against "
                        "TABOO. polar-motion: the cap at colatitude 25, "
                        "longitude 75 with rotation, polar motion against "
                        "T02-03.")
    p.add_argument("--mesh", default=os.path.join(HERE, "b2_coarse_ar7.msh"),
                   help="the gmsh file from generate_selfgrav_sphere.py")
    p.add_argument("--nmax", type=int, default=10,
                   help="spherical-harmonic truncation degree of the load, "
                        "of the reference and of the projection")
    p.add_argument("--dtn-degree", type=int, default=5,
                   help="degree L of SphericalDtN on both truncation spheres")
    p.add_argument("--bulk-shear-ratio", type=float, default=100.0,
                   help="K / mu in every layer")
    p.add_argument("--displacement-degree", type=int, default=3,
                   help="polynomial degree of the CG displacement")
    p.add_argument("--internal-variable-degree", type=int, default=2,
                   help="polynomial degree of the DG internal variables")
    p.add_argument("--epochs", type=float, nargs="+",
                   default=[0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0],
                   help="output epochs in kyr; the run stops at the last one")
    p.add_argument("--dt-yr", type=float, default=None,
                   help="replace the graded time-step ladder with one uniform "
                        "step, in years")
    p.add_argument("--outer-rtol", type=float, default=1e-6,
                   help="relative tolerance of the outer FGMRES")
    p.add_argument("--block0-rtol", type=float, default=1e-4,
                   help="relative tolerance of the mechanics-potential block "
                        "solve inside the outer FGMRES")
    p.add_argument("--quad-degree", type=int, default=40,
                   help="quadrature degree of the Legendre projections")
    p.add_argument("--label", default=None,
                   help="output file label; default the case name")
    p.add_argument("--output", default=HERE,
                   help="directory for the checkpoint and the VTK files")
    p.add_argument("--vtk", action="store_true",
                   help="also write VTK files at each epoch")
    p.add_argument("--dry-run", action="store_true",
                   help="build the meshes, the load, the solver and the "
                        "reference, assemble the residual once, and exit "
                        "before any solve")
    return p.parse_args()


def main():
    """Run one benchmark case."""
    args = parse_args()
    label = args.label or args.case
    epochs = sorted(set(args.epochs))
    theta_fine = np.linspace(0.0, np.pi, 4001)
    npz_path = os.path.join(HERE, "reference.npz")

    say("=" * 78)
    say(f"Spada et al. (2011) benchmark, case '{args.case}'")
    say("=" * 78)
    colat, lon = LOAD_CENTRE_DEG[args.case]
    say(f"  mesh {args.mesh}")
    say(f"  load: parabolic ice cap, 1500 m, 10 deg half width, 931 kg/m^3, "
        f"Heaviside step, centre at colatitude {colat:g}, longitude {lon:g}")
    say(f"  rotation {'on' if args.case == 'polar-motion' else 'off'}   "
        f"load and projection degrees 2..{args.nmax}   "
        f"SphericalDtN(L={args.dtn_degree})   K/mu {args.bulk_shear_ratio:g}")
    say(f"  displacement CG{args.displacement_degree}   internal variables "
        f"DG{args.internal_variable_degree}   potential CG2")
    say(f"  tolerances: outer {args.outer_rtol:g}, block 0 "
        f"{args.block0_rtol:g}")
    say(f"  Lambda {LAMBDA:.6f}   B_mu {B_MU:.6f}   t_bar {T_BAR_YR} yr")
    if args.case == "polar-motion":
        say(f"  C {refstate.C_NONDIM:.6f}   C-A {refstate.C_MINUS_A_PRIMARY:.8f}"
            f"   Omega^2 {refstate.OMEGA_SQ:.7e}")

    tic = time.time()
    parent, mantle = build_meshes(args.mesh)
    say(f"  meshes: parent {parent.comm.allreduce(parent.cell_set.size)} "
        f"cells, mantle {mantle.comm.allreduce(mantle.cell_set.size)} cells "
        f"({time.time() - tic:.1f} s)")

    ref = taboo.TabooReference(npz_path)
    # Dimensional load coefficients in kg m^-2, for the reference.
    sigma_dim = taboo.cap_load(args.nmax)

    t_end_yr = max(epochs) * 1000.0
    ladder = truncated_ladder(t_end_yr, args.dt_yr)
    segments = time_ladder(epochs, ladder)
    say(f"  epochs (kyr) {epochs}")
    say(f"  time-step ladder (t_end_yr, dt_yr) {ladder}")
    say(f"  {len(segments)} segments, {sum(s[3] for s in segments)} steps")

    # One live time step for the whole run. Its first value is the elastic
    # step if t = 0 is requested, and the first ladder step otherwise.
    solve_elastic = any(abs(e) < 1e-12 for e in epochs)
    first_dt = DT_ELASTIC if solve_elastic or not segments \
        else segments[0][2] / T_BAR_YR
    dt = Constant(first_dt)

    tic = time.time()
    solver, z, layout, _ = build_solver(parent, mantle, args, dt)
    say(f"  solver built ({time.time() - tic:.1f} s): layout "
        f"{'condensed' if layout.condensed else 'full'}, DtN representation "
        f"{layout.dtn_representation}, "
        f"{len(layout.multipliers)} DtN multipliers, core pressure field "
        f"{layout.core_pressure}, rotation fields {layout.rotation}")
    say(f"  unknowns: {z.function_space().dim()}")

    if args.dry_run:
        tic = time.time()
        residual = assemble(solver.F)
        with residual.dat.vec_ro as vec:
            norm = vec.norm()
        say(f"  residual of the zero state assembled ({time.time() - tic:.1f} s):"
            f" l2 norm {norm:.6e}")
        if args.case == "polar-motion":
            say(f"  rotation values of the zero state: {solver.rotation_values()}")
        say(f"  reference: TABOO degrees {ref.nmin_available}.."
            f"{ref.nmax_available}; polar motion at t = 0: "
            f"{reference_polar_motion(npz_path, 0.0)}")
        say("DRY RUN complete: no solve was run.")
        return

    h5 = os.path.join(args.output, f"spada-{label}.h5")
    vtk_mechanics = vtk_potential = None
    if args.vtk:
        # One VTK file per mesh: `VTKFile.write` needs every function on one
        # mesh, and the displacement and the potential live on two.
        vtk_mechanics = VTKFile(os.path.join(args.output,
                                             f"spada-{label}-mechanics.pvd"))
        vtk_potential = VTKFile(os.path.join(args.output,
                                             f"spada-{label}-potential.pvd"))

    rows = []

    def report_epoch(chk, t_kyr):
        """Compare, checkpoint and optionally write VTK at one epoch."""
        if args.case == "cap":
            U_n, V_n, N_n = surface_spectra(solver, parent, mantle, args.nmax,
                                            args.quad_degree)
            row = compare_cap_epoch(t_kyr, U_n, V_n, N_n, ref, sigma_dim,
                                    args.nmax, theta_fine)
            ratio = abs(row["U0"] / row["U0_ref"])
            if not (0.1 < ratio < 10.0):
                raise RuntimeError(
                    f"U(0) ratio {ratio:.4g} at t = {t_kyr} kyr is outside "
                    "[0.1, 10]. The solved state is wrong; stopping.")
        else:
            row = compare_polar_motion_epoch(t_kyr, solver, npz_path)
        rows.append(row)
        save_state(chk, z, layout, len(rows) - 1)
        if vtk_mechanics is not None:
            vtk_mechanics.write(z.subfunctions[layout.displacement],
                                time=t_kyr)
            vtk_potential.write(z.subfunctions[layout.potential], time=t_kyr)
        say(f"      written: checkpoint index {len(rows) - 1}, "
            f"t = {t_kyr:g} kyr")

    with CheckpointFile(h5, "w") as chk:
        chk.save_mesh(mantle)
        chk.save_mesh(parent)

        if solve_elastic:
            # The load is a Heaviside step, so the t = 0 state is the
            # instantaneous elastic response: the limit dt -> 0 of one step.
            # The first marched step already includes some relaxation, so this
            # state is solved on its own.
            say(f"\n  t = 0: the elastic response, one step of "
                f"{DT_ELASTIC:g} Maxwell times")
            dt.assign(DT_ELASTIC)
            tic = time.time()
            solver.solve()
            say(f"      elastic solve {time.time() - tic:.1f} s")
            report_epoch(chk, 0.0)
            # The march starts from rest. With the load held from t = 0, the
            # first marched step reproduces the elastic response by itself;
            # starting from the elastic state would count it twice.
            # `solution_old` must be reset with `solution`: the solver copied
            # the elastic state into it after the solve, and the
            # backward-Euler history term reads the internal variables from
            # `solution_old`, not from `solution`.
            z.assign(0.0)
            solver.solution_old.assign(0.0)

        previous_dt_yr = None
        for t0, t1, dt_yr, nsteps, is_epoch in segments:
            if dt_yr != previous_dt_yr:
                # The forms read the live Constant, and the preconditioners
                # rebuild the operators they cache when its value changes.
                dt.assign(dt_yr / T_BAR_YR)
                previous_dt_yr = dt_yr
            say(f"\n  {t0 / 1000:7.3f} -> {t1 / 1000:7.3f} kyr   "
                f"dt {dt_yr:7.2f} yr ({float(dt):.6g} Maxwell times)   "
                f"{nsteps:4d} steps")
            for k in range(nsteps):
                tic = time.time()
                solver.solve()
                elapsed = time.time() - tic
                t_step_kyr = (t0 + (k + 1) * dt_yr) / 1000.0
                say(f"TIMESTEP t_kyr={t_step_kyr:.9g} dt_yr={dt_yr:.9g} "
                    f"segment_step={k + 1} segment_steps={nsteps} "
                    f"wall_s={elapsed:.6f}")
            if is_epoch:
                report_epoch(chk, t1 / 1000.0)

    say("\n" + "=" * 78)
    say("SUMMARY  model / TABOO")
    say("=" * 78)
    if args.case == "cap":
        say(f"  {'t (kyr)':>8}{'U(0)':>10}{'N(0)':>10}{'max V':>10}"
            f"{'V peak deg':>12}{'ref deg':>10}")
        for r in rows:
            say(f"  {r['t_kyr']:>8g}{r['U0'] / r['U0_ref']:>10.4f}"
                f"{r['N0'] / r['N0_ref']:>10.4f}"
                f"{r['Vmax'] / r['Vmax_ref']:>10.4f}"
                f"{r['Vth']:>12.2f}{r['Vth_ref']:>10.2f}")
    else:
        say(f"  {'t (kyr)':>8}{'|m| model':>12}{'|m| TABOO':>12}{'ratio':>8}"
            f"{'phase':>10}{'difference':>12}")
        for r in rows:
            say(f"  {r['t_kyr']:>8g}{r['absm']:>12.7f}{r['absm_ref']:>12.7f}"
                f"{r['absm'] / r['absm_ref']:>8.4f}{r['phase']:>10.4f}"
                f"{r['phase'] - REFERENCE_PHASE_DEG:>+12.4f}")
    say(f"\n  checkpoint: {h5}")
    say(f"RESULT case={args.case} label={label} completed_epochs="
        f"{','.join(f'{e:g}' for e in epochs)} checkpoint={h5}")


if __name__ == "__main__":
    main()
