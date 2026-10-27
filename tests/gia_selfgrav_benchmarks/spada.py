r"""The Spada et al. (2011) self-gravitating GIA benchmark in 3-D.

THE BENCHMARK
    Spada, G., et al. (2011), A benchmark study for glacial isostatic
    adjustment codes, Geophys. J. Int. 185(1), 106-132.

    The benchmark compares GIA codes against the normal-mode code TABOO on
    the spherically layered, self-gravitating Maxwell Earth M3-L70-V01 under
    a surface ice load. This driver solves the whole problem on a 3-D
    tetrahedral mesh: viscoelastic mechanics, the perturbed gravitational
    potential and, in one case, the rotational feedback, all in one
    monolithic system. The Earth model, the mesh, the time steps and the
    solver settings are in `selfgrav_common.py`.

THE LOAD
    --load_shape cap (default): a parabolic ice cap, 1500 m thick at its
    centre, 10 degrees in angular radius, ice density 931 kg m^-3:

        sigma(gamma) = rho_ice h sqrt((cos gamma - cos alpha) / (1 - cos alpha))

    --load_shape disc: a uniform 1000 m thick disc, also of radius 10 degrees
    and density 931 kg m^-3 (Spada et al., Table 4).

    Here gamma is the angular distance from the load centre. The load is
    switched on at t = 0 and then held. By default, --load_representation
    spatial constructs the load directly and subtracts only
    sigma_0 + sigma_1 cos(gamma), retaining all higher degrees before mesh
    interpolation. Degrees 0 and 1 are left out, as in the benchmark.
    Alternatively, --load_representation spectral uses the Legendre series
    sum_{n=2}^{load_nmax} sigma_n P_n(cos gamma). Both representations are
    interpolated into the same CG2 space and used consistently as a normal
    traction in the mechanics and a mass sheet in the Poisson equation.
    --load_nmax sets the spectral load cutoff (default 10); it is unused for
    the spatial load. --diagnostic_nmax independently sets the harmonic
    projection cutoff (default 10). The projected TABOO comparison uses the
    same diagnostic degrees and accounts for the chosen load cutoff. Direct
    profiles from --write_profiles retain all detail resolved by the mesh.
    --dtn_degree sets the spherical-harmonic cutoff on both gravity boundary
    spheres (default 5). Use 2 for the coarser boundary approximation while
    retaining the quadrupole needed for polar motion. It is independent of
    the load and diagnostic cutoffs.
    --outer_rtol and --block0_rtol set the coupled and inner block relative
    tolerances (defaults 1e-6 and 1e-4). A coarser solver check uses 1e-5 and
    1e-3, with the same cached Schur preconditioner.

THE TWO CASES
    --case cap
        The selected load on the north pole, rotation off. The case name is
        retained for compatibility, including when --load_shape disc is used.
        At each epoch the radial displacement U, the colatitudinal
        displacement V and the geoid N on
        Re are projected onto Legendre polynomials in cos(colatitude), and
        each degree n = 2..diagnostic_nmax is compared with TABOO. The summary
        also gives U and N at both poles and the maximum of V reconstructed
        from those diagnostic degrees, rather than the full spatial fields.

    --case polar-motion
        The selected load centred at colatitude 25, longitude 75 degrees, rotation
        on. At each epoch the polar motion m = (m_x, m_y), its size |m| and
        its phase are compared with the TABOO series of test T02-03 (Chandler
        wobble excluded). The reference phase is lambda_c + 180 deg = -105
        degrees at every epoch: it depends on the load position only.
        The disc reference scales the cap series by sigma_2(disc)/sigma_2(cap),
        the ratio of their geometrical excitation factors (paper eq. 31).

        The moment difference C - A is 2.6952e35 kg m^2, the value that goes
        with the secular Love number k_s = 0.96672389 of the reference (see
        `selfgrav_common.C_MINUS_A`). The reference itself uses 2.63e35 in its
        excitation, which is 2.4 percent smaller, so the polar motion of this
        model is below the reference. An earlier run on a 500 km mesh gave
        |m| / |m_ref| = 0.974 at t = 0 and 0.968 at 20 kyr; the 1000 km smoke
        mesh gives 0.973 at t = 0 and at 0.1 kyr. Of the 2.6 percent at
        t = 0, 0.15 percent is the model's own error: its elastic tidal Love
        number is 0.107 percent above the TABOO value.

    The two cases need separate runs. The rotational feedback adds a
    degree-2, order-1 signal to U, V and N, so a rotating run does not
    reproduce the non-rotating reference. The off-axis load is not zonal
    about the north pole, and the rotating response is not axisymmetric
    about the load centre either. --write_profiles samples either case.
    Rotating profiles default to north-south and east-west great circles
    through the load centre, with signed distance from that centre. Their
    plots show direct fields; the non-rotating TABOO spatial reference is
    only overlaid for the pole-centred meridian comparison.

DISCRETISATION
    Mesh      `selfgrav_common.MESHES["spada"]`, generated in the job: 250 km
              lateral cell size, two 35 km lithosphere layers, curved to P2.
    Unknowns  CG3 displacement and DG2 internal variables on the mantle
              submesh, CG2 potential on the whole mesh, `Real` unknowns for
              the core pressure and, in the polar-motion case, the three
              rotation components.
    Frame     --centre_of_mass adds three Real multipliers fixing the total
              mass dipole of mantle, fluid core and load to zero. Off by
              default; independent of the load's analytic degree-0/1 removal.
    Time      Backward Euler: 10 yr steps to 0.1 kyr, 50 yr to 1 kyr, 100 yr
              to 10 kyr and 500 yr to 20 kyr. The t = 0 state is the elastic
              response, solved on its own.
              --timestep_scale multiplies this ladder (coarse check: 2).
              --mesh_h_km sets the bulk mesh size (coarse check: 500);
              --litho_layers sets lithosphere subdivisions (coarse check: 1).
              These options are recorded and checked on restart.
    Elastic   K / mu = 1000 by default (`--bulk_shear_ratio`). TABOO is
              incompressible, which is one source of the remaining
              difference.

RUN
    mpiexec -np <N> python3 spada.py --case cap
    mpiexec -np <N> python3 spada.py --case cap --load_shape disc --output_path disc/
    mpiexec -np <N> python3 spada.py --case polar-motion

    `--smoke` runs the same code on a very coarse mesh to 0.1 kyr. It checks
    that the code runs; its numbers are not results.

    The run writes a checkpoint every `--checkpoint_every` steps (default
    50) into `checkpoint_<case>.h5`. To continue a run that stopped, run the
    same command on the same number of ranks with `--restart <file>`. The
    run continues after the last checkpoint in the file and appends its own
    checkpoints to the same file.

OUTPUT (in --output_path)
    params_<case>.log      One line per time step: the step, the time, the
                           step length, the wall time of the solve, the
                           iteration counts and one number of the state (the
                           rms radial displacement on Re for the cap, the
                           polar motion for the other case).
    summary_<case>.json    The comparison with TABOO at every epoch, the
                           mesh record (gmsh version, MD5, cell counts) and
                           the run settings. `test_benchmarks.py` reads it.
    checkpoint_<case>.h5   The checkpoints of the run (Firedrake
                           `CheckpointFile`), for `--restart`.
    spada_<case>.msh       The mesh of the run.
    spada_<case>_mechanics.pvd, spada_<case>_potential.pvd
                           Volume VTK files, with --write_output. The
                           displacement and the potential live on different
                           meshes, so they go to two files (nondimensional).
    spada_<case>_surface.h5
                           With --write_surface_output: surface-only HDF5
                           displacement, gravitational/rotational/total geoid
                           and adjacent-step rates, in m and m/yr. Geometry
                           is in metres, times in kyr. Uses the actual
                           curved boundary facets; no point sampling.
                           Independent of --write_output and --write_profiles.
                           Convert offline with convert_spada_surface.py.
                           --surface_output_format vtk writes PVD/VTU directly.
    profiles_<case>_*.npz  With --write_profiles: displacement/geoid and
                           adjacent-step rates (m/yr). Rates at t=0 are
                           undefined, saved as NaN with rate_valid=False.

    --load_centre COLAT LON overrides the load location independently of
    rotation: --case cap --load_centre 25 75 gives the nonrotating control
    for the default polar-motion load. Its harmonic diagnostics use angular
    distance and horizontal displacement away from the load centre.
    Off-axis cases default to the same load-centred great circles.
    Rotating summaries and step logs include m3 = delta(Omega)/Omega and
    lod_ms = -86400e3*m3, positive for a longer day. This uses the nominal
    86400 s day of the paper, not 2*pi/Omega.

RESULTS OF AN EARLIER RUN
    The cap case on the coarser 500 km mesh with one 70 km lithosphere
    layer, to 20 kyr, model / TABOO:

        t (kyr)   U(0)     N(0)     max V
        0         1.0251   0.9971   0.9641
        1         1.0032   0.9997   0.9803
        10        1.0011   1.0025   0.9900
        20        1.0017   0.9989   0.9916

    No run on the mesh of this driver exists yet.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import selfgrav_common as common
from spada_loads import (
    CAP_RADIUS_DEG,
    CAP_THICKNESS_M,
    DISC_THICKNESS_M,
    LOAD_SHAPES,
    RHO_ICE,
    load_coefficients,
    polar_motion_scale,
)
from spada_loads import cap_load as cap_load  # noqa: PLC0414
from spada_profile_geometry import profile_geometry, profile_settings

from gadopt import (
    Constant,
    Function,
    FunctionSpace,
    ParameterLog,
    SpatialCoordinate,
    SphericalDtN,
    VTKFile,
    assemble,
    avg,
    conditional,
    dot,
    ds,
    log,
    sqrt,
)
from gadopt.gia_gravity import (
    SelfGravitatingGIASolver,
    rigid_rotation_nullspace,
    self_gravitating_gia_space,
)
from gadopt.utility import vertical_component

# --------------------------------------------------------------------------
# Settings of the benchmark
# --------------------------------------------------------------------------

#: Default cutoffs for the spectral load and the independent diagnostics.
NMAX = 10
#: The quadrature degree of the Legendre projections. P_n(z / r) is a
#: rational expression, and the automatic degree estimate for it runs to
#: several hundred, which exhausts memory. The integrand has a polynomial
#: degree of about 2n after the geometry map.
QUAD_DEGREE = 40
#: The output epochs, kyr.
EPOCHS_KYR = (0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0)
#: The output epochs of a smoke run, kyr: the elastic state and 10 steps.
SMOKE_EPOCHS_KYR = (0.0, 0.1)
#: The load centre of each case, as (colatitude, longitude) in degrees.
LOAD_CENTRE_DEG = {"cap": (0.0, 0.0), "polar-motion": (25.0, 75.0)}
#: The reference phase of the polar motion, lambda_c + 180 deg, wrapped to
#: (-180, 180]. It is exact and does not depend on time.
REFERENCE_PHASE_DEG = LOAD_CENTRE_DEG["polar-motion"][1] + 180.0 - 360.0
#: The TABOO spectra and the polar-motion series of test T02-03.
REFERENCE_NPZ = os.path.join(HERE, "reference.npz")

# --------------------------------------------------------------------------
# The TABOO reference
# --------------------------------------------------------------------------


def legendre_and_dtheta(nmax, theta):
    """P_n(cos theta) and dP_n(cos theta)/dtheta for n = 0..nmax, in NumPy.

    Bonnet's recurrence for P_n, and (1 - x^2) P_n'(x) = n (P_{n-1} - x P_n)
    with dx/dtheta = -sin(theta) for the derivative. The derivative is zero
    at both poles, and the zeros left there by the guard are correct.

    Args:
      nmax: the highest degree.
      theta: colatitudes in radians, a 1-D array.

    Returns:
      `(P, dP)`, two arrays of shape `(nmax + 1, theta.size)`.
    """
    ct, st = np.cos(theta), np.sin(theta)
    P = np.zeros((nmax + 1, theta.size))
    P[0] = 1.0
    if nmax >= 1:
        P[1] = ct
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * ct * P[n] - n * P[n - 1]) / (n + 1)
    dP = np.zeros_like(P)
    ok = st > 1e-13
    for n in range(1, nmax + 1):
        dP[n, ok] = -n * (P[n - 1, ok] - ct[ok] * P[n, ok]) / st[ok]
    return P, dP


class TabooReference:
    """The TABOO spectral reference of the benchmark, from `reference.npz`.

    Equations (17) to (19) of the paper, for a load switched on at t = 0:

        hbar(t) = h_e - sum_j h_j (1 - exp(s_j t)) / s_j
        kbar(t) = 1 + k_e - sum_j k_j (1 - exp(s_j t)) / s_j
        {U, V, N}(theta, t) = (3 / rho_bar) sum_n {hbar, lbar, kbar}_n(t)
                              sigma_n / (2n + 1) {P_n, dP_n/dtheta, P_n}

    with s_j in kyr^-1 and t in kyr. The 1 in kbar is the direct term of the
    load's own potential. Without it N is about 60 percent low.
    """

    def __init__(self, npz_path=REFERENCE_NPZ):
        """Read the spectra.

        Args:
          npz_path: the reference file.
        """
        self.data = np.load(npz_path, allow_pickle=False)
        self.nmin = int(self.data["degrees"][0])
        self.nmax = int(self.data["degrees"][-1])

    def love_time(self, t_kyr, nmax):
        """hbar, lbar and kbar for degrees nmin..nmax at `t_kyr`.

        Args:
          t_kyr: the epoch, kyr.
          nmax: the highest degree.

        Returns:
          Three arrays, one value per degree.
        """
        if not self.nmin <= nmax <= self.nmax:
            raise ValueError(f"TABOO reference supports degrees "
                             f"{self.nmin}..{self.nmax}, requested {nmax}")
        rows = slice(0, nmax - self.nmin + 1)
        s = self.data["spectrum_s"][rows]
        shape = (1.0 - np.exp(s * t_kyr)) / s
        out = []
        for symbol in ("h", "l", "k"):
            direct = 1.0 if symbol == "k" else 0.0
            residues = self.data[f"{symbol}_residues"][rows]
            out.append(direct + self.data[f"{symbol}_elastic"][rows]
                       - (residues * shape).sum(axis=1))
        return tuple(out)

    def coefficients(self, t_kyr, sigma_n, nmax):
        """The reference U_n, V_n and N_n in metres, for degrees nmin..nmax.

        Args:
          t_kyr: the epoch, kyr.
          sigma_n: the load coefficients, kg m^-2, indexed from degree 0.
          nmax: the highest degree.

        Returns:
          Three arrays, one value per degree.
        """
        hbar, lbar, kbar = self.love_time(t_kyr, nmax)
        n = np.arange(self.nmin, nmax + 1)
        c = 3.0 / common.RHO_BAR * sigma_n[self.nmin:nmax + 1] / (2 * n + 1)
        return c * hbar, c * lbar, c * kbar

    def polar_motion(self, t_kyr, load_shape="cap", load_centre=None):
        """The T02-03 polar motion (m_x, m_y) at `t_kyr`, in degrees.

        The series is tabulated at 35 epochs between 0 and 20 kyr, including
        every output epoch of this driver. Other epochs are interpolated
        linearly. Disc results scale the cap series by sigma_2, as in eq. (31).
        A custom centre changes sin(2*colatitude) and the longitude phase.
        """
        t = self.data["pm_cap_t_kyr"]
        scale = polar_motion_scale(load_shape)
        m = complex(np.interp(t_kyr, t, self.data["pm_cap_mx_deg"]),
                    np.interp(t_kyr, t, self.data["pm_cap_my_deg"]))
        if load_centre is not None:
            colat, lon = np.deg2rad(load_centre)
            tc, pc = np.deg2rad(LOAD_CENTRE_DEG["polar-motion"])
            scale *= np.sin(2 * colat) / np.sin(2 * tc)
            m *= np.exp(1j * (lon - pc))
        return float(scale * m.real), float(scale * m.imag)

    def length_of_day_eq33(self, t_kyr, load_shape="cap", load_centre=None):
        """Published Eq. (33) LOD reference in ms, preserving its sign.

        This is reconstructed from independent TABOO loading k_2(t).
        Keep it distinct from the physical -LOD*m3 solver diagnostic: for
        the degree-0-free load the inertia integral has the opposite sign
        to the geometrical factor printed in Eq. (33).
        """
        centre = LOAD_CENTRE_DEG["polar-motion"] if load_centre is None else load_centre
        cosine = np.cos(np.deg2rad(centre[0]))
        sigma2 = load_coefficients(2, load_shape)[2]
        polar_moment = common.C_NONDIM * common.RHO_BAR * common.D_SCALE**5
        factor = (4*np.pi * common.A_EARTH**4 * sigma2 / (15*polar_moment)
                  * (3*cosine**2 - 1))
        kbar = self.love_time(t_kyr, 2)[2][0]
        return float(86400e3 * factor * kbar)


# --------------------------------------------------------------------------
# Legendre series on the mesh: the load and the projection of the answer
# --------------------------------------------------------------------------


def legendre_ufl(nmax, x):
    """P_0..P_nmax as UFL expressions in `x`, by Bonnet's recurrence."""
    P = [Constant(1.0), x]
    for n in range(1, nmax):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return P[:nmax + 1]


def dlegendre_ufl(nmax, x, P):
    """dP_n / dtheta for x = cos(theta), as UFL expressions.

    dP_n/dtheta = -n (P_{n-1} - x P_n) / sin(theta). sin(theta) is bounded
    below by 1e-7 so that the expression stays finite at the poles, where the
    tangential displacement of an axisymmetric field is zero anyway.
    """
    s = sqrt(conditional(1.0 - x * x > 1e-14, 1.0 - x * x, 1e-14))
    return [Constant(0.0)] + [
        -n * (P[n - 1] - x * P[n]) / s for n in range(1, nmax + 1)]


def load_axis(case, load_centre=None):
    """The unit vector from the centre of the Earth to the load centre."""
    colat, lon = np.deg2rad(LOAD_CENTRE_DEG[case] if load_centre is None else load_centre)
    return np.array([np.sin(colat) * np.cos(lon),
                     np.sin(colat) * np.sin(lon), np.cos(colat)])


def load_field(mesh, sigma_n, axis, *, representation="spatial", load_shape="cap"):
    """The degree-0/1-free load, built spectrally or spatially, in CG2.

    cos(gamma) = (X . axis) / |X| is the cosine of the angular distance from
    the load centre, so the field depends on direction only and is defined at
    every radius. The solver reads it on the Re facets. The series is
    interpolated into CG2, the space of the potential, so that the load the
    solver sees is a load the mesh can carry.

    Args:
      mesh: the mesh to build the field on.
      sigma_n: the non-dimensional coefficients, indexed from degree 0.
        For a spatial load, only coefficients 0 and 1 are needed.
      axis: the unit vector to the load centre.
      representation: "spectral" sums degrees 2..len(sigma_n)-1;
        "spatial" evaluates the load and subtracts degrees 0 and 1.
      load_shape: "cap" or "disc", with the Table 4 thickness and radius.

    Returns:
      A CG2 `Function` of the non-dimensional surface density.
    """
    X = SpatialCoordinate(mesh)
    cos_gamma = dot(X, Constant(axis)) / sqrt(dot(X, X))
    if representation == "spectral":
        nmax = len(sigma_n) - 1
        P = legendre_ufl(nmax, cos_gamma)
        expr = Constant(0.0)
        for n in range(2, nmax + 1):
            expr = expr + Constant(sigma_n[n]) * P[n]
    elif representation == "spatial":
        cos_alpha = np.cos(np.deg2rad(CAP_RADIUS_DEG))
        scale = common.RHO_BAR * common.D_SCALE
        if load_shape == "cap":
            # Clamp before sqrt, including for points outside the cap.
            cap_fraction = conditional(cos_gamma > cos_alpha,
                                       (cos_gamma - cos_alpha) / (1 - cos_alpha), 0.0)
            surface_load = RHO_ICE * CAP_THICKNESS_M / scale * sqrt(cap_fraction)
        elif load_shape == "disc":
            surface_load = conditional(cos_gamma >= cos_alpha,
                                       RHO_ICE * DISC_THICKNESS_M / scale, 0.0)
        else:
            raise ValueError(f"Unknown load shape {load_shape!r}")
        expr = surface_load - Constant(sigma_n[0]) - Constant(sigma_n[1]) * cos_gamma
    else:
        raise ValueError(f"Unknown load representation {representation!r}")
    return Function(FunctionSpace(mesh, "CG", 2),
                    name="sigma_load").interpolate(expr)


def project_surface(field_expr, mesh, measure, interior, basis="P", *, nmax=NMAX,
                    axis=None):
    """Legendre coefficients f_n = int f B_n dS / int B_n^2 dS on Re.

    B_n is P_n(cos theta) for `basis="P"` and dP_n/dtheta for `basis="dP"`,
    with theta measured from `axis` (north pole by default).
    The denominator is assembled and not taken
    from the closed form 4 pi R^2 / (2n + 1), so that the discretisation
    error of the surface cancels between numerator and denominator.

    Args:
      field_expr: the UFL expression to project.
      mesh: the mesh of `measure`.
      measure: `ds(tag)` for exterior facets, `dS(tag)` for interior ones.
      interior: whether `measure` is an interior-facet measure, which needs
        `avg`.
      basis: `"P"` or `"dP"`.
      nmax: the diagnostic cutoff, independent of the load cutoff.

    Returns:
      A NumPy array of the coefficients for degrees 0..nmax.
    """
    X = SpatialCoordinate(mesh)
    axis = np.array([0., 0., 1.]) if axis is None else axis
    x = dot(X, Constant(axis)) / sqrt(dot(X, X))
    P = legendre_ufl(nmax, x)
    if basis == "dP":
        P = dlegendre_ufl(nmax, x, P)
    measure = measure(metadata={"quadrature_degree": QUAD_DEGREE})
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


def surface_spectra(solver, parent, mantle, nmax=NMAX, *, load_centre=None):
    """U_n, V_n and N_n in metres from a solved state of the cap case.

    U is radial, V points away from the load centre and is expanded in
    dP_n/dgamma, and N is the gravitational geoid psi / g_0 on Re.
    The geoid is built from parent coordinates, because psi lives on the
    parent, and a quotient with a g_0 built on the mantle would mix two
    meshes in one expression.

    Args:
      solver: the solved `SelfGravitatingGIASolver`.
      parent, mantle: the two meshes.
      nmax: the diagnostic cutoff.

    Returns:
      `(U_n, V_n, N_n)`, NumPy arrays for degrees 0..nmax, in metres.
    """
    ds_mantle = ds(common.SURF_RE, domain=mantle)
    axis = load_axis("cap", load_centre)
    u = solver.displacement
    # The mantle mesh is marked non-Cartesian, so `vertical_component` is the
    # radial component u . X / |X|.
    U_n = project_surface(vertical_component(u), mantle, ds_mantle,
                          interior=False, nmax=nmax, axis=axis)

    # Tangential direction away from the load centre, with a bounded
    # denominator at its axis and antipode. At the north pole this is south.
    Xm = SpatialCoordinate(mantle)
    radial = Xm / sqrt(dot(Xm, Xm))
    cos_gamma = dot(radial, Constant(axis))
    sin_gamma = sqrt(conditional(1 - cos_gamma**2 > 1e-24,
                                 1 - cos_gamma**2, 1e-24))
    u_theta = dot(u, cos_gamma * radial - Constant(axis)) / sin_gamma
    V_n = project_surface(u_theta, mantle, ds_mantle, interior=False,
                          basis="dP", nmax=nmax, axis=axis)

    Xp = SpatialCoordinate(parent)
    geoid = solver.potential / common.gravity_exact_ufl(sqrt(dot(Xp, Xp)))
    N_n = project_surface(geoid, parent, solver.form.dS(common.SURF_RE),
                          interior=True, nmax=nmax, axis=axis)
    # Non-dimensional lengths to metres.
    return U_n * common.D_SCALE, V_n * common.D_SCALE, N_n * common.D_SCALE


def series(coeffs, theta, kind="P"):
    """sum_{n >= 2} f_n P_n(cos theta), or the same sum over dP_n/dtheta.

    The sum starts at degree 2, because the load and the reference have no
    degree 0 or 1. Any degree-0 or degree-1 content of the model would
    otherwise enter the pole values with nothing on the reference side.

    Args:
      coeffs: coefficients indexed from degree 0.
      theta: colatitudes, radians.
      kind: `"P"` or `"dP"`.

    Returns:
      The series at `theta`.
    """
    c = np.array(coeffs, dtype=float, copy=True)
    c[:2] = 0.0
    P, dP = legendre_and_dtheta(len(c) - 1, np.asarray(theta, dtype=float))
    return c @ (P if kind == "P" else dP)


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


def build_solver(case, parent, mantle, dt, bulk_shear_ratio, *,
                 load_nmax=NMAX, load_representation="spatial", load_shape="cap",
                 dtn_degree=common.DTN_DEGREE, outer_rtol=common.OUTER_RTOL,
                 block0_rtol=common.BLOCK0_RTOL, load_centre=None,
                 centre_of_mass=False):
    """The coupled self-gravitating solver of one Spada case.

    Args:
      case: `"cap"` or `"polar-motion"`.
      parent, mantle: the two meshes.
      dt: the time step `Constant`, in Maxwell times.
      bulk_shear_ratio: K / mu.
      load_nmax: the spectral load cutoff; unused for a spatial load.
      load_representation: "spectral" or "spatial".
      load_shape: "cap" or "disc".
      dtn_degree: harmonic cutoff on both gravity boundary spheres.
      outer_rtol, block0_rtol: coupled and inner block relative tolerances.
      centre_of_mass: fix the total mantle, fluid-core and load mass dipole
        to zero using the library's frame multipliers.

    Returns:
      `(solver, layout)`.
    """
    rotation = case == "polar-motion"
    axis = load_axis(case, load_centre)
    # The load coefficients, scaled by rho_bar D.
    sigma_n = load_coefficients(
        load_nmax if load_representation == "spectral" else 1, load_shape)
    sigma_n /= common.RHO_BAR * common.D_SCALE
    sigma_parent = load_field(parent, sigma_n, axis, representation=load_representation,
                              load_shape=load_shape)
    sigma_mantle = load_field(mantle, sigma_n, axis, representation=load_representation,
                              load_shape=load_shape)

    # The Poisson equation: DtN maps on the two truncation spheres, and the
    # ice as a mass sheet on the interior facets of Re. Without the sheet, U
    # is unchanged and N changes sign.
    gravity_bcs = {
        common.SURF_OUTER: {"dtn": SphericalDtN(L=dtn_degree)},
        common.SURF_INNER: {"dtn": SphericalDtN(L=dtn_degree)},
        common.SURF_RE: {"interior_sigma": sigma_parent},
    }
    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        fluid_core=True, centre_of_mass=centre_of_mass,
        self_gravity_number=common.LAMBDA,
        displacement_degree=common.DISPLACEMENT_DEGREE,
        internal_variable_degree=common.INTERNAL_VARIABLE_DEGREE,
        potential_degree=common.POTENTIAL_DEGREE,
        dtn_representation=common.DTN_REPRESENTATION)
    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    # The ice load as a normal traction on the surface, in units of mu_bar:
    # B_mu sigma_hat = rho_bar g_bar D sigma_hat / mu_bar.
    bcs = {common.SURF_RE: {"normal_stress": common.B_MU * sigma_mantle}}

    # A rigid rotation of the mantle is in the kernel of the coupled
    # operator: the core is fluid, the surface carries a traction, and the
    # rotation has no strain. The discrete operator removes it only to about
    # 2e-6, so it is declared and projected out after each solve.
    nullspace = rigid_rotation_nullspace(Z, layout)

    # The rheology is Newtonian and the load is fixed, so the residual is
    # linear in the unknowns and one linear solve per step is exact to the
    # outer tolerance.
    # Both Spada layouts have a narrow Real block, for which the preset
    # normally selects this cached Schur apply. Name it explicitly so a
    # loose-tolerance study keeps the same preconditioner. The preset guards
    # its automatic selection above block0_rtol=1e-4 because loose solves
    # can degrade the Schur columns; convergence must still be checked.
    parameters = common.solver_parameters(
        layout, "ksponly", outer_rtol=outer_rtol, block0_rtol=block0_rtol,
        ainvb=True)
    solver = SelfGravitatingGIASolver(
        z, common.maxwell_approximation(mantle, bulk_shear_ratio),
        layout=layout, dt=dt, bcs=bcs, fluid_core=common.fluid_core(),
        dtn_representation=common.DTN_REPRESENTATION,
        rotation_moments={"C": common.C_NONDIM,
                          "C_minus_A": common.C_MINUS_A["ks"]},
        Omega_sq=common.OMEGA_SQ,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=parameters)
    return solver, layout


# --------------------------------------------------------------------------
# Comparison with the reference
# --------------------------------------------------------------------------


def build_profile_samplers(parent, mantle, depth_fraction, geometry=None):
    """Locate near-surface profiles once; default to the legacy meridians."""
    if geometry is None:
        geometry = profile_geometry("cap", LOAD_CENTRE_DEG["cap"])
    samplers = []
    for directions in geometry["unit_vectors"]:
        points = common.RE * (1 - depth_fraction) * directions
        samplers.append((common.PointSampler(mantle, points),
                         common.PointSampler(parent, points)))
    return geometry, depth_fraction, samplers


class ProfileRateHistory:
    """Previous solved fields for profile/surface rates, including restart.

    The solver overwrites its own old state during solve(). Keep independent
    copies, and preserve the elastic diagnostic across march()'s reset to rest.
    No derivative of the instantaneous load jump is reported at t=0.
    """

    def __init__(self, solver, initial_time=None):
        self.displacement = Function(solver.displacement.function_space())
        self.potential = Function(solver.potential.function_space())
        self.last_time = initial_time
        self.previous_time = initial_time
        self.keep_elastic = False
        self.dt_yr = 0.0
        self.rotation = {}

    def capture(self, solver):
        self.displacement.assign(solver.displacement)
        self.potential.assign(solver.potential)
        self.rotation = dict(getattr(solver, "rotation_values", dict)())
        self.previous_time = self.last_time

    def before_step(self, solver, t_kyr):
        if t_kyr == 0:
            return
        if self.keep_elastic:
            self.keep_elastic = False
        else:
            self.capture(solver)

    def after_step(self, solver, t_kyr):
        self.last_time = t_kyr
        if t_kyr == 0:
            self.capture(solver)
            self.keep_elastic = True
        self.dt_yr = (t_kyr - self.previous_time) * 1000.0

    def metadata(self):
        return {"rate_method": "adjacent-step backward difference",
                "rate_valid": self.dt_yr > 0,
                "rate_interval_start_kyr": self.previous_time,
                "rate_interval_end_kyr": self.last_time,
                "rate_dt_yr": self.dt_yr}


def write_profiles(path, solver, mantle, profiles, t_kyr, *, rates=None):
    """Save displacement/geoid in m, and optional adjacent-step rates in m/yr."""
    geometry, depth_fraction, samplers = profiles
    gravity = common.gravity_exact_ufl(Constant(common.RE))
    geoid = solver.potential / gravity
    components = {"U_m": "unit_vectors", "V_m": "south_vectors",
                  "W_m": "east_vectors", "along_m": "tangent_vectors",
                  "across_m": "right_vectors"}
    values = {name: [] for name in components}
    rate_values = {name.replace("_m", "_m_per_yr"): [] for name in components}
    rate_values["N_m_per_yr"] = []
    N, found = [], []
    for i, (mechanics, potential) in enumerate(samplers):
        displacement = np.column_stack([mechanics(solver.displacement[j])
                                         for j in range(3)]) * common.D_SCALE
        for name, basis in components.items():
            values[name].append(np.sum(displacement * geometry[basis][i], axis=1))
        N.append(potential(geoid) * common.D_SCALE)
        found.append(mechanics.found & potential.found)
        if rates is not None:
            if rates.dt_yr > 0:
                velocity = np.column_stack([
                    mechanics((solver.displacement[j] - rates.displacement[j]) / rates.dt_yr)
                    for j in range(3)]) * common.D_SCALE
                for name, basis in components.items():
                    rate_values[name.replace("_m", "_m_per_yr")].append(
                        np.sum(velocity * geometry[basis][i], axis=1))
                rate_values["N_m_per_yr"].append(
                    potential((solver.potential - rates.potential) / gravity / rates.dt_yr)
                    * common.D_SCALE)
            else:
                for data in rate_values.values():
                    data.append(np.full(displacement.shape[0], np.nan))
    arrays = {**geometry, "t_kyr": t_kyr, "depth_fraction": depth_fraction,
              "geoid_definition": "gravitational potential / g0; excludes centrifugal potential",
              **{name: np.array(data) for name, data in values.items()}, "N_m": np.array(N),
              "found": np.array(found)}
    if rates is not None:
        arrays.update(rates.metadata())
        arrays.update({name: np.array(data) for name, data in rate_values.items()})
    if common.COMM_WORLD.rank == 0:
        np.savez(path, **arrays)
    missing = np.count_nonzero(~arrays["found"])
    log(f"  profiles written: {path} ({missing} missing points)")


def diagnostic_load_coefficients(nmax, *, load_nmax=NMAX,
                                 load_representation="spatial", load_shape="cap"):
    """Load coefficients for the reference on the diagnostic degree range.

    A finite spectral load has zero coefficients above its cutoff. The
    spatial load retains every degree >= 2 before mesh interpolation.
    """
    sigma_n = load_coefficients(nmax, load_shape)
    sigma_n[:2] = 0.0
    if load_representation == "spectral":
        sigma_n[load_nmax + 1:] = 0.0
    return sigma_n


def compare_cap(t_kyr, solver, parent, mantle, reference, *,
                diagnostic_nmax=NMAX, load_nmax=NMAX,
                load_representation="spatial", load_shape="cap", load_centre=None):
    """The cap-case comparison at one epoch: printed, and returned as a row.

    Args:
      t_kyr: the epoch.
      solver: the solved solver.
      parent, mantle: the two meshes.
      reference: the `TabooReference`.
      diagnostic_nmax: the projection and reference comparison cutoff.
      load_nmax, load_representation, load_shape: the load used by the solver.

    Returns:
      A dictionary of the model and reference values at the epoch, in
      metres and degrees.
    """
    U_n, V_n, N_n = surface_spectra(solver, parent, mantle, diagnostic_nmax,
                                  load_centre=load_centre)
    sigma_n = diagnostic_load_coefficients(
        diagnostic_nmax, load_nmax=load_nmax,
        load_representation=load_representation, load_shape=load_shape)
    U_ref_n, V_ref_n, N_ref_n = reference.coefficients(
        t_kyr, sigma_n, diagnostic_nmax)
    degrees = np.arange(reference.nmin, diagnostic_nmax + 1)

    # Degree 0 is the breathing mode. The load has no degree-0 content and
    # the fluid core keeps its volume, so |U_0| / |U_2| must be small.
    u0_over_u2 = abs(U_n[0]) / max(abs(U_n[2]), 1.0e-300)

    # The spatial series on a fine grid of colatitudes, from degree 2 on,
    # for both sides. The reference coefficients go into the same series.
    theta = np.linspace(0.0, np.pi, 4001)
    full = np.zeros(diagnostic_nmax + 1)

    def reference_series(coeffs, kind="P"):
        full[:] = 0.0
        full[reference.nmin:] = coeffs
        return series(full, theta, kind)

    Um, Nm, Vm = series(U_n, theta), series(N_n, theta), \
        series(V_n, theta, "dP")
    Ur, Nr, Vr = reference_series(U_ref_n), reference_series(N_ref_n), \
        reference_series(V_ref_n, "dP")
    # The signed maximum of V and not the maximum of |V|: the reference
    # peaks positive, and a maximum of |V| would hide a sign error.
    jm, jr = int(np.argmax(Vm)), int(np.argmax(Vr))

    row = {"t_kyr": t_kyr,
           "U0": Um[0], "U0_ref": Ur[0], "N0": Nm[0], "N0_ref": Nr[0],
           "U180": Um[-1], "U180_ref": Ur[-1],
           "N180": Nm[-1], "N180_ref": Nr[-1],
           "Vmax": Vm[jm], "Vmax_ref": Vr[jr],
           "Vmax_colatitude_deg": np.rad2deg(theta[jm]),
           "Vmax_colatitude_ref_deg": np.rad2deg(theta[jr]),
           "U0_over_U2": u0_over_u2,
           "degrees": degrees,
           "U_n": U_n[reference.nmin:], "U_n_ref": U_ref_n,
           "V_n": V_n[reference.nmin:], "V_n_ref": V_ref_n,
           "N_n": N_n[reference.nmin:], "N_n_ref": N_ref_n}

    log(f"\n  t = {t_kyr:g} kyr   (lengths in m)   |U_0| / |U_2| "
        f"{u0_over_u2:.3e}")
    log(f"    spatial quantities below are reconstructed from "
        f"degrees 2..{diagnostic_nmax}")
    log(f"    {'quantity':<12}{'model':>13}{'TABOO':>13}{'ratio':>9}")
    for name, key in (("U(0)", "U0"), ("N(0)", "N0"), ("U(180)", "U180"),
                      ("N(180)", "N180"), ("max V", "Vmax")):
        log(f"    {name:<12}{row[key]:>13.5f}{row[key + '_ref']:>13.5f}"
            f"{row[key] / row[key + '_ref']:>9.4f}")
    # Per degree. This separates an error flat in n (compressibility, a
    # scale factor) from one that grows with n (mesh resolution).
    log(f"    {'n':>5} {'U_n ratio':>10} {'V_n ratio':>10} {'N_n ratio':>10}")
    for j, n in enumerate(degrees):
        # Degrees above a finite load cutoff have zero reference response.
        ratios = [f"{values[j] / expected[j]:>10.4f}" if expected[j] != 0
                  else f"{'--':>10}" for values, expected in
                  ((row["U_n"], U_ref_n), (row["V_n"], V_ref_n),
                   (row["N_n"], N_ref_n))]
        log(f"    {n:>5} {' '.join(ratios)}")

    # A wrong solved state (a failed solve that PETSc did not report, or a
    # sign error) shows here first, and the rest of the run would be wasted.
    ratio = abs(row["U0"] / row["U0_ref"])
    if not 0.1 < ratio < 10.0:
        raise RuntimeError(
            f"U(0) ratio {ratio:.4g} at t = {t_kyr} kyr is outside "
            "[0.1, 10]. The solved state is wrong; stopping.")
    return row


def polar_motion_deg(solver):
    """(m_x, m_y) in degrees, from the solved rotation unknowns.

    m_1 and m_2 are the direction cosines of the displaced rotation axis
    along x and y. At a size of 1e-4 radians they are angles in radians to
    far better than the comparison needs.
    """
    values = solver.rotation_values()
    return float(np.rad2deg(values["m1"])), float(np.rad2deg(values["m2"]))


def rotation_diagnostics(solver):
    """Rotation-rate change and physical Delta LOD, positive for a longer day.

    m3 = delta(Omega)/Omega = -delta(I33)/C. Use the paper's nominal
    LOD of 86400 s, and its linearized conversion Delta LOD = -LOD * m3.
    """
    m3 = float(solver.rotation_values()["m3"])
    return {"m3": m3, "lod_ms": -86400.0 * 1000.0 * m3}


def compare_polar_motion(t_kyr, solver, reference, *, load_shape="cap", load_centre=None):
    """The polar-motion comparison at one epoch: printed, and returned as a row.

    Args:
      t_kyr: the epoch.
      solver: the solved solver.
      reference: the `TabooReference`.
      load_shape: the geometry used by the solver.

    Returns:
      A dictionary of the model and reference values, in degrees.
    """
    mx, my = polar_motion_deg(solver)
    mxr, myr = reference.polar_motion(t_kyr, load_shape, load_centre)
    reference_size = np.hypot(mxr, myr)
    phase_ref = float(np.degrees(np.arctan2(myr, mxr))) if reference_size > 1e-15 else None
    row = {"t_kyr": t_kyr, "mx": mx, "my": my, "absm": np.hypot(mx, my),
           "phase": np.degrees(np.arctan2(my, mx)),
           "mx_ref": mxr, "my_ref": myr, "absm_ref": np.hypot(mxr, myr),
           "phase_ref": phase_ref, **rotation_diagnostics(solver),
           "lod_eq33_ref_ms": reference.length_of_day_eq33(t_kyr, load_shape, load_centre)}
    log(f"\n  t = {t_kyr:g} kyr   polar motion (degrees)")
    log(f"    {'quantity':<12}{'model':>14}{'TABOO':>14}{'ratio':>11}")
    for name, key in (("m_x", "mx"), ("m_y", "my"), ("|m|", "absm")):
        expected = row[key + "_ref"]
        ratio = f"{row[key] / expected:>11.6f}" if abs(expected) > 1e-15 else "         --"
        log(f"    {name:<12}{row[key]:>14.7f}{expected:>14.7f}{ratio}")
    log(f"    phase model {row['phase']:.4f} deg; reference {phase_ref}")
    log(f"    m3 {row['m3']:.9e}; Delta LOD {row['lod_ms']:.9e} ms")
    log(f"    published Eq.33 LOD reference {row['lod_eq33_ref_ms']:.9e} ms "
        "(separate sign convention; no relative error scored)")
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def harmonic_degree(value):
    """Benchmark cutoffs retain degree 2, including for the DtN comparison."""
    value = int(value)
    if value < 2:
        raise argparse.ArgumentTypeError("harmonic cutoff must be at least 2")
    return value


def relative_tolerance(value):
    """A finite, positive relative tolerance smaller than one."""
    value = float(value)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("relative tolerance must be in (0, 1)")
    return value


def run_settings(args):
    """Settings recorded in checkpoints, including the load and diagnostics."""
    return {"case": args.case, "bulk_shear_ratio": args.bulk_shear_ratio,
            "load_centre_deg": list(args.load_centre),
            "load_shape": args.load_shape,
            "smoke": args.smoke, "load_representation": args.load_representation,
            "load_nmax": (args.load_nmax if args.load_representation == "spectral"
                          else None),
            "diagnostic_nmax": args.diagnostic_nmax, "dtn_degree": args.dtn_degree,
            "centre_of_mass": args.centre_of_mass,
            "write_frame_diagnostics": args.write_frame_diagnostics,
            "extra_epochs_kyr": list(args.extra_epochs_kyr),
            "outer_rtol": args.outer_rtol, "block0_rtol": args.block0_rtol,
            "timestep_scale": args.timestep_scale,
            "mesh_h_km": args.mesh_h_km, "litho_layers": args.litho_layers}


def checkpoint_run_settings(record, mesh_info=None):
    """Upgrade the old checkpoint record, whose one cutoff was shared.

    Cap records contain their diagnostic degrees, hence also the old load
    cutoff. Old polar-motion records have no degree information and are
    interpreted as the original default degree-10 configuration.
    """
    written = dict(record["run"])
    written.setdefault("load_centre_deg", list(LOAD_CENTRE_DEG[written["case"]]))
    written.setdefault("load_shape", "cap")
    # Before the CLI option, both boundary maps always used degree 5.
    written.setdefault("dtn_degree", 5)
    written.setdefault("outer_rtol", 1.0e-6)
    written.setdefault("block0_rtol", 1.0e-4)
    # Older Spada runs did not constrain the centre-of-mass frame.
    written.setdefault("centre_of_mass", False)
    written.setdefault("write_frame_diagnostics", False)
    written.setdefault("extra_epochs_kyr", [])
    mesh = mesh_info or common.MESHES["smoke" if written["smoke"] else "spada"]
    written.setdefault("timestep_scale", 1.0)
    written.setdefault("mesh_h_km", mesh["h_km"])
    written.setdefault("litho_layers", mesh["litho_layers"])
    if "load_representation" not in written:
        rows = record["rows"]
        nmax = int(rows[-1]["degrees"][-1]) if (
            written["case"] == "cap" and rows) else NMAX
        written.update(load_representation="spectral", load_nmax=nmax,
                       diagnostic_nmax=nmax)
    return written


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def time_segments(args, epochs):
    """Scale the step ladder, retaining exact solved states at output epochs."""
    ladder = tuple((end, dt * args.timestep_scale) for end, dt in common.STEP_LOAD_LADDER_YR)
    return common.time_segments(epochs, common.truncated_ladder(ladder, max(epochs) * 1000.))


def output_epochs(args):
    base = SMOKE_EPOCHS_KYR if args.smoke else EPOCHS_KYR
    return tuple(sorted(set(base) | set(args.extra_epochs_kyr)))


def parse_args(argv=None):
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["cap", "polar-motion"], default="cap",
                   help="cap: the selected load on the pole, U, V and N against TABOO. "
                        "polar-motion: the selected load at colatitude 25, longitude "
                        "75 with rotation, polar motion against T02-03.")
    p.add_argument("--load_shape", choices=LOAD_SHAPES, default="cap",
                   help="Table 4 geometry: 1500 m parabolic cap (default) or "
                        "1000 m uniform disc; both have a 10 degree radius")
    p.add_argument("--load_centre", nargs=2, type=float, metavar=("COLAT", "LON"),
                   help="load location in degrees, independent of rotation; "
                        "defaults: cap 0 0, polar-motion 25 75")
    p.add_argument("--bulk_shear_ratio", type=float, default=1000.0,
                   help="K / mu in every layer (default 1000; use 100 for "
                        "the coarser compressibility approximation)")
    p.add_argument("--timestep_scale", type=positive_float, default=1.0,
                   help="multiply the graded timestep ladder (default 1; coarse check 2)")
    p.add_argument("--mesh_h_km", type=positive_float,
                   help="bulk mesh size in km (default 250, or 1000 for --smoke)")
    p.add_argument("--litho_layers", type=positive_integer,
                   help="subdivisions of the 70 km lithosphere (default 2, or 1 for --smoke)")
    p.add_argument("--load_representation", choices=["spectral", "spatial"],
                   default="spatial",
                   help="spatial (default): evaluate the load and subtract only "
                        "degrees 0 and 1; spectral: sum load degrees 2..load_nmax")
    p.add_argument("--load_nmax", type=harmonic_degree, default=NMAX,
                   help="highest spectral load degree (default 10); "
                        "unused for --load_representation spatial")
    p.add_argument("--diagnostic_nmax", type=harmonic_degree, default=NMAX,
                   help="highest projected diagnostic degree, independently "
                        "of the load (default 10; TABOO reference extends to 256)")
    p.add_argument("--dtn_degree", type=harmonic_degree, default=common.DTN_DEGREE,
                   help="harmonic cutoff on both gravity boundary spheres "
                        "(default 5; use 2 for the coarser approximation; minimum 2)")
    p.add_argument("--outer_rtol", type=relative_tolerance, default=common.OUTER_RTOL,
                   help="coupled FGMRES relative tolerance (default 1e-6)")
    p.add_argument("--block0_rtol", type=relative_tolerance, default=common.BLOCK0_RTOL,
                   help="inner mechanics-potential FGMRES relative tolerance (default 1e-4)")
    p.add_argument("--centre_of_mass", action="store_true",
                   help="constrain the total mass dipole of mantle, fluid core and load "
                        "to zero (default off); must match the checkpoint on restart")
    p.add_argument("--write_frame_diagnostics", action="store_true",
                   help="record assembled load moments, every-step centre of mass and "
                        "frame multipliers; must match the checkpoint on restart")
    p.add_argument("--extra_epochs_kyr", nargs="+", type=float, default=[],
                   help="extra profile/surface epochs on existing solved timesteps; "
                        "does not change the timestep ladder; must match on restart")
    p.add_argument("--smoke", action="store_true",
                   help="run on a very coarse mesh to 0.1 kyr, to check that "
                        "the code runs; the numbers are not results")
    p.add_argument("--write_output", action="store_true",
                   help="write Paraview VTK files at every epoch")
    p.add_argument("--write_surface_output", action="store_true",
                   help="write surface-only displacement, geoid and rates "
                        "in physical units; independent of volume --write_output")
    p.add_argument("--surface_output_format", choices=("hdf5", "vtk"), default="hdf5",
                   help="surface storage format (default hdf5); vtk writes parallel "
                        "VTU pieces directly, useful for small debugging runs")
    p.add_argument("--write_profiles", action="store_true",
                   help="write direct near-surface displacement and geoid profiles "
                        "for either case and load shape")
    locations = p.add_mutually_exclusive_group()
    locations.add_argument("--profile_longitudes", nargs="+", type=float,
                           help="geographic meridians in degrees (cap default: 0 90 180)")
    locations.add_argument("--profile_bearings", nargs="+", type=float,
                           help="great-circle bearings clockwise from north at the "
                                "profile centre (polar-motion default: 0 90)")
    p.add_argument("--profile_centre", nargs=2, type=float, metavar=("COLAT", "LON"),
                   help="great-circle centre in degrees (default: load centre)")
    p.add_argument("--profile_extent_deg", type=float, default=180.0,
                   help="signed-distance half-range, or maximum meridian colatitude "
                        "(default 180 degrees)")
    p.add_argument("--profile_spacing_deg", type=float, default=0.1,
                   help="maximum angular spacing of profile samples (default 0.1 degrees)")
    p.add_argument("--profile_depth_fraction", type=float, default=1.0e-6,
                   help="inward sampling offset as a fraction of Earth radius")
    p.add_argument("--output_path", default="./",
                   help="the directory of every output file")
    p.add_argument("--checkpoint_every", type=int,
                   default=common.CHECKPOINT_EVERY,
                   help="steps between two checkpoints in "
                        "checkpoint_<case>.h5; 0 writes none")
    p.add_argument("--restart", default=None, metavar="FILE",
                   help="continue the run after the last checkpoint in FILE, "
                        "on the same number of ranks")
    args = p.parse_args(argv)
    centre = LOAD_CENTRE_DEG[args.case] if args.load_centre is None else args.load_centre
    if not np.all(np.isfinite(centre)) or not 0 <= centre[0] <= 180:
        p.error("--load_centre requires finite angles and colatitude in [0, 180]")
    args.load_centre = [float(centre[0]), float(centre[1] % 360)]
    mesh = common.MESHES["smoke" if args.smoke else "spada"]
    args.mesh_h_km = mesh["h_km"] if args.mesh_h_km is None else args.mesh_h_km
    args.litho_layers = mesh["litho_layers"] if args.litho_layers is None else args.litho_layers
    base_epochs = SMOKE_EPOCHS_KYR if args.smoke else EPOCHS_KYR
    original = time_segments(args, base_epochs)
    for epoch in args.extra_epochs_kyr:
        if not np.isfinite(epoch) or not 0 < epoch <= max(base_epochs):
            p.error("--extra_epochs_kyr must be finite, positive and within the run")
        aligned = any(t0 < epoch * 1000 <= t1 + 1e-9
                      and np.isclose((epoch * 1000 - t0) / step,
                                     round((epoch * 1000 - t0) / step), rtol=0, atol=1e-9)
                      for t0, t1, step, _, _ in original)
        if not aligned:
            p.error("--extra_epochs_kyr must coincide with existing solved timesteps")
    args.extra_epochs_kyr = sorted(set(args.extra_epochs_kyr))
    return args


def main():
    """Run one Spada case."""
    args = parse_args()
    if args.write_profiles and not 0 < args.profile_depth_fraction < 0.01:
        raise ValueError("--profile_depth_fraction must be in (0, 0.01)")
    geometry = (profile_geometry(
        args.case, args.load_centre, longitudes=args.profile_longitudes,
        bearings=args.profile_bearings, centre=args.profile_centre,
        extent_deg=args.profile_extent_deg, spacing_deg=args.profile_spacing_deg)
        if args.write_profiles else None)
    sampling_settings = (profile_settings(geometry, args.profile_depth_fraction)
                         if geometry is not None else None)
    if sampling_settings is not None:
        sampling_settings["rates"] = "adjacent-step backward difference, m/yr; t=0 undefined"
    reference = TabooReference()
    if args.case == "cap" and args.diagnostic_nmax > reference.nmax:
        raise ValueError(f"--diagnostic_nmax must not exceed the TABOO "
                         f"reference maximum {reference.nmax}")
    settings = run_settings(args)
    tic_run = time.time()
    os.makedirs(args.output_path, exist_ok=True)
    epochs = output_epochs(args)
    mesh_kind = "smoke" if args.smoke else "spada"

    log("=" * 78)
    log(f"Spada et al. (2011) benchmark, case '{args.case}'"
        f"{' (smoke run)' if args.smoke else ''}")
    log("=" * 78)
    load_description = (f"spectral degrees 2..{args.load_nmax}"
                        if args.load_representation == "spectral" else
                        "spatial load minus degrees 0 and 1 (no load cutoff)")
    log(f"  {args.load_shape} load, centre (colatitude, longitude) {args.load_centre}, "
        f"{load_description}, SphericalDtN(L={args.dtn_degree}), K/mu "
        f"{args.bulk_shear_ratio:g}")
    if args.case == "cap":
        log(f"  projected diagnostic degrees 2..{args.diagnostic_nmax}")
    log(f"  solver tolerances: outer {args.outer_rtol:g}, inner {args.block0_rtol:g}")
    log(f"  centre-of-mass constraint: {'on' if args.centre_of_mass else 'off'}")

    # A new run generates its mesh; a restarted run loads it from the
    # checkpoint file, because the stored fields can only be loaded onto the
    # mesh that the same file holds.
    restart = None
    if args.restart is None:
        parent, mantle, mesh_info = common.build_meshes(
            mesh_kind, args.output_path, f"spada_{args.case}",
            mesh_overrides={"h_km": args.mesh_h_km, "litho_layers": args.litho_layers})
    else:
        parent, mantle, mesh_info, restart = common.load_checkpoint(
            args.restart, mesh_kind)
        # The record says what run wrote the file. A restart with another
        # case or another K / mu would continue a different problem from
        # this state without any error.
        written = checkpoint_run_settings(restart.record, mesh_info)
        if written != settings:
            raise RuntimeError(f"{args.restart} was written by the run "
                               f"{written}; this run is {settings}")
    segments = time_segments(args, epochs)
    log(f"  mesh size {args.mesh_h_km:g} km, lithosphere subdivisions {args.litho_layers}, "
        f"timestep scale {args.timestep_scale:g}")
    log(f"  epochs (kyr) {list(epochs)}, "
        f"{sum(s[3] for s in segments)} steps after the elastic solve")

    dt = Constant(common.DT_ELASTIC)
    tic = time.time()
    solver, layout = build_solver(args.case, parent, mantle, dt,
                                  args.bulk_shear_ratio,
                                  load_nmax=args.load_nmax,
                                  load_representation=args.load_representation,
                                  load_shape=args.load_shape, dtn_degree=args.dtn_degree,
                                  outer_rtol=args.outer_rtol, block0_rtol=args.block0_rtol,
                                  load_centre=args.load_centre,
                                  centre_of_mass=args.centre_of_mass)
    log(f"  solver built ({time.time() - tic:.1f} s): "
        f"{solver.solution.function_space().dim()} unknowns, "
        f"{len(layout.real_fields)} Real rows")
    frame = None
    if args.write_frame_diagnostics:
        from spada_frame_diagnostics import FrameDiagnostics

        frame = FrameDiagnostics(solver)
        log(f"  assembled frame diagnostics: {frame.metadata}")

    vtk = None
    if args.write_output:
        # One VTK file per mesh: `VTKFile.write` needs every function on one
        # mesh, and the displacement and the potential live on two.
        vtk = (VTKFile(os.path.join(args.output_path,
                                    f"spada_{args.case}_mechanics.pvd")),
               VTKFile(os.path.join(args.output_path,
                                    f"spada_{args.case}_potential.pvd")))

    surface_output = None
    if args.write_surface_output:
        from spada_surface_output import SurfaceOutput

        surface_output = SurfaceOutput(args.output_path, args.case, parent, mantle,
                                       output_format=args.surface_output_format,
                                       restart_time=None if restart is None else restart.t_kyr)

    # One line per time step. The columns are the same for both cases apart
    # from the last ones: the rms radial displacement on Re (m) for the cap,
    # the polar motion (degrees) for the other case.
    # The lines of the parameter log and the epoch comparisons so far. A
    # restarted run takes them from the checkpoint and writes the log again,
    # because `ParameterLog` opens its file for writing and the log of the
    # stopped run can hold steps after the checkpoint.
    log_lines = [] if restart is None else restart.record["log_lines"]
    if args.case == "polar-motion":
        # Old logs predate m3/LOD export. Keep their missing values explicit
        # while preserving a rectangular table under the expanded header.
        log_lines = [line + " nan nan" if len(line.split()) == 8 else line
                     for line in log_lines]
    rows = [] if restart is None else restart.record["rows"]
    wall_before_s = 0.0 if restart is None else restart.record["wall_s"]
    # Every restart of the case so far, this one last, for the summary.
    restarts = ([] if restart is None else restart.record["restarts"]
                + [{"file": args.restart, "step": restart.step}])
    plog = ParameterLog(os.path.join(args.output_path,
                                     f"params_{args.case}.log"), parent)
    state_columns = ("urms_Re_m" if args.case == "cap" else "mx_deg my_deg m3 lod_ms")
    if frame is not None:
        state_columns += " " + " ".join(frame.columns)
    plog.log_str(f"step t_kyr dt_yr wall_s newton outer {state_columns}")
    for line in log_lines:
        plog.log_str(line)
    area_Re = assemble(Constant(1.0) * ds(common.SURF_RE, domain=mantle))
    rate_history = (ProfileRateHistory(
        solver, initial_time=None if restart is None else restart.t_kyr)
        if args.write_profiles or args.write_surface_output else None)

    def on_step(t_kyr, dt_yr, step, wall_s):
        if rate_history is not None:
            rate_history.after_step(solver, t_kyr)
        newton, outer = common.iteration_counts(solver)
        if args.case == "cap":
            u_r = vertical_component(solver.displacement)
            state = [sqrt(assemble(u_r * u_r * ds(common.SURF_RE,
                                                  domain=mantle)) / area_Re)
                     * common.D_SCALE]
        else:
            rotation = rotation_diagnostics(solver)
            state = [*polar_motion_deg(solver), rotation["m3"], rotation["lod_ms"]]
        if frame is not None:
            state += frame.sample()
        line = " ".join(str(v) for v in
                        [step, t_kyr, dt_yr, wall_s, newton, outer, *state])
        log_lines.append(line)
        plog.log_str(line)
        log(f"  step {step:4d}  t {t_kyr:9.4f} kyr  outer {outer:3d}  "
            f"wall {wall_s:8.1f} s")

    profiles = (build_profile_samplers(parent, mantle, args.profile_depth_fraction, geometry)
                if args.write_profiles else None)

    def on_epoch(t_kyr):
        if args.case == "cap":
            rows.append(compare_cap(
                t_kyr, solver, parent, mantle, reference,
                diagnostic_nmax=args.diagnostic_nmax, load_nmax=args.load_nmax,
                load_representation=args.load_representation, load_shape=args.load_shape,
                load_centre=args.load_centre))
        else:
            rows.append(compare_polar_motion(t_kyr, solver, reference,
                                              load_shape=args.load_shape,
                                              load_centre=args.load_centre))
        if frame is not None:
            rows[-1].update(frame.latest)
        if profiles is not None:
            write_profiles(os.path.join(args.output_path,
                                        f"profiles_{args.case}_{t_kyr:g}kyr.npz"),
                           solver, mantle, profiles, t_kyr, rates=rate_history)
        if vtk is not None:
            vtk[0].write(solver.displacement, time=t_kyr)
            vtk[1].write(solver.potential, time=t_kyr)
        if surface_output is not None:
            surface_output.write(solver, rate_history, t_kyr)

    def record():
        """The driver's part of a checkpoint: enough to continue the outputs."""
        return {"run": settings, "profile_sampling": sampling_settings,
                "frame_diagnostics": None if frame is None else frame.metadata,
                "surface_output": None if surface_output is None else surface_output.metadata(),
                "log_lines": log_lines, "rows": rows, "restarts": restarts,
                "wall_s": wall_before_s + time.time() - tic_run}

    # A restarted run appends to the file it came from, so that one file
    # holds every checkpoint of the case.
    checkpoint = common.RunCheckpoint(
        args.restart or common.checkpoint_path(args.output_path, args.case),
        parent, solver, args.checkpoint_every, mesh_info, record,
        restart=restart)
    start_step = 0
    if restart is not None:
        common.restore_checkpoint(restart, solver, parent)
        start_step = restart.step

    common.march(solver, dt, segments, elastic=True, on_step=on_step,
                 on_epoch=on_epoch, checkpoint=checkpoint,
                 start_step=start_step,
                 before_step=(None if rate_history is None else
                              lambda t: rate_history.before_step(solver, t)))
    plog.close()

    common.write_json(
        os.path.join(args.output_path, f"summary_{args.case}.json"),
        {"benchmark": "Spada et al. (2011)", **settings,
         "rotation": args.case == "polar-motion",
         "diagnostic_axis": "load-centred" if args.case == "cap" else None,
         "profile_sampling": sampling_settings,
         "frame_diagnostics": None if frame is None else frame.metadata,
         "surface_output": None if surface_output is None else surface_output.metadata(),
         # Keep the old key for readers of the diagnostic coefficient arrays.
         "nmax": args.diagnostic_nmax,
         "ranks": parent.comm.size, "mesh": mesh_info,
         # The wall time of every leg together, so that a restarted run
         # reports the cost of the whole case.
         "wall_s": wall_before_s + time.time() - tic_run,
         "restarts": restarts, "epochs": rows})


if __name__ == "__main__":
    main()
