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
    A parabolic ice cap, 1500 m thick at its centre, 10 degrees in angular
    radius, ice density 931 kg m^-3:

        sigma(gamma) = rho_ice h sqrt((cos gamma - cos alpha) / (1 - cos alpha))

    with gamma the angular distance from the cap centre. The load is switched
    on at t = 0 and then held. It enters as the Legendre series
    sum_{n=2}^{10} sigma_n P_n(cos gamma) in two places that must agree: as a
    normal traction on the surface of the mechanics, and as a mass sheet in
    the Poisson equation. Degrees 0 and 1 are left out, as in the benchmark,
    which removes the question of the reference frame. The TABOO reference is
    built at the same truncation, so both sides see the same truncated load.

THE TWO CASES
    --case cap
        The cap on the north pole, rotation off. At each epoch the radial
        displacement U, the colatitudinal displacement V and the geoid N on
        Re are projected onto Legendre polynomials in cos(colatitude), and
        each degree n = 2..10 is compared with TABOO. The summary also gives
        U and N at both poles and the maximum of V.

    --case polar-motion
        The same cap centred at colatitude 25, longitude 75 degrees, rotation
        on. At each epoch the polar motion m = (m_x, m_y), its size |m| and
        its phase are compared with the TABOO series of test T02-03 (Chandler
        wobble excluded). The reference phase is lambda_c + 180 deg = -105
        degrees at every epoch: it depends on the load position only.

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
    reproduce the non-rotating reference, and an off-axis load has no zonal
    spectrum.

DISCRETISATION
    Mesh      `selfgrav_common.MESHES["spada"]`, generated in the job: 250 km
              lateral cell size, two 35 km lithosphere layers, curved to P2.
    Unknowns  CG3 displacement and DG2 internal variables on the mantle
              submesh, CG2 potential on the whole mesh, `Real` unknowns for
              the core pressure and, in the polar-motion case, the three
              rotation components.
    Time      Backward Euler: 10 yr steps to 0.1 kyr, 50 yr to 1 kyr, 100 yr
              to 10 kyr and 500 yr to 20 kyr. The t = 0 state is the elastic
              response, solved on its own.
    Elastic   K / mu = 100 by default (`--bulk_shear_ratio`). TABOO is
              incompressible, which is one source of the remaining
              difference.

RUN
    mpiexec -np <N> python3 spada.py --case cap
    mpiexec -np <N> python3 spada.py --case polar-motion

    `--smoke` runs the same code on a very coarse mesh to 0.1 kyr. It checks
    that the code runs; its numbers are not results.

OUTPUT (in --output_path)
    params_<case>.log      One line per time step: the step, the time, the
                           step length, the wall time of the solve, the
                           iteration counts and one number of the state (the
                           rms radial displacement on Re for the cap, the
                           polar motion for the other case).
    summary_<case>.json    The comparison with TABOO at every epoch, the
                           mesh record (gmsh version, MD5, cell counts) and
                           the run settings. `test_benchmarks.py` reads it.
    spada_<case>.msh       The mesh of the run.
    spada_<case>_*.pvd     VTK files, with `--write_output` only. The
                           displacement and the potential live on different
                           meshes, so they go to two files.

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

import numpy as np  # noqa: E402
from gadopt import (Constant, Function, FunctionSpace,  # noqa: E402
                    ParameterLog, SpatialCoordinate, SphericalDtN, VTKFile,
                    as_vector, assemble, avg, conditional, dot, ds, log, sqrt)
from gadopt.gia_gravity import (SelfGravitatingGIASolver,  # noqa: E402
                                rigid_rotation_nullspace,
                                self_gravitating_gia_space)
from gadopt.utility import vertical_component  # noqa: E402

import selfgrav_common as common  # noqa: E402

# --------------------------------------------------------------------------
# Settings of the benchmark
# --------------------------------------------------------------------------

#: The truncation degree of the load, of the TABOO reference and of the
#: Legendre projection of the answer.
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
#: The ice cap: thickness at the centre (m), angular radius (degrees) and
#: density (kg m^-3).
CAP_THICKNESS_M, CAP_RADIUS_DEG, RHO_ICE = 1500.0, 10.0, 931.0
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


def cap_load(nmax):
    """The Legendre coefficients sigma_n of the ice cap, kg m^-2.

    Table 4 of Spada et al. (2011), for degrees 0..nmax, of
    sigma(theta) = rho h sqrt((cos theta - cos alpha) / (1 - cos alpha)).

    Args:
      nmax: the highest degree.

    Returns:
      An array of `nmax + 1` coefficients, indexed from degree 0.
    """
    a = np.deg2rad(CAP_RADIUS_DEG)
    n = np.arange(0, nmax + 1)

    def T(k):
        return np.cos(k * a)

    return -RHO_ICE * CAP_THICKNESS_M / (4 * (1 - np.cos(a))) * (
        (T(n + 1) - T(n + 2)) / (n + 1.5) - (T(n - 1) - T(n)) / (n - 0.5))


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

    def love_time(self, t_kyr, nmax):
        """hbar, lbar and kbar for degrees nmin..nmax at `t_kyr`.

        Args:
          t_kyr: the epoch, kyr.
          nmax: the highest degree.

        Returns:
          Three arrays, one value per degree.
        """
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

    def polar_motion(self, t_kyr):
        """The T02-03 polar motion (m_x, m_y) at `t_kyr`, in degrees.

        The series is tabulated at 35 epochs between 0 and 20 kyr, including
        every output epoch of this driver. Other epochs are interpolated
        linearly.
        """
        t = self.data["pm_cap_t_kyr"]
        return (float(np.interp(t_kyr, t, self.data["pm_cap_mx_deg"])),
                float(np.interp(t_kyr, t, self.data["pm_cap_my_deg"])))


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


def load_axis(case):
    """The unit vector from the centre of the Earth to the load centre."""
    colat, lon = np.deg2rad(LOAD_CENTRE_DEG[case])
    return np.array([np.sin(colat) * np.cos(lon),
                     np.sin(colat) * np.sin(lon), np.cos(colat)])


def load_field(mesh, sigma_n, axis):
    """The truncated cap series sum_{n=2}^{NMAX} sigma_n P_n(cos gamma), in CG2.

    cos(gamma) = (X . axis) / |X| is the cosine of the angular distance from
    the load centre, so the field depends on direction only and is defined at
    every radius. The solver reads it on the Re facets. The series is
    interpolated into CG2, the space of the potential, so that the load the
    solver sees is a load the mesh can carry.

    Args:
      mesh: the mesh to build the field on.
      sigma_n: the non-dimensional coefficients, indexed from degree 0.
      axis: the unit vector to the load centre.

    Returns:
      A CG2 `Function` of the non-dimensional surface density.
    """
    X = SpatialCoordinate(mesh)
    cos_gamma = dot(X, Constant(axis)) / sqrt(dot(X, X))
    P = legendre_ufl(NMAX, cos_gamma)
    expr = Constant(0.0)
    for n in range(2, NMAX + 1):
        expr = expr + Constant(sigma_n[n]) * P[n]
    return Function(FunctionSpace(mesh, "CG", 2),
                    name="sigma_load").interpolate(expr)


def project_surface(field_expr, mesh, measure, interior, basis="P"):
    """Legendre coefficients f_n = int f B_n dS / int B_n^2 dS on Re.

    B_n is P_n(cos theta) for `basis="P"` and dP_n/dtheta for `basis="dP"`,
    with theta the colatitude. The denominator is assembled and not taken
    from the closed form 4 pi R^2 / (2n + 1), so that the discretisation
    error of the surface cancels between numerator and denominator.

    Args:
      field_expr: the UFL expression to project.
      mesh: the mesh of `measure`.
      measure: `ds(tag)` for exterior facets, `dS(tag)` for interior ones.
      interior: whether `measure` is an interior-facet measure, which needs
        `avg`.
      basis: `"P"` or `"dP"`.

    Returns:
      A NumPy array of the coefficients for degrees 0..NMAX.
    """
    X = SpatialCoordinate(mesh)
    x = X[2] / sqrt(dot(X, X))
    P = legendre_ufl(NMAX, x)
    if basis == "dP":
        P = dlegendre_ufl(NMAX, x, P)
    measure = measure(metadata={"quadrature_degree": QUAD_DEGREE})
    out = np.zeros(NMAX + 1)
    for n in range(NMAX + 1):
        if interior:
            num = assemble(avg(field_expr * P[n]) * measure)
            den = assemble(avg(P[n] * P[n]) * measure)
        else:
            num = assemble(field_expr * P[n] * measure)
            den = assemble(P[n] * P[n] * measure)
        out[n] = num / den if abs(den) > 0.0 else 0.0
    return out


def surface_spectra(solver, parent, mantle):
    """U_n, V_n and N_n in metres from a solved state of the cap case.

    U is the radial displacement, V the colatitudinal (southward)
    displacement expanded in dP_n/dtheta, and N the geoid psi / g_0 on Re.
    The geoid is built from parent coordinates, because psi lives on the
    parent, and a quotient with a g_0 built on the mantle would mix two
    meshes in one expression.

    Args:
      solver: the solved `SelfGravitatingGIASolver`.
      parent, mantle: the two meshes.

    Returns:
      `(U_n, V_n, N_n)`, NumPy arrays for degrees 0..NMAX, in metres.
    """
    ds_mantle = ds(common.SURF_RE, domain=mantle)
    u = solver.displacement
    # The mantle mesh is marked non-Cartesian, so `vertical_component` is the
    # radial component u . X / |X|.
    U_n = project_surface(vertical_component(u), mantle, ds_mantle,
                          interior=False)

    # e_theta |X| sin(theta) = (z x, z y, -(x^2 + y^2)): the colatitude
    # direction without its normalisation, divided by r and by the distance
    # from the axis. That distance is bounded below to keep the expression
    # finite on the axis.
    Xm = SpatialCoordinate(mantle)
    e_theta = as_vector((Xm[2] * Xm[0], Xm[2] * Xm[1],
                         -(Xm[0]**2 + Xm[1]**2)))
    rho_axis = sqrt(Xm[0]**2 + Xm[1]**2)
    u_theta = dot(u, e_theta) / (
        sqrt(dot(Xm, Xm)) * conditional(rho_axis > 1e-12, rho_axis,
                                        Constant(1e-12)))
    V_n = project_surface(u_theta, mantle, ds_mantle, interior=False,
                          basis="dP")

    Xp = SpatialCoordinate(parent)
    geoid = solver.potential / common.gravity_exact_ufl(sqrt(dot(Xp, Xp)))
    N_n = project_surface(geoid, parent, solver.form.dS(common.SURF_RE),
                          interior=True)
    # Non-dimensional lengths to metres.
    return U_n * common.D_SCALE, V_n * common.D_SCALE, N_n * common.D_SCALE


def series(coeffs, theta, kind="P"):
    """sum_{n >= 2} f_n P_n(cos theta), or the same sum over dP_n/dtheta.

    The sum starts at degree 2, because the load and the reference have no
    degree 0 or 1. Any degree-0 or degree-1 content of the model would
    otherwise enter the pole values with nothing on the reference side.

    Args:
      coeffs: coefficients for degrees 0..NMAX.
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


def build_solver(case, parent, mantle, dt, bulk_shear_ratio):
    """The coupled self-gravitating solver of one Spada case.

    Args:
      case: `"cap"` or `"polar-motion"`.
      parent, mantle: the two meshes.
      dt: the time step `Constant`, in Maxwell times.
      bulk_shear_ratio: K / mu.

    Returns:
      `(solver, layout)`.
    """
    rotation = case == "polar-motion"
    axis = load_axis(case)
    # The load coefficients, scaled by rho_bar D.
    sigma_n = cap_load(NMAX) / (common.RHO_BAR * common.D_SCALE)
    sigma_parent = load_field(parent, sigma_n, axis)
    sigma_mantle = load_field(mantle, sigma_n, axis)

    # The Poisson equation: DtN maps on the two truncation spheres, and the
    # ice as a mass sheet on the interior facets of Re. Without the sheet, U
    # is unchanged and N changes sign.
    gravity_bcs = {
        common.SURF_OUTER: {"dtn": SphericalDtN(L=common.DTN_DEGREE)},
        common.SURF_INNER: {"dtn": SphericalDtN(L=common.DTN_DEGREE)},
        common.SURF_RE: {"interior_sigma": sigma_parent},
    }
    Z, layout = self_gravitating_gia_space(
        mantle, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        fluid_core=True, self_gravity_number=common.LAMBDA,
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
    solver = SelfGravitatingGIASolver(
        z, common.maxwell_approximation(mantle, bulk_shear_ratio),
        layout=layout, dt=dt, bcs=bcs, fluid_core=common.fluid_core(),
        dtn_representation=common.DTN_REPRESENTATION,
        rotation_moments={"C": common.C_NONDIM,
                          "C_minus_A": common.C_MINUS_A["ks"]},
        Omega_sq=common.OMEGA_SQ,
        nullspace=nullspace, transpose_nullspace=nullspace,
        solver_parameters=common.solver_parameters(layout, "ksponly"))
    return solver, layout


# --------------------------------------------------------------------------
# Comparison with the reference
# --------------------------------------------------------------------------


def compare_cap(t_kyr, solver, parent, mantle, reference):
    """The cap-case comparison at one epoch: printed, and returned as a row.

    Args:
      t_kyr: the epoch.
      solver: the solved solver.
      parent, mantle: the two meshes.
      reference: the `TabooReference`.

    Returns:
      A dictionary of the model and reference values at the epoch, in
      metres and degrees.
    """
    U_n, V_n, N_n = surface_spectra(solver, parent, mantle)
    U_ref_n, V_ref_n, N_ref_n = reference.coefficients(
        t_kyr, cap_load(NMAX), NMAX)
    degrees = np.arange(reference.nmin, NMAX + 1)

    # Degree 0 is the breathing mode. The load has no degree-0 content and
    # the fluid core keeps its volume, so |U_0| / |U_2| must be small.
    u0_over_u2 = abs(U_n[0]) / max(abs(U_n[2]), 1.0e-300)

    # The spatial series on a fine grid of colatitudes, from degree 2 on,
    # for both sides. The reference coefficients go into the same series.
    theta = np.linspace(0.0, np.pi, 4001)
    full = np.zeros(NMAX + 1)

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
    log(f"    {'quantity':<12}{'model':>13}{'TABOO':>13}{'ratio':>9}")
    for name, key in (("U(0)", "U0"), ("N(0)", "N0"), ("U(180)", "U180"),
                      ("N(180)", "N180"), ("max V", "Vmax")):
        log(f"    {name:<12}{row[key]:>13.5f}{row[key + '_ref']:>13.5f}"
            f"{row[key] / row[key + '_ref']:>9.4f}")
    # Per degree. This separates an error flat in n (compressibility, a
    # scale factor) from one that grows with n (mesh resolution).
    log(f"    {'n':>5} {'U_n ratio':>10} {'V_n ratio':>10} {'N_n ratio':>10}")
    for j, n in enumerate(degrees):
        log(f"    {n:>5} {row['U_n'][j] / U_ref_n[j]:>10.4f} "
            f"{row['V_n'][j] / V_ref_n[j]:>10.4f} "
            f"{row['N_n'][j] / N_ref_n[j]:>10.4f}")

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


def compare_polar_motion(t_kyr, solver, reference):
    """The polar-motion comparison at one epoch: printed, and returned as a row.

    Args:
      t_kyr: the epoch.
      solver: the solved solver.
      reference: the `TabooReference`.

    Returns:
      A dictionary of the model and reference values, in degrees.
    """
    mx, my = polar_motion_deg(solver)
    mxr, myr = reference.polar_motion(t_kyr)
    row = {"t_kyr": t_kyr, "mx": mx, "my": my, "absm": np.hypot(mx, my),
           "phase": np.degrees(np.arctan2(my, mx)),
           "mx_ref": mxr, "my_ref": myr, "absm_ref": np.hypot(mxr, myr),
           "phase_ref": REFERENCE_PHASE_DEG}
    log(f"\n  t = {t_kyr:g} kyr   polar motion (degrees)")
    log(f"    {'quantity':<12}{'model':>14}{'TABOO':>14}{'ratio':>11}")
    for name, key in (("m_x", "mx"), ("m_y", "my"), ("|m|", "absm")):
        log(f"    {name:<12}{row[key]:>14.7f}{row[key + '_ref']:>14.7f}"
            f"{row[key] / row[key + '_ref']:>11.6f}")
    log(f"    {'phase':<12}{row['phase']:>14.4f}{REFERENCE_PHASE_DEG:>14.4f}")
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    """The command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["cap", "polar-motion"], default="cap",
                   help="cap: the cap on the pole, U, V and N against TABOO. "
                        "polar-motion: the cap at colatitude 25, longitude "
                        "75 with rotation, polar motion against T02-03.")
    p.add_argument("--bulk_shear_ratio", type=float, default=100.0,
                   help="K / mu in every layer")
    p.add_argument("--smoke", action="store_true",
                   help="run on a very coarse mesh to 0.1 kyr, to check that "
                        "the code runs; the numbers are not results")
    p.add_argument("--write_output", action="store_true",
                   help="write Paraview VTK files at every epoch")
    p.add_argument("--output_path", default="./",
                   help="the directory of every output file")
    return p.parse_args()


def main():
    """Run one Spada case."""
    args = parse_args()
    tic_run = time.time()
    os.makedirs(args.output_path, exist_ok=True)
    epochs = SMOKE_EPOCHS_KYR if args.smoke else EPOCHS_KYR
    mesh_kind = "smoke" if args.smoke else "spada"

    log("=" * 78)
    log(f"Spada et al. (2011) benchmark, case '{args.case}'"
        f"{' (smoke run)' if args.smoke else ''}")
    log("=" * 78)
    log(f"  load centre (colatitude, longitude) {LOAD_CENTRE_DEG[args.case]}, "
        f"degrees 2..{NMAX}, SphericalDtN(L={common.DTN_DEGREE}), K/mu "
        f"{args.bulk_shear_ratio:g}")

    parent, mantle, mesh_info = common.build_meshes(
        mesh_kind, args.output_path, f"spada_{args.case}")
    reference = TabooReference()
    segments = common.time_segments(
        epochs, common.truncated_ladder(common.STEP_LOAD_LADDER_YR,
                                        max(epochs) * 1000.0))
    log(f"  epochs (kyr) {list(epochs)}, "
        f"{sum(s[3] for s in segments)} steps after the elastic solve")

    dt = Constant(common.DT_ELASTIC)
    tic = time.time()
    solver, layout = build_solver(args.case, parent, mantle, dt,
                                  args.bulk_shear_ratio)
    log(f"  solver built ({time.time() - tic:.1f} s): "
        f"{solver.solution.function_space().dim()} unknowns, "
        f"{len(layout.real_fields)} Real rows")

    vtk = None
    if args.write_output:
        # One VTK file per mesh: `VTKFile.write` needs every function on one
        # mesh, and the displacement and the potential live on two.
        vtk = (VTKFile(os.path.join(args.output_path,
                                    f"spada_{args.case}_mechanics.pvd")),
               VTKFile(os.path.join(args.output_path,
                                    f"spada_{args.case}_potential.pvd")))

    # One line per time step. The columns are the same for both cases apart
    # from the last ones: the rms radial displacement on Re (m) for the cap,
    # the polar motion (degrees) for the other case.
    plog = ParameterLog(os.path.join(args.output_path,
                                     f"params_{args.case}.log"), parent)
    state_columns = ("urms_Re_m" if args.case == "cap" else "mx_deg my_deg")
    plog.log_str(f"step t_kyr dt_yr wall_s newton outer {state_columns}")
    area_Re = assemble(Constant(1.0) * ds(common.SURF_RE, domain=mantle))

    def on_step(t_kyr, dt_yr, step, wall_s):
        newton, outer = common.iteration_counts(solver)
        if args.case == "cap":
            u_r = vertical_component(solver.displacement)
            state = [sqrt(assemble(u_r * u_r * ds(common.SURF_RE,
                                                  domain=mantle)) / area_Re)
                     * common.D_SCALE]
        else:
            state = list(polar_motion_deg(solver))
        plog.log_str(" ".join(str(v) for v in
                              [step, t_kyr, dt_yr, wall_s, newton, outer,
                               *state]))
        log(f"  step {step:4d}  t {t_kyr:9.4f} kyr  outer {outer:3d}  "
            f"wall {wall_s:8.1f} s")

    rows = []

    def on_epoch(t_kyr):
        if args.case == "cap":
            rows.append(compare_cap(t_kyr, solver, parent, mantle, reference))
        else:
            rows.append(compare_polar_motion(t_kyr, solver, reference))
        if vtk is not None:
            vtk[0].write(solver.displacement, time=t_kyr)
            vtk[1].write(solver.potential, time=t_kyr)

    common.march(solver, dt, segments, elastic=True, on_step=on_step,
                 on_epoch=on_epoch)
    plog.close()

    common.write_json(
        os.path.join(args.output_path, f"summary_{args.case}.json"),
        {"benchmark": "Spada et al. (2011)", "case": args.case,
         "smoke": args.smoke, "bulk_shear_ratio": args.bulk_shear_ratio,
         "nmax": NMAX, "dtn_degree": common.DTN_DEGREE,
         "ranks": parent.comm.size, "mesh": mesh_info,
         "wall_s": time.time() - tic_run, "epochs": rows})


if __name__ == "__main__":
    main()
