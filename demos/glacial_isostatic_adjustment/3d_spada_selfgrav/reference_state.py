r"""The M3-L70-V01 reference state: rho_0, Phi_0 and g_0 on the parent sphere.

Handoff §2.1 (the model), §2.2 (the non-dimensionalisation) and §2.3 (why Phi_0
is *computed* rather than prescribed).

The point of computing it is discrete consistency.  What the momentum
equation's buoyancy and the coupled potential must agree about is not the
paper's gravity column but the field the operator actually sees, so Phi_0 is
solved from the same rho_0, on the same mesh, with the same DtN treatment, by
the same `GravitySolver` the coupled residual uses.  The paper's column is then
an *independent check* on that solve rather than its definition.

## Non-dimensionalisation, and the one factor that is easy to get wrong

Lengths by D = Re - Rc = 2891 km, densities by rho_bar = 5511.68 kg m^-3,
accelerations by g_bar = 9.81555 m s^-2.  Writing psi = psi_hat g_bar D turns
nabla^2 psi = -4 pi G rho into

    nabla_hat^2 psi_hat = -(4 pi G rho_bar D / g_bar) rho_hat = -Lambda rho_hat

so the number handed to `GravitySolver` as `gravitational_constant` is
**Lambda / (4 pi)**, not Lambda and not G.  Lambda = 1.361325 (§2.2).  Getting
this wrong by 4 pi is a factor of 12.6 in g_0 and shows up immediately in G-1,
which is one reason G-1 is cheap to isolate.

The shipped solver's psi satisfies nabla^2 psi = -4 pi G rho, so psi is *minus*
the Newtonian potential and the gravitational acceleration is

    g_0 = +grad(psi),

pointing inward, with |g_0| = G M(<r) / r^2.  The sign is the same one road map
§4 records for the geoid (+psi/g_0) and the same reason.

## The unmeshed core

The mesh starts at 0.5 Rc, so the ball r < 0.5 Rc is not represented.  It is
spherically symmetric and its whole effect on the domain is an l = 0 exterior
field, which enters as a prescribed flux on the inner boundary.  The domain's
outward normal there points *towards* the origin, n = -r_hat, so

    d(psi_hat)/d(n) = +(Lambda / 4 pi) M_hat_core / r^2   at r = 0.5 Rc

with M_hat_core = rho_hat_core (4/3) pi (0.5 Rc)^3.  The interior DtN handles
every l >= 1 exactly and is homogeneous at l = 0 (the regular interior solution
is a constant), which is precisely the part the flux supplies -- so the two
conditions are complementary rather than double-counted, and both are declared
on tag 5.
"""
import numpy as np

import gadopt  # noqa: F401  BEFORE firedrake
from firedrake import (Constant, Function, FunctionSpace, SpatialCoordinate,
                       assemble, conditional, ds, dx, grad, inner, sqrt)
from gadopt import GravitySolver, SphericalDtN

import generate_selfgrav_sphere as gen

# Physical constants, prescribed by §2.1 and not to be derived.
G_NEWTON = 6.6732e-11          # m^3 kg^-1 s^-2
A_EARTH = 6.371e6              # m
D_SCALE = 2.891e6              # m, = Re - Rc
RHO_BAR = 5511.68              # kg m^-3, the model's own mean density
G_BAR = 9.81555                # m s^-2

#: Layers of M3-L70-V01, outermost first: (r_outer_km, r_inner_km, rho).
#: The core row runs to r = 0, so it covers both the meshed inner region and
#: the unmeshed ball.
LAYERS_KM = [(6371.0, 6301.0, 3037.0),
             (6301.0, 5951.0, 3438.0),
             (5951.0, 5701.0, 3871.0),
             (5701.0, 3480.0, 4978.0),
             (3480.0, 0.0, 10750.0)]

#: The paper's gravity column, §2.1, at the five radii above, outermost first.
GRAVITY_COLUMN = {6371.0: 9.815, 6301.0: 9.854, 5951.0: 9.978,
                  5701.0: 10.024, 3480.0: 10.457}

#: Which tagged surface sits at each of those radii.  6371 and 3480 are Re and
#: Rc; the other three are the density interfaces A2 tags additively.
SURFACE_TAGS = {6371.0: gen.SURF_RE, 6301.0: gen.SURF_D3, 5951.0: gen.SURF_D2,
                5701.0: gen.SURF_D1, 3480.0: gen.SURF_RC}
INTERIOR_TAGS = {gen.SURF_RE, gen.SURF_RC, gen.SURF_D1, gen.SURF_D2,
                 gen.SURF_D3}

LAMBDA = 4 * np.pi * G_NEWTON * RHO_BAR * D_SCALE / G_BAR


# ---------------------------------------------------------------------------
# Rotation: the moments of the reference hydrostatic figure, and Omega^2.
# ---------------------------------------------------------------------------
# **These live here, in the leaf, because two drivers need them.**  They were
# a literal in `b1_elastic.py` and a definition in `b4_polar_motion.py`, and
# the literal was the *secondary* value, truncated.  Putting them in the
# rotation driver and importing sideways would have made the elastic driver
# depend on the polar-motion driver; both now import downward from here, which
# is where `RHO_BAR`, `G_BAR`, `D_SCALE` and `LAMBDA` already come from.

#: Non-dimensionalising constant for the moments of inertia, rho_bar D^5.
RHO_D5 = RHO_BAR * D_SCALE**5

#: C is common to both C-A choices; it matters to `m_3`, the rotation-rate
#: change, and not at all to polar motion.
C_NONDIM = 8.0394e37 / RHO_D5                       # 72.226935

#: **The C-A choice, declared in advance.**  `ks` is PRIMARY.  The prescribed
#: C - A = 2.63e35 implies k_s = 0.94334 against the 0.96672389 the benchmark's
#: own rotation calculation demonstrably used -- 2.42% inconsistent, amplified
#: by 1/(1 - k_T/k_s) to +3.6% at t = 0 and +7-12% by 20 kyr.  Handoff §2.2 and
#: `NOTES/REVIEW-SPADA-B4-PRECOMMIT.md` §4.  Selecting the wrong key here
#: recreates the trap one indirection deeper, so callers assert on the value.
C_MINUS_A = {"ks": 2.6952e35 / RHO_D5,              # 0.24214001  <- USE THIS
             "prescribed": 2.63e35 / RHO_D5}        # 0.23628236  secondary

#: The primary choice, by name, so a caller cannot pick the wrong key silently.
C_MINUS_A_PRIMARY = C_MINUS_A["ks"]

#: Omega^2 non-dimensionalised, Omega = 7.292115e-5 rad/s.
OMEGA_SQ = 7.292115e-5**2 * D_SCALE / G_BAR          # 1.5661757e-03



def layers_nondim():
    """(r_outer, r_inner, rho) non-dimensional, outermost first."""
    return [(ro / (D_SCALE / 1e3), ri / (D_SCALE / 1e3), rho / RHO_BAR)
            for ro, ri, rho in LAYERS_KM]


def analytic_gravity(r_km):
    """|g| at radius `r_km` from the layered rho_0, in m s^-2.

    Derived, not prescribed: §2.1 is explicit that the paper's column is the
    result of this integral and that reproducing it is the point.
    """
    m = 0.0
    for ro, ri, rho in sorted(LAYERS_KM, key=lambda x: x[1]):
        lo, hi = ri * 1e3, min(ro * 1e3, r_km * 1e3)
        if hi > lo:
            m += 4 / 3 * np.pi * (hi**3 - lo**3) * rho
    return G_NEWTON * m / (r_km * 1e3) ** 2


def total_mass():
    """Total mass of the model, kg.  §2.1 records 5.97029e24."""
    return sum(4 / 3 * np.pi * ((ro * 1e3)**3 - (ri * 1e3)**3) * rho
               for ro, ri, rho in LAYERS_KM)


def mean_density():
    """Mean density of the model.  §2.1 records 5511.68."""
    return total_mass() / (4 / 3 * np.pi * A_EARTH**3)


def density_field(mesh, degree=0):
    """rho_0 as a layered DG field on the parent, zero in the buffer.

    DG0 by default and mesh-conforming by construction (the generator makes
    every interface a geometric entity), so each cell lies wholly inside one
    layer and the representation is exact.  §2.3's proviso -- rho_0 and Phi_0
    must share level surfaces -- is therefore satisfied structurally.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    expr = Constant(0.0)  # the buffer is vacuum
    for r_out, r_in, rho in layers_nondim():
        expr = conditional(r < r_out, Constant(rho), expr)
    rho_0 = Function(FunctionSpace(mesh, "DG", degree), name="rho_0")
    return rho_0.interpolate(expr)


def core_flux():
    """Prescribed d(psi)/dn on tag 5, carrying the unmeshed ball r < 0.5 Rc."""
    rho_core = LAYERS_KM[-1][2] / RHO_BAR
    mass = rho_core * 4 / 3 * np.pi * gen.R_INNER**3
    return (LAMBDA / (4 * np.pi)) * mass / gen.R_INNER**2


def solve_reference_potential(mesh, dtn_degree=5, degree=2, rho_0=None,
                              solver_parameters=None,
                              solver_parameters_extra=None, **kwargs):
    """Phi_0 on the parent, with the DtN treatment the coupled solve uses.

    Returns `(psi, rho_0)`.  `psi` is minus the Newtonian potential, so
    `g_0 = grad(psi)`.
    """
    rho_0 = density_field(mesh) if rho_0 is None else rho_0
    psi = Function(FunctionSpace(mesh, "CG", degree), name="Phi_0")
    GravitySolver(
        psi, rho_0,
        bcs={gen.SURF_OUTER: {"dtn": SphericalDtN(L=dtn_degree)},
             gen.SURF_INNER: {"dtn": SphericalDtN(L=dtn_degree),
                              "flux": core_flux()}},
        gravitational_constant=LAMBDA / (4 * np.pi),
        solver_parameters=solver_parameters,
        solver_parameters_extra=solver_parameters_extra, **kwargs).solve()
    return psi, rho_0


def enclosed_mass_ufl(r):
    """M(<r), non-dimensional, as UFL: piecewise cubic in r.

    Exact for the layered rho_0, including the unmeshed core, so it is the
    closed form the computed g_0 should reproduce.
    """
    layers = sorted(layers_nondim(), key=lambda t: t[1])  # ascending r_inner
    m = Constant(0.0)
    acc = 0.0
    expr = Constant(0.0)
    for r_out, r_in, rho in layers:
        below = acc
        acc += 4 / 3 * np.pi * rho * (r_out**3 - r_in**3)
        here = Constant(below) + Constant(4 / 3 * np.pi * rho) * (
            r**3 - Constant(r_in**3))
        expr = conditional(r < r_out, conditional(r >= r_in, here, expr), expr)
    return conditional(r >= layers[-1][0], Constant(acc), expr) + m


def gravity_exact_ufl(r):
    """|g_0|(r) = (Lambda/4pi) M(<r) / r^2, non-dimensional, as UFL."""
    return (LAMBDA / (4 * np.pi)) * enclosed_mass_ufl(r) / r**2


def volume_gravity_error(psi, cell_tag, quad_degree=6):
    r"""Relative L2 error of g_0 over a cell group, against the closed form.

    **This, and not the surface average, is the quantity the momentum equation
    is sensitive to.**  The buoyancy term integrates `rho_0 grad(psi)` over
    volume; it never evaluates a surface average of `|grad psi|` at a density
    jump, which is the worst-conditioned functional of the solution available
    and is exactly what G-1 asks for.  Reporting both separates "the field is
    inaccurate" from "the estimator is".

    Returns `(relative_L2, relative_L2_radial_component)`.
    """
    mesh = psi.function_space().mesh()
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    rhat = X / r
    g_exact = -gravity_exact_ufl(r) * rhat   # inward
    g = grad(psi)
    dm = dx(cell_tag, domain=mesh, degree=quad_degree)
    num = assemble(inner(g - g_exact, g - g_exact) * dm)
    den = assemble(inner(g_exact, g_exact) * dm)
    d_r = inner(g, rhat) - inner(g_exact, rhat)
    num_r = assemble(d_r**2 * dm)
    return np.sqrt(num / den), np.sqrt(num_r / den)


def surface_gravity(psi, tag, interior, quad_degree=12):
    """Surface averages of g_0 = grad(psi) on a tagged sphere.

    Returns `(mean_radial, rms_tangential, area)` in non-dimensional units,
    with `mean_radial` negative because gravity points inward.  A surface
    average rather than a point evaluation: grad(psi) of a CG field is
    discontinuous across cells and a density interface is exactly where its
    cell-to-cell jump is largest, so point values there are the worst possible
    estimator of a quantity that is in fact continuous.

    The radial direction is X/r rather than `FacetNormal`, deliberately.  On an
    *interior* facet `avg(inner(g, n))` is identically zero because
    `n('+') = -n('-')`, and picking a side with `('+')` is unsafe: restriction
    sides are consistent only by gmsh's cell ordering.  X/r costs an explicit
    quadrature degree, because UFL's estimate for the 1/r is 40 -- a rule that
    is both slow and no more accurate, since r is very nearly constant on a
    facet of a curved sphere.

    **The degree matters, and 6 is an unlucky choice.**  Measured at
    `--coarse`, the relative error of this estimator against the closed form is
    identical to four digits for degrees 4, 6, 8, 12, 16 and 24 at four of the
    five interfaces -- and at r = 5701 km reads 6.6e-04, 1.8e-02, 9.2e-04,
    7.8e-04, 1.1e-03, 1.2e-03.  Degree 6 alone is off by an order.  That
    interface bounds a shell thinner than one lateral cell, so its facets carry
    the flattened tetrahedra A2's subdivision mechanism produces, and a rule
    that happens to sample them badly is not caught by any smoothness argument.
    The default is 12; the wobble at the 1e-3 level across 8..24 is the honest
    resolution limit of the estimator there, not a rule that has converged.
    """
    from firedrake import avg, dS  # noqa: PLC0415

    mesh = psi.function_space().mesh()
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    rhat = X / r
    g = grad(psi)
    g_r = inner(g, rhat)
    g_t = g - g_r * rhat
    if interior:
        dm = dS(tag, domain=mesh, degree=quad_degree)
        area = assemble(avg(1) * dm)
        mean_r = assemble(avg(g_r) * dm) / area
        rms_t = np.sqrt(assemble(avg(inner(g_t, g_t)) * dm) / area)
    else:
        dm = ds(tag, domain=mesh, degree=quad_degree)
        area = assemble(1 * dm)
        mean_r = assemble(g_r * dm) / area
        rms_t = np.sqrt(assemble(inner(g_t, g_t) * dm) / area)
    return mean_r, rms_t, area


def radiality(psi, cell_tag):
    """max |g - (g.rhat)rhat| / max |g| over a cell group.

    G-2 of §2.3.  Computed from an L-infinity proxy on a DG0 projection, so
    that it is a genuine per-cell maximum rather than a quadrature sample.
    """
    from firedrake import FunctionSpace as FS  # noqa: PLC0415

    mesh = psi.function_space().mesh()
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    rhat = X / r
    g = grad(psi)
    g_t = g - inner(g, rhat) * rhat
    V = FS(mesh, "DG", 0)
    tang = Function(V).interpolate(sqrt(inner(g_t, g_t)))
    total = Function(V).interpolate(sqrt(inner(g, g)))
    cells = mesh.cell_subset(cell_tag).indices
    comm = mesh.comm
    owned = mesh.cell_set.size
    cells = cells[cells < owned]
    t_max = comm.allreduce(float(tang.dat.data_ro[cells].max()) if cells.size
                           else 0.0, op=max)
    g_max = comm.allreduce(float(total.dat.data_ro[cells].max()) if cells.size
                           else 0.0, op=max)
    return t_max / g_max, t_max, g_max
