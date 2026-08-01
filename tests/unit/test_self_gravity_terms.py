"""Unit tests for the self-gravity and rotational body-force terms.

These terms are cheap to assemble and almost impossible to check by eye, because
getting either sign wrong produces a residual of exactly the right magnitude
pointing exactly the wrong way. Every test below therefore pins a *direction*
against a physical statement, and where it can, the magnitude too.

The convention under test, documented on `BaseGIAApproximation`: psi is
`GravitySolver`'s potential, i.e. minus the Newtonian one, so the perturbation
gravity is +grad(psi) and both body-force terms carry a MINUS in the residual.
"""

import firedrake as fd
import pytest
from tsfc.exceptions import MismatchingDomainError

from gadopt.approximations import BaseGIAApproximation
from gadopt.equations import Equation
from gadopt.momentum_equation import (
    rotational_potential,
    rotational_potential_term,
    self_gravity_term,
)

B_MU = 2.5
RHO_0 = 3.0


def centred_square(n=8):
    """A square mesh on [-1, 1]^2, so that the coordinate origin is interior.

    `rotational_potential` is a polynomial about the origin and the mass excess
    in the sign tests sits there, so neither is well posed on a mesh whose
    origin is a corner.
    """
    mesh = fd.RectangleMesh(n, n, 1.0, 1.0, originX=-1.0, originY=-1.0)
    mesh.cartesian = True
    return mesh


def parent_and_submesh(n=4):
    """A parent mesh and the submesh covering its left half.

    The two-mesh arrangement of the coupled solver: the potential is solved on
    the whole domain and the mechanics only on the mantle. Built from a DG0
    indicator rather than a mesh file so that this test file stays standalone.
    """
    parent = fd.UnitSquareMesh(n, n)
    x, _ = fd.SpatialCoordinate(parent)
    indicator = fd.Function(fd.FunctionSpace(parent, "DG", 0))
    indicator.interpolate(fd.conditional(x < 0.5, 1.0, 0.0))

    parent = fd.RelabeledMesh(parent, [indicator], [999])
    sub = fd.Submesh(parent, parent.topological_dimension, 999)
    parent.cartesian = sub.cartesian = True
    return parent, sub


def make_equation(mesh, term, eq_attrs, **kwargs):
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    approximation = BaseGIAApproximation(RHO_0, 1, 1, B_mu=B_MU)
    eq = Equation(
        fd.TestFunction(V),
        V,
        term,
        eq_attrs=eq_attrs,
        approximation=approximation,
        **kwargs,
    )
    return V, eq


def force_along(eq, V, direction):
    """The body force the term applies, projected onto a direction field.

    The residual is the term as written, i.e. everything on the left-hand side,
    so the force appearing on the right-hand side of `A u = b` is its negative.
    Returning the force rather than the residual keeps the assertions below in
    the language of the physics.
    """
    residual = eq.residual(fd.TrialFunction(V))
    w = fd.Function(V).interpolate(direction)
    return -fd.assemble(fd.replace(residual, {eq.test: w}))


# -- self_gravity_term -----------------------------------------------------


def test_self_gravity_term_assembles():
    mesh = centred_square()
    x, y = fd.SpatialCoordinate(mesh)
    V, eq = make_equation(mesh, self_gravity_term, {"psi": -(x**2 + y**2)})

    assert fd.assemble(eq.residual(fd.TrialFunction(V))).dat.norm > 0


def test_self_gravity_term_requires_psi():
    mesh = centred_square(2)
    with pytest.raises(ValueError, match="psi"):
        make_equation(mesh, self_gravity_term, {})


def test_self_gravity_term_pulls_towards_a_mass_excess():
    """The sign test. A mass excess must attract, not repel.

    A uniform positive density rho_e centred on the origin has
    psi = -pi G rho_e r^2 up to a constant: laplacian(psi) = -4 pi G rho_e, which
    is this convention's Poisson equation, and psi is largest at the mass. Take
    psi = -r^2, whose gradient -2(x, y) points towards the origin everywhere;
    the body force rho_0 grad(psi) must point the same way.

    The magnitude is checked too, because a factor is as easy to lose as a sign:
    projected on the outward field w = (x, y), the force is exactly
    -2 B_mu rho_0 int r^2 dx.
    """
    mesh = centred_square()
    x, y = fd.SpatialCoordinate(mesh)
    r_sq = x**2 + y**2
    V, eq = make_equation(mesh, self_gravity_term, {"psi": -r_sq})

    outward = force_along(eq, V, fd.as_vector([x, y]))
    inward = force_along(eq, V, fd.as_vector([-x, -y]))

    # Attraction: negative work along the outward direction, positive inward.
    assert outward < 0, (
        f"The force along the OUTWARD direction must be negative, and it is "
        f"{outward:+.6e}. A mass excess attracts: g_1 = +grad(psi), so the residual "
        "carries -B_mu rho_0 grad(psi) . w. A positive value here means the MINUS "
        "has been dropped from self_gravity_term and gravity now repels."
    )
    assert inward > 0
    assert outward == pytest.approx(-inward)

    expected = -2 * B_MU * RHO_0 * fd.assemble(r_sq * fd.dx(mesh))
    assert outward == pytest.approx(expected, rel=1e-12)


def test_self_gravity_term_is_the_transpose_of_the_poisson_source():
    """The symmetry the whole coupled formulation rests on.

    The (u, psi) block is d/d(psi) of this term and the (psi, u) block is d/du of
    the potential source, -Lambda int rho_0 u . grad(v), scaled by
    theta_psi = B_mu / Lambda. Both must carry the *same* constant -- that is
    what makes the coupled Jacobian symmetric -- so the two forms below must
    agree rather than differ by a sign.
    """
    mesh = centred_square(4)
    x, y = fd.SpatialCoordinate(mesh)

    lam = 1.7
    theta_psi = B_MU / lam
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    P = fd.FunctionSpace(mesh, "CG", 1)
    approximation = BaseGIAApproximation(
        RHO_0, 1, 1, B_mu=B_MU, self_gravity_number=lam
    )

    psi = fd.Function(P).interpolate(fd.sin(x) * fd.cos(y))
    u = fd.Function(V).interpolate(fd.as_vector([fd.cos(x), fd.sin(y)]))

    eq = Equation(
        fd.TestFunction(V),
        V,
        self_gravity_term,
        eq_attrs={"psi": psi},
        approximation=approximation,
    )
    j_u_psi = fd.assemble(fd.replace(eq.residual(fd.TrialFunction(V)), {eq.test: u}))

    lambda_constant = approximation.self_gravity_number
    j_psi_u = theta_psi * fd.assemble(
        -lambda_constant * RHO_0 * fd.dot(u, fd.grad(psi)) * fd.dx(mesh)
    )

    assert j_u_psi == pytest.approx(j_psi_u, rel=1e-12)


# -- rotational_potential_term ---------------------------------------------


def test_rotational_potential_term_assembles():
    mesh = centred_square()
    psi_rot = rotational_potential([fd.Constant(0.1)], mesh, Omega_sq=1.6e-3)
    V, eq = make_equation(mesh, rotational_potential_term, {"psi_rot": psi_rot})

    assert fd.assemble(eq.residual(fd.TrialFunction(V))).dat.norm > 0


def test_rotational_potential_term_requires_psi_rot():
    mesh = centred_square(2)
    with pytest.raises(ValueError, match="psi_rot"):
        make_equation(mesh, rotational_potential_term, {})


def test_rotational_potential_term_spins_material_outwards():
    """The sign test. Spinning faster must throw material out, not pull it in.

    In 2-D the only rotation mode is m_3 = delta_Omega / Omega, and the extra
    centrifugal acceleration for m_3 > 0 is 2 Omega^2 m_3 r, outward. The body
    force rho_0 grad(psi_rot) reproduces it including the factor of two, which is
    checked here -- that factor is the whole content of the claim that psi_rot is
    the *negated* centrifugal potential rather than the centrifugal potential.
    """
    mesh = centred_square()
    x, y = fd.SpatialCoordinate(mesh)
    omega_sq, m_3 = 1.6e-3, 0.1

    psi_rot = rotational_potential([fd.Constant(m_3)], mesh, Omega_sq=omega_sq)
    V, eq = make_equation(mesh, rotational_potential_term, {"psi_rot": psi_rot})

    outward = force_along(eq, V, fd.as_vector([x, y]))
    assert outward > 0, (
        f"Speeding the rotation up (m_3 > 0) must throw material OUTWARD, and the "
        f"force along the outward direction is {outward:+.6e}. A negative value "
        "means either the MINUS in rotational_potential_term or the negation in "
        "rotational_potential has been dropped -- one of the two, since dropping "
        "both would leave this test passing."
    )

    expected = B_MU * fd.assemble(
        RHO_0 * 2 * omega_sq * m_3 * (x**2 + y**2) * fd.dx(mesh)
    )
    assert outward == pytest.approx(expected, rel=1e-12)

    # Slowing the rotation reverses it, and changes nothing else.
    psi_slow = rotational_potential([fd.Constant(-m_3)], mesh, Omega_sq=omega_sq)
    _, eq_slow = make_equation(mesh, rotational_potential_term, {"psi_rot": psi_slow})
    assert force_along(eq_slow, V, fd.as_vector([x, y])) == pytest.approx(-outward)


def test_rotational_potential_shape():
    """The polynomial itself: three components in 3-D, only m_3 in 2-D."""
    mesh_2d = centred_square(2)
    mesh_3d = fd.UnitCubeMesh(2, 2, 2)

    m_1, m_2, m_3 = 0.3, -0.7, 0.11
    omega_sq = 1.6e-3

    x, y = fd.SpatialCoordinate(mesh_2d)
    psi_2d = rotational_potential([fd.Constant(m_3)], mesh_2d, Omega_sq=omega_sq)
    difference = fd.assemble(
        (psi_2d - omega_sq * m_3 * (x**2 + y**2)) ** 2 * fd.dx(mesh_2d)
    )
    assert difference == pytest.approx(0.0, abs=1e-24)

    X, Y, Z = fd.SpatialCoordinate(mesh_3d)
    psi_3d = rotational_potential(
        [fd.Constant(m_1), fd.Constant(m_2), fd.Constant(m_3)],
        mesh_3d,
        Omega_sq=omega_sq,
    )
    reference = -omega_sq * (m_1 * X * Z + m_2 * Y * Z - m_3 * (X**2 + Y**2))
    difference = fd.assemble((psi_3d - reference) ** 2 * fd.dx(mesh_3d))
    assert difference == pytest.approx(0.0, abs=1e-24)

    # psi_rot never enters the Dirichlet-to-Neumann treatment because it is not
    # harmonic: laplacian(psi_rot) = +4 Omega^2 m_3, in 2-D and in 3-D alike.
    for psi_rot, mesh in ((psi_2d, mesh_2d), (psi_3d, mesh_3d)):
        laplacian = fd.assemble(fd.div(fd.grad(psi_rot)) * fd.dx(mesh))
        area = fd.assemble(fd.Constant(1.0) * fd.dx(mesh))
        assert laplacian / area == pytest.approx(4 * omega_sq * m_3, rel=1e-10)


@pytest.mark.parametrize("n_rot", [0, 2, 3])
def test_rotational_potential_rejects_the_wrong_component_count(n_rot):
    mesh = centred_square(2)
    with pytest.raises(ValueError, match="rotation"):
        rotational_potential([fd.Constant(0.1)] * n_rot, mesh)


# -- the approximation's accessors -----------------------------------------


def test_self_gravity_number_defaults_to_none():
    """`None` means "this system has no potential to couple to", and every
    approximation that predates self-gravity must keep meaning that."""
    assert BaseGIAApproximation(1, 1, 1).self_gravity_number is None

    approximation = BaseGIAApproximation(1, 1, 1, self_gravity_number=1.1116)
    assert float(approximation.self_gravity_number) == pytest.approx(1.1116)


def test_gravity_accessor_splits_background_from_perturbation():
    mesh = centred_square(4)
    x, y = fd.SpatialCoordinate(mesh)
    approximation = BaseGIAApproximation(RHO_0, 1, 1, g=9.81)

    psi = -(x**2 + y**2)
    psi_rot = rotational_potential([fd.Constant(0.1)], mesh, Omega_sq=1.6e-3)
    gravity = approximation.gravity(psi, psi_rot, mesh=mesh)

    # Background: -g in the upward direction, which is +y on a Cartesian mesh.
    area = fd.assemble(fd.Constant(1.0) * fd.dx(mesh))
    upward = fd.assemble(fd.dot(gravity.background, fd.as_vector([0.0, 1.0])) * fd.dx(mesh))
    assert upward / area == pytest.approx(-9.81)

    # Perturbation: grad(psi) + grad(psi_rot), with the SAME sign for both.
    def squared_difference(a, b):
        return fd.assemble(fd.inner(a - b, a - b) * fd.dx(mesh))

    reference = fd.grad(psi) + fd.grad(psi_rot)
    assert squared_difference(gravity.perturbation, reference) == pytest.approx(
        0.0, abs=1e-24
    )
    assert squared_difference(
        gravity.total, gravity.background + gravity.perturbation
    ) == pytest.approx(0.0, abs=1e-24)

    # Background only, when there is no potential in the system.
    background_only = approximation.gravity(mesh=mesh)
    assert squared_difference(
        background_only.total, gravity.background
    ) == pytest.approx(0.0, abs=1e-24)


def test_gravity_needs_a_mesh_or_a_potential():
    with pytest.raises(ValueError, match="mesh"):
        BaseGIAApproximation(1, 1, 1).gravity()


def test_geoid_is_positive_over_a_mass_excess():
    """The sign that reaches the science, and the only thing in the suite that
    checks it.

    A mass excess raises the geoid: sea level rises towards the load. Under this
    convention a mass excess gives psi > 0 above it, so N = +psi/g_0 comes out
    positive. Nothing else fails if this sign is inverted -- the geoid feeds
    `SL = SL_0 + dphi - du_r`, so a flipped sign inverts self-attraction and
    loading while leaving the magnitude entirely plausible. Hence the explicit
    expectation in the assertion below.
    """
    mesh = centred_square(4)
    x, y = fd.SpatialCoordinate(mesh)
    g_0 = 9.81
    approximation = BaseGIAApproximation(RHO_0, 1, 1, g=g_0)

    # psi of a positive mass excess at the origin: laplacian(psi) = -4 pi G rho
    # with rho > 0, and psi is largest where the mass is.
    psi = 1.0 - (x**2 + y**2)
    assert fd.assemble(fd.div(fd.grad(psi)) * fd.dx(mesh)) < 0  # a mass EXCESS

    geoid = approximation.geoid(psi)
    mean_geoid = fd.assemble(geoid * fd.dx(mesh)) / fd.assemble(
        fd.Constant(1.0) * fd.dx(mesh)
    )
    assert mean_geoid > 0, (
        f"The geoid over a mass excess must be POSITIVE, and it is {mean_geoid:+.6e}. "
        "N = +psi/g_0 under this convention: a mass excess has Newtonian Phi_1 < 0, "
        "psi = -Phi_1 > 0 and N = -Phi_1/g_0 > 0, so the geoid bulges towards the "
        "mass. A negative value here means geoid() has been 'corrected' to "
        "-psi/g_0, which inverts self-attraction and loading. See the derivation "
        "in BaseGIAApproximation.geoid before changing either."
    )
    assert fd.assemble((geoid - psi / g_0) ** 2 * fd.dx(mesh)) == pytest.approx(
        0.0, abs=1e-24
    )


# -- Equation's intersected measures ---------------------------------------


def test_intersect_measures_defaults_are_unchanged():
    """The default path must build exactly the measures it always did."""
    mesh = centred_square(2)
    x, y = fd.SpatialCoordinate(mesh)
    _, eq = make_equation(mesh, self_gravity_term, {"psi": x * y})

    assert eq.intersect_measures == ()
    assert eq.dx == fd.dx(domain=mesh, degree=3)
    assert eq.ds == fd.ds(domain=mesh, degree=3)
    assert eq.dS == fd.dS(domain=mesh, degree=3)


def test_intersect_measures_reaches_all_three_measures():
    """A mesh at the call site becomes that mesh's cell measure, on dx, ds and dS."""
    parent, sub = parent_and_submesh(2)
    x, y = fd.SpatialCoordinate(sub)
    _, eq = make_equation(
        sub, self_gravity_term, {"psi": x * y}, intersect_measures=parent
    )

    expected = (fd.Measure("cell", domain=parent),)
    assert eq.intersect_measures == expected
    for measure in (eq.dx, eq.ds, eq.dS):
        assert measure.intersect_measures() == expected


def test_intersect_measures_is_what_makes_the_cross_mesh_term_assemble():
    """The point of the hook: psi on the parent, w on the mantle submesh.

    This is the Phase 4 arrangement in miniature, and it is checked against an
    exact value rather than merely for not raising -- an intersected measure that
    finds nothing assembles to zero without complaint, so a term built this way
    is only as trustworthy as the number it is compared against.
    """
    parent, sub = parent_and_submesh(4)
    x_p, y_p = fd.SpatialCoordinate(parent)
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 1)).interpolate(x_p * y_p)

    V, eq = make_equation(
        sub, self_gravity_term, {"psi": psi}, intersect_measures=parent
    )
    force = force_along(eq, V, fd.as_vector([1.0, 0.0]))

    # grad(psi) = (y, x), so the x-component of the force is B_mu rho_0 int_sub y.
    _, y_s = fd.SpatialCoordinate(sub)
    expected = B_MU * RHO_0 * fd.assemble(y_s * fd.dx(sub))
    assert force == pytest.approx(expected, rel=1e-12)

    # Without the hook the same form is not assemblable at all, which is the
    # reason `Equation` needed one.
    _, plain = make_equation(sub, self_gravity_term, {"psi": psi})
    with pytest.raises(MismatchingDomainError):
        fd.assemble(plain.residual(fd.TrialFunction(V)))


@pytest.mark.parametrize("supplied", ["measure", "measure_tuple", "mesh_list"])
def test_intersect_measures_accepts_measures_and_sequences(supplied):
    mesh = centred_square(2)
    other = fd.UnitSquareMesh(2, 2)
    measure = fd.Measure("cell", domain=other)
    argument = {
        "measure": measure,
        "measure_tuple": (measure,),
        "mesh_list": [other],
    }[supplied]

    x, y = fd.SpatialCoordinate(mesh)
    _, eq = make_equation(
        mesh, self_gravity_term, {"psi": x * y}, intersect_measures=argument
    )
    assert eq.intersect_measures == (measure,)


def test_intersect_measures_refused_on_extruded_meshes():
    """`CombinedSurfaceMeasure` cannot carry the keyword, so say so rather than
    dropping it silently."""
    mesh = fd.ExtrudedMesh(fd.UnitIntervalMesh(2), 2)
    other = fd.UnitSquareMesh(2, 2)
    x, y = fd.SpatialCoordinate(mesh)

    with pytest.raises(NotImplementedError, match="extruded"):
        make_equation(mesh, self_gravity_term, {"psi": x * y}, intersect_measures=other)
