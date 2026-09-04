import firedrake as fd
import gadopt
import pytest
import ufl
from gadopt.equations import Equation, interior_penalty_factor
from gadopt.momentum_equation import viscosity_term


def dev_stress_per_mu(gradient, compressible):
    r"""The deviatoric stress per $\mu$, $A(G)$, written out here in the test.

    For a gradient-like tensor $G$ this is $2\,\mathrm{sym}(G)$, minus
    $\tfrac{2}{3}\,\mathrm{tr}(G)\,I$ when the stress is compressible. Spelling
    it out in the test file rather than calling
    `approximation.stress_per_mu_from_grad` keeps a bug in the operator under
    test from hiding by appearing identically on both sides of an identity.
    """
    stress_per_mu = 2 * fd.sym(gradient)
    if compressible:
        dim = gradient.ufl_shape[0]
        stress_per_mu = stress_per_mu - 2 / 3 * fd.tr(gradient) * fd.Identity(dim)
    return stress_per_mu


def exterior_facet_form(form):
    """The exterior-facet (`ds`) part of a UFL form.

    The weak boundary conditions are the only source of `ds` integrals in the
    forms built here, so this isolates the terms under test from the volume
    term of `viscosity_term`.
    """
    return ufl.Form(
        [i for i in form.integrals() if "exterior_facet" in i.integral_type()]
    )


def generic_velocity(mesh):
    """A generic velocity field for symmetry tests.

    Unlike u = X (identity), this has n.u != 0 on every boundary and an
    anisotropic strain, so a symmetric-but-wrong penalty coefficient in the
    weak boundary terms becomes observable. At u = X the normal jump n.w is
    zero on every boundary face in this suite, which hides such errors.
    """
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    return X + fd.Constant([float(i + 1) for i in range(dim)]) + 0.3 * X[0] * X


def assert_symmetric(petscmat, rtol=1e-11):
    """Assert a PETSc matrix is symmetric relative to its own norm.

    A relative test (||A - A^T|| <= rtol ||A||) is used instead of PETSc's
    absolute isSymmetric() because a generic, strongly strained linearisation
    point can give matrix entries of very different magnitudes (e.g. O(1e7) on
    a manifold mesh), where an absolute threshold is meaningless.
    """
    transpose = petscmat.duplicate(copy=True)
    transpose.transpose()
    difference = petscmat.copy()
    difference.axpy(-1.0, transpose)
    assert difference.norm() <= rtol * petscmat.norm()


def nonlinear_mu(u, compressible):
    """A solution-dependent viscosity for symmetry tests.

    For compressible stress the viscosity must depend on the invariant that
    matches the -2/3 deviatoric stress operator; otherwise the volume Jacobian
    itself is asymmetric at a generic linearisation point and the test would be
    checking an unsatisfiable property. For incompressible stress the full
    strain invariant is the matching one.
    """
    eps = fd.sym(fd.grad(u))
    if compressible:
        dim = eps.ufl_shape[0]
        invariant = fd.inner(eps, eps - fd.tr(eps) / 3 * fd.Identity(dim))
    else:
        invariant = fd.inner(eps, eps)
    return fd.Constant(1.0) + invariant


N = 4  # resolution in all directions
mesh1d = fd.UnitIntervalMesh(N)
mesh1dcircle = fd.CircleManifoldMesh(N)
mesh2dtri = fd.UnitSquareMesh(N, N, quadrilateral=False)
mesh2dquad = fd.UnitSquareMesh(N, N, quadrilateral=True)
mesh2dcs = fd.UnitCubedSphereMesh()
mesh2dico = fd.IcosahedralSphereMesh(1)
meshes = {
    "2D-tri": mesh2dtri,
    "2D-quad": mesh2dquad,
    "2D-extruded": fd.ExtrudedMesh(mesh1d, N),
    "3D-tet": fd.UnitCubeMesh(N, N, N, hexahedral=False),
    "3D-hex": fd.UnitCubeMesh(N, N, N, hexahedral=True),
    "3D-extruded": fd.ExtrudedMesh(mesh2dquad, N),
    "3D-extruded-prism": fd.ExtrudedMesh(mesh2dtri, N),
    "2D-cylinder": fd.ExtrudedMesh(mesh1dcircle, N),
    "3D-cubed-sphere": fd.ExtrudedMesh(mesh2dcs, N),
    "3D-icosahedral-sphere": fd.ExtrudedMesh(mesh2dico, N)
}


@pytest.fixture(scope="module", params=meshes.items(), ids=meshes.keys())
def mesh(request):
    id, mesh = request.param
    mesh.cartesian = not any(x in id for x in ['cylinder', 'sphere'])
    return mesh


@pytest.fixture(scope="module",
                params=[gadopt.BoussinesqApproximation,
                        gadopt.ExtendedBoussinesqApproximation,
                        gadopt.TruncatedAnelasticLiquidApproximation,
                        gadopt.AnelasticLiquidApproximation])
def approximation(request):
    Ra = 1
    Di = 1
    if request.param is gadopt.BoussinesqApproximation:
        return request.param(Ra)
    else:
        return request.param(Ra, Di)


@pytest.fixture(scope="module", params=["TaylorHood",])
def solution_space(request, mesh):
    # at the moment only P2-P1 is supported
    # would like to test discontinuous velocity, but
    # that requires the pressure gradient term to handle that
    match request.param:
        case "TaylorHood":
            V = fd.VectorFunctionSpace(mesh, "CG", 2)
            W = fd.FunctionSpace(mesh, "CG", 1)
            return V * W
        case _:
            raise ValueError("Unknown discretisation type")


def test_stokes_symmetry(approximation, mesh, solution_space):
    """Test symmetry of discretised Stokes matrix where expected

    In particular, tests symmetry of weak bc terms."""
    z = fd.Function(solution_space)
    u, p = z.subfunctions
    # use a velocity that's not divergence free, to test symmetry of div(u) terms:
    X = fd.SpatialCoordinate(mesh)
    u.interpolate(X)

    T = fd.Function(solution_space.sub(1))
    boundary = gadopt.get_boundary_ids(mesh)
    bids = list(boundary)
    bcs = {bids[0]: {'un': 0}, bids[1]: {'normal_stress': 0}}
    # cylindrical/spherical meshes only have 2 boundaries
    # if we have more, let's test some more bc types
    if len(bids) > 2:
        dim = mesh.geometric_dimension
        zero_vec = fd.Constant([0] * dim)
        bcs[bids[2]] = {'stress': zero_vec}
        # note that we are only testing the weak bc terms here
        # weak "u" is not actually supported at the moment
        # (but will need to be for future element pairs)
        # at the moment type "u" is convert to a strong DirichletBC()
        bcs[bids[3]] = {'u': zero_vec}
    solver = gadopt.StokesSolver(z, approximation, T, bcs=bcs)

    if approximation.compressible:
        # only the velocity block will be symmetric
        M = fd.assemble(fd.derivative(solver.F, z), mat_type='nest')
        # the velocity block is assembled as type 'baij' for which .isSymmetric()
        # appears to not work (always returns False); so convert to type 'aij'
        M00 = M.petscmat.getNestSubMatrix(0, 0).convert('aij')
        assert M00.isSymmetric(1e-13)
    else:
        # test symmetry of entire matrix
        M = fd.assemble(fd.derivative(solver.F, z), mat_type='aij')
        assert M.petscmat.isSymmetric(1e-13)


@pytest.mark.parametrize("approx_class", [
    gadopt.BoussinesqApproximation,
    gadopt.TruncatedAnelasticLiquidApproximation,
])
def test_stokes_symmetry_nonlinear_viscosity(approx_class, mesh, solution_space):
    """Test that the true Jacobian is symmetric with nonlinear viscosity.

    With a strain-rate-dependent viscosity the weak (SIPG) boundary terms use
    the tangent stress and a penalty-derivative term, so the boundary residual
    is the exact first variation of a boundary functional. The true Jacobian
    derivative(F, z) is therefore symmetric by construction: no custom Jacobian
    is built (solver.J is None), and the raw derivative is what we test.

    The weak "un" branch is exercised here; the weak "u" branch is converted to
    a strong DirichletBC by StokesSolver, so it is tested at the Equation level
    in test_viscosity_term_weak_u_symmetry instead. The generic linearisation
    point makes the penalty-derivative coefficient observable at this level too.
    """
    compressible = approx_class is not gadopt.BoussinesqApproximation

    z = fd.Function(solution_space)
    u_sub, p_sub = z.subfunctions
    u_sub.interpolate(generic_velocity(mesh))

    u, _ = fd.split(z)
    mu = nonlinear_mu(u, compressible)

    Ra = 1
    if approx_class is gadopt.BoussinesqApproximation:
        approximation = approx_class(Ra, mu=mu)
    else:
        approximation = approx_class(Ra, Di=1, mu=mu)

    T = fd.Function(solution_space.sub(1))
    boundary = gadopt.get_boundary_ids(mesh)
    bids = list(boundary)
    bcs = {bids[0]: {'un': 0}, bids[1]: {'normal_stress': 0}}
    if len(bids) > 2:
        dim = mesh.geometric_dimension
        zero_vec = fd.Constant([0] * dim)
        bcs[bids[2]] = {'stress': zero_vec}
        # weak "u" is converted to a strong DirichletBC here, so this does not
        # exercise the weak "u" SIPG branch (see test_viscosity_term_weak_u_symmetry)
        bcs[bids[3]] = {'u': zero_vec}
    solver = gadopt.StokesSolver(z, approximation, T, bcs=bcs)

    # The residual is symmetrised, so no custom Jacobian is built.
    assert solver.J is None

    # The raw derivative(F, z) is the true Jacobian and must be symmetric.
    if approximation.compressible:
        M = fd.assemble(fd.derivative(solver.F, z), mat_type='nest')
        assert_symmetric(M.petscmat.getNestSubMatrix(0, 0).convert('aij'))
    else:
        M = fd.assemble(fd.derivative(solver.F, z), mat_type='aij')
        assert_symmetric(M.petscmat)


@pytest.mark.parametrize("approx_class", [
    gadopt.BoussinesqApproximation,
    gadopt.TruncatedAnelasticLiquidApproximation,
])
def test_viscosity_term_weak_u_symmetry(approx_class, mesh):
    """Symmetry of the weak "u" SIPG branch for nonlinear viscosity.

    StokesSolver converts a "u" boundary condition to a strong DirichletBC, so
    the weak "u" branch of viscosity_term is never reached from the solver. Here
    we drive viscosity_term directly on a velocity-only space so that "u" stays
    weak, and check that its true Jacobian is symmetric. A generic linearisation
    point (n.w != 0, anisotropic strain) is required so that the "u" penalty
    coefficient is actually exercised.
    """
    compressible = approx_class is not gadopt.BoussinesqApproximation

    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    u = fd.Function(V)
    u.interpolate(generic_velocity(mesh))
    mu = nonlinear_mu(u, compressible)

    if approx_class is gadopt.BoussinesqApproximation:
        approximation = approx_class(1, mu=mu)
    else:
        approximation = approx_class(1, Di=1, mu=mu)

    dim = mesh.geometric_dimension
    zero_vec = fd.Constant([0] * dim)
    bids = list(gadopt.get_boundary_ids(mesh))
    # exercise both the weak "u" and weak "un" branches
    bcs = {bids[0]: {'u': zero_vec}, bids[1]: {'un': 0}}

    eq = Equation(
        fd.TestFunction(V),
        V,
        viscosity_term,
        eq_attrs={"stress": approximation.stress(u)},
        approximation=approximation,
        bcs=bcs,
        quad_degree=6,
    )
    F = eq.residual(u)
    M = fd.assemble(fd.derivative(F, u), mat_type='aij')
    assert_symmetric(M.petscmat)


@pytest.mark.parametrize("compressible", [False, True])
@pytest.mark.parametrize("bc_kind", ["u", "un"])
@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet", "2D-cylinder"])
def test_viscosity_term_variational_structure(mesh_key, bc_kind, compressible):
    r"""Pin the weak boundary residual to the first variation of its functional.

    Symmetry cannot see an error that keeps the residual symmetric: a mis-scaled
    penalty, or a wrong constant in a term that is still the first variation of
    some functional, leaves the Jacobian symmetric. Being consistent it also
    keeps the optimal convergence order, so the solver-output tests do not see it
    either. The only property that pins such an error is that the residual is the
    first variation of the boundary functional the code documents,

    $$ E_{bdy} = \int_{\partial\Omega} \left[ -w \cdot \sigma(u) n
       + \sigma_{pen}\,\mu\,\langle G, A(G) \rangle \right] ds, $$

    with $G = n \otimes w$, $A$ the deviatoric stress per $\mu$
    ($A(G) = 2\,\mathrm{sym}(G)$ for incompressible, minus
    $\tfrac{2}{3}\mathrm{tr}(G) I$ for compressible), and $w = u - u_D$ for weak
    "u" or $w = (n \cdot u - u_n) n$ for weak "un". A symmetric wrong sign or
    constant breaks $F_{bdy} = \delta E_{bdy}$. Both branches and both
    compressibilities are covered, with inhomogeneous boundary data so the flux
    and penalty terms are exercised away from the trivial $w = 0$ point that
    would hide a wrong constant.

    The flux part of E_bdy reuses approximation.stress, so a bug inside stress
    itself sits on both sides of the identity and is not caught here; the MMS
    tests in test_weak_bc_solution.py pin stress absolutely against a manufactured
    field.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = "cylinder" not in mesh_key
    dim = mesh.geometric_dimension

    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    u = fd.Function(V).interpolate(generic_velocity(mesh))
    mu = nonlinear_mu(u, compressible)

    if compressible:
        approximation = gadopt.TruncatedAnelasticLiquidApproximation(1, Di=1, mu=mu)
    else:
        approximation = gadopt.BoussinesqApproximation(1, mu=mu)

    bids = list(gadopt.get_boundary_ids(mesh))
    # Nonzero boundary data keeps the jump w = u - u_D (or n.u - un) away from
    # zero, so a wrong coefficient in a term proportional to w is observable.
    u_D = fd.Constant([0.1 * (i + 1) for i in range(dim)])
    un = 0.2
    bcs = {bid: ({"u": u_D} if bc_kind == "u" else {"un": un}) for bid in bids}

    eq = Equation(
        fd.TestFunction(V),
        V,
        viscosity_term,
        eq_attrs={"stress": approximation.stress(u)},
        approximation=approximation,
        bcs=bcs,
        quad_degree=8,
    )
    F = eq.residual(u)
    F_bdy = exterior_facet_form(F)

    sigma = interior_penalty_factor(eq)
    sigma *= fd.FacetArea(mesh) / fd.avg(fd.CellVolume(mesh))
    n = eq.n
    stress_u = approximation.stress(u)
    # For "un" the jump keeps only its normal component; both branches then share
    # the same functional with G = outer(n, w).
    w = (u - u_D) if bc_kind == "u" else (fd.dot(n, u) - un) * n
    G = fd.outer(n, w)
    E_bdy = sum(
        (-fd.dot(w, fd.dot(stress_u, n))
         + sigma * mu * fd.inner(G, dev_stress_per_mu(G, compressible)))
        * eq.ds(bid)
        for bid in bids
    )
    dE = fd.derivative(E_bdy, u, fd.TestFunction(V))

    r = fd.assemble(F_bdy - dE)
    ref = fd.assemble(dE)
    assert r.dat.norm <= 1e-12 * ref.dat.norm


def test_internal_variable_symmetry(mesh):
    """Test symmetry of discretised (viscoelastic) Stokes matrix where expected

    In particular, tests symmetry of weak bc terms."""
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    u = fd.Function(V)
    m = fd.Function(S)
    # use a velocity that's not divergence free, to test symmetry of div(u) terms:
    X = fd.SpatialCoordinate(mesh)
    u.interpolate(X)
    density = fd.Function(DG0).assign(1)
    approximation = gadopt.MaxwellApproximation(
        bulk_modulus=1,
        viscosity=1,
        shear_modulus=1,
        B_mu=1.27,
        density=density)
    boundary = gadopt.get_boundary_ids(mesh)
    bids = list(vars(boundary).values())
    bcs = {bids[0]: {'un': 0}, bids[1]: {'free_surface': {}}}
    # cylindrical/spherical meshes only have 2 boundaries
    # if we have more, let's test some more bc types
    if len(bids) > 2:
        dim = mesh.geometric_dimension
        zero_vec = fd.Constant([0] * dim)
        bcs[bids[2]] = {'stress': zero_vec}
        # note that we are only testing the weak bc terms here
        # weak "u" is not actually supported at the moment
        # (but will need to be for future element pairs)
        # at the moment type "u" is convert to a strong DirichletBC()
        bcs[bids[3]] = {'u': zero_vec}
    solver = gadopt.InternalVariableSolver(u, approximation, dt=1, internal_variables=m, bcs=bcs)

    M = fd.assemble(fd.derivative(solver.F, u), mat_type='aij')
    assert M.petscmat.isSymmetric(1e-13)


def penalty_coefficient(eq):
    r"""The SIPG penalty coefficient $\sigma_{pen}$ used by `viscosity_term`.

    `interior_penalty_factor` returns the mesh-independent safety factor; the
    weak boundary terms scale it by the facet-area-to-cell-volume ratio, which
    carries the $1/h$ of the Nitsche penalty. Written out here so that a change
    to the scaling in the code under test is visible.
    """
    sigma = interior_penalty_factor(eq)
    return sigma * fd.FacetArea(eq.mesh) / fd.avg(fd.CellVolume(eq.mesh))


def explicit_weak_boundary_form(
    eq,
    trial,
    bc_id,
    boundary_value,
    *,
    bc_kind,
    stress,
    tangent,
    mu_penalty,
    compressible,
    bulk=None,
    dmu=None,
):
    r"""A weak velocity boundary residual, written out term by term.

    This is the reference the code under test must reproduce. The jump is
    $w = u - u_D$ for a weak "u" condition and $w = (n \cdot u - u_n) n$ for a
    weak "un" one. With $G = n \otimes w$ and $A$ the deviatoric stress per
    $\mu$, the contributions are

      * the penalty $2\sigma_{pen}\,\langle n \otimes \phi,\ \mu A(G) \rangle$,
      * the symmetrising term $-w \cdot (T n)$ with $T$ the tangent stress,
      * the flux, $-\langle n \otimes \phi,\ \sigma(u) \rangle$ for "u" and its
        normal component $-(n \cdot \phi)\,(n \cdot \sigma(u)\, n)$ for "un",
      * for a stress with a bulk part, the matching pair of bulk penalty and
        bulk symmetrising terms with coefficient
        `bulk_shear_ratio * bulk_modulus`, both driven by
        $\mathrm{tr}(G) = n \cdot w$,
      * for a solution-dependent viscosity, the derivative of the penalty
        through $\mu$.

    Every coefficient is supplied by the caller from raw approximation
    attributes, so nothing is taken from the operator under test.

    Args:
      eq: the `Equation` supplying the measures, the facet normal and the test
        function.
      trial: the field the boundary condition constrains.
      bc_id: the boundary identifier the condition applies to.
      boundary_value: the prescribed velocity for "u", or its prescribed normal
        component for "un".
      bc_kind: "u" or "un".
      stress: the full stress $\sigma(u)$ entering the flux term.
      tangent: the tangent stress $D\sigma(u)[\phi]$ entering the symmetrising
        term. For a stress with a bulk part this is the deviatoric tangent
        only; the bulk part is added through `bulk`.
      mu_penalty: the shear coefficient multiplying the deviatoric penalty.
      compressible: whether the deviatoric stress carries the $-2/3$ trace term.
      bulk: the bulk coefficient, or None for a purely deviatoric stress.
      dmu: the directional derivative of $\mu$ in the direction of the test
        function, or None for a solution-independent viscosity.

    Returns:
      A UFL form for the weak boundary residual on `bc_id`.
    """
    n = eq.n
    dim = eq.mesh.geometric_dimension
    sigma = penalty_coefficient(eq)
    ds = eq.ds(bc_id)

    if bc_kind == "u":
        w = trial - boundary_value
    else:
        w = (fd.dot(n, trial) - boundary_value) * n
    # Trace of the jump tensor, which is what a bulk modulus responds to.
    normal_jump = fd.dot(n, w)
    G = fd.outer(n, w)
    A_G = dev_stress_per_mu(G, compressible)

    # Penalty on the deviatoric part of the jump.
    F = 2 * sigma * fd.inner(fd.outer(n, eq.test), mu_penalty * A_G) * ds
    # Symmetrising term: the transpose of the flux integration by parts.
    F -= fd.dot(w, fd.dot(tangent, n)) * ds
    # Flux term. The "un" condition leaves the tangential traction free, so only
    # the normal component of the traction is removed there.
    if bc_kind == "u":
        F -= fd.inner(fd.outer(n, eq.test), stress) * ds
    else:
        F -= fd.dot(n, eq.test) * fd.dot(n, fd.dot(stress, n)) * ds

    if bulk is not None:
        # Volumetric jump tensor: the bulk stress responds to tr(G) = n . w.
        bulk_jump = fd.Identity(dim) * normal_jump
        F += 2 * sigma * fd.inner(fd.outer(n, eq.test), bulk * bulk_jump) * ds
        F -= fd.inner(bulk * fd.nabla_grad(eq.test), bulk_jump) * ds

    if dmu is not None:
        # First variation of the penalty functional through mu itself.
        F += sigma * dmu * fd.inner(G, A_G) * ds

    return F


def assert_forms_agree(form, reference, solution, rtol=1e-13):
    """Assert two residual forms have the same value and the same Jacobian.

    Comparing the assembled residual alone would miss a difference that is
    zero at the current linearisation point, and comparing the Jacobian alone
    would miss a difference that is constant in the solution, so both are
    checked. Tolerances are relative to the norm of the reference.
    """
    residual = fd.assemble(form - reference)
    residual_ref = fd.assemble(reference)
    assert residual.dat.norm <= rtol * residual_ref.dat.norm

    jacobian = fd.assemble(
        fd.derivative(form - reference, solution), mat_type="aij"
    ).petscmat
    jacobian_ref = fd.assemble(
        fd.derivative(reference, solution), mat_type="aij"
    ).petscmat
    assert jacobian.norm() <= rtol * jacobian_ref.norm()


def raw_deviatoric_strain(u):
    r"""$\mathrm{dev}(\mathrm{sym}(\nabla u))$, written out from `u` alone.

    The internal-variable formulation splits the strain into this deviatoric
    part and the volumetric part `div(u)`. The trace is removed with a factor
    1/3 in every dimension, matching the 3D convention the viscoelastic
    benchmarks are built on.
    """
    strain = fd.sym(fd.grad(u))
    return strain - fd.tr(strain) / 3 * fd.Identity(len(u))


def raw_maxwell_times(approximation):
    """Maxwell relaxation times, rebuilt from raw viscosity and shear modulus."""
    return [
        eta / mu for eta, mu in zip(approximation.viscosity, approximation.shear_modulus)
    ]


def raw_effective_viscosity(approximation, dt):
    r"""$\eta_{eff} = \sum_i \eta_i / (\tau_i + \Delta t)$ from raw attributes.

    This is the shear coefficient the internal-variable solvers put into
    `approximation.mu`, and therefore the coefficient the weak boundary penalty
    must carry.
    """
    return sum(
        eta / (tau + dt)
        for eta, tau in zip(approximation.viscosity, raw_maxwell_times(approximation))
    )


def raw_internal_variables_update(approximation, u, internal_variables, dt):
    r"""Backward-Euler update of the internal variables, from raw attributes.

    Each internal variable relaxes towards the deviatoric strain with its own
    Maxwell time, $\dot m_i = (\mathrm{dev}\,\varepsilon(u) - m_i)/\tau_i$, so
    one implicit step gives
    $m_i^{new} = (m_i + \Delta t\,\mathrm{dev}\,\varepsilon(u)/\tau_i)
    / (1 + \Delta t/\tau_i)$.
    """
    dev_strain = raw_deviatoric_strain(u)
    return [
        (m + dt / tau * dev_strain) / (1 + dt / tau)
        for m, tau in zip(internal_variables, raw_maxwell_times(approximation))
    ]


def raw_internal_variable_stress(approximation, u, internal_variables):
    r"""The internal-variable stress, written out from raw attributes.

    $$ \sigma = \kappa_r\,\kappa\,(\nabla \cdot u)\,I
       + 2\mu_0\,\mathrm{dev}\,\varepsilon(u)
       - \sum_i 2\mu_i m_i, $$

    with $\kappa$ the bulk modulus, $\kappa_r$ the bulk-to-shear ratio and
    $\mu_0 = \sum_i \mu_i$ the unrelaxed (elastic) shear modulus.
    """
    identity = fd.Identity(len(u))
    stress = (
        approximation.bulk_shear_ratio
        * approximation.bulk_modulus
        * fd.div(u)
        * identity
    )
    stress += 2 * sum(approximation.shear_modulus) * raw_deviatoric_strain(u)
    for shear_modulus, m in zip(approximation.shear_modulus, internal_variables):
        stress -= 2 * shear_modulus * m
    return stress


def raw_nonlinear_dmu(u, direction, compressible):
    r"""Directional derivative of `nonlinear_mu` in the direction `direction`.

    `nonlinear_mu` is $1 + \langle \varepsilon, \varepsilon \rangle$ for the
    incompressible invariant and
    $1 + \langle \varepsilon, \varepsilon - \tfrac{1}{3}\mathrm{tr}
    (\varepsilon) I \rangle$ for the compressible one, so its derivative is
    $2\langle \varepsilon(\phi), \varepsilon \rangle$ and
    $2\langle \varepsilon(\phi), \varepsilon - \tfrac{1}{3}\mathrm{tr}
    (\varepsilon) I \rangle$ respectively. Written out by hand so the test does
    not lean on UFL differentiation of the same expression the code
    differentiates.
    """
    eps = fd.sym(fd.grad(u))
    eps_direction = fd.sym(fd.grad(direction))
    if compressible:
        dim = eps.ufl_shape[0]
        return 2 * fd.inner(eps_direction, eps - fd.tr(eps) / 3 * fd.Identity(dim))
    return 2 * fd.inner(eps_direction, eps)


# Boundary data for the weak "un" reference tests. A nonzero value keeps the
# normal jump w_n = n.u - un away from zero, so a wrong coefficient in any term
# proportional to w_n is observable.
WEAK_UN_VALUE = 0.3
# Time step and material constants shared by the viscoelastic reference cases.
GIA_DT = 0.25
GIA_BULK_MODULUS = 3.0
GIA_BULK_SHEAR_RATIO = 1.5


def build_stokes_weak_un_case(mesh, approx_class, compressible, nonlinear, bc_id):
    """Assemble the weak "un" form and its reference for a Stokes approximation.

    Returns the form produced by `viscosity_term` restricted to the boundary,
    the reference form written out from raw quantities, and the Function the
    Jacobians are taken with respect to.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    Z = V * W
    z = fd.Function(Z)
    z.subfunctions[0].interpolate(generic_velocity(mesh))
    u, _ = fd.split(z)

    mu = nonlinear_mu(u, compressible) if nonlinear else fd.Constant(2.0)
    kwargs = {} if approx_class is gadopt.BoussinesqApproximation else {"Di": 1}
    approximation = approx_class(1, mu=mu, **kwargs)

    T = fd.Function(W)
    solver = gadopt.StokesSolver(
        z, approximation, T, bcs={bc_id: {"un": WEAK_UN_VALUE}},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    # Reference stress and tangent, written from mu and the strain alone.
    stress = mu * dev_stress_per_mu(fd.grad(u), compressible)
    tangent = mu * dev_stress_per_mu(fd.grad(eq.test), compressible)
    dmu = None
    if nonlinear:
        # A solution-dependent mu adds Dmu[phi] sigma/mu to the tangent, and a
        # penalty-derivative term to the residual.
        dmu = raw_nonlinear_dmu(u, eq.test, compressible)
        tangent = tangent + dmu * dev_stress_per_mu(fd.grad(u), compressible)

    reference = explicit_weak_boundary_form(
        eq, u, bc_id, WEAK_UN_VALUE, bc_kind="un",
        stress=stress, tangent=tangent, mu_penalty=mu,
        compressible=compressible, dmu=dmu,
    )
    return form, reference, z


def build_internal_variable_weak_un_case(mesh, shear_moduli, viscosities, bc_id):
    """Assemble the weak "un" form and its reference for `InternalVariableSolver`.

    One entry in `shear_moduli` gives Maxwell rheology, two give a Burgers
    body. The solver substitutes the backward-Euler update of the internal
    variables into the stress, so the tangent of that stress in the direction
    of the test function carries the effective viscosity, not the elastic
    shear modulus.
    """
    dim = mesh.geometric_dimension
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    X = fd.SpatialCoordinate(mesh)
    internal_variables = []
    for i in range(len(shear_moduli)):
        # A nonzero, non-isotropic history state, so that a term proportional to
        # the internal variables cannot vanish by accident.
        m = fd.Function(S).interpolate(
            0.1 * (i + 1) * fd.sym(fd.outer(X, fd.as_vector([1.0] * dim)))
        )
        internal_variables.append(m)

    approximation = gadopt.CompressibleInternalVariableApproximation(
        GIA_BULK_MODULUS,
        fd.Function(DG0).assign(1),
        list(shear_moduli),
        list(viscosities),
        bulk_shear_ratio=GIA_BULK_SHEAR_RATIO,
        B_mu=1.27,
    )
    solver = gadopt.InternalVariableSolver(
        u, approximation, dt=GIA_DT,
        internal_variables=internal_variables,
        bcs={bc_id: {"un": WEAK_UN_VALUE}},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    updated = raw_internal_variables_update(
        approximation, u, internal_variables, GIA_DT
    )
    stress = raw_internal_variable_stress(approximation, u, updated)
    eta_eff = raw_effective_viscosity(approximation, GIA_DT)
    tangent = eta_eff * dev_stress_per_mu(fd.grad(eq.test), True)

    reference = explicit_weak_boundary_form(
        eq, u, bc_id, WEAK_UN_VALUE, bc_kind="un",
        stress=stress, tangent=tangent, mu_penalty=eta_eff,
        compressible=True,
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    return form, reference, u


def build_internal_variable_weak_u_case(mesh, bc_id):
    """Assemble the weak "u" form and its reference for the internal-variable stress.

    Every `StokesSolverBase` subclass turns a "u" boundary condition into a
    strong `DirichletBC`, so this drives the `Equation` directly on a
    displacement-only space. The effective viscosity is put into
    `approximation.mu` here, which is what the solvers do before assembly, and
    the history is advanced with the same backward-Euler update
    `InternalVariableSolver` uses.

    The weak "u" branch constrains all components of the displacement, so its
    penalty and its symmetrising term act on the full jump. A stress with a
    bulk modulus therefore has to contribute a volumetric penalty here as well,
    driven by the normal part of that jump.
    """
    dim = mesh.geometric_dimension
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    X = fd.SpatialCoordinate(mesh)
    m = fd.Function(S).interpolate(
        0.1 * fd.sym(fd.outer(X, fd.as_vector([1.0] * dim)))
    )

    approximation = gadopt.CompressibleInternalVariableApproximation(
        GIA_BULK_MODULUS,
        fd.Function(DG0).assign(1),
        [2.0],
        [2.0],
        bulk_shear_ratio=GIA_BULK_SHEAR_RATIO,
        B_mu=1.27,
    )
    # The solvers put the effective viscosity into `approximation.mu` before
    # assembly. The Equation is driven directly here, so the same assignment is
    # made from the raw material parameters, wrapped as a UFL expression the way
    # the approximations wrap a bare number.
    eta_eff = raw_effective_viscosity(approximation, GIA_DT)
    approximation.mu = ufl.as_ufl(eta_eff)

    # The equation carries the approximation's own stress at its own
    # backward-Euler update; the reference below rebuilds both from raw
    # attributes.
    strain = approximation.deviatoric_strain(u)
    updated_equation = [
        (m + GIA_DT / maxwell_time * strain) / (1 + GIA_DT / maxwell_time)
        for maxwell_time in approximation.maxwell_times
    ]
    # Inhomogeneous boundary data, so the jump stays away from zero.
    u_D = fd.Constant([0.1 * (i + 1) for i in range(dim)])
    eq = Equation(
        fd.TestFunction(V),
        V,
        viscosity_term,
        eq_attrs={
            "stress": approximation.stress(u, internal_variables=updated_equation)
        },
        approximation=approximation,
        bcs={bc_id: {"u": u_D}},
        quad_degree=6,
    )
    form = exterior_facet_form(eq.residual(u))

    updated = raw_internal_variables_update(approximation, u, [m], GIA_DT)
    reference = explicit_weak_boundary_form(
        eq, u, bc_id, u_D, bc_kind="u",
        stress=raw_internal_variable_stress(approximation, u, updated),
        tangent=eta_eff * dev_stress_per_mu(fd.grad(eq.test), True),
        mu_penalty=eta_eff,
        compressible=True,
        bulk=GIA_BULK_SHEAR_RATIO * GIA_BULK_MODULUS,
    )
    return form, reference, u


def build_incompressible_maxwell_weak_un_case(mesh, bc_id):
    """Assemble the weak "un" form and its reference for `ViscoelasticStokesSolver`.

    The incremental-displacement formulation carries the previous deviatoric
    stress as data, so the stress is affine in the unknown and its tangent is
    the effective viscosity times the incompressible deviatoric shape. There is
    no bulk part.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    Z = V * W

    z = fd.Function(Z)
    z.subfunctions[0].interpolate(generic_velocity(mesh))
    u, _ = fd.split(z)
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    stress_old = fd.Function(S).interpolate(
        0.2 * fd.sym(fd.outer(X, fd.as_vector([1.0] * dim)))
    )
    displacement = fd.Function(V)

    shear_modulus, viscosity = 2.0, 2.0
    approximation = gadopt.IncompressibleMaxwellApproximation(
        fd.Function(DG0).assign(1), shear_modulus, viscosity
    )
    solver = gadopt.ViscoelasticStokesSolver(
        z, approximation, stress_old, displacement, dt=GIA_DT,
        bcs={bc_id: {"un": WEAK_UN_VALUE}},
        solver_parameters="direct",
    )
    eq = solver.equations[0]
    form = exterior_facet_form(viscosity_term(eq, u))

    # Zhong et al. (2003) incremental-displacement effective viscosity, with the
    # Maxwell time rebuilt from the raw viscosity and shear modulus.
    maxwell_time = viscosity / shear_modulus
    eta_eff = viscosity / (maxwell_time + GIA_DT / 2)
    stress = 2 * eta_eff * fd.sym(fd.grad(u)) + stress_old
    tangent = eta_eff * dev_stress_per_mu(fd.grad(eq.test), False)

    reference = explicit_weak_boundary_form(
        eq, u, bc_id, WEAK_UN_VALUE, bc_kind="un",
        stress=stress, tangent=tangent, mu_penalty=eta_eff,
        compressible=False,
    )
    return form, reference, z


# Mantle-convection cases as (class, compressible stress, solution-dependent mu).
# The compressibility of each approximation is stated here rather than read back
# from the instance, so that a class silently changing it fails the test.
WEAK_BOUNDARY_CASES = {
    "Boussinesq-linear": (gadopt.BoussinesqApproximation, False, False),
    "Boussinesq-nonlinear": (gadopt.BoussinesqApproximation, False, True),
    "EBA-linear": (gadopt.ExtendedBoussinesqApproximation, False, False),
    "EBA-nonlinear": (gadopt.ExtendedBoussinesqApproximation, False, True),
    "TALA-linear": (gadopt.TruncatedAnelasticLiquidApproximation, True, False),
    "TALA-nonlinear": (gadopt.TruncatedAnelasticLiquidApproximation, True, True),
    "ALA-linear": (gadopt.AnelasticLiquidApproximation, True, False),
    "ALA-nonlinear": (gadopt.AnelasticLiquidApproximation, True, True),
    "Maxwell": None,
    "Maxwell-weak-u": None,
    "Burgers": None,
    "IncompressibleMaxwell": None,
}


@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet"])
@pytest.mark.parametrize("case", list(WEAK_BOUNDARY_CASES))
def test_viscosity_term_weak_boundary_matches_explicit_forms(case, mesh_key):
    """Pin the weak boundary residual to explicitly written-out terms.

    `viscosity_term` builds the symmetrising term by differentiating the stress
    expression the equation carries, and the penalty from the approximation's
    stress-from-gradient helper. Both are generic, so nothing in the code says
    which coefficient ends up in front of which term. This test writes the
    contributions out with coefficients rebuilt from raw approximation
    attributes (`bulk_modulus`, `bulk_shear_ratio`, `viscosity`,
    `shear_modulus`, `dt`) and requires the assembled residual and Jacobian to
    agree.

    The reference is the same form for every approximation, so a wrong bulk
    coefficient, a dropped bulk penalty, a penalty raised from the effective
    viscosity to the elastic shear modulus, or a sign flip in the symmetrising
    term all show up. All seven shipped approximations are covered, in 2D and
    3D, with the mantle-convection ones in both a linear and a
    strain-rate-dependent viscosity variant.

    The "Maxwell-weak-u" case covers the weak "u" branch with the same
    internal-variable stress. Symmetry cannot see a bulk term missing from that
    branch, because dropping it removes a symmetric pair, so this is the only
    check on the volumetric part of the weak "u" penalty.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = True
    bc_id = list(gadopt.get_boundary_ids(mesh))[0]

    match case:
        case "Maxwell":
            form, reference, solution = build_internal_variable_weak_un_case(
                mesh, [2.0], [2.0], bc_id
            )
        case "Maxwell-weak-u":
            form, reference, solution = build_internal_variable_weak_u_case(
                mesh, bc_id
            )
        case "Burgers":
            form, reference, solution = build_internal_variable_weak_un_case(
                mesh, [2.0, 0.5], [2.0, 0.1], bc_id
            )
        case "IncompressibleMaxwell":
            form, reference, solution = build_incompressible_maxwell_weak_un_case(
                mesh, bc_id
            )
        case _:
            approx_class, compressible, nonlinear = WEAK_BOUNDARY_CASES[case]
            form, reference, solution = build_stokes_weak_un_case(
                mesh, approx_class, compressible, nonlinear, bc_id
            )

    assert_forms_agree(form, reference, solution)


def test_viscosity_term_rejects_stress_independent_of_trial():
    """A stress that does not involve the trial must be refused.

    The symmetrising term is the derivative of the stress expression with
    respect to the trial. If the caller supplies a stress built from a
    different Function, that derivative is identically zero and the weak
    boundary condition silently loses its symmetrising term: the residual stays
    consistent, so no convergence test would catch it, but the Jacobian is no
    longer symmetric. `viscosity_term` raises instead.
    """
    mesh = meshes["2D-tri"]
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    trial = fd.Function(V).interpolate(generic_velocity(mesh))
    unrelated = fd.Function(V).interpolate(generic_velocity(mesh))

    approximation = gadopt.BoussinesqApproximation(1)
    bc_id = list(gadopt.get_boundary_ids(mesh))[0]
    eq = Equation(
        fd.TestFunction(V),
        V,
        viscosity_term,
        eq_attrs={"stress": approximation.stress(unrelated)},
        approximation=approximation,
        bcs={bc_id: {"un": WEAK_UN_VALUE}},
        quad_degree=6,
    )
    with pytest.raises(ValueError, match="does not depend on"):
        eq.residual(trial)
