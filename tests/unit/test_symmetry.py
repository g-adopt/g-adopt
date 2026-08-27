import firedrake as fd
import gadopt
import pytest
import ufl
from gadopt.equations import Equation, interior_penalty_factor
from gadopt.momentum_equation import viscosity_term


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

    # The weak boundary residual is symmetrised directly, so no custom Jacobian
    # is built and derivative(F, z) is the true (symmetric) Jacobian.
    assert solver.J is None

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


@pytest.mark.parametrize("mesh_key", ["2D-tri", "3D-tet", "2D-cylinder"])
def test_viscosity_term_un_variational_structure(mesh_key):
    """The weak "un" boundary residual is the first variation of a functional.

    Symmetry tests cannot see a sign or constant error that is itself symmetric.
    This pins the boundary residual to an independently written boundary
    functional E_bdy, so that any such slip is caught. Incompressible "un" only;
    a smooth polynomial viscosity keeps entries O(1) for a tight tolerance.
    """
    mesh = meshes[mesh_key]
    mesh.cartesian = "cylinder" not in mesh_key

    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    u = fd.Function(V)
    X = fd.SpatialCoordinate(mesh)
    u.interpolate(X)
    epsilon = fd.sym(fd.grad(u))
    mu = fd.Constant(1.0) + fd.inner(epsilon, epsilon)

    approximation = gadopt.BoussinesqApproximation(1, mu=mu)
    bids = list(gadopt.get_boundary_ids(mesh))
    bcs = {bid: {'un': 0} for bid in bids}

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
    F_bdy = ufl.Form(
        [i for i in F.integrals() if "exterior_facet" in i.integral_type()]
    )

    # independently stated boundary functional (incompressible, g = 0):
    # E_bdy = -w (n.sigma.n) + 2 sigma_pen mu w^2, with w = n.u
    sigma = interior_penalty_factor(eq)
    sigma *= fd.FacetArea(mesh) / fd.avg(fd.CellVolume(mesh))
    w = fd.dot(eq.n, u)
    stress_u = approximation.stress(u)
    E_bdy = sum(
        (-w * fd.dot(eq.n, fd.dot(stress_u, eq.n)) + 2 * sigma * mu * w**2)
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
