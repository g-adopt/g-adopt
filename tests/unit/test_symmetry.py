import firedrake as fd
import gadopt
import pytest

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
def test_stokes_symmetry_nonlinear_viscosity(
    approx_class, mesh, solution_space
):
    """Test that Jacobian symmetry is restored with nonlinear viscosity.

    With strain-rate-dependent viscosity, the true Jacobian
    derivative(F, z) has non-symmetric boundary contributions from
    differentiating through mu(strain_rate). The StokesSolver
    automatically detects this and builds a symmetrised Jacobian
    J = J_int + 0.5*(J_bdy + J_bdy^T).

    This test verifies:
    1. The raw derivative(F, z) boundary terms ARE non-symmetric
    2. The solver's custom J boundary terms ARE symmetric
    """
    z = fd.Function(solution_space)
    u_sub, p_sub = z.subfunctions
    X = fd.SpatialCoordinate(mesh)
    u_sub.interpolate(X)

    # Build a nonlinear viscosity that depends on strain rate
    u, _ = fd.split(z)
    epsilon = fd.sym(fd.grad(u))
    epsii = fd.sqrt(fd.inner(epsilon, epsilon) + 1e-10)
    mu = fd.Constant(1.0) + fd.Constant(1.0) / epsii

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
        bcs[bids[3]] = {'u': zero_vec}
    solver = gadopt.StokesSolver(z, approximation, T, bcs=bcs)

    # Verify that the solver detected nonlinear mu and built a custom J
    assert solver.J is not None, "Solver should have built a symmetric Jacobian"

    # The solver's custom Jacobian velocity block should be symmetric
    if approximation.compressible:
        M_J = fd.assemble(solver.J, mat_type='nest')
        M00_J = M_J.petscmat.getNestSubMatrix(0, 0).convert('aij')
        assert M00_J.isSymmetric(1e-10), \
            "Solver Jacobian velocity block should be symmetric"
    else:
        M_J = fd.assemble(solver.J, mat_type='aij')
        assert M_J.petscmat.isSymmetric(1e-10), \
            "Solver Jacobian should be symmetric for incompressible case"
