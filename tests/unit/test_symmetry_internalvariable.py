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


def test_internal_variable_symmetry(mesh):
    """Test symmetry of discretised Stokes matrix where expected

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
