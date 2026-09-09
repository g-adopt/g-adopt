"""Symmetry of the assembled Stokes and viscoelastic matrices.

These tests assemble the whole matrix over the full mesh grid. The tests of the
weak boundary terms on their own live in `tests/weak_bc_stokes` and
`tests/weak_bc_gia`, where each mesh runs as its own doit step; the cost of
those is firedrake kernel compilation, which is serial inside the one process
this file runs in.
"""

import firedrake as fd
import gadopt
import pytest
from gadopt.equations import Equation
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


def taylor_hood(mesh):
    """The P2-P1 velocity-pressure mixed space on `mesh`.

    Only this pair is supported at the moment. Discontinuous velocity would be
    worth testing too, but that needs the pressure gradient term to handle it.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    return V * W


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
    match request.param:
        case "TaylorHood":
            return taylor_hood(mesh)
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


# Boundary data for the weak "un" reference tests. A nonzero value keeps the
# normal jump w_n = n.u - un away from zero, so a wrong coefficient in any term
# proportional to w_n is observable.
WEAK_UN_VALUE = 0.3


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
