import firedrake as fd
import numpy as np
import gadopt
import ufl


def test_slope_limiters():
    mesh1d = fd.UnitIntervalMesh(10)
    meshes = [
        fd.UnitSquareMesh(10, 10, quadrilateral=True),
        fd.UnitSquareMesh(10, 10, quadrilateral=False),
        fd.ExtrudedMesh(mesh1d, 10)
    ]
    for mesh in meshes:
        x, y = fd.SpatialCoordinate(mesh)
        elt = fd.FiniteElement("DG", mesh.ufl_cell(), 1, variant='equispaced')
        P1DG = fd.FunctionSpace(mesh, elt)
        limiter = gadopt.VertexBasedP1DGLimiter(P1DG)

        # test that linear function is not flattened at boundaries (fails with firedrake's VertexBasedLimiter)
        u = fd.interpolate(x, P1DG)
        v = u.copy(deepcopy=True)
        limiter.apply(u)
        np.testing.assert_allclose(u.dat.data, v.dat.data)

        # same thing in y-direction to test bottom/top boundaries for extruded case
        u = fd.interpolate(y, P1DG)
        v = u.copy(deepcopy=True)
        limiter.apply(u)
        np.testing.assert_allclose(u.dat.data, v.dat.data)

        # test hat function
        u = fd.interpolate(0.5-abs(x-0.5), P1DG)
        vol0 = fd.assemble(u*fd.dx)
        np.testing.assert_allclose(vol0, 0.25)
        np.testing.assert_allclose(u.dat.data[:].max(), 0.5)
        limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
        limiter.apply(u)
        vol1 = fd.assemble(u*fd.dx)
        # volume should be the same
        np.testing.assert_allclose(vol1, 0.25)
        # but cells with a x=0.5 vertex, should be limited to maximum cell average
        # adjacent to that vertex
        if mesh.ufl_cell() == ufl.triangle:
            # maximum from triangle with two x=0.5 vertices
            np.testing.assert_allclose(u.dat.data[:].max(), (2*0.5+1*0.4)/3)
        else:
            np.testing.assert_allclose(u.dat.data[:].max(), 0.45)

        # repeat for vector functionspace
        P1DG = fd.VectorFunctionSpace(mesh, elt)
        limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
        u = fd.interpolate(fd.as_vector((0.5-abs(x-0.5), 0)), P1DG)
        vol0 = fd.assemble(u[0]*fd.dx)
        np.testing.assert_allclose(vol0, 0.25)
        np.testing.assert_allclose(u.dat.data[:].max(), 0.5)
        limiter.apply(u)
        vol1 = fd.assemble(u[0]*fd.dx)
        # volume should be the same
        np.testing.assert_allclose(vol1, 0.25)
        # but cells with a x=0.5 vertex, should be limited to maximum cell average
        # adjacent to that vertex
        if mesh.ufl_cell() == ufl.triangle:
            # maximum from triangle with two x=0.5 vertices
            np.testing.assert_allclose(u.dat.data[:].max(), (2*0.5+1*0.4)/3)
        else:
            np.testing.assert_allclose(u.dat.data[:].max(), 0.45)

def test_least_squares_limiter():
    mesh1d = fd.UnitIntervalMesh(1)
    mesh = fd.ExtrudedMesh(mesh1d, 1)
    mesh1d = fd.CircleManifoldMesh(8, 1.22, degree=2)
    mesh = fd.ExtrudedMesh(mesh1d, 2, extrusion_type='radial')
    V = gadopt.utility.get_functionspace(mesh, "DG", 1, "DG", 1)
    u = fd.Function(V)
    u0 = fd.Function(V)
    x, y = fd.SpatialCoordinate(mesh)
    P0 = fd.FunctionSpace(mesh, "DQ", 0)
    up0 = fd.interpolate(x, P0)
    u.interpolate(x)
    u0.assign(u)
    limiter = gadopt.VertexBasedLeastSquaresLimiter(V)
    limiter.apply(u)
    print(u0.dat.data[:])
    print(u.dat.data[:])
