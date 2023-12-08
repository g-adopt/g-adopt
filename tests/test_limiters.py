import firedrake as fd
import numpy as np
import gadopt
import ufl
from gadopt.limiter import ExteriorFacetAverager


def test_slope_limiters():
    mesh1d = fd.UnitIntervalMesh(10)
    meshes = [
        fd.UnitSquareMesh(10, 10, quadrilateral=True),
        fd.UnitSquareMesh(10, 10, quadrilateral=False),
        fd.ExtrudedMesh(mesh1d, 10)
    ]
    meshes.append(fd.ExtrudedMesh(meshes[0], 10))
    for mesh in meshes:
        dim = mesh.geometric_dimension()
        xyz = fd.SpatialCoordinate(mesh)
        elt = fd.FiniteElement("DG", mesh.ufl_cell(), 1, variant='equispaced')
        P1DG = fd.FunctionSpace(mesh, elt)
        limiter = gadopt.VertexBasedP1DGLimiter(P1DG)

        # test that linear function is not flattened at boundaries (fails with firedrake's VertexBasedLimiter)
        u = fd.interpolate(xyz[0], P1DG)
        v = u.copy(deepcopy=True)
        limiter.apply(u)
        np.testing.assert_allclose(u.dat.data, v.dat.data)

        # same thing in vertical direction to test bottom/top boundaries for extruded case
        u = fd.interpolate(xyz[dim-1], P1DG)
        v = u.copy(deepcopy=True)
        limiter.apply(u)
        np.testing.assert_allclose(u.dat.data, v.dat.data)

        # test hat function
        u = fd.interpolate(0.5-abs(xyz[0]-0.5), P1DG)
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
        vec = [0] * dim
        vec[0] = 0.5-abs(xyz[0]-0.5)
        u = fd.interpolate(fd.as_vector(vec), P1DG)
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
    nhcells = 31
    nlayers = 16
    # assert nhcells % 2 == 1  # need odd n/o cells to ensure atan2 jump is inside
    mesh1d = fd.CircleManifoldMesh(nhcells, 1.22, degree=1)
    mesh = fd.ExtrudedMesh(mesh1d, nlayers, extrusion_type='radial')

    V = gadopt.utility.get_functionspace(mesh, "DG", 2, "DG", 2)
    P0 = fd.FunctionSpace(mesh, "DQ", 0)
    u = fd.Function(V, name='limited')
    u0 = fd.Function(V, name='orig')
    x, y = fd.SpatialCoordinate(mesh)
    u.interpolate(fd.atan2(y, x)+0.1*(x**2+y**2))
    u0.assign(u)
    limiter = gadopt.VertexBasedLeastSquaresLimiter(V)
    limiter.apply(u)
    # mask out cells that contain the atan2 jump and its neighbours
    mask = fd.Function(P0, name='mask')
    mask.assign(1.)
    mask.dat.data[(nhcells//2-1)*nlayers:(nhcells//2+2)*nlayers] = 0.

    # anywhere else, no limiting should be needed
    diff = fd.Function(V, name='diff')
    diff.interpolate(mask*(u-u0))
    #np.testing.assert_allclose(diff.dat.data, 0, atol=1e-12)

    iterations = fd.Function(P0, name='iterations')
    iterations.dat.data[:] = limiter.iterations.dat.data.astype('float64')
    fd.File('test.pvd').write(mask, u0, u, diff, iterations)


def _lexsort(array):
    # lexical sort on columns with first column as primary key
    ind = np.lexsort(array.T[::-1])
    return array[ind]


def test_exterior_facet_averager():
    mesh1d = fd.UnitIntervalMesh(2)
    meshes = [
        fd.UnitSquareMesh(2, 2, quadrilateral=True),
        fd.ExtrudedMesh(mesh1d, 2)
    ]
    for mesh in meshes:
        x, y = fd.SpatialCoordinate(mesh)

        # test averager in non-trivial scalar space
        DPC2 = fd.FunctionSpace(mesh, "DPC", 2)
        u = fd.project(3*x**2 + 30*y**2, DPC2)
        efa = ExteriorFacetAverager(u)
        efa.average()
        expected = [0, 0, 0, 0, 0.25, 1.75, 2.5, 5.5, 17.5, 20.5, 30.25, 31.75]
        np.testing.assert_allclose(np.sort(efa.facet_average.dat.data[:]), expected, atol=1e-12)

        # test in non-trivial vector space
        RT1 = fd.FunctionSpace(mesh, "RTCF", 1)
        u = fd.project(fd.as_vector((x, y)), RT1)
        efa = ExteriorFacetAverager(u)
        efa.average()
        expected = [[0.0, 0.0]]*4 # 4 interior facets set to zero
        expected += [[0.0, 0.25], [0.0, 0.75], [0.25, 0.0], [0.25, 1.0], [0.75, 0.0], [0.75, 1.0], [1.0, 0.25], [1.0, 0.75]]
        np.testing.assert_allclose(_lexsort(efa.facet_average.dat.data[:]), expected, atol=1e-12)
