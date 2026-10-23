import firedrake as fd
import numpy as np
import gadopt
import ufl
import pytest


@pytest.mark.parametrize(
    "mesh",
    [
        fd.UnitSquareMesh(10, 10, quadrilateral=True),
        fd.UnitSquareMesh(10, 10, quadrilateral=False),
        fd.ExtrudedMesh(fd.UnitIntervalMesh(10), 10),
    ],
    ids=["quad-true", "quad-false", "extruded"],
)
def test_slope_limiters(mesh):
    x, y = fd.SpatialCoordinate(mesh)
    mesh_cell = mesh.ufl_cell()
    element = "DG" if mesh_cell == ufl.triangle else "DQ"
    elt = fd.FiniteElement(element, mesh_cell, 1, variant="equispaced")
    P1DG = fd.FunctionSpace(mesh, elt)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)

    # test that linear function is not flattened at boundaries (fails with firedrake's VertexBasedLimiter)
    u = fd.Function(P1DG).interpolate(x)
    v = u.copy(deepcopy=True)
    limiter.apply(u)
    np.testing.assert_allclose(u.dat.data, v.dat.data)

    # same thing in y-direction to test bottom/top boundaries for extruded case
    u.interpolate(y)
    v = u.copy(deepcopy=True)
    limiter.apply(u)
    np.testing.assert_allclose(u.dat.data, v.dat.data)

    # test hat function
    u.interpolate(0.5 - abs(x - 0.5))
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
    if mesh_cell == ufl.triangle:
        # maximum from triangle with two x=0.5 vertices
        np.testing.assert_allclose(u.dat.data[:].max(), (2*0.5+1*0.4)/3)
    else:
        np.testing.assert_allclose(u.dat.data[:].max(), 0.45)

    # repeat for vector functionspace
    P1DG = fd.VectorFunctionSpace(mesh, elt)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
    u = fd.Function(P1DG).interpolate(fd.as_vector((0.5 - abs(x - 0.5), 0)))
    vol0 = fd.assemble(u[0]*fd.dx)
    np.testing.assert_allclose(vol0, 0.25)
    np.testing.assert_allclose(u.dat.data[:].max(), 0.5)
    limiter.apply(u)
    vol1 = fd.assemble(u[0]*fd.dx)
    # volume should be the same
    np.testing.assert_allclose(vol1, 0.25)
    # but cells with a x=0.5 vertex, should be limited to maximum cell average
    # adjacent to that vertex
    if mesh_cell == ufl.triangle:
        # maximum from triangle with two x=0.5 vertices
        np.testing.assert_allclose(u.dat.data[:].max(), (2*0.5+1*0.4)/3)
    else:
        np.testing.assert_allclose(u.dat.data[:].max(), 0.45)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral", "prism"])
def test_vector_limiter_matches_components(cell):
    """Limiting a vector field gives the same result as limiting each component."""
    if cell == "prism":
        mesh = fd.ExtrudedMesh(fd.UnitSquareMesh(6, 5), 3, layer_height=1/3)
        hcell, vcell = mesh.ufl_cell().sub_cells
        elt = fd.TensorProductElement(
            fd.FiniteElement("DG", hcell, 1, variant="equispaced"),
            fd.FiniteElement("DG", vcell, 1, variant="equispaced"),
        )
    else:
        mesh = fd.UnitSquareMesh(6, 5, quadrilateral=(cell == "quadrilateral"))
        element = "DG" if cell == "triangle" else "DQ"
        elt = fd.FiniteElement(element, mesh.ufl_cell(), 1, variant="equispaced")

    X = fd.SpatialCoordinate(mesh)
    # One hat per component, each peaking at a mesh vertex, so the limiter changes every component
    peaks = [0.5, 0.4, 1/3]
    exprs = [peaks[i] - abs(X[i] - peaks[i]) for i in range(mesh.geometric_dimension)]

    V_vec = fd.VectorFunctionSpace(mesh, elt)
    V_scalar = fd.FunctionSpace(mesh, elt)

    u = fd.Function(V_vec).interpolate(fd.as_vector(exprs))
    u_initial = u.copy(deepcopy=True)
    gadopt.VertexBasedP1DGLimiter(V_vec).apply(u)

    scalar_limiter = gadopt.VertexBasedP1DGLimiter(V_scalar)
    for i, expr in enumerate(exprs):
        assert fd.assemble((u[i] - u_initial[i])**2*fd.dx) > 1e-6
        u_i = fd.Function(V_scalar).interpolate(expr)
        scalar_limiter.apply(u_i)
        np.testing.assert_allclose(u.dat.data_ro[:, i], u_i.dat.data_ro, atol=1e-12)
