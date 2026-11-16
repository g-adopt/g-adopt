import firedrake as fd
import numpy as np
import gadopt
import ufl
import pytest

# Meshes for the limiter tests. Each value is (constructor, number of cells across x),
# and every mesh has a vertex column at x = 0.5 so that the hat function peaks at a vertex.
# The meshes are built inside the tests, so that a deselected case builds nothing.
MESHES = {
    "quadrilateral": (lambda: fd.UnitSquareMesh(10, 10, quadrilateral=True), 10),
    "triangle": (lambda: fd.UnitSquareMesh(10, 10), 10),
    "extruded-interval": (lambda: fd.ExtrudedMesh(fd.UnitIntervalMesh(10), 10), 10),
    "prism": (lambda: fd.ExtrudedMesh(fd.UnitSquareMesh(4, 4), 2), 4),
}


def p1dg_element(mesh):
    """Equispaced P1DG element on the cell of `mesh`."""
    cell = mesh.ufl_cell()
    if hasattr(cell, "sub_cells"):
        hcell, vcell = cell.sub_cells
        return fd.TensorProductElement(
            fd.FiniteElement("DG", hcell, 1, variant="equispaced"),
            fd.FiniteElement("DG", vcell, 1, variant="equispaced"),
        )
    family = "DG" if cell == ufl.triangle else "DQ"
    return fd.FiniteElement(family, cell, 1, variant="equispaced")


@pytest.mark.parametrize("cell", MESHES)
def test_slope_limiters(cell):
    make_mesh, n = MESHES[cell]
    mesh = make_mesh()
    X = fd.SpatialCoordinate(mesh)
    x = X[0]
    elt = p1dg_element(mesh)
    P1DG = fd.FunctionSpace(mesh, elt)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)

    # A linear function is not flattened at boundaries (fails with firedrake's VertexBasedLimiter).
    # Each direction tests a different boundary, including the bottom and top of extruded meshes.
    u = fd.Function(P1DG)
    for xi in X:
        u.interpolate(xi)
        v = u.copy(deepcopy=True)
        limiter.apply(u)
        np.testing.assert_allclose(u.dat.data_ro, v.dat.data_ro)

    # Cells with a vertex at x = 0.5 are limited to the largest cell average next to that vertex.
    # The largest average comes from cells with the most vertices at x = 0.5: two of three
    # on triangles and prisms, two of four on quadrilaterals. The other vertices sit at 0.5 - h.
    h = 1/n
    if cell in ("triangle", "prism"):
        limited_max = (2*0.5 + (0.5 - h))/3
    else:
        limited_max = (0.5 + (0.5 - h))/2

    # Hat function
    u.interpolate(0.5 - abs(x - 0.5))
    vol0 = fd.assemble(u*fd.dx)
    np.testing.assert_allclose(vol0, 0.25)
    np.testing.assert_allclose(u.dat.data_ro.max(), 0.5)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
    limiter.apply(u)
    # The limiter conserves the cell averages, so the volume does not change
    np.testing.assert_allclose(fd.assemble(u*fd.dx), vol0)
    np.testing.assert_allclose(u.dat.data_ro.max(), limited_max)

    # Repeat for a vector function space
    P1DG = fd.VectorFunctionSpace(mesh, elt)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
    hat = [0.5 - abs(x - 0.5)] + [0]*(mesh.geometric_dimension - 1)
    u = fd.Function(P1DG).interpolate(fd.as_vector(hat))
    vol0 = fd.assemble(u[0]*fd.dx)
    np.testing.assert_allclose(vol0, 0.25)
    np.testing.assert_allclose(u.dat.data_ro.max(), 0.5)
    limiter.apply(u)
    np.testing.assert_allclose(fd.assemble(u[0]*fd.dx), vol0)
    np.testing.assert_allclose(u.dat.data_ro.max(), limited_max)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral", "prism"])
def test_vector_limiter_matches_components(cell):
    """Limiting a vector field gives the same result as limiting each component."""
    if cell == "prism":
        mesh = fd.ExtrudedMesh(fd.UnitSquareMesh(6, 5), 3, layer_height=1/3)
    else:
        mesh = fd.UnitSquareMesh(6, 5, quadrilateral=(cell == "quadrilateral"))
    elt = p1dg_element(mesh)

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


@pytest.mark.parametrize("cell", MESHES)
def test_limiter_clip_bounds(cell):
    """Clipped bounds keep the limited field within [clip_min, clip_max]."""
    mesh = MESHES[cell][0]()
    x = fd.SpatialCoordinate(mesh)[0]
    P1DG = fd.FunctionSpace(mesh, p1dg_element(mesh))

    # Nodal values overshoot [0, 1] at x = 0 and x = 1, but all cell averages stay inside,
    # which the limiter needs to bring every nodal value into the clipped bounds
    u = fd.Function(P1DG).interpolate(0.5 + 1.05*(x - 0.5))

    assert u.dat.data_ro.min() < 0 and u.dat.data_ro.max() > 1
    vol0 = fd.assemble(u*fd.dx)

    limiter = gadopt.VertexBasedP1DGLimiter(P1DG, clip_min=0.0, clip_max=1.0)
    limiter.apply(u)

    assert limiter.min_field.dat.data_ro.min() >= 0.0
    assert limiter.max_field.dat.data_ro.max() <= 1.0
    assert u.dat.data_ro.min() >= -1e-12
    assert u.dat.data_ro.max() <= 1.0 + 1e-12
    np.testing.assert_allclose(fd.assemble(u*fd.dx), vol0)


def test_limiter_rejects_other_function_space():
    """compute_bounds refuses a field from a space other than the one it was built for."""
    mesh = fd.UnitSquareMesh(4, 4)
    limiter = gadopt.VertexBasedP1DGLimiter(fd.FunctionSpace(mesh, p1dg_element(mesh)))

    # The boundary loops index the field with the limiter's own node layout, so a field
    # on another space must raise instead of being read with the wrong layout.
    # `apply` is not used here because Firedrake's `VertexBasedLimiter.apply` makes the
    # same check with an assert first.
    other = fd.Function(fd.FunctionSpace(fd.UnitSquareMesh(5, 5), p1dg_element(mesh)))
    with pytest.raises(ValueError, match="function space"):
        limiter.compute_bounds(other)
