import firedrake as fd
import numpy as np
import gadopt
import ufl
import pytest
from mpi4py import MPI

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


# Reductions over all ranks, so that every rank takes the same branch of each assertion.
# A rank-local assertion can fail on one rank only, which then leaves while the other
# ranks wait in the next collective call.
def global_min(comm, values):
    return comm.allreduce(np.min(values, initial=np.inf), MPI.MIN)


def global_max(comm, values):
    return comm.allreduce(np.max(values, initial=-np.inf), MPI.MAX)


def assert_allclose_global(comm, actual, desired, rtol=1e-7, atol=0.0):
    """Assert `np.allclose(actual, desired)` on every rank, with the tolerances of `assert_allclose`."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    excess = np.abs(actual - desired) - (atol + rtol*np.abs(desired))
    worst = global_max(comm, excess)
    assert worst <= 0.0, f"values differ by {worst} more than the tolerance on some rank"


@pytest.mark.parametrize("cell", MESHES)
def test_slope_limiters(cell):
    make_mesh, n = MESHES[cell]
    mesh = make_mesh()
    comm = mesh.comm
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
        assert_allclose_global(comm, u.dat.data_ro, v.dat.data_ro)

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
    np.testing.assert_allclose(global_max(comm, u.dat.data_ro), 0.5)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
    limiter.apply(u)
    # The limiter conserves the cell averages, so the volume does not change
    np.testing.assert_allclose(fd.assemble(u*fd.dx), vol0)
    np.testing.assert_allclose(global_max(comm, u.dat.data_ro), limited_max)

    # Repeat for a vector function space
    P1DG = fd.VectorFunctionSpace(mesh, elt)
    limiter = gadopt.VertexBasedP1DGLimiter(P1DG)
    hat = [0.5 - abs(x - 0.5)] + [0]*(mesh.geometric_dimension - 1)
    u = fd.Function(P1DG).interpolate(fd.as_vector(hat))
    vol0 = fd.assemble(u[0]*fd.dx)
    np.testing.assert_allclose(vol0, 0.25)
    np.testing.assert_allclose(global_max(comm, u.dat.data_ro), 0.5)
    limiter.apply(u)
    np.testing.assert_allclose(fd.assemble(u[0]*fd.dx), vol0)
    np.testing.assert_allclose(global_max(comm, u.dat.data_ro), limited_max)


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
        assert_allclose_global(mesh.comm, u.dat.data_ro[:, i], u_i.dat.data_ro, atol=1e-12)


@pytest.mark.parametrize("cell", MESHES)
def test_limiter_clip_bounds(cell):
    """Clipped bounds keep the limited field within [clip_min, clip_max]."""
    mesh = MESHES[cell][0]()
    comm = mesh.comm
    x = fd.SpatialCoordinate(mesh)[0]
    P1DG = fd.FunctionSpace(mesh, p1dg_element(mesh))

    # Nodal values overshoot [0, 1] at x = 0 and x = 1, but all cell averages stay inside,
    # which the limiter needs to bring every nodal value into the clipped bounds
    u = fd.Function(P1DG).interpolate(0.5 + 1.05*(x - 0.5))

    assert global_min(comm, u.dat.data_ro) < 0 and global_max(comm, u.dat.data_ro) > 1
    vol0 = fd.assemble(u*fd.dx)

    limiter = gadopt.VertexBasedP1DGLimiter(P1DG, clip_min=0.0, clip_max=1.0)
    limiter.apply(u)

    assert global_min(comm, limiter.min_field.dat.data_ro) >= 0.0
    assert global_max(comm, limiter.max_field.dat.data_ro) <= 1.0
    assert global_min(comm, u.dat.data_ro) >= -1e-12
    assert global_max(comm, u.dat.data_ro) <= 1.0 + 1e-12
    np.testing.assert_allclose(fd.assemble(u*fd.dx), vol0)
