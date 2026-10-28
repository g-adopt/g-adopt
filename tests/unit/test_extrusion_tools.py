import firedrake as fd
from gadopt.utility import extend_function_to_3d
import numpy as np
import pytest


def test_extend_function_to_3d():
    mesh2d = fd.UnitSquareMesh(5, 5)
    mesh3d = fd.ExtrudedMesh(mesh2d, 5, layer_height=1.)
    V2d = fd.FunctionSpace(mesh2d, "CG", 1)
    u2d = fd.Function(V2d).interpolate(fd.SpatialCoordinate(mesh2d)[0])
    u3d = extend_function_to_3d(u2d, mesh3d)
    np.testing.assert_allclose(fd.assemble(u3d * fd.dx), 2.5)


def base_and_extruded_mesh(base):
    """Returns a base mesh and its extrusion."""
    if base == "interval":
        mesh = fd.UnitIntervalMesh(7)
    else:
        mesh = fd.UnitSquareMesh(4, 3, quadrilateral=(base == "quadrilateral"))
    return mesh, fd.ExtrudedMesh(mesh, 3, layer_height=1/3)


def horizontal_expression(x, dim, vector):
    """Returns an expression of the horizontal coordinates `x[:dim]`."""
    if vector:
        return fd.as_vector([x[i]**2 + (i + 1) * x[dim - 1] for i in range(dim)])
    return sum((i + 1) * x[i] for i in range(dim)) + x[0]**2


@pytest.mark.parametrize("base", ["interval", "triangle", "quadrilateral"])
@pytest.mark.parametrize("degree,vector,variant", [
    (1, False, None), (1, True, None), (2, False, None),
    (2, False, "spectral"), (2, False, "integral"), (2, False, "fdm"),
])
def test_extend_function_to_3d_shares_data(base, degree, vector, variant):
    """The extended function has the source values, and the two share their data."""
    if variant == "fdm" and base == "triangle":
        pytest.skip("The fdm variant is not defined on triangles")
    mesh, mesh_extruded = base_and_extruded_mesh(base)
    dim = mesh.geometric_dimension
    space = fd.VectorFunctionSpace if vector else fd.FunctionSpace
    u2d = fd.Function(space(mesh, fd.FiniteElement("CG", mesh.ufl_cell(), degree, variant=variant)))
    u2d.interpolate(horizontal_expression(fd.SpatialCoordinate(mesh), dim, vector))

    u3d = extend_function_to_3d(u2d, mesh_extruded)
    assert u3d.source is u2d
    # Reference: the same expression interpolated directly into the V x R space.
    x3d = fd.SpatialCoordinate(mesh_extruded)
    expr3d = horizontal_expression(x3d, dim, vector)
    ref = fd.Function(u3d.function_space()).interpolate(expr3d)
    assert fd.errornorm(ref, u3d) < 1e-12

    # A change of the owned source dofs shows in the extended function. The references get the
    # same change of dofs, because for the integral and fdm variants a dof is not a point value.
    u2d.dat.data[:] += 1
    ref.dat.data[:] += 1
    assert fd.errornorm(ref, u3d) < 1e-12

    # A change of the owned extended dofs shows in the source.
    u3d.dat.data[:] += 1
    ref2d = fd.Function(u2d.function_space())
    ref2d.interpolate(horizontal_expression(fd.SpatialCoordinate(mesh), dim, vector))
    ref2d.dat.data[:] += 2
    assert fd.errornorm(ref2d, u2d) < 1e-12


def test_extend_function_to_3d_rejects_unsupported_source():
    """Sources whose data layout differs from the V x R space give an error."""
    mesh, mesh_extruded = base_and_extruded_mesh("interval")
    Q = fd.FunctionSpace(mesh, "CG", 1)

    z = fd.Function(Q * Q)
    with pytest.raises(ValueError, match="mixed"):
        extend_function_to_3d(z.subfunctions[1], mesh_extruded)

    u_dg = fd.Function(fd.FunctionSpace(mesh, "DG", 1))
    with pytest.raises(ValueError, match="unsupported element"):
        extend_function_to_3d(u_dg, mesh_extruded)

    u_restricted = fd.Function(fd.RestrictedFunctionSpace(Q, boundary_set=[1]))
    with pytest.raises(ValueError, match="restricted"):
        extend_function_to_3d(u_restricted, mesh_extruded)

    other_mesh = fd.ExtrudedMesh(fd.UnitIntervalMesh(7), 3, layer_height=1/3)
    with pytest.raises(ValueError, match="base mesh"):
        extend_function_to_3d(fd.Function(Q), other_mesh)
