import firedrake as fd
import numpy as np

from gadopt.utility import extend_function_to_3d


def test_extend_function_to_3d():
    mesh2d = fd.UnitSquareMesh(5, 5)
    V2d = fd.FunctionSpace(mesh2d, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh2d)
    u2d = fd.Function(V2d).interpolate(x)

    mesh3d = fd.ExtrudedMesh(mesh2d, 5, layer_height=1.0)
    u3d = extend_function_to_3d(u2d, mesh3d)

    np.testing.assert_allclose(fd.assemble(u3d * fd.dx), 2.5)
