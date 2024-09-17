from gadopt import *


def test_layer_averaging():
    n = 10
    mesh = UnitSquareMesh(n, n)
    mesh.cartesian = True

    # because this is not an extruded mesh we have to provide r1d for layer averaging
    y_disc = np.linspace(0.0, 1.0, n + 1)

    X = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 2)
    q = Function(Q).interpolate(X[1])

    averager = LayerAveraging(mesh, y_disc if mesh.layers is None else None)
    q_profile = averager.get_layer_average(q)

    assert np.allclose(q_profile, y_disc, atol=1e-10)
